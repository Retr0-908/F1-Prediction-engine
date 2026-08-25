"""
elo_ratings.py — Driver & Constructor Glicko-2 Rating System for F1
Upgraded from basic Elo to Glicko-2 (Glickman 1999, 2012).

Glicko-2 adds two key parameters over simple Elo:
  - Rating Deviation (RD / φ):  uncertainty in the rating estimate.
    Rookies & drivers returning from breaks have high RD → predictions are
    appropriately less confident.
  - Volatility (σ):  how erratic the driver's performance is.
    Consistent drivers (Verstappen) get low σ; erratic drivers get high σ.

References:
  - Glickman, M.E. (1999). "Parameter estimation in large dynamic paired comparison experiments."
    Journal of the Royal Statistical Society, Series C.
  - Glickman, M.E. (2012). "Example of Glicko-2 system."
    http://www.glicko.net/glicko/glicko2.pdf
  - Elo, A.E. (1978). "The Rating of Chess Players, Past and Present."
  - Cattelan (2012) "Models for Paired Comparison Data"

In F1 every race produces N*(N-1)/2 pairwise outcomes. We credit a Glicko-2
"win" for every driver a given driver finishes ahead of. Each pairwise
comparison update happens within a rating period (one race).
"""

import math
import pickle
import time
from pathlib import Path
import datetime
from typing import Optional

from engine.core.data_fetcher import get_season_results, get_season_schedule, HISTORICAL_SEASONS
from engine.core.paths import ELO_CACHE_DIR as _ELO_CACHE_DIR
from engine.core.config import CURRENT_SEASON
import logging


logger = logging.getLogger("f1_predictor.elo_ratings")
# ─────────────────────────────────────────────
# GLICKO-2 CONSTANTS
# ─────────────────────────────────────────────
GLICKO2_MU            = 1500.0   # initial rating (displayed scale)
GLICKO2_PHI           = 350.0    # initial rating deviation (high = new driver)
GLICKO2_SIGMA         = 0.06     # initial volatility
GLICKO2_TAU           = 0.5      # system constant (controls volatility change)
GLICKO2_EPSILON       = 0.000001 # convergence threshold for volatility update
GLICKO2_SCALE         = 173.7178 # converts displayed scale ↔ internal Glicko-2 scale

ELO_BASE              = 1500.0   # kept for backward compat

# Between-season RD inflation constants
BETWEEN_PERIOD_PHI_INCREASE = 50.0   # how much RD grows over a full off-season
ROOKIE_INITIAL_PHI          = 350.0  # full uncertainty for first-timers
VETERAN_PHI_FLOOR           = 80.0   # minimum RD a stable veteran can reach


# ─────────────────────────────────────────────
# GLICKO-2 MATH HELPERS
# ─────────────────────────────────────────────
def _to_internal(mu_display: float) -> float:
    """Convert displayed rating to internal Glicko-2 mu."""
    return (mu_display - GLICKO2_MU) / GLICKO2_SCALE


def _to_display(mu_internal: float) -> float:
    """Convert internal Glicko-2 mu to displayed rating."""
    return GLICKO2_SCALE * mu_internal + GLICKO2_MU


def _g(phi: float) -> float:
    """The g(φ) function used in Glicko-2 expected score calculation."""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi ** 2 / (math.pi ** 2))


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    """Expected score of driver with rating mu against opponent with rating mu_j, RD phi_j."""
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


class Glicko2Driver:
    """Stores the three-parameter Glicko-2 state for one driver."""
    __slots__ = ("mu", "phi", "sigma", "name", "race_count")

    def __init__(self, name: str, mu: float = GLICKO2_MU, phi: float = GLICKO2_PHI,
                 sigma: float = GLICKO2_SIGMA):
        self.name       = name
        self.mu         = mu        # displayed rating
        self.phi        = phi       # rating deviation
        self.sigma      = sigma     # volatility
        self.race_count = 0         # total races processed

    @property
    def mu_internal(self) -> float:
        return _to_internal(self.mu)

    @property
    def phi_internal(self) -> float:
        return self.phi / GLICKO2_SCALE

    def confidence_interval_95(self) -> tuple:
        """Return 95% CI: (lower, upper) in displayed scale."""
        margin = 1.96 * self.phi
        return (round(self.mu - margin, 0), round(self.mu + margin, 0))


# ─────────────────────────────────────────────
# MAIN GLICKO-2 RATING SYSTEM
# ─────────────────────────────────────────────
class Glicko2RatingSystem:
    """
    Glicko-2 rating system for F1 drivers.

    Each race is treated as a rating period. Within each race, every pair of
    finishers produces a win/loss outcome (higher finisher = win).

    Between seasons:
      - φ (RD) is inflated to model off-season rust.
      - Drivers who didn't race get more inflation.
      - Volatility (σ) is updated per race via the Glicko-2 algorithm.
    """

    def __init__(self):
        self.drivers: dict[str, Glicko2Driver] = {}
        self.history: dict[str, list]          = {}   # driver → [(race_key, mu, phi), ...]
        self.seasons_processed: list[int]       = []

    def _get_driver(self, name: str) -> Glicko2Driver:
        if name not in self.drivers:
            self.drivers[name] = Glicko2Driver(name)
        return self.drivers[name]

    def get_rating(self, name: str) -> float:
        """Return current displayed rating."""
        return self.drivers[name].mu if name in self.drivers else GLICKO2_MU

    def get_rd(self, name: str) -> float:
        """Return current rating deviation (φ)."""
        return self.drivers[name].phi if name in self.drivers else GLICKO2_PHI

    def get_volatility(self, name: str) -> float:
        """Return current volatility (σ)."""
        return self.drivers[name].sigma if name in self.drivers else GLICKO2_SIGMA

    def get_all_ratings(self) -> dict[str, float]:
        return {n: d.mu for n, d in self.drivers.items()}

    def get_ranked_drivers(self) -> list[tuple[str, float]]:
        return sorted(self.drivers.items(), key=lambda x: x[1].mu, reverse=True)

    def get_elo_features(self, driver: str, roster: set = None) -> dict:
        """
        Return rich Glicko-2-based features for ML feature vector:
          elo_rating   : displayed Glicko-2 rating (backward-compat name)
          elo_rank     : rank among rated drivers (1 = best)
          elo_zscore   : std deviations above/below mean
          elo_rd       : rating deviation (uncertainty) — NEW
          elo_volatility: volatility — NEW
          elo_confidence: narrowness (high = confident estimate)

        roster: optional set of driver names restricting the comparison pool
        (e.g. the current lineup). Without it, retired drivers accumulated in
        `self.drivers` pollute rank/z-score/percentile features.
        """
        d = self.drivers.get(driver)
        rating  = d.mu    if d else GLICKO2_MU
        phi     = d.phi   if d else GLICKO2_PHI
        sigma   = d.sigma if d else GLICKO2_SIGMA

        pool = [dd.mu for n, dd in self.drivers.items()
                if roster is None or n in roster or n == driver]
        if not pool:
            pool = [rating]
        mean = sum(pool) / len(pool)
        std  = math.sqrt(sum((r - mean) ** 2 for r in pool) / len(pool)) or 1.0
        rank = sum(1 for r in pool if r > rating) + 1

        return {
            "elo_rating":     round(rating, 2),
            "elo_rank":       rank,
            "elo_zscore":     round((rating - mean) / std, 3),
            "elo_percentile": round(100 * (1 - rank / max(len(pool), 1)), 1),
            "elo_rd":         round(phi, 2),
            "elo_volatility": round(sigma, 4),
            "elo_confidence": round(max(0.0, min(1.0, 1.0 - phi / GLICKO2_PHI)), 3),
        }

    # ─────────────────────────────────────────
    # RATING PERIOD UPDATE (one race)
    # ─────────────────────────────────────────
    def process_race(self, results: list[dict], season: int, round_num: int):
        """
        Update Glicko-2 ratings from a single race result.
        results: list of {name, position, dnf, ...} sorted by finishing position.
        """
        finishers = sorted(results, key=lambda r: (r.get("dnf", False), r.get("position", 99)))
        if len(finishers) < 2:
            return

        race_key = f"{season}_R{round_num:02d}"
        n = len(finishers)

        # ── Step 1: collect all opponents' data for each driver ──
        # Build internal-scale state snapshot (pre-update)
        snap: dict[str, tuple] = {}  # name → (mu_int, phi_int)
        for f in finishers:
            d = self._get_driver(f["name"])
            snap[f["name"]] = (d.mu_internal, d.phi_internal)

        # ── Step 2: compute new ratings driver by driver ──
        updates: dict[str, Glicko2Driver] = {}
        for i, drv_a in enumerate(finishers):
            name_a = drv_a["name"]
            d_a    = self._get_driver(name_a)
            mu_a, phi_a = snap[name_a]

            # Collect outcomes against all other finishers in this race
            v_sum       = 0.0   # v^-1 accumulator
            delta_sum   = 0.0
            for j, drv_b in enumerate(finishers):
                if i == j:
                    continue
                # Two retired drivers have no meaningful ordering — their
                # stable-sort order is arbitrary noise in the pairwise sums.
                if drv_a.get("dnf", False) and drv_b.get("dnf", False):
                    continue
                name_b = drv_b["name"]
                mu_b, phi_b = snap[name_b]

                g_b = _g(phi_b)
                E_ab = _E(mu_a, mu_b, phi_b)

                # A beats B if i < j (lower index = better finish)
                s = 1.0 if i < j else 0.0

                v_sum     += g_b ** 2 * E_ab * (1.0 - E_ab)
                delta_sum += g_b * (s - E_ab)

            # Canonical Glicko-2 accumulators: v⁻¹ = Σ g²·E·(1−E), Δ = v·Σ g·(s−E)
            # (summed over ALL opponents — no per-comparison division)
            v_inv = max(v_sum, 1e-9)
            v     = 1.0 / v_inv
            delta = v * delta_sum

            # ── Step 3: Glicko-2 volatility update (Illinois algorithm) ──
            sigma_new = self._update_volatility(d_a, v, delta)

            # ── Step 4: Update phi* (pre-rating-period RD with new volatility) ──
            phi_star = math.sqrt(phi_a ** 2 + sigma_new ** 2)

            # ── Step 5: Update rating and RD ──
            phi_new = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
            mu_new  = mu_a + phi_new ** 2 * delta_sum

            # Convert back to display scale
            new_mu_disp  = _to_display(mu_new)
            new_phi_disp = phi_new * GLICKO2_SCALE

            # Clamp RD to reasonable range
            new_phi_disp = max(VETERAN_PHI_FLOOR, min(ROOKIE_INITIAL_PHI, new_phi_disp))

            updated = Glicko2Driver(name_a, mu=new_mu_disp,
                                    phi=new_phi_disp, sigma=sigma_new)
            updated.race_count = d_a.race_count + 1
            updates[name_a] = updated

        # Apply all updates atomically
        for name, updated in updates.items():
            self.drivers[name] = updated
            if name not in self.history:
                self.history[name] = []
            self.history[name].append((race_key, round(updated.mu, 1), round(updated.phi, 1)))

    def _update_volatility(self, d: Glicko2Driver, v: float, delta: float) -> float:
        """Illinois algorithm for Glicko-2 volatility update."""
        tau   = GLICKO2_TAU
        sigma = d.sigma
        phi   = d.phi_internal
        delta_sq = delta ** 2

        a = math.log(sigma ** 2)

        def f(x):
            ex = math.exp(x)
            top = ex * (delta_sq - phi ** 2 - v - ex)
            bot = 2 * (phi ** 2 + v + ex) ** 2
            if bot == 0:
                return float("nan")
            return top / bot - (x - a) / (tau ** 2)

        # Bracket the root
        A = a
        if delta_sq > phi ** 2 + v:
            B = math.log(delta_sq - phi ** 2 - v)
        else:
            k = 1
            while f(a - k * tau) < 0:
                k += 1
                if k > 200:   # safety: never spin forever
                    break
            B = a - k * tau

        fa = f(A)
        fb = f(B)

        # Guard: if initial bracket is degenerate, return current sigma unchanged
        if not math.isfinite(fa) or not math.isfinite(fb):
            return sigma

        for _ in range(100):  # max iterations
            denom = fb - fa
            if denom == 0:
                break  # converged or degenerate — stop
            C  = A + (A - B) * fa / denom
            fc = f(C)
            if not math.isfinite(fc):
                break
            if fc * fb < 0:
                A, fa = B, fb
            else:
                fa /= 2
            B, fb = C, fc
            if abs(B - A) < GLICKO2_EPSILON:
                break

        return math.exp(B / 2)

    # ─────────────────────────────────────────
    # BUILD FROM HISTORY
    # ─────────────────────────────────────────
    def build_from_history(self, seasons: list[int] = None, verbose: bool = False):
        """Build ratings by processing all historical seasons chronologically."""
        if seasons is None:
            seasons = sorted(HISTORICAL_SEASONS)

        all_seasons = sorted(seasons)
        for year in all_seasons:
            if verbose:
                print(f"    Processing {year} Glicko-2 ratings...", end=" ", flush=True)
            try:
                results = get_season_results(year)
                if not results:
                    if verbose:
                        print("no data")
                    continue

                from collections import defaultdict
                by_round: dict[int, list] = defaultdict(list)
                for r in results:
                    by_round[r["round"]].append(r)

                for rnd in sorted(by_round.keys()):
                    self.process_race(by_round[rnd], year, rnd)

                self.seasons_processed.append(year)
                if verbose:
                    top3 = sorted(self.drivers.items(), key=lambda x: x[1].mu, reverse=True)[:3]
                    top3_str = ", ".join(f"{n} ({d.mu:.0f}±{d.phi:.0f})" for n, d in top3)
                    print(f"done. Top-3: {top3_str}")

                # Between-season RD inflation (models off-season rust)
                if year != all_seasons[-1]:
                    self._inflate_rds_off_season()

            except Exception as e:
                if verbose:
                    print(f"error: {e}")
                continue

    def _inflate_rds_off_season(self):
        """
        Inflate all drivers' RDs to model off-season uncertainty.
        The standard Glicko-2 formula: φ* = sqrt(φ^2 + τ_between^2)
        We use BETWEEN_PERIOD_PHI_INCREASE as the tau_between.
        """
        tau_between = BETWEEN_PERIOD_PHI_INCREASE / GLICKO2_SCALE  # internal scale
        for d in self.drivers.values():
            phi_int = d.phi_internal
            new_phi_int = math.sqrt(phi_int ** 2 + tau_between ** 2)
            d.phi = min(ROOKIE_INITIAL_PHI, new_phi_int * GLICKO2_SCALE)


# ─────────────────────────────────────────────
# CONSTRUCTOR GLICKO-2 SYSTEM
# ─────────────────────────────────────────────
class ConstructorEloSystem:
    """
    Constructor rating = weighted average of their two drivers' Glicko-2 ratings.
    Weight = 1/phi (more certain driver contributes more to constructor rating).
    Kept name ConstructorEloSystem for backward compatibility.
    """

    def __init__(self, driver_elo):
        # driver_elo can be Glicko2RatingSystem or legacy EloRatingSystem
        self.driver_elo = driver_elo
        self.ratings: dict[str, float] = {}

    def compute_constructor_ratings(self, team_roster: dict[str, str]) -> dict[str, float]:
        """
        Compute constructor rating as RD-weighted average of their drivers.
        Drivers with lower RD (more certain) contribute more weight.
        """
        team_data: dict[str, list] = {}
        for driver, team in team_roster.items():
            if team not in team_data:
                team_data[team] = []
            mu  = self.driver_elo.get_rating(driver)
            phi = self.driver_elo.get_rd(driver) if hasattr(self.driver_elo, "get_rd") else 300.0
            team_data[team].append((mu, phi))

        for team, entries in team_data.items():
            if not entries:
                self.ratings[team] = GLICKO2_MU
                continue
            # Precision-weighted average (1/φ²): an uncertain rookie's rating
            # contributes proportionally less than a settled veteran's.
            total_weight = sum(1.0 / max(phi, 1.0) ** 2 for _, phi in entries)
            self.ratings[team] = (
                sum(mu / max(phi, 1.0) ** 2 for mu, phi in entries) / total_weight
            )

        return dict(self.ratings)

    def get_rating(self, constructor: str) -> float:
        return self.ratings.get(constructor, GLICKO2_MU)


# ─────────────────────────────────────────────
# LEGACY EloRatingSystem stub for backward compatibility
# (backtest.py and other modules may reference it directly)
# ─────────────────────────────────────────────
class EloRatingSystem(Glicko2RatingSystem):
    """
    Backward-compatible alias: EloRatingSystem now delegates to Glicko-2.
    All existing call sites (get_rating, build_from_history, get_ranked_drivers,
    get_elo_features, get_all_ratings) work unchanged.
    """
    pass


# ─────────────────────────────────────────────
# CONVENIENCE: singleton Glicko-2 system
# ─────────────────────────────────────────────
_cached_elo: Optional[Glicko2RatingSystem] = None
_ELO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _latest_completed_round() -> int:
    """Latest completed round of the current season, or -1 if unknown.

    Embedded in the cache key so a newly completed race invalidates stale
    ratings immediately instead of serving them for up to 7 days.
    """
    try:
        from engine.core.data_fetcher import data_fetcher

        today = datetime.date.today()
        done = [
            r["round"] for r in data_fetcher.get_season_schedule(CURRENT_SEASON)
            if datetime.date.fromisoformat(r["date"]) < today
        ]
        return max(done) if done else 0
    except Exception:
        return -1


def get_elo_system(
    seasons: list[int] = None,
    verbose: bool = True,
    force_rebuild: bool = False,
) -> Glicko2RatingSystem:
    """
    Returns a Glicko-2 system built from historical data, cached to disk.
    The cache key includes the latest completed round: after every race the
    ratings rebuild once, then stay stable between rounds (7-day TTL as a
    safety net only).
    """
    global _cached_elo

    if seasons is None:
        seasons = sorted(HISTORICAL_SEASONS)

    latest_round = _latest_completed_round()
    key_str = "_".join(str(s) for s in seasons) + f"_r{latest_round}"
    cache_key = f"glicko2_{key_str}.pkl"
    cache_path = _ELO_CACHE_DIR / cache_key

    # In-process singleton: reuse ONLY when it was built for this exact key
    if (
        not force_rebuild
        and _cached_elo is not None
        and getattr(_cached_elo, "_cache_key", None) == cache_key
    ):
        return _cached_elo

    if not force_rebuild and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < 7 * 86400:  # TTL safety net — the round-keyed name does the real work
            try:
                with open(cache_path, "rb") as f:
                    loaded = pickle.load(f)
                loaded._cache_key = cache_key
                _cached_elo = loaded
                if verbose:
                    print(f"  Loaded cached Glicko-2 ratings ({seasons[0]}–{seasons[-1]}, through round {latest_round})")
                return _cached_elo
            except Exception:
                logger.warning("Suppressed error", exc_info=True)
                pass

    if verbose:
        print(f"  Building Glicko-2 ratings ({seasons[0]}–{seasons[-1]}, through round {latest_round})...")

    g2 = Glicko2RatingSystem()
    g2.build_from_history(seasons, verbose=verbose)
    g2._cache_key = cache_key

    try:
        with open(cache_path, "wb") as f:
            pickle.dump(g2, f)
    except Exception:
        logger.warning("Suppressed error", exc_info=True)
        pass

    _cached_elo = g2
    return g2
