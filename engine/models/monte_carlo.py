"""
monte_carlo.py — F1 Race Weekend Monte Carlo Simulator

Runs N probabilistic simulations of an F1 race weekend to produce
point distributions rather than single-point predictions.

Each simulation:
  1. Applies DNF events (per-driver probability)
  2. Optionally deploys a Safety Car at a random lap (using circuit SC probability)
  3. Applies weather variance noise if rain_risk is significant
  4. Computes fantasy points for that simulation iteration

Outputs: per-driver DistributionStats {mean, std, p10, p90, upside_pct, p_top3}

Research backing:
  - F1FantasyTools Monte Carlo methodology (dominant tool in expert community)
  - Expected Value optimization over single-point prediction (plainenglish.io)
  - Safety car "free stop" modeling (axiorablogs.com)
"""

import random
import math
import os
from typing import Optional, Dict, List

# Julia engine is DISABLED until behavioral parity with the Python path is
# achieved: it currently omits rain/sprint/circuit-feature effects and uses
# EV-style (not Bernoulli) FL/DOTD bonuses. Flip to True only after validating
# distribution equivalence against the Python engine.
USE_JULIA_ENGINE = False

try:
    from juliacall import Main as jl
    # Load the high-performance Julia engine
    engine_path = os.path.join(os.path.dirname(__file__), "julia", "monte_carlo_engine.jl")
    jl.include(engine_path)
    _JULIA_AVAILABLE = True
except ImportError:
    _JULIA_AVAILABLE = False
except Exception as e:
    print(f"  [monte_carlo] Julia engine load failed: {e}")
    _JULIA_AVAILABLE = False

import logging

from engine.core.config import (
    RACE_POSITION_POINTS, QUALI_POSITION_POINTS,
    QUALI_Q2_BONUS, QUALI_Q3_BONUS, POLE_BONUS,
    POSITIONS_GAINED_PER, POSITIONS_LOST_PER, DNF_PENALTY,
    FASTEST_LAP_BONUS, DRIVER_OF_DAY_BONUS,
    SPRINT_RACE_POINTS, SC_PROBABILITY, VSC_PROBABILITY,
)

logger = logging.getLogger("f1_predictor.monte_carlo")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DEFAULT_N_SIMULATIONS = 1000   # balance speed vs accuracy
RACE_LAPS_TYPICAL     = 57     # typical race length for SC timing

# How much qualifying performance weighs vs form in SC-reshuffled races
SC_POSITION_NOISE     = 2.5    # ±σ position noise under SC conditions


def _compute_driver_simulation_sigma(consistency_rating: float = 75.0, base_sigma: float = 0.15) -> float:
    """
    Computes per-driver Gaussian simulation standard deviation scaled by their consistency rating.
    Higher consistency yields tighter lap-time and pace distributions around predicted mean.
    Formula: sigma = base_sigma * (1.0 - 0.50 * (C / 100.0))
    """
    c = max(0.0, min(100.0, float(consistency_rating)))
    return float(base_sigma * (1.0 - 0.50 * (c / 100.0)))


class DistributionStats:
    """Holds Monte Carlo output statistics for one driver."""
    __slots__ = (
        "driver", "team", "mean_pts", "std_pts",
        "p10_pts", "p50_pts", "p90_pts",
        "upside_pct", "p_top3", "p_points_finish",
        "p_dnf", "n_sims",
    )

    def __init__(self, driver: str, team: str, pts_list: list[float]):
        self.driver  = driver
        self.team    = team
        pts_list     = list(pts_list)
        self.n_sims  = len(pts_list)
        if self.n_sims == 0:
            # Empty merge path (e.g. Julia counts all-zero) — neutral stats
            self.mean_pts = self.std_pts = 0.0
            self.p10_pts = self.p50_pts = self.p90_pts = 0.0
            self.upside_pct = self.p_top3 = self.p_points_finish = self.p_dnf = 0.0
            return

        sorted_pts   = sorted(pts_list)
        n            = len(sorted_pts)

        def _quantile(q: float) -> float:
            return sorted_pts[max(0, min(n - 1, int(q * (n - 1))))]

        self.mean_pts = sum(pts_list) / n
        variance      = sum((p - self.mean_pts) ** 2 for p in pts_list) / n
        self.std_pts  = math.sqrt(variance)
        self.p10_pts  = _quantile(0.10)
        self.p50_pts  = _quantile(0.50)
        self.p90_pts  = _quantile(0.90)

        # Upside: share of sims landing in the driver's own top quartile.
        # NOTE: by construction this is ~25% for every driver (tautological) —
        # it measures tail-shape consistency only. Kept for UI compat; do NOT
        # use it for cross-driver ranking.
        q75 = _quantile(0.75)
        self.upside_pct = sum(1 for p in pts_list if p >= q75) / n * 100

        # Probability of a top-3-equivalent points haul (P3 + typical extras ≈ 15+)
        self.p_top3 = sum(1 for p in pts_list if p >= 15.0) / n * 100

        # Points finish probability (a genuine points finish usually clears 5)
        self.p_points_finish = sum(1 for p in pts_list if p > 5.0) / n * 100

        # DNF probability (pts < -5 — negative from DNF penalty)
        self.p_dnf = sum(1 for p in pts_list if p < -5.0) / n * 100

    def to_dict(self) -> dict:
        return {
            "driver":           self.driver,
            "team":             self.team,
            "mean_pts":         round(self.mean_pts, 2),
            "std_pts":          round(self.std_pts, 2),
            "p10_pts":          round(self.p10_pts, 1),
            "p50_pts":          round(self.p50_pts, 1),
            "p90_pts":          round(self.p90_pts, 1),
            "upside_pct":       round(self.upside_pct, 1),
            "p_top3":           round(self.p_top3, 1),
            "p_points_finish":  round(self.p_points_finish, 1),
            "p_dnf":            round(self.p_dnf, 1),
            "n_sims":           self.n_sims,
        }


# ─────────────────────────────────────────────
# CORE SIMULATION ENGINE
# ─────────────────────────────────────────────
def _simulate_one_race(
    race_order: list[dict],
    quali_order: list[dict],
    sc_prob: float,
    vsc_prob: float,
    rain_risk: str,
    is_sprint: bool,
    sprint_order: Optional[list[dict]],
    rng: random.Random,
    circuit_features: Optional[dict] = None,  # NEW: dict from track_features_loader
    grid_penalties: Optional[dict] = None,   # {driver_name: penalty_positions_int}
) -> dict[str, float]:
    """
    Run one simulation of the race weekend.
    Returns {driver_name: simulated_fantasy_points}.
    """
    n_drivers = len(race_order)
    
    # ── 1. Build initial grid/race order positions ──
    # race_order already sorted by predicted_rank
    positions = {d["driver"]: d["predicted_rank"] for d in race_order}
    raw_effective = []
    for q in quali_order:
        drv_name = q["driver"]
        base_grid = q.get("predicted_grid", 20)
        # Post-quali mode: predicted_grid already reflects penalties (locked
        # actual grid). Only add penalties to a raw pre-quali prediction.
        if q.get("is_actual", False):
            target_grid = base_grid
        else:
            target_grid = base_grid + (grid_penalties or {}).get(drv_name, 0)
        raw_effective.append((target_grid, base_grid, drv_name))

    # Resolve penalties: sort primarily by penalized target grid,
    # break ties stably using original qualifying position
    raw_effective.sort(key=lambda x: (x[0], x[1]))
    effective_grid: dict[str, int] = {
        drv_name: i + 1 for i, (_, _, drv_name) in enumerate(raw_effective)
    }
    grid      = effective_grid
    # Plan 7c: grid integrity check
    if len(set(effective_grid.values())) != len(effective_grid):
        dupes = [d for d, g in effective_grid.items()
                 if list(effective_grid.values()).count(g) > 1]
        raise ValueError(f"MC grid integrity failure — duplicate grids for: {dupes}")
    dnf_probs = {d["driver"]: d.get("dnf_prob_pct", 7.0) / 100.0 for d in race_order}

    # ── 2. Apply DNF events (probabilistic) ──
    dnf_set = set()
    for drv, prob in dnf_probs.items():
        if rng.random() < prob:
            dnf_set.add(drv)

    # ── 2b. Correlated multi-car incident (high-SC circuits) ──
    # Real F1: a single crash can retire 2–3 cars simultaneously.
    # Independent Bernoulli misses this. At circuits with SC prob > 0.55,
    # add a 12% chance of a "shunt" that additionally DNFs 1–2 back-markers
    # regardless of their individual DNF probability.
    if sc_prob > 0.55 and rng.random() < 0.12:
        active_sorted = sorted(
            [d for d in positions if d not in dnf_set],
            key=lambda d: positions[d]
        )
        # Shunts disproportionately hit the back half of the field
        back_half = active_sorted[len(active_sorted)//2:]
        n_shunted = rng.randint(1, min(2, len(back_half)))
        for drv in rng.sample(back_half, n_shunted):
            dnf_set.add(drv)

    # ── 3. Weather + circuit variance (single coherent pass, plan 9-M1) ──
    # Base weather sigma from rain risk; circuit features modulate it via a
    # multiplicative factor applied to the ONE draw (the old code drew at full
    # sigma then added an abs() "delta" second draw — calm circuits ended up
    # MORE chaotic than baseline).
    weather_noise = 0.0
    if rain_risk == "high":
        weather_noise = 3.5
    elif rain_risk == "medium":
        weather_noise = 1.5

    if circuit_features:
        overtake_diff = circuit_features.get("overtaking_difficulty", 3)
        weather_var = circuit_features.get("weather_variability", 3)
        sm_eff = circuit_features.get("overtake_mode_efficiency", 0.5)
    else:
        overtake_diff = 3
        weather_var = 3
        sm_eff = 0.5

    consistency_lookup = {d["driver"]: d.get("consistency_rating", 75.0) for d in race_order}

    if weather_noise > 0:
        factor = (1.0 + (weather_var - 3) * 0.2) if circuit_features else 1.0
        base_sigma = max(0.4, weather_noise * factor)
        for drv in positions:
            if drv not in dnf_set:
                c = consistency_lookup.get(drv, 75.0)
                drv_sigma = _compute_driver_simulation_sigma(c, base_sigma)
                positions[drv] = max(1.0, positions[drv] + rng.gauss(0, drv_sigma))
    else:
        # Dry race: overtaking difficulty creates position variance scaled by consistency
        base_overtake_noise = max(
            0.0, ((1.0 + (3 - overtake_diff) * 0.15) - 1.0) if circuit_features else 0.0)
        if base_overtake_noise > 0:
            for drv in positions:
                if drv not in dnf_set:
                    c = consistency_lookup.get(drv, 75.0)
                    drv_sigma = _compute_driver_simulation_sigma(c, base_overtake_noise)
                    positions[drv] = max(1.0,
                                         positions[drv] + rng.gauss(0, drv_sigma))

    if circuit_features:
        # First lap incident risk
        first_lap_risk = circuit_features.get("first_lap_incident_risk", 0.07)
        if rng.random() < first_lap_risk:
            active_sorted_fl = sorted(
                [d for d in positions if d not in dnf_set],
                key=lambda d: positions[d]
            )
            mid_field = active_sorted_fl[5:15]  # P6-P15 most affected
            if mid_field:
                n_affected = rng.randint(1, min(2, len(mid_field)))
                for drv in rng.sample(mid_field, n_affected):
                    if rng.random() < 0.3:
                        dnf_set.add(drv)
                    else:
                        positions[drv] += rng.uniform(1.5, 4.0)

    # ── 4. Safety car deployment ──
    sc_triggered  = rng.random() < sc_prob
    vsc_triggered = (not sc_triggered) and (rng.random() < vsc_prob)

    if sc_triggered or vsc_triggered:
        # SC compresses the field and enables cheap pit stops: mid-field
        # (P6–P15 of the live order) gains; front-runners may lose slightly.
        # Tie-break deterministically so equal positions resolve by pace.
        sorted_drivers = sorted(
            [d for d in positions if d not in dnf_set],
            key=lambda d: (positions[d], -dnf_probs.get(d, 0.0)),
        )
        for i, drv in enumerate(sorted_drivers):
            rank = i + 1  # 1-indexed rank among active drivers
            if rank > 5 and rank <= 15:
                # SM efficiency reduces SC benefit (easier to re-overtake after restart)
                sc_gain_base = rng.uniform(0, 2.5) if sc_triggered else rng.uniform(0, 1.5)
                sc_gain = sc_gain_base * (1.0 - 0.3 * sm_eff)  # reduce by up to 30%
                positions[drv] = max(2.0, positions[drv] - sc_gain)
            elif rank <= 5:
                # Front runners may lose slightly if they pit under SC
                sc_loss = rng.uniform(0, 1.0)
                positions[drv] = positions[drv] + sc_loss

    # ── 5. Re-rank to integer positions ──
    active_drivers = [d for d in positions if d not in dnf_set]
    # Deterministic ordering: set iteration is PYTHONHASHSEED-dependent, which
    # would break the seeded-reproducibility contract across processes.
    dnf_drivers    = sorted(dnf_set, key=lambda d: positions.get(d, 99))
    active_sorted  = sorted(active_drivers, key=lambda d: positions[d])

    # Assign integer race positions
    final_pos: dict[str, int] = {}
    for i, drv in enumerate(active_sorted):
        final_pos[drv] = i + 1
    # DNF drivers go to the back
    for i, drv in enumerate(dnf_drivers):
        final_pos[drv] = len(active_sorted) + 1 + i

    # ── 6. Compute fantasy points for this simulation ──
    sim_pts: dict[str, float] = {}

    for drv_info in race_order:
        drv     = drv_info["driver"]
        race_p  = final_pos.get(drv, n_drivers)
        grid_p  = grid.get(drv, 10)
        is_dnf  = drv in dnf_set
        pts     = 0.0

        # Qualifying points — F1 Fantasy rules: top-10 → Q3 bonus (+ pole),
        # P11–15 → Q2 bonus only (never both). grid_p here is the effective
        # grid; bonus eligibility follows it as an approximation.
        if grid_p <= 10:
            pts += QUALI_POSITION_POINTS.get(grid_p, 0) + QUALI_Q3_BONUS
            if grid_p == 1:
                pts += POLE_BONUS
        elif grid_p <= 15:
            pts += float(QUALI_Q2_BONUS)

        if is_dnf:
            pts += DNF_PENALTY
        else:
            # Race position points
            pts += RACE_POSITION_POINTS.get(race_p, 0)

            # Positions gained/lost
            delta = grid_p - race_p
            if delta > 0:
                pts += delta * POSITIONS_GAINED_PER
            else:
                pts += abs(delta) * POSITIONS_LOST_PER

            # Fastest lap — only top-10 finishers are eligible
            fl_threshold = max(0.0, 0.09 - (race_p - 1) * 0.006)
            if race_p <= 10 and rng.random() < fl_threshold:
                pts += FASTEST_LAP_BONUS

            # Driver of the Day (random ~10% for top finishers)
            dotd_threshold = max(0.0, 0.10 - (race_p - 1) * 0.008)
            if rng.random() < dotd_threshold:
                pts += DRIVER_OF_DAY_BONUS

        # Sprint weekend bonus (simplified)
        if is_sprint and sprint_order:
            sprint_entry = next((s for s in sprint_order if s["driver"] == drv), None)
            if sprint_entry:
                sp_rank = sprint_entry.get("predicted_sprint_rank", race_p)
                sp_dnf  = rng.random() < (sprint_entry.get("dnf_prob_pct", 5.0) / 100.0 * 0.6)
                if not sp_dnf:
                    pts += SPRINT_RACE_POINTS.get(sp_rank, 0)

        sim_pts[drv] = pts

    return sim_pts


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# PARALLEL WORKER (plan I3/3c) — module-level for spawn safety; imports only
# stdlib + this module's own functions (no TF/predictor transitive imports).
# ─────────────────────────────────────────────
def _simulate_chunk(base_seed: int, n_sims: int, chunk_seed: int, payload: dict) -> dict:
    """Run a chunk of simulations with an isolated RNG. Returns
    {driver: [pts, ...]} plain-dict (spawn-picklable)."""
    rng = random.Random(chunk_seed)
    out: dict[str, list[float]] = {d["driver"]: [] for d in payload["race_order"]}
    for _ in range(n_sims):
        sim = _simulate_one_race(
            payload["race_order"], payload["quali_order"],
            payload["sc_prob"], payload["vsc_prob"], payload["rain_risk"],
            payload["is_sprint"], payload["sprint_order"],
            rng,
            circuit_features=payload["circuit_features"],
            grid_penalties=payload["grid_penalties"],
        )
        for drv, pts in sim.items():
            if drv in out:
                out[drv].append(pts)
    return out


def simulate_race_weekend(
    race_order: list[dict],
    quali_order: list[dict],
    circuit_config: dict,
    weather: dict,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    is_sprint: bool = False,
    sprint_order: Optional[list[dict]] = None,
    seed: int = 42,
    progress_callback = None,
    grid_penalties: Optional[dict] = None,
) -> dict[str, DistributionStats]:
    """
    Run N Monte Carlo simulations of the race weekend.

    Args:
        race_order:     predictor.predict_finishing_order() output
        quali_order:    predictor.predict_qualifying_order() output
        circuit_config: circuit config dict (contains "key" for SC prob lookup)
        weather:        weather dict (rain_risk key)
        n_simulations:  number of MC iterations
        is_sprint:      True for sprint weekends
        sprint_order:   predictor.predict_sprint_order() output (if sprint)
        seed:           random seed for reproducibility
        progress_callback: callback function for simulation progress
        grid_penalties: dict of {driver_name: penalty_int}

    Returns:
        Dict {driver_name: DistributionStats}
    """
    circuit_key = circuit_config.get("key", "")
    # Load per-circuit track features; use JSON probs if available, else fall back
    try:
        from engine.core.track_features_loader import load_track_features as _ltf2
        _track_data = _ltf2(circuit_key) or {}
    except Exception:
        _track_data = {}
    sc_prob  = _track_data.get("sc_probability",  SC_PROBABILITY.get(circuit_key, 0.40))
    vsc_prob = _track_data.get("vsc_probability", VSC_PROBABILITY.get(circuit_key, 0.30))
    rain_risk = weather.get("rain_risk", "low")
    
    # Extract driver-to-team mapping for the results
    driver_teams = {d["driver"]: d.get("team", "") for d in race_order}

    # ── Julia-Accelerated Simulation (Phase 5) ──
    # NOTE: gated behind USE_JULIA_ENGINE — see constant comment. When enabled,
    # Julia arrays are 1-BASED: convert to numpy before indexing.
    if USE_JULIA_ENGINE and _JULIA_AVAILABLE and n_simulations >= 5000:
        print(f"  [monte_carlo] Running {n_simulations} iterations via Julia engine...")
        
        # Prepare driver data for Julia
        # We derive a "base_lap" from the predicted_rank: P1 gets ~85s, P20 gets ~90s
        jl_drivers = []
        for d in race_order:
            drv_name = d["driver"]
            pred_rank = d.get("predicted_rank", 11.0)
            base_lap = 85.0 + (pred_rank - 1.0) * 0.25  # spread of ~5s across field
            
            # Glicko-2 rating for overtaking probability
            glicko = d.get("elo_rating", 1500.0)
            dnf_p = d.get("dnf_prob_pct", 7.0) / 100.0
            q_entry = next((q for q in quali_order if q["driver"] == drv_name), None)
            raw_grid = q_entry["predicted_grid"] if q_entry else 10
            if q_entry is not None and q_entry.get("is_actual", False):
                grid_p = int(raw_grid)          # already penalty-adjusted
            else:
                grid_p = int(raw_grid) + (grid_penalties or {}).get(drv_name, 0)
            
            jl_drivers.append({
                "name": drv_name,
                "base_lap": float(base_lap),
                "glicko": float(glicko),
                "dnf_p": float(dnf_p),
                "grid": int(grid_p)
            })

        # Phase 5 Physics Parameters
        # These would ideally come from the tire_model and config
        δ_vals = {"MEDIUM": 0.04, "SOFT": 0.06, "HARD": 0.02}
        γ_vals = {"MEDIUM": 0.0, "SOFT": -0.8, "HARD": 1.0}
        λ_fuel = 0.035      # seconds/lap improvement from fuel burn
        α_overtake = 0.15   # base overtaking scaling
        
        try:
            # Seed the Julia RNG so results are reproducible like the Python path
            try:
                jl.seval(f"import Random; Random.seed!({int(seed)})")
            except Exception:
                logger.warning("Suppressed error", exc_info=True)
                pass

            # Call Julia: returns (rank_counts_matrix, dnf_counts_vector)
            # rank_counts[driver_idx, rank] — JULIA ARRAYS ARE 1-BASED
            rank_counts_jl, dnf_counts_jl = jl.run_batch_simulation(
                jl_drivers, 
                57, # laps
                δ_vals, γ_vals, λ_fuel, α_overtake, sc_prob,
                n_simulations
            )
            import numpy as np

            rank_counts = np.asarray(rank_counts_jl)   # copy out of juliacall
            dnf_counts  = np.asarray(dnf_counts_jl)
            
            results = {}
            for i, d in enumerate(jl_drivers):
                drv_name = d["name"]
                pts_list = []
                grid_p = d["grid"]
                
                # Base qualifying points — same rules as the Python engine
                quali_pts = 0.0
                if grid_p <= 10:
                    quali_pts += QUALI_POSITION_POINTS.get(grid_p, 0) + QUALI_Q3_BONUS
                    if grid_p == 1:
                        quali_pts += POLE_BONUS
                elif grid_p <= 15:
                    quali_pts += float(QUALI_Q2_BONUS)
                
                dnf_count = int(dnf_counts[i])       # numpy: 0-based OK
                remaining_dnfs = dnf_count
                
                n_field = rank_counts.shape[1]
                # Traverse ranks from worst (n_field) to best (1);
                # column j holds count for finishing position j+1
                for rank in range(n_field, 0, -1):
                    count = int(rank_counts[i, rank-1])
                    if count == 0:
                        continue
                    
                    c_dnf = min(count, remaining_dnfs)
                    c_finish = count - c_dnf
                    
                    if c_dnf > 0:
                        pts_dnf = quali_pts + DNF_PENALTY
                        pts_list.extend([pts_dnf] * c_dnf)
                        remaining_dnfs -= c_dnf
                        
                    if c_finish > 0:
                        pts_finish = quali_pts + RACE_POSITION_POINTS.get(rank, 0)
                        
                        # Positions gained/lost
                        delta = grid_p - rank
                        if delta > 0:
                            pts_finish += delta * POSITIONS_GAINED_PER
                        else:
                            pts_finish += abs(delta) * POSITIONS_LOST_PER
                            
                        # Average fastest lap/DOTD probability for this rank
                        # (FL only for top-10 finishers, matching Python engine)
                        if rank <= 10:
                            fl_prob = max(0.0, 0.09 - (rank - 1) * 0.006)
                            pts_finish += FASTEST_LAP_BONUS * fl_prob
                        dotd_prob = max(0.0, 0.10 - (rank - 1) * 0.008)
                        pts_finish += DRIVER_OF_DAY_BONUS * dotd_prob
                        
                        pts_list.extend([pts_finish] * c_finish)
                
                results[drv_name] = DistributionStats(drv_name, driver_teams.get(drv_name, ""), pts_list)
            
            return results
        except Exception as e:
            print(f"  [monte_carlo] Julia execution failed: {e}. Falling back to Python...")

    # ── Python Implementation (serial or process-parallel, plan I3/3c) ──
    from engine.core.hardware import profile as _hw
    workers = _hw()["mc_workers"]

    if workers > 1 and n_simulations >= 2000:
        # Parallel path: chunk sims across processes with per-chunk derived
        # seeds (deterministic per seed). NOTE: results match the serial mode
        # in DISTRIBUTION, not bitwise — F1E_MC_WORKERS=serial is the exact
        # reproduction escape hatch.
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        chunk_size = math.ceil(n_simulations / workers)
        chunks = [(i * chunk_size, min(chunk_size, n_simulations - i * chunk_size))
                  for i in range((n_simulations + chunk_size - 1) // chunk_size)]

        payload = {
            "race_order": race_order, "quali_order": quali_order,
            "sc_prob": sc_prob, "vsc_prob": vsc_prob,
            "rain_risk": rain_risk, "is_sprint": is_sprint,
            "sprint_order": sprint_order,
            "circuit_features": _track_data, "grid_penalties": grid_penalties,
        }

        driver_pts_lists: dict[str, list[float]] = {d["driver"]: [] for d in race_order}
        done = 0
        ctx = mp.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                futs = [ex.submit(_simulate_chunk, base, size, seed * 1000003 + i, payload)
                        for i, (base, size) in enumerate(chunks)]
                for fut in as_completed(futs):
                    chunk_res = fut.result()
                    done += sum(len(v) for v in chunk_res.values())
                    for drv, pts in chunk_res.items():
                        if drv in driver_pts_lists:
                            driver_pts_lists[drv].extend(pts)
                    if progress_callback:
                        intermediate_ev = {
                            d: sum(v) / len(v)
                            for d, v in driver_pts_lists.items() if v
                        }
                        top_3 = sorted(intermediate_ev.items(),
                                       key=lambda x: x[1], reverse=True)[:3]
                        top_3_str = ", ".join(f"{d}: {p:.1f} EV" for d, p in top_3)
                        progress_callback(done, n_simulations, top_3_str)
        except Exception as e:
            print(f"  [monte_carlo] parallel MC failed ({e}) — falling back to serial")
            driver_pts_lists = {d["driver"]: [] for d in race_order}
            rng = random.Random(seed)
            for sim in range(n_simulations):
                sim_result = _simulate_one_race(
                    race_order, quali_order, sc_prob, vsc_prob, rain_risk,
                    is_sprint, sprint_order, rng,
                    circuit_features=_track_data, grid_penalties=grid_penalties)
                for drv, pts in sim_result.items():
                    if drv in driver_pts_lists:
                        driver_pts_lists[drv].append(pts)
    else:
        rng = random.Random(seed)
        driver_pts_lists: dict[str, list[float]] = {d["driver"]: [] for d in race_order}
        for sim in range(n_simulations):
            sim_result = _simulate_one_race(
                race_order, quali_order,
                sc_prob, vsc_prob, rain_risk,
                is_sprint, sprint_order,
                rng,
                circuit_features=_track_data,
                grid_penalties=grid_penalties,
            )
            for drv, pts in sim_result.items():
                if drv in driver_pts_lists:
                    driver_pts_lists[drv].append(pts)

            if progress_callback and (sim + 1) % 100 == 0:
                intermediate_ev = {}
                for d_name, pts_list in driver_pts_lists.items():
                    if pts_list:
                        intermediate_ev[d_name] = sum(pts_list) / len(pts_list)
                top_3 = sorted(intermediate_ev.items(), key=lambda x: x[1], reverse=True)[:3]
                top_3_str = ", ".join(f"{d_name}: {pts:.1f} EV" for d_name, pts in top_3)
                progress_callback(sim + 1, n_simulations, top_3_str)

    results: dict[str, DistributionStats] = {}
    for drv, pts_list in driver_pts_lists.items():
        if pts_list:
            results[drv] = DistributionStats(drv, driver_teams.get(drv, ""), pts_list)

    return results


def format_mc_summary(mc_results: dict[str, DistributionStats]) -> list[dict]:
    """
    Convert MC results to a sorted list of dicts for display/export.
    Sorted by mean_pts descending.
    """
    rows = [s.to_dict() for s in mc_results.values()]
    rows.sort(key=lambda r: r["mean_pts"], reverse=True)
    return rows


def get_expected_value_pts(mc_results: dict[str, DistributionStats]) -> dict[str, float]:
    """Return {driver: mean_pts} for use in optimizer (replaces deterministic pts)."""
    return {drv: stats.mean_pts for drv, stats in mc_results.items()}


def run_monte_carlo_simulation(
    race_order: list[dict],
    quali_order: list[dict],
    sc_prob: float = 0.40,
    vsc_prob: float = 0.30,
    rain_risk: str = "low",
    n_sims: int = DEFAULT_N_SIMULATIONS,
    seed: int = 42,
    circuit_key: str = "",
    grid_penalties: Optional[dict] = None,
    progress_callback = None,
) -> dict[str, DistributionStats]:
    """Convenience wrapper for simulate_race_weekend."""
    circuit_cfg = {"key": circuit_key}
    weather_cfg = {"rain_risk": rain_risk}
    return simulate_race_weekend(
        race_order=race_order,
        quali_order=quali_order,
        circuit_config=circuit_cfg,
        weather=weather_cfg,
        n_simulations=n_sims,
        seed=seed,
        progress_callback=progress_callback,
        grid_penalties=grid_penalties,
    )
