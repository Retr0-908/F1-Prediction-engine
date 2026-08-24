"""
predictor.py — F1 Race Prediction Engine (v3 — Phase 3 ML Upgrade)
Ensemble: RandomForest + XGBoost + LightGBM (4-model stack)
Features: 35 variables (up from 33)

New in v3 (Phase 3):
  - LSTM + Multi-Head Attention temporal form model (temporal_model.py)
    Output merged as `lstm_momentum_pos` feature
  - Multi-task tire degradation model (tire_model.py)
    Output merged as `driver_tire_efficiency_score` feature

Prior features (v2):
  - Glicko-2 ratings, safety car probability, FP2 practice pace,
    qualifying sector times, EWMA momentum trend, grid penalties,
    regulation-era season weights, weekend modes.

Research backing:
  - Random Forest:      Breiman (2001), ML 45(1)
  - XGBoost:            Chen & Guestrin (2016), KDD
  - LightGBM:           Ke et al. (2017), NeurIPS
  - Glicko-2:           Glickman (1999, 2012)
  - Ensemble weighting: Wolpert (1992), Neural Networks
  - LSTM + Attention:   Singh et al. (2024), JAAFR 26; Vaswani et al. (2017)
  - Tire degradation:   Kelly (2008) PhD Thesis, Leeds; Bekker & Lotz (2009)
"""

import warnings
import datetime
import json
import hashlib
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    import xgboost as xgb
    import lightgbm as lgb

from config import (
    DRIVER_TEAMS_2025, CURRENT_SEASON, HISTORICAL_SEASONS,
    TRACK_TYPE_ENC, OVERTAKING_ENC, POWER_UNIT_ENC, DOWNFORCE_ENC, CONDITION_ENC,
    RACE_POSITION_POINTS, QUALI_POSITION_POINTS,
    QUALI_Q2_BONUS, QUALI_Q3_BONUS, FASTEST_LAP_BONUS, DRIVER_OF_DAY_BONUS,
    POSITIONS_GAINED_PER, POSITIONS_LOST_PER, DNF_PENALTY, SPRINT_ROUNDS, AVG_PIT_STOP_TEAM_POINTS,
    CIRCUITS, SPRINT_RACE_POINTS, SC_PROBABILITY, VSC_PROBABILITY,
)
from data_fetcher import (
    get_season_results, get_qualifying_results, get_race_results,
    get_circuit_history, compute_driver_form, compute_constructor_reliability,
    get_driver_standings, get_constructor_standings, get_season_schedule,
    get_pitstop_data,
    compute_multiseason_driver_form, compute_multiseason_constructor_stats,
    compute_practice_pace, get_best_practice_pace, get_qualifying_sector_times, get_grid_penalties,
    get_actual_qualifying_results,
)
from elo_ratings import (
    Glicko2RatingSystem, ConstructorEloSystem, get_elo_system,
    EloRatingSystem, GLICKO2_MU as ELO_BASE,
    VETERAN_PHI_FLOOR,  # canonical phi floor constant
)

# Phase 3 ML models — imported lazily via singleton getters to avoid
# loading TensorFlow on every predictor import (slow; only needed at inference).
try:
    from temporal_model import get_temporal_model as _get_temporal_model
    from tire_model import get_tire_model as _get_tire_model
    _PHASE3_AVAILABLE = True
except ImportError:
    _PHASE3_AVAILABLE = False

try:
    from bayesian_model import get_bayesian_model as _get_bayesian_model
    _PHASE4_AVAILABLE = True
except ImportError:
    _PHASE4_AVAILABLE = False

# ─────────────────────────────────────────────
# HISTORICALLY WET RACE LOOKUP (P8 fix)
# Authoritative list of races where weather significantly affected results.
# Used in training to correctly label weather features instead of relying on
# the Jolpica "status" field which doesn't reliably encode weather conditions.
# Format: {(year, round): "wet" | "mixed"}
# ─────────────────────────────────────────────
_KNOWN_WET_RACES: dict[tuple[int, int], str] = {
    # 2019
    (2019, 11): "wet",    # German GP (Hockenheim) — heavy rain, multiple incidents
    (2019, 20): "mixed",  # Brazilian GP — rain mid-race
    # 2020
    (2020, 14): "wet",    # Turkish GP (Istanbul) — extremely wet, Perez win
    (2020, 13): "mixed",  # Emilia Romagna GP — wet/drying
    # 2021
    (2021, 12): "wet",    # Belgian GP (Spa) — red-flagged after 2 laps, half points
    (2021, 16): "wet",    # Turkish GP — wet all race
    (2021, 10): "mixed",  # British GP — rain at start
    # 2022
    (2022, 18): "wet",    # Japanese GP (Suzuka) — wet/half points controversy
    (2022, 5):  "mixed",  # Miami GP — dry but some rain
    # 2023
    (2023, 3):  "mixed",  # Australian GP — SC periods, light rain
    (2023, 9):  "mixed",  # Canadian GP — light rain
    (2023, 11): "mixed",  # British GP — mixed conditions
    (2023, 17): "wet",    # Japanese GP — rain at start
    # 2024
    (2024, 3):  "mixed",  # Australian GP — some rain during race
    (2024, 12): "wet",    # British GP — wet race, multiple incidents
    (2024, 21): "wet",    # Brazilian GP (São Paulo) — very wet sprint + race
    # 2025
    (2025, 1):  "mixed",  # Australian GP — some light rain
}

# ─────────────────────────────────────────────
# FEATURE NAMES (32 features — up from 24)
# ─────────────────────────────────────────────
FEATURE_NAMES = [
    # Championship strength
    "drv_champ_pos",       # championship position (1–22, lower = better)
    "drv_champ_pts",       # championship points accumulated
    "ctor_champ_pos",      # constructor championship position
    "ctor_champ_pts",      # constructor championship points
    # Glicko-2 strength signals
    "elo_rating",          # Glicko-2 rating (~1500 mean)
    "elo_zscore",          # std deviations above/below mean rating
    "elo_rank",            # rank among all current rated drivers
    "ctor_elo",            # constructor Glicko-2 rating
    "elo_rd",              # rating deviation (uncertainty)
    "elo_volatility",      # performance volatility
    "elo_confidence",      # 1 - (rd / max_rd)
    # Rolling form (EWMA last 5 races)
    "form_avg_pos",        # avg finish position
    "form_avg_pts",        # avg fantasy points
    "form_dnf_rate",       # DNF rate
    "form_score",          # composite form index
    "momentum_trend",      # EWMA slope (+ve = improving)
    # Circuit history (recency-weighted: >=2022=1.0, 2020-21=0.4, <2020=0.15)
    "circuit_avg_pos",     # weighted historical avg position at this circuit
    "circuit_dnf_rate",    # weighted historical DNF rate at this circuit
    # Constructor reliability
    "ctor_dnf_rate",       # blended DNF rate
    "ctor_pts_trend",      # YoY pts trend
    # Circuit characteristics
    "track_type_enc",
    "overtaking_enc",
    "power_unit_enc",
    "downforce_enc",
    # Weather
    "weather_enc",
    "rain_risk_enc",
    # Safety car / grid
    "sc_prob",
    "grid_penalty",
    # Relative strength
    "teammate_delta",
    "wins_this_season",
    # v3 features
    "qualifying_position",             # actual/predicted grid slot
    "team_form_delta",                  # constructor form trend
    "is_rookie",                        # 1 if < 4 races current-season data
    # Phase 3 ML features
    "lstm_momentum_pos",                # LSTM predicted finish position (temporal_model.py)
    "driver_tire_efficiency_score",     # tire degradation delta vs field median (tire_model.py)
    # Phase 4 ML features
    "lap_1_risk",                       # heuristic risk based on track and grid slot
    "fresh_tires_avail",                # remaining fresh tire sets going into race
    # Phase 5: Track-specific features (from per-circuit JSON)
    "circuit_length_km",               # track length in km (3.3-7.0 range)
    "num_turns",                       # number of corners (10-27)
    "circuit_altitude_m",              # circuit altitude in meters (0-2240)
    "longest_straight_m",              # longest straight in meters (500-2200)
    "track_width_m",                   # track width in meters (10-14)
    "sm_zones",                        # 2026 Straight Mode activation zones (1-4)
    "downforce_level",                 # numeric 1-5 (replaces string downforce_enc)
    "overtake_difficulty",             # numeric 1-5 (replaces string overtaking_enc)
    "tire_degradation_track",          # track tire deg level 1-5 (not driver-specific)
    "sc_prob_track",                   # circuit-specific SC probability 0-1
    "first_lap_incident_risk",         # probability of T1 incident 0-1
    "quali_importance",                # how much quali position matters 1-5
    "weather_variability",             # weather change risk 1-5
    "pit_time_loss_s",                 # pit stop time penalty in seconds (20-25)
    "deg_compound_delta",              # tire compound performance gap (0.1-0.5)
    # Phase 5: Derived track features
    "overtake_mode_efficiency",        # how effective Straight Mode is at this circuit (0-1)
    "track_power_sensitivity",         # derived: (1/downforce) * power_unit importance
    # Fantasy features
    "driver_overtake_delta",           # net overtake position delta
    "driver_dnf_risk",                 # combined driver and constructor DNF risk
]
N_FEATURES = len(FEATURE_NAMES)

WEATHER_ENC  = {"dry": 0, "overcast": 1, "mixed": 2, "wet": 3, "unknown": 1}
RAIN_RISK_ENC = {"low": 0, "medium": 1, "high": 2, "unknown": 1}

# ─────────────────────────────────────────────
# PHASE 4: LAP 1 RISK HEURISTIC
# ─────────────────────────────────────────────
_LAP_1_RISK_BY_TRACK = {
    "monza":      {"front": 0.05, "mid": 0.15, "rear": 0.08},
    "spa":        {"front": 0.05, "mid": 0.12, "rear": 0.06},
    "singapore":  {"front": 0.06, "mid": 0.15, "rear": 0.10},
    "monaco":     {"front": 0.03, "mid": 0.10, "rear": 0.12},
    "suzuka":     {"front": 0.04, "mid": 0.10, "rear": 0.08},
}
def _get_lap_1_risk(circuit_id: str, grid_pos: float) -> float:
    risk_profile = _LAP_1_RISK_BY_TRACK.get(circuit_id, {"front": 0.03, "mid": 0.08, "rear": 0.05})
    if grid_pos <= 6:
        return risk_profile["front"]
    elif grid_pos <= 14:
        return risk_profile["mid"]
    else:
        return risk_profile["rear"]


def _clamp(val, lo=0, hi=20):
    return max(lo, min(hi, val))


# ─────────────────────────────────────────────
# MAIN PREDICTOR CLASS
# ─────────────────────────────────────────────
class F1Predictor:
    """
    4-model ensemble predictor for F1 race and qualifying position.

    Models (trained on historical 2022–present data, regulation-era weighted):
      1. RandomForest       (Breiman 2001) — handles non-linear feature interactions
      2. XGBoost            (Chen & Guestrin 2016) — gradient-boosted trees
      3. LightGBM           (Ke et al. 2017) — leaf-wise GBDT, fast and accurate
      4. Ensemble           Weighted average (Wolpert 1992 stacking via 5-fold CV RMSE)

    Feature engineering (32 total):
      - Glicko-2 ratings  (Glickman 1999, 2012) — rating + RD + volatility
      - Multi-season form — exponential-decay + regulation-era weights
      - Circuit-specific  — per-circuit historical DNF and position data
      - SC probability    — safety car likelihood per circuit
      - FP2 long-run pace — practice pace delta vs field median
      - Weekend modes     — pre-quali, post-quali, race-day progressively add signals

    Weekend modes:
      pre-quali  → uses historical + practice pace (FP2)
      post-quali → adds qualifying sector times + grid penalties
      race-day   → refreshes weather, finalizes all signals
    """

    def __init__(self):
        # Models
        self.race_rf   = None
        self.race_xgb  = None
        self.race_lgb  = None
        self.quali_rf  = None
        self.quali_xgb = None
        self.quali_lgb = None

        # Scalers (RobustScaler — better than StandardScaler for F1 outliers)
        self.scaler_race  = RobustScaler()
        self.scaler_quali = RobustScaler()

        # Weights [rf, xgb, lgb] — set by CV
        self.race_meta_model  = None
        self.quali_meta_model = None

        # Context data
        self._trained            = False
        self._driver_standings   = []
        self._ctor_standings     = []
        self._driver_form        = {}
        self._ctor_reliability   = {}
        self._circuit_history    = pd.DataFrame()
        self._circuit_config     = {}
        self._weather            = {}
        self._elo: Optional[Glicko2RatingSystem] = None
        self._ctor_elo: dict[str, float]         = {}
        self._roster: dict[str, str]             = DRIVER_TEAMS_2025   # default
        self._practice_pace: dict[str, float]    = {}    # FP2 deltas
        self._quali_sectors: dict[str, dict]     = {}    # qualifying sectors
        self._grid_penalties: dict[str, int]     = {}    # grid penalties
        self._mode: str                          = "pre-quali"  # weekend mode
        # Phase 3: lazy-loaded ML model references
        self._temporal_model                     = None  # TemporalFormModel singleton
        self._tire_model                         = None        # Additional memory for Phase 3/4
        self._recent_race_results: list[dict] = []
        self._tire_data_by_round: dict[int, dict] = {}
        self._tire_allocs_by_round: dict[int, dict] = {}
        self._is_training: bool = False
        self._bias_corrections: dict             = {}
        self._weight_adjustments: dict           = {}

    # ─────────────────────────────────────────
    # DATA LOADING
    # ─────────────────────────────────────────
    def load_context(
        self,
        circuit_config: dict,
        weather: dict,
        roster: dict[str, str] = None,
        mode: str = "pre-quali",
        race_info: dict = None,
        grid_overrides: dict = None,
    ):
        """
        Load all context: standings, form, Glicko-2 ratings, constructor stats.

        mode: "pre-quali" | "post-quali" | "race-day"
          pre-quali  → historical + practice pace
          post-quali → + qualifying sector times + grid penalties
          race-day   → same as post-quali with refreshed weather
        """
        if roster:
            self._roster = roster
        self._circuit_config = circuit_config
        self._weather        = weather
        self._mode           = mode
        self._grid_overrides = grid_overrides or {}
        self._actual_grid    = []

        # Load self-improvement bias corrections and weight adjustments
        self._bias_corrections = {}
        self._weight_adjustments = {}
        try:
            bias_file = Path(__file__).parent / "logs" / "bias_corrections.json"
            if bias_file.exists():
                with open(bias_file, "r", encoding="utf-8") as f:
                    self._bias_corrections = json.load(f)
        except Exception as e:
            print(f"  [WARN] Failed to load bias corrections: {e}")

        try:
            adj_file = Path(__file__).parent / "logs" / "weight_adjustments.json"
            if adj_file.exists():
                with open(adj_file, "r", encoding="utf-8") as f:
                    self._weight_adjustments = json.load(f)
        except Exception as e:
            print(f"  [WARN] Failed to load weight adjustments: {e}")

        print("  [1/5] Loading championship standings...")
        self._driver_standings = get_driver_standings(CURRENT_SEASON)
        self._ctor_standings   = get_constructor_standings(CURRENT_SEASON)

        print("  [2/5] Building Glicko-2 ratings (2022–now, regulation-era weighted)...")
        self._elo = get_elo_system(sorted(HISTORICAL_SEASONS), verbose=True)

        # Constructor Glicko-2
        ctor_elo_sys = ConstructorEloSystem(self._elo)
        self._ctor_elo = ctor_elo_sys.compute_constructor_ratings(self._roster)

        print("  [3/5] Computing EWMA driver form + momentum trends...")
        current_form = compute_driver_form(CURRENT_SEASON, num_races=5)
        multi_form   = compute_multiseason_driver_form(CURRENT_SEASON)
        # current_form overrides multi_form for pace metrics (more recent = more relevant),
        # BUT we preserve the career race total from multi_form so rookie detection is correct.
        # A driver with 3 races in 2026 but 37 across their career is NOT a rookie.
        self._driver_form = {**multi_form, **current_form}
        for drv, mdata in multi_form.items():
            if drv in self._driver_form:
                career_races = mdata.get("races_counted", 0)
                current_races = current_form.get(drv, {}).get("races_counted", 0)
                self._driver_form[drv]["races_counted"] = max(career_races, current_races)

        print("  [4/5] Computing regulation-era constructor performance trends...")
        ms_ctor   = compute_multiseason_constructor_stats(CURRENT_SEASON)
        curr_ctor = compute_constructor_reliability(CURRENT_SEASON)
        self._ctor_reliability = {}
        for ctor in list(ms_ctor.keys()) + list(curr_ctor.keys()):
            ms  = ms_ctor.get(ctor, {})
            cur = curr_ctor.get(ctor, {})
            if cur and ms:
                self._ctor_reliability[ctor] = {
                    "dnf_rate":     round(0.65 * cur["dnf_rate"]     + 0.35 * ms["dnf_rate"], 3),
                    "avg_points":   round(0.65 * cur["avg_points"]   + 0.35 * ms["avg_points"], 2),
                    "avg_position": round(0.65 * cur["avg_position"] + 0.35 * ms["avg_position"], 2),
                    "pts_trend":    ms.get("pts_trend", 0.0),
                }
            elif cur:
                self._ctor_reliability[ctor] = cur
            else:
                self._ctor_reliability[ctor] = ms

        circuit_id = circuit_config.get("key", "")
        if circuit_id:
            print(f"  [5/5] Loading circuit history for '{circuit_id}'...")
            self._circuit_history = get_circuit_history(circuit_id, seasons=HISTORICAL_SEASONS)
        else:
            self._circuit_history = pd.DataFrame()
            print("  [5/5] No circuit key — skipping circuit-specific history.")

        # ── Practice pace (best available: FP2 -> FP1 -> FP3) ──
        if race_info:
            gp_name = race_info.get("name", "")
            year    = int(race_info.get("season", CURRENT_SEASON))
            print("  [+] Loading practice pace (trying FP2 -> FP1 -> FP3)...")
            self._practice_pace, self._practice_session_name = get_best_practice_pace(year, gp_name)
            if self._practice_pace:
                print(f"      Found {self._practice_session_name} pace data for {len(self._practice_pace)} drivers")
            else:
                self._practice_session_name = "N/A"
                print("      No practice session data available yet — using neutral pace")

        # ── Post-quali extras ──
        if mode in ("post-quali", "race-day") and race_info:
            gp_name     = race_info.get("name", "")
            round_num   = race_info.get("round", 0)
            year        = CURRENT_SEASON
            print("  [+] Loading qualifying sector times (post-quali mode)...")
            self._quali_sectors  = get_qualifying_sector_times(year, gp_name)
            print("  [+] Loading grid penalties...")
            self._grid_penalties = get_grid_penalties(year, round_num)
            if self._grid_penalties:
                penalty_str = ", ".join(f"{d}:{p:+d}" for d, p in self._grid_penalties.items())
                print(f"      Grid penalties detected: {penalty_str}")
            
            print("  [+] Loading actual qualifying results (post-quali mode)...")
            self._actual_grid = get_actual_qualifying_results(year, round_num)

        # ── Phase 3/4: Populate rolling result caches for LSTM/Tire models ──
        if race_info:
            print("  [+] Populating Phase 3/4 temporal caches...")
            from data_fetcher import get_season_schedule, get_race_results, get_tire_stints, get_weekend_tire_allocations
            try:
                sched = get_season_schedule(CURRENT_SEASON)
                rnd = race_info.get("round", 1)
                past_races = [r for r in sched if r["round"] < rnd][-5:]
                
                self._recent_race_results = []
                self._tire_data_by_round = getattr(self, "_tire_data_by_round", {})
                self._tire_allocs_by_round = getattr(self, "_tire_allocs_by_round", {})
                
                for pr in past_races:
                    pr_num = pr["round"]
                    res = get_race_results(CURRENT_SEASON, pr_num)
                    self._recent_race_results.append({
                        "round": pr_num,
                        "name": pr["name"],
                        "results": res
                    })
                    # Tire data (cached via get_tire_stints)
                    self._tire_data_by_round[pr_num] = get_tire_stints(
                        CURRENT_SEASON, pr["name"], "R"
                    )
                    
                # Phase 4: Fetch upcoming race tire allocations
                if rnd not in self._tire_allocs_by_round:
                    self._tire_allocs_by_round[rnd] = get_weekend_tire_allocations(
                        CURRENT_SEASON, race_info.get("name", "")
                    )
                    
                print(f"      Cached results for {len(self._recent_race_results)} past races.")
            except Exception as e:
                print(f"      [WARN] Phase 3/4 cache population failed: {e}")

    # ─────────────────────────────────────────
    # FEATURE EXTRACTION (32 features)
    # ─────────────────────────────────────────
    def _build_features(
        self, driver_name: str,
        drv_standings=None, ctor_standings=None,
        driver_form=None, ctor_reliability=None,
        ctor_elo=None, elo=None,
        circuit_hist=None, circuit_cfg=None, weather=None,
        roster=None, grid_penalties=None,
    ) -> np.ndarray:
        """Build the 30-feature vector for a single driver."""
        # Defaults to instance context
        drv_standings    = drv_standings   or self._driver_standings
        ctor_standings   = ctor_standings  or self._ctor_standings
        driver_form      = driver_form     or self._driver_form
        ctor_reliability = ctor_reliability or self._ctor_reliability
        ctor_elo         = ctor_elo        if ctor_elo is not None else self._ctor_elo
        elo              = elo             or self._elo
        circuit_hist     = circuit_hist    if circuit_hist is not None else self._circuit_history
        circuit_cfg      = circuit_cfg     or self._circuit_config
        weather          = weather         or self._weather
        roster           = roster          or self._roster
        grid_penalties   = grid_penalties  if grid_penalties is not None else self._grid_penalties

        ctor_name = roster.get(driver_name, "")

        # — Championship —
        drv_entry  = next((d for d in drv_standings  if d["name"] == driver_name), None)
        ctor_entry = next((c for c in ctor_standings if c["constructor"] == ctor_name), None)

        drv_pos  = drv_entry["position"] if drv_entry else 15
        drv_pts  = drv_entry["points"]   if drv_entry else 0
        wins     = drv_entry.get("wins", 0) if drv_entry else 0
        ctor_pos = ctor_entry["position"] if ctor_entry else 6
        ctor_pts = ctor_entry["points"]   if ctor_entry else 0

        # — Glicko-2 features —
        elo_feats  = elo.get_elo_features(driver_name) if elo else {
            "elo_rating": ELO_BASE, "elo_zscore": 0.0, "elo_rank": 11,
            "elo_rd": 350.0, "elo_volatility": 0.06, "elo_confidence": 0.0,
        }
        elo_rating    = elo_feats["elo_rating"]
        elo_zscore    = elo_feats["elo_zscore"]
        elo_rank      = elo_feats["elo_rank"]
        elo_rd        = elo_feats.get("elo_rd", 350.0)
        elo_vol       = elo_feats.get("elo_volatility", 0.06)
        elo_conf      = elo_feats.get("elo_confidence", 0.0)
        c_elo         = ctor_elo.get(ctor_name, ELO_BASE) if ctor_elo else ELO_BASE

        # — Rolling form (EWMA) —
        form = driver_form.get(driver_name, {})
        form_avg_pos    = form.get("avg_position",   10.0)
        form_avg_pts    = form.get("avg_points",      5.0)
        form_dnf_rate   = form.get("dnf_rate",        0.08)
        form_score      = form.get("form_score",      0.0)
        momentum_trend  = form.get("momentum_trend",  0.0)   # NEW
        overtake_delta  = form.get("overtake_delta",  0.0)

        # — Constructor reliability —
        ctor_rel  = ctor_reliability.get(ctor_name, {})
        ctor_dnf  = ctor_rel.get("dnf_rate",   0.08)
        ctor_trend = ctor_rel.get("pts_trend",  0.0)
        driver_dnf_risk = (form_dnf_rate * 0.6) + (ctor_dnf * 0.4)

        # — Circuit history (recency-weighted) —
        # post-2022 regulation era = full weight; 2020-21 = 0.4; pre-2020 = 0.15
        # This prevents a driver's 2018 Monaco result from over-riding their current pace.
        n_circuit_races = 0
        if circuit_hist is not None and isinstance(circuit_hist, pd.DataFrame) \
                and not circuit_hist.empty and "name" in circuit_hist.columns:
            rows = circuit_hist[circuit_hist["name"] == driver_name]
            if not rows.empty:
                n_circuit_races = len(rows)
                if "year" in rows.columns:
                    def _yr_w(yr):
                        if yr >= 2022: return 1.0
                        elif yr >= 2020: return 0.4
                        else: return 0.15
                    w_vec = rows["year"].apply(_yr_w)
                    total_w = w_vec.sum()
                    if total_w > 0:
                        circ_avg = float((rows["position"] * w_vec).sum() / total_w)
                        circ_dnf = float((rows["dnf"].astype(float) * w_vec).sum() / total_w)
                    else:
                        circ_avg = float(rows["position"].mean())
                        circ_dnf = float(rows["dnf"].mean())
                else:
                    circ_avg = float(rows["position"].mean())
                    circ_dnf = float(rows["dnf"].mean())
            else:
                n_circuit_races = 0
                elo_pos_adjust = max(0, (ELO_BASE - elo_rating) / 100)
                circ_avg = _clamp(form_avg_pos + elo_pos_adjust, 1, 20)
                circ_dnf = form_dnf_rate
        else:
            elo_pos_adjust = max(0, (ELO_BASE - elo_rating) / 100)
            circ_avg = _clamp(form_avg_pos + elo_pos_adjust, 1, 20)
            circ_dnf = form_dnf_rate

        # — Rookie detection —
        # If driver has < 4 races of current-season data, use constructor form as baseline.
        races_counted = driver_form.get(driver_name, {}).get("races_counted", 0) if driver_form else 0
        is_rookie_flag = 1.0 if races_counted < 4 else 0.0
        if is_rookie_flag and n_circuit_races == 0:
            ctor_rel_inner = ctor_reliability.get(ctor_name, {}) if ctor_reliability else {}
            circ_avg = ctor_rel_inner.get("avg_position", form_avg_pos)
            circ_dnf = ctor_rel_inner.get("dnf_rate", form_dnf_rate)

        # — Form-dominates-history cap —
        # If current form is 4+ places better than circuit history, cap the drag to 2 places.
        if not is_rookie_flag and n_circuit_races > 0:
            if circ_avg - form_avg_pos > 4.0:
                circ_avg = form_avg_pos + 2.0

        # — Circuit encoding —
        track_enc = TRACK_TYPE_ENC.get(circuit_cfg.get("track_type", "permanent"), 2)
        over_enc  = OVERTAKING_ENC.get(circuit_cfg.get("overtaking",  "medium"), 2)
        power_enc = POWER_UNIT_ENC.get(circuit_cfg.get("power_unit",  "medium"), 2)
        df_enc    = DOWNFORCE_ENC.get(circuit_cfg.get("downforce",   "medium"), 2)

        # — Weather —
        weather_enc = WEATHER_ENC.get(weather.get("summary_condition", "dry"), 0)
        rain_enc    = RAIN_RISK_ENC.get(weather.get("rain_risk", "low"), 0)

        # — Safety car probability —
        circuit_key = circuit_cfg.get("key", "")
        sc_prob     = SC_PROBABILITY.get(circuit_key, 0.40)

        # — Grid penalty —
        grid_pen = grid_penalties.get(driver_name, 0) if grid_penalties else 0
        grid_pen = _clamp(grid_pen, 0, 20)

        # — Teammate delta —
        teammates = [d for d, t in roster.items() if t == ctor_name and d != driver_name]
        tm_delta = 0.0
        for tm in teammates:
            tm_entry = next((d for d in drv_standings if d["name"] == tm), None)
            if tm_entry:
                tm_delta = float(drv_pts - tm_entry["points"])
            break

        # — NEW: qualifying position proxy —
        # post-quali mode: real grid pos from quali_sectors; pre-quali: form proxy
        quali_sectors_self = getattr(self, "_quali_sectors", {})
        if quali_sectors_self and driver_name in quali_sectors_self:
            # Rank by q_lap_delta (lower delta = better); approximates grid position
            sorted_q = sorted(quali_sectors_self.items(), key=lambda x: x[1].get("q_lap_delta", 999))
            quali_pos_val = float(next((i + 1 for i, (n, _) in enumerate(sorted_q) if n == driver_name), form_avg_pos))
        else:
            quali_pos_val = _clamp(form_avg_pos, 1, 22)  # pre-quali proxy

        # — team form delta —
        ctor_rel_outer = ctor_reliability.get(ctor_name, {}) if ctor_reliability else {}
        team_form_delta = -ctor_rel_outer.get("pts_trend", 0.0) * 0.05
        team_form_delta = max(-3.0, min(3.0, team_form_delta))

        # ── Phase 3: LSTM temporal momentum ──────────────────────────────────
        # Lazy-load the TemporalFormModel singleton on first call.
        # Returns a predicted finish position (1–22); lower = better.
        # Falls back to form_avg_pos if model is unavailable.
        # NOTE: Skipped during training (_is_training=True) — no race context available.
        lstm_momentum_pos = form_avg_pos  # neutral fallback
        if _PHASE3_AVAILABLE and not getattr(self, '_is_training', False):
            try:
                if self._temporal_model is None:
                    self._temporal_model = _get_temporal_model(verbose=False)
                lstm_momentum_pos = self._temporal_model.predict_driver_momentum(
                    driver_name,
                    self._recent_race_results,
                    self._tire_data_by_round,
                    context_vec=None,   # predictor enriches at load_context time
                )
            except Exception:
                pass
        lstm_momentum_pos = _clamp(lstm_momentum_pos, 1, 22)

        # ── Phase 3: Tire efficiency score ───────────────────────────────────
        # Predicted deg_per_lap for the expected primary compound at this circuit.
        # Compared against field median to produce a delta (negative = efficient tyre use).
        # Lower deg_per_lap = more tire-efficient driver = better long-stint pace.
        # NOTE: Skipped during training (_is_training=True) — no race context available.
        tire_efficiency_score = 0.0   # neutral fallback
        if _PHASE3_AVAILABLE and not getattr(self, '_is_training', False):
            try:
                if self._tire_model is None:
                    self._tire_model = _get_tire_model(verbose=False)
                # Primary compound heuristic: soft for low-downforce circuits, medium otherwise
                primary_compound = "SOFT" if over_enc >= 3 else "MEDIUM"
                driver_deg = self._tire_model.predict_deg_per_lap(
                    primary_compound,
                    stint_length=20.0,
                    track_temp=35.0,
                    track_type=circuit_cfg.get("track_type", "permanent"),
                )
                # Score = -deg (lower degradation = positive score = better)
                # Clamped to [-0.1, 0.1] seconds/lap range
                tire_efficiency_score = float(np.clip(-driver_deg, -0.1, 0.1))
            except Exception:
                pass

        # Apply self-improvement weight adjustments to features if present
        track_type = circuit_cfg.get("track_type", "permanent")
        if hasattr(self, "_weight_adjustments") and self._weight_adjustments:
            adj = self._weight_adjustments.get("adjustments", {}).get(track_type, {})
            sc_nudge = adj.get("sc_prob", 0.0)
            form_nudge = adj.get("form_score", 0.0)
            sc_prob = max(0.0, min(1.0, sc_prob + sc_nudge))
            form_score = form_score * (1.0 + form_nudge)

        # ── Phase 4: Fantasy Features ───────────────────────────────────────
        lap_1_risk = _get_lap_1_risk(circuit_key, quali_pos_val)
        
        fresh_tires_avail = 13.0
        if not getattr(self, '_is_training', False):
            # We must figure out the current round
            current_round = self._circuit_config.get("round", 0)
            allocs = getattr(self, "_tire_allocs_by_round", {}).get(current_round, {})
            if driver_name in allocs:
                fresh_tires_avail = float(allocs[driver_name].get("total_fresh", 13.0))

        # ── Phase 5: Track-specific features from JSON ────────────────────────
        try:
            from track_features_loader import load_track_features as _ltf
            _track_data = _ltf(circuit_key) or {}
        except Exception:
            _track_data = {}
        circuit_length_km   = float(_track_data.get("circuit_length_km",    5.0))
        num_turns           = int(_track_data.get("num_turns",              15))
        circuit_altitude_m  = float(_track_data.get("altitude_m",           0))
        longest_straight_m  = float(_track_data.get("longest_straight_m",  900))
        track_width_m       = float(_track_data.get("track_width_m",        12))
        sm_zones            = int(_track_data.get("sm_zones",               2))
        downforce_level     = int(_track_data.get("downforce_level",        3))
        overtake_difficulty = int(_track_data.get("overtaking_difficulty",  3))
        tire_deg_track      = int(_track_data.get("tire_degradation",       3))
        sc_prob_track       = float(_track_data.get("sc_probability",       0.40))
        first_lap_risk      = float(_track_data.get("first_lap_incident_risk", 0.07))
        quali_importance    = int(_track_data.get("quali_importance",       3))
        weather_variability = int(_track_data.get("weather_variability",    3))
        pit_time_loss_s     = float(_track_data.get("pit_time_loss_s",      22.0))
        deg_compound_delta  = float(_track_data.get("deg_compound_delta",   0.3))
        overtake_mode_eff   = float(_track_data.get("overtake_mode_efficiency", 0.5))
        # Derived: track power sensitivity
        # High downforce + low power_unit importance = low power sensitivity
        # Low downforce + high power_unit importance = high power sensitivity
        _power_enc_val = POWER_UNIT_ENC.get(circuit_cfg.get("power_unit", "medium"), 2)
        track_power_sensitivity = (6 - downforce_level) * _power_enc_val / 10.0
        track_power_sensitivity = max(0.0, min(2.0, track_power_sensitivity))

        return np.array([
            _clamp(drv_pos,  1, 22),
            drv_pts,
            _clamp(ctor_pos, 1, 11),
            ctor_pts,
            elo_rating,
            elo_zscore,
            _clamp(elo_rank, 1, 22),
            c_elo,
            max(VETERAN_PHI_FLOOR, min(350.0, elo_rd)),
            max(0.0, min(0.15, elo_vol)),
            max(0.0, min(1.0, elo_conf)),
            _clamp(form_avg_pos, 1, 22),
            form_avg_pts,
            form_dnf_rate,
            form_score,
            max(-3.0, min(3.0, momentum_trend)),
            _clamp(circ_avg, 1, 22),
            circ_dnf,
            ctor_dnf,
            ctor_trend,
            track_enc,
            over_enc,
            power_enc,
            df_enc,
            weather_enc,
            rain_enc,
            sc_prob,
            float(grid_pen),
            tm_delta,
            wins,
            _clamp(quali_pos_val, 1, 22),   # qualifying_position
            team_form_delta,                 # team_form_delta
            is_rookie_flag,                  # is_rookie
            float(lstm_momentum_pos),        # lstm_momentum_pos     (Phase 3)
            float(tire_efficiency_score),    # driver_tire_efficiency_score (Phase 3)
            float(lap_1_risk),               # lap_1_risk            (Phase 4)
            float(fresh_tires_avail),        # fresh_tires_avail     (Phase 4)
            # Phase 5: Track-specific features
            float(circuit_length_km),        # circuit_length_km
            float(num_turns),                # num_turns
            float(circuit_altitude_m),       # circuit_altitude_m
            float(longest_straight_m),       # longest_straight_m
            float(track_width_m),            # track_width_m
            float(sm_zones),                 # sm_zones
            float(downforce_level),          # downforce_level
            float(overtake_difficulty),      # overtake_difficulty
            float(tire_deg_track),           # tire_degradation_track
            float(sc_prob_track),            # sc_prob_track
            float(first_lap_risk),           # first_lap_incident_risk
            float(quali_importance),         # quali_importance
            float(weather_variability),      # weather_variability
            float(pit_time_loss_s),          # pit_time_loss_s
            float(deg_compound_delta),       # deg_compound_delta
            float(overtake_mode_eff),        # overtake_mode_efficiency
            float(track_power_sensitivity),  # track_power_sensitivity
            float(overtake_delta),           # driver_overtake_delta
            float(driver_dnf_risk),          # driver_dnf_risk
        ], dtype=float)

    # ─────────────────────────────────────────
    # TRAINING (with disk cache)
    # ─────────────────────────────────────────
    _MODEL_CACHE_DIR = Path(__file__).parent / "cache" / "models"

    def _model_cache_key(self) -> str:
        """Hash of training seasons and feature count only.

        The ML training loop explicitly skips CURRENT_SEASON (2026), so completed
        race rounds never change the training data or model weights.  Only two things
        genuinely invalidate the model:
          1. A new historical season is added to HISTORICAL_SEASONS (e.g. end-of-year)
          2. N_FEATURES changes (new features added to the ensemble)
        Removing the round-count suffix stops the costly 5-minute full retrain that
        was being triggered after every race weekend for no accuracy benefit.
        """
        seasons_str = (
            ",".join(str(y) for y in sorted(HISTORICAL_SEASONS))
            + f"_features_{N_FEATURES}"
        )
        return hashlib.sha1(seasons_str.encode()).hexdigest()[:12]

    def _try_load_model_cache(self, verbose: bool) -> bool:
        """Return True if a valid cached model was loaded."""
        self._MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = self._MODEL_CACHE_DIR / f"ensemble_{self._model_cache_key()}.pkl"
        if not cache_file.exists():
            return False
        try:
            with open(cache_file, "rb") as f:
                state = pickle.load(f)
            # Restore all model components
            self.race_rf       = state["race_rf"]
            self.race_xgb      = state["race_xgb"]
            self.race_lgb      = state["race_lgb"]
            self.quali_rf      = state["quali_rf"]
            self.quali_xgb     = state["quali_xgb"]
            self.quali_lgb     = state["quali_lgb"]
            self.scaler_race   = state["scaler_race"]
            self.scaler_quali  = state["scaler_quali"]
            self.race_meta_model  = state.get("race_meta_model", None)
            self.quali_meta_model = state.get("quali_meta_model", None)
            self._trained      = True
            if verbose:
                print(f"  [cache hit] Loaded trained model (seasons {','.join(str(y) for y in sorted(HISTORICAL_SEASONS))})")
            return True
        except Exception as e:
            if verbose:
                print(f"  [cache] Could not load model cache ({e}), retraining...")
            return False

    def _save_model_cache(self) -> None:
        self._MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = self._MODEL_CACHE_DIR / f"ensemble_{self._model_cache_key()}.pkl"
        state = {
            "race_rf":      self.race_rf,
            "race_xgb":     self.race_xgb,
            "race_lgb":     self.race_lgb,
            "quali_rf":     self.quali_rf,
            "quali_xgb":    self.quali_xgb,
            "quali_lgb":    self.quali_lgb,
            "scaler_race":  self.scaler_race,
            "scaler_quali": self.scaler_quali,
            "race_meta_model": self.race_meta_model,
            "quali_meta_model":self.quali_meta_model,
            "trained_at":   datetime.datetime.now().isoformat(),
            "seasons":      sorted(HISTORICAL_SEASONS),
        }
        with open(cache_file, "wb") as f:
            pickle.dump(state, f, protocol=5)

    def train(self, verbose: bool = True, force: bool = False, progress_callback = None):
        """
        Train 3 models on all historical completed race data.
        Uses regulation-era weighted seasons (2019–present, heavily discounting pre-2022).
        Results are cached to disk — subsequent runs load instantly, no API calls needed.

        force=True skips the cache and retrains from scratch (same as --refresh).
        """
        if progress_callback:
            progress_callback("Training stage initialized...")
        # ── Cache hit fast-path ──
        if not force and self._try_load_model_cache(verbose):
            if progress_callback:
                progress_callback("Ensemble model loaded from cache.")
            return

        # Signal to _build_features: skip Phase 3 LSTM/tire inference during training
        self._is_training = True

        if verbose:
            seasons_used = [y for y in HISTORICAL_SEASONS if y < CURRENT_SEASON]
            print(f"  Building training dataset (regulation-era weighted, {len(seasons_used)} seasons)...")

        X_race, y_race   = [], []
        X_quali, y_quali = [], []
        race_groups, quali_groups = [], []

        try:
            from backtest import DRIVER_TEAMS_BY_YEAR
        except ImportError:
            DRIVER_TEAMS_BY_YEAR = {}

        for year in HISTORICAL_SEASONS:
            if year >= CURRENT_SEASON:
                continue
            try:
                schedule = get_season_schedule(year)
                # Build year-specific Glicko-2 from data BEFORE this season
                year_elo = Glicko2RatingSystem()
                year_elo.build_from_history([y for y in HISTORICAL_SEASONS if y < year], verbose=False)
                roster = DRIVER_TEAMS_BY_YEAR.get(year, DRIVER_TEAMS_2025)
                ctor_elo_sys  = ConstructorEloSystem(year_elo)

                multi_form = compute_multiseason_driver_form(year, [year-1, year-2])
                ctor_rel   = compute_multiseason_constructor_stats(year, [year-1, year-2])

                # Pre-fetch all season results in one call (1 API hit vs N×5).
                # Used below to warm the per-round cache so compute_driver_form
                # hits only disk on subsequent training runs.
                try:
                    _bulk = get_season_results(year)  # permanently cached for historical years
                except Exception:
                    _bulk = []

                for race in schedule:
                    rnd  = race["round"]
                    name = race["name"]
                    try:
                        # Dynamic features to prevent dataset lookahead bias
                        drv_standings  = get_driver_standings(year, max(1, rnd - 1))
                        ctor_standings = get_constructor_standings(year, max(1, rnd - 1))
                        curr_form      = compute_driver_form(year, num_races=5, until_round=rnd)
                        drv_form       = {**multi_form, **curr_form}

                        year_ctor_elo = ctor_elo_sys.compute_constructor_ratings(roster)

                        race_results  = get_race_results(year, rnd)
                        quali_results = get_qualifying_results(year, rnd)
                        if not race_results:
                            continue

                        circuit_cfg = _match_circuit_cfg(name)
                        circuit_id  = circuit_cfg.get("key", "")
                        circ_hist   = get_circuit_history(circuit_id, [year-1, year-2]) if circuit_id else pd.DataFrame()

                        # Get grid penalties for this historical race
                        hist_grid_pen = get_grid_penalties(year, rnd)

                        # Use authoritative wet-race lookup instead of the unreliable
                        # Jolpica status field (which records "Finished"/"Accident" etc.,
                        # not weather). Falls back to "dry" for unlisted races.
                        _wet_label = _KNOWN_WET_RACES.get((year, rnd), "dry")
                        weather_hist = {
                            "summary_condition": _wet_label,   # "wet" | "mixed" | "dry"
                            "rain_risk": "high" if _wet_label == "wet" else "medium" if _wet_label == "mixed" else "low",
                        }

                        for r in race_results:
                            drv = r["name"]
                            feat = self._build_features(
                                drv,
                                drv_standings=drv_standings, ctor_standings=ctor_standings,
                                driver_form=drv_form, ctor_reliability=ctor_rel,
                                ctor_elo=year_ctor_elo, elo=year_elo,
                                circuit_hist=circ_hist, circuit_cfg=circuit_cfg,
                                weather=weather_hist,
                                roster=roster,
                                grid_penalties=hist_grid_pen,
                            )
                            X_race.append(feat)
                            y_race.append(float(r["position"]))

                        for q in (quali_results or []):
                            drv = q["name"]
                            feat = self._build_features(
                                drv,
                                drv_standings=drv_standings, ctor_standings=ctor_standings,
                                driver_form=drv_form, ctor_reliability=ctor_rel,
                                ctor_elo=year_ctor_elo, elo=year_elo,
                                circuit_hist=circ_hist, circuit_cfg=circuit_cfg,
                                weather=weather_hist,
                                roster=roster,
                                grid_penalties=hist_grid_pen,
                            )
                            X_quali.append(feat)
                            y_quali.append(float(q["position"]))
                            
                        if race_results:
                            race_groups.append(len(race_results))
                        if quali_results:
                            quali_groups.append(len(quali_results))

                        # Update intra-season Elo to keep features fresh for next race
                        if race_results:
                            year_elo.update(race_results, date_str=race.get("date", ""))

                    except Exception:
                        continue
            except Exception:
                continue

        if verbose:
            print(f"  Dataset: {len(X_race)} race samples, {len(X_quali)} qualifying samples")
        if progress_callback:
            progress_callback(f"Dataset ready: {len(X_race)} race samples, {len(X_quali)} qualifying samples")

        # ── Train race models ──
        if len(X_race) >= 20:
            Xr = self.scaler_race.fit_transform(np.array(X_race))
            yr = np.array(y_race)

            self.race_rf = RandomForestRegressor(
                n_estimators=500, max_depth=10, min_samples_leaf=2,
                max_features="sqrt", random_state=42, n_jobs=-1,
            )
            self.race_xgb = xgb.XGBRanker(
                n_estimators=400, max_depth=5, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, random_state=42, verbosity=0,
            )
            self.race_lgb = lgb.LGBMRanker(
                n_estimators=400, num_leaves=31, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, random_state=42, verbose=-1,
            )
            if progress_callback:
                progress_callback("Fitting Random Forest race model...")
            self.race_rf.fit(Xr, yr)
            
            # Rankers optimize for higher values, so invert the target
            yr_inv = 25.0 - yr
            
            if progress_callback:
                progress_callback("Fitting XGBoost race model...")
            self.race_xgb.fit(Xr, yr_inv, group=race_groups)
            
            if progress_callback:
                progress_callback("Fitting LightGBM race model...")
            self.race_lgb.fit(Xr, yr_inv, group=race_groups)

            if progress_callback:
                progress_callback("Computing ensemble weights via TimeSeriesSplit OOF...")
            self.race_meta_model = _train_meta_learner(
                Xr, yr, race_groups, self.race_rf, self.race_xgb, self.race_lgb
            )
            if progress_callback:
                progress_callback(f"Race model ready. Meta-learner trained.")
            if verbose:
                print(f"  Race meta-learner trained.")

        # ── Train qualifying models ──
        if len(X_quali) >= 20:
            Xq = self.scaler_quali.fit_transform(np.array(X_quali))
            yq = np.array(y_quali)

            self.quali_rf = RandomForestRegressor(
                n_estimators=500, max_depth=10, min_samples_leaf=2,
                max_features="sqrt", random_state=42, n_jobs=-1,
            )
            self.quali_xgb = xgb.XGBRanker(
                n_estimators=400, max_depth=5, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
            )
            self.quali_lgb = lgb.LGBMRanker(
                n_estimators=400, num_leaves=31, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
            )
            if progress_callback:
                progress_callback("Fitting Random Forest qualifying model...")
            self.quali_rf.fit(Xq, yq)
            
            yq_inv = 25.0 - yq
            
            if progress_callback:
                progress_callback("Fitting XGBoost qualifying model...")
            self.quali_xgb.fit(Xq, yq_inv, group=quali_groups)
            
            if progress_callback:
                progress_callback("Fitting LightGBM qualifying model...")
            self.quali_lgb.fit(Xq, yq_inv, group=quali_groups)

            if progress_callback:
                progress_callback("Computing qualifying ensemble weights via TimeSeriesSplit OOF...")
            self.quali_meta_model = _train_meta_learner(
                Xq, yq, quali_groups, self.quali_rf, self.quali_xgb, self.quali_lgb
            )
            if verbose:
                print(f"  Quali meta-learner trained.")

        self._trained = (self.race_rf is not None)
        # Re-enable Phase 3 inference now that training is complete
        self._is_training = False
        if self._trained:
            self._save_model_cache()
            
        if verbose:
            print(f"  Training complete. Trained={self._trained}  Features={N_FEATURES}")

        if self._elo is None:
            self._elo = get_elo_system(sorted(HISTORICAL_SEASONS), verbose=False)

    # ─────────────────────────────────────────
    # INFERENCE
    # ─────────────────────────────────────────
    def _predict_position(self, driver_name: str, mode: str = "race") -> tuple[float, float]:
        """
        Returns (predicted_position, std_dev) for race or qualifying.
        ALWAYS uses Glicko-2 + ML ensemble — no random guessing.
        """
        feat = self._build_features(driver_name)

        if self._trained and self.race_rf is not None and mode == "race":
            fs = self.scaler_race.transform(feat.reshape(1, -1))
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pr_rf  = self.race_rf.predict(fs)[0]
                pr_xgb = self.race_xgb.predict(fs)[0]
                pr_lgb = self.race_lgb.predict(fs)[0]
                
            if self.race_meta_model is not None:
                # Re-invert ranker scores to position scale (25 - score = predicted position)
                pred = self.race_meta_model.predict([[pr_rf, 25.0 - pr_xgb, 25.0 - pr_lgb]])[0]
            else:
                pred = (pr_rf + (25-pr_xgb) + (25-pr_lgb)) / 3.0
                
            std  = float(np.std([pr_rf, 25-pr_xgb, 25-pr_lgb]))

        elif self._trained and self.quali_rf is not None and mode == "quali":
            fs = self.scaler_quali.transform(feat.reshape(1, -1))
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pq_rf  = self.quali_rf.predict(fs)[0]
                pq_xgb = self.quali_xgb.predict(fs)[0]
                pq_lgb = self.quali_lgb.predict(fs)[0]
                
            if self.quali_meta_model is not None:
                # Re-invert ranker scores to position scale before passing to meta-learner
                pred = self.quali_meta_model.predict([[pq_rf, 25.0 - pq_xgb, 25.0 - pq_lgb]])[0]
            else:
                pred = (pq_rf + (25-pq_xgb) + (25-pq_lgb)) / 3.0
                
            std  = float(np.std([pq_rf, 25-pq_xgb, 25-pq_lgb]))

        else:
            # Glicko-2 fallback (pure ranking — no randomness)
            if self._elo:
                ranked = self._elo.get_ranked_drivers()
                driver_names = [n for n, _ in ranked]
                if driver_name in driver_names:
                    pred = float(driver_names.index(driver_name) + 1)
                else:
                    pred = 11.0
            else:
                drv_entry = next((d for d in self._driver_standings if d["name"] == driver_name), None)
                pred = float(drv_entry["position"]) if drv_entry else 11.0

            # RD-based uncertainty — high RD → higher std (appropriate for rookies)
            rd   = self._elo.get_rd(driver_name) if self._elo else 350.0
            std  = 1.5 + 3.0 * (rd / 350.0)  # ranges 1.5–4.5

        return float(pred), std

    def _generate_explanation(self, driver_name: str, pred_pos: float, confidence: float, is_rookie: bool) -> str:
        """Return a short human-readable explanation of why this driver was ranked here."""
        ctor = self._roster.get(driver_name, "")
        form = self._driver_form.get(driver_name, {})
        form_avg = form.get("avg_position", 11.0)
        races = form.get("races_counted", 0)
        momentum = form.get("momentum_trend", 0.0)
        ctor_pos = next((c["position"] for c in self._ctor_standings if c["constructor"] == ctor), 99)

        parts = []
        if form_avg <= 4.5:
            parts.append(f"excellent form (avg P{form_avg:.1f} last {races} races)")
        elif form_avg <= 8:
            parts.append(f"solid form (avg P{form_avg:.1f})")
        elif form_avg >= 14:
            parts.append(f"poor form (avg P{form_avg:.1f})")
        else:
            parts.append(f"mid-pack form (avg P{form_avg:.1f})")

        if momentum > 0.8:
            parts.append("strong improving momentum")
        elif momentum < -0.8:
            parts.append("declining momentum")

        if ctor_pos <= 3:
            parts.append(f"{ctor} car ranked #{ctor_pos} (top team)")
        elif ctor_pos >= 8:
            parts.append(f"{ctor} car ranked #{ctor_pos} (backmarker)")

        if is_rookie:
            parts.append(f"rookie ({races} races data — low confidence, using constructor baseline)")

        conf_label = "high" if confidence >= 70 else "medium" if confidence >= 45 else "low"
        parts.append(f"confidence: {conf_label}")
        return f"Predicted P{pred_pos:.0f}: " + ", ".join(parts) + "."

    def predict_finishing_order(self) -> list[dict]:
        """Predict race finishing order for all drivers in current roster."""
        drivers = list(self._roster.keys())
        predictions = []

        for drv in drivers:
            pred, std = self._predict_position(drv, mode="race")

            # Apply FastF1 heuristics as deterministic modifiers
            fp_delta = self._practice_pace.get(drv, 0.0) if self._practice_pace else 0.0
            pred += fp_delta * 2.0
            if self._quali_sectors and drv in self._quali_sectors:
                pred += self._quali_sectors[drv].get("q_lap_delta", 0.0) * 1.5
            pred = max(1.0, min(22.0, pred))

            ctor = self._roster.get(drv, "")
            dnf_prob = self._estimate_dnf_prob(drv, ctor)
            rd = self._elo.get_rd(drv) if self._elo else 350.0
            confidence = max(20.0, min(99.0, 100.0 - std * 8.0))
            confidence *= max(0.7, 1.0 - (rd - 100) / 500.0)
            confidence = max(15.0, min(99.0, confidence))

            # — Sensibility guardrails —
            form = self._driver_form.get(drv, {})
            races_counted = form.get("races_counted", 0)
            form_avg = form.get("avg_position", 11.0)
            is_rookie = bool(races_counted < 4)
            ctor_pos_val = next((c["position"] for c in self._ctor_standings if c["constructor"] == ctor), 99)

            # Guardrail 1: Top-3 constructor + recent top-6 form => can't predict below P10
            if ctor_pos_val <= 3 and form_avg <= 6.5 and pred > 10.0:
                pred = min(pred, 10.0)

            # Guardrail 2: Rookie floor (REMOVED to allow fair weighting)

            # Guardrail 3: Last-place constructor — soft cap at P12 for backmarker teams
            if ctor_pos_val >= 9 and pred < 10.0 and not (form_avg <= 9):
                pred = max(pred, 10.0)

            pred = max(1.0, min(22.0, pred))

            # Apply self-improvement bias correction
            # Clamped to ±1.5 positions — the corrector is supplementary;
            # the ML model + form data should do the heavy lifting.
            clamped_bias = 0.0
            if hasattr(self, "_bias_corrections") and self._bias_corrections:
                circuit_type = self._circuit_config.get("track_type", "permanent")
                biases = self._bias_corrections.get("driver_biases", {}).get(drv, {})
                bias = biases.get(circuit_type, biases.get("overall", 0.0))

                # Phase 6: Per-circuit bias override
                circuit_key = self._circuit_config.get("key", "")
                circuit_biases = self._bias_corrections.get("driver_circuit_biases", {}).get(drv, {})
                if circuit_key and circuit_key in circuit_biases:
                    bias = circuit_biases[circuit_key]

                clamped_bias = max(-1.5, min(1.5, bias))
                pred = pred - clamped_bias
                pred = max(1.0, min(22.0, pred))

            explanation = self._generate_explanation(drv, pred, confidence, is_rookie)
            if abs(clamped_bias) > 0.01:
                explanation += f" [Self-Improvement Bias: {-clamped_bias:+.2f}]"

            predictions.append({
                "driver":         drv,
                "team":           ctor,
                "predicted_pos":  float(round(pred, 2)),
                "confidence_pct": float(round(confidence, 1)),
                "std_dev":        float(round(std, 2)),
                "dnf_prob_pct":   float(round(dnf_prob * 100, 1)),
                "elo_rating":     float(round(self._elo.get_rating(drv) if self._elo else ELO_BASE, 0)),
                "elo_rd":         float(round(rd, 0)),
                "momentum":       float(round(form.get("momentum_trend", 0.0), 2)),
                "fp_pace_delta":  float(round(self._practice_pace.get(drv, 0.0), 3)),
                "is_rookie":      bool(is_rookie),
                "explanation":    explanation,
            })

        predictions.sort(key=lambda x: x["predicted_pos"])
        for i, p in enumerate(predictions, 1):
            p["predicted_rank"] = i
        return predictions

    def predict_qualifying_order(self) -> list[dict]:
        """Predict qualifying grid order."""
        drivers = list(self._roster.keys())
        predictions = []

        for drv in drivers:
            pred, std = self._predict_position(drv, mode="quali")
            
            # Apply FastF1 Heuristics as deterministic modifiers
            fp_delta = self._practice_pace.get(drv, 0.0) if self._practice_pace else 0.0
            pred += (fp_delta * 2.0)
            q_delta = 0.0
            if self._quali_sectors and drv in self._quali_sectors:
                q_delta = self._quali_sectors[drv].get("q_lap_delta", 0.0)
            pred += (q_delta * 2.5)  # stronger weight in quali
            pred = max(1.0, min(22.0, pred))

            ctor = self._roster.get(drv, "")
            confidence = max(20.0, min(99.0, 100.0 - std * 8.0))

            predictions.append({
                "driver":         drv,
                "team":           ctor,
                "predicted_pos":  round(pred, 2),
                "confidence_pct": round(confidence, 1),
                "std_dev":        round(std, 2),
                "q3_likely":      pred <= 10,
                "q2_likely":      pred <= 15,
                "elo_rating":     round(self._elo.get_rating(drv) if self._elo else ELO_BASE, 0),
                "fp_pace_delta":  round(self._practice_pace.get(drv, 0.0), 3),
                "q_lap_delta":    round(self._quali_sectors.get(drv, {}).get("q_lap_delta", 0.0), 3),
                "grid_penalty":   self._grid_penalties.get(drv, 0),
            })

        predictions.sort(key=lambda x: x["predicted_pos"])
        for i, p in enumerate(predictions, 1):
            p["predicted_grid"] = i
            p["is_actual"] = False

        if getattr(self, "_actual_grid", []) or getattr(self, "_grid_overrides", {}):
            locked_order = []
            overrides = getattr(self, "_grid_overrides", {})
            
            for ml_entry in predictions:
                drv = ml_entry["driver"]
                is_actual = False
                effective_grid = ml_entry["predicted_grid"]
                
                if getattr(self, "_actual_grid", []):
                    actual_entry = next((x for x in self._actual_grid if x["driver"] == drv), None)
                    if actual_entry:
                        effective_grid = actual_entry["effective_grid"]
                        ml_entry["grid_penalty"] = actual_entry.get("grid_penalty", 0)
                        ml_entry["q1_time"] = actual_entry.get("q1_time", "")
                        ml_entry["q2_time"] = actual_entry.get("q2_time", "")
                        ml_entry["q3_time"] = actual_entry.get("q3_time", "")
                        is_actual = True
                        
                if drv in overrides:
                    effective_grid = int(overrides[drv])
                    is_actual = True
                    
                locked_order.append({
                    **ml_entry,
                    "predicted_grid": effective_grid,
                    "predicted_pos": float(effective_grid),
                    "is_actual": is_actual
                })
                
            locked_order.sort(key=lambda x: x["predicted_grid"])
            
            # Ensure unique, contiguous grid positions
            for i, p in enumerate(locked_order, 1):
                p["predicted_grid"] = i
                p["predicted_pos"] = float(i)
                
            return locked_order

        return predictions

    def predict_sprint_order(self) -> list[dict]:
        """
        Sprint race prediction: qualifying model + race model blend.
        Sprint is shorter so compress variance somewhat.
        """
        drivers = list(self._roster.keys())
        predictions = []

        for drv in drivers:
            race_pred, race_std   = self._predict_position(drv, mode="race")
            quali_pred, quali_std = self._predict_position(drv, mode="quali")
            sprint_pred = 0.6 * race_pred + 0.4 * quali_pred
            sprint_std  = (race_std + quali_std) / 2

            ctor = self._roster.get(drv, "")
            dnf_prob = self._estimate_dnf_prob(drv, ctor) * 0.6
            confidence = max(20.0, min(99.0, 100.0 - sprint_std * 8.0))

            predictions.append({
                "driver":          drv,
                "team":            ctor,
                "predicted_pos":   float(round(sprint_pred, 2)),
                "sprint_grid":     float(round(quali_pred, 2)),
                "confidence_pct":  float(round(confidence, 1)),
                "std_dev":         float(round(sprint_std, 2)),
                "dnf_prob_pct":    float(round(dnf_prob * 100, 1)),
                "elo_rating":      float(round(self._elo.get_rating(drv) if self._elo else ELO_BASE, 0)),
            })

        predictions.sort(key=lambda x: x["predicted_pos"])
        for i, p in enumerate(predictions, 1):
            p["predicted_sprint_rank"] = i
        return predictions

    def _estimate_dnf_prob(self, driver_name: str, ctor_name: str) -> float:
        """DNF probability = blended driver form + constructor reliability + weather modifier."""
        form     = self._driver_form.get(driver_name, {})
        ctor_rel = self._ctor_reliability.get(ctor_name, {})
        drv_dnf  = form.get("dnf_rate", 0.07)
        ctor_dnf = ctor_rel.get("dnf_rate", 0.07)
        # Fix: weather dict uses 'summary_condition' not 'condition_enc'
        summary_cond = self._weather.get("summary_condition", "dry") if self._weather else "dry"
        weather_enc  = CONDITION_ENC.get(summary_cond, 0)
        weather_mod  = 1.0 + 0.4 * weather_enc / 3.0
        # SC probability also elevates DNF risk slightly
        circuit_key = self._circuit_config.get("key", "")
        sc_mod = 1.0 + 0.15 * SC_PROBABILITY.get(circuit_key, 0.40)
        return max(0.0, min(0.7, (0.55 * drv_dnf + 0.45 * ctor_dnf) * weather_mod * sc_mod))

    # ─────────────────────────────────────────
    # FANTASY POINTS ESTIMATION
    # ─────────────────────────────────────────
    def estimate_fantasy_points(
        self,
        driver_name: str,
        race_order: list[dict],
        quali_order: list[dict],
        is_sprint_weekend: bool = False,
        sprint_order: list[dict] = None,
    ) -> dict:
        """Estimate F1 Fantasy points for a driver for the full race weekend."""
        race_entry  = next((r for r in race_order  if r["driver"] == driver_name), None)
        quali_entry = next((q for q in quali_order if q["driver"] == driver_name), None)
        if not race_entry or not quali_entry:
            return {"driver": driver_name, "total_pts": 0.0, "breakdown": {}}

        race_pos  = race_entry["predicted_rank"]
        grid_pos  = quali_entry["predicted_grid"]
        dnf_prob  = race_entry["dnf_prob_pct"] / 100.0

        # Apply grid penalty to effective start position for "positions gained" calculation
        grid_penalty = self._grid_penalties.get(driver_name, 0)
        effective_grid = grid_pos + grid_penalty

        breakdown = {}

        # Qualifying points
        if grid_pos <= 10:
            quali_pts = QUALI_POSITION_POINTS.get(grid_pos, 0) + QUALI_Q3_BONUS + QUALI_Q2_BONUS
        elif grid_pos <= 15:
            quali_pts = float(QUALI_Q2_BONUS)
        else:
            quali_pts = 0.0
        breakdown["qualifying"] = round(quali_pts, 1)

        # Race position points
        rp = RACE_POSITION_POINTS.get(race_pos, 0)
        breakdown["race_position"] = round(rp, 1)

        # Positions gained/lost (from effective grid after penalty)
        delta = effective_grid - race_pos
        if delta > 0:
            # Gained places
            pos_pts = delta * POSITIONS_GAINED_PER
        else:
            # Lost places (delta is negative or zero)
            pos_pts = abs(delta) * POSITIONS_LOST_PER
        breakdown["positions_delta"] = round(pos_pts, 1)

        # DNF penalty (removed from base calculation to prevent double penalty)
        breakdown["dnf_risk"] = 0.0

        # Fastest lap probability
        fl_prob = max(0.0, (0.09 - (race_pos - 1) * 0.005)) * (1.0 - dnf_prob)
        breakdown["fastest_lap"] = round(FASTEST_LAP_BONUS * fl_prob, 2)

        # Driver of the Day
        dotd_prob = max(0.0, 0.12 - (race_pos - 1) * 0.01) * (1.0 - dnf_prob)
        breakdown["driver_of_day"] = round(DRIVER_OF_DAY_BONUS * dotd_prob, 2)

        # Sprint weekend
        if is_sprint_weekend and sprint_order:
            sprint_entry = next((s for s in sprint_order if s["driver"] == driver_name), None)
            if sprint_entry:
                sp_rank = sprint_entry["predicted_sprint_rank"]
                sprint_pts = SPRINT_RACE_POINTS.get(sp_rank, 0)
                breakdown["sprint"] = round(sprint_pts, 1)
            else:
                breakdown["sprint"] = 0.0
        else:
            breakdown["sprint"] = 0.0

        total = sum(breakdown.values())

        # ── Phase 4: Bayesian Point Projection ───────────────────────────────
        # Refines the deterministic total using a Zero-Inflated Negative Binomial
        # posterior predictive distribution.
        if _PHASE4_AVAILABLE:
            try:
                bm = _get_bayesian_model()
                form_entry = self._driver_form.get(driver_name, {})
                pts_history = form_entry.get("points_list", [])
                
                if pts_history:
                    # Deterministic total acts as the prior mean for the Bayesian fit
                    bayesian_ev = bm.fit_driver(driver_name, pts_history, ensemble_pred_pts=total)
                    total = bayesian_ev
            except Exception:
                pass

        return {
            "driver":    driver_name,
            "team":      race_entry["team"],
            "total_pts": round(total, 1),
            "breakdown": breakdown,
            "race_pos":  race_pos,
            "grid_pos":  grid_pos,
            "dnf_prob":  dnf_prob,
            "confidence": race_entry["confidence_pct"],
            "elo_rating": race_entry.get("elo_rating", ELO_BASE),
            "elo_rd":     race_entry.get("elo_rd", 350.0),
            "momentum":   race_entry.get("momentum", 0.0),
        }

    def estimate_constructor_points(
        self,
        ctor_name: str,
        driver_pts_all: list[dict],
        is_sprint: bool = False,
    ) -> dict:
        """Constructor points = combined driver points + pit stop estimate."""
        drv_entries = [d for d in driver_pts_all if d.get("team") == ctor_name]
        combined    = sum(d["total_pts"] for d in drv_entries)
        dotd_ded    = sum(d["breakdown"].get("driver_of_day", 0) for d in drv_entries)
        pit_pts     = AVG_PIT_STOP_TEAM_POINTS.get(ctor_name, 7)
        total       = (combined - dotd_ded) + pit_pts
        return {
            "constructor":   ctor_name,
            "total_pts":     round(total, 1),
            "driver_pts":    round(combined - dotd_ded, 1),
            "pit_stop_pts":  pit_pts,
            "drivers":       [d["driver"] for d in drv_entries],
            "ctor_elo":      round(self._ctor_elo.get(ctor_name, ELO_BASE), 0),
        }


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
# Legacy alias kept for any external references
VETERAN_PHI = VETERAN_PHI_FLOOR   # now sourced from elo_ratings


def _train_meta_learner(
    X: np.ndarray, y: np.ndarray, groups: list[int],
    model_rf, model_xgb, model_lgb,
    n_splits: int = 5,
):
    from sklearn.linear_model import Ridge
    from sklearn.base import clone
    import numpy as np

    oof_preds = np.zeros((len(y), 3))
    
    group_boundaries = [0] + list(np.cumsum(groups))
    total_groups = len(groups)
    
    test_size = total_groups // (n_splits + 1)
    if test_size == 0:
        test_size = 1
        
    for i in range(n_splits):
        test_start_group = total_groups - (n_splits - i) * test_size
        test_end_group = test_start_group + test_size if i < n_splits - 1 else total_groups
        train_end_group = test_start_group
        
        if train_end_group == 0:
            continue
            
        train_start_idx = 0
        train_end_idx = group_boundaries[train_end_group]
        test_start_idx = group_boundaries[test_start_group]
        test_end_idx = group_boundaries[test_end_group]
        
        X_train, y_train = X[train_start_idx:train_end_idx], y[train_start_idx:train_end_idx]
        X_test, y_test = X[test_start_idx:test_end_idx], y[test_start_idx:test_end_idx]
        
        train_groups = groups[:train_end_group]
        
        rf_clone = clone(model_rf)
        xgb_clone = clone(model_xgb)
        lgb_clone = clone(model_lgb)
        
        rf_clone.fit(X_train, y_train)
        y_train_inv = 25.0 - y_train
        xgb_clone.fit(X_train, y_train_inv, group=train_groups)
        lgb_clone.fit(X_train, y_train_inv, group=train_groups)
        
        oof_preds[test_start_idx:test_end_idx, 0] = rf_clone.predict(X_test)
        # Rankers output "higher = better" scores; invert back to position scale
        oof_preds[test_start_idx:test_end_idx, 1] = 25.0 - xgb_clone.predict(X_test)
        oof_preds[test_start_idx:test_end_idx, 2] = 25.0 - lgb_clone.predict(X_test)

    first_test_start = group_boundaries[total_groups - n_splits * test_size]
    if first_test_start < len(y):
        X_meta = oof_preds[first_test_start:]
        y_meta = y[first_test_start:]
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(X_meta, y_meta)
        return meta_model
    else:
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(oof_preds, y)
        return meta_model


def _match_circuit_cfg(race_name: str) -> dict:
    """Fuzzy-match race name to circuit config."""
    race_lower = race_name.lower()
    for name, cfg in CIRCUITS.items():
        words = name.lower().split()
        if any(w in race_lower for w in words if len(w) > 3):
            return {**cfg, "name": name}
    return {}
