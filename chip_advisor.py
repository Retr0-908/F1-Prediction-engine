"""
chip_advisor.py — F1 Fantasy Chip Strategy Recommendation Engine (v2)

New in v2:
  - Persistent chip inventory tracking (cache/chip_state.json)
  - Season-level look-ahead: shows all remaining races ranked by chip opportunity
  - ChipScore dataclass — data-driven scoring with expected gain estimate
  - Smart "hold until" advice: tells you WHICH future circuit is the better opportunity
  - Works in --auto mode (no prompts, just display)

Chips modelled:
  - Limitless     (remove budget cap for one race)
  - No Negative   (floor all negative scores at 0)
  - 3x Boost      (triple one driver's score — "Extra DRS")
  - Wildcard      (unlimited free transfers)
  - Final Fix     (one swap after qualifying)
  - Turbo Driver  (2x boost — standard weekly mechanic, not a one-use chip)

Research backing:
  - Chip timing strategy (motorsportmagazine.com, reddit.com F1 Fantasy community)
  - SC probability as chaos proxy for No Negative (axiorablogs.com)
  - Sprint weekend as optimal Limitless window (community consensus)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import (
    SC_PROBABILITY, CIRCUITS,
    CURRENT_SEASON,
)
from data_fetcher import get_sprint_rounds
from track_features_loader import get_track_feature

# ─────────────────────────────────────────────
# CHIP STATE PERSISTENCE
# ─────────────────────────────────────────────

_CHIP_STATE_PATH = Path(__file__).parent / "cache" / "chip_state.json"

ALL_CHIPS = ["Limitless", "No Negative", "3x Boost", "Wildcard", "Final Fix"]

_DEFAULT_STATE = {
    "used": {c: False for c in ALL_CHIPS},
    "used_on_round": {c: None for c in ALL_CHIPS},
    "used_on_circuit": {c: None for c in ALL_CHIPS},
    "season": CURRENT_SEASON,
}


def load_chip_state() -> dict:
    """Load persisted chip state. Returns default (all unused) if no file or wrong season."""
    _CHIP_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _CHIP_STATE_PATH.exists():
        try:
            with open(_CHIP_STATE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
            # Reset if season changed
            if state.get("season") != CURRENT_SEASON:
                return dict(_DEFAULT_STATE)
            return state
        except Exception:
            pass
    return dict(_DEFAULT_STATE)


def save_chip_state(state: dict) -> None:
    """Persist chip state to disk."""
    _CHIP_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["season"] = CURRENT_SEASON
    with open(_CHIP_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def mark_chip_used(chip: str, round_num: int, circuit_name: str) -> dict:
    """Load state, mark chip as used, save, and return updated state."""
    state = load_chip_state()
    state["used"][chip] = True
    state["used_on_round"][chip] = round_num
    state["used_on_circuit"][chip] = circuit_name
    save_chip_state(state)
    return state


def reset_chip_state() -> dict:
    """Reset all chips to unused (new season / manual override)."""
    state = dict(_DEFAULT_STATE)
    save_chip_state(state)
    return state


# ─────────────────────────────────────────────
# SEASON CALENDAR LOOK-AHEAD
# ─────────────────────────────────────────────

def _circuit_value_score(circuit_key: str, is_sprint: bool) -> float:
    """
    Score circuit by chip opportunity value (0–10).
    Formula: sc_prob * 3 + sprint_bonus * 4 + overtaking_score
    """
    sc_prob = get_track_feature(circuit_key, "sc_probability", SC_PROBABILITY.get(circuit_key, 0.40))
    sprint_bonus = 1.0 if is_sprint else 0.0

    # Derive overtaking score from CIRCUITS config
    circ_cfg = next((c for c in CIRCUITS.values() if c.get("key") == circuit_key), {})
    overtaking_map = {"very_low": 0, "low": 0.25, "medium": 0.5, "high": 0.75, "very_high": 1.0}
    overtaking_score = overtaking_map.get(circ_cfg.get("overtaking", "medium"), 0.5)

    return round(sc_prob * 3.0 + sprint_bonus * 4.0 + overtaking_score, 2)


def _season_urgency(current_round: int, total_rounds: int = 22) -> float:
    """
    Returns a score bonus (0.0–3.0) that escalates as the season runs out.
    This prevents chips from being held forever and wasted at season end.

    Urgency tiers:
      < 33% elapsed  → 0.0  (plenty of time, no pressure)
      33–55% elapsed → 0.5  (mild pressure)
      55–70% elapsed → 1.2  (moderate — over halfway and still holding)
      70–85% elapsed → 2.0  (high — few good windows left)
      > 85%  elapsed → 3.0  (critical — almost no races remaining)
    """
    pct = current_round / total_rounds
    if pct < 0.33:
        return 0.0
    elif pct < 0.55:
        return 0.5
    elif pct < 0.70:
        return 1.2
    elif pct < 0.85:
        return 2.0
    else:
        return 3.0


def get_remaining_sprint_rounds(current_round: int) -> list[dict]:
    """
    Returns a list of sprint rounds that haven't happened yet.
    [{round, circuit_name, circuit_key, sc_prob, value_score}]
    """
    sprint_rounds = get_sprint_rounds()
    remaining = []
    for circuit_name, cfg in CIRCUITS.items():
        rnd = cfg.get("round", 0)
        if rnd > current_round and rnd in sprint_rounds:
            key = cfg.get("key", "")
            remaining.append({
                "round":        rnd,
                "circuit_name": circuit_name,
                "circuit_key":  key,
                "sc_prob":      get_track_feature(key, "sc_probability", SC_PROBABILITY.get(key, 0.40)),
                "value_score":  _circuit_value_score(key, True),
            })
    return sorted(remaining, key=lambda x: x["round"])


def get_remaining_high_value_rounds(current_round: int, top_n: int = 5) -> list[dict]:
    """
    Returns the top N remaining rounds ranked by chip opportunity score.
    [{round, circuit_name, circuit_key, sc_prob, is_sprint, value_score, best_chip}]
    """
    sprint_rounds = get_sprint_rounds()
    remaining = []
    for circuit_name, cfg in CIRCUITS.items():
        rnd = cfg.get("round", 0)
        if rnd <= current_round:
            continue
        key = cfg.get("key", "")
        is_sprint = rnd in sprint_rounds
        sc_prob = get_track_feature(key, "sc_probability", SC_PROBABILITY.get(key, 0.40))
        score = _circuit_value_score(key, is_sprint)

        # Best chip suggestion based on circuit characteristics
        if is_sprint and score >= 7.0:
            best_chip = "Limitless + 3x Boost"
        elif is_sprint:
            best_chip = "Limitless"
        elif sc_prob >= 0.65:
            best_chip = "No Negative"
        elif sc_prob >= 0.50:
            best_chip = "No Negative / Wildcard"
        else:
            best_chip = "Hold"

        remaining.append({
            "round":        rnd,
            "circuit_name": circuit_name,
            "circuit_key":  key,
            "sc_prob":      sc_prob,
            "is_sprint":    is_sprint,
            "value_score":  score,
            "best_chip":    best_chip,
        })

    remaining.sort(key=lambda x: x["value_score"], reverse=True)
    return remaining[:top_n]


# ─────────────────────────────────────────────
# CHIP SCORE DATACLASS
# ─────────────────────────────────────────────

@dataclass
class ChipScore:
    chip: str
    raw_score: float          # 0–10 composite score
    use_now: bool             # True if this is a genuinely good weekend
    hold_until: Optional[str] = None   # Circuit name of better upcoming opportunity
    hold_until_round: Optional[int] = None
    reason: str = ""
    urgency: str = "low"      # "high" / "medium" / "low" / "none"
    expected_gain: float = 0.0  # Estimated additional pts from playing now vs not
    target_driver: Optional[str] = None  # For 3x Boost
    already_used: bool = False
    used_on: Optional[str] = None  # Circuit name where used


# ─────────────────────────────────────────────
# CHIP RECOMMENDATION ENGINE (v2)
# ─────────────────────────────────────────────

def advise_chips(
    race_info: dict,
    circuit_config: dict,
    weather: dict,
    driver_pts: list[dict],
    mc_results: dict = None,
    current_drivers: list[str] = None,
    free_transfers: int = 1,
    chip_state: dict = None,
) -> list[ChipScore]:
    """
    Produce a list of ChipScore recommendations for the current race weekend.

    Args:
        race_info:        get_next_race() output (must contain 'round' key)
        circuit_config:   circuit config dict
        weather:          weather dict
        driver_pts:       deterministic driver points projections
        mc_results:       Monte Carlo DistributionStats (optional but enriches advice)
        current_drivers:  user's current 5 drivers (for Final Fix / penalty detection)
        free_transfers:   number of free transfers remaining
        chip_state:       loaded from load_chip_state() — tracks which chips used

    Returns:
        List of ChipScore objects.
    """
    if chip_state is None:
        chip_state = load_chip_state()

    current_round = race_info.get("round", 1)
    circuit_key   = circuit_config.get("key", "")
    sc_prob       = get_track_feature(circuit_key, "sc_probability", SC_PROBABILITY.get(circuit_key, 0.40))
    rain_risk     = weather.get("rain_risk", "low")
    sprint_rounds = get_sprint_rounds()
    is_sprint     = current_round in sprint_rounds

    # Compute season urgency — escalates the later in the season we are
    urgency_bonus = _season_urgency(current_round)

    # Count unused chips — more chips still held = more pressure to deploy
    used_flags = chip_state.get("used", {})
    unused_count = sum(1 for c in ALL_CHIPS if not used_flags.get(c, False))

    # Sort predicted points descending
    sorted_pts = sorted(driver_pts, key=lambda d: d["total_pts"], reverse=True)
    top_pts    = sorted_pts[0]["total_pts"] if sorted_pts else 0
    top5_mean  = sum(d["total_pts"] for d in sorted_pts[:5]) / 5 if len(sorted_pts) >= 5 else 0
    best_driver = sorted_pts[0]["driver"] if sorted_pts else "Unknown"

    # Future opportunities
    remaining_sprints     = get_remaining_sprint_rounds(current_round)
    remaining_high_value  = get_remaining_high_value_rounds(current_round, top_n=3)
    next_sprint_str = None
    if remaining_sprints:
        ns = remaining_sprints[0]
        next_sprint_str = f"R{ns['round']} {ns['circuit_name']} (SC {ns['sc_prob']:.0%})"

    recommendations: list[ChipScore] = []

    # ── LIMITLESS ─────────────────────────────────────────────────────
    cs_limitless = _score_limitless(
        is_sprint, top5_mean, mc_results,
        remaining_sprints, next_sprint_str, chip_state, urgency_bonus,
        circuit_key=circuit_key
    )
    recommendations.append(cs_limitless)

    # ── NO NEGATIVE ───────────────────────────────────────────────────
    cs_no_neg = _score_no_negative(
        sc_prob, rain_risk, mc_results, circuit_config,
        remaining_high_value, chip_state, urgency_bonus
    )
    recommendations.append(cs_no_neg)

    # ── 3x BOOST (EXTRA DRS) ──────────────────────────────────────────
    cs_boost = _score_3x_boost(
        is_sprint, best_driver, top_pts, mc_results,
        remaining_sprints, next_sprint_str, chip_state, urgency_bonus
    )
    recommendations.append(cs_boost)

    # ── WILDCARD ──────────────────────────────────────────────────────
    cs_wildcard = _score_wildcard(
        free_transfers, top5_mean, current_round, chip_state, urgency_bonus
    )
    recommendations.append(cs_wildcard)

    # ── FINAL FIX ─────────────────────────────────────────────────────
    cs_final_fix = _score_final_fix(
        current_drivers, driver_pts, mc_results, chip_state, urgency_bonus
    )
    recommendations.append(cs_final_fix)

    return recommendations


def _mark_used(chip: str, chip_state: dict) -> ChipScore | None:
    """If chip is used, return a terminal ChipScore object."""
    if chip_state.get("used", {}).get(chip, False):
        used_on = chip_state.get("used_on_circuit", {}).get(chip)
        used_rnd = chip_state.get("used_on_round", {}).get(chip)
        label = f"R{used_rnd} {used_on}" if used_rnd and used_on else "earlier this season"
        return ChipScore(
            chip=chip,
            raw_score=0,
            use_now=False,
            reason=f"Already played at {label}",
            urgency="none",
            expected_gain=0.0,
            already_used=True,
            used_on=used_on,
        )
    return None


def _score_limitless(
    is_sprint: bool, top5_mean: float, mc_results: dict,
    remaining_sprints: list, next_sprint_str: Optional[str],
    chip_state: dict, urgency_bonus: float = 0.0,
    circuit_key: str = "",
) -> ChipScore:
    used = _mark_used("Limitless", chip_state)
    if used:
        return used

    score = 0.0
    reasons = []

    if is_sprint:
        score += 3.0
        reasons.append("Sprint weekend — extra scoring events")
        
    quali_imp = get_track_feature(circuit_key, "quali_importance", 3)
    if quali_imp >= 4:
        score += 1.5  # High quali importance = pole sitter gets big advantage

    if top5_mean >= 28:
        score += 2.5
        reasons.append(f"Very high EV weekend (top-5 avg {top5_mean:.0f} pts)")
    elif top5_mean >= 22:
        score += 1.5
        reasons.append(f"High EV weekend (top-5 avg {top5_mean:.0f} pts)")
    elif top5_mean >= 18:
        score += 0.5
        reasons.append(f"Moderate EV weekend (top-5 avg {top5_mean:.0f} pts)")

    if mc_results:
        top_ups = sorted(
            [(d, mc_results[d].p90_pts - mc_results[d].mean_pts) for d in mc_results],
            key=lambda x: x[1], reverse=True
        )[:3]
        if top_ups and top_ups[0][1] > 18:
            score += 2.0
            reasons.append(f"{top_ups[0][0]} has +{top_ups[0][1]:.0f} pts upside")
        elif top_ups and top_ups[0][1] > 12:
            score += 1.0

    # No more sprints left → significant urgency
    if not remaining_sprints:
        score += 3.0
        reasons.append("No more sprint weekends remaining — last chance")
    elif len(remaining_sprints) == 1:
        score += 0.5  # mild nudge when only one sprint left

    # Season urgency escalation
    score += urgency_bonus
    if urgency_bonus >= 1.2:
        reasons.append(f"Season urgency: {len(remaining_sprints)} sprint(s) remaining")

    # Lower threshold: 3.0 instead of 4.0 — a good weekend is enough
    use_now = score >= 3.0

    hold_until = None
    hold_round = None
    if not use_now and remaining_sprints:
        best_future = max(remaining_sprints, key=lambda x: x["value_score"])
        hold_until = f"{best_future['circuit_name']} (R{best_future['round']}, SC {best_future['sc_prob']:.0%})"
        hold_round = best_future["round"]

    # Estimate expected gain from Limitless: can upgrade to optimal team
    expected_gain = 22.0 if is_sprint else (16.0 if score >= 3 else 8.0)

    return ChipScore(
        chip="Limitless",
        raw_score=round(score, 1),
        use_now=use_now,
        hold_until=hold_until,
        hold_until_round=hold_round,
        reason=" · ".join(reasons) if reasons else "Low-value weekend — wait for a sprint",
        urgency="high" if score >= 5 else ("medium" if score >= 3 else "low"),
        expected_gain=expected_gain,
    )


def _score_no_negative(
    sc_prob: float, rain_risk: str, mc_results: dict,
    circuit_config: dict, remaining_high_value: list,
    chip_state: dict, urgency_bonus: float = 0.0,
) -> ChipScore:
    used = _mark_used("No Negative", chip_state)
    if used:
        return used

    score = 0.0
    reasons = []

    if sc_prob >= 0.65:
        score += 4.0
        reasons.append(f"High chaos circuit (SC prob {sc_prob:.0%})")
    elif sc_prob >= 0.50:
        score += 2.5
        reasons.append(f"Elevated SC probability ({sc_prob:.0%})")
    elif sc_prob >= 0.40:
        score += 1.0
        
    circuit_key = circuit_config.get("key", "")
    sm_eff = get_track_feature(circuit_key, "overtake_mode_efficiency", 0.5)
    if sm_eff >= 0.7:
        score += 1.0  # High SM efficiency = more overtaking = more variance = NN safer
        reasons.append(f"Average SC probability ({sc_prob:.0%})")

    track_type = circuit_config.get("track_type", "permanent")
    if track_type == "street":
        score += 1.5
        reasons.append("Street circuit — wall exits and tight racing")

    if rain_risk == "high":
        score += 3.0
        reasons.append("High rain risk — chaotic finishing order")
    elif rain_risk == "medium":
        score += 1.5
        reasons.append("Medium rain risk")

    if mc_results:
        high_dnf_drivers = [d for d, s in mc_results.items() if s.p_dnf > 15]
        if len(high_dnf_drivers) >= 4:
            score += 2.5
            reasons.append(f"{len(high_dnf_drivers)} drivers with >15% DNF risk")
        elif len(high_dnf_drivers) >= 2:
            score += 1.0

    # Season urgency: if we're deep in the season and still holding No Negative,
    # a moderately chaotic circuit is good enough
    score += urgency_bonus
    if urgency_bonus >= 1.2:
        reasons.append(f"Season urgency — use on a high-SC circuit before it's too late")

    # Lower threshold: 4.0 instead of 5.0
    use_now = score >= 4.0

    hold_until = None
    hold_round = None
    if not use_now and remaining_high_value:
        best_chaos = max(remaining_high_value, key=lambda x: x["sc_prob"])
        if best_chaos["sc_prob"] > sc_prob:
            hold_until = f"{best_chaos['circuit_name']} (R{best_chaos['round']}, SC {best_chaos['sc_prob']:.0%})"
            hold_round = best_chaos["round"]

    expected_gain = 14.0 if score >= 7 else (10.0 if score >= 4 else 4.0)

    return ChipScore(
        chip="No Negative",
        raw_score=round(score, 1),
        use_now=use_now,
        hold_until=hold_until,
        hold_until_round=hold_round,
        reason=" · ".join(reasons) if reasons else f"Low chaos (SC {sc_prob:.0%}, rain {rain_risk})",
        urgency="high" if score >= 6 else ("medium" if score >= 4 else "low"),
        expected_gain=expected_gain,
    )


def _score_3x_boost(
    is_sprint: bool, best_driver: str, best_pts_val: float,
    mc_results: dict, remaining_sprints: list,
    next_sprint_str: Optional[str], chip_state: dict, urgency_bonus: float = 0.0,
) -> ChipScore:
    used = _mark_used("3x Boost", chip_state)
    if used:
        return used

    score = 0.0
    reasons = []

    if is_sprint:
        score += 3.0
        reasons.append("Sprint weekend — more scoring events to triple")

    if mc_results and best_driver in mc_results:
        mc_best = mc_results[best_driver]
        if mc_best.p90_pts > 55:
            score += 3.0
            reasons.append(f"{best_driver} P90 ceiling: {mc_best.p90_pts:.0f} pts")
        elif mc_best.p90_pts > 42:
            score += 2.0
            reasons.append(f"{best_driver} P90 ceiling: {mc_best.p90_pts:.0f} pts")
        elif mc_best.p90_pts > 32:
            score += 1.0
        if mc_best.p_top3 > 70:
            score += 2.0
            reasons.append(f"{best_driver} has {mc_best.p_top3:.0f}% podium probability")
        elif mc_best.p_top3 > 55:
            score += 1.0

    if best_pts_val > 30:
        score += 2.0
        reasons.append(f"{best_driver} projected {best_pts_val:.0f} pts")
    elif best_pts_val > 22:
        score += 1.0
        reasons.append(f"{best_driver} projected {best_pts_val:.0f} pts")

    # Season urgency
    score += urgency_bonus
    if urgency_bonus >= 1.2:
        reasons.append("Season urgency — 3x Boost should not be held much longer")

    # Lower threshold: 3.5 instead of 5.0
    use_now = score >= 3.5

    hold_until = None
    hold_round = None
    if not use_now and remaining_sprints:
        best_future = max(remaining_sprints, key=lambda x: x["value_score"])
        hold_until = f"{best_future['circuit_name']} (R{best_future['round']}, SC {best_future['sc_prob']:.0%})"
        hold_round = best_future["round"]

    expected_gain = best_pts_val * 2.0 if use_now else 0.0

    return ChipScore(
        chip="3x Boost",
        raw_score=round(score, 1),
        use_now=use_now,
        hold_until=hold_until,
        hold_until_round=hold_round,
        reason=" · ".join(reasons) if reasons else "No dominant driver this weekend — wait for a sprint",
        urgency="high" if score >= 6 else ("medium" if score >= 3.5 else "low"),
        expected_gain=round(expected_gain, 1),
        target_driver=best_driver,
    )


def _score_wildcard(
    free_transfers: int, top5_mean: float, current_round: int,
    chip_state: dict, urgency_bonus: float = 0.0,
) -> ChipScore:
    used = _mark_used("Wildcard", chip_state)
    if used:
        return used

    score = 0.0
    reasons = []

    if free_transfers == 0:
        score += 3.0
        reasons.append("No free transfers — Wildcard removes all penalties")
    elif free_transfers < 0:
        score += 4.0
        reasons.append(f"In transfer debt ({free_transfers}) — Wildcard clears penalties")

    if top5_mean >= 25:
        score += 2.0
        reasons.append(f"High-scoring weekend projected ({top5_mean:.0f} pts avg)")
    elif top5_mean >= 20:
        score += 1.0

    if current_round <= 5:
        score += 2.0
        reasons.append("Early season — good time to reset around the true pecking order")

    # Season urgency — a Wildcard held past round 16 is basically wasted
    score += urgency_bonus
    if urgency_bonus >= 2.0:
        reasons.append("Late season — Wildcard must be used soon or it's wasted")

    use_now = score >= 3.0

    return ChipScore(
        chip="Wildcard",
        raw_score=round(score, 1),
        use_now=use_now,
        hold_until="when team needs a major reshuffle (price crash, injury, regulation shift)",
        reason=" · ".join(reasons) if reasons else "Save for a full squad reset when prices shift",
        urgency="high" if score >= 5 else ("medium" if use_now else "low"),
        expected_gain=18.0 if use_now else 0.0,
    )


def _score_final_fix(
    current_drivers: Optional[list], driver_pts: list, mc_results: dict,
    chip_state: dict, urgency_bonus: float = 0.0,
) -> ChipScore:
    used = _mark_used("Final Fix", chip_state)
    if used:
        return used

    score = 1.0
    reasons = []
    use_now = False

    if current_drivers:
        high_risk = [
            d["driver"] for d in driver_pts
            if d["driver"] in current_drivers and d.get("dnf_prob_pct", 0) > 20
        ]
        if high_risk:
            score += 2.0
            reasons.append(f"High DNF risk in your team: {', '.join(high_risk)}")
            use_now = True  # genuine reason to use it this weekend

    # Late season urgency — Final Fix is least valuable but still shouldn't be wasted
    score += urgency_bonus * 0.5  # half-weight for Final Fix
    if urgency_bonus >= 2.0 and not use_now:
        reasons.append("Late season — consider deploying after qualifying")

    return ChipScore(
        chip="Final Fix",
        raw_score=round(score, 1),
        use_now=use_now,
        reason=" · ".join(reasons) if reasons else "Deploy after qualifying if a key driver takes a heavy grid penalty",
        urgency="high" if use_now else ("medium" if urgency_bonus >= 2.0 else "low"),
        expected_gain=10.0 if use_now else 8.0,
    )


# ─────────────────────────────────────────────
# SEASON CONTEXT TABLE
# ─────────────────────────────────────────────

def build_season_context_table(current_round: int, chip_state: dict) -> list[dict]:
    """
    Build a full-season chip opportunity table showing all remaining rounds
    ranked by value. Used for display in main.py.

    Returns:
        List of dicts: {round, circuit, sprint, sc_prob, value_score, best_chip, notes}
    """
    sprint_rounds = get_sprint_rounds()
    rows = []
    for circuit_name, cfg in CIRCUITS.items():
        rnd = cfg.get("round", 0)
        if rnd <= current_round:
            continue
        key = cfg.get("key", "")
        is_sprint = rnd in sprint_rounds
        sc_prob = get_track_feature(key, "sc_probability", SC_PROBABILITY.get(key, 0.40))
        score = _circuit_value_score(key, is_sprint)

        if is_sprint and score >= 7.5:
            best_chip = "Limitless ★★"
        elif is_sprint:
            best_chip = "Limitless ★"
        elif sc_prob >= 0.65:
            best_chip = "No Negative ★★"
        elif sc_prob >= 0.50:
            best_chip = "No Negative ★"
        else:
            best_chip = "—"

        # Note if a chip is already used
        notes = []
        if is_sprint and chip_state.get("used", {}).get("Limitless"):
            notes.append("[Limitless used]")
        if sc_prob >= 0.50 and chip_state.get("used", {}).get("No Negative"):
            notes.append("[No Neg used]")

        rows.append({
            "round":        rnd,
            "circuit":      circuit_name,
            "sprint":       is_sprint,
            "sc_prob":      sc_prob,
            "value_score":  score,
            "best_chip":    best_chip,
            "note":         "  ".join(notes),
        })

    rows.sort(key=lambda x: x["round"])
    return rows



