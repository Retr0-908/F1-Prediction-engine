"""
fantasy_optimizer.py — F1 Fantasy Team Optimizer (v2)
Finds optimal team changes and globally best team given budget constraints.

New in v2:
  - Monte Carlo mean pts as optional scoring input (replaces deterministic single-point)
  - find_differential_picks() — high-EV low-ownership picks for leaderboard climbing
  - find_best_turbo_driver() — 2x Boost optimizer (MC risk-adjusted)
  - Ownership differential scoring (differential_score)
"""

import itertools
import pulp
from typing import Optional, Dict, List

from engine.core.config import (
    FANTASY_BUDGET, FANTASY_NUM_DRIVERS, FANTASY_NUM_CONSTRUCTORS,
    CONSTRUCTORS_2025, DRIVER_TEAMS_2025, CURRENT_SEASON,
)

# A top-5 driver / top-3 constructor may only be swapped out for a replacement
# projected to score at least this much MORE than the star (30%).
TOP_PERFORMER_GAIN_THRESHOLD = 0.30


def get_current_team_value(
    drivers: list[str],
    constructors: list[str],
    driver_prices: dict,
    constructor_prices: dict,
) -> dict:
    """Calculate current team's total cost and per-player values."""
    players = []
    total = 0.0
    missing = []

    for d in drivers:
        matched = _fuzzy_match(d, driver_prices)
        if matched:
            price = driver_prices[matched].get("price", 0)
            players.append({"name": matched, "type": "driver", "price": price})
            total += price
        else:
            missing.append(d)

    for c in constructors:
        matched = _fuzzy_match(c, constructor_prices)
        if matched:
            price = constructor_prices[matched].get("price", 0)
            players.append({"name": matched, "type": "constructor", "price": price})
            total += price
        else:
            missing.append(c)

    return {
        "players":         players,
        "total_cost":      round(total, 1),
        "budget_spent":    round(total, 1),
        "budget_left":     round(FANTASY_BUDGET - total, 1),
        "missing_players": missing,
    }


def suggest_team_changes(
    current_drivers: list[str],
    current_constructors: list[str],
    driver_pts: list[dict],
    constructor_pts: list[dict],
    driver_prices: dict,
    constructor_prices: dict,
    free_transfers: int = 1,
    budget_remaining: float = 0.0,
    mc_pts: dict[str, float] = None,
) -> dict:
    warnings_list = []

    def effective_pts(name: str, det_pts: float) -> float:
        if mc_pts and name in mc_pts:
            return mc_pts[name]
        return det_pts

    current_d_pts   = {d: effective_pts(d, _get_driver_pts(d, driver_pts))   for d in current_drivers}
    current_c_pts   = {c: effective_pts(c, _get_ctor_pts(c, constructor_pts)) for c in current_constructors}
    current_d_price = {d: _get_driver_price(d, driver_prices)    for d in current_drivers}
    current_c_price = {c: _get_ctor_price(c, constructor_prices) for c in current_constructors}

    total_current_pts  = sum(current_d_pts.values()) + sum(current_c_pts.values())
    total_current_cost = sum(current_d_price.values()) + sum(current_c_price.values())

    # --- Top-performer protection ---
    # If a current player is in the top 5 drivers / top 3 constructors by predicted pts,
    # only allow dropping them if the replacement has >30% more expected points.
    # This prevents absurd suggestions like "drop Mercedes for Audi".
    all_d_pts_sorted = sorted(
        [(d["driver"], effective_pts(d["driver"], d["total_pts"])) for d in driver_pts],
        key=lambda x: x[1], reverse=True
    )
    all_c_pts_sorted = sorted(
        [(c["constructor"], effective_pts(c["constructor"], c["total_pts"])) for c in constructor_pts],
        key=lambda x: x[1], reverse=True
    )
    top_driver_names = {name for name, _ in all_d_pts_sorted[:5]}
    top_ctor_names   = {name for name, _ in all_c_pts_sorted[:3]}

    # Prepare candidate pools (players NOT currently in team)
    d_candidates = []
    for cand in driver_pts:
        name = cand["driver"]
        if name in current_drivers:
            continue
        pts = effective_pts(name, cand["total_pts"])
        price = _get_driver_price(name, driver_prices)
        if price <= 0:
            continue
        team = DRIVER_TEAMS_2025.get(name, "")
        d_candidates.append({"name": name, "pts": pts, "price": price, "team": team})

    c_candidates = []
    for cand in constructor_pts:
        name = cand["constructor"]
        if name in current_constructors:
            continue
        pts = effective_pts(name, cand["total_pts"])
        price = _get_ctor_price(name, constructor_prices)
        if price <= 0:
            continue
        c_candidates.append({"name": name, "pts": pts, "price": price})

    best_swaps = []
    best_net_gain = 0.0

    max_k = max(3, free_transfers + 2)

    current_players = [{"type": "driver", "name": d} for d in current_drivers] + \
                      [{"type": "ctor",   "name": c} for c in current_constructors]

    for k in range(1, max_k + 1):
        penalty = max(0, k - free_transfers) * 10

        for out_combo in itertools.combinations(current_players, k):
            out_d = [p["name"] for p in out_combo if p["type"] == "driver"]
            out_c = [p["name"] for p in out_combo if p["type"] == "ctor"]

            cash_freed = sum(current_d_price.get(d, 0) for d in out_d) + \
                         sum(current_c_price.get(c, 0) for c in out_c)
            pts_lost   = sum(current_d_pts.get(d, 0) for d in out_d) + \
                         sum(current_c_pts.get(c, 0) for c in out_c)

            budget_for_in = budget_remaining + cash_freed

            in_d_combos = itertools.combinations(d_candidates, len(out_d)) if len(out_d) > 0 else [()]

            for in_d in in_d_combos:
                in_d_cost = sum(d["price"] for d in in_d)
                if in_d_cost > budget_for_in:
                    continue

                remaining_drivers = [d for d in current_drivers if d not in out_d]
                trial_drivers = remaining_drivers + [d["name"] for d in in_d]
                remaining_ctors = [c for c in current_constructors if c not in out_c]
                if max(_count_team_assets(trial_drivers, remaining_ctors).values(), default=0) > 2:
                    continue

                in_c_combos = itertools.combinations(c_candidates, len(out_c)) if len(out_c) > 0 else [()]

                for in_c in in_c_combos:
                    total_in_cost = in_d_cost + sum(c["price"] for c in in_c)
                    if total_in_cost > budget_for_in:
                        continue
                    final_ctors = remaining_ctors + [c["name"] for c in in_c]
                    if max(_count_team_assets(trial_drivers, final_ctors).values(), default=0) > 2:
                        continue
                    new_team_total = total_current_cost - cash_freed + total_in_cost
                    if new_team_total > FANTASY_BUDGET:
                        continue

                    pts_gained = sum(d["pts"] for d in in_d) + sum(c["pts"] for c in in_c)
                    net_gain = pts_gained - pts_lost - penalty

                    # Top-performer protection: a star may only be swapped out
                    # for a replacement projected to score (1+THRESHOLD)x more.
                    justified = True
                    for od in out_d:
                        if od in top_driver_names:
                            od_idx = list(out_d).index(od)
                            replacement = list(in_d)[od_idx] if od_idx < len(in_d) else None
                            if replacement:
                                repl_pts = replacement["pts"]
                                star_pts = current_d_pts.get(od, 0)
                                if repl_pts < star_pts * (1.0 + TOP_PERFORMER_GAIN_THRESHOLD):
                                    justified = False
                                    break
                    for oc in out_c:
                        if oc in top_ctor_names and justified:
                            oc_idx = list(out_c).index(oc)
                            replacement = list(in_c)[oc_idx] if oc_idx < len(in_c) else None
                            if replacement:
                                repl_pts = replacement["pts"]
                                star_pts = current_c_pts.get(oc, 0)
                                if repl_pts < star_pts * (1.0 + TOP_PERFORMER_GAIN_THRESHOLD):
                                    justified = False
                                    break

                    if not justified:
                        continue

                    if net_gain > best_net_gain:
                        best_net_gain = net_gain

                        combo_swaps = []
                        d_in_list = list(in_d)
                        for idx, od in enumerate(out_d):
                            idrv = d_in_list[idx]
                            combo_swaps.append({
                                "type": "driver", "out": od,
                                "out_pts": round(current_d_pts.get(od, 0), 1),
                                "out_price": round(current_d_price.get(od, 0), 1),
                                "in": idrv["name"],
                                "in_pts": round(idrv["pts"], 1),
                                "in_price": round(idrv["price"], 1),
                                "pts_gain": round(idrv["pts"] - current_d_pts.get(od, 0), 1),
                                "cost_diff": round(idrv["price"] - current_d_price.get(od, 0), 1),
                            })
                        c_in_list = list(in_c)
                        for idx, oc in enumerate(out_c):
                            ictor = c_in_list[idx]
                            combo_swaps.append({
                                "type": "constructor", "out": oc, "out_pts": round(current_c_pts.get(oc,0),1),
                                "out_price": round(current_c_price.get(oc,0),1), "in": ictor["name"],
                                "in_pts": round(ictor["pts"],1), "in_price": round(ictor["price"],1),
                                "pts_gain": round(ictor["pts"] - current_c_pts.get(oc,0), 1),
                                "cost_diff": round(ictor["price"] - current_c_price.get(oc,0), 1)
                            })
                        
                        # Order: do fund-freeing swaps first (negative cost_diff = sell expensive → buy cheap).
                        # This ensures subsequent transfers in a multi-transfer plan have the budget they need.
                        combo_swaps.sort(key=lambda s: s["cost_diff"])
                        best_swaps = combo_swaps

    actual_transfers = len(best_swaps)
    penalties = max(0, actual_transfers - free_transfers) * 10
    if penalties > 0:
        warnings_list.append(f"⚠️  {actual_transfers - free_transfers} extra transfer(s) taken: -{penalties} pts penalty deducted.")
    elif free_transfers == 0 and not best_swaps:
        warnings_list.append("⚠️  No free transfers remaining — any further changes will cost -10 pts.")

    return {
        "suggested_changes":     best_swaps,
        "all_swaps_ranked":      [],
        "projected_pts_current": round(total_current_pts, 1),
        "projected_pts_new":     round(total_current_pts + best_net_gain, 1),
        "points_gain":           round(best_net_gain, 1),
        "transfers_used":        actual_transfers,
        "transfers_free":        free_transfers,
        "warnings":              warnings_list,
        "using_mc_pts":          mc_pts is not None,
    }


def find_optimal_team(
    driver_pts: list[dict],
    constructor_pts: list[dict],
    driver_prices: dict,
    constructor_prices: dict,
    budget: float = FANTASY_BUDGET,
    mc_pts: dict[str, float] = None,
    season: int = CURRENT_SEASON,
) -> dict:
    """
    Find the globally optimal team using Linear Programming (PuLP).
    Handles: $100M budget, 5 drivers, 2 constructors, max 2 assets per team,
    and 2x Turbo Driver maximization.
    """
    def get_pts(name: str, det_pts: float) -> float:
        if mc_pts and name in mc_pts:
            return mc_pts[name]
        return det_pts

    # 1. Prepare Data
    # Dynamically resolve season roster
    season_roster = None
    try:
        from engine.core.data_fetcher import get_season_roster
        season_roster = get_season_roster(season)
    except Exception:
        pass

    drivers = []
    for d in driver_pts:
        name = d["driver"]
        price = _get_driver_price(name, driver_prices)
        if price > 0:
            team = d.get("team")
            if not team or team == "Unknown":
                if season_roster and name in season_roster:
                    team = season_roster[name]
                elif season_roster:
                    matched = _fuzzy_match(name, season_roster)
                    if matched:
                        team = season_roster[matched]
                if not team or team == "Unknown":
                    matched_2025 = _fuzzy_match(name, DRIVER_TEAMS_2025)
                    team = DRIVER_TEAMS_2025.get(name) or (DRIVER_TEAMS_2025.get(matched_2025) if matched_2025 else "Unknown")
            drivers.append({
                "name": name,
                "pts": get_pts(name, d["total_pts"]),
                "price": price,
                "team": team or "Unknown"
            })

    constructors = []
    for c in constructor_pts:
        name = c["constructor"]
        price = _get_ctor_price(name, constructor_prices)
        if price > 0:
            constructors.append({
                "name": name,
                "pts": get_pts(name, c["total_pts"]),
                "price": price,
                "team": name # Team name is the constructor name
            })

    all_teams = set([d["team"] for d in drivers] + [c["team"] for c in constructors])
    
    best_overall_pts = -1.0
    best_overall_team = None

    # 2. Iterate over each possible Turbo Driver (2x boost)
    # Since only one driver can be Turbo, we can solve the LP for each case
    # or add a binary variable for it. Solving 20 small LPs is instantaneous.
    for turbo_candidate_idx in range(len(drivers)):
        prob = pulp.LpProblem(f"F1_Fantasy_Opt_Turbo_{turbo_candidate_idx}", pulp.LpMaximize)

        # Decision Variables
        d_vars = [pulp.LpVariable(f"d_{i}", cat=pulp.LpBinary) for i in range(len(drivers))]
        c_vars = [pulp.LpVariable(f"c_{j}", cat=pulp.LpBinary) for j in range(len(constructors))]

        # Objective: Maximize Points (including 2x for the chosen turbo candidate)
        # Note: Turbo only applies if that driver is actually selected in the team
        obj_terms = []
        for i, d in enumerate(drivers):
            multiplier = 2.0 if i == turbo_candidate_idx else 1.0
            obj_terms.append(d["pts"] * multiplier * d_vars[i])
        for j, c in enumerate(constructors):
            obj_terms.append(c["pts"] * c_vars[j])
        
        prob += pulp.lpSum(obj_terms)

        # Constraints
        # 1. Exactly 5 drivers
        prob += pulp.lpSum(d_vars) == FANTASY_NUM_DRIVERS
        # 2. Exactly 2 constructors
        prob += pulp.lpSum(c_vars) == FANTASY_NUM_CONSTRUCTORS
        # 3. Budget limit
        prob += pulp.lpSum([drivers[i]["price"] * d_vars[i] for i in range(len(drivers))]) + \
                pulp.lpSum([constructors[j]["price"] * c_vars[j] for j in range(len(constructors))]) <= budget
        # 4. Max 2 assets per team (Driver + Constructor)
        for t_name in all_teams:
            team_assets = []
            for i, d in enumerate(drivers):
                if d["team"] == t_name: team_assets.append(d_vars[i])
            for j, c in enumerate(constructors):
                if c["team"] == t_name: team_assets.append(c_vars[j])
            if team_assets:
                prob += pulp.lpSum(team_assets) <= 2

        # 5. Turbo driver MUST be in the team for this iteration to be valid
        prob += d_vars[turbo_candidate_idx] == 1

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] == 'Optimal':
            current_pts = pulp.value(prob.objective)
            if current_pts > best_overall_pts:
                best_overall_pts = current_pts
                
                selected_drivers = [drivers[i] for i in range(len(drivers)) if float(pulp.value(d_vars[i]) or 0) > 0.5]
                selected_ctors = [constructors[j] for j in range(len(constructors)) if float(pulp.value(c_vars[j]) or 0) > 0.5]
                
                best_overall_team = {
                    "drivers": sorted(selected_drivers, key=lambda x: x["pts"], reverse=True),
                    "constructors": sorted(selected_ctors, key=lambda x: x["pts"], reverse=True),
                    "total_pts": round(float(current_pts), 1),
                    "total_price": round(sum(float(d["price"]) for d in selected_drivers) + sum(float(c["price"]) for c in selected_ctors), 1),
                    "budget_left": round(float(budget) - (sum(float(d["price"]) for d in selected_drivers) + sum(float(c["price"]) for c in selected_ctors)), 1),
                    "turbo_driver": drivers[turbo_candidate_idx]["name"],
                    "using_mc_pts": mc_pts is not None,
                }

    return best_overall_team if best_overall_team else {
        "drivers": [], "constructors": [], "total_pts": 0.0, "total_price": 0.0, "budget_left": budget, "using_mc_pts": False
    }


def find_best_constructor(
    constructor_pts: list[dict],
    constructor_prices: dict,
) -> list[dict]:
    """Rank constructors by projected pts, value, and pts-per-million."""
    ranked = []
    for c in constructor_pts:
        name  = c["constructor"]
        pts   = c["total_pts"]
        price = _get_ctor_price(name, constructor_prices)
        value = round(pts / max(price, 1), 3)
        ranked.append({
            "constructor": name,
            "total_pts":   pts,
            "price":       price,
            "value":       value,
            "driver_pts":  c.get("driver_pts", 0),
            "pit_pts":     c.get("pit_stop_pts", 0),
            "drivers":     c.get("drivers", []),
        })
    ranked.sort(key=lambda x: x["total_pts"], reverse=True)
    return ranked


# ─────────────────────────────────────────────
# NEW: DIFFERENTIAL PICKS
# ─────────────────────────────────────────────
def find_differential_picks(
    driver_pts: list[dict],
    driver_prices: dict,
    mc_results: dict = None,
    top_n: int = 5,
) -> list[dict]:
    """
    Find high-value, low-ownership differential picks.

    differential_score = (expected_pts * upside_bonus) / (ownership_pct + 1)
    High score = high EV relative to how many managers own this driver.

    Args:
        driver_pts:  all drivers with total_pts & ownership_pct
        driver_prices: prices dict with ownership_pct
        mc_results:  Monte Carlo DistributionStats (optional — boosts upside estimate)

    Returns:
        Top N differential picks sorted by differential_score.
    """
    differentials = []
    for d in driver_pts:
        name     = d["driver"]
        det_pts  = d["total_pts"]
        price    = _get_driver_price(name, driver_prices)
        own_pct  = driver_prices.get(name, {}).get("ownership_pct", 5.0) or 5.0

        # Use MC mean if available
        if mc_results and name in mc_results:
            mc_s = mc_results[name]
            exp_pts    = mc_s.mean_pts
            upside_mul = 1.0 + max(0, (mc_s.p90_pts - mc_s.mean_pts) / max(mc_s.mean_pts, 1))
        else:
            exp_pts    = det_pts
            upside_mul = 1.0

        diff_score = (exp_pts * upside_mul) / (own_pct + 1.0)

        differentials.append({
            "driver":           name,
            "team":             d.get("team", ""),
            "price":            round(price, 1),
            "ownership_pct":    round(own_pct, 1),
            "exp_pts":          round(exp_pts, 1),
            "upside_mul":       round(upside_mul, 2),
            "differential_score": round(diff_score, 3),
            "p90_pts":          round(mc_results[name].p90_pts, 1) if mc_results and name in mc_results else round(exp_pts * 1.3, 1),
        })

    differentials.sort(key=lambda x: x["differential_score"], reverse=True)
    return differentials[:top_n]


# ─────────────────────────────────────────────
# NEW: TURBO DRIVER OPTIMIZER (moved here from chip_advisor)
# (chip_advisor.py also has this for standalone use)
# ─────────────────────────────────────────────
def find_best_turbo_driver(
    current_drivers: list[str],
    driver_pts: list[dict],
    mc_results: dict = None,
) -> dict:
    """
    Recommend which current driver to apply the 2x Turbo Boost to.
    Risk-adjusted: mean - 0.3*std to penalise volatile picks.
    """
    candidates = [d for d in driver_pts if d["driver"] in current_drivers]
    if not candidates:
        return {"driver": None, "reason": "No team entered", "expected_bonus_pts": 0}

    if mc_results:
        def risk_score(d):
            mc = mc_results.get(d["driver"])
            return (mc.mean_pts - 0.3 * mc.std_pts) if mc else d["total_pts"]

        best  = max(candidates, key=risk_score)
        mc    = mc_results.get(best["driver"])
        bonus = mc.mean_pts if mc else best["total_pts"]
        risk  = "High" if (mc and mc.std_pts > 15) else "Medium" if (mc and mc.std_pts > 8) else "Low"
        reason = (
            f"Best risk-adjusted EV: mean {mc.mean_pts:.1f} pts, "
            f"P90 ceiling {mc.p90_pts:.1f} pts, DNF {mc.p_dnf:.0f}%"
        ) if mc else f"Highest projected pts ({best['total_pts']:.1f})"
    else:
        # total_pts is now a Bayesian EV which intrinsically factors in DNF probability.
        # No need to multiply by (1 - dnf_prob_pct) again.
        best   = max(candidates, key=lambda d: d["total_pts"])
        bonus  = best["total_pts"]
        risk   = "Medium"
        reason = f"Best projected pts ({best['total_pts']:.1f} pts)"

    return {
        "driver":             best["driver"],
        "team":               best.get("team", ""),
        "reason":             reason,
        "expected_bonus_pts": round(bonus, 1),
        "risk_level":         risk,
    }


# ─────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────
def _get_driver_pts(name: str, driver_pts: list[dict]) -> float:
    for d in driver_pts:
        if d["driver"] == name:
            return d["total_pts"]
    return 0.0


def _get_ctor_pts(name: str, ctor_pts: list[dict]) -> float:
    for c in ctor_pts:
        if c["constructor"] == name:
            return c["total_pts"]
    return 0.0


import logging

logger = logging.getLogger("f1_predictor.optimizer")


def _get_driver_price(name: str, prices: dict) -> float:
    """Price for a driver, or 0.0 when unmatchable (caller excludes the
    candidate). Plan 8b: NEVER fabricate a price — invented values silently
    entered budget/LP math."""
    matched = _fuzzy_match(name, prices)
    if matched:
        return float(prices[matched].get("price", 0))
    logger.warning("No price match for driver %r — candidate excluded", name)
    return 0.0


def _get_ctor_price(name: str, prices: dict) -> float:
    matched = _fuzzy_match(name, prices)
    if matched:
        return float(prices[matched].get("price", 0))
    logger.warning("No price match for constructor %r — candidate excluded", name)
    return 0.0


def _fuzzy_match(name: str, data: dict) -> Optional[str]:
    """Exact-normalized match first; then FULL-surname-token equality.
    Bidirectional substring matching removed (plan 8b): 'sainz' inside a
    longer string, or partial tokens, must not resolve to the wrong player.
    Ambiguous surname ties are rejected rather than first-picked."""
    name_l = name.lower().strip()
    # 1. exact
    for key in data:
        if key.lower() == name_l:
            return key
    # 2. full surname-token equality (unique only)
    last = name_l.split()[-1] if name_l.split() else name_l
    hits = [key for key in data
            if key.lower().split()[-1] == last]
    if len(hits) == 1:
        logger.info("Fuzzy price match: %r -> %r (surname token)", name, hits[0])
        return hits[0]
    if len(hits) > 1:
        logger.warning("Ambiguous price match for %r (candidates: %s) — rejected",
                       name, hits)
    return None


def _count_team_assets(drivers: list[str], constructors: list[str]) -> dict[str, int]:
    """Count total assets (drivers + constructor) per team for the 2-asset-per-team cap."""
    counts: dict[str, int] = {}
    for drv in drivers:
        team = DRIVER_TEAMS_2025.get(drv, drv)
        counts[team] = counts.get(team, 0) + 1
    for ctor in constructors:
        counts[ctor] = counts.get(ctor, 0) + 1
    return counts

find_global_optimal_team = find_optimal_team


def evaluate_undercut_potential(circuit_key: str, driver_stints: Optional[dict] = None) -> dict:
    """
    Evaluates tactical undercut viability for a circuit based on pit loss time,
    tire degradation profile, and overtaking difficulty (inspired by mehmetkahya0).

    Returns:
        dict: {
            "circuit_key": str,
            "undercut_viability_score": float (0.0 to 1.0),
            "tire_deg_advantage_s": float,
            "recommended_lap_delta": int,
            "is_undercut_favored": bool,
        }
    """
    from engine.core.track_features_loader import load_track_features
    tf = load_track_features(circuit_key) or {}

    pit_loss = float(tf.get("pit_time_loss_s", 22.0))
    tire_deg = float(tf.get("tire_degradation", 3.0))       # 1 to 5
    overtake_diff = float(tf.get("overtaking_difficulty", 3.0)) # 1 to 5
    deg_delta = float(tf.get("deg_compound_delta", 0.3))

    # Fresh-tire pace advantage per lap
    deg_factor = (tire_deg / 3.0) * max(0.2, deg_delta * 3.5)
    tire_deg_advantage_s = round(float(deg_factor), 3)

    # Pit loss penalty factor
    pit_factor = max(0.5, (25.0 - pit_loss) / 10.0 + 0.5)

    # Track position factor
    track_pos_factor = 1.0 + (overtake_diff - 3.0) * 0.15

    # Viability score bounded [0.0, 1.0]
    raw_score = (deg_factor * 0.4 + pit_factor * 0.3 + track_pos_factor * 0.3) / 1.5
    viability_score = round(float(max(0.0, min(1.0, raw_score))), 3)

    recommended_lap_delta = 2 if tire_deg_advantage_s >= 1.0 else 1

    return {
        "circuit_key": circuit_key,
        "undercut_viability_score": viability_score,
        "tire_deg_advantage_s": tire_deg_advantage_s,
        "recommended_lap_delta": recommended_lap_delta,
        "is_undercut_favored": viability_score >= 0.55,
    }

