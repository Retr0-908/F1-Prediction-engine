"""
stage_comparator.py — Multi-Stage Prediction & Cross-Session Evolution Analytics

Compares predictions generated across weekend stages (pre-practice, post-practice, post-quali)
against official race classification results, computing driver position deltas,
stage MAEs, accuracy gain percentage, and ranking correlation.
"""

import json
import logging
from pathlib import Path
import datetime
from typing import Optional, Dict, Any, List

from engine.core.config import CURRENT_SEASON, DRIVER_TEAMS_2026
from engine.core.paths import OUTPUT_DIR
from engine.core.data_fetcher import get_race_by_round, get_race_results

logger = logging.getLogger("f1_predictor.stage_comparator")

STAGES_DIR = OUTPUT_DIR / "stages"


def _spearman(pred: List[float], actual: List[float]) -> float:
    """Compute Spearman's rank correlation coefficient."""
    n = len(pred)
    if n < 3:
        return 0.0

    def rank(lst):
        indexed = sorted(enumerate(lst), key=lambda x: x[1])
        ranks = [0.0] * n
        for rank_val, (idx, _) in enumerate(indexed, 1):
            ranks[idx] = float(rank_val)
        return ranks

    pred_ranks = rank(pred)
    actual_ranks = rank(actual)

    d_sq_sum = sum((pr - ar) ** 2 for pr, ar in zip(pred_ranks, actual_ranks))
    rho = 1.0 - 6.0 * d_sq_sum / (n * (n * n - 1))
    return round(float(rho), 3)


def _find_driver_in_list(driver_name: str, item_list: list) -> dict:
    """Find driver entry in list of dicts using exact or surname matching."""
    if not item_list:
        return {}
    target = driver_name.strip().lower()
    surname = target.split()[-1] if target else ""

    # 1. Exact match
    for it in item_list:
        if it.get("driver", "").strip().lower() == target:
            return it
    # 2. Surname match
    if surname:
        for it in item_list:
            it_name = it.get("driver", "").strip().lower()
            if it_name.split()[-1] == surname:
                return it
    return {}


def _load_or_generate_stage_data(
    round_num: int,
    season: int,
    stage: str,
    race_info: dict,
    predictor: Optional[Any] = None,
) -> dict:
    """Load existing stage prediction file or dynamically predict stage using F1Predictor."""
    STAGES_DIR.mkdir(parents=True, exist_ok=True)
    pattern = f"race_{round_num}_*_{stage}.json"
    matches = list(STAGES_DIR.glob(pattern))

    if matches:
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        try:
            with open(matches[0], "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("race_order"):
                    return data
        except Exception as e:
            logger.warning(f"Failed to read stage file {matches[0]}: {e}")

    # Fallback for post-quali: check primary race predictions in output/
    if stage == "post-quali":
        primaries = [
            p for p in OUTPUT_DIR.glob(f"race_{round_num}_*.json")
            if not p.name.startswith("comparison_") and p.is_file()
        ]
        if primaries:
            primaries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            try:
                with open(primaries[0], "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("race_order"):
                        return data
            except Exception as e:
                logger.warning(f"Failed to read primary file {primaries[0]}: {e}")

    # Dynamically generate prediction for this stage
    from engine.models.predictor import F1Predictor

    p = predictor or F1Predictor()
    if not hasattr(p, "_trained") or not p._trained:
        p.train(verbose=False)

    circuit_cfg = race_info.get("circuit_config", {}) if race_info else {}
    p.load_context(circuit_cfg, {}, mode=stage, race_info=race_info)
    race_order = p.predict_finishing_order()
    quali_order = p.predict_qualifying_order()
    driver_pts = [
        p.estimate_fantasy_points(d["driver"], race_order, quali_order, False, None)
        for d in race_order
    ]

    race_name = (race_info.get("name") or "race").replace(" ", "_")
    stage_file = STAGES_DIR / f"race_{round_num}_{race_name}_{stage}.json"
    stage_data = {
        "generated_at": datetime.datetime.now().isoformat(),
        "stage": stage,
        "season": season,
        "race": race_info,
        "race_order": race_order,
        "qualifying_order": quali_order,
        "driver_pts": driver_pts,
    }
    try:
        with open(stage_file, "w", encoding="utf-8") as f:
            json.dump(stage_data, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to write stage file {stage_file}: {e}")

    return stage_data


def get_stage_comparison(
    round_num: int,
    season: int = CURRENT_SEASON,
    force_refresh: bool = False,
) -> dict:
    """
    Compare pre-practice, post-practice, and post-quali predictions against
    actual classification results for a round.
    """
    STAGES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = STAGES_DIR / f"comparison_round_{round_num}.json"

    if not force_refresh and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("status") == "ok" and cached.get("round") == round_num and cached.get("matrix"):
                return cached
        except Exception as e:
            logger.warning(f"Error reading cache {cache_path}: {e}")

    race_info = get_race_by_round(round_num, season)
    if not race_info:
        race_info = {"round": round_num, "name": f"Round {round_num}", "season": season}

    # Fetch actual results
    actual_results = []
    try:
        actual_results = get_race_results(season, round_num) or []
    except Exception as e:
        logger.warning(f"Could not fetch actual results for {season} R{round_num}: {e}")

    # Shared predictor instance for stage simulations if needed
    from engine.models.predictor import F1Predictor
    shared_predictor = None

    # Load/generate all 3 stages
    stage_names = ["pre-practice", "post-practice", "post-quali"]
    stages_data: Dict[str, dict] = {}

    for st in stage_names:
        if shared_predictor is None and not list(STAGES_DIR.glob(f"race_{round_num}_*_{st}.json")):
            shared_predictor = F1Predictor()
            shared_predictor.train(verbose=False)
        stages_data[st] = _load_or_generate_stage_data(
            round_num, season, st, race_info, shared_predictor
        )

    pre_order = stages_data["pre-practice"].get("race_order", [])
    pre_pts_list = stages_data["pre-practice"].get("driver_pts", [])

    post_order = stages_data["post-practice"].get("race_order", [])
    post_pts_list = stages_data["post-practice"].get("driver_pts", [])

    quali_order = stages_data["post-quali"].get("race_order", [])
    quali_pts_list = stages_data["post-quali"].get("driver_pts", [])

    # Drivers list: prioritize actual results, fall back to post-quali order or 2026 roster
    drivers_pool = []
    if actual_results:
        for act in actual_results:
            d_name = act.get("name", "")
            if d_name and d_name not in [d["name"] for d in drivers_pool]:
                drivers_pool.append({
                    "name": d_name,
                    "team": act.get("constructor", act.get("team", "")),
                    "driver_id": act.get("driver_id", act.get("driverId", d_name.lower().replace(" ", "_"))),
                    "actual_pos": int(act.get("position", 0)),
                    "actual_pts": float(act.get("points", 0.0)),
                    "actual_status": str(act.get("status", "Finished")),
                    "actual_grid": int(act.get("grid", 0)),
                })
    else:
        # Fallback pool
        candidate_order = quali_order or post_order or pre_order
        for d in candidate_order:
            d_name = d.get("driver", "")
            drivers_pool.append({
                "name": d_name,
                "team": d.get("team", DRIVER_TEAMS_2026.get(d_name, "")),
                "driver_id": d_name.lower().replace(" ", "_"),
                "actual_pos": 0,
                "actual_pts": 0.0,
                "actual_status": "Pending",
                "actual_grid": 0,
            })

    matrix = []
    for dp in drivers_pool:
        d_name = dp["name"]
        t_name = dp["team"]
        d_id = dp["driver_id"]
        act_pos = dp["actual_pos"]
        act_pts = dp["actual_pts"]
        act_status = dp["actual_status"]

        # Pre-practice
        pre_entry = _find_driver_in_list(d_name, pre_order)
        pre_pts_info = _find_driver_in_list(d_name, pre_pts_list)
        pre_pos = round(float(pre_entry.get("predicted_pos", pre_entry.get("predicted_rank", 20.0))), 1)
        pre_err = round(abs(pre_pos - act_pos), 1) if act_pos > 0 else 0.0
        pre_pts = round(float(pre_pts_info.get("total", pre_pts_info.get("total_pts", 0.0))), 1)

        # Post-practice
        post_entry = _find_driver_in_list(d_name, post_order)
        post_pts_info = _find_driver_in_list(d_name, post_pts_list)
        post_pos = round(float(post_entry.get("predicted_pos", post_entry.get("predicted_rank", 20.0))), 1)
        post_err = round(abs(post_pos - act_pos), 1) if act_pos > 0 else 0.0
        post_delta = round(post_pos - pre_pos, 1)
        post_pts = round(float(post_pts_info.get("total", post_pts_info.get("total_pts", 0.0))), 1)
        fp_pace = round(float(post_entry.get("fp_pace_delta", 0.0)), 3)

        # Post-quali
        quali_entry = _find_driver_in_list(d_name, quali_order)
        quali_pts_info = _find_driver_in_list(d_name, quali_pts_list)
        quali_pos = round(float(quali_entry.get("predicted_pos", quali_entry.get("predicted_rank", 20.0))), 1)
        quali_err = round(abs(quali_pos - act_pos), 1) if act_pos > 0 else 0.0
        quali_delta = round(quali_pos - post_pos, 1)
        quali_pts = round(float(quali_pts_info.get("total", quali_pts_info.get("total_pts", 0.0))), 1)
        grid_pos = int(quali_entry.get("grid") or quali_pts_info.get("grid_pos") or dp["actual_grid"] or 20)

        # Trajectory
        trajectory = f"P{int(round(pre_pos))} ➔ P{int(round(post_pos))} ➔ P{int(round(quali_pos))}"

        # Convergence
        if act_pos <= 0:
            convergence = "Pending"
        elif quali_err <= 1.0:
            convergence = "Bullseye"
        elif quali_err >= 6.0 or (act_status.lower() != "finished" and quali_err >= 4.0):
            convergence = "Upset"
        elif pre_err >= post_err >= quali_err and quali_err < pre_err:
            convergence = "Converged"
        elif quali_err >= post_err >= pre_err and quali_err > pre_err:
            convergence = "Diverged"
        elif quali_err < pre_err:
            convergence = "Converged"
        else:
            convergence = "Diverged"

        matrix.append({
            "driver": d_name,
            "team": t_name or DRIVER_TEAMS_2026.get(d_name, ""),
            "driver_id": d_id,
            "pre_practice": {
                "pos": pre_pos,
                "error": pre_err,
                "pts": pre_pts,
            },
            "post_practice": {
                "pos": post_pos,
                "error": post_err,
                "delta": post_delta,
                "pts": post_pts,
                "fp_pace_delta": fp_pace,
                "fp_delta": fp_pace,
            },
            "post_quali": {
                "pos": quali_pos,
                "error": quali_err,
                "delta": quali_delta,
                "pts": quali_pts,
                "grid": grid_pos,
            },
            "actual": {
                "pos": act_pos,
                "pts": act_pts,
                "status": act_status,
            },
            "trajectory": trajectory,
            "convergence": convergence,
        })

    # Summary metrics
    valid_actuals = [r for r in matrix if r["actual"]["pos"] and r["actual"]["pos"] > 0]
    if valid_actuals:
        pre_mae = round(sum(r["pre_practice"]["error"] for r in valid_actuals) / len(valid_actuals), 2)
        post_mae = round(sum(r["post_practice"]["error"] for r in valid_actuals) / len(valid_actuals), 2)
        quali_mae = round(sum(r["post_quali"]["error"] for r in valid_actuals) / len(valid_actuals), 2)
        gain_pct = round(((pre_mae - quali_mae) / pre_mae) * 100, 1) if pre_mae > 0 else 0.0

        pre_spearman = _spearman([r["pre_practice"]["pos"] for r in valid_actuals], [r["actual"]["pos"] for r in valid_actuals])
        post_spearman = _spearman([r["post_practice"]["pos"] for r in valid_actuals], [r["actual"]["pos"] for r in valid_actuals])
        quali_spearman = _spearman([r["post_quali"]["pos"] for r in valid_actuals], [r["actual"]["pos"] for r in valid_actuals])

        top_improver_row = max(valid_actuals, key=lambda r: (r["pre_practice"]["error"] - r["post_practice"]["error"]))
        top_improver = top_improver_row["driver"]

        biggest_upset_row = max(valid_actuals, key=lambda r: r["post_quali"]["error"])
        biggest_upset = biggest_upset_row["driver"]
    else:
        pre_mae = 0.0
        post_mae = 0.0
        quali_mae = 0.0
        gain_pct = 0.0
        pre_spearman = 0.0
        post_spearman = 0.0
        quali_spearman = 0.0
        top_improver = "N/A"
        biggest_upset = "N/A"

    metrics = {
        "pre_practice_mae": pre_mae,
        "post_practice_mae": post_mae,
        "post_quali_mae": quali_mae,
        "accuracy_gain_pct": gain_pct,
        "pre_practice_spearman": pre_spearman,
        "post_practice_spearman": post_spearman,
        "post_quali_spearman": quali_spearman,
        "top_improver": top_improver,
        "biggest_upset": biggest_upset,
    }

    report = {
        "status": "ok",
        "round": round_num,
        "season": season,
        "race_name": race_info.get("name", race_info.get("race_name", f"Round {round_num}")),
        "matrix": matrix,
        "metrics": metrics,
    }

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to cache comparison report {cache_path}: {e}")

    return report
