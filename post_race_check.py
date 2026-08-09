"""
post_race_check.py — F1 Fantasy Post-Race Validation & Accuracy Tracker

Fetches actual race results after each grand prix, compares them to the
predictions stored in output/race_R*_*.json, and logs accuracy metrics.

Usage:
    python post_race_check.py --round 5
    python post_race_check.py --round 5 --season 2026

Outputs:
    - Accuracy metrics (MAE, Spearman ρ, winner hit, top-5 hits)
    - Appended entry in logs/accuracy_log.json
    - Printed comparison table (actual vs predicted)
"""

import argparse
import json
import math
import sys
from pathlib import Path
from datetime import datetime

# Rich for display
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from config import CURRENT_SEASON
from data_fetcher import get_race_results

console = Console()

ACCURACY_LOG = Path(__file__).parent / "logs" / "accuracy_log.json"
OUTPUT_DIR   = Path(__file__).parent / "output"


# ─────────────────────────────────────────────
# STAT HELPERS
# ─────────────────────────────────────────────

def _spearman(pred: list[float], actual: list[float]) -> float:
    """Spearman rank correlation between predicted and actual positions."""
    n = len(pred)
    if n < 3:
        return 0.0

    def rank(lst):
        indexed = sorted(enumerate(lst), key=lambda x: x[1])
        ranks = [0.0] * n
        for rank_val, (idx, _) in enumerate(indexed, 1):
            ranks[idx] = float(rank_val)
        return ranks

    pred_ranks   = rank(pred)
    actual_ranks = rank(actual)

    d_sq_sum = sum((pr - ar) ** 2 for pr, ar in zip(pred_ranks, actual_ranks))
    rho = 1.0 - 6.0 * d_sq_sum / (n * (n * n - 1))
    return round(rho, 4)


def _mae(pred: list[float], actual: list[float]) -> float:
    if not pred:
        return 0.0
    return round(sum(abs(p - a) for p, a in zip(pred, actual)) / len(pred), 3)


def _rmse(pred: list[float], actual: list[float]) -> float:
    if not pred:
        return 0.0
    return round(math.sqrt(sum((p - a) ** 2 for p, a in zip(pred, actual)) / len(pred)), 3)


# ─────────────────────────────────────────────
# FIND PREDICTION FILE
# ─────────────────────────────────────────────

def _find_prediction_file(round_num: int) -> Path | None:
    """Find the most recent prediction file for a given round."""
    pattern = f"race_{round_num}_*.json"
    matches = sorted(OUTPUT_DIR.glob(pattern))
    if matches:
        return matches[-1]
    # Also try zero-padded
    return None


# ─────────────────────────────────────────────
# MAIN VALIDATION LOGIC
# ─────────────────────────────────────────────

def validate_specific_prediction(pred_file: Path) -> dict | None:
    """
    Validate a specific prediction file against actual results.
    """
    if not pred_file.exists():
        console.print(f"[yellow][!] Prediction file not found: {pred_file}[/yellow]")
        return None

    with open(pred_file, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    race_name = pred_data.get("race", {}).get("name", "Unknown Race")
    round_num = pred_data.get("race", {}).get("round", 0)
    season = pred_data.get("race", {}).get("season", CURRENT_SEASON)
    
    console.print(f"\n[bold cyan][Metrics] Post-Race Validation - {race_name} (R{round_num}, {season})[/bold cyan]\n")
    console.print(f"  Prediction file: [dim]{pred_file.name}[/dim]")

    # 2. Fetch actual results
    console.print(f"  Fetching actual results from Jolpica API...")
    try:
        actual_results = get_race_results(season, round_num)
    except Exception as e:
        console.print(f"[red]  Could not fetch results: {e}[/red]")
        return None

    if not actual_results:
        console.print("[yellow]  No results available yet - race may not have finished.[/yellow]")
        return None

    # Build {driver_name: actual_position} from results
    actual_pos: dict[str, int] = {}
    for r in actual_results:
        name = r.get("name", "")
        pos  = r.get("position", 99)
        if name:
            actual_pos[name] = int(pos)

    # 3. Build comparison list
    predicted_order = pred_data.get("race_order", [])
    comparison = []

    for pred_entry in predicted_order:
        driver = pred_entry.get("driver", "")
        pred_pos = pred_entry.get("predicted_rank", 99)
        act_pos  = actual_pos.get(driver)
        if act_pos is None:
            # Try partial match
            for k, v in actual_pos.items():
                if driver.split()[-1].lower() in k.lower():
                    act_pos = v
                    break
        if act_pos is None:
            act_pos = 99  # DNF / not in results

        comparison.append({
            "driver":    driver,
            "team":      pred_entry.get("team", ""),
            "predicted": pred_pos,
            "actual":    act_pos,
            "delta":     act_pos - pred_pos,
        })

    # Sort by actual position for display
    comparison.sort(key=lambda x: x["actual"])

    # 4. Compute metrics
    pred_list   = [c["predicted"] for c in comparison if c["actual"] < 99]
    actual_list = [c["actual"]    for c in comparison if c["actual"] < 99]

    mae   = _mae(pred_list, actual_list)
    rmse  = _rmse(pred_list, actual_list)
    rho   = _spearman(pred_list, actual_list)

    # Winner accuracy
    predicted_winner = next((c["driver"] for c in sorted(comparison, key=lambda x: x["predicted"])), None)
    actual_winner    = next((c["driver"] for c in sorted(comparison, key=lambda x: x["actual"])), None)
    winner_correct   = predicted_winner == actual_winner

    # Top-5 hits (how many of predicted top-5 appear in actual top-5)
    pred_top5   = {c["driver"] for c in sorted(comparison, key=lambda x: x["predicted"])[:5]}
    actual_top5 = {c["driver"] for c in sorted(comparison, key=lambda x: x["actual"])[:5]}
    top5_hits   = len(pred_top5 & actual_top5)

    # Top-3 hits
    pred_top3   = {c["driver"] for c in sorted(comparison, key=lambda x: x["predicted"])[:3]}
    actual_top3 = {c["driver"] for c in sorted(comparison, key=lambda x: x["actual"])[:3]}
    top3_hits   = len(pred_top3 & actual_top3)

    metrics = {
        "round":            round_num,
        "season":           season,
        "race":             race_name,
        "circuit_type":     pred_data.get("race", {}).get("circuit_config", {}).get("track_type", "permanent"),
        "circuit_key":      pred_data.get("race", {}).get("circuit_config", {}).get("key", ""),  # Phase 6: per-circuit bias
        "mae":              mae,
        "rmse":             rmse,
        "spearman_rho":     rho,
        "winner_correct":   winner_correct,
        "predicted_winner": predicted_winner,
        "actual_winner":    actual_winner,
        "top3_hits":        top3_hits,
        "top5_hits":        top5_hits,
        "n_drivers":        len(pred_list),
        "validated_at":     datetime.now().isoformat(),
        "driver_comparisons": comparison,
    }

    # 5. Display comparison table
    table = Table(
        title=f"[Race] {race_name} - Prediction vs Actual",
        box=box.ROUNDED, show_header=True, header_style="bold cyan",
        border_style="cyan",
    )
    table.add_column("Driver",    width=22)
    table.add_column("Team",      width=16)
    table.add_column("Predicted", width=9,  justify="center")
    table.add_column("Actual",    width=7,  justify="center")
    table.add_column("Diff",      width=5,  justify="center")

    for c in comparison[:20]:
        delta     = c["delta"]
        delta_str = f"+{delta}" if delta > 0 else str(delta) if delta != 0 else "0"
        delta_clr = "red" if abs(delta) > 5 else "yellow" if abs(delta) > 2 else "green"
        actual_str = str(c["actual"]) if c["actual"] < 99 else "DNF"
        table.add_row(
            c["driver"],
            c["team"],
            f"P{c['predicted']}",
            actual_str,
            f"[{delta_clr}]{delta_str}[/{delta_clr}]",
        )
    console.print(table)

    # 6. Display metrics panel
    rho_color  = "bold green" if rho >= 0.65 else "green" if rho >= 0.45 else "yellow" if rho >= 0.25 else "red"
    mae_color  = "green" if mae <= 3.5 else "yellow" if mae <= 5.0 else "red"
    win_color  = "bold green" if winner_correct else "red"

    console.print(Panel(
        f"  MAE:             [{mae_color}]{mae:.2f} positions[/{mae_color}]\n"
        f"  RMSE:            {rmse:.2f}\n"
        f"  Spearman rho:    [{rho_color}]{rho:.3f}[/{rho_color}]\n"
        f"  Winner correct:  [{win_color}]{'YES' if winner_correct else 'NO'} "
        f"(predicted {predicted_winner} · actual {actual_winner})[/{win_color}]\n"
        f"  Top-3 hits:      {top3_hits}/3 ({top3_hits/3*100:.0f}%)\n"
        f"  Top-5 hits:      {top5_hits}/5 ({top5_hits/5*100:.0f}%)",
        title=f"Accuracy Metrics - R{round_num} {race_name}",
        border_style="cyan",
    ))

    # 7. Update accuracy log
    _append_accuracy_log(metrics)

    # Trigger self-improvement calculations
    try:
        import self_improvement
        self_improvement.compute_bias_corrections()
        self_improvement.compute_weight_adjustments()
        console.print("  [dim][OK] Self-improvement bias corrections updated.[/dim]")
    except Exception as e:
        console.print(f"  [red][!] Failed to run self-improvement calculations: {e}[/red]")

    # 8. Show running season summary
    _show_season_summary(season)

    return metrics

def validate_round(round_num: int, season: int = CURRENT_SEASON) -> dict | None:
    """
    Fetch actual results for round_num, compare to stored predictions,
    compute accuracy metrics, update accuracy log, and display comparison.
    """
    console.print(f"\n[bold cyan][Metrics] Post-Race Validation - Round {round_num} ({season})[/bold cyan]\n")

    # 1. Load prediction file
    pred_file = _find_prediction_file(round_num)
    if not pred_file:
        console.print(f"[yellow][!] No prediction file found for Round {round_num}. "
                      f"Run main.py before the race to generate predictions.[/yellow]")
        return None

    return validate_specific_prediction(pred_file)





def _append_accuracy_log(metrics: dict) -> None:
    """Append single race metrics to the accuracy log JSON."""
    ACCURACY_LOG.parent.mkdir(parents=True, exist_ok=True)

    log = []
    if ACCURACY_LOG.exists():
        try:
            with open(ACCURACY_LOG, "r", encoding="utf-8") as f:
                log = json.load(f)
        except Exception:
            log = []

    # Replace if same round already logged
    log = [e for e in log if not (e.get("round") == metrics["round"] and e.get("season") == metrics["season"])]
    log.append(metrics)
    log.sort(key=lambda x: (x.get("season", 0), x.get("round", 0)))

    with open(ACCURACY_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    console.print(f"  [dim][OK] Accuracy log updated: {ACCURACY_LOG}[/dim]")


def _show_season_summary(season: int) -> None:
    """Print rolling season accuracy stats from the log."""
    if not ACCURACY_LOG.exists():
        return

    with open(ACCURACY_LOG, "r", encoding="utf-8") as f:
        log = json.load(f)

    season_log = [e for e in log if e.get("season") == season]
    if not season_log:
        return

    n       = len(season_log)
    avg_mae = round(sum(e["mae"] for e in season_log) / n, 3)
    avg_rho = round(sum(e["spearman_rho"] for e in season_log) / n, 3)
    win_pct = round(sum(1 for e in season_log if e["winner_correct"]) / n * 100, 1)
    top5_avg = round(sum(e["top5_hits"] for e in season_log) / n, 2)

    rho_color = "bold green" if avg_rho >= 0.60 else "green" if avg_rho >= 0.45 else "yellow"
    console.print(Panel(
        f"  Races logged:    {n}\n"
        f"  Avg MAE:         {avg_mae:.2f} positions\n"
        f"  Avg Spearman rho:[{rho_color}]{avg_rho:.3f}[/{rho_color}]\n"
        f"  Winner hit rate: {win_pct:.1f}%\n"
        f"  Avg top-5 hits:  {top5_avg:.2f}/5",
        title=f"[Stats] {season} Season Accuracy Summary ({n} races)",
        border_style="blue",
    ))


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="F1 Fantasy Post-Race Validator — compare predictions vs actual results"
    )
    parser.add_argument("--round",  type=int, required=True,  help="Race round number to validate")
    parser.add_argument("--season", type=int, default=CURRENT_SEASON, help="Season year (default: current)")
    parser.add_argument("--summary", action="store_true", help="Show season summary only (no round validation)")
    args = parser.parse_args()

    if args.summary:
        _show_season_summary(args.season)
        return

    result = validate_round(args.round, args.season)
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
