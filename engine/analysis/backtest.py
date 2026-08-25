"""
backtest.py — F1 Prediction Model Backtester (Multi-Year)
Evaluates race AND sprint predictions against actual results.

Usage:
    python backtest.py                          # Default: 2024 + 2025 + 2026 completed
    python backtest.py --years 2024 2025        # Specific years
    python backtest.py --years 2026 --rounds 1  # Specific rounds
"""

import sys
import argparse
import datetime
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich import box

console = Console()

from engine.core.config import (
    CURRENT_SEASON, HISTORICAL_SEASONS,
    CONSTRUCTORS_2025,
)
from engine.core.data_fetcher import (
    get_season_schedule, get_race_results, get_qualifying_results,
    get_driver_standings, get_constructor_standings,
    get_circuit_history,
    compute_multiseason_driver_form, compute_multiseason_constructor_stats,
    get_sprint_rounds, get_season_roster,
)
from engine.models.predictor import F1Predictor, _match_circuit_cfg

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

# ─────────────────────────────────────────────
# ROSTER RESOLUTION — dynamic first, static fallback
# Lineups are auto-derived from championship standings each season; the tables
# below are ONLY used offline or before a season's data exists in the API.
# Team names use config-canonical names ("Audi", "Racing Bulls") so they match
# _normalize_constructor_name output. predictor.train() raises KeyError rather
# than silently using a wrong-year roster.
# ─────────────────────────────────────────────
DRIVER_TEAMS_BY_YEAR = {
    2021: {
        "Lewis Hamilton": "Mercedes",   "Valtteri Bottas": "Mercedes",
        "Max Verstappen": "Red Bull",   "Sergio Perez": "Red Bull",
        "Lando Norris": "McLaren",      "Daniel Ricciardo": "McLaren",
        "Charles Leclerc": "Ferrari",   "Carlos Sainz": "Ferrari",
        "Esteban Ocon": "Alpine",       "Fernando Alonso": "Alpine",
        "Pierre Gasly": "Racing Bulls", "Yuki Tsunoda": "Racing Bulls",
        "Sebastian Vettel": "Aston Martin", "Lance Stroll": "Aston Martin",
        "George Russell": "Williams",   "Nicholas Latifi": "Williams",
        "Kimi Raikkonen": "Audi",       "Antonio Giovinazzi": "Audi",
        "Robert Kubica": "Audi",
        "Mick Schumacher": "Haas",      "Nikita Mazepin": "Haas",
    },
    2022: {
        "Max Verstappen": "Red Bull",   "Sergio Perez": "Red Bull",
        "Charles Leclerc": "Ferrari",   "Carlos Sainz": "Ferrari",
        "Lewis Hamilton": "Mercedes",   "George Russell": "Mercedes",
        "Lando Norris": "McLaren",      "Daniel Ricciardo": "McLaren",
        "Esteban Ocon": "Alpine",       "Fernando Alonso": "Alpine",
        "Pierre Gasly": "Racing Bulls", "Yuki Tsunoda": "Racing Bulls",
        "Sebastian Vettel": "Aston Martin", "Lance Stroll": "Aston Martin",
        "Alexander Albon": "Williams",  "Nicholas Latifi": "Williams",
        "Valtteri Bottas": "Audi",      "Zhou Guanyu": "Audi",
        "Kevin Magnussen": "Haas",      "Mick Schumacher": "Haas",
        "Nico Hulkenberg": "Haas",
    },
    2023: {
        "Max Verstappen": "Red Bull",   "Sergio Perez": "Red Bull",
        "Charles Leclerc": "Ferrari",   "Carlos Sainz": "Ferrari",
        "Lewis Hamilton": "Mercedes",   "George Russell": "Mercedes",
        "Lando Norris": "McLaren",      "Oscar Piastri": "McLaren",
        "Fernando Alonso": "Aston Martin", "Lance Stroll": "Aston Martin",
        "Pierre Gasly": "Alpine",       "Esteban Ocon": "Alpine",
        "Alexander Albon": "Williams",  "Logan Sargeant": "Williams",
        "Yuki Tsunoda": "Racing Bulls", "Daniel Ricciardo": "Racing Bulls",
        "Liam Lawson": "Racing Bulls",  "Nyck De Vries": "Racing Bulls",
        "Valtteri Bottas": "Audi",      "Zhou Guanyu": "Audi",
        "Nico Hulkenberg": "Haas",      "Kevin Magnussen": "Haas",
    },
    2024: {
        "Max Verstappen": "Red Bull",   "Sergio Perez": "Red Bull",
        "Lando Norris": "McLaren",      "Oscar Piastri": "McLaren",
        "Charles Leclerc": "Ferrari",   "Carlos Sainz": "Ferrari",
        "George Russell": "Mercedes",   "Lewis Hamilton": "Mercedes",
        "Fernando Alonso": "Aston Martin", "Lance Stroll": "Aston Martin",
        "Pierre Gasly": "Alpine",       "Esteban Ocon": "Alpine",
        "Yuki Tsunoda": "Racing Bulls", "Daniel Ricciardo": "Racing Bulls",
        "Liam Lawson": "Racing Bulls",  "Alexander Albon": "Williams",
        "Logan Sargeant": "Williams",   "Franco Colapinto": "Williams",
        "Valtteri Bottas": "Audi",      "Zhou Guanyu": "Audi",
        "Nico Hulkenberg": "Haas",      "Kevin Magnussen": "Haas",
        "Oliver Bearman": "Ferrari",
    },
    2025: {
        "Max Verstappen": "Red Bull",   "Yuki Tsunoda": "Red Bull",
        "Lando Norris": "McLaren",      "Oscar Piastri": "McLaren",
        "Charles Leclerc": "Ferrari",   "Lewis Hamilton": "Ferrari",
        "George Russell": "Mercedes",   "Kimi Antonelli": "Mercedes",
        "Fernando Alonso": "Aston Martin", "Lance Stroll": "Aston Martin",
        "Pierre Gasly": "Alpine",       "Jack Doohan": "Alpine",
        "Franco Colapinto": "Alpine",
        "Isack Hadjar": "Racing Bulls", "Liam Lawson": "Racing Bulls",
        "Alexander Albon": "Williams",  "Carlos Sainz": "Williams",
        "Nico Hulkenberg": "Audi",      "Gabriel Bortoleto": "Audi",
        "Oliver Bearman": "Haas",       "Esteban Ocon": "Haas",
    },
    2026: {
        "Max Verstappen": "Red Bull",   "Isack Hadjar": "Red Bull",
        "Lando Norris": "McLaren",      "Oscar Piastri": "McLaren",
        "Charles Leclerc": "Ferrari",   "Lewis Hamilton": "Ferrari",
        "George Russell": "Mercedes",   "Kimi Antonelli": "Mercedes",
        "Fernando Alonso": "Aston Martin", "Lance Stroll": "Aston Martin",
        "Pierre Gasly": "Alpine",       "Franco Colapinto": "Alpine",
        "Liam Lawson": "Racing Bulls",  "Arvid Lindblad": "Racing Bulls",
        "Carlos Sainz": "Williams",     "Alexander Albon": "Williams",
        "Nico Hulkenberg": "Audi",      "Gabriel Bortoleto": "Audi",
        "Esteban Ocon": "Haas",         "Oliver Bearman": "Haas",
        "Sergio Perez": "Cadillac",     "Valtteri Bottas": "Cadillac",
    },
}

# Sprint rounds now come from the authoritative schedule `sprint_date` field
# via data_fetcher.get_sprint_rounds() (with a verified static fallback inside
# data_fetcher). The old local table disagreed with every other source.


def _get_year_roster(year: int) -> dict[str, str]:
    """
    Resolve a season's lineup: auto-derive from standings when the API has it,
    fall back to the static table offline / pre-season. Raises KeyError only
    when NEITHER source can produce a roster.
    """
    try:
        derived = get_season_roster(year)
        if len(derived) >= 16:
            return derived
    except Exception:
        pass
    static = DRIVER_TEAMS_BY_YEAR.get(year)
    if static is None:
        raise KeyError(
            f"No roster for {year}: API derivation failed and no static "
            "entry in DRIVER_TEAMS_BY_YEAR"
        )
    print(f"  [roster] {year}: using static fallback table (API derivation unavailable)")
    return static


def backtest_race(
    predictor, year, round_num, race_name,
    drv_standings, ctor_standings, driver_form, ctor_reliability,
    is_sprint=False, year_roster=None,
    elo_system=None, ctor_elo=None,
):
    """Backtest a single race or sprint. Returns metrics dict or None."""
    actual_race  = get_race_results(year, round_num)
    actual_quali = get_qualifying_results(year, round_num)
    if not actual_race:
        return None

    roster = year_roster or _get_year_roster(year)
    circuit_cfg = _match_circuit_cfg(race_name)
    circuit_id  = circuit_cfg.get("key", "")
    circ_hist   = get_circuit_history(circuit_id, [year - 1, year - 2]) if circuit_id else pd.DataFrame()

    actual_race_pos  = {r["name"]: r["position"]  for r in actual_race}
    actual_quali_pos = {q["name"]: q["position"]  for q in (actual_quali or [])}
    actual_team      = {r["name"]: r["constructor"] for r in actual_race}

    all_drivers  = list(actual_race_pos.keys())

    # Build all feature vectors first, then predict the whole field at once.
    # Ranker outputs are unbounded margins — converting to per-field RANKS
    # puts them on the same scale as the RF output before blending (matches
    # predictor._ensemble_rank_predictions).
    feats, valid_drivers = [], []
    for drv in all_drivers:
        team = actual_team.get(drv, roster.get(drv, ""))
        if not team:
            continue
        feats.append(predictor._build_features(
            driver_name=drv,
            drv_standings=drv_standings,
            ctor_standings=ctor_standings,
            driver_form=driver_form,
            ctor_reliability=ctor_reliability,
            circuit_hist=circ_hist,
            circuit_cfg=circuit_cfg,
            weather={"summary_condition": "dry", "rain_risk": "low", "condition_enc": 0},
            roster=roster,
            elo=elo_system,
            ctor_elo=ctor_elo,
        ))
        valid_drivers.append(drv)

    def _field_ranks(raw_scores):
        order = np.argsort(np.argsort(-np.asarray(raw_scores, dtype=float)))
        return order.astype(float) + 1.0

    driver_preds, quali_preds = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if predictor._trained and getattr(predictor, "race_rf", None) is not None and feats:
            fs = predictor.scaler_race.transform(np.array(feats))
            s_rf  = np.asarray(predictor.race_rf.predict(fs), dtype=float)
            s_xgb = _field_ranks(predictor.race_xgb.predict(fs))
            s_lgb = _field_ranks(predictor.race_lgb.predict(fs))
            fq = predictor.scaler_quali.transform(np.array(feats))
            q_rf  = np.asarray(predictor.quali_rf.predict(fq), dtype=float)
            q_xgb = _field_ranks(predictor.quali_xgb.predict(fq))
            q_lgb = _field_ranks(predictor.quali_lgb.predict(fq))

        for i, drv in enumerate(valid_drivers):
            # Race prediction
            if predictor._trained and getattr(predictor, "race_rf", None) is not None and feats:
                if getattr(predictor, "race_meta_model", None) is not None:
                    meta_X = np.array([[s_rf[i], s_xgb[i], s_lgb[i]]])
                    pr = float(predictor.race_meta_model.predict(meta_X)[0])
                else:
                    pr = float((s_rf[i] + s_xgb[i] + s_lgb[i]) / 3.0)
            else:
                f  = driver_form.get(drv, {})
                dr = next((d for d in drv_standings if d["name"] == drv), None)
                pr = 0.4 * (dr["position"] if dr else 10) + 0.6 * f.get("avg_position", 10)
            driver_preds.append({"driver": drv, "pred_pos": float(pr)})

            # Qualifying prediction
            if predictor._trained and getattr(predictor, "quali_rf", None) is not None and feats:
                if getattr(predictor, "quali_meta_model", None) is not None:
                    meta_X_q = np.array([[q_rf[i], q_xgb[i], q_lgb[i]]])
                    pq = float(predictor.quali_meta_model.predict(meta_X_q)[0])
                else:
                    pq = float((q_rf[i] + q_xgb[i] + q_lgb[i]) / 3.0)
            else:
                pq = float(pr) * 0.95
            quali_preds.append({"driver": drv, "pred_quali": float(pq)})

    if not driver_preds:
        return None

    driver_preds.sort(key=lambda x: x["pred_pos"])
    for i, d in enumerate(driver_preds, 1):
        d["pred_rank"] = i
    quali_preds.sort(key=lambda x: x["pred_quali"])
    for i, q in enumerate(quali_preds, 1):
        q["pred_grid"] = i

    pred_rank_map = {d["driver"]: d["pred_rank"] for d in driver_preds}
    pred_grid_map = {q["driver"]: q["pred_grid"] for q in quali_preds}

    common = [d for d in all_drivers if d in pred_rank_map]
    if not common:
        return None

    y_act  = [actual_race_pos[d]          for d in common]
    y_pred = [pred_rank_map[d]            for d in common]
    y_act_q  = [actual_quali_pos.get(d, 11) for d in common]
    y_pred_q = [pred_grid_map.get(d, 11)   for d in common]

    race_mae    = mean_absolute_error(y_act, y_pred)
    race_rmse   = float(np.sqrt(mean_squared_error(y_act, y_pred)))
    race_rho, _ = spearmanr(y_act, y_pred)
    quali_mae   = mean_absolute_error(y_act_q, y_pred_q)
    quali_rho,_ = spearmanr(y_act_q, y_pred_q)

    pred_top3   = {d for d in common if pred_rank_map[d] <= 3}
    actual_top3 = {d for d in common if actual_race_pos[d] <= 3}
    pred_top5   = {d for d in common if pred_rank_map[d] <= 5}
    actual_top5 = {d for d in common if actual_race_pos[d] <= 5}

    pred_winner   = min(common, key=lambda d: pred_rank_map[d])
    actual_winner = min(common, key=lambda d: actual_race_pos[d])

    misses = sorted([
        {"driver": d, "actual": actual_race_pos[d],
         "predicted": pred_rank_map[d], "error": abs(actual_race_pos[d] - pred_rank_map[d])}
        for d in common
    ], key=lambda x: x["error"], reverse=True)[:5]

    return {
        "year":           year,
        "round":          round_num,
        "race_name":      race_name,
        "is_sprint":      is_sprint,
        "race_mae":       round(race_mae, 3),
        "race_rmse":      round(race_rmse, 3),
        "race_rho":       round(float(race_rho), 3),
        "quali_mae":      round(quali_mae, 3),
        "quali_rho":      round(float(quali_rho), 3),
        "winner_correct": pred_winner == actual_winner,
        "top3_hits":      len(pred_top3 & actual_top3),
        "top5_hits":      len(pred_top5 & actual_top5),
        "pred_winner":    pred_winner,
        "actual_winner":  actual_winner,
        "biggest_misses": misses,
    }


def _compute_form_asof(year: int, as_of_round: int) -> dict:
    """Current-season rolling form truncated at `as_of_round` — mirrors the
    per-round form the live predictor uses, with zero end-of-season leakage."""
    from engine.core.data_fetcher import compute_driver_form
    try:
        return compute_driver_form(year, num_races=5, until_round=as_of_round)
    except Exception:
        return {}


def run_backtest_year(year: int, predictor: F1Predictor, specific_rounds=None):
    """Run one year's backtest using the already-trained predictor."""
    schedule = get_season_schedule(year)
    today    = datetime.date.today()
    completed = [r for r in schedule if datetime.date.fromisoformat(r["date"]) < today]
    if specific_rounds:
        completed = [r for r in completed if r["round"] in specific_rounds]
    if not completed:
        console.print(f"  No completed races for {year}.")
        return []

    roster = _get_year_roster(year)
    sprints = get_sprint_rounds(year)   # authoritative: schedule sprint_date

    # Multi-season aggregates restricted to PRIOR seasons only; the current-year
    # component is computed per-round with as_of_round below (no lookahead).
    driver_form_prior      = compute_multiseason_driver_form(year, [year - 1, year - 2])
    ctor_reliability_prior = compute_multiseason_constructor_stats(year, [year - 1, year - 2])

    from engine.models.elo_ratings import EloRatingSystem, ConstructorEloSystem
    from engine.core.config import HISTORICAL_SEASONS
    year_elo_sys = EloRatingSystem()
    year_elo_sys.build_from_history([y for y in HISTORICAL_SEASONS if y < year], verbose=False)
    year_ctor_elo_sys = ConstructorEloSystem(year_elo_sys)
    year_ctor_elo = year_ctor_elo_sys.compute_constructor_ratings(roster)

    results = []
    
    table = Table(title=f"🏎️  {year} Season Backtest", box=box.ROUNDED, header_style="bold cyan", border_style="cyan")
    table.add_column("Rnd", justify="center", width=4)
    table.add_column("Type", justify="center", width=8)
    table.add_column("Race Name", width=30)
    table.add_column("MAE", justify="right", width=6)
    table.add_column("Spearman", justify="right", width=8)
    table.add_column("Top 3", justify="center", width=7)
    table.add_column("Top 5", justify="center", width=7)
    table.add_column("Winner", justify="center", width=8)

    with Live(table, console=console, refresh_per_second=4):
        for race in completed:
            rnd  = race["round"]
            name = race["name"]
            is_sprint = rnd in sprints
            type_str = "SPRINT" if is_sprint else "RACE"

            # ── POINT-IN-TIME data only (no end-of-season knowledge) ──
            # Standings as of the PREVIOUS round; round 1 uses the prior
            # season's final championship. The old code fetched season-FINAL
            # standings once and reused them for every round — direct leakage.
            prev_rnd = rnd - 1
            if prev_rnd >= 1:
                drv_standings  = get_driver_standings(year, prev_rnd)
                ctor_standings = get_constructor_standings(year, prev_rnd)
            else:
                drv_standings  = get_driver_standings(year - 1)
                ctor_standings = get_constructor_standings(year - 1)

            if rnd >= 2:
                # Current-season form truncated at the previous round
                curr_form   = _compute_form_asof(year, prev_rnd)
                curr_ctor   = None
                driver_form = {**driver_form_prior, **curr_form}
                ctor_reliability = ctor_reliability_prior
            else:
                driver_form = dict(driver_form_prior)
                ctor_reliability = ctor_reliability_prior
            
            r = backtest_race(
                predictor, year, rnd, name,
                drv_standings, ctor_standings, driver_form, ctor_reliability,
                is_sprint=is_sprint, year_roster=roster,
                elo_system=year_elo_sys, ctor_elo=year_ctor_elo,
            )
            if r:
                results.append(r)
                wicon = "[green]✓[/green]" if r["winner_correct"] else "✗"
                table.add_row(
                    str(rnd), type_str, name,
                    f"{r['race_mae']:.2f}",
                    f"{r['race_rho']:.2f}",
                    f"{r['top3_hits']}/3",
                    f"{r['top5_hits']}/5",
                    wicon
                )
            else:
                table.add_row(str(rnd), type_str, name, "-", "-", "-", "-", "No Data")

    return results


def summarise(all_results: list, label: str):
    if not all_results:
        return
    df = pd.DataFrame(all_results)
    
    table = Table(title=f"📊 SUMMARY — {label} ({len(df)} races)", box=box.ROUNDED, header_style="bold magenta", border_style="magenta")
    table.add_column("Metric", width=35)
    table.add_column("Value", justify="right", style="bold")
    
    table.add_row("Race MAE (avg pos off)", f"{df['race_mae'].mean():.3f}")
    table.add_row("Race RMSE", f"{df['race_rmse'].mean():.3f}")
    table.add_row("Spearman ρ (1.0=perfect)", f"[{'green' if df['race_rho'].mean() > 0.5 else 'white'}]{df['race_rho'].mean():.3f}[/]")
    table.add_row("Qualifying MAE", f"{df['quali_mae'].mean():.3f}")
    table.add_row("Qualifying Spearman ρ", f"{df['quali_rho'].mean():.3f}")
    p_correct = df['winner_correct'].mean()*100
    table.add_row("Winner Correct", f"[{'green' if p_correct > 40 else 'yellow'}]{p_correct:.1f}%[/]")
    table.add_row("Avg Top-3 hits per race", f"[{'green' if df['top3_hits'].mean() > 1.5 else 'white'}]{df['top3_hits'].mean():.2f} / 3[/]")
    table.add_row("Avg Top-5 hits per race", f"[{'green' if df['top5_hits'].mean() > 2.5 else 'white'}]{df['top5_hits'].mean():.2f} / 5[/]")
    
    if "is_sprint" in df.columns and df["is_sprint"].any():
        sprint_df  = df[df["is_sprint"]]
        regular_df = df[~df["is_sprint"]]
        table.add_section()
        if not sprint_df.empty:
            table.add_row("Sprint weekends[/] (MAE | ρ)", f"{sprint_df['race_mae'].mean():.2f} | {sprint_df['race_rho'].mean():.2f}")
        if not regular_df.empty:
            table.add_row("Regular weekends[/] (MAE | ρ)", f"{regular_df['race_mae'].mean():.2f} | {regular_df['race_rho'].mean():.2f}")
            
    console.print(table)
    
    all_misses = []
    for r in all_results:
        for m in r.get("biggest_misses", []):
            m2 = dict(m); m2["race"] = r["race_name"][:25]; all_misses.append(m2)
    all_misses.sort(key=lambda x: x["error"], reverse=True)
    
    miss_table = Table(title="Top 10 Prediction Misses", box=box.SIMPLE, show_header=True, header_style="bold red")
    miss_table.add_column("Race", width=25)
    miss_table.add_column("Driver", width=22)
    miss_table.add_column("Actual", justify="right")
    miss_table.add_column("Pred", justify="right")
    miss_table.add_column("Error", justify="right", style="bold red")
    for m in all_misses[:10]:
        miss_table.add_row(m['race'], m['driver'], str(m['actual']), str(m['predicted']), str(m['error']))
    console.print(miss_table)


def main():
    parser = argparse.ArgumentParser(description="F1 Backtest")
    parser.add_argument("--years",  type=int, nargs="+", default=[2024, 2025, 2026])
    parser.add_argument("--rounds", type=int, nargs="*", default=None)
    args = parser.parse_args()

    years = sorted(args.years)
    test_years = years

    train_up_to = min(years) - 1
    
    console.print(Panel(
        f"[bold cyan]F1 Prediction Backtester[/bold cyan]\n"
        f"Testing years: {years}\n"
        f"Training on: data up to {train_up_to}",
        border_style="cyan", padding=(1, 2)
    ))

    # Build predictor and train
    predictor = F1Predictor()

    # Train on seasons STRICTLY BEFORE the first test year via the explicit
    # parameter — no monkey-patching of config.HISTORICAL_SEASONS.
    train_seasons = [y for y in HISTORICAL_SEASONS if y <= train_up_to]
    predictor._driver_standings = []
    predictor._ctor_standings   = []
    predictor._driver_form      = {}
    predictor._ctor_reliability = {}
    predictor._circuit_history  = pd.DataFrame()
    predictor._circuit_config   = {}
    predictor._weather          = {"summary_condition": "dry", "rain_risk": "low", "condition_enc": 0}

    # Force UTF-8 stdout encoding on Windows legacy consoles to avoid cp1252 charmap errors with rich emojis
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    console.print(f"\n[bold]Training model on historical data (up to {train_up_to})...[/bold]")
    predictor.train(verbose=True, seasons=train_seasons)

    all_results = []
    for year in test_years:
        results = run_backtest_year(year, predictor, args.rounds)
        all_results.extend(results)
        if results:
            summarise(results, f"{year} Season")

    if len(test_years) > 1 and all_results:
        summarise(all_results, "All Years Combined")

    # Save CSV
    if all_results:
        df = pd.DataFrame(all_results)
        years_str = "_".join(map(str, test_years))
        out = f"output/backtest_{years_str}.csv"
        df.to_csv(out, index=False)
        console.print(f"\n[bold green]✓ Results saved → {out}[/bold green]")
    
    console.print()

if __name__ == "__main__":
    main()
