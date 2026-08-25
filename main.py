"""
main.py - F1 Fantasy Prediction Tool - CLI Entry Point
Run: python main.py [--set-cookie] [--refresh] [--no-train]

Workflow:
1. Detect next race
2. Fetch weather
3. Train ML model on historical data
4. Predict race & qualifying order
5. Scrape F1 Fantasy prices
6. Prompt for current team → suggest changes
7. Show optimal team
8. Save full report
"""

import os
import sys
import json
import time
import logging
import logging.handlers
import datetime
import argparse
import contextlib
from pathlib import Path

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Rich terminal output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.text import Text

# Interactive menus (arrow-key selection)
try:
    import questionary
    from questionary import Style as QStyle
    _HAS_QUESTIONARY = True
except ImportError:
    _HAS_QUESTIONARY = False

# HTML dashboard
from dashboard import generate_html_dashboard

from config import (
    CONSTRUCTORS_2025, TEAM_COLORS,
    CURRENT_SEASON,
)
from data_fetcher import get_next_race, is_sprint_weekend
from weather import get_race_weekend_weather, format_weather_display
from fantasy_scraper import (
    scrape_driver_prices, scrape_constructor_prices,
    fuzzy_match_driver,
)
from predictor import F1Predictor
from fantasy_optimizer import (
    get_current_team_value, suggest_team_changes,
    find_optimal_team, find_best_constructor,
    find_differential_picks, find_best_turbo_driver,
)
from monte_carlo import (
    simulate_race_weekend, format_mc_summary, get_expected_value_pts,
    DistributionStats,
)
from chip_advisor import (
    advise_chips,
    load_chip_state, save_chip_state, mark_chip_used, reset_chip_state,
    build_season_context_table, ALL_CHIPS, ChipScore,
)
from price_tracker import (
    record_prices, record_ownership,
    get_price_movements, get_ownership_trends,
    get_sell_high_candidates, get_buy_low_candidates,
)

console = Console()
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# LOGGING SETUP (rotating 10MB log file)
# ─────────────────────────────────────────────
_LOG_FILE = LOGS_DIR / "f1_predictor.log"
_MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 2  # keep 2 old log files

logger = logging.getLogger("f1_predictor")
logger.setLevel(logging.DEBUG)

_file_handler = logging.handlers.RotatingFileHandler(
    _LOG_FILE, maxBytes=_MAX_LOG_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))
logger.addHandler(_file_handler)

# Also log to stderr at WARNING+ so crashes are visible
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.WARNING)
_console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(_console_handler)


# Fast progressive print - gives a "live" feel without being slow
_PRINT_DELAY = 0.0  # Set to 0 to disable artificial slow printing for better UX

def _print_slow(*args, **kwargs):
    """Print with a tiny delay so output reveals progressively."""
    if _PRINT_DELAY > 0:
        time.sleep(_PRINT_DELAY)
    console.print(*args, **kwargs)

def pause_and_clear(args):
    """(Removed) No longer clears screen or pauses, as UI is now linear and clean."""
    pass

def section_heading(title: str, style: str = "bold white", rule_style: str = "dim"):
    """Print a clear section divider with a centred heading."""
    console.print()
    console.print(Rule(f"[{style}] {title} [{style}]", style=rule_style))
    console.print()
    logger.info(f"=== {title} ===")


# ─────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="F1 Fantasy Prediction Tool")
    p.add_argument("--refresh",     action="store_true", help="Force refresh of all cached data")
    p.add_argument("--no-train",    action="store_true", help="Skip ML training (use heuristics only)")
    p.add_argument("--race",        type=str,  default=None, help="Override race name")
    p.add_argument("--report-only", action="store_true", help="Print last saved report")
    p.add_argument("--auto",        action="store_true", help="Non-interactive mode (use defaults/cache)")
    p.add_argument(
        "--mode",
        choices=["pre-quali", "post-quali", "race-day"],
        default="pre-quali",
        help=(
            "Weekend mode: "
            "pre-quali = historical + FP2 pace (default); "
            "post-quali = + qualifying sector times + grid penalties; "
            "race-day = same as post-quali with refreshed weather"
        ),
    )
    p.add_argument("--sims",        type=int, default=1000, help="Monte Carlo simulation count (default 1000)")
    p.add_argument(
        "--league",
        action="store_true",
        help="League mode: compare your team vs template and maximise rank gain",
    )
    p.add_argument("--reset-chips", action="store_true", help="Reset chip state to all-unused (new season)")
    p.add_argument(
        "--prices-file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON file of manually-curated prices to use instead of the live scraper. "
            "Format: {\"driver_name\": {\"price\": 12.5, \"ownership\": 18.3}, ...}"
        ),
    )
    return p.parse_args()



# ─────────────────────────────────────────────
# TABLE HELPERS
# ─────────────────────────────────────────────
def _team_color(team: str) -> str:
    return TEAM_COLORS.get(team, "#FFFFFF")


def _medal(pos: int) -> str:
    return {1: "", 2: "", 3: ""}.get(pos, f"P{pos}")


# ─────────────────────────────────────────────
# DISPLAY: RACE PREDICTION
# ─────────────────────────────────────────────
def display_race_prediction(race_order: list[dict], quali_order: list[dict], top_n: int = 20):
    table = Table(
        title=" Predicted Race Finishing Order",
        box=box.ROUNDED, show_header=True, header_style="bold cyan",
        border_style="cyan",
    )
    table.add_column("Pos",       width=4,  justify="center")
    table.add_column("Driver",    width=20)
    table.add_column("Team",      width=16)
    table.add_column("Grid",      width=5,  justify="center")
    table.add_column("Δ Grid",    width=6,  justify="center")
    table.add_column("DNF%",      width=6,  justify="center")
    table.add_column("Confidence",width=10, justify="center")

    # Build grid lookup
    grid_map = {q["driver"]: q["predicted_grid"] for q in quali_order}

    for drv in race_order[:top_n]:
        pos   = drv["predicted_rank"]
        name  = drv["driver"]
        team  = drv["team"]
        grid  = grid_map.get(name, "-")
        delta = (grid - pos) if isinstance(grid, int) else 0
        delta_str = f"+{delta}" if delta > 0 else str(delta) if delta < 0 else "-"
        delta_color = "green" if delta > 0 else "red" if delta < 0 else "white"
        conf  = drv.get("confidence_pct", 50)
        conf_bar = "█" * int(conf / 10) + "░" * (10 - int(conf / 10))
        dnf_p = drv.get("dnf_prob_pct", 0)

        color = _team_color(team)
        pos_str = _medal(pos) if pos <= 3 else f"P{pos}"

        table.add_row(
            pos_str,
            f"[{color}]{name}[/{color}]",
            f"[{color}]■[/{color}] {team}",
            str(grid),
            f"[{delta_color}]{delta_str}[/{delta_color}]",
            f"[{'red' if dnf_p > 15 else 'yellow' if dnf_p > 8 else 'green'}]{dnf_p:.0f}%[/]",
            f"[dim]{conf_bar}[/dim] {conf:.0f}%",
        )
    console.print(table)


def display_qualifying_prediction(quali_order: list[dict]):
    table = Table(
        title="⏱️  Predicted Qualifying Order",
        box=box.SIMPLE, show_header=True, header_style="bold magenta",
        border_style="magenta",
    )
    table.add_column("Grid", width=4, justify="center")
    table.add_column("Driver", width=22)
    table.add_column("Team",   width=16)
    table.add_column("Q3",     width=4, justify="center")
    table.add_column("Conf",   width=8, justify="center")

    for q in quali_order[:20]:
        grid  = q["predicted_grid"]
        name  = q["driver"]
        team  = q["team"]
        q3    = "[green][+][/green]" if q.get("q3_likely") else ""
        conf  = q.get("confidence_pct", 50)
        color = _team_color(team)

        table.add_row(
            f"P{grid}",
            f"[{color}]{name}[/{color}]",
            f"[{color}]■[/{color}] {team}",
            q3,
            f"{conf:.0f}%",
        )
    console.print(table)


# ─────────────────────────────────────────────
# DISPLAY: PRICE TABLE
# ─────────────────────────────────────────────
def display_prices(driver_prices: dict, constructor_prices: dict):
    """Show all fetched driver and constructor prices so the user can verify."""
    # Driver prices table
    drv_table = Table(
        title="Driver Prices (from F1 Fantasy)",
        box=box.ROUNDED, show_header=True, header_style="bold green",
        border_style="green",
    )
    drv_table.add_column("#",      width=3, justify="center")
    drv_table.add_column("Driver", width=22)
    drv_table.add_column("Team",   width=16)
    drv_table.add_column("Price",  width=8, justify="right")
    drv_table.add_column("Own%",   width=7, justify="right")
    drv_table.add_column("Source", width=12)

    sorted_drivers = sorted(driver_prices.items(),
                            key=lambda x: x[1].get("price", 0), reverse=True)
    for i, (name, info) in enumerate(sorted_drivers, 1):
        if info.get("is_constructor"):
            continue
        team = info.get("team", "")
        color = _team_color(team)
        source = info.get("source", "?")
        src_color = "green" if source in ("playwright", "api_feed") else "yellow" if source == "estimate" else "white"
        drv_table.add_row(
            str(i),
            f"[{color}]{name}[/{color}]",
            f"[{color}]\u25a0[/{color}] {team}",
            f"[bold]${info.get('price', 0):.1f}M[/bold]",
            f"{info.get('ownership_pct', 0):.1f}%",
            f"[{src_color}]{source}[/{src_color}]",
        )
    _print_slow(drv_table)

    # Constructor prices table
    ctor_table = Table(
        title="Constructor Prices (from F1 Fantasy)",
        box=box.ROUNDED, show_header=True, header_style="bold blue",
        border_style="blue",
    )
    ctor_table.add_column("#",           width=3, justify="center")
    ctor_table.add_column("Constructor", width=18)
    ctor_table.add_column("Price",       width=8, justify="right")
    ctor_table.add_column("Own%",        width=7, justify="right")
    ctor_table.add_column("Source",      width=12)

    sorted_ctors = sorted(constructor_prices.items(),
                          key=lambda x: x[1].get("price", 0), reverse=True)
    for i, (name, info) in enumerate(sorted_ctors, 1):
        color = _team_color(name)
        source = info.get("source", "?")
        src_color = "green" if source in ("playwright", "api_feed") else "yellow" if source == "estimate" else "white"
        ctor_table.add_row(
            str(i),
            f"[{color}]{name}[/{color}]",
            f"[bold]${info.get('price', 0):.1f}M[/bold]",
            f"{info.get('ownership_pct', 0):.1f}%",
            f"[{src_color}]{source}[/{src_color}]",
        )
    _print_slow(ctor_table)

    # Total cost summary
    total_all_drivers = sum(v.get("price", 0) for v in driver_prices.values() if not v.get("is_constructor"))
    total_all_ctors = sum(v.get("price", 0) for v in constructor_prices.values())
    _print_slow(f"  [dim]Total market: {len(driver_prices)} drivers (${total_all_drivers:.1f}M) + {len(constructor_prices)} constructors (${total_all_ctors:.1f}M)[/dim]")


# ─────────────────────────────────────────────
# DISPLAY: FANTASY POINTS
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# DISPLAY: MONTE CARLO DISTRIBUTION TABLE
# ─────────────────────────────────────────────
def display_monte_carlo(mc_summary: list[dict], top_n: int = 15):
    table = Table(
        title=" Monte Carlo Risk & Upside Analysis",
        box=box.ROUNDED, show_header=True, header_style="bold magenta",
        border_style="magenta",
    )
    table.add_column("Driver",     width=22)
    table.add_column("Team",       width=16)
    table.add_column("Mean Pts",   width=9,  justify="right")
    table.add_column("±Std",       width=7,  justify="right")
    table.add_column("P10",        width=6,  justify="right")
    table.add_column("P90 ▲",      width=7,  justify="right")
    table.add_column("DNF%",       width=6,  justify="right")
    table.add_column("Top-3%",     width=7,  justify="right")

    for row in mc_summary[:top_n]:
        clr      = _team_color(row.get("team", ""))
        mean     = row["mean_pts"]
        std      = row["std_pts"]
        p10      = row["p10_pts"]
        p90      = row["p90_pts"]
        dnf_pct  = row["p_dnf"]
        top3_pct = row["p_top3"]
        mean_clr = "bold green" if mean >= 25 else "green" if mean >= 15 else "yellow" if mean >= 8 else "white"
        table.add_row(
            f"[{clr}]{row['driver']}[/{clr}]",
            f"[{clr}]■[/{clr}] {row.get('team', '')}",
            f"[{mean_clr}]{mean:.1f}[/{mean_clr}]",
            f"[dim]±{std:.1f}[/dim]",
            f"[red]{p10:.1f}[/red]",
            f"[bold green]{p90:.1f}[/bold green]",
            f"[{'red' if dnf_pct > 15 else 'yellow' if dnf_pct > 8 else 'green'}]{dnf_pct:.0f}%[/]",
            f"[cyan]{top3_pct:.0f}%[/cyan]",
        )
    console.print(table)
    console.print("  [dim]P10 = floor scenario · P90 = ceiling scenario · Mean = expected value[/dim]")


# ─────────────────────────────────────────────
# DISPLAY: DIFFERENTIAL PICKS
# ─────────────────────────────────────────────
def display_differential_picks(picks: list[dict]):
    if not picks:
        return
    table = Table(
        title=" Differential Picks (High EV · Low Ownership)",
        box=box.SIMPLE, show_header=True, header_style="bold yellow",
    )
    table.add_column("Driver",    width=22)
    table.add_column("Team",      width=16)
    table.add_column("Own%",      width=6,  justify="right")
    table.add_column("Price",     width=7,  justify="right")
    table.add_column("Exp Pts",   width=8,  justify="right")
    table.add_column("P90 ▲",    width=7,  justify="right")
    table.add_column("Diff Score",width=10, justify="right")

    for p in picks:
        clr = _team_color(p.get("team", ""))
        table.add_row(
            f"[{clr}]{p['driver']}[/{clr}]",
            f"[{clr}]■[/{clr}] {p.get('team', '')}",
            f"{p['ownership_pct']:.1f}%",
            f"${p['price']:.1f}M",
            f"[bold]{p['exp_pts']:.1f}[/bold]",
            f"[green]{p['p90_pts']:.1f}[/green]",
            f"[yellow]{p['differential_score']:.3f}[/yellow]",
        )
    console.print(table)
    console.print("  [dim]Differential Score = (EV × upside) / ownership - reward for going against the crowd[/dim]")


# ─────────────────────────────────────────────
# DISPLAY: SEASON CONTEXT TABLE
# ─────────────────────────────────────────────
def display_season_context(current_round: int, chip_state: dict):
    """Show all remaining races ranked by chip opportunity."""
    rows = build_season_context_table(current_round, chip_state)
    if not rows:
        console.print("  [dim]No remaining rounds found.[/dim]")
        return

    table = Table(
        title=" Remaining Races - Chip Opportunity Calendar",
        box=box.SIMPLE, show_header=True, header_style="bold white",
    )
    table.add_column("Rnd",     width=4,  justify="center")
    table.add_column("Circuit", width=28)
    table.add_column("Sprint",  width=7,  justify="center")
    table.add_column("SC%",     width=6,  justify="center")
    table.add_column("Value",   width=7,  justify="center")
    table.add_column("Best Chip",width=24)
    table.add_column("Notes",   width=20)

    for r in rows:
        sc_color  = "red" if r["sc_prob"] >= 0.65 else "yellow" if r["sc_prob"] >= 0.50 else "white"
        val_color = "bold green" if r["value_score"] >= 7 else "green" if r["value_score"] >= 5 else "white"
        sprint_str = "[yellow]⚡[/yellow]" if r["sprint"] else ""
        table.add_row(
            str(r["round"]),
            r["circuit"],
            sprint_str,
            f"[{sc_color}]{r['sc_prob']:.0%}[/{sc_color}]",
            f"[{val_color}]{r['value_score']:.1f}[/{val_color}]",
            r["best_chip"],
            f"[dim]{r['note']}[/dim]" if r["note"] else "",
        )
    console.print(table)


# ─────────────────────────────────────────────
# DISPLAY: CHIP ADVISOR v2
# ─────────────────────────────────────────────
def display_chip_advice_v2(chip_scores: list, turbo: dict):
    """Display ChipScore objects in a rich table."""
    table = Table(
        title=" Chip Strategy Advisor",
        box=box.ROUNDED, show_header=True, header_style="bold white",
        border_style="white",
    )
    table.add_column("Chip",        width=14)
    table.add_column("Decision",    width=12, justify="center")
    table.add_column("Score",       width=6,  justify="center")
    table.add_column("Reason",      width=42)
    table.add_column("Hold Until",  width=36)
    table.add_column("+EV",         width=7,  justify="right")

    urgency_colors = {"high": "bold red", "medium": "yellow", "low": "dim", "none": "dim"}

    for cs in chip_scores:
        urg_color = urgency_colors.get(cs.urgency, "white")

        if cs.already_used:
            decision = f"[dim][+] USED ({cs.used_on or ''})[/dim]"
            score_str = "-"
            hold_str  = ""
            ev_str    = ""
        elif cs.use_now:
            decision = f"[{urg_color}][+] PLAY NOW[/{urg_color}]"
            score_str = f"[bold green]{cs.raw_score:.1f}[/bold green]"
            hold_str  = ""
            ev_str    = f"[green]+{cs.expected_gain:.0f}[/green]"
        else:
            decision = f"[{urg_color}]⏳ HOLD[/{urg_color}]"
            score_str = f"{cs.raw_score:.1f}"
            hold_str  = f"[dim]{(cs.hold_until or '')[:34]}[/dim]"
            ev_str    = ""

        table.add_row(
            f"[bold]{cs.chip}[/bold]",
            decision,
            score_str,
            f"[dim]{cs.reason[:40]}[/dim]",
            hold_str,
            ev_str,
        )
    console.print(table)

    if turbo and turbo.get("driver"):
        risk_color = "red" if turbo["risk_level"] == "High" else "yellow" if turbo["risk_level"] == "Medium" else "green"
        console.print(
            f"\n   [bold cyan]Turbo Driver (2x Boost)[/bold cyan]: "
            f"[bold]{turbo['driver']}[/bold] "
            f"([dim]{turbo['reason'][:60]}[/dim])  "
            f"Expected bonus: [green]+{turbo['expected_bonus_pts']:.1f} pts[/green]  "
            f"Risk: [{risk_color}]{turbo['risk_level']}[/{risk_color}]"
        )


# Keep legacy alias for backwards compat
def display_chip_advice(chip_recs, turbo: dict):
    display_chip_advice_v2(chip_recs, turbo)


def display_fantasy_points(driver_pts: list[dict], ctor_pts: list[dict]):
    table = Table(
        title=" Projected F1 Fantasy Points",
        box=box.ROUNDED, show_header=True, header_style="bold yellow",
        border_style="yellow",
    )
    table.add_column("Driver",      width=22)
    table.add_column("Team",        width=16)
    table.add_column("Total Pts",   width=9,  justify="right")
    table.add_column("Qualifying",  width=9,  justify="right")
    table.add_column("Race",        width=7,  justify="right")
    table.add_column("Gained",      width=7,  justify="right")
    table.add_column("DNF Risk",    width=8,  justify="right")
    table.add_column("Fastest Lap", width=10, justify="right")

    sorted_pts = sorted(driver_pts, key=lambda x: x["total_pts"], reverse=True)
    for d in sorted_pts[:20]:
        b   = d.get("breakdown", {})
        clr = _team_color(d.get("team", ""))
        pts = d["total_pts"]
        pts_color = "bold green" if pts >= 30 else "green" if pts >= 20 else "yellow" if pts >= 10 else "white"
        table.add_row(
            f"[{clr}]{d['driver']}[/{clr}]",
            f"[{clr}]■[/{clr}] {d.get('team', '')}",
            f"[{pts_color}]{pts:.1f}[/{pts_color}]",
            f"{b.get('qualifying', 0):.1f}",
            f"{b.get('race_position', 0):.1f}",
            f"{b.get('positions_delta', 0):.1f}",
            f"[red]{b.get('dnf_risk', 0):.1f}[/red]",
            f"{b.get('fastest_lap', 0):.1f}",
        )
    console.print(table)

    # Constructor table
    ctor_table = Table(
        title=" Projected Constructor Fantasy Points",
        box=box.SIMPLE, header_style="bold blue",
    )
    ctor_table.add_column("Constructor", width=16)
    ctor_table.add_column("Total Pts",   width=9, justify="right")
    ctor_table.add_column("Driver Pts",  width=10, justify="right")
    ctor_table.add_column("Pit Stop",    width=9, justify="right")
    ctor_table.add_column("Drivers",     width=35)

    for c in sorted(ctor_pts, key=lambda x: x["total_pts"], reverse=True):
        clr = _team_color(c["constructor"])
        ctor_table.add_row(
            f"[{clr}]{c['constructor']}[/{clr}]",
            f"[bold green]{c['total_pts']:.1f}[/bold green]",
            f"{c.get('driver_pts', 0):.1f}",
            f"{c.get('pit_stop_pts', 0):.0f}",
            ", ".join(c.get("drivers", [])),
        )
    console.print(ctor_table)


# ─────────────────────────────────────────────
# DISPLAY: TEAM CHANGES
# ─────────────────────────────────────────────
def display_suggestions(result: dict):
    changes = result["suggested_changes"]
    if not changes:
        console.print("[yellow]No beneficial changes found within budget.[/yellow]")
        return

    table = Table(
        title=" Suggested Team Changes",
        box=box.ROUNDED, header_style="bold green", border_style="green",
    )
    table.add_column("#",         width=3,  justify="center")
    table.add_column("Type",      width=11)
    table.add_column("OUT →",     width=22)
    table.add_column("→ IN",      width=22)
    table.add_column("Pts Gain",  width=9,  justify="right")
    table.add_column("Cost Diff", width=10, justify="right")
    table.add_column("New Total", width=10, justify="right")

    for i, c in enumerate(changes, 1):
        cost_color = "red" if c["cost_diff"] > 0 else "green"
        table.add_row(
            str(i),
            c["type"].capitalize(),
            f"[red]{c['out']}[/red] [dim](${c['out_price']}M | {c['out_pts']:.0f}pts)[/dim]",
            f"[green]{c['in']}[/green] [dim](${c['in_price']}M | {c['in_pts']:.0f}pts)[/dim]",
            f"[bold green]+{c['pts_gain']:.1f}[/bold green]",
            f"[{cost_color}]{'+' if c['cost_diff']>0 else ''}{c['cost_diff']:.1f}M[/{cost_color}]",
            f"${c['new_total']:.1f}M",
        )
    console.print(table)

    console.print(
        f"\n  Current projected: [yellow]{result['projected_pts_current']:.1f} pts[/yellow]  →  "
        f"After changes: [bold green]{result['projected_pts_new']:.1f} pts[/bold green]  "
        f"([bold green]+{result['points_gain']:.1f} pts[/bold green])"
    )

    for w in result.get("warnings", []):
        console.print(f"\n  {w}")


def display_optimal_team(optimal: dict, best_constructors: list[dict]):
    table = Table(
        title="[*] Dream Team (Best Possible Lineup)",
        box=box.ROUNDED, header_style="bold cyan", border_style="cyan",
    )
    table.add_column("Type",       width=12)
    table.add_column("Player",     width=22)
    table.add_column("Team",       width=16)
    table.add_column("Proj Pts",   width=9, justify="right")
    table.add_column("Price",      width=8, justify="right")
    table.add_column("Value",      width=8, justify="right")

    for d in optimal["drivers"]:
        clr = _team_color(d["team"])
        table.add_row(
            "Driver",
            f"[{clr}]{d['name']}[/{clr}]",
            f"[{clr}]■[/{clr}] {d['team']}",
            f"[bold]{d['pts']:.1f}[/bold]",
            f"${d['price']:.1f}M",
            f"{d['value']:.2f}",
        )
    for c in optimal["constructors"]:
        clr = _team_color(c["name"])
        table.add_row(
            "[bold]Constructor[/bold]",
            f"[{clr}]{c['name']}[/{clr}]",
            "",
            f"[bold]{c['pts']:.1f}[/bold]",
            f"${c['price']:.1f}M",
            f"{c['value']:.2f}",
        )
    console.print(table)
    console.print(
        f"  Total: [bold cyan]{optimal['total_pts']:.1f} pts[/bold cyan]  |  "
        f"Cost: [yellow]${optimal['total_price']:.1f}M[/yellow]  |  "
        f"Budget remaining: [green]${optimal['budget_left']:.1f}M[/green]"
    )


# ─────────────────────────────────────────────
# INPUT HELPERS
# ─────────────────────────────────────────────
# ── Custom questionary style to match the dark F1 theme ──────────────────
_Q_STYLE = None
if _HAS_QUESTIONARY:
    _Q_STYLE = QStyle([
        ("qmark",        "fg:#e8002d bold"),
        ("question",     "fg:#e8e8f0 bold"),
        ("answer",       "fg:#00d084 bold"),
        ("pointer",      "fg:#e8002d bold"),
        ("highlighted",  "fg:#ffffff bg:#e8002d bold"),
        ("selected",     "fg:#00d084"),
        ("separator",    "fg:#2a2a35"),
        ("instruction",  "fg:#7a7a92"),
        ("text",         "fg:#e8e8f0"),
        ("disabled",     "fg:#2a2a35"),
    ])


def _driver_choice_label(name: str, prices: dict) -> str:
    """Build a rich display label for a driver checkbox."""
    info  = prices.get(name, {})
    price = info.get("price", 0)
    team  = info.get("team", "")
    own   = info.get("ownership_pct", 0)
    return f"{name:<24} ${price:4.1f}M  {team:<18} {own:.0f}% owned"


def prompt_current_team(
    driver_prices: dict,
    constructor_prices: dict,
    old_team: dict | None = None,
) -> tuple[list, list, int, float, float]:
    old_drivers = old_team.get("drivers", []) if old_team else []
    old_ctors   = old_team.get("constructors", []) if old_team else []
    old_budget  = old_team.get("budget_remaining", 0.0) if old_team else 0.0
    old_pts     = old_team.get("current_points", 0.0) if old_team else 0.0

    console.print()
    console.print(Panel(
        "[bold white]Select exactly [bold red]5 drivers[/bold red] and [bold red]2 constructors[/bold red].\n"
        "[dim]Use [bold][UP] [DN][/bold] to navigate, [bold]Space[/bold] to select, [bold]Enter[/bold] to confirm.[/dim]",
        title="[bold cyan]Your Current F1 Fantasy Team[/bold cyan]",
        border_style="cyan", padding=(0, 2),
    ))
    console.print()

    if _HAS_QUESTIONARY:
        # ── Arrow-key checklist for drivers ───────────────────────────────
        # Sort by price descending so premium picks are at top
        sorted_driver_names = sorted(
            [n for n in driver_prices if not driver_prices[n].get("is_constructor")],
            key=lambda n: driver_prices[n].get("price", 0),
            reverse=True,
        )
        drv_choices = [
            questionary.Choice(
                title=_driver_choice_label(n, driver_prices),
                value=n,
                checked=(n in old_drivers),
            )
            for n in sorted_driver_names
        ]

        drivers = []
        while len(drivers) != 5:
            selected = questionary.checkbox(
                "Pick your 5 DRIVERS:",
                choices=drv_choices,
                style=_Q_STYLE,
                validate=lambda ans: True if len(ans) == 5 else "Please select exactly 5 drivers",
            ).ask()
            if selected is None:
                raise SystemExit(0)  # Ctrl-C
            if len(selected) == 5:
                drivers = selected
            else:
                console.print(f"[red]  ✗ You selected {len(selected)} drivers - please pick exactly 5.[/red]")

        console.print(f"  [green][+] Drivers: {', '.join(drivers)}[/green]")
        console.print()

        # ── Arrow-key checklist for constructors ──────────────────────────
        sorted_ctor_names = sorted(
            list(constructor_prices.keys()),
            key=lambda n: constructor_prices[n].get("price", 0),
            reverse=True,
        )
        ctor_choices = [
            questionary.Choice(
                title=_driver_choice_label(n, constructor_prices),
                value=n,
                checked=(n in old_ctors),
            )
            for n in sorted_ctor_names
        ]

        constructors = []
        while len(constructors) != 2:
            selected = questionary.checkbox(
                "Pick your 2 CONSTRUCTORS:",
                choices=ctor_choices,
                style=_Q_STYLE,
                validate=lambda ans: True if len(ans) == 2 else "Please select exactly 2 constructors",
            ).ask()
            if selected is None:
                raise SystemExit(0)
            if len(selected) == 2:
                constructors = selected
            else:
                console.print(f"[red]  ✗ You selected {len(selected)} constructors - please pick exactly 2.[/red]")

        console.print(f"  [green][+] Constructors: {', '.join(constructors)}[/green]")
        console.print()

        # ── Transfers / budget via questionary text ───────────────────────
        t_raw = questionary.text(
            "Free transfers available:",
            default=str(1),
            style=_Q_STYLE,
            validate=lambda v: True if v.strip().isdigit() else "Enter a whole number",
        ).ask() or "1"
        transfers = int(t_raw.strip())

        b_raw = questionary.text(
            "Budget remaining in bank ($M):",
            default=f"{old_budget:.1f}",
            style=_Q_STYLE,
            validate=lambda v: True if _is_float(v) else "Enter a number e.g. 3.5",
        ).ask() or str(old_budget)
        budget_rem = float(b_raw.strip())

        p_raw = questionary.text(
            "Your current total points:",
            default=f"{old_pts:.0f}",
            style=_Q_STYLE,
            validate=lambda v: True if _is_float(v) else "Enter a number",
        ).ask() or str(old_pts)
        current_pts = float(p_raw.strip())

    else:
        # ── Fallback: original text prompt path ───────────────────────────
        console.print("[dim](questionary not available - using text prompts)[/dim]")
        all_driver_names = sorted(driver_prices.keys())
        cols = [all_driver_names[i::3] for i in range(3)]
        max_len = max(len(c) for c in cols)
        for row in range(max_len):
            parts = []
            for col in cols:
                if row < len(col):
                    parts.append(f"  {col[row]:<25}")
            console.print("".join(parts))
        console.print()

        drivers = []
        for i in range(1, 6):
            while True:
                def_val = old_drivers[i-1] if i-1 < len(old_drivers) else None
                name = Prompt.ask(f"  Driver {i}/5", default=def_val or "").strip()
                matched = fuzzy_match_driver(name, driver_prices)
                if matched:
                    console.print(f"    [green][+] {matched}[/green]")
                    drivers.append(matched)
                    break
                console.print(f"    [red]Not found: '{name}'[/red]")

        console.print()
        all_ctor_names = sorted(constructor_prices.keys())
        console.print("  " + "   ".join(all_ctor_names))
        constructors = []
        for i in range(1, 3):
            while True:
                def_val = old_ctors[i-1] if i-1 < len(old_ctors) else None
                name = Prompt.ask(f"  Constructor {i}/2", default=def_val or "").strip()
                matched = fuzzy_match_driver(name, constructor_prices)
                if matched:
                    console.print(f"    [green][+] {matched}[/green]")
                    constructors.append(matched)
                    break
                console.print(f"    [red]Not found: '{name}'[/red]")

        transfers  = int(Prompt.ask("\n  Free transfers available", default="1"))
        budget_rem = float(Prompt.ask("  Budget remaining in bank ($M)", default=str(old_budget)))
        current_pts = float(Prompt.ask("  Current total points", default=str(old_pts)))

    return drivers, constructors, transfers, budget_rem, current_pts


def _is_float(v: str) -> bool:
    try:
        float(v.strip())
        return True
    except ValueError:
        return False


# ─────────────────────────────────────────────
# REPORT SAVING
# ─────────────────────────────────────────────
def save_report(
    race: dict,
    weather: dict,
    race_order: list,
    quali_order: list,
    driver_pts: list,
    ctor_pts: list,
    suggestions: dict,
    optimal: dict,
) -> Path:
    report = {
        "generated_at":     datetime.datetime.now().isoformat(),
        "race":             race,
        "weather_summary":  {
            k: v for k, v in weather.items() if k != "race_day_hourly"
        },
        "qualifying_order": quali_order[:20],
        "race_order":       race_order[:20],
        "driver_pts":       sorted(driver_pts, key=lambda x: x["total_pts"], reverse=True),
        "constructor_pts":  sorted(ctor_pts, key=lambda x: x["total_pts"], reverse=True),
        "suggestions":      suggestions,
        "optimal_team":     optimal,
    }
    fname = OUTPUT_DIR / f"race_{race.get('round', 'X')}_{race.get('name', 'unknown').replace(' ', '_')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved: {fname}")
    return fname


def save_recommendations(
    race: dict,
    suggestions: dict,
    optimal: dict,
    best_ctors: list[dict],
    my_points: float,
    driver_prices: dict,
    constructor_prices: dict,
) -> Path:
    """
    Save a clean, human-readable recommendations file that's easy to
    open later and see exactly what changes to make.
    """
    r = race.get("round", "X")
    race_name = race.get("name", "Unknown")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []
    lines.append(f"F1 FANTASY RECOMMENDATIONS - Round {r}: {race_name}")
    lines.append(f"Generated: {ts}")
    lines.append("=" * 60)
    lines.append("")

    # Dream Team
    lines.append("DREAM TEAM (Best $100M Lineup)")
    lines.append("-" * 40)
    for d in optimal.get("drivers", []):
        lines.append(f"  DRV | {d['name']:25s} | ${d['price']:5.1f}M | {d['pts']:5.1f} pts | value: {d['value']}")
    for c in optimal.get("constructors", []):
        lines.append(f"  CTR | {c['name']:25s} | ${c['price']:5.1f}M | {c['pts']:5.1f} pts | value: {c['value']}")
    lines.append(f"  TOTAL: ${optimal.get('total_price', 0):.1f}M | {optimal.get('total_pts', 0):.1f} pts | Budget left: ${optimal.get('budget_left', 0):.1f}M")
    lines.append("")

    # Transfer Suggestions
    changes = suggestions.get("suggested_changes", [])
    lines.append("TRANSFER SUGGESTIONS")
    lines.append("-" * 40)
    if changes:
        for i, c in enumerate(changes, 1):
            sign = "+" if c['cost_diff'] > 0 else ""
            lines.append(
                f"  {i}. {c['type'].upper()}: {c['out']} (${c['out_price']}M, {c['out_pts']:.0f}pts)"
                f"  -->  {c['in']} (${c['in_price']}M, {c['in_pts']:.0f}pts)"
                f"  | +{c['pts_gain']:.1f}pts | {sign}{c['cost_diff']:.1f}M"
            )
        lines.append(f"")
        lines.append(f"  Current projected: {suggestions.get('projected_pts_current', 0):.1f} pts")
        lines.append(f"  After transfers:   {suggestions.get('projected_pts_new', 0):.1f} pts")
        lines.append(f"  Points gain:       +{suggestions.get('points_gain', 0):.1f} pts")
    else:
        lines.append("  No beneficial changes found.")
    lines.append("")

    # Weekend Projection
    lines.append("WEEKEND PROJECTION")
    lines.append("-" * 40)
    lines.append(f"  Starting points:          {my_points:.1f}")
    lines.append(f"  Projected (no transfers): {my_points + suggestions.get('projected_pts_current', 0):.1f}")
    lines.append(f"  Projected (best):         {my_points + suggestions.get('projected_pts_new', 0):.1f}")
    lines.append("")

    # Constructor Rankings
    lines.append("CONSTRUCTOR RANKINGS")
    lines.append("-" * 40)
    for i, c in enumerate(best_ctors, 1):
        lines.append(f"  {i:2d}. {c['constructor']:18s} | ${c['price']:5.1f}M | {c['total_pts']:6.1f} pts | value: {c['value']}")
    lines.append("")

    # Warnings
    for w in suggestions.get("warnings", []):
        lines.append(f"  WARNING: {w}")

    fname = OUTPUT_DIR / f"recommendations_R{r}_{race_name.replace(' ', '_')}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Recommendations saved: {fname}")
    return fname


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def check_grid_changes(current_roster: dict[str, str], console: Console):
    """Compare current roster with cached last_grid.json and alert on changes."""
    cache_path = Path("grid_cache.json")
    last_roster = {}
    
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                last_roster = json.load(f)
        except Exception:
            pass

    if not last_roster:
        # First time or error, just save and return
        with open(cache_path, "w") as f:
            json.dump(current_roster, f)
        return

    # Check for differences
    added = [d for d in current_roster if d not in last_roster]
    removed = [d for d in last_roster if d not in current_roster]
    changed = [d for d in current_roster if d in last_roster and current_roster[d] != last_roster[d]]

    if added or removed or changed:
        console.print("\n[bold yellow] GRID CHANGE DETECTED![/bold yellow]")
        if added:
            for d in added:
                console.print(f"  [green]+[/green] {d} joined {current_roster[d]}")
        if removed:
            for d in removed:
                console.print(f"  [red]-[/red] {d} left {last_roster.get(d, 'the grid')}")
        if changed:
            for d in changed:
                console.print(f"  [yellow]Δ[/yellow] {d} moved: {last_roster[d]} → {current_roster[d]}")
        
        # Update cache
        with open(cache_path, "w") as f:
            json.dump(current_roster, f)
        console.print("[dim]Updated grid_cache.json[/dim]\n")
    else:
        console.print("[dim][+] Driver grid is unchanged.[/dim]")

def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info(f"F1 Fantasy Prediction Tool started at {datetime.datetime.now().isoformat()}")
    logger.info(f"Args: refresh={args.refresh}, no_train={args.no_train}, auto={args.auto}, race={args.race}")

    # ══════════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════════
    console.print(Panel(
        "[bold red][ F1 FANTASY PREDICTION TOOL ][/bold red]\n"
        "[dim]Powered by Jolpica API · FastF1 · OpenWeatherMap · ML Ensemble[/dim]",
        border_style="red", padding=(0, 4),
    ))

    # ══════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════
    # NEXT RACE
    # ══════════════════════════════════════════════════════════
    section_heading("NEXT RACE", style="bold green", rule_style="green")

    with Progress(TextColumn("[cyan]{task.description}"), console=console) as prog:
        t = prog.add_task("Detecting next race...", total=None)
        race = get_next_race(CURRENT_SEASON)

    if not race:
        console.print("[red]Could not determine next race. Check season schedule.[/red]")
        return

    circuit_cfg = race.get("circuit_config", {})
    is_sprint = is_sprint_weekend(race)

    _print_slow(Panel(
        f"[bold white][RACE] {race['name']}[/bold white]\n"
        f"[cyan]Round {race['round']}  ·  {race['date']}[/cyan]\n"
        f"[dim]{race['locality']}, {race['country']}  ·  {race['circuit']}[/dim]"
        + ("\n[yellow]⚡ SPRINT WEEKEND[/yellow]" if is_sprint else ""),
        border_style="green", title="Next Race",
    ))

    # ══════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════
    # WEATHER
    # ══════════════════════════════════════════════════════════
    section_heading("RACE WEEKEND", style="bold blue", rule_style="blue")

    weather = get_race_weekend_weather(race["name"], race["date"])
    if "error" not in weather:
        _print_slow(Panel(
            format_weather_display(weather),
            title=f"Weather - {weather.get('city', race['name'])}",
            border_style="blue",
        ))
    else:
        _print_slow(f"[yellow]Weather unavailable: {weather.get('error')}[/yellow]")

    # Remove Section 2b completely (early chip advisor)
    # The user will see chip recommendations later, when they have full data.

    # ══════════════════════════════════════════════════════════
    # ML TRAINING
    # ══════════════════════════════════════════════════════════
    section_heading("ANALYSING DATA", style="bold magenta", rule_style="magenta")

    _print_slow(f"  [dim]Weekend mode: [bold]{args.mode}[/bold] · Monte Carlo sims: {args.sims}[/dim]")
    predictor = F1Predictor()

    if not args.no_train:
        with console.status("[bold]Training ML ensemble (cached)...[/bold]", spinner="dots"):
            predictor.train(verbose=False, force=args.refresh)
        _print_slow("  [green][+] Model ready[/green]")
        logger.info("ML training complete")


    # ══════════════════════════════════════════════════════════
    # MARKET PRICES
    # ══════════════════════════════════════════════════════════
    section_heading("MARKET PRICES", style="bold green", rule_style="green")

    with console.status("[bold]Fetching F1 Fantasy prices & roster...[/bold]", spinner="dots"):
        with suppress_stdout():
            driver_prices, dynamic_roster = scrape_driver_prices(force_refresh=args.refresh)
            constructor_prices = scrape_constructor_prices(force_refresh=args.refresh)
    
    _print_slow(f"  [green][+] Fetched {len(driver_prices)} driver prices, {len(constructor_prices)} constructor prices[/green]")
    logger.info(f"Fetched {len(driver_prices)} driver prices, {len(constructor_prices)} constructor prices")

    # Record prices & ownership for tracking
    try:
        record_prices(race["round"], driver_prices, constructor_prices)
        record_ownership(race["round"], driver_prices, constructor_prices)
    except Exception:
        pass

    # (Prices table is now displayed in the HTML Dashboard)
    # Show price movements and sell-high/buy-low
    try:
        price_movs   = get_price_movements()
        sell_highs   = get_sell_high_candidates()
        buy_lows     = get_buy_low_candidates()
        if sell_highs:
            sh_str = "  ".join(f"[red]{c['name'].split()[-1]} {c['current_price']:.1f}M ({'+' if c['price_change']>=0 else ''}{c['price_change']:.1f}M, +{c['own_change']:.0f}% own)[/red]" for c in sell_highs[:3])
            _print_slow(f"  [Up] [bold]Sell-High candidates:[/bold] {sh_str}")
        if buy_lows:
            bl_str = "  ".join(f"[green]{c['name'].split()[-1]} {c['current_price']:.1f}M ({c['price_change']:.1f}M, {c['own_change']:.0f}% own)[/green]" for c in buy_lows[:3])
            _print_slow(f"  [Down] [bold]Buy-Low candidates:[/bold]  {bl_str}")
    except Exception:
        pass

    # ------------------------------------------
    # Optionally override prices from a user-supplied JSON file (--prices-file)
    if getattr(args, "prices_file", None):
        import json as _json
        try:
            with open(args.prices_file, "r", encoding="utf-8") as _pf:
                _override = _json.load(_pf)
            _applied = 0
            for _name, _data in _override.items():
                # Match by exact name or last-name
                for _drv in list(driver_prices.keys()):
                    if _drv == _name or _drv.split()[-1] == _name.split()[-1]:
                        driver_prices[_drv]["price"] = float(_data.get("price", driver_prices[_drv].get("price", 0)))
                        if "ownership" in _data:
                            driver_prices[_drv]["ownership"] = float(_data["ownership"])
                        driver_prices[_drv]["source"] = "manual_file"
                        _applied += 1
                        break
            _print_slow(f"[bold cyan]   Prices file applied: {_applied}/{len(_override)} entries overridden[/bold cyan]")
            logger.info(f"prices_file override applied: {_applied} drivers")
        except Exception as _e:
            _print_slow(f"[bold red]  [!] --prices-file load failed: {_e}[/bold red]")
            logger.warning(f"prices_file override failed: {_e}")

    # Price quality check - warn prominently if >50% of prices are estimates
    sources = set(v.get("source", "?") for v in driver_prices.values())
    _estimate_count = sum(1 for v in driver_prices.values() if v.get("source") == "estimate")
    _estimate_pct   = (_estimate_count / max(len(driver_prices), 1)) * 100
    if _estimate_pct > 50:
        _warn_text = Text.assemble(
            ("[!]  PRICE WARNING  [!]\n\n", "bold yellow"),
            (f"{_estimate_count}/{len(driver_prices)} driver prices ({_estimate_pct:.0f}%) are ESTIMATES - ", "yellow"),
            ("the live scraper failed to fetch real data.\n\n", "yellow"),
            ("Recommendations and budget optimisation may be significantly inaccurate.\n", "white"),
            ("Options to fix this:\n", "white"),
            ("  1. Re-run with --refresh to force a fresh scrape\n", "green"),
            ("  2. Supply corrected prices: --prices-file prices.json\n", "green"),
            ("  3. Continue anyway (results may be unreliable)", "dim white"),
        )
        console.print(Panel(_warn_text, title="[bold red]Live Prices Unavailable[/bold red]",
                            border_style="red", padding=(1, 3)))
        if not getattr(args, "auto", False):
            try:
                input("\nPress Enter to continue anyway, or Ctrl+C to abort...  ")
            except (KeyboardInterrupt, EOFError):
                console.print("[red]Aborted.[/red]")
                raise SystemExit(1)
        logger.warning(f"Continuing with {_estimate_pct:.0f}% estimated prices")
    elif "estimate" in sources:
        _print_slow(f"[yellow][!]  {_estimate_count} price(s) are estimates (scraper partial failure)[/yellow]")
    else:
        _print_slow(f"[green][+] Live prices loaded ({len(driver_prices)} drivers)[/green]")

    # Check for mid-season changes
    check_grid_changes(dynamic_roster, console)

    # Re-load context with the dynamic roster to ensure predictions match current grid
    predictor.load_context(circuit_cfg, weather, roster=dynamic_roster,
                           mode=args.mode, race_info=race)

    # Show compact price summary so user can verify before overriding
    if not args.auto:
        price_table = Table(title="Current Fantasy Prices", show_header=True, header_style="bold cyan", box=None)
        price_table.add_column("Driver", style="white")
        price_table.add_column("Price", justify="right", style="green")
        price_table.add_column("   |   ", style="dim")
        price_table.add_column("Constructor", style="white")
        price_table.add_column("Price ", justify="right", style="green")

        sorted_drivers = sorted(driver_prices.items(), key=lambda x: x[1].get('price', 0), reverse=True)
        sorted_ctors = sorted(constructor_prices.items(), key=lambda x: x[1].get('price', 0), reverse=True)
        
        max_len = max(len(sorted_drivers), len(sorted_ctors))
        for i in range(max_len):
            d_name = sorted_drivers[i][0] if i < len(sorted_drivers) else ""
            d_price = f"${sorted_drivers[i][1].get('price', 0):.1f}M" if i < len(sorted_drivers) else ""
            
            c_name = sorted_ctors[i][0] if i < len(sorted_ctors) else ""
            c_price = f"${sorted_ctors[i][1].get('price', 0):.1f}M" if i < len(sorted_ctors) else ""
            
            price_table.add_row(d_name, d_price, "   |   ", c_name, c_price)
            
        console.print()
        console.print(price_table)

    # Prompt for manual price update while we are still in the MARKET PRICES section
    if not args.auto and Confirm.ask("\nDo you want to manually update any prices? (Useful if scraping missed changes)"):
        while True:
            target = Prompt.ask("Enter driver or constructor name to update (leave blank to finish)").strip()
            if not target:
                break
            matched_d = fuzzy_match_driver(target, driver_prices)
            matched_c = fuzzy_match_driver(target, constructor_prices)

            if matched_d:
                curr  = driver_prices[matched_d]['price']
                new_p = Prompt.ask(f"New price for {matched_d} (current: {curr}M)", default=str(curr))
                try:
                    driver_prices[matched_d]['price'] = float(new_p)
                    console.print(f"  [green]Updated {matched_d} to {float(new_p)}M[/green]")
                except ValueError:
                    console.print("  [red]Invalid price format.[/red]")
            elif matched_c:
                curr  = constructor_prices[matched_c]['price']
                new_p = Prompt.ask(f"New price for {matched_c} (current: {curr}M)", default=str(curr))
                try:
                    constructor_prices[matched_c]['price'] = float(new_p)
                    console.print(f"  [green]Updated {matched_c} to {float(new_p)}M[/green]")
                except ValueError:
                    console.print("  [red]Invalid price format.[/red]")
            else:
                console.print(f"  [red]Could not find '{target}'. Try again.[/red]")

    # ══════════════════════════════════════════════════════════
    # CRUNCHING NUMBERS (Predictions + Points + MC + Differentials)
    # ══════════════════════════════════════════════════════════
    pause_and_clear(args)
    section_heading("CRUNCHING NUMBERS", style="bold cyan", rule_style="cyan")

    with console.status("[bold]Generating race & qualifying predictions...[/bold]", spinner="dots"):
        race_order  = predictor.predict_finishing_order()
        quali_order = predictor.predict_qualifying_order()

    # Show mode-specific context info
    if args.mode in ("post-quali", "race-day"):
        grid_pens = predictor._grid_penalties
        if grid_pens:
            pen_str = "  ".join(f"{d}: {p:+d}" for d, p in grid_pens.items())
            _print_slow(f"  [yellow][!]  Grid penalties: {pen_str}[/yellow]")

    # (Predictions tables are now displayed in the HTML Dashboard)

    with console.status("[bold]Estimating F1 Fantasy points...[/bold]", spinner="dots"):
        sprint_order = predictor.predict_sprint_order() if is_sprint else None
        driver_pts = [
            predictor.estimate_fantasy_points(d["driver"], race_order, quali_order, is_sprint, sprint_order)
            for d in race_order
        ]
        ctor_pts = [
            predictor.estimate_constructor_points(ctor, driver_pts, is_sprint)
            for ctor in CONSTRUCTORS_2025
        ]
    
    # (Fantasy points table is now displayed in the HTML Dashboard)
    with console.status(f"[bold]Running {args.sims} Monte Carlo simulations...[/bold]", spinner="dots"):
        try:
            with suppress_stdout():
                mc_results = simulate_race_weekend(
                    race_order=race_order,
                    quali_order=quali_order,
                    circuit_config=circuit_cfg,
                    weather=weather,
                    n_simulations=args.sims,
                    is_sprint=is_sprint,
                    sprint_order=sprint_order,
                )
            mc_summary = format_mc_summary(mc_results)
            # display_monte_carlo(mc_summary) -> Moved to HTML dashboard / advanced log
            mc_ev_pts  = get_expected_value_pts(mc_results)
        except Exception as mc_err:
            _print_slow(f"[yellow]  Monte Carlo unavailable: {mc_err}[/yellow]")
            mc_results = {}
            mc_ev_pts  = {}

    # Finding differential picks
    try:
        with console.status("[bold]Finding differential picks...[/bold]", spinner="dots"):
            diff_picks = find_differential_picks(
                driver_pts, driver_prices, mc_results=mc_results, top_n=5
            )
        # display_differential_picks(diff_picks) -> Kept internally if needed, removed from main terminal flow
    except Exception as dp_err:
        _print_slow(f"[dim]Differential picks unavailable: {dp_err}[/dim]")

    # ══════════════════════════════════════════════════════════
    # YOUR TEAM
    # ══════════════════════════════════════════════════════════
    pause_and_clear(args)
    section_heading("YOUR TEAM", style="bold cyan", rule_style="cyan")

    # Shared with the web server via config.MY_TEAM_PATH — team edits in either
    # mode are visible in the other.
    from config import MY_TEAM_PATH
    team_cache_file = Path(MY_TEAM_PATH)
    auto_team = None
    if team_cache_file.exists():
        try:
            with open(team_cache_file, "r") as f:
                auto_team = json.load(f)
            if auto_team:
                console.print("\n[dim]Found previously saved team...[/dim]")
                console.print(f"  Drivers: {', '.join(auto_team.get('drivers', []))}")
                console.print(f"  Constructors: {', '.join(auto_team.get('constructors', []))}")
                console.print(f"  Budget: ${auto_team.get('budget_remaining', 0.0)}M  |  Points: {auto_team.get('current_points', 0.0)}")
        except Exception:
            pass

    my_points = 0.0
    if auto_team and (args.auto or Confirm.ask("\nUse this exact team?", default=True)):
        my_drivers      = auto_team.get("drivers", [])
        my_constructors = auto_team.get("constructors", [])
        budget_rem      = auto_team.get("budget_remaining", 0.0)
        my_points       = auto_team.get("current_points", 0.0)
        
        if args.auto:
            transfers = 1
            console.print("  [dim]Auto mode: Defaulting to 1 free transfer.[/dim]")
        else:
            transfers = int(Prompt.ask("  Free transfers available", default="1"))
    else:
        if auto_team:
            console.print("\n[dim]You can press Enter to keep your previous selections.[/dim]")
            
        my_drivers, my_constructors, transfers, budget_rem, my_points = prompt_current_team(
            driver_prices, constructor_prices, old_team=auto_team
        )
        # Save team
        try:
            team_cache_file.parent.mkdir(exist_ok=True)
            with open(team_cache_file, "w") as f:
                json.dump({
                    "drivers": my_drivers,
                    "constructors": my_constructors,
                    "budget_remaining": budget_rem,
                    "current_points": my_points
                }, f)
        except Exception as e:
            console.print(f"[red]Could not save team to cache: {e}[/red]")

    # Show current team value
    team_value = get_current_team_value(my_drivers, my_constructors, driver_prices, constructor_prices)
    console.print(Panel(
        f"  Drivers:      [cyan]{', '.join(my_drivers)}[/cyan]\n"
        f"  Constructors: [cyan]{', '.join(my_constructors)}[/cyan]\n"
        f"  Total Cost:   [yellow]${team_value['total_cost']:.1f}M[/yellow]  |  "
        f"Bank: [green]${budget_rem:.1f}M[/green]  |  "
        f"Free Transfers: [bold]{transfers}[/bold]",
        title="Your Current Team",
        border_style="cyan",
    ))

    # ══════════════════════════════════════════════════════════
    # TRANSFER SUGGESTIONS
    # ══════════════════════════════════════════════════════════
    section_heading("CALCULATING MOVES", style="bold green", rule_style="green")

    with console.status("[bold]Analysing optimal team changes (MC-weighted)...[/bold]", spinner="dots"):
        suggestions = suggest_team_changes(
            current_drivers=my_drivers,
            current_constructors=my_constructors,
            driver_pts=driver_pts,
            constructor_pts=ctor_pts,
            driver_prices=driver_prices,
            constructor_prices=constructor_prices,
            free_transfers=transfers,
            budget_remaining=budget_rem,
            mc_pts=mc_ev_pts if mc_ev_pts else None,
        )
    # (Suggestions and Projections are now displayed in the HTML Dashboard)
    if suggestions.get("using_mc_pts"):
        _print_slow("  [dim][MC] Transfer rankings powered by Monte Carlo expected values[/dim]")

    # ══════════════════════════════════════════════════════════
    # CHIP STRATEGY ADVISOR
    # ══════════════════════════════════════════════════════════
    section_heading("CHIP STRATEGY", style="bold white", rule_style="dim")
    try:
        chip_state = load_chip_state()
        chip_recs = advise_chips(
            race_info=race, circuit_config=circuit_cfg,
            weather=weather, driver_pts=driver_pts,
            mc_results=mc_results if mc_results else None,
            current_drivers=my_drivers, free_transfers=transfers,
            chip_state=chip_state,
        )
        turbo_rec = find_best_turbo_driver(
            current_drivers=my_drivers, driver_pts=driver_pts,
            mc_results=mc_results if mc_results else None,
        )
        display_chip_advice_v2(chip_recs, turbo_rec)

        # Prompt: mark a chip as played
        if not args.auto:
            playable = [cs for cs in chip_recs if cs.use_now and not cs.already_used]
            if playable:
                chip_names = [cs.chip for cs in playable]
                console.print(f"\n  Chips recommended for this weekend: [bold cyan]{', '.join(chip_names)}[/bold cyan]")
                if Confirm.ask("  Did you play a chip this weekend? (update tracker)", default=False):
                    chip_choice = Prompt.ask(
                        "  Which chip did you play?",
                        choices=chip_names + ["Other", "None"],
                        default="None",
                    )
                    if chip_choice not in ("None", "Other") and chip_choice in ALL_CHIPS:
                        mark_chip_used(chip_choice, race["round"], race.get("name", ""))
                        console.print(f"  [green][+] {chip_choice} marked as used for R{race['round']} {race.get('name', '')}[/green]")
                    elif chip_choice == "Other":
                        other_chip = Prompt.ask("  Chip name", choices=ALL_CHIPS)
                        mark_chip_used(other_chip, race["round"], race.get("name", ""))
                        console.print(f"  [green][+] {other_chip} marked as used[/green]")

    except Exception as chip_err:
        _print_slow(f"[dim]Chip advisor unavailable: {chip_err}[/dim]")

    # ── LEAGUE MODE ──────────────────────────────────────────────────────
    if args.league:
        section_heading("LEAGUE MODE", style="bold magenta", rule_style="magenta")
        try:
            # Template team = highest-ownership drivers (approximates average manager)
            template_drivers = sorted(
                driver_pts,
                key=lambda d: driver_prices.get(d["driver"], {}).get("ownership_pct", 0),
                reverse=True
            )[:5]
            template_names = [d["driver"] for d in template_drivers]
            template_pts = sum(d["total_pts"] for d in template_drivers)
            my_pts_total = sum(
                next((d["total_pts"] for d in driver_pts if d["driver"] == drv), 0)
                for drv in my_drivers
            )
            diff_drivers = [d for d in my_drivers if d not in template_names]
            template_only = [n for n in template_names if n not in my_drivers]

            console.print(Panel(
                f"  [bold]Template team (highest ownership):[/bold] {', '.join(template_names)}\n"
                f"  Template projected:  [yellow]{template_pts:.1f} pts[/yellow]\n"
                f"  Your team projected: [cyan]{my_pts_total:.1f} pts[/cyan]\n"
                f"  Differential (you have, template doesn't): [bold green]{', '.join(diff_drivers) or 'None'}[/bold green]\n"
                f"  Template only (not in your team):          [dim]{', '.join(template_only) or 'None'}[/dim]\n"
                f"  Pts advantage vs template: [{'+' if my_pts_total>=template_pts else '-'}]"
                f"[{'green' if my_pts_total>=template_pts else 'red'}]{my_pts_total-template_pts:+.1f} pts[/]",
                title="League Mode - vs Template", border_style="magenta",
            ))
            # Differential picks that the template doesn't own
            rank_picks = [d for d in diff_picks if d["driver"] not in template_names] if 'diff_picks' in dir() else []
            if rank_picks:
                _print_slow(f"  [!] Your differentials vs template: {', '.join(d['driver'] for d in rank_picks[:3])}")
        except Exception as le:
            _print_slow(f"  [dim]League mode error: {le}[/dim]")

    # ══════════════════════════════════════════════════════════
    # SECTION 9: CONSTRUCTOR RANKINGS
    # ══════════════════════════════════════════════════════════
    # (Constructor rankings are now displayed in the HTML Dashboard)
    best_ctors = find_best_constructor(ctor_pts, constructor_prices)
    # ══════════════════════════════════════════════════════════
    # DREAM TEAM
    # ══════════════════════════════════════════════════════════
    section_heading("DREAM TEAM", style="bold yellow", rule_style="yellow")

    with console.status("[bold]Computing Dream Team (MC-weighted)...[/bold]", spinner="dots"):
        optimal = find_optimal_team(
            driver_pts, ctor_pts, driver_prices, constructor_prices,
            mc_pts=mc_ev_pts if mc_ev_pts else None,
        )
        
        # Turbo driver for dream team
        turbo_dream = find_best_turbo_driver(
            [d["name"] for d in optimal.get("drivers", [])],
            driver_pts, mc_results=mc_results if mc_results else None,
        )

    if optimal.get("using_mc_pts"):
        _print_slow("  [dim][MC] Dream Team selected using Monte Carlo Expected Values[/dim]")

    if turbo_dream.get("driver"):
        _print_slow(
            f"  [TURBO] Turbo pick: [bold cyan]{turbo_dream['driver']}[/bold cyan]  "
            f"[dim](+{turbo_dream['expected_bonus_pts']:.1f} pts · {turbo_dream['risk_level']} risk)[/dim]"
        )

    # Print Dream Team nicely
    dream_table = Table(title="[bold yellow]🏆 DREAM TEAM (Best $100M Lineup) 🏆[/bold yellow]", show_header=True, header_style="bold cyan", box=None)
    dream_table.add_column("Type", justify="right", style="dim")
    dream_table.add_column("Name", style="white")
    dream_table.add_column("Price", justify="right", style="green")
    dream_table.add_column("Projected Pts", justify="right", style="cyan")
    
    for d in optimal.get("drivers", []):
        is_turbo = turbo_dream.get("driver") == d["name"]
        t_str = " [yellow](TURBO)[/yellow]" if is_turbo else ""
        pts_val = d['pts'] * 2.0 if is_turbo else d['pts']
        dream_table.add_row("DRIVER", f"{d['name']}{t_str}", f"${d['price']:.1f}M", f"{pts_val:.1f}")
        
    for c in optimal.get("constructors", []):
        dream_table.add_row("CONSTRUCTOR", c["name"], f"${c['price']:.1f}M", f"{c['pts']:.1f}")
        
    dream_table.add_row("", "", "", "")
    dream_table.add_row("", "[bold]TOTAL[/bold]", f"[bold green]${optimal.get('total_price', 0):.1f}M[/bold green]", f"[bold cyan]{optimal.get('total_pts', 0):.1f}[/bold cyan]")
    
    console.print()
    console.print(dream_table)
    console.print(f"  [dim]Budget left: ${optimal.get('budget_left', 0):.1f}M[/dim]")

    logger.info(f"Dream team: cost=${optimal['total_price']}M, pts={optimal['total_pts']}, budget_left=${optimal['budget_left']}M")


    # ══════════════════════════════════════════════════════════
    # DONE - SAVE REPORTS & OPEN DASHBOARD
    # ══════════════════════════════════════════════════════════
    section_heading("DONE", style="bold white", rule_style="dim")

    report_path = save_report(race, weather, race_order, quali_order,
                              driver_pts, ctor_pts, suggestions, optimal)
    _print_slow(f"[dim]  JSON report saved: {report_path}[/dim]")

    # Keep the plain-text file as a quick fallback
    reco_path = save_recommendations(
        race, suggestions, optimal, best_ctors,
        my_points, driver_prices, constructor_prices,
    )
    _print_slow(f"[dim]  Text summary saved: {reco_path}[/dim]")

    # ── Phase 1: HTML Dashboard ──────────────────────────────────────────
    try:
        dash_path = generate_html_dashboard(
            race=race,
            weather=weather,
            race_order=race_order,
            quali_order=quali_order,
            driver_pts=driver_pts,
            ctor_pts=ctor_pts,
            suggestions=suggestions,
            optimal=optimal,
            best_ctors=best_ctors,
            my_drivers=my_drivers,
            my_constructors=my_constructors,
            my_points=my_points,
            driver_prices=driver_prices,
            constructor_prices=constructor_prices,
            output_dir=OUTPUT_DIR,
            mc_results=mc_results,
        )
        _print_slow(f"[bold green]  [+] HTML Dashboard saved: {dash_path}[/bold green]")
        logger.info(f"HTML dashboard saved: {dash_path}")

        # Auto-open the dashboard in the default browser
        if not getattr(args, "auto", False):
            import webbrowser
            webbrowser.open(dash_path.as_uri())
            _print_slow("  [dim]Dashboard opened in your browser.[/dim]")
    except Exception as _dash_err:
        _print_slow(f"[yellow]  [!]  Dashboard generation failed: {_dash_err}[/yellow]")
        logger.warning(f"Dashboard error: {_dash_err}")
        dash_path = reco_path  # fallback

    logger.info(f"Run completed successfully at {datetime.datetime.now().isoformat()}")

    console.print()
    console.print(Panel(
        Text.assemble(
            ("[+]  Analysis complete!\n\n", "bold green"),
            ("  Dashboard:  ", "dim"),
            (str(dash_path) + "\n", "bold white"),
            ("  JSON report: ", "dim"),
            (str(report_path) + "\n", "white"),
            ("  Log file:    ", "dim"),
            (str(_LOG_FILE) + "\n\n", "white"),
            ("  Good luck with your F1 Fantasy team this weekend! ", "bold cyan"),
        ),
        title="[bold green]Done[/bold green]",
        border_style="green",
        padding=(1, 3),
    ))


if __name__ == "__main__":
    main()
