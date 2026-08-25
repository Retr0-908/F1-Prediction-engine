"""
price_tracker.py - F1 Fantasy Price & Ownership Movement Tracker

Tracks driver/constructor price changes and ownership percentage week-over-week.
Stores history in cache/price_history.json and cache/ownership_history.json.

Use:
  from price_tracker import record_prices, get_price_movements, get_ownership_trends

Price movements are surfaced as [UP] / [DN] / - indicators in the prices table,
helping identify sell-high and buy-low opportunities before chip decisions.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import CURRENT_SEASON

_PRICE_HISTORY_PATH     = Path(__file__).parent / "cache" / "price_history.json"
_OWNERSHIP_HISTORY_PATH = Path(__file__).parent / "cache" / "ownership_history.json"

# How many rounds of history to keep (rolling window)
_MAX_HISTORY_ROUNDS = 6


# ─────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────
# PRICE RECORDING
# ─────────────────────────────────────────────

def record_prices(
    round_num: int,
    driver_prices: dict,
    constructor_prices: dict,
    season: int = None,
) -> None:
    """
    Record the current prices for all drivers/constructors at this round.
    Keeps last _MAX_HISTORY_ROUNDS rounds of history.

    Args:
        round_num:          Current round number (e.g. 5)
        driver_prices:      {name: {"price": 12.5, ...}}
        constructor_prices: {name: {"price": 22.0, ...}}
        season:             Current season year
    """
    history = _load_json(_PRICE_HISTORY_PATH)
    if season is None:
        season = CURRENT_SEASON
    key = f"{season}_R{round_num:02d}"
    ts  = datetime.now().strftime("%Y-%m-%dT%H:%M")

    snapshot = {"timestamp": ts, "prices": {}}
    for name, info in driver_prices.items():
        snapshot["prices"][name] = round(float(info.get("price", 0)), 1)
    for name, info in constructor_prices.items():
        snapshot["prices"][name] = round(float(info.get("price", 0)), 1)

    history[key] = snapshot

    # Prune old entries - keep only last N rounds within this season
    season_keys = sorted([k for k in history if k.startswith(str(season))], reverse=True)
    for old_key in season_keys[_MAX_HISTORY_ROUNDS:]:
        del history[old_key]

    _save_json(_PRICE_HISTORY_PATH, history)


def record_ownership(
    round_num: int,
    driver_prices: dict,
    constructor_prices: dict,
    season: int = None,
) -> None:
    """
    Record current ownership percentages for all players.
    """
    history = _load_json(_OWNERSHIP_HISTORY_PATH)
    if season is None:
        season = CURRENT_SEASON
    key = f"{season}_R{round_num:02d}"
    ts  = datetime.now().strftime("%Y-%m-%dT%H:%M")

    snapshot = {"timestamp": ts, "ownership": {}}
    for name, info in driver_prices.items():
        own = info.get("ownership_pct", 0)
        if own is not None:
            snapshot["ownership"][name] = round(float(own), 1)
    for name, info in constructor_prices.items():
        own = info.get("ownership_pct", 0)
        if own is not None:
            snapshot["ownership"][name] = round(float(own), 1)

    history[key] = snapshot

    season_keys = sorted([k for k in history if k.startswith(str(season))], reverse=True)
    for old_key in season_keys[_MAX_HISTORY_ROUNDS:]:
        del history[old_key]

    _save_json(_OWNERSHIP_HISTORY_PATH, history)


# ─────────────────────────────────────────────
# PRICE MOVEMENT QUERY
# ─────────────────────────────────────────────

def get_price_movements(season: int = None) -> dict[str, dict]:
    """
    Returns per-player price movement data comparing latest two snapshots.

    Returns:
        {
          "Lando Norris": {
            "current": 22.5,
            "previous": 22.0,
            "change": +0.5,
            "trend": "[UP]",
            "change_3wk": +1.0,   # vs 3 rounds ago
            "arrow": "[UP] +0.5M",
          },
          ...
        }
    """
    history = _load_json(_PRICE_HISTORY_PATH)
    if season is None:
        season = CURRENT_SEASON
    season_keys = sorted(
        [k for k in history if k.startswith(str(season))],
        reverse=True  # latest first
    )

    if len(season_keys) < 1:
        return {}

    latest_snapshot = history[season_keys[0]]["prices"]
    prev_snapshot   = history[season_keys[1]]["prices"] if len(season_keys) >= 2 else {}
    wk3_snapshot    = history[season_keys[3]]["prices"] if len(season_keys) >= 4 else {}

    movements = {}
    for name, current_price in latest_snapshot.items():
        prev_price = prev_snapshot.get(name)
        wk3_price  = wk3_snapshot.get(name)

        change = round(current_price - prev_price, 1) if prev_price is not None else 0.0
        change_3wk = round(current_price - wk3_price, 1) if wk3_price is not None else None

        if change > 0:
            trend = "UP"
            arrow = f"+{change:.1f}M"
        elif change < 0:
            trend = "DN"
            arrow = f"{change:.1f}M"
        else:
            trend = "--"
            arrow = "--"

        movements[name] = {
            "current":    current_price,
            "previous":   prev_price,
            "change":     change,
            "trend":      trend,
            "arrow":      arrow,
            "change_3wk": change_3wk,
        }

    return movements


def get_ownership_trends(season: int = None) -> dict[str, dict]:
    """
    Returns per-player ownership trend comparing latest two snapshots.

    Returns:
        {
          "Lando Norris": {
            "current": 62.3,
            "change":  +5.1,
            "trend":   "[UP]",   # rising ownership = template risk
            "arrow":   "[UP] +5.1%",
          },
          ...
        }
    """
    history = _load_json(_OWNERSHIP_HISTORY_PATH)
    if season is None:
        season = CURRENT_SEASON
    season_keys = sorted(
        [k for k in history if k.startswith(str(season))],
        reverse=True
    )

    if len(season_keys) < 1:
        return {}

    latest = history[season_keys[0]]["ownership"]
    prev   = history[season_keys[1]]["ownership"] if len(season_keys) >= 2 else {}

    trends = {}
    for name, current_own in latest.items():
        prev_own = prev.get(name)
        change = round(current_own - prev_own, 1) if prev_own is not None else 0.0

        if change > 1.0:
            trend = "UP"
            arrow = f"+{change:.1f}%"
        elif change < -1.0:
            trend = "DN"
            arrow = f"{change:.1f}%"
        else:
            trend = "--"
            arrow = "--"

        trends[name] = {
            "current": current_own,
            "previous": prev_own,
            "change":   change,
            "trend":    trend,
            "arrow":    arrow,
        }
    return trends


# ─────────────────────────────────────────────
# SELL-HIGH / BUY-LOW DETECTOR
# ─────────────────────────────────────────────

def get_sell_high_candidates(
    season: int = None,
    price_rise_threshold: float = 0.5,
    ownership_rise_threshold: float = 5.0,
) -> list[dict]:
    """
    Identify drivers whose price AND ownership have risen - prime sell-high candidates.
    These are heavily-owned, recently-appreciated assets. Good to sell before a bad race.

    Returns sorted list of {name, price_change, ownership_change, sell_high_score}.
    """
    price_movs = get_price_movements(season)
    own_trends  = get_ownership_trends(season)

    candidates = []
    for name in price_movs:
        pm = price_movs[name]
        ot = own_trends.get(name, {})

        price_change = pm["change"]
        own_change   = ot.get("change", 0.0)

        if price_change >= price_rise_threshold and own_change >= ownership_rise_threshold:
            score = price_change * 2 + own_change * 0.1
            candidates.append({
                "name":           name,
                "price_change":   price_change,
                "own_change":     own_change,
                "current_price":  pm["current"],
                "current_own":    ot.get("current", 0),
                "sell_high_score": round(score, 2),
            })

    candidates.sort(key=lambda x: x["sell_high_score"], reverse=True)
    return candidates


def get_buy_low_candidates(
    season: int = None,
    price_drop_threshold: float = -0.3,
    ownership_drop_threshold: float = -3.0,
) -> list[dict]:
    """
    Identify drivers whose price AND ownership have dropped - potential buy-low.
    These are currently discounted but may recover. Good differential candidates.

    Returns sorted list of {name, price_change, ownership_change, buy_low_score}.
    """
    price_movs  = get_price_movements(season)
    own_trends   = get_ownership_trends(season)

    candidates = []
    for name in price_movs:
        pm = price_movs[name]
        ot = own_trends.get(name, {})

        price_change = pm["change"]
        own_change   = ot.get("change", 0.0)

        if price_change <= price_drop_threshold and own_change <= ownership_drop_threshold:
            score = abs(price_change) * 2 + abs(own_change) * 0.1
            candidates.append({
                "name":           name,
                "price_change":   price_change,
                "own_change":     own_change,
                "current_price":  pm["current"],
                "current_own":    ot.get("current", 0),
                "buy_low_score":  round(score, 2),
            })

    candidates.sort(key=lambda x: x["buy_low_score"], reverse=True)
    return candidates
