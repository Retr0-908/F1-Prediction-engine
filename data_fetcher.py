"""
data_fetcher.py — Fetches historical F1 data from Jolpica (Ergast successor) + OpenF1 + FastF1
All responses are cached to disk to avoid unnecessary API calls.

Upgrades in this version:
  - compute_driver_form() now uses EWMA weighting + momentum_trend slope
  - compute_multiseason_driver_form() uses SEASON_WEIGHTS from config (reg-era aware)
  - compute_practice_pace() — FastF1 FP2 long-run pace delta (new)
  - get_qualifying_sector_times() — FastF1 Q sector time deltas (new)
  - get_grid_penalties() — detects qualifying vs grid position deltas (new)

Phase 2 additions (ML pipeline data engineering):
  - get_tire_stints() — FastF1 lap-by-lap tire compound, stint length, track temp (cached)
  - normalize_telemetry() — aligns driver telemetry streams to a unified 0.5-second grid
  - get_session_feature_weights() — empirically-derived OLR weights for session importance
    (Qualifying ~4x stronger predictor than any practice session per academic research)
"""

import os
import json
import time
import datetime
import requests
import fastf1
import pandas as pd
import numpy as np
import logging

# Silence FastF1 and requests_cache verbose warnings about limits and missing telemetry
logging.getLogger("fastf1").setLevel(logging.CRITICAL)
logging.getLogger("requests_cache").setLevel(logging.CRITICAL)

# Disable FastF1's strict client-side rate limiter (500 calls/hour) to allow full cache rebuilds of historical seasons.
# This is safe because our downloads are naturally throttled and we are downloading static timing files.
try:
    import fastf1.req
    fastf1.req._SessionWithRateLimiting._RATE_LIMITS = {}
except Exception as e:
    pass
from pathlib import Path
from typing import Optional

from config import (
    JOLPICA_BASE, OPENF1_BASE, CURRENT_SEASON,
    HISTORICAL_SEASONS, FASTF1_CACHE_DIR, CIRCUITS, SEASON_WEIGHTS,
)

# ─────────────────────────────────────────────
# CACHE SETUP
# ─────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "cache" / "api"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FF1_CACHE = Path(FASTF1_CACHE_DIR)
FF1_CACHE.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(FF1_CACHE))


def _cache_path(key: str) -> Path:
    safe = key.replace("/", "_").replace("?", "_").replace("&", "_")
    for char in [":", "\"", "{", "}", " ", "'"]:
        safe = safe.replace(char, "")
    return CACHE_DIR / f"{safe}.json"


def _load_cache(key: str, max_age_hours: float = 6, allow_stale: bool = False) -> Optional[dict]:
    p = _cache_path(key)
    if p.exists():
        if max_age_hours <= 0 or allow_stale:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        age = time.time() - p.stat().st_mtime
        if age < max_age_hours * 3600:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def _save_cache(key: str, data) -> None:
    with open(_cache_path(key), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ─────────────────────────────────────────────
# GLOBAL REQUEST THROTTLE
# ─────────────────────────────────────────────
_last_request_time: float = 0.0
# Jolpica allows ~30 req/min; 2.1s gap keeps us safely under that limit
_MIN_REQUEST_GAP = 2.1   # seconds between requests


# ─────────────────────────────────────────────
# NAME / ID NORMALISERS
# These collapse API name variants to the canonical names used in config.py
# ─────────────────────────────────────────────

# Some drivers have multi-word first names in the Jolpica API (e.g. "Andrea Kimi")
# but are known by a preferred forename everywhere else.
_PREFERRED_FORENAME = {
    "Andrea Kimi": "Kimi",   # Andrea Kimi Antonelli → Kimi Antonelli
}

def _normalize_driver_name(given: str, family: str) -> str:
    """Return the canonical driver name used across config.py / fantasy prices."""
    import unicodedata
    given_norm = _PREFERRED_FORENAME.get(given, given)
    name = f"{given_norm} {family}"
    return "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    )


# Circuit IDs returned by the Jolpica API differ from the keys we cache under.
_CIRCUIT_ID_ALIASES = {
    "canada":        "villeneuve",   # Circuit Gilles Villeneuve
    "albert_park":   "albert_park",
    # add more as needed
}

def _normalize_circuit_id(circuit_id: str) -> str:
    """Map Jolpica circuitId to the correct ID for circuit results lookups."""
    return _CIRCUIT_ID_ALIASES.get(circuit_id, circuit_id)


# Constructor name variants across API → canonical name in config DRIVER_TEAMS_2026
_CONSTRUCTOR_NAME_ALIASES = {
    "RB F1 Team":       "Racing Bulls",
    "AlphaTauri":       "Racing Bulls",
    "Toro Rosso":       "Racing Bulls",
    "Alfa Romeo":       "Audi",       # Sauber → Audi, closest mapping
    "Sauber":           "Audi",
    "Haas F1 Team":     "Haas",
    "Alpine F1 Team":   "Alpine",
    "Aston Martin":     "Aston Martin",
    "Red Bull":         "Red Bull",
    "McLaren":          "McLaren",
    "Ferrari":          "Ferrari",
    "Mercedes":         "Mercedes",
    "Williams":         "Williams",
    "Cadillac F1 Team": "Cadillac",
}

def _normalize_constructor_name(name: str) -> str:
    """Map Jolpica constructor name to canonical name used in config."""
    return _CONSTRUCTOR_NAME_ALIASES.get(name, name)


def _throttle() -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_REQUEST_GAP:
        time.sleep(_MIN_REQUEST_GAP - elapsed)
    _last_request_time = time.time()


def _jolpica_get(endpoint: str, params: dict = None, cache_hours: float = 6) -> dict:
    cache_key = f"jolpica_{endpoint}_{json.dumps(params or {})}"
    is_historical = any(
        f"/{y}/" in endpoint or endpoint.endswith(f"/{y}") or endpoint.startswith(f"/{y}")
        for y in range(2019, CURRENT_SEASON)   # covers 2019-2025
    )
    effective_cache_hours = 0 if is_historical else cache_hours

    cached = _load_cache(cache_key, effective_cache_hours)
    if cached:
        # If we have a permanent cache (hours <= 0) but it's an empty response, ignore it and re-fetch
        total_str = cached.get("MRData", {}).get("total", "1")
        if str(total_str) == "0" and effective_cache_hours <= 0:
            pass
        else:
            return cached

    url = f"{JOLPICA_BASE}{endpoint}"
    last_error = None
    for attempt in range(4):
        try:
            _throttle()
            # connect=10s, read=60s — Jolpica can be slow under load
            resp = requests.get(url, params=params or {}, timeout=(10, 60))
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 0))
                # Start at 5s (not 2s) — Jolpica 429s clear quickly but not instantly
                wait = max(retry_after, 5 * (2 ** attempt))
                print(f"    [rate limit 429] waiting {wait}s (attempt {attempt + 1}/4)...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            
            # Avoid permanently caching empty responses (e.g. race results before race finishes)
            total_str = data.get("MRData", {}).get("total", "1")
            if str(total_str) == "0" and effective_cache_hours <= 0:
                pass # Do not cache empty responses permanently
            else:
                _save_cache(cache_key, data)
                
            return data
        except requests.exceptions.HTTPError as e:
            last_error = e
            if resp.status_code not in (429, 500, 502, 503, 504):
                break
            wait = 2 ** attempt * 2
            time.sleep(wait)
        except requests.exceptions.RequestException as e:
            last_error = e
            time.sleep(2 ** attempt)
            continue
            
    stale = _load_cache(cache_key, 0, allow_stale=True)
    if stale:
        print(f"    [WARN] Jolpica server error, using stale cache for {endpoint}")
        return stale
    raise RuntimeError(f"Jolpica API failed after 4 attempts for {endpoint}: {last_error}")


def _openf1_get(endpoint: str, params: dict = None, cache_hours: float = 6) -> list:
    cache_key = f"openf1_{endpoint}_{json.dumps(params or {})}"
    cached = _load_cache(cache_key, cache_hours)
    if cached:
        return cached

    url = f"{OPENF1_BASE}{endpoint}"
    last_error = None
    for attempt in range(4):
        try:
            _throttle()
            # connect=10s, read=60s — OpenF1 can be slow under load
            resp = requests.get(url, params=params or {}, timeout=(10, 60))
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 0))
                wait = max(retry_after, 2 ** attempt * 2)
                print(f"    [OpenF1 rate limit 429] waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            _save_cache(cache_key, data)
            return data
        except requests.exceptions.HTTPError as e:
            last_error = e
            if resp.status_code not in (429, 500, 502, 503, 504):
                break
            time.sleep(2 ** attempt * 2)
        except requests.exceptions.RequestException as e:
            last_error = e
            time.sleep(2 ** attempt)
            
    stale = _load_cache(cache_key, 0, allow_stale=True)
    if stale:
        print(f"    [WARN] OpenF1 server error, using stale cache for {endpoint}")
        return stale
    raise RuntimeError(f"OpenF1 API failed after 4 attempts for {endpoint}: {last_error}")


# ─────────────────────────────────────────────
# SCHEDULE
# ─────────────────────────────────────────────
def get_season_schedule(year: int = CURRENT_SEASON) -> list[dict]:
    data = _jolpica_get(f"/{year}/races.json", cache_hours=12)
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    result = []
    for r in races:
        result.append({
            "round":    int(r["round"]),
            "name":     r["raceName"],
            "date":     r["date"],
            "time":     r.get("time", ""),
            "circuit":  r["Circuit"]["circuitName"],
            "locality": r["Circuit"]["Location"]["locality"],
            "country":  r["Circuit"]["Location"]["country"],
            "circuit_id": r["Circuit"]["circuitId"],
            "fp1_date": r.get("FirstPractice", {}).get("date", ""),
            "fp1_time": r.get("FirstPractice", {}).get("time", ""),
            "fp2_date": r.get("SecondPractice", {}).get("date", ""),
            "fp2_time": r.get("SecondPractice", {}).get("time", ""),
            "fp3_date": r.get("ThirdPractice", {}).get("date", ""),
            "fp3_time": r.get("ThirdPractice", {}).get("time", ""),
            "quali_date": r.get("Qualifying", {}).get("date", ""),
            "quali_time": r.get("Qualifying", {}).get("time", ""),
            "sprint_date": r.get("Sprint", {}).get("date", ""),
            "sprint_time": r.get("Sprint", {}).get("time", ""),
        })
    return result


def is_sprint_weekend(race: dict) -> bool:
    """True if the race schedule is verified as a sprint weekend."""
    round_num = race.get("round", 0)
    date_str = race.get("date", "")
    year = int(date_str[:4]) if date_str else 2026
    
    if year == 2026:
        return round_num in {2, 4, 5, 9, 12, 16}
    elif year == 2025:
        return round_num in {2, 6, 9, 13, 19, 22}
    elif year == 2024:
        return round_num in {2, 5, 13, 18, 19, 21}

    # Fallback to API check
    sprint_date = race.get("sprint_date", "")
    return bool(sprint_date and sprint_date.strip())


_sprint_rounds_cache: dict[int, set[int]] = {}

def get_sprint_rounds(year: int = CURRENT_SEASON) -> set[int]:
    """Get the set of sprint round numbers for the given season (cached)."""
    global _sprint_rounds_cache
    if year not in _sprint_rounds_cache:
        try:
            schedule = get_season_schedule(year)
            # Filter rounds where is_sprint_weekend is true
            rounds = {r["round"] for r in schedule if is_sprint_weekend(r)}
            if rounds:
                _sprint_rounds_cache[year] = rounds
            else:
                raise ValueError("No sprint rounds found in schedule")
        except Exception:
            # Fallback to defaults if API fails or when starting up
            if year == 2026:
                _sprint_rounds_cache[year] = {2, 4, 5, 9, 12, 16}
            elif year == 2025:
                _sprint_rounds_cache[year] = {2, 6, 9, 13, 19, 22}
            elif year == 2024:
                _sprint_rounds_cache[year] = {2, 5, 13, 18, 19, 21}
            else:
                _sprint_rounds_cache[year] = set()
    return _sprint_rounds_cache[year]



def get_next_race(year: int = CURRENT_SEASON) -> Optional[dict]:
    today = datetime.date.today()
    schedule = get_season_schedule(year)
    for race in schedule:
        race_date = datetime.date.fromisoformat(race["date"])
        if race_date >= today:
            race["circuit_config"] = _match_circuit_config(race["name"])
            return race
    if schedule:
        last_race = schedule[-1]
        last_race["circuit_config"] = _match_circuit_config(last_race["name"])
        return last_race
    return None


def get_race_by_round(round_num: int, year: int = CURRENT_SEASON) -> Optional[dict]:
    schedule = get_season_schedule(year)
    for race in schedule:
        if race["round"] == round_num:
            race["circuit_config"] = _match_circuit_config(race["name"])
            return race
    return None


def _match_circuit_config(race_name: str) -> dict:
    if race_name in CIRCUITS:
        base = CIRCUITS[race_name]
    else:
        # Fallback to loose matching
        base = {}
        for name, cfg in CIRCUITS.items():
            n_clean = name.lower().replace("grand prix", "").strip()
            if n_clean and n_clean in race_name.lower():
                base = cfg
                break

    # Phase 8: Enrich with per-circuit JSON features
    try:
        from config import get_enriched_circuit_config
        enriched = get_enriched_circuit_config(race_name)
        if enriched:
            return enriched
    except Exception:
        pass
    return base


# ─────────────────────────────────────────────
# STANDINGS
# ─────────────────────────────────────────────
def get_driver_standings(year: int = CURRENT_SEASON, round_num: int = None) -> list[dict]:
    endpoint = f"/{year}/{round_num}/driverStandings.json" if round_num else f"/{year}/driverStandings.json"
    data = _jolpica_get(endpoint, cache_hours=3)
    standings_list = (
        data.get("MRData", {})
            .get("StandingsTable", {})
            .get("StandingsLists", [{}])
    )
    if not standings_list:
        return []
    entries = standings_list[0].get("DriverStandings", [])
    result = []
    for e in entries:
        drv = e["Driver"]
        pos = e.get("position")
        if pos is None:
            pos_text = e.get("positionText", "")
            try:
                pos = int(pos_text)
            except ValueError:
                pos = len(entries)
        else:
            pos = int(pos)
        result.append({
            "position":     pos,
            "driver_id":    drv["driverId"],
            "name":         _normalize_driver_name(drv['givenName'], drv['familyName']),
            "points":       float(e["points"]),
            "wins":         int(e["wins"]),
            "constructor":  _normalize_constructor_name(e["Constructors"][0]["name"]) if e.get("Constructors") else "",
        })
    return result


def get_constructor_standings(year: int = CURRENT_SEASON, round_num: int = None) -> list[dict]:
    endpoint = f"/{year}/{round_num}/constructorStandings.json" if round_num else f"/{year}/constructorStandings.json"
    data = _jolpica_get(endpoint, cache_hours=3)
    standings_list = (
        data.get("MRData", {})
            .get("StandingsTable", {})
            .get("StandingsLists", [{}])
    )
    if not standings_list:
        return []
    entries = standings_list[0].get("ConstructorStandings", [])
    result = []
    for e in entries:
        ctor = e["Constructor"]
        pos = e.get("position")
        if pos is None:
            pos_text = e.get("positionText", "")
            try:
                pos = int(pos_text)
            except ValueError:
                pos = len(entries)
        else:
            pos = int(pos)
        result.append({
            "position":    pos,
            "constructor": ctor["name"],
            "points":      float(e["points"]),
            "wins":        int(e["wins"]),
        })
    return result


# ─────────────────────────────────────────────
# RACE RESULTS (HISTORICAL)
# ─────────────────────────────────────────────
def get_race_results(year: int, round_num: int) -> list[dict]:
    # Race results never change once posted — cache permanently (cache_hours=0)
    data = _jolpica_get(f"/{year}/{round_num}/results.json", cache_hours=0)
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return []
    results = races[0].get("Results", [])
    out = []
    for r in results:
        drv = r["Driver"]
        status = r.get("status", "")
        out.append({
            "position":     int(r["position"]) if r["position"].isdigit() else 20,
            "driver_id":    drv["driverId"],
            "name":         _normalize_driver_name(drv['givenName'], drv['familyName']),
            "constructor":  _normalize_constructor_name(r["Constructor"]["name"]),
            "grid":         int(r.get("grid", 0)),
            "laps":         int(r.get("laps", 0)),
            "status":       status,
            "dnf":          status.lower() not in ("finished", "+1 lap", "+2 laps", "+3 laps",
                                                    "+4 laps", "+5 laps", "+6 laps"),
            "points":       float(r.get("points", 0)),
            "fastest_lap":  r.get("FastestLap", {}).get("rank") == "1",
        })
    return out


def get_qualifying_results(year: int, round_num: int) -> list[dict]:
    # Qualifying results never change once posted — cache permanently (cache_hours=0)
    data = _jolpica_get(f"/{year}/{round_num}/qualifying.json", cache_hours=0)
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return []
    results = races[0].get("QualifyingResults", [])
    out = []
    for r in results:
        drv = r["Driver"]
        pos = int(r["position"])
        out.append({
            "position":    pos,
            "driver_id":   drv["driverId"],
            "name":        _normalize_driver_name(drv['givenName'], drv['familyName']),
            "constructor": _normalize_constructor_name(r["Constructor"]["name"]),
            "q1":          r.get("Q1", ""),
            "q2":          r.get("Q2", ""),
            "q3":          r.get("Q3", ""),
            "q3_set":      bool(r.get("Q3")),
            "q2_set":      bool(r.get("Q2")),
        })
    return out

def qualifying_has_happened(race: dict) -> bool:
    """Check if qualifying has completed based on current UTC time."""
    quali_date = race.get("quali_date")
    quali_time = race.get("quali_time")
    if not quali_date or not quali_time:
        return False
    try:
        # e.g., quali_time = "14:00:00Z"
        time_str = quali_time.replace("Z", "+00:00")
        dt_str = f"{quali_date}T{time_str}"
        quali_dt = datetime.datetime.fromisoformat(dt_str)
        # Add 2 hours for session duration + 30 mins buffer
        quali_end = quali_dt + datetime.timedelta(hours=2, minutes=30)
        return datetime.datetime.now(datetime.timezone.utc) > quali_end
    except Exception:
        return False

def get_actual_qualifying_results(year: int, round_num: int) -> list[dict]:
    """Get the classified grid order, combining qualifying results with penalties if available."""
    q_results = get_qualifying_results(year, round_num)
    if not q_results:
        return []
        
    penalties = get_grid_penalties(year, round_num) or {}
    
    out = []
    for r in q_results:
        drv = r["name"]
        penalty = penalties.get(drv, 0)
        effective_grid = r["position"] + penalty
        if effective_grid > 22:
            effective_grid = 22
            
        out.append({
            "driver": drv,
            "grid_pos": r["position"],
            "grid_penalty": penalty,
            "effective_grid": effective_grid,
            "q1_time": r.get("q1", ""),
            "q2_time": r.get("q2", ""),
            "q3_time": r.get("q3", "")
        })
        
    # Sort by effective_grid
    out.sort(key=lambda x: x["effective_grid"])
    
    # Reassign effective_grid incrementally to handle overlapping penalties
    for i, r in enumerate(out):
        r["effective_grid"] = i + 1
        
    return out


def get_season_results(year: int) -> list[dict]:
    # Completed seasons are immutable — cache permanently.
    # Current season grows mid-year, so use a short TTL (1h) to pick up new rounds.
    cache_hours = 0 if year < CURRENT_SEASON else 1
    data = _jolpica_get(f"/{year}/results.json", params={"limit": 1000}, cache_hours=cache_hours)
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    out = []
    for race in races:
        round_num = int(race["round"])
        race_name = race["raceName"]
        for r in race.get("Results", []):
            drv = r["Driver"]
            status = r.get("status", "")
            out.append({
                "round":        round_num,
                "race_name":    race_name,
                "position":     int(r["position"]) if r["position"].isdigit() else 20,
                "driver_id":    drv["driverId"],
                "name":         _normalize_driver_name(drv['givenName'], drv['familyName']),
                "constructor":  _normalize_constructor_name(r["Constructor"]["name"]),
                "grid":         int(r.get("grid", 0)),
                "laps":         int(r.get("laps", 0)),
                "dnf":          status.lower() not in ("finished", "+1 lap", "+2 laps", "+3 laps",
                                                        "+4 laps", "+5 laps", "+6 laps"),
                "points":       float(r.get("points", 0)),
            })
    return out


def get_circuit_history(circuit_id: str, seasons: list[int] = None) -> pd.DataFrame:
    if seasons is None:
        seasons = HISTORICAL_SEASONS

    all_rows = []
    for i, year in enumerate(seasons):
        try:
            if i > 0:
                time.sleep(0.3)
            # Normalise circuit ID — Jolpica uses 'villeneuve' for Canada, not 'canada'
            api_circuit_id = _normalize_circuit_id(circuit_id)
            data = _jolpica_get(f"/{year}/circuits/{api_circuit_id}/results.json",
                                params={"limit": 200}, cache_hours=0)
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            for race in races:
                for r in race.get("Results", []):
                    drv = r["Driver"]
                    status = r.get("status", "")
                    all_rows.append({
                        "year":        year,
                        "round":       int(race["round"]),
                        "position":    int(r["position"]) if r["position"].isdigit() else 20,
                        "driver_id":   drv["driverId"],
                        "name":        _normalize_driver_name(drv['givenName'], drv['familyName']),
                        "constructor": _normalize_constructor_name(r["Constructor"]["name"]),
                        "grid":        int(r.get("grid", 0)),
                        "dnf":         status.lower() not in ("finished", "+1 lap", "+2 laps",
                                                               "+3 laps", "+4 laps", "+5 laps"),
                    })
        except Exception:
            continue

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


def get_pitstop_data(year: int, round_num: int) -> list[dict]:
    data = _jolpica_get(f"/{year}/{round_num}/pitstops.json",
                        params={"limit": 200}, cache_hours=24)
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return []
    pit_stops = races[0].get("PitStops", [])
    out = []
    for p in pit_stops:
        try:
            dur_parts = p["duration"].split(":")
            duration_sec = float(dur_parts[-1]) if len(dur_parts) == 1 else float(dur_parts[0]) * 60 + float(dur_parts[1])
        except Exception:
            duration_sec = 99.0
        out.append({
            "driver_id":    p["driverId"],
            "stop":         int(p["stop"]),
            "lap":          int(p["lap"]),
            "duration_sec": duration_sec,
        })
    return out


# ─────────────────────────────────────────────
# GRID PENALTIES (NEW)
# Compares qualifying position against actual race grid position.
# Penalty = grid_pos - quali_pos (positive = dropped back).
# ─────────────────────────────────────────────
def get_grid_penalties(year: int, round_num: int) -> dict[str, int]:
    """
    Returns dict of {driver_name: penalty_positions}.
    Positive = grid penalty (dropped), negative = moved up via others' penalties.
    """
    cache_key = f"grid_penalties_{year}_{round_num}"
    cached = _load_cache(cache_key, 48)
    if cached:
        return cached

    try:
        quali = get_qualifying_results(year, round_num)
        race  = get_race_results(year, round_num)
    except Exception:
        return {}

    quali_pos = {r["name"]: r["position"] for r in quali}
    grid_pos  = {r["name"]: r["grid"] for r in race if r.get("grid", 0) > 0}

    penalties = {}
    for name, q_pos in quali_pos.items():
        g_pos = grid_pos.get(name, q_pos)
        delta = g_pos - q_pos
        if delta != 0:
            penalties[name] = delta

    _save_cache(cache_key, penalties)
    return penalties


# ─────────────────────────────────────────────
# OpenF1 SESSION DATA (2023+)
# ─────────────────────────────────────────────
def get_openf1_sessions(year: int, gp_name: str = None) -> list[dict]:
    params = {"year": year}
    if gp_name:
        params["meeting_name"] = gp_name
    return _openf1_get("/sessions", params, cache_hours=6)


def get_openf1_lap_times(session_key: int, driver_num: int = None) -> list[dict]:
    params = {"session_key": session_key}
    if driver_num:
        params["driver_number"] = driver_num
    return _openf1_get("/laps", params, cache_hours=6)


def get_openf1_drivers(session_key: int) -> list[dict]:
    return _openf1_get("/drivers", {"session_key": session_key}, cache_hours=24)


def get_openf1_positions(session_key: int) -> list[dict]:
    return _openf1_get("/position", {"session_key": session_key}, cache_hours=6)


def get_openf1_stints(session_key: int) -> list[dict]:
    return _openf1_get("/stints", {"session_key": session_key}, cache_hours=6)


def get_openf1_pit_stops(session_key: int) -> list[dict]:
    return _openf1_get("/pit", {"session_key": session_key}, cache_hours=6)


# ─────────────────────────────────────────────
# FASTF1 HELPERS
# ─────────────────────────────────────────────
def get_fastf1_session(year: int, gp: str, session_type: str = "R"):
    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load(telemetry=False, weather=True, messages=False)
        return session
    except Exception:
        return None


def get_driver_lap_stats(year: int, gp: str, session_type: str = "R") -> pd.DataFrame:
    session = get_fastf1_session(year, gp, session_type)
    if session is None:
        return pd.DataFrame()
    try:
        laps = session.laps
        if laps is None or laps.empty:
            return pd.DataFrame()
        stats = laps.groupby("Driver").agg(
            avg_lap_time=("LapTime", lambda x: x.dropna().mean()),
            best_lap_time=("LapTime", "min"),
            lap_count=("LapNumber", "count"),
        ).reset_index()
        return stats
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
# PHASE 2/4: TIRE STINTS & ALLOCATIONS (NEW)
# ─────────────────────────────────────────────

def get_weekend_tire_allocations(year: int, gp_name: str) -> dict[str, dict]:
    """
    Returns estimated remaining tire sets for each driver going into the Race.
    Assumes standard weekend allocation: 2 Hard, 3 Medium, 8 Soft.
    Counts unique stints across FP1, FP2, FP3, Q to decrement the allocation.
    """
    cache_key = f"tire_allocs_{year}_{gp_name.replace(' ', '_')}_v1"
    cached = _load_cache(cache_key, 72)
    if cached is not None:
        return cached

    driver_used_sets = {}
    try:
        from config import DRIVER_SHORT_2026
        sessions = ["FP1", "FP2", "FP3", "Q"]
        
        for s_name in sessions:
            try:
                session = fastf1.get_session(year, gp_name, s_name)
                session.load(telemetry=False, weather=False, messages=False)
                if session.laps is None or session.laps.empty:
                    continue
                
                # Build abbreviation mapping
                abbr_to_name = {}
                for num in session.drivers:
                    try:
                        drv_data = session.get_driver(num)
                        abbr = drv_data.get("Abbreviation", "")
                        full = f"{drv_data.get('FirstName', '')} {drv_data.get('LastName', '')}".strip()
                        if abbr and full:
                            abbr_to_name[abbr] = full
                    except Exception:
                        pass
                
                for driver_abbr, drv_laps in session.laps.groupby("Driver"):
                    full_name = abbr_to_name.get(driver_abbr, driver_abbr)
                    # Use canonical config name if possible
                    canonical_name = DRIVER_SHORT_2026.get(driver_abbr, full_name)
                    
                    if canonical_name not in driver_used_sets:
                        driver_used_sets[canonical_name] = {"HARD": 0, "MEDIUM": 0, "SOFT": 0}
                    
                    for compound, comp_laps in drv_laps.groupby("Compound"):
                        if pd.isna(compound) or compound == "UNKNOWN":
                            continue
                        compound_str = str(compound).upper()
                        if compound_str not in ["HARD", "MEDIUM", "SOFT"]:
                            continue
                        
                        unique_stints = comp_laps["Stint"].nunique()
                        driver_used_sets[canonical_name][compound_str] += unique_stints
            except Exception:
                pass
                
        result = {}
        for name, used in driver_used_sets.items():
            h_left = max(0, 2 - used["HARD"])
            m_left = max(0, 3 - used["MEDIUM"])
            s_left = max(0, 8 - used["SOFT"])
            result[name] = {
                "fresh_hard": h_left,
                "fresh_medium": m_left,
                "fresh_soft": s_left,
                "total_fresh": h_left + m_left + s_left
            }
            
        _save_cache(cache_key, result)
        return result
    except Exception as e:
        print(f"    [WARN] get_weekend_tire_allocations({year}, {gp_name}): {e}")
        return {}


def get_tire_stints(year: int, gp_name: str, session_type: str = "R") -> dict[str, list[dict]]:
    """
    Fetches lap-by-lap tire compound choices, stint lengths, and track temperature
    for each driver using the free FastF1 library.

    Returns a dict keyed by driver full name:
        {
          "Max Verstappen": [
            {
              "compound":       "MEDIUM",
              "stint_number":   1,
              "start_lap":      1,
              "end_lap":        23,
              "stint_length":   23,
              "avg_track_temp": 38.2,   # degrees C, None if unavailable
              "avg_lap_sec":    93.412, # average lap time in seconds for this stint
              "deg_per_lap":    0.032,  # estimated pace degradation per lap (s/lap)
            },
            ...
          ],
          ...
        }

    Cost & rate-limit notes:
      - FastF1 downloads from the official Ergast/F1 timing server — completely free.
      - Results are permanently cached to FASTF1_CACHE_DIR via fastf1.Cache.enable_cache().
      - Our own JSON cache layer adds a 72-hour TTL on top to avoid even re-parsing the
        FastF1 parquet files on every run.
      - `telemetry=False` keeps the session load fast — we only need laps + weather.
    """
    cache_key = f"tire_stints_{session_type}_{year}_{str(gp_name).replace(' ', '_')}_v1"
    cached = _load_cache(cache_key, 72)
    if cached is not None:
        return cached

    try:
        session = fastf1.get_session(year, gp_name, session_type)
        # telemetry=False: we don't need raw car channel data; laps + weather is enough
        session.load(telemetry=False, weather=True, messages=False)
        laps = session.laps

        if laps is None or laps.empty:
            return {}

        # Pull weather data for track temperature (free from FastF1)
        weather_df = getattr(session, "weather_data", None)
        if weather_df is not None and not weather_df.empty and "TrackTemp" in weather_df.columns:
            # Build a mapping from SessionTime → TrackTemp for fast lookup
            weather_df = weather_df.dropna(subset=["TrackTemp"])
            weather_times = weather_df["Time"].dt.total_seconds().values
            weather_temps = weather_df["TrackTemp"].values
        else:
            weather_times = None
            weather_temps = None

        # Build abbreviation → full name lookup (no extra API calls)
        abbr_to_name: dict[str, str] = {}
        for num in session.drivers:
            try:
                drv_data = session.get_driver(num)
                abbr = drv_data.get("Abbreviation", "")
                full = f"{drv_data.get('FirstName', '')} {drv_data.get('LastName', '')}".strip()
                if abbr and full:
                    abbr_to_name[abbr] = full
            except Exception:
                pass

        result: dict[str, list[dict]] = {}

        for driver_abbr, drv_laps in laps.groupby("Driver"):
            drv_laps = drv_laps.sort_values("LapNumber").reset_index(drop=True)
            full_name = abbr_to_name.get(driver_abbr, driver_abbr)

            # Identify compound changes to split into stints
            stints: list[list] = []
            current_stint: list = []
            prev_compound = None

            for _, row in drv_laps.iterrows():
                compound = row.get("Compound", None)
                if compound != prev_compound and current_stint:
                    stints.append(current_stint)
                    current_stint = []
                current_stint.append(row)
                prev_compound = compound

            if current_stint:
                stints.append(current_stint)

            driver_stints: list[dict] = []
            for stint_num, stint_rows in enumerate(stints, start=1):
                if not stint_rows:
                    continue

                compound = stint_rows[0].get("Compound", "UNKNOWN") or "UNKNOWN"
                start_lap = int(stint_rows[0]["LapNumber"])
                end_lap = int(stint_rows[-1]["LapNumber"])
                stint_length = end_lap - start_lap + 1

                # Collect valid lap times in seconds for this stint
                lap_times_sec: list[float] = []
                for row in stint_rows:
                    lt = row.get("LapTime")
                    if pd.notna(lt):
                        try:
                            lap_times_sec.append(lt.total_seconds())
                        except Exception:
                            pass

                avg_lap_sec = float(np.mean(lap_times_sec)) if lap_times_sec else None

                # Estimate pace degradation: slope of lap time over stint laps (s/lap)
                # Positive = getting slower per lap (degrading tire)
                if len(lap_times_sec) >= 3:
                    x = np.arange(len(lap_times_sec), dtype=float)
                    # Simple OLS slope — no external dependency needed
                    x_mean = x.mean()
                    y_mean = float(np.mean(lap_times_sec))
                    numer = float(np.sum((x - x_mean) * (np.array(lap_times_sec) - y_mean)))
                    denom = float(np.sum((x - x_mean) ** 2))
                    deg_per_lap = round(numer / denom, 4) if denom != 0 else 0.0
                else:
                    deg_per_lap = None

                # Approximate track temp for this stint by interpolating weather data
                avg_track_temp = None
                if weather_times is not None and len(stint_rows) > 0:
                    try:
                        # Use the SessionTime of the first lap in the stint as reference
                        ref_time = stint_rows[0].get("Time")
                        if pd.notna(ref_time):
                            ref_sec = ref_time.total_seconds()
                            # np.interp clamps extrapolated values — acceptable here
                            avg_track_temp = round(
                                float(np.interp(ref_sec, weather_times, weather_temps)), 1
                            )
                    except Exception:
                        pass

                driver_stints.append({
                    "compound":       str(compound),
                    "stint_number":   stint_num,
                    "start_lap":      start_lap,
                    "end_lap":        end_lap,
                    "stint_length":   stint_length,
                    "avg_track_temp": avg_track_temp,
                    "avg_lap_sec":    round(avg_lap_sec, 3) if avg_lap_sec is not None else None,
                    "deg_per_lap":    deg_per_lap,
                })

            if driver_stints:
                result[full_name] = driver_stints

        _save_cache(cache_key, result)
        return result

    except Exception as exc:
        exc_str = str(exc)
        if "has not been loaded yet" not in exc_str and "No data" not in exc_str:
            print(f"    [WARN] get_tire_stints({year}, {gp_name}, {session_type}): {exc}")
        return {}


def normalize_telemetry(
    driver_data: dict[str, pd.DataFrame],
    time_col: str = "SessionTime",
    value_cols: list[str] = None,
    grid_interval: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """
    Temporally normalises driver telemetry DataFrames onto a unified 0.5-second grid.

    Different cars report telemetry at slightly varying intervals. By interpolating all
    driver channels onto the same fixed time grid we ensure that comparisons (e.g. for
    the LSTM model) are spatially and temporally consistent.

    Parameters
    ----------
    driver_data  : {driver_name: DataFrame} — each DataFrame must contain `time_col`
                   (as total seconds float or timedelta) and any columns listed in value_cols.
    time_col     : Name of the column holding session time (seconds).
    value_cols   : Channels to interpolate. Defaults to ['Speed', 'X', 'Y'].
    grid_interval: Spacing of the unified time grid in seconds. Default 0.5 s.

    Returns
    -------
    Normalised DataFrames keyed by driver name, all sharing the same SessionTime index.

    Cost note: Pure NumPy — no external API calls. Runs locally at zero cost.
    """
    if value_cols is None:
        value_cols = ["Speed", "X", "Y"]

    # Determine the shared time range across all drivers
    all_min: list[float] = []
    all_max: list[float] = []
    coerced: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}

    for driver, df in driver_data.items():
        if df is None or df.empty or time_col not in df.columns:
            continue
        t = df[time_col]
        # Accept both timedelta and numeric seconds
        if hasattr(t.iloc[0], "total_seconds"):
            t_sec = t.apply(lambda x: x.total_seconds() if pd.notna(x) else np.nan).values.astype(float)
        else:
            t_sec = pd.to_numeric(t, errors="coerce").values.astype(float)

        valid_mask = ~np.isnan(t_sec)
        if valid_mask.sum() < 2:
            continue

        channels: dict[str, np.ndarray] = {}
        for col in value_cols:
            if col in df.columns:
                channels[col] = pd.to_numeric(df[col], errors="coerce").values.astype(float)

        coerced[driver] = (t_sec, channels)
        all_min.append(float(t_sec[valid_mask].min()))
        all_max.append(float(t_sec[valid_mask].max()))

    if not coerced:
        return {}

    t_start = max(all_min)   # innermost common window
    t_end   = min(all_max)

    if t_end <= t_start:
        # No overlapping window — return originals untouched
        return driver_data

    grid = np.arange(t_start, t_end, grid_interval)
    normalised: dict[str, pd.DataFrame] = {}

    for driver, (t_sec, channels) in coerced.items():
        row: dict[str, np.ndarray] = {time_col: grid}
        for col, values in channels.items():
            # np.interp fills NaN-gap edges by clamping — acceptable for telemetry
            row[col] = np.interp(grid, t_sec, values)
        normalised[driver] = pd.DataFrame(row)

    return normalised


def get_session_feature_weights() -> dict[str, float]:
    """
    Returns empirically-derived importance weights for each race-weekend session,
    based on the Ordinal Logistic Regression (OLR) academic study:

        "Re-evaluating the Qualifying/Finish Relationship in Formula One"
        Weissbock & Mills, Carleton University, 2025

    Findings (standardised β coefficients):
        Qualifying:  1.558  (strongest predictor; ~4× any practice session)
        FP3:         0.379
        FP2:         0.368
        FP1:         0.291

    Usage in predictor.py / temporal_model.py:
        weights = get_session_feature_weights()
        # e.g. weight qualifying_position feature by weights["Q"]
        weighted_qual_pos = qualifying_position * weights["Q"]

    Cost note: Pure constant lookup — no API calls.
    """
    return {
        "Q":   1.558,   # Qualifying — strongest signal
        "FP3": 0.379,   # Practice 3 — best practice session
        "FP2": 0.368,   # Practice 2
        "FP1": 0.291,   # Practice 1 — weakest signal
    }


# ─────────────────────────────────────────────
# PRACTICE LONG-RUN PACE (NEW)
# Uses FastF1 FP2 to compute long-run pace delta vs. field median.
# ─────────────────────────────────────────────
def compute_practice_pace(year: int, gp_name: str, session_type: str = "FP2") -> dict[str, float]:
    """
    Compute long-run pace delta (seconds) for each driver vs the field median.
    Works for any practice session: FP1, FP2, FP3.
    Negative = faster than median (good), Positive = slower (bad).
    Returns: {driver_name: delta_seconds} or {} if session data unavailable.
    """
    cache_key = f"practice_pace_{session_type}_{year}_{gp_name.replace(' ', '_')}_v2"
    cached = _load_cache(cache_key, 72)
    if cached:
        return cached

    try:
        session = fastf1.get_session(year, gp_name, session_type)
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps

        if laps is None or laps.empty:
            return {}

        # Filter to representative laps only
        laps = laps.pick_quicklaps(threshold=1.07)

        # Remove laps without valid lap time
        laps = laps[laps["LapTime"].notna()].copy()
        laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()

        # Build stints — group consecutive laps on same compound per driver
        driver_long_run_avgs = {}
        for driver_abbr, drv_laps in laps.groupby("Driver"):
            drv_laps = drv_laps.sort_values("LapNumber").reset_index(drop=True)

            stints = []
            current_stint = [drv_laps.iloc[0]]
            for i in range(1, len(drv_laps)):
                row = drv_laps.iloc[i]
                prev = drv_laps.iloc[i - 1]
                same_compound = row.get("Compound", "") == prev.get("Compound", "")
                consecutive = (row["LapNumber"] - prev["LapNumber"]) <= 2
                if same_compound and consecutive:
                    current_stint.append(row)
                else:
                    stints.append(current_stint)
                    current_stint = [row]
            stints.append(current_stint)

            # For FP2/FP3, use long runs (4+ laps); for FP1, accept shorter stints (3+)
            min_stint = 3 if session_type == "FP1" else 4
            long_run_laps = [lap for stint in stints if len(stint) >= min_stint for lap in stint]
            if long_run_laps:
                avg_sec = sum(l["LapTimeSec"] for l in long_run_laps) / len(long_run_laps)
                driver_long_run_avgs[driver_abbr] = avg_sec

        if not driver_long_run_avgs:
            return {}

        # Compute field median and delta
        median_pace = float(np.median(list(driver_long_run_avgs.values())))

        # Map FastF1 3-letter abbreviations to canonical names from config
        from config import DRIVER_SHORT_2026
        result = {}
        for abbr, avg_sec in driver_long_run_avgs.items():
            name = DRIVER_SHORT_2026.get(abbr)
            if not name:
                try:
                    drv_data = session.get_driver(abbr)
                    raw_name = f"{drv_data.get('FirstName', '')} {drv_data.get('LastName', '')}".strip()
                    if " " in raw_name:
                        parts = raw_name.split(" ")
                        name = _normalize_driver_name(parts[0], " ".join(parts[1:]))
                    else:
                        name = raw_name
                except Exception:
                    name = abbr
            result[name] = round(avg_sec - median_pace, 3)

        _save_cache(cache_key, result)
        return result

    except Exception:
        return {}


def get_best_practice_pace(year: int, gp_name: str) -> tuple[dict, str]:
    """
    Try practice sessions in order: FP2 -> FP1 -> FP3.
    Sprint weekends skip FP2, so FP1 will be used automatically.
    Returns (pace_data_dict, session_label_str).
    """
    for session in ["FP2", "FP1", "FP3"]:
        try:
            data = compute_practice_pace(year, gp_name, session_type=session)
            if data:
                return data, session
        except Exception:
            continue
    return {}, "N/A"


# ─────────────────────────────────────────────
# QUALIFYING SECTOR TIMES (NEW — post-quali mode)
# ─────────────────────────────────────────────
def get_qualifying_sector_times(year: int, gp_name: str) -> dict[str, dict]:
    """
    Returns per-driver qualifying sector time deltas vs. fastest sector time.
    {driver_name: {s1_delta, s2_delta, s3_delta, q_lap_delta}}

    All deltas in seconds. Negative = faster than field best (good).
    Returns {} if qualifying data not yet available.
    """
    cache_key = f"quali_sectors_{year}_{gp_name.replace(' ', '_')}_v2"
    cached = _load_cache(cache_key, 48)
    if cached:
        return cached

    try:
        session = fastf1.get_session(year, gp_name, "Q")
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps

        if laps is None or laps.empty:
            return {}

        # Get best lap per driver
        best_laps = laps.pick_quicklaps(threshold=1.05).groupby("Driver").apply(
            lambda g: g.nsmallest(1, "LapTime").iloc[0] if len(g) > 0 else None
        ).dropna()

        if best_laps.empty:
            return {}

        # Compute fastest sector times across all drivers
        s1_vals = best_laps["Sector1Time"].dropna()
        s2_vals = best_laps["Sector2Time"].dropna()
        s3_vals = best_laps["Sector3Time"].dropna()
        lap_vals = best_laps["LapTime"].dropna()

        if s1_vals.empty:
            return {}

        best_s1 = s1_vals.min().total_seconds()
        best_s2 = s2_vals.min().total_seconds() if not s2_vals.empty else None
        best_s3 = s3_vals.min().total_seconds() if not s3_vals.empty else None
        best_lap = lap_vals.min().total_seconds()

        # Map abbreviations to canonical names from config
        from config import DRIVER_SHORT_2026
        result = {}
        for abbr, row in best_laps.iterrows():
            name = DRIVER_SHORT_2026.get(abbr)
            if not name:
                try:
                    drv_data = session.get_driver(abbr)
                    raw_name = f"{drv_data.get('FirstName', '')} {drv_data.get('LastName', '')}".strip()
                    if " " in raw_name:
                        parts = raw_name.split(" ")
                        name = _normalize_driver_name(parts[0], " ".join(parts[1:]))
                    else:
                        name = raw_name
                except Exception:
                    name = abbr
            s1 = row["Sector1Time"].total_seconds() if pd.notna(row.get("Sector1Time")) else None
            s2 = row["Sector2Time"].total_seconds() if pd.notna(row.get("Sector2Time")) else None
            s3 = row["Sector3Time"].total_seconds() if pd.notna(row.get("Sector3Time")) else None
            lap = row["LapTime"].total_seconds() if pd.notna(row.get("LapTime")) else None

            result[name] = {
                "s1_delta":    round(s1 - best_s1, 3) if s1 else 0.0,
                "s2_delta":    round(s2 - best_s2, 3) if (s2 and best_s2) else 0.0,
                "s3_delta":    round(s3 - best_s3, 3) if (s3 and best_s3) else 0.0,
                "q_lap_delta": round(lap - best_lap, 3) if lap else 0.0,
            }

        _save_cache(cache_key, result)
        return result

    except Exception:
        return {}


# ─────────────────────────────────────────────
# DRIVER FORM — EWMA-weighted (improved)
# ─────────────────────────────────────────────
def compute_driver_form(year: int, num_races: int = 10, until_round: int = None) -> dict[str, dict]:
    """
    Returns per-driver form dict with EWMA-weighted positions.
    Most recent race counts ~4x more than 5 races ago (λ=0.7 decay).
    Also computes momentum_trend (slope: +ve = improving, -ve = declining).
    Disk-cached: historical years = permanent, current season = 3h TTL.
    """
    # ── Disk cache — avoids 5 API calls per round in the training loop ──
    _cache_key_form = f"driver_form_{year}_{num_races}_{until_round}_v2"
    _cache_age_form = 0 if year < CURRENT_SEASON else 3   # 0 = permanent
    _cached_form = _load_cache(_cache_key_form, _cache_age_form)
    if _cached_form is not None:
        normalized_form = {}
        for k, v in _cached_form.items():
            if " " in k:
                parts = k.split(" ")
                normalized_key = _normalize_driver_name(parts[0], " ".join(parts[1:]))
            else:
                normalized_key = k
            normalized_form[normalized_key] = v
        return normalized_form

    schedule = get_season_schedule(year)
    if until_round is not None:
        completed = [r for r in schedule if r["round"] < until_round]
    else:
        today = datetime.date.today()
        completed = [r for r in schedule if datetime.date.fromisoformat(r["date"]) < today]
    recent = completed[-num_races:] if len(completed) >= num_races else completed

    driver_records: dict[str, list] = {}
    for race in recent:
        try:
            results = get_race_results(year, race["round"])
            for r in results:
                name = r["name"]
                if name not in driver_records:
                    driver_records[name] = []
                driver_records[name].append({
                    "position": r["position"],
                    "points":   r["points"],
                    "dnf":      r["dnf"],
                    "round":    race["round"],
                })
        except Exception:
            continue

    EWMA_LAMBDA = 1.0  # Set to 1.0 (SMA) to eliminate heavy recency bias
    form_data = {}
    for drv, records in driver_records.items():
        if not records:
            continue
        # Sort by round ascending so most recent is last
        records = sorted(records, key=lambda r: r["round"])
        n = len(records)

        # EWMA weights: most recent has highest weight
        weights = [EWMA_LAMBDA ** (n - 1 - i) for i in range(n)]
        total_w = sum(weights)

        positions = [r["position"] for r in records]
        points    = [r["points"]   for r in records]
        dnfs      = [r["dnf"]      for r in records]

        avg_pos  = sum(p * w for p, w in zip(positions, weights)) / total_w
        avg_pts  = sum(p * w for p, w in zip(points,    weights)) / total_w
        dnf_rate = sum(d * w for d, w in zip(dnfs,      weights)) / total_w

        # Momentum trend: slope of position (lower = better) over last n races
        if n >= 2:
            x = list(range(n))
            x_mean = sum(x) / n
            # Use simple mean of positions (not EWMA avg_pos) so OLS is self-consistent
            y_mean = sum(positions) / n
            numer  = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, positions))
            denom  = sum((xi - x_mean) ** 2 for xi in x)
            slope  = numer / denom if denom != 0 else 0.0
            # Negate: negative slope (improving position = lower number) → positive momentum
            momentum_trend = -slope
        else:
            momentum_trend = 0.0

        form_score = (avg_pts * 2) - (avg_pos * 0.5) - (dnf_rate * 20)
        form_data[drv] = {
            "avg_position":    round(avg_pos, 2),
            "avg_points":      round(avg_pts, 2),
            "dnf_rate":        round(dnf_rate, 3),
            "form_score":      round(form_score, 2),
            "momentum_trend":  round(momentum_trend, 3),  # NEW: +ve = improving
            "races_counted":   n,
            "points_list":     points,  # NEW: for Bayesian ZINB modeling
            "source":          "current_ewma",
        }
    _save_cache(_cache_key_form, form_data)
    return form_data


# ─────────────────────────────────────────────
# MULTI-SEASON WEIGHTED PERFORMANCE
# Now uses SEASON_WEIGHTS from config (reg-era aware)
# ─────────────────────────────────────────────
def compute_multiseason_driver_form(
    current_year: int = CURRENT_SEASON,
    prior_seasons: list[int] = None,
    season_weights: dict[int, float] = None,
) -> dict[str, dict]:
    """
    Blends per-driver form across multiple seasons.
    Uses SEASON_WEIGHTS from config to discount pre-regulation-change seasons.
    """
    if prior_seasons is None:
        prior_seasons = [y for y in HISTORICAL_SEASONS if y < current_year]

    if season_weights is None:
        # Use regulation-aware weights from config
        season_weights = {
            y: SEASON_WEIGHTS.get(y, 0.5 ** (current_year - y))
            for y in sorted(prior_seasons, reverse=True)
        }
        season_weights[current_year] = SEASON_WEIGHTS.get(current_year, 1.0)

    w_str = "_".join(f"{k}-{v:.2f}" for k, v in sorted(season_weights.items()))
    cache_key = f"multiseason_driver_form_{current_year}_{w_str}_v2"
    # Historical-only query = permanent cache; includes current season = 6h TTL
    _includes_current = current_year >= CURRENT_SEASON
    cached = _load_cache(cache_key, 6 if _includes_current else 0)
    if cached:
        normalized_cached = {}
        for k, v in cached.items():
            if " " in k:
                parts = k.split(" ")
                normalized_key = _normalize_driver_name(parts[0], " ".join(parts[1:]))
            else:
                normalized_key = k
            normalized_cached[normalized_key] = v
        return normalized_cached

    all_season_data: dict[str, dict[int, dict]] = {}

    for i, (year, weight) in enumerate(season_weights.items()):
        try:
            if i > 0:
                time.sleep(0.3)
            all_results = get_season_results(year)
            if not all_results:
                continue
            df = pd.DataFrame(all_results)
            for drv, grp in df.groupby("name"):
                if drv not in all_season_data:
                    all_season_data[drv] = {}
                all_season_data[drv][year] = {
                    "avg_position": float(grp["position"].mean()),
                    "avg_points":   float(grp["points"].mean()),
                    "dnf_rate":     float(grp["dnf"].mean()),
                    "form_score":   float(grp["points"].mean() * 2 - grp["position"].mean() * 0.5 - grp["dnf"].mean() * 20),
                    "momentum_trend": 0.0,
                    "weight":       weight,
                    "race_count":   len(grp),
                }
        except Exception:
            continue

    merged: dict[str, dict] = {}
    for drv, season_data in all_season_data.items():
        total_weight = sum(d["weight"] for d in season_data.values())
        if total_weight == 0:
            continue

        def wavg(key):
            return sum(d[key] * d["weight"] for d in season_data.values()) / total_weight

        merged[drv] = {
            "avg_position":    round(wavg("avg_position"), 2),
            "avg_points":      round(wavg("avg_points"), 2),
            "dnf_rate":        round(wavg("dnf_rate"), 3),
            "form_score":      round(wavg("form_score"), 2),
            "momentum_trend":  0.0,
            "seasons_counted": len(season_data),
            "total_weight":    round(total_weight, 2),
            "races_counted":   sum(d["race_count"] for d in season_data.values()),
            "source":          "multiseason_reg_weighted",
        }
    _save_cache(cache_key, merged)
    return merged


def compute_multiseason_constructor_stats(
    current_year: int = CURRENT_SEASON,
    prior_seasons: list[int] = None,
    season_weights: dict[int, float] = None,
) -> dict[str, dict]:
    """Constructor performance across seasons with regulation-aware decay weighting."""
    if prior_seasons is None:
        prior_seasons = [y for y in HISTORICAL_SEASONS if y < current_year]

    if season_weights is None:
        season_weights = {
            y: SEASON_WEIGHTS.get(y, 0.5 ** (current_year - y))
            for y in sorted(prior_seasons, reverse=True)
        }
        season_weights[current_year] = SEASON_WEIGHTS.get(current_year, 1.0)

    w_str = "_".join(f"{k}-{v:.2f}" for k, v in sorted(season_weights.items()))
    cache_key = f"multiseason_ctor_stats_{current_year}_{w_str}"
    # Historical-only query = permanent cache; includes current season = 6h TTL
    _includes_current = current_year >= CURRENT_SEASON
    cached = _load_cache(cache_key, 6 if _includes_current else 0)
    if cached:
        return cached

    ctor_season_data: dict[str, dict[int, dict]] = {}

    for i, (year, weight) in enumerate(season_weights.items()):
        try:
            if i > 0:
                time.sleep(0.3)
            all_results = get_season_results(year)
            if not all_results:
                continue
            df = pd.DataFrame(all_results)
            for ctor, grp in df.groupby("constructor"):
                if ctor not in ctor_season_data:
                    ctor_season_data[ctor] = {}
                ctor_season_data[ctor][year] = {
                    "avg_position": float(grp["position"].mean()),
                    "avg_points":   float(grp["points"].mean()),
                    "dnf_rate":     float(grp["dnf"].mean()),
                    "weight":       weight,
                    "race_count":   len(grp),
                }
        except Exception:
            continue

    merged: dict[str, dict] = {}
    for ctor, season_data in ctor_season_data.items():
        total_weight = sum(d["weight"] for d in season_data.values())
        if total_weight == 0:
            continue

        def wavg(key):
            return sum(d[key] * d["weight"] for d in season_data.values()) / total_weight

        years_sorted = sorted(season_data.keys())
        if len(years_sorted) >= 2:
            latest_pts = season_data[years_sorted[-1]]["avg_points"]
            prior_pts  = season_data[years_sorted[-2]]["avg_points"]
            pts_trend  = round(latest_pts - prior_pts, 2)
        else:
            pts_trend = 0.0

        merged[ctor] = {
            "avg_position":    round(wavg("avg_position"), 2),
            "avg_points":      round(wavg("avg_points"), 2),
            "dnf_rate":        round(wavg("dnf_rate"), 3),
            "pts_trend":       pts_trend,
            "seasons_counted": len(season_data),
            "source":          "multiseason_reg_weighted",
        }
    _save_cache(cache_key, merged)
    return merged


def compute_constructor_reliability(year: int) -> dict[str, dict]:
    all_results = get_season_results(year)
    if not all_results:
        return {}
    df = pd.DataFrame(all_results)
    ctor_stats = {}
    for ctor, grp in df.groupby("constructor"):
        ctor_stats[ctor] = {
            "dnf_rate":     round(float(grp["dnf"].mean()), 3),
            "avg_points":   round(float(grp["points"].mean()), 2),
            "avg_position": round(float(grp["position"].mean()), 2),
        }
    return ctor_stats
