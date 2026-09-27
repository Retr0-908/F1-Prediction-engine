"""
fantasy_sync.py — Authenticated live F1 Fantasy team synchronisation.
Connects to official F1 Fantasy services (SportzInteractive platform) using the
session cookie to query user teams, bank balance, available transfers, and active chips.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import requests

from engine.core.config import F1_FANTASY_COOKIE, CURRENT_SEASON
from engine.core.paths import MY_TEAM_PATH, OUTPUT_DIR
from engine.strategy.chip_advisor import load_chip_state, save_chip_state

logger = logging.getLogger("f1_predictor.fantasy_sync")

FANTASY_BASE = "https://fantasy.formula1.com"
LOGIN_URL = f"{FANTASY_BASE}/services/session/login"
DRIVERS_CATALOG_URL = f"{FANTASY_BASE}/feeds/drivers/1_en.json"

HEADERS_BASE = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json",
    "Referer": f"{FANTASY_BASE}/",
    "Origin": FANTASY_BASE,
}


class F1FantasySyncError(Exception):
    """Raised when F1 Fantasy session authentication or team sync fails."""
    pass


def _normalize_name(name: str) -> str:
    """Normalise driver/constructor name for engine consistency."""
    import unicodedata

    name = name.strip()
    name = " ".join(name.split())
    if name.isupper():
        name = name.title()
    name = "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    )
    if name.startswith("Andrea Kimi "):
        name = "Kimi " + name[len("Andrea Kimi "):]
    return name


def _get_cookie_string(explicit_cookie: Optional[str] = None) -> str:
    """Resolve cookie string from argument, env variable, or .env file."""
    if explicit_cookie and explicit_cookie.strip():
        return explicit_cookie.strip()

    # Check config
    if F1_FANTASY_COOKIE and F1_FANTASY_COOKIE.strip():
        return F1_FANTASY_COOKIE.strip()

    # Re-check .env file dynamically in case user updated it without restarting
    from dotenv import dotenv_values
    from engine.core.paths import PROJECT_ROOT
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        vals = dotenv_values(env_file)
        c = vals.get("F1_FANTASY_COOKIE", "")
        if c and c.strip():
            return c.strip()

    return ""


def fetch_user_teams(cookie: Optional[str] = None) -> list[dict[str, Any]]:
    """
    Authenticate against F1 Fantasy and fetch all 3 user fantasy teams.
    
    Returns a list of dicts, each containing:
      - team_no: int (1, 2, or 3)
      - team_name: str
      - budget_remaining: float ($M)
      - current_points: float
      - transfers: int
      - turbo_driver: str | None
      - drivers: list[str]
      - constructors: list[str]
      - chips: dict
    """
    raw_cookie = _get_cookie_string(cookie)
    if not raw_cookie:
        raise F1FantasySyncError(
            "F1 Fantasy session cookie is not configured. "
            "Please paste your cookie into .env as F1_FANTASY_COOKIE=..."
        )

    session = requests.Session()
    session.headers.update(HEADERS_BASE)

    # Set cookies in session
    for part in raw_cookie.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            session.cookies.set(k.strip(), v.strip(), domain="formula1.com")

    # Step 1: Session login handshake
    payload = {
        "optType": 1,
        "platformId": 1,
        "platformVersion": "1",
        "platformCategory": "web",
        "clientId": 1,
    }

    try:
        login_resp = session.post(LOGIN_URL, json=payload, timeout=12)
    except Exception as exc:
        raise F1FantasySyncError(f"Network error connecting to F1 Fantasy: {exc}") from exc

    if login_resp.status_code != 200:
        raise F1FantasySyncError(
            f"F1 Fantasy session login rejected (HTTP {login_resp.status_code}). "
            "Your session cookie may have expired or is invalid."
        )

    try:
        login_json = login_resp.json()
    except Exception as exc:
        raise F1FantasySyncError(
            "F1 Fantasy returned non-JSON response during login. "
            "The site may be under maintenance or Cloudflare protected."
        ) from exc

    if not login_json.get("Meta", {}).get("Success", True) and not login_json.get("Data", {}).get("Value"):
        msg = login_json.get("Meta", {}).get("Message") or "Authentication failed"
        raise F1FantasySyncError(f"F1 Fantasy login error: {msg}")

    login_val = login_json.get("Data", {}).get("Value", {})
    guid = login_val.get("GUID")
    if not guid:
        raise F1FantasySyncError(
            "Could not obtain user GUID from F1 Fantasy session. "
            "Please re-copy your 'login-session' cookie from browser DevTools."
        )

    # Step 2: Fetch player catalog for ID resolution
    catalog: dict[str, dict] = {}
    try:
        cat_resp = session.get(DRIVERS_CATALOG_URL, timeout=10)
        if cat_resp.status_code == 200:
            cat_json = cat_resp.json()
            for p in cat_json.get("Data", {}).get("Value", []):
                pid = str(p.get("PlayerId"))
                catalog[pid] = p
    except Exception as exc:
        logger.warning(f"Could not load drivers catalog: {exc}")

    # Step 3: Fetch user gamedays to list all teams
    gamedays_url = f"{FANTASY_BASE}/services/user/gameplay/{guid}/getusergamedaysv1/1"
    try:
        gd_resp = session.get(gamedays_url, timeout=12)
        gd_resp.raise_for_status()
        raw_teams_meta = gd_resp.json().get("Data", {}).get("Value", [])
    except Exception as exc:
        raise F1FantasySyncError(f"Failed to query user teams metadata: {exc}") from exc

    if not raw_teams_meta:
        raise F1FantasySyncError("No fantasy teams found under this F1 account.")

    teams_result = []

    for meta in raw_teams_meta:
        team_no = int(meta.get("teamno", 1))
        team_name_raw = meta.get("teamname") or f"Team {team_no}"
        team_name = urllib.parse.unquote(team_name_raw).replace("+", " ")
        cumdid = meta.get("cumdid") or meta.get("prvmdid") or 1

        # Fetch detailed team lineup for this team
        team_url = f"{FANTASY_BASE}/services/user/gameplay/{guid}/getteam/1/{team_no}/{cumdid}/1"
        try:
            team_resp = session.get(team_url, timeout=12)
        except Exception as exc:
            logger.warning(f"Could not fetch team {team_no}: {exc}")
            continue

        if team_resp.status_code != 200:
            logger.warning(f"Team {team_no} returned HTTP {team_resp.status_code}")
            continue

        team_val = team_resp.json().get("Data", {}).get("Value", {})
        user_team_list = team_val.get("userTeam", [])
        if not user_team_list:
            continue

        # Match the specific team by teamno from the userTeam list
        ut = next((item for item in user_team_list if int(item.get("teamno", 0)) == team_no), None)
        if not ut:
            continue

        teambal = float(ut.get("teambal") or 0.0)
        ovpoints = float(ut.get("ovpoints") or 0.0)
        team_info = ut.get("team_info", {})
        subs_allowed = int(team_info.get("subsallowed") or 1)

        drivers = []
        constructors = []
        turbo_driver = None

        for p in ut.get("playerid", []):
            pid = str(p.get("id"))
            cat_entry = catalog.get(pid, {})
            name = _normalize_name(cat_entry.get("FUllName") or f"Player {pid}")
            pos_type = cat_entry.get("PositionName", "").upper()

            if pos_type == "CONSTRUCTOR":
                constructors.append(name)
            else:
                drivers.append(name)
                if p.get("iscaptain"):
                    turbo_driver = name

        chips_data = {
            "Wildcard": {
                "used": bool(meta.get("iswildcardtaken")),
                "round": meta.get("wildcardtakengd") or None,
            },
            "Limitless": {
                "used": bool(meta.get("islimitlesstaken")),
                "round": meta.get("limitlesstakengd") or None,
            },
            "Final Fix": {
                "used": bool(meta.get("isfinalfixtaken")),
                "round": meta.get("finalfixtakengd") or None,
            },
            "3x Boost": {
                "used": bool(meta.get("isextradrstaken")),
                "round": meta.get("extradrstakengd") or None,
            },
            "No Negative": {
                "used": bool(meta.get("isnonigativetaken")),
                "round": meta.get("nonigativetakengd") or None,
            },
            "Autopilot": {
                "used": bool(meta.get("isautopilottaken")),
                "round": meta.get("isautopilottakengd") or None,
            },
        }

        teams_result.append({
            "team_no": team_no,
            "team_name": team_name,
            "budget_remaining": round(teambal, 2),
            "current_points": round(ovpoints, 1),
            "transfers": subs_allowed,
            "turbo_driver": turbo_driver,
            "drivers": drivers,
            "constructors": constructors,
            "chips": chips_data,
        })

    teams_result.sort(key=lambda x: x["team_no"])
    return teams_result


def import_user_team(
    team_no: int = 1,
    cookie: Optional[str] = None,
    sync_chips: bool = True
) -> dict[str, Any]:
    """
    Import a specific team (1, 2, or 3) from F1 Fantasy into the local engine cache.
    Persists to MY_TEAM_PATH and syncs used chip states to CHIP_STATE_PATH.
    """
    teams = fetch_user_teams(cookie=cookie)
    target = next((t for t in teams if t["team_no"] == team_no), None)
    if not target:
        available = [t["team_no"] for t in teams]
        raise F1FantasySyncError(
            f"Team #{team_no} not found. Available teams: {available}"
        )

    saved_payload = {
        "drivers": target["drivers"],
        "constructors": target["constructors"],
        "transfers": target["transfers"],
        "budget_remaining": target["budget_remaining"],
        "current_points": target["current_points"],
        "turbo_driver": target.get("turbo_driver"),
        "team_name": target.get("team_name"),
        "team_no": target["team_no"],
        "last_synced": datetime.now().isoformat(),
    }

    # Persist to MY_TEAM_PATH
    MY_TEAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MY_TEAM_PATH, "w", encoding="utf-8") as f:
        json.dump(saved_payload, f, indent=2)

    # Legacy cache location fallback
    legacy_path = OUTPUT_DIR / "cache" / "my_team.json"
    try:
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        with open(legacy_path, "w", encoding="utf-8") as f:
            json.dump(saved_payload, f, indent=2)
    except Exception:
        logger.warning("Suppressed error", exc_info=True)

    # Sync chip inventory
    if sync_chips and "chips" in target:
        try:
            chip_state = load_chip_state()
            for chip_name, status in target["chips"].items():
                if chip_name in chip_state.get("used", {}):
                    if status["used"]:
                        chip_state["used"][chip_name] = True
                        if status.get("round"):
                            chip_state["used_on_round"][chip_name] = status["round"]
            save_chip_state(chip_state)
        except Exception as exc:
            logger.warning(f"Could not update chip state from synced team: {exc}")

    return saved_payload
