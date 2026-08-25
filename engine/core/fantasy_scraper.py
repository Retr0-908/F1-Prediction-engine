"""
fantasy_scraper.py — Scrapes current driver/constructor prices from F1 Fantasy
Approach: Public stats pages via BeautifulSoup

HOW TO GET YOUR COOKIE:
1. Log in to https://fantasy.formula1.com in Chrome/Edge
2. Open DevTools (F12) → Application tab → Cookies → fantasy.formula1.com
3. Copy the full cookie string (all cookies, semicolon-separated)
4. Paste into your .env file as F1_FANTASY_COOKIE=...
   OR run: python main.py --set-cookie and paste when prompted
"""

import re
import json
import time
import requests
from bs4 import BeautifulSoup
from typing import Optional

from engine.core.config import (F1_FANTASY_COOKIE, CURRENT_SEASON, DRIVER_TEAMS_2025,
                                CONSTRUCTORS_2025, DRIVER_TEAMS_2026)
from engine.core.data_fetcher import _load_cache, _save_cache
import logging


logger = logging.getLogger("f1_predictor.fantasy_scraper")
# ─────────────────────────────────────────────
# PLAYWRIGHT DOM SCRAPER (Primary Public Method)
# ─────────────────────────────────────────────
def _is_valid_price(price: float) -> bool:
    """Sanity check: F1 Fantasy prices are typically $3M–$50M."""
    return 3.0 <= price <= 50.0


def _is_valid_driver_name(name: str) -> bool:
    """Basic check: driver names should have at least 2 words, reasonable length."""
    parts = name.strip().split()
    if len(parts) < 2:
        return False
    if len(name) < 5 or len(name) > 40:
        return False
    # Reject strings that are obviously not names (numbers, URL fragments, etc.)
    if any(c.isdigit() for c in name):
        return False
    return True


def _try_playwright_scrape(is_constructor: bool = False) -> Optional[dict]:
    try:
        from playwright.sync_api import sync_playwright
        import time
        import re
    except ImportError:
        print("    Playwright missing. Skipping powerful headless scrape.")
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("https://fantasy.formula1.com/en/statistics/details", timeout=45000)
            time.sleep(5)
            
            # Remove any cookie consent overlays that block pointer events
            try:
                page.evaluate("""() => {
                    const ids = ['sp_message_container_1406947', 'sp_message_iframe_1406947'];
                    for (const id of ids) {
                        const el = document.getElementById(id);
                        if (el) el.remove();
                    }
                    const overlays = document.querySelectorAll('[id*="consent"], [class*="consent"], [id*="overlay"], [class*="overlay"], [class*="popup"], [id*="popup"]');
                    for (const el of overlays) el.remove();
                }""")
            except Exception:
                logger.warning("Suppressed error", exc_info=True)
                pass
            
            if is_constructor:
                try:
                    page.locator("button:has-text('CONSTRUCTOR')").click(timeout=5000)
                    time.sleep(3)
                except Exception as e:
                    print(f"Failed to click CONSTRUCTOR button: {e}")
            else:
                time.sleep(2)
            
            lines = page.locator("body").inner_text().split("\n")
            browser.close()
            
            prices = {}
            roster = {}
            
            # Known driver names for cross-validation.
            # Plan 9-H1: validate against the CURRENT season's union (curated
            # seed + last known roster), not a prior-season table — validating
            # 2026 DOM names against DRIVER_TEAMS_2025 silently dropped
            # Perez/Bottas/Lindblad from the primary scrape path.
            try:
                from engine.core.data_fetcher import get_season_roster
                _live_roster = get_season_roster(CURRENT_SEASON) or {}
            except Exception:
                _live_roster = {}
            known_drivers = (set(DRIVER_TEAMS_2026.keys())
                             | set(DRIVER_TEAMS_2025.keys())
                             | set(_live_roster.keys()))
            known_drivers_lower = {n.lower() for n in known_drivers}
            
            for i, line in enumerate(lines):
                line = line.strip()
                if re.match(r"^\$\d+(\.\d*)?M$", line):
                    price_val = float(line.replace("$", "").replace("M", ""))
                    
                    if not _is_valid_price(price_val):
                        continue
                    
                    if is_constructor:
                        # For constructors, look backwards for a team name
                        team_name = None
                        for offset in range(1, 4):
                            if i >= offset:
                                candidate = lines[i - offset].strip()
                                if candidate and len(candidate) > 2 and not re.match(r"^\$", candidate):
                                    team_name = candidate
                                    break
                        if team_name:
                            clean = _clean_name(team_name)
                            prices[clean] = {
                                "price": price_val,
                                "ownership_pct": 0.0,
                                "source": "playwright",
                            }
                    else:
                        # For drivers, search backwards for first + last name + team
                        # Try multiple offset patterns since DOM structure may vary
                        name = None
                        team_name = None
                        
                        for start_offset in range(1, 6):
                            if i < start_offset + 1:
                                continue
                            # Pattern: team is closest to price, then last name, then first name
                            candidate_team = lines[i - start_offset].strip() if i >= start_offset else ""
                            candidate_last = lines[i - start_offset - 1].strip() if i >= start_offset + 1 else ""
                            candidate_first = lines[i - start_offset - 2].strip() if i >= start_offset + 2 else ""
                            
                            if candidate_first and candidate_last:
                                test_name = f"{candidate_first.title()} {candidate_last.title()}"
                                if _is_valid_driver_name(test_name):
                                    # Cross-validate against known drivers
                                    if test_name.lower() in known_drivers_lower or test_name.lower().split()[-1] in {n.lower().split()[-1] for n in known_drivers}:
                                        name = test_name
                                        team_name = candidate_team
                                        break
                        
                        if name and team_name:
                            clean_n = _clean_name(name)
                            prices[clean_n] = {
                                "price": price_val,
                                "team": team_name,
                                "ownership_pct": 0.0,
                                "form_pts": 0.0,
                                "is_constructor": False,
                                "source": "playwright",
                            }
                            roster[clean_n] = team_name
            
            if prices:
                return (prices, roster)
            return None

    except Exception as e:
        print(f"    Playwright scrape fail: {e}")
        return None

# ─────────────────────────────────────────────
# F1 Fantasy internal API endpoints (discovered via browser DevTools)
# ─────────────────────────────────────────────
FANTASY_BASE = "https://fantasy.formula1.com"
FANTASY_API  = f"{FANTASY_BASE}/feeds/drivers"
FANTASY_STAT = f"{FANTASY_BASE}/feeds/stats"
FANTASY_TEAM = f"{FANTASY_BASE}/api/v1/teams/my"

# Public stats feed (no auth required for current season standings)
PUBLIC_DRIVER_FEED = f"{FANTASY_BASE}/feeds/drivers/drivers_{CURRENT_SEASON}.json"
PUBLIC_PICK_FEED   = f"{FANTASY_BASE}/feeds/picks/picks_{CURRENT_SEASON}.json"

HEADERS_BASE = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://fantasy.formula1.com/",
    "Origin":          "https://fantasy.formula1.com",
}


def _get_headers() -> dict:
    h = dict(HEADERS_BASE)
    if F1_FANTASY_COOKIE:
        h["Cookie"] = F1_FANTASY_COOKIE
    return h


# ─────────────────────────────────────────────
# DRIVER PRICES — PRIMARY METHOD (public feed)
# ─────────────────────────────────────────────
def scrape_driver_prices(force_refresh: bool = False) -> tuple[dict[str, dict], dict[str, str]]:
    """
    Returns (prices_dict, roster_dict)
    prices_dict: { "Driver": { price, team, ... } }
    roster_dict: { "Driver": "Team Name" }
    """
    cache_key = f"fantasy_prices_{CURRENT_SEASON}"
    if not force_refresh:
        cached = _load_cache(cache_key, max_age_hours=3)
        if cached:
            # Reconstruct roster from prices
            roster = {name: d["team"] for name, d in cached.items() if not d.get("is_constructor")}
            return cached, roster

    # Attempt 1: Playwright headless browser (Public SPA)
    result = _try_playwright_scrape(is_constructor=False)
    if isinstance(result, tuple) and result:
        prices, roster = result
        _save_cache(cache_key, prices)
        return prices, roster

    # Attempt 2: Public feed (no auth)
    result = _try_public_feed()
    if isinstance(result, tuple) and result:
        prices, roster = result
        _save_cache(cache_key, prices)
        return prices, roster

    # Attempt 3: Cookie auth
    if F1_FANTASY_COOKIE:
        result = _try_cookie_auth_api()
        if isinstance(result, tuple) and result:
            prices, roster = result
            _save_cache(cache_key, prices)
            return prices, roster

    # Attempt 4: HTML scrape
    result = _try_html_scrape()
    if isinstance(result, tuple) and result:
        prices, roster = result
        _save_cache(cache_key, prices)
        return prices, roster

    # Fallback: use estimations based on last known prices
    print("\n⚠️  WARNING: Could not fetch live F1 Fantasy prices.")
    print("   Using approximate base prices. Run with --set-cookie for live data.\n")
    prices = _fallback_prices()
    roster = {name: d["team"] for name, d in prices.items()}
    return prices, roster


def _try_public_feed() -> Optional[dict]:
    """Try the public JSON driver feed."""
    try:
        resp = requests.get(PUBLIC_DRIVER_FEED, headers=HEADERS_BASE, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            parsed = _parse_driver_feed(data)
            if parsed:
                roster = {n: d["team"] for n, d in parsed.items() if not d.get("is_constructor")}
                return parsed, roster
    except Exception:
        logger.warning("Suppressed error", exc_info=True)
        pass

    # Try alternate URL patterns
    for season in [CURRENT_SEASON, str(CURRENT_SEASON)]:
        try:
            url = f"{FANTASY_BASE}/feeds/drivers/drivers_{season}.json"
            resp = requests.get(url, headers=HEADERS_BASE, timeout=10)
            if resp.status_code == 200:
                parsed = _parse_driver_feed(resp.json())
                if parsed:
                    roster = {n: d["team"] for n, d in parsed.items() if not d.get("is_constructor")}
                    return parsed, roster
        except Exception:
            logger.warning("Suppressed error", exc_info=True)
            pass
    return None


def _try_cookie_auth_api() -> Optional[dict]:
    """Use cookie to authenticate against the F1 Fantasy internal API."""
    headers = _get_headers()
    endpoints_to_try = [
        f"{FANTASY_BASE}/api/v1/players?series=1&status=all",
        f"{FANTASY_BASE}/api/v3/players?series=1&status=all&game_period_id=current",
        f"{FANTASY_BASE}/feeds/drivers/drivers_{CURRENT_SEASON}.json",
    ]
    for i, url in enumerate(endpoints_to_try):
        try:
            if i > 0:
                time.sleep(1.0)   # polite gap between auth endpoint attempts
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10))
                print(f"    [F1 Fantasy rate limit] waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 200:
                data = resp.json()
                parsed = _parse_driver_feed(data)
                if parsed:
                    roster = {n: d["team"] for n, d in parsed.items() if not d.get("is_constructor")}
                    return parsed, roster
        except Exception:
            logger.warning("Suppressed error", exc_info=True)
            continue
    return None


def _try_html_scrape() -> Optional[tuple]:
    """BeautifulSoup scrape of the fantasy page. ALWAYS returns (prices, roster)
    or None — the embedded-JSON path used to return a bare dict, crashing the
    caller's tuple unpacking."""
    try:
        url  = f"{FANTASY_BASE}/en/game/pick-team"
        resp = requests.get(url, headers=HEADERS_BASE, timeout=15)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "lxml")

        # Look for JSON data embedded in script tags
        for script in soup.find_all("script"):
            text = script.string or ""
            if "playerPrice" in text or "player_price" in text or '"price"' in text:
                # Try to extract JSON sub-objects
                json_matches = re.findall(r'\{[^{}]+?"(?:price|playerPrice)"[^{}]+?\}', text)
                if json_matches:
                    parsed = _parse_embedded_json(json_matches)
                    if parsed:
                        roster = {
                            n: d.get("team", "") for n, d in parsed.items()
                            if isinstance(d, dict)
                        }
                        return parsed, roster

        # Look for data- attributes
        cards = soup.find_all(attrs={"data-price": True})
        if cards:
            parsed = {_clean_name(c.get("data-name", "Unknown")): {
                "price":         float(c.get("data-price", 0)) / 10,
                "team":          c.get("data-team", ""),
                "ownership_pct": float(c.get("data-selected", 0)),
                "source":        "html_scrape",
            } for c in cards}
            roster = {n: d["team"] for n, d in parsed.items()}
            return parsed, roster
    except Exception:
        logger.warning("Suppressed error", exc_info=True)
        pass
    return None


def _parse_driver_feed(data: dict | list) -> Optional[dict]:
    """Parse F1 Fantasy JSON feed into standardised dict."""
    drivers = {}

    # Handle various response shapes
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = (data.get("drivers") or data.get("players") or
                 data.get("data", {}).get("drivers") or
                 data.get("Data") or [])
        if not items:
            # Recurse into first list value found
            for v in data.values():
                if isinstance(v, list) and v:
                    items = v
                    break
    else:
        return None

    if not items:
        return None

    for item in items:
        if not isinstance(item, dict):
            continue

        name = (item.get("DisplayName") or item.get("display_name") or
                item.get("name") or item.get("fullName") or
                item.get("driver_name") or "")

        # Price in items is often in 0.1M units (e.g. 250 = $25.0M)
        raw_price = (item.get("Price") or item.get("price") or
                     item.get("value") or item.get("PlayerPrice") or 0)
        price = _normalize_price(raw_price)

        team = (item.get("TeamName") or item.get("team_name") or
                item.get("constructorName") or item.get("team") or "")

        ownership = (item.get("PercentageSelected") or item.get("ownership") or
                     item.get("selected_by_percent") or item.get("selected_by") or 0.0)
        try:
            ownership = float(ownership)
        except (ValueError, TypeError):
            ownership = 0.0

        form = (item.get("FormScore") or item.get("form") or
                item.get("points_per_game") or 0.0)
        try:
            form = float(form)
        except (ValueError, TypeError):
            form = 0.0

        player_type = item.get("PlayerType") or item.get("type") or item.get("position") or ""
        is_constructor = str(player_type).lower() in ("2", "constructor", "team", "c")

        if name and price > 0:
            clean = _clean_name(name)
            drivers[clean] = {
                "price":         round(price, 1),
                "team":          team,
                "ownership_pct": round(ownership, 1),
                "form_pts":      round(form, 1),
                "is_constructor": is_constructor,
                "source":        "api_feed",
            }

    return drivers if drivers else None


def _parse_embedded_json(json_strings: list[str]) -> Optional[dict]:
    """Try to parse JSON objects found in embedded script tags."""
    drivers = {}
    for js in json_strings:
        try:
            obj = json.loads(js)
            name  = obj.get("name") or obj.get("driver") or ""
            price = _normalize_price(obj.get("price") or obj.get("playerPrice") or 0)
            if name and price > 0:
                drivers[_clean_name(name)] = {
                    "price":         round(price, 1),
                    "team":          obj.get("team", ""),
                    "ownership_pct": float(obj.get("ownership") or 0),
                    "form_pts":      0.0,
                    "is_constructor": False,
                    "source":        "html_embedded",
                }
        except Exception:
            logger.warning("Suppressed error", exc_info=True)
            continue
    return drivers if drivers else None


def _clean_name(name: str) -> str:
    """Normalise driver names for consistent matching."""
    import unicodedata

    name = name.strip()
    # Remove extra spaces
    name = " ".join(name.split())
    # Title case
    if name.isupper():
        name = name.title()
    # Remove accents/diacritics
    name = "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    )
    # Map Andrea Kimi -> Kimi
    if name.startswith("Andrea Kimi "):
        name = "Kimi " + name[len("Andrea Kimi "):]
    return name


def scrape_constructor_prices(force_refresh: bool = False) -> dict[str, dict]:
    """
    Returns constructor prices.
    """
    cache_key = f"fantasy_constructor_prices_{CURRENT_SEASON}"
    if not force_refresh:
        cached = _load_cache(cache_key, max_age_hours=3)
        if cached:
            return cached

    # Attempt 1: Picks feed for constructor data (Most reliable for constructors)
    try:
        url  = PUBLIC_PICK_FEED
        resp = requests.get(url, headers=HEADERS_BASE, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            ctors = _extract_constructors(data)
            if ctors:
                _save_cache(cache_key, ctors)
                return ctors
    except Exception:
        logger.warning("Suppressed error", exc_info=True)
        pass

    # Attempt 2: Playwright headless browser (Public SPA)
    print("    Launching headless browser to scrape constructors DOM...")
    result = _try_playwright_scrape(is_constructor=True)
    if result:
        # _try_playwright_scrape returns (prices, roster) tuple - extract prices only
        if isinstance(result, tuple):
            prices = result[0]
        else:
            prices = result
        if prices and isinstance(prices, dict):
            _save_cache(cache_key, prices)
            return prices

    # Fallback hardcoded estimates — deliberately NOT cached: fabricated prices
    # served from cache for 72h would silently drive budget math and LP
    # optimization while looking like live market data.
    return _fallback_constructor_prices()


def _extract_constructors(data) -> Optional[dict]:
    """Extract constructor entries from feed data."""
    if isinstance(data, dict):
        constructors_raw = (data.get("constructors") or data.get("teams") or
                            data.get("Constructors") or [])
        if not constructors_raw and isinstance(data.get("data"), dict):
            constructors_raw = data["data"].get("constructors", [])
        return _parse_constructor_list(constructors_raw)
    return None


def _normalize_price(raw) -> float:
    """Coerce feed prices into $M units with sanity validation.

    Handles feeds storing 0.1M units (e.g. 55 = $5.5M) and rejects values
    outside the plausible F1 Fantasy asset range instead of letting an absurd
    price silently exclude a driver from the optimizer."""
    try:
        price = float(raw)
    except (ValueError, TypeError):
        return 0.0
    if price > 100:          # stored as 0.1M units
        price /= 10.0
    if not (2.0 <= price <= 40.0):
        return 0.0
    return price


def _parse_constructor_list(items: list) -> Optional[dict]:
    out = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        name = (item.get("Name") or item.get("name") or item.get("TeamName") or "")
        raw  = (item.get("Price") or item.get("price") or item.get("value") or 0)
        price = _normalize_price(raw)
        own  = float(item.get("PercentageSelected") or item.get("ownership") or 0)
        if name and price > 0:
            out[_clean_name(name)] = {
                "price":         round(price, 1),
                "ownership_pct": round(own, 1),
                "source":        "api_feed",
            }
    return out if out else None


# ─────────────────────────────────────────────
# FALLBACK PRICES (approximate 2025 season estimates)
# Based on publicly reported pre-season valuations
# Will be updated by successful scrape runs
# ─────────────────────────────────────────────
def _fallback_prices() -> dict[str, dict]:
    """Last-resort approximate prices if all scraping fails. (2026 estimates)"""
    return {
        "Max Verstappen":    {"price": 30.0, "team": "Red Bull",     "ownership_pct": 45.0, "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Isack Hadjar":      {"price": 9.0,  "team": "Red Bull",     "ownership_pct": 5.0,  "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Lando Norris":      {"price": 28.0, "team": "McLaren",      "ownership_pct": 40.0, "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Oscar Piastri":     {"price": 24.0, "team": "McLaren",      "ownership_pct": 35.0, "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Charles Leclerc":   {"price": 25.0, "team": "Ferrari",      "ownership_pct": 33.0, "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Lewis Hamilton":    {"price": 23.0, "team": "Ferrari",      "ownership_pct": 30.0, "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "George Russell":    {"price": 21.0, "team": "Mercedes",     "ownership_pct": 25.0, "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Kimi Antonelli":    {"price": 12.0, "team": "Mercedes",     "ownership_pct": 15.0, "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Fernando Alonso":   {"price": 14.0, "team": "Aston Martin", "ownership_pct": 18.0, "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Lance Stroll":      {"price": 9.0,  "team": "Aston Martin", "ownership_pct": 6.0,  "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Pierre Gasly":      {"price": 10.0, "team": "Alpine",       "ownership_pct": 8.0,  "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Franco Colapinto":  {"price": 8.0,  "team": "Alpine",       "ownership_pct": 6.0,  "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Liam Lawson":       {"price": 11.0, "team": "Racing Bulls", "ownership_pct": 12.0, "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Arvid Lindblad":    {"price": 7.0,  "team": "Racing Bulls", "ownership_pct": 4.0,  "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Carlos Sainz":      {"price": 16.0, "team": "Williams",     "ownership_pct": 20.0, "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Alexander Albon":   {"price": 10.0, "team": "Williams",     "ownership_pct": 10.0, "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Nico Hulkenberg":   {"price": 9.5,  "team": "Audi",         "ownership_pct": 7.0,  "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Gabriel Bortoleto": {"price": 7.5,  "team": "Audi",         "ownership_pct": 5.0,  "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Esteban Ocon":      {"price": 8.5,  "team": "Haas",         "ownership_pct": 6.0,  "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Oliver Bearman":    {"price": 8.0,  "team": "Haas",         "ownership_pct": 6.0,  "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Sergio Perez":      {"price": 11.0, "team": "Cadillac",     "ownership_pct": 10.0, "form_pts": 0, "is_constructor": False, "source": "estimate"},
        "Valtteri Bottas":   {"price": 8.0,  "team": "Cadillac",     "ownership_pct": 5.0,  "form_pts": 0, "is_constructor": False, "source": "estimate"},
    }


def _fallback_constructor_prices() -> dict[str, dict]:
    return {
        "Red Bull":      {"price": 33.5, "ownership_pct": 40.0, "source": "estimate"},
        "McLaren":       {"price": 32.0, "ownership_pct": 38.0, "source": "estimate"},
        "Ferrari":       {"price": 26.0, "ownership_pct": 35.0, "source": "estimate"},
        "Mercedes":      {"price": 22.0, "ownership_pct": 25.0, "source": "estimate"},
        "Aston Martin":  {"price": 10.0, "ownership_pct": 12.0, "source": "estimate"},
        "Williams":      {"price": 11.0, "ownership_pct": 15.0, "source": "estimate"},
        "Racing Bulls":  {"price": 9.0,  "ownership_pct": 9.0,  "source": "estimate"},
        "Alpine":        {"price": 8.5,  "ownership_pct": 7.0,  "source": "estimate"},
        "Haas":          {"price": 8.0,  "ownership_pct": 6.0,  "source": "estimate"},
        "Audi":          {"price": 7.0,  "ownership_pct": 5.0,  "source": "estimate"},
        "Cadillac":      {"price": 6.5,  "ownership_pct": 4.0,  "source": "estimate"},
    }


def fuzzy_match_driver(input_name: str, price_data: dict) -> Optional[str]:
    """Case-insensitive partial match for driver name lookup."""
    input_lower = input_name.lower().strip()
    # Exact match
    for name in price_data:
        if name.lower() == input_lower:
            return name
    # Last name match
    for name in price_data:
        if name.lower().split()[-1] == input_lower.split()[-1]:
            return name
    # Partial match
    for name in price_data:
        if input_lower in name.lower() or name.lower() in input_lower:
            return name
    return None


# Removed cookie auth my_team fetches

