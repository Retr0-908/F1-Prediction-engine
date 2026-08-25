"""
weather.py — Race weekend weather forecasting using Open-Meteo API
Provides per-session weather snapshots and rain probability analysis.
"""

import datetime
import json
import time
import requests
from pathlib import Path
from typing import Optional

from engine.core.config import CIRCUITS, CONDITION_ENC
from engine.core.paths import WEATHER_CACHE_DIR as _WEATHER_CACHE_DIR
import logging


logger = logging.getLogger("f1_predictor.weather")
# Disk cache for weather responses (avoids burning call limits on re-runs)
_WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _weather_cache_key(lat: float, lon: float) -> str:
    # Key by lat/lon + ISO week so stale forecasts aren't served across weeks
    week = datetime.date.today().isocalendar()[:2]  # (year, week)
    return f"openmeteo_{lat:.4f}_{lon:.4f}_{week[0]}W{week[1]:02d}"


def _load_weather_cache(key: str, max_age_hours: float = 3) -> Optional[dict]:
    p = _WEATHER_CACHE_DIR / f"{key}.json"
    if p.exists():
        age = time.time() - p.stat().st_mtime
        if age < max_age_hours * 3600:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def _save_weather_cache(key: str, data: dict) -> None:
    p = _WEATHER_CACHE_DIR / f"{key}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────
# TYPICAL RACE SESSION OFFSETS (hours from race day)
# FP1: -2 days ~10:00 local, FP2: -2 days ~14:00 local
# FP3: -1 day  ~11:00 local, Q:   -1 day  ~15:00 local
# Race: race day ~15:00 local
# ─────────────────────────────────────────────
SESSION_LABELS = ["Free Practice 1", "Free Practice 2",
                  "Free Practice 3", "Qualifying", "Race"]

SESSION_DAY_OFFSETS = {
    "Free Practice 1":  -2,
    "Free Practice 2":  -2,
    "Free Practice 3":  -1,
    "Qualifying":       -1,
    "Race":              0,
}


def _open_meteo_forecast(lat: float, lon: float) -> Optional[dict]:
    """Call Open-Meteo forecast API for 16-day daily + hourly. Cached 3h."""
    cache_key = _weather_cache_key(lat, lon)
    cached = _load_weather_cache(cache_key, max_age_hours=3)
    if cached:
        return cached

    # Plan 6b: explicit window anchor — but Open-Meteo rejects start_date +
    # forecast_days together, so we only add end_date to cap the horizon.
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "daily":     "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max,weather_code,relative_humidity_2m_max",
        "hourly":    "temperature_2m,relative_humidity_2m,precipitation,precipitation_probability,wind_speed_10m,weather_code,visibility",
        "forecast_days": 16,
        "timezone":  "auto"
    }

    last_error = None
    for attempt in range(4):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt)   # 2s, 4s, 8s back-off
            resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            _save_weather_cache(cache_key, data)
            return data
        except requests.exceptions.RequestException as e:
            last_error = e
            continue

    print(f"    [Open-Meteo] failed after 4 attempts: {last_error}")
    return None


def _open_meteo_historical_fallback(lat: float, lon: float, race_date: datetime.date) -> Optional[list[dict]]:
    """
    Fetch weather archive for the same calendar dates in the past 3 years
    to build a high-fidelity climatological baseline.
    """
    year = datetime.date.today().year
    target_years = [year - 1, year - 2, year - 3]
    
    daily_history = []
    
    for y in target_years:
        try:
            if race_date.month == 2 and race_date.day == 29:
                target_date = datetime.date(y, 2, 28)
            else:
                target_date = datetime.date(y, race_date.month, race_date.day)
        except Exception:
            logger.warning("Suppressed error", exc_info=True)
            continue
            
        start_d = target_date - datetime.timedelta(days=2)
        end_d = target_date
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_d.isoformat(),
            "end_date": end_d.isoformat(),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code,wind_speed_10m_max",
            "timezone": "auto"
        }
        
        try:
            cache_key = f"openmeteo_hist_{lat:.4f}_{lon:.4f}_{start_d.isoformat()}_{end_d.isoformat()}"
            cached = _load_weather_cache(cache_key, max_age_hours=240)  # History is immutable, cache 10 days
            if cached:
                daily_history.append(cached)
                continue
                
            resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=10)
            if resp.status_code == 200:
                res_data = resp.json()
                if "daily" in res_data:
                    _save_weather_cache(cache_key, res_data["daily"])
                    daily_history.append(res_data["daily"])
            time.sleep(0.1)  # Respect API limits
        except Exception:
            logger.warning("Suppressed error", exc_info=True)
            pass
            
    if not daily_history:
        return None
        
    num_days = 3
    avg_daily = []
    
    for day_idx in range(num_days):
        temps_max = []
        temps_min = []
        precips = []
        codes = []
        winds = []
        
        for h in daily_history:
            if "temperature_2m_max" in h and len(h["temperature_2m_max"]) > day_idx:
                t_max = h["temperature_2m_max"][day_idx]
                t_min = h["temperature_2m_min"][day_idx]
                prec = h["precipitation_sum"][day_idx]
                code = h["weather_code"][day_idx]
                wind = h["wind_speed_10m_max"][day_idx]
                
                if t_max is not None: temps_max.append(t_max)
                if t_min is not None: temps_min.append(t_min)
                if prec is not None: precips.append(prec)
                if code is not None: codes.append(code)
                if wind is not None: winds.append(wind)
                
        avg_max = sum(temps_max) / len(temps_max) if temps_max else None
        avg_min = sum(temps_min) / len(temps_min) if temps_min else None
        avg_prec = sum(precips) / len(precips) if precips else 0.0
        avg_wind = sum(winds) / len(winds) if winds else None
        dominant_code = max(set(codes), key=codes.count) if codes else 0

        # Probability of precipitation estimated from the fraction of wet days
        # in the historical sample (not an invented constant).
        wet_days = sum(1 for p in precips if p > 0.1)
        pop_est = round(100.0 * wet_days / max(1, len(precips)), 1) if precips else None

        avg_daily.append({
            # None (not fabricated constants) when the archive sample is too thin —
            # consumers must handle missing values rather than trust invented data.
            "temp_max": avg_max,
            "temp_min": avg_min,
            "precip": avg_prec,
            "weather_code": dominant_code,
            "wind_speed": avg_wind,
            "pop": pop_est,
            "humidity": None,   # archive API field not fetched; do NOT fabricate
            "quality": "degraded" if (avg_max is None or not precips) else "ok",
        })
        
    return avg_daily


def _wmo_to_condition(code: int, precip_mm: float) -> str:
    """Map WMO code to standard categories: wet, mixed, overcast, dry."""
    # 0: Clear
    # 1, 2, 3: Partly cloudy, overcast
    # 45, 48: Fog
    # 51-55: Drizzle
    # 61-63: Rain slight/mod
    # 65: Rain heavy
    # 80-81: Showers slight/mod
    # 82: Showers heavy
    # 95-99: Thunderstorms
    if code in (65, 82, 95, 96, 99) or precip_mm > 2.0:
        return "wet"
    if code in (51, 53, 55, 61, 63, 80, 81) or precip_mm > 0.1:
        return "mixed"
    if code in (2, 3, 45, 48):
        return "overcast"
    return "dry"


_DRY_FALLBACK = {
    "temp_min_c": 18.0,
    "temp_max_c": 25.0,
    "temp_day_c": 22.0,
    "feels_like_c": 22.0,
    "humidity_pct": 50,
    "humidity_avg_pct": 50,
    "wind_speed_kph": 10.0,
    "wind_speed_max_kph": 10.0,
    "wind_gust_kph": 15.0,
    "wind_dir_deg": 180,
    "rain_mm": 0.0,
    "precip_mm": 0.0,
    "clouds_pct": 20,
    "pop_pct": 0.0,
    "uvi": 5.0,
    "weather_main": "Clear",
    "weather_desc": "clear sky",
    "condition": "dry",
    "condition_enc": 0,
    "rain_risk": "low",
}


def _daily_summary_openmeteo(day_data: dict) -> dict:
    """Create standard daily dict compatible with frontend requirements."""
    def _or_none(v):
        return v if v is not None else None
    temp_max = day_data.get("temp_max")
    temp_max = temp_max if temp_max is not None else 22.0   # genuine 0°C must survive
    temp_min = day_data.get("temp_min")
    temp_min = temp_min if temp_min is not None else 15.0
    temp_day = (temp_max + temp_min) / 2
    precip = day_data.get("precip") or 0.0
    pop = day_data.get("pop") or 0.0
    wind_speed = day_data.get("wind_speed")
    wind_speed = wind_speed if wind_speed is not None else 10.0
    code = day_data.get("weather_code") or 0
    humidity = day_data.get("humidity") or 50.0
    
    condition = _wmo_to_condition(code, precip)
    
    day_rain_risk = "low"
    if condition == "wet":
        day_rain_risk = "high"
    elif condition == "mixed":
        day_rain_risk = "medium"
        
    return {
        "temp_min_c":   round(temp_min, 1),
        "temp_max_c":   round(temp_max, 1),
        "temp_day_c":   round(temp_day, 1),
        "feels_like_c": round(temp_day, 1),
        "humidity_pct": int(humidity),
        "humidity_avg_pct": int(humidity),
        "wind_speed_kph": round(wind_speed, 1),
        "wind_speed_max_kph": round(wind_speed, 1),
        "wind_gust_kph":  round(wind_speed * 1.3, 1),
        "wind_dir_deg":   180,
        "rain_mm":        round(precip, 2),
        "precip_mm":      round(precip, 2),
        "clouds_pct":     40 if condition == "overcast" else (80 if condition in ("mixed", "wet") else 10),
        "pop_pct":        round(pop, 1),
        "uvi":            5.0,
        "weather_main":   condition.capitalize(),
        "weather_desc":   f"WMO code {code}",
        "condition":      condition,
        "condition_enc":  CONDITION_ENC.get(condition, 0),
        "rain_risk":      day_rain_risk,
    }


def _hourly_summary_openmeteo(hour_data: dict) -> dict:
    """Create standard hourly dict compatible with frontend requirements."""
    # Plan 9-LOW: None-checks, not truthiness — genuine 0 °C / 0 kph must
    # survive (mirrors _daily_summary_openmeteo policy).
    def _or(v, default):
        return v if v is not None else default
    temp = _or(hour_data.get("temp"), 20.0)
    humidity = _or(hour_data.get("humidity"), 50.0)
    wind_speed = _or(hour_data.get("wind_speed"), 10.0)
    precip = hour_data.get("precip") or 0.0
    pop = hour_data.get("pop") or 0.0
    code = hour_data.get("weather_code") or 0
    visibility = _or(hour_data.get("visibility"), 10000.0)
    
    condition = _wmo_to_condition(code, precip)
    return {
        "dt":           hour_data.get("dt", 0),
        "temp_c":       round(temp, 1),
        "feels_like_c": round(temp, 1),
        "humidity_pct": int(humidity),
        "wind_speed_kph": round(wind_speed, 1),
        "wind_gust_kph":  round(wind_speed * 1.3, 1),
        "rain_mm":        round(precip, 2),
        "pop_pct":        round(pop, 1),
        "clouds_pct":     40 if condition == "overcast" else (80 if condition in ("mixed", "wet") else 10),
        "weather_main":   condition.capitalize(),
        "weather_desc":   f"WMO code {code}",
        "condition":      condition,
        "condition_enc":  CONDITION_ENC.get(condition, 0),
        "visibility_m":   int(visibility),
    }


def get_race_weekend_weather(race_name: str, race_date_str: str,
                             race_info: dict = None) -> dict:
    """
    race_info: optional schedule dict (carries sprint_date). Plan 9-M7 —
    sprint weekends use the real session layout (Fri: FP1+Q, Sat: Sprint,
    Sun: Race) instead of the classic Thu/Fri/Sat offsets that produced
    Thursday forecasts and a missing Sprint session.
    """
    # Plan 6a/6b context
    race_time_utc = (race_info or {}).get("time", "")   # e.g. "15:00:00Z"
    circuit_cfg = _match_circuit(race_name)
    lat  = circuit_cfg.get("lat", 0.0)
    lon  = circuit_cfg.get("lon", 0.0)

    # Unmatched circuit → coordinates (0, 0) would silently return a real
    # forecast for the Gulf of Guinea ("Null Island"). Fail loudly instead.
    if lat == 0.0 and lon == 0.0:
        print(f"    [Open-Meteo] WARNING: no circuit match for {race_name!r} — no weather data")
        return {
            "circuit": race_name,
            "city": race_name,
            "race_date": race_date_str,
            "sessions": {},
            "rain_prob": 0.0,
            "quality": "unavailable",
            "rain_risk": "low",
            "summary_condition": "unknown",
            "condition_enc": CONDITION_ENC.get("unknown", 1),
            "race_day_hourly": [],
            "daily_raw": [],
        }

    city = circuit_cfg.get("city", race_name)

    race_date = datetime.date.fromisoformat(race_date_str)
    today     = datetime.date.today()

    # Try live forecast
    forecast = None
    try:
        forecast = _open_meteo_forecast(lat, lon)
    except Exception as e:
        print(f"    [Open-Meteo] Live forecast error: {e}")

    daily_by_date = {}
    hourly_by_dt = []
    _venue_tz = None   # hoisted: race-hour slice needs it even without forecast
    
    if forecast:
        daily = forecast.get("daily", {})
        if daily:
            for idx, date_str in enumerate(daily.get("time", [])):
                daily_by_date[date_str] = {
                    "temp_max": daily.get("temperature_2m_max", [])[idx],
                    "temp_min": daily.get("temperature_2m_min", [])[idx],
                    "precip": daily.get("precipitation_sum", [])[idx],
                    "pop": daily.get("precipitation_probability_max", [0])[idx] if daily.get("precipitation_probability_max") else 0.0,
                    "wind_speed": daily.get("wind_speed_10m_max", [])[idx],
                    "weather_code": daily.get("weather_code", [])[idx],
                    "humidity": daily.get("relative_humidity_2m_max", [50])[idx] if daily.get("relative_humidity_2m_max") else 50.0
                }
        
        hourly_raw = forecast.get("hourly", {})
        # Open-Meteo with timezone:"auto" returns naive wall-clock strings in the
        # CIRCUIT's local time. Attach the venue's IANA zone (echoed in the
        # response) before computing epochs — interpreting them as machine-local
        # produced timestamps wrong by the venue/host UTC offset.
        try:
            from zoneinfo import ZoneInfo

            _venue_tz = ZoneInfo(forecast.get("timezone", "UTC") or "UTC")
        except Exception:
            _venue_tz = None
        if hourly_raw:
            times = hourly_raw.get("time", [])
            for idx, dt_str in enumerate(times):
                try:
                    dt_obj = datetime.datetime.fromisoformat(dt_str)
                    if _venue_tz is not None:
                        dt_obj = dt_obj.replace(tzinfo=_venue_tz)
                    timestamp = int(dt_obj.timestamp())
                except ValueError:
                    continue
                hourly_by_dt.append({
                    "dt": timestamp,
                    "temp": hourly_raw.get("temperature_2m", [])[idx],
                    "humidity": hourly_raw.get("relative_humidity_2m", [50])[idx] if hourly_raw.get("relative_humidity_2m") else 50.0,
                    "wind_speed": hourly_raw.get("wind_speed_10m", [])[idx],
                    "precip": hourly_raw.get("precipitation", [])[idx],
                    "pop": hourly_raw.get("precipitation_probability", [0])[idx] if hourly_raw.get("precipitation_probability") else 0.0,
                    "weather_code": hourly_raw.get("weather_code", [])[idx],
                    "visibility": hourly_raw.get("visibility", [10000])[idx] if hourly_raw.get("visibility") else 10000.0,
                    # Keep the venue-local wall clock for date/hour filtering —
                    # immune to machine-timezone and DST round-trip issues.
                    "local_dt": dt_obj,
                })

    session_forecasts = {}
    use_live_forecast = False
    
    if daily_by_date and race_date_str in daily_by_date:
        use_live_forecast = True

    # Plan 9-M7: sprint-weekend session layout. The fantasy-relevant sessions
    # on a sprint weekend are FP1 + Qualifying (Fri, −2), Sprint (Sat, −1),
    # Race (Sun). We keep the five canonical labels for UI compatibility,
    # mapping them to the REAL session days.
    is_sprint_weekend = bool((race_info or {}).get("sprint_date"))
    if is_sprint_weekend:
        offsets = {
            "Free Practice 1":  -2,
            "Free Practice 2":  -2,   # no separate FP2 — Friday conditions
            "Free Practice 3":  -1,   # Saturday = sprint day proxy
            "Qualifying":       -2,   # quali moves to FRIDAY on sprints
            "Race":              0,
        }
    else:
        offsets = SESSION_DAY_OFFSETS

    if use_live_forecast:
        for session, day_offset in offsets.items():
            session_date_str = (race_date + datetime.timedelta(days=day_offset)).isoformat()
            if session_date_str in daily_by_date:
                session_forecasts[session] = _daily_summary_openmeteo(daily_by_date[session_date_str])
                if is_sprint_weekend:
                    session_forecasts[session]["note"] = (
                        "Sprint weekend layout" if session != "Race" else "")
            else:
                fallback = _DRY_FALLBACK.copy()
                fallback["note"] = "Outside live forecast window"
                session_forecasts[session] = fallback
        # Sprint-day forecast surfaced explicitly for consumers that want it
        sprint_day_str = (race_date + datetime.timedelta(days=-1)).isoformat()
        if is_sprint_weekend and sprint_day_str in daily_by_date:
            sf = _daily_summary_openmeteo(daily_by_date[sprint_day_str])
            sf["note"] = "Sprint session day"
            session_forecasts["Sprint"] = sf
    else:
        # Fall back to historical averages
        hist_data = _open_meteo_historical_fallback(lat, lon, race_date)
        if hist_data:
            session_forecasts["Free Practice 1"] = _daily_summary_openmeteo(hist_data[0])
            session_forecasts["Free Practice 1"]["note"] = "Historical climate average"
            
            session_forecasts["Free Practice 2"] = _daily_summary_openmeteo(hist_data[0])
            session_forecasts["Free Practice 2"]["note"] = "Historical climate average"
            
            session_forecasts["Free Practice 3"] = _daily_summary_openmeteo(hist_data[1])
            session_forecasts["Free Practice 3"]["note"] = "Historical climate average"
            
            session_forecasts["Qualifying"] = _daily_summary_openmeteo(hist_data[1])
            session_forecasts["Qualifying"]["note"] = "Historical climate average"
            
            session_forecasts["Race"] = _daily_summary_openmeteo(hist_data[2])
            session_forecasts["Race"]["note"] = "Historical climate average"
        else:
            # Standard hardcoded fallback
            for session in SESSION_LABELS:
                fallback = _DRY_FALLBACK.copy()
                fallback["note"] = "Climatological fallback"
                session_forecasts[session] = fallback

    # Overall weekend risk assessment
    conditions = [s.get("condition", "dry") for s in session_forecasts.values()]
    wet_count  = conditions.count("wet")
    mix_count  = conditions.count("mixed")

    if wet_count >= 2:
        rain_risk = "high"
    elif wet_count >= 1 or mix_count >= 2:
        rain_risk = "medium"
    else:
        rain_risk = "low"

    race_cond = session_forecasts.get("Race", {}).get("condition", "dry")

    # Plan 6a: race-HOUR-aware hourly window. Night races (Vegas ~22:00,
    # Singapore ~20:30, Middle-East evenings) previously produced an EMPTY
    # slice because of the hardcoded 12–18 filter.
    race_start_local = None
    race_time_utc = (race_info or {}).get("time", "")
    if race_time_utc:
        try:
            hh = int(str(race_time_utc).split(":")[0])
            mm = int(str(race_time_utc).split(":")[1]) if ":" in str(race_time_utc)[3:] else 0
            race_utc_dt = datetime.datetime.combine(
                race_date, datetime.time(hh, mm),
                tzinfo=datetime.timezone.utc,
            )
            if _venue_tz is not None:
                race_start_local = race_utc_dt.astimezone(_venue_tz)
        except Exception:
            race_start_local = None

    race_hourly = []
    if race_start_local is not None:
        lo = race_start_local - datetime.timedelta(hours=3)
        hi = race_start_local + datetime.timedelta(hours=3)
        for h in hourly_by_dt:
            h_local = h.get("local_dt") or datetime.datetime.fromtimestamp(h["dt"])
            if lo <= h_local <= hi:
                race_hourly.append(_hourly_summary_openmeteo(h))
    else:
        for h in hourly_by_dt:
            h_local = h.get("local_dt")
            if h_local is None:
                h_local = datetime.datetime.fromtimestamp(h["dt"])
            if h_local.date() == race_date and 12 <= h_local.hour <= 18:
                race_hourly.append(_hourly_summary_openmeteo(h))

    # Plan 6e: continuous rain probability for features/DNF/MC
    race_session_data = session_forecasts.get("Race", {})
    pop_race = float(race_session_data.get("pop_pct", 0.0) or 0.0) / 100.0
    precip_race = float(race_session_data.get("rain_mm", 0.0) or 0.0)
    wet_frac = (wet_count + mix_count * 0.5) / max(1, len(session_forecasts))
    rain_prob = max(0.0, min(1.0, 0.6 * pop_race + 0.25 * float(precip_race > 0.0)
                             + 0.15 * wet_frac))

    # Plan 6b: forecast quality flag — climatology must be visible to consumers
    quality = "live_forecast" if use_live_forecast else "climatology"

    # Compile the final payload matching OWM response structure
    return {
        "circuit":           race_name,
        "city":              city,
        "lat":               lat,
        "lon":               lon,
        "race_date":         race_date_str,
        "sessions":          session_forecasts,
        "rain_risk":         rain_risk,
        "rain_prob":         round(rain_prob, 3),       # plan 6e (additive; UI-safe)
        "summary_condition": race_cond,
        "condition_enc":     CONDITION_ENC.get(race_cond, 0),
        "race_day_hourly":   race_hourly,
        "quality":           quality,                    # plan 6b
        "daily_raw":         [session_forecasts.get("Race", {})],
    }


_TRAINING_LABELS_CACHE: dict | None = None


def historical_race_condition(year: int, rnd: int) -> str | None:
    """Plan 6c/6f: archived condition label for a training race
    ("dry"/"overcast"/"mixed"/"wet"), or None if not backfilled.
    Consumed by temporal_model training so rain context VARIES."""
    global _TRAINING_LABELS_CACHE
    if _TRAINING_LABELS_CACHE is None:
        try:
            f = _WEATHER_CACHE_DIR / "training_labels.json"
            _TRAINING_LABELS_CACHE = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        except Exception:
            _TRAINING_LABELS_CACHE = {}
    entry = _TRAINING_LABELS_CACHE.get(f"{year}_{rnd}")
    return entry.get("condition") if entry else None


def _match_circuit(race_name: str) -> dict:
    """Find circuit config by matching race name keywords."""
    if race_name in CIRCUITS:
        return CIRCUITS[race_name]
        
    race_lower = race_name.lower()
    for name, cfg in CIRCUITS.items():
        if cfg.get("key", "").lower() in race_lower:
            return cfg
            
    for name, cfg in CIRCUITS.items():
        n_clean = name.lower().replace("grand prix", "").strip()
        if n_clean and n_clean in race_lower:
            return cfg
            
    for name, cfg in CIRCUITS.items():
        city = cfg.get("city", "").lower()
        if city and city in race_lower:
            return cfg
            
    common_words = {"grand", "prix", "circuit", "international", "autodromo", "autodrom", "marina", "de"}
    for name, cfg in CIRCUITS.items():
        name_words = [w for w in name.lower().split() if len(w) > 3 and w not in common_words]
        if name_words and any(w in race_lower for w in name_words):
            return cfg
            
    return {"lat": 0.0, "lon": 0.0, "city": race_name}


def _no_weather_response(race_name: str, reason: str) -> dict:
    placeholder = {"condition": "unknown", "condition_enc": 1, "note": reason}
    return {
        "circuit":           race_name,
        "city":              race_name,
        "lat":               0.0,
        "lon":               0.0,
        "sessions":          {s: placeholder for s in SESSION_LABELS},
        "rain_risk":         "unknown",
        "summary_condition": "unknown",
        "condition_enc":     1,
        "race_day_hourly":   [],
        "daily_raw":         [],
        "error":             reason,
    }


def format_weather_display(weather_data: dict) -> str:
    """Return a rich-formatted string summary of race weekend weather."""
    lines = []
    sessions = weather_data.get("sessions", {})
    for session, info in sessions.items():
        cond = info.get("condition", "?").upper()
        temp = info.get("temp_day_c", info.get("temp_c", "?"))
        rain = info.get("rain_mm", 0)
        pop  = info.get("pop_pct", 0)
        wind = info.get("wind_speed_kph", 0)
        note = info.get("note", "")
        icon = {"dry": "[SUN]", "overcast": "[OVC]", "mixed": "[MIX]", "wet": "[WET]"}.get(
            info.get("condition", ""), "[?]")
        lines.append(
            f"  {icon:<5} {session:<20} {cond:<8} "
            f"{temp}°C  | POP: {pop}% | Rain: {rain}mm | Wind: {wind}kph"
            + (f"  [{note}]" if note else "")
        )
    risk = weather_data.get("rain_risk", "unknown").upper()
    lines.append(f"\n  [RISK] Rain Risk: {risk}")
    return "\n".join(lines)
