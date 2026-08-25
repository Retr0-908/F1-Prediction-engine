"""
backfill_weather_labels.py — Plan 6c (one-time, ~70 archive calls).

Fetches real historical race-day weather for every training-season round
from the Open-Meteo archive and derives condition labels through the SAME
_wmo_to_condition mapping used at inference. Output: permanent
cache/weather/training_labels.json keyed "year_round".

Run:  python -m engine.tools.backfill_weather_labels
"""
import datetime
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.core.config import HISTORICAL_SEASONS, CIRCUITS          # noqa: E402
from engine.core.data_fetcher import get_season_schedule              # noqa: E402
from engine.core.weather import _wmo_to_condition                     # noqa: E402
from engine.core.paths import WEATHER_CACHE_DIR                       # noqa: E402

OUT = WEATHER_CACHE_DIR / "training_labels.json"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def fetch_day(lat: float, lon: float, date: str) -> dict | None:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": date, "end_date": date,
        "daily": ("temperature_2m_max,temperature_2m_min,"
                  "precipitation_sum,weather_code"),
        "timezone": "auto",
    }
    for attempt in range(3):
        try:
            r = requests.get(ARCHIVE, params=params, timeout=(10, 30))
            if r.status_code == 200:
                d = r.json().get("daily", {})
                code = (d.get("weather_code") or [0])[0]
                precip = float((d.get("precipitation_sum") or [0.0])[0] or 0.0)
                return {
                    "condition": _wmo_to_condition(int(code or 0), precip),
                    "temp_max": (d.get("temperature_2m_max") or [None])[0],
                    "temp_min": (d.get("temperature_2m_min") or [None])[0],
                    "precip_mm": precip,
                    "weather_code": int(code or 0),
                }
        except Exception as e:
            print(f"    retry {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return None


def main():
    labels = {}
    if OUT.exists():
        labels = json.loads(OUT.read_text(encoding="utf-8"))

    today = datetime.date.today()
    total_new = 0
    for year in sorted(HISTORICAL_SEASONS):
        if year >= datetime.date.today().year:
            continue   # current season handled post-hoc; keep KNOWN overrides
        sched = get_season_schedule(year)
        # venue coords per circuit key from config
        key_coords = {c.get("key"): (c.get("lat"), c.get("lon"))
                      for c in CIRCUITS.values() if c.get("lat")}
        for race in sched:
            rnd = race["round"]
            k = f"{year}_{rnd}"
            if k in labels:
                continue
            cfg = None
            # resolve via data_fetcher matcher (aliases incl São Paulo)
            from engine.core.data_fetcher import _match_circuit_config
            cfg = _match_circuit_config(race["name"]) or {}
            lat, lon = key_coords.get(cfg.get("key"), (None, None))
            if lat is None:
                print(f"  skip {k}: no coords for {race['name']}")
                continue
            res = fetch_day(lat, lon, race["date"])
            if res:
                labels[k] = res
                total_new += 1
                cond = res["condition"]
                print(f"  {k} {race['name'][:28]:28s} -> {cond}")
            else:
                print(f"  {k} FETCH FAILED")
            time.sleep(0.35)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(labels, indent=1), encoding="utf-8")
    print(f"\nDone. {total_new} new labels. Total: {len(labels)} -> {OUT}")


if __name__ == "__main__":
    main()
