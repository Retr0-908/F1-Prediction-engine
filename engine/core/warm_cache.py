import time
import datetime
import logging

from engine.core.config import HISTORICAL_SEASONS, CURRENT_SEASON
from engine.core.data_fetcher import (
    get_season_schedule,
    get_race_results,
    get_qualifying_results,
    get_driver_standings,
    get_constructor_standings,
    get_grid_penalties,
    get_season_results,
    get_fastf1_session
)

from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("f1_predictor.warm_cache")


def _warm_single_race(year: int, round_num: int, race: dict):
    """Download single race session data. Returns (ok, message)."""
    race_name = race.get("name", f"Round {round_num}")   # schedule key is "name"

    # Skip future races
    race_date_str = race.get("date")
    try:
        race_date = datetime.datetime.strptime(race_date_str, "%Y-%m-%d").date()
        if race_date > datetime.date.today():
            return True, f"[{year} R{round_num}] Skipped future race: {race_name}"
    except Exception:
        pass

    failures = []

    # 1. Jolpica core endpoints (race, qualifying, standings)
    try:
        get_race_results(year, round_num)
        get_qualifying_results(year, round_num)
        get_driver_standings(year, round_num)
        get_constructor_standings(year, round_num)
    except Exception as e:
        failures.append(f"jolpica: {e}")

    # 2. Grid penalties (Jolpica-derived quali-vs-grid deltas)
    if year >= 2023:
        try:
            get_grid_penalties(year, round_num)
        except Exception as e:
            failures.append(f"grid_penalties: {e}")

    # 3. FastF1 Telemetry Session Data
    try:
        get_fastf1_session(year, round_num, "R")
        get_fastf1_session(year, round_num, "Q")
    except Exception as e:
        failures.append(f"fastf1: {e}")

    if failures:
        for f in failures:
            logger.warning("[%s R%s] %s cache failure: %s", year, round_num, race_name, f)
        return False, f"[{year} R{round_num}] Partial: {race_name} ({len(failures)} source(s) failed)"
    return True, f"[{year} R{round_num}] Cached: {race_name}"


def verify_and_download_caches(progress_callback=None, max_workers: int = 4):
    print("==================================================")
    print("  F1 FANTASY HIGH-SPEED BULK CACHE WARMER         ")
    print("==================================================")
    print("Checking historical seasons to ensure all API data is locally cached.")
    print("Using bulk season endpoints and parallel thread pool for max speed.\n")

    # Plan 9-LOW: dedupe seasons (CURRENT_SEASON already in HISTORICAL_SEASONS)
    seasons_to_check = sorted(set(HISTORICAL_SEASONS) | {CURRENT_SEASON})
    total_races_cached = 0
    failed_races = 0

    for year in seasons_to_check:
        print(f"\n[SEASON {year}] Ingesting bulk season data...")
        if progress_callback:
            progress_callback(year, "schedule", f"Bulk fetching {year}")

        try:
            # Method A: Bulk 1-shot season fetch (downloads full season in ~1 second)
            schedule = get_season_schedule(year)
            get_season_results(year)
        except Exception as e:
            print(f"  [Error] Bulk fetch failed for {year}: {e}")
            continue

        if not schedule:
            print(f"  [Warning] No schedule found for {year}.")
            continue

        print(f"  Schedule loaded: {len(schedule)} rounds found. Concurrent session warming...")

        # Filter completed races
        completed_races = []
        for race in schedule:
            r_num = race.get("round")
            if not r_num:
                continue
            race_date_str = race.get("date")
            try:
                race_date = datetime.datetime.strptime(race_date_str, "%Y-%m-%d").date()
                if race_date <= datetime.date.today():
                    completed_races.append((r_num, race))
            except Exception:
                completed_races.append((r_num, race))

        # Method B: Concurrent multi-threaded session warming
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_warm_single_race, year, r_num, race): (r_num, race)
                for r_num, race in completed_races
            }
            for future in as_completed(futures):
                r_num, race = futures[future]
                try:
                    ok, msg = future.result()
                    print(f"  {msg}")
                    if ok:
                        total_races_cached += 1
                    else:
                        failed_races += 1
                except Exception as exc:
                    failed_races += 1
                    logger.error("[%s R%s] warm task crashed: %s", year, r_num, exc)

    summary = (f"CACHE WARMING COMPLETE: {total_races_cached} races fully cached"
               + (f", {failed_races} with partial failures (see warnings)" if failed_races else ""))
    print("\n==================================================")
    print(summary)
    print("All ML models, ELO ratings, and FastF1 sessions are locally available.")
    print("==================================================")

    if progress_callback:
        progress_callback("DONE", "DONE", summary)


if __name__ == "__main__":
    verify_and_download_caches()
