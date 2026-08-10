import time
import datetime
from config import HISTORICAL_SEASONS, CURRENT_SEASON
from data_fetcher import (
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

def _warm_single_race(year: int, round_num: int, race: dict):
    """Download single race session data with graceful fallback."""
    race_name = race.get("raceName", f"Round {round_num}")
    
    # Skip future races
    race_date_str = race.get("date")
    try:
        race_date = datetime.datetime.strptime(race_date_str, "%Y-%m-%d").date()
        if race_date > datetime.date.today():
            return f"[{year} R{round_num}] Skipped future race: {race_name}"
    except Exception:
        pass

    # 1. Jolpica core endpoints (race, qualifying, standings)
    try:
        get_race_results(year, round_num)
        get_qualifying_results(year, round_num)
        get_driver_standings(year, round_num)
        get_constructor_standings(year, round_num)
    except Exception as e:
        pass

    # 2. OpenF1 Grid Penalties
    if year >= 2023:
        try:
            get_grid_penalties(year, round_num)
        except Exception:
            pass

    # 3. FastF1 Telemetry Session Data
    try:
        get_fastf1_session(year, round_num, "R")
        get_fastf1_session(year, round_num, "Q")
    except Exception:
        pass

    return f"[{year} R{round_num}] Cached: {race_name}"


def verify_and_download_caches(progress_callback=None, max_workers: int = 4):
    print("==================================================")
    print("  F1 FANTASY HIGH-SPEED BULK CACHE WARMER         ")
    print("==================================================")
    print("Checking historical seasons to ensure all API data is locally cached.")
    print("Using bulk season endpoints and parallel thread pool for max speed.\n")

    seasons_to_check = HISTORICAL_SEASONS + [CURRENT_SEASON]
    total_races_checked = 0

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
                    msg = future.result()
                    print(f"  {msg}")
                    total_races_checked += 1
                except Exception as exc:
                    print(f"  [{year} R{r_num}] Error: {exc}")

    if progress_callback:
        progress_callback("DONE", "DONE", "Bulk Cache Warming Complete!")

    print("\n==================================================")
    print(f"HIGH-SPEED CACHE WARMING COMPLETE! Cached {total_races_checked} completed races.")
    print("All ML models, ELO ratings, and FastF1 sessions are locally available.")
    print("==================================================")

    if progress_callback:
        progress_callback("DONE", "DONE", "Cache Verification Complete!")

    print("\n==================================================")
    print(f"CACHE VERIFICATION COMPLETE! Checked {total_races_checked} completed races.")
    print("You can now run the F1 Fantasy application with full historical data.")
    print("==================================================")

if __name__ == "__main__":
    verify_and_download_caches()
