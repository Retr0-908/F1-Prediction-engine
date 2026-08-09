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

def verify_and_download_caches(progress_callback=None):
    print("==================================================")
    print("    F1 FANTASY CACHE VERIFICATION & DOWNLOAD      ")
    print("==================================================")
    print("Checking historical seasons to ensure all API data is locally cached.")
    print("This will download missing data from Jolpica and OpenF1 APIs.")
    print("If you hit rate limits, this script will automatically wait.\n")

    seasons_to_check = HISTORICAL_SEASONS + [CURRENT_SEASON]
    
    total_races_checked = 0

    for year in seasons_to_check:
        print(f"\n[SEASON {year}] Fetching schedule and full season results...")
        if progress_callback:
            progress_callback(year, "schedule", f"Fetching schedule for {year}")
            
        try:
            schedule = get_season_schedule(year)
            get_season_results(year)
        except Exception as e:
            print(f"  [Error] Failed to fetch schedule/results for {year}: {e}")
            continue
            
        if not schedule:
            print(f"  [Warning] No schedule found for {year}.")
            continue
            
        print(f"  Found {len(schedule)} rounds for {year}.")
        
        for race in schedule:
            round_num = race.get("round")
            if not round_num:
                continue
            
            # Skip future races
            race_date_str = race.get("date")
            try:
                race_date = datetime.datetime.strptime(race_date_str, "%Y-%m-%d").date()
                if race_date > datetime.date.today():
                    print(f"  [{year} R{round_num}] Skipping future race: {race.get('raceName')}")
                    continue
            except Exception:
                pass
            
            print(f"  [{year} R{round_num}] Verifying: {race.get('raceName')}... ", end="", flush=True)
            if progress_callback:
                progress_callback(year, round_num, f"Downloading: {year} R{round_num} - {race.get('raceName')}")
            
            # The get_* functions will check the local disk cache first.
            # If missing, they download and save to disk automatically.
            
            # 1. Jolpica core endpoints
            try:
                get_race_results(year, round_num)
                get_qualifying_results(year, round_num)
                get_driver_standings(year, round_num)
                get_constructor_standings(year, round_num)
            except Exception as e:
                print(f"    [Warning] Jolpica endpoints failed: {e}")
                
            # 2. OpenF1 Grid Penalties (often missing or times out)
            if year >= 2023:
                try:
                    get_grid_penalties(year, round_num)
                except Exception as e:
                    print(f"    [Warning] OpenF1 grid penalties failed: {e}")
                    
            # 3. FastF1 Session data
            try:
                # FastF1 accepts round_num as the event identifier which is 100% reliable
                get_fastf1_session(year, round_num, "R")
                get_fastf1_session(year, round_num, "Q")
                print("OK")
            except Exception as e:
                print(f"    [Warning] FastF1 session failed: {e}")
                
            total_races_checked += 1
            # Small sleep to yield, though data_fetcher handles the 2.1s rate limit throttle internally
            time.sleep(0.01)

    if progress_callback:
        progress_callback("DONE", "DONE", "Cache Verification Complete!")

    print("\n==================================================")
    print(f"CACHE VERIFICATION COMPLETE! Checked {total_races_checked} completed races.")
    print("You can now run the F1 Fantasy application with full historical data.")
    print("==================================================")

if __name__ == "__main__":
    verify_and_download_caches()
