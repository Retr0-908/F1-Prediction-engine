import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from engine.core.data_fetcher import (
    get_season_schedule,
    get_driver_standings,
    get_constructor_standings,
    get_qualifying_results,
    get_race_results,
    get_grid_penalties,
    _match_circuit_config,
    get_circuit_history
)

SEASONS = [2021, 2022, 2023, 2024, 2025]

def populate_historical_cache():
    print("Starting historical data cache verification and download...")
    
    total_checks = 0
    new_downloads = 0
    failures = []

    for year in SEASONS:
        print(f"\nChecking Season {year}...")
        try:
            schedule = get_season_schedule(year)
        except Exception as e:
            print(f"  [ERROR] Failed to fetch schedule for {year}: {e}")
            failures.append(f"{year} Schedule")
            continue

        for race in schedule:
            rnd = race["round"]
            name = race["name"]
            print(f"  Round {rnd:02d}: {name}...")

            # Endpoints to fetch for training:
            # 1. Standings for rnd - 1 (or 1 if rnd == 1)
            prev_rnd = max(1, rnd - 1)
            
            # Helper to run a fetch function and catch/report errors
            def try_fetch(label, func, *args):
                nonlocal total_checks, new_downloads
                total_checks += 1
                
                # Check if it exists in cache before calling to count new downloads
                # (our internal caching will prevent redundant API calls anyway)
                try:
                    # Let's run it. The data_fetcher handles caching.
                    # If it wasn't cached, it will fetch from Jolpica.
                    # Jolpica rate limits are handled inside data_fetcher with throttling (2.1s gap)
                    func(*args)
                except Exception as e:
                    print(f"    [FAIL] {label}: {e}")
                    failures.append(f"{year} R{rnd} {label}: {e}")

            try_fetch("Driver Standings", get_driver_standings, year, prev_rnd)
            try_fetch("Constructor Standings", get_constructor_standings, year, prev_rnd)
            try_fetch("Qualifying Results", get_qualifying_results, year, rnd)
            try_fetch("Race Results", get_race_results, year, rnd)
            try_fetch("Grid Penalties", get_grid_penalties, year, rnd)

            # Circuit History
            circuit_cfg = _match_circuit_config(name)
            circuit_id = circuit_cfg.get("key", "")
            if circuit_id:
                try_fetch(f"Circuit History ({circuit_id})", get_circuit_history, circuit_id, [year-1, year-2])

    print("\n--- CACHE POPULATION SUMMARY ---")
    print(f"Total API/Cache Checks: {total_checks}")
    print(f"Total Failures: {len(failures)}")
    if failures:
        print("\nFailed checks:")
        for f in failures[:50]:
            print(f"  - {f}")
        if len(failures) > 50:
            print(f"  ... and {len(failures) - 50} more.")
    else:
        print("\n[PASS] All historical data successfully cached with zero failures!")

if __name__ == "__main__":
    populate_historical_cache()
