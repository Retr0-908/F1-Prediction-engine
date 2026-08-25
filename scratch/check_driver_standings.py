import json
from pathlib import Path

file_path = Path(__file__).resolve().parents[1] / "cache" / "api" / "jolpica__2021_1_driverStandings.json_.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Print keys and structure
mr_data = data.get("MRData", {})
standings_table = mr_data.get("StandingsTable", {})
standings_lists = standings_table.get("StandingsLists", [])

print(f"StandingsLists length: {len(standings_lists)}")
if standings_lists:
    first_list = standings_lists[0]
    print(f"Keys of first StandingsList: {list(first_list.keys())}")
    driver_standings = first_list.get("DriverStandings", [])
    print(f"DriverStandings length: {len(driver_standings)}")
    if driver_standings:
        print(f"First entry keys: {list(driver_standings[0].keys())}")
        print(f"First entry values: {driver_standings[0]}")
