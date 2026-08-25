import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from engine.models.predictor import F1Predictor
from engine.core.data_fetcher import get_next_race
from engine.core.weather import get_race_weekend_weather
from engine.core.fantasy_scraper import scrape_driver_prices
from engine.core.config import CURRENT_SEASON

print("Initializing predictor...")
predictor = F1Predictor()

print("Fetching context data...")
race = get_next_race(CURRENT_SEASON)
weather = get_race_weekend_weather(race["name"], race["date"])
driver_prices, dynamic_roster = scrape_driver_prices(force_refresh=False)

circuit_cfg = {
    "key": race.get("circuit_id", "bahrain"),
    "track_type": "permanent",
    "overtaking": "medium",
    "power_unit": "medium",
    "downforce": "medium",
}

print(f"Loading context for race: {race['name']}...")
predictor.load_context(circuit_cfg, weather, roster=dynamic_roster, mode="pre-quali", race_info=race)

print("Predicting finishing order...")
# Mock predicting finishing order by checking how is_rookie is computed
drivers = list(predictor._roster.keys())
print("\nRoster keys vs driver_form keys:")
print(f"Roster size: {len(drivers)}, Form size: {len(predictor._driver_form)}")

for drv in ["Nico Hulkenberg", "Sergio Perez", "Andrea Kimi Antonelli", "Gabriel Bortoleto"]:
    # Check if name exists in roster
    in_roster = drv in predictor._roster
    # Check if name exists in driver_form
    in_form = drv in predictor._driver_form
    form = predictor._driver_form.get(drv, {})
    races_counted = form.get("races_counted", 0)
    is_rookie = races_counted < 4
    print(f"\nDriver: {drv}")
    print(f"  In roster: {in_roster}")
    print(f"  In form: {in_form}")
    print(f"  Races counted: {races_counted}")
    print(f"  Is rookie: {is_rookie}")

print("\nAll roster keys:")
print(list(predictor._roster.keys()))
print("\nAll driver_form keys:")
print(list(predictor._driver_form.keys()))
