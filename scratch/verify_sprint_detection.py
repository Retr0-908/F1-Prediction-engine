from engine.core.data_fetcher import get_next_race, get_season_schedule, is_sprint_weekend
from engine.core.config import CURRENT_SEASON

schedule = get_season_schedule(CURRENT_SEASON)
sprint_races = [(r['round'], r['name']) for r in schedule if is_sprint_weekend(r)]
print("Detected sprint weekends:", sprint_races)
# Expected: China(2), Miami(4), Canada(5), Britain(9), Netherlands(12), Singapore(16)
# Monaco must NOT appear in this list
