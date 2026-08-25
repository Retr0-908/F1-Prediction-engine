import json
from engine.core.data_fetcher import compute_practice_pace
import fastf1

session = fastf1.get_session(2026, "Miami Grand Prix", "FP1")
session.load(telemetry=False, weather=False, messages=False)

abbr_to_name = {}
for num in session.drivers:
    try:
        drv_data = session.get_driver(num)
        drv_abbr = drv_data.get('Abbreviation')
        full = f"{drv_data.get('FirstName', '')} {drv_data.get('LastName', '')}".strip()
        if drv_abbr and full:
            abbr_to_name[drv_abbr] = full
    except Exception:
        pass

print(json.dumps(abbr_to_name, indent=2))
