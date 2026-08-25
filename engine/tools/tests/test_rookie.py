from engine.core.data_fetcher import compute_driver_form, compute_multiseason_driver_form
from engine.core.config import CURRENT_SEASON

current_form = compute_driver_form(CURRENT_SEASON, num_races=5)
multi_form   = compute_multiseason_driver_form(CURRENT_SEASON)

# New predictor logic
driver_form = {**multi_form, **current_form}
for drv, mdata in multi_form.items():
    if drv in driver_form:
        career_races  = mdata.get("races_counted", 0)
        current_races = current_form.get(drv, {}).get("races_counted", 0)
        driver_form[drv]["races_counted"] = max(career_races, current_races)

print("=== Result after fix ===")
for veteran in ["Fernando Alonso", "Lewis Hamilton", "Max Verstappen", "Andrea Kimi Antonelli", "Gabriel Bortoleto", "Isack Hadjar", "Nico Hulkenberg", "Sergio Perez"]:
    d = driver_form.get(veteran, {})
    rc = d.get("races_counted", 0)
    is_rookie = rc < 4
    print(f"  {veteran}: races_counted={rc}, is_rookie={is_rookie}")
