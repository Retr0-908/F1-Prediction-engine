"""
test_rookie.py — Unit test for rookie driver status resolution.
"""
import unittest
from engine.core.data_fetcher import compute_driver_form, compute_multiseason_driver_form
from engine.core.config import CURRENT_SEASON


class RookieStatusTests(unittest.TestCase):
    def test_rookie_status_resolution(self):
        current_form = compute_driver_form(CURRENT_SEASON, num_races=5)
        multi_form = compute_multiseason_driver_form(CURRENT_SEASON)

        driver_form = {**multi_form, **current_form}
        for drv, mdata in multi_form.items():
            if drv in driver_form:
                career_races = mdata.get("races_counted", 0)
                current_races = current_form.get(drv, {}).get("races_counted", 0)
                driver_form[drv]["races_counted"] = max(career_races, current_races)

        # Veterans like Verstappen, Alonso, Hamilton must have counted career races >= 4
        for veteran in ["Fernando Alonso", "Lewis Hamilton", "Max Verstappen"]:
            d = driver_form.get(veteran, {})
            rc = d.get("races_counted", 0)
            self.assertGreaterEqual(rc, 4, f"{veteran} must not be classified as a rookie")


if __name__ == "__main__":
    unittest.main(verbosity=2)
