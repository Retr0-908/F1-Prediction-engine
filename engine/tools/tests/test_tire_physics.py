import unittest
import numpy as np

class TirePhysicsDegradationTests(unittest.TestCase):
    def test_fuel_correction_recovers_positive_degradation(self):
        from engine.core.data_fetcher import _calculate_fuel_corrected_degradation
        laps = np.arange(1, 11)
        raw_times = 90.0 - 0.02 * laps
        deg = _calculate_fuel_corrected_degradation(laps, raw_times, fuel_lambda=0.035)
        self.assertAlmostEqual(deg, 0.015, places=3, msg="Fuel-corrected degradation must recover true positive wear slope")

    def test_tire_model_pure_sklearn_fallback(self):
        from engine.models.tire_model import get_tire_model
        tm = get_tire_model(verbose=False)
        score = tm.predict_stint_degradation(compound="HARD", stint_length=30, track_temp=38.0)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)

if __name__ == "__main__":
    unittest.main()
