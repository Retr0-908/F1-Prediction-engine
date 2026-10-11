import unittest
from engine.models.predictor import F1Predictor


class SprintPredictionDynamicModifierTests(unittest.TestCase):
    def setUp(self):
        self.p = F1Predictor()
        self.p._roster = {
            "Max Verstappen": "Red Bull",
            "Lando Norris": "McLaren",
            "Alexander Albon": "Williams",
            "Nico Hulkenberg": "Audi"
        }
        self.p._driver_standings = [
            {"name": "Max Verstappen", "position": 1, "points": 100},
            {"name": "Lando Norris", "position": 2, "points": 90},
            {"name": "Alexander Albon", "position": 14, "points": 10},
            {"name": "Nico Hulkenberg", "position": 17, "points": 4},
        ]
        self.p._ctor_standings = [
            {"constructor": "Red Bull", "position": 1},
            {"constructor": "McLaren", "position": 2},
            {"constructor": "Williams", "position": 8},
            {"constructor": "Audi", "position": 10},
        ]
        self.p._driver_form = {
            "Max Verstappen": {"avg_position": 2.0, "races_counted": 10},
            "Lando Norris": {"avg_position": 3.0, "races_counted": 10},
            "Alexander Albon": {"avg_position": 13.0, "races_counted": 10},
            "Nico Hulkenberg": {"avg_position": 16.0, "races_counted": 10},
        }
        self.p._circuit_config = {"key": "shanghai", "track_type": "permanent"}
        self.p._practice_pace = {}
        self.p._quali_sectors = {}
        self.p._bias_corrections = {}

    def test_sprint_order_applies_practice_pace(self):
        # Baseline with neutral pace
        self.p._practice_pace = {}
        res_baseline = {d["driver"]: d["predicted_pos"] for d in self.p.predict_sprint_order()}

        # Favor Norris by -0.8s
        self.p._practice_pace = {"Lando Norris": -0.8, "Max Verstappen": 0.0}
        res_modified = {d["driver"]: d["predicted_pos"] for d in self.p.predict_sprint_order()}

        self.assertLess(
            res_modified["Lando Norris"],
            res_baseline["Lando Norris"],
            "Negative practice delta should improve (lower) predicted position"
        )

    def test_sprint_order_applies_bias_corrections(self):
        self.p._bias_corrections = {
            "driver_biases": {
                "Lando Norris": {"permanent": 1.2}
            }
        }
        res = {d["driver"]: d for d in self.p.predict_sprint_order()}
        self.assertIn("Lando Norris", res)
        # Should record explanation and bias note
        self.assertIn("explanation", res["Lando Norris"])
        self.assertIn("Bias", res["Lando Norris"]["explanation"])

    def test_sprint_order_rank_contiguity(self):
        sprint = self.p.predict_sprint_order()
        ranks = [d["predicted_sprint_rank"] for d in sprint]
        self.assertEqual(ranks, list(range(1, len(self.p._roster) + 1)))

    def test_sprint_order_explanation_and_fields(self):
        self.p._practice_pace = {"Max Verstappen": -0.2}
        sprint = self.p.predict_sprint_order()
        for d in sprint:
            self.assertIn("explanation", d)
            self.assertIn("fp_pace_delta", d)
            self.assertIn("predicted_sprint_rank", d)
            self.assertIn("predicted_pos", d)
            self.assertIn("sprint_grid", d)
            self.assertIn("confidence_pct", d)
            self.assertIn("std_dev", d)
            self.assertIn("dnf_prob_pct", d)


if __name__ == "__main__":
    unittest.main()
