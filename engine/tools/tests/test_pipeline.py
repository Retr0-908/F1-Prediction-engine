"""
test_pipeline.py — Unit test wrapper for full pipeline execution.
"""
import unittest
from unittest.mock import patch
from engine.serving.pipeline import run_full_pipeline
from engine.core.config import DRIVER_TEAMS_2026, CONSTRUCTORS_2026


class PipelineIntegrationTests(unittest.TestCase):
    @patch("engine.serving.pipeline.scrape_constructor_prices")
    @patch("engine.serving.pipeline.scrape_driver_prices")
    def test_pipeline_smoke_execution(self, mock_scrape_drivers, mock_scrape_ctors):
        mock_scrape_drivers.return_value = (
            {d: {"price": 15.0, "team": t} for d, t in DRIVER_TEAMS_2026.items()},
            dict(DRIVER_TEAMS_2026)
        )
        mock_scrape_ctors.return_value = {c: {"price": 15.0} for c in CONSTRUCTORS_2026}

        progress_events = []

        def progress_cb(run_id, stage, status, msg, data=None):
            progress_events.append((stage, status))

        result = run_full_pipeline(
            run_id="test_run",
            my_drivers=["Max Verstappen", "Lando Norris"],
            my_constructors=["Red Bull", "McLaren"],
            budget=100.0,
            points=0.0,
            transfers=3,
            options={"race_round": 1, "sims": 10},
            progress_callback=progress_cb,
        )
        self.assertIsNotNone(result, "Pipeline should return result dictionary")
        self.assertIn("predictions", result)
        self.assertIn("race_order", result["predictions"])
        self.assertGreater(len(result["predictions"]["race_order"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
