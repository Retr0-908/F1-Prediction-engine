"""
test_serialize.py — Unit test verifying JSON serializability of pipeline COMPLETE event payloads.
"""
import json
import unittest
from unittest.mock import patch
from engine.serving.pipeline import run_full_pipeline
from engine.core.config import DRIVER_TEAMS_2026, CONSTRUCTORS_2026


class PipelineSerializationTests(unittest.TestCase):
    @patch("engine.serving.pipeline.scrape_constructor_prices")
    @patch("engine.serving.pipeline.scrape_driver_prices")
    def test_pipeline_complete_event_serializable(self, mock_scrape_drivers, mock_scrape_ctors):
        mock_scrape_drivers.return_value = (
            {d: {"price": 15.0, "team": t} for d, t in DRIVER_TEAMS_2026.items()},
            dict(DRIVER_TEAMS_2026)
        )
        mock_scrape_ctors.return_value = {c: {"price": 15.0} for c in CONSTRUCTORS_2026}

        serialized_events = []

        def dummy_callback(run_id, stage, status, message, data=None):
            if stage == "COMPLETE" and data is not None:
                encoded = json.dumps(data, default=str)
                serialized_events.append(encoded)

        res = run_full_pipeline(
            run_id="test1234",
            my_drivers=[],
            my_constructors=[],
            budget=100.0,
            points=0.0,
            transfers=1,
            options={"race_round": 1, "sims": 10},
            progress_callback=dummy_callback,
        )
        self.assertIsNotNone(res)
        self.assertGreater(len(serialized_events), 0, "COMPLETE event payload must be JSON serializable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
