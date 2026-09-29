"""
test_backtest_benchmarks.py — Unit tests for backtest accuracy and rank correlation metrics.
"""
import unittest
from engine.analysis.backtest import compute_round_metrics


class BacktestMetricsTests(unittest.TestCase):
    def test_metrics_computation(self):
        metrics = compute_round_metrics([1, 2, 3, 4, 5], [1, 3, 2, 4, 5])
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("spearman", metrics)
        self.assertIn("top3_acc", metrics)
        self.assertLess(metrics["mae"], 1.5)
        self.assertGreater(metrics["spearman"], 0.85)
        self.assertGreaterEqual(metrics["top3_acc"], 0.66)

    def test_perfect_prediction_metrics(self):
        metrics = compute_round_metrics([1, 2, 3, 4], [1, 2, 3, 4])
        self.assertEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["spearman"], 1.0, places=3)
        self.assertEqual(metrics["top3_acc"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
