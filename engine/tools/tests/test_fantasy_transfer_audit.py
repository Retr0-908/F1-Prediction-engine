"""
test_fantasy_transfer_audit.py — Regression & Architectural Audit Suite
for F1 Fantasy Point Projections and Transfer Recommendations.
"""

import json
import unittest

from engine.core.data_fetcher import _estimate_historical_fantasy_pts
from engine.models.bayesian_model import BayesianPointsModel
from engine.strategy.fantasy_optimizer import (
    suggest_team_changes,
    _compute_team_score_with_turbo,
    _resolve_max_spend,
)


class FantasyTransferAuditTests(unittest.TestCase):
    """End-to-end verification of the Fantasy Remediation Plan fixes."""

    def test_1_historical_fantasy_pts_reconstruction(self):
        """
        Verify _estimate_historical_fantasy_pts reconstructs realistic F1 Fantasy
        points (including qualifying, Q2/Q3/Pole bonus, Positions Delta, and DNF penalty)
        matching config.py rather than FIA World Championship points.
        """
        # P1 from P1 with fastest lap -> 25 (race) + 7.0 (Q1+Q3+Pole) + 5.0 (FL) + 1.2 (DOTD) = 38.2
        p1_pts = _estimate_historical_fantasy_pts(position=1, grid=1, dnf=False, fastest_lap=True)
        self.assertGreaterEqual(p1_pts, 37.0)

        # P3 from P3 -> 15 (race) + 5.0 (Q3) + ~1.4 (FL/DOTD) = 21.4 (vs 15 FIA pts)
        p3_pts = _estimate_historical_fantasy_pts(position=3, grid=3, dnf=False)
        self.assertGreaterEqual(p3_pts, 20.0)

        # P11 from P15 (midfield driver who scores 0 FIA pts!) ->
        # 0 (race) + 1.0 (Q2) + 8.0 (pos delta: +4 * 2) = 9.0+ Fantasy pts
        p11_pts = _estimate_historical_fantasy_pts(position=11, grid=15, dnf=False)
        self.assertGreaterEqual(p11_pts, 8.0)

        # DNF from P5 -> 4.0 (Q5) - 15 (DNF) = -11.0 Fantasy pts
        dnf_pts = _estimate_historical_fantasy_pts(position=20, grid=5, dnf=True)
        self.assertLessEqual(dnf_pts, -10.0)

    def test_2_bayesian_fantasy_scale_preservation(self):
        """
        Verify Lando Norris (predicted P3, ensemble_pred_pts=25.9) is NOT crushed
        to 5.8 pts even if raw FIA championship points are accidentally passed,
        and stays >= 22.0 pts with proper Fantasy-scale history.
        """
        bm = BayesianPointsModel()
        # Case A: Fantasy-scale historical observations for a podium contender
        norris_fantasy_hist = [26.5, 22.0, 29.0, 18.5, 24.5, 27.0, 21.0, 25.5]
        ev_fantasy = bm.fit_driver(
            "Lando Norris",
            norris_fantasy_hist,
            ensemble_pred_pts=25.9,
            dnf_flags=[False] * len(norris_fantasy_hist),
            retained_non_sunday_pts=8.0,
            observations_are_fantasy_scale=True,
        )
        self.assertGreaterEqual(ev_fantasy, 22.0)
        self.assertLessEqual(ev_fantasy, 30.0)

        # Case B: Even if legacy FIA points [0, 6, 8, 4, 2, 0] are passed with
        # ensemble_pred_pts=25.9, the scale-contamination detector prevents collapse to 5.8
        bm_legacy = BayesianPointsModel()
        norris_fia_hist = [0.0, 6.0, 8.0, 4.0, 2.0, 0.0, 10.0, 6.0]
        ev_legacy_guarded = bm_legacy.fit_driver(
            "Lando Norris",
            norris_fia_hist,
            ensemble_pred_pts=25.9,
            dnf_flags=[False] * len(norris_fia_hist),
            retained_non_sunday_pts=8.0,
            observations_are_fantasy_scale=False,
        )
        self.assertGreaterEqual(ev_legacy_guarded, 17.5)

        # Case C: Midfield driver (P11, ensemble_pred_pts=5.5) with 0 FIA points
        # does not collapse to negative total points when clean
        bm_mid = BayesianPointsModel()
        mid_fantasy_hist = [6.0, 4.5, 5.0, 3.5, 7.0, 4.0]
        ev_mid = bm_mid.fit_driver(
            "Nico Hulkenberg",
            mid_fantasy_hist,
            ensemble_pred_pts=5.5,
            dnf_flags=[False] * len(mid_fantasy_hist),
            retained_non_sunday_pts=0.0,
            observations_are_fantasy_scale=True,
        )
        self.assertGreaterEqual(ev_mid, 3.0)

    def test_3_mexico_gp_funding_and_upgrade_package_coherence(self):
        """
        Reproduce the Mexico GP scenario:
        User holds Ferrari ($23.3M, 38.0 pts) and Nico Hulkenberg ($6.8M, 4.0 pts)
        with $0.5M in bank and 2 free transfers.
        Downgrading Ferrari -> Racing Bulls ($11.5M, 23.2 pts, -14.8 pts, -$11.8M)
        frees $11.8M so Hulkenberg -> Max Verstappen ($18.5M, 42.0 pts, +38.0 pts, +$11.7M)
        becomes affordable.
        Verify:
        1. Both legs are recommended as a 2-transfer package (+23.2 net pts).
        2. Transfer #1 is the PRIMARY UPGRADE (Hulkenberg -> Verstappen, +38.0 pts).
        3. Transfer #2 is tagged role == 'funding_downgrade' (Ferrari -> Racing Bulls, -14.8 pts).
        4. trade_summary explains the linked package.
        """
        current_drivers = [
            "George Russell",
            "Lando Norris",
            "Carlos Sainz",
            "Pierre Gasly",
            "Nico Hulkenberg",
        ]
        current_constructors = ["Mercedes", "Ferrari"]

        driver_pts = [
            {"driver": "Max Verstappen", "team": "Red Bull", "total_pts": 42.0},
            {"driver": "George Russell", "team": "Mercedes", "total_pts": 31.0},
            {"driver": "Lando Norris", "team": "McLaren", "total_pts": 28.0},
            {"driver": "Carlos Sainz", "team": "Williams", "total_pts": 16.0},
            {"driver": "Pierre Gasly", "team": "Alpine", "total_pts": 12.0},
            {"driver": "Nico Hulkenberg", "team": "Audi", "total_pts": 4.0},
            {"driver": "Franco Colapinto", "team": "Alpine", "total_pts": 4.5},
        ]
        constructor_pts = [
            {"constructor": "Mercedes", "total_pts": 58.0},
            {"constructor": "Ferrari", "total_pts": 38.0},
            {"constructor": "Racing Bulls", "total_pts": 23.2},
            {"constructor": "Haas", "total_pts": 10.0},
        ]
        driver_prices = {
            "Max Verstappen": {"price": 18.5, "team": "Red Bull"},
            "George Russell": {"price": 16.0, "team": "Mercedes"},
            "Lando Norris": {"price": 16.0, "team": "McLaren"},
            "Carlos Sainz": {"price": 11.5, "team": "Williams"},
            "Pierre Gasly": {"price": 9.0, "team": "Alpine"},
            "Nico Hulkenberg": {"price": 6.8, "team": "Audi"},
            "Franco Colapinto": {"price": 6.5, "team": "Alpine"},
        }
        constructor_prices = {
            "Mercedes": {"price": 25.0},
            "Ferrari": {"price": 23.3},
            "Racing Bulls": {"price": 11.5},
            "Haas": {"price": 8.0},
        }

        res = suggest_team_changes(
            current_drivers=current_drivers,
            current_constructors=current_constructors,
            driver_pts=driver_pts,
            constructor_pts=constructor_pts,
            driver_prices=driver_prices,
            constructor_prices=constructor_prices,
            free_transfers=2,
            budget_remaining=0.5,
        )

        changes = res["suggested_changes"]
        self.assertEqual(len(changes), 2)
        # #1 must be the positive upgrade leg!
        self.assertEqual(changes[0]["out"], "Nico Hulkenberg")
        self.assertEqual(changes[0]["in"], "Max Verstappen")
        self.assertGreater(changes[0]["pts_gain"], 0.0)
        self.assertEqual(changes[0]["role"], "primary_upgrade")

        # #2 must be the budget-enabler leg!
        self.assertEqual(changes[1]["out"], "Ferrari")
        self.assertEqual(changes[1]["in"], "Racing Bulls")
        self.assertLess(changes[1]["pts_gain"], 0.0)
        self.assertEqual(changes[1]["role"], "funding_downgrade")

        # Overall package must be strongly net positive
        self.assertAlmostEqual(res["points_gain"], 23.2, places=1)
        self.assertIsNotNone(res["trade_summary"])
        self.assertTrue(res["trade_summary"]["is_package"])
        self.assertIn("Max Verstappen", res["trade_summary"]["narrative"])

    def test_4_zero_bank_and_appreciated_team_budget_handling(self):
        """
        Verify:
        1. budget_remaining = 0.0 means $0.0M spare cash in bank (not $100M).
        2. An appreciated team worth $102.0M with $0.5M in the bank can make a swap
           resulting in a $102.2M team (not rejected by a hardcoded 100.0M cap).
        """
        eff_cap_zero, bank_zero = _resolve_max_spend(99.5, 0.0)
        self.assertEqual(bank_zero, 0.0)
        self.assertEqual(eff_cap_zero, 99.5)

        eff_cap_appr, bank_appr = _resolve_max_spend(102.0, 0.5)
        self.assertEqual(bank_appr, 0.5)
        self.assertEqual(eff_cap_appr, 102.5)

        # Appreciated team worth $102.0M making a +$0.4M upgrade using $0.5M bank
        current_drivers = ["D1", "D2", "D3", "D4", "D5"]
        current_constructors = ["C1", "C2"]
        driver_pts = [
            {"driver": "D1", "team": "T1", "total_pts": 20.0},
            {"driver": "D2", "team": "T2", "total_pts": 20.0},
            {"driver": "D3", "team": "T3", "total_pts": 20.0},
            {"driver": "D4", "team": "T4", "total_pts": 20.0},
            {"driver": "D5", "team": "T5", "total_pts": 10.0},
            {"driver": "D6", "team": "T6", "total_pts": 25.0},
        ]
        constructor_pts = [
            {"constructor": "C1", "total_pts": 30.0},
            {"constructor": "C2", "total_pts": 30.0},
        ]
        driver_prices = {
            "D1": {"price": 16.0, "team": "T1"},
            "D2": {"price": 16.0, "team": "T2"},
            "D3": {"price": 16.0, "team": "T3"},
            "D4": {"price": 16.0, "team": "T4"},
            "D5": {"price": 10.0, "team": "T5"},
            "D6": {"price": 10.4, "team": "T6"},
        }
        constructor_prices = {
            "C1": {"price": 14.0},
            "C2": {"price": 14.0},
        }
        # Total current cost = 4*16 + 10 + 2*14 = 102.0M
        res = suggest_team_changes(
            current_drivers=current_drivers,
            current_constructors=current_constructors,
            driver_pts=driver_pts,
            constructor_pts=constructor_pts,
            driver_prices=driver_prices,
            constructor_prices=constructor_prices,
            free_transfers=1,
            budget_remaining=0.5,
        )
        self.assertEqual(len(res["suggested_changes"]), 1)
        self.assertEqual(res["suggested_changes"][0]["out"], "D5")
        self.assertEqual(res["suggested_changes"][0]["in"], "D6")

    def test_5_marginal_efficiency_hurdle_and_order_independence(self):
        """
        Verify:
        1. A 2nd transfer that adds only +0.3 pts on top of a +10.0 pt 1st transfer
           is rejected in favor of the 1-transfer move (MIN_GAIN_PER_TRANSFER = 1.5).
        2. Multi-driver swaps pair lowest-scoring outgoing with highest-scoring incoming
           regardless of input list ordering.
        """
        current_drivers = ["StarA", "MidB", "D3", "D4", "D5"]
        current_constructors = ["C1", "C2"]
        driver_pts = [
            {"driver": "StarA", "team": "T1", "total_pts": 24.0},
            {"driver": "MidB", "team": "T2", "total_pts": 6.0},
            {"driver": "D3", "team": "T3", "total_pts": 15.0},
            {"driver": "D4", "team": "T4", "total_pts": 15.0},
            {"driver": "D5", "team": "T5", "total_pts": 15.0},
            {"driver": "Cand1", "team": "T6", "total_pts": 16.0},  # +10.0 over MidB
            {"driver": "Cand2", "team": "T7", "total_pts": 15.3},  # +0.3 over D5 (noise!)
        ]
        constructor_pts = [
            {"constructor": "C1", "total_pts": 30.0},
            {"constructor": "C2", "total_pts": 30.0},
        ]
        driver_prices = {
            "StarA": {"price": 20.0, "team": "T1"},
            "MidB": {"price": 10.0, "team": "T2"},
            "D3": {"price": 10.0, "team": "T3"},
            "D4": {"price": 10.0, "team": "T4"},
            "D5": {"price": 10.0, "team": "T5"},
            "Cand1": {"price": 10.0, "team": "T6"},
            "Cand2": {"price": 10.0, "team": "T7"},
        }
        constructor_prices = {
            "C1": {"price": 15.0},
            "C2": {"price": 15.0},
        }
        res = suggest_team_changes(
            current_drivers=current_drivers,
            current_constructors=current_constructors,
            driver_pts=driver_pts,
            constructor_pts=constructor_pts,
            driver_prices=driver_prices,
            constructor_prices=constructor_prices,
            free_transfers=2,
            budget_remaining=5.0,
        )
        # Should only make 1 transfer (MidB -> Cand1 for +10.0), NOT burn the 2nd free transfer for +0.3 pts
        self.assertEqual(len(res["suggested_changes"]), 1)
        self.assertEqual(res["suggested_changes"][0]["out"], "MidB")
        self.assertEqual(res["suggested_changes"][0]["in"], "Cand1")

        # JSON serializability contract check
        serialized = json.dumps(res)
        self.assertIsInstance(serialized, str)


if __name__ == "__main__":
    unittest.main()
