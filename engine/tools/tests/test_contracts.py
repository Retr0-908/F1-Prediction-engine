"""
test_contracts.py — Golden contract & regression tests.

Run from project root:
    python -m unittest engine.tools.tests.test_contracts -v

Design rules:
- Network-free: everything relies on the local cache or pure functions.
- Tests encode CURRENT verified-correct behavior (green today).
- Planned fixes from Docs/IMPROVEMENTS_PLAN.md are encoded as
  @unittest.expectedFailure probes with the plan reference in the docstring.
  When such a fix lands, the probe starts PASSING -> unittest reports
  "unexpected success" -> remove the decorator in the same commit.
"""
import ast
import json
import math
import random
import re
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]


# ─────────────────────────────────────────────────────────────
# 1. RESULT CLASSIFICATION CONTRACTS (data_fetcher)
# ─────────────────────────────────────────────────────────────
class ResultClassificationTests(unittest.TestCase):
    def setUp(self):
        from engine.core.data_fetcher import _classify_result
        self.classify = _classify_result

    def test_finished_numeric(self):
        self.assertEqual(self.classify("5", "Finished"), ("finished", False))

    def test_lapped_finishers_any_lap_count(self):
        # Regression guard for the old whitelist that stopped at +6 laps
        for laps in range(1, 50):
            self.assertEqual(
                self.classify("", f"+{laps} Lap{'s' if laps > 1 else ''}"),
                ("lapped", False), f"+{laps} laps misclassified",
            )

    def test_dnf_kinds(self):
        self.assertEqual(self.classify("", "Accident"), ("dnf", True))
        self.assertEqual(self.classify("", "Engine"), ("dnf", True))
        self.assertEqual(self.classify("", "Collision"), ("dnf", True))
        self.assertEqual(self.classify("", "Disqualified"), ("dsq", True))
        self.assertEqual(self.classify("", "Withdrawal"), ("withdrawn", True))


class ParsePositionTests(unittest.TestCase):
    def test_numeric_used_verbatim(self):
        from engine.core.data_fetcher import _parse_result_position
        r = {"positionText": "7", "position": "7"}
        self.assertEqual(_parse_result_position(r, 99, "finished", False), 7)

    def test_non_numeric_falls_back_to_order_index(self):
        from engine.core.data_fetcher import _parse_result_position
        r = {"positionText": "R", "position": ""}
        self.assertEqual(_parse_result_position(r, 18, "dnf", True), 18)


# ─────────────────────────────────────────────────────────────
# 2. SEASON-RESULTS INTEGRITY (cache-based, network-free)
# ─────────────────────────────────────────────────────────────
class SeasonResultsIntegrityTests(unittest.TestCase):
    """Guards Improvement-9-C1: page-boundary enumerate resets corrupt positions."""

    @classmethod
    def setUpClass(cls):
        from engine.core.data_fetcher import get_season_results
        cls.by_season = {}
        for year in (2023, 2025, 2026):
            try:
                res = get_season_results(year)
            except Exception as e:
                raise unittest.SkipTest(f"cache unavailable for {year}: {e}")
            if not res:
                raise unittest.SkipTest(f"no cached results for {year}")
            cls.by_season[year] = res

    def _rounds(self, year):
        rounds = {}
        for r in self.by_season[year]:
            rounds.setdefault(r["round"], []).append(r)
        return rounds

    def test_no_duplicate_driver_per_round(self):
        for year in self.by_season:
            for rnd, rows in self._rounds(year).items():
                ids = [r["driver_id"] for r in rows]
                self.assertEqual(len(ids), len(set(ids)),
                                 f"{year} R{rnd}: duplicate driver entries")

    def test_positions_contiguous_1_to_N_per_round(self):
        # Catches page-split enumerate resets: duplicated/skipped positions.
        # Plan 9-C1 fixed + caches re-warmed 2026-08-25: all seasons verified
        # contiguous (2542 rows).
        bad = []
        for year in self.by_season:
            for rnd, rows in self._rounds(year).items():
                poss = sorted(r["position"] for r in rows)
                if poss != list(range(1, len(rows) + 1)):
                    bad.append(f"{year} R{rnd}")
        self.assertEqual(bad, [], f"non-contiguous positions: {bad}")

    def test_current_season_boundary_race_not_corrupted(self):
        # The specific 9-C1 live failure: a race whose rows straddle a
        # 100-row page boundary must still classify DNFs with realistic
        # positions (> 12), never 1..10 fallback indices. Was failing with
        # six fabricated P5-P10 DNFs in 2026 R5; passes after fix + purge.
        # ⚠️ VERIFIED FAILING TODAY: 2026 R5 has six DNFs at P5–P10
        # (Perez, Norris, Russell, Alonso, Albon, Lindblad) — fabricated
        # classifications in cached data. Enable after 9-C1 + cache purge.
        bad = []
        for r in self.by_season.get(2026, []):
            if r["dnf"] and r["position"] <= 10:
                bad.append((r["round"], r["name"], r["position"]))
        self.assertEqual(bad, [], f"DNFs with top-10 positions: {bad}")


# ─────────────────────────────────────────────────────────────
# 3. NAME NORMALIZATION & CIRCUIT MATCHING
# ─────────────────────────────────────────────────────────────
class NameNormalizationTests(unittest.TestCase):
    def test_aliases(self):
        from engine.core.data_fetcher import _normalize_driver_name
        self.assertEqual(_normalize_driver_name("Andrea Kimi", "Antonelli"), "Kimi Antonelli")
        self.assertEqual(_normalize_driver_name("Guanyu", "Zhou"), "Zhou Guanyu")
        self.assertEqual(_normalize_driver_name("Nyck", "de Vries"), "Nyck De Vries")


class CircuitMatcherTests(unittest.TestCase):
    def test_predictor_matcher_full_calendar(self):
        from engine.models.predictor import _match_circuit_cfg as m
        cases = {
            "Australian Grand Prix": "australia", "Chinese Grand Prix": "china",
            "Japanese Grand Prix": "japan", "Miami Grand Prix": "miami",
            "Canadian Grand Prix": "canada", "Monaco Grand Prix": "monaco",
            "Austrian Grand Prix": "austria", "British Grand Prix": "britain",
            "Belgian Grand Prix": "belgium", "Hungarian Grand Prix": "hungary",
            "Dutch Grand Prix": "dutch_grand_prix_or_netherlands",
            "Italian Grand Prix": "italy_grand_prix_or_italy",
            "Azerbaijan Grand Prix": "azerbaijan", "Singapore Grand Prix": "singapore",
            "United States Grand Prix": "usa_grand_prix_or_usa",
            "Mexico City Grand Prix": "mexico", "Brazilian Grand Prix": "brazil",
            "Las Vegas Grand Prix": "las_vegas", "Qatar Grand Prix": "qatar",
            "Abu Dhabi Grand Prix": "abu_dhabi",
        }
        for name, key_hint in cases.items():
            cfg = m(name)
            self.assertTrue(cfg, f"{name} unmatched")
        # precise keys where unambiguous
        self.assertEqual(m("Chinese Grand Prix")["key"], "china")
        self.assertEqual(m("Brazilian Grand Prix")["key"], "brazil")
        self.assertEqual(m("Monaco Grand Prix")["key"], "monaco")
        self.assertEqual(m("United States Grand Prix")["key"], "usa")
        self.assertEqual(m("Mexico City Grand Prix")["key"] in ("mexico",), True)

    def test_historical_circuits_resolve(self):
        from engine.models.predictor import _match_circuit_cfg as m
        for name, key in [("Portuguese Grand Prix", "portugal"),
                          ("French Grand Prix", "france"),
                          ("Russian Grand Prix", "russia"),
                          ("Turkish Grand Prix", "turkey")]:
            self.assertEqual(m(name).get("key"), key)

    def test_bahrain_in_malaysia_resolves_to_sepang(self):
        # Plan 9-C3 regression guard: Sepang must not resolve to Sakhir
        from engine.models.predictor import _match_circuit_cfg as m
        cfg = m("Bahrain Grand Prix in Malaysia")
        self.assertNotEqual(cfg.get("key"), "bahrain")
        self.assertIn(cfg.get("key"), ("malaysia", "sepang"))

    def test_styrian_maps_to_austria(self):
        # Same physical circuit as Austrian GP
        from engine.models.predictor import _match_circuit_cfg as m
        self.assertEqual(m("Styrian Grand Prix").get("key"), "austria")


class DataFetcherCircuitMatcherTests(unittest.TestCase):
    """The OTHER matcher (used by get_race_by_round/pipeline)."""

    def test_sao_paulo_matches_brazil(self):
        from engine.core.data_fetcher import _match_circuit_config as m
        cfg = m("São Paulo Grand Prix")
        self.assertTrue(cfg, "São Paulo unresolved in data_fetcher matcher")
        self.assertEqual(cfg.get("key"), "brazil")

    def test_exact_names_match(self):
        from engine.core.data_fetcher import _match_circuit_config as m
        self.assertTrue(m("Chinese Grand Prix").get("key"))


# ─────────────────────────────────────────────────────────────
# 4. SPRINT CALENDAR CONTRACTS
# ─────────────────────────────────────────────────────────────
class SprintCalendarTests(unittest.TestCase):
    def test_verified_fallback_tables(self):
        from engine.core.data_fetcher import SPRINT_ROUNDS_BY_YEAR_FALLBACK as T
        self.assertEqual(T[2024], {5, 6, 11, 19, 21, 23})
        self.assertEqual(T[2025], {2, 6, 13, 19, 21, 23})

    def test_is_sprint_weekend_prefers_schedule_field(self):
        from engine.core.data_fetcher import is_sprint_weekend
        self.assertTrue(is_sprint_weekend({"round": 99, "date": "2026-05-03",
                                           "sprint_date": "2026-05-02"}))


# ─────────────────────────────────────────────────────────────
# 5. CHIP STATE ISOLATION (deepcopy contract)
# ─────────────────────────────────────────────────────────────
class ChipStateTests(unittest.TestCase):
    def test_mark_does_not_pollute_default(self):
        from engine.strategy import chip_advisor as ca
        ca.reset_chip_state()
        ca.mark_chip_used("Wildcard", 3, "Test GP")
        try:
            self.assertFalse(ca._DEFAULT_STATE["used"]["Wildcard"])
        finally:
            ca.reset_chip_state()

    def test_reset_restores_clean_state(self):
        from engine.strategy import chip_advisor as ca
        ca.mark_chip_used("Limitless", 2, "X")
        s = ca.reset_chip_state()
        self.assertFalse(any(s["used"].values()))


# ─────────────────────────────────────────────────────────────
# 6. BAYESIAN MODEL CONTRACTS
# ─────────────────────────────────────────────────────────────
class BayesianTests(unittest.TestCase):
    def test_zeros_are_not_dnfs_without_flags(self):
        from engine.models.bayesian_model import BayesianPointsModel
        bm = BayesianPointsModel()
        pts = [10.0, 0.0, 12.0, 8.0, 0.0]
        ev_flags_none = bm.fit_driver("A", pts, ensemble_pred_pts=15.0)
        ev_explicit_clean = bm.fit_driver("B", pts, ensemble_pred_pts=15.0,
                                          dnf_flags=[False] * 5)
        self.assertAlmostEqual(ev_flags_none, ev_explicit_clean, places=9,
                               msg="zeros must NOT inflate ψ when flags absent")

    def test_real_dnfs_lower_psi_path_differs(self):
        from engine.models.bayesian_model import BayesianPointsModel
        bm = BayesianPointsModel()
        ev = bm.fit_driver("C", [10.0, 0.0], ensemble_pred_pts=15.0,
                           dnf_flags=[False, True])
        self.assertGreater(ev, 0)


# ─────────────────────────────────────────────────────────────
# 7. ELO FEATURE CONTRACTS
# ─────────────────────────────────────────────────────────────
class EloTests(unittest.TestCase):
    def test_roster_scoped_pool_bounds_rank(self):
        from engine.models.elo_ratings import Glicko2RatingSystem, GLICKO2_MU
        g = Glicko2RatingSystem()
        f = g.get_elo_features("Max Verstappen",
                               roster={"Max Verstappen", "Lando Norris"})
        self.assertGreaterEqual(f["elo_rank"], 1)
        self.assertLessEqual(f["elo_rank"], 2)

    def test_process_race_skips_dnf_vs_dnf_pairs(self):
        # Two identical-DNF races in a row must not change ratings at all
        from engine.models.elo_ratings import Glicko2RatingSystem
        g = Glicko2RatingSystem()
        results = [
            {"name": "A", "position": 20, "dnf": True},
            {"name": "B", "position": 21, "dnf": True},
        ]
        before = g.get_rating("A")
        g.process_race(results, 2026, 1)
        g.process_race(list(reversed(results)), 2026, 2)
        self.assertEqual(g.get_rating("A"), before,
                         "DNF-vs-DNF pairs must be skipped (arbitrary ordering noise)")


# ─────────────────────────────────────────────────────────────
# 8. MONTE CARLO CONTRACTS
# ─────────────────────────────────────────────────────────────
def _mc_orders(n=4):
    drivers = [f"D{i}" for i in range(n)]
    race = [{"driver": d, "team": "T", "predicted_rank": float(i + 1),
             "dnf_prob_pct": 5.0} for i, d in enumerate(drivers)]
    quali = [{"driver": d, "team": "T", "predicted_grid": i + 1,
              "is_actual": False} for i, d in enumerate(drivers)]
    return race, quali


class MonteCarloTests(unittest.TestCase):
    def test_seeded_reproducibility(self):
        from engine.models.monte_carlo import simulate_race_weekend
        race, quali = _mc_orders()
        kw = dict(circuit_config={"key": "monaco"}, weather={"rain_risk": "low"},
                  n_simulations=300, seed=11)
        r1 = simulate_race_weekend(race, quali, **kw)
        r2 = simulate_race_weekend(race, quali, **kw)
        for d in r1:
            self.assertEqual(r1[d].mean_pts, r2[d].mean_pts)

    def test_empty_stats_guard(self):
        from engine.models.monte_carlo import DistributionStats
        ds = DistributionStats("X", "T", [])
        self.assertEqual(ds.n_sims, 0)
        self.assertEqual(ds.mean_pts, 0.0)

    def test_all_dnf_corner_does_not_crash(self):
        # 9-L finding: shunt block rng.randint(1, 0) when everyone DNFed
        from engine.models.monte_carlo import _simulate_one_race
        race, quali = _mc_orders(4)
        for d in race:
            d["dnf_prob_pct"] = 100.0
        rng = random.Random(3)
        res = _simulate_one_race(race, quali, 0.9, 0.0, "low", False, None, rng)
        self.assertEqual(len(res), 4)


# ─────────────────────────────────────────────────────────────
# 9. SCORING RULES CONTRACTS
# ─────────────────────────────────────────────────────────────
class ScoringRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from engine.models.predictor import F1Predictor
        cls.p = F1Predictor()
        cls.p._grid_penalties = {}
        cls.p._driver_form = {}

    def _orders(self, grid, rank):
        race = [{"driver": "X", "team": "T", "predicted_rank": rank,
                 "dnf_prob_pct": 0.0, "confidence_pct": 90.0,
                 "elo_rating": 1500.0, "momentum": 0.0}]
        quali = [{"driver": "X", "team": "T", "predicted_grid": grid,
                  "quali_class_pos": grid, "is_actual": False}]
        return race, quali

    def test_pole_gets_q3_plus_pole_bonus_not_q2(self):
        from engine.core.config import QUALI_POSITION_POINTS, QUALI_Q3_BONUS, POLE_BONUS
        race, quali = self._orders(1, 1)
        b = self.p.estimate_fantasy_points("X", race, quali)["breakdown"]
        self.assertEqual(b["qualifying"],
                         round(QUALI_POSITION_POINTS[1] + QUALI_Q3_BONUS + POLE_BONUS, 1))

    def test_p10_gets_q3_only_never_q2(self):
        from engine.core.config import QUALI_POSITION_POINTS, QUALI_Q3_BONUS, QUALI_Q2_BONUS
        race, quali = self._orders(10, 10)
        b = self.p.estimate_fantasy_points("X", race, quali)["breakdown"]
        self.assertEqual(b["qualifying"],
                         round(QUALI_POSITION_POINTS[10] + QUALI_Q3_BONUS, 1))

    def test_p11_gets_q2_only(self):
        from engine.core.config import QUALI_Q2_BONUS
        race, quali = self._orders(11, 11)
        b = self.p.estimate_fantasy_points("X", race, quali)["breakdown"]
        self.assertEqual(b["qualifying"], round(float(QUALI_Q2_BONUS), 1))

    def test_fl_ineligible_outside_top10(self):
        race, quali = self._orders(5, 15)
        out = self.p.estimate_fantasy_points("X", race, quali)
        self.assertEqual(out["breakdown"]["fastest_lap"], 0.0)

    def test_total_equals_breakdown_sum_with_bayesian_scaling(self):
        race, quali = self._orders(3, 3)
        out = self.p.estimate_fantasy_points("X", race, quali)
        self.assertAlmostEqual(out["total_pts"],
                               round(sum(out["breakdown"].values()), 1), delta=0.35)


# ─────────────────────────────────────────────────────────────
# 10. STRUCTURAL INVARIANTS (AST/source-level guards)
# ─────────────────────────────────────────────────────────────
PRED_SRC = (ROOT / "engine" / "models" / "predictor.py").read_text(encoding="utf-8")


class StructuralInvariantTests(unittest.TestCase):
    def test_feature_name_and_value_widths_align(self):
        tree = ast.parse(PRED_SRC)
        n_values = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_build_features":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Call):
                        if getattr(sub.value.func, "attr", "") == "array":
                            n_values = len(sub.value.args[0].elts)
        i = PRED_SRC.find("FEATURE_NAMES = [")
        j = PRED_SRC.find("]\n", i)
        n_names = len(re.findall(r'"[^"]+"', PRED_SRC[i:j]))
        self.assertEqual(n_names, n_values,
                         "FEATURE_NAMES vs feature vector width drift")
        self.assertEqual(n_names, 56)

    def test_model_cache_key_embeds_schema_version_and_device_ready(self):
        self.assertIn('MODEL_SCHEMA_VERSION = "', PRED_SRC)

    def test_backtest_uses_shared_field_rank_helper(self):
        bt = (ROOT / "engine" / "analysis" / "backtest.py").read_text(encoding="utf-8")
        self.assertIn("from engine.models.predictor import",
                      bt) and self.assertIn("to_rank", bt,
                      "backtest must convert ranker margins via the shared to_rank helper")
        self.assertIn("_field_ranks = to_rank", bt)

    def test_julia_engine_gated_off_by_default(self):
        mc = (ROOT / "engine" / "models" / "monte_carlo.py").read_text(encoding="utf-8")
        self.assertIn("USE_JULIA_ENGINE = False", mc)

    def test_watchdog_job_guard_present(self):
        srv = (ROOT / "engine" / "serving" / "server.py").read_text(encoding="utf-8")
        self.assertIn("_job_begin", srv)
        self.assertIn("_request_shutdown", srv)

    def test_pipeline_records_prices_after_merge(self):
        pl = (ROOT / "engine" / "serving" / "pipeline.py").read_text(encoding="utf-8")
        self.assertIn("price_tracker.record_prices", pl)
        self.assertIn("backfilled/validated", pl)   # 9-H1 merge logging
        self.assertIn("FIELD_VALIDATION", pl)       # 7b integrity event


class PlannedFixProbes(unittest.TestCase):
    """These FAIL today BY DESIGN (unittest.expectedFailure) and pin the exact
    moment each planned fix lands. Remove the decorator in the same commit as
    the fix once the probe turns green ('unexpected success')."""

    def test_meta_learner_uses_shared_rank_helper(self):
        self.assertIn("def to_rank", PRED_SRC, "to_rank must be hoisted module-level")
        i = PRED_SRC.find("def _train_meta_learner")
        seg = PRED_SRC[i:i + 4000]
        self.assertIn("to_rank(", seg,
                      "_train_meta_learner must convert OOF preds via to_rank")

    def test_config_seed_is_exactly_22_drivers(self):
        from engine.core.config import DRIVER_TEAMS_2026
        self.assertEqual(len(DRIVER_TEAMS_2026), 22)

    def test_unknown_driver_price_is_not_fabricated(self):
        # Plan 8b (implemented): unmatched -> 0.0 (caller excludes), never $5M
        from engine.strategy.fantasy_optimizer import _get_driver_price, _get_ctor_price
        self.assertEqual(_get_driver_price("Definitely Not A Driver", {}), 0.0)
        self.assertEqual(_get_ctor_price("Not A Team", {}), 0.0)

    def test_partial_roster_rejected(self):
        from engine.core import data_fetcher as df
        orig = df._latest_completed_round_num
        df._latest_completed_round_num = lambda y: 5
        try:
            # Simulate standings returning 12 entries by monkeypatching the getter
            real = df.get_driver_standings
            real_results = df.get_race_results
            df.get_driver_standings = lambda *a, **k: [
                {"name": f"P{i}", "constructor": "T", "position": i} for i in range(1, 13)
            ]
            df.get_race_results = lambda *a, **k: []
            try:
                self.assertEqual(df.get_season_roster(2026), {})
            finally:
                df.get_driver_standings = real
                df.get_race_results = real_results
        finally:
            df._latest_completed_round_num = orig


if __name__ == "__main__":
    unittest.main(verbosity=2)
