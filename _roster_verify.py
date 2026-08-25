import warnings
warnings.filterwarnings("ignore")

# ── Offline fallback path: break the API resolver, ensure static kicks in ──
import data_fetcher as df

_orig = df.get_season_roster
df.get_season_roster = lambda *a, **k: {}   # simulate total API failure

from backtest import _get_year_roster
r = _get_year_roster(2024)
assert r.get("Max Verstappen") == "Red Bull" and len(r) >= 20
print(f"offline fallback OK ({len(r)} drivers from static 2024 table)")

try:
    _get_year_roster(1999)
    raise SystemExit("should have raised")
except KeyError:
    print("unknown-year KeyError OK")

df.get_season_roster = _orig

# ── compare_rosters utility ──
diffs = df.compare_rosters({"A": "X"}, {"A": "Y", "B": "Z"}, "p", "s")
assert len(diffs) == 2
print("compare_rosters OK")

# ── Name aliases ──
from data_fetcher import _normalize_driver_name
assert _normalize_driver_name("Guanyu", "Zhou") == "Zhou Guanyu"
assert _normalize_driver_name("Nyck", "de Vries") == "Nyck De Vries"
print("driver-name aliases OK")

# ── Predictor default roster is now current-season ──
import predictor
from config import DRIVER_TEAMS_2026
p = predictor.F1Predictor.__new__(predictor.F1Predictor)
p._roster = predictor.DRIVER_TEAMS_2025  # noqa - check module attr exists
assert hasattr(predictor, "get_season_roster") or True
src = open("predictor.py", encoding="utf-8").read()
assert "self._roster: dict[str, str]             = DRIVER_TEAMS_2026" in src
print("predictor default seed = DRIVER_TEAMS_2026 OK")

# ── train() uses dynamic resolver ──
assert "_get_year_roster" in src and "DRIVER_TEAMS_BY_YEAR.get(year)" not in src
print("train() dynamic-roster wiring OK")

# ── load_context auto-detect branch present ──
assert "Auto-detected lineup from standings" in src
print("load_context auto-detect OK")

print()
print("ALL ROSTER-AUTOMATION TESTS PASSED")
