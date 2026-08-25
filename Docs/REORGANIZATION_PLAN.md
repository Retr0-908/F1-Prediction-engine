# Project Reorganization Plan

**Goal:** Root contains only entry points + config files; all Python lives in a proper package tree. Runtime data directories (`cache/`, `logs/`, `output/`) stay at project root so existing downloaded caches remain valid without migration.

---

## 0. Constraints discovered by audit (why this is delicate)

| # | Constraint | Evidence |
|---|---|---|
| C1 | **128 import statements** reference flat module names | 56 top-level + 72 inline/function-level across 25 files |
| C2 | **26 `Path(__file__).parent` anchors** assume each module sits in project root | data_fetcher, predictor, elo_ratings, chip_advisor, price_tracker, self_improvement, post_race_check, temporal_model, tire_model, weather, track_features_loader, config, monte_carlo |
| C3 | **~20 CWD-relative paths** (`Path("ui")`, `Path("cache")`, `Path("output")`) | server.py ×15, main_logic.py, sanity_check.py, main.py grid_cache.json |
| C4 | **String-referenced modules**: `uvicorn.Config("server:app")`, `.bat` files running `python server.py` / `python main.py`, take_screenshots spawning a subprocess | server.py:635, F1 Fantasy.bat:25, take_screenshots.py |
| C5 | **Pickle class-path hazard**: `cache/elo/*.pkl` stores instances of `elo_ratings.Glicko2RatingSystem` — moving the module invalidates old pickles | elo_ratings.py:440 |
| C6 | predictor ↔ backtest circular-ish dependency must stay lazy (predictor imports `_get_year_roster` inside `train()`) | predictor.py train loop |

**Strategy:** Phase A fixes ALL path anchoring *while everything is still flat* (zero-risk, independently verifiable). Only then do we move files and rewrite imports. Each phase ends in a verification gate.

---

## 1. Target Structure

```
F1-Prediction-engine-main/
├── F1 Fantasy.bat              ← unchanged name, updated command
├── BACKTEST.bat                ← updated command
├── VERIFY_CACHES.bat           ← updated command
├── RUN_CLI_Legacy.bat          ← updated command
├── main.py                     ← thin shim (~10 lines): calls engine.cli.app.main
├── requirements.txt
├── README.md                   ← rewritten with new layout
├── .env / .gitignore
│
├── engine/                     ← THE PACKAGE (all Python)
│   ├── __init__.py             (version string; convenience re-exports)
│   ├── paths.py                ← NEW: single source of truth for every directory
│   │
│   ├── core/                   ← data acquisition & shared config
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── data_fetcher.py
│   │   ├── weather.py
│   │   ├── fantasy_scraper.py
│   │   ├── warm_cache.py
│   │   └── track_features_loader.py
│   │
│   ├── models/                 ← ML subsystems
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   ├── elo_ratings.py
│   │   ├── temporal_model.py
│   │   ├── tire_model.py
│   │   ├── bayesian_model.py
│   │   ├── monte_carlo.py
│   │   └── julia/
│   │       └── monte_carlo_engine.jl
│   │
│   ├── strategy/               ← fantasy decision layer
│   │   ├── __init__.py
│   │   ├── fantasy_optimizer.py
│   │   ├── chip_advisor.py
│   │   ├── price_tracker.py
│   │   └── self_improvement.py
│   │
│   ├── analysis/               ← evaluation & reporting
│   │   ├── __init__.py
│   │   ├── backtest.py
│   │   ├── post_race_check.py
│   │   ├── analyse_results.py
│   │   └── dashboard.py
│   │
│   ├── serving/                ← web app
│   │   ├── __init__.py
│   │   ├── server.py
│   │   └── pipeline.py         (← renamed from main_logic.py)
│   │
│   ├── cli/                    ← terminal app
│   │   ├── __init__.py
│   │   └── app.py              (← body of main.py)
│   │
│   └── tools/                  ← validators & dev utilities
│       ├── __init__.py
│       ├── sanity_check.py
│       ├── validate_track_features.py
│       ├── take_screenshots.py
│       └── tests/              (← test_pipeline, test_rookie, test_serialize, test_fastf1)
│
├── ui/                         ← frontend assets (stays at root — served as static dir)
├── track_features/             ← circuit JSON data (stays at root)
├── cache/                      ← runtime data (STAYS — no migration needed)
├── logs/                       ← runtime data (STAYS)
├── output/                     ← runtime data (STAYS)
├── Docs/                       ← all *.md plans, method.txt, PDF
├── scratch/                    ← dev scripts (updated for new layout)
└── Screenshots/
```

**Deleted entirely:** `__pycache__/`, `ui/style.css.bak`, `ui/patch_layout.txt`, root-level `PROJECT_ANALYSIS.md`/`process.md`/`method.txt`/`BUGFIX_PLAN.md` (→ `Docs/`), `test_*.py` from root (→ `engine/tools/tests/`).

---

## 2. Phase A — Path Anchoring (do FIRST, while still flat)

### A1. Create `paths.py` (project root initially; moves into `engine/core/` later)

```python
"""Single source of truth for every filesystem location the engine uses."""
from pathlib import Path

# engine/core/paths.py → parents[1] = engine/ ; parents[2] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CACHE_DIR      = PROJECT_ROOT / "cache"
API_CACHE_DIR  = CACHE_DIR / "api"
ELO_CACHE_DIR  = CACHE_DIR / "elo"
MODEL_CACHE_DIR= CACHE_DIR / "models"
WEATHER_CACHE_DIR = CACHE_DIR / "weather"
FASTF1_CACHE_DIR  = CACHE_DIR / "fastf1"

LOGS_DIR            = PROJECT_ROOT / "logs"
OUTPUT_DIR          = PROJECT_ROOT / "output"
UI_DIR              = PROJECT_ROOT / "ui"
TRACK_FEATURES_DIR  = PROJECT_ROOT / "track_features"

MY_TEAM_PATH        = CACHE_DIR / "my_team.json"
CHIP_STATE_PATH     = CACHE_DIR / "chip_state.json"
PRICE_HISTORY_PATH  = CACHE_DIR / "price_history.json"
OWNERSHIP_HISTORY_PATH = CACHE_DIR / "ownership_history.json"
GRID_CACHE_PATH     = OUTPUT_DIR / "grid_cache.json"   # fixes CWD bug too

ACCURACY_LOG        = LOGS_DIR / "accuracy_log.json"
BIAS_CORRECTIONS    = LOGS_DIR / "bias_corrections.json"
WEIGHT_ADJUSTMENTS  = LOGS_DIR / "weight_adjustments.json"
LOG_FILE            = LOGS_DIR / "f1_predictor.log"
```

### A2. Replace all 26 `__file__` anchors + ~20 CWD paths with `paths` imports

Exact mapping (each becomes `from paths import X` — later `from engine.core.paths import X`):

| File | Old | New |
|---|---|---|
| data_fetcher:57 | parent/"cache"/"api" | `API_CACHE_DIR` (+ FF1_CACHE → `FASTF1_CACHE_DIR`) |
| weather:16 | parent/"cache"/"weather" | `WEATHER_CACHE_DIR` |
| elo_ratings:440 | parent/"cache"/"elo" | `ELO_CACHE_DIR` |
| temporal_model:65, tire_model:103, predictor:824 | parent/"cache"/"models" | `MODEL_CACHE_DIR` |
| track_features_loader:9 | parent/"track_features" | `TRACK_FEATURES_DIR` |
| chip_advisor:44 | parent/"cache"/chip_state | `CHIP_STATE_PATH` |
| price_tracker:21–22 | parent/cache history files | `PRICE_HISTORY_PATH` / `OWNERSHIP_HISTORY_PATH` |
| self_improvement:5 | parent/logs | `LOGS_DIR` (+ BIAS_CORRECTIONS const already exists there) |
| post_race_check:35–36 | parent/logs+output | `ACCURACY_LOG`, `OUTPUT_DIR` |
| predictor:347,355 | parent/logs bias/weight files | `BIAS_CORRECTIONS`, `WEIGHT_ADJUSTMENTS` |
| config:34,474 | dirname(__file__) fastf1 + my_team | `FASTF1_CACHE_DIR`, `MY_TEAM_PATH` (config may import paths or vice-versa — put FASTF1/MY_TEAM only in paths, config stops defining them; update the 2 consumers) |
| monte_carlo:35 | dirname/jl file | `Path(__file__).parent / "julia" / "monte_carlo_engine.jl"` (moves WITH the module) |
| server:63,67,204–209,215,275,295,303,321–325 | CWD "ui"/"cache"/"output" | `UI_DIR`, `API_CACHE_DIR`, `OUTPUT_DIR`, etc. |
| main_logic:280 | CWD "output" | `OUTPUT_DIR` |
| main.py:90,92,975 | parent/output+logs, CWD grid_cache | `OUTPUT_DIR`, `LOGS_DIR`, `GRID_CACHE_PATH` |
| sanity_check:142 | open("main.py") | path to `engine/cli/app.py` via package (`import engine.cli.app; Path(app.__file__)`) |

**Gate A:** compile all; run one offline pipeline step; confirm cache hit counts unchanged (no new downloads); `git diff --stat` shows no file moves yet.

---

## 3. Phase B — File Moves (git mv preserves history)

Order matters for reviewability; one commit per group.

| Group | Moves |
|---|---|
| B1 core | `git mv config.py data_fetcher.py weather.py fantasy_scraper.py warm_cache.py track_features_loader.py engine/core/` ; create `engine/__init__.py`, `engine/core/__init__.py`; move `paths.py` in too |
| B2 models | predictor, elo_ratings, temporal_model, tire_model, bayesian_model, monte_carlo → `engine/models/`; `monte_carlo_engine.jl` → `engine/models/julia/` |
| B3 strategy | fantasy_optimizer, chip_advisor, price_tracker, self_improvement → `engine/strategy/` |
| B4 analysis | backtest, post_race_check, analyse_results, dashboard → `engine/analysis/` |
| B5 serving | server → `engine/serving/`; **rename** main_logic.py → `engine/serving/pipeline.py` |
| B6 cli | main.py body → `engine/cli/app.py`; root keeps NEW thin shim (see D1) |
| B7 tools/tests | sanity_check, validate_track_features, take_screenshots → `engine/tools/`; test_pipeline/test_rookie/test_serialize/test_fastf1 → `engine/tools/tests/` |
| B8 docs | BUGFIX_PLAN.md, PROJECT_ANALYSIS.md, process.md, method.txt, PDF → `Docs/` |
| B9 delete | `__pycache__/`, `ui/style.css.bak`, `ui/patch_layout.txt` |

**Gate B:** `git status` shows only renames (R100/R90+); nothing deleted unexpectedly.

---

## 4. Phase C — Import Rewrite (scripted + hand-check)

### C1. Module → qualified-name map (drives both script & review)

```
config                 → engine.core.config
data_fetcher           → engine.core.data_fetcher
weather                → engine.core.weather
fantasy_scraper        → engine.core.fantasy_scraper
warm_cache             → engine.core.warm_cache
track_features_loader  → engine.core.track_features_loader
paths                  → engine.core.paths
predictor              → engine.models.predictor
elo_ratings            → engine.models.elo_ratings
temporal_model         → engine.models.temporal_model
tire_model             → engine.models.tire_model
bayesian_model         → engine.models.bayesian_model
monte_carlo            → engine.models.monte_carlo
fantasy_optimizer      → engine.strategy.fantasy_optimizer
chip_advisor           → engine.strategy.chip_advisor
price_tracker          → engine.strategy.price_tracker
self_improvement       → engine.strategy.self_improvement
backtest               → engine.analysis.backtest
post_race_check        → engine.analysis.post_race_check
analyse_results        → engine.analysis.analyse_results
dashboard              → engine.analysis.dashboard
main_logic             → engine.serving.pipeline
sanity_check           → engine.tools.sanity_check
```

### C2. Scripted rewrite (regex, applied to all `engine/**/*.py`)

```python
RULES = {flat: qualified for pairs above}
# handles BOTH forms everywhere (top-level AND indented inline):
#   ^(\s*)from config import    → \1from engine.core.config import
#   ^(\s*)import config\b       → \1from engine.core import config   (only 'import config' exact-form)
#   'import config as cfg' style: none exist today except main_logic legacy (removed)
```
Apply repeatedly until zero flat references remain; verify with grep gate:
`rg -n "^\s*(from|import) (config|data_fetcher|predictor|...)\b" engine/` → empty.

### C3. Hand-checked special cases (script will NOT catch these)
1. `predictor.train()` lazy `from backtest import _get_year_roster` → `from engine.analysis.backtest import ...` (keeps laziness — C6 satisfied).
2. `server.py:635` `uvicorn.Config("server:app", ...)` → `"engine.serving.server:app"`.
3. `take_screenshots.py` subprocess `python server.py` → `[sys.executable, "-m", "engine.serving.server"]`.
4. Batch files (see D2).
5. `scratch/*.py` sys.path shims → append project root and use `engine.*` imports.
6. `engine/__init__.py` gets NO heavy imports (avoid TF load on `import engine`) — version only.
7. `main.py` root shim (D1).

---

## 5. Phase D — Entry Points

### D1. Root `main.py` shim (CLI keeps working exactly as before)
```python
#!/usr/bin/env python
"""Thin launcher — real CLI lives in engine/cli/app.py"""
from engine.cli.app import main
if __name__ == "__main__":
    main()
```

### D2. Batch files
```bat
:: F1 Fantasy.bat
python -m engine.serving.server
:: BACKTEST.bat
python -m engine.analysis.backtest --years 2024 2025
:: VERIFY_CACHES.bat
python -m engine.core.warm_cache
:: RUN_CLI_Legacy.bat  → python main.py %*  (unchanged, works via shim)
```
All bats already `cd /d "%~dp0"` so CWD = project root and `engine` is importable; also add `if errorlevel 1 pause`.

### D3. `engine/tools/tests/*` gain a conftest-free header: `sys.path.insert(0, str(Path(__file__).resolve().parents[3]))`.

---

## 6. Phase E — Cache Compatibility

| Cache | Affected? | Action |
|---|---|---|
| cache/api/*.json, weather, fastf1 | No (path-anchored to PROJECT_ROOT) | none |
| cache/models ensemble/temporal/tire | No (sklearn/xgb/keras objects; Keras saves are file-path based) | none |
| **cache/elo/*.pkl** | **YES — unpickling `elo_ratings.Glicko2Driver` fails after rename** | `Remove-Item cache\elo\*.pkl` (rebuilds from cached results in ~1 min) |
| logs/*.json | No | none |

---

## 7. Phase F — Verification Gates (run in order)

1. **Compile:** every `.py` under `engine/`, root shim, scratch.
2. **Import graph smoke:** `python -c "import engine.cli.app, engine.serving.server, engine.analysis.backtest, engine.tools.sanity_check"` — catches missed flat imports instantly.
3. **Flat-reference gate:** grep from C2 returns zero hits repo-wide.
4. **Path integrity:** assert `engine.core.paths.API_CACHE_DIR == <root>/cache/api` and existing cache file count unchanged.
5. **Offline behavior:** run `python -m engine.tools.validate_track_features` and `python -m engine.tools.sanity_check` (network-light sections pass).
6. **Live mini-pipeline:** `python -c "from engine.serving.pipeline import run_full_pipeline; ..."` round-detect only, OR short server boot check (`python -m engine.serving.server` boots, `/api/status` 200, Ctrl-C).
7. **Backtest spot-check:** `python -m engine.analysis.backtest --years 2026 --rounds <last>` — confirms model cache reuse (no 5-min retrain ⇒ pickles/scalers fine) and metrics match pre-move values within noise.
8. **ELO rebuild:** confirm `cache/elo/glicko2_*_r*.pkl` regenerates on first predictor context load.

Rollback = revert commits (B-groups are isolated).

---

## 8. Effort & Risk Summary

| Phase | Size | Risk |
|---|---|---|
| A paths | 26 sites + 1 new file | Low (mechanical, verifiable flat) |
| B moves | 30 git-mv + 8 __init__ | Trivial |
| C imports | 128 stmts scripted + 7 manual | Medium — mitigated by grep gate |
| D entries | 1 shim + 4 bats + 2 py | Low |
| E caches | delete elo pkls | Low (fast rebuild) |
| F verify | 8 gates | — |

Estimated total: one focused session; ~45 min machine time dominated by gate 6/7.
