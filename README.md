# F1-Prediction-engine

F1 Fantasy prediction engine for the 2026 season: ML ensemble (RF + XGBRanker + LGBMRanker + Ridge meta) over Glicko-2 ratings, LSTM form momentum, tire-degradation and per-circuit track features — with Monte Carlo risk modeling, lineup optimization, chip strategy, and a self-improving bias-correction loop.

## Quick start

| I want to... | Run |
|---|---|
| Use the web app | `F1 Fantasy.bat` (double-click) |
| Re-download API data | `VERIFY_CACHES.bat` |
| Backtest historical seasons | `BACKTEST.bat` |
| Legacy terminal CLI | `RUN_CLI_Legacy.bat` or `python main.py` |

## Layout

```
engine/                  the package
├── core/                config, data fetching (Jolpica/OpenF1/FastF1), weather, price scraping
├── models/              predictor ensemble, Glicko-2, LSTM, tire model, Bayesian EV, Monte Carlo (+ Julia engine)
├── strategy/            transfers/dream-team optimizer, chip advisor, price tracker, self-improvement
├── analysis/            backtester, post-race validation, results analysis, HTML dashboard
├── serving/             FastAPI server + pipeline orchestrator
├── cli/                 terminal app (launched via root main.py shim)
└── tools/               sanity checks, validators, dev scripts/tests

ui/                      web frontend (vanilla JS SPA)
track_features/          per-circuit JSON feature files
cache/ logs/ output/     runtime data (auto-created; safe to delete cache/)
Docs/                    plans & reference documents
scratch/                 one-off dev scripts
```

Lineups are **auto-detected** each run from championship standings (`engine.core.data_fetcher.get_season_roster`) — static tables in `engine/core/config.py` are only pre-round-1/offline seeds.

## Setup

```powershell
pip install -r requirements.txt
python -m playwright install chromium   # optional: live fantasy price scraping
```

See `Docs/REORGANIZATION_PLAN.md` for the layout rationale and `Docs/BUGFIX_PLAN.md` for the correctness audit this codebase went through.
