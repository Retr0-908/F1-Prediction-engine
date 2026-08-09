---
name: f1-prediction-engine
description: Specialized workflow skill for the F1 Fantasy Prediction Engine. Provides procedures to validate track feature schemas, warm FastF1 telemetry caches, execute multi-season backtests, and run prediction pipelines safely without data leakage.
---

# F1 Prediction Engine Skill & Runbook

This skill provides step-by-step procedures for managing, testing, and retraining the F1 Fantasy Prediction Engine with track-specific feature enrichment.

---

## 1. Quick Verification & Schema Checks

Always run schema validation before retraining or generating fantasy recommendations:

```bash
python validate_track_features.py
python sanity_check.py
```

### Key Verification Metrics:
- **Feature Count**: Must strictly equal **54 features** (37 base + 15 track features + 2 derived metrics).
- **Track Enrichment**: All 24 circuits (Albert Park through Saudi Arabia) must show `enriched OK`.

---

## 2. Multi-Season Backtesting Workflow

Evaluate the accuracy of the 54-feature ensemble stack against historical race rounds:

```bash
# Backtest specific 2025 rounds
python backtest.py --years 2025 --rounds 1 2 3 4 5

# Backtest completed season
python backtest.py --years 2024 2025
```

### Metrics Benchmark:
- **Race MAE**: Target $< 2.5$ positions.
- **Spearman Rank Correlation**: Target $> 0.85$.
- **Winner Top-3 Hit Rate**: Target $> 90\%$.

---

## 3. Safe Model Retraining & Cache Invalidation

If new historical telemetry or track features are updated, invalidate stale model caches:

```powershell
python -c "import os, glob; [os.remove(f) for f in glob.glob('cache/models/ensemble_*.pkl')]; print('Stale model caches cleared.')"
```

The next call to `F1Predictor.train()` will re-build the ensemble using `TimeSeriesSplit` across historical seasons (2021–2025) with regulation-era weighting:
- 2021: 0.05
- 2022: 0.18
- 2023: 0.30
- 2024: 0.50
- 2025: 0.75
- 2026: 1.00

---

## 4. Running Web Dashboard & REST API

Start the FastAPI live dashboard with real-time SSE updates:

```bash
python server.py
```
Open your browser to `http://localhost:8000` to view live race predictions, Monte Carlo position distributions, and PuLP fantasy lineup recommendations.
