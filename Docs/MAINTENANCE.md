# F1 Fantasy Prediction Engine — Maintenance Runbook

This runbook documents operational procedures for maintaining the F1 Fantasy Prediction Engine throughout the Formula 1 season and transitioning across championship years.

---

## Table of Contents
1. [Race-Weekend Operations Runbook](#1-race-weekend-operations-runbook)
2. [Post-Race Validation & Self-Improvement Loop](#2-post-race-validation--self-improvement-loop)
3. [Handling Driver Swaps & Roster Updates](#3-handling-driver-swaps--roster-updates)
4. [Annual Transition & Multi-Season Git Workflow](#4-annual-transition--multi-season-git-workflow)
5. [Routine Maintenance & Cache Invalidation](#5-routine-maintenance--cache-invalidation)

---

## 1. Race-Weekend Operations Runbook

During an active race weekend, follow this schedule to ensure predictions, live odds, and fantasy lineups are accurate.

```
Wednesday / Thursday       Friday / Saturday            Saturday Post-Qualy / Sunday Pre-Race        Sunday Post-Race
┌────────────────────┐     ┌──────────────────────┐     ┌───────────────────────────────────┐     ┌──────────────────────┐
│  Warm Caches &     │ ──> │ Ingest Practice &    │ ──> │ Generate Predictions, Lineups,    │ ──> │ Run Post-Race        │
│  Verify Telemetry  │     │ Qualy Session Data   │     │ & Chip Strategy Recommendations   │     │ Validation & EWMA    │
└────────────────────┘     └──────────────────────┘     └───────────────────────────────────┘     └──────────────────────┘
```

### Phase A: Midweek Preparation (Wednesday / Thursday)
1. **Warm data caches**:
   Download and verify all upstream Ergast/Jolpica and OpenF1 timing data:
   ```bash
   python -m engine.core.warm_cache
   # Alternatively on Windows:
   .\VERIFY_CACHES.bat
   ```
2. **Verify track feature schemas**:
   Ensure the upcoming circuit's JSON configuration in `track_features/` is valid:
   ```bash
   python -m engine.tools.validate_track_features
   python -m engine.tools.sanity_check
   ```
3. **Scrape live fantasy pricing & sentiment**:
   If live scraping is enabled, ensure Playwright can access F1 Fantasy price feeds:
   ```bash
   python -m engine.core.fantasy_scraper
   ```

### Phase B: Practice & Qualifying Ingestion (Friday / Saturday)
1. **Qualifying data capture**:
   As soon as Qualifying concludes, timing data becomes available via OpenF1 / FastF1 / Jolpica.
2. **Launch the web application or CLI**:
   - **Web UI:**
     ```bash
     python -m engine.serving.server
     # or double-click "F1 Fantasy.bat"
     ```
     Navigate to `http://localhost:8000` to monitor live pipeline runs and SSE updates.
   - **Terminal CLI:**
     ```bash
     python main.py
     # or run .\RUN_CLI_Legacy.bat
     ```
3. **Inspect grid positions**:
   Verify that qualifying grid positions, grid penalties, and weather forecasts (via Open-Meteo) have loaded correctly for the targeted round.

### Phase C: Pre-Race Lineup Optimization (Saturday Night / Sunday Morning)
1. **Execute prediction ensemble**:
   Run the prediction pipeline for the upcoming race round.
   - Predictions are computed across the ML ensemble (Random Forest, XGBRanker, LGBMRanker, Ridge meta-learner) along with Glicko-2 ratings, tire degradation models, and LSTM form momentum.
   - Monte Carlo simulations generate expected fantasy point distributions and confidence intervals.
2. **Run Fantasy Optimizer & Chip Advisor**:
   - The integer linear programming (PuLP) optimizer recommends optimal driver and constructor lineups within the $100M budget constraint.
   - The Chip Advisor evaluates whether to deploy 3X Boost, Limitless, Wildcard, No Negative, or Extra DRC.
3. **Verify prediction artifacts**:
   Ensure the output JSON file has been written to:
   ```
   output/race_R<round>_<circuit>.json
   ```
   This artifact is required for post-race accuracy verification.

---

## 2. Post-Race Validation & Self-Improvement Loop

Once the Grand Prix finishes and official race classifications are published (typically 1–2 hours after checkered flag), run post-race validation.

### Step 1: Execute Post-Race Validation
Run `engine.analysis.post_race_check` specifying the race round:
```bash
python -m engine.analysis.post_race_check --round <ROUND_NUMBER> --season 2026
```

Example output:
```text
[Metrics] Post-Race Validation - Round 5 (2026)
Accuracy Metrics - R5 Saudi Arabian Grand Prix
  MAE:             2.14 positions
  RMSE:            2.85
  Spearman rho:    0.782
  Winner correct:  YES (predicted Max Verstappen · actual Max Verstappen)
  Top-3 hits:      3/3 (100%)
  Top-5 hits:      4/5 (80%)
  [OK] Accuracy log updated: logs/accuracy_log.json
  [OK] Self-improvement bias corrections updated.
```

### Step 2: How the Self-Improvement Feedback Loop Works
The post-race script automatically invokes `engine.strategy.self_improvement`:
1. **Accuracy Logging**: Appends round metrics to `logs/accuracy_log.json`.
2. **EWMA Error Computation**: Calculates Exponentially Weighted Moving Average (EWMA) position prediction bias for each driver (`compute_bias_corrections(decay=0.85, last_n=10)`).
3. **Bayesian Shrinkage**:
   - Minimum sample thresholds (`MIN_SAMPLES_OVERALL = 4`, `MIN_SAMPLES_CIRCUIT = 5`) prevent early-season overreaction.
   - Bayesian shrinkage factor $\frac{n}{n + 24}$ scales bias corrections proportionally to sample size.
   - Extreme single-race anomalies (such as lap 1 mechanical DNFs) are clamped at `MAX_SINGLE_ERROR = 4.0`.
4. **Weight Adjustments**: Computes adaptive model weighting adjustments based on recent performance and writes them to `logs/weight_adjustments.json` and `logs/bias_corrections.json`.

### Step 3: Inspect Season-Long Accuracy Summary
To review aggregate accuracy across all evaluated rounds in the current season:
```bash
python -m engine.analysis.post_race_check --summary --season 2026
```

---

## 3. Handling Driver Swaps & Roster Updates

Driver substitutions (e.g., reserve drivers standing in for injured drivers or mid-season seat reassignments) must be handled cleanly without corrupting ratings or breaking the optimizer.

### Architecture of Roster Discovery
- **Live / Online Mode**: `engine.core.data_fetcher.get_season_roster()` dynamically queries Jolpica/Ergast championship standings and cross-references active F1 Fantasy entry lists. Driver changes reported by official feeds are detected automatically.
- **Offline / Pre-Round-1 Seed**: When running offline or prior to Round 1 standings being populated, the engine falls back to static seeds in `engine/core/config.py`.

### Procedure for Driver Substitutions
When a driver change occurs:

1. **Verify Official API Feeds**:
   Run cache warmer or inspect the driver roster to confirm Jolpica has registered the new driver:
   ```bash
   python -c "from engine.core.data_fetcher import get_season_roster; print(get_season_roster(2026))"
   ```

2. **Update Fallback Seed (if running offline or pre-season)**:
   In `engine/core/config.py`:
   - Update `DRIVER_TEAMS_2026` to reflect the 22 active seats.
   - Add the driver's three-letter abbreviation to `DRIVER_SHORT_2026` (e.g., `"BEA": "Oliver Bearman"`).

3. **Rookie & Reserve Driver Ratings**:
   - If the substitute driver has prior F1 race history (e.g. Liam Lawson, Nico Hülkenberg), Glicko-2 ratings will pull historical performance from `HISTORICAL_SEASONS`.
   - If the driver is an F1 rookie with zero historical starts, `engine/models/elo_ratings.py` applies a rookie prior initialized from the constructor baseline with an uncertainty penalty (`test_rookie.py`).

4. **Phantom Driver Filtering**:
   If a driver was replaced mid-season, their name may still appear in championship standings with accumulated points. The pipeline cross-references active F1 Fantasy entries to ensure departed drivers are excluded as "phantoms" from lineup generation.

---

## 4. Annual Transition & Multi-Season Git Workflow

At the end of a championship season, follow this procedure to prepare the engine for the upcoming championship year.

### Git Branch & Tagging Strategy
1. **Archive the Completed Season**:
   Tag the final commit of the championship season and push to GitHub:
   ```bash
   git checkout main
   git tag -a v2026.final -m "Final release for 2026 F1 Championship season"
   git push origin v2026.final
   ```
2. **Create a Maintenance Branch (Optional)**:
   If long-term bugfixes for the previous season are desired:
   ```bash
   git checkout -b season/2026
   git push -u origin season/2026
   git checkout main
   ```

### Codebase Configuration Updates for New Season
In `engine/core/config.py`:
1. **Increment `CURRENT_SEASON`**:
   ```python
   CURRENT_SEASON = 2027
   ```
2. **Update `HISTORICAL_SEASONS`**:
   Add the completed season to the historical training window:
   ```python
   HISTORICAL_SEASONS = [2022, 2023, 2024, 2025, 2026, 2027]
   ```
3. **Calibrate `SEASON_WEIGHTS`**:
   Decay older seasons and assign full weight to the newly concluded and upcoming seasons:
   ```python
   SEASON_WEIGHTS = {
       2022: 0.10,
       2023: 0.20,
       2024: 0.35,
       2025: 0.55,
       2026: 0.80,
       2027: 1.00,
   }
   ```
4. **Update Regulatory Flags**:
   If the new season introduces major aerodynamic or power unit regulation changes, add the season year to `REGULATION_CHANGE_YEARS`.
5. **Update Calendar & Circuit Lists**:
   - Update `CIRCUIT_CALENDAR` or circuit schedule mappings for any new or relocated Grands Prix.
   - Adjust `SC_PROBABILITY` and `VSC_PROBABILITY` for any reconfigured tracks.
   - Add new circuit JSON definition files to `track_features/<circuit_key>.json` and validate them with `python -m engine.tools.validate_track_features`.
6. **Update Driver Registry Seeds**:
   Define `DRIVER_TEAMS_<YEAR>` and `CONSTRUCTORS_<YEAR>` with confirmed seat announcements for pre-round-1 runs.

### Run Verification & Backtests
After updating season configurations:
1. Run all unit and contract tests:
   ```bash
   python -m unittest engine.tools.tests.test_contracts
   python -m unittest engine.tools.tests.test_ui_smoke
   ```
2. Run historical multi-season backtests:
   ```bash
   python -m engine.analysis.backtest --years 2025 2026
   ```

---

## 5. Routine Maintenance & Cache Invalidation

### Clearing Model and Data Caches
- **Clear Trained Model Checkpoints**:
  If feature schemas or historical data weights change, delete cached ensemble models to force retraining:
  ```powershell
  Get-ChildItem -Path "cache/models" -Filter "*.pkl" | Remove-Item -Force
  ```
- **Clear API Data Cache**:
  Cached API responses from Jolpica or OpenF1 are stored in `cache/`. If upstream timing data has been corrected:
  ```powershell
  # Delete specific cached seasons or rounds
  Remove-Item -Recurse -Force "cache/jolpica"
  Remove-Item -Recurse -Force "cache/openf1"
  ```
  Then re-run `python -m engine.core.warm_cache` to repopulate.
