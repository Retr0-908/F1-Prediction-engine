# Fantasy Feature Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `driver_overtake_delta` and `driver_dnf_risk` features to the predictor ensemble to maximize F1 Fantasy points.

**Architecture:** We will modify `data_fetcher.py` to calculate a rolling average of positions gained (start vs finish), and update `predictor.py` to ingest this along with a weighted DNF risk. The model's feature space will expand from 54 to 56 features.

**Tech Stack:** Python, Pandas, Numpy

## Global Constraints

- Retrain caches using the standard procedure.
- The ensemble feature count must correctly update to 56.

---

### Task 1: Update Data Fetcher with Overtake Delta

**Files:**
- Modify: `data_fetcher.py`

**Interfaces:**
- Consumes: The `grp` pandas DataFrame grouping inside `compute_multiseason_driver_form`.
- Produces: `overtake_delta` added to the `all_season_data[drv][year]` dictionary, and properly weighted in the returned dictionary for each driver.

- [ ] **Step 1: Compute overtake_delta per season**
In `data_fetcher.py`, inside `compute_multiseason_driver_form`, update the dictionary creation for `all_season_data[drv][year]` around line 1432. Add the `overtake_delta` field. It should compute the mean difference between `grid` and `position`, strictly where `grid > 0`.

```python
                    # Inside the loop building all_season_data[drv][year]
                    valid_grid = grp[grp["grid"] > 0]
                    overtake_delta = float((valid_grid["grid"] - valid_grid["position"]).mean()) if not valid_grid.empty else 0.0
                    
                    all_season_data[drv][year] = {
                        "avg_position": float(grp["position"].mean()),
                        "avg_points":   float(grp["points"].mean()),
                        "dnf_rate":     float(grp["dnf"].mean()),
                        "overtake_delta": overtake_delta,
                        "form_score":   float(grp["points"].mean() * 2 - grp["position"].mean() * 0.5 - grp["dnf"].mean() * 20),
                        "momentum_trend": 0.0,
                        "weight":       weight,
                        "race_count":   len(grp),
                    }
```

- [ ] **Step 2: Aggregate in weighted average**
In `data_fetcher.py`, inside `wavg(key)` aggregation logic around line 1450, add `overtake_delta` to the `merged[drv]` assignment.

```python
        merged[drv] = {
            "avg_position": wavg("avg_position"),
            "avg_points":   wavg("avg_points"),
            "dnf_rate":     wavg("dnf_rate"),
            "overtake_delta": wavg("overtake_delta"),
            "form_score":   wavg("form_score"),
            "momentum_trend": wavg("momentum_trend"),
            "races_counted": int(sum(d["race_count"] for d in season_data.values())),
        }
```

- [ ] **Step 3: Commit**
```bash
git add data_fetcher.py
git commit -m "feat: add overtake_delta to driver form stats"
```

---

### Task 2: Add Features to Predictor Ensemble

**Files:**
- Modify: `predictor.py`
- Modify: `sanity_check.py`

**Interfaces:**
- Consumes: `form.get("overtake_delta", 0.0)` from the driver form dictionary.
- Produces: A 56-feature vector returned by `_build_features`.

- [ ] **Step 1: Update FEATURE_NAMES**
In `predictor.py`, append the two new features to the end of the `FEATURE_NAMES` list around line 190. `N_FEATURES` will automatically update to 56 since it uses `len()`.

```python
    "overtake_mode_efficiency",        # how effective Straight Mode is at this circuit (0-1)
    "track_power_sensitivity",         # derived: (1/downforce) * power_unit importance
    # Phase 6: Fantasy optimized features
    "driver_overtake_delta",           # Positions gained/lost during race
    "driver_dnf_risk",                 # Combined incident and reliability risk
]
```

- [ ] **Step 2: Extract features in _build_features**
In `predictor.py`, inside `_build_features`, extract `overtake_delta` and compute `driver_dnf_risk` around line 504.

```python
        # — Rolling form (EWMA) —
        form = driver_form.get(driver_name, {})
        form_avg_pos    = form.get("avg_position",   10.0)
        form_avg_pts    = form.get("avg_points",      5.0)
        form_dnf_rate   = form.get("dnf_rate",        0.08)
        overtake_delta  = form.get("overtake_delta",  0.0)   # NEW
        form_score      = form.get("form_score",      0.0)
        momentum_trend  = form.get("momentum_trend",  0.0)
```

And compute the DNF risk near the constructor reliability section (line 512):
```python
        # — Constructor reliability —
        ctor_rel  = ctor_reliability.get(ctor_name, {})
        ctor_dnf  = ctor_rel.get("dnf_rate",   0.08)
        ctor_trend = ctor_rel.get("pts_trend",  0.0)
        
        # — Fantasy DNF Risk —
        driver_dnf_risk = (form_dnf_rate * 0.6) + (ctor_dnf * 0.4)
```

- [ ] **Step 3: Append to vector**
At the end of `_build_features` in `predictor.py`, append the new variables to the numpy array return statement:

```python
            float(overtake_mode_eff),        # overtake_mode_efficiency
            float(track_power_sensitivity),  # track_power_sensitivity
            float(overtake_delta),           # driver_overtake_delta
            float(driver_dnf_risk),          # driver_dnf_risk
        ], dtype=float)
```

- [ ] **Step 4: Update sanity_check.py**
In `sanity_check.py`, update line 9 to assert `N_FEATURES == 56`.

```python
    assert N_FEATURES == 56, f"Expected 56 features, got {N_FEATURES}"
```

- [ ] **Step 5: Verify feature count**
Run `python sanity_check.py` to verify the predictor feature count matches 56 and all assertions pass.
Expected: `[PASS] predictor.py — 56 features verified`

- [ ] **Step 6: Commit**
```bash
git add predictor.py sanity_check.py
git commit -m "feat: add fantasy features to predictor ensemble"
```

---

### Task 3: Retrain the Ensemble

**Files:**
- Modify: `cache/models/` (Delete stale cache)

**Interfaces:**
- Consumes: The updated `predictor.py` with 56 features.
- Produces: Updated `ensemble_*.pkl` cached models.

- [ ] **Step 1: Clear stale model caches**
Run the following command to invalidate the old model cache:
```powershell
python -c "import os, glob; [os.remove(f) for f in glob.glob('cache/models/ensemble_*.pkl')]"
```

- [ ] **Step 2: Run Backtest to trigger training**
Run a backtest on the completed seasons to force the predictor to retrain its models with the new 56 features:
```bash
python backtest.py --years 2025
```
Wait for the training to finish and the backtest metrics to output.

- [ ] **Step 3: Commit**
*(No commit necessary for cache invalidation)*
