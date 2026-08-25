# Improvements Implementation Plan

**Scope:** (1) live cache-download progress bar · (2) accurate driver-count reporting ·
(3) full-hardware utilization incl. RTX 5060 GPU · (4) richer live analysis telemetry.
**Machine profile (verified):** AMD-class laptop, **16 logical cores**, **NVIDIA RTX 5060 Laptop GPU (8 GB)**,
xgboost **3.3.0**, lightgbm **4.7.0** (pip wheel), scikit-learn 1.9, TensorFlow **not installed**.

---

## Improvement 1 — Live Cache Download Progress Bar

### Current state
`warm_cache.verify_and_download_caches(progress_callback)` emits coarse `(year, "schedule", msg)`
events; the server forwards them as SSE `DOWNLOADING` events; the Settings screen shows only a
text status span (`app.js rebuildCache()` ~L2958). No totals, no fraction complete.

### Design
1. **Pre-compute the work manifest** before downloading anything:
   ```python
   # warm_cache.py
   def _build_manifest(seasons):
       """[(season, round, name, [endpoints...]), ...] for every completed race."""
       units = []
       for year in seasons:
           sched = get_season_schedule(year)
           for race in completed_races(sched):
               eps = ["results", "quali", "drv_standings", "ctor_standings"]
               if year >= 2023: eps.append("grid_penalties")
               eps += ["fastf1_R", "fastf1_Q"]
               units.append((year, race["round"], race["name"], eps))
       return units
   ```
   `TOTAL_UNITS = sum(len(eps) for units)` — typically ~900–1100 units for 2021–2026.
   A unit is *skipped instantly* if its cache file exists & fresh (manifest check uses
   `_load_cache(..., allow_stale=True)` existence probe — no network).
2. **Thread-safe counter**: `ProgressState(done, failed, current_item, phase)` updated by each
   worker task (both Jolpica and FastF1 endpoints become individually countable).
3. **Callback contract upgrade** (backwards-compatible):
   ```python
   progress_callback({
     "done": 412, "total": 1043, "failed": 2,
     "current": "2024 R14 · FastF1 Q session",
     "phase": "downloading" | "verifying" | "complete",
     "rate_per_min": 21.5,          # rolling window
     "eta_min": 30.1,
   })
   ```
   Server (`/api/cache/rebuild` `_bg`) wraps it into
   `sync_progress_callback(run_id, "DOWNLOADING", "loading", current, data={...})`.
4. **UI**: inside the Settings screen next to the existing status span add:
   ```html
   <progress id="cache-progress-bar" max="100" value="0"></progress>
   <span id="cache-progress-text">412 / 1043 · ETA 30m</span>
   ```
   `app.js` DOWNLOADING handler sets `bar.value = done/total*100`. On COMPLETE → 100 %,
   green state, existing reload behavior unchanged. Failed count shown in amber when > 0.

### Files touched
`engine/core/warm_cache.py` (manifest + counter), `engine/serving/server.py` (~5 lines),
`ui/app.js` (+~25 lines), `ui/index.html` (+3 lines), `ui/style.css` (progress styling).

---

## Improvement 2 — Accurate Driver Count in Analysis Log

### Current state
`pipeline.py`: `progress_callback(..., f"{len(driver_prices)} drivers loaded")` reports the
*price-table* size (20 when the fantasy site omits drivers), while the prediction roster is now
the merged set (23). Misleading.

### Fix
After roster merge:
```python
merged_n   = len(dynamic_roster)
scraped_n  = len(driver_prices)
gap_note   = f" ({merged_n - scraped_n} backfilled from standings)" if merged_n > scraped_n else ""
progress_callback(run_id, "PRICES", "done",
                  f"Prices for {scraped_n} drivers · predicting field of {merged_n}{gap_note}")
```
Also emit the merged names list in `data={"field": sorted(dynamic_roster)}` so the UI log can
expand it. No model changes.

---

## Improvement 3 — Hardware Utilization (16 cores + RTX 5060)

### Research conclusions (verified against installed versions/docs)
| Component | Method | Feasible here? |
|---|---|---|
| XGBoost GPU | xgboost ≥2.0 pip wheels ship CUDA; `tree_method="hist", device="cuda"` | ✅ **Yes — biggest single win** |
| LightGBM GPU | requires custom OpenCL build; pip wheels are CPU-only | ❌ stay CPU (`n_jobs=-1`, already default-ish) |
| RandomForest | `n_jobs=-1` | ✅ already set |
| Monte Carlo (pure-Python loop, 1000–5000+ sims) | embarrassingly parallel → `ProcessPoolExecutor` with per-chunk RNG seeds | ✅ near-linear speedup on 16 cores |
| Pipeline stage overlap | weather/prices/standings fetches are I/O-bound → run concurrently in threads | ✅ saves ~10–20 s/run |
| TensorFlow models | not installed; native Windows TF dropped GPU after 2.11 (needs WSL2 or `tensorflow-directml`) | ⛔ out of scope now; LSTM/tire fall back gracefully anyway |
| NumPy BLAS | ensure MKL/OpenBLAS threading not fighting job pools (`OMP_NUM_THREADS` tuning in workers) | minor |

### 3a. `engine/core/hardware.py` (new, stdlib-only detection)
```python
def profile() -> dict:
    return {
      "cpu_threads": os.cpu_count(),
      "gpu": detect_gpu(),        # try nvidia-smi subprocess; None if absent
      "xgb_device": "cuda" if gpu else "cpu",
    }
```
Detection: `shutil.which("nvidia-smi")` → parse name/memory; plus a 10-row xgboost CUDA probe
at train time wrapped in try/except → automatic CPU fallback (driver-less laptops, WSL quirks).

### 3b. Predictor training changes
```python
hw = hardware.profile()
self.race_xgb = xgb.XGBRanker(..., tree_method="hist", device=hw["xgb_device"])
# meta-learner clones inherit params automatically
```
Emit `"ML_MODEL"` progress payload `{"device": hw["xgb_device"], "threads": cpu_count}`.
Expected effect: XGB ranker fits drop from ~60–90 s to ~5–15 s; overall retrain ~7 min → ~3–4 min.
Model-cache key must embed `xgb_device` — a CUDA-trained booster can predict on CPU but the
pickle should record provenance; simplest is appending `hw["xgb_device"]` to `seasons_str`.

### 3c. Monte Carlo parallelization
- Split `n_simulations` into `cpu_threads` chunks; each process runs
  `_simulate_chunk(chunk_size, seed_base+i)` returning `dict[driver -> list[float]]`;
  parent concatenates lists per driver into `DistributionStats`.
- Seeds derived deterministically (`seed*1000003 + chunk_idx`) so results stay reproducible
  and identical to serial mode statistically.
- Guard: `multiprocessing.get_context("spawn")` on Windows; worker function must be
  module-level; pass only plain dicts across the boundary (race/quali orders are JSON-ish).
- Skip pool when `n_simulations < 2000` (pool startup overhead ≈ 1 s).

### 3d. Concurrent pipeline prelude
In `run_full_pipeline`, wrap weather + price-scrape + API-roster in
`ThreadPoolExecutor(max_workers=3)`; keep ordering of progress events by emitting each as it
completes. Network-bound, GIL-free benefit.

### Acceptance test
Backtest one round twice (`--rounds N`) with old/new code path timings logged;
expect ≥2× wall-clock improvement on retrain and ≥4× on MC-heavy runs (5000 sims).

---

## Improvement 4 — Live Analysis Telemetry

### Current state
SSE stages exist (NEXT_RACE→…→COMPLETE); app.js renders a fixed node track + a few HUD fields
(round/circuit/weather/ensemble weights/sims). Most internals invisible.

### Design — structured substage events
Extend `sync_progress_callback` payloads (`data={...}`) — no new endpoint needed:

| Stage | New `data` payload | UI rendering |
|---|---|---|
| NEXT_RACE | `{sprint: bool, mode}` | badge on track node |
| WEATHER | `{sessions_loaded, source: live/historical}` | HUD subtext |
| PRICES | `{scraped, merged_field, gap_filled:[names], source_tier}` | log expandable row |
| DATA (new substage) | `{cache_hits, api_calls, elo_drivers, form_races}` emitted during load_context via a lightweight hook | counters grid |
| ML_MODEL | `{dataset_samples, skipped_races, device, threads, per_model_ms:{rf,xgb,lgb,meta}, cv_folds}` | telemetry panel rows |
| PREDICTIONS | existing sims progress + `{pole, winner, fastest_predicted_lap?}` | existing HUD + new rows |
| ANALYSIS | `{lookahead_rounds_done, lp_vars}` | log lines |

Implementation notes:
- `predictor.load_context/train` accept an optional `telemetry_cb(dict)` (default no-op);
  pipeline passes a closure forwarding into `progress_callback`. Keeps engine decoupled.
- **UI**: new right-hand "Engine Telemetry" column on the Analysis screen (index.html + ~80
  lines app.js): key/value grid auto-updating from any arriving payload keys, plus per-stage
  duration badges computed client-side from event timestamps. Log lines gain timestamps +
  collapsible `data` JSON (`<details>` element).
- Throttle: telemetry_cb fires at most every 250 ms per stage (simple time-gate in the closure)
  so SSE isn't flooded during feature building.

### Files touched
`engine/models/predictor.py` (hooks), `engine/serving/pipeline.py` (payloads),
`ui/index.html` (panel markup), `ui/app.js` (~120 lines), `ui/style.css` (~40 lines).

---

## Execution Order & Risk

| Step | Depends on | Risk | Effort |
|---|---|---|---|
| 2. count fix | none | none | 15 min |
| 4. telemetry payloads + panel | none | low (additive) | 3–4 h |
| 1. download progress bar | none | low | 2 h |
| 3a/3b. hardware detect + XGB CUDA | none | medium (fallback path must be tested with GPU absent — simulate via env var) | 1–2 h |
| 3c. MC multiprocessing | 3a | medium (Windows spawn pickling) | 2–3 h |
| 3d. concurrent prelude | none | low | 1 h |
| Final gates | all | — | compile, sanity_check, one live pipeline, backtest spot-round, timings vs baseline |

Total estimate: 1–1.5 focused sessions. Every change is additive; rollback = revert commit.
