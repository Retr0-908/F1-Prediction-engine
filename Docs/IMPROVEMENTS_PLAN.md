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
   A unit is *skipped instantly* if its cache file exists & fresh.
   **⚠️ Amendment (audit):** existence-probing must NOT use `allow_stale` semantics for
   FastF1 units — those live in FastF1's own parquet cache (`cache/fastf1/`), not our JSON
   layer, so JSON-key probes would report false "already cached". Simpler & safe: **count
   every manifest unit as work** and let cached hits complete in milliseconds — totals stay
   truthful by construction.
2. **Thread-safe counter**: `ProgressState(done, failed, current_item, phase)` updated by each
   worker task (both Jolpica and FastF1 endpoints become individually countable).
3. **Callback contract change**:
   ```python
   progress_callback({
     "done": 412, "total": 1043, "failed": 2,
     "current": "2024 R14 · FastF1 Q session",
     "phase": "downloading" | "verifying" | "complete",
     "rate_per_min": 21.5,          # rolling window
     "eta_min": 30.1,
   })
   ```
   **⚠️ Amendment:** this is *not* backwards-compatible (old signature was positional
   `(year, stage, msg)`) — but there is exactly ONE consumer (`server.py` rebuild `_bg`)
   plus `__main__` prints; both are updated in the same commit, so the risk is nil.
   Server wraps it into `sync_progress_callback(run_id, "DOWNLOADING", "loading", current, data={...})`.
4. **UI**: inside the Settings screen next to the existing status span add:
   ```html
   <progress id="cache-progress-bar" max="100" value="0"></progress>
   <span id="cache-progress-text">412 / 1043 · ETA 30m</span>
   ```
   `app.js` DOWNLOADING handler sets `bar.value = done/total*100`. On COMPLETE → 100 %,
   green state, existing reload behavior unchanged. Failed count shown in amber when > 0.

### ⚠️ Amendment (audit): SSE queue lifecycle must be fixed first
`server.py:525-527` pops `run_queues[run_id]` in the stream generator's `finally:` — i.e. on
**client disconnect**, not only on terminal events. A page refresh mid-download therefore orphans
the queue while the producer keeps writing into it: every subsequent event is silently dropped and
reconnecting yields "Invalid run_id". Today this only loses text lines; with a progress bar it
would freeze at e.g. 43 % forever. **Bundled fix:** pop the queue ONLY after a terminal event
(`COMPLETE`/`ERROR`/`_TERMINATE` observed); on disconnect, keep the queue alive so a reconnecting
EventSource resumes mid-stream (events buffered while detached are replayed — bounded by the
existing terminal-event cleanup).

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

## Improvement 3 — Universal Hardware Utilization

**Design principle:** every machine gets *the best execution plan that machine supports*,
discovered at runtime — never tuned to one specific box. A Raspberry Pi, an 8-core desktop,
and a 16-core laptop with an RTX GPU must all work out of the box, degrading gracefully.

### Research conclusions (verified against installed versions/docs)
| Component | Method | Portability |
|---|---|---|
| XGBoost GPU | xgboost ≥2.0 pip wheels ship CUDA; `tree_method="hist", device="cuda"` | ✅ NVIDIA + pip install, any OS |
| XGBoost AMD | ROCm/HIP builds exist but are not pip-installable | detected only if user built it — same `device=` param path |
| XGBoost Apple Silicon | no Metal/CUDA backend | → CPU fallback |
| LightGBM GPU | requires custom OpenCL source build; pip wheels are CPU-only | ❌ always CPU `n_jobs=-1` unless the user's wheel reports GPU support |
| RandomForest | `n_jobs=-1` | ✅ universal |
| Monte Carlo loop | embarrassingly parallel → process pool with per-chunk seeds | ✅ scales to whatever core count exists |
| Pipeline stage overlap | I/O-bound fetches in a thread pool | ✅ universal |
| TensorFlow | native-Windows GPU dropped after TF 2.11; Linux CUDA / `tensorflow-directml` / Apple `tensorflow-macos` vary | ⛔ never assumed; models are optional and load lazily |

### 3a. `engine/core/hardware.py` — runtime capability profile (stdlib-only)

```python
def profile(force: bool = False) -> dict:
    """Capability profile, cached per-process. Never raises."""
    return {
      "cpu_threads":      os.cpu_count() or 2,
      "gpu_vendor":       "nvidia" | None,          # via nvidia-smi probe
      "gpu_name":         "RTX 5060 ..." | None,
      "xgb_device":       "cuda" | "cpu",           # VERIFIED by micro-fit probe
      "lgb_device":       "cpu",                    # until a GPU wheel is proven
      "mc_workers":       recommended_process_pool_size(),
      "io_workers":       min(8, max(3, cpu_threads // 4)),
      "blas_note":        "OMP_NUM_THREADS guidance for pools",
    }
```

**Detection ladder (portable, no hard deps):**
1. `os.cpu_count()` — baseline everywhere.
2. GPU vendor: `shutil.which("nvidia-smi")` → run `nvidia-smi --query-gpu=name --format=csv,noheader`
   (Windows + Linux); absence ⇒ no NVIDIA assumption. No other vendors are *assumed*;
   AMD/Intel GPU users simply get the CPU path (documented in README).
3. **Ground-truth verification, not trust**: a 10-row × 20-round XGBoost fit with
   `device="cuda"` inside try/except. If it errors (no CUDA driver, WSL quirk, non-NVIDIA),
   silently record `"cpu"`. This makes the feature work on ANY machine regardless of what
   detection guessed.
4. `mc_workers`: `max(1, min(cpu_threads - 1, 15))` minus 2 reserved cores when
   `cpu_threads >= 8`; capped at 32; further reduced if `total memory < 6 GB`
   (`sysctl`/`/proc/meminfo`/Windows GlobalMemoryStatusEx via ctypes — best effort, default full).
5. **User override contract** (env vars, all optional):
   - `F1E_XGB_DEVICE=auto|cuda|cpu`
   - `F1E_MC_WORKERS=<int>` / `F1E_MC_WORKERS=serial`
   - `F1E_IO_WORKERS=<int>`
   Overrides exist so a user can pin behavior without editing code (e.g. leave cores free
   while gaming, or force serial for deterministic debugging).

### 3b. Predictor training changes (device-agnostic)
```python
hw = hardware.profile()
self.race_xgb = xgb.XGBRanker(..., tree_method="hist", device=hw["xgb_device"])
```
- Model-cache key embeds `hw["xgb_device"]` (a CUDA-trained booster pickle records provenance
  and won't be reused by a CPU-only machine sharing a cache dir).
- Training emits telemetry `{"device": ..., "threads": ...}` so any user sees which path ran.
- Expected effect scales with hardware: reference laptop (16C + RTX 5060) retrain ~7 → ~3–4 min;
  a 4-core CPU-only box still gains from RF `n_jobs=-1`, LGBM threads, and parallel MC.

### 3c. Monte Carlo parallelization (adaptive)
- Chunk count = `hw["mc_workers"]`; chunks = `ceil(n_sims / chunk_target)`;
  **skip pool entirely** when `n_simulations < 2000` OR `mc_workers == 1` (small machines and
  serial override take the zero-overhead path).
- Deterministic seeding: `seed*1000003 + chunk_idx` — results reproducible on ANY machine.
- Windows/macOS spawn-safe: module-level worker fn, plain-dict payloads only.
- Memory guard: chunk target shrinks if driver field > 30 (never expected, but safe).

**⚠️ Amendments (audit):**
1. **Reuse one pool per process.** `run_full_pipeline` calls `simulate_race_weekend` up to
   **4×** (main race + 3 lookahead rounds). A fresh `ProcessPoolExecutor` per call pays
   spawn + import cost (~1–3 s on Windows) four times over. Use a lazily-created,
   module-level cached executor (created on first parallel call, reused; never at import).
2. **Default sims sit below the threshold.** Pipeline default is `sims=1000`, so typical runs
   take the *serial* path and see no MC speedup. Once parallel lands, raise the pipeline
   default to `sims=2500` — faster than today's 1000-serial *and* statistically tighter.
3. **Statistical, not bitwise, equivalence.** Chunked per-chunk RNG streams draw different
   samples than the single serial stream: same-seed parallel ≠ serial numerically (only in
   distribution). No golden-value tests exist, so nothing breaks — but this must be stated
   wherever reproducibility is claimed (docstring), and `F1E_MC_WORKERS=serial` remains the
   bit-exact fallback for debugging.

### 3d. Concurrent pipeline prelude (thread pool sized by `io_workers`)
Weather + price-scrape + standings fetch overlap; on a 2-core machine this degrades to
sequential-ish automatically.

### Capability matrix (what each class of machine gets)
| Machine | XGB | RF/LGBM | MC | Stage overlap |
|---|---|---|---|---|
| 16C + RTX 5060 (reference) | CUDA | 16-thread CPU | 13 workers | 4 IO threads |
| 8C desktop, no GPU | CPU | 8-thread | 6 workers | 3 |
| 4C old laptop | CPU | 4-thread | 3 workers | 3 |
| Apple Silicon M-series | CPU | M-thread | M−1 workers | 4 |
| 2C VPS / CI | CPU | 2-thread | serial (<2000 sims always serial) | 3 |

### Acceptance test (runs on ANY machine)
1. `python -c "from engine.core.hardware import profile; print(profile())"` prints a sane
   profile with no exceptions.
2. Force-fallback check: `F1E_XGB_DEVICE=cpu python -m engine.analysis.backtest --rounds N`
   works identically (deterministic seeds) — guards against accidental hard GPU dependency.
3. Timing log line emitted once per run:
   `engine-hw: xgb=cuda mc=13 io=4 (detected in 0.4s)` — so users can verify utilization.
4. On the reference machine: retrain ≥2× faster, ≥4× on 5000-sim MC vs serial baseline.

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
- **⚠️ Amendment (audit):** the `DATA` substage counters (cache_hits / api_calls) require a
  global stats singleton in `data_fetcher` incremented in `_load_cache`/`_jolpica_get`. To keep
  the hot path free of overhead when telemetry is off, use a plain module-level dict with
  integer increments (nanoseconds) and **no locks** — approximate counts are fine for display;
  reset at pipeline start via `telemetry_reset()`.
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

---

## Audit Verdict (post-review amendments applied)

**Breakage-risk register — what could have broken, and why it now can't:**

| # | Risk | Mitigation in plan |
|---|---|---|
| R1 | Progress bar freezes mid-download after page refresh | SSE queue pop moved to terminal-events-only (I1 amendment) |
| F1 | FastF1 units misreported as cached | Manifest counts all units; cache hits complete fast — totals always truthful |
| C1 | Callback signature change breaks a hidden consumer | Verified: exactly one consumer (`server._bg`) + `__main__`; updated in same commit |
| G1 | CUDA-trained pickle reused on CPU-only machine → predict failure | `xgb_device` embedded in model-cache key |
| G2 | GPU absent/driver broken at runtime | 10-row micro-fit probe = ground truth; silent CPU fallback; env override to pin |
| M1 | Pool spawn cost ×4 per pipeline run | Module-level persistent executor |
| M2 | Windows spawn pickling failures | Module-level worker fn; plain-dict payloads; tested via acceptance run on Windows |
| M3 | Silent numeric drift breaks reproducibility assumptions | Documented statistical-equivalence + `F1E_MC_WORKERS=serial` bit-exact escape hatch |
| T1 | Telemetry counters slow hot paths | Lock-free integer dict increments; zero-cost when telemetry unused |

**Invariants preserved:** model-cache schema semantics (device now part of key); deterministic
backtest results (backtest MC path unaffected — backtester doesn't call Monte Carlo);
`F1 Fantasy.bat` / CLI entry contracts unchanged; no DB/migrations; all UI additions additive.
**One intentional behavior change, flagged:** pipeline default sims 1000 → 2500 *after* parallel
MC lands (net-faster than today's serial 1000).
