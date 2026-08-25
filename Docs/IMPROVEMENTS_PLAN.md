# Improvements Implementation Plan

> **IMPLEMENTATION STATUS (2026-08-25):**
> ✅ **LANDED & VERIFIED:** I7 hotfix · 9-C1/C2/C3/C4 · 9-H1–H4 · 8a/8b/8d/8e(41 tests green) ·
> 8c sweep (46 sites) · 5a/5b/5c · 9-M1/M6/M8/M9/M10/M12/M13/M14 · I2 · I3 (CUDA verified live:
> `xgb=cuda` on RTX 5060, retrain validated, field=22, sane top-5) · I3/3c parallel MC
> (5000 sims ≈ 3.2 s across 14 workers)
>
> ⏳ **REMAINING:** I1 progress-bar UI · I4 telemetry panel · 5d/5e host-split/breaker ·
> 6a–6f weather intelligence · 9-M2/M3/M4/M5/M7 + LOW batch · final schema v7 bump + retrain
> (after which the two contiguity probes and remaining expectedFailures flip).

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

**⚠️ Amendment (second audit): attach-once guard required with resume.**
Once queues survive disconnects, two live generators attached to the same run_id would
*split* events between them (each `queue.get()` goes to one consumer). Add an
`attached: set[str]` registry — a second concurrent SSE connect to the same run receives
HTTP 409 / an error event instead of silently stealing half the progress updates.

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
merged_n = len(field)          # post-phantom-exclusion field from I7b
scraped_n = len(driver_prices)
gap_note  = f" ({merged_n - scraped_n} backfilled)" if merged_n > scraped_n else ""
progress_callback(run_id, "PRICES", "done",
                  f"Prices for {scraped_n} drivers · predicting field of {merged_n}{gap_note}")
```
Also emit the merged names list in `data={"field": sorted(field)}` so the UI log can
expand it. No model changes. *(Field size must be the I7b-validated value, not the raw
union — the earlier "23" illustration predates the phantom-driver fix.)*

---

## ✅ Regression Test Suite — IMPLEMENTED (`engine/tools/tests/test_contracts.py`)

I8e is now partially landed ahead of schedule: a 41-test stdlib-unittest suite guards the
codebase's core contracts, network-free, runs in ~2 s:

```
python -m unittest engine.tools.tests.test_contracts -v
```

| Group | Locks |
|---|---|
| Result classification | finished / lapped (+1..+49 laps) / DSQ / withdrawal table |
| Parse-position | numeric verbatim; non-numeric → per-race order index |
| Season-results integrity | unique driver/round; contiguous positions 1..N; DNFs never carry top-10 positions |
| Name normalization | Antonelli / Zhou / de Vries aliases |
| Circuit matching | full calendar + historical circuits (Portugal/France/Russia/Turkey/Styrian→austria) |
| Sprint calendar | verified fallback tables; schedule-field precedence |
| Chip state | deepcopy isolation; clean reset |
| Bayesian | zeros ≠ DNFs without flags; explicit-flag path sane |
| Elo | roster-scoped pool bounds rank; DNF-vs-DNF pairs skipped (ratings stable) |
| Monte Carlo | seeded reproducibility; empty-stats guard; all-DNF corner crash guard |
| Scoring rules | pole=Q3+pole; P10=Q3-only; P11=Q2-only; FL ineligible >P10; total==Σbreakdown |
| Structural (AST/source) | feature widths 56==56; schema-version key; backtest field-ranks; juliacall gated; watchdog job-guard; pipeline price-recording |
| Planned-fix probes | 7 × `expectedFailure` pins for 7a/7b/8b/9-C3/9-H4/9-M14 |

**Protocol:** probes fail today *by design*; when a fix lands its probe turns green
("unexpected success") → remove the decorator in the same commit.

**Third-audit live discovery:** the integrity tests immediately detected REAL corruption in
cached data — non-contiguous positions in **2025 R11/R16/R21** and **2026 R5/R10**, including
six 2026 R5 DNFs carrying fabricated P5–P10 classifications (Perez, Norris, Russell, Alonso,
Albon, Lindblad). This is 9-C1 confirmed against production caches. Remediation therefore
REQUIRES: land 9-C1 → purge `cache/api/*results*.json` AND `*qualifying*.json` → re-warm →
flip the skipped/expected-failure probes on. The corrupted rounds list above doubles as the
verification checklist.

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

**⚠️ Amendments (second audit):**
4. **Gate the juliacall import behind `USE_JULIA_ENGINE`.** Today lines 32–36 of
   monte_carlo.py *import and include the engine unconditionally* even though dispatch is
   gated. With multiprocessing spawn, EVERY worker process re-imports the module → each pays
   Julia runtime startup (~seconds) when juliacall is installed (it's in requirements.txt).
   Wrap the whole try-import in `if USE_JULIA_ENGINE:` — also removes dead work for everyone.
5. **Windows spawn re-import invariant (verified safe, keep it that way):** spawned children
   re-import the server entry module as `__mp_main__`, so the uvicorn boot under
   `if __name__ == "__main__":` (server.py:648) will NOT start a second server. Acceptance
   test must include running the full parallel pipeline from inside the live server once,
   to lock this invariant against future refactors.
6. **Worker import diet:** the MC worker function must not transitively import TF/predictor
   (monte_carlo imports only config today — preserve that boundary; asserted in final gates).

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

## Improvement 5 — Resilient & Faster Cache Downloading (anti-429)

**Observed (production log):** repeated `[rate limit 429] waiting 5s (attempt 1/4)` during a
full re-download after cache wipe. Three root causes identified:

| # | Root cause | Evidence |
|---|---|---|
| A | **Thundering herd**: all ThreadPool workers that hit a 429 sleep the *same* 5 s, then retry simultaneously → another synchronized burst | 429s repeat every few seconds across the log |
| B | **Backoff floor too low & uncoordinated**: 5 s minimum doesn't clear a server-side token bucket, and there is no shared state — one worker's 429 doesn't slow the others | gaps between 429s ≈ constant |
| C | **Request volume**: ~2 standalone standings calls per race (driver + constructor) × ~114 races ≈ **230 avoidable requests** per full warm, on top of results/quali | endpoint inventory |

### Methods (in priority order)

#### 5a. Adaptive pacer with global cooldown (fixes A + B)
Replace the fixed-gap `_throttle()` for Jolpica with a shared AIMD pacer:
```python
class _Pacer:
    gap = 2.1                 # seconds between requests (all workers share ONE slot)
    cooldown_until = 0.0      # global: EVERY worker waits past this after any 429
    successes_since_drop = 0

    def on_429(self, retry_after: int):
        # Global cooldown: nobody sends until it expires (kills the herd)
        self.cooldown_until = max(self.cooldown_until,
                                  time.time() + max(retry_after, self.gap * 3))
        self.gap = min(10.0, self.gap * 1.5)          # multiplicative increase, capped
        self.successes_since_drop = 0

    def slot(self):
        """Called under lock before every request."""
        self._next_slot = max(self._next_slot, time.time()) + self.gap
        return max(self._next_slot - self.gap, self.cooldown_until)

    def on_success(self):
        self.successes_since_drop += 1
        if self.successes_since_drop >= 25:           # additive decrease when healthy
            self.gap = max(2.1, self.gap * 0.9)
```
Retry sleeps additionally get ±20 % jitter to desynchronize stragglers. Expected effect:
429s drop to near zero; when one occurs, the whole fleet pauses once instead of stampeding.

#### 5b. Eliminate ~230 requests: derive standings from already-cached results
Championship standings are fully derivable from per-race result rows (each row carries the
official `points`, incl. fastest-lap bonus). `get_season_results` already downloads them in
~5 paginated calls per season.
- New `data_fetcher.standings_asof(year, round_num)`:
  1. If the API per-round standings file is already cached → serve it (zero change).
  2. Else accumulate points from cached season results up to `round_num`
     (tie-break: wins desc, then points-desc finish consistency — matches Ergast closely),
     return in the identical dict shape.
  3. Never writes fake API-cache entries; it's a distinct local computation.
- Consumers switched to it: `warm_cache`, `predictor.train()` per-round standings, backtest
  per-round standings. API standings remain authoritative wherever already cached and for
  live/current-season use.
- Net effect: full-warm Jolpica volume drops roughly **from ~700 to ~350 requests**;
  combined with 5a this is the difference between "constant 429s" and "no 429s".

#### 5c. Connection reuse (speed, not rate)
Module-level `requests.Session()` with `HTTPAdapter(pool_maxsize=8)` for Jolpica/OpenF1 —
removes a TCP+TLS handshake per call (~100–300 ms each on Windows). Same request count,
noticeably faster wall-clock, gentler on the remote.
**⚠️ Amendment (second audit):** `requests.Session` is not documented thread-safe; urllib3's
connection pool *is*. Keep ONE session whose adapters own the pool and never mutate session
state (headers set once at construction); alternatively use `threading.local()` sessions if
any mutation is ever needed. Add a warm-up GET at first use so the TLS handshake isn't paid
inside the throttle slot.

#### 5d. Host-aware worker split
Jolpica (rate-limited) and FastF1 (own generous limits, throttled at 3 s) currently share one
ThreadPool. Split into two executors so FastF1's slower downloads don't serialize behind
Jolpica pacing, and vice versa — overall wall-clock improves without raising Jolpica pressure.

#### 5e. Circuit breaker + resumability UX
- Rolling 60 s window: if ≥6 429s occur, enter cooldown mode (pause 5 min, emit progress event
  `"phase": "cooling_down"`), then resume with the raised gap. Prevents hammering into a ban.
- Combined with Improvement 1's manifest: restarting a broken download resumes where it left
  off instead of starting over — the practical answer to "I hit limits mid-download".

#### 5f. Observability hooks (feeds Improvements 1 & 4)
Progress payload gains `{throttle_gap_s, cooldown_remaining_s, http_429_count}`; telemetry
panel shows a live "API health" row. Users SEE the pacer adapting instead of staring at raw
429 prints.

### Files touched
`engine/core/data_fetcher.py` (pacer, Session, standings_asof), `engine/core/warm_cache.py`
(split executors, breaker), `engine/models/predictor.py` + `engine/analysis/backtest.py`
(use `standings_asof`), small SSE/UI additions shared with I1/I4.

### Risks & mitigations
- Derived standings may differ from API on exotic tie-breaks → only used as a *fallback*
  when the API file isn't cached; live/current-season paths keep API precedence.
- Pacer adds latency when healthy (2.1 s floor unchanged) → no regression vs today.
- AIMD oscillation → capped at 10 s, floored at 2.1 s, decay only after 25 clean successes.

---

## Improvement 6 — Weather Intelligence (audit-driven)

### Audit findings (verified against code, Aug 2026)
| # | Finding | Where | Severity |
|---|---|---|---|
| W1 | Hourly race-day slice hardcodes `12 <= hour <= 18` venue-local → **empty/wrong window for night races** (Vegas 22:00, Singapore 20:30, Middle-East evenings) | weather.py:465 | 🔴 |
| W2 | Forecast request not anchored to race weekend (fetches today→+16d); >16-day races silently degrade to climatology | weather.py:60–74, 406–409 | 🟡 |
| W3 | Training weather labels are a 17-entry hardcoded list (`_KNOWN_WET_RACES`) covering ~26 % of train races incl. dead 2019/2020 entries; all other races train as `"dry"` | predictor.py:96 | 🔴 |
| W4 | Inference emits `"overcast"` (from live WMO codes) — an encoding **training never sees** | predictor vs weather | 🟠 |
| W5 | Tire-degradation model called with hardcoded `track_temp=35.0` although the forecast carries real temps and the model accepts them | predictor.py:702 | 🔴 |
| W6 | Race-day PoP % / precip mm present in payload but never used; rain_risk is a 3-bucket count of wet *sessions* | weather.py:411–420 | 🠔 |
| W7 | LSTM context `rain_enc` fed at inference but trained as constant 0.0 | temporal_model | 🟠 |

Verdict: **location correct; date correct within 16 days; factor-into-prediction currently
superficial** — encodings exist but are trained on sparse synthetic labels, and the strongest
physical signal (temperature → tire degradation) is discarded.

### Fixes

#### 6a. Race-hour-aware hourly slice (W1)
Parse the schedule's race `time` (e.g. `"15:00:00Z"`), convert UTC → venue-local via the
Open-Meteo echoed timezone, slice `[start−3h, start+3h]`; fall back to 12–18 when no time
is published. Removes empty hourly payloads for night races.

#### 6b. Anchored forecast request + quality flag (W2)
Pass explicit `start_date=today, end_date=race_date+1` (capped at 16 d) so the payload is
manifestly the race window. Payload gains `"quality": "live_forecast" | "climatology"` and
`"forecast_horizon_days"` — surfaced in telemetry/UI so degraded forecasts are visible.

#### 6c. Real training labels from the immutable archive (W3 + W4)
One-time backfill script (`engine/tools/backfill_weather_labels.py`): for every
`(season, round)` in training seasons, fetch archive-api daily data for the race date at the
venue coords (~70 calls, permanent cache `cache/weather/training_labels.json`), map through
the SAME `_wmo_to_condition` used at inference → conditions include overcast/mixed/wet/dry
with realistic frequencies. `_KNOWN_WET_RACES` stays as a manual override layer. Eliminates
both the sparsity and the encoding-mismatch problems; also deletes the dead 2019/2020 entries.

#### 6d. Temperature-aware tire degradation (W5)
In `_build_features`, replace `track_temp=35.0` with the forecast race-day temp:
`track_temp ≈ 0.65·temp_max + 0.35·temp_min − 3` (air→track offset), falling back to 35.0
under climatology. This feeds the existing Keras input — no model change, just honest input.

#### 6e. Continuous rain probability (W6)
New derived value `rain_prob ∈ [0,1] = clamp(0.6·PoP_race + 0.25·(precip_mm>0) + 0.15·wet_session_fraction)`
emitted in the weather payload; replaces the bucketed `rain_enc` in features, DNF modifier,
and MC noise scaling (noise magnitude ∝ rain_prob instead of 3 tiers).

**⚠️ Amendments (second audit):**
- **Keep the legacy `rain_risk` bucket field in the payload.** The UI reads
  `weather.rain_risk` directly (app.js:347, 356) for the HUD badge — removing/renaming it
  breaks the display. rain_prob is *additive* to the payload; only the ML feature column is
  replaced.
- **Feature count invariant:** `rain_prob` must REPLACE the `rain_enc` feature column
  (not append) — `sanity_check.py:9` asserts `N_FEATURES == 56`, and keeping that assertion
  green is an intentional regression guard against accidental schema drift.
- **Coordinated single retrain:** 5b (derived standings), 6c–6f (weather labels, temp,
  rain_prob, LSTM align) and I3's device-key all alter training data or the cache key.
  Bundle them into ONE `MODEL_SCHEMA_VERSION = "7"` bump at the end of the rollout —
  otherwise users pay multiple ~4-min retrains. The plan's execution order is sequenced so
  v7 lands exactly once, after all data-affecting steps are merged.

#### 6f. LSTM rain-context alignment (W7)
Pass the same `rain_prob`-derived encoding in the temporal context vector; requires schema
bump → **MODEL_SCHEMA_VERSION v7**, one retrain (~3–4 min post-I3) plus deletion of stale
temporal/tire caches.

#### Deferred (documented non-goals)
Wind-direction/humidity features (low expected signal, adds dimensions); elevation-adjusted
air density; in-race dynamic weather.

### Files touched
`engine/core/weather.py` (6a/6b), new `engine/tools/backfill_weather_labels.py` (6c),
`engine/models/predictor.py` (6c consumption + 6d + 6f), `engine/models/monte_carlo.py`
(6e noise scaling), `engine/serving/pipeline.py` (payload passthrough).

### Risks & mitigations
- Archive API gaps for some venues/years → label falls back to `"dry"` + `quality:"unlabelled"`
  flag rather than fabricating.
- v7 retrain invalidates cached models once — acceptable, scheduled with I3 rollout.
- PoP-based rain_prob shifts EV distributions modestly; before/after comparison captured in
  the standard backtest spot-round gate.

---

## 🔴 Improvement 7 — Prediction Integrity Hotfix (PRIORITY 0)

**Incident:** finishing order contained **23 drivers** (field limit 22); ordering nonsensical
(Bearman/Hulkenberg P1–P2, top performers outside top 10). Two independent regressions,
**both introduced by recent changes**, interacting with each other:

### Root cause A — meta-learner representation mismatch (CRITICAL)
The ensemble-rank refactor changed *inference* to feed the Ridge meta-model
`[rank_rf, rank_xgb, rank_lgb]` (per-field integer ranks 1..N), but `_train_meta_learner`
still builds its out-of-fold training matrix as
`[raw_rf_prediction, 25.0 − xgb_margin, 25.0 − lgb_margin]`.
The Ridge coefficients were learned on margin-scale inputs and are applied to rank-scale
inputs → the blend output is statistically meaningless → arbitrary ordering. Verified by
code inspection: `oof_preds[...,1] = 25.0 - xgb_clone.predict(...)` vs inference
`vals = [r_rf[i], r_xgb[i], r_lgb[i]]`. Schema bumps v4→v6 did NOT fix this because they
invalidate the tree models, while the meta model was *retrained on the same wrong
representation* each time.

### Root cause B — phantom driver via blind roster union
Pipeline merge `{scrape ∪ api_standings}` produced a 23-driver field: Jolpica season
standings retain drivers who left the seat mid-season (phantom), and the merge adds them as
prediction subjects. Knock-ons: ranks clamp at 22 with ties, `RACE_POSITION_POINTS[>22]=0`,
MC grid collisions, optimizer sees a non-existent asset. Compounded by an earlier mistake:
the static 2026 config seed was "synced" to the 23-entry standings view instead of staying
a curated 22-seat table.

### Fixes

#### 7a. Unify the meta-learner representation (root cause A)
- Extract a shared module-level `to_rank(scores)` helper (already exists inside
  `_ensemble_rank_predictions`; hoist it).
- `_train_meta_learner`: per fold, convert OOF predictions of ALL THREE models to per-race
  ranks via `to_rank` before stacking — meta trains on exactly what inference feeds.
- Backtest race path already feeds ranks ✓ consistent after this change.
- No user-visible behavior until retrain → bundled into the single schema v7 bump.

#### 7b. Authoritative field list with integrity validation (root cause B)
Precedence rule, replacing the blind union:
1. **Field authority = fantasy game entry list** (the scrape defines who can score points).
2. Gaps are backfilled ONLY from the curated static seed (`DRIVER_TEAMS_2026`, restored to
   exactly 22 curated seats), never from raw standings rows.
3. Standings are used solely for TEAM attribution of drivers already accepted by 1–2.
4. Anyone present in standings but absent from (scrape ∪ seed) is logged as
   `"excluded phantom: <name>"` and dropped — this is the Tsunoda-class case.
5. Hard validation gates in pipeline + `predict_finishing_order()`:
   - `18 <= len(field) <= 24` else raise (fail loudly, no predictions);
   - warn when `!= 22`;
   - assert unique driver names;
   - emit new SSE event `FIELD_VALIDATION {size, backfilled:[], excluded_phantoms:[]}`.
6. Config seed hygiene: restore curated-22 seed; any future "sync from API" must go through
   rule 4's filter, never wholesale copy.

#### 7c. Downstream hardening (defense in depth)
- Monte Carlo: assert `len(set(effective_grid.values())) == field_size` before simulating.
- `estimate_fantasy_points`: explicit guard/error if `race_pos > 22` or `grid_pos > 22`.
- Final-gate addition: prediction run must produce `len(race_order) == len(field)` and all
  ranks contiguous 1..N.

#### 7d. Immediate remediation sequence (executed with the fix)
1. Land 7a+7b+7c → bump `MODEL_SCHEMA_VERSION = "7"` (was already earmarked; this adds
   root-cause A as mandatory content of that bump).
2. `Remove-Item cache\models\*.pkl` (force clean retrain).
3. Run one analysis; verify: 22 drivers, contiguous ranks, plausible top-5 vs standings,
   backtest spot-round MAE within historical band.
4. Until deployed: treat current predictions as invalid.

### Files touched
`engine/models/predictor.py` (7a, 7b gates, 7c), `engine/serving/pipeline.py` (7b merge
rewrite + FIELD_VALIDATION event), `engine/core/config.py` (curated seed restore),
`engine/models/monte_carlo.py` (7c assertion), `engine/tools/sanity_check.py`
(new field-integrity section).

### Risk
Low-to-moderate: 7a changes meta inputs → full retrain required anyway under v7 (no extra
cost). 7b changes pipeline data flow — covered by FIELD_VALIDATION logging during the
verification gate. Residual ambiguity: which specific driver is the 23rd phantom requires
one manual confirmation against the real 2026 entry list during implementation.

---

## 🔴 Improvement 8 — Cross-Cutting Contract Integrity (incident-class sweep)

**Rationale:** the Improvement-7 incident was not a one-off bug — it was a *class*: silent
cross-module contract mismatches and fabricated data that produce plausible-looking garbage
with zero errors. A full-codebase sweep for that CLASS found four more live instances:

### Findings (verified by code inspection + AST analysis)

| # | Finding | Evidence | Severity |
|---|---|---|---|
| X1 | **Feature name↔value binding is POSITIONAL ONLY**. `FEATURE_NAMES` (56 entries) and the returned `np.array` (56 values, AST-counted) are aligned today, but nothing binds them: reordering one without the other trains/predicts with permuted feature semantics — **no error, wrong model** | predictor.py; no `FEATURE_NAMES.index`/dict-zip mechanism exists | 🔴 |
| X2 | **Silent fallback prices fabricate money data.** Unmatched driver → `$5.0M`, constructor → `$8.0M` (`_get_driver_price/_get_ctor_price`) — silently feeds budget checks and the LP solver. Combined with `_fuzzy_match`'s bidirectional *substring* containment, a near-miss name resolves to the WRONG player's price or an invented one | fantasy_optimizer.py:520–531 | 🔴 |
| X3 | **47 remaining `except Exception: pass/continue`** across 16 files (data_fetcher ×12, scraper ×7, cli ×4, server ×4, self_improvement ×4 …) — the same masking pattern that hid root cause A until it exploded | repo-wide regex audit | 🟠 |
| X4 | **Meta rank-input is field-size dependent.** Meta trains on ranks from historical fields (~20 rows/race) but serves 22–23-driver fields; Ridge inputs shift distribution slightly. Not garbage (monotonic), but avoidable noise | consequence of 7a design | 🟡 |

### Fixes

#### 8a. Dict-bound feature construction + startup assertion (X1)
Rebuild `_build_features` to assemble an ordered dict `{name: value}` and return
`np.array([feat[n] for n in FEATURE_NAMES])`. Add a module-level self-check (runs once at
import, ~free):
```python
assert set(feat.keys()) == set(FEATURE_NAMES), "feature drift"
```
Extend `sanity_check` with a canary vector test: build features for two synthetic drivers
differing in EXACTLY ONE known feature (e.g. dnf_rate) and assert only that column moves.
Reorder/rename on either side now fails loudly at import/test time instead of poisoning
training. Feature count stays 56 (S1 invariant intact).

#### 8b. Honest pricing: no fabricated numbers (X2)
- `_get_driver_price/_get_ctor_price`: on match failure raise/log-and-skip the candidate
  (candidate excluded from optimizer with a visible reason), NEVER invent a price.
- Tighten `_fuzzy_match`: exact-normalized match first; then token-based match requiring the
  FULL surname token equality (mirroring post_race_check fix); ambiguous matches (≥2 keys
  tie) are rejected, not first-picked; every non-exact resolution logs
  `"fuzzy: 'X' -> 'Y' (score)"`.
- Backfilled drivers without real prices are labelled `"price_source": "estimate"` end-to-end
  so UI/telemetry shows estimate vs live.

#### 8c. Silent-swallow elimination sweep (X3)
Convert all 47 sites to `logger.warning(..., exc_info=True)` minimum; keep `continue`
control-flow where intentional but always log. Mechanical change per file; each file gets its
logger if missing. Priority order: data_fetcher, fantasy_scraper, server, cli,
self_improvement, then the rest. Acceptance: `grep -rE "except Exception:\s*\n\s*(pass|continue)"`
returns 0 hits in engine/.

#### 8d. Field-size-invariant rank encoding (X4)
In the shared `to_rank()` helper (from 7a), emit normalized ranks `rank / (N + 1)` instead of
raw integers — identical transformation applied in `_train_meta_learner` folds, inference, and
backtest. Makes meta input distribution invariant to field size; costs nothing; rides the
same v7 bump.

#### 8e. Golden contract tests (regression net for this entire class)
New `engine/tools/tests/test_contracts.py`:
1. **Micro end-to-end**: synthetic 3-season mini-dataset → train → predict → assert
   field size == input roster, ranks contiguous, better-form drivers rank better on average.
2. **Meta representation equality**: structural assert that `_train_meta_learner` OOF path and
   `_ensemble_rank_predictions` both route through the SAME `to_rank` helper (source-level
   check or shared-function monkeypatch counting calls).
3. **Price honesty**: optimizer given an unknown driver asserts exclusion-with-log, not $5M.
4. **Weather payload keys**: `rain_risk` present (UI contract) alongside rain_prob.
These run in seconds (tiny data), no network, and would have caught BOTH incident regressions.

### Files touched
`engine/models/predictor.py` (8a, 8d), `engine/strategy/fantasy_optimizer.py` (8b),
16 files for the swallow sweep (8c), new `engine/tools/tests/test_contracts.py` (8e),
`engine/tools/sanity_check.py` (canary test hook).

### Risk
Low: 8a/8d ride the already-planned v7 retrain; 8b changes optimizer candidate sets only when
data was previously fabricated (strictly more honest); 8c is logging-only. 8e adds a fast
test file with zero runtime impact.

---

## 🔴 Improvement 9 — Full-Spectrum Bug Sweep (third audit round, ~35 verified findings)

Four parallel line-by-line audits of every remaining module. Findings are NEW (none overlap
I1–I8). Several directly explain production symptoms already observed. Ordered by severity;
fixes are precise and scoped.

### CRITICAL

| ID | Bug | Evidence / failure | Fix |
|---|---|---|---|
| **9-C1** | **Pagination splits races across pages; per-page `enumerate` resets** → DNF positions corrupted for any boundary-straddling race. Jolpica paginates result ROWS not races; 2026 has 22-car grids so page_size=100 is *never* aligned (verified live: 2026 R5 rows straddle offset=100). Corrupts CURRENT-season form/constructor stats/ELO/temporal data with full season weight | data_fetcher.py:803–851; live API probe showed Miami-2023 sliced mid-list | Paginate **per round** via schedule × `/{year}/{round}/results.json`, or carry a running per-round counter across pages + `(round, driver_id)` dedupe |
| **9-C2** | **Post-race validation runs against the WRONG SEASON**: prediction files save `"race"` without a `season` key → `.get("season", CURRENT_SEASON)` always yields 2026; `validate_round()` accepts `--season` but silently drops it. A 2025 file compared today corrupts the accuracy log entry AND the bias-correction feedback loop | post_race_check.py:121,129,284–298; pipeline.py:292 | Save `report["season"]` explicitly; thread `season` param through `validate_specific_prediction`; log when file-season ≠ requested-season |
| **9-C3** | **2026 calendar drift**: live R16 = *"Bahrain Grand Prix **in Malaysia**"* (Sepang). Config has no Sepang → loose matcher returns Sakhir config + **Sakhir weather forecast (desert lat/lon)** for a humid Malaysian round; Singapore→Abu Dhabi all off-by-one (config says 24 races, calendar has 23 + Malaysia) | config.py:186,279–326; `_match_circuit_config` substring pass verified against live schedule | Add Sepang entry under the exact API name (key `malaysia`, true coords), renumber R17–R23, move Sakhir to historical block (`round: 0`) |
| **9-C4** | **Merged-roster substitutes reach the LP optimizer with a fabricated $5M price**, evade the 2-assets-per-team cap (counted as their own team via `DRIVER_TEAMS_2025.get(drv, drv)`), are omitted from lookahead constructor EV but included in deterministic totals, and `race_order[:20]` truncation drops a real driver from saved reports | pipeline.py:123–146,387–403; fantasy_optimizer.py:288,520–556 | Fold into **I7b**: merged-only drivers either excluded from optimizer inputs or given synthesized price/team entries; LP/cap lookups use `dynamic_roster`; remove `[:20]` truncation |

### HIGH

| ID | Bug | Failure | Fix |
|---|---|---|---|
| **9-H1** | **Playwright scrape validates names against `DRIVER_TEAMS_2025`** — Lindblad/Perez/Bottas rejected by the primary path. *This is why your scrape returns 20 drivers* | fantasy_scraper.py:90–91,135 | Validate against `DRIVER_TEAMS_2026 ∪ get_season_roster()`; log dropped-price lines |
| **9-H2** | **No mutual exclusion `/api/cache/rebuild` ↔ `/api/run`** — exactly the concurrent click seen in your logs: pipeline trains while its cache dirs are deleted underneath it; Windows rmtree silently skips open files → mixed-vintage features; two rebuild clicks collide on literal run_id `"cache_rebuild"` (first stream orphaned forever) | server.py:215–284,468–469 | Shared busy gate both directions; unique uuid run_id per rebuild |
| **9-H3** | **Temporal LSTM: `rain_enc` and `ctor_elo_z` trained as constants 0.0 but fed REAL values at inference** (the comment claiming parity is false). Untrained sensitivity axes shift momentum output arbitrarily per wet forecast / Elo z-score | temporal_model.py:274–291 vs predictor.py:657–670 | Short-term: predictor feeds 0.0 for both (true parity); long-term folded into 6c archive-label work. Extend 6f |
| **9-H4** | **`_match_circuit_config("São Paulo Grand Prix")` returns `{}`** for every 2021–2025 Brazilian GP (no alias, no city fallback unlike weather's matcher) → empty circuit features in all historical/backtest paths | data_fetcher.py:486–506 | Add `"São Paulo Grand Prix"` alias → interlagos/brazil |

### MEDIUM

| ID | Bug | Fix |
|---|---|---|
| **9-M1** | MC weather-variance "delta" adds `abs(extra)` Gaussian noise ON TOP of full-σ base draw → low-variability circuits get MORE chaos than baseline (inverted purpose); variance math wrong (√(σ²+(f−1)²σ²) ≠ fσ) | Draw once: `sigma_eff = weather_noise * factor` after computing factor |
| **9-M2** | Lazy singletons (bayesian/temporal/tire/elo) have no init lock → concurrent first-calls run DUPLICATE TF trainings (minutes, possible OOM) and non-atomic `.keras`/pickle saves | Double-checked locking + tmp-file atomic saves |
| **9-M3** | Glicko-2: no RD inflation for drivers MISSING rating periods (injury/substitute gaps keep veteran confidence indefinitely), violating Glickman §missing-periods | Track last-period index; apply φ←√(φ²+c²) for absent drivers each period |
| **9-M4** | Tire model TRAINED path ignores `tire_degradation`/`sm_zones`/compound-delta (dead vars; only altitude multiplier applied) → Bahrain(deg 5) and Silverstone(deg 2) return identical deg_per_lap | Apply physics multiplier symmetric with fallback path, or add scalars as NN inputs (v7 retrain covers cost) |
| **9-M5** | CLI records price history BEFORE `--prices-file` overrides and manual corrections → estimate-vs-real snapshots create phantom ±$2M+ "movements" feeding sell-high/buy-low banners | Move recording after all correction points |
| **9-M6** | Dashboard weather strip reads nonexistent keys (`race_day_temp_c`, `temp_c`, wind variants) → temp/wind render `?°C`/`? km/h` EVERY run | Read `weather["sessions"]["Race"]["temp_day_c"/"wind_speed_kph"]` like pipeline does |
| **9-M7** | Weather session offsets assume classic 3-day format — sprint weekends get Thursday FP forecasts, shifted quali, and NO Sprint session forecast despite sprint_date being available in the schedule dict | Build sessions from actual fp1/quali/sprint dates passed in race dict |
| **9-M8** | Permanent caching freezes IN-PROGRESS race results: empty-response guard only catches pre-race; a lap-30 running order (total>0) is cached forever for the current season | TTL 3h for current-season results until race_date+1d passes, then promote to permanent |
| **9-M9** | Backtest form off-by-one: standings snapshot includes round rnd−1 but `compute_driver_form(until_round=rnd−1)` EXCLUDES it (exclusive bound) → form lags standings by one race inside the same feature vector | Call `_compute_form_asof(year, rnd)` |
| **9-M10** | Dream-team turbo doubling exists only in terminal table; saved .txt and HTML dashboard show un-doubled rows → TOTAL ≠ sum(parts), turbo assignment (the actionable part) unmarked | Mark + double turbo row in both artifacts via `optimal["turbo_driver"]` |
| **9-M11** | One failed fetch aborts entire multi-year backtest (no try/except → hours lost, CSV never written); empty quali silently fabricates P11 baselines producing fake quali MAE | Per-race try/except → "No Data" row; None quali metrics excluded from summary |
| **9-M12** | Manual grid overrides permanently outrank ACTUAL qualifying (applied after `_actual_grid`, labeled LOCKED, never auto-cleared client-side) — Saturday's drag experiment pins Sunday's reality | Overrides only fill drivers missing from actual grid; UI auto-clears on COMPLETE |
| **9-M13** | CLI `--prices-file` writes `"ownership"` key; every consumer reads `"ownership_pct"` → documented override feature is a silent no-op | Accept both keys, store canonical; fix help text |
| **9-M14** | `get_season_roster` accepts 10–15-entry rosters after only a warning → partial API responses become THE lineup | Return `{}` below realistic seat count (20/22 season-aware) |

### LOW (batched fixes)
`--report-only`/`--race` flags parsed but never implemented · `--auto` first-run falls into interactive prompts · `_medal` glyphs stripped to empty strings · dashboard constructor tags look up nonexistent `team` key (always grey) · tire strategy head labels use hindsight full-race sequences + `remaining_laps` outside trained range · Bayesian ψ/μ branches disagree on dnf_flags length mismatch · `upside_pct` tautologically ≈25% · grid sentinel `0` flows into LSTM sequences/scaler ranges · MC missing-quali driver defaults to P10 + collects Q3 bonus every sim · DNF finishing-order iterates a set (PYTHONHASHSEED nondeterminism vs seed contract) · transfers input not restored from saved team · `activeOverrides` accumulates/stale-pins market prices against refresh · poll-fallback spins 10 min after an instant failure (error swallowed server-side: `None` stored) · `analyse_results.py` hardcodes CSV name backtester no longer produces · fallback price table half-migrated (Tsunoda missing, Hadjar mis-team) · warm_cache double-warms current season + fires DONE twice · hourly summary coerces genuine 0°C→20°C via truthiness.

---

## Execution Order & Risk

| Step | Depends on | Risk | Effort |
|---|---|---|---|
| **7. PREDICTION HOTFIX** | none — **do first** | medium | 2–3 h |
| **9-C1/C2. pagination + season-integrity** | none — corrupts live data NOW | medium | 1–2 h |
| **9-C3. Sepang/calendar drift** | none | low | 30 min |
| **9-H1..H4 (scrape roster, rebuild gate, LSTM parity, São Paulo)** | 9-H1 folds into 7b | low–med | 2 h |
| **8e. golden contract tests** | 7a | none | 1–2 h |
| **8a/8d. dict-bound features + normalized ranks** | v7 bundle | low | 2 h |
| **8b. honest pricing** (+ 9-C4 optimizer scope) | none | low | 1 h |
| 2. count fix | none | none | 15 min |
| **5a/5c. pacer + Session** (+ 9-M8 TTL) | none | low | 1–2 h |
| **5b. derived standings** (+ fixes 9-M9 off-by-one) | none | medium | 2 h |
| **5d/5e. host split + breaker** (+ 9-H2 mutual exclusion) | 5a | low | 1–2 h |
| **9-M1..M14 + LOW batch** | grouped by module | low each | 4–6 h total |
| 4. telemetry payloads + panel | 5f hooks ideal but optional | low | 3–4 h |
| 1. download progress bar | SSE lifecycle fix | low | 2 h |
| 3a/3b. hardware detect + XGB CUDA | none | medium | 1–2 h |
| 3c. MC multiprocessing (+ 9-M1 noise fix, L-findings) | 3a | medium | 2–3 h |
| 3d. concurrent prelude | 5d | low | 1 h |
| **6a–6f weather intelligence** (+ absorbs 9-H3 long-term, 9-M7) | 6c, 7a, 8a/8d | medium | 2 h |
| Final gates | all | — | compile, sanity_check + contract tests, live pipeline, backtest spot-round, timings |

Total estimate: ~3 focused sessions. Rollback = revert commits.
Recommended sequence: **7 → 9-criticals → 8e → 2+5a+5c+1 → rest**, with schema v7 landing once
after 7a + 8a/8d + 5b + 6c–6f are all merged (single retrain).

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
| H1 | juliacall imported (and Julia started) in every MC spawn worker despite engine being disabled | Import gated behind USE_JULIA_ENGINE (verified: currently unconditional at monte_carlo.py:32-36) |
| H2 | Spawned child re-boots the uvicorn server on Windows `-m` launch | Verified safe: boot is under `if __name__ == "__main__"` and children re-import as `__mp_main__`; acceptance test locks this invariant |
| W1b | Renaming/removing `rain_risk` payload field breaks UI HUD badge (app.js:347,356) | rain_prob is additive to payload; only the ML feature column changes |
| S1 | sanity_check `N_FEATURES == 56` assertion trips after weather rework | rain_prob REPLACES rain_enc column — count stays 56 by design; assertion kept as drift guard |
| R2 | Multiple data-affecting steps each forcing separate ~4-min retrains | All bundled into ONE schema v7 bump at rollout end (5b + 6c-6f + I3 device key) |
| X1b | Feature name/value positional drift (silent mislabeling — the incident class) | Dict-bound construction + import-time key assertion + canary vector test (8a) |
| X2b | Fabricated $5M/$8M prices silently entering budget/LP math; substring fuzzy match resolving wrong player | Match failure = exclude-with-log; full-token matching; ambiguity rejection; estimate tagging (8b) |
| X3b | 47 silent `except: pass/continue` sites masking failures (the pattern that hid root cause A) | Repo-wide sweep to logged warnings; acceptance grep enforces zero (8c) |
| X4b | Meta rank inputs shift with field size (20 historical vs 22-23 live) | Normalized ranks rank/(N+1) in train+inference+backtest via shared helper (8d) |
| E1 | Incident-class regressions recur unnoticed | **IMPLEMENTED**: 41-test contract suite (`engine/tools/tests/test_contracts.py`, unittest, 2 s, network-free) — already detected live 9-C1 corruption in cached data; remaining golden tests (micro e2e, meta-representation equality) land with their fixes as expectedFailure probes |
| P1b | 2026 22-car grids never page-align (100 % 5×20) → DNF positions corrupt for every boundary race of the CURRENT season | Per-round pagination or cross-page running counter + dedupe (9-C1) — verified live against API |
| P2b | Cross-season prediction files validated against wrong season; accuracy log + bias loop poisoned | Explicit season saved at report time; season threaded through validators (9-C2) |
| P3b | R16 Sepang gets Sakhir config + desert weather forecast; calendar rounds drift | Exact-name entry + renumbering (9-C3) |
| S2 | Rebuild deletes cache dirs under a running pipeline (observed in production logs) | Mutual-exclusion gate both directions + unique rebuild run_ids (9-H2) |
| S3 | Scrape roster validated against last year's teams silently drops current drivers | Validate against current-season union (9-H1) |
| T2 | Duplicate TF trainings on concurrent singleton first-calls | Double-checked locking + atomic model saves (9-M2) |

**Invariants preserved:** model-cache schema semantics (device now part of key); deterministic
backtest results (backtester doesn't call Monte Carlo); `F1 Fantasy.bat` / CLI entry contracts unchanged; no DB/migrations; all UI additions additive;
weather payload remains backward-compatible (`rain_risk` buckets retained alongside rain_prob);
feature count stays 56 (replacement, not addition).
**One intentional behavior change, flagged:** pipeline default sims 1000 → 2500 *after* parallel
MC lands (net-faster than today's serial 1000). Second flagged change: optimizer now *excludes*
unpriceable candidates instead of inventing prices (strictly more honest; may reduce candidate
pool when the scrape is degraded).
