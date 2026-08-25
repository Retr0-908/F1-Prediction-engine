# Post-Implementation Report

**Date:** 2026-08-25/26
**Scope:** Complete rollout of the 9-improvement plan from `Docs/IMPROVEMENTS_PLAN.md`
**Commits:** `11b57d5` → `0f65d47` → `216e53d` → `9929abd` → `5b67da5` → `ebc54e6` (and earlier)
**Final verification:** ALL 12 GATES PASSED · 41 contract tests OK

---

## What Was Done

### Phase 1 — Prediction Integrity Hotfix (Priority 0)

**Root cause A — meta-learner representation mismatch.**
The ensemble-rank refactor changed inference to feed per-field integer ranks to the Ridge
meta-model, but `_train_meta_learner` still built OOF features as `[raw_rf, 25−margin, 25−margin]`.
Ridge coefficients learned on one scale were applied to a different scale → arbitrary orderings.

**Fix:** Hoisted `to_rank()` to module level with normalized output (`rank/(N+1)`), applied it
identically in `_train_meta_learner` folds, `_ensemble_rank_predictions`, and backtest.

**Root cause B — phantom driver via blind roster union.**
The pipeline merged `{scrape ∪ api_standings}`, importing a phantom 23rd driver from season
standings churn. Compounded by the static config seed having been synced to that same 23-driver view.

**Fix:** Authoritative field precedence (fantasy scrape > curated seed > standings for team
attribution only). Phantom drivers logged and excluded. Hard 18–24 size gate raises on failure.
`FIELD_VALIDATION` SSE event emitted. Config seed restored to exactly 22 seats / 2 per team.

**Verified:** Field = exactly 22 contiguous ranks, unique drivers, plausible top-5 ordering.

### Phase 2 — Critical Data Bugs

| Bug | Fix | Verified |
|---|---|---|
| **Pagination splits races** — Jolpica paginates result ROWS not races; 22-car grids never align with page_size=100 | Running per-round counter across pages + `(round,driverId)` dedupe | All seasons re-warmed; 2,542 rows contiguous |
| **Post-race validation wrong season** | Season saved explicitly in reports; threaded through validators; mismatch refuses | Unit-tested path |
| **Sepang calendar drift** — R16 was "Bahrain Grand Prix in Malaysia" resolving to Sakhir config + desert weather | Exact-name entry added under key `malaysia`; R17–R23 renumbered; SC/VSC probs | Matcher test green |
| **Scrape drops new drivers** — validated against DRIVER_TEAMS_2025 | Validates against current-season union | Perez/Bottas/Lindblad included |

### Phase 3 — Contract Integrity (Improvement 8)

- **8a Dict-bound features:** `_build_features` now constructs an ordered `{name: value}` dict,
  returns `np.array([feat[n] for n in FEATURE_NAMES])`. Runtime guard raises on any missing/extra
  keys. Positional mislabeling structurally impossible.
- **8b Honest pricing:** Unmatched candidates logged+excluded (never fabricated $5M). Fuzzy match
  tightened to full-surname tokens with ambiguity rejection.
- **8c Silent-swallow sweep:** 46/47 sites converted to `logger.warning(exc_info=True)`.
  Acceptance grep enforces zero bare swallows.
- **8d Normalized ranks:** `rank/(N+1)` field-size-invariant encoding.
- **8e Golden contract tests:** 41-test unittest suite (`engine/tools/tests/test_contracts.py`),
  network-free, ~2s. Covers classification table, results integrity, name aliases, circuit
  matching, sprint calendar, chip-state isolation, Bayesian zeros-parity, Elo contracts,
  MC reproducibility, scoring rules, structural AST guards. Probe protocol documented.

### Phase 4 — Performance & Hardware (Improvement 3)

- **`hardware.py`:** stdlib capability profile. Detection ladder: cpu_count → nvidia-smi probe →
  ground-truth CUDA micro-fit. Env overrides `F1E_XGB_DEVICE/MC_WORKERS/IO_WORKERS`.
- **XGBoost CUDA verified live:** Both rankers use `tree_method="hist", device=_dev`.
  Device embedded in model-cache key provenance.
- **Parallel Monte Carlo:** Chunked ProcessPoolExecutor (spawn-safe module-level worker).
  Deterministic per-chunk seeds. Serial fallback. **5,000 sims ≈ 3.2 s across 14 workers.**
- Full CUDA retrain validated end-to-end: field=22, plausible top-5, contiguous ranks.

### Phase 5 — Rate-Limit Relief (Improvement 5 partial)

- **5a AIMD pacer:** shared request slot, global 429 cooldown (whole fleet pauses),
  multiplicative gap increase capped 10s, decay after 25 clean successes, jittered retries.
- **5c Pooled Session:** headers immutable, adapter pool_maxsize=8.
- **5b Derived standings:** `standings_asof(year, round_num)` serves cached API or derives
  from season results (~230 fewer requests per full warm).

### Phase 6 — Weather Intelligence (Improvement 6)

- **6a Race-hour-aware slice:** parses race time UTC → venue-local, slices ±3h around start.
  Night races no longer produce empty hourly payloads.
- **6b Quality flag:** payload gains `"quality": "live_forecast" | "climatology"`.
- **6c Archive label backfill:** ~113 real conditions fetched from Open-Meteo archive into
  permanent `training_labels.json`. Replaces the 17-entry hardcoded list. Labels consumed by
  predictor training AND temporal_model rain_enc (now varies in training AND inference).
- **6d Temperature→tire:** forecast race-day air temp → track temp estimate fed to tire model
  (was hardcoded 35 °C).
- **6e Rain probability feature:** continuous `rain_prob ∈ [0,1]` replaces binary `rain_enc`
  column (count stays 56). DNF modifier uses rain_prob directly. Legacy `rain_risk` bucket
  retained for UI compatibility.
- **9-M7 Sprint offsets:** session forecasts use real Fri/Sat/Sun layout on sprint weekends;
  Sprint day surfaced explicitly.

### Phase 7 — Progress Bar & Telemetry (Improvements 1 & 4)

- **I1 Download bar:** manifest-based unit counting (~1000 units), thread-safe ProgressState
  with rate/ETA, SSE DOWNLOADING events carry dict payloads, UI `<progress>` bar injected into
  Settings screen showing done/total + failures + ETA.
- **I4 Telemetry panel:** Analysis screen gains "Engine Telemetry" grid auto-populated from
  any arriving payload keys. Log lines gain timestamps + collapsible JSON payloads.
  ML_MODEL done event carries device/dataset_samples/skipped_races.

### Phase 8 — Remaining Medium/Low Fixes

| ID | Fix |
|---|---|
| 9-M1 | MC weather variance drawn once at effective sigma (inverted math fixed) |
| 9-M2 | Singleton double-checked locks + atomic tmp+os.replace saves |
| 9-M3 | Glicko missed-period RD inflation via `_inflate_absent()` |
| 9-M4 | Tire trained-path physics multiplier (Bahrain deg5 ≠ Silverstone deg2) |
| 9-M5 | Price recording moved after all corrections |
| 9-M7 | Sprint weather layout via race_info param |
| 9-M11 | Backtest per-race try/except → error row, never aborts |
| 9-M12 | Manual grid overrides fill gaps only pre-quali |
| LOW batch | analyse_results argv/glob/baseline · hourly 0°C coercion · grid sentinel · DNF determinism · --report-only/--race implemented · --auto fail-fast · _medal glyphs · warm_cache dedupe · transfers input restore |

### Phase 9 — Infrastructure (from reorganization)

- `engine/` package with `paths.py` anchoring all filesystem locations
- 128 imports rewritten to qualified names
- Server logging handler configured (warnings persist)
- Elo singleton invalidates per completed round; roster-scoped pools; precision-weighted constructors
- Roster auto-detection from standings (cross-checked against fantasy scrape)
- Driver name aliases (Zhou Guanyu, Nyck De Vries, Kimi Antonelli)
- Historical circuits added (Portugal/France/Russia/Turkey/Styrian→austria)
- Team-name canonicalization ("Red Bull Racing" → "Red Bull")
- Thread-safe throttle + atomic cache writes + corrupt-cache recovery
- FastF1 dedicated throttle (3s gap across 5 call sites)

---

## How It Was Done

Every change followed the same discipline:

1. **Verify the bug exists** by reading code line-by-line or running a live probe
2. **Write the fix** as a targeted edit (never a rewrite of working code)
3. **Compile check** after every file
4. **Run the 41-test contract suite** — green before proceeding
5. **Commit with detailed message** referencing plan item IDs

The contract test suite was the single most important tool: it caught the compute_driver_form
UnboundLocalError, the intra-season Elo `.update()` bug, the pagination boundary corruption,
and several others that would have shipped silently without it.

## Verification Results

```
Final Gates:        ALL 12 PASSED
Contract Tests:     41/41 OK (0 expected failures remaining)
Compile Sweep:      ALL PASS (44 files)
Field Size:         22 (exactly)
Rank Contiguity:    1..22 (verified)
CUDA Training:      Verified on RTX 5060 Laptop GPU
Parallel MC:        5000 sims ≈ 3.2s (14 workers)
Rate Limits:        Zero failures since pacer landed
```

## Remaining Known Limitations (documented non-goals)

- TensorFlow not installed on this machine — LSTM/tire models load lazily and fall back gracefully
- LightGBM GPU requires custom build — CPU n_jobs=-1 is correct for pip wheels
- Wind/humidity features deferred (low expected signal)
- In-race dynamic weather not modelled
- `ctor_elo_z` context pinned at 0 both sides (full Elo replay is heavy; parity preserved)
