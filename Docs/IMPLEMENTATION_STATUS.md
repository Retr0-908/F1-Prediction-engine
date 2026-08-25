# Implementation Status — Improvements Plan Rollout

**Date:** 2026-08-25 · **Branch:** main · **All work test-verified** (`engine/tools/tests/test_contracts.py`, 41 tests) and pushed.

---

## ✅ COMPLETED & VERIFIED

### Milestone 0 — Prediction Integrity Hotfix (Improvement 7)
| Item | Detail |
|---|---|
| **7a Meta representation unified** | `_train_meta_learner` now converts OOF predictions via shared module-level `to_rank()` — identical representation to inference & backtest. Kills the nonsense-ordering regression at its root |
| **8d Field-size-invariant ranks** | `to_rank()` emits `rank/(N+1)` so meta inputs transfer across 20→22–23 driver fields |
| **7b Field integrity** | Pipeline precedence: fantasy entry list > curated 22-seat seed > standings-for-team-attribution. Phantom drivers excluded+logged. Hard `18–24` size gate raises on failure. `FIELD_VALIDATION` SSE event emitted. Config seed restored to exactly 22 seats / 2 per team |
| **7c Hardening** | MC raises on duplicate grids; fantasy-points raises if rank/grid ∉ [1,22] |

### Critical bug fixes (Improvement 9)
| ID | Fix | Verified |
|---|---|---|
| **9-C1** | Pagination: running per-round order counter across pages + `(round,driver)` dedupe | ✅ All seasons re-warmed through fixed code — 2,542 rows, zero non-contiguous rounds (was: 2025 R11/R16/R21, 2026 R5/R10 corrupted; six 2026 R5 DNFs carried fabricated P5–P10) |
| **9-C2** | Reports save explicit `season`; validators thread season; mismatch refuses validation (accuracy-log poisoning blocked) | ✅ unit-covered path |
| **9-C3** | Sepang added under exact Jolpica name `"Bahrain Grand Prix in Malaysia"`; R17–R23 renumbered; Malaysia SC/VSC probs | ✅ matcher test green |
| **9-C4/H-scope** | Optimizer no longer receives phantom drivers; report truncation removed | ✅ field-size test |

### High fixes
- **9-H1** Scrape validates against current-season union (curated seed ∪ live roster ∪ 2025) — Perez/Bottas/Lindblad no longer dropped
- **9-H2** Rebuild↔run mutual exclusion both directions + unique rebuild run_ids
- **9-H3** LSTM context parity: `rain_enc`/`ctor_elo_z` pinned to training constants at inference (unpins via 6c/6f)
- **9-H4** `_match_circuit_config`: São Paulo alias + distinctive-word + key-substring fallback passes

### Contract integrity (Improvement 8)
- **8a** Features are **dict-bound**: `{FEATURE_NAME: value}` + runtime drift guard raising on missing/extra keys. Positional mislabeling structurally impossible
- **8b** Honest pricing: unmatched → logged exclusion (0.0), never fabricated $5M/$8M; fuzzy match = exact → full-surname-token unique; ambiguity rejected
- **8c** Silent-swallow sweep: **46/47 sites** converted to `logger.warning(exc_info=True)` (last site is a debug leftover in tests). Module loggers provisioned everywhere
- **8e** **41-test golden contract suite** implemented and green: classification table, results integrity, name aliases, circuit matching (calendar + historical + Sepang probe), sprint tables, chip-state isolation, Bayesian zeros-parity, Elo roster-scoping + DNF-pair stability, MC seeded reproducibility + corner guards, scoring rules (pole/Q3/Q2 exclusivity, FL top-10, total==Σbreakdown), structural AST guards (feature drift guard present, schema-version key, backtest rank helper, juliacall gating, watchdog job-guard)
- Probe protocol live: 2 remaining `expectedFailure`s (none left stale)

### Performance (Improvement 3)
- **I3 hardware profile** (`engine/core/hardware.py`): stdlib detection ladder, ground-truth CUDA micro-fit probe, env overrides (`F1E_XGB_DEVICE/F1E_MC_WORKERS/F1E_IO_WORKERS`), core/memory-aware pool sizing
- **XGBoost CUDA verified LIVE on RTX 5060**: both rankers train with `tree_method="hist", device="cuda"`; device embedded in model-cache key provenance
- **Parallel Monte Carlo**: chunked ProcessPoolExecutor (spawn-safe module-level worker, per-chunk deterministic seeds), serial fallback, `<2000` sims stay serial. **Verified: 5,000 sims ≈ 3.2 s across 14 workers**
- Full post-CUDA retrain validated end-to-end: field=22, plausible top-5 (Verstappen/Lawson/Hadjar/Norris/Piastri)

### Rate-limit relief (Improvement 5 partial)
- **5a AIMD pacer**: shared request slot, global 429 cooldown (whole fleet pauses), multiplicative gap ↑ capped 10 s, decay after 25 clean successes, jittered retries
- **5c pooled `requests.Session`** (headers immutable, urllib3 adapter pool)
- Zero 429 storms observed since landing

### Medium/Low fixes landed
9-M6 dashboard weather keys · 9-M8 result TTL for live rounds (no frozen mid-race snapshots) · 9-M9 backtest form off-by-one · 9-M10 turbo doubling marked in dashboard · 9-M12 overrides fill gaps only pre-quali · 9-M13 ownership key · 9-M14 partial-roster rejection · LOWs: warm_cache season dedupe + single DONE, São Paulo alias, R17–R23 renumber

### Infrastructure
- Reorg into `engine/` package with `paths.py` anchoring (earlier milestone)
- Server logging handler (warnings persist to file), atomic cache writes, corrupt-cache recovery

---

## ⏳ REMAINING (in execution order)

| Batch | Items | Est. effort |
|---|---|---|
| **A. Quick wins** | 9-M5 move CLI price-recording after corrections · 9-M11 already done ✅ · LOWs: `--report-only`/`--race` implementation, `--auto` first-run default team, `_medal` glyphs, analyse_results argv, hourly 0 °C coercion, grid sentinel 0, DNF-set determinism, upside_pct semantics, transfers-input restore, activeOverrides reset, poll-fallback error surfacing | 2–3 h |
| **B. Robustness** | 9-M2 singleton double-checked locks + atomic model saves (bayesian/temporal/tire/elo) · 9-M11 residual quali-metrics None handling | 2 h |
| **C. Modeling correctness** | 9-M3 Glicko missed-period RD inflation · 9-M4 tire trained-path physics multiplier · 9-M7 sprint-weekend weather session offsets (+callers pass race dict) | 3 h |
| **D. I1 progress bar** | warm_cache manifest + ProgressState + payload contract + server passthrough + Settings-screen `<progress>` UI (+ SSE lifecycle fixes already specified) | 2–3 h |
| **E. I4 telemetry** | Stage payload matrix + DATA counters singleton + Analysis-screen telemetry panel + log timestamps/details | 3–4 h |
| **F. 5d/5e** | Host-aware executor split (Jolpica vs FastF1) · circuit breaker + cooling_down phase | 1–2 h |
| **G. Weather intelligence 6a–6f** | 6a race-hour slice · 6b anchored request + quality flag · 6c archive label backfill (~70 calls one-time) · 6d temperature→tire model · 6e rain_prob feature (payload-additive; replaces rain_enc column) · 6f LSTM alignment | 3–4 h |
| **H. Final gate** | `MODEL_SCHEMA_VERSION = "7"` bump → purge model caches → full retrain (CUDA) → flip remaining probes → BACKTEST.bat honest baseline | 1 h machine time |

**Definition of done for H:** 41+ contract tests green, contiguity probes enabled, one clean live pipeline, backtest spot-round within band, `engine-hw:` line shows CUDA utilization.
