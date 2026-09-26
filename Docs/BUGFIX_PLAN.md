# Bug Fix Implementation Plan — F1 Prediction Engine

**Scope:** All defects identified during full-codebase audit (Aug 2026).
**Total issues:** ~70 across 30+ files. Organized into 9 phases ordered by dependency and risk.
**Severity legend:** 🔴 Critical (wrong predictions/state silently) · 🟠 High (feature broken / crash) · 🟡 Medium · 🟢 Low

Every phase ends with a verification step. Do **not** batch multiple phases into one commit — several fixes invalidate disk caches, so each needs isolated validation.

---

## Newly Discovered Bugs (deep-dive round 2)

The initial audit surfaced 8 defects. The second, line-by-line hunt across all layers found **~60 additional issues**. The highest-impact NEW discoveries (all already integrated into the phase plan below):

| ID | Sev | Discovery | Where | Plan ref |
|---|---|---|---|---|
| NEW-1 | 🔴 | **Every race matched to Australian GP config** — `"grand"`/`"prix"` pass the word filter and Australia is first in `CIRCUITS`; all training circuit features are wrong | predictor.py:1577 | B02.1 |
| NEW-2 | 🔴 | **Lapped finishers `+7 laps` flagged as DNF + forced to fake P20** (whitelist stops at `+6 laps`; 3 divergent copies) — corrupts form/labels everywhere | data_fetcher.py:471,578,611 | B01.1 |
| NEW-3 | 🔴 | **Chip-state default corrupted via shallow copy** — `mark_chip_used` mutates module-level `_DEFAULT_STATE`; reset restores polluted state | chip_advisor.py:62–91 | B05.1 |
| NEW-4 | 🟠 | **Grid penalties double-counted** in fantasy points (post-quali `predicted_grid` already contains penalty) | predictor.py:1398, monte_carlo.py:134 | B02.2 |
| NEW-5 | 🟠 | **Julia MC path can never work** — Python 0-based indexing into Julia 1-based arrays; BoundsError swallowed → silent fallback | monte_carlo.py:417 | B04.1 |
| NEW-6 | 🟠 | **Self-improvement EWMA inverted** — newest races get weight 0.15 instead of 0.85; feedback loop dominated by stale form | self_improvement.py:115 | B07.1 |
| NEW-7 | 🟠 | **`.env` git-tracked despite ignore rule** — auth cookie in commit history (`a5e6143`) | repo index | B00.1 |
| NEW-8 | 🟠 | **Ranker group vector desyncs** when `_build_features` throws mid-race (partial append + continue) → silent cross-boundary ranking corruption | predictor.py:894 | B02.3 |
| NEW-9 | 🟠 | **Model pickle cache ignores feature schema** — only feature COUNT hashed; renamed features load stale models silently | predictor.py:771 | B02.4 |
| NEW-10 | 🟠 | **Bayesian model treats 0 fantasy pts as DNF** — classified P11 zero-scorers inflate ψ; midfield EV deflated up to ~50% | bayesian_model.py:80 | B03.3 |
| NEW-11 | 🟠 | **Current-weekend quali sectors leak into training features** (block not gated by `_is_training`) | predictor.py:600 | B02.5 |
| NEW-12 | 🟠 | **Elo singleton never expires in-process**; 7-day pickle serves stale ratings across races | elo_ratings.py:449 | B03.2 |
| NEW-13 | 🟠 | **Thread-unsafe request throttle + non-atomic cache writes** under warm_cache's 4 workers | data_fetcher.py:149,79 | B01.5 |
| NEW-14 | 🟠 | **Watchdog kills server during cache-rebuild/post-race jobs** (`_pipeline_running` only set by /api/run); `os._exit(0)` truncates concurrent JSON writes | server.py:337 | B06.1 |
| NEW-15 | 🟠 | **Cache-rebuild SSE never emits COMPLETE** — success indistinguishable from failure in UI | server.py:253 / app.js:2931 | B06.2 |
| NEW-16 | 🟠 | **2025 roster alias = 2026 roster** — Hamilton→Ferrari in 2023, phantom drivers (Lindblad) in historical rosters | config.py:368 | B01.3 |
| NEW-17 | 🟡 | Scorer rules wrong: Q2+Q3 bonuses double-awarded, POLE_BONUS unused, FL ignores top-10 eligibility | predictor.py:1408 etc. | B02.9 |
| NEW-18 | 🟡 | RobustScaler fit on full dataset before meta-learner CV (leakage) | predictor.py:975 | B03.5 |
| NEW-19 | 🟡 | `or`-fallbacks inject CURRENT season standings into empty historical fetches | predictor.py:469 | B02.6 |
| NEW-20 | 🟡 | Grid-penalty cache poisons `{}` for 48h when queried between quali and race | data_fetcher.py:650 | B01.6 |

Full inventory of remaining medium/low findings is embedded in the phase sections below (B01.8, B03.x, B04.x, B05.x, B07.x, B09.x).

---

## Table of Contents

- [Phase 0 — Repo Hygiene & Secrets](#phase-0)
- [Phase 1 — Data Layer Correctness](#phase-1) (DNF parsing, sprint rounds, rosters)
- [Phase 2 — Predictor Core Bugs](#phase-2) (circuit matching, scoring, training integrity)
- [Phase 3 — Modeling Subsystems](#phase-3) (Elo, Bayesian, Temporal, tire, scaler)
- [Phase 4 — Monte Carlo Engine](#phase-4)
- [Phase 5 — Fantasy Strategy Layer](#phase-5) (optimizer, chips, scraper, prices)
- [Phase 6 — Server / UI Layer](#phase-6)
- [Phase 7 — Feedback Loop & Validation Tooling](#phase-7)
- [Phase 8 — Backtest De-biasing](#phase-8)
- [Phase 9 — Hardening & Cleanup](#phase-9)
- [Newly Discovered Bugs (round 2 register)](#newly-discovered-bugs-deep-dive-round-2)
- [Impact Analysis — Performance & Accuracy](#impact)
- [Appendix A — Cache Invalidation Matrix](#appendix-a)
- [Appendix B — Regression Test Strategy](#appendix-b)

---

<a name="phase-0"></a>
## Phase 0 — Repo Hygiene & Secrets

### B00.1 🟠 `.env` is tracked by git (secret committed)
- **Where:** git index; added in commit `a5e6143`. `.gitignore` line 8 already lists `.env`, but ignore rules don't apply to tracked files.
- **Risk:** F1 Fantasy auth cookie (`F1_FANTASY_COOKIE`) is in history. If repo was ever pushed, credential is exposed.
- **Fix:**
  ```powershell
  git rm --cached .env
  git commit -m "chore: stop tracking .env"
  ```
- **Then:** rotate/regenerate the F1 Fantasy cookie (logout/login invalidates it). If the repo was pushed publicly, consider history rewrite (`git filter-repo --path .env --invert-paths`) — only if pushed.
- **Also tighten `.gitignore`:**
  ```gitignore
  output/*
  !output/.gitkeep
  ```
  (current pattern only ignores typed extensions; a future `output/foo.parquet` would slip through).

### B00.2 🟢 Broken absolute paths in `scratch/*.py`
Three scripts reference an old machine layout (`c:/Users/<developer>/Desktop/Tools/f1-predictor`):
- `scratch/test_rookie_predictor.py:3`
- `scratch/download_missing_cache.py:6`
- `scratch/check_driver_standings.py:4`

**Fix (uniform pattern):**
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
```
For `check_driver_standings.py`, resolve the cache file relative to project root:
```python
file_path = Path(__file__).resolve().parents[1] / "cache" / "api" / "jolpica__2021_1_driverStandings.json"
```
(verify exact filename against `cache/api/` contents first). Alternatively delete these one-off scripts — they're superseded by `warm_cache.py`.

**Verify:** run each scratch script; confirm imports resolve.

---

<a name="phase-1"></a>
## Phase 1 — Data Layer Correctness

> These fixes change what every downstream module reads. Complete Phase 1 before any model-side work.

### B01.1 🔴 DNF whitelist misclassifies lapped finishers as DNF + assigns fake P20
- **Where:** `data_fetcher.py` lines 471–472, duplicated at 578–579, worse variant at 611–612 (three divergent copies).
- **Root cause:** `"dnf": status.lower() not in ("finished", "+1 lap", ..., "+6 laps")` — any `+N laps` with N≥7 becomes a DNF, and `position` falls into `int(...) if digit else 20`, fabricating P20.
- **Impact:** corrupts `compute_driver_form`, `compute_multiseason_*`, `compute_constructor_reliability`, and all ML training labels.
- **Implementation:**
  1. Create ONE helper in `data_fetcher.py`:
     ```python
     import re
     _LAPPED_RE = re.compile(r"^\+\d+\s+laps?$", re.IGNORECASE)
     def classify_result(position_text: str, status: str) -> tuple[int | None, bool, str]:
         """Returns (position or None, did_not_finish, classification_kind).
         kind ∈ {"finished","lapped","dnf_mechanical","dsq","withdrawn"}"""
         s = (status or "").lower()
         p = (position_text or "").strip()
         if p.isdigit():
             return int(p), False, "finished"
         if _LAPPED_RE.match(s):
             return None, False, "lapped"        # finished but unclassified position
         if "disqualif" in s or s == "dsq":
             return None, True, "dsq"
         if s in ("withdrawn", "wd"):
             return None, True, "withdrawn"
         return None, True, "dnf_mechanical"
     ```
  2. Replace all three call sites. For lapped finishers, derive position from list order among classified runners instead of hardcoding 20 (iterate the API result list; assign running index when position text isn't numeric but status says lapped).
  3. Keep DSQ/W separate from mechanical DNF — add `"dsq"` flag to the result dict so form/reliability math can decide whether to count them.
- **Cache impact:** results for completed seasons are permanently cached (`cache_hours=0`) → **must purge affected cache files** after deploy (see Appendix A) or bump every consuming cache key.
- **Verify:** pull 2021–2025 seasons; assert zero results have `dnf=True` AND numeric positionText; compare `dnf_rate` per driver before/after — expect small decreases for backmarkers.

### B01.2 🔴 EWMA-vs-SMA lie + formula not encoded in cache key
- **Where:** `data_fetcher.py` 1284–1373 (`EWMA_LAMBDA = 1.0`, docstring claims λ=0.7, output field says `"source": "current_ewma"`, cache key `driver_form_{year}_{n}_{until}_v2` omits λ).
- **Decision required:** keep SMA (deliberate de-biasing choice) or restore EWMA.
  - **Recommendation: keep SMA** (recency over-weighting amplified noise in a 24-race season), but make it honest.
- **Implementation:**
  1. Hoist to config: `FORM_DECAY_LAMBDA: float = 1.0` with comment explaining SMA choice.
  2. Fix docstring; rename `"source": "current_sma"`.
  3. Encode formula in cache key: `f"driver_form_{year}_{num_races}_{until_round}_lam{EWMA_LAMBDA}"`.
  4. Delete dead weight scaffolding if λ stays 1.0 (or keep generic code — but key must embed λ either way).
- **Same treatment** for `compute_multiseason_driver_form`'s internal weighting.

### B01.3 🔴 `DRIVER_TEAMS_2025` alias points at the 2026 roster
- **Where:** `config.py` 368–369, 376, 391. Consumers: `backtest.py:106`, `predictor.py:279,877`, `temporal_model.py:48`, `fantasy_optimizer.py` (multiple).
- **Impact:** Hamilton→Ferrari in 2023, Perez→Cadillac in 2024, phantom drivers (Lindblad) in historical rosters — corrupts teammate deltas, constructor ELO grouping, reliability lookups, and backtest metrics.
- **Implementation:**
  1. Write a genuine `DRIVER_TEAMS_2025` dict (2025 lineup: Verstappen/Perez? — no: Verstappen–Lawson→Tsunoda saga; Hadjar at Racing Bulls; no Lindblad/Bortoleto at Sauber—Bortoleto WAS at Sauber in 2025; Antonelli at Mercedes; no Cadillac/Audi-as-Cadillac — Audi replaced Sauber mid-name). Cross-check each seat against the 2025 season results in the Jolpica cache before committing.
  2. Same for `CONSTRUCTORS_2025` (10 teams, no Cadillac) and `DRIVER_SHORT_2025` (drop `LIN`).
  3. Make `backtest.DRIVER_TEAMS_BY_YEAR` cover **every** year in `HISTORICAL_SEASONS` (currently 2021 missing → predictor.py:869 falls back to 2026).
  4. In `predictor.py:869–877`, replace silent fallback with a loud failure:
     ```python
     roster = DRIVER_TEAMS_BY_YEAR.get(year)
     if roster is None:
         raise KeyError(f"No roster defined for {year}; add it to DRIVER_TEAMS_BY_YEAR")
     ```

### B01.4 🔴 Sprint-round tables: three sources, three different answers, none verified
- **Where:** `config.py:330–337` (names), `data_fetcher.py:293–298` + `get_sprint_rounds()` 320–328 (round sets), `backtest.py:87–91` (different round sets).
- **Verified reality:** 2024 sprints were China R5, Miami R6, Austria R11, USA R19, Brazil R21, Qatar R23 — none of the hardcoded tables match. 2025: China, Miami, Belgium, USA(Austin), Brazil, Qatar.
- **Implementation:**
  1. Make the schedule's own field authoritative. In `is_sprint_weekend()`:
     ```python
     def is_sprint_weekend(year, round_num):
         schedule = get_season_schedule(year)
         for r in schedule:
             if r["round"] == round_num:
                 sd = r.get("sprint_date") or ""
                 return bool(str(sd).strip())
         return False
     ```
     (Jolpica/Ergast schedules include `SprintDate` for all sprint seasons ≥2021.)
  2. Keep one shared table ONLY as last-resort fallback, moved to `config.SPRINT_ROUNDS_BY_YEAR`, populated with **verified** values:
     - 2024: `{5, 6, 11, 19, 21, 23}`
     - 2025: `{2, 6, 12, 19, 21, 22}` ← verify rounds for Belgium/USA/Brazil/Qatar from the fetched 2025 schedule before finalizing
     - 2026: derive from announced calendar; until confirmed, log a warning whenever the fallback is used.
  3. When fallback is used, `log.warning("sprint detection using static table for %s — verify against schedule", year)`.
  4. Delete `SPRINT_ROUNDS_BY_YEAR` from `backtest.py`; import the shared config table. Convert `config.SPRINT_ROUNDS` (names frozenset) to derived-from-schedule or delete (it's imported unused in predictor/backtest).
  5. In `get_sprint_rounds()`, don't raise on honest empty API result; return empty set + warning.
- **Verify:** `scratch/verify_sprint_detection.py` output vs official calendars for 2023–2025.

### B01.5 🟠 Thread-unsafe throttle + non-atomic cache writes
- **Where:** `data_fetcher.py:149–154` (`_throttle` check-then-act race), `:79–81` (`_save_cache` in-place write), `json.load` unprotected in `_load_cache`; triggered by `warm_cache.py:100` (`max_workers=4`).
- **Implementation:**
  ```python
  _throttle_lock = threading.Lock()
  _cache_io_lock = threading.Lock()

  def _throttle():
      global _last_request_time
      with _throttle_lock:
          wait = _MIN_REQUEST_GAP - (time.time() - _last_request_time)
          if wait > 0:
              time.sleep(wait)
          _last_request_time = time.time()

  def _save_cache(key, data):
      with _cache_io_lock:
          tmp = _cache_path(key).with_suffix(".tmp")
          with open(tmp, "w", encoding="utf-8") as f:
              json.dump(data, f, indent=2, default=str)
          os.replace(tmp, _cache_path(key))
  ```
  In `_load_cache`, wrap `json.load` in try/except `(json.JSONDecodeError, OSError)` → unlink corrupt file, return None (treat as miss).

### B01.6 🟠 Grid-penalty cache poisons post-race weekend for 48h
- **Where:** `data_fetcher.py:650–677`. Between quali and race completion, `get_race_results` returns `[]` → `{}` computed and persisted 48h.
- **Implementation:**
  ```python
  try:
      race = get_race_results(year, round_num)
      if not race:                       # race not run yet — DO NOT CACHE
          return {}
      quali = get_qualifying_results(year, round_num)
  except Exception:
      return {}
  ...
  _save_cache(cache_key, penalties)      # only reached when race exists
  ```
  Reduce TTL to 6h (results are immutable once race completes anyway; the permanent-cache path of `get_race_results` makes recompute cheap).

### B01.7 🟠 FastF1 rate-limiter monkey-patch is silent and ban-risky
- **Where:** `data_fetcher.py:33–39`.
- **Implementation:**
  1. Remove the blanket suppression; instead configure deliberately:
     ```python
     try:
         import fastf1.req
         fastf1.req._SessionWithRateLimiting._RATE_LIMITS = {}
         logger.info("FastF1 rate limiter disabled (own throttle active)")
     except AttributeError:
         logger.warning("FastF1 internals changed — rate limiter ACTIVE")
     ```
  2. Add a real gap for FastF1 traffic too (FastF1 hits the same endpoints): route FastF1 calls through a second throttle instance with ≥0.5s gap.
  3. Note in README that disabling their limiter increases ban risk; consider removing entirely if warm_cache runtime is acceptable.

### B01.8 🟡 Assorted data_fetcher fixes
| Issue | Lines | Fix |
|---|---|---|
| Quali sector delta `0.0` conflates "missing" with "fastest" | 1265–1270 | emit `None` for missing components; rebuild `best_laps` with `laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]` |
| Antonelli splits into two identities (FastF1 vs Jolpica paths) | 99–101, 1253–1257, 1163–1172 | map FastF1 abbreviations through `DRIVER_SHORT_2026` first; extend `_PREFERRED_FORENAME` lookup to token-based surname matching |
| Tire allocations cache failure for 72h (`cached is not None`) | 755 | use truthiness like siblings |
| NaN compound fragments stints (`NaN != NaN`) | 899–905 | `if pd.isna(compound) or compound == prev_compound: continue stint` |
| `Retry-After` header HTTP-date crashes retry loop | 182, 231 | `int(h) if h.isdigit() else 30` inside try/except |
| `normalize_telemetry` feeds unsorted x to `np.interp` | 1052–1059 | sort by `t_sec` first |
| Dead `effective_grid > 22` clamp | 533–551 | delete or make clamp apply post-reassignment |
| Placeholder `momentum_trend: 0.0` presented as feature | 1442, 1464 | compute real momentum or remove field + downstream feature |

---

<a name="phase-2"></a>
## Phase 2 — Predictor Core Bugs

> ⚠️ Fixes in this phase change training data/features → all saved models must be invalidated (delete `cache/models/`). Coordinate with Appendix A.

### B02.1 🔴 `_match_circuit_cfg` returns Australian GP config for EVERY race
- **Where:** `predictor.py:1577–1584`.
- **Root cause:** word filter keeps `"grand"`/`"prix"`; `"Australian Grand Prix"` is first in `CIRCUITS`, and `"grand" in <any race name>` is always true.
- **Impact:** every training sample uses Australia's circuit config, circuit history, sc_prob, and Phase-5 track encodings; inference uses caller-supplied configs → train/inference skew on an entire feature family.
- **Implementation:**
  ```python
  _GENERIC_TOKENS = {"grand", "prix"}
  _NAME_ALIASES = {
      # Jolpica name -> CIRCUITS key (extend as renames happen)
      "são paulo grand prix": "brazil",
      "brazilian grand prix": "brazil",
      "spanish grand prix": "spain",       # pre-2026 Barcelona
      "miami international autodrome": "miami",
      # ... audit all historical Jolpica names against CIRCUITS keys
  }
  def _match_circuit_cfg(race_name: str) -> dict:
      key = _NAME_ALIASES.get(race_name.lower().strip())
      if key:
          cfg = CIRCUITS.get(key)
          return {**cfg, "name": key} if cfg else {}
      race_lower = race_name.lower()
      best_name, best_cfg, best_score = "", {}, 0
      for name, cfg in CIRCUITS.items():
          words = [w for w in name.lower().split()
                   if len(w) > 3 and w not in _GENERIC_TOKENS]
          if not words:
              continue
          score = sum(1 for w in words if w in race_lower)
          if score == len(words) and score > best_score:
              best_name, best_cfg, best_score = name, cfg, score
      if best_score:
          return {**best_cfg, "name": best_name}
      logger.warning("_match_circuit_cfg: no match for %r", race_name)
      return {}
  ```
  Build the alias map empirically: iterate every cached schedule 2021–2026, run the matcher, list misses, add aliases.
- **Verify:** unit test asserting correct key for all 24+ names in cached schedules; assert `_match_circuit_cfg("Chinese Grand Prix")["key"] != "australia"`.

### B02.2 🟠 Grid penalties double-counted in fantasy points
- **Where:** `predictor.py:1398–1403` (adds `self._grid_penalties` to a `predicted_grid` that post-quali ALREADY contains the penalty via `effective_grid`); same double-add in `monte_carlo.py:134–137`.
- **Implementation:**
  1. In `predict_qualifying_order()`, emit both fields per entry:
     `{"quali_position": <classification position>, "predicted_grid": effective_grid}`.
  2. In `estimate_fantasy_points()` and `monte_carlo.simulate_race_weekend`, compute
     `effective_grid = entry.get("predicted_grid")` and STOP re-adding penalties when post-quali; only add when mode is pre-quali (predicted raw grid). Gate on a new flag carried through context: `self._post_quali = bool(self._actual_quali)`.
- **Verify:** construct a synthetic penalized driver; assert positions-gained term equals `race_pos − effective_grid` exactly once.

### B02.3 🟠 Ranker `group` vector desync on partial failure
- **Where:** `predictor.py:894–964`. If `_build_features` throws mid-race, rows appended but `race_groups.append(...)` skipped → XGBoost/LightGBM fit gets misaligned groups (silent cross-boundary ranking corruption).
- **Implementation:** stage rows per race; commit atomically:
  ```python
  rows = []
  for r in race_results:
      feat = self._build_features(...)
      rows.append((feat, float(r["position"])))
  X_race.extend(f for f, _ in rows)
  y_race.extend(p for _, p in rows)
  race_groups.append(len(rows))
  ```
  Wrap only the whole-race block in try/except.

### B02.4 🟠 Model pickle cache keyed only on season list + feature COUNT
- **Where:** `predictor.py:771–786`. Renaming/reordering features with constant count loads a stale model trained on a different layout — garbage predictions, no error.
- **Implementation:**
  ```python
  MODEL_SCHEMA_VERSION = "3"   # bump on ANY change to FEATURE_NAMES order,
                               # target construction, or hyperparameters
  seasons_str = (",".join(map(str, sorted(HISTORICAL_SEASONS)))
                 + "_features_" + "|".join(FEATURE_NAMES)
                 + f"_v{MODEL_SCHEMA_VERSION}")
  ```
  Also hash into the key: ranker objectives (B02.7), label construction, and `EWMA/formula` choices from B01.2.

### B02.5 🟠 Current-weekend `_quali_sectors` leak into training features
- **Where:** `predictor.py:600–604` — block not gated by `self._is_training`, unlike LSTM/tire paths.
- **Fix:** gate identically:
  ```python
  if (quali_sectors_self and driver_name in quali_sectors_self
          and not getattr(self, '_is_training', False)):
  ```
  During training the fallback chain (historical quali → form proxy) already handles it.

### B02.6 🟡 Empty-container `or` fallbacks can inject current-season data into historical rows
- **Where:** `predictor.py:469–479` (`drv_standings or self._driver_standings`, etc.). Legitimate empty fetches (round 0, API hiccup) get replaced by TODAY's standings.
- **Fix:** sentinel-based overrides throughout:
  ```python
  _SENTINEL = object()
  drv_standings = ... : param default _SENTINEL
  if drv_standings is _SENTINEL:
      drv_standings = self._driver_standings
  ```
  Apply to all eight falsy-or patterns in that function.

### B02.7 🟡 Ranker outputs treated as bounded position scale; NDCG label distortion; objectives unspecified
- **Where:** `predictor.py:982–991, 1026–1033` (no `objective=` passed), `:997, 1087–1110` (`25.0 - pr_xgb` averaged with RF positions; `std` mixes scales).
- **Implementation:**
  1. Pass explicit objectives: `XGBRanker(objective="rank:ndcg")`, `LGBMRanker(objective="lambdarank")`.
  2. Replace raw-margin mixing with per-race rank conversion at inference:
     ```python
     def margins_to_rank(scores):           # scores within one race's field
         order = np.argsort(np.argsort(-scores))
         return order + 1.0                  # expected rank 1..N, same scale as RF
     pr_xgb_rank = margins_to_rank(pr_xgb_scores)
     pred = float(np.mean([pr_rf, pr_xgb_rank, pr_lgb_rank]))
     std  = float(np.std([pr_rf, pr_xgb_rank, pr_lgb_rank]))
     ```
  3. Keep `25.0 - pos` only as monotonic label transform for lambdarank (fine), but clip labels to field size and document.

### B02.8 🟠 Blanket silent exception swallowing
- **Where:** `predictor.py:963–966, 629–630, 654–655, 453–454, 1468–1469`; `elo_ratings.py:469–470, 481–482`; `warm_cache.py` throughout.
- **Implementation:**
  1. Replace `except Exception: continue` with:
     ```python
     except Exception:
         logger.exception("Skipping race %s %s (feature build failed)", year, name)
         skipped += 1
         continue
     ```
  2. Print skip count in dataset summary: `Trained on N samples (skipped K races — see warnings)`.
  3. In `warm_cache._warm_single_race`, count successes separately from failures; report both in the banner; remove duplicate COMPLETE printout; fix `race.get("raceName",...)` → `race["name"]` (line 19).

### B02.9 🟡 Scoring-rule fixes (money-relevant, apply in BOTH predictor.py and monte_carlo.py)
| Rule bug | Location | Correct behavior |
|---|---|---|
| Q2+Q3 bonuses awarded together | predictor.py:1408–1413, mc 272–275, 412–415 | top-10 → Q3 bonus; P11–15 → Q2 bonus; never both |
| POLE_BONUS=0.5 never used | config:139, predictor | add `+ POLE_BONUS if grid_pos==1` |
| Fastest-lap ignores top-10 eligibility | predictor:1434, mc:290–293, 446–447 | multiply `fl_prob` by `1[race_p <= 10]` |
| Deterministic path omits DNF penalty while MC includes it | predictor:1430 vs mc:278 | discount race-position pts by `(1−dnf_prob)` deterministically, or document MC-only and exclude from optimizer comparison |

Extract shared scoring into a single `scoring.py` module consumed by both engines so they can't drift again.

---

<a name="phase-3"></a>
## Phase 3 — Modeling Subsystems

### B03.1 🟠 Temporal model train/inference context mismatch (LSTM branch is a stub)
- **Where:** training `temporal_model.py:274–283` (only `grid_pos` varies; rain/sc/temp/ctor_elo are constants); inference `predictor.py:618–631` passes `context_vec=None` → constant `[11, 35, 0, 0.40, 0]` — including freezing the ONE feature training varied (grid) at 11.0.
- **Implementation (two steps):**
  1. **Minimal high-value fix (do first):** build and pass the real context in `predictor._build_features`:
     ```python
     ctx_vec = np.array([
         float(quali_pos_val),
         35.0,                                            # placeholder temp (see step 2)
         1.0 if self._weather.get("rain_risk") in ("medium","high") else 0.0,
         float(track_feature(circuit_key, "sc_probability", 0.40)),
         float(z_of_ctor_elo(driver_constructor)),        # z-score within current pool
     ], dtype=np.float32)
     lstm_momentum_pos = self._temporal_model.predict_driver_momentum(
         driver_name, self._recent_race_results, self._tire_data_by_round,
         context_vec=ctx_vec)
     ```
  2. **Full fix:** populate real historical context in `build_training_dataset` (rain/sc from track-feature tables + cached weather; ctor_elo_z from the Glicko system rebuilt per season). Then retrain; delete `cache/models/temporal_*`.
  3. Remove false comments ("predictor enriches at inference").
- **Verify:** assert `predict_driver_momentum` differs materially between P1 qualifier and P15 qualifier for the same history.

### B03.2 🟠 Elo singleton + 7-day pickle serve stale ratings
- **Where:** `elo_ratings.py:449–484`. In-process singleton returned forever; TTL checked only on load.
- **Fix:**
  1. Key cache filename on `seasons + latest_completed_round`: fetch `max(round)` of completed races cheaply (cached schedule scan) and embed in the key.
  2. Re-check freshness before returning the singleton (store build timestamp on the system object).
- **Related Glicko-2 cleanups (same file):**
  - Line 231–234: comment claims averaging but code sums — divide `v_sum`/`delta_sum` by `(n−1)` **or** delete the comment (decide: current behavior ≈ intentional amplification? recommend dividing to match canonical Glicko-2).
  - Feature pool includes retired drivers since 2021 — pass current roster into `get_elo_features()` and restrict rank/z-score computation to roster members.
  - Constructor weights: use `1/max(phi,1)**2` (precision) not `1/RD`; delete unreachable `total_weight == 0` branch.
  - Skip pairwise updates when both drivers DNFed (arbitrary stable-sort winner pollutes sums).

### B03.3 🟠 Bayesian model treats 0 fantasy points as DNF
- **Where:** `bayesian_model.py:80–99`. A classified P11 with zero points counts as DNF → ψ inflated → EV deflated up to ~50% for midfielders. Also multiplies away qualifying points retained on DNF weekends.
- **Fix:**
  1. Change signature: `fit_driver(..., dnf_flags: list[bool] | None = None)`. Caller (predictor.py:1458–1467) derives flags from form/results (`position == 99` convention).
  2. Fall back to zeros-as-DNF ONLY if flags absent, with a logged warning.
  3. EV: `robust_ev = (1 − ψ)·μ_points + ψ·E[quali_pts_on_dnf]` where `E[quali_pts_on_dnf]` is the driver's mean quali bonus (small, nonzero).
  4. Pass `circuit_key=self._circuit_config.get("key")` from predictor so the dead track prior activates.
  5. Fix class docstring (it's a Beta-Bernoulli × Normal conjugate approximation, not ZINB); delete vestigial `traces`/PyTensor setup; drop unused pandas import.

### B03.4 🟡 Bayesian override breaks `total_pts == sum(breakdown)`
- **Where:** `predictor.py:1455–1467` replaces `total` without touching breakdown.
- **Fix:** scale breakdown proportionally:
  ```python
  if total > 0:
      scale = bayesian_ev / total
      breakdown = {k: v * scale for k, v in breakdown.items()}
  total = bayesian_ev
  ```

### B03.5 🟡 RobustScaler fit on full data before meta-learner CV (leakage)
- **Where:** `predictor.py:975, 1019` vs fold slicing in `_train_meta_learner`.
- **Fix:** move scaling inside `_train_meta_learner`: fit scaler on train slice per fold; refit on full data for the final stacked models; report CV RMSE from scaled-per-fold values. Expect CV metrics to get slightly worse — that's honesty, not regression.

### B03.6 🟡 Tire model dead features & semantics
- `tire_model.py`: `rain_enc` always 0 (wire real rain or remove input); strategy head trained on *actual* strategies, not optimal — relabel as "typical strategy prediction" or source better labels; delete unused pandas/tf imports; accept-but-ignore kwargs (`circuit_key`, `sm_zones`) documented.

### B03.7 🟢 track_features_loader robustness
- `load_all_track_features` magic `len(_cache) >= 24` early-return → replace with explicit `all_loaded` flag set after full directory scan.
- Negative-cache invalid files (sentinel) so `_build_features` doesn't re-parse bad JSON per driver per call; dedupe `_validation_errors`.
- Type check: reject `bool` for numeric fields (`isinstance(v, bool)` exclusion).
- Extend range validation to `first_lap_incident_risk`, `vsc_probability`, `overtake_mode_efficiency`.

---

<a name="phase-4"></a>
## Phase 4 — Monte Carlo Engine

### B04.1 🟠 Julia path indexes 1-based arrays with Python 0-based indices
- **Where:** `monte_carlo.py:417–427` vs `monte_carlo_engine.jl:48,142` (`zeros(Int, n, n)`). First access raises BoundsError → swallowed by `except Exception` → silently runs slow Python path forever. If tolerated, results would be off-by-one (driver i↔i+1).
- **Fix:**
  ```python
  ranks_np  = np.asarray(rank_counts)      # juliacall → numpy copy
  dnfs_np   = np.asarray(dnf_counts)
  dnf_count = int(dnfs_np[i])
  count     = int(ranks_np[i, rank-1])
  ```
  Remove the bare `except Exception` around delegation; log delegation success/failure explicitly.

### B04.2 🟡 Julia/Python behavioral parity
The Julia engine omits rain, sprint points, circuit features, uses EV-style FL/DOTD bonuses instead of Bernoulli draws, applies SC differently, and never receives the seed.
- **Short-term (recommended):** gate Julia behind a flag defaulting OFF until parity work is done; document divergence at the dispatch site.
- **Long-term:** forward seed (`jl.seval(f"Random.seed!({seed})")`), port rain/sprint/circuit effects, convert bonuses to Bernoulli draws, then validate: run both engines at n=5000 with same seed, assert KS-test similarity on point distributions.

### B04.3 🟡 SC simulation cleanup (`monte_carlo.py:228–246`)
- Three contradictory ranges (comment P5–15, comment P6–20, code P6–15). Pick one (recommend P6–15), align comments.
- Decide the reference order: initialize sim positions from `effective_grid` (quali-derived) so pace moves cars from a consistent baseline — removes the grid/race-order conflation.
- Deterministic tie-break: sort by `(position, -mean_pts)`.
- Delete dead `sc_lap` RNG draw (line 230) or actually implement lap-timed SC.

### B04.4 🟢 DistributionStats guards & semantics
- Guard `n == 0` (ZeroDivisionError possible on empty merge path): quantile index `max(0, int(q*(n-1)))`.
- Rename/compute `p_top3`, `p_points_finish` from simulated finishing POSITIONS (≤3 / ≤10), not point thresholds.

---

<a name="phase-5"></a>
## Phase 5 — Fantasy Strategy Layer

### B05.1 🔴 Chip state default corrupted via shallow copy
- **Where:** `chip_advisor.py:62,66,79–84,87–91`. `dict(_DEFAULT_STATE)` shares nested dicts; `mark_chip_used` mutates the module-level default; `reset_chip_state` then persists polluted state.
- **Fix:**
  ```python
  import copy
  def _fresh_state(): return copy.deepcopy(_DEFAULT_STATE)
  ```
  Use in `load_chip_state` (both return sites) and `reset_chip_state`.
- **Also (atomicity, B05.1b 🟡):** wrap read-modify-write in a module-level `threading.Lock` and write atomically (`tmp` + `os.replace`) — server handlers run concurrently (double-click on "mark used" fires two POSTs). On JSONDecodeError in `load_chip_state`, log loudly instead of silently returning defaults.

### B05.2 🔴 Scraper fallback-chain type mismatch crashes tuple unpacking
- **Where:** `fantasy_scraper.py:316–318` (`_try_html_scrape` embedded-JSON path returns bare dict) vs caller `:231–235` (`prices, roster = result`).
- **Fix:** normalize both return paths of `_try_html_scrape` to `(prices_dict, roster_dict)`; additionally harden callers with `isinstance(result, tuple)` checks (mirroring `scrape_constructor_prices:482–489`).
- **Related scraper fixes:**
  - 🟡 Don't persist fallback constructor estimates into the live price cache (`:491–494`) — return without `_save_cache`, or tag `"_fallback": true` and surface a warning flag in the API/UI.
  - 🟡 Validate feed-derived prices with `_is_valid_price` in `_parse_driver_feed`/`_parse_constructor_list` (`:370–373, 516–518`); replace naive `price /= 10` heuristic with unit detection (0.1M units ⇒ ÷10 always when max observed > threshold).
  - 🟡 Constructor DOM heuristic (`:103–109`): validate candidate team strings against `CONSTRUCTORS_2026` like the driver branch does.
  - 🟢 Replace rich-markup-in-plain-print (`[yellow]...`) with real logging.
  - 🟢 Move consent-element IDs to module constants with a "site changed?" comment.

### B05.3 🟠 Optimizer threshold/budget inconsistencies
- **Where:** `fantasy_optimizer.py:186/196` enforces `repl_pts < star_pts * 1.05` though `TOP_PERFORMER_GAIN_THRESHOLD = 0.30` documents 30% (dead variable, line 100).
- **Fix:** `if repl_pts < star_pts * (1 + TOP_PERFORMER_GAIN_THRESHOLD): continue`. Re-tune threshold empirically after fix (behavior tightens considerably).
- **Budget invariant (🟡):** `suggest_team_changes` gates affordability against global `FANTASY_BUDGET` in one place and user bank elsewhere — settle on one semantic (recommend: `budget_remaining + freed cash`) and thread it through.
- **Dream team budget (🟡):** `run_full_pipeline` (main_logic.py:183) calls `find_optimal_team` with flat $100M — pass `budget=user_total_funds` (= remaining budget + current squad cost) so recommendations are affordable.

### B05.4 🟡 Price tracker season desync + blind spot
- Defaults hardcoded `season=2026` in five signatures (`price_tracker.py:51–56, 87–92, 123, 183`) → import `CURRENT_SEASON`.
- `record_prices`/`record_ownership` called only from CLI `main.py` — web users accumulate no history. Add recording to `run_full_pipeline` right after the price-scrape step.
- Namespace snapshots per player type (`prices_drivers` / `prices_constructors`) to avoid name collisions.
- Pruning: also prune keys from previous seasons beyond a keep-limit.

### B05.5 🟢 Misc strategy-layer items
- `main_logic.py:201–212`: include `"hold_until_round": cs.hold_until_round` in chip serialization.
- `chip_advisor.py:605`: set `hold_until=None` when `use_now=True` (UI currently prints "PLAY NOW / Hold until…" simultaneously).
- `chip_advisor._season_urgency`: derive `total_rounds` from schedule length instead of hardcoded 22.
- `fantasy_optimizer`: delete dead `_count_team_drivers()`.

---

<a name="phase-6"></a>
## Phase 6 — Server / UI Layer

### B06.1 🟠 Watchdog can kill server mid-operation; `os._exit(0)` everywhere
- **Where:** `server.py:337–363` (watchdog only respects `/api/run`'s `_pipeline_running`), `:244–257` (cache rebuild thread), `:464–487` (post-race thread), `:552–555` (`/api/shutdown` responds after exit).
- **Implementation:**
  1. Job-aware guard:
     ```python
     _active_jobs = 0                    # guarded by threading.Lock
     # increment synchronously in each endpoint handler BEFORE spawning the thread;
     # decrement in the thread's finally-block.
     if _heartbeat_received and _active_jobs == 0 and (
             time.time() - _last_heartbeat > _HEARTBEAT_TIMEOUT):
         ...
     ```
  2. Graceful exit: `uvicorn_server.should_exit = True` (store the Server instance globally) instead of `os._exit(0)`; `/api/shutdown` returns `{"status":"shutting_down"}` then schedules exit via `loop.add_callback(server.should_exit setattr)`.
  3. Frontend (app.js:1490–1494): don't fully trust `beforeunload` backdating — require N consecutive missed beats server-side (counter, not single timestamp), and start the heartbeat interval on `DOMContentLoaded` rather than after dashboard init (app.js:73 coupling).

### B06.2 🟠 SSE lifecycle cluster (fix together)
1. **Rebuild stream never emits COMPLETE** (`server.py:244–253` vs `app.js:2931–2947`): emit `sync_progress_callback(run_id, "COMPLETE", "done", "")` on success before termination; UI reloads only on COMPLETE.
2. **Client gives up on first error** (`app.js:834–841`): on `onerror`, fall back to polling `GET /api/results/{run_id}` every 5s until terminal status; keep Run button state consistent. Add exponential reconnect attempts (3×) before falling back.
3. **`startAnalysis` has no try/catch** (`app.js:673–690`): wrap `fetch('/api/run')` in try/catch/finally; finally resets `_isPipelineRunning` and re-enables the button.
4. **Server queue/result leak** (`server.py:325–326`): in the SSE generator's `finally`, pop `run_queues[run_id]`; cap `run_results` to last 10 runs (OrderedDict + evict); evict queues whose terminal event was sent >1h ago in the watchdog loop.
5. **Double-consumer hazard:** attach-once flag per run_id; second SSE connect gets 409.

### B06.3 🟡 `/api/results/{run_id}` 500s on numpy payload
- **Where:** `server.py:459–463` returns unsanitized dict (SSE path sanitizes; this one doesn't).
- **Fix:** `return {"status": "ok", "data": sanitize_for_json(run_results[run_id])}` — prerequisite for B06.2's polling fallback.

### B06.4 🟠 `my_team.json` dual-path split
- Server writes `output/cache/my_team.json` (`server.py:143,160`); CLI writes `cache/my_team.json` (`main.py:1304`). Team edits in one mode are invisible to the other.
- **Fix:** define `MY_TEAM_PATH` once in `config.py`; both consumers use it; on first read, migrate legacy location if newer (mtime compare). Keep `reset_memory` wiping both for one release, then drop legacy.

### B06.5 🟡 Misc server items
- `async def rebuild_cache` does blocking `shutil.rmtree` on the event loop (`:232–238`) → `await run_in_threadpool(...)`.
- `DELETE /api/shutdown` handled in B06.1(2).
- Duplicate inline imports (`shutil`, `time`, `json as _json`) hoisted to top.
- Stub endpoints: implement `/api/circuits` or remove route; `/api/prices/override` echoes data without effect — wire it to `price_tracker.record_prices` or delete.
- Port/URL constants hoisted to config.

### B06.6 🟠 CLI chip-advisor NameError (`main.py:1393`)
- Add `chip_state = load_chip_state()` before the `try:` at line 1387 (matches `main_logic.py:189`). Optionally ALSO make `advise_chips` tolerant (already self-heals on None) — but pass explicitly for symmetry.
- While in `main.py`: delete dead display functions superseded by dashboard (grep callers first: `display_race_prediction`, `display_monte_carlo`, `display_differential_picks`…), fix `check_grid_changes` writing `grid_cache.json` to CWD (use project-root path), remove no-op `pause_and_clear`.

### B06.7 🟡 Lookahead EV cost & methodology (`main_logic.py:167–176, 271–336`)
1. Add `predictor.load_context_light()` that swaps only circuit/weather/grid state without re-fetching standings or rebuilding Glicko (snapshot mutable refs once, restore after).
2. Normalize `total_ev` to per-race average before feeding `mc_pts` into transfer/dream-team functions (they assume single-race scales).
3. Unify DOTD computation into one helper used by both main path and lookahead (two conventions currently coexist).
4. Pass `roster=dynamic_roster` through to lookahead predictions.

---

<a name="phase-7"></a>
## Phase 7 — Feedback Loop & Validation Tooling

### B07.1 🟠 Self-improvement EWMA recency weighting inverted
- **Where:** `self_improvement.py:115–118, 139–141`. Iterating oldest→newest with `ewma = decay*ewma + (1-decay)*err` gives the NEWEST observation weight `(1-d)` = 0.15 — old races dominate, contradicting both comments and weakening the feedback loop.
- **Fix (choose one, align docs):**
  ```python
  ewma = recent_errors[-1]                      # newest
  for err in reversed(recent_errors[:-1]):
      ewma = decay * ewma + (1 - decay) * err   # now newest carries weight `decay`
  ```
  Set docstring default description to match actual `decay=0.85` ("recent errors weighted ~6x"). Fix `CONFIDENCE_POOL` example comment (pool=8 → actual 24).
- **Note:** correction SIGN verified correct end-to-end (error = predicted − actual; predictor subtracts clamped bias). No sign flip needed.
- **Atomic write (🟢):** `bias_corrections.json` written non-atomically while `predictor.load_context` may read → temp-file + `os.replace`.

### B07.2 🟡 Post-race check selection & matching
- `post_race_check.py:80–87`: picks lexicographically-last prediction file, not latest. Embed timestamp in filenames going forward; for existing files, select max by `generated_at` parsed from JSON content.
- `:139–147`: surname substring containment can misattribute (false hits, fabricated `act_pos=99` for matched-none). Require exact normalized surname-token equality; leave unmatched drivers explicitly flagged (exclude from metrics, log warning) instead of inventing DNFs.

### B07.3 🟡 Weather robustness
- **Venue-timezone epochs (🟠 latent):** `weather.py:343–347` epoch-izes venue-local naive strings using machine-local tz; `:414` filters via symmetric round-trip (works by accident, breaks under DST edges, emits wrong `dt` downstream). Fix:
  ```python
  from zoneinfo import ZoneInfo
  tz = ZoneInfo(forecast.get("timezone", "UTC"))     # Open-Meteo echoes resolved IANA zone
  dt_obj = datetime.datetime.fromisoformat(dt_str).replace(tzinfo=tz)
  timestamp = int(dt_obj.timestamp())
  # filtering:
  h_local = datetime.datetime.fromtimestamp(h["dt"], tz=tz)
  ```
- **Zero-coercion `or` defaults** (`:234–241, 277–283`): genuine 0.0 °C / 0 kph become 22 °C / 10 kph → use `v if v is not None else default`.
- **Null Island fallback** (`:310–311, 460`): unmatched race yields lat/lon (0,0) and a real ocean forecast — return `_no_weather_response()` (already defined, never called) + loud warning instead.
- **Fabricated fallback climatology** (`:169–183`): humidity hardcoded 60%, invented PoP — pull `relative_humidity_2m_mean` from archive API, derive PoP from fraction of wet days, emit `None` when sample too thin (and mark `"quality": "degraded"` in payload).
- **Unguarded parallel-array indexing** (`:330–339`): zip `time` with `.get(field, [])` lengths safely.
- **Dead OWM remnants**: remove `OWM_API_KEY`/URL imports (module is Open-Meteo-only) + corresponding config constants.

### B07.4 🟢 Sanity check hygiene
- Section 10 writes fake records under `season=9999` into real history files and never cleans → use a temp directory override or clean up in finally.
- Section 9 mutates real chip state → operate on a temp state path (add optional `path` param to chip persistence funcs).
- Relax brittle `N_FEATURES == 56` assertion to match post-fix count (and update after Phase 2 changes).
- `analyse_results.py`: accept CSV path via argv; derive baseline from driver count in data.

---

<a name="phase-8"></a>
## Phase 8 — Backtest De-biasing (Critical)

### B08.1 🔴 Lookahead bias — end-of-season knowledge used for every round
- **Leak site A:** `backtest.py:243–246` computes season-FINAL standings + full multiseason form ONCE per year.
- **Leak site B:** `:268–279` reuses them for every round.
- **Also:** `main()` lines 370–373 set `predictor._driver_form` from full-history aggregates.

**Implementation (in order):**
1. Per-round standings — move inside the loop:
   ```python
   prev_rnd = rnd - 1
   if prev_rnd >= 1:
       drv_st = get_driver_standings(year, prev_rnd)
       ctor_st = get_constructor_standings(year, prev_rnd)
   else:
       # Round 1: prior season's final standings
       drv_st = get_driver_standings(year - 1)
       ctor_st = get_constructor_standings(year - 1)
   ```
   APIs already support round scoping (`data_fetcher.py:383–384, 417–418`) and per-round cache keys exist.
2. Form cutoff — add `as_of_round` to `compute_multiseason_driver_form` / `compute_multiseason_constructor_stats`:
   ```python
   def compute_multiseason_driver_form(year, seasons=None, as_of_round=None):
       ...
       if as_of_round is not None:
           results = [r for r in results if r.get("round", 0) <= as_of_round]
   ```
   Call with `as_of_round=prev_rnd` (or omit current-year component entirely for round 1).
3. ELO replay already processes races sequentially — verify `year_elo_sys` isn't pre-built on full season before the loop (audit `build_from_history` call site).
4. Pre-warm per-round standings caches via an extended `warm_cache` pass BEFORE benchmarking (per-round keys have 3h TTL; hitting Jolpica mid-backtest is slow).
5. After fixing, **re-run BACKTEST.bat and expect metrics to worsen** (higher MAE, lower Spearman). Document the honest numbers in PROJECT_ANALYSIS.md; archive the old (inflated) numbers for comparison.
- **Verify:** tamper check — modify a late-season result in cache and confirm early-round predictions DON'T change.

### B08.2 🟡 Remaining backtest items
- Monkey-patching `cfg_module.HISTORICAL_SEASONS` during training — replace with explicit parameter threading through `F1Predictor.train(seasons=...)`.
- Align fallback ensemble weights (0.35/0.35/0.30) with predictor's fallback (equal thirds) or centralize in one constant.
- Delete dead import of `SPRINT_ROUNDS`; use shared sprint table (Phase 1).
- `analyse_results.py` hardcoded baseline MAE 6.67 → compute from actual driver count.

---

<a name="phase-9"></a>
## Phase 9 — Hardening & Cleanup

### B09.1 🟢 Config hygiene
- Delete dead `OWM_BASE_URL`/`OWM_GEO_URL`/`OWM_API_KEY` (+ weather.py imports).
- Add `"madring"` SC/VSC probability entries (currently missing → Madrid lookups miss).
- Resolve Barcelona/Spanish GP dual entries (round 7 `spain` vs round 14 `madring`): mark Barcelona `"round": 0` like other placeholders or gate by season.
- Header says 24 races; `CIRCUITS` has 25 (3 with `round: 0`) — fix comment; audit anything iterating rounds.
- Remove misleading identity encodings (`DOWNFORCE_LEVEL_ENC = {1:1,...}`).
- `HISTORICAL_SEASONS` includes 2026 — enforce the "caller must skip current season" contract in `train()` itself (assert/filter) rather than trusting callers.

### B09.2 🟢 Dead code removal (after grep-confirming zero callers)
| Item | File |
|---|---|
| `normalize_telemetry`, `get_session_feature_weights` | data_fetcher.py |
| `display_race_prediction`, `display_monte_carlo`, `display_differential_picks`, et al. | main.py |
| `pause_and_clear`, `_medal` stubs | main.py |
| orphaned `import copy` | main_logic.py:269 |
| `_no_weather_response` (activate instead per B07.3) | weather.py |
| `EloRatingSystem` deprecated subclass, `datetime` import | elo_ratings.py |
| `_count_team_drivers`, `TOP_PERFORMER_GAIN_THRESHOLD` (use per B05.3) | fantasy_optimizer.py |
| `pandas` imports | bayesian_model.py, tire_model.py |
| `TimeSeriesSplit` import | predictor.py |
| `style.css.bak`, `patch_layout.txt` | ui/ |
| `test_fastf1.py`, `test_serialize.py` (one-off leftovers) | root — move to scratch/ or delete |

### B09.3 🟢 Predictor misc
- Comment/code drift: guardrail "soft cap at P12" vs `max(pred, 10.0)` (predictor.py:1202–1204) — align.
- Weather `"unknown"` encoding inconsistent between `WEATHER_ENC` (1) and `CONDITION_ENC` lookup defaulting to dry — unify.
- Compound heuristic (`SOFT if over_enc >= 3`): derive from `tire_degradation` track feature instead.
- `fresh_tires_avail` round lookup uses circuit-config round (0 for placeholders) — take from `race_info`.
- Stale docstrings: "32/35 features" → generate count dynamically in logs.

### B09.4 🟢 UI polish
- Inline `onclick` string interpolation (app.js:1036–1038) → `data-*` attributes + event delegation.
- style.css: duplicate `:root` variables; rename misleading `--f1-red: #00F0FF`.

### B09.5 🟢 RUN_CLI_Legacy.bat
Dependency install line omits fastapi/uvicorn/pulp/tensorflow/juliacall/sse-starlette — replace with `pip install -r requirements.txt`.

---

<a name="impact"></a>
## Impact Analysis — How the Fixes Affect Performance & Accuracy

### 1. Accuracy effects

**Grouped by mechanism, with expected direction and rough magnitude. Magnitudes are estimates — the de-biased backtest (B08.1) is the only trustworthy measurement, and it must be re-run after Phases 1–3.**

#### A. Fixes that should IMPROVE real prediction quality

| Fix | Mechanism | Expected effect |
|---|---|---|
| **B02.1 circuit matcher** | Circuit history, SC probability, downforce/overtaking/tire-deg encodings currently Australia's for *every* training sample AND mismatched vs inference. After fix, an entire feature family (~8–10 of ~58 features) becomes informative for the first time. | **Largest single accuracy gain in the plan.** Circuit-sensitive predictions (Monaco vs Monza style tracks) should differentiate materially. Expect visible MAE/Spearman improvement on top of honest baselines. |
| **B01.1 DNF parsing** | Removes fake P20s from form/labels; `dnf_rate`, `avg_position`, overtake deltas become truthful. | Moderate gain, concentrated on backmarkers/lapped finishers; better-calibrated `_estimate_dnf_prob`. |
| **B01.3 real rosters** | Teammate deltas, constructor ELO grouping, reliability lookups stop attributing drivers to teams they weren't in (2021–2025). | Moderate gain on constructor-level features and backtest validity; phantom-driver artifacts disappear. |
| **B01.4 sprint detection** | Sprint points adjustments and labels apply to the correct rounds. | Small but real gain; removes systematic error on 6 rounds/season. |
| **B03.1 temporal context vector** | The LSTM branch stops being a constant stub; qualifying position actually feeds momentum predictions. | Uncertain sign until retrained — could be a genuine gain or neutral/noise. Validate via A/B backtest (feature flag); drop the feature if it doesn't earn its keep. |
| **B03.2 Elo freshness + roster-scoped pool** | Ratings reflect latest results within hours, not up to 7 days; rank/z-score features compare drivers against actual peers instead of retired ghosts since 2021. | Small-to-moderate gain early season (stale off-season ratings were most wrong then) and cleaner feature distributions. |
| **B07.1 EWMA inversion** | Bias corrections track recent driver form instead of stale early-season errors — the feedback loop actually converges toward current reality. | Compounding small gain across a season; largest effect from race ~5 onward. |

#### B. Fixes that make fantasy-point EVs CORRECT (values will shift, direction varies)

These change the numbers users see. They are corrections, not tuning:

| Fix | Direction of EV change |
|---|---|
| B02.9 Q3/Q2 exclusivity + FL top-10 rule | **Down** ~0–1 pt for top-10 qualifiers, ~0–0.5 for midfield (FL eligibility). Systematic downward correction of an upward bias. |
| B02.9 pole bonus wired in | **Up** +0.5 for pole sitters only. |
| B02.2 grid-penalty double-count removed | Penalized drivers' EV **up** (positions-gained term was being crushed twice). |
| B03.3 Bayesian zeros-as-DNF fix | Midfield drivers' EV **up** substantially (ψ was inflated; some drivers deflated ~50%). Front-runners barely affected. |
| B03.4 breakdown invariant | No EV change; dashboards/optimizer stop disagreeing with headline totals. |
| B04.4 p_top3/p_points_finish from positions | Reported probabilities become meaningful; no EV change. |

Net: absolute EVs get more accurate but generally **lower** — expect fewer "value" flags from the optimizer until thresholds are re-tuned (noted in B05.3).

#### C. Fix that will make MEASURED metrics WORSE — by design

| Fix | Effect |
|---|---|
| **B08.1 backtest lookahead removal** | Every published metric (MAE, RMSE, Spearman, winner/top-N hit rates) will **degrade** — likely noticeably (est. +15–30% MAE, several points of Spearman). This is not regression: the old numbers measured leakage, not skill. The new numbers are the honest baseline all future improvements must beat. Document both in PROJECT_ANALYSIS.md. |
| B03.5 scaler-in-fold | CV RMSE slightly worse (honest), final-model accuracy unchanged. |
| B02.6/B02.5 leakage gates | Training signal slightly weaker where leakage existed; live/inference consistency improves — net win where it matters (real predictions). |

**Overall accuracy verdict:** real-world prediction quality should improve meaningfully (circuit features alone justify the effort), while reported benchmark numbers will first DROP (leakage removal) and then recover above pre-fix levels as corrected features take effect. Do not compare post-fix backtests against pre-fix ones without noting the baseline change.

### 2. Performance effects

| Area | Change | Runtime/memory effect |
|---|---|---|
| **Pipeline wall-clock (F1 Fantasy.bat / server run)** — B06.7 `load_context_light()` | Eliminates 3 lookahead reloads + 1 restore reload; each previously refetched standings over network and rebuilt Glicko-2 across all historical seasons. Also avoids 3 duplicate full Monte Carlo passes if sims count is normalized per-race. | **Biggest win: est. 30–50% faster pipeline runs**, dominated by network/Glicko cost today. |
| **Request throttle lock (B01.5)** | Serializes the 4 warm_cache workers correctly (they currently burst 4× past the limit). | warm_cache wall-clock **increases** somewhat (correct pacing) — the trade is avoiding Jolpica 429s/IP bans that would cost far more time. Atomic writes prevent corrupt-cache crashes/retries. |
| **Julia gating (B04.2 short-term)** | Today every ≥5000-sim run throws BoundsError, pays exception handling, then silently reruns Python anyway. Gating removes that overhead. | Neutral-to-slightly-faster at n≥5000. If Julia parity is completed later (B04.2 long-term), n≥5000 runs get **significantly faster** than pure Python. Until then, consider capping default sims below 5000 to stay on one engine. |
| **Elo cache keyed on latest round (B03.2)** | Rebuilds once per newly completed race instead of at most weekly. | One Glicko rebuild over ~6 seasons costs seconds and is cached after; negligible steady-state impact, big correctness win. |
| **track_features_loader negative caching (B03.7)** | Stops re-parsing invalid JSON per driver per call. | Minor CPU savings during training loops; unbounded `_validation_errors` growth eliminated. |
| **Model cache key hardening (B02.4)** | Key string longer; hash cost identical. | None. One-time full model retrain required after Phase 2/3 merge. |
| **Server memory (B06.2.4)** | `run_results` capped to last 10 runs; queues popped after terminal event. | Stops unbounded growth (each result payload is multi-MB) — matters for long-lived sessions; no effect on single runs. |
| **SSE polling fallback (B06.2.2)** | Only activates on stream failure. | Zero cost in happy path; converts "locked UI → page reload" into automatic recovery. |
| **Watchdog job-guard (B06.1)** | Counter increments/decrements per job. | Negligible; prevents mid-download kills that today waste entire multi-minute warm_cache sessions. |
| **Backtest (B08.1)** | Per-round standings fetches replace one fetch per year; needs one-time pre-warm download pass. | First post-fix backtest slower (cache warming); subsequent runs comparable or faster via permanent caches. Honest metrics may also allow shorter seasons in future tuning iterations. |
| **Scoring module extraction (B02.9)** | Shared code path replaces duplicated logic. | None measurable; prevents future drift. |

**Net performance summary:** interactive pipeline runs get **faster** (lookahead fix dominates); cache-warming gets **slower but safe** (correct rate limiting); server long-run memory becomes **bounded**; MC stays as-is unless Julia parity is pursued. There is no fix in this plan whose steady-state runtime cost is significant.

### 3. Rollout risk notes

- **One-time costs:** full model retrain + cache purge (Appendix A) + backtest re-baseline. Budget ~an evening of machine time.
- **Behavioral changes users will notice:** lower EVs (scoring corrections), different chip advice timing (EWMA inversion), dream teams that respect actual budget, honest backtest numbers.
- **Highest-risk fixes needing A/B validation:** B03.1 (temporal context — could hurt), B05.3 threshold tightening (transfer suggestions will change sharply), B08.1 (expect and communicate metric drop).

---

<a name="appendix-a"></a>
## Appendix A — Cache Invalidation Matrix

Fixes that change cached-data semantics. Apply in this order during rollout:

| Fix | Cache affected | Action |
|---|---|---|
| B01.1 DNF parsing | `cache/api/jolpica__*Results*.json` (permanent!) | Purge all `*Results*` files; re-run VERIFY_CACHES.bat |
| B01.2 form λ in key | `driver_form_*`, multiseason keys | Old keys simply orphan; optionally delete `driver_form_*` |
| B01.6 grid penalties | `grid_penalties_*` | Delete all; TTL reduced to 6h |
| B02.1 circuit matcher | none (pure compute) | n/a |
| B02.4 model schema | `cache/models/*.pkl` | Delete entire dir (auto-rebuilds on next train) |
| B03.1 temporal retrain | `cache/models/temporal_*` | Delete |
| B03.2 Elo key change | `cache/elo/*` | Delete (auto-rebuild) |
| B08.1 per-round standings | `jolpica__{year}_{rnd}_*Standings*` | Pre-warm via extended warm_cache |
| B07.3 weather tz | `cache/weather/*` | Delete (cheap to re-fetch) |

**Rollout rule:** after any Phase 1/2/3 merge, run in sequence: `VERIFY_CACHES.bat` → `sanity_check.py` → `BACKTEST.bat` → one live `F1 Fantasy.bat` smoke run.

---

<a name="appendix-b"></a>
## Appendix B — Regression Test Strategy

Currently there is no pytest suite. Introduce minimal tests alongside fixes (tests/ directory):

1. **test_scoring_rules.py** — pure-function tests for the shared `scoring.py` (B02.9): quali bonuses (Q3 vs Q2 exclusivity, pole bonus), FL top-10 eligibility, DNF penalty consistency between deterministic and MC paths.
2. **test_circuit_matching.py** — every race name 2021–2026 from cached schedules maps to the correct `CIRCUITS` key (guards B02.1 forever).
3. **test_classification.py** — `classify_result` table-driven cases ("Finished", "+1 lap"…" +46 laps", "Disqualified", "Withdrawal", "Accident").
4. **test_group_alignment.py** — inject a raising `_build_features` mid-race; assert `sum(groups) == len(X)`.
5. **test_chip_state.py** — mark/reset/deepcopy isolation; concurrent mark via threads; corrupt-file recovery.
6. **test_no_lookahead.py** — backtest harness assertion: perturb round-N result, round-1..N−1 features unchanged (guards B08.1).
7. **test_scraper_contracts.py** — `_try_html_scrape` returns `(dict, dict)` on both paths with fixture HTML.

Run via `python -m pytest tests/ -q`; wire into sanity flow before BACKTEST.bat.

---

## Suggested Execution Order (dependency-aware)

```
Week 1:  Phase 0  (secrets FIRST — independent)
         Phase 1  (data truth; everything downstream depends on it)
Week 2:  Phase 2  (predictor core; requires Phase 1 data fixed)
         Phase 3  (model subsystems; overlaps predictor work)
Week 3:  Phase 4 + Phase 5 (MC + strategy layers, parallelizable)
         Phase 6  (server/UI)
Week 4:  Phase 7 + Phase 8 (feedback loop, then de-biased backtest)
         Phase 9  (cleanup sweep)
Final:   Full cache purge → warm → sanity → BACKTEST.bat → publish honest metrics
```

**Expected outcome after all phases:** honest (lower) backtest metrics, correct circuit-specific modeling, correct fantasy-point EVs, no silent state corruption, and a regression net preventing reintroduction.
