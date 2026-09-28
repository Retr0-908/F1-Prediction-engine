# Multi-Stage Weekend Prediction & Consolidated Model Analytics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement weekend stage-aware predictions (Pre-Practice, Post-Practice, Post-Qualifying), an interactive comparison table showing prediction evolution across stages vs actual race results, and consolidate the duplicate "Model Health" and "Past Archive" screens into a unified, bug-free, Apple-grade "Archive & Model Analytics" hub.

**Architecture:**
- Extend `engine/models/predictor.py` and `engine/serving/pipeline.py` to support explicit weekend stages (`pre-practice`, `post-practice`, `post-quali`), ensuring practice telemetry and qualifying grid data are used strictly when appropriate.
- Create `engine/analysis/stage_comparator.py` to generate and cache multi-stage prediction matrices vs actual race results with progression metrics (MAE, rank correlation, driver deltas).
- Expose REST API endpoint `GET /api/analysis/stage-comparison/{round_num}` in `engine/serving/server.py`.
- Consolidate `#past-archive-screen` and `#pred-analysis-screen` in `ui/index.html` into a unified `#archive-analytics-screen` with a single sidebar link (`#nav-archive-analytics`), containing 4 clean tabs: **Stage Comparison**, **Model Health & Trends**, **Past Predictions**, and **Historical Races**.
- Update `ui/apple-springs.js` `NAV_ORDER` to prevent navigation directionality glitches, and bind Chart.js resize observers so charts never collapse to 0px upon tab switching.
- Update the top-bar mode selector to provide explicit weekend stage selection (`Auto`, `Pre-Practice`, `Post-Practice`, `Post-Qualifying`).

**Tech Stack:** Python 3.10+, FastAPI, NumPy, Pandas, Vanilla JavaScript (ES6+), CSS3 with Glassmorphism, Chart.js, HTML5.

## Global Constraints & UI Anti-Bug Invariants
- **No Canvas Collapse Bug**: Whenever a tab containing a Chart.js canvas (`accuracy-trend-chart`, `past-race-telemetry-chart`) is activated from a hidden state (`display: none`), explicitly call `chart.resize()` and `chart.update()`.
- **No Nav Indicator Drift**: Ensure `ui/apple-springs.js` `NAV_ORDER` includes `'archive-analytics-screen'` so index calculations never return `-1`.
- **Sticky Column Alignment**: On mobile and tablet horizontally-scrollable tables, make the first two columns (Finish Pos & Driver Name) sticky (`position: sticky; left: 0; background: var(--bg-card); z-index: 5`) so context is never lost while scrolling across stages.
- **Tabular Numerics**: Apply `font-variant-numeric: tabular-nums;` to all position and points columns to guarantee numbers stay perfectly aligned without jitter.
- **Zero Overflow & Text Collision**: All stage column headers and cells must use explicit min/max width constraints, clean line breaks, and responsive padding.
- **100% Green Test Suites**: All existing and new smoke/contract tests must pass (`test_ui_smoke.py`, `test_contracts.py`).
- **Nuclear Secret Scanner**: 0 secrets or tokens committed.
- **Git Push Cadence**: Strictly tracked via `.agents/git_tracker.json`.

---

### Task 1: Multi-Stage Prediction Engine & Stage-Aware Pipeline

**Files:**
- Modify: `engine/models/predictor.py:358-375, 465-495`
- Modify: `engine/serving/pipeline.py:40-56, 330-365`
- Test: `engine/tools/tests/test_contracts.py`

**Interfaces:**
- Consumes: `options["mode"]` with values `"auto" | "pre-practice" | "post-practice" | "pre-quali" | "post-quali"`
- Produces: Predictions using strictly appropriate data per stage, and saves stage-tagged JSON to `output/stages/race_{round}_{name}_{stage}.json`.

- [ ] **Step 1: Write failing contract test for stage-aware predictor context**

In `engine/tools/tests/test_contracts.py`:
```python
    def test_predictor_stage_isolation(self):
        """Verify pre-practice mode strictly ignores practice pace and quali data."""
        from engine.models.predictor import F1Predictor
        from engine.core.data_fetcher import get_race_by_round
        race = get_race_by_round(15, 2026)
        p = F1Predictor()
        p.load_context(race.get("circuit_config", {}), {}, mode="pre-practice", race_info=race)
        self.assertEqual(len(p._practice_pace), 0, "Pre-practice must have empty practice pace")
        self.assertEqual(len(p._actual_grid), 0, "Pre-practice must have empty actual grid")
        self.assertEqual(p._mode, "pre-practice")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest engine/tools/tests/test_contracts.py -k test_predictor_stage_isolation`
Expected: FAIL

- [ ] **Step 3: Implement stage isolation in `engine/models/predictor.py`**

In `predictor.py` (`load_context`):
- If `mode == "pre-practice"`:
  - Do NOT load practice pace (`self._practice_pace = {}`, `self._practice_session_name = "N/A"`).
  - Do NOT load qualifying sectors or actual grid.
- If `mode in ("post-practice", "pre-quali")`:
  - Load practice pace (`self._practice_pace`).
  - Do NOT load qualifying sectors or actual grid (`self._actual_grid = []`).
- If `mode in ("post-quali", "race-day")`:
  - Load practice pace, qualifying sectors, grid penalties, and actual qualifying results.

- [ ] **Step 4: Update `engine/serving/pipeline.py` stage handling and archival**

In `pipeline.py`:
- Map `user_mode`:
  - `"auto"`: detect if qualifying completed (`post-quali`), else if practice completed (`post-practice`), else (`pre-practice`).
  - `"pre-practice"`, `"post-practice"`, `"post-quali"` handled explicitly.
- In post-run saving:
  - Create directory `output/stages/` if not exists.
  - Save `output/stages/race_{round}_{name}_{stage}.json` containing the stage tag and predictions.
  - Keep updating primary `output/race_{round}_{name}.json`.

- [ ] **Step 5: Run tests and verify they pass**

Run: `python -m unittest engine/tools/tests/test_contracts.py -k test_predictor_stage_isolation`
Expected: PASS

- [ ] **Step 6: Commit Task 1**

```bash
git add engine/models/predictor.py engine/serving/pipeline.py engine/tools/tests/test_contracts.py
git commit -m "feat(engine): add strict stage-aware predictor context and stage archival"
```

---

### Task 2: Stage Comparison Analytics Module & REST Endpoints

**Files:**
- Create: `engine/analysis/stage_comparator.py`
- Modify: `engine/serving/server.py:370-405`
- Test: `engine/tools/tests/test_ui_smoke.py`

**Interfaces:**
- Consumes: Stored stage prediction files and Jolpica actual race classification.
- Produces: `GET /api/analysis/stage-comparison/{round_num}` returning driver matrix and MAE evolution across stages.

- [ ] **Step 1: Write failing smoke test for stage comparison endpoint**

In `engine/tools/tests/test_ui_smoke.py`:
```python
    def test_stage_comparison_api(self):
        """Verify /api/analysis/stage-comparison endpoint structure and calculation."""
        res = self.client.get("/api/analysis/stage-comparison/15")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("matrix", data)
        self.assertIn("metrics", data)
        self.assertIn("round", data)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py -k test_stage_comparison_api`
Expected: FAIL (404 Not Found)

- [ ] **Step 3: Implement `engine/analysis/stage_comparator.py`**

Create `stage_comparator.py`:
1. `get_stage_comparison(round_num: int, season: int = CURRENT_SEASON) -> dict`:
   - Checks `output/stages/` for pre-practice, post-practice, and post-quali predictions.
   - If a stage file is missing for a completed/cached round, dynamically computes that stage's prediction order via `F1Predictor` with the appropriate `mode`.
   - Fetches actual race results via `get_race_results(season, round_num)`.
   - For each driver:
     - `driver`: Name
     - `team`: Constructor name
     - `pre_practice`: `{"pos": round(p, 1), "error": round(abs(p - actual), 1), "pts": round(pts, 1)}`
     - `post_practice`: `{"pos": round(p, 1), "error": round(abs(p - actual), 1), "delta": round(p_post - p_pre, 1), "pts": round(pts, 1), "fp_delta": fp_delta}`
     - `post_quali`: `{"pos": round(p, 1), "error": round(abs(p - actual), 1), "delta": round(p_q - p_post, 1), "pts": round(pts, 1), "grid": grid_pos}`
     - `actual`: `{"pos": int(actual_pos), "pts": float(actual_pts), "status": "Finished" / "DNF"}`
   - Computes summary metrics:
     - `pre_practice_mae`, `post_practice_mae`, `post_quali_mae`
     - `pre_practice_spearman`, `post_practice_spearman`, `post_quali_spearman`
     - `accuracy_gain_pct` (improvement from pre-practice to post-qualifying)
     - `top_improver` (driver whose prediction accuracy gained the most post-FP)
     - `biggest_upset` (driver whose race result defied post-quali predictions)
   - Caches computed report to `output/stages/comparison_round_{round_num}.json`.

- [ ] **Step 4: Register endpoints in `engine/serving/server.py`**

In `server.py`:
```python
@app.get("/api/analysis/stage-comparison/{round_num}")
def get_stage_comparison_api(round_num: int, season: Optional[int] = None):
    from engine.analysis.stage_comparator import get_stage_comparison
    from engine.core.config import CURRENT_SEASON
    target_season = season if season else CURRENT_SEASON
    res = get_stage_comparison(round_num, target_season)
    return res

@app.get("/api/analysis/available-rounds")
def get_available_rounds_api():
    from engine.core.data_fetcher import get_season_schedule, race_has_happened
    from engine.core.config import CURRENT_SEASON
    sched = get_season_schedule(CURRENT_SEASON)
    completed = [r for r in sched if race_has_happened(r, CURRENT_SEASON)]
    return {"status": "ok", "rounds": completed}
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py -k test_stage_comparison_api`
Expected: PASS

- [ ] **Step 6: Commit Task 2**

```bash
git add engine/analysis/stage_comparator.py engine/serving/server.py engine/tools/tests/test_ui_smoke.py
git commit -m "feat(analysis): implement stage comparator analytics and REST endpoints"
```

---

### Task 3: Consolidate "Model Health" and "Past Archive" into Unified Screen & Fix Nav Order

**Files:**
- Modify: `ui/index.html:56-65, 620-770`
- Modify: `ui/apple-springs.js:111-125`
- Modify: `ui/style.css:790-820, 1600-1650`
- Modify: `ui/app.js:100-120, 2690-2720`
- Test: `engine/tools/tests/test_ui_smoke.py`

**Interfaces:**
- Consumes: Combined Screen 9 (`past-archive-screen`) and Screen 10 (`pred-analysis-screen`)
- Produces: Single unified Screen `#archive-analytics-screen` with unified sidebar navigation `#nav-archive-analytics`, updated `NAV_ORDER`, and 4 distinct sub-tabs.

- [ ] **Step 1: Write test verifying unified screen, NAV_ORDER update, and removal of redundant sidebar nav buttons**

In `engine/tools/tests/test_ui_smoke.py`:
```python
    def test_consolidated_archive_analytics_screen(self):
        """Verify Model Health and Past Archive are consolidated into a single unified screen."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("archive-analytics-screen", res.text)
        self.assertIn("nav-archive-analytics", res.text)
        self.assertNotIn("nav-past-archive", res.text)
        self.assertNotIn("nav-pred-analysis", res.text)

        res_springs = self.client.get("/static/apple-springs.js")
        self.assertIn("archive-analytics-screen", res_springs.text)
        self.assertNotIn("past-archive-screen", res_springs.text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py -k test_consolidated_archive_analytics_screen`
Expected: FAIL

- [ ] **Step 3: Update Navigation & Combine HTML Structure in `ui/index.html`**

1. In `#main-nav .nav-links`:
   - Replace separate `#nav-past-archive` and `#nav-pred-analysis` with:
     `<a href="#" data-target="archive-analytics-screen" class="nav-btn" id="nav-archive-analytics">Archive & Analytics</a>`
2. Update Mode Selector dropdown in header:
   ```html
   <select id="mode-selector" class="mode-select">
       <option value="auto" style="color: black;">⚡ Auto (Live Detection)</option>
       <option value="pre-practice" style="color: black;">1. Pre-Practice (Baseline)</option>
       <option value="post-practice" style="color: black;">2. Post-Practice (FP Telemetry)</option>
       <option value="post-quali" style="color: black;">3. Post-Qualifying (Grid Locked)</option>
   </select>
   ```
3. Replace `#past-archive-screen` and `#pred-analysis-screen` with unified `#archive-analytics-screen`:
   - Title: `ARCHIVE & MODEL ANALYTICS`
   - Subtitle: `Cross-stage prediction evolution, season accuracy verification, model health, and past race telemetry.`
   - Cupertino segmented tab row:
     - `tab-btn active` -> `data-tab="tab-stage-comparison"`: **Stage Comparison**
     - `tab-btn` -> `data-tab="tab-model-health"`: **Model Health & Biases**
     - `tab-btn` -> `data-tab="tab-past-predictions"`: **Past Predictions**
     - `tab-btn` -> `data-tab="tab-past-races"`: **Race Telemetry**
   - Retain all existing Model Health components (Post-race verification, accuracy trend chart, EWMA bias table, circuit feature blend weights).
   - Retain all existing Past Archive components (Past prediction selector, year selector, race results, and tire stint degradation chart).

- [ ] **Step 4: Update `NAV_ORDER` and Tab Resizing in `ui/apple-springs.js`**

1. Update `NAV_ORDER` array:
   ```javascript
   const NAV_ORDER = [
     'dashboard-screen',
     'team-screen',
     'prices-screen',
     'standings-screen',
     'archive-analytics-screen',
     'settings-screen',
     'analysis-screen',
     'results-screen'
   ];
   ```
2. In `attachTabSprings()`: When `tab-model-health` or `tab-past-races` becomes active, trigger a window resize event (`window.dispatchEvent(new Event('resize'))`) or invoke `Chart.getChart(canvas).resize()` so Chart.js canvases never collapse to 0px height.

- [ ] **Step 5: Update JavaScript Routing & Controllers in `ui/app.js`**

- In navigation router (`setupNavigation`):
  - When switching to `archive-analytics-screen`, initialize active sub-tab (or load `initStageComparison()`).
- Bind tab switcher to toggle sub-tabs seamlessly using existing Apple-spring transitions.

- [ ] **Step 6: Run tests and verify they pass**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py -k test_consolidated_archive_analytics_screen`
Expected: PASS

- [ ] **Step 7: Commit Task 3**

```bash
git add ui/index.html ui/apple-springs.js ui/app.js ui/style.css engine/tools/tests/test_ui_smoke.py
git commit -m "feat(ui): consolidate Model Health and Past Archive into unified Archive & Analytics screen"
```

---

### Task 4: Interactive Multi-Stage Comparison UI View & Styling

**Files:**
- Modify: `ui/index.html` (under `tab-stage-comparison`)
- Modify: `ui/app.js` (add `initStageComparison()`, `loadStageComparison()`, `renderStageComparisonTable()`)
- Modify: `ui/style.css` (add comparison table, evolution pills, metric KPI styling)
- Test: `engine/tools/tests/test_ui_smoke.py`

**Interfaces:**
- Consumes: `GET /api/analysis/stage-comparison/{round_num}`
- Produces: Interactive table comparing Pre-Practice, Post-Practice, Post-Qualifying against Actual Finish with color-coded evolution badges, sticky headers, and MAE progress bar.

- [ ] **Step 1: Write test verifying stage comparison table markup and JS handlers**

In `engine/tools/tests/test_ui_smoke.py`:
```python
    def test_stage_comparison_ui_elements(self):
        """Verify presence of stage comparison controls and table container."""
        res_html = self.client.get("/")
        self.assertEqual(res_html.status_code, 200)
        self.assertIn("tab-stage-comparison", res_html.text)
        self.assertIn("stage-round-selector", res_html.text)
        self.assertIn("stage-comparison-table", res_html.text)

        res_js = self.client.get("/static/app.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertIn("initStageComparison", res_js.text)
        self.assertIn("loadStageComparison", res_js.text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py -k test_stage_comparison_ui_elements`
Expected: FAIL

- [ ] **Step 3: Build Stage Comparison View Markup in `ui/index.html`**

Inside `<div id="tab-stage-comparison" class="tab-content active">`:
- Filter header (`.stage-comparison-toolbar`):
  - Round dropdown (`#stage-round-selector`) dynamically populated with completed 2026 races.
  - "📊 Compare Stages" button.
  - Visual legend pills: Pre-Practice, Post-Practice, Post-Qualifying, Actual Finish.
- Top KPI Summary Banner (`#stage-summary-cards`):
  - Card 1: Pre-Practice Baseline MAE (e.g. `±3.4 pos`).
  - Card 2: Post-Practice Telemetry MAE (`±2.5 pos` with `▲ 26% improvement` pill).
  - Card 3: Post-Qualifying Grid MAE (`±1.6 pos` with `▲ 53% overall gain` pill).
  - Card 4: Prediction-to-Actual Spearman $\rho$ rank correlation.
- Interactive Comparison Table (`#stage-comparison-table`):
  - Sticky header (`thead th` with glassmorphic backdrop-filter blur).
  - Columns:
    1. FINISH (Podium gold/silver/bronze badges, points badges, or DNF tag).
    2. DRIVER (Driver thumbnail + Name).
    3. TEAM (Team badge with official color).
    4. PRE-PRACTICE (Baseline predicted pos & error).
    5. POST-PRACTICE (Telemetry predicted pos, FP pace delta, error).
    6. POST-QUALI (Grid locked predicted pos, grid penalty delta, error).
    7. ACTUAL RACE (Official finishing pos & Fantasy points).
    8. TRAJECTORY & EVOLUTION (`P3 ➔ P2 ➔ P1` spark progression + Bullseye/Converged status badge).

- [ ] **Step 4: Implement JavaScript Logic in `ui/app.js`**

Add:
- `initStageComparison()`: fetches completed rounds from `/api/analysis/available-rounds`, populates dropdown, automatically selects latest completed race.
- `loadStageComparison()`: fetches `/api/analysis/stage-comparison/{round_num}`, animates summary metrics, and renders the comparison matrix table.
- Trajectory calculation: computes whether the driver prediction became more accurate at each stage.
- Filter chips: `All Drivers`, `Points (Top 10)`, `Podium (Top 3)`, `DNFs`.
- Skeleton loading placeholder with shimmer animation during fetch.
- Color coding:
  - Green (`var(--color-green)`): Spot-on or within $\pm 1$ position.
  - Amber / Yellow (`var(--color-yellow)`): Minor drift ($\pm 2-3$ positions).
  - Red (`var(--f1-red)`): Major error or DNF.
  - Cyan (`var(--telemetry-cyan)`): Post-practice telemetry pace tags.
  - Purple (`var(--accent-purple)`): Post-qualifying grid tags.

- [ ] **Step 5: Add Professional Glassmorphic Table Styling in `ui/style.css`**

Add:
- `.stage-comparison-table`: sticky header, crisp border separation, high legibility, `font-variant-numeric: tabular-nums`.
- `.stage-pill-pre`, `.stage-pill-fp`, `.stage-pill-quali`, `.stage-pill-actual`: distinct visual tags for each stage.
- `.trajectory-arrow`: fluid arrows showing position changes across sessions.
- `.metric-kpi-card`: Cupertino glassmorphism with subtle glow.
- Responsive styles: sticky left column for driver name on scrollable tablet view.

- [ ] **Step 6: Run tests and verify they pass**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py -k test_stage_comparison_ui_elements`
Expected: PASS

- [ ] **Step 7: Commit Task 4**

```bash
git add ui/index.html ui/app.js ui/style.css engine/tools/tests/test_ui_smoke.py
git commit -m "feat(ui): add interactive multi-stage weekend prediction comparison table"
```

---

### Task 5: End-to-End Verification, Asset Cache Bumping & Cadence Check

**Files:**
- Modify: `ui/index.html` (bump cache buster to `?v=12`)
- Test: Full unit and smoke test suites

- [ ] **Step 1: Bump asset version query parameters in `ui/index.html`**

Update `style.css?v=11` -> `style.css?v=12` and `app.js?v=11` -> `app.js?v=12`.

- [ ] **Step 2: Run full unit, contract, and UI smoke test suites**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py engine/tools/tests/test_contracts.py`
Expected: 100% PASS (all tests green)

- [ ] **Step 3: Run nuclear secret scanner**

Run: `python engine/tools/secret_scanner.py`
Expected: PASS with 0 secrets detected.

- [ ] **Step 4: Commit and push per GEMINI.md git cadence**

Update `.agents/git_tracker.json` and push verified changes to `origin main`.
