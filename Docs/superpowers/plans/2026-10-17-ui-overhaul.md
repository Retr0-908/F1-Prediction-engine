# F1 Fantasy Predictor — UI Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the F1 Fantasy Predictor's UI onto one coherent F1-themed design language, make idle states honest, remove or relocate sections that don't earn their place, make the app responsive, and clean up the CSS — without breaking any existing behaviour.

**Architecture:** The design system is rebuilt in `ui/style.css` `:root` (tokens) and inherited by all nine screens; markup changes are limited to the content decisions in the spec (D3–D6) and the nav shell (D7). Motion stays owned by GSAP in `ui/apple-springs.js`, which is extended for the mobile drawer and adjusted so nothing animates at idle. No framework, no bundler, no new dependencies.

**Tech Stack:** Vanilla HTML/CSS/JS (`ui/index.html`, `ui/style.css`, `ui/app.js`, `ui/apple-springs.js`), GSAP (already present) for motion, Python `unittest` + FastAPI `TestClient` for the guarding test suite, FastAPI static serving.

**Spec:** `docs/superpowers/specs/2026-10-17-ui-overhaul-design.md`

## Global Constraints

- **Stack is fixed.** Vanilla HTML/CSS/JS served as static files. No bundler, no build step, no new dependencies.
- **Fonts stay:** `Space Grotesk` (headings), `Inter` (body), `JetBrains Mono` (numerals).
- **One accent.** F1 red is the only UI accent. `--telemetry-cyan` / `--neon-cyan` are retired **as UI accent colour**; they may remain defined as tokens. Telemetry purple/green/amber/red are permitted **as data semantics only**, never as general chrome.
- **All nine screens stay.** Launch, Dashboard, My Team, Prices, Standings, Analysis, Results, Settings & Cache, Archive & Analytics.
- **The suites must stay 100% green** (per `GEMINI.md`):
  `python -m unittest engine/tools/tests/test_ui_smoke.py engine/tools/tests/test_contracts.py`
- **No change to prediction logic or backend endpoints.** No new API routes.
- **GSAP owns `transform`.** CSS must not add `transform` transitions to `.panel` / `.btn` / nav buttons.
- **Existing test-pinned strings** must either survive or have their assertion updated **in the same task** that changes them. Pinned today: `F1 FANTASY PREDICTOR`, `id="pipeline-visualizer"`, `id="mc-iterations-counter"`, `id="ensemble-bar-wrap"`, `id="pipeline-active-phase"`, `initPipelineVisualizer`, `updateEnsembleSegments`, `animateMonteCarloCounter`, `.pipeline-visualizer`, `.ro-col-telemetry`, `.table-responsive-container`, `.card-title`, `.node-label`, `.node-circle`, `@media (max-width: 1360px)`, `nav-spring-indicator` behaviour in `apple-springs.js`.
- **Frequent commits.** One commit per task (message given in the task).

## Review Focus

Inputs and conditions the spec implies but no task's unit tests exercise — each has a verification added to the task that owns the code:

1. **A very long race name** (e.g. "Emilia Romagna Grand Prix", "São Paulo Grand Prix") at ≈375px — must wrap to a second line, never truncate with an ellipsis or overflow its panel. *(Task 5)*
2. **A run that errors mid-stream** (SSE `error` / connection drop) — the Analysis screen must return to a clean standby state, never strand as "running". *(Task 4)*
3. **A reduced-motion user** — with `prefers-reduced-motion: reduce`, no screen shows motion that implies activity, and the drawer still opens/closes. *(Tasks 3, 4)*
4. **Missing or failed weather data** (`/api/weather` unreachable or empty sessions) — the Weekend Weather panel shows a neutral "unavailable" state, never blank or fabricated values. *(Task 5)*
5. **No upcoming race / zero sessions** — Dashboard hero, session-times list and Weekend Weather degrade gracefully with neutral copy, no "undefined" or stuck spinners. *(Task 5)*

---

### Task 1: Baseline and workspace

**Files:**
- Read: `ui/index.html`, `ui/style.css`, `ui/app.js`, `ui/apple-springs.js`
- Read: `engine/tools/tests/test_ui_smoke.py`

**Interfaces:**
- Produces: a known-green baseline commit that every later task diffs against.

- [ ] **Step 1: Confirm the working tree and record the baseline**

Run: `git status --short` and `git log --oneline -1`
Expected: note any pre-existing uncommitted changes. Do **not** discard them.

- [ ] **Step 2: Run the guarding suites and record the result**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py engine/tools/tests/test_contracts.py`
Expected: note the exact pass/skip count. If anything is already failing, stop and report — later "green" claims depend on this baseline.

- [ ] **Step 3: Start the app and capture a "before" reference**

Run the server (per `GEMINI.md` / README) and load the app at ≈375px, ≈768px and ≥1280px.
Expected: screenshots of the three dashboard states and the Analysis idle state, saved outside the repo (e.g. temp dir), for before/after comparison.

- [ ] **Step 4: Create an isolated worktree for the overhaul**

Use the `superpowers:using-git-worktrees` skill (native tool if available, else `git worktree add`).
Expected: a branch `ui-overhaul` in a worktree; all later work happens there.

---

### Task 2: Design tokens (`:root`)

**Files:**
- Modify: `ui/style.css:1-78` (the `:root` block and focus rules)
- Test: `engine/tools/tests/test_ui_smoke.py:72-79`

**Interfaces:**
- Produces: the token set every later task consumes — `--f1-red`, `--f1-red-hover`, `--accent-primary`, surface tokens (`--bg-dark`, `--bg-panel`, `--bg-card`, `--bg-card-hover`), `--border-subtle`, `--border-dim`, text tokens (`--text-primary`, `--text-dim`, `--text-muted`), radius tokens (`--radius-xs/sm/md/lg/pill/panel/inner/badge`), `--space-1..10`, shadow tokens (`--shadow-panel`, `--shadow-floating`).
- Value targets (from the approved mockup, "Modern F1 App"): page `#0A0A0C`; raised card `linear-gradient(180deg,#141418,#0F0F12)`; borders `rgba(255,255,255,.06)`; button gradient `linear-gradient(180deg,#FF2D0F,#D50500)`; flat accent `#E10600`; card radius 16px, button/input radius 12px.

- [ ] **Step 1: Update the failing assertion first**

In `test_ui_telemetry_tokens_and_visualizer_css`, replace the `--telemetry-cyan` assertion with the new accent contract:

```python
self.assertIn("--f1-red", css)
self.assertIn("--accent-primary", css)
self.assertIn(".pipeline-visualizer", css)
self.assertNotIn("f1-drive", css)
```

- [ ] **Step 2: Run it to confirm it fails or passes for the right reason**

Run: `python -m unittest engine.tools.tests.test_ui_smoke.UISmokeTests.test_ui_telemetry_tokens_and_visualizer_css -v`
Expected: FAIL only if `--f1-red`/`--accent-primary` are absent (they exist already, so this may PASS — that is fine; the assertion now guards the new contract).

- [ ] **Step 3: Rewrite the `:root` tokens**

Update the `:root` block in `ui/style.css` to the D1 values above. Keep token **names** stable wherever they already exist (`--f1-red`, `--accent-primary`, `--radius-*`, `--space-*`, `--text-*`) so screens inherit without renaming. Leave `--telemetry-*` defined (data semantics) but stop pointing `--neon-cyan` at UI chrome.

- [ ] **Step 4: Update focus states to the red/white ring (D2)**

Replace the `--neon-cyan` focus outline in the `:focus-visible` rules with a high-contrast ring that reads on dark surfaces (e.g. `outline: 2px solid #FFFFFF; outline-offset: 2px;` plus `box-shadow: 0 0 0 4px rgba(225,6,0,.45)`).

- [ ] **Step 5: Run the suite**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py engine/tools/tests/test_contracts.py`
Expected: PASS (skip count unchanged from baseline).

- [ ] **Step 6: Visual check**

Reload the app; confirm page/card surfaces and focus rings look consistent, and nothing lost its background.
Expected: no unstyled/transparent panels; focus-visible ring clearly visible on the nav, buttons and inputs.

- [ ] **Step 7: Commit**

```bash
git add ui/style.css engine/tools/tests/test_ui_smoke.py
git commit -m "feat(ui): rebuild design tokens for F1 modern theme"
```

---

### Task 3: Global components (panels, buttons, headers, typography)

**Files:**
- Modify: `ui/style.css` (`.panel`, `.panel:hover`, `.panel-header`, `.btn`, `.btn-primary` at ≈210 and ≈2650, `h1/h2/h3`, `.dim-text`)
- Test: `engine/tools/tests/test_ui_smoke.py`

**Interfaces:**
- Consumes: Task 2 tokens.
- Produces: consistent `.panel` (16px radius, soft gradient surface, quiet border), `.panel-header` (sentence case, single accent mark, no cyan), `.btn` / `.btn-primary` (12px radius, red gradient, one definition only).

- [ ] **Step 1: Add a regression assertion for a single `.btn-primary` definition**

Add to `test_ui_smoke.py`:

```python
def test_single_primary_button_definition(self):
    css = self.client.get("/static/style.css").text
    self.assertEqual(css.count(".btn-primary {"), 1)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m unittest engine.tools.tests.test_ui_smoke.UISmokeTests.test_single_primary_button_definition -v`
Expected: FAIL — `.btn-primary {` appears twice today (≈210 and ≈2650).

- [ ] **Step 3: Consolidate and restyle the components**

In `ui/style.css`: delete the duplicate `.btn-primary` (≈2650) and fold any unique declarations into the one at ≈210. Restyle `.panel`, `.panel-header`, `.btn`, `.btn-primary`, `h1/h2/h3` and `.dim-text` to the D1 language (sentence-case headers, red accent, 12px button radius, softer card surfaces). Remove the `!important` on `.panel` radius once duplicates are gone.

- [ ] **Step 4: Run the suite**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py engine/tools/tests/test_contracts.py`
Expected: PASS, including the new single-definition assertion.

- [ ] **Step 5: Visual check across screens**

Load every screen; confirm panel headers, buttons and type are consistent and no button renders cyan.
Expected: primary buttons red everywhere; no cyan chrome.

- [ ] **Step 6: Commit**

```bash
git add ui/style.css engine/tools/tests/test_ui_smoke.py
git commit -m "feat(ui): unify panels, buttons and typography"
```

---

### Task 4: Responsive drawer navigation (D7)

**Files:**
- Modify: `ui/index.html:49-71` (`.app-container`, `#main-nav`, nav actions)
- Modify: `ui/style.css` (`.app-container`, `.sidebar`, `.main-canvas` ≈2854-2880; new `.mobile-topbar`, `.nav-backdrop`, `body.nav-open`; breakpoint matrix ≈2710-2771)
- Modify: `ui/app.js` (new `initMobileNav()`, called from `initApp()` at ≈50)
- Modify: `ui/apple-springs.js:319-412` (`initNavIndicator` — keep vertical indicator; add drawer-aware behaviour)
- Test: `engine/tools/tests/test_ui_smoke.py:185-194`, `:213-224`

**Interfaces:**
- Produces in `ui/app.js`: `function initMobileNav(): void` — wires the topbar hamburger to toggle `body.nav-open`, wires `.nav-backdrop` and nav-link clicks to close, and keeps `#btn-run-analysis` reachable in the topbar.
- Produces in `ui/index.html`: `#mobile-topbar` containing `#nav-toggle` (hamburger), the brand, and a `#mobile-run-analysis` button calling `startAnalysis()`; `#nav-backdrop` element.
- Consumes: existing `showScreen(screenId)` (app.js:210) and `setupNavigation()` (app.js:91) unchanged.

- [ ] **Step 1: Update the nav assertions to the new intent**

Replace `test_nav_indicator_orientation`'s `apple-springs.js` string checks with structural intent:

```python
def test_nav_indicator_orientation(self):
    js = self.client.get("/static/apple-springs.js").text
    html = self.client.get("/").text
    self.assertIn("nav-spring-indicator", js)
    self.assertIn('id="nav-toggle"', html)
    self.assertIn('id="nav-backdrop"', html)
    self.assertIn('id="mobile-run-analysis"', html)
```

Keep `test_consolidated_archive_analytics_screen` as-is (it still holds).

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m unittest engine.tools.tests.test_ui_smoke.UISmokeTests.test_nav_indicator_orientation -v`
Expected: FAIL — the new ids are not in `index.html` yet.

- [ ] **Step 3: Add the topbar and backdrop markup**

In `ui/index.html`, inside `.app-container` before `#main-nav`, add `#mobile-topbar` (hamburger `#nav-toggle`, brand, `#mobile-run-analysis`), and after `#main-nav` add `#nav-backdrop`. Keep `#main-nav` markup otherwise intact (its `.nav-links` and `.nav-actions` move behind the drawer).

- [ ] **Step 4: Implement `initMobileNav()`**

In `ui/app.js`, add `initMobileNav()` (toggle `body.nav-open` on `#nav-toggle`, close on backdrop click and on any `.nav-btn` click) and call it from `initApp()`. Do not remove `setupNavigation()`.

- [ ] **Step 5: Style the drawer**

In `ui/style.css`: hide `#mobile-topbar` ≥1200px; make `.sidebar` a fixed off-canvas panel translated off-screen and slide it in on `body.nav-open` for <1200px; show `#nav-backdrop` only when open; ensure `#mobile-run-analysis` is visible <1200px and hidden ≥1200px.

- [ ] **Step 6: Make the nav indicator drawer-aware**

In `apple-springs.js` `initNavIndicator`, keep the vertical `left: 0; width: 3px;` indicator; ensure it repositions when the drawer opens (recompute on `body.nav-open` toggle) and never renders `bottom: 0; height: 2px;`.

- [ ] **Step 7: Run the suite**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py engine/tools/tests/test_contracts.py`
Expected: PASS.

- [ ] **Step 8: Verification — responsive + reduced motion (Review Focus 3)**

At ≈375px and ≈768px: hamburger opens the drawer over content, backdrop tap and link tap close it, Run Analysis is one tap in the topbar. With OS reduced-motion enabled: the drawer still opens/closes (instantly).
Expected: no horizontal squeeze; content reflows to full width in every state.

- [ ] **Step 9: Commit**

```bash
git add ui/index.html ui/style.css ui/app.js ui/apple-springs.js engine/tools/tests/test_ui_smoke.py
git commit -m "feat(ui): responsive slide-in drawer navigation"
```

---

### Task 5: Analysis screen — idle means idle (D3)

**Files:**
- Modify: `ui/index.html:225-320` (analysis screen: standby state + pipeline/HUD)
- Modify: `ui/app.js` (`startAnalysis` ≈1250, `initPipelineVisualizer` ≈1172, `stopPipelineVisualizer` ≈1188, `animateMonteCarloCounter` ≈1225, `connectStream` ≈1422, `displayResults` ≈1647)
- Modify: `ui/style.css` (`.radar-sweep`, `@keyframes radar-spin`, `.pulse-text`, pipeline/`monty-carlo` styles)
- Modify: `ui/apple-springs.js` (`initStatusPulse` ≈252)
- Test: `engine/tools/tests/test_ui_smoke.py:81-98`, `:158-165`

**Interfaces:**
- Keeps (tests pin these): ids `#pipeline-visualizer`, `#mc-iterations-counter`, `#ensemble-bar-wrap`, `#pipeline-active-phase`; functions `initPipelineVisualizer`, `updateEnsembleSegments`, `animateMonteCarloCounter`.
- Produces: `function setAnalysisIdleState(): void` in `ui/app.js` — puts the screen into standby (stops visualizer, clears counters/labels, shows `#analysis-standby`, hides `#pipeline-visualizer`, sets `#mc-iterations-counter` text to `""`).
- Produces: `function setAnalysisRunningState(): void` — hides `#analysis-standby`, shows `#pipeline-visualizer`, starts `initPipelineVisualizer()`.

- [ ] **Step 1: Add assertions for the idle contract**

Add to `test_ui_smoke.py`:

```python
def test_analysis_idle_standby_contract(self):
    html = self.client.get("/").text
    js = self.client.get("/static/app.js").text
    self.assertIn('id="analysis-standby"', html)
    self.assertNotIn("READY (10,000 SIMS)", html)
    self.assertIn("setAnalysisIdleState", js)
    self.assertIn("setAnalysisRunningState", js)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m unittest engine.tools.tests.test_ui_smoke.UISmokeTests.test_analysis_idle_standby_contract -v`
Expected: FAIL — `analysis-standby` and the two functions do not exist, and the HTML still contains `READY (10,000 SIMS)`.

- [ ] **Step 3: Add the standby markup and remove idle theatre**

In `ui/index.html` analysis screen: add `#analysis-standby` (grey static dot + "Standby — no analysis running" + one-line description + the inputs list + last-run timestamp slot `#analysis-last-run`). Remove the idle text `READY (10,000 SIMS)` (leave the empty `#mc-iterations-counter` element). Keep the pipeline/HUD markup for the running state.

- [ ] **Step 4: Implement the two state functions and call them at the right moments**

In `ui/app.js`: implement `setAnalysisIdleState()` and `setAnalysisRunningState()`; call `setAnalysisRunningState()` at the top of `startAnalysis()` (≈1250) after `showScreen('analysis-screen')`, and `setAnalysisIdleState()` in `stopPipelineVisualizer()` and in the error/finally path of `connectStream()` (so a failed run returns to standby). Populate `#analysis-last-run` from the `/api/predictions` `generated_at` when the screen shows (timestamp only, no duration).

- [ ] **Step 5: Make idle CSS static**

In `ui/style.css`: stop the radar sweep, pulse text and any infinite idle animation — these may only run under a `.running` class that `setAnalysisRunningState()` adds and `setAnalysisIdleState()` removes. Ensure `@media (prefers-reduced-motion: reduce)` kills them regardless.

- [ ] **Step 6: Gate the status pulse**

In `apple-springs.js` `initStatusPulse`, ensure nothing pulses while `#analysis-standby` is visible.

- [ ] **Step 7: Run the suite**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py engine/tools/tests/test_contracts.py`
Expected: PASS, including `test_ui_analysis_screen_canvas_and_hud`, `test_pipeline_visualizer_cleanliness`, `test_ui_app_js_telemetry_controller`.

- [ ] **Step 8: Verification — idle, running, failure, reduced motion (Review Focus 2, 3)**

(a) Load Analysis with no run: no motion, no counter, no "ACTIVE".
(b) Start a run: the counter shows real progress and the pipeline animates.
(c) Force a mid-run error (stop the server / point at a bad run id): the screen returns to standby.
(d) With reduced-motion on: standby has no motion.
Expected: idle never looks active; a failed run never strands as "running".

- [ ] **Step 9: Commit**

```bash
git add ui/index.html ui/app.js ui/style.css ui/apple-springs.js engine/tools/tests/test_ui_smoke.py
git commit -m "feat(ui): analysis screen idle standby and honest run state"
```

---

### Task 6: Dashboard — weather panel, full race name, status fix (D4)

**Files:**
- Modify: `ui/index.html:77-180` (dashboard: hero, `#dash-race-name`, system status, replace `AI INSIGHTS & TOP MOVE`)
- Modify: `ui/app.js` (`loadDashboardData` ≈441, `displayResults` ≈1647 — remove the dead `renderInsightsPanel(data)` call at ≈1779, keep `updateQuickSummary` ≈2220 call)
- Modify: `ui/style.css` (`.hero-text`, `.hero-top`, `.status-val`, new `.weather-panel`)
- Test: `engine/tools/tests/test_ui_smoke.py`

**Interfaces:**
- Produces in `ui/index.html`: `#weekend-weather-panel` with `#weather-summary`, `#weather-session-list`, `#weather-impact-notes`. Replaces the `AI INSIGHTS & TOP MOVE` panel.
- Produces in `ui/app.js`: `function renderWeekendWeather(sessions, summary): void` — renders per-session forecast and the authored impact notes (selected by rain-risk thresholds), and a neutral state when data is missing.
- Consumes: existing `/api/weather` fetch in `loadDashboardData()` (≈488-521) and the race-day weather summary already shown in `#dash-race-stats`.

- [ ] **Step 1: Add assertions**

Add to `test_ui_smoke.py`:

```python
def test_dashboard_weather_panel_and_no_dead_call(self):
    html = self.client.get("/").text
    js = self.client.get("/static/app.js").text
    self.assertIn('id="weekend-weather-panel"', html)
    self.assertNotIn("AI INSIGHTS & TOP MOVE", html)
    self.assertNotIn("renderInsightsPanel", js)
    self.assertIn("renderWeekendWeather", js)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m unittest engine.tools.tests.test_ui_smoke.UISmokeTests.test_dashboard_weather_panel_and_no_dead_call -v`
Expected: FAIL — old panel present, `renderInsightsPanel` still called, new id/fn absent.

- [ ] **Step 3: Replace the panel and fix truncation**

In `ui/index.html`: remove the `AI INSIGHTS & TOP MOVE` panel and `#quick-summary-content`; add `#weekend-weather-panel`. In `ui/style.css`: allow `#dash-race-name` to wrap (remove any ellipsis/nowrap on the hero heading; give `.hero-text` `min-width: 0`), and let the `System Status` `Next Race` value (`.status-val`) wrap instead of clipping.

- [ ] **Step 4: Remove the dead call and render the panel**

In `ui/app.js`: delete `renderInsightsPanel(data);` (≈1779). Keep the `updateQuickSummary(data)` call only if `updateQuickSummary` no longer targets the removed element; otherwise remove that call too. Add `renderWeekendWeather(...)` and call it from `loadDashboardData()` with the weather data already fetched.

- [ ] **Step 5: Run the suite**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py engine/tools/tests/test_contracts.py`
Expected: PASS. Also confirm no console error at `displayResults` time (the throw is gone).

- [ ] **Step 6: Verification — long names, missing data, no race (Review Focus 1, 4, 5)**

(a) Force a long race name at ≈375px: it wraps to two lines, no ellipsis.
(b) Force `/api/weather` to fail/return empty: the panel shows a neutral "weather unavailable" state.
(c) Force no upcoming race: hero, session list and weather panel show neutral copy, no "undefined", no stuck spinner.
(d) Confirm the Optimal Dream Team grid renders after a run (the previously-blocked code path).
Expected: all four behave gracefully.

- [ ] **Step 7: Commit**

```bash
git add ui/index.html ui/app.js ui/style.css engine/tools/tests/test_ui_smoke.py
git commit -m "feat(ui): dashboard weekend weather panel, full race name, status fix"
```

---

### Task 7: Results — legend affordance and merged Driver Analysis (D5)

**Files:**
- Modify: `ui/index.html:322-520` (results: `#race-order-legend`, Analytics tab panels)
- Modify: `ui/app.js` (`toggleResultsLegend` ≈3003, `renderDriverRadar` ≈2450, `setupComparisonTool` ≈2524, `updateComparison` ≈2545)
- Modify: `ui/style.css` (`.legend-panel`, `.legend-grid`, `.dt-*` duplicates ≈987-1009 / ≈2424-2502)
- Test: `engine/tools/tests/test_ui_smoke.py`

**Interfaces:**
- Produces: a compact `?` toggle replacing the full-width legend bar; the legend still toggles via `toggleResultsLegend()`.
- Produces: one `#driver-analysis-panel` combining attribute radar and the comparison tool; keep ids used by `renderDriverRadar` and `setupComparisonTool`.

- [ ] **Step 1: Assert the merged panel**

Add to `test_ui_smoke.py`:

```python
def test_merged_driver_analysis_panel(self):
    html = self.client.get("/").text
    self.assertIn('id="driver-analysis-panel"', html)
    self.assertNotIn("DRIVER ATTRIBUTE ANALYSIS", html)
    self.assertNotIn("DRIVER COMPARISON TOOL", html)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m unittest engine.tools.tests.test_ui_smoke.UISmokeTests.test_merged_driver_analysis_panel -v`
Expected: FAIL.

- [ ] **Step 3: Merge the panels and restyle the legend toggle**

In `ui/index.html`: wrap the radar + comparison markup in `#driver-analysis-panel` with one header; replace the legend bar with a small `?` button that calls `toggleResultsLegend()`. In `ui/style.css`: style the `?` toggle; remove the duplicate `.dream-team-grid` / `.dt-card` / `.dt-rank` / `.dt-thumb` blocks (keep one).

- [ ] **Step 4: Keep the JS handlers working**

In `ui/app.js`: ensure `renderDriverRadar`, `setupComparisonTool` and `updateComparison` still find their elements inside the merged panel; no signature changes.

- [ ] **Step 5: Run the suite**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py engine/tools/tests/test_contracts.py`
Expected: PASS.

- [ ] **Step 6: Verification**

Load Results → Analytics: the merged panel renders radar and comparison; the legend toggles from the `?` and is hidden by default.
Expected: no orphaned panels, no console errors.

- [ ] **Step 7: Commit**

```bash
git add ui/index.html ui/app.js ui/style.css engine/tools/tests/test_ui_smoke.py
git commit -m "feat(ui): merged driver analysis panel and compact legend toggle"
```

---

### Task 8: Archive — rename to "Model Ops" (D6)

**Files:**
- Modify: `ui/index.html:623-660` (archive sub-tab bar; the "Model Health & Biases" button and pane)
- Modify: `ui/app.js` (`switchArchiveAnalyticsSubtab` ≈159; any `tab-model-health` references)
- Modify: `ui/style.css` (sub-tab styling)
- Test: `engine/tools/tests/test_ui_smoke.py`

**Interfaces:**
- Produces: sub-tab labelled `Model Ops` with `data-subtab="tab-model-health"` unchanged (so `switchArchiveAnalyticsSubtab` keeps working).
- Consumes: `renderBiasTable` (app.js:3292) and `renderWeightAdjustments` (app.js:3351) unchanged.

- [ ] **Step 1: Assert the rename**

Add to `test_ui_smoke.py`:

```python
def test_archive_model_ops_subtab(self):
    html = self.client.get("/").text
    self.assertIn("Model Ops", html)
    self.assertNotIn("Model Health & Biases", html)
    self.assertIn('data-subtab="tab-model-health"', html)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m unittest engine.tools.tests.test_ui_smoke.UISmokeTests.test_archive_model_ops_subtab -v`
Expected: FAIL.

- [ ] **Step 3: Rename and tidy**

Update the sub-tab button label to `Model Ops`, keep `data-subtab="tab-model-health"`, and restyle the pane to the new language. Do not move the Bias Corrections / Blend Nudges panels — they are already correct here.

- [ ] **Step 4: Run the suite**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py engine/tools/tests/test_contracts.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ui/index.html ui/app.js ui/style.css engine/tools/tests/test_ui_smoke.py
git commit -m "feat(ui): rename model health sub-tab to Model Ops"
```

---

### Task 9: CSS cleanup and breakpoint rationalisation (D8)

**Files:**
- Modify: `ui/style.css` (dead rules ≈357, 496, 502, 653, 2714; media queries ≈797, 1267, 1412, 1421, 1441, 1448, 1974, 2125, 2713, 2747, 2760, 3717; `!important` usage; inline-style extraction targets in `ui/index.html` F1-sync modal and archive sub-tabs)
- Test: `engine/tools/tests/test_ui_smoke.py:100-114`

**Interfaces:**
- Produces: breakpoint set `phone ≤599`, `tablet 600–1199`, `desktop ≥1200`; no dead `#main-nav { }`; no `top: 70px` / `calc(100vh - 70px)` relics; reduced `!important`.

- [ ] **Step 1: Update the breakpoint assertion**

In `test_ui_adaptive_leaderboard_classes`, replace `self.assertIn("@media (max-width: 1360px)", css)` with an assertion on the new set, e.g. `self.assertIn("@media (max-width: 599px)", css)` and `self.assertIn("@media (min-width: 1200px)", css)`.

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m unittest engine.tools.tests.test_ui_smoke.UISmokeTests.test_ui_adaptive_leaderboard_classes -v`
Expected: FAIL.

- [ ] **Step 3: Remove dead CSS**

Delete the empty `#main-nav { }` rules, the `top: 70px` / `height: calc(100vh - 70px)` remnants, and any rules for markup that no longer exists.

- [ ] **Step 4: Consolidate media queries**

Collapse the scattered breakpoints to the three-tier set, checking each screen at 375/768/1280 after the change.

- [ ] **Step 5: Reduce `!important` and extract inline styles**

Remove `!important` where specificity now allows it. Move the F1-sync modal and archive sub-tab inline styles into classes.

- [ ] **Step 6: Run the suite**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py engine/tools/tests/test_contracts.py`
Expected: PASS.

- [ ] **Step 7: Verification — full visual pass**

Walk every screen at 375/768/1280 and the reduced-motion setting; confirm nothing unstyled or mis-laid-out.
Expected: consistent at all three widths.

- [ ] **Step 8: Commit**

```bash
git add ui/style.css ui/index.html engine/tools/tests/test_ui_smoke.py
git commit -m "refactor(ui): remove dead CSS, rationalise breakpoints, drop !important"
```

---

### Task 10: Whole-branch verification

**Files:**
- Read: everything changed

**Interfaces:**
- Produces: the green, verified branch.

- [ ] **Step 1: Run the full guarding suites**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py engine/tools/tests/test_contracts.py`
Expected: PASS with skip count equal to the Task 1 baseline.

- [ ] **Step 2: Run any broader project suite referenced by GEMINI.md**

Run the commands in `GEMINI.md` §verification.
Expected: PASS.

- [ ] **Step 3: End-to-end manual pass**

Load the app: Dashboard (weather panel, full race name), My Team, Prices, Standings, a real Analysis run (idle → running → results), Results tabs, Settings, Archive/Model Ops. At 375/768/1280.
Expected: no console errors; no truncation; idle never looks active.

- [ ] **Step 4: Confirm no regression in the previously-dead path**

After a real run, confirm the Optimal Dream Team grid renders (Review Focus side-effect of Task 6).
Expected: grid renders.

- [ ] **Step 5: Hand a diff summary to review**

Use `superpowers:requesting-code-review` for a whole-branch review before integrating.
Expected: reviewer sign-off, or a fix list that returns to the relevant task.
