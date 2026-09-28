# UI Fixes & Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve all 4 UI defects identified in `Screenshots/Errors/`: fix the missing circuit map for Sepang/Malaysia, replace the redundant/fake telemetry speed/throttle waveform with a professional AI pipeline simulation visualizer, fix the overlapping progress stage text on the Analysis screen, and fix the rogue horizontal blue line stuck under the "Model Health" navigation tab.

**Architecture:** 
- Fix asset routing and circuit fallback rendering in `ui/app.js` by adding local SVG track support for Sepang and robust inline SVG fallbacks.
- Replace the fake waveform canvas with a sleek Apple-grade rotating radar pulse / Monte Carlo pipeline visualizer with real computation status in `ui/index.html`, `ui/app.js`, and `ui/style.css`.
- Overhaul the horizontal stage tracker in `ui/style.css` and `ui/app.js` with responsive numbered step pills, clamped typography, and line-wrapped labels so text never collides.
- Correct `initNavIndicator` in `ui/apple-springs.js` from an erroneously horizontal `bottom: 0` bar into a sleek vertical accent indicator matching the vertical sidebar.

**Tech Stack:** Vanilla JavaScript (ES6+), CSS3 with CSS variables & Flexbox/Grid, GSAP / Apple Springs, FastAPI backend, Python unittest smoke suite.

## Global Constraints
- Zero external runtime framework dependencies (pure native web standards: HTML5, CSS3, ES6 JS).
- Adhere to Apple design principles: fluid transitions, spatial consistency, high typography standards, respect `prefers-reduced-motion`.
- Maintain 100% green test status on `test_ui_smoke.py` and `test_contracts.py`.
- No sensitive data or API tokens committed (guarded by `secret_scanner.py`).
- Follow the 8-change Git cadence tracked in `.agents/git_tracker.json`.

---

### Task 1: Resolve Missing Track Map for Sepang & Resilient Circuit Card Fallback

**Files:**
- Create: `ui/images/tracks/sepang.svg`
- Modify: `ui/app.js:193-205, 335-350`
- Test: `engine/tools/tests/test_ui_smoke.py`

**Interfaces:**
- Consumes: `GET /api/race/next` returning `circuit_id: "sepang"` or `"malaysia"`
- Produces: Reliable image URL mapping to `/static/images/tracks/sepang.svg` and graceful inline fallback for unknown circuits.

- [ ] **Step 1: Write failing UI smoke test for Sepang track map and circuit asset serving**

Add test to `engine/tools/tests/test_ui_smoke.py`:
```python
    def test_sepang_track_map_asset(self):
        """Verify that Sepang circuit map asset exists and is served cleanly."""
        res = self.client.get("/static/images/tracks/sepang.svg")
        self.assertEqual(res.status_code, 200)
        self.assertIn("<svg", res.text.lower())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py -k test_sepang_track_map_asset`
Expected: FAIL (404 Not Found)

- [ ] **Step 3: Create directory and add `sepang.svg` track outline**

Download / write vector SVG map of Sepang International Circuit to `ui/images/tracks/sepang.svg`.

- [ ] **Step 4: Update `trackMaps` and fallback rendering in `ui/app.js`**

In `ui/app.js`:
1. Add `'sepang': '/static/images/tracks/sepang.svg'` and `'malaysia': '/static/images/tracks/sepang.svg'` to the track maps lookup.
2. In `loadDashboardData()`, inspect whether the URL is a local `/static/` path or external F1 CDN URL:
   - If local path: `<img src="${mapUrl}" ... alt="Track Map">`
   - If external CDN: `<img src="${mapUrl}" onerror="this.onerror=null; this.parentElement.innerHTML = renderCircuitFallbackSvg('${race.name}', '${race.circuit}');" ...>`
3. Implement `renderCircuitFallbackSvg(raceName, circuitName)` so even if completely offline or CDN drops, a sleek futuristic circuit card with GP name, country, and track icon is rendered instead of `[ Map Unavailable ]`.

- [ ] **Step 5: Run tests and verify they pass**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py -k test_sepang_track_map_asset`
Expected: PASS

- [ ] **Step 6: Commit Task 1**

```bash
git add ui/images/tracks/sepang.svg ui/app.js engine/tools/tests/test_ui_smoke.py
git commit -m "fix(ui): add Sepang track map and resilient circuit fallback renderer"
```

---

### Task 2: Replace Redundant Telemetry Waveforms with AI Simulation Pipeline Visualizer

**Files:**
- Modify: `ui/index.html:231-255`
- Modify: `ui/app.js:750-836, 1190-1205`
- Modify: `ui/style.css:820-890`
- Test: `engine/tools/tests/test_ui_smoke.py`

**Interfaces:**
- Consumes: Server-Sent Events from `/api/run/stream/{run_id}` (`NEXT_RACE`, `WEATHER`, `PRICES`, `ML_MODEL`, `PREDICTIONS`, `ANALYSIS`, `COMPLETE`)
- Produces: Professional AI Strategy Engine radar pulse & Monte Carlo simulation progress visualizer without fake speed/throttle/gear traces.

- [ ] **Step 1: Write test verifying obsolete telemetry waveform elements are replaced**

In `engine/tools/tests/test_ui_smoke.py`:
```python
    def test_pipeline_visualizer_cleanliness(self):
        """Ensure fake telemetry traces (speed/throttle/gear) are replaced by pipeline visualizer."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertNotIn("telemetry-channel-legend", res.text)
        self.assertNotIn("SPEED (KM/H)", res.text)
        self.assertIn("pipeline-visualizer", res.text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py -k test_pipeline_visualizer_cleanliness`
Expected: FAIL

- [ ] **Step 3: Replace HTML structure in `ui/index.html`**

Replace `#analysis-animation-container` and `.telemetry-console-card`:
- Remove: `<canvas id="telemetry-waveform-canvas">`, `.telemetry-channel-legend`, `.chan-speed`, `.chan-throttle`, `.chan-gear`.
- Add: `#pipeline-visualizer` featuring:
  - Header: `STRATEGY ENGINE PIPELINE` with active pulsating status dot and live simulation badge.
  - Centerpiece: Sleek rotating radar ring / orbital particle sweep representing active ML inference and stochastic calculations.
  - Active Phase Display: Real-time dynamic stage description (`#pipeline-active-phase`), e.g. "Synthesizing 54 Track Telemetry Features", "Running 10,000 Monte Carlo Simulations", "Optimizing Integer Linear Programming Squad".
  - Monte Carlo Progress Metric: Clean badge showing live iteration count (`#mc-iterations-counter`).

- [ ] **Step 4: Update JavaScript animation in `ui/app.js`**

- Remove: `_telemetryPhase`, `_telemetrySpeedMultiplier`, `initTelemetryWaveform()`, and canvas sinusoids.
- Add: `initPipelineVisualizer()` using smooth canvas or high-performance CSS animation (orbital radar pulse that speeds up subtly during `PREDICTIONS` Monte Carlo runs, and halts smoothly on `COMPLETE` or `ERROR`).
- Update SSE handlers to update `#pipeline-active-phase` with informative, human-readable status text.

- [ ] **Step 5: Update styling in `ui/style.css`**

Add styling for `.pipeline-visualizer`, `.radar-core`, `.radar-sweep`, `.pipeline-phase-badge`, and `.sim-counter-pill`. Ensure high-contrast, dark glassmorphism, crisp borders, and zero clutter.

- [ ] **Step 6: Run tests and verify they pass**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py -k test_pipeline_visualizer_cleanliness`
Expected: PASS

- [ ] **Step 7: Commit Task 2**

```bash
git add ui/index.html ui/app.js ui/style.css engine/tools/tests/test_ui_smoke.py
git commit -m "feat(ui): replace fake telemetry waveforms with AI pipeline simulation visualizer"
```

---

### Task 3: Redesign Analysis Progress Stepper to Eliminate Overlap & Text Collision

**Files:**
- Modify: `ui/style.css:805-820`
- Modify: `ui/app.js:980-990, 1134-1150`
- Test: `engine/tools/tests/test_ui_smoke.py`

**Interfaces:**
- Consumes: List of stages `["NEXT_RACE", "WEATHER", "PRICES", "ML_MODEL", "PREDICTIONS", "ANALYSIS", "COMPLETE"]`
- Produces: Responsive 7-step horizontal stepper where labels never touch, collide, or truncate awkwardly.

- [ ] **Step 1: Write test verifying progress stepper styling classes and labels**

In `engine/tools/tests/test_ui_smoke.py`:
```python
    def test_progress_track_layout_contract(self):
        """Verify progress stepper styling handles all 7 stages cleanly."""
        res_css = self.client.get("/static/style.css")
        self.assertEqual(res_css.status_code, 200)
        self.assertIn(".progress-track", res_css.text)
        self.assertIn(".node-label", res_css.text)
        # Verify font size is scaled down and text-align is centered
        self.assertNotIn(".node-label { font-family: var(--font-heading); font-size: 1.1rem;", res_css.text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py -k test_progress_track_layout_contract`
Expected: FAIL

- [ ] **Step 3: Refactor CSS in `ui/style.css`**

Replace existing `.progress-track`, `.track-node`, `.node-circle`, `.node-label`:
- `.progress-track`: `display: flex; justify-content: space-between; align-items: flex-start; padding: 16px 8px; margin-bottom: 20px; width: 100%;`
- `.track-node`: `flex: 1; display: flex; flex-direction: column; align-items: center; text-align: center; position: relative; max-width: 80px; min-width: 0;`
- `.node-circle`: `width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 700;`
- `.node-label`: `font-family: var(--font-body); font-size: 0.68rem; font-weight: 600; line-height: 1.2; color: var(--text-dim); margin-top: 8px; text-transform: uppercase; letter-spacing: 0.4px; word-break: break-word;`
- Progress connector line: `position: absolute; top: 26px; height: 2px; left: 24px; right: 24px; background: rgba(255,255,255,0.08);`
- Active state: Subtle cyan/amber glow, clear label highlight. Done state: Green circle with checkmark `✓`.

- [ ] **Step 4: Update `ui/app.js` stage rendering**

Format stage labels cleanly:
- `NEXT_RACE` -> `Next<br>Race`
- `WEATHER` -> `Weather`
- `PRICES` -> `Prices`
- `ML_MODEL` -> `ML<br>Model`
- `PREDICTIONS` -> `Predict`
- `ANALYSIS` -> `Analysis`
- `COMPLETE` -> `Done`
Inside `.node-circle`, render step number `1` to `7` (or `✓` when done).

- [ ] **Step 5: Run tests and verify they pass**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py -k test_progress_track_layout_contract`
Expected: PASS

- [ ] **Step 6: Commit Task 3**

```bash
git add ui/style.css ui/app.js engine/tools/tests/test_ui_smoke.py
git commit -m "fix(ui): eliminate text overlapping in analysis stage stepper with responsive layout"
```

---

### Task 4: Fix Navigation Indicator (Remove Rogue Horizontal Blue Line Under "Model Health")

**Files:**
- Modify: `ui/apple-springs.js:293-345`
- Test: `engine/tools/tests/test_ui_smoke.py`

**Interfaces:**
- Consumes: Navigation click events on `.nav-btn` in `#main-nav .nav-links`
- Produces: Smooth vertical indicator pill sliding along the left edge of the active sidebar item, with zero rogue horizontal lines at `bottom: 0`.

- [ ] **Step 1: Write test verifying navigation indicator orientation and attributes**

In `engine/tools/tests/test_ui_smoke.py`:
```python
    def test_nav_indicator_orientation(self):
        """Verify nav indicator does not use hardcoded horizontal bottom: 0 positioning."""
        res = self.client.get("/static/apple-springs.js")
        self.assertEqual(res.status_code, 200)
        # Should not set bottom: 0 in style.cssText
        self.assertNotIn("bottom: 0;", res.text)
        self.assertIn("nav-spring-indicator", res.text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py -k test_nav_indicator_orientation`
Expected: FAIL

- [ ] **Step 3: Fix `initNavIndicator()` in `ui/apple-springs.js`**

1. Change indicator styling from a horizontal bar to a vertical accent pill:
   ```javascript
   indicator.id = 'nav-spring-indicator';
   indicator.style.cssText = `
     position: absolute;
     left: 0;
     width: 3px;
     background: var(--neon-cyan);
     border-radius: 0 3px 3px 0;
     pointer-events: none;
     box-shadow: 0 0 10px rgba(0, 240, 255, 0.6);
     transition: none;
   `;
   ```
2. Update `moveIndicator(btn)`:
   - Calculate vertical offset: `const top = btnRect.top - navRect.top;`
   - Calculate height: `const height = btnRect.height;`
   - Animate `y: top, height: height, opacity: 1, scaleY: 1`
3. Remove the broken horizontal `left` and `width` animations.
4. Ensure indicator hides cleanly when on viewports where sidebar is horizontal or hidden (mobile <= 767px).

- [ ] **Step 4: Run tests and verify they pass**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py -k test_nav_indicator_orientation`
Expected: PASS

- [ ] **Step 5: Commit Task 4**

```bash
git add ui/apple-springs.js engine/tools/tests/test_ui_smoke.py
git commit -m "fix(ui): convert nav indicator to vertical accent pill, eliminating rogue line under Model Health"
```

---

### Task 5: End-to-End Verification, Asset Cache Bumping & Cadence Check

**Files:**
- Modify: `ui/index.html` (bump cache buster to `?v=11`)
- Test: All suites

- [ ] **Step 1: Bump asset version query parameters in `ui/index.html`**

Update `style.css?v=10` -> `style.css?v=11` and `app.js?v=10` -> `app.js?v=11`.

- [ ] **Step 2: Run full unit & smoke test suites**

Run: `python -m unittest engine/tools/tests/test_ui_smoke.py engine/tools/tests/test_contracts.py`
Expected: 100% PASS (all tests green)

- [ ] **Step 3: Run secret scanner**

Run: `python engine/tools/secret_scanner.py`
Expected: PASS with 0 secrets detected.

- [ ] **Step 4: Commit and push per GEMINI.md git cadence**

Update `.agents/git_tracker.json` and push verified changes to `origin main`.
