# F1 Fantasy Predictor — UI Overhaul & Live Telemetry Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the F1 Fantasy Predictor frontend into an authentic, elite F1 Pit Wall Live Telemetry engineering console, permanently eliminating all text collisions and replacing the crude SVG car animation with a 60 FPS multi-channel telemetry waveform and dynamic Monte Carlo dial.

**Architecture:** 
1. Rebuild CSS tokens in `ui/style.css` with FIA Formula 1 racing red (`#E10600`), Apple Cupertino dark glass materials, and timing screen telemetry semantics.
2. Replace legacy SVG car keyframes in `ui/index.html` with a high-performance HTML5 Canvas telemetry waveform and a spring-interpolated Monte Carlo ticker in `ui/app.js`.
3. Decouple `.race-order-row` into dedicated columns with an isolated "Risk & Form" telemetry pill group and fluid single-column reflow below $1360\text{px}$.
4. Eliminate hardcoded inline table grids in `ui/app.js` and harden My Team cards and Dashboard status rows.

**Tech Stack:** Vanilla JavaScript (ES6+), HTML5 Canvas 2D API, CSS3 Grid/Flexbox with Backdrop Filter, GSAP 3.12, Apple Spring Physics (`apple-springs.js`), FastAPI static serving.

## Global Constraints
- Target Files: `ui/style.css`, `ui/index.html`, `ui/app.js`, `ui/apple-springs.js`, `engine/tools/tests/test_ui_smoke.py`.
- No external node/npm build dependencies (vanilla browser-native stack).
- Zero PII or API keys in code or commit messages. Commit identity must remain `Retr0-908 <Retr0-908@users.noreply.github.com>`.
- All tests in `engine.tools.tests.test_contracts` and `engine.tools.tests.test_ui_smoke` must pass.

---

### Task 1: Design System & CSS Token Architecture Refactor

**Files:**
- Modify: `ui/style.css:1-75` (tokens), `ui/style.css:1290-1445` (remove car animation, add canvas & HUD styles)
- Test: `engine/tools/tests/test_ui_smoke.py`

**Interfaces:**
- Consumes: Existing UI layout and CSS variables.
- Produces: Clean `:root` tokens, `#telemetry-waveform-canvas` styling, `.ensemble-bar-wrap` segmented progress bar, and `.ro-col-telemetry` pill container.

- [ ] **Step 1: Write UI smoke test checking for token and canvas presence**

Update `engine/tools/tests/test_ui_smoke.py` to assert that `style.css` contains `--telemetry-cyan` and `#telemetry-waveform-canvas`:

```python
    def test_ui_telemetry_tokens_and_canvas_css(self):
        """Verify style.css includes new telemetry tokens and waveform canvas."""
        res = self.client.get("/static/style.css")
        self.assertEqual(res.status_code, 200)
        css = res.text
        self.assertIn("--telemetry-cyan", css)
        self.assertIn("#telemetry-waveform-canvas", css)
        self.assertNotIn("f1-drive", css)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest engine.tools.tests.test_ui_smoke.TestUISmoke.test_ui_telemetry_tokens_and_canvas_css -v`
Expected: FAIL (token `--telemetry-cyan` not found).

- [ ] **Step 3: Update `ui/style.css` `:root` tokens and remove legacy SVG keyframes**

In `ui/style.css`:
1. Replace lines 1–40 with unified clean tokens:
```css
:root {
    /* ── Official F1 & Cockpit Brand Accents ── */
    --f1-red: #E10600;
    --f1-red-hover: #FF1801;
    --accent-primary: #E10600;
    --f1-carbon: #0E1017;

    /* ── Canonical F1 Timing & Telemetry Semantic Accents ── */
    --telemetry-purple: #B138DD;
    --telemetry-green: #00D2BE;
    --telemetry-amber: #FF9500;
    --telemetry-danger: #FF3B30;
    --telemetry-cyan: #00E5FF;
    --color-green: #00D2BE;
    --color-yellow: #FF9500;
    --color-red: #FF3B30;
    --color-sprint: #FF8000;
    --color-grey: #94A3B8;

    /* ── Cupertino Dark Obsidian Glass Materials ── */
    --bg-dark: #07090E;
    --bg-panel: rgba(12, 17, 29, 0.85);
    --bg-card: rgba(16, 22, 38, 0.70);
    --bg-card-hover: rgba(26, 35, 60, 0.85);
    --bg-glass: rgba(12, 17, 29, 0.82);
    --border-subtle: rgba(255, 255, 255, 0.07);
    --border-dim: rgba(0, 229, 255, 0.18);
    --glass-border: rgba(255, 255, 255, 0.08);
    --glass-border-hover: rgba(0, 229, 255, 0.35);

    /* ── Typography & Contrast ── */
    --text-primary: #F8FAFC;
    --text-main: #F8FAFC;
    --text-dim: #94A3B8;
    --text-muted: #64748B;
    --text-on-accent: #FFFFFF;
    --font-heading: 'Space Grotesk', sans-serif;
    --font-body: 'Inter', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;

    /* ── Apple Continuous Radius Scale ── */
    --radius-xs: 6px;
    --radius-sm: 10px;
    --radius-md: 14px;
    --radius-lg: 20px;
    --radius-pill: 9999px;
    --radius-panel: 14px;
    --radius-inner: 10px;
    --radius-badge: 6px;
```

2. Replace lines 1290–1445 in `ui/style.css` (removing `.f1-svg-loader`, `#f1-car-svg`, `.spinning-tire`, `f1-drive`, etc.) with:
```css
/* ── F1 Pit Wall Telemetry Waveform Console ── */
#analysis-animation-container {
    overflow: hidden;
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 16px 0;
    position: relative;
}

.telemetry-console-card {
    width: 100%;
    background: rgba(4, 7, 14, 0.75);
    border: 1px solid var(--border-dim);
    border-radius: var(--radius-md);
    padding: 16px;
    position: relative;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.6);
}

#telemetry-waveform-canvas {
    width: 100%;
    height: 110px;
    display: block;
    border-radius: var(--radius-sm);
    background: rgba(6, 10, 18, 0.95);
    border: 1px solid rgba(0, 229, 255, 0.15);
}

.telemetry-channel-legend {
    display: flex;
    gap: 16px;
    margin-top: 10px;
    font-size: 0.75rem;
    font-family: var(--font-mono);
}

.chan-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
}
.chan-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
.chan-speed { background: var(--telemetry-cyan); box-shadow: 0 0 6px var(--telemetry-cyan); }
.chan-throttle { background: var(--telemetry-green); }
.chan-gear { background: var(--telemetry-purple); }

/* ── Monte Carlo Dynamic Iterations Arc ── */
.monte-carlo-counter-wrap {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    margin-top: 12px;
    padding: 8px 14px;
    background: rgba(255, 255, 255, 0.02);
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-subtle);
}

.monte-carlo-label {
    font-family: var(--font-heading);
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--text-dim);
    letter-spacing: 1px;
}

.monte-carlo-val {
    font-family: var(--font-mono);
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--telemetry-cyan);
    font-variant-numeric: tabular-nums;
}

/* ── Segmented Ensemble Progress Bar ── */
.ensemble-bar-wrap {
    display: flex;
    height: 10px;
    width: 100%;
    border-radius: var(--radius-pill);
    overflow: hidden;
    background: rgba(255, 255, 255, 0.08);
    margin-top: 6px;
    gap: 2px;
}

.ensemble-segment {
    height: 100%;
    transition: width 0.4s ease;
}
.ensemble-lgb { background: var(--telemetry-green); }
.ensemble-xgb { background: var(--telemetry-purple); }
.ensemble-rf  { background: var(--telemetry-amber); }

.ensemble-labels {
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-top: 4px;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest engine.tools.tests.test_ui_smoke.TestUISmoke.test_ui_telemetry_tokens_and_canvas_css -v`
Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```powershell
git add ui/style.css engine/tools/tests/test_ui_smoke.py
git commit -m "style: modernize tokens with F1 red, cupertino glass, and telemetry console styles"
```

---

### Task 2: Analysis Screen Structure in `ui/index.html`

**Files:**
- Modify: `ui/index.html:224-325`
- Test: `engine/tools/tests/test_ui_smoke.py`

**Interfaces:**
- Consumes: CSS classes from Task 1.
- Produces: Canvas element `#telemetry-waveform-canvas`, `#mc-iterations-counter`, and segmented ensemble HUD markup.

- [ ] **Step 1: Write UI smoke test for Analysis Screen markup**

Add test method to `engine/tools/tests/test_ui_smoke.py`:

```python
    def test_ui_analysis_screen_canvas_and_hud(self):
        """Verify index.html contains telemetry canvas and segmented ensemble container."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.text
        self.assertIn('id="telemetry-waveform-canvas"', html)
        self.assertIn('id="mc-iterations-counter"', html)
        self.assertIn('id="ensemble-bar-wrap"', html)
        self.assertNotIn('class="f1-svg-loader"', html)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest engine.tools.tests.test_ui_smoke.TestUISmoke.test_ui_analysis_screen_canvas_and_hud -v`
Expected: FAIL (`id="telemetry-waveform-canvas"` not found in HTML).

- [ ] **Step 3: Update `ui/index.html` analysis screen section**

Replace lines 228–320 of `ui/index.html` with:

```html
            <div class="analysis-split-layout">
                <div class="analysis-left">
                    <div id="analysis-animation-container">
                        <div class="telemetry-console-card">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                                <div style="font-family:var(--font-heading); font-size:0.85rem; font-weight:700; color:var(--telemetry-cyan); letter-spacing:1.5px;">
                                    LIVE TELEMETRY STREAM
                                </div>
                                <div id="live-hz-badge" class="telemetry-hz-badge" style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-dim);">
                                    60 Hz ● SYNCED
                                </div>
                            </div>
                            <canvas id="telemetry-waveform-canvas" width="600" height="110"></canvas>
                            <div class="telemetry-channel-legend">
                                <div class="chan-indicator"><span class="chan-dot chan-speed"></span> SPEED (KM/H)</div>
                                <div class="chan-indicator"><span class="chan-dot chan-throttle"></span> THROTTLE %</div>
                                <div class="chan-indicator"><span class="chan-dot chan-gear"></span> GEAR SHIFT</div>
                            </div>
                            <div class="monte-carlo-counter-wrap">
                                <span class="monte-carlo-label">MONTE CARLO PROJECTION</span>
                                <span id="mc-iterations-counter" class="monte-carlo-val">READY (10,000 SIMS)</span>
                            </div>
                        </div>
                        <h3 style="margin-top: 16px; font-weight: 700; color: var(--telemetry-cyan); letter-spacing: 2px; font-size: 0.9rem;" id="animation-status-text" class="pulse-text">INITIALIZING ENGINE...</h3>
                        <p style="color: var(--text-dim); font-size: 0.78rem; margin-top: 4px;">Real-time FastF1 telemetry &amp; stochastic simulation pipeline</p>
                    </div>
                    
                    <!-- Diagnostics HUD -->
                    <div class="hud-panel">
                        <div class="hud-header">MODEL CALCULATION PARAMETERS</div>
                        <div class="hud-grid">
                            <div class="hud-item">
                                <div class="hud-label">WEEKEND ROUND</div>
                                <div class="hud-val" id="hud-round">Round — <span class="hud-unit">—</span></div>
                            </div>
                            <div class="hud-item">
                                <div class="hud-label">CIRCUIT LAYOUT</div>
                                <div class="hud-val" id="hud-circuit">Detecting... <span class="hud-unit">—</span></div>
                            </div>
                            <div class="hud-item">
                                <div class="hud-label">WEATHER CONDITIONS</div>
                                <div class="hud-val" id="hud-weather">— <span class="hud-unit">—</span></div>
                            </div>
                            <div class="hud-item">
                                <div class="hud-label">TRACK CHARACTERISTICS</div>
                                <div class="hud-val" id="hud-track">— <span class="hud-unit">—</span></div>
                            </div>
                            <div class="hud-item" style="grid-column: 1 / -1;">
                                <div class="hud-label">ENSEMBLE MODEL WEIGHTS</div>
                                <div id="ensemble-bar-wrap" class="ensemble-bar-wrap">
                                    <div id="ens-seg-lgb" class="ensemble-segment ensemble-lgb" style="width: 33.3%;"></div>
                                    <div id="ens-seg-xgb" class="ensemble-segment ensemble-xgb" style="width: 33.3%;"></div>
                                    <div id="ens-seg-rf"  class="ensemble-segment ensemble-rf"  style="width: 33.4%;"></div>
                                </div>
                                <div class="ensemble-labels">
                                    <span id="ens-lbl-lgb">LGBM: —%</span>
                                    <span id="ens-lbl-xgb">XGBoost: —%</span>
                                    <span id="ens-lbl-rf">Random Forest: —%</span>
                                </div>
                            </div>
                            <div class="hud-item" style="grid-column: 1 / -1;">
                                <div class="hud-label">SIMULATION PROGRESS &amp; SAFETY CAR RISK</div>
                                <div class="hud-val" id="hud-sims">Pending... <span class="hud-unit">—</span></div>
                            </div>
                        </div>
                    </div>
                </div>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest engine.tools.tests.test_ui_smoke.TestUISmoke.test_ui_analysis_screen_canvas_and_hud -v`
Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```powershell
git add ui/index.html engine/tools/tests/test_ui_smoke.py
git commit -m "feat(ui): mount telemetry waveform canvas and segmented ensemble HUD markup"
```

---

### Task 3: Live Telemetry Canvas & Monte Carlo Engine in `ui/app.js`

**Files:**
- Modify: `ui/app.js:610-945` (canvas waveform animation loop, SSE handler updates for segmented bars and ticker)
- Test: `engine/tools/tests/test_ui_smoke.py`

**Interfaces:**
- Consumes: `#telemetry-waveform-canvas`, `#mc-iterations-counter`, `#ens-seg-lgb`, SSE messages from `/api/run/stream/{task_id}`.
- Produces: 60 FPS animated multi-channel waveform with throttle/speed/gear traces, smooth numerical ticker during Monte Carlo stage.

- [ ] **Step 1: Write UI smoke test for app.js telemetry waveform controller**

Add test to `engine/tools/tests/test_ui_smoke.py`:

```python
    def test_ui_app_js_telemetry_controller(self):
        """Verify app.js includes canvas telemetry loop and segmented HUD logic."""
        res = self.client.get("/static/app.js")
        self.assertEqual(res.status_code, 200)
        js = res.text
        self.assertIn("initTelemetryWaveform", js)
        self.assertIn("updateEnsembleSegments", js)
        self.assertIn("animateMonteCarloCounter", js)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest engine.tools.tests.test_ui_smoke.TestUISmoke.test_ui_app_js_telemetry_controller -v`
Expected: FAIL (`initTelemetryWaveform` not found in `app.js`).

- [ ] **Step 3: Implement Telemetry Waveform Engine & SSE HUD updaters in `ui/app.js`**

Add the waveform canvas loop functions to `ui/app.js`:

```javascript
// ── Live F1 Pit Wall Telemetry Waveform Engine ────────────────────────
let _telemetryAnimFrame = null;
let _telemetryPhase = 0;
let _telemetrySpeedMultiplier = 1.0;

function initTelemetryWaveform() {
    const canvas = document.getElementById('telemetry-waveform-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high DPI displays
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
    }
    const width = rect.width || 600;
    const height = rect.height || 110;

    function renderFrame() {
        ctx.fillStyle = 'rgba(6, 10, 18, 0.35)'; // Slight trail fade
        ctx.fillRect(0, 0, width, height);

        // Draw background grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
        ctx.lineWidth = 1;
        for (let x = 0; x < width; x += 40) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        for (let y = 0; y < height; y += 22) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        _telemetryPhase += 0.04 * _telemetrySpeedMultiplier;

        // Channel 1: Speed km/h Trace (Cyan)
        ctx.strokeStyle = '#00E5FF';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let x = 0; x < width; x += 3) {
            const t = (x / 60) + _telemetryPhase;
            const y = height * 0.45 + Math.sin(t) * 22 + Math.cos(t * 2.3) * 12 + (Math.sin(t * 0.5) * 8);
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Channel 2: Throttle % Trace (Teal/Green)
        ctx.strokeStyle = '#00D2BE';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let x = 0; x < width; x += 4) {
            const t = (x / 45) + _telemetryPhase * 1.2;
            const rawThrottle = Math.sin(t * 1.5) > -0.2 ? (Math.sin(t * 1.5) * 25) : -20;
            const y = height * 0.70 + rawThrottle;
            if (x === 0) ctx.moveTo(x, Math.max(10, Math.min(height - 10, y)));
            else ctx.lineTo(x, Math.max(10, Math.min(height - 10, y)));
        }
        ctx.stroke();

        // Channel 3: Gear Shift Impulses (Purple vertical pulses)
        ctx.strokeStyle = 'rgba(177, 56, 221, 0.6)';
        ctx.lineWidth = 1.5;
        const pulseSpacing = 120;
        const shiftOffset = (_telemetryPhase * 35) % pulseSpacing;
        for (let sx = shiftOffset; sx < width; sx += pulseSpacing) {
            ctx.beginPath();
            ctx.moveTo(sx, height - 15);
            ctx.lineTo(sx, height - 35);
            ctx.stroke();
        }

        if (_isPipelineRunning) {
            _telemetryAnimFrame = requestAnimationFrame(renderFrame);
        }
    }

    if (_telemetryAnimFrame) cancelAnimationFrame(_telemetryAnimFrame);
    _telemetryAnimFrame = requestAnimationFrame(renderFrame);
}

function updateEnsembleSegments(rfWeight, xgbWeight, lgbWeight) {
    const rf = Math.round((rfWeight || 0.25) * 100);
    const xgb = Math.round((xgbWeight || 0.35) * 100);
    const lgb = Math.round((lgbWeight || 0.40) * 100);

    const segLgb = document.getElementById('ens-seg-lgb');
    const segXgb = document.getElementById('ens-seg-xgb');
    const segRf  = document.getElementById('ens-seg-rf');

    if (segLgb) segLgb.style.width = `${lgb}%`;
    if (segXgb) segXgb.style.width = `${xgb}%`;
    if (segRf)  segRf.style.width  = `${rf}%`;

    const lblLgb = document.getElementById('ens-lbl-lgb');
    const lblXgb = document.getElementById('ens-lbl-xgb');
    const lblRf  = document.getElementById('ens-lbl-rf');

    if (lblLgb) lblLgb.textContent = `LGBM: ${lgb}%`;
    if (lblXgb) lblXgb.textContent = `XGBoost: ${xgb}%`;
    if (lblRf)  lblRf.textContent  = `RF: ${rf}%`;
}

function animateMonteCarloCounter(current, target) {
    const el = document.getElementById('mc-iterations-counter');
    if (!el) return;
    let val = current;
    const step = Math.max(1, Math.round((target - current) / 10));
    const timer = setInterval(() => {
        val = Math.min(target, val + step);
        el.textContent = `${val.toLocaleString()} / 10,000 SIMS`;
        if (val >= target) clearInterval(timer);
    }, 25);
}
```

Hook `initTelemetryWaveform()` into `startAnalysis()` in `ui/app.js`:
- Call `initTelemetryWaveform()` right after `showScreen('analysis-screen')`.
- On `ML_MODEL` and `PREDICTIONS` stages, set `_telemetrySpeedMultiplier = 2.2`.
- Feed `updateEnsembleSegments(payload.rf_weight, payload.xgb_weight, payload.lgb_weight)`.
- Feed `animateMonteCarloCounter(0, payload.sims_done || 10000)`.
- When `COMPLETE` stage fires, cancel `_telemetryAnimFrame` and reset.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest engine.tools.tests.test_ui_smoke.TestUISmoke.test_ui_app_js_telemetry_controller -v`
Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```powershell
git add ui/app.js engine/tools/tests/test_ui_smoke.py
git commit -m "feat(ui): add 60fps telemetry waveform canvas and monte carlo ticker engine"
```

---

### Task 4: Adaptive Telemetry Leaderboard in `ui/style.css` & `ui/app.js`

**Files:**
- Modify: `ui/style.css:1000-1060` (grid layout & pill container), `ui/app.js:1200-1240` (`renderRaceOrder()`)
- Test: `engine/tools/tests/test_ui_smoke.py`

**Interfaces:**
- Consumes: Prediction output array from backend.
- Produces: `.ro-col-telemetry` risk & form pill group, truncated driver/team cells, responsive reflow below $1360\text{px}$.

- [ ] **Step 1: Write UI smoke test for adaptive leaderboard structure**

Add test to `engine/tools/tests/test_ui_smoke.py`:

```python
    def test_ui_adaptive_leaderboard_classes(self):
        """Verify CSS contains .ro-col-telemetry and responsive race columns breakpoint."""
        res = self.client.get("/static/style.css")
        self.assertEqual(res.status_code, 200)
        css = res.text
        self.assertIn(".ro-col-telemetry", css)
        self.assertIn("@media (max-width: 1360px)", css)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest engine.tools.tests.test_ui_smoke.TestUISmoke.test_ui_adaptive_leaderboard_classes -v`
Expected: FAIL (`.ro-col-telemetry` not found).

- [ ] **Step 3: Update `ui/style.css` for Leaderboard Grid & Risk/Form Pill Group**

In `ui/style.css`, replace lines 1000–1055 with:

```css
/* ── Predicted Race / Sprint Order Table ───────────────────────── */
.race-order-table { display: flex; flex-direction: column; gap: 4px; overflow-x: auto; }

.race-order-header {
    display: grid;
    grid-template-columns: 46px minmax(130px, 1.4fr) minmax(100px, 1fr) 95px 50px minmax(75px, 105px);
    gap: 8px; padding: 6px 12px;
    font-family: var(--font-heading); font-size: 0.72rem; font-weight: 700;
    color: var(--text-dim); letter-spacing: 1px; border-bottom: 1px solid var(--border-dim);
    margin-bottom: 4px;
    align-items: center;
}

.race-order-row {
    display: grid;
    grid-template-columns: 46px minmax(130px, 1.4fr) minmax(100px, 1fr) 95px 50px minmax(75px, 105px);
    gap: 8px; align-items: center;
    padding: 7px 12px;
    border-radius: var(--radius-sm) !important;
    background: linear-gradient(90deg, rgba(14, 19, 32, 0.85) 0%, rgba(6, 10, 18, 0.45) 100%);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-left: 3px solid transparent;
    transition: background-color 0.18s ease, border-color 0.18s ease;
}

.race-order-row:hover {
    background: linear-gradient(90deg, rgba(24, 32, 54, 0.92) 0%, rgba(12, 17, 30, 0.75) 100%);
    border-color: rgba(0, 229, 255, 0.25);
}

.ro-col-pos {
    font-family: var(--font-heading); font-size: 1rem; font-weight: 700;
    text-align: center; width: 42px; border-radius: var(--radius-xs) !important;
    padding: 2px 0; background: rgba(255,255,255,0.06); flex-shrink: 0;
}
.ro-pos-top3    { background: rgba(255, 215, 0, 0.18); color: #FFD700; border: 1px solid rgba(255, 215, 0, 0.4); }
.ro-pos-points  { background: rgba(0, 210, 190, 0.12); color: var(--telemetry-green); border: 1px solid rgba(0, 210, 190, 0.25); }
.ro-pos-out     { color: var(--text-dim); }

/* Driver & Team Columns with Strict Ellipsis */
.ro-col-driver {
    display: flex; align-items: center; gap: 8px; min-width: 0; overflow: hidden;
}
.ro-thumb {
    width: 28px; height: 28px; object-fit: contain; border-radius: 50% !important;
    background: rgba(255,255,255,0.04); flex-shrink: 0; border: 1px solid var(--border-subtle);
}
.ro-name {
    font-weight: 600; font-size: 0.88rem; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; min-width: 0; flex: 1;
}

.ro-col-team {
    font-size: 0.78rem; font-weight: 600; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; min-width: 0;
}

/* ── Dedicated Risk & Form Pill Group ── */
.ro-col-telemetry {
    display: flex;
    align-items: center;
    gap: 4px;
    flex-shrink: 0;
}
.telemetry-mini-pill {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 5px;
    border-radius: var(--radius-xs);
    white-space: nowrap;
}
.pill-rookie { background: rgba(177, 56, 221, 0.2); color: var(--telemetry-purple); border: 1px solid rgba(177, 56, 221, 0.4); }
.pill-dnf    { background: rgba(255, 59, 48, 0.2);  color: var(--telemetry-danger); border: 1px solid rgba(255, 59, 48, 0.4); }
.pill-momentum-up   { color: var(--telemetry-green); }
.pill-momentum-down { color: var(--telemetry-danger); }
.pill-momentum-flat { color: var(--text-dim); }

.ro-col-pred {
    font-family: var(--font-heading); font-size: 0.95rem; font-weight: 700;
    color: var(--text-main); text-align: center;
}

.ro-col-conf {
    display: flex; align-items: center; gap: 6px; min-width: 0;
}
.ro-conf-bar {
    flex: 1; height: 5px; background: rgba(255,255,255,0.08); border-radius: var(--radius-pill); overflow: hidden;
}
.ro-conf-fill { height: 100%; border-radius: var(--radius-pill); }
.ro-conf-pct { font-family: var(--font-mono); font-size: 0.75rem; font-weight: 600; }

/* ── Responsive Reflow ── */
@media (max-width: 1360px) {
    .race-columns {
        grid-template-columns: 1fr !important;
        gap: 16px;
    }
}
```

- [ ] **Step 4: Update `renderRaceOrder()` in `ui/app.js`**

Modify `renderRaceOrder()` in `ui/app.js` around lines 1210–1235:

```javascript
        const rookiePill = d.is_rookie ? `<span class="telemetry-mini-pill pill-rookie" title="Rookie (< 4 races)">ROOKIE</span>` : '';
        const dnfPill    = dnf >= 15 ? `<span class="telemetry-mini-pill pill-dnf" title="DNF Risk: ${dnf.toFixed(0)}%">⚠ ${dnf.toFixed(0)}%</span>` : '';
        const momentumPill = `<span class="telemetry-mini-pill ${momentum > 0.5 ? 'pill-momentum-up' : momentum < -0.5 ? 'pill-momentum-down' : 'pill-momentum-flat'}" title="EWMA Form Momentum: ${momentum > 0 ? '+' : ''}${momentum.toFixed(2)}">${mArrow}</span>`;

        return `<div class="race-order-row">
            <span class="ro-col-pos ${posClass}">${rank}</span>
            <span class="ro-col-driver">
                ${thumbHtml}
                <span class="ro-name" title="${name}">${name}</span>
            </span>
            <span class="ro-col-team" style="color:${teamColor}" title="${team}">${team}</span>
            <span class="ro-col-telemetry">
                ${rookiePill}
                ${dnfPill}
                ${momentumPill}
            </span>
            <span class="ro-col-pred">P${pred}</span>
            <span class="ro-col-conf">
                <div class="ro-conf-bar">
                    <div class="ro-conf-fill" style="width:${confW}%;background:${confColor}"></div>
                </div>
                <span class="ro-conf-pct" style="color:${confColor}">${conf.toFixed(0)}%</span>
            </span>
        </div>`;
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m unittest engine.tools.tests.test_ui_smoke.TestUISmoke.test_ui_adaptive_leaderboard_classes -v`
Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```powershell
git add ui/style.css ui/app.js engine/tools/tests/test_ui_smoke.py
git commit -m "feat(ui): implement adaptive leaderboard grid with dedicated risk-form telemetry pill"
```

---

### Task 5: Global Component Hardening & Table Responsiveness

**Files:**
- Modify: `ui/style.css` (clamp card titles, responsive tables), `ui/app.js` (replace inline grids in bias and health tables)
- Test: `engine/tools/tests/test_ui_smoke.py`

**Interfaces:**
- Consumes: Driver and Constructor cards, Bias table data, Status board.
- Produces: Responsive table containers (`.table-responsive-container`), single-line clamped cards, wrap-safe team controls.

- [ ] **Step 1: Write UI smoke test for global responsive hardening classes**

Add test to `engine/tools/tests/test_ui_smoke.py`:

```python
    def test_ui_table_responsive_container(self):
        """Verify style.css includes .table-responsive-container for bias and accuracy grids."""
        res = self.client.get("/static/style.css")
        self.assertEqual(res.status_code, 200)
        css = res.text
        self.assertIn(".table-responsive-container", css)
        self.assertIn(".card-title", css)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest engine.tools.tests.test_ui_smoke.TestUISmoke.test_ui_table_responsive_container -v`
Expected: FAIL (`.table-responsive-container` not found).

- [ ] **Step 3: Update `ui/style.css` for Card Clamping, Status Row & Responsive Container**

Add to `ui/style.css`:

```css
/* ── Responsive Table Scroll Container ── */
.table-responsive-container {
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    border-radius: var(--radius-sm);
    margin-top: 10px;
}

.table-responsive-container::-webkit-scrollbar {
    height: 6px;
}
.table-responsive-container::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.15);
    border-radius: var(--radius-pill);
}

/* ── Card Title & Team Clamping ── */
.card-title {
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 2px;
    padding: 0 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.card-team-name {
    font-size: 0.75rem;
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Team Controls Bar Wrapping ── */
.team-controls {
    display: flex;
    gap: 14px;
    align-items: center;
    flex-wrap: wrap;
}

/* ── Status Row Overflow Protection ── */
.status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 14px;
    min-width: 0;
}
.status-label {
    flex: 1;
    font-weight: 500;
    font-size: 0.95rem;
    min-width: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.status-val {
    font-size: 0.95rem;
    text-align: right;
    min-width: 0;
    max-width: 60%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
```

- [ ] **Step 4: Update `renderBiasTable()` and `renderAccuracyMetrics()` in `ui/app.js`**

In `ui/app.js`, wrap the bias table output in `.table-responsive-container` and use flexible `minmax` grids instead of hardcoded `180px 140px 100px 100px 100px 100px`:

```javascript
    let html = `
        <div class="table-responsive-container">
            <div class="race-order-header" style="grid-template-columns: minmax(140px, 1.5fr) minmax(110px, 1fr) repeat(4, minmax(70px, 1fr)); font-weight: bold; border-bottom: 2px solid var(--border-dim); padding-bottom: 8px;">
                <div>Driver</div>
                <div>Team</div>
                <div style="text-align:center;">Permanent</div>
                <div style="text-align:center;">Street</div>
                <div style="text-align:center;">Hybrid</div>
                <div style="text-align:center;">Overall</div>
            </div>
    `;

    // Row loop using matching responsive grid:
    html += `
        <div class="race-order-row" style="grid-template-columns: minmax(140px, 1.5fr) minmax(110px, 1fr) repeat(4, minmax(70px, 1fr));">
            <div class="ro-col-driver">
                <span style="border-left: 3px solid ${color}; padding-left: 8px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="${drv}">${drv}</span>
            </div>
            <div class="ro-col-team" style="color: var(--text-dim);" title="${team}">${team}</div>
            <div style="text-align:center;">${getChipHtml(b.permanent)}</div>
            <div style="text-align:center;">${getChipHtml(b.street)}</div>
            <div style="text-align:center;">${getChipHtml(b.hybrid)}</div>
            <div style="text-align:center;">${getChipHtml(b.overall)}</div>
        </div>
    `;
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m unittest engine.tools.tests.test_ui_smoke.TestUISmoke.test_ui_table_responsive_container -v`
Expected: PASS.

- [ ] **Step 6: Commit Task 5**

```powershell
git add ui/style.css ui/app.js engine/tools/tests/test_ui_smoke.py
git commit -m "fix(ui): harden responsive table containers, status rows, and card text clamping"
```

---

### Task 6: Final Verification & Secret Audit

**Files:**
- Verify: Full codebase
- Test: `engine/tools/tests/test_contracts.py`, `engine/tools/tests/test_ui_smoke.py`, `engine/tools/secret_scanner.py`

- [ ] **Step 1: Run complete UI smoke test suite**

Run: `python -m unittest engine.tools.tests.test_ui_smoke -v`
Expected: PASS (all UI smoke tests passing).

- [ ] **Step 2: Run core contract test suite**

Run: `python -m unittest engine.tools.tests.test_contracts -v`
Expected: PASS (all 46 contract tests passing).

- [ ] **Step 3: Run nuclear secret & PII scanner**

Run: `python engine/tools/secret_scanner.py`
Expected: PASS (0 secrets, 0 API keys, 0 PII detected).

- [ ] **Step 4: Check git status and branch cleanliness**

Run: `git status`
Expected: clean working tree, on `main`.
