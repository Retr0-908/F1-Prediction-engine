# F1 Fantasy Predictor — UI Overhaul & Live Telemetry Console Design

**Date:** 2026-09-26  
**Status:** Approved  
**Author:** Retr0-908  
**Scope:** Frontend overhaul (`ui/style.css`, `ui/index.html`, `ui/app.js`, `ui/apple-springs.js`)

---

## 1. Executive Summary & Problem Analysis

### 1.1 Problems Identified in Current UI
1. **Calculation Animation ("Looking Like Shite"):**
   - The calculation loading state relied on a rudimentary 2D cyan SVG car sliding across a `viewBox="0 0 200 60"` using hardcoded CSS keyframes (`f1-drive` over 2.4s). It abruptly vanished at `translateX(220px)` and popped back in at `-100px`.
   - Neon green dasharray circles simulated spinning tires, and three static dashed SVG lines slid backwards as "speed lines".
   - The animation was disconnected from the actual backend pipeline stages (Glicko-2 ratings $\to$ FastF1 telemetry $\to$ LightGBM training $\to$ 10,000 Monte Carlo stochastic runs $\to$ Mixed Integer Fantasy Optimization).
2. **Text Overflow & Collision Issues:**
   - **Diagnostics HUD (`.hud-item`):** Grid of `1fr 1fr` where `.hud-val` had `font-size: 1.35rem; font-weight: 700;` with no text wrapping or overflow handling. Strings such as `RF:35% | XGB:40% | LGB:25% WEIGHTS` or long circuit names burst through card boundaries and overlapped neighboring containers.
   - **Leaderboard Table Rows (`.race-order-row`):** Packed into fixed columns where the driver cell stuffed a headshot ($28\text{px}$), driver name, rookie badge, DNF risk pill (`⚠ 18%`), and momentum arrow (`▲`) into a $130\text{px}$ slot. Long names (*Gabriel Bortoleto*, *Franco Colapinto*) collided into badges and the team column.
   - **Dual-Column Widescreen Squeeze (`.race-columns`):** Displaying two side-by-side columns on displays $< 1400\text{px}$ caused severe column squishing and text bleeding.
   - **Hardcoded Inline Data Grids:** Tables in Model Health and Driver Bias views used hardcoded inline styles (`grid-template-columns: 180px 140px 100px 100px 100px 100px;`) with no horizontal scroll containment.
   - **Dashboard Status Board & My Team Cards:** Lack of single-line clamping caused uneven card heights and overlapping text on smaller displays.
3. **Design System & Color Conflicts:**
   - `:root` contained conflicting and duplicate definitions (e.g. `--f1-red` defined as cyan `#00F0FF`, `--color-green` also defined as `#00F0FF`, and `--color-yellow` as neon lime `#CCFF00`).
   - Inconsistent border radii (mix of `14px` and legacy `border-radius: 0 !important;`).

---

## 2. Core Architecture & System Specifications

### 2.1 Unified Design Tokens & CSS Architecture
* **Brand Accents:**
  - `--f1-red: #E10600` (Official FIA Formula 1 Crimson)
  - `--f1-red-hover: #FF1801`
  - `--f1-carbon: #0E1017` (Deep cockpit carbon background)
* **Telemetry & Timing Semantic Colors:**
  - `--telemetry-purple: #B138DD` (Fastest sector / optimal strategy / dream team highlight)
  - `--telemetry-green: #00D2BE` (Positive delta / high confidence $\ge 70\%$ / P4–P10 points)
  - `--telemetry-amber: #FF9500` (Medium confidence $45–69\%$ / moderate warning)
  - `--telemetry-danger: #FF3B30` (Low confidence $< 45\%$ / high DNF risk $\ge 15\%$ / negative delta)
  - `--telemetry-cyan: #00E5FF` (Live sensor stream / active pipeline stage indicator)
* **Cupertino Glass Materials:**
  - `--bg-app: #07090E`
  - `--bg-glass: rgba(14, 19, 32, 0.78)` with `backdrop-filter: blur(20px) saturate(180%)`
  - `--bg-glass-card: rgba(20, 27, 45, 0.65)`
  - `--glass-border: rgba(255, 255, 255, 0.08)`
  - `--glass-border-hover: rgba(255, 255, 255, 0.18)`
  - `--glass-border-active: rgba(225, 6, 0, 0.40)`
* **Typography:**
  - **Headings & Badges:** `Space Grotesk`, sans-serif
  - **Body & Controls:** `Inter`, sans-serif
  - **Telemetry & Numbers:** `JetBrains Mono`, tabular figures (`tabular-nums`)
* **Radius Scale:**
  - `--radius-xs: 6px` (badges, chips)
  - `--radius-sm: 10px` (buttons, inputs, pills)
  - `--radius-md: 14px` (cards, leaderboard rows, HUD tiles)
  - `--radius-lg: 20px` (main panels, modals)
  - Elimination of all `border-radius: 0 !important;` overrides.

---

### 2.2 F1 Pit Wall Live Telemetry Calculation Console
* **Decommissioning Legacy Visuals:**
  - Remove `.f1-svg-loader`, `#f1-car-svg`, `.speed-line`, `.spinning-tire`, and `@keyframes f1-drive`.
* **HTML5 Canvas Telemetry Waveform (`#telemetry-waveform-canvas`):**
  - Renders a continuous 60 FPS multi-channel telemetry waveform simulating live vehicle speed, throttle application, brake pressure, and gear shifts across track micro-sectors.
  - Channels rendered in high-tech telemetry cyan (`#00E5FF`) and timing purple (`#B138DD`) over a subtle dark telemetry grid.
  - When backend transitions to `ML_MODEL` and `PREDICTIONS`, waveform frequency and amplitude dynamically surge to reflect computational load.
* **Live Monte Carlo Simulation Arc & Ticker:**
  - An interactive circular dial / progress ring displaying simulated iterations (`0 ➔ 10,000 runs`).
  - Spring-interpolated numerical ticker that updates smoothly via `requestAnimationFrame` during the `PREDICTIONS` SSE stage.
* **Responsive Diagnostics HUD:**
  - Standardized `.hud-item` layout with `overflow: hidden; min-width: 0;`.
  - **Ensemble Weights:** Rendered as a sleek segmented triple-progress bar (`LGB` teal, `XGB` purple, `RF` amber) with clean numeric percentages, replacing the text string that caused overflows.
  - **Circuit & Weather:** Single-line ellipsis with hover tooltips for long circuit names.
  - **Monte Carlo Status:** Dedicated numeric counter with Safety Car risk badge.
* **Streamlined Pipeline Stage Track & Terminal Log:**
  - Pipeline stage pills with active neon glow and duration badges.
  - High-contrast, dark-mode terminal log with styled monospace output, color-coded status prefixes, and automated smooth autoscroll.

---

### 2.3 Adaptive Telemetry Leaderboard (Results View)
* **Decoupled Column Structure:**
  - New `.race-order-row` grid template:
    ```css
    grid-template-columns: 46px minmax(140px, 1.4fr) minmax(110px, 1fr) 95px 50px minmax(80px, 110px);
    gap: 10px;
    align-items: center;
    ```
* **Dedicated "Risk & Form" Telemetry Pill (`.ro-col-telemetry`):**
  - Driver names and team names are strictly isolated with `overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0;`.
  - Badges (Rookie, DNF probability, Momentum arrow) are extracted from the driver cell and rendered inside a consolidated pill group with rich hover tooltips.
* **Fluid Responsive Reflow (`.race-columns`):**
  - $\ge 1360\text{px}$: Two balanced columns (P1–P10 on left, P11–P22 on right).
  - $< 1360\text{px}$: Fluid single-column continuous leaderboard with a subtle visual divider between points positions (P10) and non-points positions (P11).
  - $< 768\text{px}$: Compact mode, hiding confidence bar graph while retaining exact percentage and predicted position.
* **Sprint Weekend Consistency:**
  - Identical resilient column structure across both Sprint Race predictions and Grand Prix predictions.

---

### 2.4 Global Component Hardening
* **Dashboard Status Board (`.status-row`):**
  - Explicit `min-width: 0;` on `.status-label` and `.status-val`.
  - Ellipsis truncation with hover tooltip on long status messages.
* **Hero Race Card Stats Row (`.hero-stats-row`):**
  - Equalized padding and tabular numerals across all 4 mini-stat cards (`weather-temp`, `precip`, `humidity`, `wind`).
* **My Team & Market Price Cards (`.card`):**
  - Single-line clamping on driver and constructor titles (`-webkit-line-clamp: 1; text-overflow: ellipsis;`).
  - Equal height across `.grid-picker` cards via `align-items: stretch`.
  - `.team-controls` bar updated with `flex-wrap: wrap; gap: 12px;` to prevent input collision.
* **Model Health, Accuracy & Bias Tables:**
  - Elimination of all inline hardcoded pixel column grids (`style="grid-template-columns: 180px 140px ..."`).
  - Introduction of `.table-responsive-container` with custom sleek webkit scrollbars and sticky driver identifier column.
* **Apple Spring & Gesture Polish (`ui/apple-springs.js`):**
  - Universal pointer-down tactile feedback (`transform: scale(0.97)`) on all clickable elements.
  - Zero-latency tap response (`touch-action: manipulation`).

---

## 3. File Modification Matrix

| File | Target Modifications |
|---|---|
| [`ui/style.css`](file:///C:/Users/Aaron%20Kunnath/Desktop/Tools/AI%20Assisted/Projects/F1-Prediction-engine-main/ui/style.css) | 1. Clean `:root` variables, establish F1 red & telemetry semantic palette.<br>2. Remove legacy SVG car keyframes & speed line animations.<br>3. Add styles for `#telemetry-waveform-canvas`, circular Monte Carlo dial, and segmented ensemble bar.<br>4. Rebuild `.race-order-row`, `.race-columns`, and `.ro-col-telemetry`.<br>5. Add `.table-responsive-container`, fix `.status-row`, and clamp My Team card titles. |
| [`ui/index.html`](file:///C:/Users/Aaron%20Kunnath/Desktop/Tools/AI%20Assisted/Projects/F1-Prediction-engine-main/ui/index.html) | 1. Replace `.f1-svg-loader` block in `analysis-screen` with `#telemetry-waveform-canvas` and Monte Carlo dial container.<br>2. Update HUD structure to support segmented ensemble bars.<br>3. Ensure responsive meta viewport and font links are pristine. |
| [`ui/app.js`](file:///C:/Users/Aaron%20Kunnath/Desktop/Tools/AI%20Assisted/Projects/F1-Prediction-engine-main/ui/app.js) | 1. Add Canvas telemetry waveform animation loop with dynamic frequency scaling.<br>2. Update `renderRaceOrder()` to render the dedicated Risk & Form pill group.<br>3. Update HUD updating logic (`ML_MODEL`, `PREDICTIONS`) to feed segmented bars and numerical counter.<br>4. Replace inline grid styles in `renderBiasTable()`, `renderAccuracyMetrics()`, and verification tables with CSS classes. |
| [`ui/apple-springs.js`](file:///C:/Users/Aaron%20Kunnath/Desktop/Tools/AI%20Assisted/Projects/F1-Prediction-engine-main/ui/apple-springs.js) | 1. Ensure spring hover and press handlers attach smoothly to new telemetry cards and buttons.<br>2. Export numerical spring interpolation helper for the Monte Carlo counter. |

---

## 4. Verification & Testing Strategy

1. **Automated Smoke & Contract Tests:**
   - Execute `python -m unittest engine.tools.tests.test_ui_smoke` (verifies static mounts, assets, endpoints).
   - Execute `python -m unittest engine.tools.tests.test_contracts` (verifies data integrity and prediction engines).
2. **Security & Anonymity Audit:**
   - Run `python engine/tools/secret_scanner.py` to confirm zero credentials, API keys, or personal identifiers exist.
3. **Responsive Visual Verification:**
   - Test UI responsiveness at $1920\times 1080$ (widescreen), $1366\times 768$ (laptop), and $768\times 1024$ (tablet/split screen).
   - Verify zero text overflow or collision on driver names, circuit names, status rows, and bias tables.
   - Verify 60 FPS smooth canvas telemetry animation and Monte Carlo dial during `startAnalysis()`.
