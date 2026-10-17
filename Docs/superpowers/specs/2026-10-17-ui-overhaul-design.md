# F1 Fantasy Predictor — UI Overhaul Design

**Date:** 2026-10-17
**Author:** pairing session (Aaron + agent)
**Status:** awaiting approval

---

## 1. Intent and the why

The app works, but its interface reads as amateur and, in places, actively
misleads. Two complaints drove this work:

1. **The Analysis screen looks like a prediction is running when nothing is.**
   The radar sweep, pulse dots, pipeline banner and a "READY (10,000 SIMS)"
   counter animate at rest, so an idle app looks busy and a stalled run looks
   the same as a healthy one.
2. **The look is inconsistent across screens.** Red and cyan fight for the role
   of accent colour, panel headers disagree with buttons, spacing is uneven,
   text gets truncated, and several panels hold prime space without earning it
   (chief among them the Dashboard's "AI Insights & Top Move").

What success looks like: the app looks like a deliberate product built by one
designer, an idle screen is unmistakably idle, and every panel either informs a
decision or shows real data. The user is also the active developer of this
project, so the interface must stay legible as a working surface — dense model
internals belong somewhere, but not in the middle of the main view.

## 2. Goals

- **G1 — Coherent visual language.** One accent colour, one panel style, one
  radius/spacing/typography system, applied across all nine screens.
- **G2 — Honest state.** Nothing animates, counts, or claims activity unless
  something is genuinely happening.
- **G3 — Every element earns its place.** Remove, replace, collapse or relocate
  sections that duplicate, mislead or sit empty.
- **G4 — Responsive.** Usable and comfortable from phone (≈375px) through
  desktop (≥1280px), with no fixed width squeezing content on small screens.
- **G5 — Cleaner CSS.** Remove dead and duplicated rules, reduce `!important`
  and inline styles, and consolidate the design tokens.

## 3. Non-goals

- No change to prediction logic, model behaviour, or the Python engine.
- No framework migration — the UI stays vanilla HTML/CSS/JS served as static
  files by FastAPI. No bundler, no build step.
- No screens are removed; all nine stay (the user uses all of them).
- No new backend endpoints unless a UI change strictly requires data the API
  cannot already supply.

## 4. Anti-goals (what would make this a failure even if it "works")

- **Decorative activity.** Any animation that implies computation while idle is
  a failure, not a flourish — including on screens we don't otherwise touch.
- **A house divided.** Two competing accents, or a red button next to a cyan
  header, is a failure even if each looks fine alone.
- **Regression by restyle.** Breaking an existing interaction (tab switching,
  file import, cache controls, SSE log rendering) to make something prettier is
  a failure.
- **Polish that hides data.** The user is building this project; over-simplifying
  must not remove information they actually rely on. Trim by relocating or
  collapsing, not by deleting capability.

## 5. Constraints

- **Stack, as it is:** `ui/index.html` (single-page shell, all screens inline),
  `ui/style.css` (design system + screen styling), `ui/app.js` (data + DOM
  logic), `ui/apple-springs.js` (GSAP motion layer). Served by
  `engine/serving/server.py` at `/static/*`.
- **Fonts already in use:** `Space Grotesk` (headings), `Inter` (body),
  `JetBrains Mono` (numerals). Keep this stack.
- **Motion layer:** screen/tab transitions, hovers and press states run through
  GSAP in `apple-springs.js`, gated by `prefers-reduced-motion`. CSS must not
  fight it (GSAP owns `transform`).
- **Existing tokens:** `ui/style.css` `:root` already defines colour, radius,
  spacing and shadow tokens. Extend and rationalise these rather than inventing
  a parallel set.

## 6. Decisions

### D1 — Visual language: "Modern F1 App" (chosen by the user)

The user reviewed three F1-themed directions (Classic race-poster; Modern F1
App; Carbon/pit-wall timing) and chose **Modern F1 App**, applied throughout.
Its character:

- **Surfaces:** near-black page (`#0A0A0C`) with soft dark-grey raised cards
  (a subtle top-to-bottom gradient, e.g. `#141418 → #0F0F12`), low-contrast
  1px borders (`rgba(255,255,255,.06)`), soft large shadows.
- **Accent:** F1 red gradient for primary actions
  (`linear-gradient(180deg, #FF2D0F, #D50500)`), flat `--f1-red` (`#E10600`)
  for small marks (header ticks, status dots).
- **Corners:** large and friendly — cards ≈ 16px, buttons/inputs ≈ 12px,
  chips ≈ 12px. (Supersedes the current mixed sharp/large radius.)
- **Typography:** headings keep `Space Grotesk`; body `Inter`; numerals
  `JetBrains Mono`. Case: sentence case for headings and labels (drop the
  shouty all-caps), except small overline labels. Numbers get real weight and
  size (this is a data app — positions, points, percentages lead).
- **Spacing:** use the existing `--space-*` scale; increase panel padding and
  inter-panel gaps so panels breathe. Target a visible, repeatable rhythm.
- **Density:** generous — fewer things per viewport, larger type. Sections that
  need density (data tables) may keep it, but chrome around them gets quieter.

### D2 — One accent, plus semantic data colours (agent's call, flagged)

Red is the single UI accent. The current cyan-everywhere (`--neon-cyan`,
`--telemetry-cyan`) that drives buttons, borders, focus rings and glows is
**retired as an accent**. Rationale: the red/cyan mix is the single biggest
cause of the "inconsistent" read, and the user asked for F1 theming.

Kept as **data semantics only** (never as general UI chrome): purple = fastest
lap, green/teal = gain/positive, amber = warning/rain, red = loss/danger.
These already exist as `--telemetry-*` / `--color-*` tokens.

*Flagged as the agent's call:* focus rings currently use cyan
(`--neon-cyan`). These will be restyled to a high-contrast ring that reads on
the dark surfaces (proposed: white/red two-tone) so accessibility does not
regress when cyan leaves.

### D3 — Analysis screen: idle means idle (user's decision)

- **At rest:** a calm standby panel. Grey (not pulsing) status dot, the sentence
  "Standby — no analysis running", a one-line description of what running will
  do, the inputs that feed the run, and the last run's timestamp (read from the
  existing `/api/predictions` `generated_at`; no duration is persisted, so none
  is shown). No radar
  sweep, no pulse dots, no "ACTIVE" badge, **no sims number**.
- **During a genuine run:** motion and numbers are allowed because they are
  real — an elapsed timer, a phase label, a filling progress bar, and the live
  SSE log. The Monte Carlo counter reappears **only while counting**
  (e.g. "4,120 / 10,000").
- **Sims number rule (user's decision):** the 10,000 figure never appears when
  idle. It shows only during/after a run where it represents actual progress.
- **Model Calculation Parameters HUD (user's decision):** stays, but fills with
  real values only as a run progresses. When idle it is either hidden or
  visibly empty/dormant — no `LGBM: —%` placeholders masquerading as content.

### D4 — Dashboard (user's decisions)

- **Weekend Hub: kept.** It is the user's favourite panel. The race name must
  render **in full** — long names wrap instead of truncating. Its existing
  race-day weather row stays as an at-a-glance **summary**; the new panel is the
  detail beneath it (summary over detail, not duplication).
- **"AI Insights & Top Move": removed**, replaced by a **Weekend Weather**
  panel. Contents:
  - a headline summary of the weekend's conditions;
  - a per-session forecast (FP1…Race) with temp and rain probability;
  - "How it could affect the race" — plain-English notes (e.g. wet race raises
    DNF risk, increases safety-car likelihood, makes grid order less
    predictive). These notes are **authored copy** selected by simple rain-risk
    thresholds, not model output, so no backend work is required.
  This panel is useful before any run, which is precisely why it beats the panel
  it replaces.
- **"System Status": kept**, because the user relies on it while developing.
  It stays on the main view, but its "Next Race" value must stop truncating.

### D5 — Results (user's decisions)

- **Legend block** above the results table: it already defaults to collapsed
  (`class="legend-panel collapsed"`), so the collapse itself needs no work. The
  change is affordance — replace the full-width "ℹ️ Prediction Symbols & Color
  Legend" bar with a compact "?" toggle, restyled to the new language.
- **Analytics tab:** `Driver Attribute Analysis` and `Driver Comparison Tool`
  merge into a single **Driver Analysis** panel.

### D6 — Archive & Analytics (user's decision)

- `Active Driver Bias Corrections` and `Circuit-Type Feature Blend Nudges`
  already live in the Archive & Analytics screen, inside the **"Model Health &
  Biases"** sub-tab — they are not in the main analysis view. The change is
  therefore a **rename and tidy**: that sub-tab becomes **"Model Ops"** and is
  restyled, keeping model internals reachable in one dedicated home. No new
  sub-tab is built.

### D7 — Responsive navigation (user's decision)

- **Desktop (≥1200px):** full fixed sidebar as today.
- **Tablet and phone (<1200px):** the sidebar becomes a **slide-in drawer**
  opened by a hamburger/menu button in a compact top bar, over a dimmed
  backdrop, closing on selection or backdrop tap.
- **Primary action stays reachable:** the top bar also carries a **Run Analysis**
  button, so the app's main action never hides inside the drawer. The run-mode
  selector and secondary controls move into the drawer.
- The exact breakpoint between "fixed" and "drawer" is the builder's call,
  consistent with the existing breakpoint set (there are currently
  `767 / 900 / 1100 / 1200 / 1360 / 1400` — these should be rationalised, see
  D8).

### D8 — CSS cleanup (agent's call, flagged)

Scope, described so it can be planned without the original conversation:

- **Remove dead layout CSS** left over from a former horizontal top-nav
  (`top: 70px`, `height: calc(100vh - 70px)`, the empty `#main-nav { }` rule).
- **Deduplicate** rules that are defined twice — notably `.dream-team-grid`,
  `.dt-card`, `.dt-rank`, `.dt-thumb` (defined near lines ≈987–1009 and again
  ≈2424–2502), and `.btn-primary` (red ≈210, then overridden cyan ≈2650 with
  `!important`).
- **Reduce `!important`** usage, especially on radii, once duplicates are gone.
- **Move inline styles into classes** where practical — the F1-sync modal and
  the archive sub-tabs are the worst offenders.
- **Rationalise breakpoints** to a small, consistent set (proposed:
  `phone ≤599`, `tablet 600–1199`, `desktop ≥1200`), replacing the seven
  scattered values.
- **Convert existing tokens** to the D1 language rather than adding new ones:
  update `--radius-*`, surface colours, shadows and the accent set in `:root`,
  then let screens inherit. This is the minimum-diff path to visual consistency.
- **Update the UI contract tests.** `engine/tools/tests/test_ui_smoke.py` pins
  literal strings that D2/D3/D7/D8 change (the `1360px` breakpoint,
  `--telemetry-cyan`, the pipeline ids, `nav-spring-indicator`). Those
  assertions are rewritten to check **intent and structure** (the nav exists,
  the drawer opens) rather than exact literals, and the suite is kept 100% green
  per `GEMINI.md`.

## 7. Left to the builder

- Exact token values within the D1 character (final hexes, gradient stops,
  shadow spreads) — tune against the real screens.
- The precise breakpoint value where the drawer takes over.
- Component-level composition of the merged **Driver Analysis** panel.
- Layout of the new **Model Ops** sub-tab.
- Whether the Analysis HUD is hidden or shown-dormant when idle (either
  satisfies D3; pick what looks best).
- Ordering of the work and how the CSS refactor is staged so the app stays
  runnable throughout.
- Focus-ring treatment within the "reads clearly on dark" requirement (D2).
- Any additional dead CSS found beyond the list in D8.

## 8. How it gets made

Existing repo, existing stack. Work happens in `ui/index.html`,
`ui/style.css`, `ui/app.js` and (where motion changes are needed)
`ui/apple-springs.js`, served by the existing FastAPI server. No dependencies
are added. The design system is rebuilt in `style.css` `:root` and inherited by
screens; markup changes are limited to the content decisions above (D3–D6) and
the nav shell (D7).

## 9. Calls the agent made

Settled by the agent, not the user. Any of these can be vetoed.

- **Last run shown as a timestamp, not a duration.** No duration is persisted;
  the standby panel shows the timestamp from `/api/predictions`.
- **Weather impact notes are authored copy**, selected by rain-risk thresholds,
  not model output.
- **Launch screen keeps its motion.** The particle field and spinner are honest
  boot feedback (the app really is loading), so they stay. The all-caps
  `F1 FANTASY PREDICTOR` title stays all-caps as a logotype — brand marks are
  exempt from the sentence-case rule.
- **Legend affordance** is a compact "?" toggle; the collapse behaviour already
  exists.
- **Dead call removed.** `renderInsightsPanel(data)` is called at
  `ui/app.js:1779` but defined nowhere, so it throws and blocks the Optimal
  Dream Team grid (from `app.js:1782`) and `updateQuickSummary` (`app.js:1813`)
  from rendering. Deleting the "AI Insights & Top Move" panel removes that call,
  and the downstream rendering is expected to start working again. This is a bug
  fix, not a feature.

## 10. Verification

- Project suites run green:
  `python -m unittest engine/tools/tests/test_ui_smoke.py engine/tools/tests/test_contracts.py`
  (per `GEMINI.md`), with the UI smoke assertions updated to intent-based checks.
- Manual pass at three widths (≈375px phone, ≈768px tablet, ≥1280px desktop):
  no horizontal squeeze, the drawer opens and closes, no truncated race names.
- Idle-state check: with no run in progress, the Analysis screen shows no
  motion, no counters and no "active" language anywhere.
- Console cleanliness: no thrown errors on a normal load and during a normal
  run.
