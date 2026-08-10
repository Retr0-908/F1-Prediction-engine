# UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the "Apple Pro" UI redesign defined in `Docs/superpowers/specs/2026-08-10-ui-redesign.md`.

**Architecture:** The UI shifts from a floating nav to a persistent left sidebar with a main content canvas. The palette changes to deep blacks/grays without heavy shadows or neon glassmorphism.

**Tech Stack:** HTML, CSS, JavaScript (GSAP)

## Global Constraints

- No zebra striping on tables.
- Do not use heavy box shadows or glowing borders.
- Only the sidebar may have a translucent/glass background.
- Typography strictly uses `Inter`.

---

### Task 1: Global Layout & CSS Variables

**Files:**
- Modify: `ui/index.html`
- Modify: `ui/style.css`

**Interfaces:**
- Consumes: Existing DOM nodes in `index.html`.
- Produces: A new `app-container` wrapping a `sidebar` and a `main-canvas`.

- [ ] **Step 1: CSS Variables & Typography**
In `ui/style.css`, update the `:root` variables:
```css
:root {
  --bg-dark: #09090B;
  --bg-sidebar: rgba(18, 18, 20, 0.7);
  --bg-card: #18181B;
  --bg-card-hover: #27272A;
  --text-primary: #FFFFFF;
  --text-muted: #A1A1AA;
  --accent-primary: #E10600; /* F1 Red */
  --border-subtle: rgba(255,255,255,0.05);
  --font-main: 'Inter', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}
```
Update `body` to use `font-family: var(--font-main); background: var(--bg-dark); color: var(--text-primary);`.

- [ ] **Step 2: Structural CSS Layout**
Add layout rules to `ui/style.css`:
```css
.app-container {
  display: flex;
  min-height: 100vh;
  width: 100vw;
  overflow: hidden;
}
.sidebar {
  width: 260px;
  background: var(--bg-sidebar);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-right: 1px solid var(--border-subtle);
  display: flex;
  flex-direction: column;
  padding: 24px;
}
.main-canvas {
  flex: 1;
  position: relative;
  overflow-y: auto;
  padding: 0;
}
```

- [ ] **Step 3: Update HTML Structure**
In `ui/index.html`, wrap the entire app (excluding the launch screen and modals) inside a `<div class="app-container">`.
Convert `<nav id="main-nav">` into `<aside id="main-nav" class="sidebar">`.
Move all `<div class="screen">` elements inside a `<main class="main-canvas">`.

- [ ] **Step 4: Update Sidebar Styling**
Remove floating properties from `#main-nav`. Ensure `.nav-links` are displayed as a vertical flex column with gap. Update `.nav-btn` to have flat transparent backgrounds and rounded corners, turning slightly gray (`#27272A`) on hover and red on active.

- [ ] **Step 5: Commit**
```bash
git add ui/index.html ui/style.css
git commit -m "ui: refactor global layout to apple pro sidebar architecture"
```

---

### Task 2: Card Styling & Spacing Overhaul

**Files:**
- Modify: `ui/style.css`

**Interfaces:**
- Consumes: `.panel`, `.card`, `.hero-race-card` class components.

- [ ] **Step 1: Flatten Cards and Increase Padding**
In `ui/style.css`, target `.panel`, `.card`, and `.hero-race-card`.
- Remove all `box-shadow` properties (e.g., `box-shadow: 0 4px...`).
- Set `background: var(--bg-card);`.
- Set `border: 1px solid var(--border-subtle);`.
- Increase padding to `24px` instead of current padding (usually 10-15px).

- [ ] **Step 2: Add CSS Active State (Tactile Feedback)**
Add a global active state rule in `ui/style.css`:
```css
.card, .panel, .btn, .nav-btn, .driver-card {
  transition: transform 0.1s ease-out, background 0.2s ease, border-color 0.2s ease;
}
.card:active, .panel:active, .btn:active, .nav-btn:active, .driver-card:active {
  transform: scale(0.98);
}
```

- [ ] **Step 3: Hierarchy via Weight**
Update headers (`h1`, `h2`, `h3`, `.panel-header`) to use `font-weight: 600` or `700`, and `letter-spacing: -0.02em`. Remove uppercase transformations if they look too harsh, or keep them with loose tracking (`letter-spacing: 0.05em`) if they are small headers.

- [ ] **Step 4: Commit**
```bash
git add ui/style.css
git commit -m "ui: flatten cards, increase padding, and add tactile active states"
```

---

### Task 3: Page Refinements & Clean Tables

**Files:**
- Modify: `ui/index.html`
- Modify: `ui/style.css`
- Modify: `ui/apple-springs.js`

**Interfaces:**
- Consumes: Tables, Toggle inputs, Animation JS.

- [ ] **Step 1: Clean Tables**
In `ui/style.css`, update table styling:
- Remove zebra striping (`tbody tr:nth-child(even)` background).
- Add a subtle border beneath rows: `border-bottom: 1px solid var(--border-subtle);`.
- Ensure numerical columns use `font-family: var(--font-mono);`.

- [ ] **Step 2: iOS Toggle Settings**
In `ui/index.html`, wrap checkboxes in `settings-screen` with a custom label (if not already). In `ui/style.css`, create an iOS-style toggle using `input[type="checkbox"]` pseudo-elements (width ~40px, height ~24px, green background when checked, white circle sliding).

- [ ] **Step 3: Update Screen Transitions**
In `ui/apple-springs.js`, locate the `navigateToScreen` logic around line 91. The current transitions use fixed durations (`duration: 0.14`). Update them to use a custom GSAP CustomEase if available, or just a critically damped equivalent `ease: 'power3.out'` with `duration: 0.3` to match the "spring" feel without bouncing.

- [ ] **Step 4: Commit**
```bash
git add ui/index.html ui/style.css ui/apple-springs.js
git commit -m "ui: clean tables, add ios toggles, refine screen transitions"
```
