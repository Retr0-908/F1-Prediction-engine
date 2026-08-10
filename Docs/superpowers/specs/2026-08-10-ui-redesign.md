# F1 Prediction Engine - UI Redesign Spec

## Objective
Redesign the F1 Prediction Engine web UI to remove the chaotic, "AI-generated" glassmorphism aesthetic and replace it with a highly intuitive, structured, and refined layout inspired by Apple's Pro applications. 

## 1. Global Layout & Architecture
- **Sidebar Navigation:** Implement a persistent, fixed-width (`260px`) left sidebar containing all main navigation links, replacing any top/bottom floating nav bars.
- **Main Canvas:** The remaining viewport width (`calc(100vw - 260px)`) acts as the primary content canvas, providing ample space for dense F1 telemetry data.
- **Header:** A minimalist top bar inside the main canvas for page titles and status indicators.

## 2. Color Palette & Materials
- **Main Background:** Deep slate/black (`#09090B`).
- **Sidebar:** Slightly elevated (`#121214`) with a subtle translucent blur (glassmorphism is restricted exclusively to the sidebar/nav).
- **Cards/Containers:** Solid, flat dark grays (`#18181B`) with an ultra-subtle `1px` border (`rgba(255,255,255,0.05)`). All heavy drop shadows and glowing outlines must be removed.
- **Accents:** Use "F1 Red" (`#E10600`) sparingly for primary actions/active states. Use muted text (`#A1A1AA`) for secondary data labels.

## 3. Typography & Spacing
- **Font Family:** Strictly enforce `Inter` (or system UI fonts) across the app.
- **Hierarchy:** Use heavy font weights (`600`/`700`) and tight letter-spacing for headers. Use regular weights for body text, and tabular numbers/monospace fonts for data tables to ensure perfect vertical alignment.
- **Whitespace:** Double the internal padding of cards (e.g., target `24px`) to reduce visual clutter and let data breathe.

## 4. Page-by-Page Requirements
- **Dashboard (`dashboard-screen`):** Rebuild using CSS Grid. Include a large top hero card for "Next Race Prediction" and a bottom row of 3 smaller cards for key metrics.
- **Team/Lineup (`team-screen`):** Convert standard lists into visual "Driver Cards" displaying driver headshots/helmets, price, and predicted points in a grid layout.
- **Past Archive (`past-archive-screen`):** Refine tables by removing zebra striping, utilizing pure whitespace and thin bottom borders for row separation.
- **Settings (`settings-screen`):** Style toggle switches to mimic iOS toggles (smooth sliding motion, green active states).

## 5. Motion & Interaction
- **Instant Response (Tactile Feedback):** Implement CSS `:active` states on interactive elements (cards, buttons) that instantly scale down by 2% (`transform: scale(0.98)`).
- **Interruptible Navigation:** Update `apple-springs.js` to ensure transitions between sidebar tabs feel physical and smooth, utilizing critically damped springs (no bouncy overshoot) for fast, predictable settling.
