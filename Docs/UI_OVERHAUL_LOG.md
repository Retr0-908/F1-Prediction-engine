================================================================================
F1 FANTASY PREDICTOR - UI OVERHAUL: COMPLETE WORK LOG
================================================================================
File written : 2026-10-18
Written by   : the AI agent, at the user's request
Scope        : the full UI overhaul (13 commits) PLUS the three post-launch bug
               fixes, including root causes, evidence and verification.
Repo         : https://github.com/Retr0-908/F1-Prediction-engine.git (origin)
Primary dir  : C:\Users\<USER>\Desktop\Tools\AI Assisted\Projects\F1-Prediction-engine-main
Worktree dir : C:\Users\<USER>\.local\share\opencode\worktree\201494\ui-overhaul
Base commit  : f5f31f7 "docs: add UI overhaul design and implementation plan"
HEAD commit  : a27b6a5 (both `main` and `ui-overhaul`)

--------------------------------------------------------------------------------
0. READ THIS FIRST - THE TWO-DIRECTORY TRAP
--------------------------------------------------------------------------------
There are TWO copies of this repository on disk and it caused a full wasted cycle:

  1. The PRIMARY checkout (where you actually run the app):
     C:\Users\<USER>\Desktop\Tools\AI Assisted\Projects\F1-Prediction-engine-main
     - branch `main`
     - this is what `F1 Fantasy.bat` launches

  2. A git WORKTREE (where the agent did the work):
     C:\Users\<USER>\.local\share\opencode\worktree\201494\ui-overhaul
     - branch `ui-overhaul`
     - a separate directory with its own checkout of the same repo

Why that mattered:
  - `F1 Fantasy.bat` starts with `cd /d "%~dp0"`, so it runs from the PRIMARY
    checkout.
  - `engine/core/paths.py` computes PROJECT_ROOT from `__file__` (the location of
    the package on disk), NOT from the current working directory:
        _here = Path(__file__).resolve()
        PROJECT_ROOT = _here.parents[2]      # ...\F1-Prediction-engine-main
  - `engine/serving/server.py` then does:
        UI_DIR = PROJECT_ROOT / "ui"
        app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")
    and serves index.html from UI_DIR.
  - The server binds 127.0.0.1:5762.

  So the running app ALWAYS serves the PRIMARY checkout's `ui/` folder, no matter
  where the worktree is. For the whole overhaul the primary checkout was still
  on `main` at the base commit f5f31f7, so the app looked completely unchanged.
  That is what "nothing has changed when i tried opening f1 fantasy.bat" was.
  The agent's fault: it treated "merge/deploy" as an optional hand-off step
  instead of confirming how the app is actually launched.

  Current status: `main` and `ui-overhaul` are BOTH at a27b6a5. The primary
  checkout now contains every change. Nothing has been pushed to origin.


================================================================================
1. SUMMARY OF THE WHOLE JOB
================================================================================
Objective: overhaul the F1 Fantasy Predictor UI end-to-end - visual polish,
responsive layout, CSS cleanup, replaced/merged sections - so it looks
professional and NEVER implies activity when idle, while verifying nothing breaks.

Hard requirements given by the user:
  - The app must not look like a prediction is running when it is idle.
  - The "10,000 sims" number must not appear while idle (only during a real run).
  - "Amateur look" spans ALL of: colours/typography, spacing/layout, and
    broken/pointless bits.
  - The user uses ALL screens - this is a section-level sweep, not screen removal.
  - The visual theme must be F1-themed and applied THROUGHOUT the app.
  - "implement it but verify nothing breaks."
  - Full test suite must stay 100% green.

Delivered: 13 commits, 6 files changed, 1162 insertions / 612 deletions.
  engine/serving/server.py            |   27 +-
  engine/tools/tests/test_ui_smoke.py |  158 ++-
  ui/app.js                           |  301 ++--
  ui/apple-springs.js                 |    2 +-
  ui/index.html                       |  130 ++--
  ui/style.css                        | 1156 ++++++----
Test suite: 91 tests, all passing, zero skips (89 at the end of the overhaul,
+2 guards added in the final bug-fix commit).


================================================================================
2. PHASE A - PLANNING (before any code)
--------------------------------------------------------------------------------
2.1 BRAINSTORMING
    Ran the superpowers `brainstorming` skill to pin down the visual direction and
    to stop the agent guessing at scope.

2.2 MOCKUPS + DESIGN CHOICE
    Presented visual-language options. The user chose:
      OPTION B - "Modern F1 App"
        - soft dark greys (not pure black, not the previous navy/glass look)
        - a SINGLE red accent: #E10600 (official F1 red)
        - large rounded cards
        - big, confident numbers
      Replacing the previous red/cyan mix.

2.3 APPROVED DESIGN DOC
    docs/superpowers/specs/2026-10-17-ui-overhaul-design.md
    Key content decisions recorded there:
      - DASHBOARD: keep the Weekend Hub (full race name is allowed to wrap);
        replace "AI Insights" and "Top Move" with a new WEEKEND WEATHER panel;
        keep SYSTEM STATUS but fix its truncation.
      - ANALYSIS: calm standby when idle; the sims counter and the "Model
        Calculation Parameters" HUD appear ONLY during a real run.
      - RESULTS: the legend collapses to a compact "?" button; "Driver Attribute
        Analysis" and "Driver Comparison Tool" merge into ONE "Driver Analysis"
        panel.
      - ARCHIVE: the "Model Health & Biases" sub-tab is renamed "Model Ops".
        The `data-subtab="tab-model-health"` attribute is deliberately NOT
        changed (tests/JS key off it).
      - RESPONSIVE: hamburger button + slide-in drawer below 1200px; the
        "Run Analysis" button stays in the mobile top bar.

2.4 PLAN
    docs/superpowers/plans/2026-10-17-ui-overhaul.md - a 10-task plan.
    Progress ledger (gitignored, on disk only):
      .superpowers/sdd/2026-10-17-ui-overhaul/progress.md

2.5 ISOLATED WORKSPACE
    Created a git worktree + branch so the overhaul could not disturb `main`:
      branch `ui-overhaul` from f5f31f7 in ...\worktree\201494\ui-overhaul
    (This decision is exactly what caused the later "nothing changed" confusion -
    see section 0 and section 6.)

2.6 RULINGS MADE DURING PLANNING (kept for the record)
    - The agent cannot render pixels. "Visual check" tasks were verified
      STRUCTURALLY and deferred to the user's manual pass. (Later proven wrong in
      a good way: a headless browser CAN render and measure - see section 7.0.)
    - Panel-header text stays UPPERCASE (acronym-dense; the sweep was not
      allocated to that).
    - `ui/apple-springs.js` left functionally unchanged.
    - `initStatusPulse` left unchanged.
    - Task 7 needed no JS change (all lookups are by id).
    - Cyan is retired as UI chrome and repointed to the red accent; it survives
      only as a DATA-SERIES colour (`.legend-fp .legend-dot`).
    - Semantic green (--color-green / --telemetry-green) is retained for positive
      status dots, points, bias chips and ensemble segments.
    - The F1-sync modal's inline styles were NOT extracted to classes
      (`display:none` is load-bearing JS state there); the visual goal was met by
      swapping tokens.
    - Dream-team avatars keep the documented Task 7 intent (60px circle); the
      assets are 1:1 transparent cutouts so `object-fit` is a no-op.
    - The `--team-color` whitelist in the token test is intentional - it fails
      loudly if future code invents a var() that is never defined.
    - Declined/deferred from review: drawer focus-trap + `inert`, and Weekend
      Weather column headers (pre-existing / behavioural, outside the idle-honesty
      remit).


================================================================================
3. PHASE B - IMPLEMENTATION (the 10 tasks, as 12 commits)
--------------------------------------------------------------------------------
Every commit below is in order, oldest first. All dated 2026-10-18.

TASK 1 - BASELINE
    Captured the starting state of ui/index.html, ui/style.css, ui/app.js and the
    two guarding test files so regressions could be detected.

TASK 2 - DESIGN TOKENS
    806887f  feat(ui): rebuild design tokens for F1 modern theme
    Rewrote the :root token set: soft dark grey surfaces (--surface-hi / --surface-lo),
    a single --accent-primary: #E10600, semantic greens retained, radii and
    spacing scales. This is the foundation the rest of the sweep uses.

TASK 3 - COMPONENTS
    db764c0  feat(ui): unify panels, buttons and typography
    Panels: one glass treatment, one radius, no per-panel overrides. Buttons: one
    primary/secondary/ghost system. Typography: consistent heading/body/mono roles.
    Important constraint discovered here: GSAP owns `transform` on `.panel`, `.btn`
    and nav buttons, so CSS must NOT add transform transitions to those (it would
    fight the spring animations).

TASK 4 - RESPONSIVE DRAWER NAVIGATION
    d594568  feat(ui): add responsive hamburger drawer navigation
    Added `initMobileNav()` in ui/app.js (hamburger toggle, backdrop, slide-in
    drawer, close on nav click / Escape / backdrop click) plus `#nav-toggle`,
    `#nav-backdrop` markup and the drawer CSS. Active below 1200px. "Run Analysis"
    deliberately remains in the mobile top bar so a run is always one tap away.

TASK 5 - ANALYSIS SCREEN: IDLE STANDBY + HONEST RUN STATE
    f53e1e5  feat(ui): analysis screen idle standby and honest run state
    The core "never imply activity when idle" change:
      - New `#analysis-standby` block ("Standby - no analysis running").
      - New `setAnalysisIdleState()` / `setAnalysisRunningState()` in ui/app.js.
      - CSS gates the HUD, the pipeline visualiser and the empty right-hand column
        behind `#analysis-screen.running`. When idle the screen is calm; the
        "10,000 SIMS" counter and the Model Calculation Parameters HUD exist only
        during a real run.
    NOTE: this is the mechanism that later made the stale-cache bug so confusing
    (old JS never added `.running`, so the new CSS kept the screen calm forever).

TASK 6 - DASHBOARD: WEEKEND WEATHER PANEL + STATUS FIX
    40734a3  feat(ui): dashboard weekend weather panel, full race name, status fix
      - New `#weekend-weather-panel` with `#weather-summary`,
        `#weather-session-list`, `#weather-impact-notes`.
      - New `renderWeekendWeather(weather)` in ui/app.js, wired into
        `loadDashboardData()` which fetches `/api/weather?race_name=...&date=...`
        and calls renderWeekendWeather(weather) - or renderWeekendWeather(null) when
        there is no forecast, which renders "Weather unavailable for this race
        weekend."
      - The Weekend Hub is allowed to wrap the full race name.
      - Status board: an initial, partial attempt at the truncation fix (see
        section 7.3 - it did NOT actually solve it; the real fix came in a27b6a5).

TASK 7 - MERGED DRIVER ANALYSIS + COMPACT LEGEND
    0bfc9ed  feat(ui): merged driver analysis panel and compact legend toggle
      - "Driver Attribute Analysis" + "Driver Comparison Tool" merged into a
        single "Driver Analysis" panel (two `.analysis-subsection` blocks separated
        by a divider).
      - The results legend collapses behind a compact "?" toggle to stop it
        dominating the results header.

TASK 8 - ARCHIVE SUB-TAB RENAME
    fdfc308  feat(ui): rename archive sub-tab to Model Ops
    Display text "Model Health & Biases" -> "Model Ops".
    `data-subtab="tab-model-health"` deliberately left unchanged.

TASK 9a - LAYOUT / BREAKPOINT / !important CLEANUP
    caadb6d  refactor(ui): remove 70px top-nav relics, rationalise breakpoints,
             drop !important
      - Removed the 70px top-nav relics from `.main-layout` / `.content-padding`
        (leftovers from a layout that no longer exists).
      - Consolidated the media queries: 14 @media blocks -> 7.
      - `!important` usage reduced from 109 -> 48.
      - Dead code removed.

TASK 9b - TOKEN AUDIT (in two commits)
    c7b448f  fix(ui): define --accent-red alias (spinner and import button
             rendered unstyled)
             Found by an audit, not by eye: `--accent-red` was referenced by the
             F1-sync spinner and the team import button but was NEVER DEFINED, so
             those two elements silently rendered unstyled.
    13388f2  refactor(ui): finish token audit for Task 9b - retire cyan HUD chrome
             to F1 accent, define --accent-danger, unify red, drop dead tokens
      - Cyan HUD chrome -> red.
      - `--accent-danger: var(--color-red)` defined.
      - The two different reds unified.
      - 6 dead tokens deleted.
    Token audit method (repeatable):
      definitions regex: (?m)^\s*(--[a-z0-9-]+)\s*:
      usages regex     : var\(\s*(--[a-z0-9-]+)
      Result: only `--team-color` is "undefined" and that is CORRECT - it is set
      per-element from JS and every use has a CSS fallback. Not a bug.
    Also: every legacy `rgba(232,0,45,...)` was changed to `rgba(225,6,0,...)` so
    there is exactly one brand red.

TASK 10 - FULL VERIFICATION + REVIEW
    See sections 4 (reviews) and 5 (verification). The Task 10 work produced these
    commits:
    b569c09  fix(ui): resolve code-review findings on the idle-honesty and accent
             consistency
    841bbe5  test(ui): clear re-review minors - drop stale nav-indicator CSS,
             harden guards
    a27b6a5  fix(ui): stop stale-asset caching and repair the squashed System
             Status panel      <-- the post-launch bug-fix commit (section 7)


================================================================================
4. PHASE C - CODE REVIEWS
--------------------------------------------------------------------------------
4.1 WHOLE-BRANCH REVIEW (f5f31f7..13388f2) -> verdict "No - with fixes"
    Findings and resolutions:

    CRITICAL - fixed in b569c09:
      `#analysis-standby` was a CHILD of `#analysis-animation-container`, and the
      SSE COMPLETE and ERROR handlers set that container to `display:none`.
      Consequence: after ANY run (especially a FAILED run) the idle screen was
      completely blank - no standby message, nothing. This is precisely the class
      of bug the overhaul was supposed to eliminate.
      Fix: removed both `display:none` assignments; `setAnalysisIdleState()` now
      explicitly restores `display:flex` on the container.

    IMPORTANT - fixed in b569c09:
      - The HUD and the empty `.analysis-right` were not properly gated off when
        idle; they are now hidden under `#analysis-screen:not(.running)` and the
        standby spans full width.
      - Residual cyan UI chrome repointed to the red accent: nav pill, nav spring
        glow, input focus ring, selected card, toast, driver hover, teal HUD.
      - The Spearman chart series colour changed to #FF9500 so it no longer
        collides with the brand red.
      - `.dt-thumb` / `.dt-explanation` duplicate rules merged.
      - Dead `.insights-icon` / `.summary-*` / `engine-running` rules removed.

    Also in that commit: 4 new test guards, and the nav-indicator orientation
    assertions were restored.

4.2 FOCUSED RE-REVIEW (13388f2..b569c09) -> verdict "Yes - ready to merge"
    Actionable minors cleared in 841bbe5:
      - Deleted the stale `#nav-spring-indicator` CSS block. Its
        `bottom: 0; height: 2px` was the forbidden horizontal orientation; GSAP
        owns height/opacity, so the CSS was both dead and wrong.

4.3 DECLINED / DEFERRED (explicitly, with reasons)
    - Mobile drawer focus trap and `inert` on background content: real
      accessibility work, but pre-existing and outside the reported remit.
    - Weekend Weather column headers: a content/behaviour question, not a bug.
    - The `--team-color` "undefined token": intentional (see 2.6).


================================================================================
5. PHASE D - VERIFICATION METHODOLOGY (and exactly what was run)
--------------------------------------------------------------------------------
5.1 THE STANDING TEST COMMAND
    python -m unittest engine/tools/tests/test_ui_smoke.py engine/tools/tests/test_contracts.py
    Output was filtered through:
      Select-String -Pattern 'Ran \d+ tests?|^OK|^FAILED|skipped|^FAIL: |^ERROR: '
    DELIBERATELY never filtered on "AssertionError": an `assertIn` failure dumps
    the entire served file as one giant line and makes the output unreadable.

    Results:
      - End of overhaul (Task 10): Ran 89 tests, OK, zero skips.
      - After a27b6a5             : Ran 91 tests, OK, zero skips (3.8s, cache warm).

5.2 JS SYNTAX
    node --check ui/app.js
    node --check ui/apple-springs.js
    Essential: the Python suite CANNOT catch a JavaScript syntax error, and the SPA
    would silently do nothing. Both exit 0.

5.3 CSS BRACE BALANCE
    Counted '{' vs '}' in ui/style.css with [regex]::Matches.
    Final: open=642 close=642, balanced.

5.4 HTML / CSS / TOKEN AUDITS
    - CSS token definition/usage cross-check (see 3, Task 9b).
    - Removed-sensitive-string checks after each rename (e.g. no leftover
      "Model Health & Biases" in shipped UI text).

5.5 THIS IS THE ONE THAT SHOULD HAVE BEEN USED FROM THE START: A REAL BROWSER
    Playwright + Chromium was available all along. The agent drove the actual
    running server headlessly and read:
      - console messages and pageerrors,
      - every network response and status code,
      - computed styles (getComputedStyle),
      - exact element geometry (getBoundingClientRect, scrollWidth/clientWidth),
      - and it captured PNG screenshots which the agent could then LOOK AT.
    That is how the three post-launch bugs were root-caused instead of guessed at.
    Scripts were written to C:\Users\<USER>\AppData\Local\Temp\opencode\:
      diag_ui.py, diag_geom.py, diag_verify.py, diag_final.py
    Screenshots:
      diag_dashboard.png, diag_analysis.png (first diagnosis)
      diag_dash_1440.png, diag_dash_1280.png, diag_dash_1100.png, diag_dash_768.png
      verify_1920.png, verify_1440.png, verify_1280.png (after the status fix)

5.6 SERVED-HEADERS CHECK
    Invoke-WebRequest -Uri http://127.0.0.1:5762/ -UseBasicParsing
    ...then reading .Headers['Cache-Control'].

5.7 OPERATIONAL GOTCHAS DISCOVERED
    - PowerShell has no `grep`; use the dedicated grep tool.
    - In PowerShell, `Write-Output "EXIT=$LASTEXITCODE"` is needed to read a real
      exit code; the "Exited with code 1" line is a stderr deprecation artefact
      and is NOT the exit code.
    - Edits via `[System.IO.File]::ReadAllText/WriteAllText` preserve BOM and line
      endings; the edit tool does the same automatically.
    - The commit hook runs engine/tools/secret_scanner.py (a "nuclear" secret/PII
      scanner) and BLOCKS a commit on failure. It passed on every commit.
    - A PowerShell here-string containing unescaped `"` characters TERMINATES
      EARLY and mangles `git commit -m`. Use `git commit -F <file>`.
    - TestClient tests require httpx; `test_stage_comparison_api` is cache-warm via
      output/stages/comparison_round_15.json, so the suite is fast (3.8s) once warm.


================================================================================
6. PHASE E - THE "NOTHING CHANGED" INCIDENT
--------------------------------------------------------------------------------
Symptom reported by the user:
    "nothing has changed when i tried opening f1 fantasy.bat. what did you even do
     all this time?"

Root cause (fully traced, every link verified):
    1. F1 Fantasy.bat does  cd /d "%~dp0"  -> runs from the PRIMARY checkout.
    2. It runs  python -m engine.serving.server  from there.
    3. engine/core/paths.py derives PROJECT_ROOT from __file__ (parents[2]), i.e.
       the PRIMARY checkout - not the worktree.
    4. server.py sets UI_DIR = PROJECT_ROOT / "ui" and mounts it at /static,
       serving index.html from it. Port 127.0.0.1:5762.
    5. The primary checkout's `main` was still at the BASE commit f5f31f7 - the
       original, untouched UI.
    => The 12 overhaul commits existed only inside the worktree directory, which
       the launcher never reads.

Fix applied:
    `main` was sitting exactly at the branch base, so a fast-forward was possible
    with no merge commit and no conflicts:
      git -C "<primary checkout>" merge --ff-only ui-overhaul
    -> `main` advanced f5f31f7 -> 841bbe5.
    The user's uncommitted README.md change was NOT touched (verified before and
    after; the branch never modified README.md).
    Verified by booting the server once: `/` and `/static/style.css` returned the
    new markup and the new tokens, then the test server was stopped to free 5762.

    Undo, if ever needed:
      git -C "<primary checkout>" reset --hard f5f31f7
    (The commits stay reachable on the `ui-overhaul` branch.)

Lesson recorded: confirm the LAUNCH PATH and deploy to it before reporting
completion.


================================================================================
7. PHASE F - THE THREE POST-LAUNCH BUGS (reported by the user after section 6)
--------------------------------------------------------------------------------
User report, verbatim in substance:
    1. "when i click analyse, the analysis screen does not change to show analysis
        is currently happening. it only shows the results at the end."
    2. "the system status in the main page is still squashed."
    3. "the weather weekend section does not update at all. it simply says loading
        session forecast and stays like that"

--------------------------------------------------------------------------------
7.0 METHOD
--------------------------------------------------------------------------------
Ran the `systematic-debugging` skill: Phase 1 root-cause investigation FIRST, no
fixes until the cause was proven. Evidence came from a headless Chromium driving
the user's actual server. No guessing.

--------------------------------------------------------------------------------
7.1 BUGS 1 AND 3 - ONE ROOT CAUSE: STALE CACHED ASSETS
--------------------------------------------------------------------------------
Evidence trail:
  a) The user COULD see the new Weekend Weather panel (new HTML was loaded) but it
     never left its placeholder. The placeholder text "Loading session forecast..."
     lives INSIDE `#weather-session-list` in the new index.html, so the element
     existed - meaning `renderWeekendWeather()` either ran and returned early
     (impossible: the element exists) or was NEVER CALLED. The only way it is
     never called is if the running app.js has no such function.
  b) In a CLEAN browser, the panel rendered perfectly: summary, all five session
     rows (FP1/FP2/FP3/QUALI/RACE with temps, rain %, wind) and the RACE IMPACT
     notes. So the committed code is correct.
  c) In a CLEAN browser, clicking Analyse DID show the live state: the
     `#analysis-screen` gained `.running`, the standby hid, and the visualiser,
     HUD and right column all appeared (counter "10,000 / 10,000 SIMS", phase
     "Optimizing Team Roster via Integer Linear Programming"). So bug 1 was also
     not a code bug.
  d) The smoking gun - `ui/index.html` loads the app with cache-buster tokens, and
     they were IDENTICAL before and after the overhaul:
          /static/style.css?v=12      (unchanged)
          /static/app.js?v=36         (unchanged)
          /static/apple-springs.js?v=2(unchanged)
     `git diff f5f31f7..HEAD -- ui/index.html` contained NO change to any `?v=`
     line, despite app.js and style.css being rewritten by 1000+ lines.
  e) `app.mount("/static", StaticFiles(...))` set no `Cache-Control` header, so
     browsers applied HEURISTIC freshness and happily reused the old files.

  Mechanism: the browser loaded the NEW index.html but kept the OLD
  `app.js?v=36` and `style.css?v=12` from cache. Old JS has no
  `renderWeekendWeather` (bug 3) and never adds the `.running` class the new CSS
  gates on, so the new CSS kept the Analysis screen calm forever while the pipeline
  ran server-side and results appeared at the end (bug 1).

Fix (in a27b6a5):
  1. Bumped the tokens:
        style.css      v12 -> v13
        app.js         v36 -> v37
        apple-springs.js v2 -> v3
  2. Made staleness impossible in future. In engine/serving/server.py:
        class _NoCacheStaticFiles(StaticFiles):
            async def get_response(self, path, scope):
                response = await super().get_response(path, scope)
                response.headers["Cache-Control"] = "no-cache"
                return response
        app.mount("/static", _NoCacheStaticFiles(directory=str(UI_DIR)), name="static")

        @app.get("/")
        def read_index():
            ...
            return FileResponse(index_path, headers={"Cache-Control": "no-cache"})
     Deliberately NOT an `@app.middleware("http")`: an app-level middleware wraps
     every request including the SSE progress stream, and Starlette's
     BaseHTTPMiddleware has historically interfered with streaming responses.
     The subclass approach cannot touch the SSE path.
     `no-cache` (NOT `no-store`) was chosen because it still permits efficient 304
     revalidation - the browser just must revalidate before reusing a file.

--------------------------------------------------------------------------------
7.2 BUG 2 - "SYSTEM STATUS STILL SQUASHED" - A GENUINE, PRE-EXISTING LAYOUT BUG
--------------------------------------------------------------------------------
The user's word "still" was accurate. Reproduced with a fresh browser, so this was
NOT a cache artefact.

Measured evidence (headless Chromium, 1440x900, fresh load):
  - Dashboard grid resolved to  THREE columns: 470.578px | 261.438px | 351.969px
    (Weekend Hub | System Status | Weekend Weather).
  - `.col-center` was therefore only ~261px wide (and ~218px at 1280px).
  - `.status-label` width was 45.7px, so "Next Race" rendered as "Nex...";
    "Predictive Engine" -> "Pre...", "Local Cache" -> "Local...",
    "Weather Service" -> "Weat...", "User Profile" -> "Use...".
  - `.status-val` height was 68px on the first row (the race name wrapped to 3
    lines) while the label stayed 1 line, vertically centred against it - which is
    the "jumble/overlap" look the user described.

  The offending CSS was the ORIGINAL (pre-overhaul) row:
      .status-row { display:flex; align-items:center; justify-content:space-between; }
      .status-label { flex:1; white-space:nowrap; overflow:hidden;
                      text-overflow:ellipsis; }
      .status-val   { max-width:60%; white-space:nowrap; overflow:hidden;
                      text-overflow:ellipsis; }
  The 3-column split `grid-template-columns: 1.8fr 1fr 1.2fr` at min-width:1200px
  also predates the overhaul (it is present at f5f31f7).
  The overhaul's Task 6 "D4" change (`.status-val { white-space: normal }`) turned
  the clipped value into a WRAPPING value - honest, but it made the tall wrapped
  value sit against a 1-line label, which is why it looked worse/overlapping.

Fix (in a27b6a5) - an ADAPTIVE row instead of a rigid inline row:
    .status-row {
        display: flex;
        flex-wrap: wrap;              /* the key change */
        align-items: baseline;
        column-gap: 12px;
        row-gap: 2px;
        ...
    }
    .status-indicator { flex: 0 0 auto; align-self: center; }
    .status-label {
        font-size: 0.9rem;
        flex: 0 0 auto;               /* NEVER shrinks -> never truncated */
        white-space: nowrap;
    }
    .status-val {
        font-size: 0.9rem;
        text-align: right;
        min-width: 0;
        white-space: normal;
        overflow-wrap: break-word;    /* words wrap whole, not mid-word */
        flex: 1 1 120px;              /* sits beside the label when there is
                                         room; drops to its own full-width line
                                         when there is not */
    }
  Plus two supporting changes:
    - `.status-val` was REMOVED from the mono font group
      (`.telemetry-num, .lap-time, .delta-val, .metric-pill-val`) - mono rendered
      the values ~25% wider and is what caused mid-word shattering like
      "Baye/sian/(MCM/C)" at 1280px. Status values are prose, not telemetry.
    - Desktop split changed 1.8fr 1fr 1.2fr -> 1.8fr 1.1fr 1.1fr. Weekend Hub stays
      at 1.8fr so the "SESSION TIMES" header keeps fitting on one line; the extra
      width for System Status comes out of the weather column.
      (Note discovered while measuring: the weather panel has a min-content width
      of ~352px, so it cannot actually shrink below that - the fr change mostly
      redistributes Hub vs Status.)

  Iterations that were tried and REJECTED, for the record:
    - A `grid-template-columns: auto auto minmax(0,1fr)` row: fixed truncation but
      at 1280px the value column was ~45px and even proportional text shattered
      ("Ba/ye/si/an"). Rejected.
    - `overflow-wrap: anywhere`: breaks mid-word eagerly. Rejected in favour of
      `break-word`.

--------------------------------------------------------------------------------
7.3 VERIFICATION OF THE BUG FIXES (all green)
--------------------------------------------------------------------------------
Headless Chromium, dashboard, BEFORE the fix:
      1440  470.578 | 261.438 | 351.969   label width 45.7px  ("Nex...")
      1280     369.547 | 218.484 | 335.969   label width 38.5px  ("Nex...")

AFTER the fix:
      1920  703.797 | 430.094 | 430.109   all labels full, no overlap
      1440  454.359 | 277.656 | 351.969   all labels full, no overlap
      1280  364.984 | 223.031 | 335.969   all labels full, no overlap
    At every width: all five `.status-label` and all five `.status-val` reported
    `scrollWidth <= clientWidth` (i.e. `truncated: false`), and no row overlapped
    the next.

Final end-to-end check against the PRIMARY checkout's own server:
      /            Cache-Control: no-cache
      /static/app.js Cache-Control: no-cache
      index.html references: /static/style.css?v=13
                             /static/apple-springs.js?v=3
                             /static/app.js?v=37
      status labels : Next Race / Predictive Engine / Local Cache / Weather Service
                      / User Profile          -> all truncated: false
      status values : "United States Grand Prix . R18" / "Bayesian (MCMC)" /
                      "649 sessions" / "Low risk" / "5 drivers selected"
                                              -> all truncated: false
      weather       : summary present, 5 session rows, placeholder GONE
      analysis      : startAnalysis() enters `running`; standby display:none;
                      visualiser display:block; HUD display:block;
                      counter "10,000 / 10,000 SIMS"
      console errors: none
      page errors   : none
      port 5762     : released

Two guard tests were added so these cannot silently return
(in engine/tools/tests/test_ui_smoke.py):
    test_ui_assets_are_never_served_stale
        - asserts Cache-Control: no-cache on "/" and on /static/app.js and
          /static/style.css,
        - asserts index.html references versioned assets ("?v=").
    test_status_board_rows_do_not_truncate_labels
        - asserts .status-row has flex-wrap: wrap,
        - asserts .status-label has NO text-overflow: ellipsis and IS flex: 0 0 auto,
        - asserts .status-val is flex: 1 1 120px with overflow-wrap: break-word.

One operational discovery: a STALE PYTHON SERVER the user had left running on
port 5762 (started 12:02) was answering the agent's verification requests with the
OLD (main@841bbe5) files and masking results. It was identified via
Get-NetTCPConnection -LocalPort 5762 -> OwningProcess and killed. Every subsequent
run asserted the port was free before starting and that the served CSS contained a
unique new marker, so the agent could prove it was testing its own build.


================================================================================
8. CURRENT STATE (as of this file being written)
--------------------------------------------------------------------------------
Branch `ui-overhaul` (worktree) HEAD : a27b6a5
Branch `main`       (primary)  HEAD : a27b6a5
Commits on the overhaul              : 13   (f5f31f7 -> a27b6a5)
Files changed (cumulative)           : 6    (+1162 / -612)
Tests                                : 91, OK, zero skips
Push state                           : NOT pushed. origin is
                                       https://github.com/Retr0-908/F1-Prediction-engine.git
Untouched in the primary checkout    : README.md (user's own modification)
                                       Screenshots/Errors/ (user's untracked folder)
Port 5762                            : free (agent's throwaway servers all stopped)

Full commit list, oldest first:
  806887f  feat(ui): rebuild design tokens for F1 modern theme
  db764c0  feat(ui): unify panels, buttons and typography
  d594568  feat(ui): add responsive hamburger drawer navigation
  f53e1e5  feat(ui): analysis screen idle standby and honest run state
  40734a3  feat(ui): dashboard weekend weather panel, full race name, status fix
  0bfc9ed  feat(ui): merged driver analysis panel and compact legend toggle
  fdfc308  feat(ui): rename archive sub-tab to Model Ops
  caadb6d  refactor(ui): remove 70px top-nav relics, rationalise breakpoints,
           drop !important
  c7b448f  fix(ui): define --accent-red alias (spinner and import button rendered
           unstyled)
  13388f2  refactor(ui): finish token audit for Task 9b - retire cyan HUD chrome
           to F1 accent, define --accent-danger, unify red, drop dead tokens
  b569c09  fix(ui): resolve code-review findings on the idle-honesty and accent
           consistency
  841bbe5  test(ui): clear re-review minors - drop stale nav-indicator CSS, harden
           guards
  a27b6a5  fix(ui): stop stale-asset caching and repair the squashed System Status
           panel


================================================================================
9. WHAT YOU NEED TO DO NOW
--------------------------------------------------------------------------------
  1. Close any open "F1 Fantasy Engine" server window.
  2. Run `F1 Fantasy.bat` (from the primary checkout - that is where it lives).
  3. Open the app and press  Ctrl + Shift + R  ONCE (a hard refresh).
     Why this matters: your browser is still holding the old index.html and the
     old app.js?v=36 from before the fix. The hard refresh forces it to pick up
     the new asset URLs. From that point on the new `Cache-Control: no-cache`
     headers keep the app current automatically - this should never happen again.

  What you should then see:
    - Dashboard: the System Status panel with full labels ("Predictive Engine",
      "Weather Service", ...) and values that wrap cleanly instead of being cut
      off. The Weekend Weather panel populated with FP1/FP2/FP3/QUALI/RACE.
    - Analysis: a calm "standby" panel when idle; when you click Run Analysis, the
      screen visibly switches into the running state (pipeline visualiser, the sims
      counter and the Model Calculation Parameters HUD) and returns to standby
      afterwards - including after a FAILED run.

  Nothing has been pushed to GitHub. Say the word if you want me to push `main`.


================================================================================
10. KNOWN OUTSTANDING ITEMS (not bugs in the reported set)
--------------------------------------------------------------------------------
  a) Between 1200px and ~1399px the 3-column desktop dashboard is inherently tight:
     the System Status column lands around 220-280px. It is legible now (no
     truncation, no mid-word breaks), but the values do wrap to two lines. If you
     would rather the dashboard fall back to the 2-column layout below ~1400px
     (Weekend Hub full width on top, System Status and Weather side by side below),
     that is a small change - just say so.
  b) COSMETIC, PRE-EXISTING, NOT FIXED: at widths up to roughly 1366px the
     "SESSION TIMES" panel header wraps onto three lines ("SESSI / ON / TIMES")
     because the timezone <select> is as wide as its longest option
     ("Africa/Johannesburg") and squeezes the header text. Independent of the
     overhaul's changes at 1280px; it only became visible at 1440px briefly during
     an intermediate iteration and was resolved by keeping Weekend Hub at 1.8fr.
  c) Deferred from review, documented, still open: the mobile drawer has no focus
     trap and does not set `inert` on the background content; and the Weekend
     Weather rows have no column headers.
  d) The git worktree at ...\.local\share\opencode\worktree\201494\ui-overhaul still
     exists and now points at the same commit as `main`. It can be removed with
     `git worktree remove` once you are happy - nothing is stranded on it.


================================================================================
11. APPENDIX A - KEY FILE INVENTORY (what lives where)
--------------------------------------------------------------------------------
  ui/style.css
      :root token block near the top; the D3 idle-gating block around line 3750+;
      `.status-board` / `.status-row` / `.status-label` / `.status-val` around
      lines 600-660; the D4 note around line 3940; the responsive breakpoint matrix
      near lines 2650-2710 (Tier 2 tablet 600-1199, Tier 3 desktop >=1200).
  ui/app.js
      DOMContentLoaded bootstrap at line 8 (setupHeartbeat, initApp,
      setupNavigation, initMobileNav, setupTabs, setupArchiveAnalyticsSubtabs,
      setupF1SyncButton).
      initApp() ~line 73; loadDashboardData() ~line 466; the weather fetch ~line
      513 with renderWeekendWeather at ~549; loadAnalysisStandbyInfo() ~1267;
      startAnalysis() ~1349; setAnalysisIdleState()/setAnalysisRunningState() in
      the analysis section; renderWeekendWeather() ~2346; the _WX_* helper tables
      ~2310; the app.js <script> tag at line 1038.
  ui/index.html
      Head: style.css?v=13, apple-springs.js?v=3; body ends with app.js?v=37.
      Screens: launch (20), dashboard (83), team (192), prices (216), analysis
      (234), results (342), settings (547), standings (628), archive-analytics
      (646). Dashboard panels: Weekend Hub (col-left, 85), System Status
      (col-center, 144), Weekend Weather (col-right, 176, with
      #weather-session-list at 180). Analysis screen 234-339 contains
      #analysis-animation-container, #analysis-standby and .hud-panel.
  ui/apple-springs.js
      Nav spring indicator: inline styles + GSAP own its height/opacity.
  engine/serving/server.py
      UI_DIR = PROJECT_ROOT / "ui"  (~line 77);
      class _NoCacheStaticFiles(StaticFiles)  (new);
      app.mount("/static", _NoCacheStaticFiles(...), name="static");
      @app.get("/") returns FileResponse(index_path,
      headers={"Cache-Control": "no-cache"});
      uvicorn.Config("engine.serving.server:app", host="127.0.0.1", port=5762)
      ~line 800.
  engine/core/paths.py
      PROJECT_ROOT derived from __file__.parents[2]; UI_DIR = PROJECT_ROOT / "ui".
  engine/tools/tests/test_ui_smoke.py
      The guarding suite, including the generic asset tests and the two new guards
      described in 7.3.
  engine/tools/tests/test_contracts.py
      The other guarding suite.


================================================================================
12. APPENDIX B - EXACT COMMANDS USED (copy/paste reference)
--------------------------------------------------------------------------------
Run the full guarding suite:
    python -m unittest engine/tools/tests/test_ui_smoke.py engine/tools/tests/test_contracts.py

Fast, readable pass/fail:
    ... 2>&1 | Select-String -Pattern 'Ran \d+ tests?|^OK|^FAILED|skipped|^FAIL: |^ERROR: '

JS syntax:
    node --check ui/app.js
    node --check ui/apple-springs.js

CSS brace balance:
    $css = [System.IO.File]::ReadAllText("$PWD\ui\style.css")
    [regex]::Matches($css,'\{').Count ; [regex]::Matches($css,'\}').Count

Token audit:
    definitions: (?m)^\s*(--[a-z0-9-]+)\s*:
    usages     : var\(\s*(--[a-z0-9-]+)

Start the server by hand (what the .bat does):
    python -m engine.serving.server          # binds 127.0.0.1:5762

Check what the server is really serving:
    Invoke-WebRequest http://127.0.0.1:5762/ -UseBasicParsing
    Invoke-WebRequest http://127.0.0.1:5762/static/style.css -UseBasicParsing

Find what is holding the port (this caught the stale server):
    Get-NetTCPConnection -State Listen -LocalPort 5762 | Select OwningProcess
    Get-Process -Id <pid>

Fast-forward the primary checkout after committing on the branch:
    git -C "<primary checkout>" merge --ff-only ui-overhaul

Commit with a multi-line message safely (PowerShell here-strings mangle quotes):
    git add <files>
    git commit -F <message-file>

Undo the whole overhaul on main (commits stay on the branch):
    git -C "<primary checkout>" reset --hard f5f31f7


================================================================================
13. APPENDIX C - LESSONS LEARNED
--------------------------------------------------------------------------------
  1. A git worktree is a DIFFERENT DIRECTORY. If the app is launched from the
     primary checkout, work committed in a worktree is invisible until it is
     merged/fast-forwarded there. Confirm the launch path and deploy to it BEFORE
     reporting completion.
  2. "I cannot render pixels" was wrong. A headless browser (Playwright/Chromium)
     can load the real app, execute the real JS, read geometry and computed styles,
     capture console errors, and take screenshots the agent can look at. Use it
     FIRST for any UI claim.
  3. Asset cache-buster tokens (?v=N) must be bumped whenever the asset changes.
     Otherwise the browser runs OLD JS against NEW HTML and produces nonsensical
     symptoms that look like broken code. Serve an unversioned or no-cache shell.
  4. Don't declare "not a bug" from a partial measurement. The first status-board
     check only measured `.status-val` (the values) and concluded "not squashed";
     the user was right and the label truncation/vertical jumble was measurable the
     whole time.
  5. A leftover server process on the app's port will silently serve old files and
     invalidate verification. Assert the port is free and assert a unique marker is
     present in what is served.
  6. In PowerShell, unescaped double quotes inside a here-string terminate it
     early. Use `git commit -F file` for multi-line messages.
  7. Keep the test suite as the safety net, but know its blind spots: it cannot
     catch a JS syntax error (hence `node --check`) and it cannot see layout. The
     headless browser and the syntax checks are what cover those.


================================================================================
END OF LOG
================================================================================
