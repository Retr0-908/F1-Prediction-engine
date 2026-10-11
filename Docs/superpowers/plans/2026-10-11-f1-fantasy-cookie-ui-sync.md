# In-App F1 Fantasy Cookie Input & Live Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow users to paste and save their F1 Fantasy `login-session` cookie directly inside the web UI when authentication fails, persisting it to `.env` automatically, with an interactive step-by-step guide explaining how to retrieve it from browser DevTools.

**Architecture:** 
1. **Backend:** Provide a `POST /api/f1-sync/cookie` endpoint that sanitizes the submitted cookie, updates the in-memory runtime configuration (`engine.core.config.F1_FANTASY_COOKIE`), and persists it to `.env`.
2. **Frontend:** In `ui/index.html` and `ui/app.js`, enhance `#f1-sync-error` to automatically reveal an in-modal cookie input box, a "Save & Sync" button, and an expandable "📖 How to find your cookie" instructions card with visual steps for Chrome/Edge/Firefox.

**Tech Stack:** FastAPI, Python `dotenv`, Vanilla JavaScript, HTML5/CSS3

## Global Constraints
- Never commit active API keys, session tokens, or personal identifiers to git (guarded by `engine/tools/secret_scanner.py`).
- Maintain 100% green status on test suites (`test_ui_smoke.py`, `test_contracts.py`).
- Sanitize the user-pasted cookie so it handles both raw values (e.g. `%7B...`) and key-value formats (e.g. `login-session=%7B...`).
- Preserve any other existing keys in `.env` when updating `F1_FANTASY_COOKIE`.
- Track edits in `.agents/git_tracker.json`.

---

### Task 1: Backend Endpoint to Save & Persist Cookie (`POST /api/f1-sync/cookie`)

**Files:**
- Modify: `engine/core/fantasy_sync.py`
- Modify: `engine/serving/server.py`
- Test: `engine/tools/tests/test_ui_smoke.py`

**Interfaces:**
- Consumes: JSON payload `{"cookie": "..."}`
- Produces: `{"status": "ok", "message": "Cookie saved successfully"}`

- [ ] **Step 1: Write helper function `save_cookie_to_env(cookie_str: str) -> None` in `engine/core/fantasy_sync.py`**
  - Sanitizes input: strips quotes, ensures format `login-session=...` if key is omitted.
  - Updates in-memory: updates `os.environ["F1_FANTASY_COOKIE"]` and `config.F1_FANTASY_COOKIE`.
  - Persists to file: reads `.env`, updates or appends `F1_FANTASY_COOKIE=...`, writes atomically.

- [ ] **Step 2: Add API endpoint `POST /api/f1-sync/cookie` in `engine/serving/server.py`**
  - Accepts `{"cookie": str}`.
  - Calls `save_cookie_to_env(cookie)`.
  - Optionally does an immediate handshake test against `fetch_user_teams(cookie)` to validate.
  - Returns JSON `{"status": "ok", "message": "Cookie saved successfully", "teams": [...]}` on success or `{"status": "error", "message": "..."}` on invalid cookie.

- [ ] **Step 3: Add unit test in `engine/tools/tests/test_ui_smoke.py`**
  - Verify `POST /api/f1-sync/cookie` accepts payload, sanitizes, and returns expected JSON status.

---

### Task 2: Frontend UI - Cookie Input Form & Interactive Guide in Sync Modal

**Files:**
- Modify: `ui/index.html` (inside `#f1-sync-modal`)
- Modify: `ui/style.css` (modal form and accordion styling)
- Modify: `ui/app.js` (`openF1SyncModal`, `saveF1CookieAndSync`, `toggleCookieGuide`)

**Interfaces:**
- Consumes: `/api/f1-sync/teams` error response & `/api/f1-sync/cookie`
- Produces: In-modal reactive cookie submission, automatic retry, expandable guide.

- [ ] **Step 1: Add Cookie Input Form & Instructions Accordion to `ui/index.html`**
  Inside `#f1-sync-error` in `#f1-sync-modal`:
  - Input container:
    - Textarea/input `#f1-sync-cookie-input` with placeholder `"Paste login-session cookie here (starts with %7B or login-session=)..."`.
    - Button `#f1-sync-save-btn`: `"Save Cookie & Retry Sync"`.
    - Button `#f1-sync-help-btn`: `"📖 How to find your cookie"`.
  - Instructions panel `#f1-sync-guide` (hidden by default):
    - **Step 1:** Log in to [fantasy.formula1.com](https://fantasy.formula1.com).
    - **Step 2:** Press `F12` to open Developer Tools.
    - **Step 3:** Navigate to **Application** (Chrome/Edge) or **Storage** (Firefox) $\rightarrow$ **Cookies** $\rightarrow$ `https://fantasy.formula1.com`.
    - **Step 4:** Double-click the Value of `login-session` and copy it.
    - **Step 5:** Paste it into the box above and click **Save Cookie & Retry Sync**.

- [ ] **Step 2: Style the components in `ui/style.css`**
  - Modern dark-theme textarea with monospace font, accent focus border.
  - Styled action buttons with hover states matching Apple Design system tokens.
  - Animated smooth reveal for the instructions panel.

- [ ] **Step 3: Implement client-side logic in `ui/app.js`**
  - Update `openF1SyncModal()`: if sync fails with an authentication/cookie error, show `#f1-sync-cookie-box` and pre-focus the input.
  - Implement `saveF1CookieAndSync()`:
    - Validates non-empty input.
    - Sends `POST /api/f1-sync/cookie` with `{ cookie: val }`.
    - On success: hides error, shows loading spinner, and loads user teams into the cards grid.
    - On failure: displays the specific rejection message.
  - Implement `toggleCookieGuide()`: toggles display of `#f1-sync-guide`.

---

### Task 3: Verification & Test Suite Execution

**Files:**
- Run: `python -m unittest engine/tools/tests/test_ui_smoke.py engine/tools/tests/test_contracts.py`
- Run: `python engine/tools/secret_scanner.py`

- [ ] **Step 1: Run unit tests**
  Confirm test suites pass with 100% green status.

- [ ] **Step 2: Run security scan**
  Verify zero secrets or PII leaked to git.

- [ ] **Step 3: Update `.agents/git_tracker.json`**
  Increment and verify change tracking cadence.

---

### Task 4: User Walkthrough and Review
- Present completed feature flow to user with screenshots / DOM overview.
