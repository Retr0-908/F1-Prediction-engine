# Public Release & Repository Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the F1-Prediction-engine repository into a secure, showcase-grade public open-source project under GNU GPLv3 with purged credentials, streamlined dependencies, automated CI, and verified UI.

**Architecture:** Purge historical git credentials via `git-filter-repo`; refactor dependencies into tiered requirements; mount UI static paths reliably in FastAPI; add GNU GPLv3 license, CI workflows, and a showcase README with architecture diagrams, screenshots, and disclaimers.

**Tech Stack:** Python 3.10+, FastAPI, FastF1, Scikit-learn, XGBoost, LightGBM, GitHub Actions, Git, Vanilla JS / CSS.

## Global Constraints
- Target License: GNU General Public License v3.0 (GPL-3.0) — author: Retr0-908.
- Sensitive files: `.env` must NEVER be tracked or exist in any Git commit or remote packfile.
- Remote URL: Must NOT contain any embedded personal access tokens (`ghp_...`).
- Dependencies: `pymc` and `arviz` removed; `juliacall` kept optional; tiered `requirements.txt` and `requirements-dev.txt`.
- Existing Windows launchers (`.bat`) must be preserved; POSIX `.sh` launchers added for cross-platform compatibility.

---

### Task 1: Pre-Flight Safety Backup & Working Tree Stabilization

**Files:**
- Modify: `engine/__init__.py`, `engine/analysis/analyse_results.py`, `engine/analysis/post_race_check.py`, `engine/core/warm_cache.py`, `engine/models/elo_ratings.py`, `engine/models/predictor.py`, `engine/serving/pipeline.py`, `engine/serving/server.py`, `main.py`

**Interfaces:**
- Consumes: Current uncommitted working tree diffs.
- Produces: Clean working tree on `main` and safety backup branch `backup-pre-release`.

- [ ] **Step 1: Create local safety git branch**

```bash
git branch backup-pre-release
```

- [ ] **Step 2: Commit pending bugfixes and hardening changes**

```bash
git add engine/__init__.py engine/analysis/analyse_results.py engine/analysis/post_race_check.py engine/core/warm_cache.py engine/models/elo_ratings.py engine/models/predictor.py engine/serving/pipeline.py engine/serving/server.py main.py
git commit -m "fix: engine hardening, import hygiene, and roster bounds validation"
```

- [ ] **Step 3: Run existing contract tests to verify zero regressions**

Run: `python -m unittest engine.tools.tests.test_contracts`  
Expected: `Ran 41 tests ... OK`

---

### Task 2: Git History Sanitization & Remote Token Scrub

**Files:**
- Modify: Git commit history and `.git/config`

**Interfaces:**
- Consumes: Commit `a5e6143` containing sensitive `.env` file.
- Produces: Clean git history where `.env` does not exist in any commit; sanitized remote origin URL without PAT.

- [ ] **Step 1: Check and install `git-filter-repo`**

```bash
python -m pip install git-filter-repo
```

- [ ] **Step 2: Execute history filter to permanently purge `.env` from all commits**

```bash
git filter-repo --path .env --invert-paths --force
```

- [ ] **Step 3: Verify git history is completely clean of `.env`**

Run: `git log --all --full-history -- .env`  
Expected: Zero commits output.

- [ ] **Step 4: Sanitize remote origin URL**

```bash
git remote add origin https://github.com/Retr0-908/F1-Prediction-engine.git || git remote set-url origin https://github.com/Retr0-908/F1-Prediction-engine.git
```

- [ ] **Step 5: Verify remote URL has no embedded token**

Run: `git remote -v`  
Expected: `origin  https://github.com/Retr0-908/F1-Prediction-engine.git (fetch)` and `(push)` with zero tokens.

---

### Task 3: License (GNU GPLv3) & Safe Environment Template

**Files:**
- Create: `LICENSE`
- Create: `.env.example`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Project legal requirements and environment configuration.
- Produces: GNU GPLv3 legal protection and template for end-user environment variables.

- [ ] **Step 1: Write `LICENSE` with GNU General Public License v3.0**

Write full GPL-3.0 text with Copyright (C) 2026 Retr0-908 to root `LICENSE`.

- [ ] **Step 2: Create `.env.example`**

```env
# ==============================================================================
# F1 Prediction Engine - Environment Configuration Template
# Copy this file to .env and populate your personal values (do not commit .env)
# ==============================================================================

# Optional: OpenWeatherMap API key for circuit weather enrichment
# (If omitted, engine defaults to free Open-Meteo forecasts)
OPENWEATHERMAP_API_KEY=your_openweathermap_api_key_here

# Optional: F1 Fantasy session cookie for automated lineup syncing & live pricing
# Paste your cookie from the browser session after logging into fantasy.formula1.com
F1_FANTASY_COOKIE=
```

- [ ] **Step 3: Confirm `.gitignore` protects sensitive and generated files**

Ensure `.gitignore` contains:
```gitignore
cache/
__pycache__/
*.pyc
Screenshots/
logs/*
!logs/.gitkeep
*.log
.env
output/*.csv
output/*.json
output/*.txt
output/*.html
!output/.gitkeep
```

- [ ] **Step 4: Commit license and environment template**

```bash
git add LICENSE .env.example .gitignore
git commit -m "chore: add GNU GPLv3 license, .env.example template, and verify gitignore"
```

---

### Task 4: Dependency Management & Cross-Platform Launchers

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `run_app.sh`
- Create: `backtest.sh`

**Interfaces:**
- Consumes: Current unpruned `requirements.txt`.
- Produces: Tiered dependencies (`requirements.txt` vs `requirements-dev.txt`) and POSIX executable scripts.

- [ ] **Step 1: Update `requirements.txt` with clean, pruned dependencies**

```text
fastf1>=3.3.0
requests>=2.31.0
beautifulsoup4>=4.12.0
pandas>=2.0.0
numpy>=1.26.0,<2.0.0
scikit-learn>=1.4.0
scipy>=1.13.0
xgboost>=2.0.0
lightgbm>=4.0.0
python-dotenv>=1.0.0
rich>=13.7.0
tabulate>=0.9.0
joblib>=1.3.0
lxml>=5.0.0
questionary>=2.0.0
fastapi>=0.110.0
uvicorn>=0.29.0
sse-starlette>=2.0.0
tensorflow>=2.15.0
pulp>=2.7.0
```

- [ ] **Step 2: Create `requirements-dev.txt`**

```text
-r requirements.txt
pytest>=8.0.0
playwright>=1.40.0
ruff>=0.3.0
```

- [ ] **Step 3: Create POSIX launch scripts `run_app.sh` and `backtest.sh`**

`run_app.sh`:
```bash
#!/usr/bin/env bash
set -e
echo "Starting F1 Fantasy Prediction Engine..."
python server.py
```

`backtest.sh`:
```bash
#!/usr/bin/env bash
set -e
echo "Running Multi-Season Backtest..."
python -m engine.analysis.backtest "$@"
```

- [ ] **Step 4: Commit dependency and launcher updates**

```bash
git add requirements.txt requirements-dev.txt run_app.sh backtest.sh
git commit -m "refactor: tier requirements, prune unused dependencies, and add POSIX launchers"
```

---

### Task 5: UI Robustness & Server Mounting Fix

**Files:**
- Modify: `engine/serving/server.py:80-82`
- Test: Smoke test script `engine/tools/tests/test_ui_smoke.py`

**Interfaces:**
- Consumes: UI directory and FastAPI application mount.
- Produces: Reliable static file delivery and verified HTTP API endpoints.

- [ ] **Step 1: Anchor static mount to `UI_DIR` in `engine/serving/server.py`**

In `engine/serving/server.py`:
Change:
```python
app.mount("/static", StaticFiles(directory="ui"), name="static")
```
To:
```python
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")
```

- [ ] **Step 2: Write UI & API smoke test script**

Create `engine/tools/tests/test_ui_smoke.py`:
```python
import unittest
from fastapi.testclient import TestClient
from engine.serving.server import app

class UISmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_index_html(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("F1 FANTASY PREDICTOR", res.text)

    def test_static_assets_exist(self):
        res_css = self.client.get("/static/style.css")
        self.assertEqual(res_css.status_code, 200)
        res_js = self.client.get("/static/app.js")
        self.assertEqual(res_js.status_code, 200)
        res_springs = self.client.get("/static/apple-springs.js")
        self.assertEqual(res_springs.status_code, 200)

    def test_api_status(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json().get("status"), "ok")

    def test_api_system_health(self):
        res = self.client.get("/api/system/health")
        self.assertEqual(res.status_code, 200)
        self.assertIn("status", res.json())
```

- [ ] **Step 3: Run smoke tests and verify passing**

Run: `python -m unittest engine.tools.tests.test_ui_smoke`  
Expected: `Ran 4 tests ... OK`

- [ ] **Step 4: Commit UI fix and smoke test**

```bash
git add engine/serving/server.py engine/tools/tests/test_ui_smoke.py
git commit -m "fix(ui): anchor static directory to absolute project path and add UI smoke tests"
```

---

### Task 6: Automated CI Workflow & Community Scaffolding

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `CONTRIBUTING.md`
- Create: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Create: `.github/ISSUE_TEMPLATE/feature_request.yml`
- Create: `Docs/MAINTENANCE.md`

**Interfaces:**
- Consumes: Test suite and repository maintenance procedures.
- Produces: GitHub Actions CI workflow, community guidelines, and race-weekend update runbook.

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run Contract Tests
        run: |
          python -m unittest engine.tools.tests.test_contracts
      - name: Run UI Smoke Tests
        run: |
          python -m unittest engine.tools.tests.test_ui_smoke
```

- [ ] **Step 2: Create `CONTRIBUTING.md`**

Include setup guide, branch naming conventions (`feat/`, `fix/`), testing expectations, and code of conduct reference.

- [ ] **Step 3: Create GitHub issue templates**

Add `.github/ISSUE_TEMPLATE/bug_report.yml` and `.github/ISSUE_TEMPLATE/feature_request.yml`.

- [ ] **Step 4: Create `Docs/MAINTENANCE.md`**

Document the step-by-step procedure for:
1. Fetching new race telemetry (`VERIFY_CACHES.bat` or `python -m engine.core.warm_cache`).
2. Post-race validation and self-improvement bias updates.
3. Updating driver & team rosters for mid-season driver swaps.
4. Git branch & PR workflow for pushing regular updates safely.

- [ ] **Step 5: Commit CI and community files**

```bash
git add .github/ CONTRIBUTING.md Docs/MAINTENANCE.md
git commit -m "ci: add GitHub Actions workflow, community templates, and maintenance runbook"
```

---

### Task 7: Showcase `README.md` Overhaul

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: Full repository capabilities, screenshots, and legal disclaimers.
- Produces: Professional open-source showcase README.

- [ ] **Step 1: Write comprehensive `README.md`**

Include:
- Badges: Python 3.10+, GNU GPLv3, CI Passing, FastAPI.
- Showcase overview and key features (54-circuit features, ML ensemble, Monte Carlo, PuLP optimizer).
- Embedded screenshots (`Screenshots/` images).
- Mermaid architecture diagram.
- 3-step Quick Start guide (Windows `.bat` + Linux/macOS `.sh` + manual commands).
- Project directory layout.
- Legal disclaimer regarding Formula 1 and F1 Fantasy.
- License section citing GNU General Public License v3.0.

- [ ] **Step 2: Commit showcase README**

```bash
git add README.md
git commit -m "docs: overhaul README with architecture, visual showcase, GPLv3 badge, and disclaimers"
```

---

### Task 8: End-to-End Verification & Release Readiness Gate

**Files:**
- None (verification only)

**Interfaces:**
- Consumes: Entire repository state.
- Produces: Verification evidence and final push command.

- [ ] **Step 1: Run complete test suite**

Run: `python -m unittest discover -s engine/tools/tests -p "test_*.py"`  
Expected: All tests pass.

- [ ] **Step 2: Verify zero sensitive files in working tree or git history**

Run:
```bash
git status
git log --all --full-history -- .env
```
Expected: Clean working tree, 0 results for `.env`.

- [ ] **Step 3: Check remote configuration**

Run: `git remote -v`  
Expected: Clean HTTPS URL without tokens.

- [ ] **Step 4: Prepare final push and publish instructions for user**
