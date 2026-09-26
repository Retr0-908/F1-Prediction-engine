# Public Release & Repository Hardening Design

**Date**: 2026-09-26  
**Status**: Approved for Implementation  
**Target Repository**: `Retr0-908/F1-Prediction-engine`  
**License**: GNU General Public License v3.0 (GPL-3.0) — Strong Copyleft (protects against code theft, enforces attribution)  

---

## 1. Executive Summary & Goals

The objective of this design is to prepare the **F1-Prediction-engine** repository for public open-source release on GitHub and showcase platforms (such as Reddit `r/F1Fantasy`, `r/formula1`, and LinkedIn). 

The release strategy ensures:
1. **Absolute Security & Privacy**: Complete eradication of sensitive credentials (API keys, personal JWT session tokens, and embedded GitHub Personal Access Tokens) from Git history, commit trees, and configuration files.
2. **Frictionless Installation**: Tiered dependencies separating core runtime from development and optional heavy modules, eliminating install failures on Windows, macOS, and Linux.
3. **Showcase-Grade Presentation**: A polished `README.md` featuring architecture diagrams, embedded dashboard screenshots, technical feature highlights, and essential legal disclaimers.
4. **Automated Continuous Integration**: GitHub Actions CI workflow running the test suite to ensure green passing badges.
5. **Long-Term Maintainability**: A clear, repeatable workflow enabling the author to easily update data, retrain models, and push improvements throughout future race weekends without risking credential leaks or regression bugs.

---

## 2. Security Sanitization & Git History Scrubbing

### 2.1 Credential Discovery & Vulnerabilities Identified
- **Historical Git Commit Leak**: In commit `a5e6143534583fdc0c858e674e9f5ccf1a2ab6c0`, a raw `.env` file was committed containing:
  - An `OPENWEATHERMAP_API_KEY`.
  - An active `F1_FANTASY_COOKIE` consisting of user JWTs, account IDs, and personal name metadata.
  - While subsequently deleted in commit `0994ac0fadfcbb1694f3faec3ea1d81b7440332d`, git objects retain this data permanently unless purged.
- **Embedded Remote Token**: The `.git/config` remote URL contains an embedded GitHub Personal Access Token (`https://Retr0-908:ghp_...`).

### 2.2 Sanitization Procedure
1. **Safety Backup**: Create a local safety backup branch before touching git history:
   ```bash
   git branch backup-pre-release
   ```
2. **Commit Current Working Tree**: Commit the 9 pending modified files (`engine/__init__.py`, `engine/analysis/analyse_results.py`, `engine/analysis/post_race_check.py`, `engine/core/warm_cache.py`, `engine/models/elo_ratings.py`, `engine/models/predictor.py`, `engine/serving/pipeline.py`, `engine/serving/server.py`, `main.py`).
3. **History Purge via `git-filter-repo`**:
   - Execute `git-filter-repo` to permanently erase `.env` from all historical commits:
     ```bash
     python -m pip install git-filter-repo
     git filter-repo --path .env --invert-paths --force
     ```
   - Verify that `git log --all --full-history -- .env` produces zero output.
4. **Sanitize Remote Origin**:
   - Re-point remote to clean HTTPS:
     ```bash
     git remote set-url origin https://github.com/Retr0-908/F1-Prediction-engine.git
     ```
5. **Environment Template**:
   - Retain local `.env` on disk (strictly ignored by `.gitignore`).
   - Create `.env.example`:
     ```env
     # Optional: Live weather data enrichment via Open-Meteo or OpenWeatherMap
     OPENWEATHERMAP_API_KEY=your_openweathermap_api_key_here

     # Optional: Live F1 Fantasy cookie for automated lineup imports & real-time pricing
     F1_FANTASY_COOKIE=
     ```

---

## 3. Packaging & Dependency Architecture

### 3.1 Pruning & Dependency Organization
- **Remove Unused Libraries**: `pymc` and `arviz` are declared in `requirements.txt` but are unused across the entire codebase. Removing them prevents compilation errors on user machines.
- **Isolate Experimental/Optional Engines**: Julia acceleration (`juliacall`) is experimental and currently disabled (`USE_JULIA_ENGINE = False`). It will be documented as an optional enhancement rather than a mandatory requirement.
- **Tiered Structure**:
  - `requirements.txt`: Core application dependencies for running the prediction engine, backtests, CLI, and FastAPI web app:
    - `fastf1>=3.3.0`
    - `requests>=2.31.0`
    - `beautifulsoup4>=4.12.0`
    - `pandas>=2.0.0`
    - `numpy>=1.26.0,<2.0.0`
    - `scikit-learn>=1.4.0`
    - `scipy>=1.13.0`
    - `xgboost>=2.0.0`
    - `lightgbm>=4.0.0`
    - `pulp>=2.7.0`
    - `fastapi>=0.110.0`
    - `uvicorn>=0.29.0`
    - `sse-starlette>=2.0.0`
    - `python-dotenv>=1.0.0`
    - `rich>=13.7.0`
    - `tabulate>=0.9.0`
    - `joblib>=1.3.0`
    - `lxml>=5.0.0`
    - `questionary>=2.0.0`
    - `tensorflow>=2.15.0`
  - `requirements-dev.txt`: Development, scraping, and testing tools:
    - `-r requirements.txt`
    - `pytest>=8.0.0`
    - `playwright>=1.40.0`
    - `ruff>=0.3.0`

### 3.2 Cross-Platform Launchers
- **Windows**: Retain `.bat` scripts (`F1 Fantasy.bat`, `BACKTEST.bat`, `VERIFY_CACHES.bat`).
- **macOS / Linux**: Provide equivalent bash scripts (`run_app.sh`, `backtest.sh`) with executable permissions.

---

## 4. Documentation & Showcase Presentation

### 4.1 GNU General Public License v3.0 (`LICENSE`)
Include the full GNU GPLv3 license text. This provides strong copyleft protection:
- Ensures full attribution and credit to Retr0-908.
- Explicitly prohibits proprietary or closed-source reuse/theft of components.
- Mandates that any derivative works or integrations must also be published as open-source under GPLv3.
- Protects against patent abuse and provides zero personal warranty/liability.

### 4.2 Comprehensive `README.md`
- **Hero & Status Badges**:
  - Python 3.10+ | License: MIT | CI: Passing | Framework: FastAPI & Vanilla JS
- **Visual Preview Section**:
  - Embedded screenshots from `Screenshots/` showing:
    - Web UI Live Dashboard
    - Monte Carlo Finishing Probability Distribution
    - PuLP Optimal Fantasy Lineup Recommendation
- **Core Technology & Innovation Breakdown**:
  - *54-Feature Circuit Enrichment*: Circuit altitude, downforce index, cornering speed, full-throttle %, tire degradation index, braking energy.
  - *Ensemble Machine Learning*: Blended XGBRanker, LGBMRanker, Random Forest, and Ridge Meta-estimator.
  - *Dynamic Performance Modeling*: Dual Glicko-2 ratings (driver + constructor) and LSTM form momentum.
  - *Stochastic Risk Modeling & Optimization*: 5,000 Monte Carlo race simulations driving a PuLP integer programming optimizer with 3-transfer budget optimization and chip advisory.
- **Architecture Diagram**:
  ```mermaid
  flowchart TD
      subgraph Ingestion ["Data Ingestion"]
          F1["FastF1 Telemetry"] --> PP["Data Preprocessor"]
          JP["Jolpica / Ergast API"] --> PP
          WX["Open-Meteo Weather"] --> PP
          TF["54 Track Features JSON"] --> PP
      end

      subgraph Modeling ["Predictive Engine"]
          PP --> G2["Dual Glicko-2 Ratings"]
          PP --> LSTM["LSTM Form Momentum"]
          PP --> ENS["ML Ensemble (XGB + LGBM + RF)"]
          G2 --> META["Ridge Meta-Model"]
          LSTM --> META
          ENS --> META
      end

      subgraph Optimization ["Simulation & Strategy"]
          META --> MC["Monte Carlo (5,000 Sims)"]
          MC --> DIST["Position & Points Distributions"]
          DIST --> PULP["PuLP Lineup & Chip Optimizer"]
      end

      subgraph Delivery ["User Experience"]
          PULP --> API["FastAPI SSE Server"]
          API --> UI["Web App Dashboard"]
          API --> CLI["Interactive CLI App"]
      end
  ```
- **3-Step Setup Instructions**:
  ```bash
  # 1. Clone repository
  git clone https://github.com/Retr0-908/F1-Prediction-engine.git
  cd F1-Prediction-engine

  # 2. Install dependencies
  pip install -r requirements.txt

  # 3. Launch dashboard
  python server.py
  ```
- **Mandatory Legal Disclaimer**:
  > *"This repository is an independent, open-source fan project and is not affiliated, endorsed, or associated with Formula One Group, Formula 1, FIA, or the official F1 Fantasy game. All trademarks belong to their respective owners."*

---

## 5. Automated CI & Quality Control

### 5.1 GitHub Actions Workflow (`.github/workflows/ci.yml`)
- Trigger on `push` to `main` and `pull_request`.
- Environment: Ubuntu Latest, Python 3.11.
- Steps:
  1. Checkout code.
  2. Install core requirements and dev packages.
  3. Run validation test suite: `python -m unittest engine.tools.tests.test_contracts`.
  4. Run track schema validation: `python engine/tools/validate_track_features.py` (or sanity checks).

### 5.2 Community Scaffolding
- `CONTRIBUTING.md`: Contributing guide for reporting bugs, adding track data, and testing models.
- `.github/ISSUE_TEMPLATE/bug_report.yml`: Structured bug report template.
- `.github/ISSUE_TEMPLATE/feature_request.yml`: Feature proposal template.

---

## 6. Long-Term Maintenance & Race-Weekend Runbook

To accommodate the user's requirement of ongoing updates throughout the season:
- **Maintenance Guide (`Docs/MAINTENANCE.md`)**:
  - Documenting how to fetch new telemetry after race weekends (`python -m engine.core.warm_cache` or `VERIFY_CACHES.bat`).
  - Documenting how to run post-race validation to evaluate accuracy and trigger the self-improving bias correction.
  - Documenting the branch and PR workflow so future updates never accidentally expose sensitive data or break existing contracts.

---

## 7. Execution Order

1. **Working Tree Commit**: Commit pending 9 modified files.
2. **Git Backup & History Sanitization**: Create backup branch, scrub `.env` with `git-filter-repo`, sanitize remote URL, verify history.
3. **Environment & License Setup**: Create `.env.example`, create `LICENSE` (MIT), audit `.gitignore`.
4. **Dependency Refactoring**: Update `requirements.txt`, create `requirements-dev.txt`, add POSIX launcher scripts.
5. **CI & Community Templates**: Create `.github/workflows/ci.yml`, `CONTRIBUTING.md`, issue templates, and `Docs/MAINTENANCE.md`.
6. **Showcase README**: Replace `README.md` with complete showcase documentation.
7. **UI Audit & Smoke Test Gate**:
   - Verify `server.py` static file mount uses `str(UI_DIR)`.
   - Spin up local server on test port and verify all core API endpoints (`/api/status`, `/api/race/next`, `/api/prices`, `/api/model/health`, `/api/system/health`).
   - Validate UI asset loading (CSS, JS, fonts, driver/team images) without 404s.
8. **Test Suite Verification Gate**: Run test suite locally (`Ran 41 tests -> OK`), check git status and git diff.
9. **Final User Review & Release Command**: Provide clean instructions for pushing to GitHub and flipping the repo toggle to Public.
