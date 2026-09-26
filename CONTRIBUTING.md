# Contributing to F1 Prediction Engine

Thank you for your interest in contributing to the **F1 Prediction Engine**! This project provides machine learning models, statistical simulations, and lineup optimization for Formula 1 Fantasy.

---

## Code of Conduct & Licensing

### GNU GPLv3 Compliance
This repository is open-sourced under the **GNU General Public License v3.0 (GPLv3)**. By contributing to this repository:
1. You agree that all contributed code and documentation will be licensed under the GNU GPLv3.
2. You must preserve existing copyright and license headers.
3. You must not include proprietary, confidential, or unlicensed third-party code.
4. When incorporating open-source libraries or data snippets, verify their licenses are compatible with GPLv3.

---

## Development Environment Setup

### Prerequisites
- **Python 3.10+** (Python 3.11 recommended)
- **Git**
- Optional: Google Chrome or Chromium (required if running live Fantasy price scraping via Playwright)

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Retr0-908/F1-Prediction-engine.git
   cd F1-Prediction-engine
   ```

2. **Create and activate a virtual environment:**
   - **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt):**
     ```cmd
     python -m venv venv
     venv\Scripts\activate.bat
     ```

3. **Install dependencies:**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **(Optional) Set up Playwright for live price scraping:**
   ```bash
   python -m playwright install chromium
   ```

5. **Configuration (.env):**
   Copy `.env.example` (or create a `.env` file) in the project root if you need custom API configurations or cookies:
   ```ini
   F1_FANTASY_COOKIE=""
   ```

---

## Project Structure & Architecture Conventions

The codebase follows a modular architecture under `engine/`:

- `engine/core/`: Configuration (`config.py`), filesystem paths (`paths.py`), external API clients (`data_fetcher.py`, `weather.py`, `scraper.py`).
- `engine/models/`: Machine learning ensemble (`predictor.py`), Glicko-2 ratings (`glicko.py`), tire degradation model (`tire_model.py`), Bayesian expected value (`bayesian.py`), Monte Carlo risk simulations (`monte_carlo.py`).
- `engine/strategy/`: Integer linear programming lineup optimizer (`fantasy_optimizer.py`), chip advisor (`chip_advisor.py`), dynamic price tracker (`price_tracker.py`), and self-improvement bias correction (`self_improvement.py`).
- `engine/analysis/`: Historical season backtester (`backtester.py`), post-race accuracy verification (`post_race_check.py`), report generation (`race_reporter.py`).
- `engine/serving/`: FastAPI application server (`server.py`) and pipeline orchestrator (`pipeline.py`).
- `engine/cli/`: Terminal interface application (`app.py`), invoked via root `main.py`.
- `engine/tools/tests/`: Unit, contract, and smoke tests.
- `ui/`: Vanilla JavaScript Single-Page Application (SPA) frontend.
- `track_features/`: Canonical JSON database of track characteristics for all championship circuits.

### Key Code Conventions
1. **Filesystem Paths**: Never hardcode relative or absolute string paths in modules. Always import path constants from `engine.core.paths` (e.g., `CACHE_DIR`, `OUTPUT_DIR`, `LOGS_DIR`, `TRACK_FEATURES_DIR`).
2. **Data Leakage Prevention**: When training or evaluating models, strictly separate historical context from future outcomes. `CURRENT_SEASON` results must never leak into model training sets for active-season predictions.
3. **Dynamic Rosters**: Driver and constructor rosters are dynamically resolved at runtime from official championship standings (`engine.core.data_fetcher.get_season_roster`). Do not hardcode static lineup changes in model logic; use `engine.core.config.DRIVER_TEAMS_2026` strictly as an offline or pre-round-1 seed.
4. **Code Quality**: Follow PEP 8 guidelines. Write clear docstrings for public classes and functions. Keep functions focused and unit-testable.

---

## Testing Guidelines

Before submitting changes, ensure all tests pass locally.

### Running Contract Tests
Validates schema compliance, path constants, track feature structures, and model input/output contracts:
```bash
python -m unittest engine.tools.tests.test_contracts
```

### Running UI Smoke Tests
Validates that the FastAPI server routes and frontend asset serving initialize properly without crashes:
```bash
python -m unittest engine.tools.tests.test_ui_smoke
```

### Running the Full Test Suite
```bash
python -m unittest discover -s engine/tools/tests -p "test_*.py"
```

---

## Pull Request Guidelines

1. **Branch Naming**:
   - `feature/your-feature-name`
   - `fix/bug-fix-name`
   - `docs/documentation-update`
   - `refactor/cleanup-name`

2. **Commit Messages**:
   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat: add new telemetry feature to ML ensemble`
   - `fix: correct tire degradation calculation for wet compound`
   - `docs: update race weekend runbook in MAINTENANCE.md`
   - `test: add unit test for budget constraint in optimizer`

3. **PR Submission Checklist**:
   - [ ] Branch is up to date with `main`.
   - [ ] All unit, contract, and UI smoke tests pass (`python -m unittest engine.tools.tests.test_contracts`).
   - [ ] New functionality includes relevant unit tests in `engine/tools/tests/`.
   - [ ] Documentation and comments are updated where applicable.
   - [ ] Code is formatted and free of unnecessary debugging statements or print calls.
