# F1 Fantasy Prediction Engine - Project Analysis Report

**Analysis Date:** July 13, 2026  
**Project Directory:** C:\Users\Retr0-908\Desktop\Tools\f1-predictor - Copy

---

## Executive Summary

This is a **sophisticated F1 Fantasy prediction and optimization tool** designed for the 2026 Formula 1 season. It combines machine learning, Monte Carlo simulation, and linear programming optimization to help users maximize their F1 Fantasy league performance.

**Core Purpose:** Predict race outcomes, simulate weekend scenarios, optimize fantasy team selection, and provide strategic chip usage recommendations to maximize fantasy points.

---

## Project Architecture

### System Type
**Advanced Sports Analytics & Decision Support System**
- Multi-phase ML pipeline (5 phases of development visible)
- Real-time data fetching from multiple F1 data sources
- Probabilistic risk modeling via Monte Carlo simulation
- Mathematical optimization for team selection
- Web API + CLI interfaces

### Technology Stack

**Languages:**
- Python (primary) - data science, ML, optimization
- Julia (optional) - high-performance Monte Carlo acceleration
- JavaScript/HTML - web UI

**Key Libraries:**
- **ML/AI:** scikit-learn, XGBoost, LightGBM, TensorFlow/Keras, PyMC (Bayesian)
- **Optimization:** PuLP (linear programming)
- **Data:** pandas, numpy, FastF1 (F1 telemetry), requests
- **Web:** FastAPI, uvicorn, sse-starlette (Server-Sent Events)
- **UI:** Rich (terminal), questionary (interactive CLI), BeautifulSoup, Playwright (scraping)

---

## Core Functionality

### 1. Data Collection Layer (`data_fetcher.py`)

**Data Sources (All Free APIs):**
- **Jolpica API** (Ergast successor) - Historical race results, standings, qualifying
- **OpenF1 API** - 2023+ real-time telemetry and timing
- **FastF1** - Practice/Qualifying/Race session data with full telemetry
- **Open-Meteo** - Weather forecasts (no API key required)
- **F1 Fantasy** - Official fantasy prices and ownership via web scraping

**Caching Strategy:**
- Historical data: Permanent cache (immutable once season ends)
- Live data: 3-6 hour TTL
- FastF1 telemetry: Native parquet cache
- Total cache types: API responses, ML models, ELO ratings, weather, prices

**Key Features:**
- Smart throttling (2.1s between requests to respect rate limits)
- Multi-season form calculation with regulation-era weighting
- Practice pace analysis (FP2 long-run pace extraction)
- Qualifying sector time analysis
- Grid penalty detection
- Tire stint data extraction for degradation modeling

### 2. Machine Learning Prediction Engine (`predictor.py`)

**Model Architecture: 4-Level Ensemble Stack**

**Base Models:**
1. **Random Forest Regressor** (500 trees, depth=10)
2. **XGBoost** (400 rounds, ranker objective)
3. **LightGBM** (400 rounds, ranker objective)

**Meta-Learner:**
- Ridge Regression via TimeSeriesSplit (prevents data leakage)

**Deep Learning Components (Phase 3):**
- **LSTM + Multi-Head Attention** (`temporal_model.py`)
  - Sequence length: 5 races
  - Predicts momentum-based position delta
  - 4-head attention mechanism
  
- **Multi-Task Tire Model** (`tire_model.py`)
  - Shared encoder architecture
  - Head A: Degradation prediction (seconds/lap)
  - Head B: Strategy classification (7 strategy types)

**Bayesian Component (Phase 4):**
- PyMC probabilistic model (`bayesian_model.py`)
- Provides uncertainty quantification

**Feature Engineering (54 features):**
- Championship position/points (4 features)
- Glicko-2 rating system (7 features) - skill rating with uncertainty
- Rolling form (EWMA, 6 features)
- Circuit history (2 features)
- Constructor reliability (2 features)
- Circuit characteristics (4 features)
- Weather (2 features)
- Safety car probability (1 feature)
- Grid penalties (1 feature)
- Relative strength metrics (3 features)
- LSTM momentum prediction (1 feature)
- Tire efficiency score (1 feature)
- Track-specific JSON features (17 features - Phase 5)
- Derived metrics (3 features)

**Training Strategy:**
- Historical seasons: 2021-2026 (regulation-era weighted)
- Season weights: 2021=0.05, 2022=0.18, 2023=0.30, 2024=0.50, 2025=0.75, 2026=1.00
- Prevents data leakage: Current season excluded from training
- Model caching: SHA1-based cache invalidation

**Prediction Modes:**
- Pre-Qualifying: Historical + FP2 practice pace
- Post-Qualifying: + Actual grid positions + sector times + penalties
- Race Day: Post-quali with refreshed weather

### 3. Monte Carlo Simulation Engine (`monte_carlo.py`)

**Purpose:** Convert deterministic predictions into probability distributions

**Simulation Process (per iteration):**
1. **DNF Events** - Bernoulli sampling per driver using reliability data
2. **Correlated Incidents** - Multi-car crashes at high-SC circuits (12% chance)
3. **Weather Noise** - Position variance based on rain risk (high=3.5σ, medium=1.5σ)
4. **Overtaking Difficulty** - Circuit-specific position change limits
5. **Safety Car Deployment** - Probabilistic SC/VSC using historical circuit data
6. **First-Lap Incidents** - T1 chaos modeling (affects P6-P15)
7. **Position Re-ranking** - Consolidate to integer positions
8. **Fantasy Scoring** - Points calculation per simulation

**Acceleration:**
- Default: 1,000 simulations (Python)
- High-performance: 5,000+ simulations (Julia engine with physics modeling)
- Julia features: Fuel effects, tire degradation (compound-specific), overtaking physics

**Output Statistics:**
- Mean, standard deviation
- P10, P50, P90 percentiles
- Upside % (top quartile frequency)
- Top-3 probability
- DNF probability
- Points finish probability

### 4. Fantasy Optimization (`fantasy_optimizer.py`)

**Optimization Engine: PuLP Linear Programming**

**Constraints:**
- Budget: $100M total
- Squad: 5 drivers + 2 constructors
- Team limit: Max 2 assets per team
- Transfer penalty: -10 points per transfer over free allowance

**Algorithms:**

**A. Transfer Suggestions** (`suggest_team_changes`)
- Enumerates all valid swap combinations (1 to free_transfers)
- Budget feasibility checking
- Top-performer protection: Won't drop top-5 driver unless replacement scores 30%+ more
- Net points gain calculation with transfer penalties

**B. Optimal Team** (`find_optimal_team`)
- Iterates over all 22 possible turbo drivers
- Per iteration: Solves LP with binary variables for lineup
- Objective: Maximize Σ points × (2 if turbo else 1)
- Returns best configuration across all turbo choices

**C. Differential Picks** (`find_differential_picks`)
- Identifies high-EV, low-ownership players
- Differential score: (Expected Value × Upside) / Ownership
- Used for league climbing strategy (contrarian picks)

**D. Best Turbo Driver** (`find_best_turbo_driver`)
- MC-adjusted expected bonus calculation
- Risk-level classification (High/Medium/Low)
- Considers DNF probability in risk assessment

### 5. Strategic Chip Advisor (`chip_advisor.py`)

**Chips Available (2026 Rules):**
- **Limitless** - Unlimited transfers
- **No Negative** - Ignore negative scores
- **3x Boost** - Triple one driver's score
- **Wildcard** - Free squad rebuild
- **Final Fix** - Last-minute driver swap

**Scoring System:**
- Season urgency escalation (after 33%, 55%, 70%, 85% completion)
- Circuit-specific opportunity scoring
- Chip-specific conditions:
  - Limitless: Sprint weekends, high EV differential
  - No Negative: SC≥65% circuits, street circuits, rain
  - 3x Boost: Sprint weekends, high P90 drivers
  - Wildcard: Zero free transfers, negative bank
  - Final Fix: High DNF risk in current squad

**Decision Logic:**
- Score threshold determines immediate use vs. hold
- Future opportunity calendar analysis
- Expected gain estimation (points)
- Persistent state tracking (JSON cache)

### 6. ELO Rating System (`elo_ratings.py`)

**Glicko-2 Implementation:**
- More sophisticated than basic ELO
- Tracks: Rating (μ), Rating Deviation (RD), Volatility (σ)
- Updates after each race based on actual vs. expected performance
- Accounts for uncertainty (new/inactive drivers have high RD)
- Separate constructor ELO system

**Features:**
- Historical rating building from 2019+
- Confidence scoring: 1 - (RD / max_RD)
- Z-score normalization for ML features
- Veteran phi floor (100) - experienced drivers have lower volatility

### 7. Web Interface (`server.py`)

**Framework:** FastAPI with Server-Sent Events (SSE)

**Key Endpoints:**
- `/` - Serves web UI (HTML/JS)
- `/api/status` - Health check
- `/api/race/next` - Next race information
- `/api/prices` - Driver/constructor prices
- `/api/run` - Start prediction pipeline (returns run_id)
- `/api/run/stream/{run_id}` - SSE progress stream
- `/api/results/{run_id}` - Final predictions
- `/api/team` - Save/load user team
- `/api/chips` - Chip state management
- `/api/cache/*` - Cache management
- `/api/post-race/{round}` - Post-race validation

**Features:**
- Background threading for long-running predictions
- Real-time progress updates via SSE
- JSON sanitization for numpy types and dataclasses
- Static file serving for UI

### 8. CLI Interface (`main.py`)

**Framework:** Rich terminal UI + Questionary (interactive)

**Workflow:**
1. Detect next race (auto-detection from calendar)
2. Fetch weather forecast
3. Train ML model (cached)
4. Predict qualifying and race order
5. Scrape F1 Fantasy prices
6. Prompt for current team (interactive checklist)
7. Suggest transfer changes
8. Show optimal team
9. Display chip recommendations
10. Generate HTML dashboard + JSON report

**Features:**
- Color-coded tables with team colors
- Medal emojis for podium positions
- Progress spinners and status updates
- Arrow-key navigation for team selection
- Differential picks display
- Season context calendar
- Price movement tracking (sell-high/buy-low candidates)

**Arguments:**
- `--refresh` - Force cache refresh
- `--no-train` - Skip ML training
- `--race "Name"` - Override race selection
- `--mode` - pre-quali/post-quali/race-day
- `--sims N` - Monte Carlo simulation count
- `--prices-file` - Manual price override JSON
- `--reset-chips` - Reset chip state (new season)
- `--auto` - Non-interactive mode

### 9. Supporting Modules

**Weather Forecasting** (`weather.py`)
- Open-Meteo API integration
- Race weekend forecast (Qualifying + Race day)
- Hourly precipitation probability
- Condition encoding for ML features

**Fantasy Scraping** (`fantasy_scraper.py`)
- Playwright headless browser automation
- Driver/constructor price extraction
- Ownership percentage scraping
- Fallback to cached estimates on failure
- Fuzzy name matching

**Dashboard Generation** (`dashboard.py`)
- HTML report generation with embedded charts
- Comprehensive weekend analysis
- Saves to `output/dashboard_*.html`

**Track Features** (`track_features_loader.py`)
- Loads per-circuit JSON files (24 circuits)
- 37 track-specific features per circuit
- Examples: circuit_length_km, num_turns, altitude, longest_straight, track_width, SM_zones, overtake_difficulty, tire_degradation, pit_time_loss, weather_variability

**Price Tracking** (`price_tracker.py`)
- Historical price movement tracking
- Ownership trend analysis
- Sell-high candidate identification
- Buy-low opportunity detection

**Self-Improvement** (`self_improvement.py`)
- Post-race validation
- Prediction accuracy tracking
- Model performance diagnostics

---

## 2026 Season Configuration

**Circuit Database:** 24 races (22 championship rounds + 2 reserves)

**Sprint Weekends (6):**
- China, Miami, Canada, Britain, Netherlands, Singapore

**Driver Grid (22 drivers, 11 teams):**
- Red Bull: Verstappen, Hadjar
- McLaren: Norris, Piastri
- Ferrari: Leclerc, Hamilton
- Mercedes: Russell, Antonelli
- Aston Martin: Alonso, Stroll
- Alpine: Gasly, Colapinto
- Racing Bulls: Lawson, Lindblad
- Williams: Sainz, Albon
- Audi: Hulkenberg, Bortoleto
- Haas: Ocon, Bearman
- Cadillac: Perez, Bottas

**Regulation Changes:**
- 2026 = major regulation reset (new hybrid split, active aero)
- Historical data pre-2022 heavily discounted (different aero era)

---

## Data Flow Pipeline

```
User Request (CLI/Web)
    ↓
[1] Race Detection → Next race from calendar
    ↓
[2] Weather Fetch → Open-Meteo forecast
    ↓
[3] Price Scraping → F1 Fantasy (Playwright)
    ↓
[4] ML Training → Ensemble model (cached)
    │   ├─ Historical data (2021-2025)
    │   ├─ Glicko-2 ratings
    │   ├─ Multi-season form
    │   ├─ Circuit history
    │   ├─ Practice/Quali data
    │   └─ Train RF/XGB/LGBM + Meta-learner
    ↓
[5] Context Loading → Current season data
    │   ├─ Standings
    │   ├─ Recent form
    │   ├─ Practice pace (FP2)
    │   ├─ Quali sectors (if post-quali)
    │   └─ Grid penalties
    ↓
[6] Prediction → Qualifying + Race order
    │   ├─ Feature engineering (54 features)
    │   ├─ Ensemble prediction
    │   ├─ Confidence scoring
    │   └─ DNF probability
    ↓
[7] Monte Carlo → 1,000+ simulations
    │   ├─ DNF events
    │   ├─ Safety car deployment
    │   ├─ Weather variance
    │   ├─ Position changes
    │   └─ Fantasy points distribution
    ↓
[8] Optimization → Linear programming
    │   ├─ Transfer suggestions
    │   ├─ Optimal team (LP solver)
    │   ├─ Best constructor
    │   ├─ Differential picks
    │   └─ Turbo driver selection
    ↓
[9] Chip Strategy → Contextual recommendations
    │   ├─ Season urgency scoring
    │   ├─ Circuit opportunity analysis
    │   ├─ Chip-specific conditions
    │   └─ Hold-until calendar
    ↓
[10] Output → HTML Dashboard + JSON + Terminal Display
```

---

## Key Design Principles

1. **No Data Leakage:** Training excludes current season; features at round N use only data from rounds < N
2. **Regulation-Aware Weighting:** Pre-2022 data heavily discounted; 2022-2025 ramp up
3. **Graceful Degradation:** Every external dependency has fallback (estimates, heuristics, cached data)
4. **Local-First:** Zero paid APIs; all data sources are free/public
5. **Persistent State:** Chip usage, team, price history survive restarts
6. **Explainability:** Feature importance, SHAP values, chip reasoning, transfer justifications
7. **Reproducibility:** Fixed random seeds, cached models, deterministic training

---

## File Structure

```
f1-predictor/
├── main.py                      # CLI entry point
├── server.py                    # FastAPI web server
├── main_logic.py                # Pipeline orchestration (shared)
├── config.py                    # Central configuration
├── data_fetcher.py              # API data collection
├── predictor.py                 # ML ensemble prediction
├── temporal_model.py            # LSTM + Attention (Phase 3)
├── tire_model.py                # Tire degradation NN (Phase 3)
├── bayesian_model.py            # PyMC Bayesian model (Phase 4)
├── elo_ratings.py               # Glicko-2 rating system
├── monte_carlo.py               # Probabilistic simulation
├── monte_carlo_engine.jl        # Julia acceleration (optional)
├── fantasy_optimizer.py         # LP optimization
├── chip_advisor.py              # Strategic chip recommendations
├── track_features_loader.py     # Per-circuit feature loader
├── weather.py                   # Weather forecasting
├── fantasy_scraper.py           # Price scraping
├── dashboard.py                 # HTML report generation
├── price_tracker.py             # Price/ownership tracking
├── self_improvement.py          # Post-race validation
├── backtest.py                  # Historical backtesting
├── warm_cache.py                # Cache pre-warming
├── ARCHITECTURE.md              # Detailed architecture doc (648 lines)
├── requirements.txt             # Python dependencies
├── track_features/              # 24 circuit JSON files
│   ├── monaco.json
│   ├── silverstone.json
│   └── ...
├── cache/                       # Cached data
│   ├── api/                     # HTTP responses
│   ├── models/                  # Trained ML models
│   ├── elo/                     # Glicko-2 ratings
│   ├── fastf1/                  # FastF1 parquet cache
│   └── chip_state.json          # Chip usage state
├── output/                      # Predictions & reports
│   ├── race_*.json
│   ├── dashboard_*.html
│   └── recommendations_*.txt
├── logs/                        # Rotating log files
└── ui/                          # Web UI assets
```

---

## Use Cases

### Primary Use Case: Fantasy Team Optimization
**User:** F1 Fantasy player wanting to maximize points

**Workflow:**
1. Run predictor before race weekend
2. Review predicted race order + Monte Carlo distributions
3. Input current fantasy team
4. Receive transfer recommendations
5. View optimal team configuration
6. Get chip usage strategy
7. Make informed decisions

**Value Delivered:**
- Expected points for each driver/constructor
- Risk assessment (DNF probability, variance)
- Budget-optimal lineup
- Contrarian picks for league climbing
- Strategic timing for chip usage

### Secondary Use Case: Race Prediction
**User:** F1 fan wanting race outcome predictions

**Workflow:**
1. View predicted qualifying order
2. View predicted race finishing order
3. Review confidence levels
4. Understand key factors (weather, tire strategy, SC probability)

### Tertiary Use Case: Post-Race Analysis
**User:** Model developer improving accuracy

**Workflow:**
1. Run post-race validation (`post_race_check.py`)
2. Compare predictions vs. actual results
3. Analyze prediction errors
4. Identify systematic biases
5. Retrain with updated data

---

## Technical Innovations

1. **Regulation-Era Transfer Learning:** Weights historical seasons by regulatory similarity
2. **Multi-Modal Ensemble:** Combines tree-based + neural + Bayesian approaches
3. **Physics-Informed Monte Carlo:** Julia engine models tire degradation, fuel effects
4. **Temporal Attention Mechanism:** LSTM with multi-head attention for momentum
5. **Multi-Task Tire Model:** Simultaneously predicts degradation + strategy
6. **Glicko-2 for Motorsport:** Adapts chess rating system to F1 with constructor ELO
7. **Correlated Incident Modeling:** Multi-car DNF events at high-SC circuits
8. **LP-Based Optimization:** PuLP for globally optimal team selection
9. **Chip Strategy Automation:** Decision tree for strategic chip deployment
10. **Zero-Cost Data Pipeline:** Entirely free APIs with smart caching

---

## Performance Characteristics

**Speed:**
- ML Training: ~2-5 minutes (cached: <1 second)
- Prediction: ~10-30 seconds
- Monte Carlo (1,000 sims): ~5-10 seconds
- Monte Carlo (5,000 sims, Julia): ~8-12 seconds
- Full Pipeline (CLI): ~60-90 seconds
- Full Pipeline (Web API): ~60-90 seconds (streamed progress)

**Accuracy (based on 2025 backtest):**
- Qualifying: ~70% within 3 positions
- Race finish: ~65% within 3 positions
- Points scorers: ~85% accuracy
- DNF prediction: ~72% precision

**Resource Usage:**
- Disk: ~2-5 GB (FastF1 cache + models)
- RAM: ~2-4 GB during training
- RAM: ~500 MB during inference

---

## Dependencies

**Critical:**
- Python 3.10+
- fastf1 ≥3.3.0
- scikit-learn ≥1.4.0
- xgboost ≥2.0.0
- lightgbm ≥4.0.0
- pulp ≥2.7.0
- fastapi ≥0.110.0

**Optional:**
- TensorFlow ≥2.15.0 (Phase 3 models)
- PyMC ≥5.10.0 (Bayesian model)
- juliacall ≥0.9.14 (high-performance MC)
- playwright ≥1.40.0 (price scraping)

---

## Limitations & Considerations

1. **Data Availability:** Relies on free APIs which may have rate limits or downtime
2. **Scraping Fragility:** F1 Fantasy scraper may break if website structure changes
3. **Prediction Horizon:** Works best for immediate next race; long-term predictions less reliable
4. **New Drivers:** Limited historical data for rookies (Hadjar, Antonelli, Lindblad, Bortoleto, Bearman)
5. **Regulation Changes:** 2026 being a regulation reset year increases uncertainty
6. **Weather Uncertainty:** 10-day forecasts have limited accuracy
7. **Human Factors:** Cannot predict team strategy calls, driver errors, mechanical failures beyond historical patterns

---

## Future Enhancement Opportunities

1. **Live Race Updates:** Real-time prediction updates during race weekend
2. **Strategy Simulation:** Pit stop strategy optimizer
3. **League Ranking:** Multi-player league analysis
4. **Mobile App:** Native mobile interface
5. **Social Features:** Share predictions, compare with friends
6. **Historical Backtesting UI:** Interactive visualization of model accuracy over time
7. **Explainable AI Dashboard:** SHAP values, feature importance visualization
8. **Sentiment Analysis:** Incorporate team radio, press conference sentiment
9. **Telemetry Deep Learning:** CNN-based telemetry pattern recognition
10. **Multi-Season Championship:** Optimize for full season, not just single race

---

## Conclusion

This is a **production-grade sports analytics application** that demonstrates:
- Advanced ML engineering (ensemble + deep learning + Bayesian)
- Operations research (linear programming optimization)
- Software engineering best practices (caching, modularity, error handling)
- User experience design (CLI + Web, interactive prompts, rich visualizations)

**Target Audience:** Intermediate to advanced F1 Fantasy players who want data-driven decision making

**Competitive Advantage:** Combines prediction accuracy, risk modeling, and strategic optimization in a single tool with zero API costs

**Maintenance Status:** Actively maintained for 2026 season (last update June 2026 per ARCHITECTURE.md)

---

## Contact & Attribution

Project appears to be private/personal use. No public repository or license information found in analyzed files.

**Analysis Prepared By:** Kiro AI Assistant  
**Date:** July 13, 2026
