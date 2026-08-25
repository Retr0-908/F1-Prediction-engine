# F1 Fantasy Predictor: Technical Process & Architecture

This document outlines the internal mechanics, data analysis methodologies, and mathematical calculations used by the F1 Fantasy Predictor tool.

## 1. System Architecture & Workflow
The program follows a modular pipeline orchestrated by `main.py`:

1.  **Context Detection (`data_fetcher.py`)**: Identifies the current date, the upcoming Grand Prix round, and circuit characteristics.
2.  **Environmental Analysis (`weather.py`)**: Fetches real-time weather forecasts for the circuit coordinates. Rain risk is converted into a heuristic modifier for DNF probabilities and pace variance.
3.  **Performance Modeling (`elo_ratings.py`)**: Calculates Glicko-2 ratings for all drivers based on historical pairwise comparisons.
4.  **Predictive Engine (`predictor.py`)**:
    *   **Ensemble ML**: Combines Random Forest, XGBoost, and LightGBM models.
    *   **Temporal Analysis**: Uses an LSTM (Long Short-Term Memory) network to capture driver "form" and momentum.
5.  **Simulation Layer (`monte_carlo.py`)**: Executes 1,000+ probabilistic simulations of the race weekend to account for stochastic events (Safety Cars, DNFs, Rain).
6.  **Optimization Layer (`fantasy_optimizer.py`)**: Solves the knapsack-style problem of maximizing fantasy points within budget constraints using linear programming.
7.  **Strategy Layer (`chip_advisor.py`)**: Analyzes the season context to recommend optimal "Chip" usage (e.g., Wildcard, Limitless).

---

## 2. Data Analysis Methodologies

### Glicko-2 Rating System
Unlike standard Elo, Glicko-2 accounts for **Rating Deviation (RD)** and **Volatility**.
*   **RD (φ)**: Measures the uncertainty of a driver's rating. High RD for rookies or returning drivers leads to larger rating swings.
*   **Volatility (σ)**: Measures the consistency of performance. Consistent finishers like Max Verstappen maintain low volatility.
*   **Pairwise Comparisons**: Every race is treated as a tournament where a driver "wins" against everyone they finish ahead of. The rating is updated based on the "surprise" factor of the result.

### Multi-Model Ensemble
The prediction engine doesn't rely on a single algorithm. It uses a **weighted average** of:
*   **Gradient Boosted Trees (XGB/LGBM)**: Excellent at capturing non-linear relationships between track features (e.g., downforce levels) and performance.
*   **LSTM Neural Network**: Specifically looks at the sequence of the last 10 races to detect rising or falling "form" that static stats might miss.

---

## 3. Key Calculations & Formulas

### Expected Value (EV) Points
The system prioritizes **Expected Value** over single-point "best case" scenarios.
$$EV = \sum (Points_{Outcome} \times Probability_{Outcome})$$
This is calculated via the Monte Carlo engine by averaging the results of all successful (non-DNF) simulation runs.

### Monte Carlo Simulation Logic
Each iteration of the simulation follows this logic:
1.  **DNF Check**: For each driver, a random float $r \in [0,1]$ is generated. If $r < P(DNF)_{driver}$, the driver is marked as DNF.
2.  **Safety Car (SC) Impact**: If a random event triggers a Safety Car (based on circuit historical frequency), positions are "shuffled" using a Gaussian noise function:
    $$Pos_{new} = Pos_{old} + \mathcal{N}(0, SC\_Noise)$$
3.  **Weather Modifier**: Rain risk increases the $SC\_Noise$ and $P(DNF)$ variables.

### Team Optimization (Linear Programming)
The tool uses the `PuLP` library to solve the following optimization:
*   **Objective**: Maximize $\sum (Points_{Driver}) + \sum (Points_{Constructor})$
*   **Constraints**:
    *   $\sum (Price_{Driver}) + \sum (Price_{Constructor}) \le Budget$
    *   Count(Drivers) $= 5$
    *   Count(Constructors) $= 2$
    *   Max 3 players from a single F1 Team.

### Differential Score
To help users climb leaderboards, a **Differential Score** is calculated:
$$DiffScore = \frac{EV\_Points}{Ownership\% + \epsilon}$$
Drivers with high points but low ownership are flagged as "Differential Picks."

---

## 4. Chip Strategy Calculations
The `chip_advisor.py` assigns a **Confidence Score** to each chip based on:
*   **Wildcard**: High score if current team EV is $> 15\%$ lower than the Global Optimal Team.
*   **Final Fix**: High score if qualifying results significantly deviate from pre-race projections (e.g., a top driver starting P20).
*   **Autopilot**: High score if the top two drivers have within $2\%$ projected points of each other (high risk of wrong captain choice).
