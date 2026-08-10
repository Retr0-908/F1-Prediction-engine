# F1 Fantasy Feature Engineering Design

## Objective
Maximize F1 Fantasy points by adding targeted features to the machine learning ensemble that directly correlate with Fantasy scoring mechanisms (specifically overtakes and DNF avoidance).

## Architecture

We are adding two new features to the main prediction model pipeline. 

### 1. Overtake Propensity (`driver_overtake_delta`)
- **Purpose**: Accurately predict positions gained during the race. In F1 Fantasy, positions gained from qualifying to the race finish award significant bonus points.
- **Data Source**: Historical race results vs qualifying results.
- **Calculation**: A rolling average of (Starting Grid Position - Finishing Position) over the driver's last 10 races.
- **Integration**: Added as a core feature to the `F1Predictor` ensemble in `predictor.py`.

### 2. DNF Risk Factor (`driver_dnf_risk`)
- **Purpose**: Avoid recommending drivers who are highly likely to crash or suffer mechanical failures, as DNFs incur massive negative points in F1 Fantasy.
- **Data Source**: Historical incident rates and constructor reliability tracking.
- **Calculation**: A combined metric (0.0 to 1.0) derived from the driver's recent crash frequency and the team's mechanical failure rate.
- **Integration**: Added as a core feature to the `F1Predictor` ensemble in `predictor.py`.

## Implementation Plan

1. **Update Data Fetching**: Modify data pipelines (e.g., `data_fetcher.py` or `track_features_loader.py`) to extract and calculate `driver_overtake_delta` and `driver_dnf_risk`.
2. **Update Predictor Ensemble**: Add the two new features to the `FEATURE_NAMES` list in `predictor.py`.
3. **Validate Schema**: Run `python validate_track_features.py` and `python sanity_check.py` to ensure the new feature count matches expectations (now 56 total features).
4. **Cache Invalidation & Retraining**: Clear stale model caches (`cache/models/ensemble_*.pkl`) and execute a backtest (`python backtest.py --years 2024 2025`) to train the models with the new fantasy-optimized features.

## Testing & Validation
- The `backtest.py` output will be used to verify that the Mean Absolute Error (MAE) and Top-5 Hit rates remain stable, while verifying the model successfully penalizes high-DNF-risk drivers during the Monte Carlo simulations.
