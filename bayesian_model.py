import os
import warnings
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

# ── SUPPRESS COMPILER & ONEDNN WARNINGS ────────────────────────────────────
# oneDNN custom operations warning (TensorFlow/NumPy)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# PyTensor compiler warning (silence the g++ missing alert)
# We check if g++ exists; if not, we set the flag to avoid the warning.
if not shutil.which("g++"):
    os.environ["PYTENSOR_FLAGS"] = "cxx="
# ────────────────────────────────────────────────────────────────────────────

# PyMC removed for performance. Using analytical approximation.

class BayesianPointsModel:
    """
    Fits a Zero-Inflated Negative Binomial model to a driver's historical fantasy points.
    Returns the robust Expected Value (EV) and the posterior predictive distribution.
    """

    def __init__(self):
        self.traces = {}
        self.evs = {}

    def fit_driver(
        self,
        driver_name: str,
        historical_points: list[float],
        ensemble_pred_pts: float = 15.0,
        circuit_key: str = None,
    ) -> float:
        """
        Fit a ZINB model for a single driver with track-informed Bayesian priors.
        
        Parameters
        ----------
        driver_name: Full name of the driver
        historical_points: List of fantasy points scored in recent races
        ensemble_pred_pts: The deterministic point prediction from the ML ensemble (used to center the prior)
        circuit_key: Optional circuit key to load track-specific DNF and variance priors
        """
        # Ensure points are integers and >= 0
        points_data = np.array([max(0, int(round(p))) for p in historical_points])
        
        sc_prob, fli_risk, track_width, alt_m, weather_var, tire_deg = 0.40, 0.05, 14.0, 0.0, 1.0, 3.0
        if circuit_key:
            try:
                from track_features_loader import load_track_features
                tf = load_track_features(circuit_key)
                if tf:
                    sc_prob = float(tf.get("sc_probability", sc_prob))
                    fli_risk = float(tf.get("first_lap_incident_risk", fli_risk))
                    track_width = float(tf.get("track_width_m", track_width))
                    alt_m = float(tf.get("altitude_m", alt_m))
                    weather_var = float(tf.get("weather_variability", weather_var))
                    tire_deg = float(tf.get("tire_degradation", tire_deg))
            except Exception:
                pass

        # Dynamic Prior Mean for DNF probability
        mu_dnf_prior = 0.05 + 0.18 * sc_prob + 0.22 * fli_risk + 0.12 * max(0.0, 1.0 - track_width / 16.0) + 0.06 * (alt_m / 2250.0)
        mu_dnf_prior = float(np.clip(mu_dnf_prior, 0.02, 0.45))

        N0 = 10.0
        alpha0 = N0 * mu_dnf_prior
        beta0 = N0 * (1.0 - mu_dnf_prior)

        # If very little data, use prior expected value
        if len(points_data) < 3:
            robust_ev = (1.0 - mu_dnf_prior) * ensemble_pred_pts
            self.evs[driver_name] = robust_ev
            return robust_ev

        # 1. Update DNF probability psi_post via Beta conjugate update
        num_races = len(points_data)
        zero_count = sum(1 for p in points_data if p <= 0)
        psi_dnf_post = (alpha0 + zero_count) / (N0 + num_races)

        # 2. Update mean non-zero points mu_post via Heteroscedastic Normal conjugate update
        non_zero_points = [p for p in points_data if p > 0]
        var_prior = 100.0 * (1.0 + 0.25 * (weather_var - 1.0) + 0.18 * (tire_deg - 3.0) + 0.20 * max(0.0, 1.0 - track_width / 16.0))
        weight_prior = 1.0 / max(10.0, var_prior)

        if non_zero_points:
            emp_mu = float(np.mean(non_zero_points))
            emp_var = float(np.var(non_zero_points)) if len(non_zero_points) > 1 else 25.0
            weight_data = len(non_zero_points) / max(5.0, emp_var)
            mu_post = (ensemble_pred_pts * weight_prior + emp_mu * weight_data) / (weight_prior + weight_data)
        else:
            mu_post = ensemble_pred_pts

        # 3. Expected Value of Zero-Inflated Model
        robust_ev = float((1.0 - psi_dnf_post) * mu_post)
        self.evs[driver_name] = robust_ev
        return robust_ev

    def get_ev(self, driver_name: str, fallback: float) -> float:
        """Get the computed robust Expected Value, or fallback if not fitted."""
        return self.evs.get(driver_name, fallback)

# Singleton accessor
_bayesian_model_instance = None

def get_bayesian_model() -> BayesianPointsModel:
    global _bayesian_model_instance
    if _bayesian_model_instance is None:
        _bayesian_model_instance = BayesianPointsModel()
    return _bayesian_model_instance
