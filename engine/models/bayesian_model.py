import os
import numpy as np

# ── SUPPRESS COMPILER & ONEDNN WARNINGS ────────────────────────────────────
# oneDNN custom operations warning (TensorFlow/NumPy)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging

logger = logging.getLogger("f1_predictor.bayesian")


class BayesianPointsModel:
    """
    Robust Expected-Value refiner for fantasy points.

    Analytical conjugate approximation (NOT a sampled ZINB — the docstring
    history claimed PyMC/PyTensor; that stack was removed for performance):
      1. Beta-Bernoulli posterior over per-race DNF probability ψ
      2. Known-variance Normal conjugate posterior over non-DNF mean points μ
      3. EV = (1 − ψ) · μ + ψ · E[points retained on a DNF weekend]

    DNF identification: callers should pass explicit `dnf_flags`. Zeros in the
    points list are NOT treated as DNFs (a classified P11 legitimately scores
    zero) unless no flags are available.
    """

    # Fantasy points typically retained on a DNF weekend (quali bonus etc.)
    DNF_QUALI_POINTS_RETAINED = 2.0

    def __init__(self):
        self.evs = {}

    def fit_driver(
        self,
        driver_name: str,
        historical_points: list[float],
        ensemble_pred_pts: float = 15.0,
        circuit_key: str = None,
        dnf_flags: list[bool] = None,
    ) -> float:
        """
        Fit the model for a single driver with track-informed priors.

        Parameters
        ----------
        driver_name: Full name of the driver
        historical_points: List of fantasy points scored in recent races
        ensemble_pred_pts: The deterministic point prediction from the ML ensemble (prior mean)
        circuit_key: Optional circuit key to load track-specific DNF and variance priors
        dnf_flags: Optional per-race DNF flags aligned with historical_points.
                   When None, zeros are NOT counted as DNFs (only the prior informs ψ).
        """
        points_data = np.array([max(0.0, float(p)) for p in historical_points])

        sc_prob, fli_risk, track_width, alt_m, weather_var, tire_deg = 0.40, 0.05, 14.0, 0.0, 1.0, 3.0
        if circuit_key:
            try:
                from engine.core.track_features_loader import load_track_features
                tf = load_track_features(circuit_key)
                if tf:
                    sc_prob = float(tf.get("sc_probability", sc_prob))
                    fli_risk = float(tf.get("first_lap_incident_risk", fli_risk))
                    track_width = float(tf.get("track_width_m", track_width))
                    alt_m = float(tf.get("altitude_m", alt_m))
                    weather_var = float(tf.get("weather_variability", weather_var))
                    tire_deg = float(tf.get("tire_degradation", tire_deg))
            except Exception as e:
                logger.warning("Track-feature priors unavailable for %s: %s", circuit_key, e)

        # Dynamic Prior Mean for DNF probability
        mu_dnf_prior = 0.05 + 0.18 * sc_prob + 0.22 * fli_risk + 0.12 * max(0.0, 1.0 - track_width / 16.0) + 0.06 * (alt_m / 2250.0)
        mu_dnf_prior = float(np.clip(mu_dnf_prior, 0.02, 0.45))

        N0 = 10.0
        alpha0 = N0 * mu_dnf_prior
        beta0 = N0 * (1.0 - mu_dnf_prior)

        # If very little data, use prior expected value
        if len(points_data) < 3:
            robust_ev = (
                (1.0 - mu_dnf_prior) * ensemble_pred_pts
                + mu_dnf_prior * self.DNF_QUALI_POINTS_RETAINED
            )
            self.evs[driver_name] = robust_ev
            return robust_ev

        num_races = len(points_data)

        # 1. Update DNF probability ψ via Beta conjugate update.
        # Use EXPLICIT flags when provided; a zero-point classified finish is
        # not a retirement and must not inflate ψ.
        if dnf_flags is not None and len(dnf_flags) == num_races:
            dnf_count = sum(1 for f in dnf_flags if f)
        else:
            dnf_count = 0
            if any(p <= 0 for p in points_data):
                logger.debug(
                    "%s: no DNF flags supplied; zeros excluded from ψ update",
                    driver_name,
                )
        psi_dnf_post = (alpha0 + dnf_count) / (N0 + num_races)

        # 2. Update mean scoring-race points μ via known-variance Normal conjugate
        # (all non-DNF races count toward μ, including low-scoring ones)
        var_prior = 100.0 * (1.0 + 0.25 * (weather_var - 1.0) + 0.18 * (tire_deg - 3.0) + 0.20 * max(0.0, 1.0 - track_width / 16.0))
        weight_prior = 1.0 / max(10.0, var_prior)

        non_dnf_points = [
            p for p, f in zip(points_data, (dnf_flags or [False] * num_races))
            if not f
        ]
        if non_dnf_points:
            emp_mu = float(np.mean(non_dnf_points))
            emp_var = float(np.var(non_dnf_points)) if len(non_dnf_points) > 1 else 25.0
            weight_data = len(non_dnf_points) / max(5.0, emp_var)
            mu_post = (ensemble_pred_pts * weight_prior + emp_mu * weight_data) / (weight_prior + weight_data)
        else:
            mu_post = ensemble_pred_pts

        # 3. Zero-inflated EV: scoring races at μ_post, DNF weekends retain a
        # small expected value (qualifying bonus survives a Sunday retirement).
        robust_ev = float((1.0 - psi_dnf_post) * mu_post + psi_dnf_post * self.DNF_QUALI_POINTS_RETAINED)
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
