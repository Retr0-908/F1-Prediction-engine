"""
tire_model.py — Multi-Task Neural Network for Tire Degradation & Strategy Prediction
Phase 3 implementation for the F1 Predictor project.

Architecture (multi-task, shared encoder):
  Input:  [compound_enc, stint_length, track_temp, air_temp, lap_number,
           circuit_track_type, rain_enc, avg_lap_sec_normalized]
  Shared: Dense(64, relu) → BatchNorm → Dropout(0.2)
          Dense(32, relu) → BatchNorm → Dropout(0.2)

  Task A — Degradation Head (regression):
    Dense(16, relu) → Dense(1, linear)   → deg_per_lap (seconds/lap)

  Task B — Strategy Head (classification):
    Dense(16, relu) → Dense(N_STRATEGIES, softmax) → optimal pit strategy index

Research references:
  - Multi-task learning for sports: Caruana (1997), "Multitask Learning", ML 28(1)
  - Tire thermal modelling: Kelly (2008), PhD Thesis, University of Leeds
  - F1 compound degradation: Bekker & Lotz (2009), in Taylor & Francis F1 compendium

Training data source:
  - FastF1 historical cache (2022–2025): all race sessions with tire stint data
  - 100% free — FastF1 pulls from the official F1 timing service; no API key needed
  - Cached to FASTF1_CACHE_DIR — zero repeated downloads after first run

Cost & rate-limit notes:
  - get_tire_stints() uses 72h JSON cache; FastF1 uses its own parquet cache
  - No cloud API calls — model trains and infers entirely on local hardware
  - Model saved to cache/models/tire_model.keras (Keras SavedModel format)
"""

import os
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

from engine.core.config import CURRENT_SEASON, HISTORICAL_SEASONS, CIRCUITS
from engine.core.data_fetcher import get_tire_stints, get_season_schedule

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Tire compound encoding
COMPOUND_ENC = {
    "SOFT":        0,
    "MEDIUM":      1,
    "HARD":        2,
    "INTERMEDIATE": 3,
    "WET":         4,
    "UNKNOWN":     1,   # default to Medium
}
N_COMPOUNDS = len(COMPOUND_ENC)

# Pit strategy classes (most common F1 strategies)
STRATEGIES = [
    "SOFT→MEDIUM",
    "SOFT→HARD",
    "MEDIUM→HARD",
    "SOFT→MEDIUM→HARD",
    "SOFT→SOFT",
    "MEDIUM→MEDIUM",
    "OTHER",
]
N_STRATEGIES = len(STRATEGIES)

# Track type encoding (matches config.py TRACK_TYPE_ENC)
TRACK_TYPE_ENC = {"street": 0, "hybrid": 1, "permanent": 2}

# Input feature names for the shared encoder
INPUT_FEATURES = [
    "compound_enc",           # 0–4 ordinal
    "stint_length",           # laps (0–60)
    "track_temp",             # °C (15–55)
    "air_temp",               # °C (15–40)
    "lap_number",             # start lap of stint (1–60)
    "track_type_enc",         # 0–2
    "rain_enc",               # 0=dry, 1=mixed, 2=wet
    "avg_lap_sec_norm",       # lap time normalised to [0,1] within-session
]
N_INPUT = len(INPUT_FEATURES)

# Training hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 60
BATCH_SIZE = 64
DROPOUT_RATE = 0.2
DEG_LOSS_WEIGHT = 0.7    # higher weight on regression task (primary goal)
STR_LOSS_WEIGHT = 0.3    # strategy classification is secondary

from engine.core.paths import MODEL_CACHE_DIR
DEG_MODEL_PATH  = MODEL_CACHE_DIR / "tire_model.keras"
DEG_SCALER_PATH = MODEL_CACHE_DIR / "tire_scaler.pkl"


# ─────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────

def _infer_strategy_label(stint_compounds: list[str]) -> int:
    """
    Convert a list of compounds (in race order) to a strategy class index.
    e.g. ["SOFT", "MEDIUM"] → "SOFT→MEDIUM" → 0
    """
    key = "→".join(c for c in stint_compounds)
    if key in STRATEGIES:
        return STRATEGIES.index(key)
    # Partial match
    for i, s in enumerate(STRATEGIES[:-1]):
        if s in key:
            return i
    return STRATEGIES.index("OTHER")


def _circuit_track_type(gp_name: str) -> int:
    """Look up track type for a GP name from config; default to permanent (2)."""
    for cname, cfg in CIRCUITS.items():
        if cname.lower() in gp_name.lower() or gp_name.lower() in cname.lower():
            return TRACK_TYPE_ENC.get(cfg.get("track_type", "permanent"), 2)
    return 2  # permanent


def build_tire_dataset(
    seasons: list[int],
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds X (stint features), y_deg (degradation), y_strat (strategy label) arrays.

    For each race in each season, calls get_tire_stints() which:
      - Reads from FastF1 cache (free, no API call if cached)
      - Falls back to downloading from official F1 timing server (free)
      - Applies our 72h JSON cache layer on top

    Returns
    -------
    X        : (N, N_INPUT) float32 — per-stint feature matrix
    y_deg    : (N,) float32         — deg_per_lap regression targets
    y_strat  : (N,) int32           — strategy class labels per driver per race
    """
    X_list:      list[list[float]] = []
    y_deg_list:  list[float]       = []
    y_strat_list: list[int]        = []

    training_seasons = [y for y in seasons if y < CURRENT_SEASON]

    for year in training_seasons:
        if verbose:
            print(f"  [tire_model] Loading {year} tire data...")
        try:
            schedule = get_season_schedule(year)
        except Exception:
            continue

        for race in schedule:
            gp_name = race["name"]
            try:
                tire_data = get_tire_stints(year, gp_name, session_type="R")
            except Exception:
                continue

            if not tire_data:
                continue

            track_enc = _circuit_track_type(gp_name)

            # Compute per-session avg_lap_sec for normalisation
            all_laps = [
                s["avg_lap_sec"]
                for stints in tire_data.values()
                for s in stints
                if s.get("avg_lap_sec") is not None
            ]
            session_avg_lap = float(np.mean(all_laps)) if all_laps else 90.0
            session_lap_std = float(np.std(all_laps)) if len(all_laps) > 1 else 5.0

            for driver, stints in tire_data.items():
                if not stints:
                    continue

                # Driver's full compound sequence for this race → strategy label
                driver_compounds = [s.get("compound", "UNKNOWN") for s in stints]
                strat_label = _infer_strategy_label(driver_compounds)

                for s in stints:
                    compound_enc = COMPOUND_ENC.get(s.get("compound", "UNKNOWN"), 1)
                    stint_length = float(s.get("stint_length", 15))
                    track_temp   = float(s.get("avg_track_temp") or 35.0)
                    air_temp     = track_temp - 10.0          # approximate: air ≈ track - 10°C
                    lap_number   = float(s.get("start_lap", 1))
                    avg_lap_sec  = float(s.get("avg_lap_sec") or session_avg_lap)
                    deg_per_lap  = s.get("deg_per_lap")

                    # Skip stints without degradation data (< 3 laps: can't compute slope)
                    if deg_per_lap is None:
                        continue

                    # Normalise lap time within session (z-score, then clip to [-3, 3])
                    avg_lap_norm = np.clip(
                        (avg_lap_sec - session_avg_lap) / max(session_lap_std, 0.1), -3.0, 3.0
                    )

                    X_list.append([
                        float(compound_enc),
                        float(stint_length),
                        float(track_temp),
                        float(air_temp),
                        float(lap_number),
                        float(track_enc),
                        0.0,              # rain_enc: not available from get_tire_stints
                        float(avg_lap_norm),
                    ])
                    y_deg_list.append(float(deg_per_lap))
                    y_strat_list.append(strat_label)

    if not X_list:
        raise RuntimeError(
            "No tire stint data found. Run the predictor at least once to populate the FastF1 cache."
        )

    X      = np.array(X_list,     dtype=np.float32)
    y_deg  = np.array(y_deg_list, dtype=np.float32)
    y_strat = np.array(y_strat_list, dtype=np.int32)

    if verbose:
        print(f"  [tire_model] Dataset: {len(y_deg)} stints from {len(training_seasons)} seasons.")

    return X, y_deg, y_strat


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────

def build_tire_model(n_input: int = N_INPUT, n_strategies: int = N_STRATEGIES) -> keras.Model:
    """
    Builds the multi-task tire degradation + strategy neural network.

    Shared Encoder → two task-specific heads:
      Head A: Degradation regression (deg_per_lap in seconds/lap)
      Head B: Pit strategy classification (which compound sequence is optimal)
    """
    inp = keras.Input(shape=(n_input,), name="stint_features")

    # ── Shared Encoder ───────────────────────────────────────────────────────
    x = layers.Dense(64, name="shared_dense_1")(inp)
    x = layers.BatchNormalization(name="shared_bn_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(DROPOUT_RATE, name="shared_dropout_1")(x)

    x = layers.Dense(32, name="shared_dense_2")(x)
    x = layers.BatchNormalization(name="shared_bn_2")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(DROPOUT_RATE, name="shared_dropout_2")(x)

    # ── Task A: Degradation Head (regression) ────────────────────────────────
    deg_x  = layers.Dense(16, activation="relu", name="deg_dense")(x)
    deg_out = layers.Dense(1, activation="linear", name="deg_output")(deg_x)

    # ── Task B: Strategy Head (classification) ───────────────────────────────
    strat_x  = layers.Dense(16, activation="relu", name="strat_dense")(x)
    strat_out = layers.Dense(n_strategies, activation="softmax", name="strat_output")(strat_x)

    model = keras.Model(
        inputs=inp,
        outputs={"deg_output": deg_out, "strat_output": strat_out},
        name="F1TireDegradationModel",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            "deg_output":   "huber",                # robust to stint outliers
            "strat_output": "sparse_categorical_crossentropy",
        },
        loss_weights={
            "deg_output":   DEG_LOSS_WEIGHT,
            "strat_output": STR_LOSS_WEIGHT,
        },
        metrics={
            "deg_output":   ["mae"],
            "strat_output": ["accuracy"],
        },
    )
    return model


# ─────────────────────────────────────────────
# NORMALISATION (simple min-max per feature)
# ─────────────────────────────────────────────

class TireScaler:
    """Stores feature-wise min/max for input and output normalisation."""

    def __init__(self):
        self.X_min: Optional[np.ndarray] = None
        self.X_max: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        self.X_min = X.min(axis=0)
        self.X_max = X.max(axis=0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        denom = self.X_max - self.X_min
        denom = np.where(denom == 0, 1.0, denom)
        return (X - self.X_min) / denom

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

class TireDegradationModel:
    """
    Wraps the Keras multi-task model with train / predict / persist methods.

    Usage in predictor.py / monte_carlo.py:
        from engine.models.tire_model import TireDegradationModel
        tdm = TireDegradationModel()
        tdm.load_or_train()
        delta = tdm.predict_deg_per_lap("MEDIUM", stint_length=20, track_temp=38)
        strategy = tdm.predict_strategy("Max Verstappen", stint_compounds=["SOFT"])
    """

    def __init__(self):
        self.model: Optional[keras.Model] = None
        self.scaler: Optional[TireScaler] = None
        self._trained: bool = False

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self):
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.model.save(str(DEG_MODEL_PATH))
        with open(DEG_SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f, protocol=5)
        print(f"  [tire_model] Saved to {DEG_MODEL_PATH}")

    def load(self) -> bool:
        if not DEG_MODEL_PATH.exists() or not DEG_SCALER_PATH.exists():
            return False
        try:
            self.model = keras.models.load_model(str(DEG_MODEL_PATH))
            with open(DEG_SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
            self._trained = True
            print(f"  [tire_model] Loaded cached model from {DEG_MODEL_PATH}")
            return True
        except Exception as exc:
            print(f"  [tire_model] Could not load cache ({exc}), will retrain.")
            return False

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, verbose: bool = True, force: bool = False):
        """
        Train on 2022–2025 FastF1 cached tire stint data (100% free, local).
        Saves model to disk; subsequent calls load instantly from cache.
        """
        if not force and self.load():
            return

        if verbose:
            print("  [tire_model] Building tire stint dataset from FastF1 cache...")

        # Only 2022+ is used: ground-effect era tires are meaningfully different
        # from earlier Pirelli compounds — avoids introducing systematic bias
        relevant_seasons = [y for y in HISTORICAL_SEASONS if y >= 2022]
        X, y_deg, y_strat = build_tire_dataset(relevant_seasons, verbose=verbose)

        self.scaler = TireScaler()
        X_n = self.scaler.fit_transform(X)

        self.model = build_tire_model()

        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=5, min_lr=1e-5, verbose=0
            ),
        ]

        if verbose:
            print(f"  [tire_model] Training on {len(y_deg)} stints for up to {EPOCHS} epochs...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(
                X_n,
                {"deg_output": y_deg, "strat_output": y_strat},
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=0.15,
                callbacks=callbacks,
                verbose=1 if verbose else 0,
            )

        self._trained = True
        self.save()

    def load_or_train(self, verbose: bool = True, force: bool = False):
        self.train(verbose=verbose, force=force)

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_deg_per_lap(
        self,
        compound: str,
        stint_length: float = 20.0,
        track_temp: float = 35.0,
        air_temp: Optional[float] = None,
        lap_number: float = 1.0,
        track_type: str = "permanent",
        rain_enc: float = 0.0,
        avg_lap_sec_norm: float = 0.0,
        circuit_key: Optional[str] = None,
        altitude_m: Optional[float] = None,
        tire_degradation: Optional[float] = None,
        deg_compound_delta: Optional[float] = None,
        sm_zones: Optional[float] = None,
    ) -> float:
        """
        Predict tire pace degradation rate for a given stint enriched with track physics.

        Returns
        -------
        float: estimated pace loss per lap in seconds (e.g. 0.045 = 45ms/lap).
               Positive = getting slower. Used as δ_compound in the Julia MC engine.
        """
        alt_val = altitude_m or 0.0
        deg_val = tire_degradation or 3.0
        delta_val = deg_compound_delta or 0.20
        sm_val = sm_zones or 2.0

        if circuit_key and (altitude_m is None or tire_degradation is None):
            try:
                from engine.core.track_features_loader import load_track_features
                tf = load_track_features(circuit_key)
                if tf:
                    alt_val = float(tf.get("altitude_m", alt_val))
                    deg_val = float(tf.get("tire_degradation", deg_val))
                    delta_val = float(tf.get("deg_compound_delta", delta_val))
                    sm_val = float(tf.get("sm_zones", sm_val))
            except Exception:
                pass

        if not self._trained or self.model is None:
            # Fallback: historical averages per compound (Pirelli data)
            _FALLBACK = {"SOFT": 0.065, "MEDIUM": 0.040, "HARD": 0.022,
                         "INTERMEDIATE": 0.030, "WET": 0.020, "UNKNOWN": 0.040}
            base_deg = _FALLBACK.get(compound.upper(), 0.040)
            c_idx = COMPOUND_ENC.get(compound.upper(), 1)
            deg_adj = base_deg * (1.0 + 0.15 * (deg_val - 3.0)) + delta_val * max(0, 1 - c_idx)
            alt_mult = 1.0 + 0.04 * (alt_val / 1000.0)
            return max(0.005, deg_adj * alt_mult)

        compound_enc = float(COMPOUND_ENC.get(compound.upper(), 1))
        air_temp_v   = float(air_temp) if air_temp is not None else track_temp - 10.0
        track_enc    = float(TRACK_TYPE_ENC.get(track_type, 2))

        X = np.array([[
            compound_enc, float(stint_length), float(track_temp),
            air_temp_v, float(lap_number), track_enc,
            float(rain_enc), float(avg_lap_sec_norm),
        ]], dtype=np.float32)

        X_n = self.scaler.transform(X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = self.model.predict(X_n, verbose=0)

        deg = float(preds["deg_output"][0, 0])
        alt_mult = 1.0 + 0.04 * (alt_val / 1000.0)
        return max(0.001, deg * alt_mult)

    def predict_strategy(
        self,
        current_compounds_used: list[str],
        remaining_laps: int = 30,
        track_temp: float = 35.0,
        track_type: str = "permanent",
    ) -> tuple[str, float]:
        """
        Predict the optimal remaining pit strategy given already-used compounds.

        Returns
        -------
        (strategy_name, confidence_probability)
        e.g. ("MEDIUM→HARD", 0.72)
        """
        if not self._trained or self.model is None:
            return ("MEDIUM→HARD", 0.5)

        # Encode current stint as if it were about to start
        last_compound = current_compounds_used[-1] if current_compounds_used else "SOFT"
        compound_enc  = float(COMPOUND_ENC.get(last_compound.upper(), 1))
        track_enc     = float(TRACK_TYPE_ENC.get(track_type, 2))

        X = np.array([[
            compound_enc, float(remaining_laps), float(track_temp),
            float(track_temp) - 10.0, 1.0, track_enc, 0.0, 0.0,
        ]], dtype=np.float32)

        X_n = self.scaler.transform(X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = self.model.predict(X_n, verbose=0)

        probs = preds["strat_output"][0]
        best_idx = int(np.argmax(probs))
        return STRATEGIES[best_idx], float(probs[best_idx])


# ─────────────────────────────────────────────
# SINGLETON ACCESSOR
# ─────────────────────────────────────────────
_tire_model_instance: Optional[TireDegradationModel] = None


def get_tire_model(verbose: bool = False, force: bool = False) -> TireDegradationModel:
    """
    Returns the singleton TireDegradationModel, loading from cache or training.
    Call this from predictor.py / monte_carlo.py to avoid repeated model loads.
    """
    global _tire_model_instance
    if _tire_model_instance is None:
        _tire_model_instance = TireDegradationModel()
        _tire_model_instance.load_or_train(verbose=verbose, force=force)
    return _tire_model_instance
