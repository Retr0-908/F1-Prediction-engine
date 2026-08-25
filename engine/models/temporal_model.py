"""
temporal_model.py - LSTM + Multi-Head Attention Temporal Form Model
Phase 3 implementation for the F1 Predictor project.

Architecture (dual-branch, fused at output):
  Branch 1 - Historical Trend (LSTM):
    Input:  sequence of last SEQ_LENGTH=5 race results per driver
    Layers: LSTM(64, return_sequences=True)   MultiHeadAttention(4 heads, key_dim=64)
              GlobalAveragePooling   Dense(32, relu)   Dropout(0.3)
    Output: 32-dimensional driver_momentum tensor

  Branch 2 - Race-Day Context (Dense):
    Input:  [qualifying_position, track_temp, rain_risk, sc_prob, ctor_elo_zscore]
    Layers: Dense(16, relu)   Dropout(0.2)
    Output: 16-dimensional context vector

  Fusion:
    Concatenate([branch1, branch2])   Dense(64, relu)   Dropout(0.2)
      Dense(32, relu)   Dense(1, linear)   # outputs expected pace delta

Research references:
  - LSTM for sequential sports data: Singh et al. (2024), JAAFR Vol. 26
  - Multi-Head Attention: Vaswani et al. (2017), "Attention Is All You Need"
  - Hyperparameters (LSTM=64, Heads=4, SEQ=5): as per Singh et al. Table A-1

Cost & data notes:
  - All training data comes from local FastF1 cache (100% free, no API key)
  - Model weights saved to cache/models/temporal_model.keras (local disk only)
  - No cloud inference - model runs locally via TensorFlow/Keras
"""

import os
import threading
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # suppress TF startup noise
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

from engine.core.config import CURRENT_SEASON, HISTORICAL_SEASONS, DRIVER_TEAMS_2025
from engine.core.data_fetcher import get_season_results, get_season_schedule, get_tire_stints

#  
# CONSTANTS
#  
SEQ_LENGTH = 5          # number of historical races per sequence (per Singh et al.)
CONTEXT_DIM = 5         # [quali_pos, track_temp, rain_risk_enc, sc_prob, ctor_elo_zscore]
LSTM_UNITS = 64
ATTN_HEADS = 4
ATTN_KEY_DIM = 64
DROPOUT_TEMPORAL = 0.3
DROPOUT_FUSION = 0.2
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32

from engine.core.paths import MODEL_CACHE_DIR
import logging
logger = logging.getLogger("f1_predictor.temporal_model")

MODEL_PATH = MODEL_CACHE_DIR / "temporal_model.keras"
SCALER_PATH = MODEL_CACHE_DIR / "temporal_scaler.pkl"


#  
# FEATURE EXTRACTION FOR LSTM SEQUENCES
#  

# Per-race sequence features (what the LSTM sees at each time-step)
SEQ_FEATURES = [
    "position",       # finish position (1-22)
    "dnf",            # 0/1 DNF flag
    "points",         # race points scored
    "grid",           # starting grid position
    "position_gain",  # grid - position (positive = moved forward)
    "deg_per_lap",    # tire degradation slope from get_tire_stints() (0 if unavailable)
]
N_SEQ_FEATURES = len(SEQ_FEATURES)


def _extract_sequence_for_driver(
    driver_name: str,
    race_results_by_round: list[dict],
    tire_data_by_round: dict[int, dict],
    seq_length: int = SEQ_LENGTH,
) -> np.ndarray:
    """
    Build a (seq_length, N_SEQ_FEATURES) array for one driver from their last
    `seq_length` race results. Pads with zeros if fewer races are available.

    Parameters
    ----------
    driver_name          : Full driver name, e.g. "Max Verstappen"
    race_results_by_round: List of {round, name, results:[...]} dicts (sorted by round asc)
    tire_data_by_round   : {round_num: {driver_name: [stint_dicts]}} from get_tire_stints
    seq_length           : Number of past races to include (default 5)

    Returns
    -------
    np.ndarray of shape (seq_length, N_SEQ_FEATURES), dtype float32
    """
    seq: list[list[float]] = []

    for race in race_results_by_round:
        round_num = race["round"]
        result = next((r for r in race.get("results", []) if r["name"] == driver_name), None)
        if result is None:
            continue

        position = float(result.get("position", 20))
        dnf_flag = 1.0 if result.get("dnf", False) else 0.0
        points   = float(result.get("points", 0.0))
        grid     = float(result.get("grid", position))
        pos_gain = grid - position   # positive = gained places

        # Tire degradation: average deg_per_lap across all stints in this race
        stint_list = tire_data_by_round.get(round_num, {}).get(driver_name, [])
        deg_vals = [s["deg_per_lap"] for s in stint_list if s.get("deg_per_lap") is not None]
        deg_per_lap = float(np.mean(deg_vals)) if deg_vals else 0.0

        seq.append([position, dnf_flag, points, grid, pos_gain, deg_per_lap])

    # Trim to last seq_length entries (most recent)
    seq = seq[-seq_length:]

    # Pad from the front with zeros if fewer than seq_length races available
    while len(seq) < seq_length:
        seq.insert(0, [0.0] * N_SEQ_FEATURES)

    return np.array(seq, dtype=np.float32)  # shape: (seq_length, N_SEQ_FEATURES)


#  
# MODEL ARCHITECTURE
#  

def build_temporal_model(
    seq_length: int = SEQ_LENGTH,
    n_seq_features: int = N_SEQ_FEATURES,
    context_dim: int = CONTEXT_DIM,
) -> keras.Model:
    """
    Build dual-branch LSTM + Multi-Head Attention architecture for driver form.

    Branch 1: Sequence of recent race performance vectors (seq_length x n_seq_features)
    Branch 2: Static race-day context vector (context_dim)

    Returns compiled Keras Model targeting position delta (actual_finish - expected_finish).
    """
    #   Branch 1: Sequential Driver Form  
    seq_input = keras.Input(shape=(seq_length, n_seq_features), name="sequence_input")

    # Multi-head self-attention over race sequence
    attn_out = layers.MultiHeadAttention(
        num_heads=ATTN_HEADS,
        key_dim=ATTN_KEY_DIM,
        dropout=DROPOUT_TEMPORAL,
        name="sequence_mha",
    )(seq_input, seq_input)
    attn_add = layers.Add(name="seq_residual")([seq_input, attn_out])
    attn_norm = layers.LayerNormalization(name="seq_norm")(attn_add)

    # Bi-LSTM feature extraction
    bilstm = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_sequences=False, dropout=DROPOUT_TEMPORAL),
        name="bilstm_extractor",
    )(attn_norm)

    # Dense sequence representation
    hist_out = layers.Dense(32, activation="relu", name="history_dense")(bilstm)

    #   Branch 2: Race-Day Context (Expanded to 32 units for 10 features)  
    ctx_input = keras.Input(shape=(context_dim,), name="race_day_context")
    ctx_dense = layers.Dense(32, activation="relu", name="context_dense")(ctx_input)
    ctx_out = layers.Dropout(DROPOUT_FUSION, name="context_dropout")(ctx_dense)

    #   Fusion Head  
    merged = layers.Concatenate(name="fusion_concat")([hist_out, ctx_out])
    fuse1 = layers.Dense(64, activation="relu", name="fusion_dense_1")(merged)
    fuse1 = layers.Dropout(DROPOUT_FUSION, name="fusion_dropout_1")(fuse1)
    fuse2 = layers.Dense(32, activation="relu", name="fusion_dense_2")(fuse1)
    output = layers.Dense(1, activation="linear", name="position_delta")(fuse2)

    model = keras.Model(
        inputs=[seq_input, ctx_input],
        outputs=output,
        name="F1TemporalFormModel",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="huber",          # Huber loss is robust to DNF position outliers
        metrics=["mae"],
    )
    return model


#  
# TRAINING DATASET BUILDER
#  

def build_training_dataset(
    seasons: list[int],
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Builds X_seq, X_ctx, y arrays from historical FastF1 and Jolpica data.
    # Returns (X_seq, X_ctx, y) tuple.
    X_seq_list: list[np.ndarray] = []
    X_ctx_list: list[list[float]] = []
    y_list: list[float] = []

    training_seasons = [y for y in seasons if y < CURRENT_SEASON]

    for year in training_seasons:
        if verbose:
            print(f"  [LSTM dataset] Processing {year}...")

        try:
            schedule = get_season_schedule(year)
            all_results = get_season_results(year)   # 1 cached API call
        except Exception as exc:
            if verbose:
                print(f"    [WARN] Could not load {year}: {exc}")
            continue

        # Organise results by round for O(1) lookup
        results_by_round: dict[int, list[dict]] = {}
        for r in all_results:
            rnd = r["round"]
            if rnd not in results_by_round:
                results_by_round[rnd] = []
            results_by_round[rnd].append(r)

        # Build cumulative list of (round, results) - used to build rolling sequences
        cumulative_rounds: list[dict] = []
        for race in sorted(schedule, key=lambda x: x["round"]):
            rnd = race["round"]
            rnd_results = results_by_round.get(rnd, [])
            if not rnd_results:
                continue

            # Tire stints for this race (cached, free, no API rate limit)
            tire_data: dict[str, list[dict]] = {}
            try:
                tire_data = get_tire_stints(year, race["name"], session_type="R")
            except Exception:
                logger.warning("Suppressed error", exc_info=True)
                pass  # missing tire data is fine - deg_per_lap defaults to 0

            # Build the rolling sequence: up to SEQ_LENGTH races *before* this one
            past_rounds = cumulative_rounds[-SEQ_LENGTH:]  # at most last 5 rounds
            past_tire_data = {}
            for pr in past_rounds:
                try:
                    past_tire_data[pr["round"]] = get_tire_stints(year, pr["name"], "R")
                except Exception:
                    past_tire_data[pr["round"]] = {}

            for r in rnd_results:
                drv = r["name"]

                # Build historical sequence input
                seq = _extract_sequence_for_driver(
                    drv,
                    past_rounds,
                    past_tire_data,
                    seq_length=SEQ_LENGTH,
                )

                # Race-day context: [quali_pos, track_temp, rain_enc, sc_prob, ctor_elo_z]
                # quali_pos = actual grid (varies); sc_prob = real per-circuit value
                # from local JSON/config (no API hit). rain/temp/ctor-elo stay at
                # neutral placeholders — predictor.py feeds the same encodings at
                # inference so the two never diverge structurally.
                grid_pos = float(r.get("grid", 11.0))
                rain_enc = 0.0
                try:
                    from engine.core.data_fetcher import _match_circuit_config as _mcc
                    from engine.core.config import SC_PROBABILITY as _SC_PROB

                    _ckey = (_mcc(race["name"]) or {}).get("key", "")
                    sc_prob = float(_SC_PROB.get(_ckey, 0.40))
                except Exception:
                    sc_prob = 0.40
                track_temp = 35.0   # neutral placeholder (same at inference)
                ctor_elo_z = 0.0    # neutral placeholder (same at inference)

                ctx = np.array([grid_pos, track_temp, rain_enc, sc_prob, ctor_elo_z], dtype=np.float32)

                X_seq_list.append(seq)
                X_ctx_list.append(ctx)
                y_list.append(float(r.get("position", 20.0)))

            # Add this round to the cumulative buffer for the next race's sequence
            cumulative_rounds.append({"round": rnd, "name": race["name"], "results": rnd_results})

    if not X_seq_list:
        raise RuntimeError("No training data could be built - check that historical data is cached.")

    X_seq = np.array(X_seq_list, dtype=np.float32)   # (N, SEQ_LENGTH, N_SEQ_FEATURES)
    X_ctx = np.array(X_ctx_list, dtype=np.float32)   # (N, CONTEXT_DIM)
    y     = np.array(y_list,     dtype=np.float32)   # (N,)

    if verbose:
        print(f"  [LSTM dataset] Built {len(y)} samples from {len(training_seasons)} seasons.")

    return X_seq, X_ctx, y


#  
# NORMALIZATION
#  

class TemporalScaler:
    # Simple per-feature min-max scaler for sequence and context arrays.
    # Stored alongside Keras model so inference is stateless.

    def __init__(self):
        self.seq_min: Optional[np.ndarray] = None
        self.seq_max: Optional[np.ndarray] = None
        self.ctx_min: Optional[np.ndarray] = None
        self.ctx_max: Optional[np.ndarray] = None
        self.y_min: float = 1.0
        self.y_max: float = 22.0

    def fit(self, X_seq: np.ndarray, X_ctx: np.ndarray, y: np.ndarray):
        # Flatten time dimension for per-feature stats
        flat = X_seq.reshape(-1, X_seq.shape[-1])
        self.seq_min = flat.min(axis=0)
        self.seq_max = flat.max(axis=0)
        self.ctx_min = X_ctx.min(axis=0)
        self.ctx_max = X_ctx.max(axis=0)
        self.y_min = float(y.min())
        self.y_max = float(y.max())

    def _scale(self, arr: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
        denom = mx - mn
        denom = np.where(denom == 0, 1.0, denom)
        return (arr - mn) / denom

    def transform_seq(self, X_seq: np.ndarray) -> np.ndarray:
        return self._scale(X_seq, self.seq_min, self.seq_max)

    def transform_ctx(self, X_ctx: np.ndarray) -> np.ndarray:
        return self._scale(X_ctx, self.ctx_min, self.ctx_max)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_min) / max(self.y_max - self.y_min, 1.0)

    def inverse_y(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * (self.y_max - self.y_min) + self.y_min


#  
# PUBLIC API
#  

class TemporalFormModel:
    # Wraps Keras dual-branch LSTM model with train / predict / persist methods.

    def __init__(self):
        self.model: Optional[keras.Model] = None
        self.scaler: Optional[TemporalScaler] = None
        self._trained: bool = False

    #   Persistence  

    def save(self):
        # Plan 9-M2: atomic saves — a reader mid-write previously got a
        # half-written .keras/.pkl and triggered spurious retrains.
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_model = Path(str(MODEL_PATH) + ".tmp")
        tmp_scaler = Path(str(SCALER_PATH) + ".tmp")
        self.model.save(str(tmp_model))
        with open(tmp_scaler, "wb") as f:
            pickle.dump(self.scaler, f, protocol=5)
        os.replace(str(tmp_model), str(MODEL_PATH))
        os.replace(str(tmp_scaler), str(SCALER_PATH))
        print(f"  [temporal_model] Saved to {MODEL_PATH}")

    def load(self) -> bool:
        # Return True if a saved model was successfully loaded.
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            return False
        try:
            self.model = keras.models.load_model(str(MODEL_PATH))
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
            self._trained = True
            print(f"  [temporal_model] Loaded cached model from {MODEL_PATH}")
            return True
        except Exception as exc:
            print(f"  [temporal_model] Could not load cache ({exc}), will retrain.")
            return False

    #   Training  

    def train(self, verbose: bool = True, force: bool = False):
        # Train dual-branch LSTM on historical race data.
        # Results are cached to disk.
        if not force and self.load():
            return

        if verbose:
            print("  [temporal_model] Building training dataset (this may take a few minutes)...")

        X_seq, X_ctx, y = build_training_dataset(HISTORICAL_SEASONS, verbose=verbose)

        self.scaler = TemporalScaler()
        self.scaler.fit(X_seq, X_ctx, y)

        X_seq_n = self.scaler.transform_seq(X_seq)
        X_ctx_n = self.scaler.transform_ctx(X_ctx)
        y_n     = self.scaler.transform_y(y)

        self.model = build_temporal_model()

        callbacks = [
            keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5, verbose=0),
        ]

        if verbose:
            print(f"  [temporal_model] Training on {len(y)} samples for up to {EPOCHS} epochs...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(
                [X_seq_n, X_ctx_n], y_n,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=0.15,
                callbacks=callbacks,
                verbose=1 if verbose else 0,
            )

        self._trained = True
        self.save()

    def load_or_train(self, verbose: bool = True, force: bool = False):
        # Convenience method: load from cache or train if not available.
        self.train(verbose=verbose, force=force)

    #   Inference  

    def predict_driver_momentum(
        self,
        driver_name: str,
        recent_race_results: list[dict],
        tire_data_by_round: dict[int, dict],
        context_vec: Optional[np.ndarray] = None,
    ) -> float:
        # Predict driver expected finishing position delta vs. field median.
        # Returns float: predicted finish position (1=best, 22=last).
        if not self._trained or self.model is None:
            return 11.0   # neutral fallback

        seq = _extract_sequence_for_driver(
            driver_name, recent_race_results, tire_data_by_round, SEQ_LENGTH
        )

        if context_vec is None:
            context_vec = np.array([11.0, 35.0, 0.0, 0.40, 0.0], dtype=np.float32)

        seq_n = self.scaler.transform_seq(seq[np.newaxis])       # (1, SEQ, FEAT)
        ctx_n = self.scaler.transform_ctx(context_vec[np.newaxis])  # (1, CTX)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_norm = self.model.predict([seq_n, ctx_n], verbose=0)[0, 0]

        return float(np.clip(self.scaler.inverse_y(np.array([y_norm]))[0], 1.0, 22.0))

    def get_momentum_tensor(
        self,
        driver_name: str,
        recent_race_results: list[dict],
        tire_data_by_round: dict[int, dict],
        context_vec: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Return the raw 32-dimensional driver_momentum tensor from temporal branch.
        if not self._trained or self.model is None:
            return np.zeros(32, dtype=np.float32)

        seq = _extract_sequence_for_driver(
            driver_name, recent_race_results, tire_data_by_round, SEQ_LENGTH
        )
        if context_vec is None:
            # 5 features matching CONTEXT_DIM: [quali_pos, track_temp, rain_enc, sc_prob, ctor_elo_z]
            context_vec = np.array([11.0, 35.0, 0.0, 0.40, 0.0], dtype=np.float32)

        seq_n = self.scaler.transform_seq(seq[np.newaxis])
        ctx_n = self.scaler.transform_ctx(context_vec[np.newaxis])

        # Build a sub-model that outputs the history_dense layer (32-dim)
        tensor_model = keras.Model(
            inputs=self.model.inputs,
            outputs=self.model.get_layer("history_dense").output,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tensor = tensor_model.predict([seq_n, ctx_n], verbose=0)[0]

        return tensor.astype(np.float32)


#  
# SINGLETON ACCESSOR
#  
_temporal_model_instance: Optional[TemporalFormModel] = None


_model_init_lock = threading.Lock()


def get_temporal_model(verbose: bool = False, force: bool = False) -> TemporalFormModel:
    # Singleton with double-checked locking (plan 9-M2): concurrent first
    # calls previously raced into DUPLICATE TF trainings.
    global _temporal_model_instance
    if _temporal_model_instance is None:
        with _model_init_lock:
            if _temporal_model_instance is None:
                m = TemporalFormModel()
                m.load_or_train(verbose=verbose, force=force)
                _temporal_model_instance = m   # publish only after fully fitted
    return _temporal_model_instance
