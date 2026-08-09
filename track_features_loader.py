"""
track_features_loader.py - Per-circuit track feature loader with caching and validation.
Loads JSON files from track_features/ directory, validates schema, provides lookup API.
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any

_TRACK_DIR = Path(__file__).parent / "track_features"

# Required keys with expected types for validation
_REQUIRED_KEYS = {
    "circuit_key": str,
    "circuit_name": str,
    "grand_prix": str,
    "city": str,
    "country": str,
    "circuit_type": str,
    "circuit_length_km": (int, float),
    "num_laps": int,
    "num_turns": int,
    "direction": str,
    "altitude_m": (int, float),
    "longest_straight_m": (int, float),
    "track_width_m": (int, float),
    "sm_zones": int,
    "downforce_level": (int, float),
    "overtaking_difficulty": (int, float),
    "tire_degradation": (int, float),
    "sc_probability": (int, float),
    "vsc_probability": (int, float),
    "first_lap_incident_risk": (int, float),
    "quali_importance": (int, float),
    "weather_variability": (int, float),
    "pit_time_loss_s": (int, float),
    "deg_compound_delta": (int, float),
    "avg_speed_kmh": (int, float),
    "elevation_change_m": (int, float),
    "overtake_mode_efficiency": (int, float),
}

# In-memory cache: circuit_key -> dict
_cache: Dict[str, Dict[str, Any]] = {}
_validation_errors: list = []


def _validate(data: dict, filepath: str) -> bool:
    """Validate that a loaded JSON has all required keys with correct types."""
    errors = []
    for key, expected_type in _REQUIRED_KEYS.items():
        if key not in data:
            errors.append(f"Missing key: {key}")
        elif not isinstance(data[key], expected_type):
            errors.append(
                f"Key '{key}': expected {expected_type}, got {type(data[key]).__name__}"
            )
    # Range checks for critical numeric fields
    if "downforce_level" in data and not (1 <= data["downforce_level"] <= 5):
        errors.append("downforce_level must be 1-5")
    if "overtaking_difficulty" in data and not (1 <= data["overtaking_difficulty"] <= 5):
        errors.append("overtaking_difficulty must be 1-5")
    if "tire_degradation" in data and not (1 <= data["tire_degradation"] <= 5):
        errors.append("tire_degradation must be 1-5")
    if "sc_probability" in data and not (0 <= data["sc_probability"] <= 1):
        errors.append("sc_probability must be 0.0-1.0")
    if errors:
        _validation_errors.append(f"{filepath}: {'; '.join(errors)}")
        return False
    return True


def load_track_features(circuit_key: str) -> Optional[Dict[str, Any]]:
    """Load track features for a given circuit_key. Returns None if not found."""
    if circuit_key in _cache:
        return _cache[circuit_key]
    filepath = _TRACK_DIR / f"{circuit_key}.json"
    if not filepath.exists():
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        _validation_errors.append(f"{filepath}: {e}")
        return None
    if not _validate(data, str(filepath)):
        return None
    _cache[circuit_key] = data
    return data


def load_all_track_features() -> Dict[str, Dict[str, Any]]:
    """Load all track features from the track_features/ directory."""
    if _cache and len(_cache) >= 24:
        return dict(_cache)
    for json_file in sorted(_TRACK_DIR.glob("*.json")):
        circuit_key = json_file.stem
        if circuit_key not in _cache:
            load_track_features(circuit_key)
    return dict(_cache)


def get_track_feature(circuit_key: str, feature_name: str, default=None):
    """Get a single feature value for a circuit. Returns default if not found."""
    data = load_track_features(circuit_key)
    if data is None:
        return default
    return data.get(feature_name, default)


def get_validation_errors() -> list:
    """Return any validation errors from loading."""
    return list(_validation_errors)


def clear_cache():
    """Clear the in-memory cache."""
    _cache.clear()
    _validation_errors.clear()
