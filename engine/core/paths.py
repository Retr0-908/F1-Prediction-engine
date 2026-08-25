"""
paths.py — Single source of truth for every filesystem location the engine uses.

Anchored at PROJECT_ROOT so modules can live anywhere in the package tree while
runtime data (cache/, logs/, output/) stays at the project root.
"""
from pathlib import Path

# While flat: parents[0] = project root.
# After moving into engine/core/: parents[2] = project root.
# Computed defensively so BOTH layouts resolve identically:
_here = Path(__file__).resolve()
if _here.parent.name == "core":
    PROJECT_ROOT = _here.parents[2]
else:
    PROJECT_ROOT = _here.parent

CACHE_DIR         = PROJECT_ROOT / "cache"
API_CACHE_DIR     = CACHE_DIR / "api"
ELO_CACHE_DIR     = CACHE_DIR / "elo"
MODEL_CACHE_DIR   = CACHE_DIR / "models"
WEATHER_CACHE_DIR = CACHE_DIR / "weather"
FASTF1_CACHE_DIR  = str(CACHE_DIR / "fastf1")   # fastf1.Cache expects str

LOGS_DIR   = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"

UI_DIR             = PROJECT_ROOT / "ui"
TRACK_FEATURES_DIR = PROJECT_ROOT / "track_features"

MY_TEAM_PATH           = CACHE_DIR / "my_team.json"
CHIP_STATE_PATH        = CACHE_DIR / "chip_state.json"
PRICE_HISTORY_PATH     = CACHE_DIR / "price_history.json"
OWNERSHIP_HISTORY_PATH = CACHE_DIR / "ownership_history.json"
GRID_CACHE_PATH        = OUTPUT_DIR / "grid_cache.json"

ACCURACY_LOG       = LOGS_DIR / "accuracy_log.json"
BIAS_CORRECTIONS   = LOGS_DIR / "bias_corrections.json"
WEIGHT_ADJUSTMENTS = LOGS_DIR / "weight_adjustments.json"
LOG_FILE           = LOGS_DIR / "f1_predictor.log"


def ensure_dirs() -> None:
    """Create all runtime directories (idempotent)."""
    for d in (API_CACHE_DIR, ELO_CACHE_DIR, MODEL_CACHE_DIR,
              WEATHER_CACHE_DIR, CACHE_DIR / "fastf1",
              LOGS_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
