"""
config.py — Central configuration for F1 Predictor (2026 Season)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Filesystem locations come from paths.py (single source of truth)
from engine.core.paths import FASTF1_CACHE_DIR, MY_TEAM_PATH  # noqa: E402,F401  (re-exported)

# ─────────────────────────────────────────────
# API CONFIGURATION
# NOTE: weather is served by Open-Meteo (no key needed) — the old OpenWeatherMap
# constants were removed. See weather.py.
# ─────────────────────────────────────────────
F1_FANTASY_COOKIE = os.getenv("F1_FANTASY_COOKIE", "")

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
OPENF1_BASE  = "https://api.openf1.org/v1"

F1_FANTASY_API   = "https://fantasy.formula1.com/feeds/drivers/drivers_{season}.json"
F1_FANTASY_STATS = "https://fantasy.formula1.com/feeds/stats/stats_{season}_{round}.json"

# ─────────────────────────────────────────────
# SEASON SETTINGS
# ─────────────────────────────────────────────
CURRENT_SEASON    = 2026
# HISTORICAL_SEASONS: years used by Glicko-2, multiseason form, and circuit history.
# 2026 is intentionally included so that:
#   - Glicko-2 ratings incorporate completed 2026 race results (current ratings)
#   - compute_multiseason_driver_form uses 2026 results for live context features
# The ML training loop in predictor.py explicitly skips year >= CURRENT_SEASON
# to prevent any data leakage from the season being predicted.
HISTORICAL_SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
# FASTF1_CACHE_DIR and MY_TEAM_PATH are imported from paths.py (top of file).

# ─────────────────────────────────────────────
# REGULATION CHANGE YEARS & SEASON WEIGHTS
# ─────────────────────────────────────────────
# Major regulatory reset years — historical data before these is
# structurally less predictive for the new regulation era.
REGULATION_CHANGE_YEARS = {2022, 2026}

# How much each historical season contributes to multi-season form.
# Pre-2022 = turbo-hybrid era (different aero/PU), very heavily discounted.
# Pre-2026 data is partially discounted because the new hybrid split /
# active aero fundamentally changes car characteristics.
SEASON_WEIGHTS = {
    2019: 0.05,   # turbo-hybrid era — included only for sample-size benefit
    2020: 0.05,   # COVID season — limited calendar, anomalous results
    2021: 0.05,   # Max/Lewis title fight era — some relevance for driver profiles
    2022: 0.18,   # ground-effect era start — some relevance but different PU concept
    2023: 0.30,   # same regs, moderate weight
    2024: 0.50,   # same regs, higher weight
    2025: 0.75,   # immediately prior season, most comparable
    2026: 1.00,   # current season = full weight
}

# ─────────────────────────────────────────────
# SAFETY CAR & VIRTUAL SAFETY CAR PROBABILITIES
# per circuit_key — derived from 2019–2025 historical SC/VSC deployment data
# ─────────────────────────────────────────────
SC_PROBABILITY = {
    "australia":    0.55,   # Albert Park — often incidents at Turn 1/3
    "china":        0.40,
    "japan":        0.35,   # Suzuka — few incidents, high grip
    "bahrain":      0.35,   # Wide track, easy to avoid
    "saudi_arabia": 0.65,   # Jeddah street circuit — frequent SC
    "miami":        0.45,   # Some tight sections, Turn 1 bunching
    "imola":        0.45,   # Narrow track, limited run-off
    "monaco":       0.70,   # Highest in calendar — guardrails everywhere
    "spain":        0.25,   # Barcelona — spacious, few incidents
    "madring":      0.45,   # Madrid street circuit — estimate pending real data
    "canada":       0.55,   # Wall of Champions, frequent incidents
    "britain":      0.40,   # Silverstone — high speed but safe
    "austria":      0.35,   # Red Bull Ring — some first-lap incidents
    "belgium":      0.45,   # Spa — weather + high speed exits
    "hungary":      0.30,   # Hungaroring — few incidents
    "netherlands":  0.35,   # Zandvoort — banking protects
    "italy":        0.40,   # Monza — slipstream battles can cause contact
    "azerbaijan":   0.65,   # Baku — wall exits, very frequent
    "singapore":    0.75,   # Highest alongside Monaco
    "usa":          0.45,   # COTA — Turn 1 drama
    "mexico":       0.40,
    "brazil":       0.55,   # Interlagos — wet + tight
    "las_vegas":    0.50,   # Street circuit, drain covers historically problematic
    "qatar":        0.30,   # Lusail — low SC rate
    "abu_dhabi":    0.30,   # Yas Marina — good visibility, space
    # Historical venues (estimates from their era on the calendar)
    "portugal":     0.35,   # Portimão — wide, fast but few SCs
    "france":       0.25,   # Paul Ricard — huge run-offs
    "russia":       0.45,   # Sochi — long walls near Turn 2-3
    "turkey":       0.40,   # Istanbul Park — Turn 1 multi-car risk
    "malaysia":     0.50,   # Sepang — monsoon downpours, evening storms
}

VSC_PROBABILITY = {
    "australia":    0.40,
    "china":        0.30,
    "japan":        0.25,
    "bahrain":      0.30,
    "saudi_arabia": 0.45,
    "miami":        0.35,
    "imola":        0.40,
    "monaco":       0.55,
    "spain":        0.20,
    "madring":      0.35,   # Madrid street circuit — estimate pending real data
    "canada":       0.40,
    "britain":      0.35,
    "austria":      0.25,
    "belgium":      0.40,
    "hungary":      0.25,
    "netherlands":  0.25,
    "italy":        0.35,
    "azerbaijan":   0.50,
    "singapore":    0.55,
    "usa":          0.35,
    "mexico":       0.30,
    "brazil":       0.45,
    "las_vegas":    0.35,
    "qatar":        0.25,
    "abu_dhabi":    0.25,
    "portugal":     0.25,
    "france":       0.20,
    "russia":       0.30,
    "turkey":       0.30,
    "malaysia":     0.40,
}

# ─────────────────────────────────────────────
# F1 FANTASY — BUDGET & SQUAD RULES (2026)
# ─────────────────────────────────────────────
FANTASY_BUDGET           = 100.0   # $100M total
FANTASY_NUM_DRIVERS      = 5
FANTASY_NUM_CONSTRUCTORS = 2
FANTASY_MAX_PER_TEAM     = 2

# ─────────────────────────────────────────────
# F1 FANTASY — SCORING SYSTEM 2026
# ─────────────────────────────────────────────
RACE_POSITION_POINTS = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8,  7: 6,  8: 4,  9: 2,  10: 1,
}
QUALI_POSITION_POINTS = {
    1: 5.5, 2: 4.5, 3: 4, 4: 3.5, 5: 3,
    6: 2.5, 7: 2,   8: 1.5, 9: 1, 10: 0.5,
}
QUALI_Q2_BONUS    = 1
QUALI_Q3_BONUS    = 1
POLE_BONUS        = 0.5

# Sprint
SPRINT_POLE_BONUS = 3
SPRINT_SQ3_BONUS  = 1
SPRINT_RACE_POINTS = {
    1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1,
}
SPRINT_QUALI_POINTS = {
    1: 5, 2: 4, 3: 3, 4: 2, 5: 1,
    6: 0.5, 7: 0.5, 8: 0, 9: 0, 10: 0,
}

POSITIONS_GAINED_PER = 2
POSITIONS_LOST_PER   = -2
DNF_PENALTY          = -15
DSQ_PENALTY          = -20
NO_TIME_SET_PENALTY  = -5
FASTEST_LAP_BONUS    = 5
DRIVER_OF_DAY_BONUS  = 10

CONSTRUCTOR_DSQ_PENALTY   = -20
PIT_STOP_POINTS = {
    2.0: 20, 2.5: 15, 3.0: 10, 3.5: 5, 4.0: 2,
}
FASTEST_PIT_STOP_BONUS = 5
WORLD_RECORD_PIT_BONUS = 15

AVG_PIT_STOP_TEAM_POINTS = {
    "Red Bull":     12, "McLaren":    14, "Mercedes":  11, "Ferrari":    10,
    "Aston Martin":  8, "Alpine":      7, "Williams":   9, "Racing Bulls": 8,
    "Audi":          7, "Haas":        7, "Cadillac":    6,
}

# ─────────────────────────────────────────────
# 2026 CIRCUIT DATABASE (24 races + sprint rounds)
# ─────────────────────────────────────────────
CIRCUITS = {
    "Australian Grand Prix": {
        "key": "australia", "city": "Melbourne", "country": "AU",
        "lat": -37.8497, "lon": 144.9680,
        "track_type": "hybrid", "overtaking": "low",
        "power_unit": "medium", "downforce": "high", "round": 1,
    },
    "Chinese Grand Prix": {
        "key": "china", "city": "Shanghai", "country": "CN",
        "lat": 31.3389, "lon": 121.2201,
        "track_type": "permanent", "overtaking": "medium",
        "power_unit": "medium", "downforce": "medium", "round": 2,
    },
    "Japanese Grand Prix": {
        "key": "japan", "city": "Suzuka", "country": "JP",
        "lat": 34.8431, "lon": 136.5407,
        "track_type": "permanent", "overtaking": "low",
        "power_unit": "medium", "downforce": "high", "round": 3,
    },
    "Miami Grand Prix": {
        "key": "miami", "city": "Miami", "country": "US",
        "lat": 25.9581, "lon": -80.2389,
        "track_type": "street", "overtaking": "medium",
        "power_unit": "medium", "downforce": "medium", "round": 4,
    },
    "Canadian Grand Prix": {
        "key": "canada", "city": "Montreal", "country": "CA",
        "lat": 45.5000, "lon": -73.5228,
        "track_type": "hybrid", "overtaking": "high",
        "power_unit": "high", "downforce": "low", "round": 5,
    },
    "Monaco Grand Prix": {
        "key": "monaco", "city": "Monte Carlo", "country": "MC",
        "lat": 43.7347, "lon": 7.4206,
        "track_type": "street", "overtaking": "very_low",
        "power_unit": "low", "downforce": "very_high", "round": 6,
    },
    "Barcelona Grand Prix": {
        "key": "spain", "city": "Barcelona", "country": "ES",
        "lat": 41.5700, "lon": 2.2610,
        "track_type": "permanent", "overtaking": "medium",
        "power_unit": "medium", "downforce": "medium", "round": 7,
    },
    "Austrian Grand Prix": {
        "key": "austria", "city": "Spielberg", "country": "AT",
        "lat": 47.2197, "lon": 14.7647,
        "track_type": "permanent", "overtaking": "high",
        "power_unit": "medium", "downforce": "medium", "round": 8,
    },
    "British Grand Prix": {
        "key": "britain", "city": "Silverstone", "country": "GB",
        "lat": 52.0786, "lon": -1.0169,
        "track_type": "permanent", "overtaking": "medium",
        "power_unit": "medium", "downforce": "high", "round": 9,
    },
    "Belgian Grand Prix": {
        "key": "belgium", "city": "Spa-Francorchamps", "country": "BE",
        "lat": 50.4372, "lon": 5.9714,
        "track_type": "permanent", "overtaking": "high",
        "power_unit": "high", "downforce": "low", "round": 10,
    },
    "Hungarian Grand Prix": {
        "key": "hungary", "city": "Budapest", "country": "HU",
        "lat": 47.5789, "lon": 19.2486,
        "track_type": "permanent", "overtaking": "low",
        "power_unit": "low", "downforce": "very_high", "round": 11,
    },
    "Dutch Grand Prix": {
        "key": "netherlands", "city": "Zandvoort", "country": "NL",
        "lat": 52.3888, "lon": 4.5431,
        "track_type": "permanent", "overtaking": "very_low",
        "power_unit": "medium", "downforce": "high", "round": 12,
    },
    "Italian Grand Prix": {
        "key": "italy", "city": "Monza", "country": "IT",
        "lat": 45.6156, "lon": 9.2811,
        "track_type": "permanent", "overtaking": "high",
        "power_unit": "very_high", "downforce": "very_low", "round": 13,
    },
    "Spanish Grand Prix": {
        "key": "madring", "city": "Madrid", "country": "ES",
        "lat": 40.4168, "lon": -3.7038,
        "track_type": "street", "overtaking": "medium",
        "power_unit": "medium", "downforce": "medium", "round": 14,
    },
    "Azerbaijan Grand Prix": {
        "key": "azerbaijan", "city": "Baku", "country": "AZ",
        "lat": 40.3725, "lon": 49.8533,
        "track_type": "street", "overtaking": "high",
        "power_unit": "high", "downforce": "low", "round": 15,
    },
    # Plan 9-C3: the live 2026 calendar inserts a Sepang round under the
    # Jolpica name "Bahrain Grand Prix in Malaysia" — the exact-name entry
    # below wins over any substring match against Sakhir. R17+ shifted +1.
    "Bahrain Grand Prix in Malaysia": {
        "key": "malaysia", "city": "Kuala Lumpur", "country": "MY",
        "lat": 2.7608, "lon": 101.7381,
        "track_type": "permanent", "overtaking": "high",
        "power_unit": "high", "downforce": "medium", "round": 16,
    },
    "Singapore Grand Prix": {
        "key": "singapore", "city": "Singapore", "country": "SG",
        "lat": 1.2914, "lon": 103.8640,
        "track_type": "street", "overtaking": "very_low",
        "power_unit": "low", "downforce": "very_high", "round": 17,
    },
    "United States Grand Prix": {
        "key": "usa", "city": "Austin", "country": "US",
        "lat": 30.1328, "lon": -97.6411,
        "track_type": "permanent", "overtaking": "medium",
        "power_unit": "medium", "downforce": "medium", "round": 18,
    },
    "Mexico City Grand Prix": {
        "key": "mexico", "city": "Mexico City", "country": "MX",
        "lat": 19.4042, "lon": -99.0907,
        "track_type": "permanent", "overtaking": "medium",
        "power_unit": "high", "downforce": "low", "round": 19,
    },
    "Brazilian Grand Prix": {
        "key": "brazil", "city": "São Paulo", "country": "BR",
        "lat": -23.7036, "lon": -46.6997,
        "track_type": "permanent", "overtaking": "high",
        "power_unit": "medium", "downforce": "medium", "round": 20,
    },
    "Las Vegas Grand Prix": {
        "key": "las_vegas", "city": "Las Vegas", "country": "US",
        "lat": 36.1699, "lon": -115.1398,
        "track_type": "street", "overtaking": "high",
        "power_unit": "very_high", "downforce": "very_low", "round": 21,
    },
    "Qatar Grand Prix": {
        "key": "qatar", "city": "Lusail", "country": "QA",
        "lat": 25.4700, "lon": 51.4538,
        "track_type": "permanent", "overtaking": "medium",
        "power_unit": "medium", "downforce": "medium", "round": 22,
    },
    "Abu Dhabi Grand Prix": {
        "key": "abu_dhabi", "city": "Abu Dhabi", "country": "AE",
        "lat": 24.4672, "lon": 54.6031,
        "track_type": "permanent", "overtaking": "low",
        "power_unit": "medium", "downforce": "medium", "round": 23,
    },
    "Bahrain Grand Prix": {
        "key": "bahrain", "city": "Sakhir", "country": "BH",
        "lat": 26.0325, "lon": 50.5106,
        "track_type": "permanent", "overtaking": "high",
        "power_unit": "high", "downforce": "medium", "round": 0,
    },

    # ── HISTORICAL circuits (no 2026 round) ─────────────────────────────
    # Needed so 2021-era training races get real circuit features instead of
    # matcher warnings + empty configs. "round": 0 = not on the current calendar.
    "Portuguese Grand Prix": {
        "key": "portugal", "city": "Portimão", "country": "PT",
        "lat": 37.2270, "lon": -8.6268,
        "track_type": "permanent", "overtaking": "medium",
        "power_unit": "high", "downforce": "medium", "round": 0,
    },
    "French Grand Prix": {
        "key": "france", "city": "Le Castellet", "country": "FR",
        "lat": 43.2504, "lon": 5.7916,
        "track_type": "permanent", "overtaking": "low",
        "power_unit": "high", "downforce": "high", "round": 0,
    },
    "Russian Grand Prix": {
        "key": "russia", "city": "Sochi", "country": "RU",
        "lat": 43.4057, "lon": 39.9578,
        "track_type": "street", "overtaking": "medium",
        "power_unit": "medium", "downforce": "low", "round": 0,
    },
    "Turkish Grand Prix": {
        "key": "turkey", "city": "Istanbul", "country": "TR",
        "lat": 40.9517, "lon": 29.4053,
        "track_type": "permanent", "overtaking": "medium",
        "power_unit": "high", "downforce": "high", "round": 0,
    },
    "Saudi Arabian Grand Prix": {
        "key": "saudi_arabia", "city": "Jeddah", "country": "SA",
        "lat": 21.6318, "lon": 39.1044,
        "track_type": "street", "overtaking": "low",
        "power_unit": "high", "downforce": "low", "round": 0,
    },
    "Emilia Romagna Grand Prix": {
        "key": "imola", "city": "Imola", "country": "IT",
        "lat": 44.3439, "lon": 11.7167,
        "track_type": "permanent", "overtaking": "low",
        "power_unit": "medium", "downforce": "high", "round": 0,
    },
}

# Sprint weekends 2026 (by GP name) — LEGACY reference only.
# The authoritative source is the schedule's `sprint_date` field via
# data_fetcher.is_sprint_weekend(); do NOT add new consumers of this constant.
SPRINT_ROUNDS = frozenset([
    "Chinese Grand Prix",
    "Miami Grand Prix",
    "Canadian Grand Prix",
    "British Grand Prix",
    "Dutch Grand Prix",
    "Singapore Grand Prix"
])

# ─────────────────────────────────────────────
# STATIC ROSTER SEEDS (fallback only)
# The live lineup is AUTO-DETECTED each run from Jolpica championship standings
# (data_fetcher.get_season_roster) and cross-checked against the fantasy price
# scrape. These static tables are used ONLY:
#   - before round 1 of a new season (standings don't exist yet), or
#   - fully offline.
# Update DRIVER_TEAMS_<year> once when a season's seats are announced, then
# forget about it — everything else is dynamic.
# ─────────────────────────────────────────────
# 2026 DRIVER REGISTRY (11 teams, 22 drivers)
# ─────────────────────────────────────────────
DRIVER_TEAMS_2026 = {
    # CURATED 22-seat seed (exactly 2 per team). The live pipeline's field
    # authority is the fantasy entry list; standings-only extras (e.g. a
    # driver who lost a seat mid-season but remains in season standings)
    # are EXCLUDED as phantoms per plan I7b. Do NOT bulk-sync this table
    # from the API — edit seats deliberately.
    "Max Verstappen":    "Red Bull",
    "Liam Lawson":       "Red Bull",
    "Isack Hadjar":      "Racing Bulls",
    "Arvid Lindblad":    "Racing Bulls",
    "Lando Norris":      "McLaren",
    "Oscar Piastri":     "McLaren",
    "Charles Leclerc":   "Ferrari",
    "Lewis Hamilton":    "Ferrari",
    "George Russell":    "Mercedes",
    "Kimi Antonelli":    "Mercedes",
    "Fernando Alonso":   "Aston Martin",
    "Lance Stroll":      "Aston Martin",
    "Pierre Gasly":      "Alpine",
    "Franco Colapinto":  "Alpine",
    "Carlos Sainz":      "Williams",
    "Alexander Albon":   "Williams",
    "Nico Hulkenberg":   "Audi",
    "Gabriel Bortoleto": "Audi",
    "Esteban Ocon":      "Haas",
    "Oliver Bearman":    "Haas",
    "Sergio Perez":      "Cadillac",
    "Valtteri Bottas":   "Cadillac",
}

# ─────────────────────────────────────────────
# 2025 DRIVER REGISTRY (real lineup — NOT an alias for 2026)
# Mid-season moves resolved to the majority-season seat holder:
#   Tsunoda ↔ Lawson swap (Red Bull/Racing Bulls, from R3)
#   Doohan → Colapinto at Alpine (from ~R9)
# ─────────────────────────────────────────────
DRIVER_TEAMS_2025 = {
    "Max Verstappen":    "Red Bull",
    "Yuki Tsunoda":      "Red Bull",
    "Lando Norris":      "McLaren",
    "Oscar Piastri":     "McLaren",
    "Charles Leclerc":   "Ferrari",
    "Lewis Hamilton":    "Ferrari",
    "George Russell":    "Mercedes",
    "Kimi Antonelli":    "Mercedes",
    "Fernando Alonso":   "Aston Martin",
    "Lance Stroll":      "Aston Martin",
    "Pierre Gasly":      "Alpine",
    "Franco Colapinto":  "Alpine",
    "Liam Lawson":       "Racing Bulls",
    "Isack Hadjar":      "Racing Bulls",
    "Carlos Sainz":      "Williams",
    "Alexander Albon":   "Williams",
    "Nico Hulkenberg":   "Audi",       # raced as Sauber in 2025; canonicalized
    "Gabriel Bortoleto": "Audi",
    "Esteban Ocon":      "Haas",
    "Oliver Bearman":    "Haas",
    # Jack Doohan also raced rounds 1–8 for Alpine (substituted by Colapinto)
}

CONSTRUCTORS_2026 = [
    "Red Bull", "McLaren", "Ferrari", "Mercedes",
    "Aston Martin", "Alpine", "Williams", "Racing Bulls",
    "Audi", "Haas", "Cadillac",
]
CONSTRUCTORS_2025 = [
    "Red Bull", "McLaren", "Ferrari", "Mercedes",
    "Aston Martin", "Alpine", "Williams", "Racing Bulls",
    "Audi", "Haas",
]

DRIVER_SHORT_2026 = {
    "VER": "Max Verstappen",    "TSU": "Yuki Tsunoda",
    "HAD": "Isack Hadjar",      "LIN": "Arvid Lindblad",
    "NOR": "Lando Norris",      "PIA": "Oscar Piastri",
    "LEC": "Charles Leclerc",   "HAM": "Lewis Hamilton",
    "RUS": "George Russell",    "ANT": "Kimi Antonelli",
    "ALO": "Fernando Alonso",   "STR": "Lance Stroll",
    "GAS": "Pierre Gasly",      "COL": "Franco Colapinto",
    "LAW": "Liam Lawson",
    "SAI": "Carlos Sainz",      "ALB": "Alexander Albon",
    "HUL": "Nico Hulkenberg",   "BOR": "Gabriel Bortoleto",
    "OCO": "Esteban Ocon",      "BEA": "Oliver Bearman",
    "PER": "Sergio Perez",      "BOT": "Valtteri Bottas",
}
DRIVER_SHORT_2025 = {
    "VER": "Max Verstappen",    "TSU": "Yuki Tsunoda",
    "NOR": "Lando Norris",      "PIA": "Oscar Piastri",
    "LEC": "Charles Leclerc",   "HAM": "Lewis Hamilton",
    "RUS": "George Russell",    "ANT": "Kimi Antonelli",
    "ALO": "Fernando Alonso",   "STR": "Lance Stroll",
    "GAS": "Pierre Gasly",      "COL": "Franco Colapinto",
    "LAW": "Liam Lawson",       "HAD": "Isack Hadjar",
    "SAI": "Carlos Sainz",      "ALB": "Alexander Albon",
    "HUL": "Nico Hulkenberg",   "BOR": "Gabriel Bortoleto",
    "OCO": "Esteban Ocon",      "BEA": "Oliver Bearman",
    "DOO": "Jack Doohan",
}

# ─────────────────────────────────────────────
# ENCODING MAPS FOR ML FEATURES
# ─────────────────────────────────────────────
TRACK_TYPE_ENC  = {"street": 0, "hybrid": 1, "permanent": 2}
OVERTAKING_ENC  = {"very_low": 0, "low": 1, "medium": 2, "high": 3, "very_high": 4}
POWER_UNIT_ENC  = {"very_low": 0, "low": 1, "medium": 2, "high": 3, "very_high": 4}
DOWNFORCE_ENC   = {"very_low": 0, "low": 1, "medium": 2, "high": 3, "very_high": 4}
CONDITION_ENC   = {"dry": 0, "overcast": 1, "mixed": 2, "wet": 3}

# ------------------------------------------------------------------
# TRACK FEATURE ENCODING MAPS (Phase 5)
# ------------------------------------------------------------------
CIRCUIT_TYPE_ENC     = {"street": 0, "hybrid": 1, "permanent": 2}
DOWNFORCE_LEVEL_ENC  = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}  # Already numeric 1-5
OVERTAKE_DIFF_ENC    = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}  # Already numeric 1-5
TIRE_DEG_ENC         = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}  # Already numeric 1-5
DIRECTION_ENC        = {"clockwise": 0, "counter-clockwise": 1, "figure-8": 2}
QUALI_IMPORTANCE_ENC = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}  # Already numeric 1-5
WEATHER_VAR_ENC      = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}  # Already numeric 1-5


# MY_TEAM_PATH defined in paths.py; imported above.

TEAM_COLORS = {    "Red Bull":     "#3671C6", "McLaren":    "#FF8000",
    "Ferrari":      "#E8002D", "Mercedes":   "#27F4D2",
    "Aston Martin": "#358C75", "Alpine":     "#FF87BC",
    "Williams":     "#64C4FF", "Racing Bulls":"#6692FF",
    "Audi":         "#BB0000", "Haas":       "#B6BABD",
    "Cadillac":     "#D6E4FF",
}


def get_enriched_circuit_config(gp_name: str) -> dict:
    """
    Return circuit config enriched with per-track JSON features.
    Falls back to the basic CIRCUITS dict if JSON is unavailable.
    Only adds new keys — does NOT replace existing CIRCUITS values.
    """
    try:
        from engine.core.track_features_loader import load_track_features
    except ImportError:
        return CIRCUITS.get(gp_name, {})
    base_cfg = CIRCUITS.get(gp_name, {})
    circuit_key = base_cfg.get("key", "")
    if not circuit_key:
        return base_cfg
    track_data = load_track_features(circuit_key)
    if track_data is None:
        return base_cfg
    # Merge: JSON data supplements (does NOT replace) base config
    enriched = dict(base_cfg)
    for key, value in track_data.items():
        if key not in enriched:  # Only add new keys
            enriched[key] = value
    return enriched
