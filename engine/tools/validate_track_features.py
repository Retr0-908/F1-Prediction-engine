#!/usr/bin/env python3
"""Validate all track feature JSON files and the loader module."""

from engine.core.track_features_loader import load_all_track_features, get_validation_errors

def main():
    print("Loading all track features...")
    data = load_all_track_features()
    print(f"Loaded {len(data)} circuits")

    # Check all 24 circuits
    expected_keys = [
        "australia", "china", "japan", "miami", "canada", "monaco",
        "spain", "austria", "britain", "belgium", "hungary", "netherlands",
        "italy", "madring", "azerbaijan", "singapore", "usa", "mexico",
        "brazil", "las_vegas", "qatar", "abu_dhabi", "bahrain", "saudi_arabia",
    ]

    missing = [k for k in expected_keys if k not in data]
    if missing:
        print(f"MISSING circuits: {missing}")
    else:
        print("All 24 circuits present")

    # Validate ranges
    for key, features in data.items():
        assert 1 <= features["downforce_level"] <= 5, f"{key}: downforce out of range"
        assert 1 <= features["overtaking_difficulty"] <= 5, f"{key}: overtake out of range"
        assert 1 <= features["tire_degradation"] <= 5, f"{key}: tire_deg out of range"
        assert 0 <= features["sc_probability"] <= 1, f"{key}: sc_prob out of range"
        assert 0 <= features["vsc_probability"] <= 1, f"{key}: vsc_prob out of range"
        assert 0 <= features["first_lap_incident_risk"] <= 1, f"{key}: fl_risk out of range"
        assert 0 <= features["overtake_mode_efficiency"] <= 1, f"{key}: sm_eff out of range"
    print("All range assertions passed: OK")

    errors = get_validation_errors()
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("No validation errors")

    # Check loader integration with config.py
    from engine.core.config import CIRCUITS, get_enriched_circuit_config
    for gp_name in CIRCUITS:
        enriched = get_enriched_circuit_config(gp_name)
        if "downforce_level" not in enriched:
            print(f"WARN: {gp_name} not enriched with track features")
        else:
            print(f" {gp_name}: enriched OK (downforce_level={enriched['downforce_level']})")

if __name__ == "__main__":
    main()
