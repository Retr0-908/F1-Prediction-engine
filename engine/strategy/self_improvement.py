import json
import os
from pathlib import Path

from engine.core.paths import LOGS_DIR, ACCURACY_LOG, BIAS_CORRECTIONS, WEIGHT_ADJUSTMENTS


# MIN_SAMPLES_CIRCUIT: require this many races at a circuit type before trusting
# a circuit-specific bias. Prevents a single bad/good race on a street circuit
# from permanently skewing all street-circuit predictions.
MIN_SAMPLES_CIRCUIT = 5
MIN_SAMPLES_OVERALL = 4  # require at least 4 logged races before any overall bias
# MAX_SINGLE_ERROR: clip each individual race error before EWMA. This stops a
# one-off catastrophic result (e.g. a race-ending mechanical DNF that still
# registered as a finish, or a safety-car-distorted result) from dominating.
MAX_SINGLE_ERROR = 4.0
# CONFIDENCE_POOL: Bayesian shrinkage denominator.  The final bias is multiplied
# by n / (n + CONFIDENCE_POOL) so it grows gradually from 0 as n increases
# rather than jumping to full strength the moment MIN_SAMPLES is reached.
# With CONFIDENCE_POOL=24 a driver at 4 races gets 4/(4+24)=0.14x weight;
# at 8 races: 0.25x; at 16 races: 0.40x; at 24 races: 0.50x.
CONFIDENCE_POOL = 24

def compute_bias_corrections(decay=0.85, last_n=10):
    """Compute EWMA bias corrections from the accuracy log.

    Parameters
    ----------
    decay : float
        EWMA recency factor (0 < decay < 1). The most recent race carries
        weight `decay` (default 0.85 ≈ recent errors weighted ~6x vs the
        previous one); lower values smooth more.
    last_n : int
        How many of the most recent races to include in the EWMA window.
        Default 10 ensures we need more races before large corrections kick in.
    """
    if not ACCURACY_LOG.exists():
        return
    
    try:
        with open(ACCURACY_LOG, "r", encoding="utf-8") as f:
            log = json.load(f)
    except Exception:
        return
    
    if not log:
        return
    
    # driver -> circuit_type -> list of errors
    # driver -> "overall" -> list of errors
    driver_errors = {}
    driver_circuit_errors = {}  # driver -> circuit_key -> list of errors (Phase 6)
    
    for entry in log:
        circuit_type = entry.get("circuit_type", "permanent")
        comparisons = entry.get("driver_comparisons", [])
        for comp in comparisons:
            driver = comp.get("driver")
            predicted = comp.get("predicted")
            actual = comp.get("actual")
            if not driver or predicted is None or actual is None:
                continue
            if actual >= 99: # DNF
                continue
            
            # Clip extreme individual errors before accumulating — prevents a
            # single catastrophic race result (e.g. car issue that didn't score
            # a DNF but finished far back) from permanently distorting the bias.
            error = max(-MAX_SINGLE_ERROR, min(MAX_SINGLE_ERROR, predicted - actual))
            
            if driver not in driver_errors:
                driver_errors[driver] = {
                    "permanent": [],
                    "street": [],
                    "hybrid": [],
                    "overall": []
                }
            
            if circuit_type in driver_errors[driver]:
                driver_errors[driver][circuit_type].append(error)
            driver_errors[driver]["overall"].append(error)

            # Phase 6: per-circuit tracking
            circuit_key_entry = entry.get("circuit_key", "")
            if circuit_key_entry:
                if driver not in driver_circuit_errors:
                    driver_circuit_errors[driver] = {}
                if circuit_key_entry not in driver_circuit_errors[driver]:
                    driver_circuit_errors[driver][circuit_key_entry] = []
                driver_circuit_errors[driver][circuit_key_entry].append(error)
            
    # Now compute EWMA
    driver_biases = {}
    for driver, types in driver_errors.items():
        driver_biases[driver] = {}
        for ctype, errors in types.items():
            if not errors:
                if ctype != "overall":
                    continue
                driver_biases[driver][ctype] = 0.0
                continue
            
            # Circuit-specific biases need enough races to be trustworthy.
            # With only 1-2 races at a given track type, a single unusual result
            # would dominate — skip the circuit-specific bias in that case.
            min_samples = MIN_SAMPLES_CIRCUIT if ctype != "overall" else MIN_SAMPLES_OVERALL
            if len(errors) < min_samples:
                # Not enough data — skip (don't emit a bias for this circuit type)
                continue
            
            # Use last_n
            recent_errors = errors[-last_n:]
            n = len(recent_errors)

            # EWMA iterating NEWEST → OLDEST so the most recent race carries
            # weight `decay` (0.85) and weights halve going back in time.
            # (The old forward iteration gave the newest error only (1−decay).)
            ewma = recent_errors[-1]
            for err in reversed(recent_errors[:-1]):
                ewma = decay * ewma + (1.0 - decay) * err

            # Bayesian shrinkage: scale toward zero when n is small.
            # This prevents early-season corrections from being overconfident.
            shrinkage = n / (n + CONFIDENCE_POOL)
            biased_val = round(ewma * shrinkage, 2)
            
            driver_biases[driver][ctype] = biased_val

    # Phase 6: Per-circuit EWMA biases
    driver_circuit_biases = {}
    for driver, circuits in driver_circuit_errors.items():
        driver_circuit_biases[driver] = {}
        for ckey, errors in circuits.items():
            if not errors:
                continue
            # Require at least 2 races at same circuit before trusting per-circuit bias
            if len(errors) < 2:
                continue
            recent_errors = errors[-last_n:]
            n = len(recent_errors)
            # Newest-weighted EWMA — same direction as the per-type loop above
            ewma = recent_errors[-1]
            for err in reversed(recent_errors[:-1]):
                ewma = decay * ewma + (1.0 - decay) * err
            shrinkage = n / (n + CONFIDENCE_POOL)
            driver_circuit_biases[driver][ckey] = round(ewma * shrinkage, 2)

    # Save — atomically, so predictor.load_context never reads a torn file
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_data = {
        "driver_biases": driver_biases,
        "driver_circuit_biases": driver_circuit_biases,  # Phase 6: per-circuit
    }
    try:
        tmp = BIAS_CORRECTIONS.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        os.replace(tmp, BIAS_CORRECTIONS)
    except Exception as e:
        print(f"Error saving bias corrections: {e}")

def compute_weight_adjustments():
    if not ACCURACY_LOG.exists():
        return
    
    try:
        with open(ACCURACY_LOG, "r", encoding="utf-8") as f:
            log = json.load(f)
    except Exception:
        return
    
    if not log:
        return
        
    # Gather MAEs by circuit type and overall
    maes_by_type = {"permanent": [], "street": [], "hybrid": []}
    all_maes = []
    
    for entry in log:
        ctype = entry.get("circuit_type", "permanent")
        mae = entry.get("mae")
        if mae is not None:
            if ctype in maes_by_type:
                maes_by_type[ctype].append(mae)
            all_maes.append(mae)
            
    if not all_maes:
        return
        
    overall_mae = sum(all_maes) / len(all_maes)
    
    adjustments = {
        "permanent": {"sc_prob": 0.0, "form_score": 0.0},
        "street": {"sc_prob": 0.0, "form_score": 0.0},
        "hybrid": {"sc_prob": 0.0, "form_score": 0.0}
    }
    
    for ctype, maes in maes_by_type.items():
        if not maes:
            continue
        avg_ctype_mae = sum(maes) / len(maes)
        
        # simple nudge logic
        if ctype == "street":
            # If street MAE is higher than overall, it is more chaotic -> rely more on safety car
            if avg_ctype_mae > overall_mae:
                nudge = min(0.15, max(-0.15, (avg_ctype_mae - overall_mae) * 0.1))
                adjustments[ctype]["sc_prob"] = round(nudge, 3)
                adjustments[ctype]["form_score"] = round(-nudge, 3)
        elif ctype == "permanent":
            # If permanent MAE is lower than overall, it is more predictable -> rely more on form
            if avg_ctype_mae < overall_mae:
                nudge = min(0.15, max(-0.15, (overall_mae - avg_ctype_mae) * 0.1))
                adjustments[ctype]["form_score"] = round(nudge, 3)
                adjustments[ctype]["sc_prob"] = round(-nudge, 3)
        elif ctype == "hybrid":
            # mild nudge
            diff = overall_mae - avg_ctype_mae
            nudge = min(0.10, max(-0.10, diff * 0.05))
            adjustments[ctype]["form_score"] = round(nudge, 3)
            adjustments[ctype]["sc_prob"] = round(-nudge, 3)
            
    # Save
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_data = {"adjustments": adjustments}
    try:
        with open(WEIGHT_ADJUSTMENTS, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
    except Exception as e:
        print(f"Error saving weight adjustments: {e}")

def get_model_health_report():
    report = {
        "accuracy_log": [],
        "bias_corrections": {},
        "weight_adjustments": {}
    }
    
    if ACCURACY_LOG.exists():
        try:
            with open(ACCURACY_LOG, "r", encoding="utf-8") as f:
                report["accuracy_log"] = json.load(f)
        except Exception:
            pass
            
    if BIAS_CORRECTIONS.exists():
        try:
            with open(BIAS_CORRECTIONS, "r", encoding="utf-8") as f:
                report["bias_corrections"] = json.load(f)
        except Exception:
            pass
            
    if WEIGHT_ADJUSTMENTS.exists():
        try:
            with open(WEIGHT_ADJUSTMENTS, "r", encoding="utf-8") as f:
                report["weight_adjustments"] = json.load(f)
        except Exception:
            pass
            
    return report

def reset_corrections():
    for p in [BIAS_CORRECTIONS, WEIGHT_ADJUSTMENTS]:
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass
    return {"status": "ok", "message": "Corrections reset successfully"}
