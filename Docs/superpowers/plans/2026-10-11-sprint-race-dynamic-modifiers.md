# Sprint Race Dynamic Modifiers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `predict_sprint_order()` in `engine/models/predictor.py` to incorporate live weekend telemetry (FP1 practice pace, sector deltas), self-improvement driver/circuit biases, and sanity guardrails instead of using a static, unadjusted 60/40 blend.

**Architecture:** Extend the sprint prediction flow to mirror `predict_finishing_order()`. After computing the raw 60/40 blend of race and qualifying models, apply deterministic practice pace modifiers (`fp_delta`), qualifying sector modifiers (`q_lap_delta`), sensibility guardrails (top-team form protection and backmarker floors), and Bayesian bias corrections. Include an explanation field for UI transparency.

**Tech Stack:** Python 3.12+, NumPy, Scikit-Learn, unittest

## Global Constraints
- Preserve backward compatibility of all dictionary keys emitted by `predict_sprint_order()` (`driver`, `team`, `predicted_pos`, `sprint_grid`, `confidence_pct`, `std_dev`, `dnf_prob_pct`, `elo_rating`, `predicted_sprint_rank`).
- Keep `predicted_sprint_rank` strictly contiguous from $1$ to $N$.
- Zero regressions in Monte Carlo simulation input contracts (`engine/models/monte_carlo.py`).
- Pass security scanner `python engine/tools/secret_scanner.py` with zero leaks.
- All modifications tracked in `.agents/git_tracker.json`.

---

### Task 1: Create Dedicated Unit Tests for Sprint Dynamic Modifiers

**Files:**
- Create: `engine/tools/tests/test_sprint_prediction.py`

**Interfaces:**
- Consumes: `F1Predictor` from `engine.models.predictor`
- Produces: Test suite validating practice pace, bias corrections, guardrails, and rank contiguity in `predict_sprint_order()`.

- [ ] **Step 1: Write the failing unit tests for sprint dynamic modifiers**

Create `engine/tools/tests/test_sprint_prediction.py` testing:
1. `test_sprint_order_applies_practice_pace`: Set `_practice_pace = {"Driver A": -0.5, "Driver B": +0.5}`, verify Driver A improves in predicted position relative to Driver B.
2. `test_sprint_order_applies_bias_corrections`: Set `_bias_corrections`, verify `clamped_bias` shifts `predicted_pos` and appears in `explanation`.
3. `test_sprint_order_applies_guardrails`: Verify a top-constructor driver with strong form is prevented from dropping past P10.
4. `test_sprint_order_rank_contiguity`: Verify `predicted_sprint_rank` is contiguous 1..N and matches ascending order of `predicted_pos`.

```python
import unittest
from engine.models.predictor import F1Predictor

class SprintPredictionDynamicModifierTests(unittest.TestCase):
    def setUp(self):
        self.p = F1Predictor()
        self.p._roster = {"Max Verstappen": "Red Bull", "Lando Norris": "McLaren"}
        self.p._driver_standings = [
            {"name": "Max Verstappen", "position": 1, "points": 100},
            {"name": "Lando Norris", "position": 2, "points": 90},
        ]
        self.p._ctor_standings = [
            {"constructor": "Red Bull", "position": 1},
            {"constructor": "McLaren", "position": 2},
        ]
        self.p._driver_form = {}
        self.p._circuit_config = {"key": "shanghai", "track_type": "permanent"}

    def test_sprint_order_applies_practice_pace(self):
        # Without practice pace
        self.p._practice_pace = {}
        res_baseline = {d["driver"]: d["predicted_pos"] for d in self.p.predict_sprint_order()}

        # With practice pace favoring Norris by -0.8s
        self.p._practice_pace = {"Lando Norris": -0.8, "Max Verstappen": 0.0}
        res_modified = {d["driver"]: d["predicted_pos"] for d in self.p.predict_sprint_order()}

        self.assertLess(res_modified["Lando Norris"], res_baseline["Lando Norris"])
```

- [ ] **Step 2: Run the test to confirm failure**
Run: `python -m unittest engine/tools/tests/test_sprint_prediction.py`
Confirm it fails because `_practice_pace` currently does not modify `predict_sprint_order()`.

---

### Task 2: Implement Dynamic Modifiers in `predict_sprint_order()`

**Files:**
- Modify: `engine/models/predictor.py` (around lines 1604–1640)

**Interfaces:**
- Consumes: `self._practice_pace`, `self._quali_sectors`, `self._bias_corrections`, `self._circuit_config`, `self._driver_form`, `self._ctor_standings`
- Produces: Enhanced `predict_sprint_order() -> list[dict]` with dynamic modifiers and explanation.

- [ ] **Step 1: Update `predict_sprint_order()` in `engine/models/predictor.py`**
1. Add practice pace delta:
   ```python
   fp_delta = self._practice_pace.get(drv, 0.0) if self._practice_pace else 0.0
   sprint_pred += fp_delta * 1.5
   ```
2. Add qualifying sector delta if available:
   ```python
   if self._quali_sectors and drv in self._quali_sectors:
       sprint_pred += self._quali_sectors[drv].get("q_lap_delta", 0.0) * 1.0
   ```
3. Add sensibility guardrails:
   ```python
   ctor_pos_val = next((c["position"] for c in self._ctor_standings if c["constructor"] == ctor), 99)
   form = self._driver_form.get(drv, {})
   form_avg = form.get("avg_position", 11.0)
   races_counted = form.get("races_counted", 0)
   is_rookie = bool(races_counted < 4)

   if ctor_pos_val <= 3 and form_avg <= 6.5 and sprint_pred > 10.0:
       sprint_pred = min(sprint_pred, 10.0)
   if ctor_pos_val >= 9 and sprint_pred < 10.0 and not (form_avg <= 9):
       sprint_pred = max(sprint_pred, 10.0)
   sprint_pred = max(1.0, min(22.0, sprint_pred))
   ```
4. Add self-improvement bias correction:
   ```python
   clamped_bias = 0.0
   if hasattr(self, "_bias_corrections") and self._bias_corrections:
       circuit_type = self._circuit_config.get("track_type", "permanent")
       biases = self._bias_corrections.get("driver_biases", {}).get(drv, {})
       bias = biases.get(circuit_type, biases.get("overall", 0.0))
       circuit_key = self._circuit_config.get("key", "")
       circuit_biases = self._bias_corrections.get("driver_circuit_biases", {}).get(drv, {})
       if circuit_key and circuit_key in circuit_biases:
           bias = circuit_biases[circuit_key]
       clamped_bias = max(-1.5, min(1.5, bias))
       sprint_pred = sprint_pred - clamped_bias
       sprint_pred = max(1.0, min(22.0, sprint_pred))
   ```
5. Add `explanation` and metadata fields to dictionary:
   ```python
   explanation = f"Sprint P{sprint_pred:.0f}: blended race/quali pace"
   if abs(fp_delta) > 0.05:
       explanation += f", practice delta ({fp_delta:+.2f}s)"
   if abs(clamped_bias) > 0.01:
       explanation += f" [Bias: {-clamped_bias:+.2f}]"
   ```
   Include `"explanation": explanation` and `"fp_pace_delta": round(fp_delta, 3)` in output dict.

- [ ] **Step 2: Run tests to verify they pass**
Run: `python -m unittest engine/tools/tests/test_sprint_prediction.py`
Verify all sprint dynamic modifier unit tests pass.

---

### Task 3: Verify Integration with Monte Carlo and Serving Pipeline

**Files:**
- Modify: `engine/serving/pipeline.py` (ensure telemetry/progress logging handles enriched sprint output)
- Test: `engine/tools/tests/test_contracts.py`

- [ ] **Step 1: Check Monte Carlo simulation contract**
Verify `simulate_race_weekend()` in `engine/models/monte_carlo.py` correctly digests the enriched sprint dictionaries without schema errors.

- [ ] **Step 2: Run full contract suite**
Run: `python -m unittest engine/tools/tests/test_contracts.py`
Verify contracts remain 100% green.

---

### Task 4: Security Scan and Git Push Cadence

**Files:**
- Check: `.agents/git_tracker.json`

- [ ] **Step 1: Run secret scanner**
Run: `python engine/tools/secret_scanner.py`
Ensure zero secrets or PII detected.

- [ ] **Step 2: Update change tracker and verify push cadence**
Check `change_count` in `.agents/git_tracker.json`. If 8 changes reached, execute commit and push per workspace guidelines.
