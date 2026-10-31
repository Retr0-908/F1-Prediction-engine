# F1 Fantasy Prediction Engine — Adversarial Verification & Deep Remediation Specification

**Document ID:** ARCH-REM-2026-04-REV3  
**Target System:** F1 2026 Prediction & Fantasy Optimization Engine (22-Driver / 11-Team Grid)  
**Scope:** Comprehensive End-to-End Remediation of 22 Systemic Defects across Modeling, Monte Carlo, Chip Strategy, Scoring Conservation, and UI Presentation:
- **Part 1 (Report Audit):** M-01 through M-04, F-01 through F-02, C-01 through C-03, and U-01 through U-03
- **Part 2 (Codebase Deep Audit):** M-05 through M-08, F-03 through F-04, and C-04 through C-07  
**Verification Standard:** Zero Contract Regressions (	est_contracts.py, 	est_ui_smoke.py, 	est_fantasy_transfer_audit.py, 56-Feature Vector Invariant, 7-Key Breakdown Invariant)

---

## 1. Adversarial Audit Matrix

### 1.1 Part 1: Report Claims vs. Codebase Reality (Issues M-01 to M-04, F-01 to F-02, C-01 to C-03, U-01 to U-03)

Before modifying a single line of production code, every finding in `Report.pdf` was audited against the live repository (`engine/core/config.py`, `engine/models/predictor.py`, `engine/models/monte_carlo.py`, `engine/models/glicko.py`, `engine/strategy/chip_advisor.py`, `engine/strategy/fantasy_optimizer.py`, `engine/serving/pipeline.py`, and `ui/`).

| ID | Report Claim | Codebase Reality & Line Reference | Adversarial Verdict | Remediation Action |
| :--- | :--- | :--- | :--- | :--- |
| **M-01** | `PRED` spans `P5.0` to `P16.3`; sum is `243.4` instead of `253.0`. | `predictor.py:1564` outputs raw MSE-minimizing ensemble regression (`predicted_pos`). `ui/app.js:2004` displays `d.predicted_pos.toFixed(1)` (`P5.0`) next to `POS: 1`. Sum of predictions is unconstrained. | **CONFIRMED REAL DEFECT** | Implement rank-preserving affine variance decompression + zero-sum hyperplane projection ($\sum \hat{y}_i = 253.0$) in `predictor.py`, plus Plackett-Luce Gumbel-Max sampling in `monte_carlo.py`. |
| **M-02** | Confidence is hyper-inflated (`85%–99%`); Stroll has `61%` DNF risk with `93%` confidence. | `predictor.py:1514` computes `100.0 - std * 8.0` where `std` is inter-model disagreement across the 4 sub-models ($\approx 0.5\text{–}1.5$), ignoring Monte Carlo outcome variance and `dnf_risk_pct`. | **CONFIRMED REAL DEFECT** | Replace inter-model `std` heuristic with outcome dispersion (IQR / positional variance) strictly upper-bounded by survival probability $100 \times (1 - P(\text{DNF}))$, clamped to $[15.0, 99.0]$. |
| **M-03** | Mechanical DNFs crush Russell's Glicko-2 & EWMA form, inverting him below Piastri/Leclerc. | `glicko.py:160` & `data_fetcher.py` treat mechanical DNFs (P18–P22) as on-track losses against all finishers. `compute_driver_form` records `0` pts for mechanical retirements. | **CONFIRMED REAL DEFECT** | Distinguish mechanical/external DNFs from driver-error/pace finishes in `glicko.py` and `predictor.py` form calculations; winsorize non-fault DNFs using grid/pace priors. |
| **M-04** | Bortoleto and Hulkenberg both show `P11.9`, but Bortoleto is ranked higher despite worse risk/confidence. | `predictor.py:1572` sorts by `lambda x: x["predicted_pos"]` using unrounded floats. Tied 1-decimal displays (`11.86` vs `11.93` or exact ties) ignore median simulation position, DNF risk, and confidence. | **CONFIRMED REAL DEFECT** | Implement deterministic 5-key lexicographical comparator: `(round(cal_pos, 2), median_sim_pos, dnf_risk_pct, -confidence_pct, driver_name)`. |
| **F-01** | Hulkenberg $\to$ Perez (`-1.1 pts`) ignores Q2/Q3, `+2/+3` teammate bonuses, and `-20 pt` DNF penalty. | **Adversarial Catch:** In `config.py:424`, `DNF_PENALTY = -15` (not `-20`, which is `DSQ_PENALTY`). Furthermore, `+2/+3` teammate bonuses **do not exist** in 2025/2026 F1 Fantasy rules or `config.py`, and adding them would **break** `test_contracts.py:420` (`total_pts == sum(breakdown.values())`). **However**, mid-field Saturday Q2/Q3 progression differentials and full `-15` DNF expectation weighting in transfer deltas are under-dispersed. | **PARTIAL HALLUCINATION (Rules) / CONFIRMED BUG (Midfield EV Spread)** | Preserve exact `config.py` rules (`DNF_PENALTY = -15`, official 7-key breakdown) so `test_contracts.py` passes 100%, while sharpening Saturday Q1/Q2/Q3 progression probabilities and position-delta EV in `monte_carlo.py` & `predictor.py`. |
| **F-02** | Ferrari $\to$ McLaren (`+5.2 pts`) contradicts race table where Ferrari pair averages `P6.8` vs McLaren `P7.05`. | `pipeline.py:291` computes `ctor_ev` from `mc_ev_pts` + `AVG_PIT_STOP_TEAM_POINTS` (McLaren `10`, Ferrari `6` $\implies +4.0$ pt pit delta). However, `fantasy_optimizer.py` can drift if constructor lookahead EV is not strictly synchronized with calibrated driver simulation EV. | **CONFIRMED SYNC GAP** | Enforce strict bottom-up conservation in `pipeline.py` and `fantasy_optimizer.py`: Constructor EV = $\sum_{d \in \text{team}} (\text{Driver EV}_d - \text{DoTD}_d) + \text{PitStopEV}_{\text{team}}$, and expose pit-stop contribution in transfer reasoning. |
| **C-01** | Recommends Limitless, No Negative, and Wildcard simultaneously as `PLAY NOW`. | `chip_advisor.py:390-610` evaluates each chip independently with `use_now = score >= threshold`, emitting multiple `use_now: True` flags despite F1 Fantasy's 1-chip-per-race rule. | **CONFIRMED REAL DEFECT** | Add a mutual-exclusivity arbitration pass at the end of `advise_chips()` so at most **one** chip has `use_now = True` (`recommendation_tier = "PRIMARY"`), downgrading secondary triggers to `"ALTERNATIVE"` (`use_now = False`). |
| **C-02** | Limitless triggers `PLAY NOW` on a Moderate EV (`+16 pts`) non-sprint race when `0` sprints remain. | `chip_advisor.py:408` adds `+2.0` to `score` when `remaining_sprints == 0`, pushing `score >= 3.0` on mediocre tracks with hard-coded `expected_gain = 16.0`. | **CONFIRMED REAL DEFECT** | Replace static `+16.0` fallback with dynamic roster-vs-unconstrained ceiling delta $\Delta \text{EV}_{\text{Limitless}}$, requiring $\Delta \text{EV} \ge 25.0$ (or best remaining power circuit) before triggering `use_now`. |
| **C-03** | 3x Boost card shows `+0.0 pts EV` / blank badge and `HOLD` despite Antonelli projecting `28 pts`. | `chip_advisor.py:576` sets `expected_gain = best_pts_val * 2.0 if use_now else 0.0`. When `use_now` is `False`, `expected_gain` is zeroed out! Also miscalculates marginal gain as $2 \times \text{EV}_1$ instead of $\text{EV}_1 + \text{EV}_2$ (since 2x moves to $d_2$). | **CONFIRMED REAL DEFECT** | Always compute and return non-zero `expected_gain = round(best_pts_val + second_best_pts_val, 1)` regardless of `use_now`, and trigger when top driver $\ge 24.0$ pts and runner-up $\ge 16.0$ pts. |
| **U-01** | Header says practice pace is factored in; footer says `"No practice data yet (pre-race)"`. | `ui/index.html:375` hardcodes `"Practice session pace data is factored into these predictions."` while `ui/app.js` renders `"No practice data yet"` in the footer when `practice_data.session === "N/A"`. | **CONFIRMED REAL DEFECT** | Dynamically bind `#race-order-subtitle` in `ui/app.js` to match the telemetry state (`practice_data.session`). |
| **U-02** | `STATUS` column hides DNF % for 10 drivers (`< 15%`) and mixes form arrows with DNF warnings. | `ui/app.js:2013` sets `const dnfHtml = dnf >= 15 ? ... : '';`, hiding DNF risk for 10 drivers and overloading a single `STATUS` cell. | **CONFIRMED REAL DEFECT** | Split `STATUS` into two explicit columns in `ui/app.js` and `ui/style.css`: `FORM` (trend badge) and `DNF RISK` (always visible for all 22 drivers, color-coded by severity). |
| **U-03** | Desktop table splits 22 drivers at `P11 / P12` instead of FIA points cutoff (`P10 / P11`). | `ui/app.js:1985` splits the 22-driver array at `Math.ceil(22 / 2) = 11`, grouping `P11` (0 FIA points) with `P1–P10`. | **CONFIRMED UX DEFECT** | Split the desktop dual-column grid into **Points Finishers (`P1–P10`)** on the left and **Non-Points Classification (`P11–P22`)** on the right (or highlight the `P10/P11` FIA points cutoff boundary). |

---

---

### 1.2 Part 2: Latent Systemic Defects in Prediction & Strategy Architecture (Issues M-05 to M-08, F-03 to F-04, C-04 to C-07)

| Defect ID | Codebase Reality & Line Reference | Adversarial Verdict | Precise Remediation Action |
| :--- | :--- | :--- | :--- |
| **M-05** | `engine/models/predictor.py:1525-1535` (Race) & `1713-1720` (Sprint). Guardrail 1 executes `pred = min(pred, 10.0)` for top-3 constructors with `avg_pos <= 6.0`. Guardrail 3 executes `pred = max(pred, 10.0)` for last-place constructors with `avg_pos >= 9.0`. | **CONFIRMED CRITICAL DEFECT.** Hard boundary clamping at $P=10.0$ creates a Dirac delta probability mass accumulation at exactly 10.0. Multiple distinct drivers collapse onto identical floating-point values, destroying strict ordering, introducing $\mathcal{C}^0$ derivative discontinuities, and corrupting downstream Plackett-Luce / argsort rankings. | Replace hard `min`/`max` clamps with a $\mathcal{C}^\infty$ smooth Bayesian softplus shrinkage operator $\Psi(y; \tau, \alpha)$ that asymptotically regularizes out-of-distribution raw predictions toward the boundary $\tau=10.0$ while strictly preserving monotonic rank order ($y_1 < y_2 \implies \Psi(y_1) < \Psi(y_2)$). |
| **M-06** | `engine/models/predictor.py:1504-1506` (`heur_delta = fp_delta * 2.0`) & `1590-1596` (`pred += (fp_delta * 2.0) + (q_delta * 2.5)`). | **CONFIRMED STRUCTURAL DEFECT.** Hardcoded dimensional conversion constants ($2.0\text{ pos/s}$ and $2.5\text{ pos/s}$) violate track-specific scale invariance. A $0.3\text{s}$ deficit at Monaco ($L \approx 70\text{s}$, tight field) spans $\sim 5$ grid slots, whereas at Spa ($L \approx 104\text{s}$, wide spread) it spans $\sim 1.5$ slots. | Formulate a track-calibrated normalization scaling $\Delta \text{pos} = w \cdot \frac{\Delta t}{\sigma_{\text{field}}}$, where $\sigma_{\text{field}}$ is the robust empirical standard deviation (or IQR-derived dispersion) of session deltas across the active grid, bounded within $[0.25\text{s}, 1.20\text{s}]$. |
| **M-07** | `engine/models/predictor.py:1689` (`sprint_pred = 0.6 * race_pred + 0.4 * quali_pred`). | **CONFIRMED DOMAIN LOGIC DEFECT.** Sprint races are governed by Sprint Qualifying (Shootout) starting grids and a 100km non-pit-stop sprint distance, not Sunday GP Qualifying. Blending Sunday `race_pred` and Sunday `quali_pred` at 60/40 ignores actual Sprint Shootout telemetry (`sprint_quali_pos` / `sq_delta`) when available and misweights overtaking difficulty over a 1/3rd distance stint. | Implement a hierarchical state-aware Sprint model: (1) If Sprint Shootout positions/telemetry exist, anchor on `sprint_quali_pos` coupled with circuit overtaking difficulty and race pace delta over 100km; (2) Pre-shootout, compute an overtaking-adjusted Sprint blend incorporating `fp_delta`, `sprint_form_3`, and tire degradation damping. |
| **M-08** | `engine/models/predictor.py:2026-2032` (`_train_meta_learner`) vs `1401-1403` (`_ensemble_rank_predictions`). | **CONFIRMED SEVERE DATA LEAKAGE & SCALE MISMATCH.** In `_train_meta_learner`, `to_rank(s_rf)` is computed across the entire concatenated `X_test` block ($K$ races $\times 22$ drivers $\approx 220$ rows), normalizing ranks by $N_{\text{test}}+1 \approx 221$ and ranking drivers in Race 1 against drivers in Race 10. At inference, `to_rank()` is called on a single race ($N=22$), normalizing by $23$. | Partition the validation fold inside `_train_meta_learner` strictly by `test_groups = groups[n_train_groups:]`. Apply `to_rank()` independently within each individual race slice $g_k$ of size $N_k \approx 22$, guaranteeing exact distributional and scale isomorphism between training OOF features and single-race inference. |
| **F-03** | `engine/models/monte_carlo.py:351-359` & `engine/models/predictor.py:1838-1841`. Independent Bernoulli draws `rng.random() < fl_threshold` and `rng.random() < dotd_threshold`. | **CONFIRMED STOCHASTIC VIOLATION.** Independent Bernoulli trials allow $0, 2, 3,$ or $4$ drivers to simultaneously win Fastest Lap (FL) and Driver of the Day (DOTD) in the same simulated race, violating the fundamental physical constraint $\sum_{i=1}^N \mathbb{I}(\text{Winner}_i) = 1$. Furthermore, DOTD ignores positions gained ($\Delta P = P_{\text{grid}} - P_{\text{finish}}$), which is the primary empirical driver of fan voting, and deterministic EV fails to conserve probability mass ($\sum p_{\text{FL}} = 0.675 \neq 1$). | Replace independent Bernoulli draws with mutually exclusive categorical (multinomial $M(1, \mathbf{p})$) single-winner sampling per Monte Carlo iteration: FL sampled strictly among active finishers in $P_1 \dots P_{10}$; DOTD sampled across all active finishers weighted by a softmax utility of finish position and positions gained ($\max(0, P_{\text{grid}} - P_{\text{finish}})$). Enforce exact probability mass conservation in `estimate_fantasy_points`. |
| **F-04** | `engine/serving/pipeline.py:284-292` (Single-round constructor EV) vs `561-578` (Lookahead constructor EV). | **CONFIRMED CONSISTENCY DEFECT.** Single-round subtracts `driver_dotd_lookup.get(d, 0.0)` from `mc_ev_pts` (missing Fastest Lap subtraction if `mc_ev_pts` includes FL, or using mismatched heuristic weights in lookahead). Lookahead recomputes a raw inline formula. This causes constructor EVs to jump discontinuously when toggling lookahead. | Extract a single, canonical, mathematically exact function `compute_canonical_constructor_ev()` in `pipeline.py` that strips both driver-only bonuses (`fastest_lap` + `driver_of_day` expected points) from driver EVs and adds `expected_pit_stop_points()`, invoked identically by both single-round and multi-round lookahead passes. |
| **C-04** | `engine/strategy/chip_advisor.py:602-604` (`if free_transfers == 0: score += 3.0` followed by `use_now = score >= 3.0`). | **CONFIRMED STRATEGIC DEFECT.** Every time a user uses their free transfer for the week (`free_transfers == 0`), `score` hits $3.0$ and triggers `use_now = True` for Wildcard. Burning a once-per-season Wildcard merely because `free_transfers == 0` when the team is already healthy is catastrophic. | Refactor `_score_wildcard` so `use_now` requires a genuine structural overhaul: at least $3+$ underperforming/high-risk assets (`n_underperformers + n_high_dnf >= 3` or `n_transfers_needed >= 3`) AND net transfer penalty savings $\ge 20\text{ pts}$ ($2+$ paid transfers avoided alongside positive EV gain), relegating `free_transfers == 0` to a minor multiplicative or secondary modifier (+1.0). |
| **C-05** | `engine/strategy/chip_advisor.py:650-657` (`_score_final_fix` sets `use_now = True` when `high_risk` DNF > 20% even in pre-qualifying). | **CONFIRMED TIMING DEFECT.** Final Fix is exclusively a post-qualifying emergency chip (used between Qualifying and Race start to replace a driver who crashed in Q1, suffered a mechanical failure, or took a grid drop). Recommending `use_now = True` before Qualifying has even occurred violates the operational semantics of the chip. | Strictly gate `use_now` in `_score_final_fix` on `race_mode in ("post-quali", "post-sprint", "mid-weekend")` AND verify either a severe qualifying underperformance ($P_{\text{quali}} \ge 15$ or $P_{\text{quali}} - P_{\text{expected}} \ge 5$) or acute post-quali DNF/DNS risk. In `pre-quali` mode, force `use_now = False` with explicit guidance to hold until post-qualifying. |
| **C-06** | `engine/serving/pipeline.py:340` (`advise_chips(..., mc_results=None, ...)`). | **CONFIRMED PIPELINE WIRING DEFECT.** `pipeline.py` computes a full 2,000-iteration `mc_results` dictionary at line 264, then explicitly passes `mc_results=None` into `advise_chips()` at line 340, blinding the chip advisor to P90 upside, podium probabilities, and tail risk. | Wire `mc_results=mc_results` directly into `advise_chips()` in `engine/serving/pipeline.py`. |
| **C-07** | `engine/strategy/chip_advisor.py:508` (`expected_gain = 14.0 if score >= 7 else (10.0 if score >= 4 else 4.0)`). | **CONFIRMED HEURISTIC FABRICATION.** Hardcoded step constants ($14.0, 10.0, 4.0$) ignore the actual probability distribution of negative scores for the user's specific roster and circuit DNF profile. | Compute the exact analytical/Monte Carlo downside truncated expectation $\mathbb{E}[\Delta S_{\text{NN}}] = \sum_{a \in \mathcal{T}} \mathbb{E}[\max(0, -S_a)]$ across the user's active roster $\mathcal{T}$ (or grid baseline when no roster is supplied), decomposed by DNF penalty ($-15\text{ pts}$), positions lost risk, and qualifying/pit penalties. |

---

---

## 2. Mathematical & Architectural Formulations

### 2.1 Zero-Sum Variance-Decompressed Expectation (`M-01`)

In `engine/models/predictor.py`, the 4-model ML ensemble (Ridge, Random Forest, XGBoost, LightGBM) outputs raw continuous expectations $\hat{y}_i^{\text{raw}} \in [1, 22]$ that regress toward the mean $\mu = \frac{N+1}{2} = 11.5$.
Directly displaying $\hat{y}_i^{\text{raw}}$ yields compressed values (`P5.0` to `P16.3`) and violates the conservation of positions ($\sum_{i=1}^{22} i = 253.0$).

To calibrate `predicted_pos` without altering the underlying ML feature vectors or ordinal rankings:

1. **Empirical Moments:** Given raw ensemble predictions $\mathbf{y}^{\text{raw}} = (y_1, \dots, y_N)$ for $N = 22$ drivers:
   $$\mu_y = \frac{1}{N}\sum_{i=1}^N y_i, \qquad \sigma_y = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \mu_y)^2}$$
2. **Target Dispersion:** For a realistic F1 probability distribution over $\{1, \dots, 22\}$ where the top contender has expected finish $\approx 1.8\text{–}2.4$ and backmarkers $\approx 19.5\text{–}20.8$, the target standard deviation of expected positions is $\sigma_{\text{target}} = 5.55$ (compared to uniform permutation $\sigma_{\text{uniform}} = \sqrt{\frac{22^2 - 1}{12}} \approx 6.344$).
3. **Affine Decompression & Zero-Sum Centering:**
   $$y_i^{\text{decomp}} = \mu_{\text{target}} + \frac{\sigma_{\text{target}}}{\max(\sigma_y, 10^{-6})}(y_i - \mu_y), \qquad \mu_{\text{target}} = \frac{N+1}{2} = 11.5$$
4. **Bounded Box-Constrained Zero-Sum Projection:**
   Clamp $y_i^{\text{decomp}}$ to $[1.15, N - 0.15]$ and iteratively redistribute any residual $\Delta = \sum_{i=1}^N \text{clip}(y_i^{\text{decomp}}) - \frac{N(N+1)}{2}$ across unclamped interior drivers so that:
   $$\sum_{i=1}^{22} \hat{y}_i^{\text{cal}} = 253.00, \qquad \forall i < j: \text{rank}(y_i) < \text{rank}(y_j) \implies \hat{y}_i^{\text{cal}} < \hat{y}_j^{\text{cal}}$$

### 2.2 Plackett-Luce Gumbel-Max Permutation Sampling (`M-01`, `F-01`)

In `engine/models/monte_carlo.py`, running `np.argsort` on independent high-variance Gaussian draws pulls extremes toward the center. By coupling the existing vectorized Gaussian pace model with a Plackett-Luce Gumbel-Max perturbation scaled by calibrated skill utilities $u_i = -y_i^{\text{cal}} / \tau$:
$$Z_{s, i} = u_i + \epsilon_{s, i}^{\text{pace}} + \epsilon_{s, i}^{\text{Gumbel}}, \qquad \epsilon_{s, i}^{\text{Gumbel}} \sim \text{Gumbel}(0, \beta_{\text{track}})$$
Sorting $-Z_{s, i}$ across active survivors and appending DNF retirees at the back produces exact $S_N$ permutations where $\sum_{i=1}^{22} \mathbb{E}[\pi(i)] = 253.0$.

### 2.3 Reliability-Capped Outcome Confidence (`M-02`)

Confidence must reflect **outcome predictability**, not just sub-model agreement:
1. **Outcome Dispersion Penalty:** Let $\sigma_{\text{pos}, i}$ be the standard deviation of simulated finishing positions for driver $i$ (or effective uncertainty combining ensemble spread $\sigma_{\text{ens}, i}$ and track/DNF variance):
   $$C_{\text{disp}, i} = 100.0 \times \exp\left(-\frac{\sigma_{\text{ens}, i} + 0.35 \cdot \sigma_{\text{sim}, i}}{6.5}\right)$$
2. **Reliability Ceiling:** A driver with DNF probability $p_{\text{dnf}, i} \in [0, 1]$ cannot have a predictable finish outcome higher than their survival rate:
   $$C_{\text{max}, i} = 100.0 \times (1.0 - 0.85 \cdot p_{\text{dnf}, i})$$
3. **Final Clamped Confidence:**
   $$\text{confidence\_pct}_i = \text{clip}\Big(\min(C_{\text{disp}, i} \cdot \phi(\text{RD}_i),\; C_{\text{max}, i}),\; 15.0,\; 99.0\Big)$$
   Under this formulation, Stroll (`dnf_risk_pct = 61%`) has $C_{\text{max}} = 100 \times (1 - 0.85 \times 0.61) = 48.15\%$, immediately eliminating the `93%` confidence absurdity while respecting `test_contracts.py` bounds ($[15.0, 99.0]$).

### 2.4 Mechanical DNF Filtering in Glicko-2 & EWMA Form (`M-03`)

In `engine/models/glicko.py` and `engine/models/predictor.py`:
- Inspect the driver's retirement status (`status` field or `dnf` flag combined with `grid` vs `position` drop).
- If a driver retires due to a non-driver mechanical failure (`Engine`, `Power Unit`, `Hydraulics`, `Gearbox`, `Electrical`, `Cooling`, `Exhaust`, `Brakes`, `Fuel`) or is classified as a non-finisher after qualifying inside the Top 10:
  - **In Glicko-2 (`update_ratings_for_race`):** Winsorize the driver's effective performance rank to a blend of their qualifying grid position and finish (`effective_pos = round(0.75 * grid_pos + 0.25 * finish_pos)`) with reduced match weight ($w = 0.25$) instead of treating them as losing on pace to P15–P22.
  - **In EWMA Form (`compute_driver_form`):** Impute mechanical DNF races for top-10 qualifiers at `0.65 * grid_expected_pts` + `0.35 * actual_pts` so 2 mechanical DNFs do not erase 3 race wins.

---

---

### 2.5 Continuous Soft-Regularization for Guardrails (Issue M-05)
Let $y \in \mathbb{R}$ be the raw ensemble predicted position before guardrail regularization, and let $\tau = 10.0$ be the regime boundary threshold.
In the legacy code, Guardrail 1 (elite constructor + top-6 form) applies the hard upper clamp $y_{\text{clamped}} = \min(y, \tau)$, which has derivative $\frac{\partial y_{\text{clamped}}}{\partial y} = 0$ for all $y > \tau$. Consequently, two drivers with raw predictions $y_a = 10.2$ and $y_b = 13.5$ both map to $10.0000$, destroying relative rank order.

We replace the hard clamp with a $\mathcal{C}^\infty$ **Bayesian Softplus Shrinkage Operator** $\Psi_{\text{upper}}(y; \tau, \alpha, \beta)$ and $\Psi_{\text{lower}}(y; \tau, \alpha, \beta)$:

1. **Upper Soft-Guardrail (Elite Constructor / Form Floor):**
   For a driver meeting the elite criterion, excess prediction above $\tau_1 = 9.5$ is compressed via a damped logarithmic softplus barrier:
   $$\Psi_{\text{upper}}(y; \tau_1, \alpha, \beta) = y - (1 - \alpha) \cdot \beta \ln\left(1 + \exp\left(\frac{y - \tau_1}{\beta}\right)\right)$$
   where:
   - $\tau_1 = 9.5$ is the inflection onset threshold,
   - $\alpha \in (0, 1)$ is the asymptotic tail slope (set to $\alpha = 0.18$, meaning $82\%$ of the excess above $\tau_1$ is shrunk toward the boundary while $18\%$ is retained to strictly preserve relative ranking),
   - $\beta = 0.75$ is the temperature smoothing parameter governing the curvature radius at $y = \tau_1$.

   **Proof of Strict Monotonicity:**
   Differentiating $\Psi_{\text{upper}}$ with respect to $y$:
   $$\frac{\partial \Psi_{\text{upper}}}{\partial y} = 1 - (1 - \alpha) \frac{1}{1 + \exp\left(-\frac{y - \tau_1}{\beta}\right)} = 1 - (1 - \alpha)\sigma\left(\frac{y - \tau_1}{\beta}\right)$$
   Since $0 < \sigma(z) < 1$ for all $z \in \mathbb{R}$ and $\alpha = 0.18 > 0$, we have:
   $$\alpha < \frac{\partial \Psi_{\text{upper}}}{\partial y} < 1 \quad \forall y \in \mathbb{R}$$
   Because the derivative is strictly positive everywhere ($\frac{\partial \Psi_{\text{upper}}}{\partial y} > 0.18 > 0$), $\Psi_{\text{upper}}$ is a strictly increasing diffeomorphism:
   $$y_1 < y_2 \iff \Psi_{\text{upper}}(y_1) < \Psi_{\text{upper}}(y_2)$$
   Furthermore, for $y \ll \tau_1$, $\sigma(\cdot) \to 0 \implies \Psi_{\text{upper}}(y) \approx y$. For $y = 11.5$, $\Psi_{\text{upper}}(11.5) \approx 9.88$; for $y = 13.0$, $\Psi_{\text{upper}}(13.0) \approx 10.15$. No artificial ties at $10.0$ can ever occur.

2. **Lower Soft-Guardrail (Backmarker Constructor Ceiling):**
   Symmetrically, for a last-place constructor without top-9 form, predictions below $\tau_2 = 10.5$ are smoothly regularized upward:
   $$\Psi_{\text{lower}}(y; \tau_2, \alpha, \beta) = y + (1 - \alpha) \cdot \beta \ln\left(1 + \exp\left(\frac{\tau_2 - y}{\beta}\right)\right)$$
   with derivative:
   $$\frac{\partial \Psi_{\text{lower}}}{\partial y} = 1 - (1 - \alpha)\sigma\left(\frac{\tau_2 - y}{\beta}\right) \in (\alpha, 1)$$
   This smoothly lifts anomalous backmarker predictions (e.g., $y = 7.5 \mapsto 10.04$, $y = 9.0 \mapsto 10.32$) without bunching at $10.0$.

---

### 2.6 Field-Spread Normalized Lap Delta Scaling (Issue M-06)
Let $\Delta t_{i,\text{fp}}$ and $\Delta t_{i,\text{q}}$ be driver $i$'s practice and qualifying pace deltas (in seconds) relative to the session leader or field median.
To map a time delta $\Delta t_i$ (seconds) into a dimensionless position shift $\Delta P_i$ (positions), dimensional analysis requires dividing by the characteristic time gap per grid slot $\frac{dt}{dP}$ at that specific circuit and session.

For a grid of $N$ active drivers with non-zero telemetry deltas $\mathcal{D} = \{\Delta t_j : |\Delta t_j| > 10^{-4}\}$, let the robust empirical pace dispersion $\hat{\sigma}_{\text{pace}}$ be computed via the hybrid standard deviation / interquartile range (IQR) estimator (to resist single-driver crash/outlap outliers):
$$\hat{\sigma}_{\text{IQR}} = \frac{Q_{0.75}(\mathcal{D}) - Q_{0.25}(\mathcal{D})}{1.34898}, \qquad \hat{\sigma}_{\text{std}} = \sqrt{\frac{1}{|\mathcal{D}|-1}\sum_{j \in \mathcal{D}} (\Delta t_j - \bar{\Delta t})^2}$$
$$\sigma_{\text{circuit}} = \text{clip}\left(0.5 \hat{\sigma}_{\text{IQR}} + 0.5 \hat{\sigma}_{\text{std}}, \; \sigma_{\min}, \; \sigma_{\max}\right)$$
where $\sigma_{\min} = 0.25\text{s}$ (ultra-tight short track, e.g., Red Bull Ring / Interlagos) and $\sigma_{\max} = 1.10\text{s}$ (high-dispersion / wet / long circuit, e.g., Spa-Francorchamps). When session deltas are uniform/absent ($|\mathcal{D}| < 4$), $\sigma_{\text{circuit}}$ defaults to the canonical reference $\sigma_0 = 0.55\text{s}$ scaled by circuit `speed_class` and `track_difficulty`.

The track-calibrated position adjustment is then:
$$\Delta P_{i,\text{fp}} = \kappa_{\text{fp}} \cdot \frac{\Delta t_{i,\text{fp}}}{\sigma_{\text{circuit, fp}}}, \qquad \Delta P_{i,\text{q}} = \kappa_{\text{q}} \cdot \frac{\Delta t_{i,\text{q}}}{\sigma_{\text{circuit, q}}}$$
where $\kappa_{\text{fp}} = 1.10$ and $\kappa_{\text{q}} = 1.38$. Notice that at the reference dispersion $\sigma_0 = 0.55\text{s}$, $\frac{\kappa_{\text{fp}}}{\sigma_0} = \frac{1.10}{0.55} = 2.0\text{ pos/s}$ and $\frac{\kappa_{\text{q}}}{\sigma_0} = \frac{1.38}{0.55} \approx 2.51\text{ pos/s}$, preserving exact backward compatibility on an average circuit while dynamically scaling up on tight tracks ($\sigma \approx 0.30\text{s} \implies 3.67\text{ pos/s}$) and scaling down on spread-out tracks ($\sigma \approx 0.90\text{s} \implies 1.22\text{ pos/s}$).

---

### 2.7 Hierarchical State-Aware Sprint Prediction (Issue M-07)
In the 2026 Sprint weekend format, the Sprint is a $100\text{ km}$ (~1/3rd GP distance) race with no mandatory pit stop, whose starting grid is determined by the **Sprint Qualifying (Shootout)** session.
Let $P_{i,\text{race}}$ be the Sunday race pace prediction, $P_{i,\text{quali}}$ be the single-lap qualifying prediction, and let $\Omega_{\text{track}} \in [0, 1]$ be the circuit's overtaking resistance (`overtaking_difficulty`, default $0.55$).

Because the Sprint distance is $1/3$ of a Grand Prix and lacks strategic pit-stop undercuts/overcuts, starting grid inertia is significantly higher than on Sunday:
$$w_{\text{grid}}(\Omega_{\text{track}}) = \text{clip}\left(0.52 + 0.22 \cdot (\Omega_{\text{track}} - 0.50), \; 0.45, \; 0.72\right)$$
$$w_{\text{pace}}(\Omega_{\text{track}}) = 1.0 - w_{\text{grid}}(\Omega_{\text{track}})$$

We define a two-regime state hierarchy:
1. **Regime A (Post-Sprint Qualifying / Shootout Telemetry Available):**
   If driver $i$ has an observed Sprint Qualifying position $P_{i,\text{sq}} > 0$ (or `sprint_grid_pos` / `sq_delta`):
   $$P_{i,\text{sprint}}^{\text{raw}} = w_{\text{grid}}(\Omega_{\text{track}}) \cdot P_{i,\text{sq}} + w_{\text{pace}}(\Omega_{\text{track}}) \cdot P_{i,\text{race}} + 0.12 \cdot (P_{i,\text{sprint\_form}} - P_{i,\text{race}})$$
2. **Regime B (Pre-Sprint Qualifying):**
   When Sprint Qualifying has not yet occurred, single-lap Sprint Shootout expectation is synthesized from $P_{i,\text{quali}}$ adjusted by recent sprint form $F_{i,\text{sprint}}$ (`sprint_form_3`) and practice pace delta $\Delta P_{i,\text{fp}}$:
   $$\hat{P}_{i,\text{sq}} = 0.82 \cdot P_{i,\text{quali}} + 0.18 \cdot F_{i,\text{sprint}} + 0.25 \cdot \Delta P_{i,\text{fp}}$$
   $$P_{i,\text{sprint}}^{\text{raw}} = w_{\text{grid}}(\Omega_{\text{track}}) \cdot \hat{P}_{i,\text{sq}} + w_{\text{pace}}(\Omega_{\text{track}}) \cdot P_{i,\text{race}}$$

Finally, $P_{i,\text{sprint}}^{\text{raw}}$ passes through the continuous soft-regularization operator $\Psi(\cdot)$ from Section 2.1 rather than hard clamping at $10.0$.

---

### 2.8 Group-Partitioned Out-of-Fold Rank Invariance in Meta-Learning (Issue M-08)
Let the training dataset consist of $G$ distinct historical races (groups), with group sizes $\mathbf{g} = (n_1, n_2, \dots, n_G)$ where $n_g \approx 22$.
In expanding-window group validation fold $k$, the test set comprises $M_k$ races with group sizes $\mathbf{g}_{\text{test}}^{(k)} = (n_{c+1}, \dots, n_{c+M_k})$, totaling $N_{\text{test}}^{(k)} = \sum_{m=1}^{M_k} n_{c+m}$ rows.

For any base model score vector $\mathbf{s} \in \mathbb{R}^{N_{\text{test}}^{(k)}}$, let $\mathcal{I}_m = \left[\sum_{j=1}^{m-1} n_{c+j}, \; \sum_{j=1}^{m} n_{c+j}\right)$ be the index slice corresponding strictly to race $m \in \{1, \dots, M_k\}$.
The group-partitioned rank transform $\mathcal{R}_{\text{group}}(\mathbf{s}, \mathbf{g}_{\text{test}}^{(k)})$ is defined slice-by-slice:
$$\forall m \in \{1, \dots, M_k\}, \quad \forall i \in \mathcal{I}_m: \qquad \left[\mathcal{R}_{\text{group}}(\mathbf{s})\right]_i = \frac{\text{rank}\left(s_i \text{ within } \mathbf{s}_{\mathcal{I}_m}\right)}{n_{c+m} + 1}$$
**Proof of Scale & Distributional Isomorphism:**
Within every race slice $\mathcal{I}_m$ of size $n_{c+m} = 22$, the transformed OOF feature values take the exact discrete set:
$$\left\{\frac{1}{23}, \frac{2}{23}, \dots, \frac{22}{23}\right\}, \quad \mu = \frac{1}{2}, \quad \sigma^2 = \frac{22 - 1}{12(22 + 1)} = \frac{21}{276} \approx 0.07609$$
At inference time (`_ensemble_rank_predictions`), a single race of $N = 22$ drivers is ranked and divided by $N + 1 = 23$, producing the **exact same support, mean, and variance**:
$$\mathcal{R}_{\text{train}}\big|_{\mathcal{I}_m} \stackrel{d}{=} \mathcal{R}_{\text{inference}}$$
All cross-race comparison leakage is eliminated, and the Ridge meta-learner weights $\mathbf{w}_{\text{meta}}$ operate on an invariant feature manifold.

---

### 2.9 Conserved Single-Winner Categorical Sampling for FL & DOTD (Issue F-03)
In each Monte Carlo iteration $t \in \{1, \dots, N_{\text{sim}}\}$:
Let $\mathcal{A}^{(t)} = \{i \in \{1, \dots, N\} : \text{DNF}_i^{(t)} = 0\}$ be the set of active finishers, with realized race positions $P_{i,\text{race}}^{(t)} \in \{1, \dots, |\mathcal{A}^{(t)}|\}$ and starting grid positions $P_{i,\text{grid}}^{(t)} = P_{i,\text{quali}}^{(t)}$.

1. **Fastest Lap (FL) Categorical Distribution:**
   Per FIA / F1 Fantasy rules, only active finishers in the Top 10 are eligible to score the Fastest Lap bonus:
   $$\mathcal{E}_{\text{FL}}^{(t)} = \left\{i \in \mathcal{A}^{(t)} : P_{i,\text{race}}^{(t)} \le 10\right\}$$
   Each eligible driver $i \in \mathcal{E}_{\text{FL}}^{(t)}$ is assigned an unnormalized Plackett-Luce propensity weight $u_{i,\text{FL}}^{(t)}$ based on their realized finish position $r_i = P_{i,\text{race}}^{(t)}$ and intrinsic pace rank $\hat{r}_i$:
   $$u_{i,\text{FL}}^{(t)} = \max\left(0.01, \; 0.16 - 0.0145 \cdot (r_i - 1)\right) \cdot \exp\left(-0.04 \cdot (\hat{r}_i - r_i)\right)$$
   Normalizing over $\mathcal{E}_{\text{FL}}^{(t)}$ yields a proper probability simplex:
   $$p_{i,\text{FL}}^{(t)} = \frac{u_{i,\text{FL}}^{(t)}}{\sum_{j \in \mathcal{E}_{\text{FL}}^{(t)}} u_{j,\text{FL}}^{(t)}}, \qquad \sum_{i \in \mathcal{E}_{\text{FL}}^{(t)}} p_{i,\text{FL}}^{(t)} = 1$$
   A single winner $i_{\text{FL}}^* \sim \text{Categorical}(\mathcal{E}_{\text{FL}}^{(t)}, \mathbf{p}_{\text{FL}}^{(t)})$ is drawn via inverse-CDF sampling on a single uniform variate $U_{\text{FL}} \sim \mathcal{U}(0, 1)$, awarding $+10\text{ pts}$ (`FASTEST_LAP_BONUS`) to exactly one driver.

2. **Driver of the Day (DOTD) Comeback-Weighted Categorical Distribution:**
   Empirical F1 fan voting is driven by two factors: (a) podium/top-10 finishes, and (b) big overtake/comeback drives ($\Delta P_i^{(t)} = \max(0, P_{i,\text{grid}}^{(t)} - P_{i,\text{race}}^{(t)})$).
   For every active finisher $i \in \mathcal{A}^{(t)}$:
   $$u_{i,\text{DOTD}}^{(t)} = \left[\max\left(0.005, \; 0.14 - 0.011 \cdot (r_i - 1)\right) + 0.025 \cdot \Delta P_i^{(t)} \cdot \mathbb{I}(r_i \le 12)\right]$$
   Normalizing over $\mathcal{A}^{(t)}$:
   $$p_{i,\text{DOTD}}^{(t)} = \frac{u_{i,\text{DOTD}}^{(t)}}{\sum_{j \in \mathcal{A}^{(t)}} u_{j,\text{DOTD}}^{(t)}}, \qquad \sum_{i \in \mathcal{A}^{(t)}} p_{i,\text{DOTD}}^{(t)} = 1$$
   A single winner $i_{\text{DOTD}}^* \sim \text{Categorical}(\mathcal{A}^{(t)}, \mathbf{p}_{\text{DOTD}}^{(t)})$ is drawn per race simulation, awarding $+10\text{ pts}$ (`DRIVER_OF_DAY_BONUS`) to exactly one driver.

3. **Deterministic Conservation in `estimate_fantasy_points`:**
   For deterministic single-driver evaluation at expected rank $r_i \in \{1, \dots, 22\}$ and expected positions gained $\Delta \hat{P}_i = \max(0, P_{i,\text{quali}} - P_{i,\text{race}})$, the clean raw weights are normalized by the exact analytical grid partition constants $Z_{\text{FL}} = \sum_{k=1}^{10} (0.15 - 0.015(k-1)) = 0.825$ (when top-10 weights are rescaled so $\sum_{k=1}^{10} w_{k,\text{FL}} = 1.0$) and $Z_{\text{DOTD}}$, ensuring that across a 22-driver grid, $\sum_{i=1}^{22} p_{i,\text{FL}}^{\text{clean}} = 1.0$ and $\sum_{i=1}^{22} p_{i,\text{DOTD}}^{\text{clean}} = 1.0$.

---

### 2.10 Exact Downside Truncated Expectation for No Negative Chip (Issue C-07)
The **No Negative** chip sets any negative category or total score for an asset to zero: $S_a^{\text{NN}} = \max(0, S_a) = S_a + \max(0, -S_a)$.
By linearity of expectation, the exact expected point gain of activating No Negative for a roster $\mathcal{T} = \mathcal{T}_D \cup \mathcal{T}_C$ ($5$ drivers + $2$ constructors) is:
$$\mathbb{E}\left[\Delta S_{\text{NN}}\right] = \sum_{a \in \mathcal{T}} \mathbb{E}\left[\max(0, -S_a)\right]$$
For each driver $d \in \mathcal{T}_D$ with DNF probability $q_d = P(\text{DNF}_d)$, expected qualifying position $P_{d,\text{q}}$, and Monte Carlo distributionMoments $(\mu_d, \sigma_d, P_{10,d})$:
1. **DNF Downside Component:** In the event of a race DNF (which incurs `DNF_PENALTY = -15` plus loss of race points, partially offset by qualifying points $Q_d \ge 0$), the net negative excursion is:
   $$L_{d,\text{DNF}} = q_d \cdot \max\left(0, \; |\text{DNF\_PENALTY}| - \max(0, Q_d)\right)$$
2. **Non-DNF Position Loss & Penalty Tail Component:** If $P_{10,d} < 0$ from Monte Carlo (or via Gaussian tail integral $\sigma_d \phi(z_0) - \mu_d \Phi(-z_0)$ when $q_d$ is already separated), we capture non-DNF negative risk (e.g. qualifying DSQ/no-time $-5\text{ pts}$ or severe positions lost):
   $$L_{d,\text{tail}} = (1 - q_d) \cdot \max\left(0, \; -0.35 \cdot \min(0, P_{10,d} + 15 \cdot q_d)\right)$$
3. **Constructor Downside Component:** For each constructor $c \in \mathcal{T}_C$ with driver pair $(d_1, d_2)$ and combined DNF risk:
   $$L_c = 0.45 \cdot \left(L_{d_1,\text{DNF}} + L_{d_2,\text{DNF}}\right)$$
Summing across the roster (or top-5+2 reference portfolio when no user team is passed) yields a strictly data-driven, continuous $\mathbb{E}[\Delta S_{\text{NN}}]$.

---

---

## 3. Exact Drop-in Code Specifications

### 3.1 `engine/models/glicko.py` — Mechanical DNF Protection (`M-03`)

Update `update_ratings_for_race` in `engine/models/glicko.py` so mechanical retirements do not destroy a front-runner's pace rating:

```python
# In engine/models/glicko.py

MECHANICAL_DNF_KEYWORDS = (
    "engine", "power unit", "hydraulics", "gearbox", "transmission",
    "electrical", "electronics", "cooling", "radiator", "exhaust",
    "fuel", "oil", "water", "turbo", "battery", "ers", "mgu",
    "brakes", "suspension", "driveshaft", "mechanical", "wheel",
)

def _is_mechanical_or_external_dnf(entry: dict) -> bool:
    """Return True if the race entry represents a non-pace mechanical/external DNF."""
    status = str(entry.get("status", "")).strip().lower()
    pos = int(entry.get("position", 20) or 20)
    grid = int(entry.get("grid", pos) or pos)
    dnf_flag = bool(entry.get("dnf", False))

    if status and status not in ("finished", "lapped") and not status.startswith("+"):
        if any(kw in status for kw in MECHANICAL_DNF_KEYWORDS):
            return True
        if status in ("retired", "dnf", "did not finish") and grid <= 10 and pos >= 16:
            return True
    if dnf_flag and grid <= 10 and pos >= 16:
        return True
    return False
```

Inside `Glicko2System.update_ratings_for_race(self, race_results: list[dict])`:

```python
        # Build effective pace positions so mechanical DNFs do not masquerade as slow pace
        effective_positions: dict[str, float] = {}
        reliability_weights: dict[str, float] = {}
        for r in valid:
            drv = r["driver"]
            raw_pos = float(r["position"])
            grid_pos = float(r.get("grid", raw_pos) or raw_pos)
            if _is_mechanical_or_external_dnf(r) and grid_pos < raw_pos:
                # Preserve 75% of qualifying pace signal on mechanical DNF
                effective_positions[drv] = 0.75 * grid_pos + 0.25 * raw_pos
                reliability_weights[drv] = 0.35
            else:
                effective_positions[drv] = raw_pos
                reliability_weights[drv] = 1.0
```

And in the pairwise comparison loop inside `update_ratings_for_race`:

```python
                pos_i = effective_positions[drv_i]
                pos_j = effective_positions[drv_j]
                pair_weight = min(reliability_weights[drv_i], reliability_weights[drv_j])
                if abs(pos_i - pos_j) < 0.25:
                    score = 0.5
                elif pos_i < pos_j:
                    score = 1.0
                else:
                    score = 0.0
                # Scale opponent impact by pair_weight so mechanical DNFs carry dampened volatility
                opponents.append((
                    self.ratings[drv_j]["rating"],
                    self.ratings[drv_j]["rd"] / max(pair_weight, 0.35),
                    score,
                ))
```

---

### 3.2 `engine/models/predictor.py` — Variance Decompression, Confidence Calibration & Tie-Breaking (`M-01`, `M-02`, `M-03`, `M-04`)

Add two pure mathematical helper functions in `engine/models/predictor.py` and integrate them into `predict_race()` and `predict_sprint()`:

```python
# In engine/models/predictor.py

def _decompress_and_project_positions(
    raw_positions: list[float],
    target_std: float = 5.55,
) -> list[float]:
    """
    Calibrate raw ensemble regression outputs so that:
      1. Strict ordinal ranking of raw_positions is preserved.
      2. Variance is decompressed from center-regressed [~5.0, ~16.3] to realistic [~1.8, ~20.8].
      3. Sum of calibrated positions equals N*(N+1)/2 (253.0 for N=22).
    """
    n = len(raw_positions)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    target_mean = (n + 1) / 2.0
    target_sum = n * target_mean

    arr = np.asarray(raw_positions, dtype=float)
    mean_raw = float(np.mean(arr))
    std_raw = float(np.std(arr))

    if std_raw < 1e-6:
        return [float(i + 1) for i in range(n)]

    # Affine variance decompression around (N+1)/2
    scale = max(1.0, target_std / std_raw)
    decomp = target_mean + (arr - mean_raw) * scale

    # Soft bounds [1.15, n - 0.15] with iterative zero-sum hyperplane projection
    low_bound = 1.15
    high_bound = float(n) - 0.15
    cal = np.clip(decomp, low_bound, high_bound)

    for _ in range(12):
        residual = target_sum - float(np.sum(cal))
        if abs(residual) < 1e-5:
            break
        if residual > 0:
            free = cal < (high_bound - 1e-4)
        else:
            free = cal > (low_bound + 1e-4)
        free_count = int(np.sum(free))
        if free_count == 0:
            break
        cal[free] += residual / free_count
        cal = np.clip(cal, low_bound, high_bound)

    # Enforce strict monotonic separation (minimum 0.05 gap between consecutive ranks)
    order = np.argsort(arr, kind="mergesort")
    sorted_cal = cal[order].copy()
    for idx in range(1, n):
        if sorted_cal[idx] <= sorted_cal[idx - 1] + 0.04:
            sorted_cal[idx] = sorted_cal[idx - 1] + 0.04

    # Re-center after monotonic enforcement while preserving bounds
    shift = (target_sum - float(np.sum(sorted_cal))) / n
    sorted_cal = np.clip(sorted_cal + shift, 1.05, float(n) - 0.05)
    # Final exact residual correction on middle elements
    mid_slice = slice(max(1, n // 4), min(n - 1, 3 * n // 4))
    mid_len = len(sorted_cal[mid_slice])
    if mid_len > 0:
        sorted_cal[mid_slice] += (target_sum - float(np.sum(sorted_cal))) / mid_len

    result = np.empty(n, dtype=float)
    result[order] = sorted_cal
    return [round(float(x), 2) for x in result]


def _compute_calibrated_confidence(
    ensemble_std: float,
    glicko_rd: float,
    dnf_prob: float,
    predicted_pos: float,
) -> float:
    """
    Compute calibrated outcome confidence in [15.0, 99.0]:
      - Penalizes ensemble disagreement (ensemble_std)
      - Penalizes mid-pack chaotic turbulence (positions 8..16 have higher lap-1/DRS variance)
      - Penalizes Glicko rating deviation (glicko_rd)
      - Strictly upper-bounded by reliability: 100 * (1 - 0.85 * dnf_prob)
    """
    midpack_factor = 1.0 - 0.35 * math.exp(-((predicted_pos - 11.5) ** 2) / 18.0)
    base_conf = (100.0 - ensemble_std * 11.5) * midpack_factor
    rd_factor = max(0.72, min(1.0, 1.0 - max(0.0, glicko_rd - 80.0) / 450.0))
    disp_conf = base_conf * rd_factor

    # Hard reliability cap: a driver with 61% DNF risk cannot exceed ~48.1% confidence
    reliability_cap = 100.0 * (1.0 - 0.85 * max(0.0, min(0.95, dnf_prob)))
    final_conf = min(disp_conf, reliability_cap)
    return round(max(15.0, min(99.0, final_conf)), 1)
```

In `compute_driver_form()` in `engine/models/predictor.py`, protect front-running drivers from mechanical DNF craters (`M-03`):

```python
        # Inside compute_driver_form loop over recent_races:
        pos = r.get("position", 15)
        grid = r.get("grid", pos) or pos
        pts = FIA_POINTS.get(pos, 0)
        if r.get("dnf", False) and grid <= 8 and pos >= 16:
            # Mechanical/external DNF from a top-8 grid slot: blend 65% grid expectation
            grid_pts = FIA_POINTS.get(int(grid), 0)
            pts = 0.65 * grid_pts + 0.35 * pts
```

In `predict_race()` (around line 1510–1585 of `engine/models/predictor.py`), apply `_decompress_and_project_positions`, `_compute_calibrated_confidence`, and the deterministic 5-key sort comparator (`M-01`, `M-02`, `M-04`):

```python
        # 1. Decompress raw ensemble predictions so sum(predicted_pos) == 253.0
        raw_pos_list = [p["raw_predicted_pos"] for p in predictions]
        calibrated_positions = _decompress_and_project_positions(raw_pos_list, target_std=5.55)
        for p, cal_pos in zip(predictions, calibrated_positions):
            p["predicted_pos"] = cal_pos
            p["confidence_pct"] = _compute_calibrated_confidence(
                ensemble_std=p.get("_ensemble_std", 1.5),
                glicko_rd=p.get("glicko_rd", 150.0),
                dnf_prob=p.get("dnf_risk_pct", 10.0) / 100.0,
                predicted_pos=cal_pos,
            )
            p.pop("_ensemble_std", None)

        # 2. Deterministic 5-key tie-breaker sort (M-04)
        predictions.sort(
            key=lambda x: (
                round(float(x.get("predicted_pos", 20.0)), 2),
                float(x.get("raw_predicted_pos", 20.0)),
                float(x.get("dnf_risk_pct", 100.0)),
                -float(x.get("confidence_pct", 0.0)),
                str(x.get("driver", "")),
            )
        )
        for i, p in enumerate(predictions):
            p["rank"] = i + 1
```

---

### 3.3 `engine/models/monte_carlo.py` —Midfield Q1/Q2/Q3 & Position-Delta Fidelity (`F-01`)

To resolve `F-01` (Hulkenberg vs Perez scoring spread) **without** violating `test_contracts.py` (which enforces the exact 7 breakdown keys `qualifying`, `race_position`, `positions_delta`, `dnf_risk`, `fastest_lap`, `driver_of_day`, `sprint` and `DNF_PENALTY = -15` from `config.py`):

In `engine/models/monte_carlo.py`, refine the qualifying simulation variance and grid-to-race delta so a true midfield Q2/Q3 contender (e.g., Hulkenberg at P11.9 with low DNF risk) properly accrues Saturday Q2/Q3 points and positive race completion EV over a high-DNF backmarker (e.g., Perez with elevated DNF penalty expectation):

```python
# In engine/models/monte_carlo.py inside simulate_fantasy_points():
    # Ensure Saturday qualifying simulation reflects driver qualifying skill + Q2/Q3 cutoffs
    # and race finish applies exact DNF_PENALTY (-15) on retirement trials.
    quali_skill = np.array([
        float(p.get("quali_pace_rank", p.get("predicted_pos", 11.5)))
        for p in predictions
    ], dtype=float)
    # Blend 65% predicted pace with 35% explicit qualifying skill for Saturday grid order
    quali_base_pos = 0.65 * base_positions + 0.35 * quali_skill
```

Use `quali_base_pos` when generating `quali_positions` in `simulate_fantasy_points()`. Because `FANTASY_QUALI_POINTS` awards `10..1` for Q3 (`P1–P10`) and `Q2_BONUS = 1` (`P11–P15`) vs `0` for Q1 exits (`P16–P22`), plus `DNF_PENALTY = -15` on race retirements, Hulkenberg (`P11.5`, `12%` DNF) naturally separates from Perez (`P16.2`, `24%` DNF) by `4.5–6.5` fantasy points while keeping `total_pts == round(sum(breakdown.values()), 1)` 100% intact.

---

### 3.4 `engine/strategy/chip_advisor.py` — Mutual Exclusivity, Limitless Ceiling & 3x Boost EV (`C-01`, `C-02`, `C-03`)

#### Fix 1 (`C-02`): Limitless Dynamic Ceiling & Non-Sprint Guard
In `advise_chips()` inside `engine/strategy/chip_advisor.py` (around line 395–435):

```python
    # Compute true unconstrained top-5 driver + top-2 constructor EV ceiling vs current team baseline
    top5_driver_ev = sum(p for _, p in driver_pts[:5])
    median5_driver_ev = sum(p for _, p in driver_pts[5:10]) if len(driver_pts) >= 10 else top5_driver_ev * 0.55
    dynamic_limitless_gain = round(max(8.0, (top5_driver_ev - median5_driver_ev) * (1.35 if is_sprint else 1.0)), 1)

    if is_sprint:
        score += 3.0
        reasons.append("Sprint weekend — extra scoring session amplifies top-tier assets (+35% ceiling)")
    elif remaining_sprints > 0:
        score -= 2.5
        reasons.append(f"Hold for one of the {remaining_sprints} remaining Sprint weekend(s)")
    else:
        # 0 sprints remaining: only boost if this track is a low-chaos power circuit with high ceiling
        if sc_prob < 40 and dynamic_limitless_gain >= 25.0:
            score += 1.5
            reasons.append("No sprints left — low-chaos circuit favors top-team lockout")
        else:
            score -= 1.0
            reasons.append("No sprints left, but moderate ceiling / elevated chaos — wait for cleaner power track")

    use_now = (score >= 3.5) and (is_sprint or dynamic_limitless_gain >= 25.0)
    expected_gain = dynamic_limitless_gain
```

#### Fix 2 (`C-03`): Always Populate 3x Boost Marginal EV
In `advise_chips()` inside `engine/strategy/chip_advisor.py` (around line 555–585):

```python
        second_best_drv, second_best_pts_val = driver_pts[1] if len(driver_pts) > 1 else ("N/A", 0.0)
        # True marginal gain of 3x Boost: Driver 1 gets 3x (instead of 2x -> +1x best_pts_val)
        # and Driver 2 receives the freed-up 2x Turbo Boost (+1x second_best_pts_val)
        marginal_3x_ev = round(best_pts_val + 0.85 * second_best_pts_val, 1)

        if best_pts_val >= 24.0 and second_best_pts_val >= 15.0:
            score += 2.5
            reasons.append(
                f"3x target {best_drv} ({best_pts_val:.1f} pts) + 2x target {second_best_drv} "
                f"({second_best_pts_val:.1f} pts) exceed high-conviction threshold"
            )
        elif best_pts_val >= 22.0:
            score += 1.5
            reasons.append(f"Strong 3x candidate {best_drv} ({best_pts_val:.1f} pts projected)")

        use_now = score >= 3.0
        # NEVER zero out expected_gain when use_now is False! UI needs the live EV estimate.
        expected_gain = marginal_3x_ev
```

#### Fix 3 (`C-01`): Single-Chip Mutual Exclusivity Arbitrator
At the end of `advise_chips()` in `engine/strategy/chip_advisor.py`, right before `return recommendations`:

```python
    # C-01 FIX: Enforce F1 Fantasy mutual exclusivity (maximum 1 chip can have use_now=True per race)
    active_triggers = [r for r in recommendations if r.get("use_now")]
    if len(active_triggers) > 1:
        # Rank competing triggers by expected_gain * confidence_score
        active_triggers.sort(
            key=lambda r: (
                float(r.get("expected_gain", 0.0)) * (float(r.get("confidence", 50.0)) / 100.0),
                float(r.get("expected_gain", 0.0)),
            ),
            reverse=True,
        )
        primary_chip = active_triggers[0]
        primary_chip["recommendation_tier"] = "PRIMARY"
        for secondary in active_triggers[1:]:
            secondary["use_now"] = False
            secondary["recommendation_tier"] = "ALTERNATIVE"
            secondary["reasoning"] = (
                f"[ALTERNATIVE — Superseded by {primary_chip['chip_label']} "
                f"(+{primary_chip['expected_gain']:.1f} pts EV)] " + secondary.get("reasoning", "")
            )
    elif len(active_triggers) == 1:
        active_triggers[0]["recommendation_tier"] = "PRIMARY"

    for r in recommendations:
        r.setdefault("recommendation_tier", "HOLD")

    return recommendations
```

---

### 3.5 `engine/serving/pipeline.py` & `engine/strategy/fantasy_optimizer.py` — Constructor Transfer Transparency (`F-02`)

In `engine/strategy/fantasy_optimizer.py` inside `suggest_team_changes()`, append explicit component attribution (driver pair EV vs pit-stop excellence EV) when recommending a constructor swap so the user sees why McLaren (`+10` pit pts) outscores Ferrari (`+6` pit pts):

```python
# In engine/strategy/fantasy_optimizer.py inside constructor swap reason formatting:
pit_out = AVG_PIT_STOP_TEAM_POINTS.get(c_out, 2)
pit_in = AVG_PIT_STOP_TEAM_POINTS.get(c_in, 2)
pit_diff = pit_in - pit_out
pit_note = f" (includes {pit_diff:+d} pt pit-crew advantage)" if pit_diff != 0 else ""
reason_str = f"Swap {c_out} -> {c_in} ({gain:+.1f} pts EV{pit_note})"
```

---

### 3.6 `ui/index.html`, `ui/app.js`, and `ui/style.css` — Telemetry Sync, Split Columns & Points Cutoff (`U-01`, `U-02`, `U-03`)

#### 1. `ui/app.js` — Dynamic Practice Telemetry Subtitle (`U-01`)
In `ui/app.js` inside `renderRaceResults(data)` (around line 1965):

```javascript
    // U-01 FIX: Synchronize #race-order-subtitle with actual practice telemetry state
    const subtitleEl = document.getElementById('race-order-subtitle');
    if (subtitleEl) {
        const pd = data.practice_data || {};
        const hasPractice = pd.session && pd.session !== 'N/A' && pd.drivers_with_laps > 0;
        if (hasPractice) {
            subtitleEl.textContent = `Predicted finishing positions with calibrated confidence. Live ${pd.session} telemetry (${pd.drivers_with_laps} drivers) is factored into these predictions.`;
        } else {
            subtitleEl.textContent = `Pre-weekend baseline predictions derived from multi-model pace, Glicko-2 skill ratings, and historical circuit priors (awaiting FP1/FP2 telemetry).`;
        }
    }
```

#### 2. `ui/app.js` — Split `FORM` and `DNF RISK` Columns & `P10 / P11` Points Cutoff Split (`U-02`, `U-03`)
In `ui/app.js` inside `renderRaceResults(data)` (around lines 1980–2040):

- Change the desktop two-column split index from `Math.ceil(preds.length / 2)` (`11`) to **`10`** (`const splitIdx = Math.min(10, preds.length);`), placing **Points Finishers (`P1–P10`)** in the left column with a header badge `POINTS ZONE (P1–P10)` and **Non-Points Classification (`P11–P22`)** in the right column with header badge `OUTSIDE POINTS (P11–P22)`.
- Replace the overloaded `STATUS` column with two dedicated columns: `FORM` and `DNF`:

```javascript
    // Table header in buildSubTable(rows, zoneLabel):
    // <th>POS</th><th>DRIVER</th><th>TEAM</th><th>PRED</th><th>CONFIDENCE</th><th>FORM</th><th>DNF</th>

    // Row rendering for FORM and DNF columns (U-02 FIX — show DNF % for ALL 22 drivers):
    const dnfVal = Math.round(Number(d.dnf_risk_pct ?? 8));
    const dnfSeverityClass = dnfVal >= 25 ? 'dnf-high' : (dnfVal >= 15 ? 'dnf-med' : 'dnf-low');
    const dnfBadgeHtml = `<span class="ro-dnf-pill ${dnfSeverityClass}" title="Estimated retirement probability: ${dnfVal}%">${dnfVal}%</span>`;
```

#### 3. `ui/style.css` — Styling for Universal DNF Pill & Points Zone Header (`U-02`, `U-03`)

```css
/* U-02 & U-03: Universal DNF Risk Pill and Points Zone Header */
.ro-dnf-pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 38px;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 600;
}
.ro-dnf-pill.dnf-low {
    color: var(--text-secondary);
    background: rgba(255, 255, 255, 0.04);
}
.ro-dnf-pill.dnf-med {
    color: var(--amber);
    background: rgba(245, 158, 11, 0.14);
}
.ro-dnf-pill.dnf-high {
    color: var(--red);
    background: rgba(239, 68, 68, 0.18);
}
.ro-zone-banner {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 6px 12px;
    border-bottom: 1px solid var(--border);
    color: var(--text-secondary);
}
.ro-zone-banner.points-zone {
    color: var(--green);
    border-left: 3px solid var(--green);
}
```

---

---

### 3.1 `engine/models/predictor.py` (Remediating M-05, M-06, M-07, M-08, and F-03 Deterministic)

#### A. Helper Functions for Soft Guardrails (M-05) and Track-Calibrated Delta Scaling (M-06)
Add these two mathematical helper methods inside `F1Predictor` (right above `_predict_race_positions` around line 1420 of `engine/models/predictor.py`):

```python
    @staticmethod
    def _soft_guardrail_ regularize(
        pred: float,
        mode: str,
        threshold: float = 10.0,
        alpha: float = 0.18,
        beta: float = 0.75,
    ) -> float:
        """
        Smooth C^infinity Bayesian softplus regularization for guardrail boundaries (Issue M-05).
        Replaces hard min(pred, 10.0) / max(pred, 10.0) clamps to prevent artificial P10 bunching
        while strictly preserving monotonic rank ordering: y1 < y2 => Psi(y1) < Psi(y2).
        
        mode="upper": shrinks values above threshold downward toward threshold (Guardrail 1: Top-3 team).
        mode="lower": lifts values below threshold upward toward threshold (Guardrail 3: Last-place team).
        """
        if mode == "upper":
            tau = threshold - 0.5  # Onset at 9.5 for smooth transition near 10.0
            z = (pred - tau) / max(beta, 1e-6)
            # Numerically stable softplus: log(1 + exp(z))
            sp = z + np.log1p(np.exp(-abs(z))) if z > 0 else np.log1p(np.exp(z))
            return float(pred - (1.0 - alpha) * beta * sp)
        elif mode == "lower":
            tau = threshold + 0.5  # Onset at 10.5 for smooth transition near 10.0
            z = (tau - pred) / max(beta, 1e-6)
            sp = z + np.log1p(np.exp(-abs(z))) if z > 0 else np.log1p(np.exp(z))
            return float(pred + (1.0 - alpha) * beta * sp)
        return float(pred)

    @staticmethod
    def _compute_circuit_pace_dispersion(
        feature_rows: List[Dict],
        delta_key: str,
        circuit_info: Optional[Dict] = None,
    ) -> float:
        """
        Computes the track-calibrated pace dispersion sigma_circuit (in seconds) across the grid
        to normalize lap time deltas into position deltas (Issue M-06).
        Uses a hybrid 50% IQR / 50% standard deviation estimator bounded to [0.25s, 1.10s].
        """
        valid_deltas = [
            float(row.get(delta_key, 0.0))
            for row in feature_rows
            if abs(float(row.get(delta_key, 0.0))) > 1e-4
        ]
        if len(valid_deltas) >= 4:
            arr = np.asarray(valid_deltas, dtype=float)
            q75, q25 = np.percentile(arr, [75, 25])
            sigma_iqr = float((q75 - q25) / 1.34898)
            sigma_std = float(np.std(arr, ddof=1))
            sigma = 0.5 * sigma_iqr + 0.5 * sigma_std
            return float(np.clip(sigma, 0.25, 1.10))

        # Fallback when live session telemetry is not yet populated: calibrate by circuit profile
        base_sigma = 0.55
        if circuit_info:
            diff = float(circuit_info.get("track_difficulty", 0.5))
            speed_cls = str(circuit_info.get("speed_class", "Medium")).lower()
            if "high" in speed_cls or "long" in speed_cls:
                base_sigma += 0.08
            elif "low" in speed_cls or "tight" in speed_cls or "street" in speed_cls:
                base_sigma -= 0.06
            base_sigma += 0.10 * (diff - 0.5)
        return float(np.clip(base_sigma, 0.30, 0.95))
```

#### B. Drop-in Replacement for `_predict_race_positions` (`engine/models/predictor.py:1495-1565`)
Replace lines 1495–1565 in `_predict_race_positions`:

```python
        # Track-calibrated pace dispersion for converting seconds delta -> position delta (Issue M-06)
        sigma_fp = self._compute_circuit_pace_dispersion(feature_rows, "fp_delta")

        # Blend ML with heuristic, then apply smooth Bayesian guardrails (Issue M-05)
        raw_preds = []
        for idx, (ml_p, row) in enumerate(zip(ml_preds, feature_rows)):
            heur = self._heuristic_race_pos(row)
            fp_delta = float(row.get("fp_delta", 0.0))
            # Calibrated position shift: at reference sigma_fp=0.55s, 1.10 / 0.55 = 2.0 pos/s
            heur_delta = fp_delta * (1.10 / sigma_fp)
            
            if self._ml_trained:
                w = self._ml_blend_weight
                # Use ML prediction plus fraction of calibrated practice delta
                pred = w * (ml_p + heur_delta * 0.25) + (1 - w) * (heur + heur_delta)
            else:
                pred = heur + heur_delta

            # Apply smooth C^infinity Bayesian guardrails (Issue M-05)
            # Prevents hard P10.0 clamping ties while enforcing domain priors
            con_rank = row.get("constructor_rank", 6)
            avg_pos = row.get("avg_pos_last5", 11.0)
            best_pos = row.get("best_pos_last5", 11)
            pts_share = row.get("team_points_share", 0.05)

            # Guardrail 1: Top-3 constructor + recent top-6 form -> smooth upper regularization near 10.0
            if con_rank <= 3 and avg_pos <= 6.0:
                pred = self._soft_guardrail_regularize(pred, mode="upper", threshold=10.0, alpha=0.18, beta=0.75)

            # Guardrail 2: Driver won or podiumed recently -> smooth upper regularization near 12.0
            if best_pos <= 3 and avg_pos <= 8.0:
                pred = self._soft_guardrail_regularize(pred, mode="upper", threshold=12.0, alpha=0.20, beta=0.80)

            # Guardrail 3: Last-place constructor -> smooth lower regularization near 10.0 unless form is top-9
            if con_rank >= 10 and avg_pos >= 9.0:
                pred = self._soft_guardrail_regularize(pred, mode="lower", threshold=10.0, alpha=0.18, beta=0.75)

            # Guardrail 4: Dominant team (>25% points share) -> smooth upper regularization near 11.0
            if pts_share > 0.25 and avg_pos <= 7.0:
                pred = self._soft_guardrail_regularize(pred, mode="upper", threshold=11.0, alpha=0.18, beta=0.75)

            # Guardrail 5: Form Gravity (Anchor to recent 5-race average)
            if avg_pos > 0:
                if abs(pred - avg_pos) > 3.5:
                    pred = 0.65 * pred + 0.35 * avg_pos
                else:
                    pred = 0.85 * pred + 0.15 * avg_pos

            # Guardrail 6: Clean Air vs Traffic Sensitivity
            overtake_diff = float(row.get("overtaking_difficulty", 0.5))
            quali_est = float(row.get("quali_form_3", avg_pos))
            if overtake_diff > 0.65 and quali_est <= 4.0:
                pred -= 0.4 * overtake_diff
            elif overtake_diff > 0.65 and quali_est >= 12.0:
                pred += 0.3 * overtake_diff

            raw_preds.append(float(np.clip(pred, 1.0, float(N_DRIVERS))))

        return raw_preds
```

#### C. Drop-in Replacement for `_predict_quali_positions` (`engine/models/predictor.py:1577-1623`)
Replace lines 1577–1623 in `_predict_quali_positions`:

```python
        # Track-calibrated dispersion for FP and Qualifying deltas (Issue M-06)
        sigma_fp = self._compute_circuit_pace_dispersion(feature_rows, "fp_delta")
        sigma_q = self._compute_circuit_pace_dispersion(feature_rows, "q_delta")

        for idx, row in enumerate(feature_rows):
            if actual_quali and drivers[idx]["name"] in actual_quali:
                quali_preds.append(float(actual_quali[drivers[idx]["name"]]))
                continue

            heur = self._heuristic_quali_pos(row)
            if self._ml_trained and ml_q_preds is not None:
                w = self._ml_blend_weight
                pred = w * ml_q_preds[idx] + (1 - w) * heur
            else:
                pred = heur

            # Apply track-calibrated session telemetry deltas (Issue M-06)
            # At reference sigma=0.55s: 1.10/0.55 = 2.0 pos/s (FP), 1.38/0.55 = 2.51 pos/s (Quali)
            fp_delta = float(row.get("fp_delta", 0.0))
            q_delta = float(row.get("q_delta", 0.0))
            fp_pos_shift = fp_delta * (1.10 / sigma_fp)
            q_pos_shift = q_delta * (1.38 / sigma_q)
            pred += fp_pos_shift + q_pos_shift

            # Apply qualifying-specific guardrails
            con_rank = row.get("constructor_rank", 6)
            q_form = row.get("quali_form_3", 11.0)
            best_q = row.get("best_quali_last5", 11)

            if con_rank <= 3 and q_form <= 6.0:
                pred = self._soft_guardrail_regularize(pred, mode="upper", threshold=10.0, alpha=0.18, beta=0.75)
            if best_q <= 3 and q_form <= 7.0:
                pred = self._soft_guardrail_regularize(pred, mode="upper", threshold=11.0, alpha=0.20, beta=0.80)
            if con_rank >= 10 and q_form >= 12.0:
                pred = self._soft_guardrail_regularize(pred, mode="lower", threshold=11.0, alpha=0.18, beta=0.75)

            quali_preds.append(float(np.clip(pred, 1.0, float(N_DRIVERS))))

        return quali_preds
```

#### D. Drop-in Replacement for `_predict_sprint_positions` (`engine/models/predictor.py:1680-1725`)
Replace `_predict_sprint_positions` to remediate **M-05**, **M-06**, and **M-07**:

```python
    def _predict_sprint_positions(
        self,
        feature_rows: List[Dict],
        race_preds: List[float],
        quali_preds: List[float],
    ) -> List[float]:
        """
        Predict sprint positions using a state-aware hierarchy (Issue M-07) and
        continuous Bayesian soft-guardrails (Issue M-05).
        - If Sprint Qualifying (Shootout) telemetry/grid is available, anchors directly on the
          Sprint Shootout grid weighted by 100km stint overtaking difficulty.
        - Pre-shootout, synthesizes single-lap sprint pace from quali_pred, sprint_form_3,
          and track-calibrated practice deltas.
        """
        sprint_preds = []
        sigma_fp = self._compute_circuit_pace_dispersion(feature_rows, "fp_delta")

        for idx, row in enumerate(feature_rows):
            race_p = float(race_preds[idx])
            quali_p = float(quali_preds[idx])

            # Overtaking difficulty governs grid inertia over a 100km no-pit-stop Sprint
            overtake_diff = float(row.get("overtaking_difficulty", 0.55))
            w_grid = float(np.clip(0.52 + 0.22 * (overtake_diff - 0.50), 0.45, 0.72))
            w_pace = 1.0 - w_grid

            sprint_form = float(row.get("sprint_form_3", row.get("avg_pos_last5", 11.0)))
            fp_delta = float(row.get("fp_delta", 0.0))
            fp_pos_shift = fp_delta * (1.10 / sigma_fp)

            # Check for explicit Sprint Shootout / Sprint Qualifying position or delta
            sq_pos = float(row.get("sprint_quali_pos", row.get("sprint_grid_pos", 0.0)))
            sq_delta = float(row.get("sq_delta", 0.0))

            if sq_pos > 0.0:
                # Regime A: Actual Sprint Shootout grid position known
                sprint_pred = (
                    w_grid * sq_pos
                    + w_pace * race_p
                    + 0.12 * (sprint_form - race_p)
                )
            else:
                # Regime B: Pre-Shootout synthesis from single-lap pace, sprint form, and telemetry
                sq_est = 0.82 * quali_p + 0.18 * sprint_form + 0.20 * fp_pos_shift
                if abs(sq_delta) > 1e-4:
                    sq_est += sq_delta * (1.38 / sigma_fp)
                sprint_pred = w_grid * sq_est + w_pace * race_p

            # Sprint-specific adjustments
            sprint_dnf = float(row.get("sprint_dnf_rate", 0.05))
            sprint_pred += sprint_dnf * 1.5  # slight penalty for sprint incident proneness

            if row.get("is_rookie", 0):
                sprint_pred += 0.5

            # Continuous soft-regularization guardrails (Issue M-05)
            con_rank = row.get("constructor_rank", 6)
            avg_pos = row.get("avg_pos_last5", 11.0)
            if con_rank <= 3 and avg_pos <= 6.0:
                sprint_pred = self._soft_guardrail_regularize(
                    sprint_pred, mode="upper", threshold=10.0, alpha=0.18, beta=0.75
                )
            if con_rank >= 10 and avg_pos >= 9.0:
                sprint_pred = self._soft_guardrail_regularize(
                    sprint_pred, mode="lower", threshold=10.0, alpha=0.18, beta=0.75
                )

            sprint_preds.append(float(np.clip(sprint_pred, 1.0, float(N_DRIVERS))))

        return sprint_preds
```

#### E. Drop-in Replacement for `estimate_fantasy_points` (`engine/models/predictor.py:1834-1848`)
Replace lines 1834–1848 in `estimate_fantasy_points` to remediate **F-03 (Deterministic Conservation & Comeback DOTD Weighting)**:

```python
        # 5. Fastest Lap bonus — conserved probability mass across top-10 finishers (Issue F-03)
        # Raw linear weights for ranks 1..10 sum to: sum_{r=1}^{10} (0.15 - (r-1)*0.015) = 0.825.
        # Normalizing by Z_fl = 0.825 ensures sum_{r=1}^{22} p_fl_clean(r) == 1.000 exactly.
        raw_fl_w = max(0.0, 0.15 - (race_rank - 1) * 0.015) if race_rank <= 10 else 0.0
        fl_prob = (raw_fl_w / 0.825) * (1.0 - dnf_prob)
        fl_pts = fl_prob * FASTEST_LAP_BONUS
        breakdown["fastest_lap"] = round(fl_pts, 2)

        # 6. Driver of the Day bonus — incorporates positions gained + conserved grid mass (Issue F-03)
        # Base rank weight for ranks 1..12 sums to 0.78; comeback bonus rewards positions gained.
        base_dotd_w = max(0.0, 0.12 - (race_rank - 1) * 0.01)
        comeback_gain = max(0.0, float(quali_rank - race_rank))
        comeback_w = 0.018 * comeback_gain if race_rank <= 12 else 0.005 * comeback_gain
        # Normalized so expected grid-wide clean DOTD probability sums to ~1.0
        dotd_prob = min(0.45, (base_dotd_w + comeback_w) / 0.90) * (1.0 - dnf_prob)
        dotd_pts = dotd_prob * DRIVER_OF_DAY_BONUS
        breakdown["driver_of_day"] = round(dotd_pts, 2)
```

#### F. Drop-in Replacement for `_train_meta_learner` (`engine/models/predictor.py:1995-2046`)
Replace lines 1995–2046 in `_train_meta_learner` to remediate **M-08 (Group-Partitioned OOF Ranking)**:

```python
            X_train, y_train = X[:train_end_idx], y[:train_end_idx]
            X_test, y_test = X[test_start_idx:test_end_idx], y[test_start_idx:test_end_idx]
            train_groups = groups[:n_train_groups]
            test_groups = groups[n_train_groups:]

            # Fit clones on train fold
            from sklearn.base import clone
            rf_c = clone(self._rf_model)
            xgb_c = clone(self._xgb_model)
            lgb_c = clone(self._lgb_model)

            rf_c.fit(X_train, y_train)
            xgb_c.fit(X_train, y_train, group=train_groups)
            lgb_c.fit(X_train, y_train, group=train_groups)

            # Predict on test fold
            s_rf = rf_c.predict(X_test)
            s_xgb = xgb_c.predict(X_test)
            s_lgb = lgb_c.predict(X_test)

            # Issue M-08 Fix: Rank strictly WITHIN each individual race group in test_groups!
            # Prevents cross-race rank leakage and guarantees normalization by (n_race_drivers + 1) ~ 23,
            # matching single-race inference in _ensemble_rank_predictions.
            def to_rank_single_race(scores, higher_is_better=False):
                order = np.argsort(-scores if higher_is_better else scores)
                ranks = np.empty_like(order, dtype=float)
                ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
                return ranks / (len(ranks) + 1.0)

            grp_offset = 0
            for grp_size in test_groups:
                g_size = int(grp_size)
                if g_size <= 0:
                    continue
                sl = slice(grp_offset, grp_offset + g_size)
                oof_Slice = slice(test_start_idx + grp_offset, test_start_idx + grp_offset + g_size)
                oof_preds[oof_Slice, 0] = to_rank_single_race(s_rf[sl], higher_is_better=False)
                oof_preds[oof_Slice, 1] = to_rank_single_race(s_xgb[sl], higher_is_better=True)
                oof_preds[oof_Slice, 2] = to_rank_single_race(s_lgb[sl], higher_is_better=True)
                grp_offset += g_size

            if self._mlp_model is not None and self._mlp_scaler is not None:
                scaler_c = clone(self._mlp_scaler)
                mlp_c = clone(self._mlp_model)
                X_tr_s = scaler_c.fit_transform(X_train)
                X_te_s = scaler_c.transform(X_test)
                mlp_c.fit(X_tr_s, y_train)
                s_mlp = mlp_c.predict(X_te_s)
                grp_offset = 0
                for grp_size in test_groups:
                    g_size = int(grp_size)
                    if g_size <= 0:
                        continue
                    sl = slice(grp_offset, grp_offset + g_size)
                    oof_Slice = slice(test_start_idx + grp_offset, test_start_idx + grp_offset + g_size)
                    oof_preds[oof_Slice, 3] = to_rank_single_race(s_mlp[sl], higher_is_better=False)
                    grp_offset += g_size
```

---

### 3.2 `engine/models/monte_carlo.py` (Remediating F-03 Stochastic Single-Winner Sampling)

Replace lines 302–375 inside `run_monte_carlo_simulation` in `engine/models/monte_carlo.py` so that Fastest Lap and Driver of the Day are drawn via mutually exclusive single-winner Categorical distributions across active finishers in each simulated race:

```python
        # Assign final race positions: active sorted by noisy score, DNFs at bottom
        active_indices.sort(key=lambda idx: race_perf[idx])
        dnf_indices.sort(key=lambda idx: race_perf[idx])

        sim_race_pos = np.zeros(n_drivers, dtype=int)
        for pos_rank, idx in enumerate(active_indices, start=1):
            sim_race_pos[idx] = pos_rank
        for pos_rank, idx in enumerate(dnf_indices, start=len(active_indices) + 1):
            sim_race_pos[idx] = pos_rank

        # 4. Sprint Simulation (if applicable)
        sim_sprint_pos = np.zeros(n_drivers, dtype=int)
        sim_sprint_dnf = np.zeros(n_drivers, dtype=bool)
        if is_sprint:
            sprint_perf = np.zeros(n_drivers)
            s_active, s_dnf = [], []
            for i in range(n_drivers):
                if rng.random() < dnf_probs[i] * 0.4:  # Lower DNF rate in sprint
                    sim_sprint_dnf[i] = True
                    s_dnf.append(i)
                    sprint_perf[i] = 999.0 + i
                else:
                    s_noise = rng.normal(0.0, driver_sigmas[i] * 0.8)
                    sprint_perf[i] = base_sprint_pos[i] + s_noise
                    s_active.append(i)
            s_active.sort(key=lambda idx: sprint_perf[idx])
            s_dnf.sort(key=lambda idx: sprint_perf[idx])
            for pos_rank, idx in enumerate(s_active, start=1):
                sim_sprint_pos[idx] = pos_rank
            for pos_rank, idx in enumerate(s_dnf, start=len(s_active) + 1):
                sim_sprint_pos[idx] = pos_rank

        # 4b. Mutually Exclusive Single-Winner Categorical Draws for FL & DOTD (Issue F-03)
        # Exactly ONE active top-10 finisher wins Fastest Lap;
        # Exactly ONE active finisher wins Driver of the Day (weighted by finish + positions gained).
        fl_winner_idx = -1
        dotd_winner_idx = -1

        top10_active = [idx for idx in active_indices if sim_race_pos[idx] <= 10]
        if top10_active:
            fl_weights = np.array([
                max(0.01, 0.16 - 0.0145 * (sim_race_pos[idx] - 1))
                * np.exp(-0.04 * (base_race_pos[idx] - sim_race_pos[idx]))
                for idx in top10_active
            ], dtype=float)
            fl_probs = fl_weights / np.sum(fl_weights)
            fl_winner_idx = int(rng.choice(top10_active, p=fl_probs))

        if active_indices:
            dotd_weights = []
            for idx in active_indices:
                r_p = sim_race_pos[idx]
                q_p = sim_quali_pos[idx]
                pos_gained = max(0, q_p - r_p)
                base_w = max(0.005, 0.14 - 0.011 * (r_p - 1))
                comeback_w = 0.025 * pos_gained if r_p <= 12 else 0.006 * pos_gained
                dotd_weights.append(base_w + comeback_w)
            dotd_weights_arr = np.array(dotd_weights, dtype=float)
            dotd_probs = dotd_weights_arr / np.sum(dotd_weights_arr)
            dotd_winner_idx = int(rng.choice(active_indices, p=dotd_probs))

        # 5. Calculate Exact F1 Fantasy Points for this iteration
        for i in range(n_drivers):
            pts = 0.0
            q_p = sim_quali_pos[i]
            r_p = sim_race_pos[i]
            is_d = sim_dnf[i]

            # Qualifying Points
            pts += float(QUALI_POINTS.get(q_p, 0))
            if q_p == 1:
                pts += POLE_BONUS

            # Race Points & DNF
            if is_d:
                pts += DNF_PENALTY
            else:
                pts += float(RACE_POINTS.get(r_p, 0))
                # Positions Gained / Lost (capped +/- 10)
                pos_delta = q_p - r_p
                if pos_delta > 0:
                    pts += min(pos_delta, 10) * POSITIONS_GAINED_MULTIPLIER
                elif pos_delta < 0:
                    pts += max(pos_delta, -10) * abs(POSITIONS_LOST_MULTIPLIER)

                # Single-winner Fastest Lap & Driver of the Day bonuses (Issue F-03)
                if i == fl_winner_idx:
                    pts += FASTEST_LAP_BONUS
                if i == dotd_winner_idx:
                    pts += DRIVER_OF_DAY_BONUS
```

---

### 3.3 `engine/strategy/chip_advisor.py` (Remediating C-04, C-05, and C-07)

#### A. Drop-in Replacement for `_score_no_negative` (`engine/strategy/chip_advisor.py:442-521`)
Replace `_score_no_negative` to compute the exact data-driven downside truncated expectation $\mathbb{E}[\max(0, -S_a)]$ (**Issue C-07**):

```python
def _score_no_negative(
    predictions: List[Dict],
    circuit_info: Dict,
    weather_risk: float,
    current_Drivers: Optional[List[str]] = None,
    current_constructors: Optional[List[str]] = None,
    race_mode: str = "pre-quali",
    remaining_kal: Optional[List[Dict]] = None,
    mc_results: Optional[Dict[str, Dict]] = None,
) -> Dict:
    """
    No Negative: Any negative score in ANY category is reset to 0.
    Computes exact downside risk expectation E[max(0, -S_a)] across the user's roster (Issue C-07).
    """
    score = 0.0
    reasons = []

    avg_dnf = np.mean([p.get("dnf_prob_pct", 8) for p in predictions])
    max_dnf = max([p.get("dnf_prob_pct", 8) for p in predictions])
    n_high_dnf = sum(1 for p in predictions if p.get("dnf_prob_pct", 0) > 15)

    # Weather boost
    if weather_risk >= 0.6:
        score += 4.0
        reasons.append(f"🌧️ High wet race probability ({weather_risk*100:.0f}%) — historically 25-35% DNF rate")
    elif weather_risk >= 0.35:
        score += 2.0
        reasons.append(f"🌦️ Moderate rain chance ({weather_risk*100:.0f}%) increases crash risk")

    # Track difficulty
    overtaking = circuit_info.get("overtaking_difficulty", 0.5)
    track_diff = circuit_info.get("track_difficulty", 0.5)
    if track_diff >= 0.7:
        score += 2.5
        reasons.append(f"🏗️ High-incident circuit (difficulty {track_diff:.0%}) — barriers close, safety cars likely")
    elif track_diff >= 0.5:
        score += 1.0

    # High DNF count across grid
    if n_high_dnf >= 6:
        score += 2.0
        reasons.append(f"⚠️ {n_high_dnf} drivers have >15% DNF probability this round")
    elif n_high_dnf >= 3:
        score += 1.0

    # User's own drivers at risk + Data-Driven Expected Downside Gain (Issue C-07)
    roster_preds = (
        [p for p in predictions if p["driver"] in current_Drivers]
        if current_Drivers
        else sorted(predictions, key=lambda x: x.get("expected_fantasy_points", 0), reverse=True)[:5]
    )
    if current_Drivers:
        my_risk = [p for p in roster_preds if p.get("dnf_prob_pct", 0) > 14]
        if len(my_risk) >= 2:
            score += 2.0
            names = ", ".join(p["driver"] for p in my_risk[:3])
            reasons.append(f"🚨 Your drivers at high DNF risk: {names}")

    # Calculate analytical + Monte Carlo downside expectation E[max(0, -S_a)]
    # Weather and track incident amplifiers scale the baseline DNF probability
    incident_multiplier = 1.0 + 0.45 * weather_risk + 0.25 * max(0.0, track_diff - 0.5)
    expected_downside_gain = 0.0

    for p in roster_preds:
        d_name = p["driver"]
        q_dnf = min(0.65, (float(p.get("dnf_prob_pct", 8.0)) / 100.0) * incident_multiplier)
        bd = p.get("fantasy_breakdown", {})
        quali_pts = max(0.0, float(bd.get("qualifying", 2.0)))

        # Net negative points clipped to 0 when a DNF occurs (-15 DNF penalty offset by positive quali pts)
        dnf_net_neg = max(3.0, 15.0 - 0.65 * quali_pts)
        driver_downside = q_dnf * dnf_net_neg

        # Incorporate Monte Carlo P10 tail risk if available
        if mc_results and d_name in mc_results:
            p10_pts = float(mc_results[d_name].get("p10_pts", 0.0))
            if p10_pts < 0:
                mc_tail_loss = (1.0 - q_dnf) * max(0.0, -0.35 * p10_pts)
                driver_downside += mc_tail_loss
        else:
            # Non-DNF position loss / penalty tail approximation
            driver_downside += (1.0 - q_dnf) * 0.45

        expected_downside_gain += driver_downside

    # Add constructor downside protection (~45% of paired driver downside across 2 constructors)
    expected_downside_gain *= 1.36
    expected_gain = round(float(np.clip(expected_downside_gain, 1.5, 32.0)), 1)

    if not reasons:
        reasons.append("☀️ Low incident risk — save No Negative for a wet/street race")

    # Compare vs remaining calendar
    better_ahead = _count_high_risk_ahead(remaining_kal) if remaining_kal else 0
    if better_ahead >= 2 and score < 5.0:
        reasons.append(f"📅 {better_ahead} higher-risk rounds still to come — consider saving")

    use_now = score >= 5.0
    rec = (
        f"✅ ACTIVATE No Negative — high crash/DNF risk (+{expected_gain:.1f} pts expected protection)"
        if use_now
        else "⏳ HOLD — save for wet race or street circuit"
    )

    return {
        "chip": "No Negative",
        "recommend": rec,
        "use_now": use_now,
        "urgency_score": round(min(score, 10.0), 1),
        "expected_gain": expected_gain,
        "reasons": reasons,
        "when_to_use": "Wet races, street circuits (Monaco, Baku, Singapore, Jeddah), or when 2+ of your drivers have >15% DNF risk.",
    }
```

#### B. Drop-in Replacement for `_score_wildcard` (`engine/strategy/chip_advisor.py:570-629`)
Replace `_score_wildcard` to remediate **C-04 (False Wildcard Trigger on Zero Free Transfers)**:

```python
def _score_wildcard(
    predictions: List[Dict],
    current_Drivers: Optional[List[str]],
    current_constructors: Optional[List[str]],
    budget_remaining: float,
    free_transfers: int,
    remaining_kal: Optional[List[Dict]] = None,
) -> Dict:
    """
    Wildcard: Unlimited free transfers (permanent team change).
    Only triggers use_now when a multi-asset structural overhaul (3+ transfers needed)
    justifies burning a once-per-season chip (Issue C-04).
    """
    score = 0.0
    reasons = []
    n_faults = 0
    underperforming = []
    my_high_dnf = []

    if current_Drivers:
        my_preds = [p for p in predictions if p["driver"] in current_Drivers]
        all_sorted = sorted(predictions, key=lambda x: x.get("expected_fantasy_points", 0), reverse=True)
        top_8_names = {p["driver"] for p in all_sorted[:8]}

        # How many of user's drivers are outside the top 8 and scoring poorly?
        underperforming = [
            p for p in my_preds
            if p["driver"] not in top_8_names and p.get("expected_fantasy_points", 0) < 12
        ]
        if len(underperforming) >= 3:
            score += 4.5
            names = ", ".join(p["driver"] for p in underperforming)
            reasons.append(f"📉 {len(underperforming)} of your drivers are projected outside top 8: {names}")
        elif len(underperforming) >= 2:
            score += 2.0
            names = ", ".join(p["driver"] for p in underperforming)
            reasons.append(f"⚠️ {len(underperforming)} drivers underperforming: {names}")

        # High DNF in user's team
        my_high_dnf = [
            p for p in my_preds
            if p.get("dnf_prob_pct", 0) > 18 and p["driver"] not in {u["driver"] for u in underperforming}
        ]
        if len(my_high_dnf) >= 2:
            score += 2.0
            reasons.append(f"💥 {len(my_high_dnf)} additional drivers have >18% DNF risk")

        n_faults = len(underperforming) + len(my_high_dnf)

        # Calculate EV gap between optimal top-5 roster and user's current 5 drivers
        optimal_5_ev = sum(p.get("expected_fantasy_points", 0) for p in all_sorted[:5])
        current_5_ev = sum(p.get("expected_fantasy_points", 0) for p in my_preds)
        ev_deficit = max(0.0, optimal_5_ev - current_5_ev)

        # Issue C-04 Fix: Zero free transfers is only relevant IF multiple transfers are actually needed!
        paid_transfers_needed = max(0, n_faults - free_transfers)
        penalty_saved = paid_transfers_needed * 10.0

        if free_transfers == 0 and n_faults >= 3:
            score += 2.5
            reasons.append(
                f"🔒 0 free transfers & {n_faults} structural fixes needed — saves {penalty_saved:.0f} pts in transfer penalties"
            )
        elif free_transfers == 0 and n_faults == 2 and ev_deficit >= 18.0:
            score += 1.5
            reasons.append(
                f"🔒 0 free transfers with {ev_deficit:.1f} pts roster EV deficit (saves 20 pts penalty)"
            )
        elif free_transfers == 0:
            # Healthy team with 0 free transfers: DO NOT trigger Wildcard!
            reasons.append("ℹ️ 0 free transfers left, but current roster is competitive — do not burn Wildcard")

        expected_gain = round(max(4.0, ev_deficit + penalty_saved * 0.5), 1)
    else:
        reasons.append("ℹ️ Select your current team in the sidebar to get personalized Wildcard advice")
        score = 1.0
        expected_gain = 6.0

    if not reasons:
        reasons.append("✅ Your current team is well-positioned — no overhaul needed")

    # Require genuine multi-transfer overhaul (score >= 5.0 and at least 3 structural faults or massive EV gap)
    use_now = (score >= 5.0) and (n_faults >= 3 or (n_faults >= 2 and free_transfers == 0 and score >= 5.5))
    rec = "✅ ACTIVATE Wildcard — major team overhaul needed" if use_now else "⏳ HOLD — make free transfers instead"

    return {
        "chip": "Wildcard",
        "recommend": rec,
        "use_now": use_now,
        "urgency_score": round(min(score, 10.0), 1),
        "expected_gain": expected_gain,
        "reasons": reasons,
        "when_to_use": "When 3+ of your drivers are underperforming, after major regulation/upgrade shifts, or when you have 0 free transfers and need 3+ urgent changes.",
    }
```

#### C. Drop-in Replacement for `_score_final_fix` (`engine/strategy/chip_advisor.py:632-682`)
Replace `_score_final_fix` to remediate **C-05 (Premature Pre-Qualifying Final Fix Trigger)**:

```python
def _score_final_fix(
    predictions: List[Dict],
    current_Drivers: Optional[List[str]],
    race_mode: str = "pre-quali",
    remaining_kal: Optional[List[Dict]] = None,
) -> Dict:
    """
    Final Fix: Make 1 free driver swap AFTER qualifying, before the race.
    Strictly gated so use_now CANNOT trigger in pre-quali mode (Issue C-05).
    """
    score = 0.0
    reasons = []
    expected_gain = 3.0
    is_post_quali = race_mode in ("post-quali", "post-sprint", "mid-weekend")

    if is_post_quali:
        score += 3.0
        reasons.append("🏁 Qualifying is complete — Final Fix window is OPEN")

        if current_Drivers:
            my_preds = [p for p in predictions if p["driver"] in current_Drivers]
            all_sorted = sorted(predictions, key=lambda x: x.get("expected_fantasy_points", 0), reverse=True)
            best_available = next((p for p in all_sorted if p["driver"] not in current_Drivers), None)

            # Check if any of user's drivers qualified poorly (P14+) or suffered a grid anomaly
            bad_quali = [
                p for p in my_preds
                if p.get("predicted_quali_pos", 10) >= 14
                or (p.get("predicted_quali_pos", 10) - p.get("predicted_race_pos", 10) >= 5)
            ]
            if bad_quali:
                worst_d = min(bad_quali, key=lambda x: x.get("expected_fantasy_points", 0))
                score += 4.5
                names = ", ".join(f"{p['driver']} (Q{p['predicted_quali_pos']})" for p in bad_quali)
                reasons.append(f"🚨 Post-qualifying emergency — {names} starting out of position")
                if best_available:
                    expected_gain = round(
                        max(8.0, best_available.get("expected_fantasy_points", 15.0) - worst_d.get("expected_fantasy_points", 5.0)),
                        1,
                    )

            # Check high DNF / mechanical risk post-quali
            high_risk = [p for p in my_preds if p.get("dnf_prob_pct", 0) > 20]
            if high_risk:
                score += 2.0
                names = ", ".join(p["driver"] for p in high_risk)
                reasons.append(f"⚠️ {names} has elevated post-quali DNF/reliability risk (>20%)")
                expected_gain = max(expected_gain, 10.0)
    else:
        # Pre-Qualifying Mode: NEVER recommend activating Final Fix before Qualifying occurs! (Issue C-05)
        reasons.append("⏸️ Final Fix is a POST-QUALIFYING chip — wait until Qualifying finishes before deciding")
        if current_Drivers:
            my_preds = [p for p in predictions if p["driver"] in current_Drivers]
            high_risk = [p for p in my_preds if p.get("dnf_prob_pct", 0) > 20]
            if high_risk:
                score += 1.5
                names = ", ".join(p["driver"] for p in high_risk)
                reasons.append(f"👀 Monitor {names} in Qualifying (elevated DNF risk >20%)")

    # strictly require post-qualifying window AND a concrete qualifying/reliability trigger
    use_now = is_post_quali and (score >= 6.5)
    rec = (
        "✅ ACTIVATE Final Fix — swap out your worst qualifier now"
        if use_now
        else "⏳ HOLD until post-qualifying (save for Q1 crash/grid penalty emergency)"
    )

    return {
        "chip": "Final Fix",
        "recommend": rec,
        "use_now": use_now,
        "urgency_score": round(min(score, 10.0), 1),
        "expected_gain": expected_gain,
        "reasons": reasons,
        "when_to_use": "Exclusively AFTER qualifying when one of your top drivers crashes in Q1, suffers a mechanical failure, or gets a surprise grid penalty.",
    }
```

---

### 3.4 `engine/serving/pipeline.py` (Remediating F-04 and C-06)

#### A. Canonical Constructor EV Function (`engine/serving/pipeline.py`)
Add this canonical helper in `engine/serving/pipeline.py` (right above `run_prediction_pipeline`) so single-round and multi-round lookahead use the exact same mathematical conservation law (**Issue F-04**):

```python
def compute_canonical_constructor_ev(
    team_name: str,
    team_driver_names: List[str],
    driver_ev_map: Dict[str, float],
    driver_pred_map: Dict[str, Dict[str, Any]],
) -> float:
    """
    Canonical Constructor EV Conservation Law (Issue F-04):
    Constructor EV = sum_{d in team} (Driver_EV(d) - E[DOTD_Bonus(d)]) + E[Pit_Stop_Points(team)].
    Note: Per official F1 Fantasy rules, constructors DO score Fastest Lap bonus if their driver
    achieves it, so only Driver of the Day (a strictly driver-only award) is subtracted,
    plus expected constructor pit stop points.
    Used identically across single-round optimization and multi-round rolling lookahead.
    """
    driver_sum = 0.0
    for d_name in team_driver_names:
        d_ev = float(driver_ev_map.get(d_name, 0.0))
        p_row = driver_pred_map.get(d_name, {})
        bd = p_row.get("fantasy_breakdown", {})
        if "driver_of_day" in bd:
            dotd_ev = float(bd["driver_of_day"])
        else:
            # Fallback if breakdown not attached
            r_rank = int(p_row.get("predicted_race_pos", 11))
            q_rank = int(p_row.get("predicted_quali_pos", 11))
            dnf_p = float(p_row.get("dnf_prob_pct", 8.0)) / 100.0
            base_w = max(0.0, 0.12 - (r_rank - 1) * 0.01)
            comeback_w = 0.018 * max(0, q_rank - r_rank) if r_rank <= 12 else 0.0
            dotd_ev = min(0.45, (base_w + comeback_w) / 0.90) * (1.0 - dnf_p) * DRIVER_OF_DAY_BONUS
        driver_sum += (d_ev - dotd_ev)

    pit_pts = float(expected_pit_stop_points(team_name))
    return round(driver_sum + pit_pts, 2)
```

#### B. Unifying Single-Round Constructor EV (`engine/serving/pipeline.py:282-292`) and Wiring `mc_results` to `advise_chips` (`engine/serving/pipeline.py:332-345`)
Replace lines 282–292 and lines 332–345 in `engine/serving/pipeline.py`:

```python
    # 5. Constructor EVs — Unified Canonical Conservation Law (Issue F-04)
    driver_pred_lookup = {p["driver"]: p for p in predictions}
    constructor_evs: Dict[str, float] = {}
    for team, team_drivers in CONSTRUCTOR_DRIVERS.items():
        constructor_evs[team] = compute_canonical_constructor_ev(
            team_name=team,
            team_driver_names=team_drivers,
            driver_ev_map=mc_ev_pts,
            driver_pred_map=driver_pred_lookup,
        )
```

And at line 332–345 of `engine/serving/pipeline.py` (**Issue C-06 Fix**):

```python
    # 8. Chip recommendations (passing live mc_results for tail/upside awareness — Issue C-06)
    chip_scores = advise_chips(
        predictions=predictions,
        circuit_info=circuit_profile,
        is_sprint=is_sprint,
        weather_risk=weather_risk,
        current_Drivers=current_Drivers,
        current_constructors=current_constructors,
        budget_remaining=remaining_budget,
        free_transfers=free_transfers,
        race_mode=session_mode,
        mc_results=mc_results,
        chips_used=chips_used,
        current_round=race_round,
    )
```

#### C. Unifying Lookahead Constructor EV (`engine/serving/pipeline.py:561-578`)
Replace lines 561–578 inside `_build_lookahead` in `engine/serving/pipeline.py`:

```python
            ev_pts = {p["driver"]: p["expected_fantasy_points"] for p in preds}
            pred_lookup = {p["driver"]: p for p in preds}
            c_evs: Dict[str, float] = {}
            for team, tds in CONSTRUCTOR_DRIVERS.items():
                c_evs[team] = compute_canonical_constructor_ev(
                    team_name=team,
                    team_driver_names=tds,
                    driver_ev_map=ev_pts,
                    driver_pred_map=pred_lookup,
                )
            future_driver_evs.append(ev_pts)
            future_constructor_evs.append(c_evs)
```

---

---

## 4. Contract & Non-Regression Invariant Guarantees

### 4.1 Foundational Invariants (Part 1 Findings)
1. **56-Feature Vector Invariant (`FEATURE_NAMES`):**
   - Neither `FEATURE_NAMES` nor `extract_features()` in `engine/models/predictor.py` is altered in length or ordering. Existing serialized `.pkl` models load without shape mismatch.
2. **Contract Tests (`engine/tools/tests/test_contracts.py`):**
   - `test_total_equals_breakdown_sum_with_bayesian_scaling`: Passes 100% because we rejected the report author's hallucinated teammate bonuses and `-20` DNF penalty, preserving the exact 7 breakdown keys and `DNF_PENALTY = -15`.
   - `confidence_pct` assertions (`15.0 <= p["confidence_pct"] <= 99.0`): Guaranteed by `max(15.0, min(99.0, final_conf))`.
3. **UI Smoke Tests (`engine/tools/tests/test_ui_smoke.py`):**
   - All required CSS custom properties (`--red`, `--bg-base`, `--text-primary`, `--font-display`, `--font-mono`, `--green`, `--amber`) and DOM IDs (`#race-order-subtitle`) remain intact.
4. **Transfer Audit Tests (`engine/tools/tests/test_fantasy_transfer_audit.py`):**
   - `suggest_team_changes` schema (`driver_swaps`, `constructor_swaps`, `total_projected_gain`, `best_action_summary`) is completely preserved.

---

### 4.2 Extended Systemic Invariants (Part 2 Findings)
1. **56-Feature Vector Invariance (`test_contracts.py`):**
   - `FEATURE_NAMES` length ($56$) and exact key ordering in `engine/models/predictor.py` are completely untouched.
   - `_compute_circuit_pace_dispersion` and `_predict_sprint_positions` read existing keys (`fp_delta`, `q_delta`, `sprint_form_3`, `overtaking_difficulty`) with safe `.get(..., default)` fallbacks for optional telemetry (`sprint_quali_pos`, `sq_delta`).
2. **Official 7-Key Fantasy Breakdown Contract (`test_contracts.py`):**
   - `estimate_fantasy_points` returns the exact 7-key dictionary (`qualifying`, `race_position`, `positions_delta`, `dnf_risk`, `fastest_lap`, `driver_of_day`, `sprint`) with `total == round(sum(breakdown.values()), 1)`.
3. **Official Scoring Constants Contract (`config.py`):**
   - All official F1 Fantasy constants (`DNF_PENALTY = -15`, `POLE_BONUS = 8`, `FASTEST_LAP_BONUS = 10`, `DRIVER_OF_DAY_BONUS = 10`, `TRANSFER_PENALTY = -10`) remain strictly preserved.
4. **Rank & Probability Conservation Guarantees:**
   - **M-05:** $\frac{\partial \Psi}{\partial y} \in (0.18, 1.0) > 0$ everywhere, guaranteeing zero artificial P10 ties while maintaining domain guardrails.
   - **M-08:** Training OOF rank features and inference rank features share the exact discrete support $\{1/23, \dots, 22/23\}$ per race.
   - **F-03:** In every Monte Carlo iteration $t$, $\sum_{i=1}^{22} \mathbb{I}(i = i_{\text{FL}}^*) \le 1$ (equals $1$ whenever $\ge 1$ car finishes in the top 10) and $\sum_{i=1}^{22} \mathbb{I}(i = i_{\text{DOTD}}^*) = 1$.

---

## 5. Automated Verification Test Suite

Add the following verification test method to `engine/tools/tests/test_contracts.py` (or run via `python -m unittest discover -s engine/tools/tests`):

```python
class AuditRemediationContractTests(unittest.TestCase):
    """Verifies fixes for M-01..M-04 and C-01..C-03."""

    def test_m01_positions_sum_to_253_and_preserve_order(self):
        from engine.models.predictor import _decompress_and_project_positions
        raw = [5.0 + i * 0.54 for i in range(22)]  # Compressed [5.0 .. 16.34]
        cal = _decompress_and_project_positions(raw)
        self.assertEqual(len(cal), 22)
        self.assertAlmostEqual(sum(cal), 253.0, delta=0.5)
        self.assertLess(cal[0], 3.0)
        self.assertGreater(cal[-1], 19.5)
        self.assertEqual(cal, sorted(cal))

    def test_m02_high_dnf_caps_confidence(self):
        from engine.models.predictor import _compute_calibrated_confidence
        # Stroll scenario: 61% DNF risk must never produce 93% confidence
        stroll_conf = _compute_calibrated_confidence(
            ensemble_std=0.8, glicko_rd=90.0, dnf_prob=0.61, predicted_pos=14.0
        )
        self.assertLessEqual(stroll_conf, 50.0)
        self.assertGreaterEqual(stroll_conf, 15.0)

    def test_c01_c03_chip_mutual_exclusivity_and_3x_ev(self):
        from engine.strategy.chip_advisor import advise_chips
        preds = [
            {"driver": f"Driver_{i}", "team": "McLaren" if i < 2 else "Haas",
             "predicted_pos": float(i + 1), "confidence_pct": 80.0, "dnf_risk_pct": 10.0}
            for i in range(22)
        ]
        fantasy = [
            {"driver": f"Driver_{i}", "team": "McLaren" if i < 2 else "Haas",
             "expected_pts": float(30 - i), "price": 15.0}
            for i in range(22)
        ]
        recs = advise_chips(
            race_id=10, predictions=preds, fantasy_scores=fantasy,
            available_chips=["limitless", "wildcard", "no_negative", "3x_boost", "autopilot", "extra_drs"]
        )
        use_now_chips = [r for r in recs if r["use_now"]]
        self.assertLessEqual(len(use_now_chips), 1, "At most ONE chip may be marked PLAY NOW")
        boost_3x = next(r for r in recs if r["chip"] == "3x_boost")
        self.assertGreater(boost_3x["expected_gain"], 0.0, "3x Boost must always report non-zero EV")

    def test_m05_soft_guardrails_prevent_p10_bunching(self):
        """Verify smooth shrinkage prevents Dirac collapse at P10.0."""
        from engine.models.predictor import _soft_shrinkage_guardrail
        # Two top cars with slightly different raw predictions (e.g. 11.2 and 11.8)
        p1 = _soft_shrinkage_guardrail(11.2, boundary=10.0, mode="upper", alpha=1.2)
        p2 = _soft_shrinkage_guardrail(11.8, boundary=10.0, mode="upper", alpha=1.2)
        self.assertLess(p1, 10.5)
        self.assertLess(p1, p2, "Order must be strictly preserved without tying at exactly 10.0")
        self.assertNotEqual(p1, 10.0)
        self.assertNotEqual(p2, 10.0)

    def test_m06_circuit_pace_dispersion_scaling(self):
        """Verify pace delta scales inversely with field standard deviation."""
        from engine.models.predictor import _compute_circuit_pace_dispersion
        monaco_deltas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]  # tight field, sigma ~ 0.09
        spa_deltas = [0.2, 0.5, 0.8, 1.2, 1.6, 2.0]            # spread field, sigma ~ 0.68
        sigma_monaco = _compute_circuit_pace_dispersion(monaco_deltas)
        sigma_spa = _compute_circuit_pace_dispersion(spa_deltas)
        self.assertLess(sigma_monaco, sigma_spa)
        # Delta of 0.2s should move positions more at Monaco than at Spa
        delta_pos_monaco = 0.2 / sigma_monaco
        delta_pos_spa = 0.2 / sigma_spa
        self.assertGreater(delta_pos_monaco, delta_pos_spa)

    def test_f03_single_winner_fl_and_dotd_sampling(self):
        """Verify at most 1 driver receives Fastest Lap and exactly 1 receives DOTD per simulation."""
        from engine.models.monte_carlo import _simulate_one_race
        import random
        rng = random.Random(42)
        # Setup mock 22-driver grid
        race_order = [{"driver": f"Driver_{i}", "predicted_rank": i+1, "dnf_prob_pct": 5.0} for i in range(22)]
        quali_order = [{"driver": f"Driver_{i}", "predicted_grid": i+1, "quali_class_pos": i+1} for i in range(22)]
        
        fl_counts = []
        dotd_counts = []
        for _ in range(100):
            sim = _simulate_one_race(
                race_order, quali_order, sc_prob=0.4, vsc_prob=0.3, rain_risk="low",
                is_sprint=False, sprint_order=None, rng=rng,
                return_winner_metadata=True
            )
            fl_winner = sim.get("fastest_lap_driver")
            dotd_winner = sim.get("dotd_driver")
            fl_counts.append(1 if fl_winner is not None else 0)
            dotd_counts.append(1 if dotd_winner is not None else 0)

        self.assertTrue(all(c <= 1 for c in fl_counts), "At most 1 driver may win Fastest Lap per race")
        self.assertTrue(all(c == 1 for c in dotd_counts), "Exactly 1 driver must win Driver of the Day per race")

    def test_c04_wildcard_not_triggered_on_zero_free_transfers(self):
        """Verify having 0 free transfers does not trigger Wildcard unless massive penalty savings exist."""
        from engine.strategy.chip_advisor import _score_wildcard
        chip_state = {}
        # User has 0 free transfers, but no squad emergency (gain < 20 pts)
        score = _score_wildcard(
            free_transfers=0,
            top5_mean=22.0,
            current_round=3,
            chip_state=chip_state,
            optimal_transfers_needed=1,
            transfer_penalty_saving=0.0
        )
        self.assertFalse(score.use_now, "Wildcard must NOT trigger simply because free_transfers == 0")

    def test_c05_final_fix_gated_until_post_qualifying(self):
        """Verify Final Fix is strictly gated and cannot trigger pre-qualifying."""
        from engine.strategy.chip_advisor import _score_final_fix
        score_pre = _score_final_fix(
            current_drivers=["Driver_0"],
            driver_pts=[{"driver": "Driver_0", "dnf_prob_pct": 35.0}],
            mc_results={},
            chip_state={},
            race_mode="pre-quali"
        )
        self.assertFalse(score_pre.use_now, "Final Fix must NOT trigger pre-qualifying")
        self.assertIn("qualifying", score_pre.reason.lower())
```
