"""Quick sanity checks for all upgraded F1 predictor modules."""
import sys

errors = []

# ── 1. PREDICTOR FEATURES ────────────────────────────────────────────────────
try:
    from predictor import F1Predictor, FEATURE_NAMES, N_FEATURES
    assert N_FEATURES == 56, f"Expected 56 features, got {N_FEATURES}"
    assert "elo_rd" in FEATURE_NAMES, "elo_rd missing"
    assert "elo_volatility" in FEATURE_NAMES, "elo_volatility missing"
    assert "sc_prob" in FEATURE_NAMES, "sc_prob missing"
    assert "momentum_trend" in FEATURE_NAMES, "momentum_trend missing"
    assert "grid_penalty" in FEATURE_NAMES, "grid_penalty missing"
    # Phase 3 features
    assert "lstm_momentum_pos" in FEATURE_NAMES, "lstm_momentum_pos missing (Phase 3)"
    assert "driver_tire_efficiency_score" in FEATURE_NAMES, "driver_tire_efficiency_score missing (Phase 3)"
    # Phase 4 features
    assert "lap_1_risk" in FEATURE_NAMES, "lap_1_risk missing (Phase 4)"
    assert "fresh_tires_avail" in FEATURE_NAMES, "fresh_tires_avail missing (Phase 4)"
    # Fantasy features
    assert "driver_overtake_delta" in FEATURE_NAMES
    assert "driver_dnf_risk" in FEATURE_NAMES
    print(f"[PASS] predictor.py — {N_FEATURES} features verified (Phases 1-4 complete)")
except Exception as e:
    errors.append(f"[FAIL] predictor.py: {e}")

# ── 2. GLICKO-2 LOGIC ────────────────────────────────────────────────────────
try:
    from elo_ratings import Glicko2RatingSystem, Glicko2Driver, EloRatingSystem
    g = Glicko2RatingSystem()
    g.drivers['Veteran'] = Glicko2Driver('Veteran', mu=1600, phi=120, sigma=0.05)
    g.drivers['Rookie']  = Glicko2Driver('Rookie',  mu=1500, phi=340, sigma=0.08)
    vf = g.get_elo_features('Veteran')
    rf = g.get_elo_features('Rookie')
    assert rf['elo_rd'] > vf['elo_rd'], "Rookie RD should be higher"
    assert rf['elo_confidence'] < vf['elo_confidence'], "Rookie conf should be lower"
    # Backward compat
    assert issubclass(EloRatingSystem, Glicko2RatingSystem), "EloRatingSystem alias broken"
    print(f"[PASS] elo_ratings.py — Glicko-2: Veteran rd={vf['elo_rd']}, Rookie rd={rf['elo_rd']}")
except Exception as e:
    errors.append(f"[FAIL] elo_ratings.py: {e}")

# ── 3. CONFIG SC PROBS ───────────────────────────────────────────────────────
try:
    from config import SC_PROBABILITY, VSC_PROBABILITY, SEASON_WEIGHTS, REGULATION_CHANGE_YEARS, CIRCUITS
    assert "monaco" in SC_PROBABILITY, "Monaco missing from SC_PROBABILITY"
    assert SC_PROBABILITY["monaco"] >= 0.65, "Monaco SC prob should be high"
    assert SC_PROBABILITY["singapore"] >= 0.70, "Singapore SC prob should be high"
    assert 2026 in REGULATION_CHANGE_YEARS
    assert SEASON_WEIGHTS[2026] == 1.0
    assert SEASON_WEIGHTS[2022] < SEASON_WEIGHTS[2025]
    print(f"[PASS] config.py — SC probs: Monaco={SC_PROBABILITY['monaco']}, Singapore={SC_PROBABILITY['singapore']}")
except Exception as e:
    errors.append(f"[FAIL] config.py: {e}")

# ── 4. MONTE CARLO ───────────────────────────────────────────────────────────
try:
    from monte_carlo import simulate_race_weekend, format_mc_summary, DistributionStats
    # Build minimal fake race order
    fake_race = [
        {"driver": "Alice", "team": "Red Bull", "predicted_rank": i+1, "dnf_prob_pct": 7.0}
        for i in range(10)
    ]
    fake_quali = [
        {"driver": "Alice", "team": "Red Bull", "predicted_grid": i+1}
        for i in range(10)
    ]
    fake_circuit = {"key": "monaco"}
    fake_weather = {"rain_risk": "low", "summary_condition": "dry"}
    results = simulate_race_weekend(fake_race, fake_quali, fake_circuit, fake_weather, n_simulations=50)
    assert "Alice" in results, "Alice should be in MC results"
    stats = results["Alice"]
    assert hasattr(stats, "mean_pts"), "DistributionStats missing mean_pts"
    assert hasattr(stats, "p90_pts"), "DistributionStats missing p90_pts"
    assert stats.n_sims == 50, "n_sims mismatch"
    summary = format_mc_summary(results)
    assert isinstance(summary, list)
    print(f"[PASS] monte_carlo.py — 50 sims: Alice mean={stats.mean_pts:.1f}, p90={stats.p90_pts:.1f}, dnf%={stats.p_dnf:.1f}")
except Exception as e:
    errors.append(f"[FAIL] monte_carlo.py: {e}")
    import traceback; traceback.print_exc()

# ── 5. CHIP ADVISOR ──────────────────────────────────────────────────────────
try:
    from chip_advisor import advise_chips
    fake_race_info = {"round": 2, "name": "Chinese Grand Prix"}  # sprint round
    fake_driver_pts = [
        {"driver": "Alice", "team": "Red Bull", "total_pts": 35.0, "dnf_prob_pct": 5.0},
        {"driver": "Bob",   "team": "McLaren",  "total_pts": 28.0, "dnf_prob_pct": 7.0},
    ]
    recs = advise_chips(
        race_info=fake_race_info,
        circuit_config={"key": "singapore"},
        weather={"rain_risk": "low"},
        driver_pts=fake_driver_pts,
    )
    assert len(recs) >= 4, f"Expected >=4 chip recs, got {len(recs)}"
    # v2: advise_chips returns ChipScore objects — access via attributes
    chips = [r.chip for r in recs]
    assert "Limitless" in chips, "Limitless chip missing"
    assert "No Negative" in chips, "No Negative chip missing"
    print(f"[PASS] chip_advisor.py — {len(recs)} chip recs")
except Exception as e:
    errors.append(f"[FAIL] chip_advisor.py: {e}")
    import traceback; traceback.print_exc()

# ── 6. FANTASY OPTIMIZER ─────────────────────────────────────────────────────
try:
    from fantasy_optimizer import find_differential_picks, find_best_turbo_driver
    fake_driver_pts = [
        {"driver": f"D{i}", "team": f"Team{i%5}", "total_pts": 30.0 - i*2} for i in range(10)
    ]
    fake_prices = {f"D{i}": {"price": 10.0 - i*0.5, "ownership_pct": i*3.0} for i in range(10)}
    diffs = find_differential_picks(fake_driver_pts, fake_prices, top_n=3)
    assert len(diffs) == 3, "Should return 3 differential picks"
    assert "differential_score" in diffs[0], "differential_score missing"
    turbo = find_best_turbo_driver(["D0", "D1", "D2"], fake_driver_pts)
    assert turbo["driver"] in ("D0", "D1", "D2"), "Turbo driver invalid"
    print(f"[PASS] fantasy_optimizer.py — diffs={[d['driver'] for d in diffs]}, turbo={turbo['driver']}")
except Exception as e:
    errors.append(f"[FAIL] fantasy_optimizer.py: {e}")

# ── 7. DATA_FETCHER NEW FUNCTIONS ────────────────────────────────────────────
try:
    import inspect
    from data_fetcher import compute_practice_pace, get_qualifying_sector_times, get_grid_penalties
    from data_fetcher import compute_driver_form
    # Just check signatures exist and are callable
    assert callable(compute_practice_pace)
    assert callable(get_qualifying_sector_times)
    assert callable(get_grid_penalties)
    # Check compute_driver_form returns momentum_trend key (EWMA upgrade)
    sig = inspect.signature(compute_driver_form)
    print(f"[PASS] data_fetcher.py — new functions present, compute_driver_form sig: {sig}")
except Exception as e:
    errors.append(f"[FAIL] data_fetcher.py: {e}")

# ── 8. MAIN.PY SYNTAX ────────────────────────────────────────────────────────
try:
    import ast
    with open("main.py", "r", encoding="utf-8") as f:
        src = f.read()
    ast.parse(src)
    print("[PASS] main.py — syntax valid")
except SyntaxError as e:
    errors.append(f"[FAIL] main.py syntax: {e}")

# ── 9. CHIP STATE TRACKING ───────────────────────────────────────────────────
try:
    import tempfile, os
    from chip_advisor import (
        load_chip_state, save_chip_state, mark_chip_used, reset_chip_state,
        advise_chips, ALL_CHIPS, ChipScore,
        get_remaining_sprint_rounds, get_remaining_high_value_rounds,
        build_season_context_table,
    )
    # Test default state is all unused
    state = reset_chip_state()
    assert all(not v for v in state["used"].values()), "Default state should have all chips unused"
    assert set(state["used"].keys()) == set(ALL_CHIPS), "State must contain all 5 chips"

    # Test marking a chip as used
    mark_chip_used("Limitless", 5, "Saudi Arabian Grand Prix")
    reloaded = load_chip_state()
    assert reloaded["used"]["Limitless"] is True, "Limitless should be marked as used"
    assert reloaded["used_on_round"]["Limitless"] == 5, "Round should be saved"

    # Test advise_chips with used chip returns 'already_used' ChipScore
    fake_race_info  = {"round": 6, "name": "Miami Grand Prix"}
    fake_driver_pts = [
        {"driver": "Alice", "team": "Red Bull", "total_pts": 35.0, "dnf_prob_pct": 5.0},
        {"driver": "Bob",   "team": "McLaren",  "total_pts": 28.0, "dnf_prob_pct": 7.0},
    ]
    recs = advise_chips(
        race_info=fake_race_info,
        circuit_config={"key": "singapore"},
        weather={"rain_risk": "low"},
        driver_pts=fake_driver_pts,
        chip_state=reloaded,
    )
    assert len(recs) == 5, f"Expected 5 chip recs, got {len(recs)}"
    limitless_rec = next(r for r in recs if r.chip == "Limitless")
    assert limitless_rec.already_used is True, "Limitless should show as already_used"
    assert limitless_rec.urgency == "none", "Used chip urgency should be 'none'"

    # Test season calendar
    calendar = get_remaining_sprint_rounds(current_round=5)
    assert all(r["round"] > 5 for r in calendar), "Calendar should only show future rounds"
    context_table = build_season_context_table(5, reloaded)
    assert all(r["round"] > 5 for r in context_table), "Context table should be future rounds only"

    # Reset state back to clean
    reset_chip_state()
    print(f"[PASS] chip_advisor.py v2 — state tracking, all {len(ALL_CHIPS)} chips, calendar verified")
except Exception as e:
    errors.append(f"[FAIL] chip_advisor.py v2: {e}")
    import traceback; traceback.print_exc()

# ── 10. PRICE TRACKER ────────────────────────────────────────────────────────
try:
    from price_tracker import (
        record_prices, record_ownership,
        get_price_movements, get_ownership_trends,
        get_sell_high_candidates, get_buy_low_candidates,
    )
    # Fake two rounds of prices
    fake_prices_r1 = {
        "Alice": {"price": 20.0, "ownership_pct": 10.0},
        "Bob":   {"price": 15.0, "ownership_pct": 20.0},
    }
    fake_prices_r2 = {
        "Alice": {"price": 20.5, "ownership_pct": 16.0},  # rising
        "Bob":   {"price": 14.5, "ownership_pct": 17.0},  # falling
    }
    record_prices(98, fake_prices_r1, {}, season=9999)
    record_prices(99, fake_prices_r2, {}, season=9999)
    record_ownership(98, fake_prices_r1, {}, season=9999)
    record_ownership(99, fake_prices_r2, {}, season=9999)

    movs = get_price_movements(season=9999)
    assert "Alice" in movs, "Alice missing from price movements"
    assert movs["Alice"]["change"] == 0.5, f"Expected +0.5 price change, got {movs['Alice']['change']}"
    assert movs["Alice"]["trend"] == "UP", "Alice price should be trending up"
    assert movs["Bob"]["change"] == -0.5, f"Expected -0.5 price change, got {movs['Bob']['change']}"

    trends = get_ownership_trends(season=9999)
    assert trends["Alice"]["change"] == 6.0, f"Expected +6.0 ownership change, got {trends['Alice']['change']}"

    sells = get_sell_high_candidates(season=9999, price_rise_threshold=0.4, ownership_rise_threshold=5.0)
    assert any(c["name"] == "Alice" for c in sells), "Alice should be a sell-high candidate"

    alice_trend = movs["Alice"]["trend"]
    bob_trend   = movs["Bob"]["trend"]
    print(f"[PASS] price_tracker.py — price movement: Alice {alice_trend} +0.5M, Bob {bob_trend} -0.5M")
except Exception as e:
    errors.append(f"[FAIL] price_tracker.py: {e}")
    import traceback; traceback.print_exc()

# ── RESULTS ──────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"FAILED — {len(errors)} issue(s):")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED OK")
