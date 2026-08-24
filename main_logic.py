import logging
import time
from pathlib import Path
import json

from config import CONSTRUCTORS_2025, CURRENT_SEASON, SC_PROBABILITY
from data_fetcher import get_next_race, get_race_by_round, is_sprint_weekend, qualifying_has_happened
from weather import get_race_weekend_weather
from fantasy_scraper import scrape_driver_prices, scrape_constructor_prices
from predictor import F1Predictor
from fantasy_optimizer import (
    suggest_team_changes, find_optimal_team, find_best_constructor, 
    find_differential_picks, find_best_turbo_driver
)
from monte_carlo import simulate_race_weekend, get_expected_value_pts
from chip_advisor import advise_chips, load_chip_state

logger = logging.getLogger(__name__)

def run_full_pipeline(run_id: str, my_drivers: list, my_constructors: list, budget: float, points: float, transfers: int, options: dict, progress_callback):
    try:
        # 1. Next Race
        progress_callback(run_id, "NEXT_RACE", "loading", "Detecting target race...")
        selected_round = options.get("race_round")
        if selected_round:
            race = get_race_by_round(int(selected_round), CURRENT_SEASON)
        else:
            race = get_next_race(CURRENT_SEASON)
        if not race:
            progress_callback(run_id, "NEXT_RACE", "error", "Could not detect race.")
            return None
        try:
            from config import get_enriched_circuit_config
            circuit_cfg = get_enriched_circuit_config(race.get("name", ""))
            if not circuit_cfg:
                circuit_cfg = race.get("circuit_config", {})
        except Exception:
            circuit_cfg = race.get("circuit_config", {})
        is_sprint = is_sprint_weekend(race)
        
        user_mode = options.get("mode", "auto")
        if user_mode == "auto":
            detected_mode = "post-quali" if qualifying_has_happened(race) else "pre-quali"
        else:
            detected_mode = user_mode
            
        progress_callback(run_id, "NEXT_RACE", "done", f"Found: {race['name']}", data={
            "race_name": race["name"],
            "round": race["round"],
            "season": CURRENT_SEASON,
            "circuit_type": circuit_cfg.get("track_type", "N/A").upper(),
            "downforce": circuit_cfg.get("downforce", "N/A").upper(),
            "overtaking": circuit_cfg.get("overtaking", "N/A").upper(),
            "mode": detected_mode
        })

        # 2. Weather
        progress_callback(run_id, "WEATHER", "loading", "Fetching weather...")
        weather = get_race_weekend_weather(race["name"], race["date"])
        
        # Safely extract temp, rain_risk, and summary from the weather dict to avoid NoneType errors
        temp_val = weather.get("temp_c")
        if temp_val is None or temp_val == "":
            temp_val = weather.get("temp_day_c")
        if temp_val is None or temp_val == "":
            sessions = weather.get("sessions", {})
            race_session = sessions.get("Race", {})
            temp_val = race_session.get("temp_day_c") or race_session.get("temp_c")
        if temp_val is None or temp_val == "":
            temp_val = "N/A"
            
        rain_risk_val = weather.get("rain_risk")
        if rain_risk_val is None or rain_risk_val == "":
            rain_risk_val = "N/A"
            
        summary_val = weather.get("summary_condition")
        if summary_val is None or summary_val == "":
            summary_val = "N/A"

        progress_callback(run_id, "WEATHER", "done", "Weather loaded", data={
            "temp": temp_val,
            "rain_risk": str(rain_risk_val).upper(),
            "summary": str(summary_val).upper()
        })

        # 3. Prices
        progress_callback(run_id, "PRICES", "loading", "Fetching market prices...")
        refresh = options.get("refresh", False)
        driver_prices, dynamic_roster = scrape_driver_prices(force_refresh=refresh)
        constructor_prices = scrape_constructor_prices(force_refresh=refresh)
        
        # Apply manual overrides
        overrides = options.get("overrides", {})
        for drv, data in overrides.items():
            if drv in driver_prices: 
                driver_prices[drv]["price"] = float(data.get("price"))
            elif drv in constructor_prices: 
                constructor_prices[drv]["price"] = float(data.get("price"))

        progress_callback(run_id, "PRICES", "done", f"{len(driver_prices)} drivers loaded")

        # 4. ML Model
        progress_callback(run_id, "ML_MODEL", "loading", "Training ML ensemble...")
        predictor = F1Predictor()
        
        def ml_train_progress(msg, data=None):
            progress_callback(run_id, "ML_MODEL", "loading", msg, data=data)
            
        predictor.train(verbose=False, force=refresh, progress_callback=ml_train_progress)
        grid_overrides = options.get("grid_overrides", {})
        predictor.load_context(circuit_cfg, weather, roster=dynamic_roster, mode=detected_mode, race_info=race, grid_overrides=grid_overrides)
        
        # Safely extract Ridge meta-learner coefficients for the UI payload
        meta_coefs = [0.33, 0.33, 0.34]
        if hasattr(predictor, "race_meta_model") and hasattr(predictor.race_meta_model, "coef_"):
            meta_coefs = predictor.race_meta_model.coef_

        progress_callback(run_id, "ML_MODEL", "done", "Model ready", data={
            "rf_weight": float(round(meta_coefs[0], 2)),
            "xgb_weight": float(round(meta_coefs[1], 2)),
            "lgb_weight": float(round(meta_coefs[2], 2))
        })

        # 5. Predictions
        progress_callback(run_id, "PREDICTIONS", "loading", "Generating race predictions...")
        race_order = predictor.predict_finishing_order()
        quali_order = predictor.predict_qualifying_order()
        sprint_order = predictor.predict_sprint_order() if is_sprint else None

        # Send predicted Pole and predicted Winner info before Monte Carlo runs
        pole_sitter = quali_order[0]["driver"] if quali_order else "N/A"
        predicted_winner = race_order[0]["driver"] if race_order else "N/A"
        progress_callback(run_id, "PREDICTIONS", "loading", f"Pole predicted: {pole_sitter} | Winner predicted: {predicted_winner}")

        driver_pts = [predictor.estimate_fantasy_points(d["driver"], race_order, quali_order, is_sprint, sprint_order) for d in race_order]
        ctor_pts = [predictor.estimate_constructor_points(ctor, driver_pts, is_sprint) for ctor in CONSTRUCTORS_2025]

        # 6. Monte Carlo
        sims = options.get("sims", 1000)
        
        def mc_progress(sims_done, sims_total, top_ev_str):
            progress_callback(run_id, "PREDICTIONS", "loading", f"Simulating {sims_done} / {sims_total} runs...", data={
                "sims_done": sims_done,
                "sims_total": sims_total,
                "top_ev": top_ev_str
            })
            
        mc_results = simulate_race_weekend(
            race_order, quali_order, circuit_cfg, weather, sims, is_sprint, sprint_order,
            progress_callback=mc_progress
        )
        mc_ev_pts = get_expected_value_pts(mc_results)
        from config import DRIVER_TEAMS_2025, AVG_PIT_STOP_TEAM_POINTS
        driver_dotd_lookup = {d["driver"]: d["breakdown"].get("driver_of_day", 0.0) for d in driver_pts}
        for ctor in CONSTRUCTORS_2025:
            drv_list = [d for d, c in DRIVER_TEAMS_2025.items() if c == ctor]
            pit_pts = AVG_PIT_STOP_TEAM_POINTS.get(ctor, 7)
            ctor_ev = sum(mc_ev_pts.get(d, 0.0) - driver_dotd_lookup.get(d, 0.0) for d in drv_list) + pit_pts
            mc_ev_pts[ctor] = ctor_ev
        progress_callback(run_id, "PREDICTIONS", "done", "Predictions complete", data={
            "sims": sims,
            "sc_prob": int(SC_PROBABILITY.get(circuit_cfg.get("key"), 0.40) * 100)
        })

        # 7. Analysis
        progress_callback(run_id, "ANALYSIS", "loading", "Analyzing optimal moves (3-race lookahead)...")
        
        # Calculate 3-race lookahead Expected Value
        lookahead_ev_pts = calculate_lookahead_ev(
            predictor, race.get("round", 1), options, sims,
            run_id=run_id, progress_callback=progress_callback
        ) if race.get("round") else mc_ev_pts
        
        # Restore predictor context for the current race just in case
        predictor.load_context(circuit_cfg, weather, roster=dynamic_roster, mode=detected_mode, race_info=race, grid_overrides=grid_overrides)

        progress_callback(run_id, "ANALYSIS", "loading", "Running LP solver to find globally optimal team combinations...")
        if detected_mode in ["post-quali", "race-day"]:
            suggestions = {"suggested_changes": [], "locked": True, "using_mc_pts": True}
        else:
            suggestions = suggest_team_changes(my_drivers, my_constructors, driver_pts, ctor_pts, driver_prices, constructor_prices, transfers, budget, mc_pts=lookahead_ev_pts)
        optimal = find_optimal_team(driver_pts, ctor_pts, driver_prices, constructor_prices, mc_pts=lookahead_ev_pts)
        best_ctors = find_best_constructor(ctor_pts, constructor_prices)

        progress_callback(run_id, "ANALYSIS", "done", "Analysis complete")

        # Chip advisor
        chip_state = load_chip_state()
        chip_scores = advise_chips(
            race_info=race,
            circuit_config=circuit_cfg,
            weather=weather,
            driver_pts=driver_pts,
            mc_results=None,  # pass mc_results if DistributionStats serializable
            current_drivers=my_drivers,
            free_transfers=transfers,
            chip_state=chip_state,
        )
        # Serialize ChipScore dataclasses to plain dicts for JSON transport
        chips_serialized = [{
            "chip":         cs.chip,
            "raw_score":    cs.raw_score,
            "use_now":      cs.use_now,
            "urgency":      cs.urgency,
            "reason":       cs.reason,
            "hold_until":   cs.hold_until,
            "expected_gain": cs.expected_gain,
            "already_used": cs.already_used,
            "used_on":      cs.used_on,
            "target_driver": cs.target_driver,
        } for cs in chip_scores]

        # Combine results
        results = {
            "race": race,
            "weather": weather,
            "prices": {"drivers": driver_prices, "constructors": constructor_prices},
            "predictions": {
                "race_order":   race_order,
                "quali_order":  quali_order,
                "sprint_order": sprint_order,
                "driver_pts":   driver_pts,
                "ctor_pts":     ctor_pts,
            },
            "practice_data": {
                "session":  getattr(predictor, "_practice_session_name", "N/A"),
                "deltas":   getattr(predictor, "_practice_pace", {}),
            },
            "suggestions": suggestions,
            "optimal": optimal,
            "best_ctors": best_ctors,
            "chips": chips_serialized,
            "is_sprint": is_sprint,
        }

        # Save prediction file for post-race analysis
        try:
            import datetime as dt
            report = {
                "generated_at":     dt.datetime.now().isoformat(),
                "race":             race,
                "weather_summary":  {
                    k: v for k, v in weather.items() if k != "race_day_hourly"
                },
                "qualifying_order": quali_order[:20] if quali_order else [],
                "race_order":       race_order[:20] if race_order else [],
                "driver_pts":       sorted(driver_pts, key=lambda x: x.get("total_pts", 0), reverse=True),
                "constructor_pts":  sorted(ctor_pts, key=lambda x: x.get("total_pts", 0), reverse=True),
                "suggestions":      suggestions,
                "optimal_team":     optimal,
            }
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            fname = output_dir / f"race_{race.get('round', 'X')}_{race.get('name', 'unknown').replace(' ', '_')}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"UI run saved prediction file to {fname}")
        except Exception as e:
            logger.error(f"Failed to save prediction file in UI run: {e}")

        progress_callback(run_id, "COMPLETE", "done", "All tasks complete", data=results)
        return results

    except Exception as e:
        logger.error("Pipeline failed", exc_info=True)
        progress_callback(run_id, "ERROR", "error", str(e))
        return None
import copy

def calculate_lookahead_ev(predictor, current_race_round, options, sims, run_id=None, progress_callback=None):
    from data_fetcher import get_race_by_round, is_sprint_weekend
    from weather import get_race_weekend_weather
    from config import CURRENT_SEASON, CONSTRUCTORS_2025, DRIVER_TEAMS_2025, AVG_PIT_STOP_TEAM_POINTS
    from monte_carlo import simulate_race_weekend, get_expected_value_pts
    
    total_ev = {}
    
    # We will do 3 races: round, round+1, round+2
    for r in range(current_race_round, current_race_round + 3):
        race = get_race_by_round(r, CURRENT_SEASON)
        if not race:
            break
            
        if progress_callback and run_id:
            progress_callback(run_id, "ANALYSIS", "loading", f"Simulating Round {r} lookahead ({race['name'].replace(' Grand Prix', '')})...")
            
        is_sprint = is_sprint_weekend(race)
        try:
            from config import get_enriched_circuit_config
            circuit_cfg = get_enriched_circuit_config(race.get("name", ""))
            if not circuit_cfg:
                circuit_cfg = race.get("circuit_config", {})
        except Exception:
            circuit_cfg = race.get("circuit_config", {})
        weather = get_race_weekend_weather(race["name"], race["date"])
        
        # Load context for this future race
        predictor.load_context(circuit_cfg, weather, mode="pre-quali", race_info=race)
        
        race_order = predictor.predict_finishing_order()
        quali_order = predictor.predict_qualifying_order()
        sprint_order = predictor.predict_sprint_order() if is_sprint else None
        
        # Pass None as progress_callback for lookahead MC to keep it quiet and fast
        mc_results = simulate_race_weekend(race_order, quali_order, circuit_cfg, weather, sims, is_sprint, sprint_order)
        ev_pts = get_expected_value_pts(mc_results)
        
        for name, points in ev_pts.items():
            total_ev[name] = total_ev.get(name, 0.0) + points
            
        # Calculate constructor expected values for this round
        from config import DRIVER_OF_DAY_BONUS
        for ctor in CONSTRUCTORS_2025:
            drv_list = [d for d, c in DRIVER_TEAMS_2025.items() if c == ctor]
            pit_pts = AVG_PIT_STOP_TEAM_POINTS.get(ctor, 7)
            ctor_ev = 0.0
            for d in drv_list:
                d_race = next((x for x in race_order if x["driver"] == d), None)
                if d_race:
                    rank = d_race["predicted_rank"]
                    dnf_prob = d_race["dnf_prob_pct"] / 100.0
                    dotd_prob = max(0.0, 0.12 - (rank - 1) * 0.01) * (1.0 - dnf_prob)
                    dotd_ev = DRIVER_OF_DAY_BONUS * dotd_prob
                else:
                    dotd_ev = 0.0
                ctor_ev += ev_pts.get(d, 0.0) - dotd_ev
            ctor_ev += pit_pts
            total_ev[ctor] = total_ev.get(ctor, 0.0) + ctor_ev
            
    return total_ev

