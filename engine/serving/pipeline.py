import logging
import time
from pathlib import Path
import json

from engine.core.config import CONSTRUCTORS_2026, CONSTRUCTORS_2025, CURRENT_SEASON, SC_PROBABILITY
from engine.core.data_fetcher import get_next_race, get_race_by_round, is_sprint_weekend, qualifying_has_happened
from engine.core.weather import get_race_weekend_weather
from engine.core.fantasy_scraper import scrape_driver_prices, scrape_constructor_prices
from engine.models.predictor import F1Predictor
from engine.strategy.fantasy_optimizer import (
    suggest_team_changes, find_optimal_team, find_best_constructor, 
    find_differential_picks, find_best_turbo_driver
)
from engine.models.monte_carlo import simulate_race_weekend, get_expected_value_pts
from engine.strategy.chip_advisor import advise_chips, load_chip_state

logger = logging.getLogger(__name__)

def run_pipeline(options: dict = None, **kwargs):
    """Convenience entrypoint for pipeline execution supporting options dict."""
    opts = options or {}
    return run_full_pipeline(
        run_id=kwargs.get("run_id", "pipeline_run"),
        my_drivers=kwargs.get("my_drivers", []),
        my_constructors=kwargs.get("my_constructors", []),
        budget=float(kwargs.get("budget", 100.0)),
        points=float(kwargs.get("points", 0.0)),
        transfers=int(kwargs.get("transfers", 1)),
        options=opts,
        progress_callback=kwargs.get("progress_callback"),
    )


def run_full_pipeline(
    run_id: str = "default",
    my_drivers: list = None,
    my_constructors: list = None,
    budget: float = 100.0,
    points: float = 0.0,
    transfers: int = 1,
    options: dict = None,
    progress_callback = None,
):
    if my_drivers is None:
        my_drivers = []
    if my_constructors is None:
        my_constructors = []
    if options is None:
        options = {}
    if progress_callback is None:
        progress_callback = lambda *args, **kwargs: None

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
            from engine.core.config import get_enriched_circuit_config
            circuit_cfg = get_enriched_circuit_config(race.get("name", ""))
            if not circuit_cfg:
                circuit_cfg = race.get("circuit_config", {})
        except Exception:
            circuit_cfg = race.get("circuit_config", {})
        is_sprint = is_sprint_weekend(race)
        
        user_mode = options.get("mode") or options.get("stage") or "auto"
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
        weather = get_race_weekend_weather(race["name"], race["date"], race_info=race)
        
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

        # Normalize fantasy-site team spellings ("Red Bull Racing" → "Red Bull")
        try:
            from engine.core.data_fetcher import _normalize_constructor_name
            for info in driver_prices.values():
                if info.get("team"):
                    info["team"] = _normalize_constructor_name(info["team"])
            dynamic_roster = {
                n: _normalize_constructor_name(t) for n, t in dynamic_roster.items()
            }
        except Exception as e:
            logger.debug("Team-name normalization skipped: %s", e)

        # Apply manual overrides
        overrides = options.get("overrides", {})
        for drv, data in overrides.items():
            if drv in driver_prices:
                driver_prices[drv]["price"] = float(data.get("price"))
            elif drv in constructor_prices:
                constructor_prices[drv]["price"] = float(data.get("price"))

        progress_callback(run_id, "PRICES", "done", f"{len(driver_prices)} drivers loaded")
        # Record price/ownership snapshots so web-UI users accumulate movement
        # history too (previously only the CLI recorded these).
        try:
            import engine.strategy.price_tracker as price_tracker
            price_tracker.record_prices(race.get("round", 0), driver_prices, constructor_prices)
            price_tracker.record_ownership(race.get("round", 0), driver_prices, constructor_prices)
        except Exception as e:
            logger.warning("Price snapshot recording failed: %s", e)

        # Cross-check the scraped fantasy roster against Jolpica standings.
        #
        # FIELD AUTHORITY PRECEDENCE (plan I7b):
        #   1. The fantasy game entry list (scrape) defines who can score.
        #   2. Gaps are backfilled ONLY from the curated 22-seat config seed.
        #   3. Standings contribute TEAM ATTRIBUTION only — never new names.
        # Standings-only extras (drivers who lost seats but remain in season
        # standings) are EXCLUDED as phantoms; a blind union previously let a
        # phantom 23rd driver into predictions.
        try:
            from engine.core.config import DRIVER_TEAMS_2026
            from engine.core.data_fetcher import get_season_roster, compare_rosters
            api_roster = get_season_roster(CURRENT_SEASON)
            if dynamic_roster:
                seed_names = set(DRIVER_TEAMS_2026)
                accepted = seed_names
                # Phantom exclusion: standings/scrape extras not in seed
                excluded_phantoms = sorted((set(dynamic_roster) | set(api_roster or {})) - accepted)
                for name in excluded_phantoms:
                    logger.warning("Excluded phantom driver: %s", name)

                # Team attribution: curated seed wins; API fills unknowns
                merged_roster: dict[str, str] = {}
                for name in accepted:
                    merged_roster[name] = (
                        DRIVER_TEAMS_2026.get(name)
                        or (api_roster or {}).get(name)
                        or dynamic_roster.get(name, "")
                    )
                drift = compare_rosters(
                    {n: t for n, t in dynamic_roster.items() if n in merged_roster},
                    {n: t for n, t in (api_roster or {}).items() if n in merged_roster},
                    "fantasy-scrape", "jolpica-standings",
                ) if api_roster else []
                if drift:
                    progress_callback(run_id, "PRICES", "loading",
                                      f"Roster drift detected ({len(drift)} differences) — see logs")
                if excluded_phantoms:
                    logger.warning(
                        "Fantasy scrape missing %d drivers — backfilled/validated: "
                        "field=%d, phantoms_excluded=%s",
                        len(excluded_phantoms), len(merged_roster), excluded_phantoms,
                    )
                dynamic_roster = merged_roster

                # Hard integrity gate (plan I7b.5)
                n_field = len(dynamic_roster)
                if not (18 <= n_field <= 24):
                    raise RuntimeError(
                        f"Field integrity failure: {n_field} drivers "
                        "(expected 18–24). Refusing to predict."
                    )
                if n_field != 22:
                    logger.warning("Unusual field size %d (calendar standard is 22)", n_field)
                progress_callback(run_id, "PRICES", "loading",
                                  f"FIELD_VALIDATION size={n_field} "
                                  f"backfilled={max(0, n_field - len(driver_prices))} "
                                  f"phantoms_excluded={len(excluded_phantoms)}")
                # Plan I2: report the VALIDATED field size, not the price-table size
                gap_note = (f" ({n_field - len(driver_prices)} backfilled)"
                            if n_field > len(driver_prices) else "")
                progress_callback(run_id, "PRICES", "done",
                                  f"Prices for {len(driver_prices)} drivers · "
                                  f"predicting field of {n_field}{gap_note}")
            elif api_roster and not dynamic_roster:
                dynamic_roster = dict(DRIVER_TEAMS_2026)
        except RuntimeError:
            raise
        except Exception as e:
            logger.debug("Roster cross-check skipped: %s", e)

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
            "lgb_weight": float(round(meta_coefs[2], 2)),
            # Plan I4: telemetry payload
            "device": getattr(predictor, "_train_device", "unknown"),
            "dataset_samples": getattr(predictor, "_last_dataset_size", None),
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

        active_ctors = list(CONSTRUCTORS_2026 if CURRENT_SEASON >= 2026 else CONSTRUCTORS_2025)
        driver_pts = [predictor.estimate_fantasy_points(d["driver"], race_order, quali_order, is_sprint, sprint_order) for d in race_order]
        ctor_pts = [predictor.estimate_constructor_points(ctor, driver_pts, is_sprint) for ctor in active_ctors]

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
            progress_callback=mc_progress,
            grid_penalties=getattr(predictor, "_grid_penalties", {})
        )
        mc_ev_pts = get_expected_value_pts(mc_results)
        from engine.core.config import DRIVER_TEAMS_2026, DRIVER_TEAMS_2025, AVG_PIT_STOP_TEAM_POINTS
        driver_dotd_lookup = {d["driver"]: d["breakdown"].get("driver_of_day", 0.0) for d in driver_pts}
        active_driver_teams = dict(DRIVER_TEAMS_2026 if CURRENT_SEASON >= 2026 else DRIVER_TEAMS_2025)
        if dynamic_roster:
            active_driver_teams.update(dynamic_roster)
        for ctor in active_ctors:
            drv_list = [d for d, c in active_driver_teams.items() if c == ctor]
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
            run_id=run_id, progress_callback=progress_callback,
            roster=dynamic_roster, grid_overrides=grid_overrides,
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
            "hold_until_round": getattr(cs, "hold_until_round", None),
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
                "stage":            detected_mode,
                "season":           CURRENT_SEASON,   # plan 9-C2: explicit season for validators
                "race":             {**race, "season": CURRENT_SEASON},
                "weather_summary":  {
                    k: v for k, v in weather.items() if k != "race_day_hourly"
                },
                # Plan 9-C4: NO truncation — every predicted driver must appear
                # in post-race comparison, not just the first 20.
                "qualifying_order": quali_order if quali_order else [],
                "race_order":       race_order if race_order else [],
                "driver_pts":       sorted(driver_pts, key=lambda x: x.get("total_pts", 0), reverse=True),
                "constructor_pts":  sorted(ctor_pts, key=lambda x: x.get("total_pts", 0), reverse=True),
                "suggestions":      suggestions,
                "optimal_team":     optimal,
            }
            from engine.core.paths import OUTPUT_DIR
            output_dir = OUTPUT_DIR
            output_dir.mkdir(exist_ok=True)
            round_val = race.get("round", "X")
            name_val = race.get("name", "unknown").replace(" ", "_")
            fname = output_dir / f"race_{round_val}_{name_val}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"UI run saved prediction file to {fname}")

            stages_dir = output_dir / "stages"
            stages_dir.mkdir(parents=True, exist_ok=True)
            stage_fname = stages_dir / f"race_{round_val}_{name_val}_{detected_mode}.json"
            with open(stage_fname, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"UI run saved stage prediction file to {stage_fname}")
        except Exception as e:
            logger.error(f"Failed to save prediction file in UI run: {e}")

        progress_callback(run_id, "COMPLETE", "done", "All tasks complete", data=results)
        return results

    except Exception as e:
        logger.error("Pipeline failed", exc_info=True)
        progress_callback(run_id, "ERROR", "error", str(e))
        return None


def calculate_lookahead_ev(
    predictor, current_race_round, options, sims,
    run_id=None, progress_callback=None,
    roster=None, grid_overrides=None,
    season=None,
):
    """
    Sum of MC expected values over the next 3 races with temporal discounting.

    NOTE on scales: the returned totals are PER-RACE WEIGHTED AVERAGES (gamma=0.75),
    not 3-race sums — the transfer optimizer and dream-team LP operate on single-race
    point scales ($-per-point tradeoffs), so feeding them a 3x-summed EV distorted value
    rankings by construction.
    """
    from engine.core.data_fetcher import get_race_by_round, is_sprint_weekend
    from engine.core.weather import get_race_weekend_weather
    from engine.core.config import (
        CURRENT_SEASON, CONSTRUCTORS_2026, CONSTRUCTORS_2025,
        DRIVER_TEAMS_2026, DRIVER_TEAMS_2025, AVG_PIT_STOP_TEAM_POINTS,
        DRIVER_OF_DAY_BONUS
    )
    from engine.models.monte_carlo import simulate_race_weekend, get_expected_value_pts
    
    if season is None:
        season = CURRENT_SEASON

    constructors = list(CONSTRUCTORS_2026 if season >= 2026 else CONSTRUCTORS_2025)
    base_driver_teams = dict(DRIVER_TEAMS_2026 if season >= 2026 else DRIVER_TEAMS_2025)
    if roster:
        if isinstance(roster, dict):
            for d, t in roster.items():
                if t:
                    base_driver_teams[d] = t
                    if t not in constructors:
                        constructors.append(t)
        elif isinstance(roster, list):
            for d in roster:
                if isinstance(d, dict) and "driver" in d and d.get("team"):
                    base_driver_teams[d["driver"]] = d["team"]
                    if d["team"] not in constructors:
                        constructors.append(d["team"])

    total_ev = {}
    rounds_simulated = 0
    total_weight = 0.0
    gamma = 0.75  # Exponential temporal discount factor per race lookahead step

    # We will do 3 races: round, round+1, round+2
    for offset, r in enumerate(range(current_race_round, current_race_round + 3)):
        race = get_race_by_round(r, season)
        if not race:
            break
        weight = gamma ** offset
        rounds_simulated += 1
        total_weight += weight
            
        if progress_callback and run_id:
            progress_callback(run_id, "ANALYSIS", "loading", f"Simulating Round {r} lookahead ({race['name'].replace(' Grand Prix', '')})...")
            
        is_sprint = is_sprint_weekend(race)
        try:
            from engine.core.config import get_enriched_circuit_config
            circuit_cfg = get_enriched_circuit_config(race.get("name", ""))
            if not circuit_cfg:
                circuit_cfg = race.get("circuit_config", {})
        except Exception:
            circuit_cfg = race.get("circuit_config", {})
        weather = get_race_weekend_weather(race["name"], race["date"], race_info=race)
        
        # Load context for this future race.
        # Future rounds do NOT inherit current-round grid penalties/overrides.
        round_grid_overrides = grid_overrides if r == current_race_round else None
        predictor.load_context(
            circuit_cfg, weather, roster=roster,
            mode="pre-quali", race_info=race, grid_overrides=round_grid_overrides,
        )
        
        race_order = predictor.predict_finishing_order()
        quali_order = predictor.predict_qualifying_order()
        sprint_order = predictor.predict_sprint_order() if is_sprint else None
        
        # Pass None as progress_callback for lookahead MC to keep it quiet and fast.
        # Future rounds do not inherit current-round grid penalties.
        mc_results = simulate_race_weekend(
            race_order, quali_order, circuit_cfg, weather, sims, is_sprint, sprint_order,
            grid_penalties={}
        )
        ev_pts = get_expected_value_pts(mc_results)
        
        for name, points in ev_pts.items():
            total_ev[name] = total_ev.get(name, 0.0) + points * weight
            
        # Calculate constructor expected values for this round.
        # ev_pts already contains MC-integrated driver points (including sprint when is_sprint=True).
        # Constructor points = sum of both drivers' EV minus DOTD (driver-only bonus) + pit stop bonus.
        for ctor in constructors:
            drv_list = [d for d, c in base_driver_teams.items() if c == ctor]
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
            total_ev[ctor] = total_ev.get(ctor, 0.0) + ctor_ev * weight

    # Normalize by total discount weight to obtain single-race expected value scale
    if total_weight > 0.0:
        total_ev = {name: pts / total_weight for name, pts in total_ev.items()}

    return total_ev

