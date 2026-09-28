import os
import json
import asyncio
import logging
import logging.handlers
import uuid
import threading
import shutil
from pathlib import Path
from typing import Dict, Any

# Server process previously had NO logging handlers — warnings (roster drift,
# cache failures) vanished. Route engine loggers to the rotating log file.
from engine.core.paths import LOG_FILE
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.getLogger("f1_predictor").setLevel(logging.DEBUG)
_h = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024,
                                          backupCount=2, encoding="utf-8")
_h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
logging.getLogger("f1_predictor").addHandler(_h)


def sanitize_for_json(obj):
    """
    Recursively convert an object to JSON-serializable Python primitives.
    Handles:
    - numpy scalars (numpy.bool_, numpy.int64, numpy.float64, etc.)
    - numpy arrays
    - objects with __slots__ but no __dict__ (e.g. DistributionStats)
    - dataclasses, namedtuples
    - dicts, lists, tuples
    """
    import numpy as np
    # numpy scalar types
    if isinstance(obj, np.generic):
        return obj.item()
    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # dict
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    # list or tuple
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    # objects with __slots__ (no __dict__) — e.g. DistributionStats
    if hasattr(obj, '__slots__') and not hasattr(obj, '__dict__'):
        return {slot: sanitize_for_json(getattr(obj, slot, None))
                for slot in obj.__slots__}
    # objects with __dict__ (dataclasses, plain objects)
    if hasattr(obj, '__dict__'):
        return {k: sanitize_for_json(v) for k, v in obj.__dict__.items()}
    # primitives
    return obj

# ── SUPPRESS COMPILER & ONEDNN WARNINGS ────────────────────────────────────
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
if not shutil.which("g++"):
    os.environ["PYTENSOR_FLAGS"] = "cxx="
# ────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
import uvicorn
from contextlib import asynccontextmanager

from engine.serving.pipeline import run_full_pipeline

app = FastAPI(title="F1 Fantasy Predictor API")

from engine.core.paths import PROJECT_ROOT, OUTPUT_DIR, API_CACHE_DIR, MODEL_CACHE_DIR, WEATHER_CACHE_DIR, ELO_CACHE_DIR, FASTF1_CACHE_DIR as _FF1, MY_TEAM_PATH

# Ensure ui directory exists (anchored at project root, not CWD)
UI_DIR = PROJECT_ROOT / "ui"
UI_DIR.mkdir(exist_ok=True)

# Mount the UI directory as static files
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

@app.get("/")
def read_index():
    index_path = UI_DIR / "index.html"
    if not index_path.exists():
        return {"error": "UI not built yet."}
    return FileResponse(index_path)

@app.get("/favicon.ico")
def favicon():
    from fastapi import Response
    return Response(status_code=204)

@app.get("/api/status")
def status():
    return {"status": "ok", "message": "Server is running"}

@app.get("/api/race/next")
def get_next_race():
    from engine.core.data_fetcher import get_next_race as fetch_race, _latest_completed_round_num, get_race_by_round
    from engine.core.config import CURRENT_SEASON
    race = fetch_race(CURRENT_SEASON)
    if race:
        latest_round = _latest_completed_round_num(CURRENT_SEASON)
        race["latest_completed_round"] = latest_round
        if latest_round:
            latest_race = get_race_by_round(latest_round, CURRENT_SEASON)
            if latest_race:
                race["latest_completed_name"] = latest_race.get("name")
    return race

@app.get("/api/races")
def get_all_races(year: int = None):
    from engine.core.data_fetcher import get_season_schedule, race_has_happened
    from engine.core.config import CURRENT_SEASON
    target_year = year if year else CURRENT_SEASON
    schedule = get_season_schedule(target_year)
    for r in schedule:
        r["is_completed"] = race_has_happened(r, target_year)
    return schedule

@app.get("/api/race/latest-completed")
def get_latest_completed_race(year: int = None):
    from engine.core.data_fetcher import _latest_completed_round_num, get_race_by_round, get_race_results
    from engine.core.config import CURRENT_SEASON
    target_year = year if year else CURRENT_SEASON
    latest_rnd = _latest_completed_round_num(target_year)
    if not latest_rnd:
        return {"completed": False, "round": 0}
    race = get_race_by_round(latest_rnd, target_year)
    results = get_race_results(target_year, latest_rnd)
    return {
        "completed": True,
        "round": latest_rnd,
        "race": race,
        "results_count": len(results),
        "winner": results[0]["name"] if results else "N/A"
    }

@app.get("/api/race/{round_num}")
def get_race_by_round_api(round_num: int):
    from engine.core.data_fetcher import get_race_by_round
    from engine.core.config import CURRENT_SEASON
    race = get_race_by_round(round_num, CURRENT_SEASON)
    return race

@app.get("/api/qualifying/results")
def get_qualifying_results_api(round_num: int = None, year: int = None):
    from engine.core.data_fetcher import get_actual_qualifying_results, qualifying_has_happened, get_next_race, get_race_by_round
    from engine.core.config import CURRENT_SEASON
    target_year = year or CURRENT_SEASON
    if round_num:
        race = get_race_by_round(round_num, target_year)
    else:
        race = get_next_race(target_year)
    if not race:
        return {"available": False, "results": []}
    happened = qualifying_has_happened(race)
    if not happened:
        return {"available": False, "results": [], "reason": "Qualifying hasn't happened yet"}
    results = get_actual_qualifying_results(target_year, race["round"])
    return {"available": True, "race": race["name"], "results": results}


@app.get("/api/weather")
def get_weather(race_name: str, date: str):
    from engine.core.weather import get_race_weekend_weather
    return get_race_weekend_weather(race_name, date)

@app.get("/api/prices")
def get_prices(refresh: bool = False):
    from engine.core.fantasy_scraper import scrape_driver_prices, scrape_constructor_prices
    d_prices, _ = scrape_driver_prices(force_refresh=refresh)
    c_prices = scrape_constructor_prices(force_refresh=refresh)
    return {"drivers": d_prices, "constructors": c_prices}

@app.post("/api/prices/override")
async def override_prices(request: Request):
    data = await request.json()
    return {"status": "ok", "received": data}

@app.get("/api/team")
def get_team():
    from engine.core.config import MY_TEAM_PATH
    team_cache_file = Path(MY_TEAM_PATH)
    if not team_cache_file.exists():
        # Migrate legacy location (output/cache/) on first read
        legacy = OUTPUT_DIR / "cache" / "my_team.json"
        if legacy.exists():
            try:
                team_cache_file.parent.mkdir(parents=True, exist_ok=True)
                team_cache_file.write_text(legacy.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                logger.warning("Suppressed error", exc_info=True)
                pass
    if team_cache_file.exists():
        try:
            with open(team_cache_file, "r") as f:
                return json.load(f)
        except Exception:
            logger.warning("Suppressed error", exc_info=True)
            pass
    return {
        "drivers": [],
        "constructors": [],
        "budget_remaining": 100.0,
        "current_points": 0.0
    }

@app.post("/api/team")
async def save_team(request: Request):
    from engine.core.config import MY_TEAM_PATH
    data = await request.json()
    team_cache_file = Path(MY_TEAM_PATH)
    team_cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(team_cache_file, "w") as f:
        json.dump(data, f)
    return {"status": "ok"}


@app.get("/api/f1-sync/teams")
async def get_f1_teams(cookie: Optional[str] = None):
    """Fetch all user teams from F1 Fantasy via session cookie."""
    from engine.core.fantasy_sync import fetch_user_teams, F1FantasySyncError
    try:
        teams = fetch_user_teams(cookie=cookie)
        return {"status": "ok", "teams": teams}
    except F1FantasySyncError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error syncing F1 Fantasy teams: {e}", exc_info=True)
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}


@app.post("/api/f1-sync/import")
async def import_f1_team(request: Request):
    """Import a specific team (1, 2, or 3) from F1 Fantasy into local team state."""
    from engine.core.fantasy_sync import import_user_team, F1FantasySyncError
    try:
        data = await request.json()
        team_no = int(data.get("team_no", 1))
        cookie = data.get("cookie")
        imported = import_user_team(team_no=team_no, cookie=cookie)
        return {
            "status": "ok",
            "imported": imported,
            "message": f"Successfully imported Team #{team_no} ({imported.get('team_name', '')})"
        }
    except F1FantasySyncError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error(f"Failed to import team: {e}", exc_info=True)
        return {"status": "error", "message": f"Import failed: {str(e)}"}

@app.get("/api/chips")
def get_chips():
    from engine.strategy.chip_advisor import load_chip_state
    state = load_chip_state()
    return state

@app.post("/api/chips/mark")
async def mark_chip(request: Request):
    from engine.strategy.chip_advisor import mark_chip_used
    data = await request.json()
    chip_name    = data.get("chip_name", "")
    round_used   = data.get("round_used", 0)
    circuit_name = data.get("circuit_name", "Unknown")
    if not chip_name:
        return {"status": "error", "message": "chip_name required"}
    state = mark_chip_used(chip_name, round_used, circuit_name)
    return {"status": "ok", "state": state}

@app.post("/api/chips/reset")
def reset_chips():
    from engine.strategy.chip_advisor import reset_chip_state
    state = reset_chip_state()
    return {"status": "ok", "state": state}

@app.delete("/api/cache/{cache_type}")
def clear_cache(cache_type: str):
    import shutil
    targets = {
        "api":     API_CACHE_DIR,
        "models":  MODEL_CACHE_DIR,
        "weather": WEATHER_CACHE_DIR,
        "fastf1":  Path(_FF1),
        "elo":     ELO_CACHE_DIR,
        "all":     None,  # handled below
    }
    
    if cache_type == "predictions":
        out_dir = OUTPUT_DIR
        cleared = []
        if out_dir.exists():
            for f in out_dir.glob("race_*.json"):
                f.unlink()
                cleared.append(f.name)
        return {"status": "ok", "cleared": cleared}

    if cache_type == "all":
        cleared = []
        for name, path in targets.items():
            if name == "all" or path is None:
                continue
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
                path.mkdir(parents=True, exist_ok=True)
                cleared.append(name)
        
        return {"status": "ok", "cleared": cleared}
        
    path = targets.get(cache_type)
    if path is None:
        return {"status": "error", "message": f"Unknown cache type: {cache_type}"}
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
    return {"status": "ok", "cleared": cache_type}

@app.post("/api/cache/rebuild")
async def rebuild_cache():
    # Plan 9-H2: mutual exclusion — a rebuild while the pipeline is running
    # deletes cache dirs out from under it (mixed-vintage features, 429 storms).
    global _rebuild_running
    if _pipeline_running:
        return {"status": "error",
                "message": "Cannot rebuild caches while an analysis is running."}
    if _rebuild_running:
        return {"status": "error", "message": "A cache rebuild is already running."}
    _rebuild_running = True
    # Trigger background thread — heavy rmtree clearing runs OFF the event loop.
    # Unique run_id: a second click must not orphan the first stream's queue.
    run_id = f"cache_rebuild_{uuid.uuid4().hex[:8]}"
    run_queues[run_id] = asyncio.Queue()
    _job_begin()
    
    def _bg():
        try:
            clear_cache("api")
            clear_cache("weather")
            clear_cache("fastf1")
            clear_cache("elo")

            from engine.core import warm_cache
            def cache_progress(payload):
                # Plan I1: dict progress contract {done,total,current,...}
                if isinstance(payload, dict):
                    sync_progress_callback(
                        run_id, "DOWNLOADING", "loading",
                        payload.get("current", ""),
                        data=payload,
                    )
                else:
                    sync_progress_callback(run_id, "DOWNLOADING", "loading", str(payload))
            warm_cache.verify_and_download_caches(progress_callback=cache_progress)
            # Client-visible terminal event (app.js reloads on COMPLETE)
            sync_progress_callback(run_id, "COMPLETE", "done", "Cache rebuild complete")
        except Exception as e:
            sync_progress_callback(run_id, "ERROR", "error", str(e))
        finally:
            globals()["_rebuild_running"] = False
            _job_end()
            sync_progress_callback(run_id, "_TERMINATE", "done", "")

    t = threading.Thread(target=_bg, daemon=True)
    t.start()
    return {"status": "ok", "run_id": run_id}

@app.get("/api/predictions")
def list_predictions():
    out_dir = OUTPUT_DIR
    preds = []
    if out_dir.exists():
        for f in out_dir.glob("race_*.json"):
            try:
                with open(f, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    preds.append({
                        "filename": f.name,
                        "generated_at": data.get("generated_at"),
                        "race": data.get("race", {}).get("name")
                    })
            except Exception:
                logger.warning("Suppressed error", exc_info=True)
                pass
    # Sort by newest
    preds.sort(key=lambda x: x.get("generated_at", ""), reverse=True)
    return {"status": "ok", "predictions": preds}

@app.get("/api/predictions/{filename}")
def get_prediction(filename: str):
    f = OUTPUT_DIR / filename
    if f.exists() and f.name.startswith("race_") and f.name.endswith(".json"):
        with open(f, "r", encoding="utf-8") as file:
            return json.load(file)
    return {"error": "Not found"}

@app.post("/api/post-race/compare/{filename}")
def compare_prediction(filename: str):
    f = OUTPUT_DIR / filename
    if not (f.exists() and f.name.startswith("race_") and f.name.endswith(".json")):
        return {"error": "Not found"}
    
    from engine.analysis import post_race_check
    metrics = post_race_check.validate_specific_prediction(f)
    if metrics:
        return {"status": "ok", "metrics": metrics}
    return {"status": "error", "message": "Failed to validate or results not available yet."}


@app.get("/api/analysis/stage-comparison/{round_num}")
def get_stage_comparison_api(round_num: int, season: int = None):
    from engine.analysis.stage_comparator import get_stage_comparison
    from engine.core.config import CURRENT_SEASON
    target_season = season if season else CURRENT_SEASON
    return get_stage_comparison(round_num, target_season)


@app.get("/api/analysis/available-rounds")
def get_available_rounds_api():
    from engine.core.data_fetcher import get_season_schedule, race_has_happened
    from engine.core.config import CURRENT_SEASON
    sched = get_season_schedule(CURRENT_SEASON)
    completed = [r for r in sched if race_has_happened(r, CURRENT_SEASON)]
    return {"status": "ok", "rounds": completed}



@app.post("/api/memory/reset")
def reset_memory():
    """Wipe user-specific state (team, chips, price/ownership history).
    Does NOT delete downloaded API data or trained models."""
    import json as _json
    wiped = []
    from engine.core.paths import CACHE_DIR, CHIP_STATE_PATH, PRICE_HISTORY_PATH, OWNERSHIP_HISTORY_PATH
    files_to_reset = [
        MY_TEAM_PATH,
        CHIP_STATE_PATH,
        PRICE_HISTORY_PATH,
        OWNERSHIP_HISTORY_PATH,
        OUTPUT_DIR / "cache" / "my_team.json",   # legacy location
    ]
    for f in files_to_reset:
        if f.exists():
            f.unlink()
            wiped.append(str(f.name))
    from engine.strategy.chip_advisor import reset_chip_state
    reset_chip_state()
    return {"status": "ok", "wiped": wiped}



# --- Pipeline Execution & SSE ---

run_queues: Dict[str, asyncio.Queue] = {}
run_results: Dict[str, Any] = {}
# Capture the main event loop to use in background threads
main_loop = None

import time
import webbrowser


logger = logging.getLogger("f1_predictor.server")
# ── Browser-close watchdog ──────────────────────────────────────────────────
# The frontend sends POST /api/heartbeat every 5 s while the page is open.
# If no heartbeat arrives within HEARTBEAT_TIMEOUT seconds after the first
# one — AND no background jobs are active — we assume the tab was closed and
# shut down the server gracefully.
_last_heartbeat: float = 0.0
_heartbeat_received: bool = False
_HEARTBEAT_TIMEOUT: int  = 20  # seconds — 4 missed beats before exit
_pipeline_running: bool  = False  # block duplicate /api/run while ML is in flight
_rebuild_running: bool   = False  # plan 9-H2: block rebuild↔run mutual exclusion

# Job-aware shutdown guard: EVERY long-running background job (pipeline,
# cache rebuild, post-race check) registers here for its lifetime so a
# browser close mid-download can't hard-kill the process.
_active_jobs: int = 0
_jobs_lock = threading.Lock()
# Set to the running uvicorn.Server so the watchdog can request graceful exit
_uvicorn_server = None

def _job_begin():
    global _active_jobs
    with _jobs_lock:
        _active_jobs += 1

def _job_end():
    global _active_jobs
    with _jobs_lock:
        _active_jobs = max(0, _active_jobs - 1)

def _request_shutdown():
    srv = globals().get("_uvicorn_server")
    if srv is not None:
        try:
            srv.should_exit = True
            return
        except Exception:
            logger.warning("Suppressed error", exc_info=True)
            pass
    os._exit(0)   # last resort if no server handle
# ────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_loop, _uvicorn_server
    main_loop = asyncio.get_running_loop()

    # 1. Open browser
    def open_browser():
        time.sleep(0.5)
        webbrowser.open("http://127.0.0.1:5762/")
    threading.Thread(target=open_browser, daemon=True).start()

    # 3. Watchdog
    def watchdog():
        """Shut down server when browser tab is closed (heartbeat stops)."""
        missed_required = 2   # require N consecutive timed-out ticks (refresh safety)
        missed = 0
        while True:
            time.sleep(5)
            with _jobs_lock:
                busy = _active_jobs > 0
            if _heartbeat_received and not busy:
                if time.time() - _last_heartbeat > _HEARTBEAT_TIMEOUT:
                    missed += 1
                    if missed >= missed_required:
                        print("\n[F1 Engine] Browser disconnected — shutting down. Goodbye!")
                        _request_shutdown()
                        return
                else:
                    missed = 0
    threading.Thread(target=watchdog, daemon=True).start()
    
    yield

app.router.lifespan_context = lifespan

@app.post("/api/heartbeat")
def heartbeat():
    global _last_heartbeat, _heartbeat_received
    _last_heartbeat = time.time()
    _heartbeat_received = True
    return {"status": "ok"}

@app.post("/api/heartbeat/close")
def heartbeat_close():
    """Browser fired beforeunload — backdate timestamp so watchdog exits on next tick."""
    global _last_heartbeat
    # Set timestamp far in the past so the watchdog sees a timeout immediately
    _last_heartbeat = time.time() - _HEARTBEAT_TIMEOUT - 10
    return {"status": "closing"}

def sync_progress_callback(run_id: str, stage: str, status: str, message: str, data: dict = None):
    event = {
        "stage": stage,
        "status": status,
        "message": message,
    }
    if data:
        event["data"] = data
        
    if run_id in run_queues and main_loop:
        try:
            main_loop.call_soon_threadsafe(run_queues[run_id].put_nowait, event)
        except Exception as e:
            print(f"Error putting event in queue: {e}")

@app.post("/api/run")
async def trigger_run(request: Request):
    global _pipeline_running
    if _pipeline_running:
        return {"status": "error", "message": "A pipeline is already currently running"}
    # Plan 9-H2: block runs during cache rebuilds (dirs deleted underneath)
    if _rebuild_running:
        return {"status": "error",
                "message": "A cache rebuild is running — start the analysis after it completes."}

    data = await request.json()
    run_id = str(uuid.uuid4())
    run_queues[run_id] = asyncio.Queue()
    _job_begin()          # registered synchronously: no watchdog race window
    
    my_drivers = data.get("drivers", [])
    my_ctors = data.get("constructors", [])
    budget = float(data.get("budget", 100.0))
    points = float(data.get("points", 0.0))
    transfers = int(data.get("transfers", 1))
    options = data.get("options", {})
    
    def background_task():
        global _pipeline_running
        _pipeline_running = True
        try:
            res = run_full_pipeline(
                run_id, my_drivers, my_ctors, budget, points, transfers, options, sync_progress_callback
            )
            with _results_lock:
                run_results[run_id] = res
                _prune_run_results()
        finally:
            _pipeline_running = False
            _job_end()
            sync_progress_callback(run_id, "_TERMINATE", "done", "")

    thread = threading.Thread(target=background_task)
    thread.start()
    
    return {"status": "ok", "run_id": run_id}

async def event_generator(run_id: str):
    queue = run_queues.get(run_id)
    if not queue:
        yield f"data: {json.dumps({'error': 'Invalid run_id'})}\n\n"
        return
        
    try:
        while True:
            try:
                event = await queue.get()
                if event.get("stage") == "_TERMINATE":
                    break
                # Deep-sanitize to handle numpy types, __slots__ objects, etc.
                safe_event = sanitize_for_json(event)
                yield f"data: {json.dumps(safe_event)}\n\n"
                if event.get("stage") in ("COMPLETE", "ERROR"):
                    break
            except Exception as e:
                import traceback
                err_event = {"stage": "ERROR", "message": f"Serialization error: {e}", "detail": traceback.format_exc()}
                yield f"data: {json.dumps(err_event)}\n\n"
                break
    finally:
        # Free the queue + buffered events once this stream is done
        run_queues.pop(run_id, None)

@app.get("/api/run/stream/{run_id}")
async def run_stream(run_id: str):
    return StreamingResponse(event_generator(run_id), media_type="text/event-stream")

# Keep only the most recent N pipeline payloads (each can be multi-MB)
_MAX_RUN_RESULTS = 10
_results_lock = threading.Lock()

def _prune_run_results():
    while len(run_results) > _MAX_RUN_RESULTS:
        oldest = next(iter(run_results))
        run_results.pop(oldest, None)

@app.get("/api/results/{run_id}")
def get_results(run_id: str):
    with _results_lock:
        res = run_results.get(run_id)
    if res is not None:
        # Sanitize: raw payload contains numpy scalars FastAPI can't encode
        return {"status": "ok", "data": sanitize_for_json(res)}
    return {"status": "not_found"}

@app.post("/api/post-race/{round_num}")
async def trigger_post_race(round_num: int):
    run_id = str(uuid.uuid4())
    run_queues[run_id] = asyncio.Queue()
    _job_begin()
    
    def background_task():
        try:
            sync_progress_callback(run_id, "FETCH_RESULTS", "loading", f"Fetching actual results for Round {round_num}...")
            from engine.analysis import post_race_check
            metrics = post_race_check.validate_round(round_num)
            
            if metrics:
                sync_progress_callback(run_id, "COMPLETE", "done", "Post-race check complete!", data=metrics)
            else:
                sync_progress_callback(run_id, "ERROR", "error", f"Failed to run post-race check. Check if actual results are available or prediction file exists.")
        except Exception as e:
            sync_progress_callback(run_id, "ERROR", "error", str(e))
        finally:
            _job_end()
            sync_progress_callback(run_id, "_TERMINATE", "done", "")

    thread = threading.Thread(target=background_task)
    thread.start()
    
    return {"status": "ok", "run_id": run_id}

@app.get("/api/model/health")
def get_model_health():
    from engine.strategy import self_improvement
    return self_improvement.get_model_health_report()

@app.post("/api/model/reset-corrections")
def reset_model_corrections():
    from engine.strategy import self_improvement
    return self_improvement.reset_corrections()

@app.get("/api/system/health")
def get_system_health():
    import shutil
    from engine.core.data_fetcher import CACHE_DIR
    import time
    
    # Check for g++
    has_compiler = shutil.which("g++") is not None
    
    # Count cached races
    cache_count = len(list(CACHE_DIR.glob("jolpica_*.json")))
    
    return {
        "status": "ok",
        "engine": "Bayesian (MCMC)" if has_compiler else "Iterative (Fallback)",
        "compiler": "Detected" if has_compiler else "Missing (Slow Mode)",
        "cache_size": f"{cache_count} sessions",
        "last_sync": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/api/standings/drivers")
def get_drivers_standings_api(round_num: int = None):
    from engine.core.data_fetcher import get_driver_standings
    from engine.core.config import CURRENT_SEASON
    return get_driver_standings(CURRENT_SEASON, round_num)

@app.get("/api/standings/constructors")
def get_constructors_standings_api(round_num: int = None):
    from engine.core.data_fetcher import get_constructor_standings
    from engine.core.config import CURRENT_SEASON
    return get_constructor_standings(CURRENT_SEASON, round_num)

@app.get("/api/race/{round_num}/results")
def get_race_results_api(round_num: int, year: int = None):
    from engine.core.data_fetcher import get_race_results
    from engine.core.config import CURRENT_SEASON
    target_year = year if year else CURRENT_SEASON
    return get_race_results(target_year, round_num)

@app.get("/api/telemetry/{gp_name}")
def get_telemetry(gp_name: str, round_num: int = None, year: int = None):
    from engine.core.config import CURRENT_SEASON
    from engine.core.data_fetcher import get_tire_stints

    target_year = year if year else CURRENT_SEASON
    target_event = round_num if round_num else gp_name
    
    stints = get_tire_stints(target_year, target_event)
    return {"stints": stints}

@app.get("/api/circuits")
def get_circuits():
    return {"status": "not_implemented"}

@app.delete("/api/shutdown")
def shutdown():
    # Respond FIRST, then request graceful shutdown (os._exit before returning
    # meant the client never received a response).
    threading.Thread(target=lambda: (time.sleep(0.3), _request_shutdown()), daemon=True).start()
    return {"status": "shutting_down"}

if __name__ == "__main__":
    config = uvicorn.Config("engine.serving.server:app", host="127.0.0.1", port=5762)
    _uvicorn_server = uvicorn.Server(config)
    _uvicorn_server.run()
