import datetime
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from engine.core.config import HISTORICAL_SEASONS, CURRENT_SEASON
from engine.core.data_fetcher import (
    get_season_schedule,
    get_race_results,
    get_qualifying_results,
    get_driver_standings,
    get_constructor_standings,
    get_grid_penalties,
    get_season_results,
    get_fastf1_session,
)

logger = logging.getLogger("f1_predictor.warm_cache")


# ─────────────────────────────────────────────
# PROGRESS STATE (plan I1)
# ─────────────────────────────────────────────
class ProgressState:
    """Thread-safe done/total counter emitting the dict payload contract."""

    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.failed = 0
        self.current = ""
        self.phase = "downloading"
        self._lock = threading.Lock()
        self._t0 = time.time()
        self._last_done = 0
        self._last_t = self._t0

    def advance(self, ok: bool, current: str):
        with self._lock:
            self.done += 1
            if not ok:
                self.failed += 1
            self.current = current

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = max(0.001, time.time() - self._t0)
            rate = self.done / elapsed * 60.0
            remaining = max(0, self.total - self.done)
            eta = (remaining / rate) if rate > 0.5 else None
            return {
                "done": self.done,
                "total": self.total,
                "failed": self.failed,
                "current": self.current,
                "phase": self.phase,
                "rate_per_min": round(rate, 1),
                "eta_min": round(eta, 1) if eta else None,
            }


def _build_manifest(seasons: list[int]) -> list[tuple]:
    """[(year, round, name, [endpoints...])] for every completed race.
    Plan I1 amendment: EVERY unit counts as work — cached hits complete in
    milliseconds, so totals stay truthful without fragile cache probing."""
    units = []
    today = datetime.date.today()
    for year in seasons:
        try:
            sched = get_season_schedule(year)
        except Exception as e:
            logger.warning("Manifest: schedule unavailable for %s: %s", year, e)
            continue
        for race in sched:
            r_num = race.get("round")
            if not r_num:
                continue
            try:
                if datetime.datetime.strptime(race["date"], "%Y-%m-%d").date() > today:
                    continue
            except Exception:
                pass
            eps = ["jolpica_results", "jolpica_quali",
                   "jolpica_drv_standings", "jolpica_ctor_standings"]
            if year >= 2023:
                eps.append("grid_penalties")
            eps += ["fastf1_R", "fastf1_Q"]
            units.append((year, r_num, race.get("name", f"R{r_num}"), eps))
    return units


def _run_unit(year: int, round_num: int, name: str, ep: str) -> tuple[bool, str]:
    """Execute ONE manifest endpoint. Returns (ok, label)."""
    label = f"{year} R{round_num} · {ep}"
    try:
        if ep == "jolpica_results":
            get_race_results(year, round_num)
        elif ep == "jolpica_quali":
            get_qualifying_results(year, round_num)
        elif ep == "jolpica_drv_standings":
            get_driver_standings(year, round_num)
        elif ep == "jolpica_ctor_standings":
            get_constructor_standings(year, round_num)
        elif ep == "grid_penalties":
            get_grid_penalties(year, round_num)
        elif ep == "fastf1_R":
            get_fastf1_session(year, name, "R")
        elif ep == "fastf1_Q":
            get_fastf1_session(year, name, "Q")
        else:
            return True, label
        return True, label
    except Exception as e:
        logger.warning("[%s R%s] %s failed: %s", year, round_num, ep, e)
        return False, label


def verify_and_download_caches(progress_callback=None, max_workers: int = 4):
    """Bulk cache warmer with manifest-based progress (plan I1).

    progress_callback receives DICT payloads:
      {done, total, failed, current, phase, rate_per_min, eta_min}
    """
    print("==================================================")
    print("  F1 FANTASY HIGH-SPEED BULK CACHE WARMER         ")
    print("==================================================")
    seasons = sorted(set(HISTORICAL_SEASONS) | {CURRENT_SEASON})
    logger.info("Building work manifest for %d seasons...", seasons)
    units = _build_manifest(seasons)
    total_units = len(units)
    print(f"Manifest: {total_units} download units across {len(seasons)} seasons.\n")

    state = ProgressState(total_units)

    def emit():
        if progress_callback:
            try:
                progress_callback(state.snapshot())
            except Exception as e:
                logger.warning("progress callback error: %s", e)

    # Host-aware split (plan 5d): Jolpica endpoints are rate-limited and
    # paced; FastF1 has its own throttle and bigger payloads. Two pools so
    # neither serializes behind the other.
    jolpica_units = [(y, r, n, ep) for y, r, n, eps in units for ep in eps
                     if not ep.startswith("fastf1")]
    fastf1_units = [(y, r, n, ep) for y, r, n, eps in units for ep in eps
                    if ep.startswith("fastf1")]

    failures = []

    def run_pool(pool_units, workers, pool_name):
        if not pool_units:
            return
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_run_unit, *u): u for u in pool_units}
            for fut in as_completed(futs):
                y, r, n, ep = futures[fut]
                ok, label = fut.result()
                state.advance(ok, label)
                if not ok:
                    failures.append(label)
                emit()

    run_pool(jolpica_units, max_workers, "jolpica")
    state.phase = "fastf1_sessions"
    run_pool(fastf1_units, max(2, max_workers), "fastf1")

    state.phase = "complete"

    summary = (f"CACHE WARMING COMPLETE: {state.done}/{state.total} units "
               f"({state.failed} failed)")
    print("\n==================================================")
    print(summary)
    print("==================================================")

    if progress_callback:
        snap = state.snapshot()
        snap["phase"] = "complete"
        progress_callback(snap)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    verify_and_download_caches()
