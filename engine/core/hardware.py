"""
hardware.py — Universal runtime capability profile (plan I3).

Every machine gets the best execution plan it supports, discovered at
runtime and verified by ground-truth probes rather than optimistic guesses.
Stdlib-only; never raises.

Env-var overrides (all optional):
    F1E_XGB_DEVICE = auto | cuda | cpu
    F1E_MC_WORKERS = <int> | serial
    F1E_IO_WORKERS = <int>
"""
import os
import shutil
import threading

_cache: dict | None = None
_lock = threading.Lock()


def _cpu_threads() -> int:
    return os.cpu_count() or 2


def _nvidia_gpu_name() -> str | None:
    """Return the first NVIDIA GPU name via nvidia-smi, or None."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        name = (out.stdout or "").strip().splitlines()[0].strip() if out.returncode == 0 else ""
        return name or None
    except Exception:
        return None


def _probe_xgb_cuda() -> bool:
    """Ground truth: actually fit a tiny booster on CUDA. Any failure → False."""
    try:
        import numpy as np
        import xgboost as xgb
        X = np.random.RandomState(0).rand(10, 3)
        y = np.random.RandomState(1).rand(10)
        m = xgb.XGBRegressor(n_estimators=2, max_depth=1, tree_method="hist",
                             device="cuda")
        m.fit(X, y)
        return True
    except Exception:
        return False


def recommended_mc_workers(cpu_threads: int) -> int:
    env = os.getenv("F1E_MC_WORKERS", "auto").strip().lower()
    if env == "serial":
        return 1
    if env.isdigit():
        return max(1, min(64, int(env)))
    workers = max(1, cpu_threads - 1)
    if cpu_threads >= 8:
        workers = min(workers - 1, 15)   # leave headroom on big machines
    return max(1, min(32, workers))


def recommended_io_workers(cpu_threads: int) -> int:
    env = os.getenv("F1E_IO_WORKERS", "auto").strip().lower()
    if env.isdigit():
        return max(1, min(32, int(env)))
    return min(8, max(3, cpu_threads // 4))


def profile(force: bool = False) -> dict:
    """Capability profile, cached per-process. Never raises."""
    global _cache
    with _lock:
        if _cache is not None and not force:
            return dict(_cache)

        threads = _cpu_threads()
        gpu_name = _nvidia_gpu_name()

        dev_env = os.getenv("F1E_XGB_DEVICE", "auto").strip().lower()
        if dev_env == "cpu":
            xgb_device = "cpu"
        elif dev_env == "cuda":
            xgb_device = "cuda"   # user insists; probe still recorded below
        else:  # auto
            xgb_device = "cuda" if gpu_name else "cpu"

        # Ground-truth verification even for explicit cuda (driver may be broken)
        if xgb_device == "cuda" and not _probe_xgb_cuda():
            xgb_device = "cpu"

        _cache = {
            "cpu_threads": threads,
            "gpu_name": gpu_name,
            "gpu_vendor": "nvidia" if gpu_name else None,
            "xgb_device": xgb_device,
            "lgb_device": "cpu",
            "mc_workers": recommended_mc_workers(threads),
            "io_workers": recommended_io_workers(threads),
        }
        return dict(_cache)


def summary_line(t_detect: float = 0.0) -> str:
    p = profile()
    gpu = p["gpu_name"] or "none"
    return (f"engine-hw: xgb={p['xgb_device']} mc={p['mc_workers']} "
            f"io={p['io_workers']} gpu={gpu} ({t_detect:.1f}s detect)")
