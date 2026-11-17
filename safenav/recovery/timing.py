r"""
timing.py — high-resolution per-thread CPU time for latency diagnostics (round 4).

time.thread_time() on Windows has the resolution of the system timer tick (15.625 ms), which is
useless for planner calls of a few milliseconds. On Windows this module reads the thread's cycle
counter (QueryThreadCycleTime) and converts it to seconds with a rate calibrated once against the
wall clock while the thread is busy. Elsewhere it uses time.thread_time_ns().

The result is a diagnostic: comparing it with the wall clock shows whether a slow call was busy
computing or was waiting (pre-empted by other processes). It is not used for any decision.
"""

import ctypes
import sys
import time

_rate = None          # cycles per second (Windows)
_calibration = None


def _cycles() -> int:
    c = ctypes.c_ulonglong()
    ctypes.windll.kernel32.QueryThreadCycleTime(ctypes.c_void_p(-2), ctypes.byref(c))  # -2: current thread
    return c.value


def calibrate(duration_s: float = 0.15, repeats: int = 5) -> dict:
    """Cycle rate from busy loops (largest observed ratio, i.e. the least interrupted loop)."""
    global _rate, _calibration
    if sys.platform != "win32":
        _calibration = {"method": "time.thread_time_ns"}
        return _calibration
    rates = []
    for _ in range(repeats):
        c0, w0 = _cycles(), time.perf_counter()
        while time.perf_counter() - w0 < duration_s:
            pass
        rates.append((_cycles() - c0) / (time.perf_counter() - w0))
    _rate = max(rates)
    _calibration = {"method": "QueryThreadCycleTime / calibrated rate", "rate_hz": _rate,
                    "rates_hz": rates, "spread": (max(rates) - min(rates)) / max(rates)}
    return _calibration


def thread_cpu_s() -> float:
    if sys.platform != "win32":
        return time.thread_time_ns() / 1e9
    if _rate is None:
        calibrate()
    return _cycles() / _rate


def calibration() -> dict:
    if _calibration is None:
        calibrate()
    return _calibration
