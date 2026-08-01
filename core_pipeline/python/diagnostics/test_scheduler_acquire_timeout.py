"""Regression test: AdaptiveGpuScheduler.acquire() must bound its wait with a max
timeout and raise SchedulerBusyError (not hang forever) once the timeout elapses, without
leaking the reservation (_active must return to its pre-call value).
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend_api", "app"))

import gpu_scheduler  # noqa: E402


def test_scheduler_acquire_timeout() -> None:
    scheduler = gpu_scheduler.AdaptiveGpuScheduler()
    # Force forced_sequential capacity=1 so a second acquire has to wait.
    os.environ["FMT_PIPELINE_MAX_PARALLEL"] = "1"

    with scheduler.acquire("holder"):
        started = time.perf_counter()
        try:
            with scheduler.acquire("waiter", max_wait_seconds=0.5):
                raise AssertionError("acquire() should have timed out, not succeeded")
        except gpu_scheduler.SchedulerBusyError:
            elapsed = time.perf_counter() - started
            assert elapsed < 5.0, f"timeout took too long: {elapsed:.2f}s"
        assert scheduler._active == 1, f"expected active=1 (only the holder), got {scheduler._active}"

    assert scheduler._active == 0, f"expected active=0 after holder released, got {scheduler._active}"
    os.environ.pop("FMT_PIPELINE_MAX_PARALLEL", None)
    print("scheduler_acquire_timeout=pass")


if __name__ == "__main__":
    test_scheduler_acquire_timeout()
