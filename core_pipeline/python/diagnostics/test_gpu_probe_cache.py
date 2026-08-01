"""Regression test: AdaptiveGpuScheduler.gpu_memory() must cache the probe for a short
TTL instead of re-probing (nvidia-smi/torch.cuda.mem_get_info) on every call -- a fresh
probe on every /v1/health call caused a measured ~15s stall during model warmup because a
busy CUDA context makes mem_get_info() slow to return.

Fail-before: with no cache, calling gpu_memory() twice in immediate succession invokes the
underlying probe twice. Pass-after: within the TTL window, the second call reuses the
cached result (probe invoked once); after the TTL expires, a new probe happens.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend_api", "app"))

import gpu_scheduler  # noqa: E402


def test_gpu_probe_cache() -> None:
    scheduler = gpu_scheduler.AdaptiveGpuScheduler()

    call_count = {"n": 0}

    def fake_probe():
        call_count["n"] += 1
        return gpu_scheduler.GpuMemory(total_mb=8192, used_mb=1024, free_mb=7168, source="fake")

    scheduler._nvidia_smi_memory = fake_probe  # type: ignore[method-assign]
    scheduler._torch_memory = lambda: None  # type: ignore[method-assign]

    first = scheduler.gpu_memory()
    second = scheduler.gpu_memory()
    assert call_count["n"] == 1, f"expected 1 underlying probe within TTL window, got {call_count['n']}"
    assert first == second

    # Force TTL expiry and confirm a fresh probe happens.
    ttl = getattr(scheduler, "_gpu_probe_ttl_seconds", None) or gpu_scheduler.DEFAULT_GPU_PROBE_TTL_SECONDS
    time.sleep(ttl + 0.3)
    scheduler.gpu_memory()
    assert call_count["n"] == 2, f"expected a fresh probe after TTL expiry, got {call_count['n']} total calls"

    print("gpu_probe_cache=pass")


if __name__ == "__main__":
    test_gpu_probe_cache()
