from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterator


DEFAULT_MAX_PARALLEL = 2
DEFAULT_RESERVED_FREE_MB = 2048
DEFAULT_JOB_VRAM_MB = 3072
DEFAULT_WAIT_LOG_SECONDS = 8.0
DEFAULT_ADMISSION_GRACE_SECONDS = 8.0
DEFAULT_GPU_PROBE_TTL_SECONDS = 1.5
# 60s was measured to be shorter than a real page's own pipeline runtime (a live end-to-end
# run measured ~78s: step5_ocr 12.8 + step6_layout 1.4 + step7_translate 5.2 + step4_inpaint
# 3.6 + step8_typeset 54.8), which made a 2nd concurrent request at capacity=1 (the common case
# on an 8GB card post-warmup) 503 with SCHEDULER_BUSY deterministically, not occasionally.
# extension/background.js now clamps its own dispatch concurrency to this server's advertised
# capacity (effectiveParallelLimit(), fed by /v1/health's scheduler.capacity and this endpoint's
# own report.scheduler.capacityAtAcquire) and retries a SCHEDULER_BUSY response internally
# instead of dropping the page, so this wait is now a secondary safety margin rather than the
# primary defense -- it mainly protects clients that don't do that clamping (a second browser
# profile, /v1/batch, any other caller). 120s must stay under the extension's own fetch timeout
# (background.js's DEFAULT_FETCH_TIMEOUT_MS = 360_000) with margin for wait + the run itself:
# 120 + 180 (documented cold-run ceiling) = 300s, leaving ~60s. 180s would consume the whole
# 360s budget on a cold run; 240s would exceed it.
DEFAULT_MAX_WAIT_SECONDS = 120.0


class SchedulerBusyError(Exception):
    """Raised by acquire() when max_wait_seconds elapses before a slot frees up.

    Deliberately NOT a subclass of any pipeline-error type -- callers must be able to
    catch this specifically and map it to HTTP 503, distinct from the generic 500 used
    for real pipeline failures.
    """


def _env_int(name: str, default: int, minimum: int, maximum: int | None = None) -> int:
    value = os.environ.get(name, "").strip()
    try:
        parsed = int(value) if value else default
    except ValueError:
        parsed = default
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _env_float(name: str, default: float, minimum: float) -> float:
    value = os.environ.get(name, "").strip()
    try:
        parsed = float(value) if value else default
    except ValueError:
        parsed = default
    return max(minimum, parsed)


@dataclass(frozen=True)
class GpuMemory:
    total_mb: int
    used_mb: int
    free_mb: int
    source: str


@dataclass(frozen=True)
class PipelineSlot:
    active_at_acquire: int
    capacity_at_acquire: int
    wait_seconds: float
    gpu: GpuMemory | None
    mode: str

    def as_report(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "activeAtAcquire": self.active_at_acquire,
            "capacityAtAcquire": self.capacity_at_acquire,
            "waitSeconds": round(self.wait_seconds, 3),
            "gpu": self.gpu.__dict__ if self.gpu else None,
        }


class AdaptiveGpuScheduler:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = 0
        self._last_wait_log = 0.0
        # Timestamps of recent successful admissions, used only to widen the capacity
        # estimate's safety margin for a short grace window after each admission (see
        # admission_grace_seconds) -- a just-admitted job's VRAM usage may not show up in
        # the next gpu_memory() probe yet, so without this a burst of admissions can
        # oversubscribe the GPU before the probe catches up.
        self._admission_times: list[float] = []
        # Short-TTL cache for gpu_memory(): probing nvidia-smi/torch on every call caused a
        # measured ~15s stall on /v1/health during model warmup (a busy CUDA context makes
        # mem_get_info() slow). This is a leaf lock -- only ever held inside gpu_memory()
        # itself, never nested with _condition or any other lock -- so it cannot introduce
        # a new lock-ordering hazard.
        self._gpu_probe_lock = threading.Lock()
        self._gpu_probe_cache: GpuMemory | None = None
        self._gpu_probe_cached_at = 0.0

    @property
    def max_parallel(self) -> int:
        # The ceiling here (8) is not the real safety gate -- _capacity_for_memory()'s live
        # free-VRAM probe is, and it already refuses to admit more jobs than actual free
        # memory supports regardless of this value. This ceiling only bounds how high a
        # deployment CAN opt into via the env var; it was previously hard-clamped to 4, which
        # meant a machine with genuinely more free VRAM than 4 jobs' worth could never exceed
        # 4 concurrent jobs even with headroom to spare. The DEFAULT stays 2 (unchanged) --
        # this only widens what a deployment can explicitly opt into. Kaggle's own notebook
        # currently sets FMT_PIPELINE_MAX_PARALLEL=3, already close to what its estimated free
        # VRAM after base models load supports (see Cell 3's own comment) -- this change does
        # not itself raise that, it only removes the code-level wall below it.
        return _env_int("FMT_PIPELINE_MAX_PARALLEL", DEFAULT_MAX_PARALLEL, 1, 8)

    @property
    def reserved_free_mb(self) -> int:
        return _env_int("FMT_GPU_RESERVED_FREE_MB", DEFAULT_RESERVED_FREE_MB, 512, 32768)

    @property
    def job_vram_mb(self) -> int:
        return _env_int("FMT_PIPELINE_JOB_VRAM_MB", DEFAULT_JOB_VRAM_MB, 512, 32768)

    @property
    def admission_grace_seconds(self) -> float:
        return _env_float("FMT_PIPELINE_ADMISSION_GRACE_SECONDS", DEFAULT_ADMISSION_GRACE_SECONDS, 0.0)

    @property
    def _gpu_probe_ttl_seconds(self) -> float:
        return _env_float("FMT_GPU_PROBE_TTL_SECONDS", DEFAULT_GPU_PROBE_TTL_SECONDS, 0.0)

    @property
    def max_wait_seconds(self) -> float:
        return _env_float("FMT_PIPELINE_MAX_WAIT_SECONDS", DEFAULT_MAX_WAIT_SECONDS, 0.0)

    def _nvidia_smi_memory(self) -> GpuMemory | None:
        executable = shutil.which("nvidia-smi")
        if not executable:
            return None
        try:
            output = subprocess.check_output(
                [
                    executable,
                    "--query-gpu=memory.total,memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            )
        except Exception:
            return None
        first_line = next((line.strip() for line in output.splitlines() if line.strip()), "")
        if not first_line:
            return None
        try:
            total, used, free = [int(part.strip()) for part in first_line.split(",")[:3]]
        except ValueError:
            return None
        return GpuMemory(total_mb=total, used_mb=used, free_mb=free, source="nvidia-smi")

    def _torch_memory(self) -> GpuMemory | None:
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            total = int(total_bytes / (1024 * 1024))
            free = int(free_bytes / (1024 * 1024))
            return GpuMemory(total_mb=total, used_mb=max(0, total - free), free_mb=free, source="torch")
        except Exception:
            return None

    def gpu_memory(self) -> GpuMemory | None:
        ttl = self._gpu_probe_ttl_seconds
        now = time.perf_counter()
        if ttl > 0:
            with self._gpu_probe_lock:
                if now - self._gpu_probe_cached_at < ttl:
                    return self._gpu_probe_cache
        result = self._nvidia_smi_memory() or self._torch_memory()
        if ttl > 0:
            with self._gpu_probe_lock:
                self._gpu_probe_cache = result
                self._gpu_probe_cached_at = time.perf_counter()
        return result

    def _recent_admission_count(self) -> int:
        grace_seconds = self.admission_grace_seconds
        if grace_seconds <= 0:
            return 0
        now = time.perf_counter()
        with self._condition:
            self._admission_times = [t for t in self._admission_times if now - t < grace_seconds]
            return len(self._admission_times)

    def _capacity_for_memory(self, gpu: GpuMemory | None) -> tuple[int, str]:
        max_parallel = self.max_parallel
        if max_parallel <= 1:
            return 1, "forced_sequential"
        if gpu is None:
            return 1, "sequential_no_gpu_memory_probe"
        recent_admissions = self._recent_admission_count()
        usable_free = gpu.free_mb - self.reserved_free_mb - recent_admissions * self.job_vram_mb
        if usable_free < self.job_vram_mb:
            return 1, "sequential_low_vram"
        extra_slots = usable_free // self.job_vram_mb
        capacity = max(1, min(max_parallel, 1 + int(extra_slots)))
        return capacity, "adaptive_parallel" if capacity > 1 else "sequential_low_vram"

    def status(self) -> dict[str, Any]:
        gpu = self.gpu_memory()
        capacity, mode = self._capacity_for_memory(gpu)
        with self._condition:
            active = self._active
        return {
            "mode": mode,
            "active": active,
            "capacity": capacity,
            "maxParallel": self.max_parallel,
            "reservedFreeMb": self.reserved_free_mb,
            "estimatedJobVramMb": self.job_vram_mb,
            "gpu": gpu.__dict__ if gpu else None,
        }

    @contextlib.contextmanager
    def acquire(self, request_label: str = "", max_wait_seconds: float | None = None) -> Iterator[PipelineSlot]:
        started = time.perf_counter()
        wait_limit = self.max_wait_seconds if max_wait_seconds is None else max_wait_seconds
        while True:
            if wait_limit > 0 and time.perf_counter() - started >= wait_limit:
                raise SchedulerBusyError(
                    f"Timed out after {wait_limit:.1f}s waiting for a pipeline slot "
                    f"(request={request_label})"
                )
            # gpu_memory() can shell out to nvidia-smi (up to a 2s subprocess timeout) or
            # touch the torch CUDA context. It must run OUTSIDE the condition lock: probing
            # while holding the lock serializes every waiter behind each other's probe,
            # turning concurrent load into a lock convoy.
            gpu = self.gpu_memory()
            capacity, mode = self._capacity_for_memory(gpu)
            with self._condition:
                if self._active < capacity:
                    self._active += 1
                    try:
                        slot = PipelineSlot(
                            active_at_acquire=self._active,
                            capacity_at_acquire=capacity,
                            wait_seconds=time.perf_counter() - started,
                            gpu=gpu,
                            mode=mode,
                        )
                        self._admission_times.append(time.perf_counter())
                        print(
                            (
                                f"[scheduler] acquired active={self._active}/{capacity} "
                                f"mode={mode} wait={slot.wait_seconds:.2f}s request={request_label}"
                            ),
                            flush=True,
                        )
                    except Exception:
                        # Nothing below this point has run yet, so the slot was never
                        # actually handed out -- release the reservation before propagating,
                        # or _active permanently overcounts by one (a leaked slot that
                        # starves the scheduler forever).
                        self._active = max(0, self._active - 1)
                        self._condition.notify_all()
                        raise
                    break
                now = time.perf_counter()
                if now - self._last_wait_log >= DEFAULT_WAIT_LOG_SECONDS:
                    self._last_wait_log = now
                    print(
                        (
                            f"[scheduler] waiting active={self._active}/{capacity} "
                            f"mode={mode} free={gpu.free_mb if gpu else 'unknown'}MB request={request_label}"
                        ),
                        flush=True,
                    )
                self._condition.wait(timeout=1.0)
        try:
            yield slot
        finally:
            with self._condition:
                self._active = max(0, self._active - 1)
                self._condition.notify_all()
                print(f"[scheduler] released active={self._active} request={request_label}", flush=True)


PIPELINE_SCHEDULER = AdaptiveGpuScheduler()
