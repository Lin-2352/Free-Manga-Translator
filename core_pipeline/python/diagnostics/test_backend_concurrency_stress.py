from __future__ import annotations

# Stress suite for the backend concurrency stack: AdaptiveGpuScheduler (gpu_scheduler.py),
# the sample-lock + runtime-output-cache fast path (pipeline_bridge.py), and JOBS bookkeeping
# (main.py). Mirrors test_runtime_output_cache.py's monkeypatch/temp-SAMPLES_ROOT style, but
# drives real threads at the real concurrency primitives instead of simulating single calls,
# so it can demonstrate genuine races (not just unit-level behaviour) before they are fixed.

import base64
import io
import shutil
import sys
import threading
import time
from pathlib import Path

from PIL import Image

_BOOTSTRAP_FILE = Path(__file__).resolve()
for _candidate in _BOOTSTRAP_FILE.parents:
    if (_candidate / "samples").exists() and (_candidate / "python").exists():
        _PROJECT_ROOT_FOR_IMPORTS = _candidate
        break
else:
    _PROJECT_ROOT_FOR_IMPORTS = _BOOTSTRAP_FILE.parents[2]
for _rel in ("", "python/common", "python/runtime"):
    _path = str(_PROJECT_ROOT_FOR_IMPORTS / _rel)
    if _path not in sys.path:
        sys.path.insert(0, _path)
del _BOOTSTRAP_FILE, _candidate, _PROJECT_ROOT_FOR_IMPORTS, _rel, _path

from fastapi import HTTPException

from backend_api.app import pipeline_bridge
from backend_api.app import main as backend_main
from backend_api.app.gpu_scheduler import PIPELINE_SCHEDULER, GpuMemory
from backend_api.app.schemas import TranslateRequest
import run_extension_pipeline_server as runtime_bridge


# --------------------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------------------

def _png_bytes(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(output, format="PNG")
    return output.getvalue()


def _payload_for(image_bytes: bytes, *, language: str = "ja", metadata: dict | None = None) -> dict:
    return {
        "imageData": f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}",
        "sourceLanguage": language,
        "targetLanguage": "en",
        "qualityProfile": "strict",
        "metadata": metadata or {},
    }


def _write_minimal_pass_artifacts(sample_name: str, language: str) -> None:
    sample_path = runtime_bridge.SAMPLES_ROOT / sample_name
    (sample_path / "step_1_detect").mkdir(parents=True, exist_ok=True)
    (sample_path / "step_4_final").mkdir(parents=True, exist_ok=True)
    (sample_path / "step_5_ocr").mkdir(parents=True, exist_ok=True)
    (sample_path / "step_6_layout").mkdir(parents=True, exist_ok=True)
    (sample_path / "step_7_translate").mkdir(parents=True, exist_ok=True)
    (sample_path / "step_8_typeset").mkdir(parents=True, exist_ok=True)

    (sample_path / "step_1_detect" / "detections.json").write_text("[]", encoding="utf-8")
    (sample_path / "step_1_detect" / "semantic_detections.json").write_text("[]", encoding="utf-8")
    (sample_path / "step_5_ocr" / "ocr_results.json").write_text("[]", encoding="utf-8")
    (sample_path / "step_6_layout" / "layout_constraints.json").write_text('[{"id": 0}]', encoding="utf-8")
    (sample_path / "step_6_layout" / "rejected_layout_items.json").write_text("[]", encoding="utf-8")
    (sample_path / "step_7_translate" / "translation_results.json").write_text(
        '[{"id": 0, "en_text": "HELLO"}]',
        encoding="utf-8",
    )
    (sample_path / "step_8_typeset" / "typeset_report.json").write_text(
        '[{"id": 0, "status": "fit"}]',
        encoding="utf-8",
    )
    Image.new("RGB", (32, 32), (255, 255, 255)).save(sample_path / "step_4_final" / "inpainted_result.jpg")
    Image.new("RGB", (32, 32), (255, 255, 255)).save(sample_path / "step_8_typeset" / "final_output.png")
    report = runtime_bridge._collect_runtime_report(sample_name, language)
    runtime_bridge._assert_runtime_report_safe(report)
    runtime_bridge._write_runtime_cache_meta(sample_name, language, report)


_RUN_LOCK = threading.Lock()
_RUN_COUNT = 0
_CONCURRENCY_LOCK = threading.Lock()
_CONCURRENT = 0
_CONCURRENCY_HIGH_WATER = 0


def _reset_instrumentation() -> None:
    global _RUN_COUNT, _CONCURRENT, _CONCURRENCY_HIGH_WATER
    with _RUN_LOCK:
        _RUN_COUNT = 0
    with _CONCURRENCY_LOCK:
        _CONCURRENT = 0
        _CONCURRENCY_HIGH_WATER = 0


def _make_fake_run_runtime_pipeline(sleep_seconds: float = 0.15, stop_aware: bool = True):
    """A stand-in for the real (heavy, model-loading) _run_runtime_pipeline that still
    exercises the real _begin_runtime_job/_end_runtime_job/generation-token stop machinery,
    just without running any actual ML stages. Mirrors the real function's signature
    (sample_name, language, stop_generation=...) so it actually exercises the generation
    comparison in _is_stop_current rather than a simplified stand-in for it."""

    def _fake(sample_name: str, language: str, stop_generation: int | None = None, page_key: str = "") -> dict:
        global _RUN_COUNT, _CONCURRENT, _CONCURRENCY_HIGH_WATER
        job_generation = (
            stop_generation if stop_generation is not None else runtime_bridge.current_stop_generation()
        )
        with _CONCURRENCY_LOCK:
            _CONCURRENT += 1
            _CONCURRENCY_HIGH_WATER = max(_CONCURRENCY_HIGH_WATER, _CONCURRENT)
        runtime_bridge._begin_runtime_job(sample_name, stop_generation=job_generation)
        try:
            with _RUN_LOCK:
                _RUN_COUNT += 1
            deadline = time.monotonic() + sleep_seconds
            while time.monotonic() < deadline:
                if stop_aware and runtime_bridge._is_stop_current(job_generation):
                    raise RuntimeError("Translation stopped by user")
                time.sleep(0.005)
            _write_minimal_pass_artifacts(sample_name, language)
            report = runtime_bridge._collect_runtime_report(sample_name, language)
            runtime_bridge._assert_runtime_report_safe(report)
            runtime_bridge._write_runtime_cache_meta(sample_name, language, report)
            return report
        finally:
            runtime_bridge._end_runtime_job(sample_name)
            with _CONCURRENCY_LOCK:
                _CONCURRENT -= 1

    return _fake


def _reset_global_runtime_state() -> None:
    runtime_bridge.RUNTIME_STOP_EVENT.clear()
    with runtime_bridge.RUNTIME_ACTIVITY_LOCK:
        runtime_bridge.RUNTIME_ACTIVE_JOBS = 0
        runtime_bridge.ACTIVE_RUNTIME_SAMPLES.clear()
        runtime_bridge.RUNTIME_PENDING_RELEASE = False
    with PIPELINE_SCHEDULER._condition:
        PIPELINE_SCHEDULER._active = 0
    if hasattr(PIPELINE_SCHEDULER, "gpu_memory") and "gpu_memory" in PIPELINE_SCHEDULER.__dict__:
        del PIPELINE_SCHEDULER.__dict__["gpu_memory"]
    if "_capacity_for_memory" in PIPELINE_SCHEDULER.__dict__:
        del PIPELINE_SCHEDULER.__dict__["_capacity_for_memory"]


def _join_all(threads: list[threading.Thread], timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    for t in threads:
        remaining = max(0.0, deadline - time.monotonic())
        t.join(timeout=remaining)
        assert not t.is_alive(), "a worker thread did not finish within the bounded join timeout (possible deadlock)"


# --------------------------------------------------------------------------------------
# Scenario 1 -- capacity cap: with capacity forced to 2, 12 distinct images across 12
# threads must never let more than 2 run concurrently, and all must complete.
# --------------------------------------------------------------------------------------
def scenario_capacity_cap():
    PIPELINE_SCHEDULER.__dict__["_capacity_for_memory"] = lambda gpu: (2, "forced_test_capacity")
    original_runner = runtime_bridge._run_runtime_pipeline
    runtime_bridge._run_runtime_pipeline = _make_fake_run_runtime_pipeline(0.12)
    _reset_instrumentation()
    errors: list[BaseException] = []
    threads = []
    try:
        def _worker(n: int):
            try:
                payload = _payload_for(_png_bytes((n % 256, (n * 7) % 256, (n * 13) % 256)), metadata={"cacheId": f"cap-{n}"})
                pipeline_bridge.run_pipeline_payload(payload)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        for n in range(12):
            t = threading.Thread(target=_worker, args=(n,))
            threads.append(t)
            t.start()
        _join_all(threads)
    finally:
        runtime_bridge._run_runtime_pipeline = original_runner

    assert not errors, f"unexpected errors under capacity cap: {errors}"
    assert _CONCURRENCY_HIGH_WATER <= 2, f"capacity cap violated: high-water={_CONCURRENCY_HIGH_WATER}"
    assert PIPELINE_SCHEDULER._active == 0, f"scheduler slot leaked: _active={PIPELINE_SCHEDULER._active}"


# --------------------------------------------------------------------------------------
# Scenario 2 -- same-image herd: 8 threads requesting byte-identical images must produce
# exactly one real pipeline run and seven runtime-output-cache hits.
# --------------------------------------------------------------------------------------
def scenario_same_image_herd():
    original_runner = runtime_bridge._run_runtime_pipeline
    runtime_bridge._run_runtime_pipeline = _make_fake_run_runtime_pipeline(0.1)
    _reset_instrumentation()
    image_bytes = _png_bytes((10, 20, 30))
    results: list[dict] = []
    lock = threading.Lock()
    errors: list[BaseException] = []
    threads = []
    try:
        def _worker():
            try:
                payload = _payload_for(image_bytes, metadata={"cacheId": "herd"})
                result = pipeline_bridge.run_pipeline_payload(payload)
                with lock:
                    results.append(result)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        for _ in range(8):
            t = threading.Thread(target=_worker)
            threads.append(t)
            t.start()
        _join_all(threads)
    finally:
        runtime_bridge._run_runtime_pipeline = original_runner

    assert not errors, f"unexpected errors in same-image herd: {errors}"
    assert len(results) == 8
    assert _RUN_COUNT == 1, f"expected exactly one real pipeline run for a byte-identical herd, ran {_RUN_COUNT}"
    hits = sum(1 for r in results if r["report"].get("runtimeOutputCache") == "hit")
    assert hits == 7, f"expected 7 of 8 identical requests to be cache hits, got {hits}"


# --------------------------------------------------------------------------------------
# Scenario 3 -- cache-hit read vs /v1/cache/clear hammer (Bug B1). Readers repeatedly hit
# an already-cached sample while clearers repeatedly wipe the runtime output cache. No
# unhandled exception should ever escape either side.
# --------------------------------------------------------------------------------------
def scenario_cache_hit_vs_clear_hammer():
    # Pin a cheap synthetic capacity so this scenario isolates the B1 cache-hit/clear race
    # instead of also being throttled by the real nvidia-smi probe running inside the
    # scheduler's lock on every acquire (a separate, real slowdown -- see Bug B6).
    PIPELINE_SCHEDULER.__dict__["_capacity_for_memory"] = lambda gpu: (4, "forced_test_capacity")
    original_runner = runtime_bridge._run_runtime_pipeline
    runtime_bridge._run_runtime_pipeline = _make_fake_run_runtime_pipeline(0.02)
    image_bytes = _png_bytes((77, 88, 99))
    language = "ja"
    sample_name, _ = runtime_bridge._write_runtime_sample(image_bytes, language, preserve_reusable_output=False)
    _write_minimal_pass_artifacts(sample_name, language)

    stop_flag = threading.Event()
    errors: list[BaseException] = []
    errors_lock = threading.Lock()
    iterations_lock = threading.Lock()
    reader_iterations = 0
    clearer_iterations = 0

    def _reader():
        nonlocal reader_iterations
        while not stop_flag.is_set():
            try:
                payload = _payload_for(image_bytes, language=language, metadata={"cacheId": "hammer"})
                pipeline_bridge.run_pipeline_payload(payload)
                with iterations_lock:
                    reader_iterations += 1
            except BaseException as exc:  # noqa: BLE001
                with errors_lock:
                    errors.append(exc)

    def _clearer():
        nonlocal clearer_iterations
        while not stop_flag.is_set():
            try:
                runtime_bridge.clear_runtime_output_cache()
                with iterations_lock:
                    clearer_iterations += 1
            except BaseException as exc:  # noqa: BLE001
                with errors_lock:
                    errors.append(exc)
            time.sleep(0.005)

    threads = [threading.Thread(target=_reader) for _ in range(6)]
    threads += [threading.Thread(target=_clearer) for _ in range(3)]
    try:
        for t in threads:
            t.start()
        time.sleep(1.2)
    finally:
        stop_flag.set()
        runtime_bridge._run_runtime_pipeline = original_runner
        _join_all(threads, timeout=10.0)

    assert reader_iterations > 0 and clearer_iterations > 0, "hammer did not actually run concurrently"
    assert not errors, f"unhandled exception(s) racing cache-hit reads against cache clear: {errors[:3]}"


# --------------------------------------------------------------------------------------
# Scenario 4 -- stop semantics (Bug B2). With capacity forced to 1, job A runs while job B
# waits on the scheduler slot. A stop request must cancel BOTH; a fresh request submitted
# after the stop must succeed normally (generation-token semantics, not a sticky global).
# --------------------------------------------------------------------------------------
def scenario_stop_semantics():
    PIPELINE_SCHEDULER.__dict__["_capacity_for_memory"] = lambda gpu: (1, "forced_sequential_test")
    original_runner = runtime_bridge._run_runtime_pipeline
    runtime_bridge._run_runtime_pipeline = _make_fake_run_runtime_pipeline(0.35, stop_aware=True)

    outcomes: dict[str, str] = {}
    outcomes_lock = threading.Lock()

    def _run(label: str, image_bytes: bytes):
        try:
            payload = _payload_for(image_bytes, metadata={"cacheId": label})
            pipeline_bridge.run_pipeline_payload(payload)
            with outcomes_lock:
                outcomes[label] = "success"
        except BaseException as exc:  # noqa: BLE001
            with outcomes_lock:
                outcomes[label] = f"error:{exc}"

    try:
        thread_a = threading.Thread(target=_run, args=("A", _png_bytes((1, 2, 3))))
        thread_a.start()
        time.sleep(0.05)  # let A acquire the (capacity=1) scheduler slot and begin
        thread_b = threading.Thread(target=_run, args=("B", _png_bytes((4, 5, 6))))
        thread_b.start()
        time.sleep(0.05)  # B should now be blocked waiting on the scheduler slot

        runtime_bridge.request_runtime_stop(release_gpu=False, reason="stress_test_stop")
        _join_all([thread_a, thread_b], timeout=10.0)

        assert outcomes.get("A", "").startswith("error"), f"job A should abort on stop, got {outcomes.get('A')}"
        assert outcomes.get("B", "").startswith("error"), (
            f"job B (queued behind the scheduler at stop time) should also abort on stop, "
            f"got {outcomes.get('B')} -- a global stop-clearing event lets later jobs erase an earlier stop"
        )

        # A fresh request submitted after the stop has resolved must succeed normally.
        runtime_bridge.RUNTIME_STOP_EVENT.clear()
        thread_c = threading.Thread(target=_run, args=("C", _png_bytes((7, 8, 9))))
        thread_c.start()
        _join_all([thread_c], timeout=10.0)
        assert outcomes.get("C") == "success", f"a request submitted after the stop resolved should succeed, got {outcomes.get('C')}"
    finally:
        runtime_bridge._run_runtime_pipeline = original_runner
        runtime_bridge.RUNTIME_STOP_EVENT.clear()


# --------------------------------------------------------------------------------------
# Scenario 5 -- probe-failure degradation: no GPU memory probe available must degrade to
# sequential mode, not crash or deadlock.
# --------------------------------------------------------------------------------------
def scenario_probe_failure_degradation():
    PIPELINE_SCHEDULER.__dict__["gpu_memory"] = lambda: None
    original_runner = runtime_bridge._run_runtime_pipeline
    runtime_bridge._run_runtime_pipeline = _make_fake_run_runtime_pipeline(0.05)
    try:
        status = PIPELINE_SCHEDULER.status()
        assert status["mode"] == "sequential_no_gpu_memory_probe", status
        assert status["capacity"] == 1, status

        errors: list[BaseException] = []
        threads = []

        def _worker(n: int):
            try:
                payload = _payload_for(_png_bytes((n, n, n)), metadata={"cacheId": f"probe-{n}"})
                pipeline_bridge.run_pipeline_payload(payload)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        for n in range(3):
            t = threading.Thread(target=_worker, args=(n,))
            threads.append(t)
            t.start()
        _join_all(threads)
        assert not errors, f"probe-failure degradation should still complete requests: {errors}"
    finally:
        runtime_bridge._run_runtime_pipeline = original_runner


# --------------------------------------------------------------------------------------
# Scenario 6 -- slot-leak guard (Bug B3). If anything raises between incrementing _active
# and the try/finally that releases it, the slot must not leak.
# --------------------------------------------------------------------------------------
def scenario_slot_leak_guard():
    import backend_api.app.gpu_scheduler as gpu_scheduler_module

    raise_once = {"armed": True}
    real_print = print

    def _maybe_raising_print(*args, **kwargs):
        if raise_once["armed"] and args and str(args[0]).startswith("[scheduler] acquired"):
            raise_once["armed"] = False
            raise RuntimeError("simulated failure between _active increment and try/finally")
        return real_print(*args, **kwargs)

    gpu_scheduler_module.print = _maybe_raising_print
    original_runner = runtime_bridge._run_runtime_pipeline
    runtime_bridge._run_runtime_pipeline = _make_fake_run_runtime_pipeline(0.02)
    try:
        payload = _payload_for(_png_bytes((200, 201, 202)), metadata={"cacheId": "leak-guard"})
        try:
            pipeline_bridge.run_pipeline_payload(payload)
        except Exception:
            pass  # the simulated failure is expected to surface; what matters is the slot afterward

        assert PIPELINE_SCHEDULER._active == 0, (
            f"scheduler slot leaked after a failure between acquire's increment and its try/finally: "
            f"_active={PIPELINE_SCHEDULER._active}"
        )

        # The scheduler must still be usable afterward (not permanently wedged).
        payload2 = _payload_for(_png_bytes((203, 204, 205)), metadata={"cacheId": "leak-guard-followup"})
        pipeline_bridge.run_pipeline_payload(payload2)
        assert PIPELINE_SCHEDULER._active == 0
    finally:
        runtime_bridge._run_runtime_pipeline = original_runner
        del gpu_scheduler_module.print


# --------------------------------------------------------------------------------------
# Scenario 7 -- lock-map bound (Bug B4). Many distinct images processed over time must not
# leave _SAMPLE_LOCKS growing without bound.
# --------------------------------------------------------------------------------------
def scenario_lock_map_bound():
    original_runner = runtime_bridge._run_runtime_pipeline
    runtime_bridge._run_runtime_pipeline = _make_fake_run_runtime_pipeline(0.0)
    before = len(pipeline_bridge._SAMPLE_LOCKS)
    try:
        for n in range(300):
            payload = _payload_for(_png_bytes((n % 256, (n * 3) % 256, (n * 5) % 256)), metadata={"cacheId": f"lock-{n}"})
            pipeline_bridge.run_pipeline_payload(payload)
    finally:
        runtime_bridge._run_runtime_pipeline = original_runner

    grown_by = len(pipeline_bridge._SAMPLE_LOCKS) - before
    assert grown_by <= 20, (
        f"_SAMPLE_LOCKS grew by {grown_by} after 300 distinct one-off images; "
        "locks for images no longer in use should be evicted, not retained forever"
    )


# --------------------------------------------------------------------------------------
# Scenario 8 -- JOBS churn (Bug B5). A request that raises HTTPException before pipeline
# work begins must not crash with KeyError if its JOBS entry gets evicted by concurrent
# churn while its exception handler is still running.
# --------------------------------------------------------------------------------------
def scenario_jobs_churn():
    original_runner = runtime_bridge._run_runtime_pipeline
    runtime_bridge._run_runtime_pipeline = _make_fake_run_runtime_pipeline(0.0)
    original_request_payload = backend_main._request_payload

    poisoned_ready = threading.Event()
    resume_poisoned = threading.Event()

    def _slow_request_payload(request):
        if isinstance(request.metadata, dict) and request.metadata.get("poisoned"):
            poisoned_ready.set()
            resume_poisoned.wait(timeout=60)
        return original_request_payload(request)

    backend_main._request_payload = _slow_request_payload
    errors: list[BaseException] = []

    def _run_poisoned():
        try:
            request = TranslateRequest(imageData=None, base64Data=None, metadata={"poisoned": True})
            backend_main._execute_translate(request)
            errors.append(AssertionError("poisoned request should have raised HTTPException(400)"))
        except HTTPException:
            pass
        except BaseException as exc:  # noqa: BLE001 -- this is exactly what we're checking for
            errors.append(exc)

    try:
        good_image = _png_bytes((250, 251, 252))
        poisoned_thread = threading.Thread(target=_run_poisoned)
        poisoned_thread.start()
        assert poisoned_ready.wait(timeout=5), "poisoned request never reached the delay hook"

        for _ in range(backend_main._JOBS_MAX_ENTRIES + 50):
            request = TranslateRequest(imageData=f"data:image/png;base64,{base64.b64encode(good_image).decode('ascii')}")
            backend_main._execute_translate(request)

        resume_poisoned.set()
        _join_all([poisoned_thread], timeout=10.0)

        assert not errors, f"JOBS churn during a poisoned request's error handling raised: {errors}"
        assert len(backend_main.JOBS) <= backend_main._JOBS_MAX_ENTRIES, (
            f"JOBS exceeded its {backend_main._JOBS_MAX_ENTRIES}-entry bound: {len(backend_main.JOBS)}"
        )
    finally:
        runtime_bridge._run_runtime_pipeline = original_runner
        backend_main._request_payload = original_request_payload


# --------------------------------------------------------------------------------------
scenarios = [
    ("1_capacity_cap", scenario_capacity_cap),
    ("2_same_image_herd", scenario_same_image_herd),
    ("3_cache_hit_vs_clear_hammer_B1", scenario_cache_hit_vs_clear_hammer),
    ("4_stop_semantics_B2", scenario_stop_semantics),
    ("5_probe_failure_degradation", scenario_probe_failure_degradation),
    ("6_slot_leak_guard_B3", scenario_slot_leak_guard),
    ("7_lock_map_bound_B4", scenario_lock_map_bound),
    ("8_jobs_churn_B5", scenario_jobs_churn),
]


def main() -> int:
    original_samples_root = runtime_bridge.SAMPLES_ROOT
    temp_root = original_samples_root.parent / "_diagnostics_backend_concurrency_stress"
    shutil.rmtree(temp_root, ignore_errors=True)
    runtime_bridge.SAMPLES_ROOT = temp_root

    results = []
    try:
        for name, fn in scenarios:
            _reset_global_runtime_state()
            _reset_instrumentation()
            try:
                fn()
                results.append((name, "pass", None))
                print(f"[backend_concurrency_stress] {name}: pass")
            except BaseException as exc:  # noqa: BLE001
                results.append((name, "fail", exc))
                print(f"[backend_concurrency_stress] {name}: FAIL -- {exc}")
            finally:
                _reset_global_runtime_state()
    finally:
        runtime_bridge.SAMPLES_ROOT = original_samples_root
        shutil.rmtree(temp_root, ignore_errors=True)

    failed = [r for r in results if r[1] == "fail"]
    print(f"\nbackend_concurrency_stress_test summary: {len(results) - len(failed)}/{len(results)} passed")
    if failed:
        print("Failed scenarios:", ", ".join(r[0] for r in failed))
        return 1
    print("backend_concurrency_stress_test=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
