from __future__ import annotations

import sys
import threading
from pathlib import Path

_BOOTSTRAP_FILE = Path(__file__).resolve()
for _candidate in _BOOTSTRAP_FILE.parents:
    if (_candidate / "samples").exists() and (_candidate / "python").exists():
        _PROJECT_ROOT_FOR_IMPORTS = _candidate
        break
else:
    _PROJECT_ROOT_FOR_IMPORTS = _BOOTSTRAP_FILE.parents[2]
for _rel in ("python/common", "python/steps", "python/runtime"):
    _path = str(_PROJECT_ROOT_FOR_IMPORTS / _rel)
    if _path not in sys.path:
        sys.path.insert(0, _path)
del _BOOTSTRAP_FILE, _candidate, _PROJECT_ROOT_FOR_IMPORTS, _rel, _path

import run_extension_pipeline_server as server


class _ThreadingProxy:
    """Rebinds only server.py's OWN `threading.Thread` lookup to a hooked subclass, without
    touching the real stdlib threading module used by this test's own orchestration threads
    or anything else in the process."""

    def __init__(self, real_threading, thread_class):
        self._real = real_threading
        self.Thread = thread_class

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_warmup_dispatch_atomic_with_release_check() -> None:
    """warm_runtime_models()'s background-dispatch path used to set WARMUP_STATE to "running"
    BEFORE creating/assigning WARMUP_THREAD (the module global). A release_runtime_models()
    call landing in that exact window saw status=="running" but WARMUP_THREAD stale (None or a
    previous finished thread) -- _warmup_thread_running() requires BOTH, so it returned False,
    and release proceeded as if nothing were in flight: clearing RUNTIME_PENDING_RELEASE and
    overwriting WARMUP_STATE to "released", even though the warmup thread was about to start
    and would overwrite state to "pass" once it finished, silently orphaning the release the
    caller thought had happened.

    This test reproduces the race deterministically with two real threads instead of relying
    on timing: a hook fires from inside threading.Thread's constructor (the exact point
    between the state-set and the thread-assignment in the original code), signals a second
    thread to run release_runtime_models() concurrently, and only then lets construction
    finish. The fix's job is to make release's check block until dispatch is fully consistent
    (both state AND thread committed together), never observe the torn combination."""
    original_thread_class = threading.Thread
    original_state = server.WARMUP_STATE
    original_thread = server.WARMUP_THREAD
    original_pending = server.RUNTIME_PENDING_RELEASE
    original_active_jobs = server.RUNTIME_ACTIVE_JOBS
    original_worker = server._warm_runtime_models_worker
    original_threading_name = server.threading

    hook_entered = threading.Event()
    release_probe_done = threading.Event()
    worker_may_finish = threading.Event()
    captured: dict = {}

    class HookThread(original_thread_class):
        def __init__(self, *args, **kwargs):
            hook_entered.set()
            release_probe_done.wait(timeout=5)
            super().__init__(*args, **kwargs)

    def fake_worker(force: bool = False):
        # Stays "alive" until explicitly released, so once the release probe finally gets to
        # run (after the fix makes it wait for dispatch to finish), it sees a genuinely still-
        # running warmup rather than racing against how fast a real worker happens to finish.
        worker_may_finish.wait(timeout=5)
        return {"status": "pass"}

    def run_release_probe() -> None:
        hook_entered.wait(timeout=5)
        captured["result"] = server.release_runtime_models(reason="race-probe-test")
        release_probe_done.set()

    server.WARMUP_STATE = {
        "status": "idle", "models": {}, "startedAt": None, "finishedAt": None, "seconds": None,
    }
    server.WARMUP_THREAD = None
    server.RUNTIME_PENDING_RELEASE = False
    server.RUNTIME_ACTIVE_JOBS = 0
    server.threading = _ThreadingProxy(original_threading_name, HookThread)
    server._warm_runtime_models_worker = fake_worker

    try:
        dispatch_thread = original_thread_class(
            target=lambda: server.warm_runtime_models(force=True, background=True)
        )
        probe_thread = original_thread_class(target=run_release_probe)

        dispatch_thread.start()
        probe_thread.start()
        dispatch_thread.join(timeout=10)
        probe_thread.join(timeout=10)
        worker_may_finish.set()

        assert not dispatch_thread.is_alive(), "dispatch thread did not finish -- likely deadlocked"
        assert not probe_thread.is_alive(), "release probe thread did not finish -- likely deadlocked"

        result = captured.get("result")
        assert result is not None, "release probe never ran -- test setup is broken"
        assert result.get("success") is False and result.get("status") == "deferred", (
            "release_runtime_models() must never observe warm_runtime_models()'s dispatch in a "
            "torn state (WARMUP_STATE already 'running' but WARMUP_THREAD not yet committed) -- "
            f"got: {result}"
        )
    finally:
        worker_may_finish.set()
        release_probe_done.set()
        server.threading = original_threading_name
        server._warm_runtime_models_worker = original_worker
        server.WARMUP_STATE = original_state
        server.WARMUP_THREAD = original_thread
        server.RUNTIME_PENDING_RELEASE = original_pending
        server.RUNTIME_ACTIVE_JOBS = original_active_jobs


def main() -> int:
    test_warmup_dispatch_atomic_with_release_check()
    print("warmup_dispatch_release_race=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
