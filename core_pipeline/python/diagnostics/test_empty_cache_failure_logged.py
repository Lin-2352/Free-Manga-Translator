"""Regression test: a failing torch.cuda.empty_cache() during release_runtime_models()
must be logged (at minimum), not silently swallowed by a bare `except Exception: pass` --
previously a failed VRAM release reported success with zero trace of the failure.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_BOOTSTRAP_FILE = Path(__file__).resolve()
for _candidate in _BOOTSTRAP_FILE.parents:
    if (_candidate / "samples").exists() and (_candidate / "python").exists():
        _PROJECT_ROOT_FOR_IMPORTS = _candidate
        break
else:
    _PROJECT_ROOT_FOR_IMPORTS = _BOOTSTRAP_FILE.parents[2]
for _rel in (
    "python/common",
    "python/steps",
    "python/validation",
    "python/runtime",
    "python/downloaders",
    "python/reference",
    "python/diagnostics",
):
    _path = str(_PROJECT_ROOT_FOR_IMPORTS / _rel)
    if _path not in sys.path:
        sys.path.insert(0, _path)
del _BOOTSTRAP_FILE, _candidate, _PROJECT_ROOT_FOR_IMPORTS, _rel, _path

import run_extension_pipeline_server as bridge  # noqa: E402


class _FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def empty_cache():
        raise RuntimeError("simulated empty_cache failure")

    @staticmethod
    def ipc_collect():
        pass


class _FakeTorch:
    cuda = _FakeCuda()


def test_empty_cache_failure_logged() -> None:
    captured: list[dict] = []

    def _fake_write_diagnostic_event(event, details=None, **kwargs):
        captured.append({"event": event, "details": details or {}})
        return None

    original_write = bridge.write_diagnostic_event
    original_torch = sys.modules.get("torch")
    bridge.write_diagnostic_event = _fake_write_diagnostic_event
    sys.modules["torch"] = _FakeTorch()
    try:
        result = bridge.release_runtime_models(reason="test_empty_cache_failure")
        assert result.get("success") is not False or "status" in result

        matches = [c for c in captured if "empty_cache" in c["event"].lower() or "cuda" in c["event"].lower()]
        assert matches, (
            f"expected release_runtime_models to log the empty_cache() failure, "
            f"captured events: {[c['event'] for c in captured]}"
        )
        print("empty_cache_failure_logged=pass")
    finally:
        bridge.write_diagnostic_event = original_write
        if original_torch is not None:
            sys.modules["torch"] = original_torch
        else:
            sys.modules.pop("torch", None)


if __name__ == "__main__":
    test_empty_cache_failure_logged()
