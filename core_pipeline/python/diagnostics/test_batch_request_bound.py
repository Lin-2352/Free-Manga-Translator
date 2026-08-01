"""Regression test: BatchRequest.images must be bounded -- an unbounded list is a
resource-exhaustion risk (each item spins up pipeline work on a shared executor).
"""
from __future__ import annotations

import sys
from pathlib import Path

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

from pydantic import ValidationError

from backend_api.app.schemas import BatchRequest, TranslateRequest


def test_batch_request_bound() -> None:
    one_image = {"imageData": "data:image/png;base64,x"}

    # A reasonable-sized batch must still be accepted.
    ok = BatchRequest(images=[TranslateRequest(**one_image) for _ in range(20)])
    assert len(ok.images) == 20

    # An oversized batch must be rejected at validation time, not accepted and then
    # exhaust the executor/thread pool.
    try:
        BatchRequest(images=[TranslateRequest(**one_image) for _ in range(21)])
        raise AssertionError("expected ValidationError for an oversized batch, none raised")
    except ValidationError:
        pass

    print("batch_request_bound=pass")


if __name__ == "__main__":
    test_batch_request_bound()
