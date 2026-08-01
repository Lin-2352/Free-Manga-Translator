"""Regression test: main._diagnostic_json must redact sensitive keys nested inside lists
of dicts, not only bare dicts -- {"frames": [{"imageData": "..."}]} previously passed the
imageData through unredacted because the redaction loop only recursed into dict values.
Also checks a depth cap protects against pathological deep nesting.
"""
from __future__ import annotations

import json
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

from backend_api.app import main


def test_diagnostic_json_list_redaction() -> None:
    payload = {"frames": [{"imageData": "SECRET_BYTES_SHOULD_NOT_APPEAR"}]}
    safe = json.loads(main._diagnostic_json(payload))
    assert safe["frames"][0]["imageData"] == "<redacted>", safe

    # Depth cap: pathologically deep nesting must not blow the stack / run away.
    deep: dict = {"imageData": "leaf"}
    for _ in range(2000):
        deep = {"nested": deep}
    result = main._diagnostic_json(deep)  # must return, not recurse unbounded
    assert isinstance(result, str)

    print("diagnostic_json_list_redaction=pass")


if __name__ == "__main__":
    test_diagnostic_json_list_redaction()
