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
for _rel in ("python/common", "python/steps"):
    _path = str(_PROJECT_ROOT_FOR_IMPORTS / _rel)
    if _path not in sys.path:
        sys.path.insert(0, _path)
del _BOOTSTRAP_FILE, _candidate, _PROJECT_ROOT_FOR_IMPORTS, _rel, _path

import run_step7_translate as step7


def test_scrub_redacts_numbered_rotated_key_env_vars() -> None:
    """Regression test for Tier B #8: SECRET_ENV_NAMES only listed the bare
    plural form (GEMINI_API_KEYS), but this project's actual .env convention is
    numbered/rotated keys (GEMINI_API_KEY_1, _2, _3, ...), which _csv_env
    already expands for real key lookup -- just not for the scrub/redaction
    list previously. A bare raw key from GEMINI_API_KEY_2 could then leak
    into a log/error message unredacted. The fix derives the scrub list from
    API_MANAGER.providers[*].env_names on every call (also picking up a
    mid-session key rotation, which a frozen-at-import list could not)."""
    original = {}
    probe_names = ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"]
    for name in probe_names:
        original[name] = os.environ.get(name)
    try:
        os.environ["GEMINI_API_KEY_1"] = "sk-probe-numbered-one-abc123"
        os.environ["GEMINI_API_KEY_2"] = "sk-probe-numbered-two-def456"
        os.environ.pop("GEMINI_API_KEY_3", None)

        message = "request failed for key sk-probe-numbered-one-abc123 and sk-probe-numbered-two-def456"
        scrubbed = step7._scrub_secret(message)

        assert "sk-probe-numbered-one-abc123" not in scrubbed, (
            f"GEMINI_API_KEY_1's value leaked unredacted: {scrubbed}"
        )
        assert "sk-probe-numbered-two-def456" not in scrubbed, (
            f"GEMINI_API_KEY_2's value leaked unredacted: {scrubbed}"
        )
        assert scrubbed.count("[REDACTED]") == 2, f"expected both numbered keys redacted, got: {scrubbed}"

        # Mid-session rotation: a key that didn't exist when the module was
        # imported must still be scrubbed once it's set later.
        os.environ["GEMINI_API_KEY_3"] = "sk-probe-numbered-three-ghi789"
        rotated_message = "later failure mentioning sk-probe-numbered-three-ghi789"
        rotated_scrubbed = step7._scrub_secret(rotated_message)
        assert "sk-probe-numbered-three-ghi789" not in rotated_scrubbed, (
            f"a key rotated in mid-session (added after import) must still be scrubbed, got: {rotated_scrubbed}"
        )
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main() -> int:
    test_scrub_redacts_numbered_rotated_key_env_vars()
    print("step7_secret_scrub_numbered_keys=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
