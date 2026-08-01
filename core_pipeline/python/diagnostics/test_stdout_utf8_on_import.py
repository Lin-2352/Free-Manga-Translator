from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_BOOTSTRAP_FILE = Path(__file__).resolve()
for _candidate in _BOOTSTRAP_FILE.parents:
    if (_candidate / "samples").exists() and (_candidate / "python").exists():
        PROJECT_ROOT = _candidate
        break
else:
    PROJECT_ROOT = _BOOTSTRAP_FILE.parents[2]
del _BOOTSTRAP_FILE, _candidate


def test_module_import_sets_utf8_stdout_before_any_cjk_print() -> None:
    """Every launcher other than start_backend.ps1 (which sets PYTHONIOENCODING=utf-8 itself
    before starting uvicorn) can import run_extension_pipeline_server.py under a non-UTF-8
    console encoding -- a Windows service, Task Scheduler, or an IDE run config all default to
    the system code page (cp1252 on this machine). Several unconditional print()s of raw CJK
    OCR/translation text exist (run_step5_ocr.py, run_step7_translate.py), and previously the
    only sys.stdout.reconfigure(encoding="utf-8") call in run_extension_pipeline_server.py
    lived inside its `if __name__ == "__main__":` guard -- so any entry point that imports this
    module without running that guard (uvicorn's `backend_api.app.main:app` does exactly this)
    got no UTF-8 guarantee at all, and crashed with UnicodeEncodeError on the first
    Japanese/Korean/Chinese OCR result.

    This spawns a real subprocess with PYTHONIOENCODING explicitly forced to cp1252 (simulating
    a launcher that never sets it), imports the module the same way uvicorn does (plain
    `import`, never touching __main__), then prints raw CJK text and asserts it doesn't crash."""
    script = (
        "import run_extension_pipeline_server\n"
        "print('\\u4fde\\u5409\\u6fc6\\u6c0f \\u52de\\u52d5\\u591c\\u5b78\\u6703\\u9867\\u554f')\n"
    )
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp1252"
    env.pop("PYTHONUTF8", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT / "python" / "runtime"), env.get("PYTHONPATH", "")]
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        "printing CJK text right after `import run_extension_pipeline_server` crashed under a "
        f"cp1252 console encoding (returncode={result.returncode}). This is exactly the entry "
        "path uvicorn uses -- stderr:\n"
        f"{result.stderr.decode('utf-8', errors='replace')[-2000:]}"
    )


def main() -> int:
    test_module_import_sets_utf8_stdout_before_any_cjk_print()
    print("stdout_utf8_on_import=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
