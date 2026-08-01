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


def test_main_print_loop_survives_a_cp1252_console() -> None:
    """Regression test: test_env_api_keys.py's main() prints CheckResult.detail, which can
    carry live API response text including non-ASCII (e.g. the Japanese translation-probe
    prompt's echoed response). On a plain Windows console (cp1252, the default for a fresh
    PowerShell/Task Scheduler/IDE run config with no PYTHONIOENCODING set), that print()
    raised UnicodeEncodeError and failed the test for a reason unrelated to the API keys it
    actually checks. Confirmed as the only diagnostics/test_*.py entrypoint exposed to this:
    every sibling either imports run_extension_pipeline_server (which reconfigures stdout to
    UTF-8 on import) or reconfigures stdout itself.

    Drives the REAL test_env_api_keys.main() (not a hand-written reproduction of its print
    shape) under a forced cp1252 console, in a real subprocess -- the crash is a real
    OS-level stdout encoding failure that only reproduces through an actual process with that
    console encoding active. The subprocess monkeypatches the network-calling test_*()
    functions to return instantly with a synthetic CJK-carrying CheckResult, so this stays
    fast and offline while still executing the real file's own main()/print()/reconfigure
    code, not a substitute for it."""
    script = (
        "import sys\n"
        "sys.path.insert(0, r'" + str((PROJECT_ROOT / 'python' / 'diagnostics')) + "')\n"
        "import test_env_api_keys as tek\n"
        "\n"
        "def _fake_summary(provider):\n"
        "    detail = '\\u3053\\u3093\\u306b\\u3061\\u306f translated as hello'\n"
        "    result = tek.CheckResult(provider=provider, check='text/translation', status='pass',\n"
        "                             detail=detail, model='test-model', http_status=200)\n"
        "    return tek.ProviderSummary(provider=provider, configured=True,\n"
        "                                uses_current_pipeline=True,\n"
        "                                required_for_current_pipeline=False, results=[result])\n"
        "\n"
        "for _name in ('test_gemini', 'test_mistral', 'test_github', 'test_openrouter',\n"
        "              'test_groq', 'test_nvidia', 'test_cerebras', 'test_fireworks',\n"
        "              'test_cloudflare'):\n"
        "    setattr(tek, _name, lambda n=_name: _fake_summary(n))\n"
        "\n"
        "raise SystemExit(tek.main())\n"
    )
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp1252"
    env.pop("PYTHONUTF8", None)

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        timeout=30,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, (
        "running the real test_env_api_keys.main() with a CJK-carrying detail string under a "
        f"forced cp1252 console crashed (returncode={result.returncode}) -- the UTF-8 "
        f"reconfigure guard in main() isn't working. stderr:\n"
        f"{result.stderr.decode('utf-8', errors='replace')[-2000:]}"
    )
    stdout_text = result.stdout.decode("utf-8", errors="replace")
    assert "translated as hello" in stdout_text, (
        "main() ran but didn't print the synthetic CJK detail text -- the monkeypatch didn't "
        f"take effect as expected. stdout:\n{stdout_text[-2000:]}"
    )


def test_env_api_keys_main_has_the_reconfigure_guard() -> None:
    """Static check that the guard actually landed in the real file (not just proven in a
    synthetic reproduction above) -- main() must reconfigure stdout to UTF-8 before its first
    print() call."""
    source = (PROJECT_ROOT / "python" / "diagnostics" / "test_env_api_keys.py").read_text(encoding="utf-8")
    main_start = source.index("def main() -> int:")
    # Strip comment-only lines before searching for the first real print() CALL -- a naive
    # substring search for "print(" would also match the word "print()" inside a comment
    # explaining the fix (as this very file's docstring does), giving a false position.
    main_body_lines = source[main_start:].splitlines()
    code_only = "\n".join(
        line for line in main_body_lines if not line.strip().startswith("#")
    )
    first_print = code_only.index("print(")
    reconfigure_call = code_only.find("sys.stdout.reconfigure(encoding=")
    assert reconfigure_call != -1, "main() no longer reconfigures stdout to UTF-8"
    assert reconfigure_call < first_print, (
        "sys.stdout.reconfigure(...) must run BEFORE the first print() in main(), "
        "otherwise an early CJK print can still crash before the guard takes effect"
    )


def main() -> int:
    test_main_print_loop_survives_a_cp1252_console()
    test_env_api_keys_main_has_the_reconfigure_guard()
    print("env_api_keys_cp1252_safe=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
