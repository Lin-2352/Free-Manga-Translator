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
for _rel in ("python/common", "python/steps", "python/runtime"):
    _path = str(_PROJECT_ROOT_FOR_IMPORTS / _rel)
    if _path not in sys.path:
        sys.path.insert(0, _path)
del _BOOTSTRAP_FILE, _candidate, _PROJECT_ROOT_FOR_IMPORTS, _rel, _path

import run_step7_translate
import run_extension_pipeline_server


def test_release_runtime_models_clears_nllb_translator_globals() -> None:
    """release_runtime_models() is called by the extension's GPU-release button, the
    hard-stop path, and the idle-unload timer -- its whole job is to free VRAM. Before this
    fix it freed step5 (OCR/detection), step4 (inpaint) and step8 (font cache) but never
    touched run_step7_translate's _LOCAL_TRANSLATOR / _LOCAL_TOKENIZERS globals, which hold
    the ~2.4 GB NLLB-200 translation model on CUDA whenever LOCAL_NLLB_TRANSLATION is used.
    That model stayed resident through every "release" the user could trigger."""
    original_translator = run_step7_translate._LOCAL_TRANSLATOR
    original_tokenizers = dict(run_step7_translate._LOCAL_TOKENIZERS)
    try:
        run_step7_translate._LOCAL_TRANSLATOR = ("fake_model", "cuda")
        run_step7_translate._LOCAL_TOKENIZERS = {"eng_Latn": "fake_tokenizer"}

        report = run_extension_pipeline_server.release_runtime_models(reason="test")

        assert report.get("success") is not False, f"release should not be deferred/failed in this test: {report}"
        assert run_step7_translate._LOCAL_TRANSLATOR is None, (
            "release_runtime_models() must clear run_step7_translate._LOCAL_TRANSLATOR -- "
            f"it is still set to {run_step7_translate._LOCAL_TRANSLATOR!r}"
        )
        assert run_step7_translate._LOCAL_TOKENIZERS == {}, (
            "release_runtime_models() must clear run_step7_translate._LOCAL_TOKENIZERS -- "
            f"it still holds {run_step7_translate._LOCAL_TOKENIZERS!r}"
        )
        released = report.get("released", [])
        assert any("run_step7_translate" in str(item) for item in released), (
            f"release report should list the freed step7 globals, got: {released}"
        )
    finally:
        run_step7_translate._LOCAL_TRANSLATOR = original_translator
        run_step7_translate._LOCAL_TOKENIZERS = original_tokenizers


def main() -> int:
    test_release_runtime_models_clears_nllb_translator_globals()
    print("release_frees_nllb_translator=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
