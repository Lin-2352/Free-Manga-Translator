from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

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

import run_step5_ocr
import ml_region_lib


class _ProbeStop(Exception):
    """Raised deliberately right after step5's detection-model reload guard runs, so
    this test can verify the guard's reload decision without needing to fake the rest
    of the OCR/consolidation pipeline (real detection results, bubble masks, etc)."""


def test_reload_guard_repairs_a_partial_detection_model_wedge(tmp_path) -> None:
    """Regression test for the 500 crash: 'NoneType' object has no attribute
    'predict_detections_and_associations'.

    Root cause: the runtime server's startup warmup worker used to commit
    _TEXT_HANDLE / _BUBBLE_MODEL / _SEMANTIC_HANDLE one at a time as each model
    loaded. A request arriving mid-warmup (or a warmup that failed between the text
    and semantic loads) could observe _TEXT_HANDLE set but _SEMANTIC_HANDLE still
    None. Step 5's reload guard was `if text_handle is None`, which used one handle
    as a proxy for "all three are loaded" -- so it never repaired that partial
    state, and every request after it reused the stale non-None text_handle and
    crashed on the still-None semantic_handle.

    This test reproduces the exact partial state directly (bypassing the timing-
    dependent race) and asserts the broadened guard (`if text_handle is None or
    bubble_model is None or semantic_handle is None`) reloads all three instead of
    silently reusing the stale ones.
    """
    load_calls: list[str] = []
    detect_text_calls: list[object] = []

    def fake_load_text_model(path):
        load_calls.append("text")
        return "reloaded_text_handle"

    def fake_load_bubble_model(path):
        load_calls.append("bubble")
        return "reloaded_bubble_model", "cpu"

    def fake_load_semantic_model(name):
        load_calls.append("semantic")
        return "reloaded_semantic_handle"

    def fake_detect_text(handle, image, cfg):
        detect_text_calls.append(handle)
        raise _ProbeStop()

    original = {
        "load_text_model": run_step5_ocr.load_text_model,
        "load_bubble_model": run_step5_ocr.load_bubble_model,
        "load_semantic_model": run_step5_ocr.load_semantic_model,
        "detect_text": run_step5_ocr.detect_text,
        "_TEXT_HANDLE": run_step5_ocr._TEXT_HANDLE,
        "_BUBBLE_MODEL": run_step5_ocr._BUBBLE_MODEL,
        "_BUBBLE_DEVICE": run_step5_ocr._BUBBLE_DEVICE,
        "_SEMANTIC_HANDLE": run_step5_ocr._SEMANTIC_HANDLE,
    }
    try:
        run_step5_ocr.load_text_model = fake_load_text_model
        run_step5_ocr.load_bubble_model = fake_load_bubble_model
        run_step5_ocr.load_semantic_model = fake_load_semantic_model
        run_step5_ocr.detect_text = fake_detect_text

        # The exact partial wedge state: text handle present and non-None (as if warmup
        # or a prior request already loaded it), semantic handle still None.
        run_step5_ocr._TEXT_HANDLE = "stale_text_handle"
        run_step5_ocr._BUBBLE_MODEL = "stale_bubble_model"
        run_step5_ocr._BUBBLE_DEVICE = "cpu"
        run_step5_ocr._SEMANTIC_HANDLE = None

        sample_name = "wedge_probe_sample"
        sample_dir = tmp_path / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(sample_dir / "input.jpg")
        # No step_1_detect/ folder -- forces needs_step1_detection=True, which is the
        # branch containing the reload guard.

        raised = False
        try:
            run_step5_ocr.run_step5_ocr(sample_map={sample_name: "input.jpg"}, samples_dir=tmp_path)
        except _ProbeStop:
            raised = True
        assert raised, "expected the probe to reach detect_text() and stop there"

        assert load_calls == ["text", "bubble", "semantic"], (
            f"expected the reload guard to reload all three detection models when semantic_handle "
            f"was None even though text_handle was not, got load_calls={load_calls} -- if this is "
            f"empty, the guard regressed back to `if text_handle is None` only, which cannot repair "
            f"this exact wedge"
        )
        assert detect_text_calls == ["reloaded_text_handle"], (
            "detect_text should have been called with the freshly reloaded handle, not the stale one"
        )
        assert run_step5_ocr._SEMANTIC_HANDLE == "reloaded_semantic_handle", (
            "the module-global semantic handle should be repaired after the reload for subsequent calls"
        )
    finally:
        run_step5_ocr.load_text_model = original["load_text_model"]
        run_step5_ocr.load_bubble_model = original["load_bubble_model"]
        run_step5_ocr.load_semantic_model = original["load_semantic_model"]
        run_step5_ocr.detect_text = original["detect_text"]
        run_step5_ocr._TEXT_HANDLE = original["_TEXT_HANDLE"]
        run_step5_ocr._BUBBLE_MODEL = original["_BUBBLE_MODEL"]
        run_step5_ocr._BUBBLE_DEVICE = original["_BUBBLE_DEVICE"]
        run_step5_ocr._SEMANTIC_HANDLE = original["_SEMANTIC_HANDLE"]


def test_detect_semantic_text_regions_rejects_none_model() -> None:
    """Fix 1c: the final safety net. Even if some future caller reintroduces a path
    that can hand a None semantic_model handle to this function, it must fail with a
    clear, named RuntimeError instead of the cryptic
    'NoneType' object has no attribute 'predict_detections_and_associations'."""
    cfg = ml_region_lib.MLConfig()
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    try:
        ml_region_lib.detect_semantic_text_regions(None, image, cfg)
        raised = False
    except RuntimeError as error:
        raised = True
        assert "semantic_model=None" in str(error), f"expected a clear message, got: {error}"
    except AttributeError as error:
        raise AssertionError(
            f"detect_semantic_text_regions(None, ...) raised the old cryptic AttributeError "
            f"instead of a clear RuntimeError -- Fix 1c regressed: {error}"
        ) from error
    assert raised, "detect_semantic_text_regions(None, ...) should raise RuntimeError"


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_reload_guard_repairs_a_partial_detection_model_wedge(Path(tmp))
    test_detect_semantic_text_regions_rejects_none_model()
    print("step5_detection_model_wedge=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
