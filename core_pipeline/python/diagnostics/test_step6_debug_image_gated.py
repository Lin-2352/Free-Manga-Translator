"""Regression test: run_step6_layout's debug_layout_boxes.jpg write (diagnostic only --
real outputs layout_constraints.json/rejected_layout_items.json/sfx_artwork_regions.json
are written earlier and are unaffected) must be gated behind MANGA_PIPELINE_DEBUG, off by
default (matching prior behavior minus the debug write only when explicitly requested).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
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

import cv2
import numpy as np

import run_step6_layout as step6  # noqa: E402


def _build_minimal_sample(samples_dir: Path, sample_name: str) -> None:
    sample_path = samples_dir / sample_name
    (sample_path / "step_5_ocr").mkdir(parents=True, exist_ok=True)
    with open(sample_path / "step_5_ocr" / "ocr_results.json", "w", encoding="utf-8") as f:
        json.dump([], f)
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(sample_path / "input.jpg"), image)


def test_step6_debug_image_gated() -> None:
    assert step6._debug_images_enabled() is False, "must default to off with no env var set"
    os.environ["MANGA_PIPELINE_DEBUG"] = "1"
    try:
        assert step6._debug_images_enabled() is True
    finally:
        os.environ.pop("MANGA_PIPELINE_DEBUG", None)
    assert step6._debug_images_enabled() is False, "must return to off once env var is cleared"

    tmp_dir = Path(tempfile.mkdtemp(prefix="fmt_step6_debug_gate_"))
    old_env = os.environ.get("MANGA_PIPELINE_DEBUG")
    try:
        sample_name = "debug_gate_sample"
        _build_minimal_sample(tmp_dir, sample_name)

        # Default (off): no debug_layout_boxes.jpg should be written.
        os.environ.pop("MANGA_PIPELINE_DEBUG", None)
        step6.run_step6_layout(sample_map={sample_name: "input.jpg"}, samples_dir=tmp_dir)
        debug_path = tmp_dir / sample_name / "step_6_layout" / "debug_layout_boxes.jpg"
        assert not debug_path.exists(), "debug image must NOT be written when MANGA_PIPELINE_DEBUG is unset"
        real_output = tmp_dir / sample_name / "step_6_layout" / "layout_constraints.json"
        assert real_output.exists(), "real layout_constraints.json output must still be written"

        # Enabled: the debug image must now be written.
        os.environ["MANGA_PIPELINE_DEBUG"] = "1"
        step6.run_step6_layout(sample_map={sample_name: "input.jpg"}, samples_dir=tmp_dir)
        assert debug_path.exists(), "debug image must be written when MANGA_PIPELINE_DEBUG=1"

        print("step6_debug_image_gated=pass")
    finally:
        if old_env is None:
            os.environ.pop("MANGA_PIPELINE_DEBUG", None)
        else:
            os.environ["MANGA_PIPELINE_DEBUG"] = old_env
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_step6_debug_image_gated()
