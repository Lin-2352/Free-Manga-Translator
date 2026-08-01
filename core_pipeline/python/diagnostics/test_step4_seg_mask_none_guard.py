from __future__ import annotations

import contextlib
import io
import sys
import tempfile
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

import run_step4_inpaint


def _build_sample_with_corrupt_seg_mask(tmp_path: Path, sample_name: str) -> None:
    sample_dir = tmp_path / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(sample_dir / "input.jpg")

    detect_dir = sample_dir / "step_1_detect"
    detect_dir.mkdir(parents=True, exist_ok=True)
    # seg_mask.png exists (Path.exists() is True) but is truncated/undecodable.
    (detect_dir / "seg_mask.png").write_bytes(b"not a real png file")

    layout_dir = sample_dir / "step_6_layout"
    layout_dir.mkdir(parents=True, exist_ok=True)
    (layout_dir / "layout_constraints.json").write_text("[]", encoding="utf-8")


def test_corrupt_seg_mask_is_skipped_gracefully_not_crashed() -> None:
    """Regression test for Tier B #5: Path.exists() doesn't guarantee a file is
    decodable. A truncated/corrupt seg_mask.png used to make _imread_grayscale_2d
    return None, and a later `seg_mask[y1:y2, x1:x2]` subscript would then raise
    an uncaught 'NoneType' object is not subscriptable -- this function has
    exactly one try/except in ~2300 lines and its caller has none, so this
    crashed the entire page. The fix must detect the None right after load and
    skip the sample gracefully (mirroring the existing "file missing" SKIP),
    not let it reach the later subscript."""
    original_get_lama_session = run_step4_inpaint._get_lama_session
    original_get_anime_lama = run_step4_inpaint._get_anime_lama_model
    try:
        run_step4_inpaint._get_lama_session = lambda *_args, **_kwargs: None
        run_step4_inpaint._get_anime_lama_model = lambda: (None, None)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sample_name = "corrupt_seg_mask_sample"
            _build_sample_with_corrupt_seg_mask(tmp_path, sample_name)

            captured = io.StringIO()
            raised = None
            with contextlib.redirect_stdout(captured):
                try:
                    run_step4_inpaint.run_step4_inpaint(
                        sample_map={sample_name: "input.jpg"}, samples_dir=tmp_path
                    )
                except Exception as error:
                    raised = error

            assert raised is None, (
                f"a corrupt seg_mask.png must not crash the pipeline with an uncaught exception, got: {raised!r}"
            )
            output = captured.getvalue()
            assert "SKIP" in output and sample_name in output, (
                f"expected a graceful SKIP message for the sample with the corrupt seg_mask, got:\n{output}"
            )
    finally:
        run_step4_inpaint._get_lama_session = original_get_lama_session
        run_step4_inpaint._get_anime_lama_model = original_get_anime_lama


def main() -> int:
    test_corrupt_seg_mask_is_skipped_gracefully_not_crashed()
    print("step4_seg_mask_none_guard=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
