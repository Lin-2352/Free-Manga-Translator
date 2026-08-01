from __future__ import annotations

import contextlib
import io
import json
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


def _build_sample(tmp_path: Path, sample_name: str) -> None:
    sample_dir = tmp_path / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(sample_dir / "input.jpg")

    detect_dir = sample_dir / "step_1_detect"
    detect_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(detect_dir / "seg_mask.png")

    layout_dir = sample_dir / "step_6_layout"
    layout_dir.mkdir(parents=True, exist_ok=True)
    (layout_dir / "layout_constraints.json").write_text("[]", encoding="utf-8")
    # Deliberately corrupt: not valid JSON.
    (layout_dir / "sfx_artwork_regions.json").write_text("{not valid json", encoding="utf-8")


def test_sfx_registry_load_failure_is_logged_loudly() -> None:
    """Regression test for Tier A #3: a failed load of sfx_artwork_regions.json
    used to be silently swallowed (bare `except: pass`), which drops
    onomatopoeia protection -- the inpainter can then erase the artist's own
    SFX lettering thinking it's translatable text. The fix is log-only (no
    behavior change, to avoid pixel risk): the failure must now be printed
    loudly with the actual exception, not swallowed silently."""
    original_get_lama_session = run_step4_inpaint._get_lama_session
    original_get_anime_lama = run_step4_inpaint._get_anime_lama_model
    try:
        run_step4_inpaint._get_lama_session = lambda *_args, **_kwargs: None
        run_step4_inpaint._get_anime_lama_model = lambda: (None, None)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sample_name = "sfx_probe_sample"
            _build_sample(tmp_path, sample_name)

            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                try:
                    run_step4_inpaint.run_step4_inpaint(
                        sample_map={sample_name: "input.jpg"}, samples_dir=tmp_path
                    )
                except Exception:
                    # Whether or not the rest of the (empty-layout) page finishes
                    # cleanly is not what this test is checking -- only that the
                    # sfx-registry load failure itself was surfaced before any
                    # later unrelated failure.
                    pass

            output = captured.getvalue()
            assert "sfx_artwork_regions.json" in output and (
                "step4-warn" in output or "SFX" in output.upper()
            ), (
                "expected a loud warning naming sfx_artwork_regions.json's load failure; got no such "
                f"message in captured output:\n{output}"
            )
            assert "Expecting" in output or "JSONDecodeError" in output or "delimiter" in output, (
                f"expected the actual exception detail to appear in the warning, got:\n{output}"
            )
    finally:
        run_step4_inpaint._get_lama_session = original_get_lama_session
        run_step4_inpaint._get_anime_lama_model = original_get_anime_lama


def main() -> int:
    test_sfx_registry_load_failure_is_logged_loudly()
    print("step4_sfx_registry_load_failure_logged=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
