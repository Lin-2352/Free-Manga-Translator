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
for _rel in ("python/common", "python/steps"):
    _path = str(_PROJECT_ROOT_FOR_IMPORTS / _rel)
    if _path not in sys.path:
        sys.path.insert(0, _path)
del _BOOTSTRAP_FILE, _candidate, _PROJECT_ROOT_FOR_IMPORTS, _rel, _path

import numpy as np
import run_step8_typeset as step8


def test_measure_then_render_matches_single_call_render() -> None:
    """Step 8's hot search loop tries ~800 (font size x width ratio) candidates per text
    region and used to fully rasterize (FreeType render + stroke) every single one before
    checking if it even fits the bubble -- profiling showed that rasterization is ~70% of
    step 8's wall time, and measure-before-rasterize (skip the render for candidates that fail
    the cheap bounding-box check) cuts it by ~3x on real samples with zero pixel change.

    This is the fast, standing regression guard for that split: _measure_text_block() +
    _render_text_block_from_measurement() must always produce IDENTICAL output to the original
    single-call _render_text_block() for the same inputs. If someone edits one without keeping
    the other in sync, this catches it in under a second instead of requiring a full pipeline
    run + pixel-diff."""
    font = step8._load_font(28, step8.FONT_PATH)
    cases = [
        (["Hello there"], 28, 2),
        (["A longer line of dialogue", "that wraps to two lines"], 24, 3),
        ([""], 16, 0),
        (["X"], 96, 4),
    ]
    for lines, size, outline_width in cases:
        expected_block, expected_alpha, expected_metrics = step8._render_text_block(
            lines, font, size, outline_width,
            fill_color=(0, 0, 0, 255), stroke_color=(255, 255, 255, 255),
        )

        measurement = step8._measure_text_block(lines, font, size, outline_width)
        actual_block, actual_alpha, actual_metrics = step8._render_text_block_from_measurement(
            lines, font, measurement, outline_width,
            fill_color=(0, 0, 0, 255), stroke_color=(255, 255, 255, 255),
        )

        assert actual_block.size == expected_block.size, (
            f"block size differs for {lines!r}@{size}: {actual_block.size} vs {expected_block.size}"
        )
        assert np.array_equal(np.array(actual_block), np.array(expected_block)), (
            f"rendered pixels differ for {lines!r}@{size} -- the measure/render split is not "
            "output-identical to the original single-call path"
        )
        assert np.array_equal(actual_alpha, expected_alpha), f"alpha mask differs for {lines!r}@{size}"
        assert actual_metrics == expected_metrics, f"metrics differ for {lines!r}@{size}"

        # The hot-loop's bounds pre-check must agree exactly with what the post-render check
        # would have decided -- this is the actual optimization's correctness invariant.
        assert (
            measurement["block_width"] + 2 == expected_block.width
            and measurement["block_height"] + 2 == expected_block.height
        ), f"measured dims don't match rendered block dims for {lines!r}@{size}"


def main() -> int:
    test_measure_then_render_matches_single_call_render()
    print("step8_measure_render_split=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
