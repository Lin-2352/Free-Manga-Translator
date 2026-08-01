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
del _BOOTSTRAP_FILE, _candidate

import cv2
import numpy as np
import run_step8_typeset as step8


def _original_structural_art_mask(gray: np.ndarray) -> np.ndarray:
    """A frozen copy of the pre-vectorization implementation, kept ONLY as ground truth for
    this test -- profiling found this loop's per-component `labels == component_idx` full-page
    scan was 96% of the function's time on real pages (thousands of components), and it is
    called 1-2x per floating text region. The real function was rewritten to a vectorized
    label-lookup; this copy exists so a future edit to either side can be caught by comparison
    instead of trusting the rewrite was equivalent."""
    edges = cv2.Canny(gray, 45, 135) > 0
    dark_ink = gray < 124
    seed = (edges | dark_ink).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    structural = np.zeros(seed.shape, dtype=bool)
    for component_idx in range(1, component_count):
        x, y, width, height, area = stats[component_idx]
        if (
            area >= 24
            or max(width, height) >= 18
            or (area >= 10 and min(width, height) <= 3 and max(width, height) >= 12)
        ):
            structural[labels == component_idx] = True
    return structural


def test_vectorized_matches_original_on_real_samples() -> None:
    """The vectorized _structural_art_mask() must produce bit-identical output to the original
    per-component loop on real page images, not just a synthetic case."""
    samples_dir = _PROJECT_ROOT_FOR_IMPORTS / "samples"
    checked = 0
    for sample_dir in sorted(samples_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        image_paths = list(sample_dir.glob("sample*.jpg")) + list(sample_dir.glob("sample *.jpg"))
        if not image_paths:
            continue
        img = cv2.imread(str(image_paths[0]))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        expected = _original_structural_art_mask(gray)
        actual = step8._structural_art_mask(gray)
        assert actual.dtype == np.dtype(bool), f"{sample_dir.name}: expected bool dtype, got {actual.dtype}"
        assert np.array_equal(expected, actual), (
            f"{sample_dir.name}: vectorized _structural_art_mask output diverged from the "
            f"original per-component loop (mismatched pixels: {int(np.count_nonzero(expected != actual))})"
        )
        checked += 1
    assert checked >= 3, f"expected to check at least 3 real sample images, only found {checked}"


def test_background_label_zero_is_excluded() -> None:
    """Regression guard for the specific bug this vectorization could silently introduce: cv2's
    connectedComponentsWithStats always assigns label 0 to the background. The original loop
    started at range(1, component_count), NEVER evaluating the background component. The
    vectorized version computes the same predicate across the WHOLE stats array (including index
    0) and must explicitly force keep[0] = False, or a background component -- which trivially
    has a huge area and max dimension, so it always satisfies the predicate -- would flip the
    entire mask to True. This constructs an image where the background is the overwhelming
    majority of pixels, so a keep[0]=True bug would be immediately and unmistakably visible."""
    # A large blank (uniform) image: after Canny + dark-ink thresholding, the seed is essentially
    # empty, so there is exactly one component (label 0, the background) covering the whole image.
    gray = np.full((200, 200), 200, dtype=np.uint8)
    result = step8._structural_art_mask(gray)
    assert result.dtype == np.dtype(bool)
    assert not result.any(), (
        "a uniform blank image (background-only, no real components) must produce an all-False "
        "structural mask -- any True pixel here means the background label (0) leaked into the "
        "result, which would make _apply_floating_art_avoidance treat the entire page as "
        "protected structural art"
    )


def main() -> int:
    test_vectorized_matches_original_on_real_samples()
    test_background_label_zero_is_excluded()
    print("step8_structural_art_mask_vectorized=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
