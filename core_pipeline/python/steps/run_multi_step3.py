"""
Step 3 Multi-Sample Test Runner (v3)
=====================================
Processes all 6 samples through Step 3 dynamic mask builder.
Outputs strictly 1-channel binary masks (0=keep, 255=erase).

ZERO debug text. ZERO bounding boxes. ZERO RGB. ZERO cyan.

Output structure (per workspace rules):
  samples/sample1/step3/mask.png
  samples/sample2/step3/mask.png
  ...
  samples/sample6/step3/mask.png

Previous contents of step3/ are deleted before each run.
"""

# --- Clean-copy path bootstrap ---
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_BOOTSTRAP_FILE = _BootstrapPath(__file__).resolve()
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
    if _path not in _bootstrap_sys.path:
        _bootstrap_sys.path.insert(0, _path)
del _BootstrapPath, _bootstrap_sys, _BOOTSTRAP_FILE, _candidate, _PROJECT_ROOT_FOR_IMPORTS, _rel, _path
# --- End clean-copy path bootstrap ---
import cv2
import shutil
import numpy as np
from pathlib import Path
from pipeline_paths import DEFAULT_SAMPLES_ROOT, sample_root_from_env
from ml_region_lib import (
    MLConfig, load_text_model, load_bubble_model, load_semantic_model, detect_text,
    detect_bubbles, detect_semantic_text_regions, build_step2_routing_state, build_step3_dynamic_mask,
    SAMPLE_MAP,
)


def run_step3_test():
    cfg = MLConfig(
        text_model_path="models/comictextdetector.pt.onnx",
        bubble_model_path="models/manga109_bubble/best.pt",
    )

    print("Loading models...")
    text_handle = load_text_model(cfg.text_model_path, allow_cpu=False)
    bubble_model, bubble_device = load_bubble_model(cfg.bubble_model_path, allow_cpu=False)
    semantic_handle = load_semantic_model("magi", allow_cpu=False)

    samples_dir = sample_root_from_env(DEFAULT_SAMPLES_ROOT)

    for sample_name, img_file in SAMPLE_MAP.items():
        img_path = samples_dir / sample_name / img_file
        if not img_path.exists():
            print(f"  SKIP {sample_name}: {img_file} not found")
            continue

        print(f"Processing {sample_name}...")
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]

        # Step 1: Detection
        text_result = detect_text(text_handle, image, cfg)

        # Step 2: Routing
        bubble_masks = detect_bubbles(bubble_model, bubble_device, image, cfg)
        semantic_result = detect_semantic_text_regions(semantic_handle, image, cfg)
        routed = build_step2_routing_state(text_result, semantic_result, bubble_masks, cfg, w, h)

        # Step 3: Build strictly binary 1-channel mask (clipped to bubble contours)
        step3_mask = build_step3_dynamic_mask(image, routed, text_result.seg_mask, cfg, bubble_masks=bubble_masks)

        # Output to samples/sampleN/step3/mask.png
        out_dir = samples_dir / sample_name / "step3"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        mask_path = out_dir / "mask.png"
        cv2.imwrite(str(mask_path), step3_mask)

        # Strict binary assertions
        assert step3_mask.ndim == 2, f"FAIL {sample_name}: {step3_mask.ndim} channels"
        unique_vals = np.unique(step3_mask)
        assert all(v in (0, 255) for v in unique_vals), f"FAIL {sample_name}: non-binary {unique_vals}"

        # Stats
        bubble_count = sum(1 for r in routed if r.route_state == "bubble_dialogue")
        floating_count = sum(1 for r in routed if r.route_state == "floating_dialogue")
        sfx_count = sum(1 for r in routed if r.route_state == "onomatopoeia")
        mask_ratio = np.count_nonzero(step3_mask) / max(1, step3_mask.size) * 100
        print(f"  {bubble_count} bubble, {floating_count} floating, {sfx_count} SFX | coverage: {mask_ratio:.2f}%")
        print(f"  -> {mask_path}")

    print("\nDone. All masks: 1-channel, binary, no debug artifacts.")


if __name__ == "__main__":
    run_step3_test()
