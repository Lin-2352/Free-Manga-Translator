"""External inpainting command adapter for Step 4.

Usage through Step 4:
    set MANGA_INPAINT_COMMAND=python python/runtime/iopaint_inpaint_backend.py --input "{input}" --mask "{mask}" --output "{output}" --model migan

Supported models depend on the local IOPaint installation: migan, mat, fcf,
zits. This adapter is intentionally opt-in because the focused validation
showed that stronger generic models can improve flat paper cleanup but can
damage dense manga art when used blindly.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import cv2
import numpy as np


def _install_py313_imghdr_stub() -> None:
    if "imghdr" not in sys.modules:
        module = types.ModuleType("imghdr")
        module.what = lambda file, h=None: None
        sys.modules["imghdr"] = module


def _load_model_class(name: str):
    _install_py313_imghdr_stub()
    normalized = name.strip().lower()
    if normalized == "migan":
        from iopaint.model.mi_gan import MIGAN

        return MIGAN
    if normalized == "mat":
        from iopaint.model.mat import MAT

        return MAT
    if normalized == "fcf":
        from iopaint.model.fcf import FcF

        return FcF
    if normalized == "zits":
        from iopaint.model.zits import ZITS

        return ZITS
    raise ValueError(f"Unsupported IOPaint model: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="migan", choices=["migan", "mat", "fcf", "zits"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crop-trigger-size", type=int, default=512)
    parser.add_argument("--crop-margin", type=int, default=64)
    args = parser.parse_args()

    image_path = Path(args.input)
    mask_path = Path(args.mask)
    output_path = Path(args.output)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read input image: {image_path}")
    if mask is None:
        raise FileNotFoundError(f"Could not read mask image: {mask_path}")

    _install_py313_imghdr_stub()
    from iopaint.schema import HDStrategy, InpaintRequest

    model_class = _load_model_class(args.model)
    if not model_class.is_downloaded():
        model_class.download()
    model = model_class(args.device)
    config = InpaintRequest(
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_trigger_size=args.crop_trigger_size,
        hd_strategy_crop_margin=args.crop_margin,
    )
    result_bgr = model(image_bgr[:, :, ::-1], (mask > 0).astype(np.uint8) * 255, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result_bgr):
        raise RuntimeError(f"Could not write output image: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
