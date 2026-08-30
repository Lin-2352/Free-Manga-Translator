"""Route the pipeline's GPU model calls to a remote bridge instead of the local GPU.

The point of this module is a laptop that orchestrates the whole pipeline locally --
steps 5 -> 6 -> 7 -> 4 -> 8 in local processes, reading and writing local artifacts --
while every heavyweight model runs on a remote GPU. That is the inverse of the Kaggle
backend deployment, where the laptop posts an image and all eight stages run remotely.

DESIGN: the switch is the HANDLE TYPE, not a check inside each detect_* call.

Each load_*() in ml_region_lib returns a RemoteHandle instead of a real model when the
bridge is enabled, and the matching detect_*() dispatches on that. Deciding once, at load
time, buys two things a per-call env lookup would not:

  * The local model is never loaded at all. This is the whole claim -- loading a torch or
    ONNX-CUDA model locally creates a CUDA context on the laptop even if inference then
    happens elsewhere, so a design that loads locally and calls remotely would quietly
    fail the "local GPU idle" test while looking correct.
  * Local and remote cannot be mixed by accident inside one call: a RemoteHandle simply
    has no session to run.

Default is OFF. With FMT_GPU_BRIDGE unset, ml_region_lib behaves exactly as before --
this module is not even imported on the hot path.
"""

from __future__ import annotations

import base64
import os
from dataclasses import asdict, is_dataclass
from typing import Any

import cv2
import numpy as np

BRIDGE_ENV = "FMT_GPU_BRIDGE"
BRIDGE_CLIENT_PATH_ENV = "FMT_GPU_BRIDGE_CLIENT"

# Task names registered by the bridge server. Kept here so client and server share one
# spelling; a typo becomes an immediate "unknown task" instead of a silent local fallback.
TASK_DETECT_TEXT = "fmt_detect_text"
TASK_DETECT_BUBBLES = "fmt_detect_bubbles"
TASK_DETECT_SEMANTIC = "fmt_detect_semantic"
TASK_OCR_BATCH = "fmt_ocr_batch"
TASK_INPAINT = "fmt_inpaint"

_BRIDGE = None


class RemoteHandle:
    """Stands in for a locally-loaded model when inference is offloaded.

    Deliberately inert: it holds no session, so any code path that forgets to check for
    it fails loudly at the point of misuse instead of silently loading a local model.
    """

    __slots__ = ("kind",)

    def __init__(self, kind: str):
        self.kind = kind

    def __repr__(self) -> str:
        return f"<RemoteHandle {self.kind}>"


def bridge_enabled() -> bool:
    return os.environ.get(BRIDGE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def get_bridge():
    """Lazily build the GPUBridge client. Import is deferred so a normal local run never
    needs `requests` or the gpu-bridge checkout on sys.path."""
    global _BRIDGE
    if _BRIDGE is None:
        import sys
        client_dir = os.environ.get(BRIDGE_CLIENT_PATH_ENV) or str(
            _default_client_dir())
        if client_dir not in sys.path:
            sys.path.insert(0, client_dir)
        from gpu_client import GPUBridge  # noqa: PLC0415
        _BRIDGE = GPUBridge()
    return _BRIDGE


def _default_client_dir():
    from pathlib import Path
    # common/ -> python/ -> core_pipeline/ -> <repo> -> <container>/gpu-bridge
    return Path(__file__).resolve().parents[3].parent / "gpu-bridge"


# -- wire format -----------------------------------------------------------------------
# Images cross as PNG bytes, not raw arrays: PNG is lossless (a JPEG round trip would
# perturb detection inputs) and an order of magnitude smaller than a raw uint8 dump, which
# matters when every call is a round trip over a tunnel.

def encode_image(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("cv2.imencode failed while packing an image for the bridge")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_gray(b64: str) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    out = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if out is None:
        raise RuntimeError("cv2.imdecode failed while unpacking a mask from the bridge")
    return out


def encode_cfg(cfg: Any) -> dict:
    """MLConfig is a flat dataclass of scalars, so asdict round-trips as JSON."""
    return asdict(cfg) if is_dataclass(cfg) else dict(cfg)


# -- remote calls ----------------------------------------------------------------------
# One round trip per detection call, not per tensor. These mirror the three coarse
# functions in ml_region_lib that already take (handle, image, cfg) and return plain data,
# which is what makes offloading tractable here at all.
#
# Tiling stays SERVER-side on purpose. detect_text and detect_bubbles split extreme-aspect
# pages into overlapping tiles above _BUBBLE_TILE_ASPECT_THRESHOLD; doing that split on the
# laptop would turn one webtoon strip into a dozen round trips and would also fork the
# tiling logic into two places. The server calls the real detect_* and returns the stitched
# result, so the tiling rules live in exactly one file, as they do today.

def remote_detect_text(image: np.ndarray, cfg: Any):
    """Returns (boxes_as_xyxy_tuples, seg_mask). Caller rebuilds Box/TextDetectionResult
    so this module never has to import ml_region_lib (which would be circular)."""
    out = get_bridge().run(
        TASK_DETECT_TEXT,
        {"image": encode_image(image), "cfg": encode_cfg(cfg)},
        max_wait=900.0,
    )
    boxes = [tuple(int(v) for v in b) for b in out["boxes"]]
    return boxes, decode_gray(out["seg_mask"])


def remote_detect_bubbles(image: np.ndarray, cfg: Any) -> list:
    """Returns a LIST of per-instance masks. Kept as a list rather than flattened into one
    canvas: detect_bubbles' contract is one mask per bubble, and merging them would lose
    the per-bubble identity every downstream consolidation step depends on."""
    out = get_bridge().run(
        TASK_DETECT_BUBBLES,
        {"image": encode_image(image), "cfg": encode_cfg(cfg)},
        max_wait=900.0,
    )
    return [decode_gray(m) for m in out["masks"]]


def remote_detect_semantic(image: np.ndarray, cfg: Any) -> list:
    """Returns a list of plain dicts, one per SemanticTextRegion; the caller rebuilds the
    dataclasses."""
    out = get_bridge().run(
        TASK_DETECT_SEMANTIC,
        {"image": encode_image(image), "cfg": encode_cfg(cfg)},
        max_wait=900.0,
    )
    return out["regions"]


def remote_ocr_batch(crops: list, language: str, engine: str = "manga_ocr") -> list:
    """OCR N crops in ONE round trip; returns N result dicts.

    Batched deliberately. A page can carry dozens of text regions, and one request per
    crop through a tunnel serialised at max_concurrent=1 would make the round trips, not
    the GPU, the whole cost of step 5. Sending the crops together keeps it to one.
    """
    out = get_bridge().run(
        TASK_OCR_BATCH,
        {"crops": [encode_image(c) for c in crops],
         "language": language, "engine": engine},
        max_wait=1800.0,
    )
    return out["results"]


def remote_inpaint(image: np.ndarray, mask: np.ndarray, variant: str = "lama_onnx"):
    """Inpaint one page remotely. `variant` selects which inpainter the server uses
    (lama_onnx / anime_lama / manga_cleaner), so the local variant-selection logic stays
    authoritative and the server stays a dumb executor."""
    out = get_bridge().run(
        TASK_INPAINT,
        {"image": encode_image(image), "mask": encode_image(mask), "variant": variant},
        max_wait=1800.0,
    )
    raw = np.frombuffer(base64.b64decode(out["image"]), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)
