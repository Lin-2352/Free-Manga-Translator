"""
Step 4 — Layout-Driven Inpainting
=================================

Bubble text uses ONNX LaMa on local crops clipped to eroded bubble masks.

Floating text uses the tight Step 6 text mask first. Flat/paper backgrounds are
filled directly, detailed art uses small-radius OpenCV stroke repair, and the
Anime/Manga LaMa checkpoint is reserved for masks that are safe to synthesize.
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
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import threading
from typing import Sequence

# torch must be imported before cv2 in this process -- see run_step5_ocr.py
# for the full explanation (verified import-order segfault reproduction).
import torch  # noqa: F401  (import-order guard, see comment above)
import cv2
import numpy as np

from pipeline_paths import DEFAULT_SAMPLES_ROOT, sample_root_from_env

try:
    import torch
except ImportError:  # pragma: no cover - fallback for environments without torch
    torch = None

from ml_region_lib import (
    MLConfig,
    SAMPLE_MAP,
    _extract_text_strokes,
    lama_inpaint,
    load_lama_model,
)


ANIME_LAMA_PATH = Path("models/lama/anime-manga-big-lama.pt")
MANGA_CLEANER_MODEL_DIR = Path("models/manga_cleaner/ComfyUI/models/lama")
EXTERNAL_INPAINT_CWD = Path(__file__).resolve().parents[2]
_LAMA_SESSION = None
_ANIME_LAMA_MODEL = None
_ANIME_LAMA_DEVICE = None
_ANIME_LAMA_LOAD_ATTEMPTED = False
_MANGA_CLEANER_MODELS = None
_MANGA_CLEANER_LOAD_ATTEMPTED = False
# Guards the check-then-act singleton loads below. The GPU scheduler (backend_api/app/
# gpu_scheduler.py) allows 2-4 concurrent pipeline runs, so two cold-start requests can otherwise
# race into loading the same model twice (wasted VRAM/time, last-write-wins on the global).
_MODEL_LOAD_LOCK = threading.Lock()


def _is_renderable_translation(text: str) -> bool:
    if not text or text.startswith("[TL:"):
        return False
    return any(char.isalnum() for char in text)


def _load_renderable_translation_ids(sample_path: Path) -> set[int] | None:
    trans_path = sample_path / "step_7_translate" / "translation_results.json"
    if not trans_path.exists():
        return None

    try:
        trans_data = json.loads(trans_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    return {
        int(item["id"])
        for item in trans_data
        if "id" in item and _is_renderable_translation(item.get("en_text", ""))
    }


def _load_translation_text_map(sample_path: Path) -> dict[int, str]:
    trans_path = sample_path / "step_7_translate" / "translation_results.json"
    if not trans_path.exists():
        return {}
    try:
        trans_data = json.loads(trans_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        int(item["id"]): str(item.get("en_text", "") or "")
        for item in trans_data
        if "id" in item
    }


def _dominant_stroke_hue(
    image: np.ndarray,
    seg_mask: np.ndarray,
    coords: tuple[int, int, int, int],
    kernel_3: np.ndarray,
) -> tuple[int, int] | None:
    """Dominant OpenCV hue of the saturated glyph strokes in `coords`, with
    the number of contributing pixels. None when the strokes are mostly
    unsaturated (plain black/grey lettering)."""
    x1, y1, x2, y2 = coords
    img_h, img_w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    strokes = _floating_erase_mask_for_box(image, seg_mask, (x1, y1, x2, y2), kernel_3)
    on = strokes > 0
    if int(np.count_nonzero(on)) < 40:
        return None
    hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    saturated = on & (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 40)
    count = int(np.count_nonzero(saturated))
    if count < 40:
        return None
    hist = np.bincount(hsv[:, :, 0][saturated].astype(np.int32), minlength=180)
    return int(np.argmax(hist)), count


def _page_dialogue_lettering_hue(
    image: np.ndarray,
    seg_mask: np.ndarray,
    layout_data: list[dict],
    translation_text_map: dict[int, str],
    kernel_3: np.ndarray,
) -> int | None:
    """The page's dominant colored dialogue-lettering hue (e.g. the green all
    narration on a page is set in), or None when dialogue is plain black.
    Sampled only from constraints that are clearly text (bubbles, or floating
    regions whose translation is a multi-word sentence)."""
    hist = np.zeros(180, dtype=np.int64)
    total = 0
    for constraint in layout_data:
        cid = int(constraint.get("id", -1))
        en_text = str(translation_text_map.get(cid, "") or "")
        is_bubble = constraint.get("bubble_idx", -1) != -1
        if not is_bubble and len(en_text.split()) < 2:
            continue
        red_box = constraint.get("red_box")
        if not red_box:
            continue
        sample = _dominant_stroke_hue(
            image, seg_mask, tuple(int(v) for v in red_box), kernel_3
        )
        if sample is None:
            continue
        hue, count = sample
        hist[hue] += count
        total += count
    if total < 300:
        return None
    return int(np.argmax(hist))


def _floating_sfx_signature(
    image: np.ndarray,
    seg_mask: np.ndarray,
    coords: tuple[int, int, int, int],
    en_text: str,
    kernel_3: np.ndarray,
    dialogue_hue: int | None = None,
) -> bool:
    """Stylized onomatopoeia lettering drawn as part of the artwork. SFX must
    be preserved untouched (no erase, no typeset) -- they belong to the artist.

    Three signals must agree, so plain dialogue is never misclassified:
    1. the translation is a single short transliteration-like token
       ("PIRAA", "THUMP"), not a sentence;
    2. the source glyph strokes are stylized SFX lettering: saturated colored
       strokes and/or hollow outline with bright cores, unlike solid dark
       dialogue glyphs;
    3. the lettering is large relative to the page (SFX display lettering,
       not caption-sized text)."""
    token = re.sub(r"[^A-Za-z]+", " ", str(en_text or "")).strip()
    if not token or " " in token or len(token) > 8:
        return False

    x1, y1, x2, y2 = coords
    img_h, img_w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return False

    strokes = _floating_erase_mask_for_box(image, seg_mask, (x1, y1, x2, y2), kernel_3)
    stroke_count = int(np.count_nonzero(strokes > 0))
    if stroke_count < 60:
        return False

    # Large display lettering: the stroke bbox must be tall relative to the page.
    ys, xs = np.where(strokes > 0)
    stroke_height = int(ys.max()) - int(ys.min()) + 1
    if stroke_height < max(30, int(img_h * 0.025)):
        return False

    roi = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    on_stroke = strokes > 0
    saturated = on_stroke & (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 40)
    saturated_stroke = float(np.mean(saturated[on_stroke]))
    bright_core = float(np.mean(gray[on_stroke] >= 205))
    # Stylized SFX lettering is colored (or hollow with a colored rim).
    if saturated_stroke < 0.20 and not (bright_core >= 0.28 and saturated_stroke >= 0.10):
        return False
    # Lettering in exactly the page's dialogue color is text, not art.
    if dialogue_hue is not None and int(np.count_nonzero(saturated)) >= 40:
        hist = np.bincount(hsv[:, :, 0][saturated].astype(np.int32), minlength=180)
        hue = int(np.argmax(hist))
        hue_dist = min(abs(hue - dialogue_hue), 180 - abs(hue - dialogue_hue))
        if hue_dist <= 8:
            return False
    # Brush-art anatomy: art SFX have fat brush strokes and/or hollow bright
    # cores deep inside the stroke body. Solid caption-weight lettering (even
    # colored onomatopoeia like a teal "thump") is translatable text -- the
    # pro reference translates it in place over a real reconstruction.
    dist = cv2.distanceTransform(on_stroke.astype(np.uint8), cv2.DIST_L2, 3)
    deep_thresh = max(2.5, float(np.percentile(dist[on_stroke], 70)))
    deep = dist >= deep_thresh
    if int(np.count_nonzero(deep)) < 30:
        return False
    bright_deep = float(np.mean(gray[deep] >= 205))
    if bright_deep < 0.38 and deep_thresh < 12.0:
        return False
    return True


def _bridge_enabled() -> bool:
    """True when GPU work is offloaded to the remote bridge."""
    try:
        from gpu_bridge_backend import bridge_enabled
        return bridge_enabled()
    except Exception:
        return False


def _load_anime_lama_model(model_path: Path = ANIME_LAMA_PATH):
    if _bridge_enabled():
        # (None, None) is this function's existing "not available, use ONNX LaMa" signal,
        # and ONNX LaMa is bridged -- so floating text still gets inpainted, remotely.
        print("  [AnimeLaMa] bridge enabled; deferring to remote LaMa")
        return None, None
    if torch is None:
        print("  [AnimeLaMa] torch unavailable; floating text will use ONNX LaMa fallback")
        return None, None
    if not model_path.exists():
        print(f"  [AnimeLaMa] missing {model_path}; floating text will use ONNX LaMa fallback")
        return None, None
    if not torch.cuda.is_available():
        print("  [AnimeLaMa] CUDA unavailable; floating text will use ONNX LaMa fallback")
        return None, None

    device = torch.device("cuda")
    try:
        model = torch.jit.load(str(model_path), map_location="cpu").to(device).eval()
    except Exception as error:
        # Mirrors the sibling manga-cleaner loader's guard below. A corrupt/bad
        # checkpoint must fail once and be remembered (see _ANIME_LAMA_LOAD_ATTEMPTED
        # in _get_anime_lama_model, set BEFORE this call is attempted), not retried
        # on every single request forever.
        print(f"  [AnimeLaMa] load failed; floating text will use ONNX LaMa fallback. {str(error)[:160]}")
        return None, None
    print(f"  [AnimeLaMa] floating-text inpainter: CUDA LOCKED ({model_path})")
    return model, device


def _get_lama_session(model_path: str):
    global _LAMA_SESSION
    if _LAMA_SESSION is None:
        with _MODEL_LOAD_LOCK:
            if _LAMA_SESSION is None:
                _LAMA_SESSION = load_lama_model(model_path)
    return _LAMA_SESSION


def _get_anime_lama_model():
    global _ANIME_LAMA_MODEL, _ANIME_LAMA_DEVICE, _ANIME_LAMA_LOAD_ATTEMPTED
    if not _ANIME_LAMA_LOAD_ATTEMPTED:
        with _MODEL_LOAD_LOCK:
            if not _ANIME_LAMA_LOAD_ATTEMPTED:
                # Set the attempted-flag BEFORE attempting the load (not after
                # success) so that even an exception escaping _load_anime_lama_model
                # (belt-and-suspenders on top of its own try/except) is remembered
                # as "already tried" instead of retrying the load on every request.
                _ANIME_LAMA_LOAD_ATTEMPTED = True
                _ANIME_LAMA_MODEL, _ANIME_LAMA_DEVICE = _load_anime_lama_model()
    return _ANIME_LAMA_MODEL, _ANIME_LAMA_DEVICE


def _manga_cleaner_enabled() -> bool:
    return os.getenv("MANGA_CLEANER_BACKEND", "off").strip().lower() not in {"0", "false", "no", "off"}


def _manga_cleaner_paths() -> tuple[Path, Path]:
    model_dir = Path(os.getenv("MANGA_CLEANER_MODEL_DIR", str(MANGA_CLEANER_MODEL_DIR)))
    if not model_dir.is_absolute():
        model_dir = EXTERNAL_INPAINT_CWD / model_dir
    return model_dir / "manga_inpaintor.jit", model_dir / "erika.jit"


def _load_manga_cleaner_models():
    global _MANGA_CLEANER_MODELS, _MANGA_CLEANER_LOAD_ATTEMPTED
    if _MANGA_CLEANER_LOAD_ATTEMPTED:
        return _MANGA_CLEANER_MODELS
    with _MODEL_LOAD_LOCK:
        if _MANGA_CLEANER_LOAD_ATTEMPTED:
            return _MANGA_CLEANER_MODELS
        _MANGA_CLEANER_LOAD_ATTEMPTED = True

        if torch is None or not _manga_cleaner_enabled():
            return None
        inpaintor_path, line_path = _manga_cleaner_paths()
        if not inpaintor_path.exists() or not line_path.exists():
            print(
                "  [MangaCleaner] missing manga cleaner weights; using local/AnimeLaMa fallback "
                f"({inpaintor_path}, {line_path})",
                flush=True,
            )
            return None

        if _bridge_enabled():
            print("  [MangaCleaner] bridge enabled; deferring to remote LaMa")
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            inpaintor = torch.jit.load(str(inpaintor_path), map_location="cpu").to(device).eval()
            line = torch.jit.load(str(line_path), map_location="cpu").to(device).eval()
        except Exception as error:
            print(f"  [MangaCleaner] load failed; using fallback. {str(error)[:160]}", flush=True)
            return None

        _MANGA_CLEANER_MODELS = (inpaintor, line, device)
        print(f"  [MangaCleaner] dedicated manga cleaner: {device.type.upper()} ({inpaintor_path})", flush=True)
        return _MANGA_CLEANER_MODELS


def _pad_to_modulo(arr: np.ndarray, modulo: int = 8, is_mask: bool = False) -> np.ndarray:
    height, width = arr.shape[:2]
    out_height = ((height + modulo - 1) // modulo) * modulo
    out_width = ((width + modulo - 1) // modulo) * modulo

    if arr.ndim == 2:
        pad_width = ((0, out_height - height), (0, out_width - width))
    else:
        pad_width = ((0, out_height - height), (0, out_width - width), (0, 0))

    return np.pad(arr, pad_width, mode="constant" if is_mask else "symmetric")


def _anime_lama_inpaint(anime_model, anime_device, crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = _pad_to_modulo(crop_rgb, modulo=8, is_mask=False)
    mask_pad = _pad_to_modulo(mask, modulo=8, is_mask=True)

    image_tensor = (crop_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    mask_tensor = (mask_pad.astype(np.float32) / 255.0)[None, None]

    with torch.no_grad():
        output = anime_model(
            torch.from_numpy(image_tensor).to(anime_device),
            torch.from_numpy(mask_tensor).to(anime_device),
        )[0]

    output = output.permute(1, 2, 0).detach().cpu().numpy()
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    output = output[: crop_bgr.shape[0], : crop_bgr.shape[1]]
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


def _roi_is_manga_cleaner_candidate(crop_bgr: np.ndarray, crop_mask: np.ndarray) -> bool:
    if crop_bgr.size == 0 or np.count_nonzero(crop_mask > 0) < 8:
        return False
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    saturation_p90 = float(np.percentile(hsv[:, :, 1], 90))
    channel_spread = float(np.mean(crop_bgr.max(axis=2).astype(np.int16) - crop_bgr.min(axis=2).astype(np.int16)))
    edge_density = float(np.mean(cv2.Canny(gray, 45, 135) > 0))
    force = os.getenv("MANGA_CLEANER_BACKEND", "auto").strip().lower() == "force"
    return force or (saturation_p90 <= 48.0 and channel_spread <= 36.0 and edge_density >= 0.035)


def _manga_cleaner_inpaint(crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    models = _load_manga_cleaner_models()
    if models is None or not _roi_is_manga_cleaner_candidate(crop_bgr, mask):
        return None
    inpaintor, line_model, device = models

    height, width = crop_bgr.shape[:2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = _pad_to_modulo(crop_rgb, modulo=16, is_mask=False)
    mask_pad = _pad_to_modulo((mask > 0).astype(np.uint8) * 255, modulo=16, is_mask=True)

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray_tensor = torch.from_numpy(gray[np.newaxis, np.newaxis, :, :].astype(np.float32)).to(device)
    mask_tensor = torch.from_numpy(mask_pad[np.newaxis, :, :, np.newaxis].astype(np.float32)).to(device)
    mask_tensor = mask_tensor.permute(0, 3, 1, 2)
    mask_tensor = torch.where(mask_tensor > 0.5, 1.0, 0.0)

    with torch.no_grad():
        line_tensor = torch.clamp(line_model(gray_tensor), 0, 255)
        noise = torch.zeros_like(mask_tensor)
        ones = torch.ones_like(mask_tensor)
        gray_norm = gray_tensor / 255.0 * 2.0 - 1.0
        line_norm = line_tensor / 255.0 * 2.0 - 1.0
        output = inpaintor(gray_norm, line_norm, mask_tensor, noise, ones)

    output_np = output[0].permute(1, 2, 0).detach().cpu().numpy()
    output_np = np.clip(output_np * 127.5 + 127.5, 0, 255).astype(np.uint8)
    output_np = output_np[:height, :width]
    if output_np.ndim == 3 and output_np.shape[2] == 1:
        output_np = output_np[:, :, 0]
    return cv2.cvtColor(output_np, cv2.COLOR_GRAY2BGR)


def _context_crop_bounds(img_h: int, img_w: int, x1: int, y1: int, x2: int, y2: int):
    box_width = x2 - x1
    box_height = y2 - y1
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    half_size = max(256, min(512, max(box_width, box_height) // 2 + 220))

    crop_x1 = max(0, center_x - half_size)
    crop_x2 = min(img_w, center_x + half_size)
    crop_y1 = max(0, center_y - half_size)
    crop_y2 = min(img_h, center_y + half_size)
    target_size = half_size * 2

    if crop_x2 - crop_x1 < min(img_w, target_size):
        if crop_x1 == 0:
            crop_x2 = min(img_w, target_size)
        elif crop_x2 == img_w:
            crop_x1 = max(0, img_w - target_size)

    if crop_y2 - crop_y1 < min(img_h, target_size):
        if crop_y1 == 0:
            crop_y2 = min(img_h, target_size)
        elif crop_y2 == img_h:
            crop_y1 = max(0, img_h - target_size)

    return crop_x1, crop_y1, crop_x2, crop_y2


def _anime_lama_local_crop(
    anime_model,
    anime_device,
    image,
    mask,
    img_h,
    img_w,
    x1,
    y1,
    x2,
    y2,
    blend_mask=None,
):
    crop_x1, crop_y1, crop_x2, crop_y2 = _context_crop_bounds(img_h, img_w, x1, y1, x2, y2)
    crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    if not np.any(crop_mask > 0):
        return

    if blend_mask is None:
        crop_blend_mask = crop_mask
    else:
        crop_blend_mask = blend_mask[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        if not np.any(crop_blend_mask > 0):
            crop_blend_mask = crop_mask

    inpainted_crop = _anime_lama_inpaint(anime_model, anime_device, crop_img, crop_mask)
    alpha = cv2.GaussianBlur((crop_blend_mask > 0).astype(np.float32), (0, 0), 1.8)
    alpha = np.clip(alpha[..., None], 0.0, 1.0)

    view = image[crop_y1:crop_y2, crop_x1:crop_x2]
    blended = (
        inpainted_crop.astype(np.float32) * alpha
        + view.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)
    view[crop_blend_mask > 0] = blended[crop_blend_mask > 0]


def _reinpaint_ghost_residue(
    anime_model,
    anime_device,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
    max_rounds: int = 2,
    container_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Self-check after a model inpaint: when the erase mask missed the
    anti-aliased white outlines of the source lettering, the model re-embosses
    glyph-shaped ghosts (seen on framed captions over pink gradients). Ghosts
    are EDGES inside the cleaned area that its surroundings don't have —
    genuine reconstruction (door slats, tatami) continues the surrounding
    texture, so its inside/outside edge densities match. On detection, grow
    the mask through the residue edges and re-run the model.

    `container_mask` (page-sized, pre-union of every traced bubble/container
    outline) clips the ring/regrow to the SAME container the cleaned area
    came from. Organic bubble shapes make the ring cross the bubble's own
    wall into busy surrounding art, manufacturing a false tone/edge mismatch
    against art that was never part of this region to begin with (verified:
    destroyed new_sample_3's correctly-filled bubble interiors and
    new_sample_5's black-bubble fill, both re-reconstructed into the
    character art behind them).

    Returns the (possibly grown) mask ROI."""
    if anime_model is None or anime_device is None:
        return mask_roi
    x1, y1, x2, y2 = coords
    img_h, img_w = target.shape[:2]
    container_roi = None
    if container_mask is not None:
        crop = container_mask[y1:y2, x1:x2]
        if crop.size and np.any(crop):
            container_roi = crop > 0
    for _ in range(max_rounds):
        result_roi = target[y1:y2, x1:x2]
        if result_roi.size == 0:
            return mask_roi
        gray = cv2.cvtColor(result_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120) > 0
        inside = cv2.erode(
            (mask_roi > 0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        ).astype(bool)
        ring = (
            cv2.dilate(
                (mask_roi > 0).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            ).astype(bool)
            & ~(mask_roi > 0)
        )
        if container_roi is not None:
            ring = ring & container_roi
        if int(np.count_nonzero(inside)) < 80 or int(np.count_nonzero(ring)) < 80:
            return mask_roi
        edge_inside = float(np.mean(edges[inside]))
        edge_ring = float(np.mean(edges[ring]))
        luma_inside = float(np.mean(gray[inside].astype(np.float32)))
        luma_ring = float(np.mean(gray[ring].astype(np.float32)))
        ghost_edges = edge_inside > max(0.015, edge_ring * 1.6)
        # The inverse failure: a flat fill ERASED texture the surroundings
        # have (pale rectangle over flower art / a gradient) — the cleaned
        # area is suspiciously smoother or tonally shifted vs its ring.
        flattened_fill = edge_ring >= 0.06 and edge_inside <= edge_ring * 0.5
        tone_delta = abs(luma_inside - luma_ring)
        tone_mismatch = edge_inside <= 0.02 and (
            tone_delta >= 14.0
            # On white paper even a slight gray cast reads as a dirty smudge.
            or (tone_delta >= 6.0 and luma_ring >= 240.0)
            or (tone_delta >= 8.0 and luma_ring >= 235.0)
        )
        if container_roi is not None and (flattened_fill or tone_mismatch):
            # `ring`'s tone/edge stats are only a trustworthy "what this
            # surface should look like" reference when a meaningful share of
            # the traced container actually lies OUTSIDE the current mask --
            # otherwise the ring is squeezed into a thin boundary sliver,
            # which is exactly where a container polygon's own tracing
            # imprecision concentrates (antialiased outline, flood-fill
            # leaking a pixel or two past the true edge into adjacent art).
            # A fully-filled container (mask_roi already claims ~all of it)
            # has no legitimate "rest of the same container" left to sample;
            # whatever tone the ring finds is leak noise, not evidence the
            # fill is wrong (verified: new_sample_5's id12 -- a correctly
            # flat-black 57-luma bubble interior, whose polygon's boundary
            # sliver sampled a bright adjacent highlight at luma 217, a false
            # 160-point "mismatch" that triggered a full re-reconstruct into
            # hair/shoulder texture). ghost_edges is unaffected: it depends
            # on edge density, not ring tone, so it isn't vulnerable to this.
            container_total = int(np.count_nonzero(container_roi))
            container_margin = int(np.count_nonzero(container_roi & ~(mask_roi > 0)))
            if container_total > 0 and (container_margin / container_total) < 0.20:
                flattened_fill = False
                tone_mismatch = False
        elif container_roi is None and (flattened_fill or tone_mismatch) and luma_inside <= 140.0 and edge_inside <= 0.15:
            # Same failure family, no traced polygon to fall back on (a
            # floating device-UI panel/banner, not a bubble). The ring can
            # legitimately be dominated by a DIFFERENT, brighter surface --
            # not because this fill erased shared texture, but because the
            # mask sits right at the true edge of a small bounded dark
            # object (a UI banner, a dark button) embedded in a much
            # brighter surrounding page/screen. A clean (low edge_inside,
            # so not ghost-riddled) dark fill next to an overwhelmingly
            # bright ring is that boundary, not evidence of a mistake.
            # Verified: new_sample_4's black promo banner on a phone
            # screenshot -- ring_bright_frac 0.79, luma_ring 210.7 vs a
            # correctly-dark luma_inside 96.2 -- triggered flattened_fill
            # and got fully re-reconstructed by the model into the
            # surrounding screen's light prior (~92% blown to white).
            ring_bright_frac = float(np.mean(gray[ring] > 180)) if int(np.count_nonzero(ring)) else 0.0
            if ring_bright_frac >= 0.60:
                flattened_fill = False
                tone_mismatch = False
        if not (ghost_edges or flattened_fill or tone_mismatch):
            return mask_roi
        if tone_mismatch and luma_ring >= 240.0 and edge_ring <= 0.03:
            # Plain paper: the model tints large holes gray; the correct
            # repair is deterministic — normalize to the surrounding paper.
            paper_tone = np.median(
                result_roi[ring].reshape(-1, 3).astype(np.float32), axis=0
            )
            fill_zone = mask_roi > 0
            result_roi[fill_zone] = np.clip(paper_tone, 0, 255).astype(np.uint8)
            return mask_roi
        if tone_mismatch and luma_ring <= 60.0 and edge_ring <= 0.03:
            # Symmetric case: a plain DARK fill (black speech bubble, device
            # panel) that the model tinted grayish. Same deterministic fix,
            # mirrored to the dark end of the tone range.
            dark_tone = np.median(
                result_roi[ring].reshape(-1, 3).astype(np.float32), axis=0
            )
            fill_zone = mask_roi > 0
            result_roi[fill_zone] = np.clip(dark_tone, 0, 255).astype(np.uint8)
            return mask_roi
        if ghost_edges:
            residue = (edges & inside).astype(np.uint8)
            residue = cv2.dilate(
                residue, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            )
            grown = np.maximum(mask_roi, residue * 255)
        else:
            # Re-reconstruct the whole damaged region: the model continues
            # the surrounding texture/tone through it.
            grown = cv2.dilate(
                mask_roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            )
        if container_roi is not None:
            # A "re-reconstruct" can never grow past the container's own
            # wall -- that would erase into art the container was supposed
            # to protect.
            grown = np.where(container_roi, grown, mask_roi)
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = grown
        _anime_lama_local_crop(
            anime_model,
            anime_device,
            target,
            full_mask,
            img_h,
            img_w,
            x1,
            y1,
            x2,
            y2,
        )
        if not ghost_edges and int(np.count_nonzero(grown)) == int(
            np.count_nonzero(mask_roi)
        ):
            # Nothing left to grow on the next round: one full re-reconstruction
            # is the fix; a second identical pass would be a no-op.
            return grown
        mask_roi = grown
    return mask_roi


def _manga_cleaner_local_crop(image, mask, img_h, img_w, x1, y1, x2, y2) -> bool:
    crop_x1, crop_y1, crop_x2, crop_y2 = _context_crop_bounds(img_h, img_w, x1, y1, x2, y2)
    crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    if crop_img.size == 0 or np.count_nonzero(crop_mask > 0) < 8:
        return False

    inpainted_crop = _manga_cleaner_inpaint(crop_img, crop_mask)
    if inpainted_crop is None:
        return False

    alpha = cv2.GaussianBlur((crop_mask > 0).astype(np.float32), (0, 0), 0.7)
    alpha = np.clip(alpha[..., None], 0.0, 1.0)
    view = image[crop_y1:crop_y2, crop_x1:crop_x2]
    blended = (
        inpainted_crop.astype(np.float32) * alpha
        + view.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)
    view[crop_mask > 0] = blended[crop_mask > 0]
    return True


def _external_inpaint_command_local_crop(
    image: np.ndarray,
    mask: np.ndarray,
    img_h: int,
    img_w: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> bool:
    command_template = os.getenv("MANGA_INPAINT_COMMAND", "").strip()
    if not command_template:
        return False
    if _bridge_enabled():
        # Hard refusal, not a silent skip. This launches an EXTERNAL process (typically
        # iopaint_inpaint_backend.py, whose --device defaults to cuda), so it is invisible
        # to every in-process bridge guard: it would quietly consume the local GPU while
        # the code-level checks all still reported clean. Failing loudly is the only way
        # a "zero local GPU" claim stays honest.
        raise RuntimeError(
            "MANGA_INPAINT_COMMAND is set while FMT_GPU_BRIDGE is enabled. That command "
            "runs in a separate process and would use the LOCAL GPU, which the bridge "
            "cannot intercept. Unset MANGA_INPAINT_COMMAND for remote-GPU runs."
        )

    crop_x1, crop_y1, crop_x2, crop_y2 = _context_crop_bounds(img_h, img_w, x1, y1, x2, y2)
    crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    if crop_img.size == 0 or np.count_nonzero(crop_mask > 0) < 6:
        return False

    with tempfile.TemporaryDirectory(prefix="manga_inpaint_") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "input.png"
        mask_path = temp_path / "mask.png"
        output_path = temp_path / "output.png"
        cv2.imwrite(str(input_path), crop_img)
        cv2.imwrite(str(mask_path), (crop_mask > 0).astype(np.uint8) * 255)

        # Tokenize the operator's template BEFORE substituting the temp-file paths, then format
        # each token independently. This keeps the actual argv split under our control (no
        # shell=True, so no shell-metacharacter injection surface) while never re-parsing the
        # substituted paths themselves -- shlex only ever sees the operator's own literal template
        # text, so backslashes inside the runtime-substituted Windows paths can't be misread as
        # shlex escape characters. MANGA_INPAINT_COMMAND is operator-set via .env, not
        # request-derived; if its literal (non-placeholder) portion needs a path containing spaces,
        # quote it in the template as you would on a command line.
        try:
            template_tokens = shlex.split(command_template)
        except ValueError as error:
            print(f"  [ExternalInpaint] MANGA_INPAINT_COMMAND is not a valid command line: {error}", flush=True)
            return False
        argv = [
            token.format(
                image=str(input_path),
                input=str(input_path),
                mask=str(mask_path),
                output=str(output_path),
            )
            for token in template_tokens
        ]
        try:
            completed = subprocess.run(
                argv,
                shell=False,
                cwd=str(EXTERNAL_INPAINT_CWD),
                capture_output=True,
                text=True,
                timeout=float(os.getenv("MANGA_INPAINT_COMMAND_TIMEOUT", "180")),
            )
        except subprocess.TimeoutExpired:
            print(
                "  [ExternalInpaint] command timed out; using local fallback.",
                flush=True,
            )
            return False
        except OSError as error:
            print(
                f"  [ExternalInpaint] command failed to launch ({error}); using local fallback.",
                flush=True,
            )
            return False
        if completed.returncode != 0:
            print(
                "  [ExternalInpaint] command failed; using local fallback. "
                f"stderr={completed.stderr[-500:]}",
                flush=True,
            )
            return False
        if not output_path.exists():
            print("  [ExternalInpaint] command did not write output; using local fallback.", flush=True)
            return False

        external = cv2.imread(str(output_path))
        if external is None or external.shape[:2] != crop_img.shape[:2]:
            print("  [ExternalInpaint] invalid output dimensions; using local fallback.", flush=True)
            return False

    alpha = cv2.GaussianBlur((crop_mask > 0).astype(np.float32), (0, 0), 1.2)
    alpha = np.clip(alpha[..., None], 0.0, 1.0)
    view = image[crop_y1:crop_y2, crop_x1:crop_x2]
    blended = (external.astype(np.float32) * alpha + view.astype(np.float32) * (1.0 - alpha)).astype(
        np.uint8
    )
    view[crop_mask > 0] = blended[crop_mask > 0]
    print("  [ExternalInpaint] repaired crop with configured command", flush=True)
    return True


def _apply_context_tone_match(reference: np.ndarray, image: np.ndarray, mask: np.ndarray):
    if not np.any(mask > 0):
        return

    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    for component_label in range(1, component_count):
        area = stats[component_label, cv2.CC_STAT_AREA]
        if area < 80:
            continue
        cx = int(stats[component_label, cv2.CC_STAT_LEFT])
        cy = int(stats[component_label, cv2.CC_STAT_TOP])
        cw = int(stats[component_label, cv2.CC_STAT_WIDTH])
        ch = int(stats[component_label, cv2.CC_STAT_HEIGHT])
        bbox_area = max(1, cw * ch)
        bbox_density = float(area) / float(bbox_area)
        if ch >= 72 and ch > cw * 1.35 and bbox_density >= 0.55:
            continue

        component = (labels == component_label).astype(np.uint8) * 255
        ring = cv2.dilate(component, ring_kernel, iterations=1)
        ring = cv2.subtract(ring, component)

        ring_values = ref_gray[ring > 0]
        ring_values = ring_values[(ring_values > 25) & (ring_values < 250)]
        if ring_values.size < 100:
            continue

        component_values = img_gray[component > 0]
        component_values = component_values[(component_values > 25) & (component_values < 250)]
        if component_values.size < 50:
            continue

        ring_mean = float(np.mean(ring_values))
        ring_std = float(np.std(ring_values))
        component_mean = float(np.mean(component_values))
        component_std = max(1.0, float(np.std(component_values)))
        if ring_mean > 170.0 and float(np.mean(ring_values > 168)) > 0.78 and ring_std < 32.0:
            continue

        if ring_std < 18 or component_std >= ring_std * 0.85:
            continue

        scale = min(1.8, ring_std / component_std)
        adjusted_gray = (img_gray.astype(np.float32) - component_mean) * scale + ring_mean
        delta = np.clip(adjusted_gray - img_gray.astype(np.float32), -35, 35)
        affect = (component > 0) & (img_gray > 80)

        for channel_idx in range(3):
            channel = image[:, :, channel_idx].astype(np.float32)
            channel[affect] = np.clip(channel[affect] + delta[affect], 0, 255)
            image[:, :, channel_idx] = channel.astype(np.uint8)


def _final_flat_paper_cleanup(reference: np.ndarray, image: np.ndarray, mask: np.ndarray):
    if not np.any(mask > 0):
        return

    grouped = cv2.dilate(
        (mask > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)

    for component_label in range(1, component_count):
        area = int(stats[component_label, cv2.CC_STAT_AREA])
        if area < 80:
            continue
        x = max(0, int(stats[component_label, cv2.CC_STAT_LEFT]) - 4)
        y = max(0, int(stats[component_label, cv2.CC_STAT_TOP]) - 4)
        w = int(stats[component_label, cv2.CC_STAT_WIDTH])
        h = int(stats[component_label, cv2.CC_STAT_HEIGHT])
        x2 = min(reference.shape[1], x + w + 8)
        y2 = min(reference.shape[0], y + h + 8)
        if x2 <= x or y2 <= y:
            continue

        bbox_area = max(1, w * h)
        bbox_density = float(area) / float(bbox_area)
        if bbox_density >= 0.40 or bbox_area >= 12000:
            continue

        roi = reference[y:y2, x:x2]
        roi_mask = mask[y:y2, x:x2]
        pad = 22
        rx1 = max(0, x - pad)
        ry1 = max(0, y - pad)
        rx2 = min(reference.shape[1], x2 + pad)
        ry2 = min(reference.shape[0], y2 + pad)
        patch = reference[ry1:ry2, rx1:rx2]
        if patch.size:
            ring = np.ones(patch.shape[:2], dtype=bool)
            ix1 = max(0, x - rx1)
            iy1 = max(0, y - ry1)
            ix2 = min(patch.shape[1], x2 - rx1)
            iy2 = min(patch.shape[0], y2 - ry1)
            ring[iy1:iy2, ix1:ix2] = False
            if np.count_nonzero(ring) >= 80:
                ring_gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                ring_hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                ring_gray = ring_gray_patch[ring]
                ring_sat = ring_hsv_patch[:, :, 1][ring]
                gray_tone_fraction = float(
                    np.mean((ring_gray >= 95) & (ring_gray <= 218) & (ring_sat < 145))
                )
                ring_median = float(np.median(ring_gray))
                ring_edges = float(np.mean(cv2.Canny(ring_gray_patch, 45, 135)[ring] > 0))
                bright_halftone_fraction = float(
                    np.mean((ring_gray > 230) & (ring_sat < 155))
                )
                if (
                    gray_tone_fraction >= 0.45
                    and 95.0 <= ring_median <= 205.0
                    and ring_edges <= 0.11
                ) or (
                    ring_median >= 228.0
                    and bright_halftone_fraction >= 0.52
                    and ring_edges <= 0.13
                ):
                    continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        paper_pixels = (gray > 168) & (hsv[:, :, 1] < 145)
        paper_fraction = float(np.mean(paper_pixels))
        if paper_fraction < 0.68:
            continue
        cleanup = cv2.dilate(
            (roi_mask > 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=2,
        ) > 0
        background = (~cleanup) & paper_pixels
        if np.count_nonzero(background) < max(20, int(cleanup.size * 0.04)):
            continue
        paper_luma = float(np.percentile(gray[background], 76))
        clean_background = background & (gray >= paper_luma - 2.0)
        if np.count_nonzero(clean_background) < 12:
            clean_background = background
        fill_color = np.median(roi[clean_background], axis=0).astype(np.uint8)
        image_roi = image[y:y2, x:x2]
        image_roi[cleanup] = fill_color


def _fill_with_neighbor_background(reference, image, x1, y1, x2, y2, padding: int = 10):
    img_h, img_w = image.shape[:2]
    ring_x1 = max(0, x1 - padding)
    ring_y1 = max(0, y1 - padding)
    ring_x2 = min(img_w, x2 + padding)
    ring_y2 = min(img_h, y2 + padding)

    ring_mask = np.ones((ring_y2 - ring_y1, ring_x2 - ring_x1), dtype=np.uint8)
    inner_x1 = x1 - ring_x1
    inner_y1 = y1 - ring_y1
    inner_x2 = x2 - ring_x1
    inner_y2 = y2 - ring_y1
    ring_mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0

    ring_pixels = reference[ring_y1:ring_y2, ring_x1:ring_x2][ring_mask > 0]
    if ring_pixels.size == 0:
        fill_color = np.array([255, 255, 255], dtype=np.uint8)
    else:
        ring_gray = cv2.cvtColor(
            ring_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY
        ).reshape(-1)
        bright_pixels = ring_pixels[ring_gray > 180]
        usable_pixels = bright_pixels if bright_pixels.shape[0] >= 20 else ring_pixels
        fill_color = np.median(usable_pixels, axis=0).astype(np.uint8)

    image[y1:y2, x1:x2] = fill_color


def _maybe_fill_screentone_background(reference, image, x1, y1, x2, y2, padding: int = 18) -> bool:
    img_h, img_w = image.shape[:2]
    ring_x1 = max(0, x1 - padding)
    ring_y1 = max(0, y1 - padding)
    ring_x2 = min(img_w, x2 + padding)
    ring_y2 = min(img_h, y2 + padding)

    if ring_x2 <= ring_x1 or ring_y2 <= ring_y1:
        return False

    ring_mask = np.ones((ring_y2 - ring_y1, ring_x2 - ring_x1), dtype=np.uint8)
    inner_x1 = x1 - ring_x1
    inner_y1 = y1 - ring_y1
    inner_x2 = x2 - ring_x1
    inner_y2 = y2 - ring_y1
    ring_mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0

    ring_patch = reference[ring_y1:ring_y2, ring_x1:ring_x2]
    ring_gray_patch = cv2.cvtColor(ring_patch, cv2.COLOR_BGR2GRAY)
    ring_gray = ring_gray_patch[ring_mask > 0]

    if ring_gray.size < 80:
        return False

    ring_median = float(np.median(ring_gray))
    ring_edges = cv2.Canny(ring_gray_patch, 50, 150)
    edge_density = float(np.mean(ring_edges[ring_mask > 0] > 0))
    bright_fraction = float(np.mean(ring_gray > 230))

    if not (90.0 <= ring_median <= 190.0):
        return False
    if edge_density > 0.16 or bright_fraction > 0.45:
        return False

    ring_pixels = ring_patch[ring_mask > 0]
    keep_pixels = (ring_gray > max(35.0, ring_median - 45.0)) & (
        ring_gray < min(245.0, ring_median + 45.0)
    )
    background_pixels = ring_pixels[keep_pixels]

    if background_pixels.shape[0] < 50:
        background_pixels = ring_pixels[(ring_gray > 35) & (ring_gray < 245)]
    if background_pixels.shape[0] < 50:
        return False

    target_h = y2 - y1
    target_w = x2 - x1
    background_mask = ring_mask > 0
    background_mask[background_mask] = keep_pixels
    bg_y, bg_x = np.where(background_mask)
    if bg_x.size < 50:
        return False

    design = np.column_stack(
        [
            bg_x.astype(np.float32) / float(max(1, ring_patch.shape[1] - 1)),
            bg_y.astype(np.float32) / float(max(1, ring_patch.shape[0] - 1)),
            np.ones_like(bg_x, dtype=np.float32),
        ]
    )
    target_grid_y, target_grid_x = np.indices((target_h, target_w))
    target_abs_x = target_grid_x.reshape(-1) + inner_x1
    target_abs_y = target_grid_y.reshape(-1) + inner_y1
    target_design = np.column_stack(
        [
            target_abs_x.astype(np.float32) / float(max(1, ring_patch.shape[1] - 1)),
            target_abs_y.astype(np.float32) / float(max(1, ring_patch.shape[0] - 1)),
            np.ones(target_h * target_w, dtype=np.float32),
        ]
    )

    fitted = np.empty((target_h, target_w, 3), dtype=np.float32)
    for channel in range(3):
        values = ring_patch[:, :, channel][background_mask].astype(np.float32)
        coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
        fitted[:, :, channel] = (target_design @ coeffs).reshape(target_h, target_w)

    image[y1:y2, x1:x2] = np.clip(fitted, 0, 255).astype(np.uint8)
    return True


def _clean_white_bubble_residue(
    image: np.ndarray,
    region_mask: np.ndarray,
    bubble_mask: np.ndarray,
    source: np.ndarray | None = None,
    debug_label: object = None,
) -> np.ndarray:
    """`image` is the paint target (the progressively-mutated working canvas -- the fill is
    written into it). `source` is the PRISTINE page, used only for the gate and the sampled
    fill color -- reading `image` for those would test/sample pixels a prior model pass has
    already smudged, so the rescue would decline (or sample a smudged color) exactly when a
    prior pass damaged the interior enough to matter. Falls back to `image` when `source` is
    not supplied, matching the historical (buggy) behavior for any other caller."""
    if bubble_mask is None or not np.any(region_mask > 0):
        return np.zeros(region_mask.shape, dtype=np.uint8)
    if source is None:
        source = image

    kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    safe_bubble = cv2.erode(bubble_mask, kernel_3, iterations=2)
    cleanup_mask = cv2.dilate(region_mask, kernel_3, iterations=2)
    cleanup_mask = cv2.bitwise_and(cleanup_mask, safe_bubble)

    # F-1b: never flatten a pixel that is dark in the pristine source and
    # falls outside a small halo around a detected glyph stroke (region_mask
    # is already the stroke-dilated ∩ container mask F-1a passes in). Without
    # this, lifting F-1a's early-out lets cleanup run in cases where it never
    # ran before, and its flat-fill footprint can still clip adjacent
    # non-glyph ink (bubble tail marks, small art) that happens to sit inside
    # the same dilated-stroke area -- confirmed on original/sample2 id=1.
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    # NOTE: cleanup_mask above is dilate(region_mask, iterations=2) & safe_bubble.
    # wall_protect_radius must be < 2 or stroke_halo becomes a superset of
    # cleanup_mask and this subtraction is a no-op by construction -- caught by
    # rerunning and finding r=2 changed zero pixels across all 34 samples.
    wall_protect_radius = int(os.getenv("MANGA_WALL_PROTECT_RADIUS", "1"))
    stroke_halo = cv2.dilate(region_mask, kernel_3, iterations=wall_protect_radius) if wall_protect_radius > 0 else region_mask
    # Task #206: stroke_halo's only signal is "far from a detected text stroke" -- in a small
    # enough bubble the OCR text bbox is tangent to the wall itself (confirmed on
    # original/sample2 id=1: red_box x2=117 vs wall x~117), so wall ink within
    # wall_protect_radius of a stroke can NEVER be protected by that signal alone, no matter how
    # the radius is tuned (r=2 was tried and made stroke_halo a superset of cleanup_mask,
    # collapsing protection to nothing everywhere -- see the note above; do not touch that
    # constant again). Add a second, independent signal instead: wall ink is, by definition,
    # close to the bubble's own traced boundary, regardless of its distance to text -- manga
    # lettering is conventionally set with interior clearance from the wall, so this does not
    # widen protection over genuine glyph ink away from the edge.
    # Measured (original/sample2 id=1, MANGA_DEBUG_WALL_PROTECT=1): edge_px=3 protected 0 of the
    # wall pixels this fix targets -- the traced outline sits ~4-6px outside the true wall (see
    # comment above), so a 3px search radius from the outline never reaches it. edge_px=6 closes
    # that gap and was confirmed to fully resolve id=1's wall gap by direct crop comparison, with
    # newly-protected-pixel counts (not just guessed) checked across every constraint in the
    # sample before landing on this value.
    edge_protect_px = int(os.getenv("MANGA_WALL_EDGE_PROTECT_PX", "6"))
    if edge_protect_px > 0:
        dist_from_edge = cv2.distanceTransform((bubble_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        near_bubble_edge = dist_from_edge <= edge_protect_px
    else:
        near_bubble_edge = np.zeros(region_mask.shape, dtype=bool)
    protected_ink = (gray < 110) & ((stroke_halo == 0) | near_bubble_edge)
    if os.getenv("MANGA_DEBUG_WALL_PROTECT") == "1":
        n_before = int(np.count_nonzero(cleanup_mask))
        n_prot = int(np.count_nonzero(cleanup_mask.astype(bool) & protected_ink))
        n_edge_only = int(np.count_nonzero(cleanup_mask.astype(bool) & (gray < 110) & near_bubble_edge & (stroke_halo != 0)))
        print(f"    [wall-protect-debug] id={debug_label} r={wall_protect_radius} edge_px={edge_protect_px} "
              f"cleanup_mask_before={n_before} overlap_with_protected={n_prot} "
              f"newly_protected_by_edge_signal={n_edge_only}", flush=True)
    cleanup_mask[protected_ink] = 0

    if np.count_nonzero(cleanup_mask) < 20:
        return np.zeros(region_mask.shape, dtype=np.uint8)

    bubble_pixels = gray[safe_bubble > 0]
    if bubble_pixels.size < 100:
        return np.zeros(region_mask.shape, dtype=np.uint8)

    bright_fraction = float(np.mean(bubble_pixels > 210))
    bubble_median = float(np.median(bubble_pixels))
    if bright_fraction < 0.55 or bubble_median < 205:
        return np.zeros(region_mask.shape, dtype=np.uint8)

    background_area = (safe_bubble > 0) & (cleanup_mask == 0) & (gray > 190)
    if np.count_nonzero(background_area) < 50:
        return np.zeros(region_mask.shape, dtype=np.uint8)

    fill_color = np.median(source[background_area], axis=0).astype(np.uint8)
    image[cleanup_mask > 0] = fill_color
    return cleanup_mask


def _fill_bubble_text_box_with_local_background(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    bubble_mask: np.ndarray | None,
    roi_text_mask: np.ndarray | None = None,
) -> np.ndarray:
    if bubble_mask is None:
        return np.zeros(source.shape[:2], dtype=np.uint8)

    img_h, img_w = source.shape[:2]
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    safe_bubble = cv2.erode(bubble_mask, kernel_3, iterations=3)

    pad = 3
    cleanup_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if roi_text_mask is not None and np.count_nonzero(roi_text_mask) >= 10:
        text_cleanup = cv2.dilate(
            roi_text_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=3,
        )
        cleanup_mask[y1:y2, x1:x2] = text_cleanup
    else:
        cleanup_mask[
            max(0, y1 - pad) : min(img_h, y2 + pad),
            max(0, x1 - pad) : min(img_w, x2 + pad),
        ] = 255
    cleanup_mask = cv2.bitwise_and(cleanup_mask, safe_bubble)
    if np.count_nonzero(cleanup_mask) < 20:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    source_box = np.zeros((img_h, img_w), dtype=np.uint8)
    source_box[
        max(0, y1 - pad * 2) : min(img_h, y2 + pad * 2),
        max(0, x1 - pad * 2) : min(img_w, x2 + pad * 2),
    ] = 255
    background_area = (
        (safe_bubble > 0)
        & (source_box == 0)
        & (gray > 150)
        & (hsv[:, :, 1] < 115)
    )
    if np.count_nonzero(background_area) < 80:
        background_area = (safe_bubble > 0) & (source_box == 0) & (gray > 145)
    if np.count_nonzero(background_area) < 40:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    fill_color = np.median(source[background_area], axis=0).astype(np.uint8)
    target[cleanup_mask > 0] = fill_color
    return cleanup_mask


def _filled_external_component(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask.shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        kernel_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_7, iterations=2)
    return filled


def _infer_unsegmented_white_bubble_mask(
    image: np.ndarray,
    coords: tuple[int, int, int, int],
) -> np.ndarray | None:
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = coords
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    pad_x = max(48, int(box_w * 3.0))
    pad_y = max(36, int(box_h * 0.40))
    roi_x1 = max(0, x1 - pad_x)
    roi_y1 = max(0, y1 - pad_y)
    roi_x2 = min(img_w, x2 + pad_x)
    roi_y2 = min(img_h, y2 + pad_y)
    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return None

    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    bright = ((gray > 204) & (hsv[:, :, 1] < 82)).astype(np.uint8) * 255
    kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel_5, iterations=2)

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if labels_count <= 1:
        return None

    seed = np.zeros(bright.shape, dtype=np.uint8)
    seed_x1 = max(0, x1 - roi_x1 - 8)
    seed_y1 = max(0, y1 - roi_y1 - 8)
    seed_x2 = min(bright.shape[1], x2 - roi_x1 + 8)
    seed_y2 = min(bright.shape[0], y2 - roi_y1 + 8)
    seed[seed_y1:seed_y2, seed_x1:seed_x2] = 255

    box_area = box_w * box_h
    image_area = img_w * img_h
    best_mask = None
    best_score = -1.0

    for label in range(1, labels_count):
        component = (labels == label).astype(np.uint8) * 255
        overlap = int(np.count_nonzero((component > 0) & (seed > 0)))
        if overlap < max(16, int(box_area * 0.015)):
            continue

        filled = _filled_external_component(component)
        filled_area = int(np.count_nonzero(filled > 0))
        if filled_area < max(900, int(box_area * 1.35)):
            continue
        if filled_area > max(180000, int(image_area * 0.12)):
            continue

        ys, xs = np.where(filled > 0)
        if xs.size == 0 or ys.size == 0:
            continue
        bx1, bx2 = int(xs.min()), int(xs.max()) + 1
        by1, by2 = int(ys.min()), int(ys.max()) + 1
        touches_edges = int(bx1 <= 1) + int(by1 <= 1) + int(bx2 >= bright.shape[1] - 2) + int(by2 >= bright.shape[0] - 2)
        if touches_edges >= 4:
            continue

        bbox_area = max(1, (bx2 - bx1) * (by2 - by1))
        fill_ratio = filled_area / bbox_area
        if fill_ratio < 0.28:
            continue

        score = overlap * 4.0 + filled_area * min(1.0, fill_ratio)
        if score > best_score:
            best_score = score
            best_mask = filled

    if best_mask is None:
        return None

    mask_bool = best_mask > 0
    if np.count_nonzero(mask_bool) < 1:
        return None
    edge_density = float(np.mean((cv2.Canny(gray, 45, 135) > 0)[mask_bool]))
    pale_fraction = float(np.mean(((gray > 198) & (hsv[:, :, 1] < 96))[mask_bool]))
    dark_fraction = float(np.mean(gray[mask_bool] < 132))
    if pale_fraction < 0.78 or dark_fraction > 0.20:
        return None
    if edge_density > 0.16:
        return None

    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    full_mask[roi_y1:roi_y2, roi_x1:roi_x2] = best_mask
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    full_mask = cv2.erode(full_mask, kernel_3, iterations=1)
    return full_mask


def _fill_unsegmented_bubble_text_strokes(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    text_seg_mask: np.ndarray | None = None,
) -> np.ndarray:
    bubble_mask = _infer_unsegmented_white_bubble_mask(source, coords)
    if bubble_mask is None or np.count_nonzero(bubble_mask > 0) < 900:
        return np.zeros(source.shape[:2], dtype=np.uint8)

    kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    x1, y1, x2, y2 = coords
    outline_erosion = 4 if min(x2 - x1, y2 - y1) >= 70 else 3
    safe_bubble = cv2.erode(bubble_mask, kernel_3, iterations=outline_erosion)
    if np.count_nonzero(safe_bubble > 0) < 600:
        return np.zeros(source.shape[:2], dtype=np.uint8)

    cleanup_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    if text_seg_mask is not None and text_seg_mask.shape == safe_bubble.shape:
        _, seg_bin = cv2.threshold(text_seg_mask, 127, 255, cv2.THRESH_BINARY)
        candidate_mask = cv2.bitwise_and(seg_bin, safe_bubble)
        seed_pad = 14
        seed_x1 = max(0, x1 - seed_pad)
        seed_y1 = max(0, y1 - seed_pad)
        seed_x2 = min(candidate_mask.shape[1], x2 + seed_pad)
        seed_y2 = min(candidate_mask.shape[0], y2 + seed_pad)
        seed = np.zeros_like(candidate_mask)
        seed[seed_y1:seed_y2, seed_x1:seed_x2] = 255
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
        text_box_area = max(1, (x2 - x1) * (y2 - y1))
        for label in range(1, labels_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < 4:
                continue
            if area > max(9000, int(text_box_area * 1.75)):
                continue
            if width > max(140, int((x2 - x1) * 2.4)) and height > max(140, int((y2 - y1) * 0.85)):
                continue
            if width > max(180, int((x2 - x1) * 0.62)) and height <= max(12, int((y2 - y1) * 0.16)):
                continue
            component = labels == label
            if np.count_nonzero(component & (seed > 0)) < 1:
                continue
            cleanup_mask[component] = 255

    if np.count_nonzero(cleanup_mask > 0) < 20:
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
        dark = ((safe_bubble > 0) & (gray < 188) & (hsv[:, :, 2] < 205)).astype(np.uint8) * 255
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel_3, iterations=1)

        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        bubble_area = int(np.count_nonzero(safe_bubble > 0))
        for label in range(1, labels_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < 4:
                continue
            if area > max(6000, int(bubble_area * 0.12)):
                continue
            if width > max(80, int((coords[2] - coords[0]) * 2.8)) and height > max(80, int((coords[3] - coords[1]) * 0.75)):
                continue
            if width > max(180, int((coords[2] - coords[0]) * 0.62)) and height <= max(12, int((coords[3] - coords[1]) * 0.16)):
                continue
            cleanup_mask[labels == label] = 255

    if np.count_nonzero(cleanup_mask > 0) < 20:
        return np.zeros(source.shape[:2], dtype=np.uint8)

    cleanup_mask = cv2.dilate(cleanup_mask, kernel_3, iterations=2)
    cleanup_mask = cv2.bitwise_and(cleanup_mask, safe_bubble)

    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    background_area = (
        (safe_bubble > 0)
        & (cleanup_mask == 0)
        & (gray > 204)
        & (hsv[:, :, 1] < 88)
    )
    if np.count_nonzero(background_area) < 80:
        background_area = (safe_bubble > 0) & (cleanup_mask == 0) & (gray > 196)
    if np.count_nonzero(background_area) < 40:
        return np.zeros(source.shape[:2], dtype=np.uint8)

    fill_color = np.median(source[background_area], axis=0).astype(np.uint8)
    target[cleanup_mask > 0] = fill_color
    return cleanup_mask


def _fill_inferred_bubble_polygon_text_strokes(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    polygon: list,
    text_seg_mask: np.ndarray | None = None,
) -> np.ndarray:
    img_h, img_w = source.shape[:2]
    if not polygon or len(polygon) < 3:
        return np.zeros((img_h, img_w), dtype=np.uint8)
    try:
        points = np.array(
            [[int(point[0]), int(point[1])] for point in polygon if len(point) >= 2],
            dtype=np.int32,
        )
    except (TypeError, ValueError):
        return np.zeros((img_h, img_w), dtype=np.uint8)
    if points.shape[0] < 3:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    bubble_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(bubble_mask, [points.reshape((-1, 1, 2))], 255)
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    safe_bubble = cv2.erode(bubble_mask, kernel_3, iterations=4)
    if np.count_nonzero(safe_bubble > 0) < 600:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    x1, y1, x2, y2 = coords
    cleanup_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if text_seg_mask is not None and text_seg_mask.shape == safe_bubble.shape:
        _, seg_bin = cv2.threshold(text_seg_mask, 127, 255, cv2.THRESH_BINARY)
        candidate_mask = cv2.bitwise_and(seg_bin, safe_bubble)
        seed_x_pad = 8
        seed_top_pad = 3
        seed_bottom_pad = 2
        seed = np.zeros_like(candidate_mask)
        seed[
            max(0, y1 - seed_top_pad):min(candidate_mask.shape[0], y2 + seed_bottom_pad),
            max(0, x1 - seed_x_pad):min(candidate_mask.shape[1], x2 + seed_x_pad),
        ] = 255
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
        text_box_area = max(1, (x2 - x1) * (y2 - y1))
        for label in range(1, labels_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < 4 or area > max(9000, int(text_box_area * 1.75)):
                continue
            if width > max(140, int((x2 - x1) * 2.4)) and height > max(140, int((y2 - y1) * 0.85)):
                continue
            if width > max(180, int((x2 - x1) * 0.62)) and height <= max(12, int((y2 - y1) * 0.16)):
                continue
            component = labels == label
            seed_overlap = int(np.count_nonzero(component & (seed > 0)))
            if seed_overlap < max(1, int(area * 0.08)):
                continue
            cleanup_mask[component] = 255

    if np.count_nonzero(cleanup_mask > 0) < 20:
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
        dark = ((safe_bubble > 0) & (gray < 188) & (hsv[:, :, 2] < 205)).astype(np.uint8) * 255
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel_3, iterations=1)
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        bubble_area = int(np.count_nonzero(safe_bubble > 0))
        for label in range(1, labels_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < 4 or area > max(6000, int(bubble_area * 0.12)):
                continue
            if width > max(80, int((x2 - x1) * 2.8)) and height > max(80, int((y2 - y1) * 0.75)):
                continue
            if width > max(180, int((x2 - x1) * 0.62)) and height <= max(12, int((y2 - y1) * 0.16)):
                continue
            cleanup_mask[labels == label] = 255

    if np.count_nonzero(cleanup_mask > 0) < 20:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    cleanup_mask = cv2.dilate(cleanup_mask, kernel_3, iterations=2)
    cleanup_mask = cv2.bitwise_and(cleanup_mask, safe_bubble)

    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    background_area = (
        (safe_bubble > 0)
        & (cleanup_mask == 0)
        & (gray > 194)
        & (hsv[:, :, 1] < 105)
    )
    if np.count_nonzero(background_area) < 80:
        background_area = (safe_bubble > 0) & (cleanup_mask == 0) & (gray > 184)
    if np.count_nonzero(background_area) < 40:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    fill_color = np.median(source[background_area], axis=0).astype(np.uint8)
    target[cleanup_mask > 0] = fill_color
    return cleanup_mask


def _inferred_polygon_mask(
    image_shape: tuple[int, int],
    polygon: list,
) -> np.ndarray:
    img_h, img_w = image_shape
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not polygon or len(polygon) < 3:
        return mask
    try:
        points = np.array(
            [[int(point[0]), int(point[1])] for point in polygon if len(point) >= 2],
            dtype=np.int32,
        )
    except (TypeError, ValueError):
        return mask
    if points.shape[0] < 3:
        return mask
    cv2.fillPoly(mask, [points.reshape((-1, 1, 2))], 255)
    return mask


def _fill_pale_dialogue_box_with_local_background(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
) -> np.ndarray:
    img_h, img_w = source.shape[:2]
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return np.zeros((img_h, img_w), dtype=np.uint8)
    crop = source[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((img_h, img_w), dtype=np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pale = (gray > 178) & (hsv[:, :, 1] < 96)
    dark = gray < 135
    if float(np.mean(pale)) < 0.50 or float(np.mean(dark)) > 0.34:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    text_core = (gray < 205) & (hsv[:, :, 1] < 190)
    text_core |= cv2.Canny(gray, 45, 135) > 0
    repair = cv2.dilate(
        text_core.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    )
    repair = cv2.morphologyEx(
        repair,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    repair_count = int(np.count_nonzero(repair > 0))
    repair_density = repair_count / float(max(1, repair.size))
    if repair_count < max(18, int(repair.size * 0.025)) or repair_density > 0.42:
        return np.zeros((img_h, img_w), dtype=np.uint8)
    fitted, _ = _tone_fit_context_background(
        source,
        coords,
        repair,
        padding=max(18, int(max(x2 - x1, y2 - y1) * 0.10)),
        rowwise=True,
    )
    if fitted is None:
        fill_pixels = crop[pale & (~dark)]
        if fill_pixels.size < 30:
            return np.zeros((img_h, img_w), dtype=np.uint8)
        fitted = np.empty_like(crop)
        fitted[:, :] = np.median(fill_pixels.reshape(-1, 3), axis=0).astype(np.uint8)
    target_roi = target[y1:y2, x1:x2]
    if target_roi.size == 0:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    repair_bool = repair > 0
    distance = cv2.distanceTransform(repair, cv2.DIST_L2, 3)
    feather = max(6.0, min(24.0, max(x2 - x1, y2 - y1) * 0.08))
    alpha = np.clip(distance / feather, 0.0, 1.0)
    text_like = text_core
    text_like = cv2.dilate(
        text_like.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ) > 0
    alpha[text_like] = 1.0
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 1.4)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    blended = (fitted.astype(np.float32) * alpha + target_roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    write_mask = repair_bool | (alpha[:, :, 0] > 0.10)
    target_roi[write_mask] = blended[write_mask]
    cleanup_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cleanup_mask[y1:y2, x1:x2] = repair
    return cleanup_mask


def _extract_dark_text_strokes(
    image: np.ndarray,
    coords: tuple[int, int, int, int],
    source_colors: list[list[int]] | list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    x1, y1, x2, y2 = coords
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((max(1, y2 - y1), max(1, x2 - x1)), dtype=np.uint8)

    color_candidates = np.zeros(roi.shape[:2], dtype=bool)
    if source_colors:
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.int32)
        for raw_color in source_colors:
            if not isinstance(raw_color, (list, tuple)) or len(raw_color) < 3:
                continue
            target = np.array(raw_color[:3], dtype=np.int32)
            distance = np.sqrt(np.sum((rgb - target) ** 2, axis=2))
            color_candidates |= distance < 95

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    channel_max = roi.max(axis=2)
    channel_min = roi.min(axis=2)
    color_spread = channel_max.astype(np.int16) - channel_min.astype(np.int16)
    if np.count_nonzero(color_candidates) >= 10:
        dark_candidates = color_candidates
    else:
        dark_candidates = (gray < 175) & (color_spread < 75)
    pale_bubble = (gray > 148) & (hsv[:, :, 1] < 125)
    if np.count_nonzero(pale_bubble) > max(80, int(pale_bubble.size * 0.08)):
        near_pale_bubble = cv2.dilate(
            pale_bubble.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=2,
        )
        dark_candidates &= near_pale_bubble > 0
    dark = dark_candidates.astype(np.uint8) * 255

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    strokes = np.zeros_like(dark)
    for component_label in range(1, component_count):
        area = stats[component_label, cv2.CC_STAT_AREA]
        width = stats[component_label, cv2.CC_STAT_WIDTH]
        height = stats[component_label, cv2.CC_STAT_HEIGHT]
        if area < 5 or area > 1800:
            continue
        if width > 95 or height > 130:
            continue
        if width > 70 and height <= 5:
            continue
        if height > 100 and width <= 5:
            continue
        strokes[labels == component_label] = 255

    return cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))


def _high_contrast_light_text_block_mask(roi: np.ndarray) -> np.ndarray | None:
    if roi.size == 0:
        return None

    height, width = roi.shape[:2]
    area = max(1, height * width)
    if area > 26000:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    bright_background = (gray > 178) & (saturation < 145)
    bright_fraction = float(np.mean(bright_background))
    dark_fraction = float(np.mean(gray < 112))
    if (
        bright_fraction < 0.22
        or dark_fraction < 0.10
        or float(np.std(gray.astype(np.float32))) < 42.0
    ):
        return None

    if np.count_nonzero(bright_background) >= max(12, int(area * 0.05)):
        background_luma = float(np.median(gray[bright_background]))
    else:
        background_luma = 218.0

    cutoffs = [
        int(np.clip(background_luma - 35.0, 118, 208)),
        int(np.clip(background_luma - 55.0, 96, 190)),
        178,
        156,
        134,
    ]
    best_mask = None
    best_score = -1.0
    for cutoff in dict.fromkeys(cutoffs):
        raw = ((gray < cutoff) & (saturation < 220)).astype(np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
        mask = np.zeros_like(raw, dtype=np.uint8)
        for component_label in range(1, component_count):
            component_area = int(stats[component_label, cv2.CC_STAT_AREA])
            component_width = int(stats[component_label, cv2.CC_STAT_WIDTH])
            component_height = int(stats[component_label, cv2.CC_STAT_HEIGHT])
            if component_area < 3:
                continue
            if component_area > max(12000, int(area * 0.72)):
                continue
            if component_width > int(width * 0.98) and component_height <= 5:
                continue
            if component_height > int(height * 0.99) and component_width <= 5:
                continue
            mask[labels == component_label] = 255

        count = int(np.count_nonzero(mask > 0))
        density = count / float(area)
        if count < 8 or density > 0.72:
            continue
        if density < 0.035 and area > 2400:
            continue
        score = count - abs(density - 0.42) * area * 0.28
        if score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None:
        return None

    return cv2.dilate(
        best_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )


def _floating_text_erase_roi(image: np.ndarray, coords: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = coords
    height = max(1, y2 - y1)
    width = max(1, x2 - x1)
    area = height * width

    roi = image[y1:y2, x1:x2]
    paper_mask = np.zeros((height, width), dtype=np.uint8)
    if roi.size:
        raw_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        raw_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        high_contrast_mask = _high_contrast_light_text_block_mask(roi)
        if high_contrast_mask is not None:
            return high_contrast_mask

        paper_fraction = float(np.mean((raw_gray > 168) & (raw_hsv[:, :, 1] < 115)))
        if paper_fraction >= 0.45:
            paper_pixels = raw_gray[(raw_gray > 168) & (raw_hsv[:, :, 1] < 145)]
            paper_luma = float(np.median(paper_pixels)) if paper_pixels.size else 219.0
            cutoff = int(np.clip(paper_luma - 3.0, 188, 234))
            dark_candidates = ((raw_gray < cutoff) & (raw_hsv[:, :, 1] < 155)).astype(np.uint8) * 255
            component_count, labels, stats, _ = cv2.connectedComponentsWithStats(dark_candidates, connectivity=8)
            for component_label in range(1, component_count):
                component_area = int(stats[component_label, cv2.CC_STAT_AREA])
                component_width = int(stats[component_label, cv2.CC_STAT_WIDTH])
                component_height = int(stats[component_label, cv2.CC_STAT_HEIGHT])
                if component_area < 2:
                    continue
                if component_width > int(width * 0.84) and component_height <= 8:
                    continue
                if component_height > int(height * 0.90) and component_width <= 8:
                    continue
                if component_area > int(area * 0.18):
                    continue
                paper_mask[labels == component_label] = 255
            if np.count_nonzero(paper_mask > 0) >= 8:
                paper_mask = cv2.dilate(
                    paper_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=1,
                )

    stroke_roi = _extract_text_strokes(image, coords, is_bubble=False)
    if np.count_nonzero(paper_mask > 0) >= 8:
        stroke_roi = np.maximum(stroke_roi, paper_mask)
    stroke_count = int(np.count_nonzero(stroke_roi > 0))
    stroke_density = stroke_count / float(max(1, area))

    if stroke_count < 6 or stroke_density > 0.48:
        dark_roi = _extract_dark_text_strokes(image, coords)
        dark_count = int(np.count_nonzero(dark_roi > 0))
        dark_density = dark_count / float(max(1, area))
        if dark_count >= 6 and dark_density <= 0.42:
            stroke_roi = dark_roi
            stroke_count = dark_count
            stroke_density = dark_density

    if stroke_count >= 6 and stroke_density <= 0.55:
        kernel_size = 3 if min(width, height) < 44 else 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate(stroke_roi, kernel, iterations=1)

    if area <= 1400:
        return np.ones((height, width), dtype=np.uint8) * 255
    return np.zeros((height, width), dtype=np.uint8)


def _floating_erase_mask_for_box(
    image: np.ndarray,
    seg_mask: np.ndarray,
    coords: tuple[int, int, int, int],
    kernel: np.ndarray,
) -> np.ndarray:
    x1, y1, x2, y2 = coords
    seg_erase_roi = seg_mask[y1:y2, x1:x2]
    _, seg_erase_roi = cv2.threshold(seg_erase_roi, 127, 255, cv2.THRESH_BINARY)
    if np.count_nonzero(seg_erase_roi > 0) >= 6:
        seg_erase_roi = cv2.dilate(seg_erase_roi, kernel, iterations=2)
        local_erase_roi = _floating_text_erase_roi(image, coords)
        seg_count = int(np.count_nonzero(seg_erase_roi > 0))
        local_count = int(np.count_nonzero(local_erase_roi > 0))
        area = max(1, seg_erase_roi.size)
        seg_density = seg_count / float(area)
        local_density = local_count / float(area)
        if (
            local_count >= max(6, int(seg_count * 0.18))
            and seg_density >= 0.34
            and 0.025 <= local_density <= seg_density * 0.76
        ):
            return local_erase_roi
        return seg_erase_roi
    return _floating_text_erase_roi(image, coords)


def _fill_bright_paper_text_mask(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> bool:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0 or np.count_nonzero(mask_roi > 0) < 6:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    if float(np.mean(gray)) < 206.0:
        return False
    if _floating_cleanup_should_fail_closed(source, coords, mask_roi):
        return False
    unmasked = mask_roi <= 0
    if np.count_nonzero(unmasked) < 20:
        return False

    background_candidates = unmasked & (gray > 168) & (hsv[:, :, 1] < 95)
    if np.count_nonzero(background_candidates) < 20:
        background_candidates = unmasked & (gray > 185)
    if np.count_nonzero(background_candidates) < 20:
        return False

    median_luma = float(np.median(gray[background_candidates]))
    paper_fraction = float(np.mean((gray > 172) & (hsv[:, :, 1] < 110)))
    edge_density = float(np.mean(cv2.Canny(gray, 50, 150) > 0))
    if median_luma < 178 or paper_fraction < 0.48:
        return False
    if edge_density > 0.28 and paper_fraction < 0.72:
        return False
    if not _continuous_flat_background_allowed(source, coords, mask_roi):
        return False

    flat_paper = paper_fraction >= 0.72 and edge_density <= 0.20
    kernel_size = 9 if flat_paper else 5
    iterations = 2 if flat_paper else 1
    cleanup = cv2.dilate(
        (mask_roi > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        iterations=iterations,
    )
    target_roi = target[y1:y2, x1:x2]
    cleanup_bool = cleanup > 0
    if not _apply_bright_context_plane_fill(
        source,
        target_roi,
        coords,
        cleanup_bool,
        median_luma,
    ) and not _apply_local_plane_fill(roi, target_roi, cleanup_bool, background_candidates):
        fill_color = np.median(roi[background_candidates], axis=0).astype(np.uint8)
        target_roi[cleanup_bool] = fill_color
    return True


def _apply_bright_context_plane_fill(
    source: np.ndarray,
    target_roi: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_bool: np.ndarray,
    local_luma: float,
) -> bool:
    x1, y1, x2, y2 = coords
    img_h, img_w = source.shape[:2]
    pad = max(24, min(120, max(x2 - x1, y2 - y1) // 2))
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(img_w, x2 + pad)
    cy2 = min(img_h, y2 + pad)
    context = source[cy1:cy2, cx1:cx2]
    if context.size == 0:
        return False

    ctx_gray = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY)
    ctx_hsv = cv2.cvtColor(context, cv2.COLOR_BGR2HSV)
    ctx_edges = cv2.Canny(ctx_gray, 50, 150) > 0
    local_mask = np.zeros(ctx_gray.shape, dtype=bool)
    ix1, iy1 = x1 - cx1, y1 - cy1
    ix2, iy2 = ix1 + mask_bool.shape[1], iy1 + mask_bool.shape[0]
    local_mask[iy1:iy2, ix1:ix2] = mask_bool
    guard = cv2.dilate(
        local_mask.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=1,
    ) > 0
    background = (
        (~guard)
        & (~ctx_edges)
        & (ctx_hsv[:, :, 1] < 120)
        & (ctx_gray > max(168, int(local_luma - 30)))
        & (ctx_gray < min(252, int(local_luma + 18)))
    )
    if np.count_nonzero(background) < max(80, int(mask_bool.size * 0.10)):
        return False
    return _apply_context_plane_fill(context, target_roi, mask_bool, background, ix1, iy1)


def _apply_local_plane_fill(
    roi: np.ndarray,
    target_roi: np.ndarray,
    mask_bool: np.ndarray,
    background_mask: np.ndarray,
) -> bool:
    """Fill a text mask from a local color plane instead of a flat rectangle."""

    if roi.size == 0 or target_roi.size == 0:
        return False
    if mask_bool.shape != background_mask.shape or mask_bool.shape != roi.shape[:2]:
        return False
    if np.count_nonzero(mask_bool) < 4:
        return False
    if np.count_nonzero(background_mask) < max(16, int(mask_bool.size * 0.025)):
        return False

    height, width = mask_bool.shape
    bg_y, bg_x = np.where(background_mask)
    if bg_x.size < 16:
        return False

    design = np.column_stack(
        [
            bg_x.astype(np.float32) / float(max(1, width - 1)),
            bg_y.astype(np.float32) / float(max(1, height - 1)),
            np.ones_like(bg_x, dtype=np.float32),
        ]
    )
    grid_y, grid_x = np.indices((height, width))
    grid_design = np.column_stack(
        [
            grid_x.reshape(-1).astype(np.float32) / float(max(1, width - 1)),
            grid_y.reshape(-1).astype(np.float32) / float(max(1, height - 1)),
            np.ones(width * height, dtype=np.float32),
        ]
    )

    fitted = np.empty_like(roi, dtype=np.float32)
    for channel in range(3):
        values = roi[:, :, channel][background_mask].astype(np.float32)
        if values.size < 16:
            return False
        coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
        fitted[:, :, channel] = (grid_design @ coeffs).reshape(height, width)

    fitted = np.clip(fitted, 0, 255).astype(np.uint8)
    target_roi[mask_bool] = fitted[mask_bool]
    return True


def _continuous_flat_background_allowed(
    source: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> bool:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return False
    mask_bool = mask_roi > 0
    mask_count = int(np.count_nonzero(mask_bool))
    area = max(1, mask_bool.size)
    if mask_count < 6 or mask_count / float(area) > 0.78:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 45, 135) > 0
    seed_guard = cv2.dilate(
        mask_bool.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    ).astype(bool)
    background = (
        (~seed_guard)
        & (hsv[:, :, 1] < 160)
        & (gray > 48)
        & (gray < 252)
    )
    if np.count_nonzero(background) < max(28, int(area * 0.035)):
        background = (~seed_guard) & (hsv[:, :, 1] < 175) & (gray > 48)
    if np.count_nonzero(background) < max(28, int(area * 0.035)):
        return False

    bg_gray = gray[background].astype(np.float32)
    bg_sat = hsv[:, :, 1][background]
    bg_std = float(np.std(bg_gray))
    bg_edge = float(np.mean(edges[background]))
    paper_fraction = float(np.mean((bg_gray > 218) & (bg_sat < 125)))
    light_paper_fraction = float(np.mean((bg_gray > 172) & (bg_sat < 135)))
    gray_tone_fraction = float(np.mean((bg_gray >= 82) & (bg_gray <= 220) & (bg_sat < 155)))
    dark_fraction = float(np.mean(bg_gray < 92))

    if paper_fraction >= 0.58 or light_paper_fraction >= 0.78:
        return bg_std <= 25.0 and bg_edge <= 0.075 and dark_fraction <= 0.10
    if gray_tone_fraction >= 0.66:
        return bg_std <= 20.0 and bg_edge <= 0.060 and dark_fraction <= 0.16
    return False


def _apply_context_plane_fill(
    context_roi: np.ndarray,
    target_roi: np.ndarray,
    mask_bool: np.ndarray,
    context_background_mask: np.ndarray,
    inner_x1: int,
    inner_y1: int,
) -> bool:
    """Fill an ROI mask from a plane fitted to the surrounding context ring."""

    if context_roi.size == 0 or target_roi.size == 0:
        return False
    if mask_bool.shape != target_roi.shape[:2]:
        return False
    if context_background_mask.shape != context_roi.shape[:2]:
        return False
    if np.count_nonzero(mask_bool) < 4:
        return False
    if np.count_nonzero(context_background_mask) < 80:
        return False

    ctx_h, ctx_w = context_background_mask.shape
    bg_y, bg_x = np.where(context_background_mask)
    if bg_x.size < 80:
        return False

    design = np.column_stack(
        [
            bg_x.astype(np.float32) / float(max(1, ctx_w - 1)),
            bg_y.astype(np.float32) / float(max(1, ctx_h - 1)),
            np.ones_like(bg_x, dtype=np.float32),
        ]
    )
    roi_h, roi_w = mask_bool.shape
    grid_y, grid_x = np.indices((roi_h, roi_w))
    ctx_grid_x = grid_x.reshape(-1) + inner_x1
    ctx_grid_y = grid_y.reshape(-1) + inner_y1
    target_design = np.column_stack(
        [
            ctx_grid_x.astype(np.float32) / float(max(1, ctx_w - 1)),
            ctx_grid_y.astype(np.float32) / float(max(1, ctx_h - 1)),
            np.ones(roi_h * roi_w, dtype=np.float32),
        ]
    )

    fitted = np.empty((roi_h, roi_w, 3), dtype=np.float32)
    for channel in range(3):
        values = context_roi[:, :, channel][context_background_mask].astype(np.float32)
        if values.size < 80:
            return False
        coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
        fitted[:, :, channel] = (target_design @ coeffs).reshape(roi_h, roi_w)

    target_roi[mask_bool] = np.clip(fitted, 0, 255).astype(np.uint8)[mask_bool]
    return True


def _fill_high_contrast_light_text_mask(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> bool:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0 or np.count_nonzero(mask_roi > 0) < 8:
        return False

    high_contrast_mask = _high_contrast_light_text_block_mask(roi)
    if high_contrast_mask is None:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    paper_fraction = float(np.mean((gray > 170) & (hsv[:, :, 1] < 130)))
    edge_density = float(np.mean(cv2.Canny(gray, 50, 150) > 0))
    flat_paper = paper_fraction >= 0.72 and edge_density <= 0.24
    if not flat_paper and paper_fraction < 0.68:
        return False
    kernel_size = 9 if flat_paper else 3
    iterations = 2 if flat_paper else 1
    mask_bool = cv2.dilate(
        ((mask_roi > 0) | (high_contrast_mask > 0)).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        iterations=iterations,
    ) > 0
    area = max(1, mask_bool.size)
    mask_density = float(np.count_nonzero(mask_bool)) / float(area)
    if mask_density > 0.92:
        return False
    background = (~mask_bool) & (gray > 164) & (hsv[:, :, 1] < 150)
    if np.count_nonzero(background) < max(4, int(area * 0.015)):
        return False

    target_roi = target[y1:y2, x1:x2]
    if flat_paper:
        paper_pixels = gray[(gray > 168) & (hsv[:, :, 1] < 150)]
        paper_luma = float(np.percentile(paper_pixels, 78)) if paper_pixels.size else 219.0
        clean_background = (~mask_bool) & (gray >= paper_luma - 1.5) & (hsv[:, :, 1] < 150)
        if np.count_nonzero(clean_background) >= max(8, int(area * 0.02)):
            if not _apply_local_plane_fill(roi, target_roi, mask_bool, clean_background):
                fill_color = np.median(roi[clean_background], axis=0).astype(np.uint8)
                target_roi[mask_bool] = fill_color
            return True

    fallback_color = np.median(roi[background], axis=0).astype(np.uint8)
    height = roi.shape[0]
    for row_index in range(height):
        row_mask = mask_bool[row_index]
        if not np.any(row_mask):
            continue
        row_start = max(0, row_index - 4)
        row_end = min(height, row_index + 5)
        row_background = background[row_start:row_end]
        if np.count_nonzero(row_background) >= 5:
            fill_color = np.median(roi[row_start:row_end][row_background], axis=0).astype(np.uint8)
        else:
            fill_color = fallback_color
        target_roi[row_index][row_mask] = fill_color
    return True


def _fill_flat_background_text_mask(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> bool:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0 or np.count_nonzero(mask_roi > 0) < 6:
        return False

    if _floating_cleanup_should_fail_closed(source, coords, mask_roi):
        return False
    if not _continuous_flat_background_allowed(source, coords, mask_roi):
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 135)
    unmasked = mask_roi <= 0
    mask_density = float(np.count_nonzero(mask_roi > 0)) / float(max(1, mask_roi.size))
    if mask_density >= 0.72:
        if _maybe_fill_screentone_background(source, target, x1, y1, x2, y2, padding=35):
            return True

    pad = max(8, min(28, max(x2 - x1, y2 - y1) // 8))
    ctx_x1 = max(0, x1 - pad)
    ctx_y1 = max(0, y1 - pad)
    ctx_x2 = min(source.shape[1], x2 + pad)
    ctx_y2 = min(source.shape[0], y2 + pad)
    context_roi = source[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
    if context_roi.size:
        ctx_gray = cv2.cvtColor(context_roi, cv2.COLOR_BGR2GRAY)
        ctx_edges = cv2.Canny(ctx_gray, 45, 135)
        ring = np.ones(ctx_gray.shape, dtype=bool)
        inner_x1 = x1 - ctx_x1
        inner_y1 = y1 - ctx_y1
        inner_x2 = x2 - ctx_x1
        inner_y2 = y2 - ctx_y1
        ring[max(0, inner_y1):min(ring.shape[0], inner_y2), max(0, inner_x1):min(ring.shape[1], inner_x2)] = False
        context_candidates = ring & (ctx_gray > 55) & (ctx_edges == 0)
        if np.count_nonzero(context_candidates) >= 80:
            pixels = context_roi[context_candidates].astype(np.float32)
            median_color = np.median(pixels, axis=0)
            distances = np.linalg.norm(pixels - median_color, axis=1)
            inlier_pixels = pixels[distances < 46.0]
            if len(inlier_pixels) >= 80 and float(np.mean(np.std(inlier_pixels, axis=0))) <= 28.0:
                fill_color = np.median(inlier_pixels, axis=0).astype(np.uint8)
                fill_luma = float(np.mean(fill_color))
                inlier_std = float(np.mean(np.std(inlier_pixels, axis=0)))
                gray_tone_fill = 95.0 <= fill_luma < 222.0 and inlier_std <= 18.0
                if fill_luma < 222.0 and not gray_tone_fill:
                    return False
                if fill_luma > 242.0 and float(np.mean(edges > 0)) > 0.075:
                    return False
                gray_fill = fill_luma < 235.0
                cleanup = cv2.dilate(
                    (mask_roi > 0).astype(np.uint8) * 255,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3) if gray_tone_fill else ((5, 5) if gray_fill else (3, 3))),
                    iterations=1 if gray_tone_fill else (2 if gray_fill else 1),
                )
                target_roi = target[y1:y2, x1:x2]
                cleanup_bool = cleanup > 0
                inlier_context = np.zeros_like(context_candidates, dtype=bool)
                candidate_y, candidate_x = np.where(context_candidates)
                if candidate_x.size == pixels.shape[0]:
                    inlier_context[candidate_y[distances < 46.0], candidate_x[distances < 46.0]] = True
                if not _apply_context_plane_fill(
                    context_roi,
                    target_roi,
                    cleanup_bool,
                    inlier_context if np.count_nonzero(inlier_context) >= 80 else context_candidates,
                    inner_x1,
                    inner_y1,
                ):
                    target_roi[cleanup_bool] = fill_color
                return True

    border_width = max(3, min(10, min(roi.shape[:2]) // 8))
    border = np.zeros_like(gray, dtype=bool)
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    background_candidates = unmasked & border & (gray > 55) & (edges == 0)
    if np.count_nonzero(background_candidates) < 40:
        background_candidates = unmasked & border & (gray > 55)
    if np.count_nonzero(background_candidates) < 80:
        background_candidates = unmasked & (gray > 55)
    if np.count_nonzero(background_candidates) < 80:
        return False

    pixels = roi[background_candidates].astype(np.float32)
    median_color = np.median(pixels, axis=0)
    distances = np.linalg.norm(pixels - median_color, axis=1)
    inlier_pixels = pixels[distances < 42.0]
    if len(inlier_pixels) < 80:
        return False

    channel_std = np.std(inlier_pixels, axis=0)
    if float(np.mean(channel_std)) > 26.0:
        return False

    edge_density = float(np.mean(edges > 0))
    if edge_density > 0.24:
        return False

    fill_color = np.median(inlier_pixels, axis=0).astype(np.uint8)
    fill_luma = float(np.mean(fill_color))
    inlier_std = float(np.mean(channel_std))
    gray_tone_fill = 95.0 <= fill_luma < 222.0 and inlier_std <= 18.0 and edge_density <= 0.10
    if fill_luma < 222.0 and not gray_tone_fill:
        return False
    if fill_luma > 242.0 and edge_density > 0.075:
        return False
    gray_fill = fill_luma < 235.0
    cleanup = cv2.dilate(
        (mask_roi > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3) if gray_tone_fill else ((5, 5) if gray_fill else (3, 3))),
        iterations=1 if gray_tone_fill else (2 if gray_fill else 1),
    )
    target_roi = target[y1:y2, x1:x2]
    cleanup_bool = cleanup > 0
    if not _apply_local_plane_fill(roi, target_roi, cleanup_bool, background_candidates):
        target_roi[cleanup_bool] = fill_color
    return True


def _fill_dark_background_text_mask(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> bool:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0 or np.count_nonzero(mask_roi > 0) < 6:
        return False

    if _floating_cleanup_should_fail_closed(source, coords, mask_roi):
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    edge_density = float(np.mean(cv2.Canny(gray, 45, 135) > 0))
    channel_std = float(np.mean(np.std(roi.reshape(-1, 3).astype(np.float32), axis=0)))
    dark_fraction = float(np.mean(gray < 150))
    paper_fraction = float(np.mean((gray > 172) & (hsv[:, :, 1] < 115)))
    if dark_fraction < 0.82 or paper_fraction > 0.10:
        return False
    if edge_density > 0.035 or channel_std > 18.0:
        return False

    cleanup = (mask_roi > 0).astype(np.uint8) * 255
    eroded_cleanup = cv2.erode(
        cleanup,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    if np.count_nonzero(eroded_cleanup > 0) >= 6:
        cleanup = eroded_cleanup
    target_roi = target[y1:y2, x1:x2]
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(cleanup, connectivity=8)
    changed = 0
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 3 or area > max(3500, int(cleanup.size * 0.18)):
            continue
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        pad = max(5, min(18, int(max(cw, ch) * 0.35)))
        bx1 = max(0, cx - pad)
        by1 = max(0, cy - pad)
        bx2 = min(roi.shape[1], cx + cw + pad)
        by2 = min(roi.shape[0], cy + ch + pad)
        local_component = labels[by1:by2, bx1:bx2] == label
        local_cleanup = cleanup[by1:by2, bx1:bx2] > 0
        local_gray = gray[by1:by2, bx1:bx2]
        background = (~local_cleanup) & (local_gray < 175)
        if np.count_nonzero(background) < 8:
            continue
        fill_color = np.median(roi[by1:by2, bx1:bx2][background], axis=0).astype(np.uint8)
        view = target_roi[by1:by2, bx1:bx2]
        view[local_component] = fill_color
        changed += int(np.count_nonzero(local_component))

    return changed >= 6


def _fill_component_local_background_text_mask(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> np.ndarray | None:
    """Repair only individual glyph strokes when each glyph sits on flat local art.

    This is intentionally more conservative than the broad floating-text paths:
    it never fills the whole text box and refuses components whose immediate
    neighborhood contains line-art edges or high tone variance.
    """

    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0 or np.count_nonzero(mask_roi > 0) < 6:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 135) > 0
    cleanup = (mask_roi > 0).astype(np.uint8) * 255
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(cleanup, connectivity=8)
    changed_mask = np.zeros_like(cleanup)
    target_roi = target[y1:y2, x1:x2]
    target_patch = target_roi.copy()

    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 3 or area > max(1400, int(cleanup.size * 0.12)):
            continue
        if cw > max(96, int(cleanup.shape[1] * 0.70)):
            continue
        if ch > max(140, int(cleanup.shape[0] * 0.82)):
            continue

        pad = max(4, min(16, int(max(cw, ch) * 0.45)))
        bx1 = max(0, cx - pad)
        by1 = max(0, cy - pad)
        bx2 = min(roi.shape[1], cx + cw + pad)
        by2 = min(roi.shape[0], cy + ch + pad)
        local_label = labels[by1:by2, bx1:bx2] == label
        local_cleanup = cleanup[by1:by2, bx1:bx2] > 0
        local_gray = gray[by1:by2, bx1:bx2]
        local_edges = edges[by1:by2, bx1:bx2]

        ring = cv2.dilate(
            local_label.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ) > 0
        background = ring & (~local_cleanup)
        non_halo_background = background & (local_gray > 42) & (local_gray < 238)
        if np.count_nonzero(non_halo_background) >= 8:
            background = non_halo_background
        if np.count_nonzero(background) < 8:
            background = ~local_cleanup
            non_halo_background = background & (local_gray > 42) & (local_gray < 238)
            if np.count_nonzero(non_halo_background) >= 12:
                background = non_halo_background
        if np.count_nonzero(background) < 12:
            continue

        bg_edge_density = float(np.mean(local_edges[background]))
        bg_luma_std = float(np.std(local_gray[background].astype(np.float32)))
        local_edge_density = float(np.mean(local_edges))
        if bg_edge_density > 0.10 or local_edge_density > 0.22 or bg_luma_std > 32.0:
            continue

        pixels = roi[by1:by2, bx1:bx2][background].astype(np.float32)
        median_color = np.median(pixels, axis=0)
        distances = np.linalg.norm(pixels - median_color, axis=1)
        inliers = pixels[distances < 48.0]
        if len(inliers) < 8 or float(np.mean(np.std(inliers, axis=0))) > 30.0:
            continue

        component_mask = local_label.astype(np.uint8) * 255
        if area >= 12:
            component_mask = cv2.dilate(
                component_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
        view = target_patch[by1:by2, bx1:bx2]
        view[component_mask > 0] = np.median(inliers, axis=0).astype(np.uint8)
        changed_mask[by1:by2, bx1:bx2] = np.maximum(changed_mask[by1:by2, bx1:bx2], component_mask)

    changed_count = int(np.count_nonzero(changed_mask > 0))
    source_coverage = float(np.count_nonzero((changed_mask > 0) & (cleanup > 0))) / float(
        max(1, np.count_nonzero(cleanup > 0))
    )
    if changed_count >= 6 and source_coverage >= 0.78:
        target_roi[:, :] = target_patch
        return changed_mask
    return None


def _opencv_local_stroke_repair(
    target: np.ndarray,
    region_mask: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    radius: float = 2.0,
) -> bool:
    pad = 10
    crop_x1 = max(0, x1 - pad)
    crop_y1 = max(0, y1 - pad)
    crop_x2 = min(target.shape[1], x2 + pad)
    crop_y2 = min(target.shape[0], y2 + pad)
    crop_mask = region_mask[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop_mask.size == 0 or np.count_nonzero(crop_mask > 0) < 6:
        return False

    crop_img = target[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    repair_mask = (crop_mask > 0).astype(np.uint8) * 255
    repaired = cv2.inpaint(crop_img, repair_mask, radius, cv2.INPAINT_TELEA)
    target_view = target[crop_y1:crop_y2, crop_x1:crop_x2]
    target_view[repair_mask > 0] = repaired[repair_mask > 0]
    return True


def _local_repair_tone_match(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    repair_mask: np.ndarray,
) -> None:
    x1, y1, x2, y2 = coords
    source_roi = source[y1:y2, x1:x2]
    target_roi = target[y1:y2, x1:x2]
    if source_roi.size == 0 or target_roi.size == 0 or repair_mask.size == 0:
        return

    repair = (repair_mask > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(repair, connectivity=8)
    for label in range(1, component_count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        ring = cv2.dilate(
            component.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=1,
        ) > 0
        ring &= ~(repair > 0)
        if np.count_nonzero(ring) < 12:
            continue

        ring_pixels = source_roi[ring].astype(np.float32)
        output_pixels = target_roi[component].astype(np.float32)
        if output_pixels.size == 0:
            continue
        ring_median = np.median(ring_pixels, axis=0)
        output_median = np.median(output_pixels, axis=0)
        ring_gray = cv2.cvtColor(ring_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY).reshape(-1)
        ring_luma_std = float(np.std(ring_gray.astype(np.float32)))
        color_delta = ring_median - output_median
        delta_norm = float(np.linalg.norm(color_delta))
        if delta_norm < 34.0:
            continue
        blend_strength = 0.70 if ring_luma_std < 42.0 else 0.48
        corrected = np.clip(output_pixels + color_delta * blend_strength, 0, 255).astype(np.uint8)
        target_roi[component] = corrected


def _restore_source_outside_caption_guard(
    source: np.ndarray,
    target: np.ndarray,
    repair_coords: tuple[int, int, int, int],
    seed_coords: tuple[int, int, int, int],
    seed_roi: np.ndarray,
    repair_mask: np.ndarray,
) -> np.ndarray | None:
    rx1, ry1, rx2, ry2 = repair_coords
    sx1, sy1, sx2, sy2 = seed_coords
    source_roi = source[ry1:ry2, rx1:rx2]
    target_roi = target[ry1:ry2, rx1:rx2]
    if source_roi.size == 0 or target_roi.size == 0 or seed_roi.size == 0 or repair_mask.size == 0:
        return None

    height, width = source_roi.shape[:2]
    repair = repair_mask > 0
    if repair.shape[:2] != (height, width) or int(np.count_nonzero(repair)) < 12:
        return None

    guard = np.zeros((height, width), dtype=np.uint8)
    ox1 = max(0, sx1 - rx1)
    oy1 = max(0, sy1 - ry1)
    ox2 = min(width, sx2 - rx1)
    oy2 = min(height, sy2 - ry1)
    if ox2 <= ox1 or oy2 <= oy1:
        return None

    seed_x1 = max(0, rx1 - sx1)
    seed_y1 = max(0, ry1 - sy1)
    seed_x2 = seed_x1 + (ox2 - ox1)
    seed_y2 = seed_y1 + (oy2 - oy1)
    guard[oy1:oy2, ox1:ox2] = (seed_roi[seed_y1:seed_y2, seed_x1:seed_x2] > 0).astype(np.uint8) * 255
    if int(np.count_nonzero(guard > 0)) < 8:
        return None

    guard_width = max(5, min(17, (int(round(width * 0.16)) | 1)))
    guard_height = max(7, min(23, (int(round(height * 0.055)) | 1)))
    guard = cv2.dilate(
        guard,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (guard_width, guard_height)),
        iterations=1,
    )
    guard = cv2.morphologyEx(
        guard,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    restore = repair & ~(guard > 0)
    restore_count = int(np.count_nonzero(restore))
    if restore_count < max(12, int(np.count_nonzero(repair) * 0.10)):
        return None

    target_roi[restore] = source_roi[restore]
    restored_mask = np.zeros((height, width), dtype=np.uint8)
    restored_mask[restore] = 255
    return restored_mask


def _stroke_only_inpaint_repair(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
    radius: float = 2.0,
    max_seed_density: float = 0.46,
    max_repair_density: float = 0.42,
) -> np.ndarray | None:
    """Remove only detected source glyph strokes, never the whole text box.

    This path is for floating dialogue over character/background art. It is not a
    semantic redraw model; it uses a tight glyph/halo mask, small-radius inpaint,
    and local tone correction so failed masks do not turn into white rectangles.
    """

    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return None

    height, width = roi.shape[:2]
    area = max(1, height * width)
    seed = mask_roi > 0
    seed_count = int(np.count_nonzero(seed))
    if seed_count < 6:
        return None
    seed_density = seed_count / float(area)
    if seed_density > max_seed_density:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    unseed = ~seed
    if np.count_nonzero(unseed) < max(20, int(area * 0.05)):
        return None
    background_luma = float(np.median(gray[unseed]))
    tophat = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    )
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    )
    near_seed = cv2.dilate(
        seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    ) > 0
    bright_cutoff = max(184.0, min(244.0, background_luma + 34.0))
    dark_cutoff = max(32.0, min(118.0, background_luma - 34.0))
    bright_halo = near_seed & (gray > bright_cutoff) & ((gray > 224) | (tophat > 7))
    dark_glyph = near_seed & (gray < dark_cutoff) & ((blackhat > 6) | seed)

    repair = (seed | bright_halo | dark_glyph).astype(np.uint8) * 255
    repair = cv2.morphologyEx(
        repair,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    repair = cv2.dilate(
        repair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats((repair > 0).astype(np.uint8), 8)
    filtered = np.zeros_like(repair)
    for label in range(1, component_count):
        component = labels == label
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if component_area < 3:
            continue
        if component_area > max(4200, int(area * 0.34)):
            continue
        if cw > int(width * 0.94) and ch < max(12, int(height * 0.10)):
            continue
        if ch > int(height * 0.94) and cw < max(8, int(width * 0.08)):
            continue
        seed_overlap = int(np.count_nonzero(component & seed))
        halo_overlap = int(np.count_nonzero(component & bright_halo))
        glyph_overlap = int(np.count_nonzero(component & dark_glyph))
        if seed_overlap < 2 and halo_overlap < 3 and glyph_overlap < 2:
            continue
        filtered[component] = 255

    repair = filtered
    repair_count = int(np.count_nonzero(repair > 0))
    if repair_count < 6:
        return None
    repair_density = repair_count / float(area)
    if repair_density > max_repair_density:
        return None

    pad = max(12, min(34, int(max(width, height) * 0.10)))
    crop_x1 = max(0, x1 - pad)
    crop_y1 = max(0, y1 - pad)
    crop_x2 = min(source.shape[1], x2 + pad)
    crop_y2 = min(source.shape[0], y2 + pad)
    crop_img = target[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    crop_mask = np.zeros(crop_img.shape[:2], dtype=np.uint8)
    crop_mask[y1 - crop_y1:y2 - crop_y1, x1 - crop_x1:x2 - crop_x1] = repair

    try:
        telea = cv2.inpaint(crop_img, crop_mask, radius, cv2.INPAINT_TELEA)
        navier = cv2.inpaint(crop_img, crop_mask, max(1.0, radius * 0.85), cv2.INPAINT_NS)
    except cv2.error:
        return None

    repaired = cv2.addWeighted(telea, 0.72, navier, 0.28, 0)
    target_view = target[crop_y1:crop_y2, crop_x1:crop_x2]
    target_view[crop_mask > 0] = repaired[crop_mask > 0]
    _local_repair_tone_match(source, target, coords, repair)
    return repair


def _fill_paper_tone_repair(source_roi: np.ndarray, target_roi: np.ndarray, repair: np.ndarray) -> bool:
    if source_roi.size == 0 or target_roi.size == 0 or repair.size == 0:
        return False

    repair_bool = repair > 0
    if int(np.count_nonzero(repair_bool)) < 8:
        return False

    height, width = repair.shape[:2]
    area = max(1, height * width)
    gray = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(source_roi, cv2.COLOR_BGR2HSV)
    guard = cv2.dilate(
        repair.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ) > 0
    background = (~guard) & (gray >= 168) & (hsv[:, :, 1] < 130)
    if int(np.count_nonzero(background)) < max(24, int(area * 0.045)):
        background = (~repair_bool) & (gray >= 156) & (hsv[:, :, 1] < 150)
    if int(np.count_nonzero(background)) < max(16, int(area * 0.025)):
        return False

    background_gray = gray[background].astype(np.float32)
    paper_floor = max(156.0, float(np.percentile(background_gray, 28)) - 4.0)
    clean_background = background & (gray.astype(np.float32) >= paper_floor) & (hsv[:, :, 1] < 150)
    if int(np.count_nonzero(clean_background)) >= max(16, int(area * 0.025)):
        background = clean_background

    bg_y, bg_x = np.where(background)
    if bg_x.size < 16:
        return False

    fallback = np.percentile(source_roi[background].astype(np.float32), 68, axis=0)
    row_colors = np.empty((height, 3), dtype=np.float32)
    row_band = max(5, min(24, height // 14))
    fallback_rows = 0
    repair_rows = repair_bool.any(axis=1)
    for row_index in range(height):
        near_row = np.abs(bg_y - row_index) <= row_band
        if int(np.count_nonzero(near_row)) >= 8:
            row_colors[row_index, :] = np.percentile(
                source_roi[bg_y[near_row], bg_x[near_row]].astype(np.float32),
                68,
                axis=0,
            )
        else:
            row_colors[row_index, :] = fallback
            if repair_rows[row_index]:
                fallback_rows += 1
    # When most repaired rows have no local background sample the fill is
    # just the global tone in a box shape — on anything but flat paper that
    # renders as a bright rectangle. Decline and let a real inpainter run.
    repair_row_count = int(np.count_nonzero(repair_rows))
    if repair_row_count and fallback_rows > int(repair_row_count * 0.30):
        return False
    if height >= 7:
        row_colors = cv2.GaussianBlur(
            row_colors.reshape(height, 1, 3),
            (1, 0),
            sigmaX=0,
            sigmaY=max(1.6, min(8.0, height / 32.0)),
        ).reshape(height, 3)

    fitted = np.repeat(row_colors[:, None, :], width, axis=1)
    target_roi[repair_bool] = np.clip(fitted, 0, 255).astype(np.uint8)[repair_bool]
    return True


def _pure_paper_dark_glyph_mask(source_roi: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    if source_roi.size == 0 or seed_mask.size == 0:
        return np.zeros(seed_mask.shape, dtype=np.uint8)

    height, width = seed_mask.shape[:2]
    area = max(1, height * width)
    gray = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(source_roi, cv2.COLOR_BGR2HSV)
    dark = ((gray < 154) & (hsv[:, :, 1] < 235)).astype(np.uint8)
    if int(np.count_nonzero(dark)) < 8:
        return np.zeros(seed_mask.shape, dtype=np.uint8)

    seed_near = cv2.dilate(
        (seed_mask > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29)),
        iterations=1,
    ) > 0
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    keep = np.zeros(seed_mask.shape, dtype=np.uint8)
    for label in range(1, component_count):
        component = labels == label
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        if component_area < 3 or component_area > max(3600, int(area * 0.16)):
            continue
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if cw <= 0 or ch <= 0:
            continue
        fill_ratio = component_area / float(max(1, cw * ch))
        long_axis = max(cw, ch)
        short_axis = max(1, min(cw, ch))
        axis_ratio = long_axis / float(short_axis)
        touches_border = cx <= 1 or cy <= 1 or (cx + cw) >= width - 1 or (cy + ch) >= height - 1
        compact_text_like = (
            component_area <= 1200
            and fill_ratio >= 0.24
            and axis_ratio <= 3.8
            and long_axis <= max(72, int(min(width, height) * 0.34))
        )
        border_art = touches_border and not compact_text_like and (
            component_area > 320
            or long_axis >= max(42, int(min(width, height) * 0.16))
            or (axis_ratio >= 2.2 and fill_ratio <= 0.42)
        )
        horizontal_rule = cw >= int(width * 0.56) and ch <= 8
        vertical_rule = ch >= int(height * 0.66) and cw <= 8
        bracket_like = (
            border_art
            or (touches_border and axis_ratio >= 4.2 and short_axis <= 9)
            or horizontal_rule
            or vertical_rule
            or (axis_ratio >= 9.0 and short_axis <= 6 and fill_ratio <= 0.42)
            or (axis_ratio >= 3.0 and long_axis >= max(26, int(min(width, height) * 0.08)) and fill_ratio <= 0.34)
            or (long_axis >= max(54, int(min(width, height) * 0.20)) and fill_ratio <= 0.22)
        )
        if bracket_like:
            continue
        if int(np.count_nonzero(component & seed_near)) < 2 and component_area < 18:
            continue
        keep[component] = 255
    if int(np.count_nonzero(keep > 0)) < 6:
        return np.zeros(seed_mask.shape, dtype=np.uint8)
    return keep


def _pure_paper_repair_context(
    source: np.ndarray,
    coords: tuple[int, int, int, int],
    repair_mask: np.ndarray,
) -> bool:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or repair_mask.size == 0:
        return False
    repair = repair_mask > 0
    area = max(1, repair_mask.shape[0] * repair_mask.shape[1])
    if int(np.count_nonzero(repair)) < 8:
        return False
    background = cv2.dilate(
        repair.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ) <= 0
    if int(np.count_nonzero(background)) < max(32, int(area * 0.06)):
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    background_gray = gray[background].astype(np.float32)
    background_sat = hsv[:, :, 1][background]
    if not (
        float(np.mean(background_gray > 230)) >= 0.78
        and float(np.mean((background_gray >= 82) & (background_gray <= 220))) <= 0.18
        and float(np.mean(background_gray < 82)) <= 0.12
        and float(np.percentile(background_sat, 80)) <= 90.0
        and float(np.mean((cv2.Canny(gray, 45, 135) > 0)[background])) <= 0.12
    ):
        return False
    # Pure paper is FLAT paper. A smooth wall gradient can sneak past the
    # brightness gate when the sampled fringe is dominated by its bright end;
    # a flat-tone fill then leaves a visible bright rectangle (new_sample_1).
    # Require the background tone to be consistent from top to bottom and the
    # background samples to actually span the window's rows.
    row_has_bg = background.any(axis=1)
    row_indices = np.where(row_has_bg)[0]
    if row_indices.size >= 8:
        third = max(2, row_indices.size // 3)
        top_sel = np.zeros_like(background)
        top_sel[row_indices[:third], :] = True
        bot_sel = np.zeros_like(background)
        bot_sel[row_indices[-third:], :] = True
        top_med = float(np.median(gray[background & top_sel].astype(np.float32)))
        bot_med = float(np.median(gray[background & bot_sel].astype(np.float32)))
        if abs(top_med - bot_med) > 7.0:
            return False
    if float(np.mean(row_has_bg)) < 0.45:
        return False
    # The non-repair pixels can be the lettering's own white halo rather than
    # the true surroundings: halo'd floating text on a gray wall measured 255
    # "background" while the wall itself sat at ~165, and the halo reached the
    # window border, so the fill stamped a white rectangle (new_sample_1).
    # Pure paper must continue OUTSIDE the window: sample an outer ring.
    img_h, img_w = source.shape[:2]
    ring_pad = 12
    ox1, oy1 = max(0, x1 - ring_pad), max(0, y1 - ring_pad)
    ox2, oy2 = min(img_w, x2 + ring_pad), min(img_h, y2 + ring_pad)
    outer = source[oy1:oy2, ox1:ox2]
    outer_gray = cv2.cvtColor(outer, cv2.COLOR_BGR2GRAY)
    ring = np.ones(outer_gray.shape, dtype=bool)
    ring[y1 - oy1:y2 - oy1, x1 - ox1:x2 - ox1] = False
    if int(np.count_nonzero(ring)) >= 48:
        ring_bright = float(np.percentile(outer_gray[ring].astype(np.float32), 55))
        if ring_bright < 220.0:
            return False
    return True


def _sparse_source_line_art_preserve_mask(source_roi: np.ndarray, repair_mask: np.ndarray) -> np.ndarray | None:
    if source_roi.size == 0 or repair_mask.size == 0:
        return None

    repair = repair_mask > 0
    repair_count = int(np.count_nonzero(repair))
    if repair_count < 16:
        return None

    height, width = repair_mask.shape[:2]
    area = max(1, height * width)
    if height < 32 or width < 32:
        return None

    gray = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(source_roi, cv2.COLOR_BGR2HSV)
    low_saturation = float(np.mean(hsv[:, :, 1] < 170))
    bright_fraction = float(np.mean(gray > 205))
    edge_density = float(np.mean(cv2.Canny(gray, 45, 135) > 0))
    if low_saturation < 0.78 or bright_fraction < 0.42 or edge_density < 0.022:
        return None

    edges = cv2.Canny(gray, 35, 130)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=24,
        minLineLength=30,
        maxLineGap=8,
    )
    if lines is None:
        return None

    preserve = np.zeros((height, width), dtype=np.uint8)
    for line in lines[:, 0, :]:
        lx1, ly1, lx2, ly2 = [int(value) for value in line]
        length = math.hypot(lx2 - lx1, ly2 - ly1)
        if length < 34:
            continue
        angle = abs(math.degrees(math.atan2(ly2 - ly1, lx2 - lx1)))
        if angle > 90.0:
            angle = 180.0 - angle
        if angle < 8.0 or angle > 82.0:
            continue

        line_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.line(line_mask, (lx1, ly1), (lx2, ly2), 255, 1)
        line_bool = line_mask > 0
        if int(np.count_nonzero(line_bool & repair)) < 1:
            continue

        neighborhood = cv2.dilate(
            line_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        ) > 0
        values = gray[neighborhood]
        if values.size < 8:
            continue
        dark_density = float(np.mean(values < 80))
        bright_density = float(np.mean(values > 220))
        sparse_art_line = (
            (bright_density >= 0.58 and dark_density <= 0.24)
            or (length >= 70.0 and bright_density >= 0.52 and dark_density <= 0.20)
        )
        if not sparse_art_line:
            continue
        preserve[line_bool & repair] = 255

    preserve_count = int(np.count_nonzero(preserve > 0))
    if preserve_count < 4:
        return None
    if preserve_count / float(repair_count) > 0.34 or preserve_count / float(area) > 0.055:
        return None
    preserve = cv2.dilate(
        preserve,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    preserve[~repair] = 0
    if int(np.count_nonzero(preserve > 0)) < 4:
        return None
    return preserve


def _pure_paper_source_inpaint(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return None

    repair = (mask_roi > 0).astype(np.uint8) * 255
    if np.count_nonzero(repair > 0) < 8:
        return None

    line_preserve = _sparse_source_line_art_preserve_mask(roi, repair)
    if line_preserve is not None:
        repair[line_preserve > 0] = 0
        if int(np.count_nonzero(repair > 0)) < 8:
            return None

    height, width = repair.shape[:2]
    area = max(1, height * width)
    density = float(np.count_nonzero(repair > 0)) / float(area)
    if density < 0.015 or density > 0.62:
        return None

    repair_background = cv2.dilate(
        repair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ) <= 0
    if np.count_nonzero(repair_background) < max(32, int(area * 0.08)):
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    background_gray = gray[repair_background].astype(np.float32)
    background_dark_fraction = float(np.mean(background_gray < 150))
    bright_fraction = float(np.mean(background_gray > 232))
    mid_fraction = float(np.mean((background_gray >= 82) & (background_gray <= 220)))
    dark_fraction = float(np.mean(background_gray < 82))
    saturation_p80 = float(np.percentile(hsv[:, :, 1][repair_background], 80))
    edge_map = cv2.Canny(gray, 45, 135) > 0
    edge_density = float(np.mean(edge_map[repair_background]))
    luma_std = float(np.std(background_gray))
    full_edge_density = float(np.mean(edge_map))
    full_dark_fraction = float(np.mean(gray < 82))
    full_mid_fraction = float(np.mean((gray >= 82) & (gray <= 220)))
    repair_bool = repair > 0
    repair_edge_density = float(np.mean(edge_map[repair_bool])) if np.any(repair_bool) else 0.0
    if background_dark_fraction >= 0.18 and bright_fraction <= 0.72:
        return None
    line_art_hazard = (
        density >= 0.16
        and full_edge_density >= 0.074
        and repair_edge_density >= 0.11
        and (full_dark_fraction >= 0.055 or full_mid_fraction >= 0.045)
    )
    if line_art_hazard:
        glyph_mask = _pure_paper_dark_glyph_mask(roi, repair)
        glyph_count = int(np.count_nonzero(glyph_mask > 0))
        glyph_density = glyph_count / float(area)
        if glyph_count < 8 or glyph_density > 0.24:
            return None
        cleanup = cv2.dilate(
            glyph_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        target_roi = target[y1:y2, x1:x2]
        try:
            telea = cv2.inpaint(target_roi, cleanup, 1.15, cv2.INPAINT_TELEA)
            navier = cv2.inpaint(target_roi, cleanup, 0.9, cv2.INPAINT_NS)
        except cv2.error:
            return None
        repaired = cv2.addWeighted(telea, 0.86, navier, 0.14, 0)
        target_roi[cleanup > 0] = repaired[cleanup > 0]
        restore_mask = _restore_art_lines_crossing_repair_mask(source, target, coords, cleanup, repair)
        if restore_mask is not None:
            cleanup = cv2.bitwise_or(cleanup, restore_mask)
        return cleanup

    pure_paper = (
        bright_fraction >= 0.88
        and mid_fraction <= 0.13
        and dark_fraction <= 0.10
        and saturation_p80 <= 72.0
        and edge_density <= 0.105
        and luma_std <= 46.0
    )
    if not pure_paper:
        return None

    dark_glyph_mask = _pure_paper_dark_glyph_mask(roi, repair)
    if int(np.count_nonzero(dark_glyph_mask > 0)) >= 6:
        repair = cv2.bitwise_or(repair, dark_glyph_mask)
        repair = cv2.morphologyEx(
            repair,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

    target_roi = target[y1:y2, x1:x2]
    cleanup = repair
    if density >= 0.18:
        cleanup = cv2.dilate(
            cleanup,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
    if density >= 0.42:
        cleanup = cv2.morphologyEx(
            cleanup,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
    if _fill_paper_tone_repair(roi, target_roi, cleanup):
        return cleanup

    try:
        telea = cv2.inpaint(target_roi, repair, 1.25, cv2.INPAINT_TELEA)
        navier = cv2.inpaint(target_roi, repair, 1.0, cv2.INPAINT_NS)
    except cv2.error:
        return None

    repaired = cv2.addWeighted(telea, 0.82, navier, 0.18, 0)
    target_roi[repair > 0] = repaired[repair > 0]
    return repair


def _seeded_smooth_panel_outline_mask(
    source: np.ndarray,
    coords: tuple[int, int, int, int],
    seed_roi: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or seed_roi.size == 0:
        return None

    height, width = seed_roi.shape[:2]
    area = max(1, height * width)
    if area < 700:
        return None

    seed = seed_roi > 0
    seed_count = int(np.count_nonzero(seed))
    if seed_count < 6:
        return None
    seed_density = seed_count / float(area)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    seed_context = cv2.dilate(
        seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ) > 0
    background = (
        ~seed_context
        & (gray >= 70)
        & (gray <= 232)
        & (hsv[:, :, 1] < 160)
    )
    if np.count_nonzero(background) < max(24, int(area * 0.035)):
        return None

    background_values = gray[background].astype(np.float32)
    background_median = float(np.median(background_values))
    background_std = float(np.std(background_values))
    background_edges = float(np.mean((cv2.Canny(gray, 45, 135) > 0)[background]))
    background_bright = float(np.mean(background_values > 232))
    background_dark = float(np.mean(background_values < 70))
    if (
        not (82.0 <= background_median <= 238.0)
        or background_std > 62.0
        or background_edges > 0.16
        or background_bright > 0.72
        or background_dark > 0.18
    ):
        return None
    if background_median > 224.0 and seed_density < 0.32:
        return None

    near_seed = cv2.dilate(
        seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
        iterations=1,
    ) > 0
    bright_outline = (
        near_seed
        & (gray >= max(188, int(background_median + 16)))
        & (hsv[:, :, 1] < 185)
    )
    dark_core = (
        near_seed
        & (gray <= min(156, int(background_median - 24)))
    )
    repair = (seed | bright_outline | dark_core).astype(np.uint8) * 255
    repair = cv2.morphologyEx(
        repair,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    repair = cv2.dilate(
        repair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    repair_count = int(np.count_nonzero(repair > 0))
    repair_density = repair_count / float(area)
    if repair_count < max(10, int(seed_count * 1.12)):
        return None
    if repair_density < 0.025 or repair_density > 0.94:
        return None
    return repair


def _apply_soft_repair_fill(
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    repair_mask: np.ndarray,
    fitted_roi: np.ndarray,
    seed_roi: np.ndarray,
    source_roi: np.ndarray | None = None,
) -> None:
    x1, y1, x2, y2 = coords
    target_roi = target[y1:y2, x1:x2]
    repair = repair_mask > 0
    if target_roi.size == 0 or fitted_roi.size == 0 or int(np.count_nonzero(repair)) < 2:
        return

    height, width = repair.shape[:2]
    density = float(np.count_nonzero(repair)) / float(max(1, height * width))
    if density < 0.42:
        target_roi[repair] = fitted_roi[repair]
        return

    distance = cv2.distanceTransform(repair.astype(np.uint8), cv2.DIST_L2, 3)
    feather = max(5.0, min(20.0, max(width, height) * 0.075))
    alpha = np.clip(distance / feather, 0.0, 1.0)
    seed_core = cv2.dilate(
        (seed_roi > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
        iterations=1,
    ) > 0
    alpha[seed_core & repair] = 1.0
    if source_roi is not None and source_roi.size:
        source_gray = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
        fitted_gray = cv2.cvtColor(fitted_roi, cv2.COLOR_BGR2GRAY)
        fitted_median = float(np.median(fitted_gray[repair])) if np.count_nonzero(repair) else float(np.median(fitted_gray))
        source_text_like = repair & (
            (source_gray >= max(210, int(fitted_median + 24)))
            | (source_gray <= max(0, int(fitted_median - 42)))
        )
        source_text_like = cv2.dilate(
            source_text_like.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ) > 0
        alpha[source_text_like & repair] = 1.0
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 1.0)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    blended = (
        fitted_roi.astype(np.float32) * alpha
        + target_roi.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)
    target_roi[repair] = blended[repair]


def _smooth_panel_gradient_repair(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    repair_mask: np.ndarray,
    seed_roi: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or repair_mask.size == 0:
        return None

    height, width = repair_mask.shape[:2]
    area = max(1, height * width)
    repair = (repair_mask > 0).astype(np.uint8) * 255
    repair_density = float(np.count_nonzero(repair > 0)) / float(area)
    if repair_density < 0.36:
        return None

    img_h, img_w = source.shape[:2]
    pad = max(36, min(96, int(max(width, height) * 0.34)))
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(img_w, x2 + pad)
    cy2 = min(img_h, y2 + pad)
    crop = source[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    crop_edges = cv2.dilate(
        (cv2.Canny(crop_gray, 45, 135) > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ) > 0
    crop_h, crop_w = crop_gray.shape[:2]
    box = np.zeros((crop_h, crop_w), dtype=bool)
    box[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1] = True
    seed_crop = np.zeros((crop_h, crop_w), dtype=bool)
    seed_crop[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1] = cv2.dilate(
        repair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ) > 0
    candidates = (
        ~box
        & ~seed_crop
        & ~crop_edges
        & (crop_hsv[:, :, 1] < 135)
        & (crop_gray >= 72)
        & (crop_gray <= 232)
    )
    if int(np.count_nonzero(candidates)) < max(60, int(area * 0.025)):
        return None

    candidate_values = crop_gray[candidates].astype(np.float32)
    median = float(np.median(candidate_values))
    if not (92.0 <= median <= 224.0):
        return None
    spread = float(np.percentile(candidate_values, 85) - np.percentile(candidate_values, 15))
    if spread > 82.0:
        keep = candidates & (crop_gray >= int(median - 46)) & (crop_gray <= int(median + 46))
        if int(np.count_nonzero(keep)) >= max(60, int(area * 0.02)):
            candidates = keep
            candidate_values = crop_gray[candidates].astype(np.float32)
            median = float(np.median(candidate_values))

    edge_density = float(np.mean(crop_edges[candidates]))
    bright_fraction = float(np.mean(candidate_values > 222))
    dark_fraction = float(np.mean(candidate_values < 88))
    if edge_density > 0.025 or bright_fraction > 0.55 or dark_fraction > 0.30:
        return None

    yy, xx = np.where(candidates)
    if len(xx) > 7000:
        step = max(1, len(xx) // 7000)
        xx = xx[::step]
        yy = yy[::step]
    design = np.column_stack(
        [
            xx.astype(np.float32),
            yy.astype(np.float32),
            np.ones_like(xx, dtype=np.float32),
        ]
    )
    fitted_crop = np.empty_like(crop, dtype=np.float32)
    grid_y, grid_x = np.indices((crop_h, crop_w), dtype=np.float32)
    grid_design = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)
    for channel in range(3):
        values = crop[yy, xx, channel].astype(np.float32)
        coeffs, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
        fitted_crop[:, :, channel] = (
            grid_design[:, :, 0] * coeffs[0]
            + grid_design[:, :, 1] * coeffs[1]
            + coeffs[2]
        )
    fitted_crop = cv2.GaussianBlur(
        np.clip(fitted_crop, 0, 255).astype(np.uint8),
        (0, 0),
        1.15,
    )
    fitted_roi = fitted_crop[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1]
    _apply_soft_repair_fill(target, coords, repair, fitted_roi, seed_roi, roi)

    target_roi = target[y1:y2, x1:x2]
    source_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
    local_blur = cv2.GaussianBlur(target_gray, (0, 0), 4)
    seed_near = cv2.dilate(
        (seed_roi > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        iterations=1,
    ) > 0
    ghost = (
        seed_near
        & (repair > 0)
        & (source_gray > int(median + 20))
        & (target_gray > local_blur + 5)
    )
    ghost = cv2.morphologyEx(
        ghost.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    if 4 <= int(np.count_nonzero(ghost > 0)) <= int(area * 0.24):
        target_roi[ghost > 0] = fitted_roi[ghost > 0]
        repair = cv2.bitwise_or(repair, ghost)
    line_restore = _restore_art_lines_crossing_repair_mask(
        source,
        target,
        coords,
        repair,
        seed_roi,
    )
    if line_restore is not None:
        repair = cv2.bitwise_or(repair, line_restore)
    return repair


def _cleanup_adjacent_smooth_panel_text(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    seed_mask: np.ndarray,
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    x1, y1, x2, y2 = coords
    if seed_mask.size == 0:
        return None

    height, width = seed_mask.shape[:2]
    area = max(1, height * width)
    seed_density = float(np.count_nonzero(seed_mask > 0)) / float(area)
    if seed_density < 0.24:
        return None

    source_roi = source[y1:y2, x1:x2]
    if source_roi.size:
        source_gray = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
        source_hsv = cv2.cvtColor(source_roi, cv2.COLOR_BGR2HSV)
        source_seed_guard = cv2.dilate(
            (seed_mask > 0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        ).astype(bool)
        local_background = ~source_seed_guard
        if int(np.count_nonzero(local_background)) >= max(24, int(area * 0.035)):
            background_gray = source_gray[local_background].astype(np.float32)
            background_sat = source_hsv[:, :, 1][local_background]
            paper_fraction = float(
                np.mean((background_gray >= 232) & (background_sat < 90))
            )
            paper_edge_density = float(
                np.mean((cv2.Canny(source_gray, 45, 135) > 0)[local_background])
            )
            if (
                float(np.median(background_gray)) >= 238.0
                and paper_fraction >= 0.82
                and paper_edge_density <= 0.035
            ):
                return None

    img_h, img_w = source.shape[:2]
    pad = max(24, min(74, int(max(width, height) * 0.20)))
    ex1 = max(0, x1 - pad)
    ey1 = max(0, y1 - pad)
    ex2 = min(img_w, x2 + pad)
    ey2 = min(img_h, y2 + pad)
    if ex2 <= ex1 or ey2 <= ey1:
        return None

    roi = source[ey1:ey2, ex1:ex2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    edges = cv2.dilate(
        (cv2.Canny(gray, 45, 135) > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ) > 0
    roi_h, roi_w = gray.shape[:2]
    placed_seed = np.zeros((roi_h, roi_w), dtype=np.uint8)
    placed_seed[y1 - ey1:y2 - ey1, x1 - ex1:x2 - ex1] = (seed_mask > 0).astype(np.uint8) * 255
    seed_guard = cv2.dilate(
        placed_seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    ) > 0
    background = (
        ~seed_guard
        & ~edges
        & (hsv[:, :, 1] < 135)
        & (gray >= 72)
        & (gray <= 232)
    )
    if int(np.count_nonzero(background)) < max(80, int(area * 0.035)):
        return None
    background_values = gray[background].astype(np.float32)
    median = float(np.median(background_values))
    if not (92.0 <= median <= 224.0):
        return None
    if float(np.mean(background_values > 222)) > 0.58 or float(np.mean(background_values < 88)) > 0.28:
        return None

    focus = cv2.dilate(
        placed_seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (43, 43)),
        iterations=1,
    ) > 0
    bright_text = (
        focus
        & (gray >= max(205, int(median + 22)))
        & (hsv[:, :, 1] < 170)
    )
    dark_near_text = (
        cv2.dilate(
            (bright_text | (placed_seed > 0)).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        ) > 0
    ) & (gray <= max(42, int(median - 48)))
    candidate = bright_text | dark_near_text | (placed_seed > 0)
    candidate = cv2.morphologyEx(
        candidate.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    candidate = cv2.dilate(
        candidate,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats((candidate > 0).astype(np.uint8), connectivity=8)
    filtered = np.zeros_like(candidate)
    seed_component_guard = cv2.dilate(
        placed_seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27)),
        iterations=1,
    ) > 0
    for label in range(1, component_count):
        component = labels == label
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        if component_area < 5 or component_area > max(22000, int(candidate.size * 0.42)):
            continue
        overlaps_seed = int(np.count_nonzero(component & seed_component_guard)) >= 2
        overlaps_bright = int(np.count_nonzero(component & bright_text)) >= 2
        if overlaps_seed or overlaps_bright:
            filtered[component] = 255

    filtered_count = int(np.count_nonzero(filtered > 0))
    if filtered_count < int(np.count_nonzero(placed_seed > 0) * 1.02):
        return None
    if filtered_count / float(max(1, filtered.size)) > 0.48:
        return None

    cleaned = _smooth_panel_gradient_repair(
        source,
        target,
        (ex1, ey1, ex2, ey2),
        filtered,
        placed_seed,
    )
    if cleaned is None:
        return None
    return (ex1, ey1, ex2, ey2), cleaned


def _smooth_panel_box_gradient_cleanup(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return None

    width = x2 - x1
    height = y2 - y1
    area = max(1, width * height)
    if width < 44 or height < 90 or area < 4200:
        return None

    roi = source[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_edges = cv2.Canny(roi_gray, 45, 135) > 0
    roi_edge_density = float(np.mean(roi_edges))
    roi_low_sat = float(np.mean(roi_hsv[:, :, 1] < 150))
    roi_mid_tone_fraction = float(np.mean((roi_gray >= 70) & (roi_gray <= 225) & (roi_hsv[:, :, 1] < 160)))
    roi_dark_fraction = float(np.mean(roi_gray < 112))
    roi_bright_fraction = float(np.mean(roi_gray > 235))
    if (
        roi_bright_fraction >= 0.42
        and roi_dark_fraction >= 0.08
        and roi_edge_density >= 0.075
        and roi_low_sat >= 0.92
    ):
        text_like = (
            ((roi_gray > 235) | (roi_gray < 112))
            & (roi_hsv[:, :, 1] < 185)
        ).astype(np.uint8)
        text_guard = cv2.dilate(
            text_like,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        ).astype(bool)
        plane_background = (
            (~text_guard)
            & (roi_hsv[:, :, 1] < 170)
            & (roi_gray >= 58)
            & (roi_gray <= 232)
        )
        plane_background_count = int(np.count_nonzero(plane_background))
        plane_background_ratio = plane_background_count / float(area)
        plane_background_edge = (
            float(np.mean(roi_edges[plane_background]))
            if plane_background_count >= 16
            else 1.0
        )
        if not (
            plane_background_ratio >= 0.045
            and plane_background_edge <= 0.055
            and roi_mid_tone_fraction >= 0.12
        ):
            return None
    if _floating_region_has_mixed_character_tone(source, coords, np.ones_like(roi_gray, dtype=np.uint8) * 255):
        return None
    if roi_low_sat < 0.88 or roi_edge_density > 0.24:
        return None
    if float(np.mean(roi_gray < 92)) > 0.30 and float(np.mean(roi_gray > 218)) > 0.08:
        return None
    if float(np.median(roi_gray)) >= 220.0 and roi_edge_density > 0.035 and roi_mid_tone_fraction < 0.16:
        return None

    img_h, img_w = source.shape[:2]
    pad = max(56, min(170, int(max(width, height) * 0.38)))
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(img_w, x2 + pad)
    cy2 = min(img_h, y2 + pad)
    crop = source[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    crop_edges = cv2.dilate(
        (cv2.Canny(crop_gray, 45, 135) > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ).astype(bool)
    crop_h, crop_w = crop_gray.shape[:2]
    inner = np.zeros((crop_h, crop_w), dtype=np.uint8)
    inner[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1] = 255
    inner_guard = cv2.dilate(
        inner,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=1,
    ).astype(bool)
    background = (
        ~inner_guard
        & ~crop_edges
        & (crop_hsv[:, :, 1] < 145)
        & (crop_gray >= 58)
        & (crop_gray <= 238)
    )
    background_count = int(np.count_nonzero(background))
    if background_count < max(140, int(crop_h * crop_w * 0.018)):
        return None

    background_values = crop_gray[background].astype(np.float32)
    background_median = float(np.median(background_values))
    background_std = float(np.std(background_values))
    if (
        not (82.0 <= background_median <= 218.0)
        or background_std > 62.0
        or float(np.mean(background_values < 70)) > 0.18
        or float(np.mean(background_values > 236)) > 0.36
    ):
        return None

    fallback_color = np.median(crop[background], axis=0).astype(np.float32)
    background_y, background_x = np.where(background)
    row_colors = np.empty((crop_h, 3), dtype=np.float32)
    row_band = max(10, min(38, crop_h // 12))
    for row_index in range(crop_h):
        near_row = np.abs(background_y - row_index) <= row_band
        if int(np.count_nonzero(near_row)) >= 20:
            row_colors[row_index, :] = np.median(
                crop[background_y[near_row], background_x[near_row]],
                axis=0,
            ).astype(np.float32)
        else:
            row_colors[row_index, :] = fallback_color
    row_colors = cv2.GaussianBlur(
        row_colors.reshape(crop_h, 1, 3),
        (1, 0),
        sigmaX=0,
        sigmaY=max(2.5, min(12.0, crop_h / 28.0)),
    ).reshape(crop_h, 3)

    fitted_crop = np.empty_like(crop, dtype=np.float32)
    yy, xx = np.where(background)
    if len(xx) > 9000:
        step = max(1, len(xx) // 9000)
        xx = xx[::step]
        yy = yy[::step]
    design = np.column_stack(
        [
            xx.astype(np.float32),
            yy.astype(np.float32),
            np.ones_like(xx, dtype=np.float32),
        ]
    )
    grid_y, grid_x = np.indices((crop_h, crop_w), dtype=np.float32)
    grid_design = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)
    for channel_idx in range(3):
        values = crop[yy, xx, channel_idx].astype(np.float32)
        coeffs, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
        plane = (
            grid_design[:, :, 0] * coeffs[0]
            + grid_design[:, :, 1] * coeffs[1]
            + coeffs[2]
        )
        row_fit = np.repeat(row_colors[:, channel_idx][:, None], crop_w, axis=1)
        fitted_crop[:, :, channel_idx] = plane * 0.68 + row_fit * 0.32
    fitted_crop = cv2.GaussianBlur(
        np.clip(fitted_crop, 0, 255).astype(np.uint8),
        (0, 0),
        0.9,
    )

    target_roi = target[y1:y2, x1:x2]
    fitted = fitted_crop[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1]

    baseline_candidates = (
        (roi_hsv[:, :, 1] < 170)
        & (roi_gray >= 72)
        & (roi_gray <= 238)
        & ~roi_edges
    )
    fallback_baseline = float(np.median(roi_gray[baseline_candidates])) if np.count_nonzero(baseline_candidates) else float(np.median(roi_gray))
    row_baseline = np.empty(height, dtype=np.float32)
    row_band = max(4, min(18, height // 18))
    candidate_y, candidate_x = np.where(baseline_candidates)
    for row_index in range(height):
        near_row = np.abs(candidate_y - row_index) <= row_band
        if int(np.count_nonzero(near_row)) >= 10:
            row_baseline[row_index] = float(np.median(roi_gray[candidate_y[near_row], candidate_x[near_row]]))
        else:
            row_baseline[row_index] = fallback_baseline
    row_baseline = cv2.GaussianBlur(
        row_baseline.reshape(height, 1),
        (1, 0),
        sigmaX=0,
        sigmaY=max(1.5, min(7.5, height / 32.0)),
    ).reshape(height)
    bright_floor = np.maximum(208.0, row_baseline[:, None] + 28.0)
    bright_outline = (roi_gray.astype(np.float32) > bright_floor) & (roi_hsv[:, :, 1] < 190)
    bright_guard = cv2.dilate(
        bright_outline.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ).astype(bool)
    dark = (roi_gray < 98) & (roi_hsv[:, :, 1] < 230)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark.astype(np.uint8),
        connectivity=8,
    )
    smooth_gray_text_panel = (
        float(np.median(roi_gray)) < 218.0
        and roi_mid_tone_fraction >= 0.20
        and roi_edge_density <= 0.18
    )
    art_restore = np.zeros((height, width), dtype=bool)
    text_dark = np.zeros((height, width), dtype=bool)
    for label in range(1, component_count):
        component = labels == label
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        if component_area < 3:
            continue
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        halo_ratio = float(np.count_nonzero(component & bright_guard)) / float(component_area)
        touches_border = (
            bool(np.any(component[0, :]))
            or bool(np.any(component[-1, :]))
            or bool(np.any(component[:, 0]))
            or bool(np.any(component[:, -1]))
        )
        elongated = (
            max(component_width, component_height) >= 14
            and max(component_width, component_height)
            >= max(1, min(component_width, component_height)) * 1.7
        )
        halo_text_component = (
            halo_ratio >= (0.32 if smooth_gray_text_panel else 0.48)
            and component_area <= max(1200, int(area * 0.24))
            and component_width <= max(28, int(width * 0.82))
            and component_height <= max(28, int(height * 0.82))
        )
        elongated_art_threshold = 0.10 if smooth_gray_text_panel else 0.38
        elongated_art = (
            elongated
            and halo_ratio < elongated_art_threshold
            and component_area >= max(18, int(area * 0.0025))
        )
        if (touches_border and not halo_text_component) or elongated_art:
            art_restore |= component
        else:
            text_dark |= component

    text_seed = bright_outline | text_dark
    text_seed[art_restore] = False
    if int(np.count_nonzero(text_seed)) < max(24, int(area * 0.012)):
        return None

    text_guard = cv2.dilate(
        text_seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=1,
    ).astype(bool)
    local_upper_luma = 255 if float(np.median(roi_gray)) >= 218.0 else 242
    local_background = (
        ~text_guard
        & ~art_restore
        & ~roi_edges
        & (roi_hsv[:, :, 1] < 145)
        & (roi_gray >= 44)
        & (roi_gray <= local_upper_luma)
    )
    local_background_count = int(np.count_nonzero(local_background))
    if local_background_count >= max(180, int(area * 0.16)):
        local_y, local_x = np.where(local_background)
        if len(local_x) > 6000:
            step = max(1, len(local_x) // 6000)
            local_x = local_x[::step]
            local_y = local_y[::step]
        local_design = np.column_stack(
            [
                local_x.astype(np.float32),
                local_y.astype(np.float32),
                np.ones_like(local_x, dtype=np.float32),
            ]
        )
        local_grid_y, local_grid_x = np.indices((height, width), dtype=np.float32)
        local_grid_design = np.stack(
            [local_grid_x, local_grid_y, np.ones_like(local_grid_x)],
            axis=-1,
        )
        local_fit = np.empty_like(fitted, dtype=np.float32)
        row_colors_local = np.empty((height, 3), dtype=np.float32)
        local_band = max(8, min(32, height // 10))
        for row_index in range(height):
            near_row = np.abs(local_y - row_index) <= local_band
            if int(np.count_nonzero(near_row)) >= 12:
                row_colors_local[row_index, :] = np.median(
                    roi[local_y[near_row], local_x[near_row]],
                    axis=0,
                ).astype(np.float32)
            else:
                row_colors_local[row_index, :] = np.median(
                    roi[local_background],
                    axis=0,
                ).astype(np.float32)
        row_colors_local = cv2.GaussianBlur(
            row_colors_local.reshape(height, 1, 3),
            (1, 0),
            sigmaX=0,
            sigmaY=max(1.8, min(8.5, height / 30.0)),
        ).reshape(height, 3)
        for channel_idx in range(3):
            values = roi[local_y, local_x, channel_idx].astype(np.float32)
            coeffs, _, _, _ = np.linalg.lstsq(local_design, values, rcond=None)
            plane = (
                local_grid_design[:, :, 0] * coeffs[0]
                + local_grid_design[:, :, 1] * coeffs[1]
                + coeffs[2]
            )
            row_fit = np.repeat(row_colors_local[:, channel_idx][:, None], width, axis=1)
            local_fit[:, :, channel_idx] = plane * 0.42 + row_fit * 0.58
        fitted = cv2.GaussianBlur(
            np.clip(local_fit, 0, 255).astype(np.uint8),
            (0, 0),
            0.75,
        )

    gray_smooth_panel = float(np.median(roi_gray)) < 218.0
    if gray_smooth_panel:
        texture_background = local_background.copy()
        if int(np.count_nonzero(texture_background)) < max(120, int(area * 0.055)):
            texture_background = (
                baseline_candidates
                & ~text_guard
                & ~art_restore
                & (roi_hsv[:, :, 1] < 170)
                & (roi_gray >= 54)
                & (roi_gray <= 232)
            )
        texture_count = int(np.count_nonzero(texture_background))
        if texture_count >= max(90, int(area * 0.035)):
            texture_values = roi_gray[texture_background].astype(np.float32)
            texture_std = float(np.std(texture_values))
            texture_edge_density = float(np.mean(roi_edges[texture_background]))
            background_y, background_x = np.where(texture_background)
            rng_seed = (
                (int(x1) + 1) * 73856093
                ^ (int(y1) + 1) * 19349663
                ^ (int(width) + 1) * 83492791
                ^ (int(height) + 1) * 2654435761
            ) & 0xFFFFFFFF
            rng = np.random.default_rng(rng_seed)
            sampled_texture = np.empty_like(fitted, dtype=np.uint8)
            texture_band = max(8, min(28, height // 5))
            all_indices = np.arange(texture_count)
            for row_index in range(height):
                near_row = np.where(np.abs(background_y - row_index) <= texture_band)[0]
                if len(near_row) < max(10, width // 5):
                    near_row = all_indices
                chosen = rng.choice(near_row, size=width, replace=True)
                sampled_texture[row_index, :, :] = roi[
                    background_y[chosen],
                    background_x[chosen],
                ]
            sampled_texture = cv2.GaussianBlur(sampled_texture, (3, 3), 0)
            texture_weight = 0.20 if texture_std <= 34.0 and texture_edge_density <= 0.050 else 0.42
            fitted = (
                sampled_texture.astype(np.float32) * texture_weight
                + fitted.astype(np.float32) * (1.0 - texture_weight)
            ).astype(np.uint8)
    halo_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            max(7, min(19 if gray_smooth_panel else 13, int(round(width * (0.082 if gray_smooth_panel else 0.055))) | 1)),
            max(7, min(17 if gray_smooth_panel else 13, int(round(height * (0.038 if gray_smooth_panel else 0.026))) | 1)),
        ),
    )
    ink_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    repair_core = cv2.bitwise_or(
        cv2.dilate(bright_outline.astype(np.uint8), halo_kernel, iterations=1),
        cv2.dilate(text_dark.astype(np.uint8), ink_kernel, iterations=1),
    ).astype(bool)
    repair_core[art_restore] = False
    if gray_smooth_panel:
        residual_search = cv2.dilate(
            text_seed.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)),
            iterations=1,
        ).astype(bool)
        residual_floor = np.maximum(188.0, row_baseline[:, None] + 14.0)
        residual_halo_seed = (
            residual_search
            & ~art_restore
            & (roi_hsv[:, :, 1] < 190)
            & (roi_gray.astype(np.float32) >= residual_floor)
        )
        residual_components = np.zeros((height, width), dtype=bool)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            residual_halo_seed.astype(np.uint8),
            connectivity=8,
        )
        seed_guard = cv2.dilate(
            text_seed.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
            iterations=1,
        ).astype(bool)
        for label in range(1, component_count):
            component = labels == label
            component_area = int(stats[label, cv2.CC_STAT_AREA])
            if component_area < 4 or component_area > max(9000, int(area * 0.18)):
                continue
            component_width = int(stats[label, cv2.CC_STAT_WIDTH])
            component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
            if component_width > int(width * 0.72) and component_height < 8:
                continue
            if component_height > int(height * 0.94) and component_width <= 5:
                continue
            if int(np.count_nonzero(component & seed_guard)) < max(1, int(component_area * 0.035)):
                continue
            residual_components |= component
        repair_core |= residual_components
        repair_core[art_restore] = False
    repair_core = cv2.morphologyEx(
        repair_core.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ).astype(bool)
    repair_core[art_restore] = False
    if gray_smooth_panel:
        precise_core = cv2.dilate(
            text_seed.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        ).astype(bool)
        precise_core[art_restore] = False
        precise_count = int(np.count_nonzero(precise_core))
        repair_count = int(np.count_nonzero(repair_core))
        if (
            precise_count >= max(40, int(area * 0.014))
            and repair_count > precise_count * 1.85
            and repair_count / float(area) >= 0.18
        ):
            residual_near_text = cv2.dilate(
                precise_core.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
                iterations=1,
            ).astype(bool)
            repair_core = (precise_core | (repair_core & residual_near_text)) & ~art_restore
    precise_text_repair = _precise_floating_text_repair_mask(source, coords)
    if precise_text_repair is not None:
        precise_text = precise_text_repair > 0
        precise_count = int(np.count_nonzero(precise_text))
        repair_count = int(np.count_nonzero(repair_core))
        precise_density = precise_count / float(area)
        repair_density = repair_count / float(area)
        if (
            precise_count >= max(18, int(area * 0.008))
            and precise_density <= 0.46
            and (
                repair_density >= 0.38
                or roi_edge_density >= 0.095
                or (width <= 190 and height <= 150)
            )
        ):
            precise_text = cv2.dilate(
                precise_text.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            ).astype(bool)
            repair_core = precise_text & ~art_restore
            text_seed = (text_seed & precise_text) | (precise_text_repair > 0)
    if int(np.count_nonzero(repair_core)) < max(80, int(area * 0.028)):
        return None
    texture_mask = _rowwise_midtone_texture_repair(
        source,
        target,
        coords,
        repair_core.astype(np.uint8) * 255,
    )
    if texture_mask is not None:
        return texture_mask

    distance = cv2.distanceTransform(repair_core.astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(distance / 4.5, 0.0, 1.0)
    alpha[text_seed] = 1.0
    alpha[art_restore] = 0.0
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 1.25 if gray_smooth_panel else 0.9)
    alpha = np.clip(alpha, 0.0, 1.0)
    if gray_smooth_panel:
        alpha = np.power(alpha, 1.18)
    if float(np.mean(alpha > 0.18)) > 0.68 and roi_edge_density > 0.16:
        return None

    write_threshold = 0.035 if gray_smooth_panel else 0.02
    blended = (
        fitted.astype(np.float32) * alpha[..., None]
        + target_roi.astype(np.float32) * (1.0 - alpha[..., None])
    ).astype(np.uint8)
    target_roi[alpha > write_threshold] = blended[alpha > write_threshold]
    art_restore = cv2.dilate(
        art_restore.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1,
    ).astype(bool)
    target_roi[art_restore] = roi[art_restore]

    repair_mask = (alpha > write_threshold).astype(np.uint8) * 255
    repair_mask[art_restore] = 0
    if gray_smooth_panel:
        target_gray = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
        fitted_gray = cv2.cvtColor(fitted, cv2.COLOR_BGR2GRAY)
        residual_search = cv2.dilate(
            text_seed.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    max(21, min(47, (int(round(width * 0.22)) | 1))),
                    max(21, min(57, (int(round(height * 0.12)) | 1))),
                ),
            ),
            iterations=1,
        ).astype(bool)
        residual_delta = target_gray.astype(np.float32) - fitted_gray.astype(np.float32)
        _, residual_grid_x = np.indices((height, width), dtype=np.int32)
        edge_residual_search = (
            residual_grid_x <= max(7, int(round(width * 0.075)))
        ) | (
            residual_grid_x >= width - max(8, int(round(width * 0.075)))
        )
        bright_residual = (
            (residual_search | edge_residual_search)
            & ~art_restore
            & (roi_hsv[:, :, 1] < 190)
            & (target_gray >= 168)
            & (residual_delta >= 7.0)
        )
        dark_residual = (
            residual_search
            & ~art_restore
            & (roi_hsv[:, :, 1] < 190)
            & (target_gray >= 56)
            & (fitted_gray >= 92)
            & (residual_delta <= -16.0)
        )
        visible_residual = bright_residual | dark_residual
        visible_residual = cv2.morphologyEx(
            visible_residual.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        visible_components = np.zeros((height, width), dtype=np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (visible_residual > 0).astype(np.uint8),
            connectivity=8,
        )
        seed_guard = cv2.dilate(
            text_seed.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
            iterations=1,
        ).astype(bool)
        for label in range(1, component_count):
            component = labels == label
            component_area = int(stats[label, cv2.CC_STAT_AREA])
            if component_area < 4 or component_area > max(12000, int(area * 0.26)):
                continue
            component_width = int(stats[label, cv2.CC_STAT_WIDTH])
            component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
            if component_width > int(width * 0.78) and component_height < 7:
                continue
            if int(np.count_nonzero(component & seed_guard)) < max(1, int(component_area * 0.025)):
                continue
            visible_components[component] = 255
        if int(np.count_nonzero(visible_components > 0)) >= max(8, int(area * 0.002)):
            visible_components = cv2.dilate(
                visible_components,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            visible_components[art_restore] = 0
            target_roi[visible_components > 0] = fitted[visible_components > 0]
            repair_mask = cv2.bitwise_or(repair_mask, visible_components)
            repair_mask[art_restore] = 0
    if int(np.count_nonzero(repair_mask > 0)) < max(80, int(area * 0.028)):
        return None
    return repair_mask


def _cleanup_adjacent_smooth_panel_halos(
    target: np.ndarray,
    coords: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    x1, y1, x2, y2 = coords
    img_h, img_w = target.shape[:2]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = max(14, min(34, int(round(width * 0.15))))
    pad_y = max(8, min(24, int(round(height * 0.06))))
    ex1 = max(0, x1 - pad_x)
    ey1 = max(0, y1 - pad_y)
    ex2 = min(img_w, x2 + pad_x)
    ey2 = min(img_h, y2 + pad_y)
    if ex2 <= ex1 or ey2 <= ey1:
        return None

    roi = target[ey1:ey2, ex1:ex2]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 45, 135) > 0
    roi_h, roi_w = gray.shape[:2]
    base = (
        (hsv[:, :, 1] < 170)
        & (gray >= 66)
        & (gray <= 226)
        & ~edges
    )
    if int(np.count_nonzero(base)) < max(140, int(base.size * 0.12)):
        return None

    fallback_color = np.median(roi[base], axis=0).astype(np.float32)
    fallback_gray = float(np.median(gray[base]))
    base_y, base_x = np.where(base)
    row_colors = np.empty((roi_h, 3), dtype=np.float32)
    row_gray = np.empty(roi_h, dtype=np.float32)
    row_band = max(6, min(24, roi_h // 12))
    for row_index in range(roi_h):
        near_row = np.abs(base_y - row_index) <= row_band
        if int(np.count_nonzero(near_row)) >= 14:
            row_pixels = roi[base_y[near_row], base_x[near_row]]
            row_colors[row_index, :] = np.median(row_pixels, axis=0).astype(np.float32)
            row_gray[row_index] = float(np.median(gray[base_y[near_row], base_x[near_row]]))
        else:
            row_colors[row_index, :] = fallback_color
            row_gray[row_index] = fallback_gray
    row_colors = cv2.GaussianBlur(
        row_colors.reshape(roi_h, 1, 3),
        (1, 0),
        sigmaX=0,
        sigmaY=max(1.8, min(7.5, roi_h / 30.0)),
    ).reshape(roi_h, 3)
    row_gray = cv2.GaussianBlur(
        row_gray.reshape(roi_h, 1),
        (1, 0),
        sigmaX=0,
        sigmaY=max(1.8, min(7.5, roi_h / 30.0)),
    ).reshape(roi_h)

    y_grid, x_grid = np.indices((roi_h, roi_w), dtype=np.int32)
    inner = (
        (x_grid >= x1 - ex1)
        & (x_grid < x2 - ex1)
        & (y_grid >= y1 - ey1)
        & (y_grid < y2 - ey1)
    )
    side_band = (
        (x_grid < max(0, x1 - ex1) + max(8, int(round(width * 0.06))))
        | (x_grid >= min(roi_w, x2 - ex1) - max(8, int(round(width * 0.06))))
    )
    expected_gray = row_gray[:, None]
    candidate = (
        (hsv[:, :, 1] < 190)
        & (gray.astype(np.float32) >= np.maximum(172.0, expected_gray + 18.0))
        & (inner | side_band)
    )
    vertical_guard = max(12, min(28, int(round(roi_h * 0.045))))
    candidate &= (y_grid >= vertical_guard) & (y_grid < roi_h - max(8, vertical_guard // 2))
    candidate = cv2.morphologyEx(
        candidate.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5)),
        iterations=1,
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (candidate > 0).astype(np.uint8),
        connectivity=8,
    )
    cleanup = np.zeros((roi_h, roi_w), dtype=np.uint8)
    for label in range(1, component_count):
        component = labels == label
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        if component_area < 4 or component_area > max(3200, int(candidate.size * 0.05)):
            continue
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if component_width > int(roi_w * 0.60) and component_height <= 8:
            continue
        if not np.any(component & (inner | side_band)):
            continue
        cleanup[component] = 255
    if int(np.count_nonzero(cleanup > 0)) < 8:
        return None

    fill = np.repeat(row_colors[:, None, :], roi_w, axis=1).astype(np.uint8)
    cleanup = cv2.dilate(
        cleanup,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    roi[cleanup > 0] = fill[cleanup > 0]
    return (ex1, ey1, ex2, ey2), cleanup


def _smooth_gradient_source_text_fill(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return None

    height, width = roi.shape[:2]
    area = max(1, height * width)
    if height < 54 or width < 24:
        return None
    narrow_vertical = width <= 170 and height >= width * 1.35

    outlined = _outlined_floating_source_mask(source, coords, mask_roi)
    if outlined is None:
        outlined = _seeded_smooth_panel_outline_mask(source, coords, mask_roi)
    repair = outlined if outlined is not None else mask_roi
    if outlined is not None:
        combined_repair = cv2.bitwise_or(outlined, (mask_roi > 0).astype(np.uint8) * 255)
        combined_density = float(np.count_nonzero(combined_repair > 0)) / float(area)
        if combined_density <= 0.62:
            repair = combined_repair
    repair = (repair > 0).astype(np.uint8) * 255
    repair = cv2.morphologyEx(
        repair,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    repair_count = int(np.count_nonzero(repair > 0))
    if repair_count < 10:
        return None
    repair_density = repair_count / float(area)
    compact_dense = width <= 130 and height <= 170 and height >= 45
    if repair_density < 0.025:
        return None
    if repair_density > 0.44 and not (
        (narrow_vertical and repair_density <= 0.88)
        or (compact_dense and repair_density <= 1.001)
    ):
        return None

    if compact_dense and repair_density >= 0.86:
        fitted, stats = _tone_fit_context_background(
            target,
            coords,
            repair,
            padding=max(34, min(88, int(max(width, height) * 0.42))),
            rowwise=True,
        )
        if (
            fitted is not None
            and 76.0 <= stats.get("median", 0.0) <= 236.0
            and stats.get("std", 99.0) <= 44.0
            and stats.get("edge_density", 1.0) <= 0.10
            and stats.get("dark_fraction", 1.0) <= 0.18
            and stats.get("bright_fraction", 1.0) <= 0.55
        ):
            target_roi = target[y1:y2, x1:x2]
            target_roi[repair > 0] = fitted[repair > 0]
            return repair

    fit_mask = cv2.dilate(
        repair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    background = fit_mask <= 0
    if np.count_nonzero(background) < max(40, int(area * 0.08)):
        fit_mask = repair
        background = repair <= 0
    if np.count_nonzero(background) < max(40, int(area * 0.08)):
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_dark_fraction = float(np.mean(gray < 118))
    roi_bright_fraction = float(np.mean(gray > 218))
    roi_luma_std = float(np.std(gray.astype(np.float32)))
    if roi_dark_fraction >= 0.18 and roi_bright_fraction >= 0.16 and roi_luma_std >= 70.0:
        return None

    bg_gray = gray[background].astype(np.float32)
    bg_bright = float(np.mean(bg_gray > 232))
    bg_mid = float(np.mean((bg_gray >= 70) & (bg_gray <= 225)))
    bg_dark = float(np.mean(bg_gray < 62))
    bg_std = float(np.std(bg_gray))
    bg_edges = float(np.mean((cv2.Canny(gray, 45, 135) > 0)[background]))
    low_saturation = float(np.mean(hsv[:, :, 1][background] < 150))

    fitted = None
    stats = None
    if narrow_vertical and repair_density >= 0.62:
        context_padding = max(42, min(118, int(max(width, height) * 0.45)))
        fitted, stats = _tone_fit_context_background(
            target,
            coords,
            fit_mask,
            padding=context_padding,
            rowwise=True,
        )
        if not (
            fitted is not None
            and 78.0 <= stats.get("median", 0.0) <= 236.0
            and stats.get("std", 99.0) <= 60.0
            and stats.get("edge_density", 1.0) <= 0.13
            and stats.get("dark_fraction", 1.0) <= 0.18
            and stats.get("bright_fraction", 1.0) <= 0.42
        ):
            fitted = None
            stats = None

    local_background = background & (gray >= 70) & (gray <= 232) & (hsv[:, :, 1] < 150)
    if fitted is None and bg_mid >= 0.28 and np.count_nonzero(local_background) >= max(36, int(area * 0.035)):
        background_pixels = roi[local_background].astype(np.float32)
        background_gray = gray[local_background].astype(np.float32)
        global_color = np.median(background_pixels, axis=0)
        fitted = np.empty_like(roi)
        row_colors = np.empty((height, 3), dtype=np.float32)
        band_radius = max(4, min(18, height // 18))
        for row_index in range(height):
            row_y1 = max(0, row_index - band_radius)
            row_y2 = min(height, row_index + band_radius + 1)
            row_background = local_background[row_y1:row_y2]
            if np.count_nonzero(row_background) >= 8:
                row_pixels = roi[row_y1:row_y2][row_background].astype(np.float32)
                row_color = np.median(row_pixels, axis=0)
            else:
                row_color = global_color
            row_colors[row_index, :] = row_color
        if height >= 7:
            sigma_y = max(1.8, min(9.0, height / 36.0))
            row_colors = cv2.GaussianBlur(
                row_colors.reshape(height, 1, 3),
                (1, 0),
                sigmaX=0,
                sigmaY=sigma_y,
            ).reshape(height, 3)
        fitted[:, :, :] = np.clip(row_colors[:, None, :], 0, 255).astype(np.uint8)
        stats = {
            "median": float(np.median(background_gray)),
            "std": float(np.std(background_gray)),
            "edge_density": float(np.mean((cv2.Canny(gray, 45, 135) > 0)[local_background])),
            "dark_fraction": float(np.mean(background_gray < 82)),
            "bright_fraction": float(np.mean(background_gray > 232)),
        }
    else:
        fitted, stats = _tone_fit_context_background(
            source,
            coords,
            fit_mask,
            padding=max(30, min(92, int(max(width, height) * 0.30))),
            rowwise=True,
        )
        if fitted is None:
            return None

    local_smooth = (
        low_saturation >= 0.82
        and bg_mid >= 0.14
        and bg_dark <= 0.22
        and bg_std <= 58.0
        and bg_edges <= 0.14
        and not (bg_bright > 0.90 and bg_mid < 0.12)
    )
    context_smooth = (
        82.0 <= stats.get("median", 0.0) <= 225.0
        and stats.get("std", 99.0) <= 44.0
        and stats.get("edge_density", 1.0) <= 0.12
        and stats.get("dark_fraction", 1.0) <= 0.18
        and stats.get("bright_fraction", 1.0) <= 0.38
    )
    if not (local_smooth or context_smooth):
        return None

    apply_mask = repair
    if (
        context_smooth
        and stats.get("std", 99.0) <= 36.0
        and stats.get("edge_density", 1.0) <= 0.04
        and stats.get("dark_fraction", 1.0) <= 0.04
        and (height >= width * 1.15 or repair_density >= 0.30)
    ):
        apply_mask = cv2.dilate(
            repair,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )

    target_roi = target[y1:y2, x1:x2]
    target_roi[apply_mask > 0] = fitted[apply_mask > 0]
    target_gray = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
    residual_near_text = cv2.dilate(
        apply_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    ) > 0
    residual_halo = (
        residual_near_text
        & (gray >= 218)
        & (target_gray >= 234)
        & (hsv[:, :, 1] < 145)
    )
    residual_halo = cv2.morphologyEx(
        residual_halo.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    residual_density = float(np.count_nonzero(residual_halo > 0)) / float(area)
    if 0.002 <= residual_density <= 0.22:
        residual_halo = cv2.dilate(
            residual_halo,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        target_roi[residual_halo > 0] = fitted[residual_halo > 0]
        apply_mask = cv2.bitwise_or(apply_mask, residual_halo)
    line_restore = _restore_art_lines_crossing_repair_mask(
        source,
        target,
        coords,
        apply_mask,
        mask_roi,
    )
    if line_restore is not None:
        apply_mask = cv2.bitwise_or(apply_mask, line_restore)
    return apply_mask


def _dense_smooth_tone_source_text_fill(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return None

    height, width = roi.shape[:2]
    area = max(1, height * width)
    if height < 54 or width < 24:
        return None

    narrow_vertical = width <= 92 and height >= width * 1.55
    max_repair_density = 1.001 if narrow_vertical else 0.96
    outlined = _outlined_floating_source_mask(source, coords, mask_roi)
    if outlined is None:
        outlined = _seeded_smooth_panel_outline_mask(source, coords, mask_roi)
    if outlined is not None:
        combined_repair = cv2.bitwise_or(outlined, (mask_roi > 0).astype(np.uint8) * 255)
        combined_density = float(np.count_nonzero(combined_repair > 0)) / float(area)
        repair = combined_repair if combined_density <= max_repair_density else mask_roi
    else:
        repair = mask_roi
    repair = (repair > 0).astype(np.uint8) * 255
    repair = cv2.morphologyEx(
        repair,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    repair_count = int(np.count_nonzero(repair > 0))
    if repair_count < 10:
        return None
    repair_density = repair_count / float(area)
    if repair_density < 0.18 or repair_density > max_repair_density:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    fit_mask = cv2.dilate(
        repair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    background = fit_mask <= 0
    if np.count_nonzero(background) < max(28, int(area * 0.035)):
        fit_mask = repair
        background = repair <= 0
    if np.count_nonzero(background) < max(28, int(area * 0.035)):
        if not narrow_vertical:
            return None
        fitted, stats = _tone_fit_context_background(
            target,
            coords,
            repair,
            padding=max(30, min(92, int(max(width, height) * 0.30))),
            rowwise=True,
        )
        if fitted is None:
            fitted, stats = _tone_fit_context_background(
                source,
                coords,
                repair,
                padding=max(30, min(92, int(max(width, height) * 0.30))),
                rowwise=True,
            )
        if fitted is None:
            return None
        if not (
            74.0 <= stats.get("median", 0.0) <= 232.0
            and stats.get("std", 99.0) <= 56.0
            and stats.get("edge_density", 1.0) <= 0.16
            and stats.get("dark_fraction", 1.0) <= 0.18
            and stats.get("bright_fraction", 1.0) <= 0.46
        ):
            return None
        gradient_repair = _smooth_panel_gradient_repair(
            source,
            target,
            coords,
            repair,
            mask_roi,
        )
        if gradient_repair is not None:
            return gradient_repair
        target_roi = target[y1:y2, x1:x2]
        target_roi[repair > 0] = fitted[repair > 0]
        return repair

    bg_gray = gray[background].astype(np.float32)
    bg_mid = float(np.mean((bg_gray >= 70) & (bg_gray <= 230)))
    bg_dark = float(np.mean(bg_gray < 62))
    bg_bright = float(np.mean(bg_gray > 238))
    bg_std = float(np.std(bg_gray))
    bg_edges = float(np.mean((cv2.Canny(gray, 45, 135) > 0)[background]))
    low_saturation = float(np.mean(hsv[:, :, 1][background] < 150))

    if low_saturation < 0.86 or bg_dark > 0.12 or bg_edges > 0.11:
        return None
    if bg_std > 42.0 and bg_mid < 0.58:
        return None
    if bg_bright > 0.92 and bg_mid < 0.08:
        return None

    local_background = background & (gray >= 70) & (gray <= 232) & (hsv[:, :, 1] < 150)
    fitted = None
    stats = None
    if bg_mid >= 0.28 and np.count_nonzero(local_background) >= max(36, int(area * 0.035)):
        background_pixels = roi[local_background].astype(np.float32)
        background_gray = gray[local_background].astype(np.float32)
        global_color = np.median(background_pixels, axis=0)
        fitted = np.empty_like(roi)
        row_colors = np.empty((height, 3), dtype=np.float32)
        band_radius = max(4, min(18, height // 18))
        for row_index in range(height):
            row_y1 = max(0, row_index - band_radius)
            row_y2 = min(height, row_index + band_radius + 1)
            row_background = local_background[row_y1:row_y2]
            if np.count_nonzero(row_background) >= 8:
                row_pixels = roi[row_y1:row_y2][row_background].astype(np.float32)
                row_color = np.median(row_pixels, axis=0)
            else:
                row_color = global_color
            row_colors[row_index, :] = row_color
        if height >= 7:
            sigma_y = max(1.8, min(9.0, height / 36.0))
            row_colors = cv2.GaussianBlur(
                row_colors.reshape(height, 1, 3),
                (1, 0),
                sigmaX=0,
                sigmaY=sigma_y,
            ).reshape(height, 3)
        fitted[:, :, :] = np.clip(row_colors[:, None, :], 0, 255).astype(np.uint8)
        stats = {
            "median": float(np.median(background_gray)),
            "std": float(np.std(background_gray)),
            "edge_density": float(np.mean((cv2.Canny(gray, 45, 135) > 0)[local_background])),
            "dark_fraction": float(np.mean(background_gray < 82)),
            "bright_fraction": float(np.mean(background_gray > 232)),
        }
    elif fitted is None:
        fitted, stats = _tone_fit_context_background(
            target,
            coords,
            fit_mask,
            padding=max(30, min(92, int(max(width, height) * 0.30))),
            rowwise=True,
        )
        if fitted is None:
            fitted, stats = _tone_fit_context_background(
                source,
                coords,
                fit_mask,
                padding=max(30, min(92, int(max(width, height) * 0.30))),
                rowwise=True,
            )
        if fitted is None:
            return None
    if not (82.0 <= stats.get("median", 0.0) <= 236.0):
        return None
    max_background_std = 56.0 if narrow_vertical else 36.0
    max_background_edge_density = 0.095 if narrow_vertical else 0.065
    if (
        stats.get("std", 99.0) > max_background_std
        or stats.get("edge_density", 1.0) > max_background_edge_density
    ):
        return None
    if stats.get("dark_fraction", 1.0) > 0.10:
        return None

    apply_mask = repair
    if repair_density >= 0.22:
        kernel_size = 5 if repair_density >= 0.28 else 3
        iterations = 2 if repair_density >= 0.55 else 1
        expanded = cv2.dilate(
            repair,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
            iterations=iterations,
        )
        if float(np.count_nonzero(expanded > 0)) / float(area) <= 0.992:
            apply_mask = expanded

    gradient_repair = _smooth_panel_gradient_repair(
        source,
        target,
        coords,
        apply_mask,
        mask_roi,
    )
    if gradient_repair is not None:
        return gradient_repair

    target_roi = target[y1:y2, x1:x2]
    target_roi[apply_mask > 0] = fitted[apply_mask > 0]
    target_gray = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
    residual_near_text = cv2.dilate(
        apply_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    ) > 0
    residual_halo = (
        residual_near_text
        & (gray >= 218)
        & (target_gray >= 234)
        & (hsv[:, :, 1] < 145)
    )
    residual_halo = cv2.morphologyEx(
        residual_halo.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    residual_density = float(np.count_nonzero(residual_halo > 0)) / float(area)
    if 0.002 <= residual_density <= 0.22:
        residual_halo = cv2.dilate(
            residual_halo,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        target_roi[residual_halo > 0] = fitted[residual_halo > 0]
        apply_mask = cv2.bitwise_or(apply_mask, residual_halo)
    return apply_mask


def _thin_edge_line_preserve_mask(source_roi: np.ndarray) -> np.ndarray:
    if source_roi.size == 0:
        return np.zeros(source_roi.shape[:2], dtype=bool)

    height, width = source_roi.shape[:2]
    gray = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(source_roi, cv2.COLOR_BGR2HSV)
    dark_line = (gray < 92) & (hsv[:, :, 1] < 230)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark_line.astype(np.uint8),
        connectivity=8,
    )
    preserve = np.zeros((height, width), dtype=bool)
    for label in range(1, component_count):
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if cw <= 0 or ch <= 0:
            continue
        touches_border = cx <= 1 or cy <= 1 or (cx + cw) >= width - 1 or (cy + ch) >= height - 1
        axis_ratio = max(cw, ch) / float(max(1, min(cw, ch)))
        if (
            touches_border
            and axis_ratio >= 5.0
            and min(cw, ch) <= 7
            and area >= max(8, int(max(width, height) * 0.18))
        ):
            preserve |= labels == label
    return preserve


def _compact_precise_outline_mask(
    source_roi: np.ndarray,
    seed_roi: np.ndarray,
    line_preserve: np.ndarray,
) -> np.ndarray | None:
    if source_roi.size == 0 or seed_roi.size == 0:
        return None

    gray = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(source_roi, cv2.COLOR_BGR2HSV)
    seed = seed_roi > 0
    area = max(1, seed_roi.size)
    if np.count_nonzero(seed) < 8:
        return None

    near_seed = cv2.dilate(
        seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        iterations=1,
    ) > 0
    seed_context = cv2.dilate(
        seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
        iterations=1,
    ) > 0
    background = (
        ~seed_context
        & ~line_preserve
        & (gray >= 55)
        & (gray <= 238)
        & (hsv[:, :, 1] < 175)
    )
    if np.count_nonzero(background) < max(24, int(area * 0.035)):
        return None

    background_level = float(np.percentile(gray[background], 35))
    bright_outline = (
        near_seed
        & ~line_preserve
        & (gray >= max(174, int(background_level + 10)))
        & (hsv[:, :, 1] < 180)
    )
    dark_glyph = (
        near_seed
        & ~line_preserve
        & (gray <= min(154, int(background_level - 28)))
    )
    repair = (seed | bright_outline | dark_glyph).astype(np.uint8) * 255
    repair[line_preserve] = 0
    repair = cv2.morphologyEx(
        repair,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    repair = cv2.dilate(
        repair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    repair[line_preserve] = 0
    density = float(np.count_nonzero(repair > 0)) / float(area)
    if density < 0.16 or density > 0.82:
        return None
    return repair


def _low_frequency_smooth_background_repair(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return None

    height, width = roi.shape[:2]
    area = max(1, height * width)
    if height < 45 or width < 24 or width > 180 or height > 240:
        return None

    repair = _outlined_floating_source_mask(source, coords, mask_roi)
    if repair is None:
        return None
    repair = (repair > 0).astype(np.uint8) * 255
    repair_density = float(np.count_nonzero(repair > 0)) / float(area)
    if repair_density < 0.42:
        return None

    line_preserve = _thin_edge_line_preserve_mask(roi)
    if repair_density >= 0.92:
        precise_repair = _compact_precise_outline_mask(roi, mask_roi, line_preserve)
        if precise_repair is not None:
            repair = precise_repair
            repair_density = float(np.count_nonzero(repair > 0)) / float(area)

    fitted, stats = _tone_fit_context_background(
        target,
        coords,
        repair,
        padding=max(34, min(88, int(max(width, height) * 0.42))),
        rowwise=True,
    )
    if fitted is None:
        return None
    if not (
        74.0 <= stats.get("median", 0.0) <= 238.0
        and stats.get("std", 99.0) <= 18.0
        and stats.get("edge_density", 1.0) <= 0.04
        and stats.get("dark_fraction", 1.0) <= 0.06
        and stats.get("bright_fraction", 1.0) <= 0.70
    ):
        return None

    img_h, img_w = source.shape[:2]
    pad = max(42, min(96, int(max(width, height) * 0.55)))
    crop_x1 = max(0, x1 - pad)
    crop_y1 = max(0, y1 - pad)
    crop_x2 = min(img_w, x2 + pad)
    crop_y2 = min(img_h, y2 + pad)
    crop = target[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    if crop.size == 0:
        return None

    min_dim = min(crop.shape[:2])
    kernel = int(max(31, min(75, max(width, height) * 0.56)))
    if kernel % 2 == 0:
        kernel += 1
    if kernel >= min_dim:
        kernel = max(3, min_dim - 1 if (min_dim - 1) % 2 == 1 else min_dim - 2)
    if kernel < 21:
        return None

    background = cv2.medianBlur(crop, kernel)
    background = cv2.GaussianBlur(background, (0, 0), 2.6)
    local_background = background[y1 - crop_y1:y2 - crop_y1, x1 - crop_x1:x2 - crop_x1]
    if local_background.shape[:2] != (height, width):
        return None

    apply_mask = repair > 0
    apply_mask &= ~line_preserve
    if np.count_nonzero(apply_mask) < 10:
        return None

    target_roi = target[y1:y2, x1:x2]
    target_roi[apply_mask] = local_background[apply_mask]
    target_roi[line_preserve] = roi[line_preserve]
    return apply_mask.astype(np.uint8) * 255


def _mixed_dark_surface_source_inpaint(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return None

    outlined = _outlined_floating_source_mask(source, coords, mask_roi)
    if outlined is None or np.count_nonzero(outlined > 0) < 10:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    dark_fraction = float(np.mean(gray < 118))
    bright_fraction = float(np.mean(gray > 218))
    luma_std = float(np.std(gray.astype(np.float32)))
    if dark_fraction < 0.18 or bright_fraction < 0.16 or luma_std < 76.0:
        return None

    repair = (outlined > 0).astype(np.uint8) * 255
    repair = cv2.morphologyEx(
        repair,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    repair_count = int(np.count_nonzero(repair > 0))
    area = max(1, repair.shape[0] * repair.shape[1])
    density = repair_count / float(area)
    if repair_count < 10 or density > 0.62:
        return None

    try:
        telea = cv2.inpaint(target[y1:y2, x1:x2], repair, 1.6, cv2.INPAINT_TELEA)
        navier = cv2.inpaint(target[y1:y2, x1:x2], repair, 1.35, cv2.INPAINT_NS)
    except cv2.error:
        return None

    repaired = cv2.addWeighted(telea, 0.72, navier, 0.28, 0)
    target_roi = target[y1:y2, x1:x2]
    target_roi[repair > 0] = repaired[repair > 0]
    return repair


def _mixed_surface_caption_fill(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    seed_roi: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or seed_roi.size == 0:
        return None

    height, width = roi.shape[:2]
    area = max(1, height * width)
    if height < max(80, int(width * 1.35)) or width > 110:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    low_saturation = float(np.mean(saturation < 170))
    bright_fraction = float(np.mean((gray > 184) & (saturation < 170)))
    dark_fraction = float(np.mean(gray < 72))
    mid_fraction = float(np.mean((gray >= 72) & (gray <= 180) & (saturation < 170)))
    luma_std = float(np.std(gray.astype(np.float32)))
    if (
        low_saturation < 0.82
        or bright_fraction < 0.16
        or dark_fraction < 0.16
        or mid_fraction < 0.08
        or luma_std < 64.0
    ):
        return None

    img_h, img_w = source.shape[:2]
    pad = max(18, min(34, int(max(width, height) * 0.12)))
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(img_w, x2 + pad)
    ry2 = min(img_h, y2 + pad)
    crop = source[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return None

    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    inner = np.zeros(crop.shape[:2], dtype=np.uint8)
    inner[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = 255
    hint = np.zeros_like(inner)
    hint[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = (seed_roi > 0).astype(np.uint8) * 255
    hint_near = cv2.dilate(
        hint,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    white_seed = (
        (crop_gray > 184)
        & (crop_hsv[:, :, 1] < 180)
        & (inner > 0)
        & ((hint_near > 0) | (hint == 0))
    ).astype(np.uint8) * 255
    if int(np.count_nonzero(white_seed > 0)) < max(24, int(area * 0.10)):
        white_seed = cv2.bitwise_and(hint, inner)
    if int(np.count_nonzero(white_seed > 0)) < 24:
        return None

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats((white_seed > 0).astype(np.uint8), 8)
    seed = np.zeros_like(white_seed)
    for label in range(1, component_count):
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if component_area < 14 or ch < 4 or cw < 2:
            continue
        if component_area > max(5200, int(area * 0.62)):
            continue
        component = labels == label
        if int(np.count_nonzero(component & (inner > 0))) < component_area * 0.82:
            continue
        seed[component] = 255
    if int(np.count_nonzero(seed > 0)) < 24:
        return None

    blob = cv2.morphologyEx(
        seed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    flood = blob.copy()
    flood_mask = np.zeros((blob.shape[0] + 2, blob.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    blob = cv2.bitwise_or(blob, holes)
    allowed = cv2.dilate(
        seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=1,
    )
    blob = cv2.bitwise_and(blob, allowed)
    blob = cv2.dilate(
        blob,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    blob = cv2.bitwise_and(blob, inner)
    repair_count = int(np.count_nonzero(blob > 0))
    repair_density = repair_count / float(max(1, blob.size))
    if repair_count < 30 or repair_density > 0.46:
        return None

    repair = blob > 0
    context = ~repair
    black_anchor = context & (crop_gray < 52)
    mid_anchor = context & (crop_gray >= 72) & (crop_gray <= 175) & (crop_hsv[:, :, 1] < 175)
    if int(np.count_nonzero(black_anchor)) < 40 or int(np.count_nonzero(mid_anchor)) < 30:
        return None
    # A caption whose TRUE background is bright (this function only runs on
    # `_legacy_bright_caption_candidate` regions) can have its bright pixels
    # fragmented by dense text/art into many components each just under the
    # seed-selection size cap, so `repair` ends up covering most of the
    # panel's plain background, not just small mixed-tone gaps. Without a
    # bright reference option, every one of those pixels was forced toward
    # black_anchor or mid_anchor -- repainting the legitimate white
    # background dark (verified: original/sample6's "RAT'S WEAPON" caption,
    # a plain white panel with bold text, went fully black). A bright anchor
    # lets pixels near the dominant light surround repaint back to it
    # instead of being forced into one of the two dark options.
    bright_anchor = context & (crop_gray > 184) & (crop_hsv[:, :, 1] < 180)
    has_bright_anchor = int(np.count_nonzero(bright_anchor)) >= 30

    distance_to_black = cv2.distanceTransform((~black_anchor).astype(np.uint8), cv2.DIST_L2, 5)
    distance_to_mid = cv2.distanceTransform((~mid_anchor).astype(np.uint8), cv2.DIST_L2, 5)
    distance_to_bright = (
        cv2.distanceTransform((~bright_anchor).astype(np.uint8), cv2.DIST_L2, 5)
        if has_bright_anchor
        else None
    )
    target_crop = target[ry1:ry2, rx1:rx2].copy()
    repaired = target_crop.astype(np.float32)
    black_default = np.median(target_crop[black_anchor].astype(np.float32), axis=0)
    mid_default = np.median(target_crop[mid_anchor].astype(np.float32), axis=0)
    bright_default = (
        np.median(target_crop[bright_anchor].astype(np.float32), axis=0)
        if has_bright_anchor
        else mid_default
    )

    row_black = np.empty((target_crop.shape[0], 3), dtype=np.float32)
    row_mid = np.empty((target_crop.shape[0], 3), dtype=np.float32)
    row_bright = np.empty((target_crop.shape[0], 3), dtype=np.float32)
    for row_index in range(target_crop.shape[0]):
        row_start = max(0, row_index - 5)
        row_end = min(target_crop.shape[0], row_index + 6)
        black_rows = black_anchor[row_start:row_end]
        mid_rows = mid_anchor[row_start:row_end]
        row_black[row_index] = (
            np.median(target_crop[row_start:row_end][black_rows].astype(np.float32), axis=0)
            if int(np.count_nonzero(black_rows)) >= 8
            else black_default
        )
        row_mid[row_index] = (
            np.median(target_crop[row_start:row_end][mid_rows].astype(np.float32), axis=0)
            if int(np.count_nonzero(mid_rows)) >= 8
            else mid_default
        )
        if has_bright_anchor:
            bright_rows = bright_anchor[row_start:row_end]
            row_bright[row_index] = (
                np.median(target_crop[row_start:row_end][bright_rows].astype(np.float32), axis=0)
                if int(np.count_nonzero(bright_rows)) >= 8
                else bright_default
            )

    repair_y, repair_x = np.where(repair)
    if has_bright_anchor:
        dist_black = distance_to_black[repair_y, repair_x]
        dist_mid = distance_to_mid[repair_y, repair_x]
        dist_bright = distance_to_bright[repair_y, repair_x]
        # Slight bias (0.86) toward mid/bright over black, matching the
        # original black-vs-mid tie-break -- a mid or bright anchor equally
        # close to a black one is usually the better match since anti-
        # aliasing halos skew distance-to-black shorter than the true tone.
        nearest = np.argmin(
            np.stack([dist_black, dist_mid * 0.86, dist_bright * 0.86], axis=0), axis=0
        )
        for pixel_y, pixel_x, choice in zip(repair_y, repair_x, nearest):
            if choice == 1:
                repaired[pixel_y, pixel_x] = row_mid[pixel_y]
            elif choice == 2:
                repaired[pixel_y, pixel_x] = row_bright[pixel_y]
            else:
                repaired[pixel_y, pixel_x] = row_black[pixel_y]
    else:
        choose_mid = distance_to_mid[repair_y, repair_x] < distance_to_black[repair_y, repair_x] * 0.86
        for pixel_y, pixel_x, use_mid in zip(repair_y, repair_x, choose_mid):
            repaired[pixel_y, pixel_x] = row_mid[pixel_y] if use_mid else row_black[pixel_y]

    repaired = np.clip(repaired, 0, 255).astype(np.uint8)
    smoothed = cv2.medianBlur(repaired, 3)
    repaired[repair] = smoothed[repair]
    target_view = target[ry1:ry2, rx1:rx2]
    target_view[repair] = repaired[repair]

    full_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    mask_view = full_mask[ry1:ry2, rx1:rx2]
    mask_view[repair] = 255
    return full_mask


def _bright_textured_source_repair(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return None

    repair = (mask_roi > 0).astype(np.uint8) * 255
    repair_count = int(np.count_nonzero(repair > 0))
    if repair_count < 8:
        return None

    area = max(1, repair.size)
    repair_density = repair_count / float(area)
    if repair_density > 0.58:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    unmasked = repair <= 0
    if np.count_nonzero(unmasked) < max(20, int(area * 0.04)):
        return None

    background_gray = gray[unmasked]
    background_sat = hsv[:, :, 1][unmasked]
    background_edges = cv2.Canny(gray, 45, 135)[unmasked] > 0
    bg_median = float(np.median(background_gray))
    bg_std = float(np.std(background_gray.astype(np.float32)))
    bg_edge = float(np.mean(background_edges))
    bright_fraction = float(np.mean(background_gray > 228))
    paper_fraction = float(np.mean((background_gray > 170) & (background_sat < 145)))
    flat_white = bg_std < 6.0 and bg_edge < 0.015 and bright_fraction > 0.90
    textured_bright = (
        bg_median >= 208.0
        and paper_fraction >= 0.58
        and bright_fraction >= 0.40
        and not flat_white
        and (bg_std >= 7.0 or bg_edge >= 0.018)
    )
    if not textured_bright:
        return None

    high_texture = bg_std >= 18.0 or bg_edge >= 0.065
    if high_texture:
        core_repair = cv2.dilate(
            repair,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
    else:
        core_repair = repair.copy()
    expansion_kernel = (7, 7) if high_texture else (5, 5)
    model_repair = cv2.dilate(
        repair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, expansion_kernel),
        iterations=1,
    )
    full_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = model_repair
    before = target[y1:y2, x1:x2].copy()

    use_manga_cleaner = high_texture and os.getenv(
        "MANGA_BRIGHT_TEXTURE_CLEANER", "off"
    ).strip().lower() in {"1", "true", "yes", "on"}
    repaired = False
    if use_manga_cleaner:
        previous_backend = os.environ.get("MANGA_CLEANER_BACKEND")
        try:
            if not previous_backend or previous_backend.strip().lower() in {"0", "false", "no", "off"}:
                os.environ["MANGA_CLEANER_BACKEND"] = "auto"
            repaired = _manga_cleaner_local_crop(
                target,
                full_mask,
                source.shape[0],
                source.shape[1],
                x1,
                y1,
                x2,
                y2,
            )
        finally:
            if previous_backend is None:
                os.environ.pop("MANGA_CLEANER_BACKEND", None)
            else:
                os.environ["MANGA_CLEANER_BACKEND"] = previous_backend

    if not repaired:
        pad = max(12, min(26, int(max(x2 - x1, y2 - y1) * 0.10)))
        crop_x1 = max(0, x1 - pad)
        crop_y1 = max(0, y1 - pad)
        crop_x2 = min(source.shape[1], x2 + pad)
        crop_y2 = min(source.shape[0], y2 + pad)
        crop_img = target[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        crop_mask = full_mask[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        if np.count_nonzero(crop_mask > 0) < 8:
            return None
        try:
            telea = cv2.inpaint(crop_img, crop_mask, 1.8, cv2.INPAINT_TELEA)
            navier = cv2.inpaint(crop_img, crop_mask, 1.4, cv2.INPAINT_NS)
        except cv2.error:
            return None
        repaired_crop = cv2.addWeighted(telea, 0.72, navier, 0.28, 0)
        target_view = target[crop_y1:crop_y2, crop_x1:crop_x2]
        target_view[crop_mask > 0] = repaired_crop[crop_mask > 0]
    elif high_texture:
        changed = np.any(before != target[y1:y2, x1:x2], axis=2)
        changed_count = int(np.count_nonzero(changed & (core_repair > 0)))
        if changed_count < 8:
            return None
        return core_repair

    ring_mask = (model_repair > 0) & ~(cv2.dilate(
        core_repair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ) > 0)
    if np.count_nonzero(ring_mask) >= 10:
        repaired_roi = target[y1:y2, x1:x2].copy()
        ring_alpha = cv2.GaussianBlur(ring_mask.astype(np.float32), (0, 0), 1.1)
        ring_strength = 0.74 if (bg_std < 14.0 and bg_edge < 0.055) else 0.62
        ring_alpha = np.clip(ring_alpha * ring_strength, 0.0, 1.0)[..., None]
        restored = (
            before.astype(np.float32) * ring_alpha
            + repaired_roi.astype(np.float32) * (1.0 - ring_alpha)
        ).astype(np.uint8)
        repaired_roi[ring_mask] = restored[ring_mask]
        target[y1:y2, x1:x2] = repaired_roi

    _local_repair_tone_match(source, target, coords, core_repair)
    source_gray_roi = gray
    repaired_gray = cv2.cvtColor(target[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    residual_similarity = (
        (core_repair > 0)
        & (np.abs(repaired_gray.astype(np.int16) - source_gray_roi.astype(np.int16)) < 20)
    )
    residual_dark = (
        (core_repair > 0)
        & (repaired_gray < max(0.0, bg_median - 20.0))
    )
    if (
        int(np.count_nonzero(residual_similarity)) >= max(18, int(repair_count * 0.16))
        or int(np.count_nonzero(residual_dark)) >= max(14, int(repair_count * 0.10))
    ):
        fallback_mask = cv2.dilate(
            core_repair,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        pad = max(12, min(26, int(max(x2 - x1, y2 - y1) * 0.10)))
        crop_x1 = max(0, x1 - pad)
        crop_y1 = max(0, y1 - pad)
        crop_x2 = min(source.shape[1], x2 + pad)
        crop_y2 = min(source.shape[0], y2 + pad)
        crop_img = target[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        crop_mask = np.zeros(crop_img.shape[:2], dtype=np.uint8)
        crop_mask[y1 - crop_y1:y2 - crop_y1, x1 - crop_x1:x2 - crop_x1] = fallback_mask
        try:
            telea = cv2.inpaint(crop_img, crop_mask, 1.8, cv2.INPAINT_TELEA)
            navier = cv2.inpaint(crop_img, crop_mask, 1.4, cv2.INPAINT_NS)
            repaired_crop = cv2.addWeighted(telea, 0.72, navier, 0.28, 0)
            target_view = target[crop_y1:crop_y2, crop_x1:crop_x2]
            target_view[crop_mask > 0] = repaired_crop[crop_mask > 0]
            _local_repair_tone_match(source, target, coords, fallback_mask)
            core_repair = fallback_mask
        except cv2.error:
            pass
    changed = np.any(before != target[y1:y2, x1:x2], axis=2)
    changed_count = int(np.count_nonzero(changed & (core_repair > 0)))
    if changed_count < 8:
        return None
    return core_repair


def _outlined_floating_source_mask(
    source: np.ndarray,
    coords: tuple[int, int, int, int],
    seed_roi: np.ndarray,
) -> np.ndarray | None:
    """Find white-outlined manga floating text, including its halo.

    Standard text detectors often catch only the black glyph core. For manga SFX
    and floating dialogue, the destructive leftover is usually the white outline.
    This mask targets that outline plus nearby dark glyph pixels while keeping the
    mask bounded to connected components that look text-like inside the layout box.
    """

    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or seed_roi.size == 0:
        return None

    height, width = roi.shape[:2]
    area = max(1, height * width)
    if area < 1000:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    seed = seed_roi > 0
    if int(np.count_nonzero(seed)) >= 6:
        seed_context = cv2.dilate(
            seed.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        ) > 0
        background_probe = ~seed_context
        if int(np.count_nonzero(background_probe)) >= max(20, int(area * 0.04)):
            background_luma = float(np.median(gray[background_probe]))
        else:
            background_luma = float(np.percentile(gray, 50))
        img_h, img_w = source.shape[:2]
        pad = max(12, min(56, int(max(width, height) * 0.18)))
        rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
        rx2, ry2 = min(img_w, x2 + pad), min(img_h, y2 + pad)
        if rx2 > rx1 and ry2 > ry1:
            context = source[ry1:ry2, rx1:rx2]
            context_gray = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY)
            context_hsv = cv2.cvtColor(context, cv2.COLOR_BGR2HSV)
            ring = np.ones(context_gray.shape, dtype=bool)
            ring[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False
            ring_candidates = ring & (context_gray >= 48) & (context_gray <= 232) & (context_hsv[:, :, 1] < 170)
            if int(np.count_nonzero(ring_candidates)) >= max(24, int(area * 0.018)):
                background_luma = min(background_luma, float(np.median(context_gray[ring_candidates])))
        if background_luma < 224.0:
            halo_context = cv2.dilate(
                seed.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
                iterations=1,
            ) > 0
            halo_cutoff = max(218.0, min(248.0, background_luma + 28.0))
            halo = halo_context & (gray >= halo_cutoff) & (saturation < 155)
            halo = cv2.morphologyEx(
                halo.astype(np.uint8) * 255,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            halo_count = int(np.count_nonzero(halo > 0))
            seed_count = int(np.count_nonzero(seed))
            if halo_count >= max(8, int(seed_count * 0.16)):
                repair = cv2.bitwise_or(halo, seed.astype(np.uint8) * 255)
                repair = cv2.morphologyEx(
                    repair,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                    iterations=1,
                )
                repair = cv2.dilate(
                    repair,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=1,
                )
                density = float(np.count_nonzero(repair > 0)) / float(area)
                narrow_vertical = width <= 92 and height >= width * 1.55
                max_density = 1.001 if narrow_vertical else 0.94
                if 0.025 <= density <= max_density:
                    return repair
    paper_fraction = float(np.mean((gray > 172) & (saturation < 115)))
    mean_luma = float(np.mean(gray))
    dark_glyph_fraction = float(np.mean(gray < 132))
    if paper_fraction >= 0.76 and mean_luma >= 176.0 and dark_glyph_fraction < 0.08:
        return None

    p50 = float(np.percentile(gray, 50))
    bright_cutoff = max(168.0, min(206.0, p50 + 42.0))
    bright = ((gray > bright_cutoff) & (saturation < 150)).astype(np.uint8) * 255
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1,
    )
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    kept = np.zeros_like(bright)
    for label in range(1, component_count):
        component = labels == label
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if component_area < 5:
            continue
        if component_area > max(12000, int(area * 0.72)):
            continue

        near = cv2.dilate(
            component.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        ) > 0
        dark_near = int(np.count_nonzero(near & (gray < 132)))
        seed_overlap = int(np.count_nonzero(component & seed))
        if dark_near < max(2, int(component_area * 0.005)) and seed_overlap < 3:
            continue

        touches_border = cx <= 1 or cy <= 1 or (cx + cw) >= width - 1 or (cy + ch) >= height - 1
        if touches_border and component_area > max(260, int(area * 0.035)):
            if dark_near < int(component_area * 0.018) and seed_overlap < 8:
                continue

        kept[component] = 255

    if np.count_nonzero(kept > 0) < 12:
        return None

    near_kept = cv2.dilate(
        (kept > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    ) > 0
    narrow_vertical = width <= 72 and height >= width * 2.0
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    )
    if narrow_vertical:
        dark_glyphs = (near_kept & (gray < 132) & ((blackhat > 6) | seed)).astype(np.uint8) * 255
    else:
        dark_glyphs = (near_kept & (gray < 132)).astype(np.uint8) * 255
    repair = cv2.bitwise_or(kept, dark_glyphs)
    repair = cv2.bitwise_or(repair, (seed.astype(np.uint8) * 255))
    repair = cv2.morphologyEx(
        repair,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    repair = cv2.dilate(
        repair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    density = float(np.count_nonzero(repair > 0)) / float(area)
    max_density = 1.001 if narrow_vertical else 0.92
    if width <= 130 and height <= 170 and height >= 45:
        max_density = 1.001
    if density < 0.025 or density > max_density:
        return None
    return repair


def _filter_precise_text_components(
    raw_mask: np.ndarray,
    area: int,
    width: int,
    height: int,
    mode: str,
) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        raw_mask.astype(np.uint8),
        connectivity=8,
    )
    filtered = np.zeros((height, width), dtype=np.uint8)
    for label in range(1, component_count):
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        if component_area < 3:
            continue
        component_x = int(stats[label, cv2.CC_STAT_LEFT])
        component_y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_border = (
            component_x <= 1
            or component_y <= 1
            or (component_x + component_width) >= width - 1
            or (component_y + component_height) >= height - 1
        )
        if mode == "color":
            if touches_border:
                continue
            if component_area > max(3500, int(area * 0.28)):
                continue
            if component_width > int(width * 0.82) or component_height > int(height * 0.82):
                continue
        else:
            if component_area > max(1800, int(area * 0.14)):
                continue
            if component_width > int(width * 0.86) or component_height > int(height * 0.86):
                continue
            if component_width > 90 and component_height <= 7:
                continue
            if component_height > 130 and component_width <= 7:
                continue
            if touches_border and (
                component_area > max(64, int(area * 0.014))
                or component_height > int(height * 0.42)
                or component_width > int(width * 0.42)
            ):
                continue
        filtered[labels == label] = 255
    return filtered


def _floating_artist_backing_silhouette(
    source: np.ndarray,
    coords: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Full silhouette (ROI-shaped uint8) of the artist's translucent light
    backing panel behind floating lettering, for whole-panel erasure ahead of
    model reconstruction: erasing only the glyph area leaves tone
    discontinuities at the panel's soft edges, while erasing the whole panel
    lets the model rebuild the raw art so text can sit transparently on it.
    Returns None when there is no such panel -- or when the bright region is a
    real speech bubble (traced by a dark outline), which must never be erased."""
    x1, y1, x2, y2 = coords
    img_h, img_w = source.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    pad = 20
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(img_w, x2 + pad), min(img_h, y2 + pad)
    window = source[cy1:cy2, cx1:cx2]
    gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(window, cv2.COLOR_BGR2HSV)
    ix1, iy1 = x1 - cx1, y1 - cy1
    ix2, iy2 = ix1 + (x2 - x1), iy1 + (y2 - y1)
    inside_gray = gray[iy1:iy2, ix1:ix2]
    inside_sat = hsv[iy1:iy2, ix1:ix2, 1]
    bright = (inside_gray >= 170) & (inside_sat <= 130)
    if float(np.mean(bright)) < 0.45:
        return None
    ring = np.ones(gray.shape, dtype=bool)
    ring[iy1:iy2, ix1:ix2] = False
    if int(np.count_nonzero(ring)) < 80:
        return None
    # A backing panel is a LOCAL luminance lift. Compare like-for-like: the
    # panel must be brighter than the ring's own bright pixels. Comparing
    # against the whole-ring median let any bright wall/floor "lift" over a
    # ring that contained dark art (a character's suit) and get erased whole —
    # the model then re-hallucinated the surrounding art (new_sample_13).
    ring_sat = hsv[:, :, 1]
    ring_bright = ring & (gray >= 170) & (ring_sat <= 130)
    if int(np.count_nonzero(ring_bright)) >= 40:
        lift = float(np.median(inside_gray[bright])) - float(
            np.median(gray[ring_bright].astype(np.float32))
        )
        if lift < 12.0:
            return None
    elif float(np.median(inside_gray[bright])) - float(np.median(gray[ring].astype(np.float32))) < 16.0:
        return None
    silhouette = cv2.morphologyEx(
        bright.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    silhouette = cv2.morphologyEx(
        silhouette,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    # Real speech bubbles are traced by a dark outline just outside the bright
    # region -- those are artwork containers, never erased wholesale.
    sil_window = np.zeros(gray.shape, dtype=np.uint8)
    sil_window[iy1:iy2, ix1:ix2] = silhouette
    border = (
        cv2.dilate(
            sil_window, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        )
        > 0
    ) & (sil_window == 0)
    if int(np.count_nonzero(border)) >= 40:
        if float(np.mean(gray[border] < 110)) >= 0.22:
            return None
    return silhouette


def _precise_floating_text_repair_mask(
    source: np.ndarray,
    coords: tuple[int, int, int, int],
    allow_dense: bool = False,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    height, width = roi.shape[:2]
    area = max(1, height * width)
    if area < 700:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    edges = cv2.Canny(gray, 45, 135) > 0
    background_candidates = (
        (saturation < 150)
        & (gray >= 42)
        & (gray <= 245)
        & ~edges
    )
    if int(np.count_nonzero(background_candidates)) >= max(16, int(area * 0.06)):
        background_luma = float(np.median(gray[background_candidates]))
    else:
        background_luma = float(np.median(gray))

    raw_color_mask = (saturation > 68) & (gray > 45) & (gray < 245)
    # The color-text branch below assumes "isolated colored text on a
    # comparatively neutral background" -- its component-count/density
    # checks are sized for that. A saturated PANEL/badge background (e.g.
    # gold) can itself satisfy the raw saturation test across most of the
    # region; shape/size filtering (_filter_precise_text_components) then
    # strips the big background blob but can still leave enough small
    # boundary-noise leftovers to cross the filtered-count threshold below,
    # incorrectly routing a bold-black-ink-on-colored-panel region into the
    # color branch (measured: new_sample_14's "신혼특강" badge -- 82.7% raw
    # coverage, entirely the gold background, not text). When the RAW mask
    # already covers most of the region, the premise is broken regardless of
    # what survives filtering, so skip straight to the dark_components path.
    color_components = None
    if float(np.count_nonzero(raw_color_mask)) / float(area) <= 0.55:
        color_components = _filter_precise_text_components(
            raw_color_mask,
            area,
            width,
            height,
            "color",
        )
    color_count = 0 if color_components is None else int(np.count_nonzero(color_components > 0))
    color_density = color_count / float(area)
    if color_count >= max(16, int(area * 0.006)) and color_density <= 0.32:
        core = color_components > 0
        near_core = cv2.dilate(
            core.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=1,
        ).astype(bool)
        bright_threshold = max(188.0, min(244.0, background_luma + 38.0))
        halo = near_core & (gray > bright_threshold) & (saturation < 180)
        antialias = near_core & (saturation > 38) & (gray > 48) & (gray < 252)
        mask = core | halo | antialias
    else:
        if background_luma >= 220.0:
            dark_threshold = max(95.0, min(174.0, background_luma - 48.0))
        elif background_luma >= 150.0:
            dark_threshold = max(44.0, min(105.0, background_luma - 50.0))
        else:
            dark_threshold = max(36.0, min(82.0, background_luma - 36.0))
        dark_components = _filter_precise_text_components(
            (gray < dark_threshold) & (saturation < 235),
            area,
            width,
            height,
            "dark",
        )
        if int(np.count_nonzero(dark_components > 0)) < 6:
            return None
        core = dark_components > 0
        near_core = cv2.dilate(
            core.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        ).astype(bool)
        bright_threshold = max(176.0, min(246.0, background_luma + 42.0))
        halo = near_core & (gray > bright_threshold) & (saturation < 185)
        near_text = cv2.dilate(
            (core | halo).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ).astype(bool)
        antialias_threshold = max(18.0, min(46.0, abs(background_luma - dark_threshold) * 0.42))
        antialias = (
            near_text
            & (saturation < 200)
            & (np.abs(gray.astype(np.float32) - background_luma) >= antialias_threshold)
        )
        mask = core | halo | antialias

    mask = cv2.morphologyEx(
        mask.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    # Dense halo'd caption columns legitimately exceed the classical-inpaint
    # density budget; the anime-manga LaMa handles large holes, so the caller
    # can raise the cap when the model will do the reconstruction.
    density_cap = 0.62 if allow_dense else 0.46
    density = float(np.count_nonzero(mask > 0)) / float(area)
    if density > density_cap:
        mask = cv2.erode(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        density = float(np.count_nonzero(mask > 0)) / float(area)
        if density > density_cap:
            return None
    if int(np.count_nonzero(mask > 0)) < max(10, int(area * 0.006)):
        return None
    return mask


def _precise_floating_text_local_cleanup(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    anime_model=None,
    anime_device=None,
    allowed_mask: np.ndarray | None = None,
    seg_mask: np.ndarray | None = None,
    container_mask: np.ndarray | None = None,
    clip_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    source_roi = source[y1:y2, x1:x2]
    target_roi = target[y1:y2, x1:x2]
    if source_roi.size == 0 or target_roi.size == 0:
        return None

    # This repair mask assumes normal polarity (dark ink on a light/plain
    # background) throughout -- its stroke/background split treats DARK
    # pixels as the strokes to erase. A reversed-polarity container (light
    # glyphs on a dark/black bubble fill) inverts that assumption: it would
    # read the dark FILL as "background" and produce a no-op or wrong-target
    # repair, silently reporting a cleaned region that still shows the
    # source glyphs (verified: new_sample_5's merged black-bubble
    # constraint). Decline so the caller falls through to the
    # reverse-dark-balloon path built for this case.
    if _reverse_dark_balloon_candidate(source, coords):
        return None

    model_available = anime_model is not None and anime_device is not None
    repair = _precise_floating_text_repair_mask(source, coords, allow_dense=model_available)
    if repair is None and seg_mask is not None and model_available:
        # A regular halftone/screentone pattern (original/sample6's dotted
        # caption panel) is hundreds of small dark dot components; the
        # per-pixel stroke detector above cannot tell one from real glyph
        # ink, and each dot's halo/antialias growth compounds until the
        # combined mask covers most of the region -- correctly triggering
        # the density-cap decline above, but leaving no path to erase the
        # real text at all. The Step-1 segmentation model does not have
        # this confusion (verified: it isolates exactly the glyph pixels on
        # this exact panel, zero dot bleed), so when the plain repair
        # mask fails, fall back to the model's own text mask instead of
        # giving up -- the caller then flat-fills or model-repairs only the
        # true glyphs, never the dots between them.
        seg_repair = _refined_floating_source_mask(source, seg_mask, coords)
        if seg_repair is not None and int(np.count_nonzero(seg_repair > 0)) >= 10:
            repair = seg_repair
    if repair is None:
        return None
    if allowed_mask is not None and allowed_mask.shape[:2] == repair.shape[:2]:
        # The window is a bounding box over several erase boxes; artwork that
        # sits BETWEEN those boxes (a face between a SFX and a text column on
        # new_sample_13) is inside the window but must never seed the repair
        # mask — its dark features read as "strokes" to the mask builder.
        repair = cv2.bitwise_and(repair, allowed_mask)
        if int(np.count_nonzero(repair > 0)) < 10:
            return None

    height, width = repair.shape[:2]
    area = max(1, height * width)
    repair_density = float(np.count_nonzero(repair > 0)) / float(area)
    if repair_density < 0.025 or repair_density > (0.60 if model_available else 0.42):
        return None

    source_gray = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
    source_hsv = cv2.cvtColor(source_roi, cv2.COLOR_BGR2HSV)
    source_edges = cv2.Canny(source_gray, 45, 135) > 0
    low_saturation = float(np.mean(source_hsv[:, :, 1] < 180))
    # Classical TELEA/NS cannot handle saturated art, but the anime-manga LaMa
    # reconstructs colored backgrounds well with a tight stroke mask -- only
    # reject colored regions when no model is available to take them.
    if low_saturation < 0.82 and anime_model is None:
        return None
    if _floating_region_has_mixed_character_tone(source, coords, repair):
        return None

    guard = cv2.dilate(
        repair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    ).astype(bool)
    background = (
        ~guard
        & ~source_edges
        & (source_hsv[:, :, 1] < 185)
        & (source_gray >= 35)
        & (source_gray <= 246)
    )
    if int(np.count_nonzero(background)) < max(24, int(area * 0.045)):
        background = (
            ~guard
            & (source_hsv[:, :, 1] < 195)
            & (source_gray >= 35)
            & (source_gray <= 248)
        )
    background_count = int(np.count_nonzero(background))
    background_edge_density = (
        float(np.mean(source_edges[background]))
        if background_count
        else 1.0
    )
    background_luma_std = (
        float(np.std(source_gray[background].astype(np.float32)))
        if background_count
        else 99.0
    )

    fitted = None
    if background_count >= max(24, int(area * 0.035)):
        background_pixels = source_roi[background].astype(np.float32)
        if float(np.mean(np.std(background_pixels, axis=0))) <= 40.0:
            bg_y, bg_x = np.where(background)
            values = source_roi[bg_y, bg_x].astype(np.float32)
            if len(bg_x) > 6000:
                step = max(1, len(bg_x) // 6000)
                bg_x = bg_x[::step]
                bg_y = bg_y[::step]
                values = values[::step]
            design = np.column_stack(
                [
                    bg_x.astype(np.float32) / float(max(1, width - 1)),
                    bg_y.astype(np.float32) / float(max(1, height - 1)),
                    np.ones_like(bg_x, dtype=np.float32),
                ]
            )
            grid_y, grid_x = np.indices((height, width), dtype=np.float32)
            grid_design = np.column_stack(
                [
                    grid_x.reshape(-1).astype(np.float32) / float(max(1, width - 1)),
                    grid_y.reshape(-1).astype(np.float32) / float(max(1, height - 1)),
                    np.ones(area, dtype=np.float32),
                ]
            )
            fitted = np.empty_like(source_roi, dtype=np.uint8)
            for channel in range(3):
                coeffs, _, _, _ = np.linalg.lstsq(design, values[:, channel], rcond=None)
                fitted[:, :, channel] = np.clip(
                    (grid_design @ coeffs).reshape(height, width),
                    0,
                    255,
                ).astype(np.uint8)
            fitted = cv2.GaussianBlur(fitted, (0, 0), 0.45)

    commit = cv2.morphologyEx(
        repair,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    commit = cv2.dilate(
        commit,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    commit_density = float(np.count_nonzero(commit > 0)) / float(area)
    commit_cap = 0.70 if model_available else 0.58
    if commit_density > commit_cap:
        commit = repair
        commit_density = repair_density
    if commit_density < 0.025 or commit_density > commit_cap:
        return None

    # Thick airbrushed halos extend past the near-core halo band the mask
    # builder captures; grow the commit mask through contiguous bright
    # low-saturation pixels so no white rim survives around erased lettering,
    # then push the boundary a little further so anti-aliased halo edges
    # (which sit below the brightness threshold) cannot survive as glassy
    # glyph-shaped rims that steer the model into re-embossing the text.
    bright_halo = (source_gray >= 178) & (source_hsv[:, :, 1] <= 130)
    grow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grown = commit.copy()
    for _ in range(8):
        grown = cv2.dilate(grown, grow_kernel)
        grown[~bright_halo] = 0
    commit = np.maximum(commit, grown)
    backing = None
    if model_available:
        # An artist backing panel is erased WHOLE so the model rebuilds the
        # raw art beneath it (partial erasure leaves its soft edges as tone
        # discontinuities). Real speech bubbles are excluded by their outline.
        backing = _floating_artist_backing_silhouette(source, coords)
        if backing is not None:
            commit = np.maximum(commit, backing)
        commit = cv2.dilate(
            commit, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
    if backing is None:
        # Without a backing panel there is no reason for the mask to stray far
        # from the lettering itself: bound halo growth to a fixed reach around
        # the detected glyph strokes so it can never crawl along a bright wall
        # or floor into surrounding artwork.
        reach = cv2.dilate(
            repair, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        )
        commit[reach == 0] = 0

    if container_mask is not None:
        # Applied here -- after every density/accept-decline gate above has
        # already judged the region on its true, unclipped extent -- so a
        # well-fitting container (the common case) can never flip a
        # borderline accept/decline outcome by shrinking the mask the gates
        # see. This only prevents the actual PAINT from crossing into
        # neighboring art the container tracer deliberately excluded (e.g. a
        # character silhouette): an erase_box drawn wider than its own
        # region's traced container is otherwise unconstrained here.
        region_container = container_mask[y1:y2, x1:x2]
        if np.any(region_container):
            commit = cv2.bitwise_and(commit, region_container)
            if int(np.count_nonzero(commit > 0)) < 10:
                return None

    if clip_mask is not None:
        # Per-constraint hard bound (Step 6's semantic_rescue_region). Applied at
        # the same point, and for the same reason, as the container clip above.
        #
        # Unlike container_mask there is deliberately NO `if np.any(...)` escape:
        # container_mask is a page-level union that may legitimately not cover a
        # given region, so it has to no-op when absent. clip_mask is passed only
        # for a constraint that exists BECAUSE step 1's semantic detector vouched
        # for that specific region -- outside it there is nothing this cleanup is
        # entitled to repaint, so an empty intersection must fail closed, not
        # silently paint unbounded.
        region_clip = clip_mask[y1:y2, x1:x2]
        # Provably-flat interior needs no generative reconstruction -- take the
        # deterministic fill and skip the model entirely (see the helper's docstring
        # for why the clip alone is not enough on a small bubble in dense art).
        if seg_mask is not None:
            flat = _semantic_clip_flat_fill(source, target, coords, region_clip, seg_mask)
            if flat is not None:
                return flat
        commit = cv2.bitwise_and(commit, region_clip)
        if int(np.count_nonzero(commit > 0)) < 10:
            return None

    # A confirmed-flat background (near-zero luma std, no edges at all) needs
    # no generative reconstruction -- the planar fit already computed above
    # is both correct and seam-free. Dense multi-character text (a narration
    # line) commonly leaves `commit` as several disjoint dilated islands, one
    # per character/word; routing that through the model has each island
    # independently reconstructed with slightly different local tone, and the
    # seams between islands show up as a scalloped/wavy pattern across what
    # should be one flat panel (verified: new_sample_14's top and bottom
    # narration bands, both plain flat strips with background_luma_std ~1,
    # came out wavy through the model branch below; a real gradient/art panel
    # on the same page measured std~15 and is correctly excluded by this
    # threshold, so it still goes through the model as before).
    if fitted is not None and background_luma_std <= 14.0 and background_edge_density <= 0.05:
        write_mask = commit > 0
        target_roi[write_mask] = fitted[write_mask]
        _local_repair_tone_match(source, target, coords, commit)
        return commit

    # Prefer the anime-manga LaMa whenever it is available: with a tight
    # stroke mask it reconstructs door slats, tatami lines, kimono gradients,
    # and colored art that classical TELEA/NS inpainting turns into milky
    # smears. The old gate limited the model to near-grayscale structured
    # regions (low_saturation >= 0.92), which routed every colored page to
    # the smear path.
    structured_background = model_available
    if structured_background:
        full_model_mask = np.zeros(source.shape[:2], dtype=np.uint8)
        full_model_mask[y1:y2, x1:x2] = commit
        before = target_roi.copy()
        _anime_lama_local_crop(
            anime_model,
            anime_device,
            target,
            full_model_mask,
            source.shape[0],
            source.shape[1],
            x1,
            y1,
            x2,
            y2,
        )
        changed = np.any(before != target[y1:y2, x1:x2], axis=2)
        if int(np.count_nonzero(changed & (commit > 0))) >= max(8, int(np.count_nonzero(commit > 0) * 0.08)):
            commit = _reinpaint_ghost_residue(
                anime_model, anime_device, target, coords, commit,
                # The one path that can still GROW the mask and repaint after the
                # clip above, so it has to inherit the same bound. None here keeps
                # today's behaviour exactly when no clip is in play.
                container_mask=clip_mask,
            )
            return commit
        target[y1:y2, x1:x2] = before

    full_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = commit
    try:
        telea = cv2.inpaint(target, full_mask, 3.0, cv2.INPAINT_TELEA)[y1:y2, x1:x2]
        navier = cv2.inpaint(target, full_mask, 2.6, cv2.INPAINT_NS)[y1:y2, x1:x2]
    except cv2.error:
        return None
    inpainted = cv2.addWeighted(telea, 0.64, navier, 0.36, 0)
    if fitted is not None:
        background_std = float(np.mean(np.std(source_roi[background].astype(np.float32), axis=0))) if background_count else 99.0
        if background_std <= 18.0 and float(np.mean(source_edges)) <= 0.10:
            inpainted = cv2.addWeighted(inpainted, 0.82, fitted, 0.18, 0)

    write_mask = commit > 0
    target_roi[write_mask] = inpainted[write_mask]
    _local_repair_tone_match(source, target, coords, commit)
    return commit


def _outlined_floating_text_repair(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    seed_roi: np.ndarray,
) -> np.ndarray | None:
    repair = _outlined_floating_source_mask(source, coords, seed_roi)
    if repair is None:
        return None

    x1, y1, x2, y2 = coords
    density = float(np.count_nonzero(repair > 0)) / float(max(1, repair.size))
    if density <= 0.58:
        repaired = _stroke_only_inpaint_repair(source, target, coords, repair, radius=2.0)
        return repaired if repaired is not None else None

    # Dense outlined manga text needs the whole halo removed. Generic semantic
    # models hallucinate here, so use small-radius classical inpaint with a large
    # context crop and only commit masked pixels.
    pad = max(18, min(48, int(max(x2 - x1, y2 - y1) * 0.14)))
    crop_x1 = max(0, x1 - pad)
    crop_y1 = max(0, y1 - pad)
    crop_x2 = min(source.shape[1], x2 + pad)
    crop_y2 = min(source.shape[0], y2 + pad)
    crop_img = target[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    crop_mask = np.zeros(crop_img.shape[:2], dtype=np.uint8)
    crop_mask[y1 - crop_y1:y2 - crop_y1, x1 - crop_x1:x2 - crop_x1] = repair

    try:
        telea = cv2.inpaint(crop_img, crop_mask, 2.0, cv2.INPAINT_TELEA)
        navier = cv2.inpaint(crop_img, crop_mask, 1.5, cv2.INPAINT_NS)
    except cv2.error:
        return None
    blended = cv2.addWeighted(telea, 0.76, navier, 0.24, 0)
    target_view = target[crop_y1:crop_y2, crop_x1:crop_x2]
    target_view[crop_mask > 0] = blended[crop_mask > 0]
    _local_repair_tone_match(source, target, coords, repair)
    return repair


def _halftone_outlined_text_model_repair(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    seed_roi: np.ndarray,
    anime_model,
    anime_device,
) -> np.ndarray | None:
    if anime_model is None:
        return None

    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or seed_roi.size == 0:
        return None

    height, width = roi.shape[:2]
    area = max(1, height * width)
    if area < 3200:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    low_saturation = float(np.mean(hsv[:, :, 1] < 170))
    bright_fraction = float(np.mean(gray > 218))
    mid_fraction = float(np.mean((gray >= 70) & (gray <= 218)))
    dark_fraction = float(np.mean(gray < 70))
    edge_density = float(np.mean(cv2.Canny(gray, 45, 135) > 0))
    luma_std = float(np.std(gray.astype(np.float32)))

    tall_caption = height >= 90 and height >= width * 1.12
    safe_dense_halftone_caption = (
        tall_caption
        and low_saturation >= 0.94
        and dark_fraction <= 0.16
        and mid_fraction >= 0.32
        and luma_std >= 48.0
    )
    safe_wide_dotted_caption = (
        width >= max(140, int(height * 1.90))
        and height >= 40
        and low_saturation >= 0.94
        and bright_fraction >= 0.52
        and mid_fraction >= 0.10
        and dark_fraction <= 0.22
        and 0.050 <= edge_density <= 0.28
        and luma_std >= 52.0
        and not _floating_region_has_mixed_character_tone(source, coords, seed_roi)
    )
    halftone_or_gradient = (
        low_saturation >= 0.84
        and 0.18 <= bright_fraction <= 0.72
        and (mid_fraction >= 0.24 or safe_wide_dotted_caption)
        and dark_fraction <= 0.18
        and edge_density >= 0.055
        and edge_density <= (0.31 if (safe_dense_halftone_caption or safe_wide_dotted_caption) else 0.26)
        and luma_std >= 32.0
    )
    if not halftone_or_gradient:
        return None

    repair = _outlined_floating_source_mask(source, coords, seed_roi)
    if repair is None:
        return None
    if safe_dense_halftone_caption:
        source_only_seed = _refined_floating_source_mask(
            source,
            np.zeros(source.shape[:2], dtype=np.uint8),
            coords,
        )
        source_only_repair = _outlined_floating_source_mask(
            source,
            coords,
            source_only_seed,
        )
        if source_only_repair is not None:
            source_only_density = float(np.count_nonzero(source_only_repair > 0)) / float(area)
            repair_density = float(np.count_nonzero(repair > 0)) / float(area)
            if repair_density + 0.10 <= source_only_density <= 0.88:
                repair = source_only_repair
    repair_count = int(np.count_nonzero(repair > 0))
    repair_density = repair_count / float(area)
    max_repair_density = 0.88 if safe_dense_halftone_caption else (0.82 if safe_wide_dotted_caption else 0.74)
    if repair_count < 24 or repair_density < 0.16 or repair_density > max_repair_density:
        return None
    if repair_density > 0.55 and _floating_region_is_art_sensitive(source, coords, seed_roi):
        if (
            not safe_dense_halftone_caption
            or _floating_region_has_mixed_character_tone(source, coords, seed_roi)
        ):
            return None

    region_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    region_mask[y1:y2, x1:x2] = repair
    before = target[y1:y2, x1:x2].copy()
    _anime_lama_local_crop(
        anime_model,
        anime_device,
        target,
        region_mask,
        source.shape[0],
        source.shape[1],
        x1,
        y1,
        x2,
        y2,
    )
    changed = np.any(before != target[y1:y2, x1:x2], axis=2)
    changed_mask = ((changed & (repair > 0)).astype(np.uint8)) * 255
    if int(np.count_nonzero(changed_mask > 0)) < 8:
        return None
    if not (
        (safe_dense_halftone_caption and repair_density > 0.55)
        or (safe_wide_dotted_caption and repair_density > 0.34)
    ):
        residual_cleanup = _opencv_cleanup_visible_source_residual(source, target, coords, repair)
        if residual_cleanup is not None:
            repair = cv2.bitwise_or(repair, residual_cleanup)
    restore_mask = _restore_art_lines_crossing_repair_mask(source, target, coords, repair, seed_roi)
    if restore_mask is not None:
        repair = cv2.bitwise_or(repair, restore_mask)
    return repair


def _safe_outlined_halftone_caption_region(
    source: np.ndarray,
    coords: tuple[int, int, int, int],
    seed_roi: np.ndarray,
) -> bool:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or seed_roi.size == 0:
        return False
    height, width = roi.shape[:2]
    if height < 90 or height < width * 1.12:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    low_saturation = float(np.mean(hsv[:, :, 1] < 170))
    bright_fraction = float(np.mean(gray > 218))
    mid_fraction = float(np.mean((gray >= 70) & (gray <= 218)))
    dark_fraction = float(np.mean(gray < 70))
    edge_density = float(np.mean(cv2.Canny(gray, 45, 135) > 0))
    luma_std = float(np.std(gray.astype(np.float32)))
    return (
        low_saturation >= 0.94
        and 0.18 <= bright_fraction <= 0.72
        and mid_fraction >= 0.32
        and dark_fraction <= 0.16
        and 0.055 <= edge_density <= 0.31
        and luma_std >= 48.0
        and not _floating_region_has_mixed_character_tone(source, coords, seed_roi)
    )


def _opencv_cleanup_visible_source_residual(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    repair_mask: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    source_roi = source[y1:y2, x1:x2]
    target_roi = target[y1:y2, x1:x2]
    if source_roi.size == 0 or target_roi.size == 0 or repair_mask.size == 0:
        return None

    repair = repair_mask > 0
    repair_count = int(np.count_nonzero(repair))
    if repair_count < 24:
        return None

    source_gray = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
    source_hsv = cv2.cvtColor(source_roi, cv2.COLOR_BGR2HSV)
    low_saturation = float(np.mean(source_hsv[:, :, 1] < 175))
    edge_density = float(np.mean(cv2.Canny(source_gray, 45, 135) > 0))
    if low_saturation < 0.82 or edge_density < 0.055:
        return None

    source_dark = repair & (source_gray < 135)
    dark_count = int(np.count_nonzero(source_dark))
    if dark_count < 28:
        return None
    visible_dark = source_dark & (target_gray < 150)
    visible_count = int(np.count_nonzero(visible_dark))
    visible_ratio = visible_count / float(max(1, dark_count))
    repair_visible_ratio = visible_count / float(max(1, repair_count))
    if visible_count < 24 or (visible_ratio < 0.18 and repair_visible_ratio < 0.045):
        return None

    cleanup = cv2.dilate(
        repair.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    full_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = cleanup
    repaired = cv2.inpaint(target, full_mask, 3.0, cv2.INPAINT_NS)
    target[y1:y2, x1:x2] = repaired[y1:y2, x1:x2]
    return cleanup


def _refined_floating_source_mask(
    source: np.ndarray,
    seg_mask: np.ndarray,
    coords: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((max(1, y2 - y1), max(1, x2 - x1)), dtype=np.uint8)

    height, width = roi.shape[:2]
    area = max(1, height * width)
    candidates: list[np.ndarray] = []
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    high_contrast_mask = _high_contrast_light_text_block_mask(roi)
    if high_contrast_mask is not None:
        high_contrast_density = float(np.count_nonzero(high_contrast_mask > 0)) / float(area)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        background = (high_contrast_mask <= 0) & (gray > 150) & (hsv[:, :, 1] < 170)
        if high_contrast_density <= 0.92 and np.count_nonzero(background) >= max(4, int(area * 0.015)):
            return high_contrast_mask

    seg_roi = seg_mask[y1:y2, x1:x2]
    _, seg_roi = cv2.threshold(seg_roi, 127, 255, cv2.THRESH_BINARY)
    floating_roi = _floating_text_erase_roi(source, coords)
    dark_roi = _extract_dark_text_strokes(source, coords)

    seg_density = np.count_nonzero(seg_roi > 0) / float(area)
    floating_density = np.count_nonzero(floating_roi > 0) / float(area)
    dark_density = np.count_nonzero(dark_roi > 0) / float(area)

    bright_background_fraction = float(np.mean((gray > 188) & (hsv[:, :, 1] < 155)))
    edge_density = float(np.mean(cv2.Canny(gray, 45, 135) > 0))
    bright_halftone_background = (
        bright_background_fraction >= 0.42
        and float(np.std(gray.astype(np.float32))) >= 24.0
        and edge_density >= 0.028
    )

    if np.count_nonzero(seg_roi > 0) >= 6 and seg_density <= 0.44:
        candidates.append(seg_roi)

    if np.count_nonzero(floating_roi > 0) >= 6 and floating_density <= 0.44:
        candidates.append(floating_roi)

    if np.count_nonzero(dark_roi > 0) >= 6 and dark_density <= 0.44:
        candidates.append(dark_roi)

    if not candidates:
        return np.zeros((height, width), dtype=np.uint8)

    mask = max(candidates, key=lambda item: np.count_nonzero(item > 0)).copy()
    bright_fraction = float(np.mean((gray > 178) & (hsv[:, :, 1] < 135)))
    dark_fraction = float(np.mean(gray < 112))
    if (
        area <= 22000
        and bright_fraction >= 0.24
        and dark_fraction >= 0.12
        and float(np.std(gray.astype(np.float32))) >= 42.0
        and float(np.count_nonzero(mask > 0)) / float(area) <= 0.62
    ):
        return mask

    dark_fraction = float(np.mean(gray < 130))
    screen_like = dark_fraction >= 0.34 and height / float(max(1, width)) <= 1.85

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    large_component_centers: list[tuple[float, float]] = []
    component_meta = []
    tiny_round_component_count = 0
    for label in range(1, component_count):
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        fill_ratio = component_area / float(max(1, cw * ch))
        aspect = cw / float(max(1, ch))
        center = (cx + cw / 2.0, cy + ch / 2.0)
        component_meta.append((label, component_area, cx, cy, cw, ch, fill_ratio, aspect, center))
        if component_area >= 28 or cw >= 10 or ch >= 10 or fill_ratio < 0.30:
            large_component_centers.append(center)
        elif component_area <= 28 and cw <= 10 and ch <= 10 and 0.45 <= aspect <= 2.2 and fill_ratio >= 0.34:
            tiny_round_component_count += 1

    filtered = np.zeros_like(mask)
    for label, component_area, cx, cy, cw, ch, fill_ratio, aspect, center in component_meta:
        if component_area < 3:
            continue
        if component_area > max(2600, int(area * 0.18)):
            continue
        if cw > int(width * 0.92) and ch <= 10:
            continue
        if ch > int(height * 0.92) and cw <= 8:
            continue
        long_axis = max(cw, ch)
        short_axis = max(1, min(cw, ch))
        axis_ratio = long_axis / float(short_axis)
        elongated_low_fill = (
            long_axis >= 18
            and axis_ratio >= 2.15
            and fill_ratio <= 0.26
            and component_area <= max(240, int(area * 0.018))
        )
        diagonal_line_art = (
            elongated_low_fill
            and 0.16 <= aspect <= 6.25
            and (
                component_area <= 96
                or fill_ratio <= 0.18
                or long_axis >= 34
            )
        )
        if diagonal_line_art:
            continue
        if screen_like and cy < int(height * 0.20) and ch < int(height * 0.20):
            continue
        if (
            bright_halftone_background
            and tiny_round_component_count >= 8
            and component_area <= 28
            and cw <= 10
            and ch <= 10
            and 0.45 <= aspect <= 2.2
            and fill_ratio >= 0.34
            and large_component_centers
        ):
            nearest_large = min(
                math.hypot(center[0] - other[0], center[1] - other[1])
                for other in large_component_centers
            )
            if nearest_large > max(13.0, float(min(width, height)) * 0.18):
                continue
        filtered[labels == label] = 255

    if np.count_nonzero(filtered > 0) < 6:
        return np.zeros((height, width), dtype=np.uint8)

    kernel_size = 3 if bright_halftone_background else (5 if screen_like else 3)
    return cv2.dilate(
        filtered,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        iterations=1,
    )


def _fill_dark_surface_source_strokes(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0 or np.count_nonzero(mask_roi > 0) < 6:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    dark_fraction = float(np.mean(gray < 140))
    if dark_fraction < 0.30:
        return None

    cleanup = (mask_roi > 0).astype(np.uint8) * 255
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(cleanup, connectivity=8)
    changed_mask = np.zeros_like(cleanup)
    target_roi = target[y1:y2, x1:x2]
    target_patch = target_roi.copy()

    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 4 or area > max(2600, int(cleanup.size * 0.18)):
            continue

        pad = max(7, min(22, int(max(cw, ch) * 0.55)))
        bx1 = max(0, cx - pad)
        by1 = max(0, cy - pad)
        bx2 = min(roi.shape[1], cx + cw + pad)
        by2 = min(roi.shape[0], cy + ch + pad)

        local_component = labels[by1:by2, bx1:bx2] == label
        local_cleanup = cleanup[by1:by2, bx1:bx2] > 0
        local_gray = gray[by1:by2, bx1:bx2]

        component_u8 = local_component.astype(np.uint8)
        ring = cv2.dilate(
            component_u8,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    max(5, min(17, (pad // 2) * 2 + 1)),
                    max(5, min(17, (pad // 2) * 2 + 1)),
                ),
            ),
            iterations=1,
        ) > 0
        inner = cv2.dilate(
            component_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        ) > 0
        background = ring & ~inner & (~local_cleanup) & (local_gray < 246)
        if np.count_nonzero(background) < 10:
            background = (~local_cleanup) & (local_gray < 210)
        if np.count_nonzero(background) < 8:
            continue

        pixels = roi[by1:by2, bx1:bx2][background].astype(np.float32)
        median_color = np.median(pixels, axis=0)
        distances = np.linalg.norm(pixels - median_color, axis=1)
        inliers = pixels[distances < 55.0]
        if len(inliers) < 8:
            continue

        component_mask = local_component.astype(np.uint8) * 255
        view = target_patch[by1:by2, bx1:bx2]
        local_luma = cv2.cvtColor(roi[by1:by2, bx1:bx2], cv2.COLOR_BGR2GRAY)
        varied_background = float(np.std(local_luma[background].astype(np.float32))) > 28.0
        if varied_background:
            fallback_fill = np.median(inliers, axis=0).astype(np.uint8)
            bg_rows = np.where(background)[0]
            for local_row in range(component_mask.shape[0]):
                row_mask = component_mask[local_row] > 0
                if not np.any(row_mask):
                    continue
                near_rows = np.abs(bg_rows - local_row) <= 5
                if np.count_nonzero(near_rows) >= 6:
                    row_background = np.zeros(background.shape, dtype=bool)
                    row_background[
                        np.where(background)[0][near_rows],
                        np.where(background)[1][near_rows],
                    ] = True
                    view[local_row][row_mask] = np.median(
                        roi[by1:by2, bx1:bx2][row_background],
                        axis=0,
                    ).astype(np.uint8)
                else:
                    view[local_row][row_mask] = fallback_fill
        else:
            view[component_mask > 0] = np.median(inliers, axis=0).astype(np.uint8)
        changed_mask[by1:by2, bx1:bx2] = np.maximum(changed_mask[by1:by2, bx1:bx2], component_mask)

    coverage = float(np.count_nonzero((changed_mask > 0) & (cleanup > 0))) / float(
        max(1, np.count_nonzero(cleanup > 0))
    )
    if np.count_nonzero(changed_mask > 0) >= 6 and coverage >= 0.55:
        target_roi[:, :] = target_patch
        return changed_mask

    cleanup_bool = cleanup > 0
    repair_density = float(np.count_nonzero(cleanup_bool)) / float(max(1, cleanup_bool.size))
    if repair_density >= 0.18:
        background = (~cleanup_bool) & (gray < 155)
        if np.count_nonzero(background) >= max(60, int(cleanup_bool.size * 0.08)):
            deep_background = background & (gray < 105)
            if np.count_nonzero(deep_background) >= max(36, int(np.count_nonzero(background) * 0.18)):
                background = deep_background
            pixels = roi[background].astype(np.float32)
            median_color = np.median(pixels, axis=0)
            distances = np.linalg.norm(pixels - median_color, axis=1)
            inliers = pixels[distances < 62.0]
            if len(inliers) >= 40 and float(np.mean(np.std(inliers, axis=0))) <= 34.0:
                fill = np.median(inliers, axis=0).astype(np.uint8)
                target_roi[cleanup_bool] = fill
                return cleanup
    return None


def _fill_reverse_dark_balloon_text(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    seed_roi: np.ndarray,
    allowed_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    height, width = roi.shape[:2]
    area = max(1, height * width)
    if area < 900:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    dark_background = (gray < 112) & (saturation < 180)
    if allowed_mask is not None and allowed_mask.shape[:2] == dark_background.shape[:2]:
        # coords is a container's BOUNDING RECTANGLE, not its organic
        # outline -- corners of that rectangle routinely fall outside the
        # actual bubble (hair, screentone) and must never be treated as
        # "more of the bubble to flatten" (verified: a widened green_box
        # let this function paint the bubble's fill color over adjoining
        # hair texture on new_sample_5, destroying it).
        dark_background &= allowed_mask > 0
    dark_fraction = float(np.mean(dark_background))
    if dark_fraction < 0.58:
        return None

    dark_values = gray[dark_background]
    if dark_values.size < max(32, int(area * 0.18)):
        return None
    if float(np.std(dark_values.astype(np.float32))) > 34.0:
        return None
    uniform_dark_balloon = dark_fraction >= 0.68
    dark_support = cv2.dilate(
        dark_background.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ) > 0

    bright_seed = (gray > 128) & (saturation < 185) & dark_support
    if seed_roi.size == gray.shape[0] * gray.shape[1]:
        seed = seed_roi > 0
        if np.any(seed):
            near_seed = cv2.dilate(
                seed.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                iterations=1,
            ) > 0
            bright_seed |= near_seed & dark_support & (gray > 104) & (saturation < 205)

    bright_seed = cv2.morphologyEx(
        bright_seed.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats((bright_seed > 0).astype(np.uint8), 8)
    cleanup = np.zeros_like(bright_seed)
    for label in range(1, component_count):
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if component_area < 4 or component_area > max(2600, int(area * 0.42)):
            continue
        if cw > int(width * 0.94) and ch <= 5:
            continue
        if ch > int(height * 0.98) and cw <= 3:
            continue
        component = labels == label
        near = cv2.dilate(
            component.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        ) > 0
        if int(np.count_nonzero(near & dark_background)) < max(6, int(component_area * 0.20)):
            continue
        cleanup[component] = 255

    cleanup = cv2.dilate(
        cleanup,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    if np.count_nonzero(cleanup > 0) >= 10:
        dark_median = float(np.median(dark_values.astype(np.float32)))
        halo = cv2.dilate(
            (cleanup > 0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        ) > 0
        anti_alias_residual = (
            halo
            & dark_support
            & (gray > max(42.0, dark_median + 10.0))
            & (saturation < 210)
        )
        cleanup[anti_alias_residual] = 255
        if dark_fraction >= 0.68:
            broad_reverse_residual = (
                (gray > max(36.0, dark_median + 8.0))
                & (gray < 214)
                & (saturation < 210)
                & dark_support
            )
            cleanup[broad_reverse_residual] = 255
            outline_halo = cv2.dilate(
                (cleanup > 0).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            ) > 0
            dark_outline_residual = (
                outline_halo
                & dark_support
                & (gray < max(6.0, dark_median - 4.0))
                & (saturation < 210)
            )
            cleanup[dark_outline_residual] = 255
            if uniform_dark_balloon:
                uniform_dark_fill = (
                    ((gray < 118) | (cleanup > 0))
                    & (saturation < 210)
                    & dark_support
                )
                cleanup[uniform_dark_fill] = 255
        cleanup = cv2.morphologyEx(
            cleanup,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        cleanup[~dark_support] = 0
    cleanup_count = int(np.count_nonzero(cleanup > 0))
    density = cleanup_count / float(area)
    if cleanup_count < 10 or (density > 0.52 and not uniform_dark_balloon):
        return None

    target_roi = target[y1:y2, x1:x2]
    background = dark_background if uniform_dark_balloon else dark_background & (cleanup <= 0)
    if int(np.count_nonzero(background)) < max(30, int(area * 0.08)):
        return None
    pixels = target_roi[background].astype(np.float32)
    median_color = np.median(pixels, axis=0)
    distances = np.linalg.norm(pixels - median_color, axis=1)
    inliers = pixels[distances < 48.0]
    if len(inliers) < 24:
        return None

    fill = np.median(inliers, axis=0).astype(np.uint8)
    target_roi[cleanup > 0] = fill
    return cleanup


def _reverse_dark_balloon_candidate(
    source: np.ndarray,
    coords: tuple[int, int, int, int],
) -> bool:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    dark = (gray < 112) & (hsv[:, :, 1] < 185)
    dark_fraction = float(np.mean(dark))
    if dark_fraction < 0.62:
        return False
    dark_values = gray[dark]
    if dark_values.size < max(32, int(gray.size * 0.22)):
        return False
    bright_fraction = float(np.mean((gray > 218) & (hsv[:, :, 1] < 185)))
    edge_density = float(np.mean(cv2.Canny(gray, 45, 135) > 0))
    return (
        float(np.median(dark_values)) <= 76.0
        and float(np.std(dark_values.astype(np.float32))) <= 34.0
        and bright_fraction <= 0.18
        and edge_density <= 0.18
    )


def _fill_local_tone_source_strokes(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0 or np.count_nonzero(mask_roi > 0) < 6:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 135) > 0
    cleanup = (mask_roi > 0).astype(np.uint8) * 255
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(cleanup, connectivity=8)
    changed_mask = np.zeros_like(cleanup)
    target_roi = target[y1:y2, x1:x2]
    target_patch = target_roi.copy()

    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 4 or area > max(2200, int(cleanup.size * 0.14)):
            continue

        pad = max(6, min(18, int(max(cw, ch) * 0.50)))
        bx1 = max(0, cx - pad)
        by1 = max(0, cy - pad)
        bx2 = min(roi.shape[1], cx + cw + pad)
        by2 = min(roi.shape[0], cy + ch + pad)

        local_component = labels[by1:by2, bx1:bx2] == label
        local_cleanup = cleanup[by1:by2, bx1:bx2] > 0
        local_gray = gray[by1:by2, bx1:bx2]
        local_edges = edges[by1:by2, bx1:bx2]

        ring = cv2.dilate(
            local_component.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        ) > 0
        background = ring & (~local_cleanup)
        if np.count_nonzero(background) < 8:
            background = ~local_cleanup
        if np.count_nonzero(background) < 12:
            continue
        mid_tone_background = background & (local_gray >= 72) & (local_gray <= 220)
        if np.count_nonzero(mid_tone_background) >= max(10, int(np.count_nonzero(background) * 0.18)):
            background = mid_tone_background
        else:
            non_text_background = background & (local_gray < 245)
            if np.count_nonzero(non_text_background) >= 10:
                background = non_text_background

        bg_edge_density = float(np.mean(local_edges[background]))
        if bg_edge_density > 0.22:
            continue

        pixels = roi[by1:by2, bx1:bx2][background].astype(np.float32)
        median_color = np.median(pixels, axis=0)
        distances = np.linalg.norm(pixels - median_color, axis=1)
        inliers = pixels[distances < 62.0]
        if len(inliers) < 10:
            continue
        if float(np.mean(np.std(inliers, axis=0))) > 42.0:
            continue
        if float(np.mean(np.median(inliers, axis=0))) > 244.0 and float(np.median(local_gray[~local_cleanup])) < 238.0:
            continue

        component_mask = local_component.astype(np.uint8) * 255
        view = target_patch[by1:by2, bx1:bx2]
        view[component_mask > 0] = np.median(inliers, axis=0).astype(np.uint8)
        changed_mask[by1:by2, bx1:bx2] = np.maximum(changed_mask[by1:by2, bx1:bx2], component_mask)

    coverage = float(np.count_nonzero((changed_mask > 0) & (cleanup > 0))) / float(
        max(1, np.count_nonzero(cleanup > 0))
    )
    if np.count_nonzero(changed_mask > 0) >= 6 and coverage >= 0.55:
        target_roi[:, :] = target_patch
        return changed_mask
    return None


def _fill_smooth_tone_caption_strokes(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> np.ndarray | None:
    """Repair vertical floating captions on smooth tone/gradient panels.

    This path is deliberately narrower than the generic floating inpaint path.
    It is for source text drawn on mostly flat gray/toned backgrounds with a
    bright outline around dark glyphs. It fills only the detected strokes and
    outline halo from nearby row-wise tone samples, so it does not create an
    artificial speech bubble or translucent badge.
    """

    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0 or np.count_nonzero(mask_roi > 0) < 8:
        return None

    height, width = roi.shape[:2]
    if height < 72 or height < width * 1.35:
        return None
    if width > max(180, int(source.shape[1] * 0.18)):
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 135) > 0
    seed = mask_roi > 0
    seed_soft = cv2.dilate(
        seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=1,
    ) > 0

    background = (~seed_soft) & (gray > 40) & (gray < 245)
    if np.count_nonzero(background) < max(40, int(width * height * 0.12)):
        return None

    bg_gray = gray[background].astype(np.float32)
    bg_median = float(np.median(bg_gray))
    bg_std = float(np.std(bg_gray))
    bg_edge_density = float(np.mean(edges[background]))
    if bg_median < 88.0 or bg_median > 236.0:
        return None
    if bg_std > 48.0 or bg_edge_density > 0.16:
        return None

    dark_threshold = max(70.0, bg_median - 62.0)
    bright_threshold = min(244.0, max(190.0, bg_median + 6.0))
    near_seed = cv2.dilate(
        seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    ) > 0
    dark_strokes = near_seed & (gray < dark_threshold)
    text_core = seed | dark_strokes
    near_core = cv2.dilate(
        seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
        iterations=1,
    ) > 0
    white_tophat = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    bright_halo = near_core & ((gray > bright_threshold) | (white_tophat > 10))

    repair = seed | dark_strokes | bright_halo
    repair = cv2.dilate(
        repair.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    repair = cv2.morphologyEx(
        repair,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (repair > 0).astype(np.uint8), connectivity=8
    )
    filtered = np.zeros_like(repair)
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        component = labels == label
        seed_overlap = int(np.count_nonzero(component & seed))
        if seed_overlap < 2:
            continue
        if area > max(3600, int(width * height * 0.92)):
            continue
        if cw > int(width * 0.95) and ch < 18:
            continue
        filtered[component] = 255

    repair_bool = filtered > 0
    if np.count_nonzero(repair_bool) < 10:
        return None
    source_coverage = float(np.count_nonzero(repair_bool & seed)) / float(
        max(1, np.count_nonzero(seed))
    )
    if source_coverage < 0.72:
        return None

    stable_background = (~repair_bool) & (gray > max(35.0, bg_median - 70.0)) & (
        gray < min(242.0, bg_median + 38.0)
    )
    stable_background &= ~edges
    if np.count_nonzero(stable_background) < max(32, int(width * height * 0.08)):
        stable_background = background & (~edges)
    if np.count_nonzero(stable_background) < 24:
        return None

    bg_y, bg_x = np.where(stable_background)
    design = np.column_stack(
        [
            bg_x.astype(np.float32) / float(max(1, width - 1)),
            bg_y.astype(np.float32) / float(max(1, height - 1)),
            np.ones_like(bg_x, dtype=np.float32),
        ]
    )
    grid_y, grid_x = np.indices((height, width))
    grid_design = np.column_stack(
        [
            grid_x.reshape(-1).astype(np.float32) / float(max(1, width - 1)),
            grid_y.reshape(-1).astype(np.float32) / float(max(1, height - 1)),
            np.ones(width * height, dtype=np.float32),
        ]
    )

    fitted = np.empty_like(roi, dtype=np.float32)
    for channel in range(3):
        values = roi[:, :, channel][stable_background].astype(np.float32)
        coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
        fitted[:, :, channel] = (grid_design @ coeffs).reshape(height, width)
    fitted = np.clip(fitted, 0, 255).astype(np.uint8)

    target_roi = target[y1:y2, x1:x2]
    target_roi[repair_bool] = fitted[repair_bool]
    return filtered


def _tone_fit_background(
    roi: np.ndarray,
    repair_mask: np.ndarray,
    *,
    prefer_low_edges: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    height, width = roi.shape[:2]
    if height == 0 or width == 0:
        return None, None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 45, 135) > 0
    background = (repair_mask <= 0) & (gray > 22) & (gray < 244) & (hsv[:, :, 1] < 185)
    if prefer_low_edges:
        edge_shell = cv2.dilate(
            edges.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        ) > 0
        low_edge_background = background & ~edge_shell
        if np.count_nonzero(low_edge_background) >= max(40, int(height * width * 0.04)):
            background = low_edge_background

    values = gray[background]
    if values.size < max(24, int(height * width * 0.018)):
        return None, None

    median = float(np.median(values))
    mad = float(np.median(np.abs(values.astype(np.float32) - median))) + 1.0
    trimmed = background & (np.abs(gray.astype(np.float32) - median) < max(34.0, mad * 3.0))
    if np.count_nonzero(trimmed) >= max(24, int(height * width * 0.018)):
        background = trimmed

    bg_values = gray[background].astype(np.float32)
    if bg_values.size < max(24, int(height * width * 0.018)):
        return None, None
    if float(np.mean(edges[background])) > 0.22:
        return None, None
    if float(np.std(bg_values)) > 72.0:
        return None, None

    bg_y, bg_x = np.where(background)
    if bg_x.size < max(24, int(height * width * 0.018)):
        return None, None

    design = np.column_stack(
        [
            bg_x.astype(np.float32) / float(max(1, width - 1)),
            bg_y.astype(np.float32) / float(max(1, height - 1)),
            np.ones_like(bg_x, dtype=np.float32),
        ]
    )
    grid_y, grid_x = np.indices((height, width))
    grid_design = np.column_stack(
        [
            grid_x.reshape(-1).astype(np.float32) / float(max(1, width - 1)),
            grid_y.reshape(-1).astype(np.float32) / float(max(1, height - 1)),
            np.ones(height * width, dtype=np.float32),
        ]
    )

    fitted = np.empty_like(roi, dtype=np.float32)
    for channel in range(3):
        samples = roi[:, :, channel][background].astype(np.float32)
        coeffs, *_ = np.linalg.lstsq(design, samples, rcond=None)
        fitted[:, :, channel] = (grid_design @ coeffs).reshape(height, width)
    return np.clip(fitted, 0, 255).astype(np.uint8), background


def _tone_fit_context_background(
    source: np.ndarray,
    coords: tuple[int, int, int, int],
    repair_mask: np.ndarray,
    *,
    padding: int,
    rowwise: bool = False,
) -> tuple[np.ndarray | None, dict[str, float]]:
    x1, y1, x2, y2 = coords
    height, width = repair_mask.shape[:2]
    if height <= 0 or width <= 0:
        return None, {}

    crop_x1 = max(0, x1 - padding)
    crop_y1 = max(0, y1 - padding)
    crop_x2 = min(source.shape[1], x2 + padding)
    crop_y2 = min(source.shape[0], y2 + padding)
    crop = source[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None, {}

    crop_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    inner_x1 = x1 - crop_x1
    inner_y1 = y1 - crop_y1
    crop_mask[inner_y1:inner_y1 + height, inner_x1:inner_x1 + width] = repair_mask

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 45, 135) > 0
    background = (crop_mask <= 0) & (gray > 24) & (gray < 242) & (hsv[:, :, 1] < 185)
    low_edge = background & ~cv2.dilate(
        edges.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ).astype(bool)
    if np.count_nonzero(low_edge) >= max(60, int(crop.shape[0] * crop.shape[1] * 0.035)):
        background = low_edge

    values = gray[background].astype(np.float32)
    if values.size < max(48, int(crop.shape[0] * crop.shape[1] * 0.018)):
        return None, {}

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) + 1.0
    trimmed = background & (np.abs(gray.astype(np.float32) - median) < max(38.0, mad * 3.2))
    if np.count_nonzero(trimmed) >= max(48, int(crop.shape[0] * crop.shape[1] * 0.018)):
        background = trimmed
        values = gray[background].astype(np.float32)

    edge_density = float(np.mean(edges[background]))
    bg_std = float(np.std(values))
    dark_fraction = float(np.mean(values < 62.0))
    bright_fraction = float(np.mean(values > 232.0))
    stats = {
        "median": float(np.median(values)),
        "std": bg_std,
        "edge_density": edge_density,
        "dark_fraction": dark_fraction,
        "bright_fraction": bright_fraction,
    }
    if edge_density > 0.24 or bg_std > 78.0:
        return None, stats

    bg_y, bg_x = np.where(background)
    if rowwise:
        fitted = np.empty((height, width, 3), dtype=np.float32)
        fallback = np.median(crop[background], axis=0).astype(np.float32)
        row_colors = np.empty((height, 3), dtype=np.float32)
        row_band = max(4, min(16, height // 18))
        for row_index in range(height):
            crop_row = inner_y1 + row_index
            near_row = np.abs(bg_y - crop_row) <= row_band
            if np.count_nonzero(near_row) >= 12:
                color = np.median(crop[bg_y[near_row], bg_x[near_row]], axis=0).astype(np.float32)
            else:
                color = fallback
            row_colors[row_index, :] = color
        if height >= 7:
            sigma_y = max(1.8, min(9.0, height / 36.0))
            row_colors = cv2.GaussianBlur(
                row_colors.reshape(height, 1, 3),
                (1, 0),
                sigmaX=0,
                sigmaY=sigma_y,
            ).reshape(height, 3)
        fitted[:, :, :] = row_colors[:, None, :]
        return np.clip(fitted, 0, 255).astype(np.uint8), stats

    design = np.column_stack(
        [
            bg_x.astype(np.float32) / float(max(1, crop.shape[1] - 1)),
            bg_y.astype(np.float32) / float(max(1, crop.shape[0] - 1)),
            np.ones_like(bg_x, dtype=np.float32),
        ]
    )
    grid_y, grid_x = np.indices((height, width))
    crop_grid_x = grid_x.reshape(-1) + inner_x1
    crop_grid_y = grid_y.reshape(-1) + inner_y1
    target_design = np.column_stack(
        [
            crop_grid_x.astype(np.float32) / float(max(1, crop.shape[1] - 1)),
            crop_grid_y.astype(np.float32) / float(max(1, crop.shape[0] - 1)),
            np.ones(height * width, dtype=np.float32),
        ]
    )

    fitted = np.empty((height, width, 3), dtype=np.float32)
    for channel in range(3):
        samples = crop[:, :, channel][background].astype(np.float32)
        coeffs, *_ = np.linalg.lstsq(design, samples, rcond=None)
        fitted[:, :, channel] = (target_design @ coeffs).reshape(height, width)
    return np.clip(fitted, 0, 255).astype(np.uint8), stats


def _fill_smooth_tone_caption_block(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0 or np.count_nonzero(mask_roi > 0) < 8:
        return None

    height, width = roi.shape[:2]
    if height < 86 or height < width * 1.25 or width > max(230, int(source.shape[1] * 0.22)):
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    seed = mask_roi > 0
    text_mask = _high_contrast_light_text_block_mask(roi)
    repair = seed.copy()
    if text_mask is not None:
        near_seed = cv2.dilate(
            seed.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
            iterations=1,
        ) > 0
        repair |= (text_mask > 0) & near_seed
    outlined = _outlined_floating_source_mask(source, coords, mask_roi)
    if outlined is not None:
        repair |= outlined > 0

    if np.count_nonzero(repair) < 8:
        return None

    repair = cv2.morphologyEx(
        repair.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    repair = cv2.dilate(
        repair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    rows, cols = np.where(repair > 0)
    if rows.size < 8:
        return None
    block = repair.copy()
    block_density = float(np.count_nonzero(block > 0)) / float(max(1, block.size))
    if block_density < 0.04 or block_density > 0.40:
        return None

    padding = max(26, min(84, int(max(width, height) * 0.28)))
    fitted, context_stats = _tone_fit_context_background(source, coords, block, padding=padding, rowwise=True)
    if fitted is None:
        return None
    context_x1 = max(0, x1 - padding)
    context_y1 = max(0, y1 - padding)
    context_x2 = min(source.shape[1], x2 + padding)
    context_y2 = min(source.shape[0], y2 + padding)
    context_gray = cv2.cvtColor(source[context_y1:context_y2, context_x1:context_x2], cv2.COLOR_BGR2GRAY)
    if float(np.mean(context_gray < 48)) > 0.18:
        return None
    if context_stats.get("dark_fraction", 1.0) > 0.12:
        return None
    if context_stats.get("std", 99.0) > 38.0 or context_stats.get("edge_density", 1.0) > 0.095:
        return None
    if not (82.0 <= context_stats.get("median", 0.0) <= 236.0):
        return None

    target_roi = target[y1:y2, x1:x2]
    alpha = cv2.GaussianBlur((block > 0).astype(np.float32), (0, 0), 4.0)
    core = cv2.erode(
        block,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ) > 0
    alpha[core] = 1.0
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    blended = (fitted.astype(np.float32) * alpha + target_roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    blend_mask = alpha[:, :, 0] > 0.02
    target_roi[blend_mask] = blended[blend_mask]
    return block


def _fill_bounded_tone_source_strokes(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0 or np.count_nonzero(mask_roi > 0) < 8:
        return None

    height, width = roi.shape[:2]
    area = max(1, height * width)
    if height < 48 or width < 22:
        return None

    seed = mask_roi > 0
    repair = seed.astype(np.uint8) * 255
    outlined = _outlined_floating_source_mask(source, coords, mask_roi)
    if outlined is not None:
        repair = outlined.copy()

    repair = cv2.morphologyEx(
        repair,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    repair = cv2.dilate(
        repair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    repair_count = int(np.count_nonzero(repair > 0))
    if repair_count < 10:
        return None
    repair_density = repair_count / float(area)
    if repair_density < 0.025 or repair_density > 0.78:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    background = repair <= 0
    if np.count_nonzero(background) < max(28, int(area * 0.04)):
        return None
    bg_gray = gray[background].astype(np.float32)
    bg_median = float(np.median(bg_gray))
    bg_std = float(np.std(bg_gray))
    bg_bright = float(np.mean(bg_gray > 232))
    bg_mid = float(np.mean((bg_gray >= 50) & (bg_gray <= 210)))
    bg_dark = float(np.mean(bg_gray < 86))
    low_saturation = float(np.mean(hsv[:, :, 1][background] < 185))
    if low_saturation < 0.82:
        return None
    if bg_bright > 0.82 and bg_mid < 0.10:
        return None
    if bg_std > 78.0:
        return None

    edges = cv2.Canny(gray, 45, 135) > 0
    bg_edge_density = float(np.mean(edges[background]))
    if bg_edge_density > 0.24:
        return None
    if bg_median < 42.0 and bg_mid < 0.18:
        return None
    if repair_density > 0.36 and (bg_std > 42.0 or bg_edge_density > 0.105 or bg_dark > 0.22):
        return None

    fitted, fit_background = _tone_fit_background(roi, repair, prefer_low_edges=True)
    if fitted is None:
        fitted, context_stats = _tone_fit_context_background(
            source,
            coords,
            repair,
            padding=max(18, min(76, int(max(width, height) * 0.22))),
        )
        if (
            fitted is not None
            and (
                context_stats.get("edge_density", 1.0) > 0.24
                or context_stats.get("std", 99.0) > 82.0
                or context_stats.get("bright_fraction", 1.0) > 0.92
            )
        ):
            fitted = None
    if fitted is None or fit_background is None:
        if fitted is None:
            return None

    target_roi = target[y1:y2, x1:x2]
    before = target_roi.copy()
    target_roi[repair > 0] = fitted[repair > 0]
    changed = np.any(before != target_roi, axis=2)
    if int(np.count_nonzero(changed & (repair > 0))) < 8:
        return None
    return repair


def _mixed_tone_outline_mask(
    source: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return None

    height, width = roi.shape[:2]
    area = max(1, height * width)
    seed = mask_roi > 0
    seed_count = int(np.count_nonzero(seed))
    if seed_count < 8:
        return None
    seed_density = seed_count / float(area)
    if seed_density > 0.48:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    seed_soft = cv2.dilate(
        seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=1,
    ) > 0
    background = (~seed_soft) & (gray < 246)
    if np.count_nonzero(background) < max(32, int(area * 0.08)):
        return None

    bg_values = gray[background].astype(np.float32)
    bg_p10, bg_p50, bg_p90 = [float(value) for value in np.percentile(bg_values, [10, 50, 90])]
    bg_range = bg_p90 - bg_p10
    bg_dark_fraction = float(np.mean(bg_values < 82))
    bg_mid_fraction = float(np.mean((bg_values >= 82) & (bg_values < 178)))
    bg_light_fraction = float(np.mean(bg_values >= 178))

    mixed_tone = (
        bg_range >= 58.0
        and bg_dark_fraction >= 0.10
        and (bg_mid_fraction >= 0.10 or bg_light_fraction >= 0.10)
    )
    if not mixed_tone:
        return None

    mostly_dark_screen = bg_dark_fraction >= 0.70 and bg_mid_fraction <= 0.10
    if mostly_dark_screen:
        return None

    white_tophat = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    bright_cutoff = min(245.0, max(205.0, bg_p50 + 42.0))
    bright_outline_raw = (gray > bright_cutoff) & ((gray > 232) | (white_tophat > 12))

    bright_outline = np.zeros_like(gray, dtype=np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        bright_outline_raw.astype(np.uint8), connectivity=8
    )
    for label in range(1, component_count):
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_border = cx <= 0 or cy <= 0 or (cx + cw) >= width or (cy + ch) >= height
        if component_area < 3:
            continue
        if touches_border and component_area > 36:
            continue
        if component_area > max(1600, int(area * 0.12)):
            continue
        if cw > int(width * 0.74) and ch <= 12:
            continue
        if ch > int(height * 0.86) and cw <= 9:
            continue
        bright_outline[labels == label] = 255

    if np.count_nonzero(bright_outline > 0) < 8:
        return None

    outline_neighborhood = cv2.dilate(
        bright_outline,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    ) > 0
    dark_cutoff = max(58.0, min(104.0, bg_p50 - 44.0))
    dark_candidates = outline_neighborhood & (gray < dark_cutoff)

    dark_glyphs = np.zeros_like(gray, dtype=np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark_candidates.astype(np.uint8), connectivity=8
    )
    for label in range(1, component_count):
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_border = cx <= 0 or cy <= 0 or (cx + cw) >= width or (cy + ch) >= height
        if component_area < 3:
            continue
        if touches_border and component_area > 18:
            continue
        if component_area > max(650, int(area * 0.05)):
            continue
        if cw > int(width * 0.48) or ch > int(height * 0.48):
            continue
        dark_glyphs[labels == label] = 255

    repair = cv2.bitwise_or(bright_outline, dark_glyphs)
    repair = cv2.morphologyEx(
        repair,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    repair = cv2.dilate(
        repair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    residual_near = cv2.dilate(
        repair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55)),
        iterations=1,
    ) > 0
    residual_bright_raw = residual_near & (gray > 202) & ((gray > 232) | (white_tophat > 7))
    residual_bright = np.zeros_like(gray, dtype=np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        residual_bright_raw.astype(np.uint8), connectivity=8
    )
    repair_neighborhood = cv2.dilate(
        repair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37)),
        iterations=1,
    ) > 0
    for label in range(1, component_count):
        component = labels == label
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_border = cx <= 0 or cy <= 0 or (cx + cw) >= width or (cy + ch) >= height
        if component_area < 3:
            continue
        if touches_border and component_area > 36:
            continue
        if component_area > max(2400, int(area * 0.10)):
            continue
        if np.count_nonzero(component & repair_neighborhood) < 2:
            continue
        residual_bright[component] = 255
    if np.count_nonzero(residual_bright) >= 4:
        repair = cv2.bitwise_or(repair, residual_bright)
        repair = cv2.dilate(
            repair,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (repair > 0).astype(np.uint8), connectivity=8
    )
    filtered = np.zeros_like(repair)
    for label in range(1, component_count):
        component = labels == label
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        seed_overlap = int(np.count_nonzero(component & seed))
        bright_overlap = int(np.count_nonzero(component & bright_outline))
        if seed_overlap < 2 and bright_overlap < 4:
            continue
        if component_area > max(5200, int(area * 0.58)):
            continue
        if cw > int(width * 0.98) and ch < 18:
            continue
        filtered[component] = 255

    if np.count_nonzero(filtered > 0) < 10:
        return None
    repair_density = float(np.count_nonzero(filtered > 0)) / float(area)
    outline_count = int(np.count_nonzero(bright_outline > 0))
    outline_coverage = float(np.count_nonzero((filtered > 0) & (bright_outline > 0))) / float(
        max(1, outline_count)
    )
    source_coverage = float(np.count_nonzero((filtered > 0) & seed)) / float(max(1, seed_count))
    if repair_density > 0.36 or outline_coverage < 0.72:
        return None
    if source_coverage < 0.22 and outline_coverage < 0.86:
        return None
    return filtered


def _nearest_tone_class_repair(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    repair_mask: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or repair_mask.size == 0:
        return None

    repair = repair_mask > 0
    if np.count_nonzero(repair) < 8:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    near_repair = cv2.dilate(
        repair.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
        iterations=1,
    ) > 0
    local_tophat = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    source_like_bright = near_repair & (gray > 188) & (local_tophat > 8)
    valid = (~repair) & (~source_like_bright)
    dark_class = valid & (gray < 78)
    mid_class = valid & (gray >= 78) & (gray < 178)
    light_class = valid & (gray >= 178) & (gray < 226)
    if np.count_nonzero(dark_class) >= 20 and np.count_nonzero(mid_class) >= 20:
        classes = [dark_class, mid_class]
    else:
        classes = [dark_class, mid_class, light_class]
    if sum(int(np.count_nonzero(cls) >= 20) for cls in classes) < 2:
        return None

    height, width = gray.shape
    best_distance = np.full((height, width), np.inf, dtype=np.float32)
    best_color = np.zeros_like(roi)
    for cls in classes:
        if np.count_nonzero(cls) < 20:
            continue
        distance_input = (~cls).astype(np.uint8)
        distance_input[cls] = 0
        distances, labels = cv2.distanceTransformWithLabels(
            distance_input,
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        source_pixels = np.argwhere(cls)
        if source_pixels.size == 0:
            continue
        better = distances < best_distance
        ys, xs = np.where(better)
        nearest_indices = np.clip(labels[ys, xs] - 1, 0, len(source_pixels) - 1)
        best_distance[ys, xs] = distances[ys, xs]
        best_color[ys, xs] = roi[
            source_pixels[nearest_indices, 0],
            source_pixels[nearest_indices, 1],
        ]

    if not np.isfinite(best_distance[repair]).all():
        return None

    target_roi = target[y1:y2, x1:x2]
    target_roi[repair] = best_color[repair]
    return repair_mask


def _piecewise_tone_class_repair(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    repair_mask: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or repair_mask.size == 0:
        return None

    repair = repair_mask > 0
    repair_count = int(np.count_nonzero(repair))
    if repair_count < 8:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    near_repair = cv2.dilate(
        repair.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        iterations=1,
    ) > 0
    local_tophat = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    source_like_bright = near_repair & (gray > 188) & ((gray > 226) | (local_tophat > 8))
    valid = (~repair) & (~source_like_bright)

    dark_class = valid & (gray < 74)
    mid_class = valid & (gray >= 74) & (gray < 180)
    light_class = valid & (gray >= 180) & (gray < 226)
    use_light = not (np.count_nonzero(dark_class) >= 24 and np.count_nonzero(mid_class) >= 24)
    class_masks = [dark_class, mid_class]
    if use_light:
        class_masks.append(light_class)

    available = [mask for mask in class_masks if np.count_nonzero(mask) >= 24]
    if len(available) < 2:
        return None

    height, width = gray.shape
    assignment = np.full((height, width), -1, dtype=np.int16)
    best_distance = np.full((height, width), np.inf, dtype=np.float32)
    class_colors: list[np.ndarray] = []
    for class_index, class_mask in enumerate(class_masks):
        if np.count_nonzero(class_mask) < 24:
            class_colors.append(np.zeros(3, dtype=np.uint8))
            continue
        distance_input = (~class_mask).astype(np.uint8)
        distance_input[class_mask] = 0
        distances = cv2.distanceTransform(distance_input, cv2.DIST_L2, 5)
        better = distances < best_distance
        assignment[better] = class_index
        best_distance[better] = distances[better]
        pixels = roi[class_mask].astype(np.float32)
        median_color = np.median(pixels, axis=0)
        deviations = np.linalg.norm(pixels - median_color, axis=1)
        inliers = pixels[deviations < 42.0]
        if len(inliers) >= 12:
            median_color = np.median(inliers, axis=0)
        class_colors.append(np.clip(median_color, 0, 255).astype(np.uint8))

    if np.any(assignment[repair] < 0):
        return None

    filled = target[y1:y2, x1:x2].copy()
    for class_index, color in enumerate(class_colors):
        class_repair = repair & (assignment == class_index)
        if np.count_nonzero(class_repair) == 0:
            continue
        filled[class_repair] = color

    target_roi = target[y1:y2, x1:x2]
    target_roi[repair] = filled[repair]
    return repair_mask


def _row_tone_class_repair(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    repair_mask: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or repair_mask.size == 0:
        return None

    repair = cv2.dilate(
        (repair_mask > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ) > 0
    repair_count = int(np.count_nonzero(repair))
    if repair_count < 8:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    dark_fraction = float(np.mean(gray < 74))
    mid_fraction = float(np.mean((gray >= 74) & (gray < 180)))
    light_fraction = float(np.mean(gray >= 206))
    if dark_fraction < 0.18 or mid_fraction < 0.08:
        return None
    if light_fraction > 0.45 and dark_fraction < 0.25:
        return None

    near_repair = cv2.dilate(
        repair.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        iterations=1,
    ) > 0
    local_tophat = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    source_like_bright = near_repair & (gray > 188) & ((gray > 226) | (local_tophat > 8))
    valid = (~repair) & (~source_like_bright) & (gray < 190)
    if np.count_nonzero(valid) < 40:
        return None

    target_roi = target[y1:y2, x1:x2]
    patched = target_roi.copy()
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        repair.astype(np.uint8), connectivity=8
    )
    changed = np.zeros_like(repair_mask, dtype=np.uint8)
    fallback_values = roi[valid]
    fallback_gray = gray[valid]
    fallback_dark = fallback_values[fallback_gray < 74]
    fallback_mid = fallback_values[(fallback_gray >= 74) & (fallback_gray < 180)]

    for label in range(1, component_count):
        component = labels == label
        if np.count_nonzero(component) < 4:
            continue
        cy = int(stats[label, cv2.CC_STAT_TOP])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        pad = 36
        last_color: np.ndarray | None = None
        for yy in range(cy, cy + ch):
            xs = np.where(component[yy])[0]
            if xs.size == 0:
                continue
            xlo = max(0, int(xs.min()) - pad)
            xhi = min(roi.shape[1], int(xs.max()) + pad + 1)
            ylo = max(0, yy - 1)
            yhi = min(roi.shape[0], yy + 2)
            local_valid = valid[ylo:yhi, xlo:xhi]
            local_values = roi[ylo:yhi, xlo:xhi][local_valid]
            if local_values.shape[0] < 6:
                local_values = fallback_values
            if local_values.shape[0] < 6:
                continue

            local_gray = cv2.cvtColor(
                local_values.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY
            ).reshape(-1)
            dark_values = local_values[local_gray < 74]
            mid_values = local_values[(local_gray >= 74) & (local_gray < 180)]
            if dark_values.shape[0] >= 4 and (
                mid_values.shape[0] < 4 or dark_values.shape[0] >= mid_values.shape[0] * 0.65
            ):
                chosen_values = dark_values
            elif mid_values.shape[0] >= 4:
                chosen_values = mid_values
            elif fallback_dark.shape[0] >= 8:
                chosen_values = fallback_dark
            elif fallback_mid.shape[0] >= 8:
                chosen_values = fallback_mid
            elif last_color is not None:
                patched[yy, xs] = last_color
                changed[yy, xs] = 255
                continue
            else:
                continue

            color = np.median(chosen_values.astype(np.float32), axis=0)
            last_color = np.clip(color, 0, 255).astype(np.uint8)
            patched[yy, xs] = last_color
            changed[yy, xs] = 255

    coverage = float(np.count_nonzero(changed > 0)) / float(max(1, repair_count))
    if coverage < 0.62:
        return None

    target_roi[changed > 0] = patched[changed > 0]
    return changed


def _mixed_tone_model_repair(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
    anime_model,
    anime_device,
) -> np.ndarray | None:
    repair_mask = _mixed_tone_outline_mask(source, coords, mask_roi)
    if repair_mask is None:
        return None

    x1, y1, x2, y2 = coords
    region_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    region_mask[y1:y2, x1:x2] = repair_mask
    if _external_inpaint_command_local_crop(
        target,
        region_mask,
        source.shape[0],
        source.shape[1],
        x1,
        y1,
        x2,
        y2,
    ):
        return repair_mask
    if _manga_cleaner_local_crop(
        target,
        region_mask,
        source.shape[0],
        source.shape[1],
        x1,
        y1,
        x2,
        y2,
    ):
        return repair_mask

    roi = source[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    dark_fraction = float(np.mean(gray < 74))
    light_fraction = float(np.mean(gray >= 206))
    if light_fraction > 0.45 and dark_fraction < 0.25:
        return None

    row_repair = _row_tone_class_repair(source, target, coords, repair_mask)
    if row_repair is not None:
        return row_repair

    if _piecewise_tone_class_repair(source, target, coords, repair_mask) is not None:
        return repair_mask

    before = target[y1:y2, x1:x2].copy()

    if anime_model is not None:
        _anime_lama_local_crop(
            anime_model,
            anime_device,
            target,
            region_mask,
            source.shape[0],
            source.shape[1],
            x1,
            y1,
            x2,
            y2,
        )
        class_target = target.copy()
        class_repair = _nearest_tone_class_repair(source, class_target, coords, repair_mask)
        if class_repair is not None:
            target_roi = target[y1:y2, x1:x2]
            class_roi = class_target[y1:y2, x1:x2]
            output_gray = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
            output_tophat = cv2.morphologyEx(
                output_gray,
                cv2.MORPH_TOPHAT,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            )
            repair_bool = repair_mask > 0
            residual_text = repair_bool & ((output_gray > 188) | (output_tophat > 12))
            if np.count_nonzero(residual_text) >= 4:
                target_roi[residual_text] = class_roi[residual_text]
    else:
        repaired = _nearest_tone_class_repair(source, target, coords, repair_mask)
        if repaired is None:
            return None

    changed = np.any(before != target[y1:y2, x1:x2], axis=2)
    changed_count = int(np.count_nonzero(changed & (repair_mask > 0)))
    if changed_count < 8:
        return None
    return repair_mask


def _dilated_anime_caption_repair(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
    anime_model,
    anime_device,
) -> np.ndarray | None:
    if anime_model is None:
        return None

    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return None

    source_only_mask = _refined_floating_source_mask(
        source,
        np.zeros(source.shape[:2], dtype=np.uint8),
        coords,
    )
    if np.count_nonzero(source_only_mask > 0) >= 6:
        repair = source_only_mask > 0
    else:
        repair = mask_roi > 0
    repair_count = int(np.count_nonzero(repair))
    if repair_count < 6:
        return None

    area = max(1, repair.size)
    density = float(repair_count) / float(area)
    if density < 0.18 or density > 0.92:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 135) > 0
    unmasked = ~repair
    unmasked_count = int(np.count_nonzero(unmasked))
    if unmasked_count < max(16, int(area * 0.015)):
        return None

    bg_std = float(np.std(gray[unmasked]))
    bg_edge = float(np.mean(edges[unmasked]))
    bg_median = float(np.median(gray[unmasked]))
    bg_bright_fraction = float(np.mean(gray[unmasked] > 230))

    simple_gray = bg_std <= 18.0 and bg_edge <= 0.12
    bright_halftone = bg_median >= 228.0 and bg_bright_fraction >= 0.52 and bg_edge <= 0.13
    if not (simple_gray or bright_halftone):
        return None

    dilated = cv2.dilate(
        (repair.astype(np.uint8)) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2,
    )
    if _floating_region_is_art_sensitive(source, coords, dilated):
        return None

    region_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    region_mask[y1:y2, x1:x2] = dilated
    before = target[y1:y2, x1:x2].copy()
    _anime_lama_local_crop(
        anime_model,
        anime_device,
        target,
        region_mask,
        source.shape[0],
        source.shape[1],
        x1,
        y1,
        x2,
        y2,
    )
    core_seed = cv2.dilate(
        (repair.astype(np.uint8)) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    ring_mask = (dilated > 0) & ~(core_seed > 0)
    if np.count_nonzero(ring_mask) >= 10:
        repaired_roi = target[y1:y2, x1:x2].copy()
        ring_alpha = cv2.GaussianBlur(ring_mask.astype(np.float32), (0, 0), 1.2)
        ring_alpha = np.clip(
            ring_alpha * (0.72 if bright_halftone else 0.58),
            0.0,
            1.0,
        )[..., None]
        restored = (
            before.astype(np.float32) * ring_alpha
            + repaired_roi.astype(np.float32) * (1.0 - ring_alpha)
        ).astype(np.uint8)
        repaired_roi[ring_mask] = restored[ring_mask]
        target[y1:y2, x1:x2] = repaired_roi
    changed = np.any(before != target[y1:y2, x1:x2], axis=2)
    changed_count = int(np.count_nonzero(changed & (dilated > 0)))
    if changed_count < 8:
        return None
    return dilated


def _legacy_bright_caption_candidate(
    source: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> bool:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return False

    height, width = roi.shape[:2]
    area = max(1, height * width)
    text_count = int(np.count_nonzero(mask_roi > 0))
    if text_count < 20:
        return False

    density = text_count / float(area)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    seed = mask_roi > 0
    seed_neighborhood = cv2.dilate(
        seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    ).astype(bool)
    nonpaper_surface = False
    if np.any(seed_neighborhood):
        near_gray = gray[seed_neighborhood]
        near_mid_tone = float(np.mean((near_gray >= 70) & (near_gray <= 210)))
        near_nonpaper = float(np.mean(near_gray < 190))
        near_bright = float(np.mean(near_gray > 224))
        nonpaper_surface = near_mid_tone >= 0.18 and near_nonpaper >= 0.38 and near_bright <= 0.64
    background = mask_roi <= 0
    if np.count_nonzero(background) < max(16, int(area * 0.025)):
        background = np.ones_like(mask_roi, dtype=bool)

    bg_gray = gray[background]
    bg_sat = hsv[:, :, 1][background]
    bg_bright = float(np.mean(bg_gray > 224))
    bg_dark = float(np.mean(bg_gray < 72))
    low_saturation = float(np.mean(bg_sat < 120))
    edge_density = float(np.mean(cv2.Canny(gray, 45, 135) > 0))

    if bg_bright < 0.68 or bg_dark > 0.15 or low_saturation < 0.76:
        return False

    tall_caption = height >= max(84, int(width * 1.16))
    wide_caption = width >= max(150, int(height * 1.70))
    dense_caption = density >= 0.62 and edge_density <= 0.24
    horizontal_bright_caption = wide_caption and density >= 0.20 and edge_density <= 0.20
    compact_bright_caption = (
        width >= 58
        and height >= 38
        and width <= 180
        and height <= 120
        and density >= 0.72
        and edge_density <= 0.30
    )
    compact_vertical_nonpaper_caption = (
        nonpaper_surface
        and width <= 72
        and height >= max(110, int(width * 2.40))
        and density >= 0.44
        and edge_density <= 0.20
    )
    if nonpaper_surface and not compact_vertical_nonpaper_caption:
        return False

    return (
        dense_caption
        or compact_bright_caption
        or horizontal_bright_caption
        or compact_vertical_nonpaper_caption
        or (tall_caption and density >= 0.48)
    )


def _striped_screen_ui_caption_candidate(
    source: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> bool:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return False

    height, width = roi.shape[:2]
    area = max(1, height * width)
    if width < 64 or height < 48:
        return False

    seed_count = int(np.count_nonzero(mask_roi > 0))
    if seed_count < max(32, int(area * 0.12)):
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    density = seed_count / float(area)
    dark_fraction = float(np.mean(gray < 96))
    bright_fraction = float(np.mean((gray > 180) & (sat < 175)))
    mid_fraction = float(np.mean((gray >= 96) & (gray <= 180)))
    low_saturation = float(np.mean(sat < 165))
    row_medians = np.median(gray, axis=1).astype(np.float32)
    row_std = float(np.std(row_medians))
    row_delta = float(np.mean(np.abs(np.diff(row_medians)))) if len(row_medians) > 1 else 0.0
    dark_rows = float(np.mean(row_medians < 105))
    bright_rows = float(np.mean(row_medians > 174))
    transitions = int(np.count_nonzero(np.abs(np.diff(row_medians)) > 28.0)) if len(row_medians) > 1 else 0
    edge_density = float(np.mean(cv2.Canny(gray, 45, 135) > 0))
    # Bold dark display text on a bright surface can satisfy every row-profile
    # condition below (glyph-dominated rows read as "dark rows"), routing a
    # plain caption to the device-panel cleaner, which then erases the BRIGHT
    # background between strokes as if it were knockout text. The reliable
    # polarity signal is topological: a genuine dark screen/panel band holds
    # light KNOCKOUT text fully enclosed by the dark surface, while dark-on-
    # bright text sits on background that stays connected to the box border.
    # Measure, within the dark-row band, the fraction of bright pixels that
    # are NOT border-connected (measured 0.543 for a real dark UI banner vs
    # 0.000 for bold dark text on a bright surface).
    band_rows = row_medians < 105
    knockout_ratio = 0.0
    if bool(np.any(band_rows)):
        bright_mask = ((gray > 170) & (sat < 190)).astype(np.uint8)
        _, bright_labels = cv2.connectedComponents(bright_mask, 8)
        border_labels = set(np.unique(bright_labels[0, :]))
        border_labels |= set(np.unique(bright_labels[-1, :]))
        border_labels |= set(np.unique(bright_labels[:, 0]))
        border_labels |= set(np.unique(bright_labels[:, -1]))
        border_labels.discard(0)
        enclosed_bright = bright_mask.astype(bool)
        if border_labels:
            enclosed_bright &= ~np.isin(bright_labels, list(border_labels))
        band_bright_total = int(np.count_nonzero(bright_mask[band_rows]))
        band_bright_enclosed = int(np.count_nonzero(enclosed_bright[band_rows]))
        if band_bright_total >= 12:
            knockout_ratio = band_bright_enclosed / float(band_bright_total)

    return (
        density >= 0.22
        and density <= 0.82
        and dark_fraction >= 0.24
        and bright_fraction >= 0.20
        and mid_fraction <= 0.18
        and low_saturation >= 0.64
        and row_std >= 38.0
        and row_delta >= 5.0
        and row_delta <= 28.0
        and dark_rows >= 0.16
        and bright_rows >= 0.10
        and transitions >= 2
        and edge_density >= 0.035
        and edge_density <= 0.22
        and knockout_ratio >= 0.30
    )


def _dark_device_surface_text_cleanup(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    height, width = roi.shape[:2]
    area = max(1, height * width)
    if width < 56 or height < 48:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    dark = (gray < 96) & (sat < 190)
    dark_panel = np.zeros((height, width), dtype=bool)
    dark_panel_top = height
    isolated_screen_band = False
    row_fraction = np.mean(dark, axis=1)
    row_mask = row_fraction >= 0.55
    row_ranges = []
    row_start = None
    for row_index, value in enumerate(row_mask):
        if value and row_start is None:
            row_start = row_index
        if (not value or row_index == height - 1) and row_start is not None:
            row_end = row_index if not value else row_index + 1
            row_ranges.append((row_start, row_end, row_end - row_start))
            row_start = None
    row_ranges = [
        item for item in row_ranges
        if item[2] >= max(18, int(height * 0.18))
    ]
    if row_ranges:
        row_start, row_end, _ = max(row_ranges, key=lambda item: item[2])
        band = dark[row_start:row_end]
        col_fraction = np.mean(band, axis=0)
        col_indices = np.flatnonzero(col_fraction >= 0.38)
        if col_indices.size >= max(32, int(width * 0.42)):
            cleanup_row_start = max(0, row_start - max(4, int(round(height * 0.045))))
            dark_panel[cleanup_row_start:row_end, int(col_indices.min()): int(col_indices.max()) + 1] = True
            dark_panel_top = cleanup_row_start
            isolated_screen_band = cleanup_row_start >= int(height * 0.12)
    if int(np.count_nonzero(dark_panel)) < max(320, int(area * 0.20)):
        loose_dark = (gray < 120) & (sat < 190)
        loose_row_mask = np.mean(loose_dark, axis=1) >= 0.25
        loose_ranges = []
        loose_start = None
        for row_index, value in enumerate(loose_row_mask):
            if value and loose_start is None:
                loose_start = row_index
            if (not value or row_index == height - 1) and loose_start is not None:
                loose_end = row_index if not value else row_index + 1
                loose_ranges.append((loose_start, loose_end, loose_end - loose_start))
                loose_start = None
        loose_candidates = [
            item for item in loose_ranges
            if item[0] >= int(height * 0.18)
            and item[2] >= max(10, int(height * 0.05))
        ]
        merged_ranges: list[list[int]] = []
        max_gap = max(6, int(height * 0.06))
        for start, end, _ in loose_candidates:
            if not merged_ranges or start - merged_ranges[-1][1] > max_gap:
                merged_ranges.append([start, end])
            else:
                merged_ranges[-1][1] = end
        if merged_ranges:
            row_start, row_end = max(merged_ranges, key=lambda item: item[1] - item[0])
            band = loose_dark[row_start:row_end]
            col_fraction = np.mean(band, axis=0)
            col_indices = np.flatnonzero(col_fraction >= 0.20)
            if (
                row_end - row_start >= max(32, int(height * 0.34))
                and col_indices.size >= max(44, int(width * 0.50))
            ):
                dark_panel[row_start:row_end, int(col_indices.min()): int(col_indices.max()) + 1] = True
                dark_panel_top = row_start
                isolated_screen_band = True
    if int(np.count_nonzero(dark_panel)) < max(320, int(area * 0.20)):
        dark_closed = cv2.morphologyEx(
            dark.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
            iterations=1,
        )
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(dark_closed, 8)
        for label in range(1, component_count):
            component_area = int(stats[label, cv2.CC_STAT_AREA])
            cy = int(stats[label, cv2.CC_STAT_TOP])
            cw = int(stats[label, cv2.CC_STAT_WIDTH])
            ch = int(stats[label, cv2.CC_STAT_HEIGHT])
            if (
                component_area >= max(320, int(area * 0.24))
                and cw >= max(44, int(width * 0.55))
                and ch >= max(40, int(height * 0.42))
                and cy <= int(height * 0.42)
            ):
                dark_panel |= labels == label
                dark_panel_top = min(dark_panel_top, cy)
    if int(np.count_nonzero(dark_panel)) < max(320, int(area * 0.20)):
        return None

    dark_support = cv2.dilate(
        dark_panel.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ).astype(bool)
    bright = (gray > 148) & (sat < 190) & dark_support
    if dark_panel_top < height:
        bright[: min(height, dark_panel_top + 4), :] = False
    bright_row_fraction = np.mean(bright, axis=1)
    bright[bright_row_fraction >= 0.72, :] = False
    bright_count = int(np.count_nonzero(bright))
    if bright_count < max(36, int(area * 0.015)):
        return None

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8), 8)
    repair = np.zeros((height, width), dtype=np.uint8)
    for label in range(1, component_count):
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        if component_area < 8:
            continue
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if cw <= 0 or ch <= 0:
            continue
        fill_ratio = component_area / float(max(1, cw * ch))
        top_ui_surface = cy < int(height * 0.22) and cw >= int(width * 0.55) and ch >= int(height * 0.18)
        bottom_hand_surface = (
            component_area >= max(700, int(area * 0.045))
            and cy >= int(height * 0.54)
            and cw >= int(width * 0.20)
            and ch >= int(height * 0.18)
        )
        side_device_edge = (
            cx >= int(width * 0.86)
            and ch >= int(height * 0.18)
            and fill_ratio <= 0.52
        )
        bottom_ui_rule = cy >= int(height * 0.92) and cw >= int(width * 0.30)
        if top_ui_surface or bottom_hand_surface or side_device_edge or bottom_ui_rule:
            continue
        if component_area >= int(area * 0.10):
            continue
        repair[labels == label] = 255

    repair = cv2.dilate(
        repair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    repair_bool = (repair > 0) & dark_support
    residual_threshold = 128 if isolated_screen_band else 138
    residual_max_area = max(420, int(area * 0.018)) if isolated_screen_band else max(260, int(area * 0.012))
    residual = (gray > residual_threshold) & (sat < 190) & dark_panel & (~repair_bool)
    residual_count, residual_labels, residual_stats, _ = cv2.connectedComponentsWithStats(
        residual.astype(np.uint8),
        8,
    )
    residual_repair = np.zeros((height, width), dtype=np.uint8)
    for label in range(1, residual_count):
        component_area = int(residual_stats[label, cv2.CC_STAT_AREA])
        if component_area < 4 or component_area > residual_max_area:
            continue
        cx = int(residual_stats[label, cv2.CC_STAT_LEFT])
        cy = int(residual_stats[label, cv2.CC_STAT_TOP])
        cw = int(residual_stats[label, cv2.CC_STAT_WIDTH])
        ch = int(residual_stats[label, cv2.CC_STAT_HEIGHT])
        if cx >= int(width * 0.84) and ch >= max(8, int(height * 0.06)):
            continue
        if cy >= int(height * 0.58) and cw >= int(width * 0.18):
            continue
        residual_repair[residual_labels == label] = 255
    if int(np.count_nonzero(residual_repair > 0)) > 0:
        residual_repair = cv2.dilate(
            residual_repair,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        repair_bool |= (residual_repair > 0) & dark_panel
    if int(np.count_nonzero(repair_bool)) < max(24, int(area * 0.01)):
        return None

    target_roi = target[y1:y2, x1:x2].copy()
    dark_pixels = roi[dark_panel & (gray < 126) & (~repair_bool)]
    if dark_pixels.size == 0:
        fill_fallback = np.array([8, 8, 8], dtype=np.uint8)
    else:
        fill_fallback = np.median(dark_pixels.reshape(-1, 3), axis=0).astype(np.uint8)

    for row in range(height):
        cols = np.flatnonzero(repair_bool[row])
        if cols.size == 0:
            continue
        row_start = max(0, row - 5)
        row_end = min(height, row + 6)
        band_mask = dark_panel[row_start:row_end] & (gray[row_start:row_end] < 126) & (~repair_bool[row_start:row_end])
        band_pixels = roi[row_start:row_end][band_mask]
        if band_pixels.size:
            fill_color = np.median(band_pixels.reshape(-1, 3), axis=0).astype(np.uint8)
        else:
            fill_color = fill_fallback
        target_roi[row, cols] = fill_color

    target[y1:y2, x1:x2] = target_roi
    full_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = (repair_bool.astype(np.uint8)) * 255
    return full_mask


def _legacy_full_box_anime_repair(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    anime_model,
    anime_device,
    pad: int = 2,
) -> np.ndarray | None:
    if anime_model is None:
        return None

    img_h, img_w = source.shape[:2]
    x1, y1, x2, y2 = coords
    fx1 = max(0, x1 - pad)
    fy1 = max(0, y1 - pad)
    fx2 = min(img_w, x2 + pad)
    fy2 = min(img_h, y2 + pad)
    if fx2 <= fx1 or fy2 <= fy1:
        return None

    region_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    region_mask[fy1:fy2, fx1:fx2] = 255
    before = target[fy1:fy2, fx1:fx2].copy()
    _anime_lama_local_crop(
        anime_model,
        anime_device,
        target,
        region_mask,
        img_h,
        img_w,
        fx1,
        fy1,
        fx2,
        fy2,
    )
    changed = np.any(before != target[fy1:fy2, fx1:fx2], axis=2)
    if int(np.count_nonzero(changed)) < 8:
        return None
    return region_mask


def _dense_mixed_surface_fullbox_model_repair(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
    anime_model,
    anime_device,
) -> np.ndarray | None:
    img_h, img_w = source.shape[:2]
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return None

    height, width = roi.shape[:2]
    area = max(1, height * width)
    if width < 86 or height < max(132, int(width * 1.45)):
        return None
    if width > max(260, int(img_w * 0.24)):
        return None

    seed = mask_roi > 0
    seed_count = int(np.count_nonzero(seed))
    if seed_count < 80:
        return None
    seed_density = seed_count / float(area)
    if seed_density < 0.22 or seed_density > 0.56:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    low_saturation = float(np.mean(saturation < 170))
    dark_fraction = float(np.mean(gray < 78))
    mid_fraction = float(np.mean((gray >= 78) & (gray < 188)))
    paper_fraction = float(np.mean((gray > 224) & (saturation < 120)))
    luma_std = float(np.std(gray.astype(np.float32)))
    near_seed = cv2.dilate(
        seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    ) > 0
    near_bright_fraction = float(np.mean((gray > 188) & near_seed))
    if not (
        low_saturation >= 0.86
        and dark_fraction >= 0.18
        and mid_fraction >= 0.12
        and 0.18 <= paper_fraction <= 0.54
        and near_bright_fraction >= 0.22
        and luma_std >= 72.0
        and _floating_region_has_mixed_character_tone(source, coords, mask_roi)
        and _floating_region_is_art_sensitive(source, coords, mask_roi)
    ):
        return None

    text_mask = _outlined_floating_source_mask(source, coords, mask_roi)
    if text_mask is None or int(np.count_nonzero(text_mask > 0)) < 24:
        return None

    cleanup = cv2.morphologyEx(
        (text_mask > 0).astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    cleanup = cv2.dilate(
        cleanup,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    cleanup_count = int(np.count_nonzero(cleanup > 0))
    cleanup_density = cleanup_count / float(area)
    if cleanup_count < 36 or cleanup_density > 0.86:
        return None

    snapshot = target[y1:y2, x1:x2].copy()
    try:
        telea = cv2.inpaint(snapshot, cleanup, 4.0, cv2.INPAINT_TELEA)
        navier = cv2.inpaint(snapshot, cleanup, 3.4, cv2.INPAINT_NS)
    except cv2.error:
        return None

    repaired = cv2.addWeighted(telea, 0.68, navier, 0.32, 0)
    target_roi = target[y1:y2, x1:x2]
    target_roi[cleanup > 0] = repaired[cleanup > 0]

    repaired_roi = target_roi
    repaired_gray = cv2.cvtColor(repaired_roi, cv2.COLOR_BGR2GRAY)
    source_like = near_seed & (gray > 188) & (text_mask > 0)
    if int(np.count_nonzero(source_like)) < 24:
        target[y1:y2, x1:x2] = snapshot
        return None
    residual_bright_fraction = float(np.mean((repaired_gray > 224)[source_like]))
    repaired_paper_fraction = float(np.mean(repaired_gray > 230))
    repaired_mid_fraction = float(np.mean((repaired_gray >= 72) & (repaired_gray < 190)))
    if (
        residual_bright_fraction > 0.38
        or repaired_paper_fraction > 0.48
        or repaired_mid_fraction < 0.18
    ):
        target[y1:y2, x1:x2] = snapshot
        return None

    repair_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    repair_mask[y1:y2, x1:x2] = cleanup
    return repair_mask


def _repair_box_crosses_foreground_line_art(
    source: np.ndarray,
    repair_coords: tuple[int, int, int, int],
    seed_coords: tuple[int, int, int, int],
    seed_roi: np.ndarray,
) -> bool:
    rx1, ry1, rx2, ry2 = repair_coords
    sx1, sy1, sx2, sy2 = seed_coords
    roi = source[ry1:ry2, rx1:rx2]
    if roi.size == 0 or seed_roi.size == 0:
        return False

    height, width = roi.shape[:2]
    area = max(1, height * width)
    seed_mask = np.zeros((height, width), dtype=np.uint8)
    ox1 = max(0, sx1 - rx1)
    oy1 = max(0, sy1 - ry1)
    ox2 = min(width, sx2 - rx1)
    oy2 = min(height, sy2 - ry1)
    if ox2 > ox1 and oy2 > oy1:
        sx_off1 = max(0, rx1 - sx1)
        sy_off1 = max(0, ry1 - sy1)
        sx_off2 = sx_off1 + (ox2 - ox1)
        sy_off2 = sy_off1 + (oy2 - oy1)
        seed_mask[oy1:oy2, ox1:ox2] = (seed_roi[sy_off1:sy_off2, sx_off1:sx_off2] > 0).astype(np.uint8) * 255

    seed_guard = cv2.dilate(
        seed_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=1,
    ) > 0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    line_candidates = (gray < 142) & (hsv[:, :, 1] < 235) & ~seed_guard
    line_candidates = cv2.morphologyEx(
        line_candidates.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(line_candidates, connectivity=8)
    significant_components = 0
    for label in range(1, component_count):
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        if component_area < 12 or component_area > max(2200, int(area * 0.18)):
            continue
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        elongated = max(cw, ch) >= 18 and max(cw, ch) >= min(cw, ch) * 2.4
        border_touch = cx <= 1 or cy <= 1 or (cx + cw) >= width - 1 or (cy + ch) >= height - 1
        lower_or_side = cy >= int(height * 0.36) or cx <= int(width * 0.22) or (cx + cw) >= int(width * 0.78)
        if elongated and (border_touch or lower_or_side):
            significant_components += 1
            if significant_components >= 1:
                return True
    return False


def _restore_foreground_art_outside_source_text(
    source: np.ndarray,
    target: np.ndarray,
    repair_mask: np.ndarray,
    seed_coords: tuple[int, int, int, int],
    seed_roi: np.ndarray,
) -> np.ndarray | None:
    if repair_mask.size == 0 or int(np.count_nonzero(repair_mask > 0)) < 8:
        return None

    ys, xs = np.where(repair_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    img_h, img_w = source.shape[:2]
    rx1 = max(0, int(xs.min()) - 1)
    ry1 = max(0, int(ys.min()) - 1)
    rx2 = min(img_w, int(xs.max()) + 2)
    ry2 = min(img_h, int(ys.max()) + 2)
    if rx2 <= rx1 or ry2 <= ry1:
        return None

    source_roi = source[ry1:ry2, rx1:rx2]
    target_roi = target[ry1:ry2, rx1:rx2]
    repair_roi = repair_mask[ry1:ry2, rx1:rx2] > 0
    if source_roi.size == 0 or target_roi.size == 0:
        return None

    sx1, sy1, sx2, sy2 = seed_coords
    seed_mask = np.zeros(repair_roi.shape, dtype=np.uint8)
    ox1 = max(0, sx1 - rx1)
    oy1 = max(0, sy1 - ry1)
    ox2 = min(seed_mask.shape[1], sx2 - rx1)
    oy2 = min(seed_mask.shape[0], sy2 - ry1)
    if ox2 > ox1 and oy2 > oy1:
        seed_src_x1 = max(0, rx1 - sx1)
        seed_src_y1 = max(0, ry1 - sy1)
        seed_src_x2 = seed_src_x1 + (ox2 - ox1)
        seed_src_y2 = seed_src_y1 + (oy2 - oy1)
        seed_mask[oy1:oy2, ox1:ox2] = (seed_roi[seed_src_y1:seed_src_y2, seed_src_x1:seed_src_x2] > 0).astype(
            np.uint8
        ) * 255

    seed_guard = cv2.dilate(
        seed_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ) > 0
    gray = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(source_roi, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 42, 126) > 0
    foreground = ((gray < 118) & (hsv[:, :, 1] < 245)) | (edges & (gray < 205))
    restore = repair_roi & foreground & ~seed_guard
    if int(np.count_nonzero(restore)) < 8:
        return None

    restore_uint = cv2.morphologyEx(
        restore.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    restore = restore_uint > 0
    if int(np.count_nonzero(restore)) < 8:
        return None

    target_roi[restore] = source_roi[restore]
    restored_mask = np.zeros_like(repair_mask)
    restored_mask[ry1:ry2, rx1:rx2] = restore.astype(np.uint8) * 255
    return restored_mask


def _restore_art_lines_crossing_repair_mask(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    repair_mask: np.ndarray,
    seed_roi: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    source_roi = source[y1:y2, x1:x2]
    target_roi = target[y1:y2, x1:x2]
    if source_roi.size == 0 or repair_mask.size == 0:
        return None

    repair = repair_mask > 0
    if int(np.count_nonzero(repair)) < 8:
        return None

    gray = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(source_roi, cv2.COLOR_BGR2HSV)
    seed_guard = cv2.dilate(
        (seed_roi > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ) > 0
    repair_guard = cv2.dilate(
        repair.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ) > 0
    line_candidates = (gray < 128) & (hsv[:, :, 1] < 235) & ~seed_guard
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        line_candidates.astype(np.uint8),
        connectivity=8,
    )
    restore = np.zeros(repair.shape, dtype=np.uint8)
    area = max(1, repair.shape[0] * repair.shape[1])
    for label in range(1, component_count):
        component = labels == label
        inside_count = int(np.count_nonzero(component & repair_guard))
        if inside_count < 2:
            continue
        outside_count = int(np.count_nonzero(component & ~repair_guard))
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        if component_area < 4 or component_area > max(3200, int(area * 0.22)):
            continue
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        elongated = max(cw, ch) >= 10 and max(cw, ch) >= min(cw, ch) * 1.8
        if outside_count >= 2 or elongated:
            restore[component & repair_guard] = 255

    if int(np.count_nonzero(restore > 0)) < 2:
        return None
    restore = cv2.dilate(
        restore,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1,
    )
    target_roi[restore > 0] = source_roi[restore > 0]
    return restore


def _union_repair_box(
    base: tuple[int, int, int, int],
    extra: Sequence[int] | None,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    img_h, img_w = image_shape[:2]
    x1, y1, x2, y2 = base
    if extra and len(extra) >= 4:
        ex1, ey1, ex2, ey2 = [int(round(float(v))) for v in extra[:4]]
        x1 = min(x1, ex1)
        y1 = min(y1, ey1)
        x2 = max(x2, ex2)
        y2 = max(y2, ey2)
    return (
        max(0, min(img_w, x1)),
        max(0, min(img_h, y1)),
        max(0, min(img_w, x2)),
        max(0, min(img_h, y2)),
    )


def _compact_caption_repair_pad(coords: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = coords
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    if width <= 180 and height <= 120:
        return 6
    return 2


def _restore_wide_caption_line_art(
    source: np.ndarray,
    target: np.ndarray,
    analysis_coords: tuple[int, int, int, int],
    limit_coords: tuple[int, int, int, int],
) -> np.ndarray | None:
    ax1, ay1, ax2, ay2 = analysis_coords
    lx1, ly1, lx2, ly2 = limit_coords
    roi = source[ay1:ay2, ax1:ax2]
    if roi.size == 0:
        return None

    height, width = roi.shape[:2]
    if width < max(220, int(height * 2.10)) or height > 170:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    bright_fraction = float(np.mean((gray > 178) & (hsv[:, :, 1] < 150)))
    edge_density = float(np.mean(cv2.Canny(gray, 45, 135) > 0))
    if bright_fraction < 0.42 or edge_density < 0.025:
        return None

    dark = (gray < 95).astype(np.uint8) * 255
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    local_restore = np.zeros_like(dark)

    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 8:
            continue

        fill_ratio = area / float(max(1, cw * ch))
        aspect = cw / float(max(1, ch))
        panel_border = cw > width * 0.38 and ch <= 9 and cy > height * 0.78
        left_line_art = (
            cx < width * 0.23
            and cy < height * 0.56
            and fill_ratio < 0.30
            and aspect >= 1.85
        )
        lower_left_shirt_line = (
            cx < width * 0.23
            and cy > height * 0.68
            and ch >= 10
            and cw <= 18
            and fill_ratio < 0.86
            and (ch / float(max(1, cw))) >= 0.75
        )
        if panel_border or left_line_art or lower_left_shirt_line:
            local_restore[labels == label] = 255

    if np.count_nonzero(local_restore > 0) < 8:
        return None

    local_restore = cv2.dilate(
        local_restore,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    restore_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    restore_mask[ay1:ay2, ax1:ax2] = local_restore
    limit = np.zeros_like(restore_mask)
    limit[ly1:ly2, lx1:lx2] = 255
    restore_mask = cv2.bitwise_and(restore_mask, limit)
    if np.count_nonzero(restore_mask > 0) < 8:
        return None

    target[restore_mask > 0] = source[restore_mask > 0]
    return restore_mask


def _restore_crossing_line_art_after_repair(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    repair_mask: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or repair_mask.size == 0:
        return None

    repair = (repair_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(repair > 0) < 8:
        return None

    height, width = repair.shape[:2]
    if height <= 0 or width <= 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    dark_components = ((gray < 104) & (hsv[:, :, 1] < 230)).astype(np.uint8)
    bright_outline = (gray > 205) & (hsv[:, :, 1] < 190)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(dark_components, connectivity=8)
    broad_repair = cv2.dilate(
        repair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    ) > 0
    repair_bool = repair > 0
    local_restore = np.zeros((height, width), dtype=np.uint8)

    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 6 or area > max(3600, int(width * height * 0.12)):
            continue
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if cw <= 0 or ch <= 0:
            continue

        fill_ratio = area / float(max(1, cw * ch))
        long_axis = max(cw, ch)
        short_axis = max(1, min(cw, ch))
        axis_ratio = long_axis / float(short_axis)
        if fill_ratio > 0.34 or long_axis < 22:
            continue
        if axis_ratio < 1.35 and area < 72:
            continue

        component = labels == label
        overlap = int(np.count_nonzero(component & repair_bool))
        if overlap < max(3, int(area * 0.035)):
            continue

        outside_broad = int(np.count_nonzero(component & ~broad_repair))
        if outside_broad < max(4, int(area * 0.14)):
            continue

        bbox_repair_density = float(np.count_nonzero(broad_repair[cy:cy + ch, cx:cx + cw])) / float(max(1, cw * ch))
        if bbox_repair_density > 0.82 and outside_broad < max(9, int(area * 0.28)):
            continue

        component_ring = cv2.dilate(
            component.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        ).astype(bool) & ~component
        ring_count = int(np.count_nonzero(component_ring))
        if ring_count:
            bright_ring_density = float(np.count_nonzero(bright_outline & component_ring)) / float(ring_count)
            if bright_ring_density > 0.20 and outside_broad < max(18, int(area * 0.44)):
                continue

        local_restore[component] = 255

    if np.count_nonzero(local_restore > 0) < 6:
        return None

    local_restore = cv2.dilate(
        local_restore,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    target_roi = target[y1:y2, x1:x2]
    target_roi[local_restore > 0] = roi[local_restore > 0]
    return local_restore


def _extend_line_art_across_repair_mask(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    repair_mask: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    source_roi = source[y1:y2, x1:x2]
    target_roi = target[y1:y2, x1:x2]
    if source_roi.size == 0 or target_roi.size == 0 or repair_mask.size == 0:
        return None

    repair = repair_mask > 0
    repair_count = int(np.count_nonzero(repair))
    if repair_count < 18:
        return None

    height, width = repair_mask.shape[:2]
    area = max(1, height * width)
    if height < 32 or width < 32:
        return None

    source_gray = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
    source_hsv = cv2.cvtColor(source_roi, cv2.COLOR_BGR2HSV)
    low_saturation = float(np.mean(source_hsv[:, :, 1] < 170))
    if low_saturation < 0.78:
        return None

    edges = cv2.Canny(target_gray, 35, 115)
    outside_edges = cv2.bitwise_and(
        edges,
        ((~repair).astype(np.uint8)) * 255,
    )
    if int(np.count_nonzero(outside_edges > 0)) < 18:
        return None

    lines = cv2.HoughLinesP(
        outside_edges,
        1,
        np.pi / 180,
        threshold=18,
        minLineLength=18,
        maxLineGap=8,
    )
    if lines is None:
        return None

    repair_near = cv2.dilate(
        repair.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ).astype(bool)
    target_bright = target_gray > 150
    local_restore = np.zeros((height, width), dtype=np.uint8)

    for line in lines[:, 0, :]:
        lx1, ly1, lx2, ly2 = [int(value) for value in line]
        length = math.hypot(lx2 - lx1, ly2 - ly1)
        if length < 18:
            continue
        line_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.line(line_mask, (lx1, ly1), (lx2, ly2), 255, 2)
        line_bool = line_mask > 0
        near_support = int(np.count_nonzero(line_bool & repair_near & ~repair))
        if near_support < 2:
            continue
        edge_support = int(np.count_nonzero(line_bool & (outside_edges > 0) & ~repair))
        if edge_support < max(4, int(length * 0.12)):
            continue

        dx = (lx2 - lx1) / length
        dy = (ly2 - ly1) / length
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if 72.0 <= angle <= 108.0 and length < 42:
            continue

        extension = max(22.0, min(110.0, length * 1.75))
        ex1 = int(round(lx1 - dx * extension))
        ey1 = int(round(ly1 - dy * extension))
        ex2 = int(round(lx2 + dx * extension))
        ey2 = int(round(ly2 + dy * extension))
        extended = np.zeros((height, width), dtype=np.uint8)
        cv2.line(extended, (ex1, ey1), (ex2, ey2), 255, 1)
        candidate = (extended > 0) & repair & target_bright
        if int(np.count_nonzero(candidate)) < 2:
            continue
        local_restore[candidate] = 255

    restore_count = int(np.count_nonzero(local_restore > 0))
    if restore_count < 6:
        return None
    if restore_count / float(area) > 0.045:
        return None

    local_restore = cv2.dilate(
        local_restore,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    source_dark = (source_gray < 88) & repair
    source_dark_near = cv2.dilate(
        source_dark.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ).astype(bool)
    glyph_risk = (local_restore > 0) & source_dark_near
    if int(np.count_nonzero(glyph_risk)) / float(max(1, np.count_nonzero(local_restore > 0))) > 0.74:
        return None

    target_roi[local_restore > 0] = (0, 0, 0)
    return local_restore


def _clean_smooth_tone_residual_halos(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    seed_mask: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    source_roi = source[y1:y2, x1:x2]
    target_roi = target[y1:y2, x1:x2]
    if source_roi.size == 0 or target_roi.size == 0 or seed_mask.size == 0:
        return None

    height, width = seed_mask.shape[:2]
    area = max(1, height * width)
    if area < 900:
        return None

    source_gray = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
    source_hsv = cv2.cvtColor(source_roi, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(target_roi, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(source_gray, 45, 135) > 0
    seed = seed_mask > 0
    background = (
        (target_gray >= 64)
        & (target_gray <= 232)
        & (target_hsv[:, :, 1] < 150)
        & ~cv2.dilate(
            edges.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        ).astype(bool)
        & ~cv2.dilate(
            seed.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        ).astype(bool)
    )

    if np.count_nonzero(background) < max(32, int(area * 0.035)):
        background = (
            (target_gray >= 64)
            & (target_gray <= 232)
            & (target_hsv[:, :, 1] < 150)
            & ~seed
        )
    if np.count_nonzero(background) < max(32, int(area * 0.035)):
        return None

    background_values = target_gray[background].astype(np.float32)
    background_median = float(np.median(background_values))
    background_q25 = float(np.percentile(background_values, 25))
    background_q35 = float(np.percentile(background_values, 35))
    background_std = float(np.std(background_values))
    background_edge_density = float(np.mean(edges[background]))
    bright_fraction = float(np.mean(background_values > 218))
    dark_fraction = float(np.mean(background_values < 72))
    mid_fraction = float(np.mean((background_values >= 76) & (background_values <= 218)))
    if (
        not (78.0 <= background_median <= 220.0)
        or background_std > 64.0
        or background_edge_density > 0.15
        or bright_fraction > 0.78
        or dark_fraction > 0.20
        or mid_fraction < 0.34
    ):
        return None

    residual_reference = min(
        background_median,
        background_q25 + 10.0,
        background_q35 + 6.0,
    )
    target_local_blur = cv2.GaussianBlur(target_gray, (0, 0), 5)
    source_bright_text = (
        (source_gray >= max(176, int(residual_reference - 14)))
        & (source_hsv[:, :, 1] < 180)
        & (target_hsv[:, :, 1] < 190)
        & (
            (source_gray >= int(residual_reference + 5))
            | (target_gray >= int(residual_reference + 5))
        )
    )
    seed_for_search = seed | source_bright_text
    near_seed = cv2.dilate(
        seed_for_search.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        iterations=1,
    ) > 0
    residual = (
        near_seed
        & (source_gray >= max(176, int(residual_reference - 8)))
        & (target_gray >= max(202, int(residual_reference + 14)))
        & (source_hsv[:, :, 1] < 185)
        & (target_hsv[:, :, 1] < 185)
    )
    residual |= (
        near_seed
        & (target_gray >= max(214, int(residual_reference + 24)))
        & (target_hsv[:, :, 1] < 135)
    )
    residual |= (
        near_seed
        & (target_gray >= max(184, int(residual_reference + 3)))
        & (source_gray >= max(186, int(residual_reference + 6)))
        & (
            (target_gray >= target_local_blur + 2)
            | (source_gray >= int(residual_reference + 18))
        )
        & (source_hsv[:, :, 1] < 190)
        & (target_hsv[:, :, 1] < 165)
    )
    residual &= ~edges
    residual = cv2.morphologyEx(
        residual.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats((residual > 0).astype(np.uint8), connectivity=8)
    clean_residual = np.zeros_like(residual)
    for label in range(1, component_count):
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        if component_area < 5 or component_area > int(area * 0.34):
            continue
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if component_width > width * 0.72 and component_height <= 5:
            continue
        if component_height > height * 0.94 and component_width <= 3:
            continue
        clean_residual[labels == label] = 255

    residual_count = int(np.count_nonzero(clean_residual > 0))
    residual_density = residual_count / float(area)
    if residual_count < 8 or residual_density > 0.28:
        return None

    local_background = background & (clean_residual <= 0)
    if np.count_nonzero(local_background) < max(32, int(area * 0.03)):
        return None
    fitted = np.empty_like(target_roi, dtype=np.float32)
    fallback = np.median(target_roi[local_background], axis=0).astype(np.float32)
    row_colors = np.empty((height, 3), dtype=np.float32)
    row_band = max(5, min(24, height // 14))
    bg_y = np.where(local_background)[0]
    for row_index in range(height):
        near_row = np.abs(bg_y - row_index) <= row_band
        row_mask = np.zeros((height, width), dtype=bool)
        if np.count_nonzero(near_row) >= 8:
            row_mask[np.where(local_background)[0][near_row], np.where(local_background)[1][near_row]] = True
            row_colors[row_index, :] = np.median(target_roi[row_mask], axis=0).astype(np.float32)
        else:
            row_colors[row_index, :] = fallback
    if height >= 7:
        row_colors = cv2.GaussianBlur(
            row_colors.reshape(height, 1, 3),
            (1, 0),
            sigmaX=0,
            sigmaY=max(1.8, min(10.0, height / 32.0)),
        ).reshape(height, 3)
    fitted[:, :, :] = row_colors[:, None, :]
    fitted = np.clip(fitted, 0, 255).astype(np.uint8)

    target_roi[clean_residual > 0] = fitted[clean_residual > 0]
    return clean_residual


def _polish_plain_background_text_residual(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    seed_mask: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    source_roi = source[y1:y2, x1:x2]
    target_roi = target[y1:y2, x1:x2]
    if source_roi.size == 0 or target_roi.size == 0 or seed_mask.size == 0:
        return None

    height, width = seed_mask.shape[:2]
    area = max(1, height * width)
    seed = seed_mask > 0
    seed_count = int(np.count_nonzero(seed))
    if seed_count < 8:
        return None

    source_gray = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
    source_hsv = cv2.cvtColor(source_roi, cv2.COLOR_BGR2HSV)
    source_edges = cv2.Canny(source_gray, 45, 135) > 0
    seed_density = seed_count / float(area)
    paper_fraction = float(np.mean((source_gray > 168) & (source_hsv[:, :, 1] < 135)))
    smooth_tone_fraction = float(np.mean((source_gray >= 82) & (source_gray <= 236) & (source_hsv[:, :, 1] < 165)))
    edge_density = float(np.mean(source_edges))
    plain_text_box = paper_fraction >= 0.76 and edge_density <= 0.22 and seed_density <= 0.70
    if seed_density > 0.70:
        return None
    if not _continuous_flat_background_allowed(source, coords, seed_mask):
        return None
    if paper_fraction < 0.48 and smooth_tone_fraction < 0.62:
        return None
    if edge_density > 0.20 and paper_fraction < 0.72:
        return None
    if _floating_region_has_mixed_character_tone(source, coords, (seed.astype(np.uint8) * 255)) and not plain_text_box:
        return None

    seed_guard = cv2.dilate(
        seed.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    ) > 0
    edge_guard = cv2.dilate(
        source_edges.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ) > 0
    background_upper = 255 if paper_fraction >= 0.70 else 246
    background = (
        (~seed_guard)
        & (~edge_guard)
        & (source_hsv[:, :, 1] < 170)
        & (source_gray > 70)
        & (source_gray <= background_upper)
    )
    if np.count_nonzero(background) < max(24, int(area * 0.025)):
        background = (
            (~seed_guard)
            & (source_hsv[:, :, 1] < 170)
            & (source_gray > 70)
            & (source_gray <= background_upper)
        )
    if np.count_nonzero(background) < max(24, int(area * 0.025)):
        return None

    background_gray = source_gray[background].astype(np.float32)
    background_std = float(np.std(background_gray))
    if background_std > (42.0 if paper_fraction >= 0.60 else 30.0):
        return None
    background_pixels = source_roi[background].astype(np.float32)
    background_color = np.median(background_pixels, axis=0)
    background_luma = float(np.median(background_gray))
    background_color_std = float(np.mean(np.std(background_pixels, axis=0)))
    continuous_flat = (
        background_color_std <= (20.0 if paper_fraction >= 0.68 else 14.0)
        and background_std <= (26.0 if paper_fraction >= 0.68 else 18.0)
        and edge_density <= (0.16 if paper_fraction >= 0.72 else 0.10)
    )

    kernel_size = 7 if paper_fraction >= 0.60 else 5
    cleanup = cv2.dilate(
        seed.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        iterations=1,
    ) > 0

    if continuous_flat:
        target_gray = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
        target_hsv = cv2.cvtColor(target_roi, cv2.COLOR_BGR2HSV)
        source_delta = np.abs(source_gray.astype(np.float32) - background_luma)
        target_delta = np.abs(target_gray.astype(np.float32) - background_luma)
        blackhat = cv2.morphologyEx(
            source_gray,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        )
        tophat = cv2.morphologyEx(
            source_gray,
            cv2.MORPH_TOPHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        )
        color_distance = np.linalg.norm(source_roi.astype(np.float32) - background_color, axis=2)
        threshold = 9.0 if paper_fraction >= 0.72 else 13.0
        residual_raw = (
            (source_hsv[:, :, 1] < 185)
            & (target_hsv[:, :, 1] < 190)
            & (
                (source_delta >= threshold)
                | (target_delta >= threshold + 4.0)
                | (blackhat >= 5)
                | (tophat >= 5)
                | (color_distance >= threshold + 6.0)
            )
        )
        if paper_fraction < 0.78:
            search = cv2.dilate(
                seed.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)),
                iterations=1,
            ).astype(bool)
            residual_raw &= search
        residual_raw &= ~edge_guard | (source_delta >= threshold + 8.0)
        residual_raw = cv2.morphologyEx(
            residual_raw.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        residual_filtered = np.zeros_like(residual_raw)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (residual_raw > 0).astype(np.uint8), connectivity=8
        )
        for label in range(1, component_count):
            component = labels == label
            component_area = int(stats[label, cv2.CC_STAT_AREA])
            cx = int(stats[label, cv2.CC_STAT_LEFT])
            cy = int(stats[label, cv2.CC_STAT_TOP])
            cw = int(stats[label, cv2.CC_STAT_WIDTH])
            ch = int(stats[label, cv2.CC_STAT_HEIGHT])
            if component_area < 3 or component_area > max(2600, int(area * 0.28)):
                continue
            touches_border = cx <= 0 or cy <= 0 or (cx + cw) >= width or (cy + ch) >= height
            if touches_border and component_area > max(32, int(area * 0.018)):
                continue
            if cw > int(width * 0.90) and ch <= 10:
                continue
            if ch > int(height * 0.92) and cw <= 10:
                continue
            component_density = component_area / float(max(1, cw * ch))
            seed_overlap = int(np.count_nonzero(component & seed_guard))
            dense_paper_residual = (
                paper_fraction >= 0.78
                and component_area <= max(900, int(area * 0.16))
                and seed_overlap >= max(2, int(component_area * 0.025))
            )
            if component_density > 0.82 and component_area > 18 and not dense_paper_residual:
                continue
            residual_filtered[component] = 255
        residual_count = int(np.count_nonzero(residual_filtered > 0))
        residual_density = residual_count / float(area)
        if 4 <= residual_count and residual_density <= (0.42 if paper_fraction >= 0.78 else 0.24):
            residual_filtered = cv2.dilate(
                residual_filtered,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            ) > 0
            cleanup = cleanup | residual_filtered

    cleanup_density = float(np.count_nonzero(cleanup)) / float(area)
    if cleanup_density > 0.48 and paper_fraction < 0.74:
        return None

    if not _apply_local_plane_fill(source_roi, target_roi, cleanup, background):
        fill_color = np.median(source_roi[background], axis=0).astype(np.uint8)
        target_roi[cleanup] = fill_color
    return cleanup.astype(np.uint8) * 255


def _clean_halftone_residual_deviation(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    seed_mask: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    source_roi = source[y1:y2, x1:x2]
    target_roi = target[y1:y2, x1:x2]
    if source_roi.size == 0 or target_roi.size == 0 or seed_mask.size == 0:
        return None

    height, width = seed_mask.shape[:2]
    area = max(1, height * width)
    if area < 900:
        return None

    seed = seed_mask > 0
    seed_count = int(np.count_nonzero(seed))
    if seed_count < 8:
        return None
    seed_density = seed_count / float(area)
    if seed_density > 0.82:
        return None

    source_gray = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
    target_hsv = cv2.cvtColor(target_roi, cv2.COLOR_BGR2HSV)
    source_hsv = cv2.cvtColor(source_roi, cv2.COLOR_BGR2HSV)

    low_saturation = float(np.mean(source_hsv[:, :, 1] < 175))
    mid_fraction = float(np.mean((source_gray >= 72) & (source_gray <= 225)))
    dark_fraction = float(np.mean(source_gray < 58))
    edge_density = float(np.mean(cv2.Canny(source_gray, 45, 135) > 0))
    if (
        low_saturation < 0.78
        or mid_fraction < 0.24
        or dark_fraction > 0.34
        or edge_density > 0.36
    ):
        return None

    search = cv2.dilate(
        seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ).astype(bool)
    background = ~cv2.dilate(
        seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    ).astype(bool)
    background &= target_hsv[:, :, 1] < 150
    background &= (target_gray > 72) & (target_gray < 238)
    if np.count_nonzero(background) < max(24, int(area * 0.025)):
        background = ~cv2.dilate(
            seed.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        ).astype(bool)
        background &= target_hsv[:, :, 1] < 170
        background &= (target_gray > 58) & (target_gray < 242)
    if np.count_nonzero(background) < max(10, int(area * 0.01)):
        return None

    fallback = np.median(target_roi[background], axis=0).astype(np.float32)
    bg_y, bg_x = np.where(background)
    row_colors = np.tile(fallback, (height, 1)).astype(np.float32)
    row_band = max(5, min(28, height // 10))
    for row_index in range(height):
        near_row = np.abs(bg_y - row_index) <= row_band
        if np.count_nonzero(near_row) >= 6:
            row_colors[row_index, :] = np.median(
                target_roi[bg_y[near_row], bg_x[near_row]],
                axis=0,
            ).astype(np.float32)
    if height >= 7:
        row_colors = cv2.GaussianBlur(
            row_colors.reshape(height, 1, 3),
            (1, 0),
            sigmaX=0,
            sigmaY=max(1.5, min(9.0, height / 32.0)),
        ).reshape(height, 3)

    row_gray = (
        row_colors[:, 0] * 0.114
        + row_colors[:, 1] * 0.299
        + row_colors[:, 2] * 0.587
    ).astype(np.float32)
    expected_gray = np.repeat(row_gray[:, None], width, axis=1)
    deviation = target_gray.astype(np.float32) - expected_gray
    source_text_signal = ((source_gray < 150) | (source_gray > 216)) & search
    residual = (
        search
        & source_text_signal
        & (target_hsv[:, :, 1] < 180)
        & ((deviation < -7.0) | (deviation > 12.0))
    )
    residual &= ~((source_gray < 90) & (target_gray < 95))
    residual = cv2.morphologyEx(
        residual.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (residual > 0).astype(np.uint8),
        connectivity=8,
    )
    clean_residual = np.zeros_like(residual)
    for label in range(1, component_count):
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        if component_area < 4 or component_area > int(area * 0.22):
            continue
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if component_width > width * 0.78 and component_height < 7:
            continue
        if component_height > height * 0.96 and component_width <= 3:
            continue
        clean_residual[labels == label] = 255

    clean_count = int(np.count_nonzero(clean_residual > 0))
    if clean_count < 8:
        return None
    clean_density = clean_count / float(area)
    if clean_density > 0.12:
        return None
    if clean_density > 0.36 and _floating_region_is_art_sensitive(source, coords, seed_mask):
        return None

    fitted = np.repeat(row_colors[:, None, :], width, axis=1)
    target_roi[clean_residual > 0] = np.clip(fitted, 0, 255).astype(np.uint8)[clean_residual > 0]
    return clean_residual


def _rowwise_midtone_texture_repair(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return None

    height, width = roi.shape[:2]
    area = max(1, height * width)
    seed_count = int(np.count_nonzero(mask_roi > 0))
    if seed_count < 40:
        return None
    seed_density = seed_count / float(area)
    if width < 24 or height < 36:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    low_saturation = float(np.mean(hsv[:, :, 1] < 170))
    mid_fraction = float(np.mean((gray >= 62) & (gray <= 228) & (hsv[:, :, 1] < 175)))
    dark_fraction = float(np.mean(gray < 58))
    bright_fraction = float(np.mean(gray > 235))
    edge_density = float(np.mean(cv2.Canny(gray, 45, 135) > 0))
    compact_outlined_caption = (
        width <= 150
        and height <= 112
        and seed_density >= 0.52
        and low_saturation >= 0.94
        and mid_fraction >= 0.24
        and 0.10 <= bright_fraction <= 0.58
        and dark_fraction <= 0.50
        and edge_density <= 0.36
    )
    if not (
        low_saturation >= 0.88
        and mid_fraction >= 0.30
        and dark_fraction <= 0.24
        and bright_fraction <= 0.58
        and edge_density <= 0.32
    ) and not compact_outlined_caption:
        return None
    if _floating_region_has_mixed_character_tone(source, coords, mask_roi) and not compact_outlined_caption:
        return None

    outlined = _outlined_floating_source_mask(source, coords, mask_roi)
    if outlined is not None:
        outlined_count = int(np.count_nonzero(outlined > 0))
        outlined_density = outlined_count / float(area)
        if outlined_density > 0.96 and seed_density < 0.96:
            repair = (mask_roi > 0).astype(np.uint8) * 255
        elif (
            outlined_density >= 0.78
            and seed_density <= 0.62
            and bright_fraction >= 0.42
            and dark_fraction >= 0.08
        ):
            repair = (mask_roi > 0).astype(np.uint8) * 255
        elif outlined_count >= seed_count * 0.55:
            repair = outlined
        else:
            repair = (mask_roi > 0).astype(np.uint8) * 255
    else:
        repair = (mask_roi > 0).astype(np.uint8) * 255
    # `_precise_floating_text_repair_mask`'s component filter assumes glyphs
    # sit with some margin inside the crop -- it rejects components that
    # touch the crop border or span most of the crop's own height/width, on
    # the assumption that such a component is background art, not text. A
    # TIGHT, unpadded box (this caller passes `coords` directly, the
    # constraint's own bounding box) breaks that assumption: real glyph
    # strokes legitimately touch the edges and span nearly the full crop
    # height when there is no margin. Measured directly on new_sample_14's
    # "썸머스쿨" label: 4 of 7 raw dark-ink connected components were
    # rejected this way, missing most of the glyph strokes -- exactly the
    # ghost-mask pattern observed in the render. Padding the coords before
    # this specific call gives real glyphs room to sit inside the margin
    # the filter expects, without touching the shared function's own
    # border/size heuristics (used by many other callers with their own,
    # already-correct padding conventions).
    src_h, src_w = source.shape[:2]
    precise_pad = 14
    ppx1 = max(0, x1 - precise_pad)
    ppy1 = max(0, y1 - precise_pad)
    ppx2 = min(src_w, x2 + precise_pad)
    ppy2 = min(src_h, y2 + precise_pad)
    precise_repair_padded = _precise_floating_text_repair_mask(source, (ppx1, ppy1, ppx2, ppy2))
    precise_repair = None
    if precise_repair_padded is not None:
        precise_repair = precise_repair_padded[
            y1 - ppy1 : y1 - ppy1 + height, x1 - ppx1 : x1 - ppx1 + width
        ]
    if precise_repair is not None:
        precise_count = int(np.count_nonzero(precise_repair > 0))
        precise_density = precise_count / float(area)
        preliminary_density = float(np.count_nonzero(repair > 0)) / float(area)
        # A precise mask that covers LESS of the seed-detected text than the
        # seed itself is under-detecting, not refining -- swapping it in
        # replaces real glyph coverage with a sparser mask, which can leave
        # small isolated fragments the model then reconstructs as visible
        # speckle instead of blank background (verified: caused green
        # text-shaped speckle on new_sample_13's vertical wall strip when
        # this guard was absent). Require the precise mask to cover most of
        # what the seed already found before trusting it as a replacement.
        if precise_count < 0.55 * max(1, seed_count):
            precise_repair = None
        elif (
            precise_count >= max(18, int(area * 0.008))
            and precise_density <= 0.46
            and (
                compact_outlined_caption
                or seed_density >= 0.42
                or preliminary_density >= 0.44
            )
        ):
            repair = precise_repair
    repair = cv2.morphologyEx(
        repair,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    repair_density_before_dilate = float(np.count_nonzero(repair > 0)) / float(area)
    if not (compact_outlined_caption and repair_density_before_dilate >= 0.78):
        repair = cv2.dilate(
            repair,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
    repair_bool = repair > 0
    repair_count = int(np.count_nonzero(repair_bool))
    repair_density = repair_count / float(area)
    max_repair_density = 1.0 if compact_outlined_caption else 0.94
    if repair_count < 60 or repair_density < 0.10 or repair_density > max_repair_density:
        return None

    img_h, img_w = source.shape[:2]
    pad = max(28, min(110, int(max(width, height) * 0.45)))
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(img_w, x2 + pad)
    cy2 = min(img_h, y2 + pad)
    crop = source[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    crop_edges = cv2.dilate(
        (cv2.Canny(crop_gray, 45, 135) > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ).astype(bool)
    crop_repair = np.zeros(crop_gray.shape, dtype=np.uint8)
    crop_repair[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1] = repair
    repair_guard = cv2.dilate(
        crop_repair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=1,
    ).astype(bool)
    background = (
        ~repair_guard
        & ~crop_edges
        & (crop_hsv[:, :, 1] < 170)
        & (crop_gray >= 48)
        & (crop_gray <= 238)
    )
    if int(np.count_nonzero(background)) < max(100, int(crop_gray.size * 0.025)):
        background = (
            ~repair_guard
            & (crop_hsv[:, :, 1] < 180)
            & (crop_gray >= 42)
            & (crop_gray <= 242)
        )
    if int(np.count_nonzero(background)) >= max(80, int(crop_gray.size * 0.018)):
        background_values = crop_gray[background].astype(np.float32)
        background_median = float(np.median(background_values))
        background_mad = float(np.median(np.abs(background_values - background_median)))
        tone_window = max(28.0, min(58.0, background_mad * 3.5 + 18.0))
        background = (
            background
            & (np.abs(crop_gray.astype(np.float32) - background_median) <= tone_window)
        )
    background_count = int(np.count_nonzero(background))
    if background_count < max(80, int(crop_gray.size * 0.018)):
        return None

    background_values = crop_gray[background].astype(np.float32)
    background_std = float(np.std(background_values))
    background_edge_density = float(np.mean(crop_edges[background]))
    smooth_low_detail_background = (
        not compact_outlined_caption
        and background_std <= 42.0
        and background_edge_density <= 0.060
        and edge_density <= 0.24
    )
    if smooth_low_detail_background:
        background_y, background_x = np.where(background)
        fit_y = background_y
        fit_x = background_x
        if len(fit_x) > 9000:
            step = max(1, len(fit_x) // 9000)
            fit_x = fit_x[::step]
            fit_y = fit_y[::step]

        fallback_color = np.median(crop[background], axis=0).astype(np.float32)
        row_colors = np.empty((crop_gray.shape[0], 3), dtype=np.float32)
        row_band = max(8, min(34, crop_gray.shape[0] // 10))
        for row_index in range(crop_gray.shape[0]):
            near_row = np.abs(background_y - row_index) <= row_band
            if int(np.count_nonzero(near_row)) >= 14:
                row_colors[row_index, :] = np.median(
                    crop[background_y[near_row], background_x[near_row]],
                    axis=0,
                ).astype(np.float32)
            else:
                row_colors[row_index, :] = fallback_color
        row_colors = cv2.GaussianBlur(
            row_colors.reshape(crop_gray.shape[0], 1, 3),
            (1, 0),
            sigmaX=0,
            sigmaY=max(1.8, min(9.0, crop_gray.shape[0] / 28.0)),
        ).reshape(crop_gray.shape[0], 3)

        design = np.column_stack(
            [
                fit_x.astype(np.float32),
                fit_y.astype(np.float32),
                np.ones_like(fit_x, dtype=np.float32),
            ]
        )
        grid_y, grid_x = np.indices(crop_gray.shape, dtype=np.float32)
        fitted_crop = np.empty_like(crop, dtype=np.float32)
        for channel_idx in range(3):
            values = crop[fit_y, fit_x, channel_idx].astype(np.float32)
            coeffs, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
            plane = grid_x * coeffs[0] + grid_y * coeffs[1] + coeffs[2]
            row_fit = np.repeat(row_colors[:, channel_idx][:, None], crop_gray.shape[1], axis=1)
            fitted_crop[:, :, channel_idx] = plane * 0.38 + row_fit * 0.62
        fitted_crop = cv2.GaussianBlur(
            np.clip(fitted_crop, 0, 255).astype(np.uint8),
            (0, 0),
            0.65,
        )

        target_roi = target[y1:y2, x1:x2]
        fitted_roi = fitted_crop[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1]
        if fitted_roi.shape[:2] != target_roi.shape[:2]:
            return None
        alpha = cv2.GaussianBlur(repair_bool.astype(np.float32), (0, 0), 0.72)
        alpha = np.clip(alpha, 0.0, 1.0)
        blended = (
            fitted_roi.astype(np.float32) * alpha[..., None]
            + target_roi.astype(np.float32) * (1.0 - alpha[..., None])
        ).astype(np.uint8)
        write_mask = alpha > 0.06
        target_roi[write_mask] = blended[write_mask]
        return (write_mask.astype(np.uint8) * 255)

    background_y, background_x = np.where(background)
    rng_seed = (
        (int(x1) + 1) * 73856093
        ^ (int(y1) + 1) * 19349663
        ^ (int(width) + 1) * 83492791
        ^ (int(height) + 1) * 2654435761
    ) & 0xFFFFFFFF
    rng = np.random.default_rng(rng_seed)
    target_roi = target[y1:y2, x1:x2]
    sampled = target_roi.copy()
    repair_y, repair_x = np.where(repair_bool)
    if len(repair_y) < 10:
        return None
    all_indices = np.arange(background_count)
    band = max(8, min(34, height // 4))
    dense_compact_sampling = compact_outlined_caption and repair_density >= 0.78
    changed = np.zeros(repair.shape, dtype=np.uint8)
    for row in range(height):
        row_positions = np.where(repair_y == row)[0]
        if len(row_positions) == 0:
            continue
        global_row = y1 + row - cy1
        near_row = all_indices if dense_compact_sampling else np.where(np.abs(background_y - global_row) <= band)[0]
        if len(near_row) < max(10, width // 5):
            near_row = all_indices
        chosen = rng.choice(near_row, size=len(row_positions), replace=True)
        sampled[row, repair_x[row_positions], :] = crop[
            background_y[chosen],
            background_x[chosen],
        ]
        changed[row, repair_x[row_positions]] = 255

    if int(np.count_nonzero(changed > 0)) < max(60, int(seed_count * 0.50)):
        return None
    sampled = cv2.GaussianBlur(sampled, (3, 3), 0)
    alpha = cv2.GaussianBlur(repair_bool.astype(np.float32), (0, 0), 0.72)
    alpha = np.clip(alpha, 0.0, 1.0)
    blended = (
        sampled.astype(np.float32) * alpha[..., None]
        + target_roi.astype(np.float32) * (1.0 - alpha[..., None])
    ).astype(np.uint8)
    write_mask = alpha > 0.06
    target_roi[write_mask] = blended[write_mask]
    return changed


def _rowwise_textured_caption_repair(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return None

    height, width = roi.shape[:2]
    area = max(1, height * width)
    mask_count = int(np.count_nonzero(mask_roi > 0))
    if mask_count < 40:
        return None
    mask_density = mask_count / float(area)
    if mask_density < 0.22 or mask_density > 0.64:
        return None
    if height < 90 or height < width * 1.12:
        return None
    if _floating_region_has_mixed_character_tone(source, coords, mask_roi):
        return None
    if _floating_region_is_art_sensitive(source, coords, mask_roi):
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    unmasked = mask_roi <= 0
    if int(np.count_nonzero(unmasked)) < max(80, int(area * 0.08)):
        return None

    bg_gray = gray[unmasked]
    bg_sat = hsv[:, :, 1][unmasked]
    bg_median = float(np.median(bg_gray))
    bg_std = float(np.std(bg_gray.astype(np.float32)))
    bg_edge = float(np.mean(cv2.Canny(gray, 45, 135)[unmasked] > 0))
    bg_paper = float(np.mean((bg_gray > 170) & (bg_sat < 145)))
    bg_dark = float(np.mean(bg_gray < 95))
    if not (
        145.0 <= bg_median <= 238.0
        and bg_paper >= 0.62
        and bg_dark <= 0.075
        and (bg_std >= 22.0 or bg_edge >= 0.080)
    ):
        return None

    repair = cv2.dilate(
        (mask_roi > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    )
    repair_count = int(np.count_nonzero(repair > 0))
    repair_density = repair_count / float(area)
    if repair_count < 60 or repair_density > 0.76:
        return None

    repair_bool = repair > 0
    target_roi = target[y1:y2, x1:x2]
    target_patch = target_roi.copy()
    valid_background = (~repair_bool) & (gray > 45) & (gray < 250)
    if int(np.count_nonzero(valid_background)) < max(80, int(area * 0.06)):
        return None
    global_fill = np.median(roi[valid_background], axis=0).astype(np.uint8)

    changed = np.zeros(repair.shape, dtype=np.uint8)
    for row in range(height):
        row_mask = repair_bool[row]
        if not np.any(row_mask):
            continue
        band_y1 = max(0, row - 5)
        band_y2 = min(height, row + 6)
        band_background = valid_background[band_y1:band_y2]
        if int(np.count_nonzero(band_background)) >= 22:
            fill = np.median(roi[band_y1:band_y2][band_background], axis=0).astype(np.uint8)
        else:
            fill = global_fill
        target_patch[row, row_mask] = fill
        changed[row, row_mask] = 255

    if int(np.count_nonzero(changed > 0)) < max(60, int(mask_count * 0.55)):
        return None
    target_roi[:, :] = target_patch
    return changed


def _semantic_clip_flat_fill(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
    region_clip: np.ndarray,
    seg_mask: np.ndarray,
) -> np.ndarray | None:
    """Deterministic fill for a semantically-rescued region with a flat interior.

    Same principle as the bubble path's flat_fill_uniform_interior (and the same
    measured thresholds): when the pristine interior around the glyphs is provably
    uniform and bright, skip the ML inpainter entirely -- it prevents invention at
    the source rather than trying to bound it afterwards.

    Clipping alone is not sufficient for these constraints. The clip stops the model
    painting OUTSIDE the text region, but inside it the model still fills from a
    +-256..512px context window, so on a small bubble surrounded by dense art it
    reconstructs that art inside the bubble (measured: external_ja_2 id 1, where the
    clip restored the outline but left character hair painted across the interior).
    A flat white bubble interior needs no reconstruction at all.

    Returns the committed mask, or None to fall through to the normal cascade.
    """
    x1, y1, x2, y2 = coords
    src_roi = source[y1:y2, x1:x2]
    if src_roi.size == 0:
        return None
    glyph = _refined_floating_source_mask(source, seg_mask, coords)
    if glyph is None:
        return None
    glyph = cv2.bitwise_and(glyph, region_clip)
    if int(np.count_nonzero(glyph > 0)) < 6:
        return None

    gray = cv2.cvtColor(src_roi, cv2.COLOR_BGR2GRAY)
    kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Judge "is the interior flat" on pixels clear of the glyphs AND their
    # anti-aliased halo, so the ink itself cannot drag the statistics down.
    grown = cv2.dilate(glyph, kernel_5, iterations=2)
    interior = (region_clip > 0) & (grown == 0)
    vals = gray[interior]
    if vals.size < 40:
        return None
    if not (float(np.percentile(vals, 30)) >= 225.0 and float(np.std(vals)) <= 12.0):
        return None

    commit = cv2.bitwise_and(cv2.dilate(glyph, kernel_5, iterations=2), region_clip)
    if int(np.count_nonzero(commit > 0)) < 10:
        return None
    fill_bgr = np.median(src_roi[interior].reshape(-1, 3), axis=0)
    target[y1:y2, x1:x2][commit > 0] = fill_bgr.astype(np.uint8)
    return commit


def _tight_floating_stroke_repair(
    source: np.ndarray,
    target: np.ndarray,
    seg_mask: np.ndarray,
    coords: tuple[int, int, int, int],
    anime_model,
    anime_device,
    clip_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Clip wrapper around the repair cascade below.

    Threading a bound through each of the cascade's ~25 branches is not
    reviewable: several helpers paint through their own grown masks and some use
    padded context crops that write outside `coords` entirely. So a caller-supplied
    clip is enforced twice -- the body clips its SEED (so the model never sees an
    oversized hole), and this wrapper reverts every page pixel the cascade touched
    outside the clip. With clip_mask=None both are inert and behaviour is
    byte-identical to before.

    Clipping the returned mask is load-bearing, not cosmetic: the caller writes it
    into the page-level floating mask that later drives the residual sweep, which
    must not be told more was cleaned than actually was.
    """
    if clip_mask is None:
        return _tight_floating_stroke_repair_body(
            source, target, seg_mask, coords, anime_model, anime_device
        )
    x1, y1, x2, y2 = coords
    region_clip = clip_mask[y1:y2, x1:x2]
    if region_clip.size == 0 or int(np.count_nonzero(region_clip)) < 10:
        return None
    flat = _semantic_clip_flat_fill(source, target, coords, region_clip, seg_mask)
    if flat is not None:
        return flat
    before = target.copy()
    mask_roi = _tight_floating_stroke_repair_body(
        source, target, seg_mask, coords, anime_model, anime_device,
        clip_mask=clip_mask,
    )
    keep = np.zeros(target.shape[:2], dtype=bool)
    keep[y1:y2, x1:x2] = region_clip > 0
    target[~keep] = before[~keep]
    if mask_roi is None:
        return None
    mask_roi = cv2.bitwise_and(mask_roi, region_clip)
    if int(np.count_nonzero(mask_roi > 0)) < 10:
        target[y1:y2, x1:x2] = before[y1:y2, x1:x2]
        return None
    return mask_roi


def _tight_floating_stroke_repair_body(
    source: np.ndarray,
    target: np.ndarray,
    seg_mask: np.ndarray,
    coords: tuple[int, int, int, int],
    anime_model,
    anime_device,
    clip_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    mask_roi = _refined_floating_source_mask(source, seg_mask, coords)
    if clip_mask is not None:
        # Clip the SEED, not just the result. Every branch below derives its repair
        # mask from mask_roi, and several hand it to AnimeLaMa -- which fills the
        # hole from a +-256..512px context window. Reverting out-of-clip pixels
        # afterwards would still leave surrounding artwork painted INSIDE the clip.
        # Bounding the hole up front is what makes the fill correct, not merely
        # contained. The seed can itself be far larger than the glyph strokes.
        mask_roi = cv2.bitwise_and(mask_roi, clip_mask[y1:y2, x1:x2])
    if np.count_nonzero(mask_roi > 0) < 6:
        return None
    mask_density = float(np.count_nonzero(mask_roi > 0)) / float(max(1, mask_roi.size))
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    precise_only = False
    if roi.size:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        precise_only = (
            mask_density >= 0.24
            and float(np.std(gray.astype(np.float32))) >= 58.0
            and float(np.mean(cv2.Canny(gray, 45, 135) > 0)) >= 0.055
            and (float(np.mean(gray < 118)) >= 0.055 or float(np.mean(gray > 220)) >= 0.36)
        )

    height, width = roi.shape[:2] if roi.size else (0, 0)
    if roi.size and _pure_paper_repair_context(source, coords, mask_roi):
        paper_cleanup = cv2.dilate(
            (mask_roi > 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        paper_cleanup = cv2.morphologyEx(
            paper_cleanup,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        if _fill_paper_tone_repair(roi, target[y1:y2, x1:x2], paper_cleanup):
            return paper_cleanup

    midtone_texture_mask = _rowwise_midtone_texture_repair(
        source,
        target,
        coords,
        mask_roi,
    )
    if midtone_texture_mask is not None:
        return midtone_texture_mask

    compact_smooth_text = width <= 130 and 45 <= height <= 170
    if compact_smooth_text:
        low_frequency_mask = _low_frequency_smooth_background_repair(source, target, coords, mask_roi)
        if low_frequency_mask is not None:
            return low_frequency_mask

    halftone_outlined_mask = _halftone_outlined_text_model_repair(
        source,
        target,
        coords,
        mask_roi,
        anime_model,
        anime_device,
    )
    if halftone_outlined_mask is not None:
        return halftone_outlined_mask

    dense_smooth_mask = _dense_smooth_tone_source_text_fill(source, target, coords, mask_roi)
    if dense_smooth_mask is not None:
        return dense_smooth_mask

    textured_caption_mask = _rowwise_textured_caption_repair(source, target, coords, mask_roi)
    if textured_caption_mask is not None:
        return textured_caption_mask

    dark_surface_seed = mask_roi
    if roi.size:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if float(np.mean(roi_gray < 140)) >= 0.30:
            reverse_dark_mask = _fill_reverse_dark_balloon_text(
                source,
                target,
                coords,
                mask_roi,
            )
            if reverse_dark_mask is not None:
                return reverse_dark_mask
            outlined_for_dark = _outlined_floating_source_mask(source, coords, mask_roi)
            if outlined_for_dark is not None:
                dark_surface_seed = outlined_for_dark
            dark_surface_seed = cv2.morphologyEx(
                (dark_surface_seed > 0).astype(np.uint8) * 255,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=2,
            )
            dark_surface_seed = cv2.dilate(
                dark_surface_seed,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            dark_surface_mask = _fill_dark_surface_source_strokes(
                source,
                target,
                coords,
                dark_surface_seed,
            )
            if dark_surface_mask is not None:
                return dark_surface_mask

    pure_paper_mask = _pure_paper_source_inpaint(source, target, coords, mask_roi)
    if pure_paper_mask is not None:
        return pure_paper_mask

    smooth_gradient_mask = _smooth_gradient_source_text_fill(source, target, coords, mask_roi)
    if smooth_gradient_mask is not None:
        return smooth_gradient_mask

    if roi.size:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if float(np.mean(roi_gray < 140)) >= 0.30:
            reverse_dark_mask = _fill_reverse_dark_balloon_text(
                source,
                target,
                coords,
                mask_roi,
            )
            if reverse_dark_mask is not None:
                return reverse_dark_mask
            outlined_for_dark = _outlined_floating_source_mask(source, coords, mask_roi)
            if outlined_for_dark is not None:
                dark_surface_seed = outlined_for_dark
            dark_surface_seed = cv2.morphologyEx(
                (dark_surface_seed > 0).astype(np.uint8) * 255,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=2,
            )
            dark_surface_seed = cv2.dilate(
                dark_surface_seed,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            dark_surface_mask = _fill_dark_surface_source_strokes(
                source, target, coords, dark_surface_seed
            )
            if dark_surface_mask is not None:
                return dark_surface_mask

    mixed_dark_mask = _mixed_dark_surface_source_inpaint(source, target, coords, mask_roi)
    if mixed_dark_mask is not None:
        return mixed_dark_mask

    smooth_caption_block = _fill_smooth_tone_caption_block(source, target, coords, mask_roi)
    if smooth_caption_block is not None:
        return smooth_caption_block

    bounded_tone_mask = _fill_bounded_tone_source_strokes(source, target, coords, mask_roi)
    if bounded_tone_mask is not None:
        return bounded_tone_mask

    dark_surface_mask = _fill_dark_surface_source_strokes(source, target, coords, dark_surface_seed)
    if dark_surface_mask is not None:
        return dark_surface_mask

    wide_caption = (x2 - x1) >= max(120, int((y2 - y1) * 1.45))
    if not wide_caption:
        early_local_tone_mask = _fill_local_tone_source_strokes(source, target, coords, mask_roi)
        if early_local_tone_mask is not None:
            return early_local_tone_mask

    bright_textured_mask = _bright_textured_source_repair(source, target, coords, mask_roi)
    if bright_textured_mask is not None:
        return bright_textured_mask

    if _fill_high_contrast_light_text_mask(source, target, coords, mask_roi):
        return mask_roi

    if precise_only:
        stroke_only_mask = _stroke_only_inpaint_repair(
            source,
            target,
            coords,
            mask_roi,
            radius=1.15,
            max_seed_density=0.64,
            max_repair_density=0.60,
        )
        if stroke_only_mask is not None:
            return stroke_only_mask

    dilated_anime_mask = _dilated_anime_caption_repair(
        source,
        target,
        coords,
        mask_roi,
        anime_model,
        anime_device,
    )
    if dilated_anime_mask is not None:
        return dilated_anime_mask

    if anime_model is not None and roi.size:
        outlined_for_model = _outlined_floating_source_mask(source, coords, mask_roi)
        if outlined_for_model is not None:
            outlined_density = float(np.count_nonzero(outlined_for_model > 0)) / float(max(1, outlined_for_model.size))
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            edge_density = float(np.mean(cv2.Canny(gray, 45, 135) > 0))
            paper_fraction = float(np.mean((gray > 172) & (hsv[:, :, 1] < 135)))
            dark_fraction = float(np.mean(gray < 82))
            low_saturation = float(np.mean(hsv[:, :, 1] < 150))
            if (
                outlined_density >= 0.48
                and height >= 90
                and height >= width * 1.20
                and low_saturation >= 0.90
                and paper_fraction >= 0.42
                and dark_fraction <= 0.22
                and edge_density <= 0.32
                and not _floating_region_is_art_sensitive(source, coords, outlined_for_model)
            ):
                region_mask = np.zeros(source.shape[:2], dtype=np.uint8)
                region_mask[y1:y2, x1:x2] = outlined_for_model
                before = target[y1:y2, x1:x2].copy()
                _anime_lama_local_crop(
                    anime_model,
                    anime_device,
                    target,
                    region_mask,
                    source.shape[0],
                    source.shape[1],
                    x1,
                    y1,
                    x2,
                    y2,
                )
                changed = np.any(before != target[y1:y2, x1:x2], axis=2)
                changed_mask = ((changed & (outlined_for_model > 0)).astype(np.uint8)) * 255
                if np.count_nonzero(changed_mask > 0) >= 6:
                    _local_repair_tone_match(source, target, coords, outlined_for_model)
                    return outlined_for_model

    outlined_mask = _outlined_floating_text_repair(source, target, coords, mask_roi)
    if outlined_mask is not None:
        return outlined_mask

    stroke_only_mask = _stroke_only_inpaint_repair(source, target, coords, mask_roi)
    if stroke_only_mask is not None:
        return stroke_only_mask

    if mask_density >= 0.72 and _fill_flat_background_text_mask(source, target, coords, mask_roi):
        return mask_roi

    mixed_tone_mask = _mixed_tone_model_repair(
        source,
        target,
        coords,
        mask_roi,
        anime_model,
        anime_device,
    )
    if mixed_tone_mask is not None:
        return mixed_tone_mask

    smooth_caption_mask = _fill_smooth_tone_caption_strokes(source, target, coords, mask_roi)
    if smooth_caption_mask is not None:
        return smooth_caption_mask

    local_tone_mask = _fill_local_tone_source_strokes(source, target, coords, mask_roi)
    if local_tone_mask is not None:
        return local_tone_mask

    if _floating_cleanup_should_fail_closed(source, coords, mask_roi):
        return None

    region_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    region_mask[y1:y2, x1:x2] = mask_roi
    if _floating_region_is_art_sensitive(source, coords, mask_roi):
        return None

    before = target[y1:y2, x1:x2].copy()
    if anime_model is not None:
        _anime_lama_local_crop(
            anime_model,
            anime_device,
            target,
            region_mask,
            source.shape[0],
            source.shape[1],
            x1,
            y1,
            x2,
            y2,
        )
    else:
        _opencv_local_stroke_repair(target, region_mask, x1, y1, x2, y2, radius=2.0)

    changed = np.any(before != target[y1:y2, x1:x2], axis=2)
    changed_mask = ((changed & (mask_roi > 0)).astype(np.uint8)) * 255
    if np.count_nonzero(changed_mask > 0) < 6:
        return None
    return mask_roi


def _floating_region_is_art_sensitive(
    source: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> bool:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return False

    area = max(1, mask_roi.shape[0] * mask_roi.shape[1])
    mask_density = float(np.count_nonzero(mask_roi > 0)) / area
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    paper_fraction = float(np.mean((gray > 172) & (hsv[:, :, 1] < 115)))
    dark_fraction = float(np.mean(gray < 58))
    edge_density = float(np.mean(cv2.Canny(gray, 45, 135) > 0))

    if mask_density > 0.42:
        return False
    if paper_fraction >= 0.72 and edge_density < 0.26:
        return False
    if dark_fraction >= 0.62 and edge_density < 0.30:
        return True
    return edge_density >= 0.10 or paper_fraction < 0.52


def _floating_region_has_mixed_character_tone(
    source: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray | None = None,
) -> bool:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    height, width = roi.shape[:2]
    area = max(1, height * width)
    if area < 900:
        return False

    mask_density = 0.0
    if mask_roi is not None and mask_roi.size:
        mask_density = float(np.count_nonzero(mask_roi > 0)) / float(max(1, mask_roi.size))

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    edges = cv2.Canny(gray, 45, 135) > 0
    dark_fraction = float(np.mean(gray < 86))
    deep_dark_fraction = float(np.mean(gray < 48))
    mid_tone_fraction = float(np.mean((gray >= 70) & (gray <= 210) & (saturation < 170)))
    bright_fraction = float(np.mean(gray > 218))
    edge_density = float(np.mean(edges))
    low_saturation = float(np.mean(saturation < 160))
    luma_std = float(np.std(gray.astype(np.float32)))

    if low_saturation < 0.82:
        return False
    if (
        dark_fraction >= 0.18
        and mid_tone_fraction >= 0.10
        and bright_fraction >= 0.12
        and luma_std >= 72.0
        and edge_density >= 0.045
    ):
        return True
    if (
        mask_density >= 0.24
        and dark_fraction >= 0.24
        and mid_tone_fraction >= 0.14
        and luma_std >= 78.0
    ):
        return True
    return deep_dark_fraction >= 0.20 and mid_tone_fraction >= 0.16 and bright_fraction >= 0.16


def _plain_paper_text_restore_exclusion_allowed(
    source: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> bool:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return False
    area = max(1, roi.shape[0] * roi.shape[1])
    mask_density = float(np.count_nonzero(mask_roi > 0)) / float(max(1, mask_roi.size))
    if mask_density > 0.72:
        return False
    if not _continuous_flat_background_allowed(source, coords, mask_roi):
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    edges = cv2.Canny(gray, 45, 135) > 0
    paper_fraction = float(np.mean((gray > 172) & (saturation < 125)))
    bright_fraction = float(np.mean(gray > 218))
    low_saturation = float(np.mean(saturation < 160))
    edge_density = float(np.mean(edges))
    dark_fraction = float(np.mean(gray < 64))
    if (
        paper_fraction < 0.76
        or bright_fraction < 0.68
        or low_saturation < 0.90
        or edge_density > 0.18
        or dark_fraction > 0.20
    ):
        return False

    seed_guard = cv2.dilate(
        (mask_roi > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=1,
    ).astype(bool)
    background = (~seed_guard) & (gray > 172) & (saturation < 125)
    if np.count_nonzero(background) < max(32, int(area * 0.05)):
        return False
    return float(np.std(gray[background].astype(np.float32))) <= 34.0


def _floating_region_requires_device_overlay(
    source: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> bool:
    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return False

    height, width = roi.shape[:2]
    area = max(1, height * width)
    if area < 1600:
        return False
    mask_density = float(np.count_nonzero(mask_roi > 0)) / float(area)
    if mask_density < 0.015 or mask_density > 0.55:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 45, 135)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(18, min(width, height) // 5),
        minLineLength=max(24, int(min(width, height) * 0.42)),
        maxLineGap=max(4, int(min(width, height) * 0.08)),
    )
    if lines is None:
        return False

    horizontal_lines = 0
    vertical_lines = 0
    diagonal_lines = 0
    for line in lines[:, 0, :]:
        x_a, y_a, x_b, y_b = [int(value) for value in line]
        dx = abs(x_b - x_a)
        dy = abs(y_b - y_a)
        length = float(np.hypot(dx, dy))
        if length < max(24, min(width, height) * 0.42):
            continue
        if dy <= max(4, int(height * 0.08)):
            horizontal_lines += 1
        elif dx <= max(4, int(width * 0.08)):
            vertical_lines += 1
        else:
            diagonal_lines += 1

    if horizontal_lines < 1 or vertical_lines < 1:
        return False
    if diagonal_lines > max(2, horizontal_lines + vertical_lines):
        return False

    saturation_p80 = float(np.percentile(hsv[:, :, 1], 80))
    flat_fraction = float(np.mean((gray > 32) & (gray < 238) & (hsv[:, :, 1] < 96)))
    border_dark = np.zeros_like(gray, dtype=bool)
    band = max(3, min(9, min(width, height) // 10))
    border_dark[:band, :] = True
    border_dark[-band:, :] = True
    border_dark[:, :band] = True
    border_dark[:, -band:] = True
    border_dark_fraction = float(np.mean(gray[border_dark] < 92)) if np.any(border_dark) else 0.0
    return saturation_p80 < 80.0 and flat_fraction >= 0.42 and border_dark_fraction >= 0.08


def _floating_region_requires_source_cover(
    source: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> bool:
    """Detect dense outlined floating dialogue where automatic redraw is unsafe."""

    if os.getenv("MANGA_SOURCE_COVER_FALLBACK", "off").strip().lower() in {"0", "false", "no", "off"}:
        return False

    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return False

    height, width = roi.shape[:2]
    area = max(1, height * width)
    if area < 1600:
        return False

    mask_density = float(np.count_nonzero(mask_roi > 0)) / float(area)
    if mask_density < 0.018 or mask_density > 0.58:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    paper_fraction = float(np.mean((gray > 170) & (saturation < 130)))
    mean_luma = float(np.mean(gray))
    if paper_fraction >= 0.62 and mean_luma >= 176:
        return False

    dark_fraction = float(np.mean(gray < 88))
    mid_fraction = float(np.mean((gray >= 88) & (gray < 178)))
    bright_fraction = float(np.mean((gray > 188) & (saturation < 150)))
    luma_std = float(np.std(gray.astype(np.float32)))
    edge_density = float(np.mean(cv2.Canny(gray, 45, 135) > 0))

    if bright_fraction < 0.035:
        return False
    if dark_fraction < 0.16 or mid_fraction < 0.045:
        return False
    if luma_std < 48.0 or edge_density < 0.045:
        return False

    mostly_dark_simple = dark_fraction >= 0.76 and mid_fraction < 0.08 and edge_density < 0.10
    return not mostly_dark_simple


def _floating_full_box_cleanup_allowed(
    source: np.ndarray,
    coords: tuple[int, int, int, int],
) -> bool:
    """Allow full-rectangle cleanup only when the surrounding panel is simple."""

    x1, y1, x2, y2 = coords
    img_h, img_w = source.shape[:2]
    roi_w = max(1, x2 - x1)
    roi_h = max(1, y2 - y1)
    pad = max(14, min(52, max(roi_w, roi_h) // 6))
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(img_w, x2 + pad)
    ry2 = min(img_h, y2 + pad)
    patch = source[ry1:ry2, rx1:rx2]
    if patch.size == 0:
        return False

    ring = np.ones(patch.shape[:2], dtype=bool)
    ix1 = max(0, x1 - rx1)
    iy1 = max(0, y1 - ry1)
    ix2 = min(patch.shape[1], x2 - rx1)
    iy2 = min(patch.shape[0], y2 - ry1)
    ring[iy1:iy2, ix1:ix2] = False
    if np.count_nonzero(ring) < 80:
        return False

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 45, 135)
    ring_gray = gray[ring]
    ring_sat = hsv[:, :, 1][ring]
    ring_median = float(np.median(ring_gray))
    paper_fraction = float(np.mean((ring_gray > 218) & (ring_sat < 115)))
    gray_tone_fraction = float(np.mean((ring_gray >= 95) & (ring_gray <= 218) & (ring_sat < 145)))
    edge_density = float(np.mean(edges[ring] > 0))

    plain_paper = paper_fraction >= 0.58 and ring_median >= 218.0 and edge_density <= 0.18
    smooth_gray_tone = gray_tone_fraction >= 0.50 and 95.0 <= ring_median <= 205.0 and edge_density <= 0.10
    return plain_paper or smooth_gray_tone


def _fill_simple_floating_caption_box(
    source: np.ndarray,
    target: np.ndarray,
    coords: tuple[int, int, int, int],
) -> np.ndarray | None:
    if not _floating_full_box_cleanup_allowed(source, coords):
        return None

    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return None
    img_h, img_w = source.shape[:2]
    width = x2 - x1
    height = y2 - y1
    pad_x = max(24, min(96, int(width * 0.24)))
    pad_y = max(18, min(72, int(height * 0.70)))
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(img_w, x2 + pad_x)
    cy2 = min(img_h, y2 + pad_y)
    context = source[cy1:cy2, cx1:cx2]
    if context.size == 0:
        return None

    gray = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(context, cv2.COLOR_BGR2HSV)
    edges = cv2.dilate(
        (cv2.Canny(gray, 45, 135) > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ).astype(bool)
    background = np.ones(gray.shape, dtype=bool)
    background[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1] = False
    background &= ~edges
    background &= hsv[:, :, 1] < 180
    background &= gray >= 48
    background &= gray <= 250
    if int(np.count_nonzero(background)) < max(120, int(width * height * 0.030)):
        return None

    roi = source[y1:y2, x1:x2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_edges = cv2.dilate(
        (cv2.Canny(roi_gray, 35, 120) > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ).astype(bool)
    dark_text = (roi_gray < 132) & (roi_hsv[:, :, 1] < 230)
    dark_guard = cv2.dilate(
        dark_text.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 25)),
        iterations=1,
    ).astype(bool)
    row_baseline = np.empty(height, dtype=np.float32)
    for row in range(height):
        row_pixels = roi_gray[row, :]
        row_sat = roi_hsv[row, :, 1]
        candidates = (row_sat < 180) & (row_pixels >= 70) & (row_pixels <= 245)
        row_baseline[row] = float(np.median(row_pixels[candidates])) if np.count_nonzero(candidates) >= 8 else float(np.median(roi_gray))
    bright_floor = np.minimum(232.0, np.maximum(205.0, row_baseline[:, None] + 4.0))
    bright_halo = (
        (roi_gray.astype(np.float32) >= bright_floor)
        & (roi_hsv[:, :, 1] < 155)
        & (roi_edges | dark_guard)
    )
    repair = dark_text | bright_halo
    repair = cv2.dilate(
        repair.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 7)),
        iterations=1,
    ).astype(bool)
    if int(np.count_nonzero(repair)) < max(32, int(width * height * 0.010)):
        return None

    target_roi = target[y1:y2, x1:x2]
    repair_mask = repair.astype(np.uint8) * 255
    telea = cv2.inpaint(target_roi, repair_mask, 3.0, cv2.INPAINT_TELEA)
    navier = cv2.inpaint(target_roi, repair_mask, 2.4, cv2.INPAINT_NS)
    repaired = cv2.addWeighted(telea, 0.68, navier, 0.32, 0)
    target_roi[repair] = repaired[repair]
    return repair.astype(np.uint8) * 255


def _floating_cleanup_should_fail_closed(
    source: np.ndarray,
    coords: tuple[int, int, int, int],
    mask_roi: np.ndarray,
) -> bool:
    """Fail closed when floating-text cleanup would likely damage artwork."""

    x1, y1, x2, y2 = coords
    roi = source[y1:y2, x1:x2]
    if roi.size == 0 or mask_roi.size == 0:
        return True

    mask_bool = mask_roi > 0
    mask_count = int(np.count_nonzero(mask_bool))
    area = max(1, mask_roi.shape[0] * mask_roi.shape[1])
    if mask_count < 6:
        return True
    if mask_count / float(area) > 0.38:
        high_contrast_mask = _high_contrast_light_text_block_mask(roi)
        high_contrast_density = (
            float(np.count_nonzero(high_contrast_mask > 0)) / float(area)
            if high_contrast_mask is not None
            else 1.0
        )
        if high_contrast_mask is not None and high_contrast_density <= 0.92:
            gray_probe = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            hsv_probe = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            background = (high_contrast_mask <= 0) & (gray_probe > 150) & (hsv_probe[:, :, 1] < 170)
            if np.count_nonzero(background) >= max(4, int(area * 0.015)):
                return False
        return True

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 45, 135) > 0
    unmasked = ~mask_bool
    dilated_mask = cv2.dilate(
        mask_bool.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    ) > 0
    near_unmasked = dilated_mask & unmasked

    paper_fraction = float(np.mean((gray > 172) & (hsv[:, :, 1] < 115)))
    dark_fraction = float(np.mean(gray < 80))
    edge_density = float(np.mean(edges))
    channel_std = float(np.mean(np.std(roi.reshape(-1, 3).astype(np.float32), axis=0)))

    if np.count_nonzero(near_unmasked) >= 20:
        near_edge_density = float(np.mean(edges[near_unmasked]))
        near_dark_density = float(np.mean(gray[near_unmasked] < 130))
    else:
        near_edge_density = edge_density
        near_dark_density = float(np.mean(gray[unmasked] < 130)) if np.count_nonzero(unmasked) else 1.0

    if np.count_nonzero(unmasked) >= 20:
        unmasked_edge_density = float(np.mean(edges[unmasked]))
        unmasked_dark_density = float(np.mean(gray[unmasked] < 130))
    else:
        unmasked_edge_density = edge_density
        unmasked_dark_density = near_dark_density

    flat_white = (
        paper_fraction >= 0.86
        and near_edge_density <= 0.035
        and unmasked_dark_density <= 0.055
        and edge_density <= 0.16
    )
    flat_dark = (
        dark_fraction >= 0.72
        and near_edge_density <= 0.045
        and unmasked_edge_density <= 0.060
        and channel_std <= 32.0
    )
    flat_light_tone = (
        paper_fraction >= 0.72
        and near_edge_density <= 0.025
        and unmasked_dark_density <= 0.040
        and edge_density <= 0.075
        and channel_std <= 18.0
    )
    return not (flat_white or flat_dark or flat_light_tone)


def _lama_local_crop(lama_session, image, mask, img_h, img_w, x1, y1, x2, y2):
    box_width = x2 - x1
    box_height = y2 - y1

    if box_width <= 512 and box_height <= 512:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        crop_x1, crop_x2 = center_x - 256, center_x + 256
        crop_y1, crop_y2 = center_y - 256, center_y + 256
        if crop_x1 < 0:
            crop_x2 -= crop_x1
            crop_x1 = 0
        if crop_y1 < 0:
            crop_y2 -= crop_y1
            crop_y1 = 0
        if crop_x2 > img_w:
            crop_x1 -= crop_x2 - img_w
            crop_x2 = img_w
        if crop_y2 > img_h:
            crop_y1 -= crop_y2 - img_h
            crop_y2 = img_h
        crop_x1, crop_y1 = max(0, crop_x1), max(0, crop_y1)
        crop_x2, crop_y2 = min(img_w, crop_x2), min(img_h, crop_y2)
    else:
        crop_x1, crop_y1 = max(0, x1 - 32), max(0, y1 - 32)
        crop_x2, crop_y2 = min(img_w, x2 + 32), min(img_h, y2 + 32)

    crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    if not np.any(crop_mask > 0):
        return

    crop_height, crop_width = crop_img.shape[:2]
    pad_bottom = max(0, 512 - crop_height)
    pad_right = max(0, 512 - crop_width)
    if pad_bottom > 0 or pad_right > 0:
        crop512 = cv2.copyMakeBorder(
            crop_img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT
        )
        mask512 = cv2.copyMakeBorder(
            crop_mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0
        )
    else:
        crop512, mask512 = crop_img, crop_mask

    inpainted512 = lama_inpaint(lama_session, crop512, mask512)
    inpainted_crop = inpainted512[:crop_height, :crop_width]

    view = image[crop_y1:crop_y2, crop_x1:crop_x2]
    view[crop_mask > 127] = inpainted_crop[crop_mask > 127]


def _sweep_residual_dark_clusters(
    source: np.ndarray,
    final_mask: np.ndarray,
    preserved_sfx_mask: np.ndarray | None = None,
    text_evidence_mask: np.ndarray | None = None,
    min_cluster_area: int = 50,
    max_cluster_area: int = 50000,
    context_ring_pad: int = 8,
    max_ring_std: float = 35.0,
    min_text_evidence_overlap: float = 0.35,
) -> tuple[np.ndarray, dict]:
    """Find unconstrained dark clusters in the source image and return them as a cleanup mask.

    These clusters are typically source-text glyphs that the constraint loop dropped
    (Step 6 classified them as ``sfx`` or ``noise``) or that the per-constraint
    inpainter failed to mask.  Without this sweep, they survive to the final output
    as visible dark residue on bright backgrounds — the strip-5 smoking gun on
    ``new_sample_13_(chi)`` was this exact failure mode (9.29 % dark pixels in a
    strip where the Ichigo example has 0.00 %).

    The sweep is deliberately conservative:

    * Only clusters with area in ``[min_cluster_area, max_cluster_area]`` are kept.
    * Clusters already covered by ``final_mask`` are skipped (no double work).
    * Clusters overlapping ``preserved_sfx_mask`` are skipped (real SFX is sacred).
    * Clusters whose immediate context ring has channel ``std > max_ring_std`` are
      skipped — high-variance rings mean the surrounding art is too busy to safely
      extrapolate a fill colour from.
    * Clusters must overlap ``text_evidence_mask`` (the Step-1 text-detector
      seg mask) by at least ``min_text_evidence_overlap`` of their own area.
      Dark ink that the text detector never flagged is ARTWORK (facial lines,
      tatami edges, speed lines) — sweeping it hands the artist's drawing to
      the inpainter, which re-hallucinates it (glasses vanished from a face on
      ``new_sample_13_(chi)``). Text residue, by definition, is something the
      detector saw and the constraint loop dropped.

    Args:
        source: BGR source image ``(H, W, 3)``.
        final_mask: Existing cleanup mask ``(H, W)`` uint8 — clusters overlapping
            this mask are skipped.
        preserved_sfx_mask: Optional mask ``(H, W)`` uint8 of regions to preserve
            (genuine SFX glyphs that should not be cleaned).
        min_cluster_area: Minimum connected-component area to consider.
        max_cluster_area: Maximum area — guards against swallowing huge dark panels.
        context_ring_pad: Padding around each cluster used for variance sampling.
        max_ring_std: Maximum mean per-channel std in the context ring.

    Returns:
        ``(mask, stats)`` where ``mask`` is a uint8 mask of dark clusters to clean
        and ``stats`` is a diagnostic counter dict.
    """
    stats = {
        "clusters_found": 0,
        "clusters_cleaned": 0,
        "clusters_skipped_already_masked": 0,
        "clusters_skipped_sfx": 0,
        "clusters_skipped_too_small": 0,
        "clusters_skipped_too_large": 0,
        "clusters_skipped_high_variance": 0,
        "clusters_skipped_no_text_evidence": 0,
    }

    H, W = source.shape[:2]
    out_mask = np.zeros((H, W), dtype=np.uint8)

    if source.size == 0 or final_mask.shape[:2] != (H, W):
        return out_mask, stats

    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    # 1. Find dark pixels (gray < 80) — typical manga glyph stroke intensity.
    dark_pixels = (gray < 80).astype(np.uint8) * 255

    # 2. Subtract existing cleanup mask (no double work).
    dark_pixels = cv2.bitwise_and(dark_pixels, cv2.bitwise_not(final_mask))

    # 3. Subtract preserved SFX mask (real SFX stays put).
    if preserved_sfx_mask is not None and preserved_sfx_mask.shape == (H, W):
        dark_pixels = cv2.bitwise_and(dark_pixels, cv2.bitwise_not(preserved_sfx_mask))

    # 4. Morphological open removes 1–2 px salt noise.
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_pixels = cv2.morphologyEx(dark_pixels, cv2.MORPH_OPEN, kernel_3, iterations=1)

    # Dilate the detector evidence slightly: seg masks hug glyph cores, while
    # the dark-pixel clusters here include anti-aliased stroke fringes.
    evidence = None
    if text_evidence_mask is not None and text_evidence_mask.shape[:2] == (H, W):
        evidence = cv2.dilate(
            (text_evidence_mask > 0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )

    # 5. Connected components.
    labels_count, labels, stats_arr, _ = cv2.connectedComponentsWithStats(
        dark_pixels, connectivity=8
    )
    if labels_count <= 1:
        return out_mask, stats

    stats["clusters_found"] = int(labels_count - 1)

    # 6. Filter and validate each cluster.
    for label_idx in range(1, labels_count):
        area = int(stats_arr[label_idx, cv2.CC_STAT_AREA])
        x = int(stats_arr[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats_arr[label_idx, cv2.CC_STAT_TOP])
        w = int(stats_arr[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats_arr[label_idx, cv2.CC_STAT_HEIGHT])

        if area < min_cluster_area:
            stats["clusters_skipped_too_small"] += 1
            continue
        if area > max_cluster_area:
            stats["clusters_skipped_too_large"] += 1
            continue

        strong_evidence = False
        if evidence is not None:
            cluster_bin = labels[y:y + h, x:x + w] == label_idx
            overlap = int(np.count_nonzero(cluster_bin & (evidence[y:y + h, x:x + w] > 0)))
            if overlap < area * min_text_evidence_overlap:
                stats["clusters_skipped_no_text_evidence"] += 1
                continue
            strong_evidence = overlap >= area * 0.6

        ring_x1 = max(0, x - context_ring_pad)
        ring_y1 = max(0, y - context_ring_pad)
        ring_x2 = min(W, x + w + context_ring_pad)
        ring_y2 = min(H, y + h + context_ring_pad)

        local_labels = labels[ring_y1:ring_y2, ring_x1:ring_x2]
        local_source = source[ring_y1:ring_y2, ring_x1:ring_x2]
        cluster_in_ring = local_labels == label_idx

        ring_pixels = local_source[~cluster_in_ring]
        if ring_pixels.size == 0:
            stats["clusters_skipped_high_variance"] += 1
            continue

        ring_std = float(
            np.mean(np.std(ring_pixels.reshape(-1, 3).astype(np.float32), axis=0))
        )
        if ring_std > max_ring_std and not strong_evidence:
            # A busy ring makes tone extrapolation unsafe — but when the text
            # detector itself confirms the cluster (strong seg overlap), it is
            # residue that MUST go; the model fallback handles mixed rings.
            stats["clusters_skipped_high_variance"] += 1
            continue

        cluster_mask = (labels == label_idx).astype(np.uint8) * 255
        out_mask = cv2.bitwise_or(out_mask, cluster_mask)
        stats["clusters_cleaned"] += 1

    return out_mask, stats


def _imread_grayscale_2d(path: Path) -> np.ndarray | None:
    # cv2.IMREAD_GRAYSCALE is documented to always return a 2D (H, W) array,
    # but under the torch/cv2 native-DLL load-order conflict already guarded
    # against at this file's own imports (see the "torch must be imported
    # before cv2" comment near the top), it can still decode a mask PNG with
    # a spurious trailing (H, W, 1) channel dimension. Verified: reproducible
    # only through the real chained backend request path (translate_image ->
    # ... -> run_step4_inpaint, all in one process after step5/6/7 already
    # ran), never in an isolated single-stage process against the same file
    # -- collapse it back to 2D unconditionally so every downstream consumer
    # (np.nonzero unpacking, direct-shape comparisons, bitwise ops against
    # other 2D masks) stays safe regardless of which environment quirk
    # triggered it this run.
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is not None and mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def run_step4_inpaint(sample_map: dict[str, str] | None = None, samples_dir: Path | None = None):
    print("=" * 60)
    print("  Step 4 — Layout-Driven Inpainting (v15: Art-Safe Stroke Repair)")
    print("=" * 60)

    cfg = MLConfig()
    samples_dir = Path(samples_dir) if samples_dir is not None else sample_root_from_env(DEFAULT_SAMPLES_ROOT)
    sample_map = sample_map or SAMPLE_MAP
    lama_session = _get_lama_session(cfg.lama_model_path)
    anime_model, anime_device = _get_anime_lama_model()
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for sample_name, img_file in sample_map.items():
        sample_path = samples_dir / sample_name
        img_path = sample_path / img_file
        layout_path = sample_path / "step_6_layout" / "layout_constraints.json"
        detect_dir = sample_path / "step_1_detect"
        seg_mask_path = detect_dir / "seg_mask.png"

        if not (img_path.exists() and layout_path.exists() and seg_mask_path.exists()):
            print(f"  SKIP {sample_name}: Missing layout/detect data")
            continue

        print(f"\nProcessing {sample_name}")
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(
                f"cv2.imread() could not read {img_path} -- the file is missing, truncated, or "
                "not a valid image. This surfaces here as a clear error instead of the cryptic "
                "'NoneType' object has no attribute 'shape'."
            )
        img_h, img_w = image.shape[:2]
        seg_mask = _imread_grayscale_2d(seg_mask_path)
        if seg_mask is None:
            # seg_mask_path.exists() was already confirmed above, but exists()
            # doesn't guarantee the file is decodable -- a truncated/corrupt
            # seg_mask.png yields None here. Every downstream use of seg_mask
            # in this function assumes a valid array (subscripting it directly),
            # so treat this exactly like the "file missing" case above instead
            # of letting a later `seg_mask[y1:y2, x1:x2]` crash the whole page
            # with an uncaught 'NoneType' object is not subscriptable.
            print(f"  SKIP {sample_name}: seg_mask.png exists but failed to decode (corrupt/truncated)")
            continue

        with open(layout_path, "r", encoding="utf-8") as layout_file:
            layout_data = json.load(layout_file)
        renderable_translation_ids = _load_renderable_translation_ids(sample_path)
        translation_text_map = _load_translation_text_map(sample_path)

        # Page-level union of every traced bubble/container outline Step 6
        # found (real detector bubbles and inferred containers alike). Used
        # by the page-level self-checks below to keep their context sampling
        # and any "re-reconstruct" grow inside the SAME container a cleaned
        # region came from, instead of crossing its wall into unrelated art.
        container_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        _bubble_disk_mask_cache: dict[int, np.ndarray | None] = {}
        for _constraint_for_container in layout_data:
            _outline = (
                _constraint_for_container.get("refined_bubble_outline")
                or _constraint_for_container.get("inferred_bubble_outline")
                or []
            )
            if len(_outline) >= 3:
                cv2.fillPoly(
                    container_mask, [np.array(_outline, dtype=np.int32)], 255
                )
                continue
            # Neither re-trace succeeded, but a real detected bubble (unusual
            # shapes -- e.g. compound/overlapping caption boxes -- can fail
            # _container_flood_outline's plainness/boundary validation even
            # though the original detector mask is fine) still has its own
            # segmentation mask on disk. Without this fallback the region is
            # invisible to container_mask and the page-level self-checks below
            # treat a real bubble/box interior as floating text on bare art,
            # blending it toward surrounding tone instead of preserving its
            # own solid fill.
            _bubble_idx = int(_constraint_for_container.get("bubble_idx", -1))
            if _bubble_idx < 0:
                continue
            if _bubble_idx not in _bubble_disk_mask_cache:
                _disk_mask_path = detect_dir / f"bubble_{_bubble_idx}.png"
                _bubble_disk_mask_cache[_bubble_idx] = (
                    _imread_grayscale_2d(_disk_mask_path)
                    if _disk_mask_path.exists()
                    else None
                )
            _disk_mask = _bubble_disk_mask_cache[_bubble_idx]
            if _disk_mask is not None and _disk_mask.shape == container_mask.shape:
                container_mask = cv2.bitwise_or(container_mask, _disk_mask)

        result = image.copy()
        final_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        floating_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        skipped_without_translation = 0
        inferred_bubble_cleanups = 0
        cleanup_status = {}
        sfx_preserved_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        # Step 6's anatomy classifier rejects brush/hollow SFX lettering as
        # artwork before it ever becomes a constraint; its region registry
        # shields those pixels from the residual sweep here.
        sfx_regions_path = sample_path / "step_6_layout" / "sfx_artwork_regions.json"
        if sfx_regions_path.exists():
            try:
                with open(sfx_regions_path, "r", encoding="utf-8") as sfx_file:
                    for sfx_box in json.load(sfx_file):
                        if len(sfx_box) >= 4:
                            sx1 = max(0, min(img_w, int(sfx_box[0])))
                            sy1 = max(0, min(img_h, int(sfx_box[1])))
                            sx2 = max(0, min(img_w, int(sfx_box[2])))
                            sy2 = max(0, min(img_h, int(sfx_box[3])))
                            if sx2 > sx1 and sy2 > sy1:
                                sfx_preserved_mask[sy1:sy2, sx1:sx2] = 255
            except (json.JSONDecodeError, OSError, ValueError, TypeError) as _sfx_load_error:
                # A failed load here silently drops onomatopoeia protection --
                # the inpainter can then erase the artist's own SFX lettering
                # thinking it's translatable text. Log loudly; this is
                # deliberately log-only (no behavior change) to avoid pixel
                # risk in the inpainting path itself.
                print(
                    f"  [step4-warn] failed to load sfx_artwork_regions.json ({sfx_regions_path}): "
                    f"{_sfx_load_error!r} -- SFX/onomatopoeia protection is DISABLED for this page"
                )
        page_dialogue_hue = _page_dialogue_lettering_hue(
            image,
            seg_mask,
            layout_data,
            translation_text_map,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )

        for constraint in layout_data:
            if (
                renderable_translation_ids is not None
                and int(constraint.get("id", -1)) not in renderable_translation_ids
            ):
                skipped_without_translation += 1
                continue

            constraint_id = str(int(constraint.get("id", -1)))
            red_box = [int(value) for value in constraint["red_box"]]
            source_red_coords = tuple(red_box)
            force_bubble_cleanup = bool(constraint.get("force_bubble_cleanup", False))
            is_bubble = constraint.get("bubble_idx", -1) != -1 or force_bubble_cleanup
            precise_layout_mask = constraint.get("mask_mode") == "svg_text"
            # Hard bound for constraints Step 6 kept ONLY on the semantic detector's
            # word (its floating_too_small rescue). They have bubble_idx == -1 and no
            # traced outline, so they contribute nothing to container_mask below and
            # the floating paths would otherwise grow unbounded into the surrounding
            # art -- see the semantic_rescue_region comment in run_step6_layout.py.
            # Unioned with red_box so a later Step-6 red_box change can never leave
            # real glyph ink outside the bound.
            semantic_clip_mask = None
            _rescue_region = constraint.get("semantic_rescue_region")
            if _rescue_region and len(_rescue_region) >= 4:
                _sr = [int(v) for v in _rescue_region[:4]]
                _cx1 = max(0, min(img_w, min(_sr[0], red_box[0])))
                _cy1 = max(0, min(img_h, min(_sr[1], red_box[1])))
                _cx2 = max(0, min(img_w, max(_sr[2], red_box[2])))
                _cy2 = max(0, min(img_h, max(_sr[3], red_box[3])))
                if _cx2 > _cx1 and _cy2 > _cy1:
                    semantic_clip_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    semantic_clip_mask[_cy1:_cy2, _cx1:_cx2] = 255
            # red_box is display-tight (glyph bbox +4px, Step 6 refinement);
            # cleanup must work from a padded window so halo and edge glyph
            # pixels just outside the tight box are still captured. Without
            # this, tightening red_box silently shrinks erase coverage and
            # leaves source-text residue (regression seen on new_sample_6).
            cleanup_pad = 10
            cleanup_box = [
                max(0, int(red_box[0]) - cleanup_pad),
                max(0, int(red_box[1]) - cleanup_pad),
                min(img_w, int(red_box[2]) + cleanup_pad),
                min(img_h, int(red_box[3]) + cleanup_pad),
            ]
            x1, y1, x2, y2 = cleanup_box

            # SFX are part of the artwork: preserve them untouched (no erase,
            # no reconstruction) and tell Step 8 not to typeset a translation.
            if not is_bubble and _floating_sfx_signature(
                image,
                seg_mask,
                (x1, y1, x2, y2),
                translation_text_map.get(int(constraint.get("id", -1)), ""),
                kernel_3,
                dialogue_hue=page_dialogue_hue,
            ):
                cleanup_status[constraint_id] = {
                    "cleaned": False,
                    "mode": "floating",
                    "reason": "sfx_preserved_artwork",
                }
                sfx_pad = 6
                sfx_preserved_mask[
                    max(0, y1 - sfx_pad):min(img_h, y2 + sfx_pad),
                    max(0, x1 - sfx_pad):min(img_w, x2 + sfx_pad),
                ] = 255
                continue

            roi_seg = seg_mask[y1:y2, x1:x2]
            _, roi_seg_bin = cv2.threshold(roi_seg, 127, 255, cv2.THRESH_BINARY)

            # seg_mask can under-segment a bubble's own text (miss most of the
            # glyphs) while still clearing the `< 20` floor below -- 770px of
            # detected stroke looks non-trivial until you compare it to the box's
            # actual ink. Measured (34-sample sweep, 2026-07-27): for bubble
            # constraints, seg_coverage / an independent Otsu dark-pixel estimate
            # of the same box is bimodal with an >8x gap -- genuine detections
            # cluster at ratio>=1.1, under-segmented ones sit below 0.15. A box
            # past that gap gets the same independent-detector fallback the `<20`
            # branch already uses for a fully-empty mask. Scoped to is_bubble only
            # -- that's what was measured; floating text can legitimately have a
            # dark background where this dark-pixel estimate isn't meaningful.
            # Without this, region_mask/dilated stays sparse and
            # _clean_white_bubble_residue can only flatten a thin halo around the
            # fragments seg_mask did find, leaving the rest of the source text
            # completely uncleaned (confirmed: new_sample_10_(chi) id=3,
            # ratio=0.134, left 5237px of legible source text once F-1a stopped
            # compensating with a solid rectangle).
            seg_looks_unreliable = False
            if is_bubble:
                roi_gray_for_coverage = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                dark_estimate = int(np.count_nonzero(roi_gray_for_coverage < 150))
                seg_looks_unreliable = (
                    200 <= dark_estimate < roi_gray_for_coverage.size * 0.5
                    and np.count_nonzero(roi_seg_bin) < dark_estimate * 0.5
                )
                # The upper bound above excludes a genuinely dark-interior bubble
                # (light text on a black background): there, dark_estimate is
                # most of the box by construction, the ratio would look just as
                # "sparse", and _extract_text_strokes(is_bubble=True) assumes the
                # opposite polarity (Otsu inverted for dark-on-light) -- it would
                # return the background, not the text. No such bubble exists in
                # the 34-sample sweep this threshold was measured on, so this
                # guard is structural, not evidenced by a counterexample.

            if force_bubble_cleanup:
                roi_mask = _extract_dark_text_strokes(
                    image,
                    (x1, y1, x2, y2),
                    constraint.get("source_colors") or None,
                )
            elif np.count_nonzero(roi_seg_bin) < 20 or seg_looks_unreliable:
                roi_mask = _extract_text_strokes(image, (x1, y1, x2, y2), is_bubble=is_bubble)
            else:
                roi_mask = roi_seg_bin

            dilated = cv2.dilate(roi_mask, kernel_3, iterations=2)

            if is_bubble:
                bubble_mask_path = detect_dir / f"bubble_{constraint['bubble_idx']}.png"
                bubble_mask = None
                refined_outline = constraint.get("refined_bubble_outline") or []
                if len(refined_outline) >= 3:
                    # Step 6 re-traced the actual container boundary; it is
                    # more faithful than the detector mask (which can hug the
                    # whitened glyph areas instead of the bubble/box walls).
                    bubble_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    cv2.fillPoly(
                        bubble_mask,
                        [np.array(refined_outline, dtype=np.int32)],
                        255,
                    )
                elif bubble_mask_path.exists():
                    bubble_mask = _imread_grayscale_2d(bubble_mask_path)
                elif not precise_layout_mask:
                    # No detected bubble segmentation for this region (e.g.
                    # force_bubble_cleanup, or the mask was filtered out
                    # upstream). Synthesize an inscribed elliptical boundary
                    # from the red_box instead of leaving the fill unconstrained
                    # — an unconstrained rectangle lets LaMa/residue-cleanup
                    # bleed past the actual bubble outline into surrounding art.
                    bubble_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    ellipse_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    ellipse_axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
                    cv2.ellipse(bubble_mask, ellipse_center, ellipse_axes, 0, 0, 360, 255, -1)

                if bubble_mask is not None:
                    roi_bubble_mask = bubble_mask[y1:y2, x1:x2]
                    # 3 iterations (was 5) — 5 iterations of a 3x3 kernel shrinks
                    # the safe cleanup zone by ~5px per side, which was leaving
                    # ghost stroke residue on tightly-curved bubble boundaries.
                    eroded_bubble = cv2.erode(roi_bubble_mask, kernel_3, iterations=3)
                    dilated = cv2.bitwise_and(dilated, eroded_bubble)

                region_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                region_mask[y1:y2, x1:x2] = dilated

                # Tinted/translucent interiors (framed caption boxes, colored
                # bubbles): the glyphs carry white halos that the stroke mask
                # misses, and white-residue cleanup would bleach streaks into
                # the tint. Grow the mask through the halo pixels and let the
                # anime-manga LaMa rebuild the interior instead.
                tinted_interior = False
                if bubble_mask is not None and np.any(roi_bubble_mask > 0):
                    # Detector bubble masks hug the whitest areas; the tint of
                    # a translucent box interior lies between/around them, so
                    # sample the mask's whole bounding box (minus glyphs).
                    bys, bxs = np.nonzero(roi_bubble_mask)
                    interior_np = np.zeros_like(roi_bubble_mask, dtype=bool)
                    interior_np[bys.min():bys.max() + 1, bxs.min():bxs.max() + 1] = True
                    interior_np &= dilated == 0
                    if int(np.count_nonzero(interior_np)) >= 60:
                        roi_bgr = image[y1:y2, x1:x2]
                        gray_int = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                        hsv_int = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                        # A round/curved bubble's OWN border wall is tangent to
                        # its rectangular bbox at the extremes, so the bbox-based
                        # interior_np above samples border ink at its corners --
                        # that ink is foreground art, not background tint, and
                        # dragging it into the percentile below makes plain white
                        # bubbles misread as "tinted" (confirmed: external_ja_1
                        # id=1 measured p30=239 <=244 from border contamination
                        # alone, routing a plain bubble into the anime-model
                        # tinted-fill path instead of a flat white fill). Exclude
                        # clearly-dark pixels (border/art ink, not tint) from the
                        # sample; a genuine translucent tint is midtone, not
                        # near-black, so this doesn't defeat real tint detection.
                        tint_sample = interior_np & (gray_int >= 80)
                        if int(np.count_nonzero(tint_sample)) >= 60:
                            # Percentiles, not medians: a tinted interior mixed
                            # with bright glyph halos can push the median above a
                            # pure-white bubble's, hiding the tint. A meaningful
                            # tinted share shows up at the 30th luma / 70th sat
                            # percentile regardless of how much halo is present.
                            tinted_interior = (
                                float(np.percentile(gray_int[tint_sample], 30)) <= 244.0
                                or float(np.percentile(hsv_int[:, :, 1][tint_sample], 70)) >= 12.0
                            )
                        if os.getenv("MANGA_DEBUG_TINT_FLIP") == "1":
                            # Regression-tracking instrument, not live logic: old_verdict
                            # recomputes the HISTORICAL (pre-border-ink-exclusion) formula
                            # on unfiltered interior_np purely so a future change here can be
                            # diffed against what shipped. Compare against the pre-fix predicate
                            # to find flips the fix caused. Two distinct
                            # mechanisms can flip the verdict -- starvation (n_tint_sample
                            # < 60 leaves tinted_interior at its False initializer, never
                            # evaluated at all -- the unintended one) versus a valid
                            # filtered sample simply reading differently (the intended
                            # effect of excluding border ink). Log both so they aren't
                            # conflated in the count.
                            old_verdict = (
                                float(np.percentile(gray_int[interior_np], 30)) <= 244.0
                                or float(np.percentile(hsv_int[:, :, 1][interior_np], 70)) >= 12.0
                            )
                            n_tint = int(np.count_nonzero(tint_sample))
                            print(
                                f"    [tint-flip-debug] id={constraint_id} "
                                f"n_interior={int(np.count_nonzero(interior_np))} n_tint_sample={n_tint} "
                                f"starved={n_tint < 60} old={old_verdict} new={tinted_interior}",
                                flush=True,
                            )
                if tinted_interior and anime_model is not None:
                    roi_bgr = image[y1:y2, x1:x2]
                    gray_int = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                    hsv_int = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                    white_halo = (gray_int >= 196) & (hsv_int[:, :, 1] <= 50)
                    grown = dilated.copy()
                    for _ in range(12):
                        grown = cv2.dilate(grown, kernel_3)
                        grown[~white_halo] = 0
                    grown = np.maximum(dilated, grown)
                    grown = cv2.bitwise_and(grown, roi_bubble_mask)
                    tinted_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    tinted_mask[y1:y2, x1:x2] = grown
                    _anime_lama_local_crop(
                        anime_model, anime_device, result, tinted_mask,
                        img_h, img_w, x1, y1, x2, y2,
                    )
                    grown = _reinpaint_ghost_residue(
                        anime_model, anime_device, result, (x1, y1, x2, y2), grown,
                        container_mask=bubble_mask,
                    )
                    tinted_mask[y1:y2, x1:x2] = grown
                    final_mask = cv2.bitwise_or(final_mask, tinted_mask)
                    cleanup_status[constraint_id] = {
                        "cleaned": True,
                        "mode": "bubble",
                        "reason": "tinted_interior_model_repair",
                    }
                    continue

                # F-2: skip the ML inpainter entirely when the pristine interior
                # (excluding glyph strokes) is provably flat -- prevents invention
                # at the source instead of cleaning it up after the fact. Strict,
                # AND'd gate (not tinted_interior's loose OR) since committing to
                # skip the model is the bigger decision; must not fire on
                # screentoned/textured interiors. p30 (not median) so a handful of
                # halo/antialiasing pixels near strokes can't hide real texture.
                interior_uniform = False
                if (
                    os.getenv("MANGA_FLAT_FILL_UNIFORM", "on").strip().lower() in {"1", "true", "yes", "on"}
                    and not tinted_interior
                    and bubble_mask is not None
                    and np.any(roi_bubble_mask > 0)
                ):
                    # NOT interior_np (raw bbox of roi_bubble_mask minus strokes) --
                    # a round/curved bubble's own border wall is tangent to its own
                    # bbox at the extremes, so interior_np's corners sample border
                    # ink and make every real bubble read as high-variance. Use the
                    # already-eroded interior instead (eroded_bubble, computed
                    # above for the stroke-clip), which stays inside the wall.
                    flat_fill_interior = (eroded_bubble > 0) & (dilated == 0)
                    if int(np.count_nonzero(flat_fill_interior)) >= 60:
                        roi_bgr_src = image[y1:y2, x1:x2]
                        gray_src = cv2.cvtColor(roi_bgr_src, cv2.COLOR_BGR2GRAY)
                        interior_vals = gray_src[flat_fill_interior]
                        interior_uniform = (
                            float(np.percentile(interior_vals, 30)) >= 225.0
                            and float(np.std(interior_vals)) <= 12.0
                        )
                if interior_uniform:
                    box_fill_mask = _fill_bubble_text_box_with_local_background(
                        image, result, (x1, y1, x2, y2), bubble_mask, dilated,
                    )
                    # _fill_bubble_text_box_with_local_background can itself
                    # return an all-zero mask (its own background_area gate
                    # starves on small bubbles -- the same box-starvation shape
                    # as F-1a's bug, in a different function). Only take this
                    # path if the fill actually painted something; otherwise
                    # fall through to the normal LaMa path below rather than
                    # abandoning the region with zero cleanup (confirmed: an
                    # unconditional `continue` here left external_ja_1 id=1's
                    # source Japanese text completely untranslated/uncleaned).
                    if np.count_nonzero(box_fill_mask) > 0:
                        region_mask = cv2.bitwise_or(region_mask, box_fill_mask)
                        # Verify the fill actually removed the source glyphs
                        # before marking this region clean -- same check the
                        # ordinary LaMa path below runs (:11279-11291). An
                        # unconditional cleaned=True here would silently
                        # bypass step 8's overlay/transparency fallback
                        # (run_step8_typeset.py:3183-3196), which was
                        # deliberately widened to bubble mode specifically
                        # because step 4 stopped hardcoding this flag.
                        residual_ink_ratio = 0.0
                        stroke_roi = dilated > 0
                        if np.count_nonzero(stroke_roi) >= 10:
                            roi_after_gray = cv2.cvtColor(result[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                            residual_ink_ratio = float(np.mean(roi_after_gray[stroke_roi] < 100))
                        if residual_ink_ratio > 0.20:
                            cleanup_status[constraint_id] = {
                                "cleaned": False,
                                "mode": "bubble",
                                "reason": "unsafe_art_preserved",
                            }
                        else:
                            cleanup_status[constraint_id] = {
                                "cleaned": True,
                                "mode": "bubble",
                                "reason": "flat_fill_uniform_interior",
                            }
                        final_mask = cv2.bitwise_or(final_mask, region_mask)
                        continue

                _lama_local_crop(lama_session, result, region_mask, img_h, img_w, x1, y1, x2, y2)
                # F-1a: residue cleanup must stay stroke-local (region_mask = dilated
                # glyph-stroke mask intersected with the container). A solid
                # red_box-derived rectangle used to be OR'd in here; it either
                # swallowed the whole eroded container (starving the paper-color
                # sampler in _clean_white_bubble_residue so cleanup silently
                # no-op'd, letting invented ML texture survive -- external_ja_1,
                # original/sample2) or, when it did run, flattened every pixel in
                # the box to one median color including bubble-wall/spike ink
                # crossing it (external_ja_2 id=4's burst wall). Measured: removing
                # it drops wall-damage to 0 on every tested sample while the
                # existing stroke-local region_mask still handles ordinary residue.
                cleanup_input = region_mask
                cleanup_mask = _clean_white_bubble_residue(result, cleanup_input, bubble_mask, source=image, debug_label=constraint_id)
                region_mask = cv2.bitwise_or(region_mask, cleanup_mask)
                if precise_layout_mask:
                    fill_coords = (x1, y1, x2, y2)
                    fill_text_mask = dilated
                    if constraint.get("full_box_cleanup"):
                        gx1, gy1, gx2, gy2 = [int(value) for value in constraint.get("green_box", [x1, y1, x2, y2])]
                        fill_coords = (
                            max(0, min(img_w, gx1)),
                            max(0, min(img_h, gy1)),
                            max(0, min(img_w, gx2)),
                            max(0, min(img_h, gy2)),
                        )
                        fill_text_mask = None
                    box_fill_mask = _fill_bubble_text_box_with_local_background(
                        image,
                        result,
                        fill_coords,
                        bubble_mask,
                        fill_text_mask,
                    )
                    region_mask = cv2.bitwise_or(region_mask, box_fill_mask)
                final_mask = cv2.bitwise_or(final_mask, region_mask)
                # Verify the fill actually removed the source glyphs before
                # marking this region clean — previously this was hardcoded
                # to True regardless of outcome, so Step 8 had no signal to
                # gate on for bubble text (unlike floating text, which already
                # checks `cleaned`). Sample the exact stroke locations for
                # remaining dark ink; a high residual ratio means the fill
                # failed to erase the source text.
                residual_ink_ratio = 0.0
                stroke_roi = dilated > 0
                if np.count_nonzero(stroke_roi) >= 10:
                    roi_after_gray = cv2.cvtColor(result[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    residual_ink_ratio = float(np.mean(roi_after_gray[stroke_roi] < 100))
                if residual_ink_ratio > 0.20:
                    cleanup_status[constraint_id] = {
                        "cleaned": False,
                        "mode": "bubble",
                        "reason": "unsafe_art_preserved",
                    }
                else:
                    cleanup_status[constraint_id] = {"cleaned": True, "mode": "bubble"}
                continue

            use_layout_stroke_mask = (
                constraint.get("mask_mode") == "svg_text"
                and np.count_nonzero(dilated) >= 20
            )
            fallback_source = str(constraint.get("fallback_source", "") or "").lower()
            allow_inferred_bubble_cleanup = any(
                token in fallback_source
                for token in ("unsegmented_bubble", "missed_bubble")
            )
            if allow_inferred_bubble_cleanup:
                # Real reconstruction first: local-background stroke fills leave
                # glyph-shaped residue on textured art (screentone), while the
                # anime-manga LaMa rebuilds the actual background. The fills
                # below remain as fallback when the model declines the region.
                if anime_model is not None:
                    # Span the erase-box union: donated fragments (rejected
                    # OCR debris inside the same container) can lie outside
                    # the padded red box and must be erased with it.
                    union_boxes = [(x1, y1, x2, y2)]
                    for eb in constraint.get("erase_boxes") or []:
                        if isinstance(eb, dict):
                            vals = [eb.get("x1"), eb.get("y1"), eb.get("x2"), eb.get("y2")]
                        else:
                            vals = list(eb[:4]) if len(eb) >= 4 else []
                        if len(vals) == 4 and all(v is not None for v in vals):
                            union_boxes.append(tuple(int(v) for v in vals))
                    ux1 = max(0, min(b[0] for b in union_boxes) - 3)
                    uy1 = max(0, min(b[1] for b in union_boxes) - 3)
                    ux2 = min(img_w, max(b[2] for b in union_boxes) + 3)
                    uy2 = min(img_h, max(b[3] for b in union_boxes) + 3)
                    inferred_allowed_mask = np.zeros(
                        (uy2 - uy1, ux2 - ux1), dtype=np.uint8
                    )
                    for bx1, by1, bx2, by2 in union_boxes:
                        ax1 = max(0, bx1 - 8 - ux1)
                        ay1 = max(0, by1 - 8 - uy1)
                        ax2 = min(ux2 - ux1, bx2 + 8 - ux1)
                        ay2 = min(uy2 - uy1, by2 + 8 - uy1)
                        if ax2 > ax1 and ay2 > ay1:
                            inferred_allowed_mask[ay1:ay2, ax1:ax2] = 255
                    precise_first = _precise_floating_text_local_cleanup(
                        image,
                        result,
                        (ux1, uy1, ux2, uy2),
                        anime_model,
                        anime_device,
                        allowed_mask=inferred_allowed_mask,
                        seg_mask=seg_mask,
                        container_mask=container_mask,
                        clip_mask=semantic_clip_mask,
                    )
                    if precise_first is not None:
                        precise_full = np.zeros((img_h, img_w), dtype=np.uint8)
                        precise_full[uy1:uy2, ux1:ux2] = precise_first
                        final_mask = cv2.bitwise_or(final_mask, precise_full)
                        floating_mask = cv2.bitwise_or(floating_mask, precise_full)
                        cleanup_status[constraint_id] = {
                            "cleaned": True,
                            "mode": "floating",
                            "reason": "precise_floating_text_local_cleanup",
                        }
                        continue
                inferred_outline = constraint.get("inferred_bubble_outline") or []
                inferred_bubble_mask = _inferred_polygon_mask((img_h, img_w), inferred_outline)
                inferred_box_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                inferred_erase_boxes = constraint.get("erase_boxes") or []
                for raw_box in inferred_erase_boxes:
                    if isinstance(raw_box, dict):
                        box_values = [
                            raw_box.get("x1", 0),
                            raw_box.get("y1", 0),
                            raw_box.get("x2", 0),
                            raw_box.get("y2", 0),
                        ]
                    else:
                        box_values = raw_box
                    if len(box_values) < 4:
                        continue
                    ex1, ey1, ex2, ey2 = [int(value) for value in box_values[:4]]
                    ex1 = max(0, min(img_w, ex1 - 3))
                    ey1 = max(0, min(img_h, ey1 - 3))
                    ex2 = max(0, min(img_w, ex2 + 3))
                    ey2 = max(0, min(img_h, ey2 + 3))
                    if ex2 <= ex1 or ey2 <= ey1:
                        continue
                    inferred_box_mask = cv2.bitwise_or(
                        inferred_box_mask,
                        _fill_bubble_text_box_with_local_background(
                            image,
                            result,
                            (ex1, ey1, ex2, ey2),
                            inferred_bubble_mask,
                            None,
                        ),
                    )
                if np.count_nonzero(inferred_box_mask > 0) >= 20:
                    inferred_bubble_cleanups += 1
                    final_mask = cv2.bitwise_or(final_mask, inferred_box_mask)
                    cleanup_status[constraint_id] = {"cleaned": True, "mode": "inferred_bubble_text_boxes"}
                    continue
                inferred_polygon_mask = _fill_inferred_bubble_polygon_text_strokes(
                    image,
                    result,
                    (x1, y1, x2, y2),
                    inferred_outline,
                    seg_mask,
                )
                if np.count_nonzero(inferred_polygon_mask > 0) >= 20:
                    inferred_bubble_cleanups += 1
                    final_mask = cv2.bitwise_or(final_mask, inferred_polygon_mask)
                    cleanup_status[constraint_id] = {"cleaned": True, "mode": "inferred_bubble_polygon"}
                    continue
                pale_box_mask = _fill_pale_dialogue_box_with_local_background(
                    image,
                    result,
                    (x1, y1, x2, y2),
                )
                if np.count_nonzero(pale_box_mask > 0) >= 20:
                    final_mask = cv2.bitwise_or(final_mask, pale_box_mask)
                    cleanup_status[constraint_id] = {"cleaned": True, "mode": "pale_bubble_box"}
                    continue
            inferred_bubble_mask = (
                _fill_unsegmented_bubble_text_strokes(
                    image,
                    result,
                    (x1, y1, x2, y2),
                    seg_mask,
                )
                if allow_inferred_bubble_cleanup
                else np.zeros((img_h, img_w), dtype=np.uint8)
            )
            if np.count_nonzero(inferred_bubble_mask > 0) >= 20:
                inferred_bubble_cleanups += 1
                final_mask = cv2.bitwise_or(final_mask, inferred_bubble_mask)
                cleanup_status[constraint_id] = {"cleaned": True, "mode": "inferred_bubble"}
                continue

            pad = 2
            float_x1 = max(0, x1 - pad)
            float_y1 = max(0, y1 - pad)
            float_x2 = min(img_w, x2 + pad)
            float_y2 = min(img_h, y2 + pad)
            device_like_floating = False

            region_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            erase_boxes = []
            cleanup_main_box = [x1, y1, x2, y2]
            box_width = x2 - x1
            box_height = y2 - y1
            if (
                constraint.get("green_box")
                and box_height >= box_width * 1.35
                and box_width <= max(180, int(img_w * 0.16))
            ):
                gx1, gy1, gx2, gy2 = [int(value) for value in constraint["green_box"]]
                red_area = max(1, (x2 - x1) * (y2 - y1))
                green_area = max(1, (gx2 - gx1) * (gy2 - gy1))
                if green_area <= red_area * 2.4:
                    candidate_cleanup_main_box = [
                        max(0, min(x1, gx1)),
                        max(0, min(y1, gy1)),
                        min(img_w, max(x2, gx2)),
                        min(img_h, max(y2, gy2)),
                    ]
                    if not _repair_box_crosses_foreground_line_art(
                        image,
                        tuple(candidate_cleanup_main_box),
                        source_red_coords,
                        dilated,
                    ):
                        cleanup_main_box = candidate_cleanup_main_box
            explicit_erase_boxes = constraint.get("erase_boxes") or []
            if not explicit_erase_boxes and cleanup_main_box != [x1, y1, x2, y2]:
                x1, y1, x2, y2 = cleanup_main_box
                float_x1 = max(0, x1 - pad)
                float_y1 = max(0, y1 - pad)
                float_x2 = min(img_w, x2 + pad)
                float_y2 = min(img_h, y2 + pad)
            raw_erase_boxes = (
                explicit_erase_boxes
                if explicit_erase_boxes and constraint.get("art_aware_erase_only")
                else [cleanup_main_box, *explicit_erase_boxes]
                if explicit_erase_boxes
                else []
            )
            for raw_box_index, raw_box in enumerate(raw_erase_boxes):
                if isinstance(raw_box, dict):
                    box_values = [
                        raw_box.get("x1", 0),
                        raw_box.get("y1", 0),
                        raw_box.get("x2", 0),
                        raw_box.get("y2", 0),
                    ]
                else:
                    box_values = raw_box
                if len(box_values) < 4:
                    continue
                ex1, ey1, ex2, ey2 = [int(value) for value in box_values[:4]]
                raw_width = max(0, ex2 - ex1)
                raw_height = max(0, ey2 - ey1)
                box_pad = 0
                if raw_box_index > 0:
                    box_pad = 6 if raw_width <= 80 and raw_height >= 38 else 3
                ex1 = max(0, min(img_w, ex1 - box_pad))
                ey1 = max(0, min(img_h, ey1 - box_pad))
                ex2 = max(0, min(img_w, ex2 + box_pad))
                ey2 = max(0, min(img_h, ey2 + box_pad))
                if ex2 > ex1 and ey2 > ey1:
                    erase_boxes.append((ex1, ey1, ex2, ey2))

            if len(erase_boxes) > 1:
                primary_box = erase_boxes[0]
                px1, py1, px2, py2 = primary_box
                erase_boxes = [
                    primary_box,
                    *[
                        box
                        for box in erase_boxes[1:]
                        if not (
                            box[0] >= px1 - 2
                            and box[1] >= py1 - 2
                            and box[2] <= px2 + 2
                            and box[3] <= py2 + 2
                        )
                    ],
                ]

            if erase_boxes:
                # A Step-6 consolidated dark bubble (multi-column same-
                # container merge) can carry a donated erase_boxes entry
                # even though it is still a single dark balloon underneath.
                # Give the purpose-built dark-balloon fill first refusal
                # before the generic multi-fragment union path below, which
                # has no balloon-polarity awareness and can leave visible
                # source-glyph residue on an otherwise-flat dark fill
                # (verified: new_sample_5's id13, a consolidated black
                # caption with 1 donated erase box, kept white-glyph streaks
                # under the union path even though the same box's reverse-
                # dark-balloon fill -- tested in isolation -- cleans it
                # fully). Every early return in _fill_reverse_dark_balloon_
                # text happens before its one write, so a decline here is a
                # guaranteed no-op and the union path below runs exactly as
                # before it.
                reverse_dark_precheck_candidates = []
                if constraint.get("green_box"):
                    pgx1, pgy1, pgx2, pgy2 = [int(v) for v in constraint["green_box"][:4]]
                    pre_green = (
                        max(0, min(img_w, pgx1)),
                        max(0, min(img_h, pgy1)),
                        max(0, min(img_w, pgx2)),
                        max(0, min(img_h, pgy2)),
                    )
                    if pre_green[2] > pre_green[0] and pre_green[3] > pre_green[1]:
                        pre_pad_x = max(6, min(18, int(round((pre_green[2] - pre_green[0]) * 0.18))))
                        pre_pad_y = max(6, min(16, int(round((pre_green[3] - pre_green[1]) * 0.08))))
                        reverse_dark_precheck_candidates.append((
                            max(0, pre_green[0] - pre_pad_x),
                            max(0, pre_green[1] - pre_pad_y),
                            min(img_w, pre_green[2] + pre_pad_x),
                            min(img_h, pre_green[3] + pre_pad_y),
                        ))
                        reverse_dark_precheck_candidates.append(pre_green)
                reverse_dark_precheck_candidates.append((x1, y1, x2, y2))
                reverse_dark_precheck_done = False
                for reverse_coords in reverse_dark_precheck_candidates:
                    rx1, ry1, rx2, ry2 = reverse_coords
                    if rx2 <= rx1 or ry2 <= ry1:
                        continue
                    if not _reverse_dark_balloon_candidate(image, reverse_coords):
                        continue
                    reverse_seed = _refined_floating_source_mask(image, seg_mask, reverse_coords)
                    reverse_allowed = container_mask[ry1:ry2, rx1:rx2]
                    reverse_mask = _fill_reverse_dark_balloon_text(
                        image,
                        result,
                        reverse_coords,
                        reverse_seed,
                        allowed_mask=reverse_allowed if np.any(reverse_allowed) else None,
                    )
                    if reverse_mask is None:
                        continue
                    precheck_region_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    precheck_region_mask[ry1:ry2, rx1:rx2] = reverse_mask
                    floating_mask = cv2.bitwise_or(floating_mask, precheck_region_mask)
                    final_mask = cv2.bitwise_or(final_mask, precheck_region_mask)
                    cleanup_status[constraint_id] = {
                        "cleaned": True,
                        "mode": "floating",
                        "reason": "reverse_dark_bubble_cleanup",
                    }
                    reverse_dark_precheck_done = True
                    break
                if reverse_dark_precheck_done:
                    continue

                routed_green_repair_done = False
                if (
                    constraint.get("art_aware_routed")
                    and constraint.get("green_box")
                    and len(erase_boxes) > 1
                ):
                    gx1, gy1, gx2, gy2 = [int(value) for value in constraint["green_box"][:4]]
                    green_coords = (
                        max(0, min(img_w, gx1)),
                        max(0, min(img_h, gy1)),
                        max(0, min(img_w, gx2)),
                        max(0, min(img_h, gy2)),
                    )
                    if green_coords[2] > green_coords[0] and green_coords[3] > green_coords[1]:
                        green_probe_mask = _refined_floating_source_mask(image, seg_mask, green_coords)
                        green_probe_count = int(np.count_nonzero(green_probe_mask > 0))
                        green_probe_density = green_probe_count / float(max(1, green_probe_mask.size))
                        green_mixed_tone = _floating_region_has_mixed_character_tone(
                            image,
                            green_coords,
                            green_probe_mask,
                        )
                        green_art_sensitive = _floating_region_is_art_sensitive(
                            image,
                            green_coords,
                            green_probe_mask,
                        )
                        if (
                            green_probe_count >= 60
                            and green_probe_density <= 0.70
                            and not green_mixed_tone
                            and (
                                not green_art_sensitive
                                or _safe_outlined_halftone_caption_region(
                                    image,
                                    green_coords,
                                    green_probe_mask,
                                )
                            )
                        ):
                            repair_snapshot = result.copy()
                            routed_repair_mask = _tight_floating_stroke_repair(
                                image,
                                result,
                                seg_mask,
                                green_coords,
                                anime_model,
                                anime_device,
                                clip_mask=semantic_clip_mask,
                            )
                            if routed_repair_mask is not None:
                                rgx1, rgy1, rgx2, rgy2 = green_coords
                                region_mask[rgy1:rgy2, rgx1:rgx2] = cv2.bitwise_or(
                                    region_mask[rgy1:rgy2, rgx1:rgx2],
                                    routed_repair_mask,
                                )
                                floating_mask = cv2.bitwise_or(floating_mask, region_mask)
                                final_mask = cv2.bitwise_or(final_mask, region_mask)
                                cleanup_status[constraint_id] = {
                                    "cleaned": True,
                                    "mode": "floating",
                                    "reason": "routed_tight_stroke_repair",
                                }
                                routed_green_repair_done = True
                            else:
                                result[:, :] = repair_snapshot
                if routed_green_repair_done:
                    continue
                if len(erase_boxes) > 1:
                    primary_box = erase_boxes[0]
                    primary_roi = _floating_erase_mask_for_box(image, seg_mask, primary_box, kernel_3)
                    primary_probe = result.copy()
                    primary_component_mask = _fill_component_local_background_text_mask(
                        image,
                        primary_probe,
                        primary_box,
                        primary_roi,
                    )
                    secondary_boxes_nested = all(
                        other[0] >= primary_box[0] - 4
                        and other[1] >= primary_box[1] - 4
                        and other[2] <= primary_box[2] + 4
                        and other[3] <= primary_box[3] + 4
                        for other in erase_boxes[1:]
                    )
                    if (
                        np.count_nonzero(primary_roi > 0) >= 20
                        and (
                            primary_component_mask is not None
                            or not _floating_cleanup_should_fail_closed(image, primary_box, primary_roi)
                        )
                        and secondary_boxes_nested
                    ):
                        erase_boxes = [primary_box]

                union_x1 = max(0, min(box[0] for box in erase_boxes) - 3)
                union_y1 = max(0, min(box[1] for box in erase_boxes) - 3)
                union_x2 = min(img_w, max(box[2] for box in erase_boxes) + 3)
                union_y2 = min(img_h, max(box[3] for box in erase_boxes) + 3)
                union_probe_mask = np.zeros((union_y2 - union_y1, union_x2 - union_x1), dtype=np.uint8)
                for ex1, ey1, ex2, ey2 in erase_boxes:
                    probe_roi = _floating_erase_mask_for_box(image, seg_mask, (ex1, ey1, ex2, ey2), kernel_3)
                    union_probe_mask[ey1 - union_y1:ey2 - union_y1, ex1 - union_x1:ex2 - union_x1] = np.maximum(
                        union_probe_mask[ey1 - union_y1:ey2 - union_y1, ex1 - union_x1:ex2 - union_x1],
                        probe_roi,
                    )
                if (
                    os.getenv("MANGA_DEVICE_OVERLAY_FALLBACK", "off").strip().lower()
                    in {"1", "true", "yes", "on"}
                    and _floating_region_requires_device_overlay(
                    image,
                    (union_x1, union_y1, union_x2, union_y2),
                    union_probe_mask,
                    )
                ):
                    cleanup_status[constraint_id] = {
                        "cleaned": False,
                        "mode": "floating",
                        "reason": "device_overlay_required",
                    }
                    continue
                if _floating_region_requires_source_cover(
                    image,
                    (union_x1, union_y1, union_x2, union_y2),
                    union_probe_mask,
                ):
                    cleanup_status[constraint_id] = {
                        "cleaned": False,
                        "mode": "floating",
                        "reason": "source_cover_required",
                    }
                    continue
                if constraint.get("force_full_line_cleanup"):
                    full_caption_mask = _fill_simple_floating_caption_box(
                        image,
                        result,
                        (union_x1, union_y1, union_x2, union_y2),
                    )
                    if full_caption_mask is not None:
                        region_mask[union_y1:union_y2, union_x1:union_x2] = cv2.bitwise_or(
                            region_mask[union_y1:union_y2, union_x1:union_x2],
                            full_caption_mask,
                        )
                        floating_mask = cv2.bitwise_or(floating_mask, region_mask)
                        final_mask = cv2.bitwise_or(final_mask, region_mask)
                        cleanup_status[constraint_id] = {
                            "cleaned": True,
                            "mode": "floating",
                            "reason": "forced_simple_caption_box",
                        }
                        continue
                precise_cleanup_mask = None
                if not _floating_region_has_mixed_character_tone(
                    image,
                    (union_x1, union_y1, union_x2, union_y2),
                    union_probe_mask,
                ):
                    union_allowed_mask = np.zeros(
                        (union_y2 - union_y1, union_x2 - union_x1), dtype=np.uint8
                    )
                    allowed_pad = 8
                    for ex1, ey1, ex2, ey2 in erase_boxes:
                        ax1 = max(0, ex1 - allowed_pad - union_x1)
                        ay1 = max(0, ey1 - allowed_pad - union_y1)
                        ax2 = min(union_x2 - union_x1, ex2 + allowed_pad - union_x1)
                        ay2 = min(union_y2 - union_y1, ey2 + allowed_pad - union_y1)
                        if ax2 > ax1 and ay2 > ay1:
                            union_allowed_mask[ay1:ay2, ax1:ax2] = 255
                    precise_cleanup_mask = _precise_floating_text_local_cleanup(
                        image,
                        result,
                        (union_x1, union_y1, union_x2, union_y2),
                        anime_model,
                        anime_device,
                        allowed_mask=union_allowed_mask,
                        seg_mask=seg_mask,
                        container_mask=container_mask,
                        clip_mask=semantic_clip_mask,
                    )
                if precise_cleanup_mask is not None:
                    region_mask[union_y1:union_y2, union_x1:union_x2] = cv2.bitwise_or(
                        region_mask[union_y1:union_y2, union_x1:union_x2],
                        precise_cleanup_mask,
                    )
                    floating_mask = cv2.bitwise_or(floating_mask, region_mask)
                    final_mask = cv2.bitwise_or(final_mask, region_mask)
                    cleanup_status[constraint_id] = {
                        "cleaned": True,
                        "mode": "floating",
                        "reason": "precise_floating_text_local_cleanup",
                    }
                    continue
                smooth_panel_mask = None
                if not _floating_region_has_mixed_character_tone(
                    image,
                    (union_x1, union_y1, union_x2, union_y2),
                    union_probe_mask,
                ):
                    smooth_panel_mask = _smooth_panel_box_gradient_cleanup(
                        image,
                        result,
                        (union_x1, union_y1, union_x2, union_y2),
                    )
                if smooth_panel_mask is not None:
                    adjacent_halo_cleanup = _cleanup_adjacent_smooth_panel_halos(
                        result,
                        (union_x1, union_y1, union_x2, union_y2),
                    )
                    region_mask[union_y1:union_y2, union_x1:union_x2] = cv2.bitwise_or(
                        region_mask[union_y1:union_y2, union_x1:union_x2],
                        smooth_panel_mask,
                    )
                    if adjacent_halo_cleanup is not None:
                        (ahx1, ahy1, ahx2, ahy2), adjacent_halo_mask = adjacent_halo_cleanup
                        region_mask[ahy1:ahy2, ahx1:ahx2] = cv2.bitwise_or(
                            region_mask[ahy1:ahy2, ahx1:ahx2],
                            adjacent_halo_mask,
                        )
                    floating_mask = cv2.bitwise_or(floating_mask, region_mask)
                    final_mask = cv2.bitwise_or(final_mask, region_mask)
                    cleanup_status[constraint_id] = {
                        "cleaned": True,
                        "mode": "floating",
                        "reason": "smooth_panel_box_gradient",
                    }
                    continue
                erase_snapshot = result[union_y1:union_y2, union_x1:union_x2].copy()
                prefilled_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                flat_restore_exclusion_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                unsafe_skipped = False
                for ex1, ey1, ex2, ey2 in erase_boxes:
                    erase_roi = _floating_erase_mask_for_box(
                        image,
                        seg_mask,
                        (ex1, ey1, ex2, ey2),
                        kernel_3,
                    )
                    if np.count_nonzero(erase_roi) >= 6:
                        if (
                            _floating_full_box_cleanup_allowed(image, (ex1, ey1, ex2, ey2))
                            and (
                                not _floating_region_has_mixed_character_tone(
                                    image,
                                    (ex1, ey1, ex2, ey2),
                                    erase_roi,
                                )
                                or _plain_paper_text_restore_exclusion_allowed(
                                    image,
                                    (ex1, ey1, ex2, ey2),
                                    erase_roi,
                                )
                            )
                        ):
                            flat_restore_exclusion_mask[ey1:ey2, ex1:ex2] = 255
                        if constraint.get("art_aware_routed"):
                            routed_roi = cv2.dilate(erase_roi, kernel_3, iterations=1)
                            if _fill_bright_paper_text_mask(
                                image,
                                result,
                                (ex1, ey1, ex2, ey2),
                                routed_roi,
                            ) or _fill_flat_background_text_mask(
                                image,
                                result,
                                (ex1, ey1, ex2, ey2),
                                routed_roi,
                            ):
                                prefilled_mask[ey1:ey2, ex1:ex2] = np.maximum(
                                    prefilled_mask[ey1:ey2, ex1:ex2],
                                    routed_roi,
                                )
                                continue
                        tight_repair_mask = _tight_floating_stroke_repair(
                            image,
                            result,
                            seg_mask,
                            (ex1, ey1, ex2, ey2),
                            anime_model,
                            anime_device,
                            clip_mask=semantic_clip_mask,
                        )
                        if tight_repair_mask is not None:
                            tight_repair_density = float(np.count_nonzero(tight_repair_mask > 0)) / float(
                                max(1, tight_repair_mask.size)
                            )
                            if tight_repair_density <= 0.68 and not _pure_paper_repair_context(
                                image,
                                (ex1, ey1, ex2, ey2),
                                tight_repair_mask,
                            ):
                                line_restore_mask = _restore_crossing_line_art_after_repair(
                                    image,
                                    result,
                                    (ex1, ey1, ex2, ey2),
                                    tight_repair_mask,
                                )
                                if line_restore_mask is not None:
                                    tight_repair_mask = cv2.bitwise_or(tight_repair_mask, line_restore_mask)
                                line_extend_mask = _extend_line_art_across_repair_mask(
                                    image,
                                    result,
                                    (ex1, ey1, ex2, ey2),
                                    tight_repair_mask,
                                )
                                if line_extend_mask is not None:
                                    tight_repair_mask = cv2.bitwise_or(tight_repair_mask, line_extend_mask)
                            residual_source_mask = _opencv_cleanup_visible_source_residual(
                                image,
                                result,
                                (ex1, ey1, ex2, ey2),
                                tight_repair_mask,
                            )
                            if residual_source_mask is not None:
                                tight_repair_mask = cv2.bitwise_or(tight_repair_mask, residual_source_mask)
                            halo_cleanup_mask = _clean_smooth_tone_residual_halos(
                                image,
                                result,
                                (ex1, ey1, ex2, ey2),
                                tight_repair_mask,
                            )
                            if halo_cleanup_mask is not None:
                                tight_repair_mask = cv2.bitwise_or(tight_repair_mask, halo_cleanup_mask)
                            prefilled_mask[ey1:ey2, ex1:ex2] = np.maximum(
                                prefilled_mask[ey1:ey2, ex1:ex2],
                                tight_repair_mask,
                            )
                            continue
                        if _floating_cleanup_should_fail_closed(image, (ex1, ey1, ex2, ey2), erase_roi):
                            unsafe_skipped = True
                            continue
                        local_component_mask = _fill_component_local_background_text_mask(
                            image,
                            result,
                            (ex1, ey1, ex2, ey2),
                            erase_roi,
                        )
                        if local_component_mask is not None:
                            halo_cleanup_mask = _clean_smooth_tone_residual_halos(
                                image,
                                result,
                                (ex1, ey1, ex2, ey2),
                                local_component_mask,
                            )
                            if halo_cleanup_mask is not None:
                                local_component_mask = cv2.bitwise_or(local_component_mask, halo_cleanup_mask)
                            prefilled_mask[ey1:ey2, ex1:ex2] = np.maximum(
                                prefilled_mask[ey1:ey2, ex1:ex2],
                                local_component_mask,
                            )
                            continue
                        if _fill_dark_background_text_mask(
                            image,
                            result,
                            (ex1, ey1, ex2, ey2),
                            erase_roi,
                        ) or _fill_flat_background_text_mask(
                            image,
                            result,
                            (ex1, ey1, ex2, ey2),
                            erase_roi,
                        ) or _fill_bright_paper_text_mask(
                            image,
                            result,
                            (ex1, ey1, ex2, ey2),
                            erase_roi,
                        ):
                            prefilled_mask[ey1:ey2, ex1:ex2] = np.maximum(
                                prefilled_mask[ey1:ey2, ex1:ex2],
                                cv2.dilate(erase_roi, kernel_3, iterations=1),
                            )
                        else:
                            region_mask[ey1:ey2, ex1:ex2] = np.maximum(
                                region_mask[ey1:ey2, ex1:ex2], erase_roi
                            )

                if not unsafe_skipped:
                    for ex1, ey1, ex2, ey2 in erase_boxes:
                        erase_roi = _floating_erase_mask_for_box(
                            image,
                            seg_mask,
                            (ex1, ey1, ex2, ey2),
                            kernel_3,
                        )
                        polish_x1, polish_y1, polish_x2, polish_y2 = ex1, ey1, ex2, ey2
                        polish_seed = erase_roi
                        if _plain_paper_text_restore_exclusion_allowed(
                            image,
                            (ex1, ey1, ex2, ey2),
                            erase_roi,
                        ):
                            expand_x = max(4, min(22, (ex2 - ex1) // 8))
                            expand_y = max(10, min(92, (ey2 - ey1) // 3))
                            candidate_coords = (
                                max(0, ex1 - expand_x),
                                max(0, ey1 - expand_y),
                                min(img_w, ex2 + expand_x),
                                min(img_h, ey2 + expand_y),
                            )
                            candidate_seed = _floating_erase_mask_for_box(
                                image,
                                seg_mask,
                                candidate_coords,
                                kernel_3,
                            )
                            if (
                                np.count_nonzero(candidate_seed > 0) >= np.count_nonzero(erase_roi > 0)
                                and _plain_paper_text_restore_exclusion_allowed(
                                    image,
                                    candidate_coords,
                                    candidate_seed,
                                )
                            ):
                                polish_x1, polish_y1, polish_x2, polish_y2 = candidate_coords
                                polish_seed = candidate_seed
                                flat_restore_exclusion_mask[
                                    polish_y1:polish_y2,
                                    polish_x1:polish_x2,
                                ] = 255
                        for _ in range(3):
                            residual_polish = _polish_plain_background_text_residual(
                                image,
                                result,
                                (polish_x1, polish_y1, polish_x2, polish_y2),
                                polish_seed,
                            )
                            if residual_polish is None:
                                break
                            prefilled_mask[polish_y1:polish_y2, polish_x1:polish_x2] = np.maximum(
                                prefilled_mask[polish_y1:polish_y2, polish_x1:polish_x2],
                                residual_polish,
                            )
                            combined_seed = cv2.bitwise_or(
                                (polish_seed > 0).astype(np.uint8) * 255,
                                residual_polish,
                            )
                            if int(np.count_nonzero(combined_seed > 0)) <= int(
                                np.count_nonzero(polish_seed > 0)
                            ):
                                break
                            polish_seed = combined_seed

                float_x1, float_y1, float_x2, float_y2 = union_x1, union_y1, union_x2, union_y2

                if np.count_nonzero(region_mask > 0) > 0:
                    region_roi = region_mask[float_y1:float_y2, float_x1:float_x2]
                    if _floating_region_is_art_sensitive(
                        image,
                        (float_x1, float_y1, float_x2, float_y2),
                        region_roi,
                    ):
                        if _manga_cleaner_local_crop(
                            result,
                            region_mask,
                            img_h,
                            img_w,
                            float_x1,
                            float_y1,
                            float_x2,
                            float_y2,
                        ):
                            pass
                        elif not _external_inpaint_command_local_crop(
                            result,
                            region_mask,
                            img_h,
                            img_w,
                            float_x1,
                            float_y1,
                            float_x2,
                            float_y2,
                        ):
                            unsafe_skipped = True
                            region_mask = np.zeros_like(region_mask)
                    elif _external_inpaint_command_local_crop(
                        result,
                        region_mask,
                        img_h,
                        img_w,
                        float_x1,
                        float_y1,
                        float_x2,
                        float_y2,
                    ):
                        pass
                    elif anime_model is not None:
                        _anime_lama_local_crop(
                            anime_model,
                            anime_device,
                            result,
                            region_mask,
                            img_h,
                            img_w,
                            float_x1,
                            float_y1,
                            float_x2,
                            float_y2,
                        )
                    else:
                        _lama_local_crop(
                            lama_session,
                            result,
                            region_mask,
                            img_h,
                            img_w,
                            float_x1,
                            float_y1,
                            float_x2,
                            float_y2,
                        )

                if unsafe_skipped:
                    result[union_y1:union_y2, union_x1:union_x2] = erase_snapshot
                    prefilled_mask = np.zeros_like(prefilled_mask)
                    region_mask = np.zeros_like(region_mask)

                combined_float_mask = cv2.bitwise_or(region_mask, prefilled_mask)
                union_halo_seed = combined_float_mask[union_y1:union_y2, union_x1:union_x2]
                union_halo_cleanup_mask = _clean_smooth_tone_residual_halos(
                    image,
                    result,
                    (union_x1, union_y1, union_x2, union_y2),
                    union_halo_seed,
                )
                if union_halo_cleanup_mask is not None:
                    combined_float_mask[union_y1:union_y2, union_x1:union_x2] = np.maximum(
                        combined_float_mask[union_y1:union_y2, union_x1:union_x2],
                        union_halo_cleanup_mask,
                    )
                if not constraint.get("art_aware_routed"):
                    restore_input_mask = combined_float_mask
                    if np.count_nonzero(flat_restore_exclusion_mask > 0) > 0:
                        restore_input_mask = cv2.bitwise_and(
                            combined_float_mask,
                            cv2.bitwise_not(flat_restore_exclusion_mask),
                        )
                    restored_foreground_mask = _restore_foreground_art_outside_source_text(
                        image,
                        result,
                        restore_input_mask,
                        source_red_coords,
                        dilated,
                    )
                    if restored_foreground_mask is not None:
                        combined_float_mask = cv2.bitwise_and(
                            combined_float_mask,
                            cv2.bitwise_not(restored_foreground_mask),
                        )
                floating_mask = cv2.bitwise_or(floating_mask, combined_float_mask)
                final_mask = cv2.bitwise_or(final_mask, combined_float_mask)
                cleaned = bool(np.count_nonzero(combined_float_mask > 0) > 0)
                cleanup_status[constraint_id] = {
                    "cleaned": cleaned,
                    "mode": "floating",
                    "reason": (
                        "partial_unsafe_art_preserved"
                        if cleaned and unsafe_skipped
                        else "cleaned" if cleaned else "unsafe_art_preserved"
                    ),
                }
                continue

            if use_layout_stroke_mask:
                region_mask[y1:y2, x1:x2] = dilated
            else:
                floating_roi = _floating_text_erase_roi(image, (x1, y1, x2, y2))
                if np.count_nonzero(floating_roi > 0) < 6:
                    cleanup_status[constraint_id] = {
                        "cleaned": False,
                        "mode": "floating",
                        "reason": "no_safe_text_mask",
                    }
                    continue
                device_like_floating = _floating_region_requires_device_overlay(
                    image,
                    (x1, y1, x2, y2),
                    floating_roi,
                )
                if (
                    os.getenv("MANGA_DEVICE_OVERLAY_FALLBACK", "off").strip().lower()
                    in {"1", "true", "yes", "on"}
                    and device_like_floating
                ):
                    cleanup_status[constraint_id] = {
                        "cleaned": False,
                        "mode": "floating",
                        "reason": "device_overlay_required",
                    }
                    continue
                if _floating_region_requires_source_cover(image, (x1, y1, x2, y2), floating_roi):
                    cleanup_status[constraint_id] = {
                        "cleaned": False,
                        "mode": "floating",
                        "reason": "source_cover_required",
                    }
                    continue
                floating_area = max(1, (x2 - x1) * (y2 - y1))
                floating_density = float(np.count_nonzero(floating_roi > 0)) / float(floating_area)
                floating_tall = (y2 - y1) >= max(92, int((x2 - x1) * 1.20))
                floating_wide = (x2 - x1) >= max(180, int((y2 - y1) * 1.75))
                prefer_box_cleanup = (
                    not constraint.get("erase_boxes")
                    and not use_layout_stroke_mask
                    and (
                        (floating_tall and floating_density >= 0.26)
                        or (
                            floating_area >= 18000
                            and floating_density >= 0.68
                            and not floating_wide
                        )
                    )
                    and _floating_full_box_cleanup_allowed(image, (x1, y1, x2, y2))
                )
                screen_ui_caption = (
                    not constraint.get("erase_boxes")
                    and not use_layout_stroke_mask
                    and _striped_screen_ui_caption_candidate(image, (x1, y1, x2, y2), floating_roi)
                )
                if screen_ui_caption:
                    screen_ui_box = _union_repair_box(
                        (x1, y1, x2, y2),
                        constraint.get("green_box"),
                        image.shape,
                    )
                    screen_ui_repair = _dark_device_surface_text_cleanup(
                        image,
                        result,
                        screen_ui_box,
                    )
                    if screen_ui_repair is not None:
                        floating_mask = cv2.bitwise_or(floating_mask, screen_ui_repair)
                        final_mask = cv2.bitwise_or(final_mask, screen_ui_repair)
                        cleanup_status[constraint_id] = {
                            "cleaned": True,
                            "mode": "floating",
                            "reason": "dark_device_surface_text_cleanup",
                        }
                    else:
                        cleanup_status[constraint_id] = {
                            "cleaned": False,
                            "mode": "floating",
                            "reason": "device_surface_preserved_overlay",
                        }
                    continue
                legacy_caption_repair = None
                legacy_caption_reason = "legacy_bright_caption_anime"
                if (
                    not constraint.get("erase_boxes")
                    and not use_layout_stroke_mask
                    and not screen_ui_caption
                    and _legacy_bright_caption_candidate(image, (x1, y1, x2, y2), floating_roi)
                ):
                    legacy_repair_box = _union_repair_box(
                        (x1, y1, x2, y2),
                        constraint.get("green_box"),
                        image.shape,
                    )
                    refined_caption_mask = _refined_floating_source_mask(
                        image,
                        seg_mask,
                        (x1, y1, x2, y2),
                    )
                    if int(np.count_nonzero(refined_caption_mask > 0)) < 6:
                        refined_caption_mask = floating_roi
                    art_boundary_risk = _repair_box_crosses_foreground_line_art(
                        image,
                        legacy_repair_box,
                        (x1, y1, x2, y2),
                        refined_caption_mask,
                    )
                    mixed_surface_caption = _floating_region_has_mixed_character_tone(
                        image,
                        (x1, y1, x2, y2),
                        floating_roi,
                    )
                    if mixed_surface_caption:
                        legacy_caption_repair = _mixed_surface_caption_fill(
                            image,
                            result,
                            (x1, y1, x2, y2),
                            refined_caption_mask,
                        )
                        if legacy_caption_repair is not None:
                            legacy_caption_reason = "mixed_surface_caption_fill"
                    if legacy_caption_repair is None and art_boundary_risk:
                        local_caption_repair = _dilated_anime_caption_repair(
                            image,
                            result,
                            (x1, y1, x2, y2),
                            refined_caption_mask,
                            anime_model,
                            anime_device,
                        )
                        if local_caption_repair is not None:
                            legacy_caption_repair = np.zeros(image.shape[:2], dtype=np.uint8)
                            legacy_caption_repair[y1:y2, x1:x2] = local_caption_repair
                            line_restore_mask = _restore_art_lines_crossing_repair_mask(
                                image,
                                result,
                                (x1, y1, x2, y2),
                                local_caption_repair,
                                refined_caption_mask,
                            )
                            if line_restore_mask is not None:
                                legacy_caption_repair[y1:y2, x1:x2] = cv2.bitwise_or(
                                    legacy_caption_repair[y1:y2, x1:x2],
                                    line_restore_mask,
                                )
                    elif legacy_caption_repair is None:
                        legacy_caption_repair = _legacy_full_box_anime_repair(
                            image,
                            result,
                            legacy_repair_box,
                            anime_model,
                            anime_device,
                            pad=_compact_caption_repair_pad(legacy_repair_box),
                        )
                    if legacy_caption_repair is not None:
                        _restore_wide_caption_line_art(
                            image,
                            result,
                            (x1, y1, x2, y2),
                            legacy_repair_box,
                        )
                        lx1, ly1, lx2, ly2 = legacy_repair_box
                        legacy_local_mask = legacy_caption_repair[ly1:ly2, lx1:lx2]
                        line_restore_mask = _restore_crossing_line_art_after_repair(
                            image,
                            result,
                            legacy_repair_box,
                            legacy_local_mask,
                        )
                        if line_restore_mask is not None:
                            legacy_caption_repair[ly1:ly2, lx1:lx2] = cv2.bitwise_or(
                                legacy_local_mask,
                                line_restore_mask,
                            )
                            legacy_local_mask = legacy_caption_repair[ly1:ly2, lx1:lx2]
                        halo_cleanup_mask = _clean_smooth_tone_residual_halos(
                            image,
                            result,
                            legacy_repair_box,
                            legacy_local_mask,
                        )
                        if halo_cleanup_mask is not None:
                            legacy_caption_repair[ly1:ly2, lx1:lx2] = cv2.bitwise_or(
                                legacy_caption_repair[ly1:ly2, lx1:lx2],
                                halo_cleanup_mask,
                            )
                        legacy_mixed_tone = _floating_region_has_mixed_character_tone(
                            image,
                            (x1, y1, x2, y2),
                            floating_roi,
                        )
                        if legacy_mixed_tone and legacy_caption_reason != "mixed_surface_caption_fill":
                            legacy_surface_mask = legacy_caption_repair[ly1:ly2, lx1:lx2]
                            if int(np.count_nonzero(legacy_surface_mask > 0)) >= 8:
                                _local_repair_tone_match(
                                    image,
                                    result,
                                    legacy_repair_box,
                                    legacy_surface_mask,
                                )
                                restored_guard_mask = _restore_source_outside_caption_guard(
                                    image,
                                    result,
                                    legacy_repair_box,
                                    (x1, y1, x2, y2),
                                    refined_caption_mask,
                                    legacy_surface_mask,
                                )
                                if restored_guard_mask is not None:
                                    legacy_surface_mask[restored_guard_mask > 0] = 0
                        elif legacy_caption_reason != "mixed_surface_caption_fill":
                            adjacent_halo_cleanup = _cleanup_adjacent_smooth_panel_halos(
                                result,
                                legacy_repair_box,
                            )
                            if adjacent_halo_cleanup is not None:
                                (ahx1, ahy1, ahx2, ahy2), adjacent_halo_mask = adjacent_halo_cleanup
                                legacy_caption_repair[ahy1:ahy2, ahx1:ahx2] = cv2.bitwise_or(
                                    legacy_caption_repair[ahy1:ahy2, ahx1:ahx2],
                                    adjacent_halo_mask,
                                )
                if legacy_caption_repair is not None:
                    floating_mask = cv2.bitwise_or(floating_mask, legacy_caption_repair)
                    final_mask = cv2.bitwise_or(final_mask, legacy_caption_repair)
                    cleanup_status[constraint_id] = {
                        "cleaned": True,
                        "mode": "floating",
                        "reason": legacy_caption_reason,
                    }
                    continue
                dense_surface_repair = _dense_mixed_surface_fullbox_model_repair(
                    image,
                    result,
                    (x1, y1, x2, y2),
                    floating_roi,
                    anime_model,
                    anime_device,
                )
                if dense_surface_repair is not None:
                    floating_mask = cv2.bitwise_or(floating_mask, dense_surface_repair)
                    final_mask = cv2.bitwise_or(final_mask, dense_surface_repair)
                    cleanup_status[constraint_id] = {
                        "cleaned": True,
                        "mode": "floating",
                        "reason": "dense_mixed_surface_text_mask",
                    }
                    continue
                prefer_box_cleanup = False
                tight_coords = (x1, y1, x2, y2)
                smooth_panel_coords = None
                reverse_dark_candidates = []
                if constraint.get("green_box"):
                    gx1, gy1, gx2, gy2 = [int(value) for value in constraint["green_box"][:4]]
                    green_candidate = (
                        max(0, min(img_w, gx1)),
                        max(0, min(img_h, gy1)),
                        max(0, min(img_w, gx2)),
                        max(0, min(img_h, gy2)),
                    )
                    if green_candidate[2] > green_candidate[0] and green_candidate[3] > green_candidate[1]:
                        reverse_pad_x = max(6, min(18, int(round((green_candidate[2] - green_candidate[0]) * 0.18))))
                        reverse_pad_y = max(6, min(16, int(round((green_candidate[3] - green_candidate[1]) * 0.08))))
                        expanded_reverse_candidate = (
                            max(0, green_candidate[0] - reverse_pad_x),
                            max(0, green_candidate[1] - reverse_pad_y),
                            min(img_w, green_candidate[2] + reverse_pad_x),
                            min(img_h, green_candidate[3] + reverse_pad_y),
                        )
                        reverse_dark_candidates.append(expanded_reverse_candidate)
                        reverse_dark_candidates.append(green_candidate)
                reverse_dark_candidates.append((x1, y1, x2, y2))
                reverse_dark_done = False
                for reverse_coords in reverse_dark_candidates:
                    rx1, ry1, rx2, ry2 = reverse_coords
                    if rx2 <= rx1 or ry2 <= ry1:
                        continue
                    if not _reverse_dark_balloon_candidate(image, reverse_coords):
                        continue
                    reverse_seed = _refined_floating_source_mask(image, seg_mask, reverse_coords)
                    reverse_allowed = container_mask[ry1:ry2, rx1:rx2]
                    reverse_mask = _fill_reverse_dark_balloon_text(
                        image,
                        result,
                        reverse_coords,
                        reverse_seed,
                        allowed_mask=reverse_allowed if np.any(reverse_allowed) else None,
                    )
                    if reverse_mask is None:
                        continue
                    region_mask[ry1:ry2, rx1:rx2] = cv2.bitwise_or(
                        region_mask[ry1:ry2, rx1:rx2],
                        reverse_mask,
                    )
                    floating_mask = cv2.bitwise_or(floating_mask, region_mask)
                    final_mask = cv2.bitwise_or(final_mask, region_mask)
                    cleanup_status[constraint_id] = {
                        "cleaned": True,
                        "mode": "floating",
                        "reason": "reverse_dark_bubble_cleanup",
                    }
                    reverse_dark_done = True
                    break
                if reverse_dark_done:
                    continue
                if not constraint.get("erase_boxes") and constraint.get("green_box"):
                    gx1, gy1, gx2, gy2 = [int(value) for value in constraint["green_box"]]
                    gx1 = max(0, min(img_w, gx1))
                    gy1 = max(0, min(img_h, gy1))
                    gx2 = max(0, min(img_w, gx2))
                    gy2 = max(0, min(img_h, gy2))
                    red_width = max(1, x2 - x1)
                    red_height = max(1, y2 - y1)
                    red_area = red_width * red_height
                    green_area = max(1, (gx2 - gx1) * (gy2 - gy1))
                    max_expand = max(10, int(max(red_width, red_height) * 0.18))
                    modest_green = (
                        gx2 > gx1
                        and gy2 > gy1
                        and green_area <= red_area * 1.55
                        and gx1 <= x1
                        and gy1 <= y1
                        and gx2 >= x2
                        and gy2 >= y2
                        and (x1 - gx1) <= max_expand
                        and (y1 - gy1) <= max_expand
                        and (gx2 - x2) <= max_expand
                        and (gy2 - y2) <= max_expand
                    )
                    if modest_green:
                        green_cleanup_coords = (gx1, gy1, gx2, gy2)
                        if not _repair_box_crosses_foreground_line_art(
                            image,
                            green_cleanup_coords,
                            (x1, y1, x2, y2),
                            floating_roi,
                        ):
                            tight_coords = green_cleanup_coords
                            smooth_panel_coords = tight_coords
                elif constraint.get("green_box"):
                    gx1, gy1, gx2, gy2 = [int(value) for value in constraint["green_box"]]
                    gx1 = max(0, min(img_w, gx1))
                    gy1 = max(0, min(img_h, gy1))
                    gx2 = max(0, min(img_w, gx2))
                    gy2 = max(0, min(img_h, gy2))
                    if gx2 > gx1 and gy2 > gy1:
                        smooth_panel_coords = (gx1, gy1, gx2, gy2)
                if smooth_panel_coords is not None:
                    spx1, spy1, spx2, spy2 = smooth_panel_coords
                    for erase_box in constraint.get("erase_boxes") or []:
                        if not erase_box or len(erase_box) < 4:
                            continue
                        ex1, ey1, ex2, ey2 = [int(value) for value in erase_box[:4]]
                        ex1 = max(0, min(img_w, ex1))
                        ey1 = max(0, min(img_h, ey1))
                        ex2 = max(0, min(img_w, ex2))
                        ey2 = max(0, min(img_h, ey2))
                        if ex2 <= ex1 or ey2 <= ey1:
                            continue
                        expand_w = max(ex2 - ex1, spx2 - spx1)
                        expand_h = max(ey2 - ey1, spy2 - spy1)
                        if (
                            abs(ex1 - spx1) <= expand_w * 1.35
                            and abs(ex2 - spx2) <= expand_w * 1.35
                            and abs(ey1 - spy1) <= expand_h * 1.35
                            and abs(ey2 - spy2) <= expand_h * 1.35
                        ):
                            spx1 = min(spx1, ex1)
                            spy1 = min(spy1, ey1)
                            spx2 = max(spx2, ex2)
                            spy2 = max(spy2, ey2)
                    smooth_panel_coords = (spx1, spy1, spx2, spy2)
                    smooth_panel_mask = None
                    original_smooth_panel_coords = smooth_panel_coords
                    panel_width = max(1, spx2 - spx1)
                    panel_height = max(1, spy2 - spy1)
                    panel_pad_x = max(6, min(18, int(round(panel_width * 0.085))))
                    panel_pad_y = max(4, min(14, int(round(panel_height * 0.035))))
                    expanded_smooth_panel_coords = (
                        max(0, spx1 - panel_pad_x),
                        max(0, spy1 - panel_pad_y),
                        min(img_w, spx2 + panel_pad_x),
                        min(img_h, spy2 + panel_pad_y),
                    )
                    smooth_panel_candidates = [expanded_smooth_panel_coords]
                    if original_smooth_panel_coords != expanded_smooth_panel_coords:
                        smooth_panel_candidates.append(original_smooth_panel_coords)
                    for candidate_smooth_panel_coords in smooth_panel_candidates:
                        if _floating_region_has_mixed_character_tone(
                            image,
                            candidate_smooth_panel_coords,
                            floating_roi,
                        ):
                            continue
                        candidate_precise_mask = _precise_floating_text_local_cleanup(
                            image,
                            result,
                            candidate_smooth_panel_coords,
                            anime_model,
                            anime_device,
                            seg_mask=seg_mask,
                            clip_mask=semantic_clip_mask,
                        )
                        if candidate_precise_mask is not None:
                            smooth_panel_coords = candidate_smooth_panel_coords
                            smooth_panel_mask = candidate_precise_mask
                            spx1, spy1, spx2, spy2 = smooth_panel_coords
                            break
                        candidate_smooth_panel_mask = _smooth_panel_box_gradient_cleanup(
                            image,
                            result,
                            candidate_smooth_panel_coords,
                        )
                        if candidate_smooth_panel_mask is not None:
                            smooth_panel_coords = candidate_smooth_panel_coords
                            smooth_panel_mask = candidate_smooth_panel_mask
                            spx1, spy1, spx2, spy2 = smooth_panel_coords
                            break
                    if smooth_panel_mask is not None:
                        spx1, spy1, spx2, spy2 = smooth_panel_coords
                        adjacent_halo_cleanup = _cleanup_adjacent_smooth_panel_halos(
                            result,
                            smooth_panel_coords,
                        )
                        region_mask[spy1:spy2, spx1:spx2] = cv2.bitwise_or(
                            region_mask[spy1:spy2, spx1:spx2],
                            smooth_panel_mask,
                        )
                        if adjacent_halo_cleanup is not None:
                            (ahx1, ahy1, ahx2, ahy2), adjacent_halo_mask = adjacent_halo_cleanup
                            region_mask[ahy1:ahy2, ahx1:ahx2] = cv2.bitwise_or(
                                region_mask[ahy1:ahy2, ahx1:ahx2],
                                adjacent_halo_mask,
                            )
                        floating_mask = cv2.bitwise_or(floating_mask, region_mask)
                        final_mask = cv2.bitwise_or(final_mask, region_mask)
                        cleanup_status[constraint_id] = {
                            "cleaned": True,
                            "mode": "floating",
                            "reason": "smooth_panel_box_gradient",
                        }
                        continue
                current_cleanup_coords = (x1, y1, x2, y2)
                green_coords = None
                if constraint.get("green_box"):
                    gx1, gy1, gx2, gy2 = [int(value) for value in constraint.get("green_box", [])[:4]]
                    green_coords = (
                        max(0, min(img_w, gx1)),
                        max(0, min(img_h, gy1)),
                        max(0, min(img_w, gx2)),
                        max(0, min(img_h, gy2)),
                    )
                if _floating_region_has_mixed_character_tone(image, current_cleanup_coords, floating_roi):
                    tight_repair_candidates = [current_cleanup_coords]
                    if tight_coords not in tight_repair_candidates:
                        tight_repair_candidates.append(tight_coords)
                    if source_red_coords not in tight_repair_candidates:
                        tight_repair_candidates.append(source_red_coords)
                else:
                    tight_repair_candidates = [source_red_coords]
                if current_cleanup_coords not in tight_repair_candidates:
                    tight_repair_candidates.append(current_cleanup_coords)
                if tight_coords not in tight_repair_candidates:
                    tight_repair_candidates.append(tight_coords)
                if (
                    green_coords is not None
                    and green_coords != source_red_coords
                ):
                    green_probe_mask = _refined_floating_source_mask(image, seg_mask, green_coords)
                    green_probe_count = int(np.count_nonzero(green_probe_mask > 0))
                    green_probe_density = green_probe_count / float(max(1, green_probe_mask.size))
                    red_probe_mask_for_green = _refined_floating_source_mask(image, seg_mask, source_red_coords)
                    red_probe_count_for_green = int(np.count_nonzero(red_probe_mask_for_green > 0))
                    green_mixed_tone = _floating_region_has_mixed_character_tone(
                        image,
                        green_coords,
                        green_probe_mask,
                    )
                    green_art_sensitive = _floating_region_is_art_sensitive(
                        image,
                        green_coords,
                        green_probe_mask,
                    )
                    green_safe_halftone = _safe_outlined_halftone_caption_region(
                        image,
                        green_coords,
                        green_probe_mask,
                    )
                    green_roi = image[
                        green_coords[1]:green_coords[3],
                        green_coords[0]:green_coords[2],
                    ]
                    green_compact_midtone = False
                    if green_roi.size:
                        green_gray = cv2.cvtColor(green_roi, cv2.COLOR_BGR2GRAY)
                        green_hsv = cv2.cvtColor(green_roi, cv2.COLOR_BGR2HSV)
                        green_height, green_width = green_gray.shape[:2]
                        green_compact_midtone = (
                            green_width <= 150
                            and green_height <= 112
                            and green_probe_density >= 0.50
                            and float(np.mean(green_hsv[:, :, 1] < 170)) >= 0.94
                            and float(np.mean((green_gray >= 62) & (green_gray <= 228) & (green_hsv[:, :, 1] < 175))) >= 0.24
                            and 0.10 <= float(np.mean(green_gray > 235)) <= 0.58
                            and float(np.mean(green_gray < 58)) <= 0.50
                            and float(np.mean(cv2.Canny(green_gray, 45, 135) > 0)) <= 0.36
                        )
                    green_coverage_gain = (
                        green_probe_count >= max(48, int(red_probe_count_for_green * 1.45))
                        or (
                            green_safe_halftone
                            and green_probe_count >= max(48, int(red_probe_count_for_green * 0.92))
                        )
                        or (
                            green_compact_midtone
                            and green_probe_count >= max(48, int(red_probe_count_for_green * 0.92))
                        )
                    )
                    if (
                        green_coverage_gain
                        and (green_probe_density <= 0.68 or green_compact_midtone)
                        and (not green_mixed_tone or green_compact_midtone)
                        and (
                            not green_art_sensitive
                            or green_safe_halftone
                            or green_compact_midtone
                        )
                    ):
                        tight_repair_candidates = [green_coords] + [
                            candidate
                            for candidate in tight_repair_candidates
                            if candidate != green_coords
                        ]
                if (
                    not constraint.get("erase_boxes")
                    and current_cleanup_coords != source_red_coords
                ):
                    red_probe_mask = _refined_floating_source_mask(image, seg_mask, source_red_coords)
                    current_probe_mask = _refined_floating_source_mask(image, seg_mask, current_cleanup_coords)
                    red_probe_count = int(np.count_nonzero(red_probe_mask > 0))
                    current_probe_count = int(np.count_nonzero(current_probe_mask > 0))
                    current_probe_density = current_probe_count / float(max(1, current_probe_mask.size))
                    if (
                        current_probe_count >= max(40, int(red_probe_count * 1.55))
                        and current_probe_density <= 0.66
                        and not _floating_region_has_mixed_character_tone(
                            image,
                            current_cleanup_coords,
                            current_probe_mask,
                        )
                        and not _floating_region_is_art_sensitive(
                            image,
                            current_cleanup_coords,
                            current_probe_mask,
                        )
                    ):
                        ordered_candidates = [current_cleanup_coords]
                        ordered_candidates.extend(
                            candidate
                            for candidate in tight_repair_candidates
                            if candidate != current_cleanup_coords
                        )
                        tight_repair_candidates = ordered_candidates
                tight_repair_mask = None
                for candidate_coords in tight_repair_candidates:
                    repair_snapshot = result.copy()
                    candidate_mask = _tight_floating_stroke_repair(
                        image,
                        result,
                        seg_mask,
                        candidate_coords,
                        anime_model,
                        anime_device,
                        clip_mask=semantic_clip_mask,
                    )
                    if candidate_mask is not None:
                        tight_coords = candidate_coords
                        tight_repair_mask = candidate_mask
                        break
                    result[:, :] = repair_snapshot
                if tight_repair_mask is not None:
                    tx1, ty1, tx2, ty2 = tight_coords
                    tight_repair_density = float(np.count_nonzero(tight_repair_mask > 0)) / float(
                        max(1, (tx2 - tx1) * (ty2 - ty1))
                    )
                    if (
                        not prefer_box_cleanup
                        or constraint.get("erase_boxes")
                        or tight_repair_density <= 0.78
                    ):
                        if tight_repair_density <= 0.68 and not _pure_paper_repair_context(
                            image,
                            tight_coords,
                            tight_repair_mask,
                        ):
                            line_restore_mask = _restore_crossing_line_art_after_repair(
                                image,
                                result,
                                tight_coords,
                                tight_repair_mask,
                            )
                            if line_restore_mask is not None:
                                tight_repair_mask = cv2.bitwise_or(tight_repair_mask, line_restore_mask)
                            line_extend_mask = _extend_line_art_across_repair_mask(
                                image,
                                result,
                                tight_coords,
                                tight_repair_mask,
                            )
                            if line_extend_mask is not None:
                                tight_repair_mask = cv2.bitwise_or(tight_repair_mask, line_extend_mask)
                        tight_roi = image[ty1:ty2, tx1:tx2]
                        if tight_roi.size:
                            tight_gray = cv2.cvtColor(tight_roi, cv2.COLOR_BGR2GRAY)
                            tight_hsv = cv2.cvtColor(tight_roi, cv2.COLOR_BGR2HSV)
                            tight_height, tight_width = tight_gray.shape[:2]
                            if (
                                tight_repair_density >= 0.78
                                and tight_width <= 150
                                and tight_height <= 112
                                and float(np.mean(tight_hsv[:, :, 1] < 170)) >= 0.94
                                and float(np.mean((tight_gray >= 62) & (tight_gray <= 228) & (tight_hsv[:, :, 1] < 175))) >= 0.24
                                and float(np.mean(cv2.Canny(tight_gray, 45, 135) > 0)) <= 0.36
                            ):
                                region_mask[ty1:ty2, tx1:tx2] = tight_repair_mask
                                floating_mask = cv2.bitwise_or(floating_mask, region_mask)
                                final_mask = cv2.bitwise_or(final_mask, region_mask)
                                cleanup_status[constraint_id] = {
                                    "cleaned": True,
                                    "mode": "floating",
                                    "reason": "tight_stroke_repair",
                                }
                                continue
                        residual_source_mask = _opencv_cleanup_visible_source_residual(
                            image,
                            result,
                            tight_coords,
                            tight_repair_mask,
                        )
                        if residual_source_mask is not None:
                            tight_repair_mask = cv2.bitwise_or(tight_repair_mask, residual_source_mask)
                        halo_cleanup_mask = _clean_smooth_tone_residual_halos(
                            image,
                            result,
                            tight_coords,
                            tight_repair_mask,
                        )
                        if halo_cleanup_mask is not None:
                            tight_repair_mask = cv2.bitwise_or(tight_repair_mask, halo_cleanup_mask)
                        refined_residual_seed = _refined_floating_source_mask(image, seg_mask, tight_coords)
                        if refined_residual_seed is not None and np.count_nonzero(refined_residual_seed > 0) >= 8:
                            halftone_residual_mask = _clean_halftone_residual_deviation(
                                image,
                                result,
                                tight_coords,
                                refined_residual_seed,
                            )
                            if halftone_residual_mask is not None:
                                tight_repair_mask = cv2.bitwise_or(tight_repair_mask, halftone_residual_mask)
                        adjacent_cleanup = _cleanup_adjacent_smooth_panel_text(
                            image,
                            result,
                            tight_coords,
                            tight_repair_mask,
                        )
                        if adjacent_cleanup is not None:
                            adjacent_coords, adjacent_mask = adjacent_cleanup
                            ax1, ay1, ax2, ay2 = adjacent_coords
                            region_mask[ay1:ay2, ax1:ax2] = cv2.bitwise_or(
                                region_mask[ay1:ay2, ax1:ax2],
                                adjacent_mask,
                            )
                        region_mask[ty1:ty2, tx1:tx2] = tight_repair_mask
                        floating_mask = cv2.bitwise_or(floating_mask, region_mask)
                        final_mask = cv2.bitwise_or(final_mask, region_mask)
                        cleanup_status[constraint_id] = {
                            "cleaned": True,
                            "mode": "floating",
                            "reason": "tight_stroke_repair",
                        }
                        continue
                if (
                    not prefer_box_cleanup
                    and _floating_cleanup_should_fail_closed(image, (x1, y1, x2, y2), floating_roi)
                ):
                    cleanup_status[constraint_id] = {
                        "cleaned": False,
                        "mode": "floating",
                        "reason": "unsafe_art_preserved",
                    }
                    continue
                if not prefer_box_cleanup:
                    local_component_mask = _fill_component_local_background_text_mask(
                        image,
                        result,
                        (x1, y1, x2, y2),
                        floating_roi,
                    )
                    if local_component_mask is not None:
                        halo_cleanup_mask = _clean_smooth_tone_residual_halos(
                            image,
                            result,
                            (x1, y1, x2, y2),
                            local_component_mask,
                        )
                        if halo_cleanup_mask is not None:
                            local_component_mask = cv2.bitwise_or(local_component_mask, halo_cleanup_mask)
                        region_mask[y1:y2, x1:x2] = local_component_mask
                        floating_mask = cv2.bitwise_or(floating_mask, region_mask)
                        final_mask = cv2.bitwise_or(final_mask, region_mask)
                        cleanup_status[constraint_id] = {
                            "cleaned": True,
                            "mode": "floating",
                            "reason": "component_local_repair",
                        }
                        continue
                    region_mask[y1:y2, x1:x2] = floating_roi
                else:
                    region_mask[float_y1:float_y2, float_x1:float_x2] = 255

            is_bottom_margin = float_y1 > int(img_h * 0.90) and (float_y2 - float_y1) <= 80
            if is_bottom_margin:
                _fill_with_neighbor_background(
                    image, result, float_x1, float_y1, float_x2, float_y2
                )
            else:
                tone_pad = 5
                tone_x1 = max(0, x1 - tone_pad)
                tone_y1 = max(0, y1 - tone_pad)
                tone_x2 = min(img_w, x2 + tone_pad)
                tone_y2 = min(img_h, y2 + tone_pad)
                current_mask_roi = region_mask[y1:y2, x1:x2]
                current_mask_density = float(np.count_nonzero(current_mask_roi > 0)) / float(max(1, current_mask_roi.size))
                if (
                    not device_like_floating
                    and not use_layout_stroke_mask
                    and current_mask_density >= 0.72
                    and _maybe_fill_screentone_background(
                        image, result, tone_x1, tone_y1, tone_x2, tone_y2
                    )
                ):
                    region_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    region_mask[tone_y1:tone_y2, tone_x1:tone_x2] = 255
                elif not device_like_floating and not use_layout_stroke_mask and (
                    _fill_dark_background_text_mask(
                        image,
                        result,
                        (x1, y1, x2, y2),
                        current_mask_roi,
                    )
                    or _fill_flat_background_text_mask(
                        image,
                        result,
                        (x1, y1, x2, y2),
                        current_mask_roi,
                    )
                    or _fill_bright_paper_text_mask(
                        image,
                        result,
                        (x1, y1, x2, y2),
                        current_mask_roi,
                    )
                ):
                    pass
                elif (
                    not use_layout_stroke_mask
                    and current_mask_density >= 0.72
                    and _maybe_fill_screentone_background(
                        image, result, tone_x1, tone_y1, tone_x2, tone_y2
                    )
                ):
                    region_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    region_mask[tone_y1:tone_y2, tone_x1:tone_x2] = 255
                elif _floating_region_is_art_sensitive(
                    image,
                    (float_x1, float_y1, float_x2, float_y2),
                    region_mask[float_y1:float_y2, float_x1:float_x2],
                ):
                    if _manga_cleaner_local_crop(
                        result,
                        region_mask,
                        img_h,
                        img_w,
                        float_x1,
                        float_y1,
                        float_x2,
                        float_y2,
                    ):
                        pass
                    elif not _external_inpaint_command_local_crop(
                        result,
                        region_mask,
                        img_h,
                        img_w,
                        float_x1,
                        float_y1,
                        float_x2,
                        float_y2,
                    ):
                        cleanup_status[constraint_id] = {
                            "cleaned": False,
                            "mode": "floating",
                            "reason": "unsafe_art_preserved",
                        }
                        continue
                elif _external_inpaint_command_local_crop(
                    result,
                    region_mask,
                    img_h,
                    img_w,
                    float_x1,
                    float_y1,
                    float_x2,
                    float_y2,
                ):
                    pass
                elif anime_model is not None:
                    _anime_lama_local_crop(
                        anime_model,
                        anime_device,
                        result,
                        region_mask,
                        img_h,
                        img_w,
                        float_x1,
                        float_y1,
                        float_x2,
                        float_y2,
                    )
                else:
                    _lama_local_crop(
                        lama_session,
                        result,
                        region_mask,
                        img_h,
                        img_w,
                        float_x1,
                        float_y1,
                        float_x2,
                        float_y2,
                    )

            restored_foreground_mask = _restore_foreground_art_outside_source_text(
                image,
                result,
                region_mask,
                source_red_coords,
                dilated,
            )
            if restored_foreground_mask is not None:
                region_mask = cv2.bitwise_and(region_mask, cv2.bitwise_not(restored_foreground_mask))
            floating_mask = cv2.bitwise_or(floating_mask, region_mask)
            final_mask = cv2.bitwise_or(final_mask, region_mask)
            cleanup_status[constraint_id] = {"cleaned": True, "mode": "floating", "reason": "cleaned"}

        # ── Residual-sweep pass (v1.1.17) ─────────────────────────────────────
        # Catch dark clusters that the constraint loop dropped (Step 6 classified
        # them as sfx/noise, or the per-constraint inpainter failed to mask).
        # Without this, source-text residue survives to the final output.
        # Evidence for the sweep: the processed constraints' red/erase boxes
        # (explicit, guarded erase intents) PLUS detector-positive ink that
        # sits right next to an already-cleaned region (dropped-fragment
        # residue). Detector-positive text FAR from any cleanup is a
        # standalone untranslated label (썸머스쿨) — fail-safe policy keeps
        # it fully intact rather than half-erased.
        near_cleanup = cv2.dilate(
            (final_mask > 0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)),
        )
        sweep_evidence = np.where(
            near_cleanup > 0, (seg_mask > 0).astype(np.uint8) * 255, 0
        ).astype(np.uint8)
        for constraint in layout_data:
            if (
                renderable_translation_ids is not None
                and int(constraint.get("id", -1)) not in renderable_translation_ids
            ):
                continue
            evidence_boxes = [constraint.get("red_box")] + list(
                constraint.get("erase_boxes") or []
            )
            for ev_box in evidence_boxes:
                if isinstance(ev_box, dict):
                    ev_vals = [ev_box.get(k) for k in ("x1", "y1", "x2", "y2")]
                else:
                    ev_vals = list(ev_box[:4]) if ev_box is not None and len(ev_box) >= 4 else []
                if len(ev_vals) != 4 or any(v is None for v in ev_vals):
                    continue
                evx1 = max(0, min(img_w, int(ev_vals[0])))
                evy1 = max(0, min(img_h, int(ev_vals[1])))
                evx2 = max(0, min(img_w, int(ev_vals[2])))
                evy2 = max(0, min(img_h, int(ev_vals[3])))
                if evx2 > evx1 and evy2 > evy1:
                    sweep_evidence[evy1:evy2, evx1:evx2] = 255
        residual_mask, residual_stats = _sweep_residual_dark_clusters(
            image,
            final_mask,
            preserved_sfx_mask=(
                sfx_preserved_mask if np.any(sfx_preserved_mask) else None
            ),
            text_evidence_mask=sweep_evidence,
        )
        if int(np.count_nonzero(residual_mask)) > 0:
            # Route each residual cluster through the flat-background fill path,
            # which is the safest fallback for dark-on-bright residue.
            cleaned_count = 0
            labels_count, res_labels, res_stats, _ = cv2.connectedComponentsWithStats(
                residual_mask, connectivity=8
            )
            for label_idx in range(1, labels_count):
                x = int(res_stats[label_idx, cv2.CC_STAT_LEFT])
                y = int(res_stats[label_idx, cv2.CC_STAT_TOP])
                w = int(res_stats[label_idx, cv2.CC_STAT_WIDTH])
                h = int(res_stats[label_idx, cv2.CC_STAT_HEIGHT])
                coords = (x, y, x + w, y + h)
                # Crop the full-size cluster mask to the ROI; _fill_*_text_mask
                # functions expect mask_roi shaped like source[y1:y2, x1:x2].
                cluster_mask_roi = (
                    res_labels[y:y + h, x:x + w] == label_idx
                ).astype(np.uint8) * 255
                # Dilate slightly so we catch halos around the glyph.
                cluster_mask_roi = cv2.dilate(
                    cluster_mask_roi,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=1,
                )
                in_container = (
                    np.count_nonzero(container_mask[y:y + h, x:x + w] > 0)
                    >= 0.6 * max(1, w * h)
                )
                if in_container:
                    # Same rationale as the unkept-promise check: LaMa's
                    # >=512px context window dwarfs a small residual cluster
                    # and reconstructs from surrounding art instead of the
                    # container's own fill (verified: destroyed
                    # new_sample_5's black-bubble interior via this exact
                    # path -- the isolated per-constraint cleanup was
                    # correct, but this sweep re-touched it afterward).
                    ring = cv2.dilate(
                        cluster_mask_roi,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
                    ) > 0
                    sample_area = ring & (cluster_mask_roi == 0) & (container_mask[y:y + h, x:x + w] > 0)
                    if int(np.count_nonzero(sample_area)) >= 20:
                        fill_color = np.median(
                            result[y:y + h, x:x + w][sample_area].reshape(-1, 3).astype(np.float32),
                            axis=0,
                        )
                        target_roi = result[y:y + h, x:x + w]
                        target_roi[cluster_mask_roi > 0] = np.clip(fill_color, 0, 255).astype(np.uint8)
                        cleaned_count += 1
                        continue
                if _fill_flat_background_text_mask(image, result, coords, cluster_mask_roi):
                    cleaned_count += 1
                elif anime_model is not None:
                    # Flat fill declined (mixed ring tones etc.) -- the model
                    # reconstructs the cluster instead of leaving marked-but-
                    # unpainted residue in the final output.
                    cluster_full = np.zeros((img_h, img_w), dtype=np.uint8)
                    cluster_full[y:y + h, x:x + w] = cluster_mask_roi
                    _anime_lama_local_crop(
                        anime_model,
                        anime_device,
                        result,
                        cluster_full,
                        img_h,
                        img_w,
                        x,
                        y,
                        x + w,
                        y + h,
                    )
                    cleaned_count += 1
            residual_stats["clusters_inpainted"] = cleaned_count
            # Treat all found residual clusters as part of floating_mask so the
            # post-loop context-tone and flat-paper cleanups see them too.
            floating_mask = cv2.bitwise_or(floating_mask, residual_mask)
            final_mask = cv2.bitwise_or(final_mask, residual_mask)
        cleanup_status["__residual_sweep__"] = residual_stats

        # Sample ring/paper tone from the pristine `image`, not `result`.
        # These two passes are a safety-net correction step: they replace
        # whatever ended up in `result` (including a bad LaMa/inpainter fill)
        # using the surrounding tone as ground truth. Sampling from `result`
        # instead let a bad fill (e.g. a white blob) poison its own
        # "correct reference" and get reinforced instead of fixed — verified
        # as a regression: new_sample_1's floating caption panel produced a
        # large white cloud blob with `result`-sampling that the original
        # `image`-sampling did not produce.
        #
        # Both passes exist for genuinely floating text (no enclosing shape,
        # so "match the surrounding tone" is the correct ground truth). A
        # traced bubble/container's interior is not floating -- its fill
        # color is defined by its OWN interior, not by whatever sits outside
        # its wall. `floating_mask` accumulates every floating constraint's
        # region_mask unconditionally (including ones that carry container
        # geometry from the step-6 dark-container tracer), so without this
        # exemption a correctly-filled dark bubble next to bright/complex art
        # (hair, screentone) would read as a high-variance "mismatched" ring
        # to `_apply_context_tone_match` -- which has no flatness/texture
        # skip, unlike `_final_flat_paper_cleanup` -- and get tone-shifted
        # toward the brighter surrounding mean. (A checkpoint trace on
        # new_sample_5's id12 showed this specific pair wasn't the active
        # cause there -- the real destroyer was the ghost-residue check,
        # see its container-margin guard below -- but the category error
        # this exemption closes is real and independent of that case.)
        floating_mask_outside_containers = cv2.bitwise_and(
            floating_mask, cv2.bitwise_not(container_mask)
        )
        _apply_context_tone_match(image, result, floating_mask_outside_containers)
        _final_flat_paper_cleanup(image, result, floating_mask_outside_containers)

        # ── Unkept-promise check ─────────────────────────────────────────
        # A cleanup path can claim an erase box in the mask while its painter
        # declines part of it (border-touching components, guarded fills).
        # Dark source ink inside final_mask that is still pixel-identical to
        # the input was promised erased and never was — erase it now.
        if anime_model is not None and np.any(final_mask):
            unpainted_same = np.all(cv2.absdiff(image, result) <= 2, axis=2)
            source_dark = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < 120
            # Evidence-gated like the residual sweep above: final_mask can
            # include a large rectangular erase box (cleanup_main_box/
            # green_box merges) that is mostly plain BACKGROUND, not glyph
            # ink. Without requiring actual Step-1 text-detector evidence
            # nearby, a correctly-dark background/panel that just sits
            # inside that rectangle reads as "unpainted ink" and gets
            # force-repainted toward the model's lighter prior (verified:
            # new_sample_4's dark UI banner whitewashed this way).
            text_evidence = cv2.dilate(
                (seg_mask > 127).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            ) > 0
            unpainted_ink = (
                (final_mask > 0)
                & unpainted_same
                & source_dark
                & (sfx_preserved_mask == 0)
                & text_evidence
            )
            promise_repairs = 0
            if int(np.count_nonzero(unpainted_ink)) >= 12:
                up_count, up_labels, up_stats, _ = cv2.connectedComponentsWithStats(
                    unpainted_ink.astype(np.uint8), connectivity=8
                )
                for up_idx in range(1, up_count):
                    if int(up_stats[up_idx, cv2.CC_STAT_AREA]) < 24:
                        continue
                    ux = int(up_stats[up_idx, cv2.CC_STAT_LEFT])
                    uy = int(up_stats[up_idx, cv2.CC_STAT_TOP])
                    uw = int(up_stats[up_idx, cv2.CC_STAT_WIDTH])
                    uh = int(up_stats[up_idx, cv2.CC_STAT_HEIGHT])
                    up_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    up_mask[uy:uy + uh, ux:ux + uw] = (
                        up_labels[uy:uy + uh, ux:ux + uw] == up_idx
                    ).astype(np.uint8) * 255
                    up_mask = cv2.dilate(
                        up_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    )
                    in_container = (
                        np.count_nonzero(container_mask[uy:uy + uh, ux:ux + uw] > 0)
                        >= 0.6 * max(1, uw * uh)
                    )
                    filled_deterministically = False
                    if in_container:
                        # Inside a traced bubble/container: LaMa's context
                        # window (>=512px, see _context_crop_bounds) dwarfs a
                        # small residual cluster and pulls in surrounding
                        # character art as context, hallucinating INTO the
                        # container instead of touching up its own fill
                        # (verified: destroyed new_sample_5's black-bubble
                        # interior). The container's own nearby pixels
                        # already carry the correct target color -- a
                        # deterministic fill is both safer and correct.
                        ring = cv2.dilate(
                            up_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                        ) > 0
                        sample_area = ring & (up_mask == 0) & (container_mask > 0)
                        if int(np.count_nonzero(sample_area)) >= 20:
                            fill_color = np.median(
                                result[sample_area].reshape(-1, 3).astype(np.float32), axis=0
                            )
                            result[up_mask > 0] = np.clip(fill_color, 0, 255).astype(np.uint8)
                            filled_deterministically = True
                    if not filled_deterministically:
                        _anime_lama_local_crop(
                            anime_model,
                            anime_device,
                            result,
                            up_mask,
                            img_h,
                            img_w,
                            ux,
                            uy,
                            ux + uw,
                            uy + uh,
                        )
                    final_mask = cv2.bitwise_or(final_mask, up_mask)
                    promise_repairs += 1
            cleanup_status["__unpainted_ink_check__"] = {
                "clusters_repainted": promise_repairs
            }

        # ── Ghost-residue self-check (page-level, all cleanup paths) ─────
        # Fills that painted glyph cores but missed anti-aliased white
        # outlines leave glyph-shaped ghosts; so can a model run whose mask
        # missed halo pixels. Every cleaned region is verified: edge density
        # inside vs around the region, re-inpainted on mismatch. SFX regions
        # are sacred and skipped.
        if anime_model is not None and np.any(final_mask):
            ghost_groups = cv2.dilate(
                (final_mask > 0).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            group_count, group_labels, group_stats, _ = cv2.connectedComponentsWithStats(
                ghost_groups, connectivity=8
            )
            ghost_repairs = 0
            for group_idx in range(1, group_count):
                gx = int(group_stats[group_idx, cv2.CC_STAT_LEFT])
                gy = int(group_stats[group_idx, cv2.CC_STAT_TOP])
                gw = int(group_stats[group_idx, cv2.CC_STAT_WIDTH])
                gh = int(group_stats[group_idx, cv2.CC_STAT_HEIGHT])
                if int(group_stats[group_idx, cv2.CC_STAT_AREA]) < 300:
                    continue
                pad = 10
                gx1 = max(0, gx - pad)
                gy1 = max(0, gy - pad)
                gx2 = min(img_w, gx + gw + pad)
                gy2 = min(img_h, gy + gh + pad)
                if np.any(
                    sfx_preserved_mask[gy1:gy2, gx1:gx2] > 0
                ):
                    continue
                group_roi_mask = np.where(
                    group_labels[gy1:gy2, gx1:gx2] == group_idx,
                    final_mask[gy1:gy2, gx1:gx2],
                    0,
                ).astype(np.uint8)
                before_count = int(np.count_nonzero(group_roi_mask))
                repaired_roi = _reinpaint_ghost_residue(
                    anime_model,
                    anime_device,
                    result,
                    (gx1, gy1, gx2, gy2),
                    group_roi_mask,
                    container_mask=container_mask,
                )
                if int(np.count_nonzero(repaired_roi)) != before_count:
                    ghost_repairs += 1
                    final_mask[gy1:gy2, gx1:gx2] = np.maximum(
                        final_mask[gy1:gy2, gx1:gx2], repaired_roi
                    )
            cleanup_status["__ghost_residue_check__"] = {"regions_reinpainted": ghost_repairs}

        out_dir = sample_path / "step_4_final"
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "inpainted_result.jpg"), result)
        cv2.imwrite(str(out_dir / "mask.png"), final_mask)
        (out_dir / "cleanup_status.json").write_text(
            json.dumps(cleanup_status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"  Inpainted {len(layout_data) - skipped_without_translation} regions"
            f" (skipped {skipped_without_translation} without renderable English,"
            f" inferred bubble cleanups {inferred_bubble_cleanups})."
        )


if __name__ == "__main__":
    run_step4_inpaint()
