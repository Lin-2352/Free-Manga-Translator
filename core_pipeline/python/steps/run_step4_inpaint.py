"""
Step 4 — Layout-Driven Inpainting
=================================

Bubble text uses ONNX LaMa on local crops clipped to eroded bubble masks.

Floating text uses the tight Step 6 text mask first. Flat/paper backgrounds are
filled directly, detailed art uses small-radius OpenCV stroke repair, and the
Anime/Manga LaMa checkpoint is reserved for masks that are safe to synthesize.
"""



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

import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Sequence

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


def _load_anime_lama_model(model_path: Path = ANIME_LAMA_PATH):
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
    model = torch.jit.load(str(model_path), map_location="cpu").to(device).eval()
    print(f"  [AnimeLaMa] floating-text inpainter: CUDA LOCKED ({model_path})")
    return model, device


def _get_lama_session(model_path: str):
    global _LAMA_SESSION
    if _LAMA_SESSION is None:
        _LAMA_SESSION = load_lama_model(model_path)
    return _LAMA_SESSION


def _get_anime_lama_model():
    global _ANIME_LAMA_MODEL, _ANIME_LAMA_DEVICE, _ANIME_LAMA_LOAD_ATTEMPTED
    if not _ANIME_LAMA_LOAD_ATTEMPTED:
        _ANIME_LAMA_MODEL, _ANIME_LAMA_DEVICE = _load_anime_lama_model()
        _ANIME_LAMA_LOAD_ATTEMPTED = True
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

        command = command_template.format(
            image=str(input_path),
            input=str(input_path),
            mask=str(mask_path),
            output=str(output_path),
        )
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(EXTERNAL_INPAINT_CWD),
            capture_output=True,
            text=True,
            timeout=float(os.getenv("MANGA_INPAINT_COMMAND_TIMEOUT", "180")),
        )
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


def _clean_white_bubble_residue(image: np.ndarray, region_mask: np.ndarray, bubble_mask: np.ndarray) -> np.ndarray:
    if bubble_mask is None or not np.any(region_mask > 0):
        return np.zeros(region_mask.shape, dtype=np.uint8)

    kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    safe_bubble = cv2.erode(bubble_mask, kernel_3, iterations=2)
    cleanup_mask = cv2.dilate(region_mask, kernel_3, iterations=2)
    cleanup_mask = cv2.bitwise_and(cleanup_mask, safe_bubble)

    if np.count_nonzero(cleanup_mask) < 20:
        return np.zeros(region_mask.shape, dtype=np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    fill_color = np.median(image[background_area], axis=0).astype(np.uint8)
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
    bright = ((gray > 214) & (hsv[:, :, 1] < 72)).astype(np.uint8) * 255
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
    bright_fraction = float(np.mean(((gray > 220) & (hsv[:, :, 1] < 80))[mask_bool]))
    if bright_fraction < 0.78:
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
    safe_bubble = cv2.erode(bubble_mask, kernel_3, iterations=2)
    if np.count_nonzero(safe_bubble > 0) < 600:
        return np.zeros(source.shape[:2], dtype=np.uint8)

    cleanup_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    if text_seg_mask is not None and text_seg_mask.shape == safe_bubble.shape:
        _, seg_bin = cv2.threshold(text_seg_mask, 127, 255, cv2.THRESH_BINARY)
        candidate_mask = cv2.bitwise_and(seg_bin, safe_bubble)
        x1, y1, x2, y2 = coords
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
        return cv2.dilate(seg_erase_roi, kernel, iterations=2)
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

    fill_color = np.median(roi[background_candidates], axis=0).astype(np.uint8)
    flat_paper = paper_fraction >= 0.72 and edge_density <= 0.20
    kernel_size = 9 if flat_paper else 5
    iterations = 2 if flat_paper else 1
    cleanup = cv2.dilate(
        (mask_roi > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        iterations=iterations,
    )
    target_roi = target[y1:y2, x1:x2]
    target_roi[cleanup > 0] = fill_color
    return True


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
        if np.count_nonzero(background) < 8:
            background = ~local_cleanup
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
    bright_fraction = float(np.mean(background_gray > 232))
    mid_fraction = float(np.mean((background_gray >= 82) & (background_gray <= 220)))
    dark_fraction = float(np.mean(background_gray < 82))
    saturation_p80 = float(np.percentile(hsv[:, :, 1][repair_background], 80))
    edge_density = float(np.mean((cv2.Canny(gray, 45, 135) > 0)[repair_background]))
    luma_std = float(np.std(background_gray))

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

    try:
        telea = cv2.inpaint(target[y1:y2, x1:x2], repair, 1.25, cv2.INPAINT_TELEA)
        navier = cv2.inpaint(target[y1:y2, x1:x2], repair, 1.0, cv2.INPAINT_NS)
    except cv2.error:
        return None

    repaired = cv2.addWeighted(telea, 0.82, navier, 0.18, 0)
    target_roi = target[y1:y2, x1:x2]
    target_roi[repair > 0] = repaired[repair > 0]
    return repair


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
    narrow_vertical = width <= 92 and height >= width * 1.55

    outlined = _outlined_floating_source_mask(source, coords, mask_roi)
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
    if repair_density < 0.025 or repair_density > 0.44:
        return None

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
    if paper_fraction >= 0.76 and mean_luma >= 176.0:
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
    if density < 0.025 or density > max_density:
        return None
    return repair


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

    return (
        dense_caption
        or compact_bright_caption
        or horizontal_bright_caption
        or (tall_caption and density >= 0.48)
    )


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


def _tight_floating_stroke_repair(
    source: np.ndarray,
    target: np.ndarray,
    seg_mask: np.ndarray,
    coords: tuple[int, int, int, int],
    anime_model,
    anime_device,
) -> np.ndarray | None:
    x1, y1, x2, y2 = coords
    mask_roi = _refined_floating_source_mask(source, seg_mask, coords)
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

    dense_smooth_mask = _dense_smooth_tone_source_text_fill(source, target, coords, mask_roi)
    if dense_smooth_mask is not None:
        return dense_smooth_mask

    pure_paper_mask = _pure_paper_source_inpaint(source, target, coords, mask_roi)
    if pure_paper_mask is not None:
        return pure_paper_mask

    smooth_gradient_mask = _smooth_gradient_source_text_fill(source, target, coords, mask_roi)
    if smooth_gradient_mask is not None:
        return smooth_gradient_mask

    mixed_dark_mask = _mixed_dark_surface_source_inpaint(source, target, coords, mask_roi)
    if mixed_dark_mask is not None:
        return mixed_dark_mask

    smooth_caption_block = _fill_smooth_tone_caption_block(source, target, coords, mask_roi)
    if smooth_caption_block is not None:
        return smooth_caption_block

    bounded_tone_mask = _fill_bounded_tone_source_strokes(source, target, coords, mask_roi)
    if bounded_tone_mask is not None:
        return bounded_tone_mask

    dark_surface_seed = mask_roi
    if roi.size:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if float(np.mean(roi_gray < 140)) >= 0.30:
            outlined_for_dark = _outlined_floating_source_mask(source, coords, mask_roi)
            if outlined_for_dark is not None:
                dark_surface_seed = outlined_for_dark
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


def run_step4_inpaint():
    print("=" * 60)
    print("  Step 4 — Layout-Driven Inpainting (v15: Art-Safe Stroke Repair)")
    print("=" * 60)

    cfg = MLConfig()
    samples_dir = sample_root_from_env(DEFAULT_SAMPLES_ROOT)
    lama_session = _get_lama_session(cfg.lama_model_path)
    anime_model, anime_device = _get_anime_lama_model()
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for sample_name, img_file in SAMPLE_MAP.items():
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
        img_h, img_w = image.shape[:2]
        seg_mask = cv2.imread(str(seg_mask_path), cv2.IMREAD_GRAYSCALE)

        with open(layout_path, "r", encoding="utf-8") as layout_file:
            layout_data = json.load(layout_file)
        renderable_translation_ids = _load_renderable_translation_ids(sample_path)

        result = image.copy()
        final_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        floating_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        skipped_without_translation = 0
        inferred_bubble_cleanups = 0
        cleanup_status = {}

        for constraint in layout_data:
            if (
                renderable_translation_ids is not None
                and int(constraint.get("id", -1)) not in renderable_translation_ids
            ):
                skipped_without_translation += 1
                continue

            constraint_id = str(int(constraint.get("id", -1)))
            red_box = [int(value) for value in constraint["red_box"]]
            force_bubble_cleanup = bool(constraint.get("force_bubble_cleanup", False))
            is_bubble = constraint.get("bubble_idx", -1) != -1 or force_bubble_cleanup
            precise_layout_mask = constraint.get("mask_mode") == "svg_text"
            cleanup_box = red_box
            x1, y1, x2, y2 = cleanup_box

            roi_seg = seg_mask[y1:y2, x1:x2]
            _, roi_seg_bin = cv2.threshold(roi_seg, 127, 255, cv2.THRESH_BINARY)

            if force_bubble_cleanup:
                roi_mask = _extract_dark_text_strokes(
                    image,
                    (x1, y1, x2, y2),
                    constraint.get("source_colors") or None,
                )
            elif np.count_nonzero(roi_seg_bin) < 20:
                roi_mask = _extract_text_strokes(image, (x1, y1, x2, y2), is_bubble=is_bubble)
            else:
                roi_mask = roi_seg_bin

            dilated = cv2.dilate(roi_mask, kernel_3, iterations=2)

            if is_bubble:
                bubble_mask_path = detect_dir / f"bubble_{constraint['bubble_idx']}.png"
                bubble_mask = None
                if bubble_mask_path.exists():
                    bubble_mask = cv2.imread(str(bubble_mask_path), cv2.IMREAD_GRAYSCALE)
                    roi_bubble_mask = bubble_mask[y1:y2, x1:x2]
                    eroded_bubble = cv2.erode(roi_bubble_mask, kernel_3, iterations=5)
                    dilated = cv2.bitwise_and(dilated, eroded_bubble)

                region_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                region_mask[y1:y2, x1:x2] = dilated

                _lama_local_crop(lama_session, result, region_mask, img_h, img_w, x1, y1, x2, y2)
                cleanup_input = region_mask
                if not precise_layout_mask:
                    box_cleanup_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    cleanup_pad = 2
                    cleanup_x1 = max(0, x1 - cleanup_pad)
                    cleanup_y1 = max(0, y1 - cleanup_pad)
                    cleanup_x2 = min(img_w, x2 + cleanup_pad)
                    cleanup_y2 = min(img_h, y2 + cleanup_pad)
                    box_cleanup_mask[cleanup_y1:cleanup_y2, cleanup_x1:cleanup_x2] = 255
                    cleanup_input = cv2.bitwise_or(cleanup_input, box_cleanup_mask)
                cleanup_mask = _clean_white_bubble_residue(result, cleanup_input, bubble_mask)
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
                    cleanup_main_box = [
                        max(0, min(x1, gx1)),
                        max(0, min(y1, gy1)),
                        min(img_w, max(x2, gx2)),
                        min(img_h, max(y2, gy2)),
                    ]
            explicit_erase_boxes = constraint.get("erase_boxes") or []
            raw_erase_boxes = (
                [cleanup_main_box, *explicit_erase_boxes]
                if explicit_erase_boxes
                else []
            )
            for raw_box in raw_erase_boxes:
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
                ex1 = max(0, min(img_w, ex1 - 1))
                ey1 = max(0, min(img_h, ey1 - 1))
                ex2 = max(0, min(img_w, ex2 + 1))
                ey2 = max(0, min(img_h, ey2 + 1))
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
                erase_snapshot = result[union_y1:union_y2, union_x1:union_x2].copy()
                prefilled_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                unsafe_skipped = False
                for ex1, ey1, ex2, ey2 in erase_boxes:
                    erase_roi = _floating_erase_mask_for_box(
                        image,
                        seg_mask,
                        (ex1, ey1, ex2, ey2),
                        kernel_3,
                    )
                    if np.count_nonzero(erase_roi) >= 6:
                        tight_repair_mask = _tight_floating_stroke_repair(
                            image,
                            result,
                            seg_mask,
                            (ex1, ey1, ex2, ey2),
                            anime_model,
                            anime_device,
                        )
                        if tight_repair_mask is not None:
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
                legacy_caption_repair = None
                if (
                    not constraint.get("erase_boxes")
                    and not use_layout_stroke_mask
                    and _legacy_bright_caption_candidate(image, (x1, y1, x2, y2), floating_roi)
                ):
                    legacy_repair_box = _union_repair_box(
                        (x1, y1, x2, y2),
                        constraint.get("green_box"),
                        image.shape,
                    )
                    legacy_caption_repair = _legacy_full_box_anime_repair(
                        image,
                        result,
                        legacy_repair_box,
                        anime_model,
                        anime_device,
                        pad=_compact_caption_repair_pad(legacy_repair_box),
                    )
                if legacy_caption_repair is not None:
                    floating_mask = cv2.bitwise_or(floating_mask, legacy_caption_repair)
                    final_mask = cv2.bitwise_or(final_mask, legacy_caption_repair)
                    cleanup_status[constraint_id] = {
                        "cleaned": True,
                        "mode": "floating",
                        "reason": "legacy_bright_caption_anime",
                    }
                    continue
                prefer_box_cleanup = False
                tight_repair_mask = _tight_floating_stroke_repair(
                    image,
                    result,
                    seg_mask,
                    (x1, y1, x2, y2),
                    anime_model,
                    anime_device,
                )
                if tight_repair_mask is not None:
                    tight_repair_density = float(np.count_nonzero(tight_repair_mask > 0)) / float(
                        max(1, (x2 - x1) * (y2 - y1))
                    )
                    if (
                        not prefer_box_cleanup
                        or constraint.get("erase_boxes")
                        or tight_repair_density <= 0.78
                    ):
                        region_mask[y1:y2, x1:x2] = tight_repair_mask
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

            floating_mask = cv2.bitwise_or(floating_mask, region_mask)
            final_mask = cv2.bitwise_or(final_mask, region_mask)
            cleanup_status[constraint_id] = {"cleaned": True, "mode": "floating", "reason": "cleaned"}

        _apply_context_tone_match(image, result, floating_mask)
        _final_flat_paper_cleanup(image, result, floating_mask)

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
