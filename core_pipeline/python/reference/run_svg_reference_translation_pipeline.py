"""
Generate exact modern CJK translation artifacts from Pepper&Carrot SVG sources.

This is a deterministic modern benchmark path: the source page, source SVG
text layer, and official English SVG text layer come from the same CC BY 4.0
project. The generated artifacts are consumed by the normal Step 4 and Step 8
scripts, so erasure/typesetting are still tested visually.
"""
from __future__ import annotations

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

import argparse
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from pipeline_paths import DEFAULT_SAMPLES_ROOT, PROJECT_ROOT, sample_root_from_env


SAMPLES_ROOT = sample_root_from_env(DEFAULT_SAMPLES_ROOT)
LANG_PACK_URL = "https://www.peppercarrot.com/0_sources/ep01_Potion-of-Flight/zip/ep01_Potion-of-Flight_lang-pack.zip"
LANG_PACK_ZIP = PROJECT_ROOT / "quality_reports" / "ep01_Potion-of-Flight_lang-pack.zip"
LANG_PACK_DIR = PROJECT_ROOT / "quality_reports" / "ep01_lang_pack"
SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


def _download_lang_pack() -> None:
    if LANG_PACK_DIR.exists() and (LANG_PACK_DIR / "lang" / "en" / "E01P01.svg").exists():
        return
    LANG_PACK_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if not LANG_PACK_ZIP.exists():
        request = Request(LANG_PACK_URL, headers={"User-Agent": "TranslatorPipelineModernValidation/1.0"})
        with urlopen(request, timeout=120) as response:
            LANG_PACK_ZIP.write_bytes(response.read())
    if LANG_PACK_DIR.exists():
        import shutil

        shutil.rmtree(LANG_PACK_DIR)
    with zipfile.ZipFile(LANG_PACK_ZIP) as archive:
        archive.extractall(LANG_PACK_DIR)


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [dict(item) for item in manifest.get("samples", [])]


def _parse_transform(transform: str) -> tuple[float, float, float, float, float, float]:
    matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    if not transform:
        return matrix
    for name, raw_args in re.findall(r"(matrix|translate|scale)\(([^)]*)\)", transform):
        values = [float(value) for value in re.split(r"[ ,]+", raw_args.strip()) if value]
        if name == "matrix" and len(values) >= 6:
            next_matrix = tuple(values[:6])
        elif name == "translate":
            next_matrix = (1.0, 0.0, 0.0, 1.0, values[0], values[1] if len(values) > 1 else 0.0)
        elif name == "scale":
            sy = values[1] if len(values) > 1 else values[0]
            next_matrix = (values[0], 0.0, 0.0, sy, 0.0, 0.0)
        else:
            continue
        a, b, c, d, e, f = matrix
        A, B, C, D, E, F = next_matrix
        matrix = (
            a * A + c * B,
            b * A + d * B,
            a * C + c * D,
            b * C + d * D,
            a * E + c * F + e,
            b * E + d * F + f,
        )
    return matrix


def _apply_matrix(matrix: tuple[float, float, float, float, float, float], x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def _style_value(style: str, key: str) -> str:
    match = re.search(rf"(?:^|;){re.escape(key)}:([^;]+)", style or "")
    return match.group(1).strip() if match else ""


def _hex_colors_from_style(style: str) -> list[tuple[int, int, int]]:
    colors = []
    for key in ("fill", "stroke"):
        value = _style_value(style, key)
        if re.fullmatch(r"#[0-9a-fA-F]{6}", value):
            colors.append((int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)))
    return colors


def _extract_svg_items(svg_path: Path) -> tuple[float, float, list[dict[str, Any]]]:
    root = ET.parse(svg_path).getroot()
    svg_width = float(root.get("width", "0"))
    svg_height = float(root.get("height", "0"))
    items: list[dict[str, Any]] = []
    for flow_root in root.findall(".//svg:flowRoot", SVG_NS):
        paras = []
        colors = _hex_colors_from_style(flow_root.get("style", ""))
        for para in flow_root.findall(".//svg:flowPara", SVG_NS):
            text = "".join(para.itertext()).strip()
            if text and text != "\xa0":
                paras.append(text)
            colors.extend(_hex_colors_from_style(para.get("style", "")))
        text = " ".join(paras).strip()
        if not text:
            continue
        rect = flow_root.find(".//svg:flowRegion/svg:rect", SVG_NS)
        if rect is None:
            continue
        x = float(rect.get("x", "0"))
        y = float(rect.get("y", "0"))
        width = float(rect.get("width", "0"))
        height = float(rect.get("height", "0"))
        transform = _parse_transform(flow_root.get("transform", ""))
        points = [
            _apply_matrix(transform, x, y),
            _apply_matrix(transform, x + width, y),
            _apply_matrix(transform, x + width, y + height),
            _apply_matrix(transform, x, y + height),
        ]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        items.append(
            {
                "text": text,
                "svg_box": [min(xs), min(ys), max(xs), max(ys)],
                "colors": list(dict.fromkeys(colors)),
            }
        )
    return svg_width, svg_height, items


def _scale_box(box: list[float], svg_width: float, svg_height: float, image_width: int, image_height: int) -> list[int]:
    sx = image_width / max(1.0, svg_width)
    sy = image_height / max(1.0, svg_height)
    return [
        max(0, min(image_width - 1, int(math.floor(box[0] * sx)))),
        max(0, min(image_height - 1, int(math.floor(box[1] * sy)))),
        max(0, min(image_width, int(math.ceil(box[2] * sx)))),
        max(0, min(image_height, int(math.ceil(box[3] * sy)))),
    ]


def _expand_box(box: list[int], image_width: int, image_height: int, pad_x: int, pad_y: int) -> list[int]:
    return [
        max(0, box[0] - pad_x),
        max(0, box[1] - pad_y),
        min(image_width, box[2] + pad_x),
        min(image_height, box[3] + pad_y),
    ]


def _union_box(boxes: list[list[int]], image_width: int, image_height: int) -> list[int]:
    valid = [box for box in boxes if box and box[2] > box[0] and box[3] > box[1]]
    if not valid:
        return [0, 0, image_width, image_height]
    return [
        max(0, min(box[0] for box in valid)),
        max(0, min(box[1] for box in valid)),
        min(image_width, max(box[2] for box in valid)),
        min(image_height, max(box[3] for box in valid)),
    ]


def _inset_box(box: list[int], image_width: int, image_height: int, inset: int) -> list[int]:
    return [
        max(0, min(image_width, box[0] + inset)),
        max(0, min(image_height, box[1] + inset)),
        max(0, min(image_width, box[2] - inset)),
        max(0, min(image_height, box[3] - inset)),
    ]


def _intersect_box(left: list[int], right: list[int]) -> list[int] | None:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _mask_bounds(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _largest_mask_polygon(mask: np.ndarray) -> list[list[int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 10:
        return []
    epsilon = max(1.5, cv2.arcLength(contour, True) * 0.006)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return [[int(x), int(y)] for x, y in approx]


def _safe_inner_bubble_mask(bubble_mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    safe = cv2.erode((bubble_mask > 0).astype(np.uint8) * 255, kernel, iterations=1)
    if np.count_nonzero(safe) < max(80, int(np.count_nonzero(bubble_mask > 0) * 0.40)):
        return (bubble_mask > 0).astype(np.uint8) * 255
    return safe


def _box_mask_overlap_ratio(box: list[int], mask: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi > 0)) / float(max(1, (x2 - x1) * (y2 - y1)))


def _select_bubble_mask(hint_box: list[int], bubble_masks: list[np.ndarray]) -> np.ndarray | None:
    best_mask = None
    best_score = 0.0
    hint_center = ((hint_box[0] + hint_box[2]) / 2.0, (hint_box[1] + hint_box[3]) / 2.0)
    for bubble_mask in bubble_masks:
        if bubble_mask is None:
            continue
        bounds = _mask_bounds(bubble_mask)
        if bounds is None:
            continue
        overlap = _box_mask_overlap_ratio(hint_box, bubble_mask)
        contains_center = (
            bounds[0] <= hint_center[0] <= bounds[2]
            and bounds[1] <= hint_center[1] <= bounds[3]
        )
        score = overlap + (0.20 if contains_center else 0.0)
        if score > best_score:
            best_score = score
            best_mask = bubble_mask
    return best_mask if best_score >= 0.28 else None


def _is_sfx(english_text: str) -> bool:
    compact = re.sub(r"[^A-Za-z]", "", english_text)
    if not compact:
        return False
    return english_text.upper() == english_text and len(compact) <= 16


def _mask_from_colors(crop: np.ndarray, colors: list[tuple[int, int, int]]) -> np.ndarray:
    if crop.size == 0:
        return np.zeros(crop.shape[:2], dtype=np.uint8)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.int32)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    for color in colors:
        target = np.array(color, dtype=np.int32)
        distance = np.sqrt(np.sum((rgb - target) ** 2, axis=2))
        mask[distance < 80] = 255
    return mask


def _mask_from_box(
    image: np.ndarray,
    box: list[int],
    colors: list[tuple[int, int, int]],
    prefer_color_only: bool = False,
) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)[1]
    color_mask = _mask_from_colors(crop, colors)
    if prefer_color_only and np.count_nonzero(color_mask) >= 10:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.dilate(color_mask, kernel, iterations=1)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    saturated = np.where((saturation > 65) & (value > 80), 255, 0).astype(np.uint8)
    mask = cv2.bitwise_or(dark, color_mask)
    if colors:
        mask = cv2.bitwise_or(mask, color_mask)
    else:
        mask = cv2.bitwise_or(mask, saturated)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(mask, kernel, iterations=1)


def _mask_bounds_in_global(mask: np.ndarray, box: list[int]) -> list[int] | None:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, y1, _, _ = box
    return [
        int(x1 + xs.min()),
        int(y1 + ys.min()),
        int(x1 + xs.max() + 1),
        int(y1 + ys.max() + 1),
    ]


def _filter_sfx_components(mask: np.ndarray) -> np.ndarray:
    if np.count_nonzero(mask) == 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, num_labels):
        x, y, width, height, area = stats[label]
        if area < 8:
            continue
        if area > 3500:
            continue
        if width > 180 or height > 180:
            continue
        if width < 2 or height < 2:
            continue
        filtered[labels == label] = 255
    if np.count_nonzero(filtered) < 8:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(filtered, kernel, iterations=1)


def _filter_dialogue_text_components(mask: np.ndarray) -> np.ndarray:
    if np.count_nonzero(mask) == 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros(mask.shape, dtype=np.uint8)
    height, width = mask.shape[:2]
    for label in range(1, num_labels):
        x, y, component_width, component_height, area = stats[label]
        if area < 4:
            continue
        if area > max(2200, int(width * height * 0.18)):
            continue
        if component_width > int(width * 0.78) and component_height <= 8:
            continue
        if component_height > int(height * 0.78) and component_width <= 8:
            continue
        if (
            (x + component_width >= width - 1 or y + component_height >= height - 1)
            and area > 20
            and (component_width > 6 or component_height > 6)
        ):
            continue
        if y > int(height * 0.70) and (area < 90 or (component_height <= 8 and component_width > 10)):
            continue
        if (
            component_height > max(34, int(height * 0.32))
            and area > 420
            and component_width > 18
        ):
            continue
        filtered[labels == label] = 255
    if np.count_nonzero(filtered) < 8:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(filtered, kernel, iterations=1)


def _actual_text_box(
    image: np.ndarray,
    hint_box: list[int],
    colors: list[tuple[int, int, int]],
    image_width: int,
    image_height: int,
    *,
    pad_x: int,
    pad_y: int,
    sfx: bool = False,
) -> list[int]:
    search_box = _expand_box(hint_box, image_width, image_height, pad_x, pad_y)
    mask = _mask_from_box(
        image,
        search_box,
        colors,
        prefer_color_only=bool(colors),
    )
    mask = _filter_sfx_components(mask) if sfx else _filter_dialogue_text_components(mask)
    actual_bounds = _mask_bounds_in_global(mask, search_box)
    if actual_bounds is None:
        return hint_box
    return _expand_box(actual_bounds, image_width, image_height, 4 if sfx else 6, 4 if sfx else 6)


def _bubble_mask_and_box(
    image: np.ndarray,
    hint_box: list[int],
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, list[int]]:
    def fallback_mask() -> tuple[np.ndarray, list[int]]:
        fallback_box = _expand_box(hint_box, image_width, image_height, 24, 18)
        fallback = np.zeros((image_height, image_width), dtype=np.uint8)
        fallback[fallback_box[1]:fallback_box[3], fallback_box[0]:fallback_box[2]] = 255
        return fallback, fallback_box

    search_box = _expand_box(hint_box, image_width, image_height, 42, 32)
    x1, y1, x2, y2 = search_box
    crop = image[y1:y2, x1:x2]
    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    if crop.size == 0:
        return fallback_mask()

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pale = (((gray > 135) & (hsv[:, :, 1] < 120)) | ((gray > 165) & (hsv[:, :, 1] < 155))).astype(np.uint8) * 255
    pale = cv2.morphologyEx(
        pale,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=1,
    )
    pale = cv2.dilate(pale, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    hint_center = ((hint_box[0] + hint_box[2]) / 2 - x1, (hint_box[1] + hint_box[3]) / 2 - y1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pale, connectivity=8)
    best_label = 0
    best_score = -1e18
    for label in range(1, num_labels):
        cx, cy = centroids[label]
        left, top, width, height, area = stats[label]
        if area < 260:
            continue
        if area > crop.shape[0] * crop.shape[1] * 0.72:
            continue
        distance = math.hypot(cx - hint_center[0], cy - hint_center[1])
        contains_hint = (
            left <= hint_center[0] <= left + width
            and top <= hint_center[1] <= top + height
        )
        score = area * (1.8 if contains_hint else 1.0) - distance * 45.0
        if score > best_score:
            best_score = score
            best_label = label

    if best_label == 0:
        return fallback_mask()

    component = (labels == best_label).astype(np.uint8) * 255
    bounds = _mask_bounds_in_global(component, search_box)
    if bounds is None:
        return fallback_mask()

    bubble_box = _expand_box(bounds, image_width, image_height, 4, 4)
    hint_area = max(1, (hint_box[2] - hint_box[0]) * (hint_box[3] - hint_box[1]))
    bubble_area = max(1, (bubble_box[2] - bubble_box[0]) * (bubble_box[3] - bubble_box[1]))
    overlap_x1 = max(hint_box[0], bubble_box[0])
    overlap_y1 = max(hint_box[1], bubble_box[1])
    overlap_x2 = min(hint_box[2], bubble_box[2])
    overlap_y2 = min(hint_box[3], bubble_box[3])
    overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
    overlap_ratio = overlap_area / float(hint_area)
    if bubble_area < hint_area * 0.75 or bubble_area > hint_area * 3.6 or overlap_ratio < 0.35:
        return fallback_mask()

    full_mask[y1:y2, x1:x2] = component
    return full_mask, bubble_box


def _modern_item_records(
    sample: dict[str, Any],
    image: np.ndarray,
    bubble_masks: list[np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    source_url = sample["source_url"]
    match = re.search(r"/([a-z]{2})_Pepper-and-Carrot_by-David-Revoy_(E\d+P\d+)\.jpg$", source_url)
    if not match:
        raise ValueError(f"Cannot parse source URL: {source_url}")
    lang_code, page_code = match.groups()
    source_svg = LANG_PACK_DIR / "lang" / lang_code / f"{page_code}.svg"
    english_svg = LANG_PACK_DIR / "lang" / "en" / f"{page_code}.svg"
    if not source_svg.exists() or not english_svg.exists():
        raise FileNotFoundError(f"Missing SVG pair for {sample['sample_name']}: {source_svg}, {english_svg}")

    source_w, source_h, source_items = _extract_svg_items(source_svg)
    english_w, english_h, english_items = _extract_svg_items(english_svg)
    image_height, image_width = image.shape[:2]
    reference_image = cv2.imread(str(SAMPLES_ROOT / sample["sample_name"] / sample.get("english_reference_file", "english_reference.jpg")))
    records = []
    for index, (source_item, english_item) in enumerate(zip(source_items, english_items)):
        english_text = english_item["text"].strip()
        source_text = source_item["text"].strip()
        if not source_text or not english_text or not re.search(r"[A-Za-z]", english_text):
            continue
        source_hint_box = _scale_box(source_item["svg_box"], source_w, source_h, image_width, image_height)
        english_box = _scale_box(english_item["svg_box"], english_w, english_h, image_width, image_height)
        if source_hint_box[2] <= source_hint_box[0] or source_hint_box[3] <= source_hint_box[1]:
            continue
        sfx = _is_sfx(english_text)
        source_box = _actual_text_box(
            image,
            source_hint_box,
            source_item.get("colors", []),
            image_width,
            image_height,
            pad_x=26 if sfx else 4,
            pad_y=22 if sfx else 4,
            sfx=sfx,
        )
        if sfx:
            if reference_image is not None and reference_image.shape[:2] == image.shape[:2]:
                reference_text_box = _actual_text_box(
                    reference_image,
                    english_box,
                    english_item.get("colors", []),
                    image_width,
                    image_height,
                    pad_x=28,
                    pad_y=24,
                    sfx=True,
                )
                reference_patch_box = _expand_box(
                    _union_box([source_box, reference_text_box], image_width, image_height),
                    image_width,
                    image_height,
                    8,
                    8,
                )
            else:
                reference_patch_box = _expand_box(
                    _union_box([source_box, english_box], image_width, image_height),
                    image_width,
                    image_height,
                    18,
                    14,
                )
            green_box = reference_patch_box
            region_type = "sfx"
            bubble_idx = -1
            bubble_box = None
            bubble_mask = None
        else:
            source_box = _actual_text_box(
                image,
                source_hint_box,
                source_item.get("colors", []),
                image_width,
                image_height,
                pad_x=20,
                pad_y=18,
                sfx=False,
            )
            reference_patch_box = _expand_box(
                _union_box([source_box, english_box], image_width, image_height),
                image_width,
                image_height,
                18,
                14,
            )
            bubble_mask = _select_bubble_mask(source_hint_box, bubble_masks or [])
            if bubble_mask is None:
                bubble_mask, bubble_box = _bubble_mask_and_box(
                    image,
                    source_hint_box,
                    image_width,
                    image_height,
                )
            else:
                bubble_box = _mask_bounds(bubble_mask) or _expand_box(source_hint_box, image_width, image_height, 24, 18)
            green_mask = _safe_inner_bubble_mask(bubble_mask)
            bubble_inner_box = _mask_bounds(green_mask) or _inset_box(bubble_box, image_width, image_height, 6)
            clipped_source = _intersect_box(source_box, bubble_box) or _intersect_box(source_hint_box, bubble_box)
            if clipped_source is not None:
                source_box = clipped_source
            green_box = bubble_inner_box
            reference_patch_box = green_box
            region_type = "bubble"
            bubble_idx = len([record for record in records if record["bubble_idx"] != -1])
            green_polygon = _largest_mask_polygon(green_mask)
        records.append(
            {
                "id": len(records),
                "source_text": source_text,
                "english_text": english_text,
                "red_box": source_box,
                "green_box": green_box,
                "english_box": english_box,
                "reference_patch_box": reference_patch_box if sfx else None,
                "region_type": region_type,
                "bubble_idx": bubble_idx,
                "bubble_box": bubble_box,
                "bubble_mask": bubble_mask,
                "green_mask": locals().get("green_mask") if not sfx else None,
                "green_polygon": locals().get("green_polygon") if not sfx else None,
                "colors": source_item.get("colors", []),
            }
        )
    return records


def _write_artifacts(sample: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, int]:
    sample_path = SAMPLES_ROOT / sample["sample_name"]
    image = cv2.imread(str(sample_path / sample.get("input_file", "input.jpg")))
    if image is None:
        raise FileNotFoundError(sample_path / sample.get("input_file", "input.jpg"))
    image_height, image_width = image.shape[:2]

    detect_dir = sample_path / "step_1_detect"
    step5_dir = sample_path / "step_5_ocr"
    step6_dir = sample_path / "step_6_layout"
    step7_dir = sample_path / "step_7_translate"
    for directory in [detect_dir, step5_dir, step6_dir, step7_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    for path in detect_dir.glob("bubble_*.png"):
        path.unlink()

    seg_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    detection_boxes = []
    ocr_rows = []
    constraints = []
    translations = []
    preserved_sfx = []
    reference_patches = []
    debug = image.copy()

    for record in records:
        rb = record["red_box"]
        gb = record["green_box"]
        x1, y1, x2, y2 = rb
        reference_patches.append(
            {
                "id": record["id"],
                "source_text": record["source_text"],
                "english_text": record["english_text"],
                "region_type": record["region_type"],
                "reference_patch_box": record.get("reference_patch_box") or gb,
            }
        )
        if record["region_type"] == "sfx":
            preserved_sfx.append(
                {
                    "id": record["id"],
                    "source_text": record["source_text"],
                    "english_text": record["english_text"],
                    "source_box": rb,
                    "reference_patch_box": record.get("reference_patch_box") or gb,
                    "reason": "decorative_sfx_reference_patch_preserves_art",
                }
            )
            continue

        item_mask = _mask_from_box(
            image,
            rb,
            record.get("colors", []),
            prefer_color_only=record["region_type"] == "sfx",
        )
        item_mask = _filter_dialogue_text_components(item_mask)
        seg_mask[y1:y2, x1:x2] = cv2.bitwise_or(seg_mask[y1:y2, x1:x2], item_mask)
        detection_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

        if record["bubble_idx"] != -1:
            bubble_mask = record.get("bubble_mask")
            if bubble_mask is None:
                bubble_mask = np.zeros((image_height, image_width), dtype=np.uint8)
                bubble_box = record.get("bubble_box") or gb
                bubble_mask[bubble_box[1]:bubble_box[3], bubble_box[0]:bubble_box[2]] = 255
            cv2.imwrite(str(detect_dir / f"bubble_{record['bubble_idx']}.png"), bubble_mask)

        fallback_poly = [[gb[0], gb[1]], [gb[2], gb[1]], [gb[2], gb[3]], [gb[0], gb[3]]]
        poly = record.get("green_polygon") or fallback_poly
        route = "bubble_dialogue" if record["bubble_idx"] != -1 else "floating_dialogue"
        ocr_rows.append(
            {
                "id": record["id"],
                "text": record["source_text"],
                "box": {"x1": rb[0], "y1": rb[1], "x2": rb[2], "y2": rb[3]},
                "green_box": {"x1": gb[0], "y1": gb[1], "x2": gb[2], "y2": gb[3]},
                "green_polygon": poly,
                "bubble_box": record.get("bubble_box"),
                "route": route,
                "bubble_idx": record["bubble_idx"],
                "mask_mode": "svg_text",
                "fallback_source": "peppercarrot_svg",
                "force_bubble_cleanup": record["bubble_idx"] != -1,
                "full_box_cleanup": record["bubble_idx"] != -1,
                "source_colors": record.get("colors", []),
            }
        )
        constraints.append(
            {
                "id": record["id"],
                "text": record["source_text"],
                "red_box": rb,
                "green_box": gb,
                "green_polygon": poly,
                "bubble_box": record.get("bubble_box"),
                "bubble_idx": record["bubble_idx"],
                "mask_mode": "svg_text",
                "route": route,
                "semantic_role": record["region_type"],
                "fallback_source": "peppercarrot_svg",
                "force_bubble_cleanup": record["bubble_idx"] != -1,
                "full_box_cleanup": record["bubble_idx"] != -1,
                "reference_patch_box": record.get("reference_patch_box") or gb,
                "reference_restored": True,
                "source_colors": record.get("colors", []),
            }
        )
        translations.append(
            {
                "id": record["id"],
                "box": {"x1": rb[0], "y1": rb[1], "x2": rb[2], "y2": rb[3]},
                "jp_text": record["source_text"],
                "en_text": record["english_text"],
                "provider": "peppercarrot_official_svg_reference",
            }
        )
        if record["bubble_idx"] != -1:
            bubble_mask = record.get("bubble_mask")
            if bubble_mask is not None:
                contours, _ = cv2.findContours(
                    bubble_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(debug, contours, -1, (255, 0, 0), 2)
            else:
                bubble_box = record.get("bubble_box") or gb
                cv2.rectangle(debug, (bubble_box[0], bubble_box[1]), (bubble_box[2], bubble_box[3]), (255, 0, 0), 2)
        green_mask = record.get("green_mask")
        if green_mask is not None:
            contours, _ = cv2.findContours(
                green_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(debug, contours, -1, (0, 255, 0), 2)
        elif poly and len(poly) >= 3:
            cv2.polylines(debug, [np.array(poly, np.int32).reshape((-1, 1, 2))], True, (0, 255, 0), 1)
        else:
            cv2.rectangle(debug, (gb[0], gb[1]), (gb[2], gb[3]), (0, 255, 0), 2)
        cv2.rectangle(debug, (rb[0], rb[1]), (rb[2], rb[3]), (0, 0, 255), 2)

    cv2.imwrite(str(detect_dir / "seg_mask.png"), seg_mask)
    (detect_dir / "detections.json").write_text(
        json.dumps({"boxes": detection_boxes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (detect_dir / "semantic_detections.json").write_text(
        json.dumps({"regions": []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (step6_dir / "preserved_sfx.json").write_text(
        json.dumps(preserved_sfx, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (step6_dir / "reference_patches.json").write_text(
        json.dumps(reference_patches, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (step5_dir / "ocr_results.json").write_text(json.dumps(ocr_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (step6_dir / "layout_constraints.json").write_text(
        json.dumps(constraints, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cv2.imwrite(str(step6_dir / "debug_layout_boxes.jpg"), debug)
    (step7_dir / "translation_results.json").write_text(
        json.dumps(translations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"translated_regions": len(constraints), "preserved_sfx": len(preserved_sfx)}


def run_svg_reference_pipeline(manifest_path: Path, sample_names: set[str] | None = None) -> dict[str, Any]:
    _download_lang_pack()
    bubble_cfg = None
    bubble_model = None
    bubble_device = None
    detect_bubbles_fn = None
    try:
        from ml_region_lib import MLConfig, detect_bubbles, load_bubble_model

        bubble_cfg = MLConfig(
            bubble_model_path="models/manga109_bubble/best.pt",
            bubble_confidence=0.25,
        )
        bubble_model, bubble_device = load_bubble_model(bubble_cfg.bubble_model_path)
        detect_bubbles_fn = detect_bubbles
    except Exception as error:
        print(f"[modern bubble warn] falling back to image heuristic: {str(error)[:160]}")

    summary: dict[str, Any] = {}
    for sample in _load_manifest(manifest_path):
        sample_name = sample["sample_name"]
        if sample_names and sample_name not in sample_names:
            continue
        image = cv2.imread(str(SAMPLES_ROOT / sample_name / sample.get("input_file", "input.jpg")))
        if image is None:
            raise FileNotFoundError(SAMPLES_ROOT / sample_name / sample.get("input_file", "input.jpg"))
        bubble_masks: list[np.ndarray] = []
        if detect_bubbles_fn is not None and bubble_model is not None and bubble_device is not None and bubble_cfg is not None:
            try:
                bubble_masks = detect_bubbles_fn(bubble_model, bubble_device, image, bubble_cfg)
            except Exception as error:
                print(f"[modern bubble warn] {sample_name}: model bubble detection failed, using fallback: {str(error)[:160]}")
                bubble_masks = []
        records = _modern_item_records(sample, image, bubble_masks)
        if not records:
            raise RuntimeError(f"No SVG records produced for {sample_name}")
        counts = _write_artifacts(sample, records)
        summary[sample_name] = {"regions": len(records), **counts}
        print(
            f"{sample_name}: wrote {counts['translated_regions']} SVG-backed translated regions; "
            f"preserved {counts['preserved_sfx']} decorative SFX"
        )
    report_dir = PROJECT_ROOT / "quality_reports" / "svg_reference"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{manifest_path.stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate modern sample Step 5/6/7 artifacts from SVG sources.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "modern_cjk_samples_manifest.json")
    parser.add_argument("--sample", action="append", help="Optional sample to process; repeatable.")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    run_svg_reference_pipeline(args.manifest, set(args.sample or []) or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
