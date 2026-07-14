"""
Step 6 — Layout Export (v4: Strict Filtering + 3-Color Debug)
=============================================================
Reads Step 5 OCR results, applies strict linguistic filtering to remove
SFX, standalone numbers, and English text, then exports layout constraints
and a clean 3-color debug image:

  Blue  (255, 0, 0)  — Speech bubble boundaries (absolute wall)
  Red   (0, 0, 255)  — Source text erasure zone (tight ink bounding box)
  Green (0, 255, 0)  — Typesetting layout zone (polygon or expanded box)

No orange, no yellow, no collision indicators.
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
# torch must be imported before cv2 in this process -- see run_step5_ocr.py
# for the full explanation (verified import-order segfault reproduction).
import torch  # noqa: F401  (import-order guard, see comment above)
import json
import os
import re
import cv2
import numpy as np
from collections import Counter
from pathlib import Path
from diagnostic_logger import write_diagnostic_event
from ml_region_lib import SAMPLE_MAP, classify_text_by_content
from pipeline_paths import DEFAULT_SAMPLES_ROOT, sample_root_from_env


CREDIT_MARKERS = ("原作", "作画", "漫画", "監修", "キャラクターデザイン")
DEVICE_LABEL_CHARS = set("0123456789０１２３４５６７８９一二三四五六七八九十零〇")


def _coerce_box4(value) -> list[int] | None:
    if isinstance(value, dict):
        for key in ("box", "bbox", "red_box", "green_box", "coords"):
            nested = value.get(key)
            if nested is not None:
                return _coerce_box4(nested)
        if all(key in value for key in ("x1", "y1", "x2", "y2")):
            value = [value["x1"], value["y1"], value["x2"], value["y2"]]
        elif all(key in value for key in ("left", "top", "right", "bottom")):
            value = [value["left"], value["top"], value["right"], value["bottom"]]
        else:
            return None
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        return None
    values = list(value)
    if len(values) < 4:
        return None
    try:
        return [int(round(float(item))) for item in values[:4]]
    except (TypeError, ValueError):
        return None


def _draw_recovered_fragment_box(image: np.ndarray, box: list[int]) -> None:
    ex1, ey1, ex2, ey2 = box
    if ex2 <= ex1 or ey2 <= ey1:
        return
    cv2.rectangle(image, (ex1, ey1), (ex2, ey2), (0, 0, 255), 2)
    inset = 2 if (ex2 - ex1) >= 8 and (ey2 - ey1) >= 8 else 1
    ix1 = min(ex2, ex1 + inset)
    iy1 = min(ey2, ey1 + inset)
    ix2 = max(ix1, ex2 - inset)
    iy2 = max(iy1, ey2 - inset)
    cv2.rectangle(image, (ix1, iy1), (ix2, iy2), (0, 255, 0), 1)


def _draw_masked_layer(image: np.ndarray, layer: np.ndarray, guard_mask: np.ndarray | None) -> None:
    line_mask = np.any(layer != 0, axis=2)
    if guard_mask is not None:
        line_mask &= guard_mask <= 0
    if np.any(line_mask):
        image[line_mask] = layer[line_mask]


def _draw_masked_rectangle(
    image: np.ndarray,
    box: list[int],
    color: tuple[int, int, int],
    thickness: int,
    guard_mask: np.ndarray | None = None,
) -> None:
    x1, y1, x2, y2 = [int(value) for value in box[:4]]
    if x2 <= x1 or y2 <= y1:
        return
    layer = np.zeros_like(image)
    cv2.rectangle(layer, (x1, y1), (x2, y2), color, thickness)
    _draw_masked_layer(image, layer, guard_mask)


def _draw_masked_polyline(
    image: np.ndarray,
    polygon: list[list[int]],
    color: tuple[int, int, int],
    thickness: int,
    guard_mask: np.ndarray | None = None,
) -> None:
    if len(polygon) < 3:
        return
    layer = np.zeros_like(image)
    pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
    cv2.polylines(layer, [pts], isClosed=True, color=color, thickness=thickness)
    _draw_masked_layer(image, layer, guard_mask)


def _draw_plain_polyline(
    image: np.ndarray,
    polygon: list[list[int]],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if len(polygon) < 3:
        return
    points = np.array(polygon, np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)


def _draw_routed_erase_outline(
    image: np.ndarray,
    boxes: list,
    image_shape,
    guard_mask: np.ndarray | None = None,
) -> None:
    img_h, img_w = image_shape[:2]
    union = np.zeros((img_h, img_w), dtype=np.uint8)
    for raw_box in boxes or []:
        box = _coerce_box4(raw_box)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        x1 = max(0, min(img_w, x1))
        x2 = max(0, min(img_w, x2))
        y1 = max(0, min(img_h, y1))
        y2 = max(0, min(img_h, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        union[y1:y2, x1:x2] = 255
    if np.count_nonzero(union) == 0:
        return
    contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    layer = np.zeros_like(image)
    cv2.drawContours(layer, contours, -1, (0, 0, 255), 2)
    _draw_masked_layer(image, layer, guard_mask)


def _external_local_mode() -> bool:
    return os.environ.get("LOCAL_NLLB_TRANSLATION", "").strip().lower() in {"1", "true", "yes", "on"}


def _script_counts(text: str) -> dict[str, int]:
    return {
        "hangul": len(re.findall(r"[\uac00-\ud7af]", text)),
        "kana": len(re.findall(r"[\u3040-\u30ff]", text)),
        "han": len(re.findall(r"[\u3400-\u9fff]", text)),
    }


def _has_usable_translation(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    if re.fullmatch(r"[\W_]+", cleaned, flags=re.UNICODE):
        return len(cleaned) <= 6 and any(marker in cleaned for marker in ("！", "?", "？", "…", "．", ".", "ー", "—"))
    if re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", cleaned):
        return True
    return bool(re.search(r"[A-Za-z0-9]", cleaned)) and len(cleaned) <= 24


def _strong_dialogue_candidate(text: str, counts: dict[str, int]) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    compact_len = len("".join(c for c in cleaned if c.isalnum()))
    cjk_total = counts["hangul"] + counts["kana"] + counts["han"]
    if cjk_total < 4:
        return False
    if counts["kana"] >= 3 or counts["hangul"] >= 3:
        return True
    if counts["han"] >= 2 and cjk_total >= 5:
        return True
    if compact_len >= 6 and re.search(r"[。！？!?…]", cleaned):
        return True
    return False


def _normalize_floating_layout_boxes(constraints: list[dict], image_shape) -> None:
    if image_shape is None:
        return
    img_h, img_w = image_shape[:2]
    for constraint in constraints:
        if constraint.get("bubble_idx", -1) != -1:
            continue
        if constraint.get("mask_mode", "stroke") != "stroke":
            continue
        if constraint.get("route", "floating_dialogue") != "floating_dialogue":
            continue
        rb = constraint.get("red_box")
        gb = constraint.get("green_box")
        if not rb or not gb or len(rb) < 4 or len(gb) < 4:
            continue
        rx1, ry1, rx2, ry2 = [int(v) for v in rb[:4]]
        gx1, gy1, gx2, gy2 = [int(v) for v in gb[:4]]
        red_w = max(1, rx2 - rx1)
        red_h = max(1, ry2 - ry1)
        green_w = max(1, gx2 - gx1)
        green_h = max(1, gy2 - gy1)
        red_area = max(1, red_w * red_h)
        green_area = max(1, green_w * green_h)
        vertical = red_h >= red_w * 1.18
        max_x_pad = max(6, min(28, int(red_w * (0.14 if vertical else 0.20)) + 4))
        max_y_pad = max(8, min(34, int(red_h * (0.08 if vertical else 0.18)) + 4))
        loose_area = green_area > int(red_area * 1.55)
        loose_left = rx1 - gx1 > max_x_pad
        loose_right = gx2 - rx2 > max_x_pad
        loose_top = ry1 - gy1 > max_y_pad
        loose_bottom = gy2 - ry2 > max_y_pad
        if not (loose_area or loose_left or loose_right or loose_top or loose_bottom):
            continue
        nx1 = max(0, min(img_w, rx1 - max_x_pad))
        ny1 = max(0, min(img_h, ry1 - max_y_pad))
        nx2 = max(0, min(img_w, rx2 + max_x_pad))
        ny2 = max(0, min(img_h, ry2 + max_y_pad))
        if nx2 <= nx1 or ny2 <= ny1:
            continue
        constraint["green_box"] = [nx1, ny1, nx2, ny2]
        constraint["green_polygon"] = [[nx1, ny1], [nx2, ny1], [nx2, ny2], [nx1, ny2]]
        constraint["layout_adjustment"] = "floating_source_box_normalized"


def _load_step1_text_mask(detect_dir: Path, image_shape) -> np.ndarray | None:
    if image_shape is None:
        return None
    seg_path = detect_dir / "seg_mask.png"
    if not seg_path.exists():
        return None
    mask = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    img_h, img_w = image_shape[:2]
    if mask.shape[:2] != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    if mask.ndim == 3:
        if mask.shape[2] == 1:
            mask = mask[:, :, 0]
        else:
            mask = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
    return (mask > 0).astype(np.uint8) * 255


def _structural_art_mask(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 55, 145)
    dark = ((gray < 128).astype(np.uint8) * 255)
    structural = cv2.bitwise_or(edges, dark)
    structural = cv2.morphologyEx(
        structural,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    return structural


def _polygon_mask(image_shape, polygon: list[list[int]]) -> np.ndarray:
    img_h, img_w = image_shape[:2]
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if len(polygon) >= 3:
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    return mask


def _notched_box_polygon(box: list[int], obstacle_box: tuple[int, int, int, int], side: str) -> list[list[int]]:
    x1, y1, x2, y2 = [int(value) for value in box[:4]]
    ox1, oy1, ox2, oy2 = [int(value) for value in obstacle_box]
    if x2 <= x1 or y2 <= y1:
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    oy1 = max(y1, min(y2 - 2, oy1))
    oy2 = max(oy1 + 1, min(y2, oy2))
    if side == "left":
        notch_x = max(x1 + 1, min(x2 - 2, ox2))
        if oy1 <= y1 + 1 and oy2 >= y2 - 1:
            return [[notch_x, y1], [x2, y1], [x2, y2], [notch_x, y2]]
        if oy1 <= y1 + 1:
            return [[notch_x, y1], [x2, y1], [x2, y2], [x1, y2], [x1, oy2], [notch_x, oy2]]
        if oy2 >= y2 - 1:
            return [[x1, y1], [x2, y1], [x2, y2], [notch_x, y2], [notch_x, oy1], [x1, oy1]]
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, oy2], [notch_x, oy2], [notch_x, oy1], [x1, oy1]]
    notch_x = max(x1 + 2, min(x2 - 1, ox1))
    if oy1 <= y1 + 1 and oy2 >= y2 - 1:
        return [[x1, y1], [notch_x, y1], [notch_x, y2], [x1, y2]]
    if oy1 <= y1 + 1:
        return [[x1, y1], [notch_x, y1], [notch_x, oy2], [x2, oy2], [x2, y2], [x1, y2]]
    if oy2 >= y2 - 1:
        return [[x1, y1], [x2, y1], [x2, oy1], [notch_x, oy1], [notch_x, y2], [x1, y2]]
    return [[x1, y1], [x2, y1], [x2, oy1], [notch_x, oy1], [notch_x, oy2], [x2, oy2], [x2, y2], [x1, y2]]


def _expand_obstacle_through_green_art(
    protected_art: np.ndarray,
    green_box: list[int],
    seed_obstacle: tuple[int, int, int, int],
    side: str,
    image_shape,
) -> tuple[int, int, int, int]:
    img_h, img_w = image_shape[:2]
    gx1, gy1, gx2, gy2 = [int(value) for value in green_box[:4]]
    gx1 = max(0, min(img_w, gx1))
    gx2 = max(0, min(img_w, gx2))
    gy1 = max(0, min(img_h, gy1))
    gy2 = max(0, min(img_h, gy2))
    if gx2 <= gx1 or gy2 <= gy1:
        return seed_obstacle

    sx1, sy1, sx2, sy2 = [int(value) for value in seed_obstacle]
    seed_local = (
        max(0, sx1 - gx1),
        max(0, sy1 - gy1),
        min(gx2 - gx1, sx2 - gx1),
        min(gy2 - gy1, sy2 - gy1),
    )
    green_art = protected_art[gy1:gy2, gx1:gx2]
    if green_art.size == 0 or np.count_nonzero(green_art) < 20:
        return seed_obstacle

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats((green_art > 0).astype(np.uint8), connectivity=8)
    selected: list[tuple[int, int, int, int]] = []
    seed_x1, seed_y1, seed_x2, seed_y2 = seed_local
    seed_center_x = (seed_x1 + seed_x2) * 0.5
    green_w = max(1, gx2 - gx1)
    green_h = max(1, gy2 - gy1)
    min_area = max(30, int(green_w * green_h * 0.004))
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        bx = int(stats[label, cv2.CC_STAT_LEFT])
        by = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        if bw < 5 or bh < 8:
            continue
        cx = bx + bw * 0.5
        horizontal_match = cx <= seed_center_x + max(18, green_w * 0.18) if side == "left" else cx >= seed_center_x - max(18, green_w * 0.18)
        if not horizontal_match:
            continue
        intersects_seed = not (
            bx + bw < seed_x1 - 12
            or bx > seed_x2 + 12
            or by + bh < seed_y1 - 18
            or by > seed_y2 + 18
        )
        vertical_near_seed = by <= seed_y2 + max(18, int(green_h * 0.10)) and by + bh >= seed_y1 - max(18, int(green_h * 0.10))
        tall_side_art = bh >= max(24, int(green_h * 0.20)) and (bx <= seed_x2 + 16 if side == "left" else bx + bw >= seed_x1 - 16)
        if intersects_seed or (vertical_near_seed and tall_side_art):
            selected.append((gx1 + bx, gy1 + by, gx1 + bx + bw, gy1 + by + bh))

    if not selected:
        return seed_obstacle

    pad_x = max(3, min(14, int(green_w * 0.04)))
    pad_y = max(4, min(18, int(green_h * 0.04)))
    return (
        max(gx1, min([seed_obstacle[0], *[box[0] for box in selected]]) - pad_x),
        max(gy1, min([seed_obstacle[1], *[box[1] for box in selected]]) - pad_y),
        min(gx2, max([seed_obstacle[2], *[box[2] for box in selected]]) + pad_x),
        min(gy2, max([seed_obstacle[3], *[box[3] for box in selected]]) + pad_y),
    )


def _source_component_erase_boxes(
    source_mask: np.ndarray,
    red_polygon_mask: np.ndarray,
    red_box: list[int],
) -> list[list[int]]:
    x1, y1, x2, y2 = [int(value) for value in red_box[:4]]
    if x2 <= x1 or y2 <= y1:
        return []
    source = cv2.bitwise_and(source_mask, red_polygon_mask)
    roi = source[y1:y2, x1:x2]
    if np.count_nonzero(roi) < 20:
        return []
    roi = cv2.dilate(
        roi,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7)),
        iterations=1,
    )
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats((roi > 0).astype(np.uint8), connectivity=8)
    boxes = []
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 12:
            continue
        bx = int(stats[label, cv2.CC_STAT_LEFT])
        by = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        if bw < 2 or bh < 3:
            continue
        boxes.append([x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh])
    if not boxes:
        return []
    boxes.sort(key=lambda box: (box[0], box[1]))
    merged: list[list[int]] = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue
        prev = merged[-1]
        close_x = max(0, max(box[0], prev[0]) - min(box[2], prev[2])) <= 10
        close_y = max(0, max(box[1], prev[1]) - min(box[3], prev[3])) <= 18
        if close_x and close_y:
            prev[:] = [min(prev[0], box[0]), min(prev[1], box[1]), max(prev[2], box[2]), max(prev[3], box[3])]
        else:
            merged.append(box)
    return merged[:12]


def _trim_box_to_source_mask(source_mask: np.ndarray, box: tuple[int, int, int, int], image_shape) -> list[int] | None:
    img_h, img_w = image_shape[:2]
    x1, y1, x2, y2 = [int(value) for value in box]
    x1 = max(0, min(img_w, x1))
    x2 = max(0, min(img_w, x2))
    y1 = max(0, min(img_h, y1))
    y2 = max(0, min(img_h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = source_mask[y1:y2, x1:x2] > 0
    if np.count_nonzero(roi) < 10:
        return None
    ys, xs = np.where(roi)
    pad_x = 2
    pad_y = 2
    return [
        max(0, x1 + int(xs.min()) - pad_x),
        max(0, y1 + int(ys.min()) - pad_y),
        min(img_w, x1 + int(xs.max()) + 1 + pad_x),
        min(img_h, y1 + int(ys.max()) + 1 + pad_y),
    ]


def _notched_source_erase_boxes(
    source_mask: np.ndarray,
    red_box: list[int],
    obstacle_box: tuple[int, int, int, int],
    side: str,
    image_shape,
) -> tuple[list[list[int]], list[int] | None]:
    rx1, ry1, rx2, ry2 = [int(value) for value in red_box[:4]]
    ox1, oy1, ox2, oy2 = [int(value) for value in obstacle_box]
    min_band = 4
    if side == "left":
        bands = [
            (rx1, ry1, rx2, oy1),
            (ox2, oy1, rx2, oy2),
            (rx1, oy2, rx2, ry2),
        ]
        preferred = 1
    else:
        bands = [
            (rx1, ry1, rx2, oy1),
            (rx1, oy1, ox1, oy2),
            (rx1, oy2, rx2, ry2),
        ]
        preferred = 1
    trimmed = []
    counts = []
    for band in bands:
        if band[2] - band[0] < min_band or band[3] - band[1] < min_band:
            continue
        box = _trim_box_to_source_mask(source_mask, band, image_shape)
        if box is None:
            continue
        trimmed.append(box)
        x1, y1, x2, y2 = box
        counts.append(int(np.count_nonzero(source_mask[y1:y2, x1:x2] > 0)))
    if not trimmed:
        return [], None
    anchor_box = None
    if preferred < len(bands):
        preferred_box = _trim_box_to_source_mask(source_mask, bands[preferred], image_shape)
        if preferred_box is not None:
            anchor_box = preferred_box
    if anchor_box is None:
        anchor_box = trimmed[int(np.argmax(counts))]
    return trimmed, anchor_box


def _clamped_layout_box_around_source(
    source_box: list[int],
    green_box: list[int],
    protected_art: np.ndarray,
    image_shape,
) -> list[list[int]] | None:
    img_h, img_w = image_shape[:2]
    sx1, sy1, sx2, sy2 = [int(value) for value in source_box[:4]]
    gx1, gy1, gx2, gy2 = [int(value) for value in green_box[:4]]
    sx1 = max(0, min(img_w, sx1))
    sx2 = max(0, min(img_w, sx2))
    sy1 = max(0, min(img_h, sy1))
    sy2 = max(0, min(img_h, sy2))
    gx1 = max(0, min(img_w, gx1))
    gx2 = max(0, min(img_w, gx2))
    gy1 = max(0, min(img_h, gy1))
    gy2 = max(0, min(img_h, gy2))
    if sx2 <= sx1 or sy2 <= sy1 or gx2 <= gx1 or gy2 <= gy1:
        return None

    width = sx2 - sx1
    height = sy2 - sy1
    margin_x = max(5, min(20, int(round(max(width * 0.08, height * 0.08)))))
    margin_y = max(5, min(16, int(round(max(height * 0.08, width * 0.04)))))
    lx1 = max(gx1, sx1 - margin_x)
    ly1 = max(gy1, sy1 - margin_y)
    lx2 = min(gx2, sx2 + margin_x)
    ly2 = min(gy2, sy2 + margin_y)
    if lx2 <= lx1 or ly2 <= ly1:
        return None

    candidate_polygon = [[lx1, ly1], [lx2, ly1], [lx2, ly2], [lx1, ly2]]
    border = np.zeros((ly2 - ly1, lx2 - lx1), dtype=np.uint8)
    cv2.rectangle(border, (0, 0), (lx2 - lx1 - 1, ly2 - ly1 - 1), 255, thickness=3)
    border_art = int(np.count_nonzero((protected_art[ly1:ly2, lx1:lx2] > 0) & (border > 0)))
    border_area = int(np.count_nonzero(border > 0))
    if border_art <= max(8, int(border_area * 0.035)):
        return candidate_polygon

    art_roi = protected_art[ly1:ly2, lx1:lx2] > 0
    if np.count_nonzero(art_roi) >= 12:
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(art_roi.astype(np.uint8), connectivity=8)
        source_cx = (sx1 + sx2) * 0.5
        source_cy = (sy1 + sy2) * 0.5
        for label in range(1, labels_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 12:
                continue
            ax1 = lx1 + int(stats[label, cv2.CC_STAT_LEFT])
            ay1 = ly1 + int(stats[label, cv2.CC_STAT_TOP])
            ax2 = ax1 + int(stats[label, cv2.CC_STAT_WIDTH])
            ay2 = ay1 + int(stats[label, cv2.CC_STAT_HEIGHT])
            art_cx = (ax1 + ax2) * 0.5
            art_cy = (ay1 + ay2) * 0.5
            if art_cx < source_cx and ax2 > lx1:
                lx1 = min(sx1 - 2, max(lx1, ax2 + 2))
            elif art_cx >= source_cx and ax1 < lx2:
                lx2 = max(sx2 + 2, min(lx2, ax1 - 2))
            if art_cy < source_cy and ay2 > ly1 and abs(art_cx - source_cx) < max(width, height):
                ly1 = min(sy1 - 2, max(ly1, ay2 + 2))
            elif art_cy >= source_cy and ay1 < ly2 and abs(art_cx - source_cx) < max(width, height):
                ly2 = max(sy2 + 2, min(ly2, ay1 - 2))
    lx1 = max(gx1, min(sx1 - 1, lx1))
    ly1 = max(gy1, min(sy1 - 1, ly1))
    lx2 = min(gx2, max(sx2 + 1, lx2))
    ly2 = min(gy2, max(sy2 + 1, ly2))
    if lx2 <= lx1 or ly2 <= ly1:
        return None
    return [[lx1, ly1], [lx2, ly1], [lx2, ly2], [lx1, ly2]]


def _green_polygons_from_source_boxes(
    erase_boxes: list[list[int]],
    green_box: list[int],
    protected_art: np.ndarray,
    image_shape,
) -> list[list[list[int]]]:
    polygons = []
    seen = set()
    for box in erase_boxes:
        polygon = _clamped_layout_box_around_source(box, green_box, protected_art, image_shape)
        if not polygon:
            continue
        key = tuple(tuple(point) for point in polygon)
        if key in seen:
            continue
        seen.add(key)
        polygons.append(polygon)
    return polygons


def _apply_art_aware_floating_routes(constraints: list[dict], image: np.ndarray | None, detect_dir: Path) -> None:
    if image is None:
        return
    image_shape = image.shape
    img_h, img_w = image_shape[:2]
    source_mask = _load_step1_text_mask(detect_dir, image_shape)
    structural = _structural_art_mask(image)
    if source_mask is None or structural is None:
        return
    source_dilate = cv2.dilate(
        source_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    protected_art = cv2.bitwise_and(structural, cv2.bitwise_not(source_dilate))
    protected_art = cv2.morphologyEx(
        protected_art,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )

    for constraint in constraints:
        if constraint.get("bubble_idx", -1) != -1:
            continue
        if constraint.get("mask_mode", "stroke") != "stroke":
            continue
        if constraint.get("route", "floating_dialogue") != "floating_dialogue":
            continue
        if "missed_bubble" in str(constraint.get("fallback_source") or ""):
            continue
        red_box = [int(value) for value in constraint.get("red_box", [0, 0, 0, 0])[:4]]
        green_box = [int(value) for value in constraint.get("green_box", red_box)[:4]]
        rx1, ry1, rx2, ry2 = red_box
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        red_w = rx2 - rx1
        red_h = ry2 - ry1
        if red_w < 28 or red_h < 42:
            continue
        pale_fraction = _pale_region_fraction(image, (rx1, ry1, rx2, ry2))
        dark_fraction = _dark_region_fraction(image, (rx1, ry1, rx2, ry2))
        if pale_fraction < 0.62 or dark_fraction > 0.22:
            continue

        red_source = source_mask[ry1:ry2, rx1:rx2] > 0
        source_count = int(np.count_nonzero(red_source))
        if source_count < max(18, int(red_w * red_h * 0.004)):
            continue
        source_y, source_x = np.where(red_source)
        source_cx = float(source_x.mean())

        region_art = protected_art[ry1:ry2, rx1:rx2]
        if np.count_nonzero(region_art) < max(60, int(red_w * red_h * 0.012)):
            continue
        labels_count, _, stats, _ = cv2.connectedComponentsWithStats((region_art > 0).astype(np.uint8), connectivity=8)
        best = None
        best_score = 0.0
        for label in range(1, labels_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            bx = int(stats[label, cv2.CC_STAT_LEFT])
            by = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < max(45, int(red_w * red_h * 0.010)):
                continue
            if bh < max(18, int(red_h * 0.22)) or bw < max(10, int(red_w * 0.08)):
                continue
            source_inside = int(np.count_nonzero(red_source[by:by + bh, bx:bx + bw]))
            if source_inside / max(1, source_count) > 0.34:
                continue
            side = ""
            if bx <= max(8, int(red_w * 0.18)) and bx + bw <= red_w - max(6, int(red_w * 0.12)) and source_cx > bx + bw * 0.58:
                side = "left"
            elif bx + bw >= red_w - max(8, int(red_w * 0.18)) and bx >= max(6, int(red_w * 0.12)) and source_cx < bx + bw * 0.42:
                side = "right"
            if not side:
                continue
            score = area * (1.0 + min(1.4, bh / max(1, red_h)))
            if score > best_score:
                best_score = score
                best = (side, bx, by, bw, bh)
        if best is None:
            continue

        side, bx, by, bw, bh = best
        pad_x = max(4, min(18, int(red_w * 0.08)))
        pad_y = max(4, min(18, int(red_h * 0.06)))
        obstacle_red = (
            max(rx1, rx1 + bx - pad_x),
            max(ry1, ry1 + by - pad_y),
            min(rx2, rx1 + bx + bw + pad_x),
            min(ry2, ry1 + by + bh + pad_y),
        )
        red_polygon = _notched_box_polygon(red_box, obstacle_red, side)
        red_poly_mask = _polygon_mask(image_shape, red_polygon)
        kept_source = int(np.count_nonzero((source_mask > 0) & (red_poly_mask > 0)))
        if kept_source < int(source_count * 0.54):
            continue

        gx1, gy1, gx2, gy2 = green_box
        obstacle_green = (
            max(gx1, obstacle_red[0] - max(3, pad_x // 2)),
            max(gy1, obstacle_red[1] - max(3, pad_y // 2)),
            min(gx2, obstacle_red[2] + max(3, pad_x // 2)),
            min(gy2, obstacle_red[3] + max(3, pad_y // 2)),
        )
        obstacle_green = _expand_obstacle_through_green_art(
            protected_art,
            green_box,
            obstacle_green,
            side,
            image_shape,
        )
        green_polygon = _notched_box_polygon(green_box, obstacle_green, side)
        erase_boxes, anchor_box = _notched_source_erase_boxes(
            source_mask,
            red_box,
            obstacle_red,
            side,
            image_shape,
        )
        if not erase_boxes:
            erase_boxes = _source_component_erase_boxes(source_mask, red_poly_mask, red_box)
            if not erase_boxes:
                continue
            anchor_box = [
                max(0, min(box[0] for box in erase_boxes) - 2),
                max(0, min(box[1] for box in erase_boxes) - 2),
                min(img_w, max(box[2] for box in erase_boxes) + 2),
                min(img_h, max(box[3] for box in erase_boxes) + 2),
            ]

        green_source = source_mask[gy1:gy2, gx1:gx2] > 0
        green_source_count = int(np.count_nonzero(green_source))
        if green_source_count >= max(24, int(source_count * 0.60)):
            routed_source = np.zeros_like(source_mask, dtype=np.uint8)
            for box in erase_boxes:
                ex1, ey1, ex2, ey2 = [int(value) for value in box[:4]]
                ex1 = max(0, min(img_w, ex1))
                ex2 = max(0, min(img_w, ex2))
                ey1 = max(0, min(img_h, ey1))
                ey2 = max(0, min(img_h, ey2))
                if ex2 > ex1 and ey2 > ey1:
                    routed_source[ey1:ey2, ex1:ex2] = 255
            green_source_full = np.zeros_like(source_mask, dtype=bool)
            green_source_full[gy1:gy2, gx1:gx2] = green_source
            routed_coverage = int(np.count_nonzero(green_source_full & (routed_source > 0))) / float(
                max(1, green_source_count)
            )
            if routed_coverage < 0.74:
                continue

        green_polygons = _green_polygons_from_source_boxes(
            erase_boxes,
            green_box,
            protected_art,
            image_shape,
        )
        if not green_polygons:
            green_polygons = [green_polygon]

        constraint["red_polygon"] = red_polygon
        constraint["red_box"] = anchor_box
        constraint["green_polygon"] = green_polygon
        constraint["green_polygons"] = green_polygons
        constraint["erase_boxes"] = erase_boxes
        constraint["art_aware_routed"] = True
        constraint["art_aware_erase_only"] = True
        existing = constraint.get("layout_adjustment")
        constraint["layout_adjustment"] = f"{existing}+art_aware_route" if existing else "art_aware_route"


_SHORT_DIALOGUE_STEMS = (
    "\u3042\u3063\u305f",  # atta / attaka variants
    "\u3042\u3064",       # hot / warm variants
    "\u3055\u3080",       # cold
    "\u3044\u305f",       # hurts
    "\u3084\u3060",       # no / don't
    "\u3060\u3081",       # no good
    "\u3044\u3084",       # no / gross
    "\u3044\u3044",       # okay / good
    "\u306d\u3048",       # hey
    "\u307b\u3089",       # look
    "\u307e\u3063\u3066",   # wait
    "\u3061\u3087\u3063\u3068", # wait / a little
    "\u3042\u306e",       # um / hey
    "\u3059\u304b",       # casual question suffix
    "\u3053\u308c",       # this
    "\u305d\u308c",       # that
    "\u306a\u3093",       # what/why fragments
    "\u3059\u3054",       # amazing
    "\u3053\u308f",       # scary
    "\u304d\u3082",       # gross
    "\u3046\u305d",       # no way / lie
    "\u3063\u3066",       # -tte (quotative/continuation particle, e.g. spoken "...\u3063\u3066")
    "\u3044\u306e",       # -ino (casual question ending, e.g. "...\u3044\u306e?")
    "\u306f\u3042\u3041", # haaa (sigh/breath reaction)
)

# Single-character Japanese sentence-final/question particles. These are
# near-universally real spoken dialogue (never decorative SFX art), so a
# lone occurrence should not be discarded purely for being short \u2014 unlike
# classify_text_by_content()'s general short-hiragana-run heuristic, which
# has no way to tell "\u304b" (a question particle) apart from a one-glyph SFX.
_SINGLE_CHAR_DIALOGUE_PARTICLES = frozenset("\u304b\u306d\u3088\u306a\u306e\u3055\u305e\u308f")


def _normalized_cjk_fragment(text: str) -> str:
    return "".join(
        c for c in str(text or "").strip()
        if c.isalnum() or "\u3040" <= c <= "\u30ff" or "\u3400" <= c <= "\u9fff" or "\uac00" <= c <= "\ud7af"
    )


def _obvious_sfx_fragment(text: str) -> bool:
    compact = _normalized_cjk_fragment(text)
    if not compact:
        return True
    if len(compact) <= 1:
        return True
    if len(set(compact)) == 1:
        return True

    sound_prefixes = (
        "\u3061\u3085\u3071",  # chupa
        "\u3061\u3085",
        "\u3074\u3085",
        "\u3073\u3085",
        "\u306b\u3085",
        "\u306f\u3041",
    )
    exact_sound_tokens = {
        "\u3071",
        "\u305a",
        "\u3050",
        "\u3054",
        "\u3069",
        "\u3042\u3063",
        "\u3093\u3063",
    }
    if compact in exact_sound_tokens or any(compact.startswith(stem) for stem in sound_prefixes):
        return True
    return False


def _short_spoken_dialogue_fragment(item: dict, image: np.ndarray | None) -> bool:
    text = str(item.get("text", "")).strip()
    compact = _normalized_cjk_fragment(text)
    if not compact:
        return False

    # A lone sentence-final/question particle (e.g. "か") is real dialogue,
    # not SFX — but _obvious_sfx_fragment() and the cjk_total<2 gate below
    # would otherwise reject any single-character fragment outright.
    is_single_char_particle = (
        len(compact) == 1 and compact in _SINGLE_CHAR_DIALOGUE_PARTICLES
    )
    if not is_single_char_particle and _obvious_sfx_fragment(text):
        return False

    counts = _script_counts(text)
    cjk_total = counts["hangul"] + counts["kana"] + counts["han"]
    if not is_single_char_particle and (cjk_total < 2 or len(compact) > 8):
        return False

    box = item.get("box", {})
    coords = (
        int(box.get("x1", 0)),
        int(box.get("y1", 0)),
        int(box.get("x2", 0)),
        int(box.get("y2", 0)),
    )
    x1, y1, x2, y2 = coords
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    if image is not None:
        img_h, img_w = image.shape[:2]
        min_readable_width = max(28, int(img_w * 0.025))
        min_readable_height = max(72, int(img_h * 0.070))
        if width < min_readable_width and height < min_readable_height and len(compact) <= 5:
            return False

    if is_single_char_particle:
        return True

    if counts["han"] >= 1 or counts["hangul"] >= 2:
        return True
    if not any(stem in compact for stem in _SHORT_DIALOGUE_STEMS):
        return False

    pale_fraction = _pale_region_fraction(image, coords)
    dark_fraction = _dark_region_fraction(image, coords)
    return pale_fraction >= 0.32 or dark_fraction >= 0.18


def _erase_only_adjacent_fragment(item: dict, image: np.ndarray | None) -> bool:
    text = str(item.get("text", "")).strip()
    compact = _normalized_cjk_fragment(text)
    if not compact or _obvious_sfx_fragment(text):
        return False

    counts = _script_counts(text)
    if counts["han"] or counts["hangul"]:
        return False
    if counts["kana"] < 2 or len(compact) > 4:
        return False

    box = item.get("box", {})
    coords = (
        int(box.get("x1", 0)),
        int(box.get("y1", 0)),
        int(box.get("x2", 0)),
        int(box.get("y2", 0)),
    )
    width = max(1, coords[2] - coords[0])
    height = max(1, coords[3] - coords[1])
    if width > max(42, int(height * 0.74)):
        return False

    pale_fraction = _pale_region_fraction(image, coords)
    dark_fraction = _dark_region_fraction(image, coords)
    return pale_fraction >= 0.34 or dark_fraction >= 0.16




def _recoverable_large_floating_dialogue(
    text: str,
    counts: dict[str, int],
    width: int,
    height: int,
    x1: int,
    y1: int,
    img_w: int,
    img_h: int,
    image: np.ndarray | None = None,
) -> bool:
    cleaned = str(text or "").strip()
    compact_len = len("".join(c for c in cleaned if c.isalnum()))
    cjk_total = counts["hangul"] + counts["kana"] + counts["han"]
    punctuated_short_cjk_caption = bool(
        cjk_total >= 5
        and compact_len >= 5
        and (
            counts["han"] >= 5
            or counts["hangul"] >= 5
            or counts["kana"] >= 5
        )
        and re.search(r"[，。、！？!?…]", cleaned)
    )
    # A healed single-line dialogue tagline (a step-5 same-line-merge that
    # reunited a shattered horizontal CJK line, e.g. "몰디브 7일") is
    # wide-and-short -- the same silhouette as a title/logo stamp -- and can
    # be short in raw glyph count (4-5 real CJK glyphs) without being
    # weak/noise. Computed early so it can waive the width/glyph-count floors
    # below for THIS shape specifically, rather than loosening them for every
    # floating box. Position doesn't distinguish a real tagline from a title
    # (verified: new_sample_14's torso sits mid-page, not near an edge), so
    # shape + a confirmed translatable reading is the signal instead.
    single_line_merged_dialogue = (
        height > 0
        and width >= height * 2.2
        and cjk_total >= 4
        and _has_usable_translation(cleaned)
    )
    if not single_line_merged_dialogue:
        if cjk_total < 6 or (compact_len < 8 and not punctuated_short_cjk_caption):
            return False
        if not _strong_dialogue_candidate(cleaned, counts):
            return False

    area_ratio = (width * height) / max(1, img_w * img_h)
    y2 = y1 + height
    # A thin, wide, low-area-ratio narration strip is equally valid dialogue
    # whether it sits at the top or bottom of a page (a caption band, not a
    # bubble) -- the geometry that matters is "near a page edge", not which
    # edge. Both ends get the same width/height/area/script-count bar below.
    near_top_or_bottom_edge = img_h > 0 and (
        y1 <= int(img_h * 0.22) or y2 >= int(img_h * 0.78)
    )
    top_narration_caption = (
        img_w > 0
        and img_h > 0
        and near_top_or_bottom_edge
        and height <= max(140, int(img_h * 0.18))
        # Matches the merge function's own width cap (0.97 of page width,
        # loosened earlier this session), minus a small margin -- a thin
        # (low area_ratio, checked below) full-width narration strip is
        # exactly what that wider merge is meant to produce; this exemption
        # must accept what it can legitimately hand over instead of
        # rejecting a correctly-merged band as "too wide" (verified:
        # new_sample_14's top AND bottom narration bands, each merged into
        # a single ~1000px-wide constraint, fell through this check when it
        # was still capped at 0.78 of page width and top-edge-only, and
        # were dropped from layout_constraints).
        and width <= max(520, int(img_w * 0.95))
        and area_ratio <= 0.075
        and (
            (counts["hangul"] >= 8 and (" " in cleaned or height >= 52))
            or (counts["han"] >= 7 and re.search(r"[，。、！？!?…]", cleaned))
            or (counts["kana"] >= 7 and re.search(r"[、。！？!?…]", cleaned))
        )
    )
    if top_narration_caption:
        coords = (x1, y1, x1 + width, y1 + height)
        pale_fraction = _pale_region_fraction(image, coords)
        dark_fraction = _dark_region_fraction(image, coords)
        if image is None or pale_fraction >= 0.58 or dark_fraction <= 0.22:
            return True

    punctuated_korean_emphasis_caption = (
        img_w > 0
        and img_h > 0
        and counts["hangul"] >= 5
        and compact_len >= 5
        and re.search(r"[！？!?…]", cleaned)
        and width <= max(460, int(img_w * 0.42))
        and height <= max(120, int(img_h * 0.10))
        and area_ratio <= 0.032
        and not any(marker in cleaned for marker in CREDIT_MARKERS)
    )
    if punctuated_korean_emphasis_caption:
        coords = (x1, y1, x1 + width, y1 + height)
        dark_fraction = _dark_region_fraction(image, coords)
        if image is None or dark_fraction <= 0.38:
            return True

    wide_pale_boxed_caption = (
        img_w > 0
        and img_h > 0
        and counts["han"] >= 5
        and compact_len >= 6
        and width <= max(420, int(img_w * 0.55))
        and height <= max(140, min(240, int(img_h * 0.040)))
        and area_ratio <= 0.016
        and not any(marker in cleaned for marker in CREDIT_MARKERS)
    )
    if wide_pale_boxed_caption:
        coords = (x1, y1, x1 + width, y1 + height)
        pale_fraction = _pale_region_fraction(image, coords)
        dark_fraction = _dark_region_fraction(image, coords)
        if image is None or (pale_fraction >= 0.50 and dark_fraction <= 0.32):
            return True

    if not single_line_merged_dialogue:
        if width > max(260, int(img_w * 0.22)) and area_ratio > 0.045:
            return False
        if width > max(360, int(img_w * 0.30)):
            return False

    vertical_dialogue = height >= width * 1.18
    compact_caption = (
        height >= width * 0.70
        and (not img_w or width <= max(260, int(img_w * 0.36)))
    )
    wide_title_like = width >= height * 1.75 and (not img_w or width >= int(img_w * 0.25))
    top_title_like = bool(img_h and y1 <= int(img_h * 0.16) and width >= height * 1.20)
    readable_chinese_caption = (
        counts["han"] >= 6
        and cjk_total >= 7
        and compact_len >= 8
        and width <= max(360, int(img_w * 0.38))
        and height <= max(130, int(width * 0.56))
        and re.search(r"[，。、！？!?…]", cleaned)
    )

    if (
        (wide_title_like or top_title_like)
        and not readable_chinese_caption
        and not single_line_merged_dialogue
    ):
        return False
    return (
        vertical_dialogue
        or compact_caption
        or readable_chinese_caption
        or single_line_merged_dialogue
    )



def _pale_region_fraction(image: np.ndarray | None, coords: tuple[int, int, int, int]) -> float:
    if image is None:
        return 0.0
    x1, y1, x2, y2 = coords
    x1 = max(0, min(image.shape[1], x1))
    x2 = max(0, min(image.shape[1], x2))
    y1 = max(0, min(image.shape[0], y1))
    y2 = max(0, min(image.shape[0], y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return float(np.mean((gray > 168) & (hsv[:, :, 1] < 125)))


def _dark_region_fraction(image: np.ndarray | None, coords: tuple[int, int, int, int]) -> float:
    if image is None:
        return 0.0
    x1, y1, x2, y2 = coords
    x1 = max(0, min(image.shape[1], x1))
    x2 = max(0, min(image.shape[1], x2))
    y1 = max(0, min(image.shape[0], y1))
    y2 = max(0, min(image.shape[0], y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray < 120))


def _numeric_device_label_text(text: str) -> bool:
    compact = _normalized_cjk_fragment(text)
    if not 1 <= len(compact) <= 4:
        return False
    return all(char in DEVICE_LABEL_CHARS for char in compact)


def _looks_like_embedded_device_label(item: dict, image_shape, image: np.ndarray | None) -> bool:
    if item.get("bubble_idx", -1) != -1:
        return False
    if not _numeric_device_label_text(str(item.get("text", ""))):
        return False
    if image_shape is None:
        return True

    box = item.get("box", {})
    img_h = int(image_shape[0])
    img_w = int(image_shape[1])
    x1 = max(0, min(img_w, int(box.get("x1", 0))))
    y1 = max(0, min(img_h, int(box.get("y1", 0))))
    x2 = max(0, min(img_w, int(box.get("x2", x1))))
    y2 = max(0, min(img_h, int(box.get("y2", y1))))
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    area_ratio = (width * height) / max(1, img_w * img_h)
    if width > max(180, int(img_w * 0.24)):
        return False
    if height > max(96, int(img_h * 0.055)):
        return False
    if area_ratio > 0.018:
        return False

    if image is None:
        return True

    pad_x = max(16, min(48, int(width * 0.45)))
    pad_y = max(14, min(42, int(height * 0.55)))
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(img_w, x2 + pad_x)
    cy2 = min(img_h, y2 + pad_y)
    context = image[cy1:cy2, cx1:cx2]
    if context.size == 0:
        return True

    gray = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY)
    dark_fraction = float(np.mean(gray < 100))
    pale_fraction = float(np.mean(gray > 170))
    edges = cv2.Canny(gray, 60, 160)
    edge_fraction = float(np.mean(edges > 0))
    roi_pale = _pale_region_fraction(image, (x1, y1, x2, y2))
    roi_dark = _dark_region_fraction(image, (x1, y1, x2, y2))

    panel_like_context = edge_fraction >= 0.035 or dark_fraction >= 0.045 or pale_fraction >= 0.60
    readable_on_art_surface = roi_pale >= 0.30 or roi_dark >= 0.20
    return panel_like_context and readable_on_art_surface and pale_fraction >= 0.18


def _matched_semantic_region(
    box: tuple[int, int, int, int], semantic_dialogue_boxes: list[tuple[int, int, int, int]]
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    area_a = max(1, (x2 - x1) * (y2 - y1))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    for region in semantic_dialogue_boxes:
        sx1, sy1, sx2, sy2 = region
        ix1, iy1 = max(x1, sx1), max(y1, sy1)
        ix2, iy2 = min(x2, sx2), min(y2, sy2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_b = max(1, (sx2 - sx1) * (sy2 - sy1))
        iou = inter / float(area_a + area_b - inter)
        center_hit = sx1 <= cx <= sx2 and sy1 <= cy <= sy2
        if iou > 0.05 or center_hit:
            return region
    return None


def _merge_adjacent_floating_line_fragments(
    ocr_data: list[dict],
    image_shape,
    semantic_dialogue_boxes: list[tuple[int, int, int, int]] | None = None,
) -> list[dict]:
    if not isinstance(ocr_data, list) or not ocr_data:
        return ocr_data
    semantic_dialogue_boxes = semantic_dialogue_boxes or []

    img_h = int(image_shape[0]) if image_shape is not None else 0
    img_w = int(image_shape[1]) if image_shape is not None else 0

    def eligible(item: dict) -> bool:
        if item.get("bubble_idx", -1) != -1:
            return False
        text = str(item.get("text", "")).strip()
        if not text or not _has_usable_translation(text):
            return False
        if _numeric_device_label_text(text):
            return False
        counts = _script_counts(text)
        if counts["hangul"] + counts["kana"] + counts["han"] <= 0:
            return False
        box = item.get("box", {})
        x1 = int(box.get("x1", 0))
        y1 = int(box.get("y1", 0))
        x2 = int(box.get("x2", x1))
        y2 = int(box.get("y2", y1))
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        # The absolute page-fraction ceiling exists to exclude genuinely tall
        # CJK dialogue COLUMNS -- but a column is tall relative to its OWN
        # width by definition, so the ceiling should not apply to a shape
        # that is merely a same-line fragment set in a notably bigger/bolder
        # emphasis font (slightly wider than tall, e.g. new_sample_14's
        # "완성" at 279x236 next to "몰디브 7일"'s ~150px-tall glyphs -- it
        # never even reaches the ratio check below, which would have
        # correctly admitted it). This must stay narrow, though: a shape
        # that is DECISIVELY wider than tall is no longer "big emphasis
        # text" but a different kind of visual element entirely (verified
        # regression: external_ja_1's "白科事典恨", a 276x190 SFX/logo-style
        # blob already independently classified as sfx_artwork on its own,
        # became newly eligible under a bare height>width precondition and
        # dragged a genuine neighboring dialogue sentence into its SFX
        # group verdict once merged). The two measured cases calibrate the
        # band: 완성's width/height ratio is 1.18 (exempt); the SFX blob's
        # is 1.45 (still excluded, strict check still applies to it).
        emphasis_font_shape = height < width and width < height * 1.3
        if not emphasis_font_shape and height > max(150, int(img_h * 0.14) if img_h else 150):
            return False
        # This absolute-floor-protected ratio check already excludes
        # genuinely tall CJK dialogue COLUMNS (many characters stacked
        # vertically, e.g. new_sample_5's 150-270px columns) while still
        # admitting small reaction fragments like "えっ！？" (19x54px) that
        # are squarish only because they're SHORT, not because they're a
        # column -- a pure ratio-only guard (no floor) wrongly excluded the
        # latter (verified regression: external_ja_2's "えっ！？" no longer
        # merged with its neighboring line, losing the utterance).
        if height > max(96, int(width * 1.35)):
            return False
        return True

    def bounds(item: dict, key: str = "box") -> tuple[int, int, int, int]:
        box = item.get(key) or item.get("box", {})
        return (
            int(box.get("x1", 0)),
            int(box.get("y1", 0)),
            int(box.get("x2", box.get("x1", 0))),
            int(box.get("y2", box.get("y1", 0))),
        )

    def same_line(group_box: tuple[int, int, int, int], item_box: tuple[int, int, int, int]) -> bool:
        gx1, gy1, gx2, gy2 = group_box
        x1, y1, x2, y2 = item_box
        group_height = max(1, gy2 - gy1)
        item_height = max(1, y2 - y1)
        v_overlap = max(0, min(gy2, y2) - max(gy1, y1))
        center_gap = abs(((gy1 + gy2) / 2.0) - ((y1 + y2) / 2.0))
        if v_overlap < max(12, int(min(group_height, item_height) * 0.45)) and center_gap > max(18, int(max(group_height, item_height) * 0.38)):
            return False
        h_gap = max(0, max(gx1, x1) - min(gx2, x2))
        return h_gap <= max(72, int(min(group_height, item_height) * 1.10))

    indexed = [
        (index, item)
        for index, item in enumerate(ocr_data)
        if isinstance(item, dict) and eligible(item)
    ]
    indexed.sort(key=lambda pair: ((bounds(pair[1])[1] + bounds(pair[1])[3]) / 2.0, bounds(pair[1])[0]))
    consumed: set[int] = set()
    merged_by_index: dict[int, dict] = {}
    member_indexes: set[int] = set()

    for index, item in indexed:
        if index in consumed:
            continue
        group = [(index, item)]
        consumed.add(index)
        gx1, gy1, gx2, gy2 = bounds(item)
        # Regions Step 1's independent semantic detector has already matched
        # to a member of this group, by that member's OWN box (not the
        # growing merged bounds -- a wide merged box can spuriously overlap
        # a neighboring region it shouldn't). Two fragments that each match
        # a DIFFERENT, non-overlapping detector-seen dialogue region are two
        # distinct detector-confirmed utterances, not one split line -- do
        # not bridge them. Validated offline against all 33 samples' current
        # merge groups: every genuine merge either matches no semantic
        # region at all, or every member matches the SAME region; only the
        # external_ja_2 "えっ！？"/"ぶち壊しですね" cross-bubble case has two
        # distinct matches.
        group_regions: set[tuple[int, int, int, int]] = set()
        first_region = _matched_semantic_region((gx1, gy1, gx2, gy2), semantic_dialogue_boxes)
        if first_region is not None:
            group_regions.add(first_region)
        changed = True
        while changed:
            changed = False
            for other_index, other in indexed:
                if other_index in consumed:
                    continue
                ox1, oy1, ox2, oy2 = bounds(other)
                if not same_line((gx1, gy1, gx2, gy2), (ox1, oy1, ox2, oy2)):
                    continue
                candidate_width = max(gx2, ox2) - min(gx1, ox1)
                # A generous ceiling, not a routine blocker: with membership
                # now restricted to landscape fragments (above) and each
                # step's gap already tightly bounded by same_line(), a chain
                # reaching this wide is a genuine full-width caption/narration
                # line (new_sample_14's top band spans ~94% of the page width
                # across two OCR-split fragments), not cross-page bridging.
                if img_w and candidate_width > max(720, int(img_w * 0.97)):
                    continue
                cand_region = _matched_semantic_region((ox1, oy1, ox2, oy2), semantic_dialogue_boxes)
                if cand_region is not None and group_regions and cand_region not in group_regions:
                    continue
                group.append((other_index, other))
                consumed.add(other_index)
                if cand_region is not None:
                    group_regions.add(cand_region)
                gx1, gy1, gx2, gy2 = min(gx1, ox1), min(gy1, oy1), max(gx2, ox2), max(gy2, oy2)
                changed = True

        if len(group) <= 1:
            continue

        ordered = sorted(group, key=lambda pair: bounds(pair[1])[0])
        primary_index, primary_item = ordered[0]
        merged = dict(primary_item)
        merged_ids = [int(part.get("id", -1)) for _, part in ordered]
        merged["text"] = " ".join(str(part.get("text", "")).strip() for _, part in ordered if str(part.get("text", "")).strip())
        merged["line_fragment_ids"] = merged_ids
        fallback_source = str(merged.get("fallback_source") or "")
        merged["fallback_source"] = f"{fallback_source}+line_fragment_merge" if fallback_source else "line_fragment_merge"

        red_boxes = [bounds(part) for _, part in ordered]
        green_boxes = [bounds(part, "green_box") for _, part in ordered]
        rx1 = min(box[0] for box in red_boxes)
        ry1 = min(box[1] for box in red_boxes)
        rx2 = max(box[2] for box in red_boxes)
        ry2 = max(box[3] for box in red_boxes)
        gx1 = min(box[0] for box in green_boxes)
        gy1 = min(box[1] for box in green_boxes)
        gx2 = max(box[2] for box in green_boxes)
        gy2 = max(box[3] for box in green_boxes)
        merged_counts = _script_counts(merged["text"])
        if merged_counts["hangul"] >= 4 and re.search(r"[！？!?…]", merged["text"]):
            merged["force_full_line_cleanup"] = True
            line_height = max(1, ry2 - ry1)
            extra_right = max(20, min(72, int(line_height * 0.80)))
            extra_x = max(4, min(14, int(line_height * 0.18)))
            extra_y = max(3, min(12, int(line_height * 0.14)))
            rx1 = max(0, rx1 - extra_x)
            ry1 = max(0, ry1 - extra_y)
            rx2 = min(img_w or rx2 + extra_right, rx2 + extra_right)
            ry2 = min(img_h or ry2 + extra_y, ry2 + extra_y)
            gx1 = max(0, gx1 - extra_x)
            gy1 = max(0, gy1 - extra_y)
            gx2 = min(img_w or gx2 + extra_right, gx2 + extra_right)
            gy2 = min(img_h or gy2 + extra_y, gy2 + extra_y)
            green_boxes = [
                (
                    max(0, box[0] - extra_x),
                    max(0, box[1] - extra_y),
                    min(img_w or box[2] + extra_right, box[2] + (extra_right if idx == len(green_boxes) - 1 else extra_x)),
                    min(img_h or box[3] + extra_y, box[3] + extra_y),
                )
                for idx, box in enumerate(green_boxes)
            ]
        merged["box"] = {"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2, "width": rx2 - rx1, "height": ry2 - ry1}
        merged["green_box"] = {"x1": gx1, "y1": gy1, "x2": gx2, "y2": gy2, "width": gx2 - gx1, "height": gy2 - gy1}
        merged["green_polygon"] = [[gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]]
        merged["erase_boxes"] = [list(box) for box in green_boxes]
        merged_by_index[primary_index] = merged
        member_indexes.update(part_index for part_index, _ in ordered if part_index != primary_index)

    if not merged_by_index:
        return ocr_data

    output: list[dict] = []
    for index, item in enumerate(ocr_data):
        if index in member_indexes:
            continue
        output.append(merged_by_index.get(index, item))
    return output


def _sfx_artwork_signature(image: np.ndarray | None, box: dict, text: str = "") -> bool:
    """Anatomy-only detector for SFX lettering drawn as ARTWORK — fat brush
    strokes (ガタ) or hollow outlined display glyphs (は/はッ). Works from the
    pixels alone: monochrome brush SFX routinely OCRs into a plausible short
    dialogue fragment ('キャッ さっき'), so text-based gates cannot be trusted.
    Dialogue lettering (3–6 px strokes, solid cores) never matches; solid
    uniform-colour onomatopoeia (teal 噗通) stays translatable text.

    The `display_glyphs` path (below) has no brush/hollow anatomy check of
    its own -- it fires on ANY large, dense, roughly-square glyph run, which
    includes a perfectly ordinary bold display/title font. That is only a
    safe signal when OCR could not extract real words from it; when `text`
    is genuine translatable content (new_sample_14's "몰디브 7일 완성" title
    and "신혼특강" badge), it is solid lettering, not brush/hollow SFX art,
    and must translate like the Ichigo reference's solid onomatopoeia does."""
    if image is None:
        return False
    img_h, img_w = image.shape[:2]
    x1 = max(0, min(img_w, int(box.get("x1", 0))))
    y1 = max(0, min(img_h, int(box.get("y1", 0))))
    x2 = max(0, min(img_w, int(box.get("x2", x1))))
    y2 = max(0, min(img_h, int(box.get("y2", y1))))
    width = x2 - x1
    height = y2 - y1
    if width < 40 or height < 40:
        return False
    if max(width, height) < max(56, int(img_h * 0.045)):
        return False

    pad = 4
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(img_w, x2 + pad), min(img_h, y2 + pad)
    roi = image[ry1:ry2, rx1:rx2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturated = (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 40)
    # Saturated pixels that share the window border's dominant hue are the
    # BACKGROUND showing through (door wood, tatami, kimono) — counting them
    # as strokes makes any text box on colored art read as "fat brush".
    ring_pad = 10
    ox1, oy1 = max(0, rx1 - ring_pad), max(0, ry1 - ring_pad)
    ox2, oy2 = min(img_w, rx2 + ring_pad), min(img_h, ry2 + ring_pad)
    outer = image[oy1:oy2, ox1:ox2]
    outer_hsv = cv2.cvtColor(outer, cv2.COLOR_BGR2HSV)
    ring = np.ones(outer_hsv.shape[:2], dtype=bool)
    ring[ry1 - oy1:ry2 - oy1, rx1 - ox1:rx2 - ox1] = False
    ring_sat = ring & (outer_hsv[:, :, 1] >= 90) & (outer_hsv[:, :, 2] >= 40)
    if int(np.count_nonzero(ring_sat)) >= 60:
        hist = np.bincount(
            outer_hsv[:, :, 0][ring_sat].astype(np.int32), minlength=180
        )
        ring_hue = int(np.argmax(hist))
        if float(hist[ring_hue]) / float(np.count_nonzero(ring_sat)) >= 0.25:
            hue = hsv[:, :, 0].astype(np.int32)
            hue_dist = np.minimum(
                np.abs(hue - ring_hue), 180 - np.abs(hue - ring_hue)
            )
            saturated &= hue_dist > 12
    strokes = ((gray < 110) | saturated).astype(np.uint8)
    strokes = cv2.morphologyEx(
        strokes,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    stroke_count = int(np.count_nonzero(strokes))
    area = max(1, strokes.shape[0] * strokes.shape[1])
    if stroke_count < 400:
        return False
    stroke_fraction = stroke_count / float(area)
    if stroke_fraction < 0.05 or stroke_fraction > 0.60:
        return False

    # Display-scale lettering: SFX/title glyphs are individually huge
    # (150–200 px components); dialogue glyphs are 30–45 px even in bold
    # columns. Structural elements passing THROUGH the box (panel frames,
    # door bars — they touch both opposite window borders) and thin rules
    # are not glyphs and must not feed any metric.
    comp_count, comp_labels, comp_stats, _ = cv2.connectedComponentsWithStats(
        strokes, connectivity=8
    )
    win_h, win_w = strokes.shape[:2]
    kept = np.zeros_like(strokes)
    biggest_glyph = 0
    small_glyphs = 0
    small_area = 0
    large_area = 0
    display_glyphs = 0
    small_cap = max(50, int(img_h * 0.03))
    for comp in range(1, comp_count):
        c_x = int(comp_stats[comp, cv2.CC_STAT_LEFT])
        c_y = int(comp_stats[comp, cv2.CC_STAT_TOP])
        c_w = int(comp_stats[comp, cv2.CC_STAT_WIDTH])
        c_h = int(comp_stats[comp, cv2.CC_STAT_HEIGHT])
        c_area = int(comp_stats[comp, cv2.CC_STAT_AREA])
        if c_area < 120:
            continue
        long_axis = max(c_w, c_h)
        short_axis = max(1, min(c_w, c_h))
        spans_vertical = c_y <= 1 and c_y + c_h >= win_h - 2
        spans_horizontal = c_x <= 1 and c_x + c_w >= win_w - 2
        if (spans_vertical or spans_horizontal) and long_axis / short_axis >= 3.0:
            continue
        if long_axis / short_axis >= 6.0 and short_axis <= 16:
            continue
        if long_axis / short_axis >= 4.0:
            # A straight uniform bar (door frame, panel rule segment cut by
            # overlapping glyphs) fills its rotated bounding rect densely;
            # brush strokes are wobbly and sparse in theirs.
            comp_mask = (comp_labels == comp).astype(np.uint8)
            comp_points = cv2.findNonZero(comp_mask)
            if comp_points is not None:
                (_, (rect_w, rect_h), _) = cv2.minAreaRect(comp_points)
                rect_area = max(1.0, float(rect_w) * float(rect_h))
                if c_area / rect_area >= 0.68:
                    continue
        kept[comp_labels == comp] = 1
        if 12 <= long_axis <= small_cap:
            small_glyphs += 1
            small_area += c_area
        else:
            large_area += c_area
        biggest_glyph = max(biggest_glyph, long_axis)
        # A solid display glyph: large, roughly square, dense — one letter
        # of poster/title lettering. Dialogue glyph chains are elongated and
        # sparse; dialogue letters are far smaller.
        if (
            c_area >= 1200
            and short_axis >= 36
            and long_axis / short_axis <= 2.2
            and c_area / float(max(1, c_w * c_h)) >= 0.22
        ):
            display_glyphs += 1
    if biggest_glyph < max(64, int(img_h * 0.05)):
        return False
    # Dialogue-sized components carrying real mass next to the big one mean
    # this is a text RUN (a bold column can chain into one tall component);
    # SFX boxes hold a few huge glyphs plus at most stray specks.
    if small_glyphs >= 4 and small_area >= int(large_area * 0.25):
        return False

    strokes = kept
    on = strokes > 0
    if int(np.count_nonzero(on)) < 400:
        return False
    dist = cv2.distanceTransform(strokes, cv2.DIST_L2, 3)
    thick_p70 = float(np.percentile(dist[on], 70))
    thick_p90 = float(np.percentile(dist[on], 90))

    closed = cv2.morphologyEx(
        strokes,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    holes = (closed > 0) & ~on
    hole_count = int(np.count_nonzero(holes))
    hole_ratio = hole_count / float(max(1, stroke_count))
    bright_holes = (
        float(np.mean(gray[holes] >= 195)) if hole_count >= 60 else 0.0
    )

    fat_brush = thick_p70 >= max(5.5, img_h * 0.0045)
    hollow_brush = (
        thick_p90 >= max(4.0, img_h * 0.003)
        and hole_ratio >= 0.25
        and bright_holes >= 0.55
    )
    # A thin glossy/gradient outline font (new_sample_13's cyan "吡啦…" SFX)
    # has real hollow-outline anatomy (hole_ratio 0.57, bright_holes 0.87)
    # but fails hollow_brush's thickness floor (thick_p90 3.28 < 4.0) -- it
    # is decorative art, not a thickness measurement artifact. hole_ratio
    # and bright_holes alone can't safely stand in for the thickness floor
    # though: thin ORDINARY katakana (new_sample_15's plain black "ラック")
    # produces near-identical hole/bright values (0.98, 0.79) purely because
    # its sparse strokes leave lots of enclosed white space -- not because
    # it's stylized. What cleanly separates the two is COLOR: dialogue ink
    # is plain black, but decorative SFX/title lettering is almost always
    # rendered in a distinct saturated color to read as "effect", not text
    # (this pipeline already special-cases colored lettering elsewhere --
    # green narration stays green, cyan SFX glosses stay cyan). Requiring
    # the strokes to be mostly saturated color, not just hollow, keeps this
    # a strict superset of hollow_brush rather than a blanket thickness
    # loosening.
    saturated_stroke_fraction = float(np.mean(saturated[on])) if int(np.count_nonzero(on)) else 0.0
    colored_hollow = (
        saturated_stroke_fraction >= 0.50
        and hole_ratio >= 0.25
        and bright_holes >= 0.55
    )
    return (
        fat_brush
        or hollow_brush
        or colored_hollow
        or (display_glyphs >= 2 and not _has_usable_translation(text))
    )


def _floating_roi_has_dominant_sfx_art(item: dict, image: np.ndarray | None) -> bool:
    """Reject mixed floating OCR boxes dominated by a large SFX glyph.

    These boxes can contain a few readable dialogue characters, but the visual
    region is mostly a large stylized SFX. Treating the whole OCR box as normal
    floating dialogue forces Step 4 to reconstruct artwork under the SFX, which
    is exactly the failure mode seen in difficult web samples.
    """

    if image is None or item.get("bubble_idx", -1) != -1:
        return False

    text = str(item.get("text", "")).strip()
    compact_text_len = len(_normalized_cjk_fragment(text))
    if compact_text_len >= 18:
        return False

    box = item.get("box", {})
    img_h, img_w = image.shape[:2]
    x1 = max(0, min(img_w, int(box.get("x1", 0))))
    y1 = max(0, min(img_h, int(box.get("y1", 0))))
    x2 = max(0, min(img_w, int(box.get("x2", x1))))
    y2 = max(0, min(img_h, int(box.get("y2", y1))))
    if x2 <= x1 or y2 <= y1:
        return False

    width = x2 - x1
    height = y2 - y1
    area = width * height
    page_area = max(1, img_w * img_h)
    if area < max(90000, int(page_area * 0.032)):
        return False

    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    dark = (gray < 76).astype(np.uint8) * 255
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if labels_count <= 1:
        return False

    largest_area = 0
    largest_width = 0
    largest_height = 0
    for label in range(1, labels_count):
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if component_area > largest_area:
            largest_area = component_area
            largest_width = component_width
            largest_height = component_height

    dark_fraction = float(np.mean(dark > 0))
    component_area_ratio = largest_area / float(max(1, area))
    component_width_ratio = largest_width / float(max(1, width))
    component_height_ratio = largest_height / float(max(1, height))
    dominant_graphic = (
        component_area_ratio >= 0.035
        and component_width_ratio >= 0.34
        and component_height_ratio >= 0.20
    )
    if not dominant_graphic:
        return False

    counts = _script_counts(text)
    has_short_dialogue_fragment = counts["kana"] + counts["hangul"] + counts["han"] >= 3
    return dark_fraction >= 0.055 and has_short_dialogue_fragment


def _recoverable_adjacent_fragment(item: dict, constraints: list[dict], image: np.ndarray | None) -> bool:
    text = str(item.get("text", "")).strip()
    if not text:
        return False
    counts = _script_counts(text)
    cjk_total = counts["hangul"] + counts["kana"] + counts["han"]
    compact_len = len("".join(c for c in text if c.isalnum()))
    if cjk_total < 2 or compact_len < 2:
        return False
    has_dialogue_script = counts["han"] >= 1 or re.search(r"[぀-ゟ]", text) or compact_len >= 5

    box = item.get("box", {})
    x1 = int(box.get("x1", 0))
    y1 = int(box.get("y1", 0))
    x2 = int(box.get("x2", x1))
    y2 = int(box.get("y2", y1))
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    image_shape = image.shape if image is not None else None
    img_h = image_shape[0] if image_shape is not None else 0
    img_w = image_shape[1] if image_shape is not None else 0
    needs_tight_anchor = False
    if img_h and img_w:
        min_readable_width = max(28, int(img_w * 0.025))
        min_readable_height = max(72, int(img_h * 0.070))
        if width < min_readable_width and height < min_readable_height and compact_len <= 5:
            needs_tight_anchor = True
    cls = classify_text_by_content(text)
    pale_fraction = _pale_region_fraction(image, (x1, y1, x2, y2))
    dark_fraction = _dark_region_fraction(image, (x1, y1, x2, y2))
    pale_fragment = pale_fraction >= 0.42
    top_pale_fragment = bool(img_h and y1 <= int(img_h * 0.26) and pale_fragment)
    safe_caption_fragment = pale_fraction >= 0.55 and dark_fraction <= 0.24
    erase_only_fragment = _erase_only_adjacent_fragment(item, image)

    if cls == "english":
        return False
    # A correctly-identified SFX element is its own distinct page element,
    # never incidental debris belonging to an adjacent dialogue bubble --
    # unlike genuine "noise" (stray glyph fragments from the SAME bubble's
    # own text), no amount of erase-only eligibility should let it be
    # donated to a neighbor's erase_boxes.
    if cls == "sfx":
        return False
    if cls == "noise" and not erase_only_fragment:
        return False

    if cls != "dialogue" and not (top_pale_fragment or safe_caption_fragment or erase_only_fragment):
        return False

    if width > max(92, int(height * 0.92)):
        return False
    if img_w and width > int(img_w * 0.16):
        return False

    center_y = (y1 + y2) / 2.0
    for constraint in constraints:
        if constraint.get("bubble_idx", -1) != -1:
            continue
        rb = constraint.get("red_box", [0, 0, 0, 0])
        rx1, ry1, rx2, ry2 = [int(v) for v in rb]
        r_width = max(1, rx2 - rx1)
        r_height = max(1, ry2 - ry1)
        if r_width > max(120, int(r_height * 1.20)):
            continue
        h_gap = max(0, max(x1, rx1) - min(x2, rx2))
        v_gap = max(0, max(y1, ry1) - min(y2, ry2))
        v_overlap = max(0, min(y2, ry2) - max(y1, ry1))
        h_overlap = max(0, min(x2, rx2) - max(x1, rx1))
        same_vertical_band = v_overlap >= max(8, int(min(height, r_height) * 0.20))
        close_y = abs(center_y - ((ry1 + ry2) / 2.0)) <= max(92, int(max(height, r_height) * 0.90))
        close_x = h_gap <= max(96, int(min(height, r_height) * 1.15))
        if needs_tight_anchor:
            tight_x = h_gap <= max(18, int(min(width, r_width) * 0.45)) or h_overlap >= max(8, int(min(width, r_width) * 0.30))
            tight_y = v_gap <= max(18, int(min(height, r_height) * 0.35)) or v_overlap >= max(12, int(min(height, r_height) * 0.22))
            if not (tight_x and tight_y):
                continue
        if close_x and (same_vertical_band or (close_y and v_gap <= max(36, int(max(height, r_height) * 0.28)))):
            if erase_only_fragment:
                return True
            if has_dialogue_script or safe_caption_fragment:
                return True
            if top_pale_fragment and img_h and ry1 <= int(img_h * 0.25):
                return True

    return False


def _layout_rejection(item: dict, reason: str, semantic_role: str | None = None, classification: str | None = None) -> dict:
    box = item.get("box", {})
    return {
        "id": item.get("id"),
        "text": item.get("text", ""),
        "reason": reason,
        "semantic_role": semantic_role,
        "classification": classification,
        "bubble_idx": item.get("bubble_idx", -1),
        "route": item.get("route", ""),
        "box": {
            "x1": box.get("x1", 0),
            "y1": box.get("y1", 0),
            "x2": box.get("x2", 0),
            "y2": box.get("y2", 0),
        },
    }


def _needs_inferred_bubble_cleanup(
    item: dict,
    image_shape,
    image: np.ndarray | None,
    semantic_role: str | None = None,
) -> bool:
    if image is None or image_shape is None:
        return False
    if item.get("bubble_idx", -1) != -1:
        return False
    if semantic_role == "sfx":
        # SFX glyphs overlaid directly on artwork have no real container --
        # tracing a "bubble" contour around them just follows whatever
        # happens to be nearby (hair, clothing, panel art), producing a
        # meaningless boundary that can misroute Step 4's cleanup mask.
        return False
    ocr_provider = str(item.get("ocr_provider") or "")
    fallback_source = str(item.get("fallback_source") or "")
    text = str(item.get("text") or "")
    korean_paddle = fallback_source.startswith("paddleocr_korean") and ocr_provider.startswith("paddleocr_ko")
    # This used to also require chinese_paddle specifically -- but
    # the actual outline tracer below (_infer_missed_bubble_outline) is
    # shape-agnostic contour tracing, not a Japanese-round-bubble fit, so
    # restricting it to two narrow OCR-provider code paths meant rectangular
    # manhua/manhwa caption boxes (which the bubble segmentation model was
    # never trained to recognize at all -- it's a single-class "balloon"
    # detector on Manga109) never got a blue outline from any OCR provider.
    # The pale/dark-fraction and size checks below already gate on whether
    # the region actually looks like a flat-background dialogue container,
    # so they remain the real safety net once the provider restriction is
    # dropped.
    box = item.get("box", {})
    img_h = int(image_shape[0])
    img_w = int(image_shape[1])
    x1 = int(box.get("x1", 0))
    y1 = int(box.get("y1", 0))
    x2 = int(box.get("x2", x1))
    y2 = int(box.get("y2", y1))
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    if korean_paddle and y1 <= int(img_h * 0.20):
        return False
    if height > max(180, int(img_h * 0.22)):
        return False
    pale_fraction = _pale_region_fraction(image, (x1, y1, x2, y2))
    dark_fraction = _dark_region_fraction(image, (x1, y1, x2, y2))
    compact = _normalized_cjk_fragment(text)
    short_pale_dialogue = (
        2 <= len(compact) <= 8
        and _has_usable_translation(text)
        and pale_fraction >= 0.72
        and dark_fraction <= 0.18
        and width >= max(42, int(img_w * 0.045))
        and height <= max(96, int(img_h * 0.035))
    )
    # A single vertical column of a reversed-polarity (dark-fill) bubble is
    # narrow but TALL, not short -- new_sample_5's black bubbles OCR into
    # one column per box (39x229px), nowhere near the 120px width floor
    # below. short_pale_dialogue's shape (small width AND small height)
    # doesn't fit a column, so it needs its own exemption.
    narrow_dark_column = (
        len(compact) >= 1
        and _has_usable_translation(text)
        and dark_fraction >= 0.55
        and pale_fraction <= 0.20
        and height >= width * 1.3
        and width >= max(20, int(img_w * 0.015))
    )
    # Tried removing this width floor (2026-07-04) to let a narrow single
    # text LINE inside a wider multi-line bubble reach _container_flood_
    # outline instead of being pre-declined (original/sample3 id7: a real
    # oval bubble whose own line box is 105px wide never got a rescue
    # attempt). Reverted: the flood still failed on that exact case (it
    # leaks straight through the bubble's thin hand-drawn outline into a
    # same-tone panel background -- a real limitation of the barrier
    # thickness, not this gate), while the gate removal DID change
    # fallback_source tagging (and therefore Step 4's reconstruction
    # routing, see allow_inferred_bubble_cleanup) for every other narrow
    # floating item suite-wide with no confirmed benefit. A change that
    # doesn't fix its own target case shouldn't stay on vibes. id7 itself
    # renders correctly as-is (white bubble on white panel -- the "floating
    # text on art" treatment is visually indistinguishable from a proper
    # bubble fill when interior and surround are the same tone), so this is
    # a debug-overlay/classification-accuracy gap, not a page defect --
    # revisit only alongside a deliberate barrier-thickness change to
    # _container_flood_outline, verified full-suite, if ever.
    if width < max(120, int(img_w * 0.10)) and not (short_pale_dialogue or narrow_dark_column):
        return False
    if (pale_fraction >= 0.58 and dark_fraction <= 0.22) or short_pale_dialogue:
        return True
    # Reversed-polarity container: light glyphs on a dark/black bubble fill
    # (e.g. new_sample_5's black speech bubbles). Mirrors the light-bubble
    # gate above with the fractions swapped.
    return (dark_fraction >= 0.58 and pale_fraction <= 0.22) or narrow_dark_column


def _refine_constraint_geometry(
    constraint: dict,
    text_mask: np.ndarray | None,
    image_shape: tuple,
    image: np.ndarray | None = None,
) -> None:
    """Final geometry polish, mutating the constraint in place:

    * red_box shrinks to tightly hug the detected glyph strokes (bbox of the
      Step-1 text mask inside the box, +4px), instead of the loose OCR line
      box that can span a whole bubble;
    * for inferred-bubble constraints, green_box/green_polygon are derived
      from the bubble outline inset a few pixels (the typeset area is the
      bubble interior), instead of a red-box expansion that can poke outside
      the bubble."""
    height, width = image_shape[:2]
    rx1, ry1, rx2, ry2 = [int(v) for v in constraint["red_box"]]
    if text_mask is not None:
        wx1, wy1 = max(0, rx1), max(0, ry1)
        wx2, wy2 = min(width, rx2), min(height, ry2)
        if wx2 > wx1 and wy2 > wy1:
            window = text_mask[wy1:wy2, wx1:wx2]
            ys, xs = np.where(window > 0)
            if len(xs) >= 30:
                pad = 4
                tight = [
                    max(0, wx1 + int(xs.min()) - pad),
                    max(0, wy1 + int(ys.min()) - pad),
                    min(width, wx1 + int(xs.max()) + 1 + pad),
                    min(height, wy1 + int(ys.max()) + 1 + pad),
                ]
                tight_area = (tight[2] - tight[0]) * (tight[3] - tight[1])
                if 0 < tight_area < (rx2 - rx1) * (ry2 - ry1):
                    constraint["red_box"] = tight

    # Real bubbles: the detector mask often hugs the whitened glyph areas
    # instead of the bubble/box boundary. Re-trace the container from the
    # (tight) red box; when a bounded container is found it becomes the blue
    # outline, the green inset, and Step 4's cleanup boundary.
    if constraint.get("bubble_idx", -1) >= 0 and image is not None:
        refined = _container_flood_outline(image, constraint["red_box"])
        if refined:
            constraint["refined_bubble_outline"] = refined

    outline = (
        constraint.get("inferred_bubble_outline")
        or constraint.get("refined_bubble_outline")
        or []
    )
    if len(outline) >= 3:
        raster = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(raster, [np.array(outline, dtype=np.int32)], 255)
        inset = cv2.erode(
            raster, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        )
        ys, xs = np.where(inset > 0)
        if len(xs) >= 100:
            tx1, ty1, tx2, ty2 = [int(v) for v in constraint["red_box"]]
            constraint["green_box"] = [
                min(int(xs.min()), tx1),
                min(int(ys.min()), ty1),
                max(int(xs.max()) + 1, tx2),
                max(int(ys.max()) + 1, ty2),
            ]
            contours, _ = cv2.findContours(
                inset, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                constraint["green_polygon"] = [
                    [int(point[0][0]), int(point[0][1])] for point in largest
                ]


def _same_utterance_vertical_columns(
    member_boxes: list[tuple[int, int, int, int]],
    img_w: int,
) -> bool:
    """True when a set of red boxes reads as side-by-side vertical CJK
    columns of ONE utterance inside one bubble (new_sample_13's three-column
    dialogue), as opposed to separate utterances in touching bubbles
    (new_sample_11's stacked horizontal-text lobes, which must stay split).

    Mirrors `_should_merge_chinese_bubble_columns` (run_step5_ocr.py), which
    already encodes this exact signal for bubble-assigned columns; this
    variant works on constraint red boxes for the floating/traced-container
    case. Same-bubble columns sit close together (gap well under a column
    width) with near-full vertical overlap; touching-bubble text has the
    bubble walls between the boxes (bigger gap) and/or is landscape-shaped."""
    if len(member_boxes) < 2:
        return False
    heights = []
    widths = []
    for bx1, by1, bx2, by2 in member_boxes:
        w = max(1, bx2 - bx1)
        h = max(1, by2 - by1)
        if h < w * 1.1:
            return False
        heights.append(h)
        widths.append(w)
    for i in range(len(member_boxes)):
        for j in range(i + 1, len(member_boxes)):
            a, b = member_boxes[i], member_boxes[j]
            y_overlap = min(a[3], b[3]) - max(a[1], b[1])
            if y_overlap < 0.5 * min(heights[i], heights[j]):
                return False
    ordered = sorted(member_boxes, key=lambda box: box[0])
    median_width = float(np.median(widths))
    for left, right in zip(ordered, ordered[1:]):
        gap = right[0] - left[2]
        if gap > max(72, int(median_width * 1.8)):
            return False
    merged_width = max(b[2] for b in member_boxes) - min(b[0] for b in member_boxes)
    if merged_width > int(img_w * 0.28):
        return False
    return True


def _rescue_strokes_belong_to_container(
    image: np.ndarray | None,
    probe_box: list[int],
    flood_points: list[list[int]],
) -> bool:
    """Reject an enclosed_dialogue_rescue candidate when its own dark strokes are
    physically connected to the traced container's boundary ring, rather than
    sitting isolated in the interior.

    A genuinely missed speech bubble's text is drawn INSIDE the bubble, with
    clearance from the bubble's own ink outline. A character's blank head
    silhouette (or other plain-enclosed art -- a bandaged limb, a sign panel)
    can flood-trace as a valid-looking container too, but any "text" OCR finds
    there is either the container's own boundary stroke (a hair tuft that IS
    the head outline) or directly fused to it -- there is no real gap between
    glyph and wall. Checked only at the enclosed_dialogue_rescue call site, not
    inside _container_flood_outline/_infer_missed_bubble_outline themselves,
    which also serve already-accepted dialogue where this signal doesn't apply
    (verified 2026-07-14: 7/7 flips on the full regression suite were confirmed
    false positives -- bandage wrap, store signage, character skin, speed
    lines, an SFX tail mark, and the original head-as-bubble case -- while the
    2 genuine rescues, both real text with interior clearance, were unaffected)."""
    if image is None or len(flood_points) < 3:
        return False
    poly = np.array(flood_points, dtype=np.int32)
    px1 = max(0, int(poly[:, 0].min()) - 4)
    py1 = max(0, int(poly[:, 1].min()) - 4)
    px2 = min(image.shape[1], int(poly[:, 0].max()) + 5)
    py2 = min(image.shape[0], int(poly[:, 1].max()) + 5)
    if px2 - px1 < 8 or py2 - py1 < 8:
        return False
    window = image[py1:py2, px1:px2]
    gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
    local = poly - np.array([[px1, py1]])

    boundary = np.zeros(gray.shape, dtype=np.uint8)
    cv2.polylines(boundary, [local], True, 255, 1)
    boundary_ring = cv2.dilate(
        boundary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ) > 0

    x1, y1, x2, y2 = [int(v) for v in probe_box[:4]]
    bx1 = max(0, x1 - 2 - px1)
    by1 = max(0, y1 - 2 - py1)
    bx2 = min(gray.shape[1], x2 + 2 - px1)
    by2 = min(gray.shape[0], y2 + 2 - py1)
    if bx2 <= bx1 or by2 <= by1:
        return False
    probe_mask = np.zeros(gray.shape, dtype=bool)
    probe_mask[by1:by2, bx1:bx2] = True

    dark = (gray < 110).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(dark, connectivity=8)
    for label in range(1, n_labels):
        comp = labels == label
        if np.any(comp & probe_mask) and np.any(comp & boundary_ring):
            return True
    return False


def _container_flood_outline(
    image: np.ndarray | None,
    red_box: list[int],
    max_area_ratio: float = 9.0,
    dark_interior: bool = False,
) -> list[list[int]]:
    """Trace the CONTAINER that encloses a text region, not the text itself.

    Floods outward from the text seed through non-dark pixels; the container's
    own line art (box frame, bubble outline, the shared wall between connected
    bubbles) bounds the flood, so the contour follows the container boundary
    and connected compartments split naturally at their wall. Returns [] when
    the flood leaks to the search-window border (text is unenclosed).

    `dark_interior=True` traces the reversed-polarity case instead: light
    (white/pale) glyphs on a dark or black bubble fill. The barrier and the
    free-space polarity both flip so the flood travels through the dark fill
    the same way it travels through a light one (verified against
    new_sample_5's solid-black speech bubbles, which the default light-mode
    barrier -- dark pixels -- cannot flood through at all)."""
    if image is None or len(red_box) < 4:
        return []
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in red_box[:4]]
    if x2 <= x1 or y2 <= y1:
        return []
    width = x2 - x1
    height = y2 - y1
    flood_pad = max(60, min(220, int(max(width, height) * 0.9)))
    fx1 = max(0, x1 - flood_pad)
    fy1 = max(0, y1 - flood_pad)
    fx2 = min(img_w, x2 + flood_pad)
    fy2 = min(img_h, y2 + flood_pad)
    if fx2 <= fx1 or fy2 <= fy1:
        return []
    flood_gray = cv2.cvtColor(image[fy1:fy2, fx1:fx2], cv2.COLOR_BGR2GRAY)
    if dark_interior:
        barrier = cv2.dilate(
            (flood_gray > 150).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
    else:
        barrier = cv2.dilate(
            (flood_gray < 100).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
    free = (barrier == 0).astype(np.uint8)
    _, flood_labels = cv2.connectedComponents(free, connectivity=4)
    sx1 = max(0, x1 - fx1 + max(2, int(width * 0.05)))
    sy1 = max(0, y1 - fy1 + max(2, int(height * 0.05)))
    sx2 = min(flood_labels.shape[1], x2 - fx1 - max(2, int(width * 0.05)))
    sy2 = min(flood_labels.shape[0], y2 - fy1 - max(2, int(height * 0.05)))
    if sx2 <= sx1 or sy2 <= sy1:
        return []
    seed_labels = flood_labels[sy1:sy2, sx1:sx2]
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size < 40:
        return []
    label_ids, label_counts = np.unique(seed_labels, return_counts=True)
    container_label = int(label_ids[int(np.argmax(label_counts))])
    region = (flood_labels == container_label).astype(np.uint8)

    # A leak into the search window's border only disqualifies the region
    # when it escapes through an interior gap; a border that coincides with
    # the PAGE's own physical edge is a bubble/panel legitimately bleeding
    # off the page, not a failed containment (verified: new_sample_5's black
    # speech bubbles are drawn flush against the page's left edge).
    border_segments = []
    if fy1 != 0:
        border_segments.append(region[0, :])
    if fy2 != img_h:
        border_segments.append(region[-1, :])
    if fx1 != 0:
        border_segments.append(region[:, 0])
    if fx2 != img_w:
        border_segments.append(region[:, -1])
    leak = float(np.mean(np.concatenate(border_segments))) if border_segments else 0.0
    region_area = int(np.count_nonzero(region))
    red_area = max(1, width * height)
    if leak > 0.02 or not (red_area * 0.85 <= region_area <= red_area * max_area_ratio):
        return []
    # A genuine bubble/caption interior is PLAIN once the glyph area is
    # excluded. Panel frames also enclose flood regions (art, screentone,
    # doors, characters) — those "containers" must be rejected or floating
    # dialogue gets a phantom blue outline and Step 4 erases artwork inside
    # it (new_sample_5 floating text, new_sample_13 噗通 door region).
    interior = cv2.erode(
        region,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=2,
    ).astype(bool)
    glyph_pad = 6
    gx1 = max(0, x1 - fx1 - glyph_pad)
    gy1 = max(0, y1 - fy1 - glyph_pad)
    gx2 = min(interior.shape[1], x2 - fx1 + glyph_pad)
    gy2 = min(interior.shape[0], y2 - fy1 + glyph_pad)
    if gx2 > gx1 and gy2 > gy1:
        interior[gy1:gy2, gx1:gx2] = False
    if int(np.count_nonzero(interior)) >= 200:
        interior_gray = flood_gray[interior].astype(np.float32)
        edge_fraction = float(np.mean((cv2.Canny(flood_gray, 45, 135) > 0)[interior]))
        if edge_fraction > 0.07 or float(np.std(interior_gray)) > 34.0:
            return []
    region = cv2.morphologyEx(
        region * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    best = max(contours, key=cv2.contourArea)
    epsilon = max(1.5, min(6.0, cv2.arcLength(best, True) * 0.003))
    approx = cv2.approxPolyDP(best, epsilon, True)
    points = []
    for point in approx.reshape(-1, 2):
        px = int(point[0]) + fx1
        py = int(point[1]) + fy1
        points.append([max(0, min(img_w - 1, px)), max(0, min(img_h - 1, py))])
    return points if len(points) >= 3 else []


def _validate_container_outline(
    image: np.ndarray,
    red_box: list[int],
    points: list[list[int]],
) -> bool:
    """Heuristic outlines (pale-region / dark-lines fallbacks) must behave
    like real containers before they may become a blue outline:

    * the interior (minus the glyph box) is plain — a bubble/caption interior,
      not screentone, wood grain or a character;
    * the polygon boundary runs along dark ink for most of its length — real
      bubbles and caption frames are drawn, ambient bright regions are not.

    Without this, floating dialogue on a bright wall gets a phantom container
    (new_sample_13 噗通; new_sample_5 floating columns)."""
    if image is None or len(points) < 3:
        return False
    img_h, img_w = image.shape[:2]
    poly = np.array(points, dtype=np.int32)
    px1 = max(0, int(poly[:, 0].min()) - 4)
    py1 = max(0, int(poly[:, 1].min()) - 4)
    px2 = min(img_w, int(poly[:, 0].max()) + 5)
    py2 = min(img_h, int(poly[:, 1].max()) + 5)
    if px2 - px1 < 12 or py2 - py1 < 12:
        return False
    window = image[py1:py2, px1:px2]
    gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
    local = poly - np.array([[px1, py1]])
    raster = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(raster, [local], 255)

    interior = cv2.erode(
        raster, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ).astype(bool)
    x1, y1, x2, y2 = [int(v) for v in red_box[:4]]
    gx1 = max(0, x1 - 6 - px1)
    gy1 = max(0, y1 - 6 - py1)
    gx2 = min(gray.shape[1], x2 + 6 - px1)
    gy2 = min(gray.shape[0], y2 + 6 - py1)
    if gx2 > gx1 and gy2 > gy1:
        interior[gy1:gy2, gx1:gx2] = False
    if int(np.count_nonzero(interior)) >= 200:
        if float(np.mean((cv2.Canny(gray, 45, 135) > 0)[interior])) > 0.07:
            return False
        if float(np.std(gray[interior].astype(np.float32))) > 34.0:
            return False

    boundary = np.zeros(gray.shape, dtype=np.uint8)
    cv2.polylines(boundary, [local], True, 255, 1)
    boundary_points = boundary > 0
    if int(np.count_nonzero(boundary_points)) < 24:
        return False
    dark_near = cv2.dilate(
        (gray < 110).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ).astype(bool)
    dark_fraction = float(np.mean(dark_near[boundary_points]))
    return dark_fraction >= 0.70


def _infer_missed_bubble_outline(
    image: np.ndarray | None,
    red_box: list[int],
    dark_interior: bool = False,
) -> list[list[int]]:
    if image is None or len(red_box) < 4:
        return []
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in red_box[:4]]
    if x2 <= x1 or y2 <= y1:
        return []
    width = x2 - x1
    height = y2 - y1

    # Primary strategy: trace the CONTAINER, not the text (see
    # _container_flood_outline). Falls through to the legacy pale-region
    # heuristics below when the text is unenclosed.
    flood_points = _container_flood_outline(image, [x1, y1, x2, y2], dark_interior=dark_interior)
    if flood_points:
        return flood_points
    if dark_interior:
        # The remaining fallbacks below assume a light/pale bubble fill with
        # dark ink -- the opposite polarity -- so they would either find
        # nothing or trace the wrong shape for a dark-filled container.
        return []

    pad_x = max(25, min(70, int(width * 0.08)))
    pad_top = max(30, min(70, int(height * 0.60)))
    pad_bottom = max(45, min(90, int(height * 0.90)))
    roi_x1 = max(0, x1 - pad_x)
    roi_y1 = max(0, y1 - pad_top)
    roi_x2 = min(img_w, x2 + pad_x)
    roi_y2 = min(img_h, y2 + pad_bottom)
    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return []

    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
    red_area = max(1, width * height)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    pale_fill = ((gray > 170) & (hsv[:, :, 1] < 82)).astype(np.uint8) * 255
    pale_fill = cv2.morphologyEx(
        pale_fill,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    seed = np.zeros(pale_fill.shape, dtype=np.uint8)
    seed_x1 = max(0, x1 - roi_x1 + max(2, int(width * 0.02)))
    seed_y1 = max(0, y1 - roi_y1 + max(2, int(height * 0.03)))
    seed_x2 = min(pale_fill.shape[1], x2 - roi_x1 - max(2, int(width * 0.02)))
    seed_y2 = min(pale_fill.shape[0], y2 - roi_y1 - max(2, int(height * 0.03)))
    if seed_x2 > seed_x1 and seed_y2 > seed_y1:
        seed_region = pale_fill[seed_y1:seed_y2, seed_x1:seed_x2] > 0
        bright_region = gray[seed_y1:seed_y2, seed_x1:seed_x2] > 185
        seed[seed_y1:seed_y2, seed_x1:seed_x2][seed_region & bright_region] = 255
    seed_count = int(np.count_nonzero(seed))
    if seed_count >= max(28, int(red_area * 0.004)):
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(pale_fill, 8)
        best_label = -1
        best_score = -1.0
        for label in range(1, labels_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < max(220, int(red_area * 0.18)):
                continue
            if area > int(red_area * 3.2):
                continue
            cx = int(stats[label, cv2.CC_STAT_LEFT])
            cy = int(stats[label, cv2.CC_STAT_TOP])
            cw = int(stats[label, cv2.CC_STAT_WIDTH])
            ch = int(stats[label, cv2.CC_STAT_HEIGHT])
            if cw < max(70, int(width * 0.38)) or ch < max(36, int(height * 0.34)):
                continue
            component = labels == label
            overlap = int(np.count_nonzero(component & (seed > 0)))
            if overlap < max(28, int(red_area * 0.004)):
                continue
            component_seed_ratio = overlap / max(1, seed_count)
            if component_seed_ratio < 0.36:
                continue
            score = overlap + area * 0.04
            if score > best_score:
                best_score = score
                best_label = label
        if best_label >= 0:
            component_mask = (labels == best_label).astype(np.uint8) * 255
            component_mask = cv2.morphologyEx(
                component_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                best_contour = max(contours, key=cv2.contourArea)
                contour_area = float(cv2.contourArea(best_contour))
                if contour_area >= max(220.0, float(red_area) * 0.18):
                    epsilon = max(1.5, min(7.0, cv2.arcLength(best_contour, True) * 0.004))
                    outline = cv2.approxPolyDP(best_contour, epsilon, True)
                    points: list[list[int]] = []
                    for point in outline.reshape(-1, 2):
                        px = int(point[0]) + roi_x1
                        py = int(point[1]) + roi_y1
                        points.append([max(0, min(img_w - 1, px)), max(0, min(img_h - 1, py))])
                    if len(points) >= 3 and _validate_container_outline(image, [x1, y1, x2, y2], points):
                        return points

    pad_x = max(18, min(36, int(width * 0.045)))
    pad_top = max(18, min(34, int(height * 0.36)))
    pad_bottom = max(22, min(42, int(height * 0.48)))
    roi_x1 = max(0, x1 - pad_x)
    roi_y1 = max(0, y1 - pad_top)
    roi_x2 = min(img_w, x2 + pad_x)
    roi_y2 = min(img_h, y2 + pad_bottom)
    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return []
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    dark_lines = (gray < 90).astype(np.uint8) * 255
    dark_lines = cv2.morphologyEx(
        dark_lines,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    dark_lines = cv2.dilate(
        dark_lines,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    seed = np.zeros(dark_lines.shape, dtype=np.uint8)
    seed[y1 - roi_y1:y2 - roi_y1, x1 - roi_x1:x2 - roi_x1] = 255
    contours, _ = cv2.findContours(dark_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contour = None
    best_score = -1.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < max(120.0, float(width * height) * 0.06):
            continue
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if cw < max(80, int(width * 0.30)) or ch < max(32, int(height * 0.35)):
            continue
        global_cx1 = cx + roi_x1
        global_cy1 = cy + roi_y1
        global_cx2 = global_cx1 + cw
        global_cy2 = global_cy1 + ch
        horizontal_slack = max(18, int(width * 0.045))
        vertical_slack_top = max(18, int(height * 0.36))
        vertical_slack_bottom = max(24, int(height * 0.48))
        if global_cx1 < x1 - horizontal_slack or global_cx2 > x2 + horizontal_slack:
            if not (roi_x2 >= img_w - 1 and global_cx2 >= img_w - 3):
                continue
        if global_cy1 < y1 - vertical_slack_top or global_cy2 > y2 + vertical_slack_bottom:
            continue
        contour_mask = np.zeros(dark_lines.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, cv2.FILLED)
        overlap = int(np.count_nonzero((contour_mask > 0) & (seed > 0)))
        if overlap < max(30, int(width * height * 0.015)):
            continue
        touches = int(cx <= 1) + int(cy <= 1) + int(cx + cw >= dark_lines.shape[1] - 2) + int(cy + ch >= dark_lines.shape[0] - 2)
        edge_clipped_bubble = roi_x2 >= img_w - 1 and x2 >= img_w - 16
        if touches >= 4 and not edge_clipped_bubble:
            continue
        score = overlap + area * 0.05
        if score > best_score:
            best_score = score
            best_contour = contour
    if best_contour is None:
        return []

    epsilon = max(1.5, min(7.0, cv2.arcLength(best_contour, True) * 0.006))
    outline = cv2.approxPolyDP(best_contour, epsilon, True)
    points: list[list[int]] = []
    for point in outline.reshape(-1, 2):
        px = int(point[0]) + roi_x1
        py = int(point[1]) + roi_y1
        points.append([max(0, min(img_w - 1, px)), max(0, min(img_h - 1, py))])
    if len(points) < 3 or not _validate_container_outline(image, [x1, y1, x2, y2], points):
        return []
    return points


def _load_semantic_dialogue_regions(detect_dir: Path) -> list[tuple[int, int, int, int]]:
    sem_path = detect_dir / "semantic_detections.json"
    if not sem_path.exists():
        return []
    try:
        data = json.loads(sem_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    regions = data.get("regions", []) if isinstance(data, dict) else data
    boxes: list[tuple[int, int, int, int]] = []
    for r in regions:
        if str(r.get("semantic_class", "")).lower() != "dialogue":
            continue
        b = r.get("box", {})
        if all(k in b for k in ("x1", "y1", "x2", "y2")):
            boxes.append((int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])))
    return boxes


def _semantic_dialogue_rescue(
    item: dict, semantic_dialogue_boxes: list[tuple[int, int, int, int]], img_w: int
) -> bool:
    # Only consumed for a "title_logo" verdict (see the one call site in the
    # main layout loop) -- NOT wired into the general content-classification
    # rejection gate. An earlier version of this rescue was tried there too
    # (to catch bubbles the flood-fill container tracer also fails to
    # enclose) and was reverted: full 33-sample verification found it
    # produced FOUR confirmed false positives (SFX/decorative lettering
    # stamped directly on artwork with no bubble at all -- a splashing-water
    # "どぱどぱ", a kissing-sound "ちゅぱ．．．", a stylized "ひとく", an emphasis
    # stamp "こいつ/ちいち") against only two genuine hits, and no signal
    # tried (confidence, IoU, region width, region aspect) cleanly separated
    # them. Restricted to title_logo, this rescue validates clean: full
    # 33-sample kept-id diff shows exactly one change (modern_zh_2's "哈…
    # 成功了！" newly kept) and zero regressions. Rescue when Step 1's
    # independent semantic detector agrees a title_logo box is dialogue AND
    # the box is narrow relative to the page (SFX artwork/narration bands
    # sprawl wide; a reaction bubble stays narrow) AND the matched detector
    # region itself is plausibly bubble-shaped, not an elongated strip.
    if not semantic_dialogue_boxes or not img_w:
        return False
    box = item.get("box", {})
    if not all(k in box for k in ("x1", "y1", "x2", "y2")):
        return False
    x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
    width = max(1, x2 - x1)
    if width / img_w >= 0.15:
        return False
    if not _has_usable_translation(item.get("text", "")):
        return False
    area_a = max(1, (x2 - x1) * (y2 - y1))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    for sx1, sy1, sx2, sy2 in semantic_dialogue_boxes:
        ix1, iy1 = max(x1, sx1), max(y1, sy1)
        ix2, iy2 = min(x2, sx2), min(y2, sy2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_b = max(1, (sx2 - sx1) * (sy2 - sy1))
        iou = inter / float(area_a + area_b - inter)
        center_hit = sx1 <= cx <= sx2 and sy1 <= cy <= sy2
        if iou > 0.05 or center_hit:
            rw = max(1, sx2 - sx1)
            rh = max(1, sy2 - sy1)
            region_aspect = max(rw, rh) / min(rw, rh)
            if region_aspect <= 2.2:
                return True
    return False


def _semantic_role_for_item(item: dict, image_shape, sample_name: str = "", image: np.ndarray | None = None) -> str:
    text = item.get("text", "")
    box = item.get("box", {})
    bubble_idx = item.get("bubble_idx", -1)
    img_h = image_shape[0] if image_shape is not None else 0
    img_w = image_shape[1] if image_shape is not None else 0
    x1 = int(box.get("x1", 0))
    y1 = int(box.get("y1", 0))
    x2 = int(box.get("x2", x1))
    y2 = int(box.get("y2", y1))
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    counts = _script_counts(text)
    strong_dialogue = _strong_dialogue_candidate(text, counts)
    if _looks_like_embedded_device_label(item, image_shape, image):
        return "embedded_device_label"
    recover_large_dialogue = (
        bubble_idx == -1
        and img_h
        and img_w
        and _recoverable_large_floating_dialogue(
            text,
            counts,
            width,
            height,
            x1,
            y1,
            img_w,
            img_h,
            image,
        )
    )
    ocr_confidence = item.get("ocr_confidence")
    ocr_provider = str(item.get("ocr_provider", ""))
    fallback_source = str(item.get("fallback_source", ""))
    trusted_korean_paddle_group = (
        ocr_provider.startswith("paddleocr_ko")
        and fallback_source.startswith("paddleocr_korean")
        and counts["hangul"] >= 2
        and (ocr_confidence is None or float(ocr_confidence) >= 0.70)
    )
    vision_ocr_provider = ocr_provider.startswith(
        ("gemini_vision_ocr", "openrouter_vision_ocr", "groq_vision_ocr", "github_vision_ocr")
    )
    external_ko_vertical_paddle = (
        sample_name.startswith("external_ko")
        and ocr_provider.startswith(("paddleocr_ko", "paddleocr_ch_mixed"))
        and (counts["hangul"] >= 2 or counts["han"] >= 2)
        and height >= width * 1.15
        and (not img_w or width <= max(380, int(img_w * 0.42)))
    )
    external_ko_vision_region = (
        sample_name.startswith("external_ko")
        and vision_ocr_provider
        and (counts["hangul"] >= 2 or counts["han"] >= 2)
    )

    if _external_local_mode():
        if sample_name.startswith("external_ko"):
            if counts["hangul"] == 0 and not (
                (ocr_provider.startswith("paddleocr") and counts["han"] >= 2)
                or external_ko_vision_region
            ):
                return "ocr_language_mismatch"
            min_korean_confidence = 0.65 if ocr_provider.startswith("paddleocr_ch_mixed") else 0.45 if ocr_provider.startswith("paddleocr_ko") else 0.55
            if ocr_confidence is not None and not external_ko_vision_region and float(ocr_confidence) < min_korean_confidence:
                return "ocr_low_confidence"
        if sample_name.startswith("external_zh"):
            min_han = 2 if bubble_idx != -1 else 3
            if counts["han"] < min_han or counts["kana"] > counts["han"]:
                return "ocr_language_mismatch"
            if ocr_confidence is not None and float(ocr_confidence) < 0.35:
                return "ocr_low_confidence"
        if bubble_idx == -1 and img_h and img_w:
            area_ratio = (width * height) / max(1, img_h * img_w)
            pale_fraction = _pale_region_fraction(image, (x1, y1, x2, y2))
            dark_fraction = _dark_region_fraction(image, (x1, y1, x2, y2))
            if (
                y1 <= int(img_h * 0.20)
                and (pale_fraction < 0.65 or dark_fraction > 0.24)
                and not external_ko_vertical_paddle
                and not external_ko_vision_region
                and not trusted_korean_paddle_group
                and not strong_dialogue
            ):
                return "title_logo"
            if (
                y1 <= int(img_h * 0.18)
                and not external_ko_vertical_paddle
                and not external_ko_vision_region
                and not trusted_korean_paddle_group
                and not strong_dialogue
            ):
                return "title_logo"
            if (
                area_ratio > 0.025
                and not external_ko_vertical_paddle
                and not external_ko_vision_region
                and not trusted_korean_paddle_group
                and not recover_large_dialogue
            ):
                return "floating_too_large"
            if (
                width > int(img_w * 0.28)
                and not external_ko_vertical_paddle
                and not external_ko_vision_region
                and not trusted_korean_paddle_group
                and not recover_large_dialogue
            ):
                return "floating_too_wide"

    if bubble_idx != -1:
        return "dialogue"

    compact_text_len = len("".join(c for c in text if c.isalnum()))

    if (
        bubble_idx == -1
        and img_h
        and img_w
        and y1 <= int(img_h * 0.18)
        and width >= max(int(img_w * 0.34), int(height * 1.55))
        and compact_text_len <= 10
        and not external_ko_vision_region
        and not trusted_korean_paddle_group
    ):
        return "title_logo"
    if any(marker in text for marker in CREDIT_MARKERS):
        return "credit"
    if (
        img_h
        and y1 >= int(img_h * 0.90)
        and height <= max(80, int(img_h * 0.06))
        and not recover_large_dialogue
        and not strong_dialogue
    ):
        # Position/geometry alone is not credit evidence -- a bottom-strip
        # narration line that reads as a real sentence (new_sample_14's
        # "리조트에 수많은 옵션이 있는 모양인데") is dialogue that happens to sit
        # low on the page, not an assistant/artist credit tag. Real credit
        # lines are short name/studio snippets, not full sentences.
        return "credit"
    if img_h and y1 <= int(img_h * 0.08) and compact_text_len <= 4:
        # A short exclamation sitting near the page top INSIDE an enclosed
        # bubble-shaped container is a speech bubble Step 1's detector
        # missed, not title/logo art -- title/logo text is stamped directly
        # on panel art, never boxed by a bubble outline. Verified regression:
        # a panel-1 bubble near the top of modern_ko_2/modern_zh_2
        # ("하...완벽해" / "哈...成功了！") was silently dropped here with no
        # rescue path (title_logo skips the enclosed_dialogue_rescue check
        # entirely), leaving the bubble untranslated with zero debug trace.
        if bubble_idx == -1 and image is not None and _has_usable_translation(text):
            coords = (x1, y1, x2, y2)
            dark_interior = (
                _dark_region_fraction(image, coords) >= 0.58
                and _pale_region_fraction(image, coords) <= 0.22
            )
            if _infer_missed_bubble_outline(image, [x1, y1, x2, y2], dark_interior=dark_interior):
                return "dialogue"
        return "title_logo"
    if bubble_idx == -1 and img_h and img_w and not external_ko_vision_region:
        area_ratio = (width * height) / max(1, img_h * img_w)
        if width < max(28, int(img_w * 0.025)) and height < max(72, int(img_h * 0.070)) and compact_text_len <= 5:
            return "floating_too_small"
        if (
            (area_ratio > 0.065 or width > int(img_w * 0.52))
            and counts["han"] == 0
            and compact_text_len < 10
            and not trusted_korean_paddle_group
            and not recover_large_dialogue
        ):
            return "floating_too_large"
    return "dialogue"


def run_step6_layout(sample_map: dict[str, str] | None = None, samples_dir: Path | None = None):
    print("=" * 60)
    print("  Step 6 — Layout Export (v4: Strict 3-Color Debug)")
    print("=" * 60)

    samples_dir = Path(samples_dir) if samples_dir is not None else sample_root_from_env(DEFAULT_SAMPLES_ROOT)
    sample_map = sample_map or SAMPLE_MAP

    for sample_name, img_file in sample_map.items():
        sample_path = samples_dir / sample_name
        ocr_path    = sample_path / "step_5_ocr" / "ocr_results.json"
        detect_dir  = sample_path / "step_1_detect"

        if not ocr_path.exists():
            continue

        print(f"Processing {sample_name}")
        with open(ocr_path, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)

        img_path = sample_path / img_file
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [WARN] Cannot load image: {img_path}")
            image_shape = None
        else:
            image_shape = image.shape
        semantic_dialogue_boxes = _load_semantic_dialogue_regions(detect_dir)

        ocr_data = _merge_adjacent_floating_line_fragments(ocr_data, image_shape, semantic_dialogue_boxes)

        # ── Strict linguistic gate ──────────────────────────────────────
        # Reject English text, SFX, standalone numbers, and all noise.
        # Only confirmed Japanese dialogue passes through.
        constraints = []
        rejected = []
        sfx_artwork_regions: list[list[int]] = []

        # ── Display-line grouping for artwork lettering ──────────────────
        # A big display title is often OCR-split into fragments that are
        # individually ambiguous ('7일') while the LINE is unambiguous
        # display lettering. Test the anatomy signature on the union of
        # same-line adjacent floating fragments; the union's verdict binds
        # every member, so a title is never half-erased/half-preserved.
        sfx_artwork_group_ids: set = set()
        # Every id that was tested as part of a 2+ fragment line group, win
        # or lose. A group verdict of "not SFX" (real translatable text,
        # e.g. new_sample_14's "몰디브 7일 완성") must bind every member --
        # re-testing an individual sub-fragment's own small box can trip
        # fat_brush/hollow_brush on font-specific artifacts the full union
        # doesn't have (bold Hangul block letters have enclosed counter
        # shapes -- e.g. inside ㅁ/ㅇ -- that read as "hollow outline" on a
        # short sub-fragment even though the whole line is ordinary type).
        sfx_group_tested_ids: set = set()
        if image is not None:
            def _landscape_fragment(it: dict) -> bool:
                b = it["box"]
                w = max(1, b["x2"] - b["x1"])
                h = max(1, b["y2"] - b["y1"])
                # Title/SFX display fragments are landscape (a few big wide
                # glyphs); a tall narrow box is a CJK dialogue COLUMN (many
                # characters stacked vertically) -- letting one into the
                # grouping pool inflates the gap-tolerance formula below
                # (which scales off box height) enough to bridge two
                # unrelated captions clear across the page (regression:
                # new_sample_13's caption column at h=535 bridged a 517px
                # gap to a second, unrelated caption and both got
                # misclassified as one giant SFX union).
                return h <= w * 1.8

            line_items = [
                it
                for it in ocr_data
                if it.get("bubble_idx", -1) == -1
                and isinstance(it.get("box"), dict)
                and all(k in it["box"] for k in ("x1", "y1", "x2", "y2"))
                and _landscape_fragment(it)
            ]
            assigned: set = set()
            for i, base in enumerate(line_items):
                if i in assigned:
                    continue
                group = [i]
                assigned.add(i)
                changed = True
                while changed:
                    changed = False
                    for j, cand in enumerate(line_items):
                        if j in assigned:
                            continue
                        cb = cand["box"]
                        for gi in list(group):
                            gb = line_items[gi]["box"]
                            overlap = min(gb["y2"], cb["y2"]) - max(gb["y1"], cb["y1"])
                            min_h = max(1, min(gb["y2"] - gb["y1"], cb["y2"] - cb["y1"]))
                            max_h = max(gb["y2"] - gb["y1"], cb["y2"] - cb["y1"])
                            gap = max(gb["x1"], cb["x1"]) - min(gb["x2"], cb["x2"])
                            if overlap >= 0.5 * min_h and gap <= max(60, int(1.2 * max_h)):
                                group.append(j)
                                assigned.add(j)
                                changed = True
                                break
                if len(group) < 2:
                    continue
                ub = {
                    "x1": min(line_items[g]["box"]["x1"] for g in group),
                    "y1": min(line_items[g]["box"]["y1"] for g in group),
                    "x2": max(line_items[g]["box"]["x2"] for g in group),
                    "y2": max(line_items[g]["box"]["y2"] for g in group),
                }
                group_text = " ".join(
                    str(line_items[g].get("text", "")).strip() for g in group
                )
                for g in group:
                    sfx_group_tested_ids.add(line_items[g].get("id"))
                if _sfx_artwork_signature(image, ub, group_text):
                    for g in group:
                        sfx_artwork_group_ids.add(line_items[g].get("id"))
        for item in ocr_data:
            semantic_role = _semantic_role_for_item(item, image_shape, sample_name, image)
            if semantic_role == "title_logo" and _semantic_dialogue_rescue(
                item, semantic_dialogue_boxes, image_shape[1] if image_shape is not None else 0
            ):
                semantic_role = "dialogue"
            if semantic_role in {
                "credit",
                "title_logo",
                "floating_too_large",
                "floating_too_wide",
                "floating_too_small",
                "embedded_device_label",
                "ocr_language_mismatch",
                "ocr_low_confidence",
            }:
                rejected.append(_layout_rejection(item, semantic_role, semantic_role=semantic_role))
                continue

            cls = classify_text_by_content(item.get("text", ""))
            item_id = item.get("id")
            is_floating = item.get("bubble_idx", -1) == -1
            if item_id in sfx_group_tested_ids:
                is_sfx_artwork = item_id in sfx_artwork_group_ids
            else:
                is_sfx_artwork = is_floating and _sfx_artwork_signature(
                    image, item.get("box", {}), item.get("text", "")
                )
            if is_floating and is_sfx_artwork:
                # Brush/hollow display lettering is the artist's drawing, not
                # text: no boxes, no erase, no typeset. Registered so Step 4
                # shields the region from sweeps and neighboring cleanups.
                rejected.append(_layout_rejection(item, "sfx_artwork", semantic_role="sfx", classification=cls))
                rej_box = item.get("box") or {}
                if all(k in rej_box for k in ("x1", "y1", "x2", "y2")):
                    sfx_artwork_regions.append(
                        [int(rej_box["x1"]), int(rej_box["y1"]), int(rej_box["x2"]), int(rej_box["y2"])]
                    )
                continue
            if (
                item.get("bubble_idx", -1) == -1
                and _floating_roi_has_dominant_sfx_art(item, image)
            ):
                rejected.append(_layout_rejection(item, "floating_sfx_art", semantic_role="sfx", classification=cls))
                rej_box = item.get("box") or {}
                if all(k in rej_box for k in ("x1", "y1", "x2", "y2")):
                    sfx_artwork_regions.append(
                        [int(rej_box["x1"]), int(rej_box["y1"]), int(rej_box["x2"]), int(rej_box["y2"])]
                    )
                continue
            keep_short_floating = (
                item.get("bubble_idx", -1) == -1
                and _short_spoken_dialogue_fragment(item, image)
            )
            keep_short_bubble = (
                cls == "sfx"
                and item.get("bubble_idx", -1) != -1
                and _has_usable_translation(item.get("text", ""))
            )
            # A fragment classified as sfx/noise that sits INSIDE a pale
            # enclosed container is bubble dialogue whose bubble the detector
            # missed (OCR often splits such bubbles into short fragments that
            # then look like SFX). Rescue it when it has a usable translation.
            enclosed_dialogue_rescue = False
            if (
                cls != "dialogue"
                and item.get("bubble_idx", -1) == -1
                and _has_usable_translation(item.get("text", ""))
                and image is not None
            ):
                probe_box = item["box"]
                probe_coords = [probe_box["x1"], probe_box["y1"], probe_box["x2"], probe_box["y2"]]
                # A short OCR fragment covers only a fraction of its bubble,
                # so the container may legitimately be tens of times larger.
                flood_points = _container_flood_outline(
                    image,
                    probe_coords,
                    max_area_ratio=45.0,
                )
                enclosed_dialogue_rescue = bool(flood_points) and not _rescue_strokes_belong_to_container(
                    image, probe_coords, flood_points
                )
            if (
                cls != "dialogue"
                and not keep_short_bubble
                and not keep_short_floating
                and not enclosed_dialogue_rescue
            ):
                rejected.append(_layout_rejection(item, f"classification_{cls}", semantic_role=semantic_role, classification=cls))
                continue
            if cls == "dialogue" and not _has_usable_translation(item.get("text", "")):
                rejected.append(_layout_rejection(item, "no_usable_translation", semantic_role=semantic_role, classification=cls))
                continue

            rb = item["box"]
            gb = item["green_box"]
            fallback_source = item.get("fallback_source")
            inferred_bubble_outline: list[list[int]] = []
            if _needs_inferred_bubble_cleanup(item, image_shape, image, semantic_role=semantic_role):
                coords = (rb["x1"], rb["y1"], rb["x2"], rb["y2"])
                dark_interior = (
                    _dark_region_fraction(image, coords) >= 0.58
                    and _pale_region_fraction(image, coords) <= 0.22
                )
                # Distinct tag for reversed-polarity containers: Step 4's
                # allow_inferred_bubble_cleanup routing (keyed off the literal
                # substring "missed_bubble") assumes light-fill/dark-ink
                # content throughout -- precise_floating_text_local_cleanup
                # and its siblings look for DARK stroke evidence and silently
                # no-op on white glyphs over a black bubble (verified: it
                # claimed new_sample_5's merged black-bubble constraint and
                # left the source text un-erased). Dark containers must reach
                # Step 4's separate reverse-dark-balloon path instead.
                tag = "missed_dark_bubble" if dark_interior else "missed_bubble"
                fallback_source = f"{fallback_source}+{tag}" if fallback_source else tag
                inferred_bubble_outline = _infer_missed_bubble_outline(
                    image,
                    [rb["x1"], rb["y1"], rb["x2"], rb["y2"]],
                    dark_interior=dark_interior,
                )
            constraint = {
                "id":              item["id"],
                "text":            item["text"],
                "red_box":         [rb["x1"], rb["y1"], rb["x2"], rb["y2"]],
                "erase_boxes":      item.get("erase_boxes", []),
                "green_box":       [gb["x1"], gb["y1"], gb["x2"], gb["y2"]],
                "green_polygon":   item.get("green_polygon", []),
                "bubble_idx":      item.get("bubble_idx", -1),
                "mask_mode":       item.get("mask_mode", "stroke"),
                "route":           item.get("route", "floating_dialogue"),
                "semantic_role":   "dialogue" if (keep_short_floating or enclosed_dialogue_rescue) else semantic_role,
                "fallback_source": fallback_source,
                "force_bubble_cleanup": bool(item.get("force_bubble_cleanup", False)),
            }
            if keep_short_floating:
                # This constraint's own box is tiny (a rescued 1-4 char
                # spoken particle, see _short_spoken_dialogue_fragment) --
                # tell Step 7 to keep the translation proportionally short
                # too. Without this, a translator can expand "って" into a
                # full explanatory sentence that has nowhere to fit in a
                # 40x65px box and visibly overflows into neighboring text.
                constraint["short_fragment"] = True
            if inferred_bubble_outline:
                constraint["inferred_bubble_outline"] = inferred_bubble_outline
                constraint["inferred_bubble_source"] = "outline_contour"
            if item.get("line_fragment_ids"):
                constraint["line_fragment_ids"] = item.get("line_fragment_ids")
            if item.get("force_full_line_cleanup"):
                constraint["force_full_line_cleanup"] = True
            if keep_short_floating:
                # Marks a short spoken-particle fragment kept as its own
                # translatable constraint (see _short_spoken_dialogue_fragment)
                # rather than dropped as SFX. It still needs to donate its
                # box to a nearby larger dialogue constraint's erase_boxes
                # below -- without that, the neighbor's own Step 4 local-crop
                # inpaint loses surrounding erased context and becomes
                # unstable (verified: produces large paint-bleed artifacts on
                # gradient backgrounds). This flag is removed before the
                # constraint list is serialized.
                constraint["_short_rescue"] = True
            constraints.append(constraint)

        # Donate each short-rescue fragment's own box to the nearest larger
        # floating dialogue constraint's erase_boxes, purely as extra
        # erasure-context for Step 4 -- the fragment keeps its own
        # independent constraint entry (and therefore its own translation/
        # typeset pass) unaffected. Reuses the same proximity heuristics as
        # the recoverable-fragment merge pass below.
        for rescued in constraints:
            if not rescued.pop("_short_rescue", False):
                continue
            fx1, fy1, fx2, fy2 = [int(v) for v in rescued["red_box"]]
            f_height = max(1, fy2 - fy1)
            f_center_y = (fy1 + fy2) / 2.0
            best_neighbor = None
            best_score = 1e18
            for constraint in constraints:
                if constraint is rescued or constraint.get("bubble_idx", -1) != -1:
                    continue
                rx1, ry1, rx2, ry2 = [int(v) for v in constraint.get("red_box", [0, 0, 0, 0])]
                r_height = max(1, ry2 - ry1)
                h_gap = max(0, max(fx1, rx1) - min(fx2, rx2))
                v_gap = max(0, max(fy1, ry1) - min(fy2, ry2))
                close_x = h_gap <= max(96, int(min(f_height, r_height) * 1.15))
                close_y = abs(f_center_y - ((ry1 + ry2) / 2.0)) <= max(92, int(max(f_height, r_height) * 0.90))
                if not close_x or not close_y or v_gap > max(72, int(max(f_height, r_height) * 0.40)):
                    continue
                score = h_gap * 2.0 + v_gap + abs(f_center_y - ((ry1 + ry2) / 2.0)) * 0.25
                if score < best_score:
                    best_score = score
                    best_neighbor = constraint
            if best_neighbor is not None:
                best_neighbor.setdefault("erase_boxes", [])
                best_neighbor["erase_boxes"].append([fx1, fy1, fx2, fy2])
                if best_neighbor.get("fallback_source"):
                    best_neighbor["fallback_source"] = f"{best_neighbor['fallback_source']}+erase_context_donation"
                else:
                    best_neighbor["fallback_source"] = "erase_context_donation"

        for item in ocr_data:
            item_id = int(item.get("id", -1))
            if any(int(c.get("id", -9999)) == item_id for c in constraints):
                continue
            if item.get("bubble_idx", -1) != -1:
                continue
            if not _recoverable_adjacent_fragment(item, constraints, image):
                continue

            box = item.get("box", {})
            fx1 = int(box.get("x1", 0))
            fy1 = int(box.get("y1", 0))
            fx2 = int(box.get("x2", fx1))
            fy2 = int(box.get("y2", fy1))
            f_height = max(1, fy2 - fy1)
            f_center_y = (fy1 + fy2) / 2.0
            erase_only_fragment = _erase_only_adjacent_fragment(item, image)

            if _short_spoken_dialogue_fragment(item, image):
                gb = item.get("green_box", box)
                gx1 = int(gb.get("x1", fx1))
                gy1 = int(gb.get("y1", fy1))
                gx2 = int(gb.get("x2", fx2))
                gy2 = int(gb.get("y2", fy2))
                constraints.append({
                    "id":              item["id"],
                    "text":            item["text"],
                    "red_box":         [fx1, fy1, fx2, fy2],
                    "erase_boxes":      item.get("erase_boxes", []),
                    "green_box":       [gx1, gy1, gx2, gy2],
                    "green_polygon":   item.get("green_polygon", [[gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]]),
                    "bubble_idx":      -1,
                    "mask_mode":       item.get("mask_mode", "stroke"),
                    "route":           item.get("route", "floating_dialogue"),
                    "semantic_role":   "dialogue",
                    "fallback_source": "adjacent_short_dialogue",
                    "force_bubble_cleanup": False,
                })
                continue

            best_constraint = None
            best_score = 1e18
            for constraint in constraints:
                if constraint.get("bubble_idx", -1) != -1:
                    continue
                rx1, ry1, rx2, ry2 = [int(v) for v in constraint.get("red_box", [0, 0, 0, 0])]
                r_height = max(1, ry2 - ry1)
                h_gap = max(0, max(fx1, rx1) - min(fx2, rx2))
                v_gap = max(0, max(fy1, ry1) - min(fy2, ry2))
                close_x = h_gap <= max(96, int(min(f_height, r_height) * 1.15))
                close_y = abs(f_center_y - ((ry1 + ry2) / 2.0)) <= max(92, int(max(f_height, r_height) * 0.90))
                if not close_x or not close_y or v_gap > max(72, int(max(f_height, r_height) * 0.40)):
                    continue
                score = h_gap * 2.0 + v_gap + abs(f_center_y - ((ry1 + ry2) / 2.0)) * 0.25
                if score < best_score:
                    best_score = score
                    best_constraint = constraint

            if best_constraint is None:
                continue

            if erase_only_fragment:
                best_constraint.setdefault("erase_boxes", [])
                best_constraint["erase_boxes"].append([fx1, fy1, fx2, fy2])
                if best_constraint.get("fallback_source"):
                    best_constraint["fallback_source"] = f"{best_constraint['fallback_source']}+erase_fragment_merged"
                else:
                    best_constraint["fallback_source"] = "erase_fragment_merged"
                continue

            rb = best_constraint["red_box"]
            gb = best_constraint["green_box"]
            best_constraint["red_box"] = [min(rb[0], fx1), min(rb[1], fy1), max(rb[2], fx2), max(rb[3], fy2)]
            best_constraint["green_box"] = [min(gb[0], fx1), min(gb[1], fy1), max(gb[2], fx2), max(gb[3], fy2)]
            gx1, gy1, gx2, gy2 = best_constraint["green_box"]
            best_constraint["green_polygon"] = [[gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]]
            best_constraint.setdefault("erase_boxes", [])
            best_constraint["erase_boxes"].append([fx1, fy1, fx2, fy2])
            if best_constraint.get("fallback_source"):
                best_constraint["fallback_source"] = f"{best_constraint['fallback_source']}+adjacent_fragment_merged"
            else:
                best_constraint["fallback_source"] = "adjacent_fragment_merged"

        _normalize_floating_layout_boxes(constraints, image_shape)
        _apply_art_aware_floating_routes(constraints, image, detect_dir)
        refine_text_mask = (
            _load_step1_text_mask(detect_dir, image.shape) if image is not None else None
        )
        for constraint in constraints:
            _refine_constraint_geometry(constraint, refine_text_mask, image_shape, image)

        # ── Multi-column same-container consolidation ────────────────────
        # Vertical CJK dialogue is often split into several side-by-side
        # text columns that each OCR cleanly and each get their OWN kept
        # floating constraint -- unlike the debris-donation case below, none
        # of them were ever rejected, so that consolidation path never runs
        # for them. When two or more floating constraints trace to the SAME
        # container (new_sample_5's black bubbles: two columns of one
        # utterance), they are one utterance in one bubble: merge into a
        # single constraint so Step 7 translates one coherent sentence and
        # Step 8 fits one readable block instead of cramming each column
        # into its own sliver.
        floating_with_outline = [
            c
            for c in constraints
            if int(c.get("bubble_idx", -1)) == -1 and len(c.get("inferred_bubble_outline") or []) >= 3
        ]
        consolidated_ids: set = set()
        for base in floating_with_outline:
            if id(base) in consolidated_ids:
                continue
            outline = base["inferred_bubble_outline"]
            raster = np.zeros(image_shape[:2], dtype=np.uint8)
            cv2.fillPoly(raster, [np.array(outline, dtype=np.int32)], 255)
            group = [base]
            for other in floating_with_outline:
                if other is base or id(other) in consolidated_ids:
                    continue
                orx1, ory1, orx2, ory2 = [int(v) for v in other["red_box"]]
                ocx = min(image_shape[1] - 1, max(0, (orx1 + orx2) // 2))
                ocy = min(image_shape[0] - 1, max(0, (ory1 + ory2) // 2))
                if raster[ocy, ocx] > 0:
                    group.append(other)
            if len(group) < 2:
                continue
            # Vertical Japanese/CJK columns read right-to-left; a landscape
            # (wide) group reads left-to-right like a normal line.
            vertical_group = all(
                (int(g["red_box"][3]) - int(g["red_box"][1]))
                >= (int(g["red_box"][2]) - int(g["red_box"][0])) * 1.1
                for g in group
            )
            ordered = sorted(
                group,
                key=lambda g: -int(g["red_box"][0]) if vertical_group else int(g["red_box"][0]),
            )
            primary = ordered[0]
            primary["text"] = " ".join(
                str(g.get("text", "")).strip() for g in ordered if str(g.get("text", "")).strip()
            )
            rxs = [int(v) for g in ordered for v in (g["red_box"][0], g["red_box"][2])]
            rys = [int(v) for g in ordered for v in (g["red_box"][1], g["red_box"][3])]
            primary["red_box"] = [min(rxs), min(rys), max(rxs), max(rys)]
            primary["line_fragment_ids"] = [int(g.get("id", -1)) for g in ordered]
            primary["consolidated_container"] = True
            for g in ordered[1:]:
                consolidated_ids.add(id(g))
            _refine_constraint_geometry(primary, refine_text_mask, image_shape, image)
        if consolidated_ids:
            constraints[:] = [c for c in constraints if id(c) not in consolidated_ids]

        # Rejected classification debris (OCR split fragments with no usable
        # translation) that sits inside a kept bubble/container constraint is
        # source-text remnant: donate its box for erasure so cleanup removes
        # it even though nothing renders for it. Art SFX outside containers
        # is never donated. Runs after geometry refinement so container
        # outlines and green boxes are final.
        def _rejected_box_owned_by_kept_constraint(box: dict) -> bool:
            # A rejected fragment that IS a kept constraint's own text region
            # (a duplicate detection of the same glyphs) must never be
            # donated to a DIFFERENT constraint: the owner already erases
            # those pixels itself, and the donation bloats the recipient's
            # step-4 erase window clear across the page (verified:
            # new_sample_13's id6 carried a donated copy of kept id8's
            # red_box, stretching its erase union from its own bubble to the
            # far side of the panel).
            bx1, by1, bx2, by2 = box["x1"], box["y1"], box["x2"], box["y2"]
            b_area = max(1, (bx2 - bx1) * (by2 - by1))
            for kept in constraints:
                kx1, ky1, kx2, ky2 = [int(v) for v in kept["red_box"]]
                iw = min(bx2, kx2) - max(bx1, kx1)
                ih = min(by2, ky2) - max(by1, ky1)
                if iw <= 0 or ih <= 0:
                    continue
                inter = iw * ih
                k_area = max(1, (kx2 - kx1) * (ky2 - ky1))
                if inter / float(b_area + k_area - inter) >= 0.45:
                    return True
            return False

        for rej_item in rejected:
            reason = str(rej_item.get("reason") or "")
            rej_box = rej_item.get("box") or {}
            if not all(k in rej_box for k in ("x1", "y1", "x2", "y2")):
                continue
            if _rejected_box_owned_by_kept_constraint(rej_box):
                continue
            if reason == "classification_sfx":
                # A correctly-identified SFX element is its own distinct page
                # element, never incidental debris belonging to a neighboring
                # dialogue constraint -- neither the container-trace donation
                # nor the plain-dark-fragment proximity fallback below should
                # ever claim it (verified regression: external_ja_2's "ズーン"
                # SFX, OCR-misread as short kana "えっ", was correctly
                # classified classification_sfx but still donated via the
                # proximity fallback to a neighboring dialogue constraint's
                # erase_boxes, causing Step 4 to erase/repaint unrelated
                # background art as if it were bubble interior).
                continue
            if reason == "title_logo":
                # A "title_logo" fragment that continues a kept caption LINE
                # is the tail of that caption (OCR split it): its Korean/CJK
                # remnant must be erased with the line, or it survives next
                # to the English translation (new_sample_14 top caption).
                # Real standalone title logos have no same-line neighbor.
                for c in constraints:
                    crx1, cry1, crx2, cry2 = [int(v) for v in c["red_box"]]
                    overlap = min(cry2, rej_box["y2"]) - max(cry1, rej_box["y1"])
                    min_h = max(1, min(cry2 - cry1, rej_box["y2"] - rej_box["y1"]))
                    gap = max(crx1 - rej_box["x2"], rej_box["x1"] - crx2)
                    if overlap >= 0.6 * min_h and gap <= 70:
                        c.setdefault("erase_boxes", [])
                        c["erase_boxes"].append(
                            [rej_box["x1"], rej_box["y1"], rej_box["x2"], rej_box["y2"]]
                        )
                        break
                continue
            if not reason.startswith("classification_"):
                continue
            container_pts = (
                _container_flood_outline(
                    image,
                    [rej_box["x1"], rej_box["y1"], rej_box["x2"], rej_box["y2"]],
                    max_area_ratio=45.0,
                )
                if image is not None
                else []
            )
            donated = False
            if len(container_pts) >= 3:
                container_raster = np.zeros(image_shape[:2], dtype=np.uint8)
                cv2.fillPoly(
                    container_raster, [np.array(container_pts, dtype=np.int32)], 255
                )
                for c in constraints:
                    crx1, cry1, crx2, cry2 = [int(v) for v in c["red_box"]]
                    ccx = min(image_shape[1] - 1, max(0, (crx1 + crx2) // 2))
                    ccy = min(image_shape[0] - 1, max(0, (cry1 + cry2) // 2))
                    if container_raster[ccy, ccx] > 0:
                        c.setdefault("erase_boxes", [])
                        c["erase_boxes"].append(
                            [rej_box["x1"], rej_box["y1"], rej_box["x2"], rej_box["y2"]]
                        )
                        donated = True
                        break
            if donated or image is None:
                continue
            # Open containers (bubble tails/outline gaps defeat the flood):
            # fall back to proximity, but ONLY for small plain-dark fragments.
            # Colored/brush lettering is potential artwork SFX and is never
            # donated for erasure.
            frag_w = rej_box["x2"] - rej_box["x1"]
            frag_h = rej_box["y2"] - rej_box["y1"]
            if frag_w * frag_h > 9000 or max(frag_w, frag_h) > 130:
                continue
            frag = image[
                max(0, rej_box["y1"]):min(image_shape[0], rej_box["y2"]),
                max(0, rej_box["x1"]):min(image_shape[1], rej_box["x2"]),
            ]
            if frag.size == 0:
                continue
            frag_hsv = cv2.cvtColor(frag, cv2.COLOR_BGR2HSV)
            saturated_frac = float(
                np.mean((frag_hsv[:, :, 1] >= 90) & (frag_hsv[:, :, 2] >= 40))
            )
            if saturated_frac >= 0.10:
                continue
            gap = 50
            for c in constraints:
                crx1, cry1, crx2, cry2 = [int(v) for v in c["red_box"]]
                if (
                    rej_box["x1"] - gap <= crx2
                    and rej_box["x2"] + gap >= crx1
                    and rej_box["y1"] - gap <= cry2
                    and rej_box["y2"] + gap >= cry1
                ):
                    c.setdefault("erase_boxes", [])
                    c["erase_boxes"].append(
                        [rej_box["x1"], rej_box["y1"], rej_box["x2"], rej_box["y2"]]
                    )
                    break

        # ── Caption-line erase extension ─────────────────────────────────
        # OCR line boxes routinely clip the first/last glyph of a caption.
        # When glyph ink continues past the box on a PLAIN background, the
        # erase must follow it to the ink's true end, or a character sliver
        # survives beside the translation (new_sample_14 top caption).
        if image is not None:
            ext_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            for c in constraints:
                if int(c.get("bubble_idx", -1)) >= 0:
                    continue
                rx1, ry1, rx2, ry2 = [int(v) for v in c["red_box"]]
                bw, bh = rx2 - rx1, ry2 - ry1
                if bw < bh * 3 or bh < 14 or bh > 120:
                    continue
                band = ext_gray[max(0, ry1):min(image_shape[0], ry2), :]
                if band.size == 0:
                    continue
                ink_cols = np.mean(band < 110, axis=0) >= 0.08
                plain_cols = np.mean(band >= 200, axis=0) >= 0.80
                max_ext = int(image_shape[1] * 0.25)
                gap_cap = 25

                def scan(start: int, step: int) -> int:
                    pos = start
                    gap = 0
                    last_ink = start
                    walked = 0
                    while 0 <= pos + step < image_shape[1] and walked < max_ext:
                        pos += step
                        walked += 1
                        if ink_cols[pos]:
                            gap = 0
                            last_ink = pos
                        elif plain_cols[pos]:
                            gap += 1
                            if gap > gap_cap:
                                break
                        else:
                            break
                    return last_ink

                left_end = scan(rx1, -1)
                right_end = scan(rx2 - 1, 1)
                if left_end < rx1 - 2:
                    c.setdefault("erase_boxes", [])
                    c["erase_boxes"].append([max(0, left_end - 4), ry1, rx1 + 4, ry2])
                if right_end > rx2 + 1:
                    c.setdefault("erase_boxes", [])
                    c["erase_boxes"].append([rx2 - 4, ry1, min(image_shape[1], right_end + 5), ry2])

        # ── Enclosed-constraint consolidation ───────────────────────────
        # A floating constraint whose fragments (donated erase boxes) share
        # one validated container is ONE utterance in ONE bubble/box: union
        # the enclosed fragment boxes into the red box, adopt the container
        # as the blue outline, and rebuild green as its inset. This replaces
        # the per-fragment box clutter (per-character red boxes on rescued
        # bubbles, per-column boxes on framed captions) with one coherent
        # red/blue/green triple.
        if image is not None:
            enclosed_absorbed: set = set()
            for c in constraints:
                if id(c) in enclosed_absorbed:
                    continue
                if int(c.get("bubble_idx", -1)) >= 0:
                    continue
                erase_box4s = [
                    b4
                    for b4 in (_coerce_box4(eb) for eb in (c.get("erase_boxes") or []))
                    if b4 is not None
                ]
                if not erase_box4s:
                    continue
                container_pts = c.get("inferred_bubble_outline") or []
                if len(container_pts) < 3:
                    container_pts = _container_flood_outline(
                        image, c["red_box"], max_area_ratio=18.0
                    )
                if len(container_pts) < 3:
                    continue
                container_raster = np.zeros(image_shape[:2], dtype=np.uint8)
                cv2.fillPoly(
                    container_raster, [np.array(container_pts, dtype=np.int32)], 255
                )
                rx1, ry1, rx2, ry2 = [int(v) for v in c["red_box"]]
                enclosed = []
                for bx1, by1, bx2, by2 in erase_box4s:
                    ccx = min(image_shape[1] - 1, max(0, (bx1 + bx2) // 2))
                    ccy = min(image_shape[0] - 1, max(0, (by1 + by2) // 2))
                    if container_raster[ccy, ccx] > 0:
                        enclosed.append((bx1, by1, bx2, by2))
                if not enclosed:
                    continue
                union_box = [
                    min([rx1] + [b[0] for b in enclosed]),
                    min([ry1] + [b[1] for b in enclosed]),
                    max([rx2] + [b[2] for b in enclosed]),
                    max([ry2] + [b[3] for b in enclosed]),
                ]

                # A touching-bubble pair presents no wall at the contact
                # point, so the flood-fill container tracer can bleed from
                # `c`'s own bubble into a NEIGHBORING, already-separate
                # constraint's bubble (new_sample_11's id9: its traced
                # outline sprawled far enough to fully contain id8's own
                # red_box, and id9 then adopted the WHOLE outline as its
                # green shape -- swallowing id8's territory and causing
                # Step 8 to drop id8 as a contained duplicate). Detect any
                # OTHER kept constraint whose red_box center falls inside
                # this container and, if found, split the raster between
                # `c`'s own union_box and each sibling via the same
                # nearest-box distance assignment `_bubble_cluster_zones`
                # (run_step5_ocr.py) already uses for real detected bubbles
                # hosting multiple clusters -- `c` only ever adopts its OWN
                # share of a shared container.
                sibling_boxes = []
                sibling_ids = []
                sibling_refs = []
                for other in constraints:
                    if other is c or id(other) in enclosed_absorbed:
                        continue
                    orb = other.get("red_box")
                    if not orb or len(orb) != 4:
                        continue
                    ox1, oy1, ox2, oy2 = [int(v) for v in orb]
                    ocx = min(image_shape[1] - 1, max(0, (ox1 + ox2) // 2))
                    ocy = min(image_shape[0] - 1, max(0, (oy1 + oy2) // 2))
                    if container_raster[ocy, ocx] > 0:
                        sibling_boxes.append((ox1, oy1, ox2, oy2))
                        sibling_refs.append(other)
                        if "id" in other:
                            sibling_ids.append(other["id"])

                # Same-utterance vertical columns sharing one container are
                # ONE dialogue, not touching-bubble neighbors: merge them
                # instead of Voronoi-splitting (which draws diagonal
                # dividers straight through the text -- new_sample_13's
                # three-column bubble). The dedicated multi-column
                # consolidation above can't catch these because it needs
                # `inferred_bubble_outline` BEFORE it runs, while these
                # constraints only get their container traced here.
                if (
                    sibling_refs
                    and all(int(s.get("bubble_idx", -1)) == -1 for s in sibling_refs)
                    and _same_utterance_vertical_columns(
                        [(rx1, ry1, rx2, ry2), *sibling_boxes], image_shape[1]
                    )
                ):
                    members = [c, *sibling_refs]
                    # Vertical CJK reads right-to-left: rightmost column first.
                    ordered = sorted(
                        members, key=lambda g: -int(g["red_box"][0])
                    )
                    c["text"] = " ".join(
                        str(g.get("text", "")).strip()
                        for g in ordered
                        if str(g.get("text", "")).strip()
                    )
                    merge_xs = [int(v) for g in members for v in (g["red_box"][0], g["red_box"][2])]
                    merge_ys = [int(v) for g in members for v in (g["red_box"][1], g["red_box"][3])]
                    c["red_box"] = [
                        min(merge_xs + [union_box[0]]),
                        min(merge_ys + [union_box[1]]),
                        max(merge_xs + [union_box[2]]),
                        max(merge_ys + [union_box[3]]),
                    ]
                    c.setdefault("erase_boxes", [])
                    for sib in sibling_refs:
                        c["erase_boxes"].append([int(v) for v in sib["red_box"]])
                        for eb in sib.get("erase_boxes") or []:
                            b4 = _coerce_box4(eb)
                            if b4 is not None:
                                c["erase_boxes"].append(list(b4))
                        enclosed_absorbed.add(id(sib))
                    c["line_fragment_ids"] = sorted(
                        {int(g.get("id", -1)) for g in members}
                    )
                    if c.get("fallback_source"):
                        c["fallback_source"] = f"{c['fallback_source']}+vertical_column_merge"
                    else:
                        c["fallback_source"] = "vertical_column_merge"
                    c["inferred_bubble_outline"] = container_pts
                    c["inferred_bubble_source"] = "enclosed_consolidation"
                    c["consolidated_container"] = True
                    c.pop("red_polygon", None)
                    c.pop("green_polygons", None)
                    c.pop("touching_container_siblings", None)
                    _refine_constraint_geometry(c, refine_text_mask, image_shape, image)
                    continue

                if sibling_boxes:
                    ux1, uy1, ux2, uy2 = union_box
                    own_seed = np.ones(image_shape[:2], dtype=np.uint8) * 255
                    cv2.rectangle(own_seed, (ux1, uy1), (max(ux1, ux2 - 1), max(uy1, uy2 - 1)), 0, -1)
                    own_dist = cv2.distanceTransform(own_seed, cv2.DIST_L2, 5)
                    sibling_dist_min = None
                    for sx1, sy1, sx2, sy2 in sibling_boxes:
                        seed = np.ones(image_shape[:2], dtype=np.uint8) * 255
                        cv2.rectangle(seed, (sx1, sy1), (max(sx1, sx2 - 1), max(sy1, sy2 - 1)), 0, -1)
                        d = cv2.distanceTransform(seed, cv2.DIST_L2, 5)
                        sibling_dist_min = d if sibling_dist_min is None else np.minimum(sibling_dist_min, d)
                    own_territory = (own_dist <= sibling_dist_min)
                    split_raster = container_raster.copy()
                    split_raster[~own_territory] = 0
                    if np.count_nonzero(split_raster) < 60:
                        continue
                    contours, _ = cv2.findContours(split_raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        epsilon = max(1.5, cv2.arcLength(largest, True) * 0.006)
                        approx = cv2.approxPolyDP(largest, epsilon, True)
                        split_pts = [point[0].astype(int).tolist() for point in approx]
                        if len(split_pts) >= 3:
                            container_pts = split_pts

                c["red_box"] = union_box
                c["inferred_bubble_outline"] = container_pts
                c["inferred_bubble_source"] = "enclosed_consolidation"
                c["consolidated_container"] = True
                if sibling_ids:
                    # Two constraints sharing one traced (touching-bubble)
                    # container can still end up with overlapping bounding
                    # green_box rectangles even after the split above (a
                    # constraint whose own outline never bled into its
                    # neighbor's space in the first place has nothing to
                    # trim, yet its rectangle still spans the neighbor's
                    # smaller box -- verified: new_sample_13's id6/id7).
                    # Step 8's duplicate-overlap drop is bounding-box-only
                    # and would otherwise treat that as the SAME region
                    # detected twice and silently drop the shorter
                    # translation. Tag both sides of a KNOWN sibling
                    # relationship so Step 8 can tell "two separate
                    # touching-bubble utterances" apart from an actual
                    # duplicate detection.
                    own_id = c.get("id")
                    existing = set(c.get("touching_container_siblings") or [])
                    existing.update(sibling_ids)
                    c["touching_container_siblings"] = sorted(existing)
                    for other in constraints:
                        if other.get("id") in sibling_ids:
                            other_existing = set(other.get("touching_container_siblings") or [])
                            if own_id is not None:
                                other_existing.add(own_id)
                            other["touching_container_siblings"] = sorted(other_existing)
                c.pop("red_polygon", None)
                c.pop("green_polygons", None)
                # Re-derive the tight red box and the green inset from the
                # container outline now that the geometry is consolidated.
                _refine_constraint_geometry(c, refine_text_mask, image_shape, image)
            if enclosed_absorbed:
                constraints[:] = [
                    k for k in constraints if id(k) not in enclosed_absorbed
                ]

        # ── Save layout constraints JSON ────────────────────────────────
        out_dir = sample_path / "step_6_layout"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "layout_constraints.json", "w", encoding="utf-8") as f:
            json.dump(constraints, f, indent=2, ensure_ascii=False)
        with open(out_dir / "rejected_layout_items.json", "w", encoding="utf-8") as f:
            json.dump(rejected, f, indent=2, ensure_ascii=False)
        # Regions of SFX drawn as artwork — Step 4 must treat them as sacred
        # (no residual sweep, no neighboring-cleanup bleed). Always written so
        # a rerun never inherits a stale registry.
        with open(out_dir / "sfx_artwork_regions.json", "w", encoding="utf-8") as f:
            json.dump(sfx_artwork_regions, f, indent=2, ensure_ascii=False)
        if rejected:
            reasons = Counter(str(item.get("reason", "unknown")) for item in rejected)
            reason_text = ", ".join(f"{reason}={count}" for reason, count in reasons.most_common(5))
            print(f"  Layout kept {len(constraints)}/{len(ocr_data)}; rejected {len(rejected)} ({reason_text})")
            for item in rejected[:8]:
                text = str(item.get("text") or "").replace("\n", " ").strip()
                if len(text) > 48:
                    text = text[:45] + "..."
                print(f"    [layout-reject] id={item.get('id')} reason={item.get('reason')} text={text!r}")
            write_diagnostic_event(
                "layout.rejections",
                {
                    "sample": sample_name,
                    "kept": len(constraints),
                    "total": len(ocr_data),
                    "rejected": len(rejected),
                    "reasons": dict(reasons),
                    "items": [
                        {
                            "id": item.get("id"),
                            "reason": item.get("reason"),
                            "text": str(item.get("text") or "")[:120],
                            "box": item.get("box"),
                            "route": item.get("route"),
                            "bubbleIdx": item.get("bubble_idx"),
                        }
                        for item in rejected[:20]
                    ],
                },
                source="layout",
                level="warning",
            )
        else:
            print(f"  Layout kept {len(constraints)}/{len(ocr_data)}; rejected 0")
            write_diagnostic_event(
                "layout.rejections",
                {"sample": sample_name, "kept": len(constraints), "total": len(ocr_data), "rejected": 0},
                source="layout",
                level="info",
            )

        # ── Build debug image ────────────────────────────────────────────
        if image is None:
            continue

        debug_text_mask = _load_step1_text_mask(detect_dir, image.shape)
        debug_glyph_guard = None
        debug_layout_guard = None
        debug_art_guard = None
        if debug_text_mask is not None:
            debug_glyph_guard = cv2.dilate(
                debug_text_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
                iterations=1,
            )
            debug_layout_guard = cv2.dilate(
                debug_text_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                iterations=1,
            )
        debug_structural = _structural_art_mask(image)
        if debug_structural is not None:
            if debug_text_mask is not None:
                debug_source_guard = cv2.dilate(
                    debug_text_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                    iterations=1,
                )
            else:
                debug_source_guard = np.zeros_like(debug_structural)
            debug_art_guard = cv2.bitwise_and(debug_structural, cv2.bitwise_not(debug_source_guard))
            debug_art_guard = cv2.dilate(
                debug_art_guard,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )

        used_bubble_indices = sorted({
            int(c.get("bubble_idx", -1))
            for c in constraints
            if int(c.get("bubble_idx", -1)) >= 0
        })

        # 1. Blue: speech bubble boundaries used by accepted layout constraints.
        # Refined container outlines supersede the detector's mask contour
        # (which often hugs whitened glyph areas instead of the boundary).
        refined_bubble_ids = {
            int(c.get("bubble_idx", -1))
            for c in constraints
            if c.get("refined_bubble_outline") and int(c.get("bubble_idx", -1)) >= 0
        }
        if detect_dir.exists() and used_bubble_indices:
            for i in used_bubble_indices:
                if i in refined_bubble_ids:
                    continue
                bm_path = detect_dir / f"bubble_{i}.png"
                if not bm_path.exists():
                    continue
                bmask = cv2.imread(str(bm_path), cv2.IMREAD_GRAYSCALE)
                if bmask is not None:
                    contours, _ = cv2.findContours(
                        bmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
        for c in constraints:
            for outline_key in ("inferred_bubble_outline", "refined_bubble_outline"):
                container_outline = c.get(outline_key) or []
                if container_outline:
                    _draw_plain_polyline(image, container_outline, (255, 0, 0), 2)

        # 2. Red (erasure) + Green (layout)
        for c in constraints:
            rb   = c["red_box"]
            gb   = c["green_box"]
            poly = c.get("green_polygon", [])
            green_polys = c.get("green_polygons") or []
            red_poly = c.get("red_polygon", [])

            consolidated = bool(c.get("consolidated_container"))
            if consolidated:
                # One utterance in one container: a single red rectangle;
                # fragment/erase boxes stay out of the debug view.
                _draw_masked_rectangle(image, rb, (0, 0, 255), 2, debug_glyph_guard)
            elif c.get("art_aware_routed") and (c.get("erase_boxes") or []):
                _draw_routed_erase_outline(image, c.get("erase_boxes") or [], image.shape, debug_glyph_guard)
            elif red_poly and len(red_poly) >= 3:
                _draw_masked_polyline(image, red_poly, (0, 0, 255), 2, debug_glyph_guard)
            else:
                _draw_masked_rectangle(image, rb, (0, 0, 255), 2, debug_glyph_guard)
            show_fragment_boxes = (
                not consolidated
                and not str(c.get("fallback_source") or "").startswith("paddleocr_korean")
            )
            if not c.get("art_aware_routed") and show_fragment_boxes:
                for erase_box in c.get("erase_boxes", []) or []:
                    coerced_box = _coerce_box4(erase_box)
                    if coerced_box is None:
                        continue
                    _draw_recovered_fragment_box(image, coerced_box)

            green_guard = debug_layout_guard
            if c.get("art_aware_routed"):
                green_guard = debug_layout_guard.copy() if debug_layout_guard is not None else None
                if debug_art_guard is not None:
                    green_guard = debug_art_guard.copy() if green_guard is None else cv2.max(green_guard, debug_art_guard)
            if green_polys:
                for green_poly in green_polys:
                    if green_poly and len(green_poly) >= 3:
                        _draw_masked_polyline(image, green_poly, (0, 255, 0), 1, green_guard)
            elif poly and len(poly) >= 3:
                _draw_masked_polyline(image, poly, (0, 255, 0), 1, green_guard)
            else:
                _draw_masked_rectangle(image, gb, (0, 255, 0), 1, green_guard)

        cv2.imwrite(str(out_dir / "debug_layout_boxes.jpg"), image)
        print(f"  Saved {len(constraints)} constraints -> {out_dir}/debug_layout_boxes.jpg")


if __name__ == "__main__":
    run_step6_layout()
