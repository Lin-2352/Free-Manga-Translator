"""
Step 8 - Automated Typesetting (v11: Mask-Aware Fit)
====================================================
Renders English text onto the clean inpainted canvas.
1. Uses Step 6 green polygons as the hard placement boundary.
2. Fits the real outlined text bitmap, not just approximate text length.
3. Anchors each translation near its original source text center.
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
import re
import shutil
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from ml_region_lib import SAMPLE_MAP
from pipeline_paths import DEFAULT_SAMPLES_ROOT, sample_root_from_env


                                                                     
           
                                                                     
MAX_FONT_DIALOGUE = 96
MIN_FONT_SIZE = 8
MAX_OUTLINE_WIDTH = 3
MIN_OUTLINE_WIDTH = 1
MIN_LINE_SPACING = 1
LOW_RES_TARGET_WIDTH = 900
LOW_RES_MAX_RENDER_SCALE = 3
DARK_BACKGROUND_MEDIAN_LUMA = 90
DARK_BACKGROUND_P75_LUMA = 130

FONT_PATH = "C:/Windows/Fonts/comicbd.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "arialbd.ttf"

MODERN_REFERENCE_FONT_PATH = "C:/Windows/Fonts/comic.ttf"
if not os.path.exists(MODERN_REFERENCE_FONT_PATH):
    MODERN_REFERENCE_FONT_PATH = FONT_PATH

FLOATING_FONT_PATH = "C:/Windows/Fonts/arialnb.ttf"
if not os.path.exists(FLOATING_FONT_PATH):
    FLOATING_FONT_PATH = "arialbd.ttf"

DENSE_FONT_PATH = "C:/Windows/Fonts/arial.ttf"
if not os.path.exists(DENSE_FONT_PATH):
    DENSE_FONT_PATH = FONT_PATH


def _external_local_mode() -> bool:
    return os.environ.get("LOCAL_NLLB_TRANSLATION", "").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=512)
def _load_font(size: int, font_path: str = FONT_PATH) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


def _has_alphabetical(text: str) -> bool:
    return bool(re.search(r"[a-zA-Z0-9]", text))


def _normalize_text(text: str, uppercase: bool = True) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text.upper() if uppercase else text


def _outline_for_size(size: int, is_floating: bool = False) -> int:
    outline = max(MIN_OUTLINE_WIDTH, min(MAX_OUTLINE_WIDTH, int(round(size * 0.06))))
    if is_floating:
        return max(2, outline)
    return outline


def _line_spacing_for_size(size: int) -> int:
    return max(MIN_LINE_SPACING, int(round(size * 0.12)))


def _text_width(text: str, font: ImageFont.FreeTypeFont, draw: ImageDraw.ImageDraw) -> float:
    if not text:
        return 0.0
    return draw.textlength(text, font=font)


def _split_word_to_fit(
    word: str,
    font: ImageFont.FreeTypeFont,
    draw: ImageDraw.ImageDraw,
    target_width: float,
) -> list[str]:
    if _text_width(word, font, draw) <= target_width:
        return [word]

    trailing = ""
    core = word
    while core and core[-1] in ".,!?;:…":
        trailing = core[-1] + trailing
        core = core[:-1]
    if not core:
        core = word
        trailing = ""
    if len(core) < 12:
        return [word]

    best_pair = None
    best_pair_score = 1e18
    split_start = max(4, len(core) // 2 - 3)
    split_end = min(len(core) - 3, len(core) // 2 + 3)
    core_upper = core.upper()
    preferred_splits = set()
    for suffix in ("STAND", "TION", "MENT", "NESS", "ABLE", "IBLE", "ALLY", "ING"):
        if core_upper.endswith(suffix) and len(core) - len(suffix) >= 4:
            preferred_splits.add(len(core) - len(suffix))
    for prefix in ("MISUNDER", "UNDER", "INTER", "COUNTER", "TRANS", "OVER"):
        if core_upper.startswith(prefix) and len(core) - len(prefix) >= 3:
            preferred_splits.add(len(prefix))
    best_pair_is_preferred = False

    for split_at in range(split_start, split_end + 1):
        first = core[:split_at] + "-"
        second = core[split_at:] + trailing
        if _text_width(first, font, draw) > target_width:
            continue
        if _text_width(second, font, draw) > target_width:
            continue
        balance_score = abs(len(first) - len(second))
        is_preferred = split_at in preferred_splits
        if is_preferred:
            balance_score -= 6
        if balance_score < best_pair_score:
            best_pair = [first, second]
            best_pair_is_preferred = is_preferred
            best_pair_score = balance_score

    if best_pair and (not preferred_splits or best_pair_is_preferred):
        return best_pair

    return [word]


def _wrap_standard(
    words: list[str],
    font: ImageFont.FreeTypeFont,
    draw: ImageDraw.ImageDraw,
    target_width: float,
    allow_word_split: bool = False,
) -> list[str]:
    """
    Standard left-to-right greedy word wrap. Long words are split only when
    a single word would otherwise force clipping.
    """
    if not words:
        return []

    wrapped_words = []
    for word in words:
        if allow_word_split:
            wrapped_words.extend(_split_word_to_fit(word, font, draw, target_width))
        else:
            wrapped_words.append(word)

    lines = []
    current_words = []
    for word in wrapped_words:
        candidate = " ".join(current_words + [word])
        if _text_width(candidate, font, draw) <= target_width:
            current_words.append(word)
        else:
            if current_words:
                lines.append(" ".join(current_words))
            current_words = [word]

    if current_words:
        lines.append(" ".join(current_words))
    return lines


def _render_text_block(
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    size: int,
    outline_width: int,
    fill_color: tuple[int, int, int, int] = (0, 0, 0, 255),
    stroke_color: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> tuple[Image.Image, np.ndarray, dict]:
    measure_img = Image.new("L", (1, 1), 0)
    measure_draw = ImageDraw.Draw(measure_img)
    line_boxes = []
    line_widths = []
    line_heights = []
    spacing = _line_spacing_for_size(size)

    for line in lines:
        bbox = measure_draw.textbbox((0, 0), line or " ", font=font, stroke_width=outline_width)
        line_boxes.append(bbox)
        line_widths.append(max(1, bbox[2] - bbox[0]))
        line_heights.append(max(1, bbox[3] - bbox[1]))

    block_width = max(1, max(line_widths, default=1))
    block_height = max(1, sum(line_heights) + spacing * max(0, len(lines) - 1))
    block = Image.new("RGBA", (block_width + 2, block_height + 2), (0, 0, 0, 0))
    block_draw = ImageDraw.Draw(block)

    cursor_y = 1
    for line, bbox, line_width, line_height in zip(lines, line_boxes, line_widths, line_heights):
        line_x = 1 + (block_width - line_width) / 2 - bbox[0]
        line_y = cursor_y - bbox[1]
        block_draw.text(
            (line_x, line_y),
            line,
            font=font,
            fill=fill_color,
            stroke_width=outline_width,
            stroke_fill=stroke_color,
        )
        cursor_y += line_height + spacing

    alpha = np.array(block.getchannel("A")) > 0
    metrics = {
        "width": block.width,
        "height": block.height,
        "spacing": spacing,
        "outline_width": outline_width,
    }
    return block, alpha, metrics


def _text_style_for_layout(background: Image.Image, allowed_mask: Image.Image) -> dict:
    mask_np = np.array(allowed_mask) > 0
    if not np.any(mask_np):
        return {
            "name": "dark_on_light",
            "fill_color": (0, 0, 0, 255),
            "stroke_color": (255, 255, 255, 255),
            "background_luma_median": None,
            "background_luma_p75": None,
        }

    rgb = np.array(background.convert("RGB")).astype(np.float32)
    luminance = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    values = luminance[mask_np]
    median_luma = float(np.median(values))
    p75_luma = float(np.percentile(values, 75))
    dark_fraction = float(np.mean(values < 135))

    if (
        median_luma < DARK_BACKGROUND_MEDIAN_LUMA and p75_luma < DARK_BACKGROUND_P75_LUMA
    ) or dark_fraction >= 0.38:
        return {
            "name": "light_on_dark",
            "fill_color": (255, 255, 255, 255),
            "stroke_color": (0, 0, 0, 255),
            "background_luma_median": median_luma,
            "background_luma_p75": p75_luma,
        }

    return {
        "name": "dark_on_light",
        "fill_color": (0, 0, 0, 255),
        "stroke_color": (255, 255, 255, 255),
        "background_luma_median": median_luma,
        "background_luma_p75": p75_luma,
    }


def _floating_dialogue_layout(layout: dict) -> bool:
    role = str(layout.get("semantic_role", "") or "").lower()
    route = str(layout.get("route", "") or "").lower()
    fallback_source = str(layout.get("fallback_source", "") or "").lower()
    return (
        layout.get("bubble_idx", -1) == -1
        and (
            "dialogue" in role
            or "dialogue" in route
            or "caption" in role
            or "narration" in role
            or "adjacent_fragment" in fallback_source
        )
    )


def _overlay_fallback_mask_for_layout(layout: dict, image_size: tuple[int, int]) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    green_box = _coerce_box(layout["green_box"], image_size)
    red_box = _coerce_box(layout.get("red_box", layout["green_box"]), image_size)
    page_width, page_height = image_size
    box_width = max(green_box[2] - green_box[0], red_box[2] - red_box[0])
    box_height = max(green_box[3] - green_box[1], red_box[3] - red_box[1])
    expand_x = min(max(28, int(box_width * 0.32)), int(page_width * 0.16))
    expand_y = min(max(16, int(box_height * 0.16)), int(page_height * 0.08))
    overlay_box = _coerce_box(
        [
            min(green_box[0], red_box[0]) - expand_x,
            min(green_box[1], red_box[1]) - expand_y,
            max(green_box[2], red_box[2]) + expand_x,
            max(green_box[3], red_box[3]) + expand_y,
        ],
        image_size,
    )
    draw.rectangle(overlay_box, fill=255)
    return mask


def _overlay_badge_style(background: Image.Image, mask: Image.Image) -> dict:
    style = _text_style_for_layout(background, mask)
    median_luma = style.get("background_luma_median")
    if median_luma is not None and median_luma < 145:
        style.update({
            "name": "uncleaned_overlay_light_on_dark",
            "fill_color": (255, 255, 255, 255),
            "stroke_color": (0, 0, 0, 255),
            "badge_fill": (0, 0, 0, 205),
            "badge_outline": (255, 255, 255, 160),
        })
    else:
        style.update({
            "name": "uncleaned_overlay_dark_on_light",
            "fill_color": (0, 0, 0, 255),
            "stroke_color": (255, 255, 255, 255),
            "badge_fill": (255, 255, 255, 218),
            "badge_outline": (0, 0, 0, 150),
        })
    return style


def _source_cover_text_style(background: Image.Image, mask: Image.Image) -> dict:
    style = _text_style_for_layout(background, mask)
    style["name"] = f"{style['name']}_source_cover"
    style["source_cover"] = True
    return style


def _composite_overlay_badge(
    base: Image.Image,
    position: tuple[int, int],
    block_size: tuple[int, int],
    font_size: int,
    text_style: dict,
) -> Image.Image:
    left, top = position
    width, height = block_size
    pad_x = max(4, int(round(font_size * 0.24)))
    pad_y = max(3, int(round(font_size * 0.18)))
    badge_box = [
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(base.size[0] - 1, left + width + pad_x),
        min(base.size[1] - 1, top + height + pad_y),
    ]
    if badge_box[2] <= badge_box[0] or badge_box[3] <= badge_box[1]:
        return base
    badge_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge_layer)
    radius = max(3, min(10, int(round(font_size * 0.28))))
    draw.rounded_rectangle(
        badge_box,
        radius=radius,
        fill=text_style.get("badge_fill", (255, 255, 255, 218)),
        outline=text_style.get("badge_outline", (0, 0, 0, 150)),
        width=1,
    )
    return Image.alpha_composite(base, badge_layer)


def _dense_external_text(text: str) -> bool:
    return _external_local_mode() and len(str(text or "")) >= 72


def _compact_text_variant(text: str, max_words: int) -> str:
    words = str(text or "").split()
    if len(words) <= max_words:
        return str(text or "").strip()
    trimmed = " ".join(words[:max_words]).rstrip(".,;:")
    return f"{trimmed}..."


def _layout_text_variants(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return []

    variants = [clean]
    if _external_local_mode() and len(clean) >= 55:
        no_parenthetical = re.sub(r"\([^)]{1,80}\)", "", clean)
        no_parenthetical = re.sub(r"\s+", " ", no_parenthetical).strip(" ,.;:")
        if no_parenthetical and no_parenthetical not in variants:
            variants.append(no_parenthetical)

        first_sentence = re.split(r"(?<=[.!?])\s+", clean, maxsplit=1)[0].strip()
        if 16 <= len(first_sentence) < len(clean) and first_sentence not in variants:
            variants.append(first_sentence)

        first_clause = re.split(r"[,;:]\s+", clean, maxsplit=1)[0].strip()
        if 16 <= len(first_clause) < len(clean) and first_clause not in variants:
            variants.append(first_clause)

        for max_words in (16, 12, 9):
            compact = _compact_text_variant(clean, max_words)
            if compact and compact not in variants:
                variants.append(compact)

    return variants


def _font_path_for_layout(layout: dict, text: str = "") -> str:
    if layout.get("fallback_source") == "peppercarrot_svg" and layout.get("bubble_idx", -1) != -1:
        return MODERN_REFERENCE_FONT_PATH
    if _dense_external_text(text):
        return DENSE_FONT_PATH
    if layout.get("bubble_idx", -1) == -1:
        return FLOATING_FONT_PATH
    return FONT_PATH


def _render_scale_for_image(image_size: tuple[int, int]) -> int:
    width, _ = image_size
    if width >= LOW_RES_TARGET_WIDTH:
        return 1
    return min(LOW_RES_MAX_RENDER_SCALE, max(1, math.ceil(LOW_RES_TARGET_WIDTH / width)))


def _scale_box(box: list[int] | tuple[int, int, int, int], scale: int) -> list[int]:
    return [int(round(value * scale)) for value in box]


def _scale_layout(layout: dict, scale: int) -> dict:
    if scale == 1:
        return dict(layout)

    scaled = dict(layout)
    scaled["red_box"] = _scale_box(layout["red_box"], scale)
    scaled["green_box"] = _scale_box(layout["green_box"], scale)
    polygon = layout.get("green_polygon") or []
    scaled["green_polygon"] = [
        [int(round(point[0] * scale)), int(round(point[1] * scale))]
        for point in polygon
    ]
    return scaled


def _coerce_box(
    box: list[int] | tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = image_size
    left = max(0, min(width, int(round(box[0]))))
    top = max(0, min(height, int(round(box[1]))))
    right = max(0, min(width, int(round(box[2]))))
    bottom = max(0, min(height, int(round(box[3]))))
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return left, top, right, bottom


def _bbox_from_mask(mask: Image.Image) -> tuple[int, int, int, int] | None:
    mask_np = np.array(mask) > 0
    ys, xs = np.where(mask_np)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _allowed_mask_for_layout(
    layout: dict,
    image_size: tuple[int, int],
    detect_dir: Path,
    protection_image: Image.Image | None = None,
) -> Image.Image:
    allowed = Image.new("L", image_size, 0)
    allowed_draw = ImageDraw.Draw(allowed)
    polygon = layout.get("green_polygon") or []
    bubble_idx = layout.get("bubble_idx", -1)

    if bubble_idx == -1:
        green_box = _coerce_box(layout["green_box"], image_size)
        box_width = green_box[2] - green_box[0]
        box_height = green_box[3] - green_box[1]
        page_width, page_height = image_size
        max_expand_x = max(10, int(round(page_width * 0.035)))
        max_expand_y = max(4, int(round(page_height * 0.010)))
        wide_horizontal = box_width >= box_height * 1.45
        if wide_horizontal:
            expand_x = min(8, max(2, int(round(box_width * 0.025))))
            expand_y = min(6, max(2, int(round(box_height * 0.05))))
        elif box_height > box_width * 1.35:
            expand_x = min(10, max(2, int(round(box_width * 0.10))))
            expand_y = min(max_expand_y, max(2, int(round(box_height * 0.025))))
        else:
            expand_x = min(max_expand_x, max(4, int(round(box_width * 0.10))))
            expand_y = min(max_expand_y, max(2, int(round(box_height * 0.04))))
        expanded_box = _coerce_box(
            [
                green_box[0] - expand_x,
                green_box[1] - expand_y,
                green_box[2] + expand_x,
                green_box[3] + expand_y,
            ],
            image_size,
        )
        allowed_draw.rectangle(expanded_box, fill=255)
    elif len(polygon) >= 3:
        points = [(int(round(point[0])), int(round(point[1]))) for point in polygon]
        allowed_draw.polygon(points, fill=255)
    else:
        allowed_draw.rectangle(_coerce_box(layout["green_box"], image_size), fill=255)

    if bubble_idx != -1:
        bubble_mask_path = detect_dir / f"bubble_{bubble_idx}.png"
        if bubble_mask_path.exists():
            bubble_mask = Image.open(str(bubble_mask_path)).convert("L")
            if bubble_mask.size != image_size:
                bubble_mask = bubble_mask.resize(image_size, Image.Resampling.NEAREST)
            allowed = ImageChops.multiply(allowed, bubble_mask)

    if _bbox_from_mask(allowed) is None:
        fallback = Image.new("L", image_size, 0)
        fallback_draw = ImageDraw.Draw(fallback)
        fallback_draw.rectangle(_coerce_box(layout["green_box"], image_size), fill=255)
        allowed = fallback

    if bubble_idx == -1 and layout.get("erase_boxes") and protection_image is not None:
        allowed_np = np.array(allowed) > 0
        gray = np.array(protection_image.convert("L"))
        if gray.shape == allowed_np.shape:
            dark_ink = (gray < 105) & allowed_np
            erase_np = np.zeros_like(allowed_np, dtype=bool)
            for erase_box in layout.get("erase_boxes", []):
                if not isinstance(erase_box, (list, tuple)) or len(erase_box) < 4:
                    continue
                ex1, ey1, ex2, ey2 = _coerce_box(erase_box[:4], image_size)
                ex1 = max(0, ex1 - 4)
                ey1 = max(0, ey1 - 4)
                ex2 = min(image_size[0], ex2 + 4)
                ey2 = min(image_size[1], ey2 + 4)
                if ex2 > ex1 and ey2 > ey1:
                    erase_np[ey1:ey2, ex1:ex2] = True
                                                                           
                                                                             
                                                                           
            dark_ink &= ~erase_np
            dark_ink = cv2.dilate(
                dark_ink.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=1,
            ) > 0
            protected_allowed = allowed_np & ~dark_ink
            if np.count_nonzero(protected_allowed) >= max(80, int(np.count_nonzero(allowed_np) * 0.18)):
                allowed = Image.fromarray((protected_allowed.astype(np.uint8) * 255), mode="L")

    return allowed


def _box_center(box: list[int] | tuple[int, int, int, int]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _clamp_top_left(
    center: tuple[float, float],
    block_size: tuple[int, int],
    bounds: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    block_width, block_height = block_size
    left, top, right, bottom = bounds
    min_x = left
    min_y = top
    max_x = right - block_width
    max_y = bottom - block_height
    if max_x < min_x or max_y < min_y:
        return None

    raw_x = int(round(center[0] - block_width / 2))
    raw_y = int(round(center[1] - block_height / 2))
    return min(max(raw_x, min_x), max_x), min(max(raw_y, min_y), max_y)


def _candidate_positions(
    block_size: tuple[int, int],
    bounds: tuple[int, int, int, int],
    anchor_center: tuple[float, float],
    green_center: tuple[float, float],
) -> list[tuple[int, int]]:
    positions = []
    seen = set()
    base_centers = [
        anchor_center,
        green_center,
        ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2),
    ]
    block_width, block_height = block_size
    search_step = max(2, min(block_width, block_height) // 10)
    offsets = [
        (0, 0),
        (-search_step, 0),
        (search_step, 0),
        (0, -search_step),
        (0, search_step),
    ]

    for center_x, center_y in base_centers:
        for offset_x, offset_y in offsets:
            position = _clamp_top_left(
                (center_x + offset_x, center_y + offset_y),
                block_size,
                bounds,
            )
            if position and position not in seen:
                positions.append(position)
                seen.add(position)

    left, top, right, bottom = bounds
    block_width, block_height = block_size
    max_x = right - block_width
    max_y = bottom - block_height
    if max_x >= left and max_y >= top:
        x_values = np.linspace(left, max_x, num=min(5, max(2, (max_x - left) // max(1, block_width // 2) + 1)))
        y_values = np.linspace(top, max_y, num=min(5, max(2, (max_y - top) // max(1, block_height // 2) + 1)))
        for grid_y in y_values:
            for grid_x in x_values:
                position = (int(round(grid_x)), int(round(grid_y)))
                if position not in seen:
                    positions.append(position)
                    seen.add(position)

    return positions


def _alpha_fits(alpha: np.ndarray, allowed_np: np.ndarray, position: tuple[int, int]) -> bool:
    left, top = position
    height, width = alpha.shape
    if left < 0 or top < 0:
        return False
    region = allowed_np[top:top + height, left:left + width]
    if region.shape != alpha.shape:
        return False
    return not np.any(alpha & ~region)


def _count_clipped_pixels(alpha: np.ndarray, allowed_np: np.ndarray, position: tuple[int, int]) -> int:
    left, top = position
    height, width = alpha.shape
    region = allowed_np[top:top + height, left:left + width]
    if region.shape != alpha.shape:
        return int(alpha.sum())
    return int(np.count_nonzero(alpha & ~region))


def _find_mask_aware_layout(
    text: str,
    layout: dict,
    allowed_mask: Image.Image,
    text_style: dict,
) -> dict:
    words = text.split()
    if not words:
        words = [text]

    image_size = allowed_mask.size
    green_box = _coerce_box(layout["green_box"], image_size)
    mask_bounds = _bbox_from_mask(allowed_mask) or green_box
    bounds_width = max(1, mask_bounds[2] - mask_bounds[0])
    bounds_height = max(1, mask_bounds[3] - mask_bounds[1])
    allowed_np = np.array(allowed_mask) > 0
    measure_draw = ImageDraw.Draw(Image.new("L", (1, 1), 0))
    anchor_center = _box_center(layout.get("red_box", green_box))
    green_center = _box_center(green_box)
    is_floating = layout.get("bubble_idx", -1) == -1
    tall_narrow_floating = bool(is_floating and bounds_height >= bounds_width * 1.35)
    modern_reference_bubble = (
        layout.get("fallback_source") == "peppercarrot_svg"
        and not is_floating
    )
    font_path = _font_path_for_layout(layout, text)

    max_size = min(MAX_FONT_DIALOGUE, max(MIN_FONT_SIZE, int(bounds_height * 0.9)))
    if is_floating:
        max_size = min(max_size, 72 if text_style.get("source_cover") else 32)
    if modern_reference_bubble:
        max_size = min(max_size, max(MIN_FONT_SIZE, int(bounds_height * 0.46)))
    font_sizes = list(range(max_size, MIN_FONT_SIZE - 1, -2))
    if MIN_FONT_SIZE not in font_sizes:
        font_sizes.append(MIN_FONT_SIZE)

    best_candidate = None
    best_score = -1e18

    for allow_word_split in (False, True):
        if tall_narrow_floating:
            width_ratios = np.linspace(
                1.0,
                0.62 if not allow_word_split else 0.34,
                9 if not allow_word_split else 10,
            )
        else:
            width_ratios = np.linspace(
                1.0,
                0.55 if not allow_word_split else 0.42,
                8 if not allow_word_split else 7,
            )

        for size in font_sizes:
            font = _load_font(size, font_path)
            outline_width = 0 if modern_reference_bubble else _outline_for_size(size, is_floating=is_floating)
            if text_style.get("source_cover"):
                outline_width = max(outline_width, min(5, int(round(size * 0.12))))

            for width_ratio in width_ratios:
                target_width = max(8.0, bounds_width * float(width_ratio))
                lines = _wrap_standard(
                    words,
                    font,
                    measure_draw,
                    target_width,
                    allow_word_split=allow_word_split,
                )
                if not lines:
                    continue

                block, alpha, metrics = _render_text_block(
                    lines,
                    font,
                    size,
                    outline_width,
                    text_style["fill_color"],
                    text_style["stroke_color"],
                )
                if block.width > bounds_width or block.height > bounds_height:
                    continue

                positions = _candidate_positions(
                    (block.width, block.height),
                    mask_bounds,
                    anchor_center,
                    green_center,
                )

                for position in positions:
                    if not _alpha_fits(alpha, allowed_np, position):
                        continue

                    block_center = (position[0] + block.width / 2, position[1] + block.height / 2)
                    anchor_distance = math.hypot(
                        block_center[0] - anchor_center[0],
                        block_center[1] - anchor_center[1],
                    )
                    aspect_ratio = block.width / max(1, block.height)
                    aspect_penalty = abs(math.log(max(0.1, min(10.0, aspect_ratio)))) * 45.0
                    line_lengths = [len(line.replace(" ", "")) for line in lines]
                    if tall_narrow_floating:
                        short_line_penalty = sum(max(0, 3 - line_length) for line_length in line_lengths) * 7.0
                        line_penalty = max(0, len(lines) - 9) * 8.0
                        split_penalty = 90.0 if allow_word_split else 0.0
                    elif is_floating:
                        short_line_penalty = sum(max(0, 4 - line_length) for line_length in line_lengths) * 12.0
                        line_penalty = max(0, len(lines) - 6) * 14.0
                        split_penalty = 160.0 if allow_word_split else 0.0
                    else:
                        short_line_penalty = sum(max(0, 4 - line_length) for line_length in line_lengths) * 22.0
                        line_penalty = max(0, len(lines) - 5) * 22.0
                        split_penalty = 240.0 if allow_word_split else 0.0
                    area_usage = (block.width * block.height) / max(1, bounds_width * bounds_height)
                    score = (
                        size * 38.0
                        + area_usage * 260.0
                        - anchor_distance * 0.8
                        - aspect_penalty
                        - line_penalty
                        - short_line_penalty
                        - split_penalty
                    )

                    candidate = {
                        "font_size": size,
                        "lines": lines,
                        "block": block,
                        "alpha": alpha,
                        "position": position,
                        "metrics": metrics,
                        "status": "fit" if not allow_word_split else "fit_with_word_split",
                        "clipped_pixels": 0,
                    }
                    if score > best_score:
                        best_candidate = candidate
                        best_score = score

        if (
            not allow_word_split
            and best_candidate is not None
            and best_candidate["font_size"] >= max(MIN_FONT_SIZE, max_size - 6)
            and len(best_candidate["lines"]) <= 6
            and best_candidate["status"] == "fit"
        ):
            return best_candidate

    if best_candidate is not None:
        return best_candidate

    fallback_font_size = MIN_FONT_SIZE
    fallback_font = _load_font(fallback_font_size, font_path)
    fallback_outline = 0 if modern_reference_bubble else _outline_for_size(fallback_font_size, is_floating=is_floating)
    if text_style.get("source_cover"):
        fallback_outline = max(fallback_outline, min(5, int(round(fallback_font_size * 0.12))))
    fallback_lines = _wrap_standard(
        words,
        fallback_font,
        measure_draw,
        max(8.0, bounds_width * 0.9),
        allow_word_split=True,
    )
    fallback_block, fallback_alpha, fallback_metrics = _render_text_block(
        fallback_lines or words,
        fallback_font,
        fallback_font_size,
        fallback_outline,
        text_style["fill_color"],
        text_style["stroke_color"],
    )
    fallback_position = _clamp_top_left(
        green_center,
        (fallback_block.width, fallback_block.height),
        mask_bounds,
    ) or (mask_bounds[0], mask_bounds[1])
    return {
        "font_size": fallback_font_size,
        "lines": fallback_lines,
        "block": fallback_block,
        "alpha": fallback_alpha,
        "position": fallback_position,
        "metrics": fallback_metrics,
        "status": "fallback_clipped",
        "clipped_pixels": _count_clipped_pixels(fallback_alpha, allowed_np, fallback_position),
    }


def run_step8_typeset():
    print("=" * 60)
    print("  Step 8 - Automated Typesetting (Mask-Aware Fit)")
    print("=" * 60)

    samples_dir = sample_root_from_env(DEFAULT_SAMPLES_ROOT)

    for sample_name, img_file in SAMPLE_MAP.items():
        sample_path = samples_dir / sample_name
        canvas_path = sample_path / "step_4_final" / "inpainted_result.jpg"
        cleanup_status_path = sample_path / "step_4_final" / "cleanup_status.json"
        layout_path = sample_path / "step_6_layout" / "layout_constraints.json"
        trans_path = sample_path / "step_7_translate" / "translation_results.json"
        detect_dir = sample_path / "step_1_detect"

        if not (canvas_path.exists() and layout_path.exists() and trans_path.exists()):
            print(f"  SKIP {sample_name}: Missing prerequisites")
            continue

        print(f"\nProcessing {sample_name}")
        image = cv2.imread(str(canvas_path))
        layout_data = json.loads(layout_path.read_text(encoding="utf-8"))
        trans_data = json.loads(trans_path.read_text(encoding="utf-8"))
        cleanup_status = {}
        if cleanup_status_path.exists():
            cleanup_status = json.loads(cleanup_status_path.read_text(encoding="utf-8"))
        trans_map = {item["id"]: item for item in trans_data}

        native_size = (image.shape[1], image.shape[0])
        render_scale = _render_scale_for_image(native_size)
        if render_scale > 1:
            image = cv2.resize(
                image,
                (native_size[0] * render_scale, native_size[1] * render_scale),
                interpolation=cv2.INTER_LANCZOS4,
            )
            layout_data = [_scale_layout(layout, render_scale) for layout in layout_data]
            print(f"  Low-res page detected; rendering final at {render_scale}x")
        else:
            layout_data = [dict(layout) for layout in layout_data]

        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
        image_size = pil_img.size
        report = []

        for layout in layout_data:
            tid = layout["id"]
            if tid not in trans_map:
                continue

            trans_item = trans_map[tid]
            en_text = trans_item["en_text"]
            if not en_text:
                continue

            if en_text.startswith("[TL:"):
                en_text = en_text.replace("[TL: ", "").replace("]", "")
                if "..." in en_text:
                    en_text = en_text.split("...")[0]

            if not _has_alphabetical(en_text):
                continue
            cleanup_reason = None
            force_overlay_badge = False
            force_source_cover = False
            if layout.get("bubble_idx", -1) == -1:
                status = cleanup_status.get(str(tid))
                if status is not None and not status.get("cleaned", True):
                    cleanup_reason = status.get("reason", "unsafe_art_preserved")
                    if cleanup_reason == "device_overlay_required":
                        force_overlay_badge = True
                    elif cleanup_reason == "source_cover_required":
                        force_source_cover = True
                    else:
                        report.append({
                            "id": tid,
                            "text": en_text,
                            "status": "skipped_unsafe_floating_cleanup",
                            "reason": cleanup_reason,
                            "bubble_idx": -1,
                            "font_role": "floating",
                        })
                        continue
            if force_overlay_badge:
                allowed_mask = _overlay_fallback_mask_for_layout(layout, image_size)
                text_style = _overlay_badge_style(pil_img, allowed_mask)
            elif force_source_cover:
                allowed_mask = _allowed_mask_for_layout(layout, image_size, detect_dir, None)
                text_style = _source_cover_text_style(pil_img, allowed_mask)
            else:
                allowed_mask = _allowed_mask_for_layout(layout, image_size, detect_dir, pil_img)
                text_style = _text_style_for_layout(pil_img, allowed_mask)
            if layout.get("bubble_idx", -1) == -1 and not force_overlay_badge and not force_source_cover:
                red_style_mask = Image.new("L", image_size, 0)
                ImageDraw.Draw(red_style_mask).rectangle(
                    _coerce_box(layout.get("red_box", layout["green_box"]), image_size),
                    fill=255,
                )
                red_style = _text_style_for_layout(pil_img, red_style_mask)
                if red_style["name"] == "light_on_dark":
                    text_style = red_style
            allowed_np = np.array(allowed_mask) > 0

            selected_text = None
            selected_fit = None
            selected_clipped_pixels = None
            preserve_case = (
                trans_item.get("provider") == "peppercarrot_official_svg_reference"
                or layout.get("fallback_source") == "peppercarrot_svg"
            )
            for variant in _layout_text_variants(en_text):
                candidate_text = _normalize_text(
                    variant,
                    uppercase=(not preserve_case and not _dense_external_text(variant)),
                )
                fitted_candidate = _find_mask_aware_layout(
                    candidate_text,
                    layout,
                    allowed_mask,
                    text_style,
                )
                clipped_candidate = _count_clipped_pixels(
                    fitted_candidate["alpha"],
                    allowed_np,
                    fitted_candidate["position"],
                )
                if selected_fit is None:
                    selected_text = candidate_text
                    selected_fit = fitted_candidate
                    selected_clipped_pixels = clipped_candidate
                if clipped_candidate == 0 and fitted_candidate["status"] != "fallback_clipped":
                    selected_text = candidate_text
                    selected_fit = fitted_candidate
                    selected_clipped_pixels = 0
                    break

            if selected_fit is None or selected_text is None:
                continue

            en_text = selected_text
            fitted = selected_fit
            text_block = fitted["block"]
            alpha = fitted["alpha"]
            left, top = fitted["position"]

            text_layer = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
            text_layer.alpha_composite(text_block, (left, top))
            if force_overlay_badge:
                pil_img = _composite_overlay_badge(
                    pil_img,
                    (left, top),
                    (text_block.width, text_block.height),
                    fitted["font_size"],
                    text_style,
                )

            clipped_pixels = int(selected_clipped_pixels or 0)
            if clipped_pixels:
                safe_alpha = ImageChops.multiply(text_layer.getchannel("A"), allowed_mask)
                text_layer.putalpha(safe_alpha)

            pil_img = Image.alpha_composite(pil_img, text_layer)

            report.append({
                "id": tid,
                "text": en_text,
                "font_size": fitted["font_size"],
                "lines": fitted["lines"],
                "position": [left, top, left + text_block.width, top + text_block.height],
                "status": fitted["status"],
                "reason": cleanup_reason,
                "overlay_badge": force_overlay_badge,
                "source_cover": force_source_cover,
                "clipped_pixels": clipped_pixels,
                "outline_width": fitted["metrics"]["outline_width"],
                "bubble_idx": layout.get("bubble_idx", -1),
                "render_scale": render_scale,
                "native_size": list(native_size),
                "output_size": [image_size[0], image_size[1]],
                "font_role": "floating" if layout.get("bubble_idx", -1) == -1 else "dialogue",
                "text_style": text_style["name"],
                "background_luma_median": (
                    None
                    if text_style["background_luma_median"] is None
                    else round(text_style["background_luma_median"], 2)
                ),
                "background_luma_p75": (
                    None
                    if text_style["background_luma_p75"] is None
                    else round(text_style["background_luma_p75"], 2)
                ),
            })

        out_dir = sample_path / "step_8_typeset"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        final_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / "final_output.jpg"), final_img, [cv2.IMWRITE_JPEG_QUALITY, 97])
        cv2.imwrite(str(out_dir / "final_output.png"), final_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        (out_dir / "typeset_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved final output to {out_dir}")


if __name__ == "__main__":
    run_step8_typeset()
