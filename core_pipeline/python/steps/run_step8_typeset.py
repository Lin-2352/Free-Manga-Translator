"""
Step 8 - Automated Typesetting (v11: Mask-Aware Fit)
====================================================
Renders English text onto the clean inpainted canvas.
1. Uses Step 6 green polygons as the hard placement boundary.
2. Fits the real outlined text bitmap, not just approximate text length.
3. Anchors each translation near its original source text center.
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
import re
import shutil
from functools import lru_cache
from pathlib import Path

# torch must be imported before cv2 in this process -- see run_step5_ocr.py
# for the full explanation (verified import-order segfault reproduction).
import torch  # noqa: F401  (import-order guard, see comment above)
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

try:
    import pyphen as _pyphen
    _HYPHEN_DICT = _pyphen.Pyphen(lang="en_US")
except Exception:
    _HYPHEN_DICT = None

from ml_region_lib import SAMPLE_MAP
from pipeline_paths import DEFAULT_SAMPLES_ROOT, sample_root_from_env


# ===================================================================
# CONSTANTS
# ===================================================================
MAX_FONT_DIALOGUE = 96
MIN_FONT_SIZE = 12
MIN_READABLE_DIALOGUE_SIZE = 16
MIN_READABLE_FLOATING_SIZE = 18
TARGET_BUBBLE_AREA_USAGE = 0.45
MAX_OUTLINE_WIDTH = 3
MIN_OUTLINE_WIDTH = 1
MIN_LINE_SPACING = 1
LOW_RES_TARGET_WIDTH = 900
LOW_RES_MAX_RENDER_SCALE = 3
DARK_BACKGROUND_MEDIAN_LUMA = 90
DARK_BACKGROUND_P75_LUMA = 130

def _bundled_font_dir() -> str:
    """Absolute path to the bundled fonts, which live INSIDE core_pipeline/.

    This file is core_pipeline/python/steps/run_step8_typeset.py, so three dirname()
    levels reach core_pipeline/.

    They used to sit at the REPO ROOT (four levels up), one directory outside
    core_pipeline/ -- and build_kaggle_dataset.ps1 stages only core_pipeline/*, so the
    fonts never reached Kaggle. The cascade silently fell through to DejaVu there: no
    error, still one font per page, just not the font anyone asked for. Keeping them
    inside the deployable unit is what makes the Kaggle and local paths agree.

    Single source of truth on purpose -- this computation used to be duplicated inline
    below, and a move that updated only one copy is exactly the producer/consumer drift
    that silently disabled the Chinese OCR path.
    """
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "fonts",
    )


FONT_PATH = "C:/Windows/Fonts/comicbd.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "arialbd.ttf"
if not os.path.exists(FONT_PATH):
    # Linux/Kaggle: bundled Comic Neue (open-source Comic Sans alternative, manga-style)
    FONT_PATH = os.path.join(_bundled_font_dir(), "ComicNeue-Bold.ttf")
if not os.path.exists(FONT_PATH):
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# MODERN_REFERENCE_FONT_PATH / FLOATING_FONT_PATH / NARROW_FLOATING_FONT_PATH /
# DENSE_FONT_PATH used to live here -- four more role-keyed cascades that
# _font_path_for_layout() picked between PER REGION, which is what put two different
# font families on 16 of 34 sample pages. Region role still drives font SIZE, outline
# width and wrapping; it no longer drives font FAMILY, so these constants had no
# remaining readers and are removed rather than left as dead code that reads as live.


# ---------------------------------------------------------------------------
# One font per page.
#
# The five constants above are five INDEPENDENT role-keyed cascades, and
# _font_path_for_layout() used to pick between them per REGION -- so a page with one
# speech bubble plus one floating caption rendered Comic Sans Bold beside Arial Bold.
# Measured across the 34-sample fixture suite: 16 of 34 pages (47%) shipped with two
# different font families on the same page. Users read that as "the fonts keep
# changing," and separately, the extension popup has had a font picker
# (popup.html `fontSelect` -> `mangaFontStyle`) whose value was never sent to the
# backend and never consulted here at all.
#
# `family` is the popup's own option value. Each entry keeps the SAME degradation
# chain the originals used (Windows font -> bundled Comic Neue -> DejaVu), so an
# unavailable font degrades to something legible instead of raising. Families we do
# not ship a real file for resolve to the default rather than silently rendering as
# something unrelated -- and _resolve_font_family() reports what it actually picked so
# the substitution is visible in typeset_report.json rather than invisible.
# Only families we can actually deliver. The popup used to also offer CC Wild Words,
# Bangers and Patrick Hand; no font file for any of them exists in this repo, in the
# extension package, or in C:/Windows/Fonts, so all three silently rendered as Comic
# Sans. They are not re-added here: those files only ever entered this repo as a
# byproduct of vendoring a third-party extension and were deliberately removed again in
# b55ac43 ("Polish repository for public release"), and CC Wild Words in particular is a
# commercial Comicraft font with no license artifact anywhere in the tree.
#
# Comic Neue is the default because it is the ONLY option that resolves to the identical
# file on Windows and on Kaggle/Linux -- the other two degrade to it off-Windows, so
# picking either of them yields different output per platform, which is the very
# inconsistency this change exists to remove.
FONT_FAMILY_FILES: dict[str, list[str]] = {
    "comic neue": ["ComicNeue-Bold.ttf"],
    "comic sans ms": ["C:/Windows/Fonts/comicbd.ttf", "ComicNeue-Bold.ttf"],
    "arial": ["C:/Windows/Fonts/arialbd.ttf", "ComicNeue-Bold.ttf"],
}

DEFAULT_FONT_FAMILY = "comic neue"


def _resolve_font_family(family: str | None) -> tuple[str, str]:
    """Resolve a popup font-family name to a concrete, existing font file.

    Returns (path, resolved_label). `resolved_label` names what was actually used --
    it differs from the requested family when we had to fall back, and it is recorded
    in typeset_report.json so a substitution is auditable instead of silent.
    """
    def _first_existing(fam: str) -> str | None:
        for candidate in FONT_FAMILY_FILES.get(fam, []):
            path = candidate if os.path.isabs(candidate) else os.path.join(_bundled_font_dir(), candidate)
            if os.path.exists(path):
                return path
        return None

    requested = (family or "").strip().lower()
    hit = _first_existing(requested)
    if hit:
        return hit, requested

    # Unknown family (or its files missing): fall back to DEFAULT_FONT_FAMILY rather than
    # to raw FONT_PATH. Those are not the same thing -- FONT_PATH resolves to Comic Sans
    # on Windows but to bundled Comic Neue on Linux/Kaggle, so returning it here made a
    # typo'd font name render a DIFFERENT typeface per platform, and made an unknown name
    # disagree with an omitted one. Caught by driving the real endpoint with
    # fontFamily="Nonexistent Font": it returned comicbd.ttf while an omitted field
    # returned ComicNeue-Bold.ttf.
    label = f"{requested}->default" if requested else "default"
    fallback = _first_existing(DEFAULT_FONT_FAMILY)
    if fallback:
        return fallback, label
    # Last resort only if even the default family's files are missing.
    return FONT_PATH, label


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
    text = text.translate({
        ord("\u00a0"): " ",
        ord("\u2010"): "-",
        ord("\u2011"): "-",
        ord("\u2012"): "-",
        ord("\u2013"): "-",
        ord("\u2014"): "-",
        ord("\u2212"): "-",
    })
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


def _readable_floor_for_bounds(bounds_width: int, bounds_height: int, is_floating: bool) -> int:
    if is_floating:
        shortest_side = max(1, min(bounds_width, bounds_height))
        proportional_floor = int(round(shortest_side * 0.18))
        return max(MIN_FONT_SIZE, min(32, max(MIN_READABLE_FLOATING_SIZE, proportional_floor)))
    shortest_side = max(1, min(bounds_width, bounds_height))
    proportional_floor = int(round(shortest_side * 0.16))
    return max(MIN_FONT_SIZE, min(30, max(MIN_READABLE_DIALOGUE_SIZE, proportional_floor)))


def _text_width(text: str, font: ImageFont.FreeTypeFont, draw: ImageDraw.ImageDraw) -> float:
    if not text:
        return 0.0
    return draw.textlength(text, font=font)


def _split_word_to_fit(
    word: str,
    font: ImageFont.FreeTypeFont,
    draw: ImageDraw.ImageDraw,
    target_width: float,
    min_split_length: int = 8,
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
    # The default 8-char gate is intentionally left unchanged for the main
    # best-candidate search (run_step8_typeset's `for allow_word_split in
    # (False, True)` loop), which scores a split candidate against a
    # split_penalty and can select it purely because it also permits a much
    # larger font size -- lowering the gate there let 6-7 char words that
    # already rendered fine unsplit (e.g. "WORKERS...") get needlessly
    # hyphenated into a bigger font just because the scoring allowed it, a
    # regression on samples that needed no fix. `min_split_length` lets the
    # narrow last-resort fallback path (used only when NO tested font size
    # avoids clipping) pass 6 instead, since ITS suffix-aware heuristic
    # (used when pyphen isn't installed) already anticipates splitting words
    # as short as 7 characters (needs only a 4-char prefix before a suffix
    # like "ING": len(core) - len(suffix) >= 4) -- one shorter than the old
    # blanket 8-char gate allowed, silently blocking exactly those words.
    # Verified regression: external_ja_1's "MOORING POST!" in a very narrow
    # bubble (target width ~28px at the font-size floor) never got
    # hyphenated because "MOORING" is 7 characters, so the whole word
    # clipped instead, rendering as "OOR1. POST!". Scoping the lower
    # threshold to only the fallback path fixes that case while a full
    # 33-sample true-before/after diff confirms zero effect on any sample
    # that didn't already need the fallback (the main search's behavior for
    # every other word, at every length, is provably unchanged since its own
    # calls still pass the default 8).
    if len(core) < min_split_length:
        return [word]

    # A word that already contains a natural hyphen (e.g. "middle-aged")
    # should break there first, rather than have the dictionary look for
    # an unrelated internal syllable boundary -- splitting at an existing
    # hyphen is always linguistically valid and needs no further lookup.
    if "-" in core:
        hyphen_at = core.index("-") + 1
        first = core[:hyphen_at]
        second = core[hyphen_at:] + trailing
        if (
            1 < hyphen_at < len(core)
            and _text_width(first, font, draw) <= target_width
            and _text_width(second, font, draw) <= target_width
        ):
            return [first, second]

    # Prefer a real dictionary hyphenation point (Liang's algorithm via
    # pyphen) over a blind midpoint split -- this is what actually fixes
    # invalid breaks like "PREPAR-ING"/"SHOU-LD'VE"/"CINN-AMON": pyphen
    # returns the linguistically valid syllable boundaries for a word, or
    # explicitly none at all for words that should never be split (e.g.
    # contractions like "should've"). When pyphen is available we trust it
    # fully -- including its "don't split this" verdict -- rather than
    # falling back to the old midpoint heuristic, which is exactly what
    # produced "SHOU-LD'VE" in the first place. The midpoint heuristic
    # below is used only when pyphen itself isn't installed.
    if _HYPHEN_DICT is not None:
        dict_positions = [
            pos for pos in _HYPHEN_DICT.positions(core)
            if 2 <= pos <= len(core) - 2
        ]
        best_dict_pair = None
        best_dict_score = 1e18
        for split_at in dict_positions:
            first = core[:split_at] + "-"
            second = core[split_at:] + trailing
            if _text_width(first, font, draw) > target_width:
                continue
            if _text_width(second, font, draw) > target_width:
                continue
            balance_score = abs(len(first) - len(second))
            if balance_score < best_dict_score:
                best_dict_pair = [first, second]
                best_dict_score = balance_score
        return best_dict_pair if best_dict_pair else [word]

    # Pure fallback for environments without pyphen installed.
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


def _multipiece_split_word(
    word: str,
    font: ImageFont.FreeTypeFont,
    draw: ImageDraw.ImageDraw,
    target_width: float,
    max_pieces: int = 6,
) -> list[str]:
    """Last-resort greedy multi-piece hyphenation for a single word that still
    doesn't fit target_width after _split_word_to_fit's normal (at most
    2-piece) split -- e.g. a long compound proper noun in a very narrow
    bubble. Only ever called from the fallback_clipped retry path in
    _find_mask_aware_layout, at the one font size that already minimized
    clipping; returns [word] unsplit on any failure so the caller can safely
    compare the result against its current best candidate and discard it.
    """
    if _text_width(word, font, draw) <= target_width:
        return [word]

    trailing = ""
    core = word
    while core and core[-1] in ".,!?;:…":
        trailing = core[-1] + trailing
        core = core[:-1]
    if not core:
        return [word]

    # Candidate internal break points: dictionary syllable boundaries when
    # available, else every character position -- either way each is just a
    # place a hyphen is ALLOWED, not a forced cut. Reusing _HYPHEN_DICT keeps
    # this consistent with _split_word_to_fit's own preference for real
    # hyphenation points over blind character counting.
    if _HYPHEN_DICT is not None:
        breaks = sorted(set(_HYPHEN_DICT.positions(core)))
    else:
        breaks = list(range(1, len(core)))
    if not breaks:
        return [word]

    pieces: list[str] = []
    start = 0
    while start < len(core):
        tail_candidate = core[start:] + trailing
        if _text_width(tail_candidate, font, draw) <= target_width:
            pieces.append(tail_candidate)
            start = len(core)
            break
        best_end = None
        for brk in breaks:
            if brk <= start:
                continue
            candidate = core[start:brk] + "-"
            if _text_width(candidate, font, draw) <= target_width:
                best_end = brk
            else:
                break
        if best_end is None:
            return [word]
        pieces.append(core[start:best_end] + "-")
        start = best_end
        if len(pieces) >= max_pieces:
            return [word]

    if len(pieces) < 3:
        return [word]
    return pieces


def _wrap_standard(
    words: list[str],
    font: ImageFont.FreeTypeFont,
    draw: ImageDraw.ImageDraw,
    target_width: float,
    allow_word_split: bool = False,
    min_split_length: int = 8,
    allow_multipiece: bool = False,
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
            split_pieces = _split_word_to_fit(word, font, draw, target_width, min_split_length)
            if (
                allow_multipiece
                and len(split_pieces) == 1
                and _text_width(split_pieces[0], font, draw) > target_width
            ):
                split_pieces = _multipiece_split_word(split_pieces[0], font, draw, target_width)
            wrapped_words.extend(split_pieces)
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


_MEASURE_IMG = Image.new("L", (1, 1), 0)
_MEASURE_DRAW = ImageDraw.Draw(_MEASURE_IMG)


def _measure_text_block(
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    size: int,
    outline_width: int,
) -> dict:
    """Compute block layout/dimensions ONLY -- no glyph rasterization. Used to reject
    oversized candidates before paying for _render_text_block_from_measurement's FreeType
    render+stroke pass, which profiling showed is the dominant cost of step 8 (~70% of wall
    time). Every value here is identical to what _render_text_block used to compute inline
    before rendering, so this changes nothing about which candidate wins or what it looks like."""
    line_boxes = []
    line_widths = []
    line_heights = []
    spacing = _line_spacing_for_size(size)

    for line in lines:
        bbox = _MEASURE_DRAW.textbbox((0, 0), line or " ", font=font, stroke_width=outline_width)
        line_boxes.append(bbox)
        line_widths.append(max(1, bbox[2] - bbox[0]))
        line_heights.append(max(1, bbox[3] - bbox[1]))

    block_width = max(1, max(line_widths, default=1))
    block_height = max(1, sum(line_heights) + spacing * max(0, len(lines) - 1))
    return {
        "block_width": block_width,
        "block_height": block_height,
        "line_boxes": line_boxes,
        "line_widths": line_widths,
        "line_heights": line_heights,
        "spacing": spacing,
    }


def _render_text_block_from_measurement(
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    measurement: dict,
    outline_width: int,
    fill_color: tuple[int, int, int, int] = (0, 0, 0, 255),
    stroke_color: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> tuple[Image.Image, np.ndarray, dict]:
    """Rasterize using a measurement already computed by _measure_text_block -- the actual
    FreeType render+stroke pass, only reached for candidates that already passed the bounds
    check the caller runs against the cheap measurement."""
    block_width = measurement["block_width"]
    block_height = measurement["block_height"]
    line_boxes = measurement["line_boxes"]
    line_widths = measurement["line_widths"]
    line_heights = measurement["line_heights"]
    spacing = measurement["spacing"]

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


def _render_text_block(
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    size: int,
    outline_width: int,
    fill_color: tuple[int, int, int, int] = (0, 0, 0, 255),
    stroke_color: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> tuple[Image.Image, np.ndarray, dict]:
    """Measure + render in one call, unchanged behavior -- kept for the fallback call sites
    (:~2870-3010) that always need the render immediately and have no size-rejection step
    before it. The hot search loop below calls _measure_text_block /
    _render_text_block_from_measurement directly instead, to skip rendering oversized
    candidates entirely."""
    measurement = _measure_text_block(lines, font, size, outline_width)
    return _render_text_block_from_measurement(
        lines, font, measurement, outline_width, fill_color, stroke_color
    )


def _text_style_for_layout(background: Image.Image, allowed_mask: Image.Image) -> dict:
    mask_np = np.array(allowed_mask) > 0
    if not np.any(mask_np):
        return {
            "name": "dark_on_light",
            "fill_color": (0, 0, 0, 255),
            "stroke_color": (255, 255, 255, 255),
            "floating_stroke_color": (255, 255, 255, 255),
            "floating_outline_cap": 4,
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
            "floating_stroke_color": (0, 0, 0, 178),
            "floating_outline_cap": 2,
            "background_luma_median": median_luma,
            "background_luma_p75": p75_luma,
        }

    return {
        "name": "dark_on_light",
        "fill_color": (0, 0, 0, 255),
        "stroke_color": (255, 255, 255, 255),
        "floating_stroke_color": (255, 255, 255, 255),
        "floating_outline_cap": 4,
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


def _inferred_bubble_layout(layout: dict) -> bool:
    if layout.get("bubble_idx", -1) != -1:
        return False
    fallback_source = str(layout.get("fallback_source", "") or "").lower()
    outline = layout.get("inferred_bubble_outline")
    return bool(outline) or "missed_bubble" in fallback_source


def _typeset_as_floating(layout: dict) -> bool:
    return layout.get("bubble_idx", -1) == -1 and not _inferred_bubble_layout(layout)


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


def _device_surface_overlay_mask_for_layout(
    layout: dict,
    image_size: tuple[int, int],
    background: Image.Image | None = None,
) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    green_box = _coerce_box(layout["green_box"], image_size)
    red_box = _coerce_box(layout.get("red_box", layout["green_box"]), image_size)
    box = _coerce_box(
        [
            min(green_box[0], red_box[0]),
            min(green_box[1], red_box[1]),
            max(green_box[2], red_box[2]),
            max(green_box[3], red_box[3]),
        ],
        image_size,
    )
    if background is not None:
        gray = np.array(background.convert("L"))
        x1, y1, x2, y2 = box
        crop = gray[y1:y2, x1:x2]
        if crop.size:
            dark = (crop < 104).astype(np.uint8)
            row_fraction = np.mean(dark > 0, axis=1)
            row_mask = row_fraction >= 0.55
            row_ranges = []
            row_start = None
            for row_index, value in enumerate(row_mask):
                if value and row_start is None:
                    row_start = row_index
                if (not value or row_index == len(row_mask) - 1) and row_start is not None:
                    row_end = row_index if not value else row_index + 1
                    row_ranges.append((row_start, row_end, row_end - row_start))
                    row_start = None
            row_ranges = [
                item for item in row_ranges
                if item[2] >= max(18, int(crop.shape[0] * 0.18))
            ]
            if row_ranges:
                row_start, row_end, _ = max(row_ranges, key=lambda item: item[2])
                band = dark[row_start:row_end] > 0
                col_fraction = np.mean(band, axis=0)
                col_indices = np.flatnonzero(col_fraction >= 0.45)
                if col_indices.size:
                    ys = np.arange(row_start, row_end)
                    xs = col_indices
                    pad_x = max(2, int(round(crop.shape[1] * 0.03)))
                    pad_y = max(2, int(round(crop.shape[0] * 0.03)))
                    box = _coerce_box(
                        [
                            x1 + int(xs.min()) - pad_x,
                            y1 + int(ys.min()) - pad_y,
                            x1 + int(xs.max()) + 1 + pad_x,
                            y1 + int(ys.max()) + 1 + pad_y,
                        ],
                        image_size,
                    )
            else:
                dark = cv2.morphologyEx(
                    dark,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
                    iterations=1,
                )
                component_count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
                selected = np.zeros_like(dark, dtype=bool)
                area = max(1, crop.shape[0] * crop.shape[1])
                for label in range(1, component_count):
                    component_area = int(stats[label, cv2.CC_STAT_AREA])
                    cy = int(stats[label, cv2.CC_STAT_TOP])
                    cw = int(stats[label, cv2.CC_STAT_WIDTH])
                    ch = int(stats[label, cv2.CC_STAT_HEIGHT])
                    if (
                        component_area >= max(180, int(area * 0.16))
                        and cw >= max(28, int(crop.shape[1] * 0.42))
                        and ch >= max(28, int(crop.shape[0] * 0.30))
                        and cy <= int(crop.shape[0] * 0.50)
                    ):
                        selected |= labels == label
                if np.any(selected):
                    ys, xs = np.where(selected)
                    pad_x = max(2, int(round(crop.shape[1] * 0.03)))
                    pad_y = max(2, int(round(crop.shape[0] * 0.03)))
                    box = _coerce_box(
                        [
                            x1 + int(xs.min()) - pad_x,
                            y1 + int(ys.min()) - pad_y,
                            x1 + int(xs.max()) + 1 + pad_x,
                            y1 + int(ys.max()) + 1 + pad_y,
                        ],
                        image_size,
                    )
    draw.rectangle(box, fill=255)
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


def _source_region_is_busy(source_img: Image.Image | None, box: tuple[int, int, int, int]) -> bool:
    """True when the ORIGINAL art under a floating caption region is complex
    (colored/patterned/high-frequency) rather than flat.

    This is the discriminator for the Ichigo-style caption-panel strategy:
    flat backgrounds inpaint cleanly and need no panel, but busy backgrounds
    (floral patterns, detailed art, gradients) are exactly where erase+
    reconstruct produces ragged white/grey blobs -- there we instead lay down
    a clean semi-transparent panel and render text on it, like a real
    scanlation caption box, rather than chasing an impossible reconstruction.
    """
    if source_img is None:
        return False
    x1, y1, x2, y2 = box
    if x2 - x1 < 8 or y2 - y1 < 8:
        return False
    crop = np.array(source_img.convert("RGB"))[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    gray = crop[:, :, 0] * 0.299 + crop[:, :, 1] * 0.587 + crop[:, :, 2] * 0.114
    luma_std = float(np.std(gray))
    # Color spread: how far from grayscale the region is (colored art vs B&W).
    channel_spread = float(np.mean(crop.max(axis=2).astype(np.int16) - crop.min(axis=2).astype(np.int16)))
    edge_density = float(np.mean(cv2.Canny(gray.astype(np.uint8), 45, 135) > 0))
    # Busy = strong local contrast OR clearly colored OR many edges. Flat
    # paper/screentone stays below all three thresholds.
    return luma_std >= 34.0 or channel_spread >= 26.0 or edge_density >= 0.14


def _load_erase_footprint(sample_path: Path, image_size: tuple[int, int]) -> np.ndarray | None:
    """Step 4's erase mask: the exact pixels where source text was removed and
    reconstruction may have left ghosting. The caption haze only needs to
    normalize this footprint (plus the new text), so surrounding art keeps its
    detail instead of being covered by a whole-box panel."""
    mask_path = sample_path / "step_4_final" / "mask.png"
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if (mask.shape[1], mask.shape[0]) != image_size:
        mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
    return mask > 127


def _artist_backing_mask(
    source_img: Image.Image | None,
    strip_box: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Silhouette of the artist's own translucent light backing panel behind
    floating lettering (the soft white column pro releases preserve). Detected
    in the SOURCE as a bright low-saturation region filling most of the strip
    AND clearly brighter than the strip's immediate surroundings -- plain
    bright backgrounds (paper, sky) fail the lift test and get no backing.
    Returns a full-canvas uint8 mask (255 = backing), or None."""
    if source_img is None:
        return None
    x1, y1, x2, y2 = strip_box
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    pad = 20
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(source_img.width, x2 + pad), min(source_img.height, y2 + pad)
    window = np.array(source_img.convert("RGB"))[cy1:cy2, cx1:cx2]
    if window.size == 0:
        return None
    gray = cv2.cvtColor(window, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(window, cv2.COLOR_RGB2HSV)
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
    inside_luma = float(np.median(inside_gray[bright]))
    ring_luma = float(np.median(gray[ring].astype(np.float32)))
    if inside_luma - ring_luma < 16.0:
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
    full = np.zeros((source_img.height, source_img.width), dtype=np.uint8)
    full[y1:y2, x1:x2] = silhouette
    return full


def _reconstruction_left_blob(
    canvas: Image.Image,
    source_img: Image.Image | None,
    erase_footprint: np.ndarray | None,
    strip_box: tuple[int, int, int, int],
) -> bool:
    """True when Step 4's cleanup left a milky blob instead of plausible art:
    the erased footprint on the CLEANED canvas reads markedly brighter and
    flatter than the region's own surroundings in the SOURCE. This is the
    caption-haze trigger -- backing appears only over damaged reconstructions,
    never over art that was restored properly (which must stay untouched)."""
    if erase_footprint is None or source_img is None:
        return False
    x1, y1, x2, y2 = strip_box
    if x2 - x1 < 8 or y2 - y1 < 8:
        return False
    pad = 16
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(canvas.width, x2 + pad), min(canvas.height, y2 + pad)
    window = erase_footprint[cy1:cy2, cx1:cx2]
    if int(np.count_nonzero(window)) < 120:
        return False
    ring = cv2.dilate(
        window.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    ).astype(bool) & ~window
    if int(np.count_nonzero(ring)) < 120:
        return False
    canvas_gray = np.array(canvas.convert("L"), dtype=np.float32)[cy1:cy2, cx1:cx2]
    source_gray = np.array(source_img.convert("L"), dtype=np.float32)[cy1:cy2, cx1:cx2]
    inside_vals = canvas_gray[window]
    source_ring_vals = source_gray[ring]
    luma_lift = float(np.median(inside_vals)) - float(np.median(source_ring_vals))
    inside_std = float(np.std(inside_vals))
    ring_std = float(np.std(source_ring_vals))
    return luma_lift >= 24.0 and inside_std <= max(16.0, ring_std * 0.75)


def _source_caption_glyph_color(
    source_img: Image.Image | None,
    strip_box: tuple[int, int, int, int],
    erase_footprint: np.ndarray | None,
) -> tuple[int, int, int] | None:
    """Dominant saturated lettering color of the SOURCE caption strip, or None
    for plain black/grey glyphs. Pro scanlation keeps colored narration and
    laughter lettering in its original color (green captions stay green, pink
    laughter stays pink); only the language changes."""
    if source_img is None:
        return None
    x1, y1, x2, y2 = strip_box
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = np.array(source_img.convert("RGB"))[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    pixels = None
    if erase_footprint is not None:
        # The erase mask is dilated past the glyph strokes; erode it back so
        # we sample lettering pixels, not the surrounding art it swept up.
        window = erase_footprint[y1:y2, x1:x2].astype(np.uint8)
        eroded = cv2.erode(
            window, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        ).astype(bool)
        if np.count_nonzero(eroded) >= 60:
            pixels = crop[eroded]
        elif np.count_nonzero(window) >= 60:
            pixels = crop[window.astype(bool)]
    if pixels is None:
        pixels = crop.reshape(-1, 3)
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    saturated = (hsv[:, 1] >= 95) & (hsv[:, 2] >= 60) & (hsv[:, 2] <= 235)
    if np.count_nonzero(saturated) < max(40, int(0.08 * len(hsv))):
        return None
    hues = hsv[saturated, 0].astype(np.int32)
    hist = np.bincount(hues, minlength=180)
    mode_hue = int(np.argmax(hist))
    # Circular hue distance; keep pixels near the dominant hue so a mixed
    # region (colored glyphs over colored art) still yields the glyph color.
    dist = np.minimum(np.abs(hues - mode_hue), 180 - np.abs(hues - mode_hue))
    near = dist <= 14
    if np.count_nonzero(near) < 30:
        return None
    # Bias toward glyph-core pixels (65th percentile): the median over
    # anti-aliased edges drags value down and renders colored lettering
    # nearly black at page scale.
    sat = int(np.percentile(hsv[saturated][near, 1], 60))
    val = int(np.percentile(hsv[saturated][near, 2], 65))
    # Clamp for readability on the light haze backing.
    val = max(96, min(val, 180))
    sat = max(130, sat)
    rgb = cv2.cvtColor(
        np.array([[[mode_hue, sat, val]]], dtype=np.uint8), cv2.COLOR_HSV2RGB
    )[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _detect_caption_frame_inner_box(
    source_img: Image.Image | None,
    box: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    """Detect a thin dark rectangular frame drawn around a caption strip in
    the SOURCE art (a deliberate framed caption box). Returns the interior box
    just inside the frame, or None. Framed captions keep their frame and get a
    uniform translucent interior instead of footprint-shaped haze."""
    if source_img is None:
        return None
    x1, y1, x2, y2 = box
    if x2 - x1 < 40 or y2 - y1 < 40:
        return None
    gray = np.array(source_img.convert("L"))
    height, width = gray.shape
    band = 9
    margin = 6  # ignore strip corners where perpendicular frame lines cross

    def innermost_dark_line(fixed_axis: str, edge: int, direction: int) -> int | None:
        best = None
        for offset in range(-band, band + 1):
            idx = edge + direction * offset
            if fixed_axis == "row":
                if not (0 <= idx < height):
                    continue
                line = gray[idx, max(0, x1 + margin):min(width, x2 - margin)]
            else:
                if not (0 <= idx < width):
                    continue
                line = gray[max(0, y1 + margin):min(height, y2 - margin), idx]
            if line.size < 20:
                continue
            if float(np.mean(line < 100)) >= 0.72:
                best = idx
        return best

    top = innermost_dark_line("row", y1, 1)
    bottom = innermost_dark_line("row", y2, -1)
    left = innermost_dark_line("col", x1, 1)
    right = innermost_dark_line("col", x2, -1)
    if top is None or bottom is None or left is None or right is None:
        return None
    inner = (left + 3, top + 3, right - 2, bottom - 2)
    if inner[2] - inner[0] < 24 or inner[3] - inner[1] < 24:
        return None
    return inner


def _caption_haze_contribution(
    image_size: tuple[int, int],
    strip_box: tuple[int, int, int, int],
    erase_footprint: np.ndarray | None,
    text_layer: Image.Image,
    font_size: int,
    framed_inner_box: tuple[int, int, int, int] | None = None,
    backing_mask: np.ndarray | None = None,
) -> np.ndarray:
    """One floating caption's contribution to the page's caption-haze layer:
    the step-4 erase footprint inside its strip, unioned with a dilated halo
    of the rendered text, then Gaussian-feathered. No corners, no borders --
    the soft milky cloud a pro scanlation lays over art it could not cleanly
    reconstruct. Returns a float32 canvas-sized array in 0..255."""
    width, height = image_size
    local = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = strip_box
    if backing_mask is not None:
        # Restore the artist's own translucent backing panel: fill its
        # original silhouette softly (below the transfer curve's saturation
        # knee so art still breathes through, like the source design), with
        # a full-strength halo under the new text for readability.
        local[backing_mask > 0] = 132
        text_alpha = np.array(text_layer.getchannel("A"))
        np.maximum(local, np.where(text_alpha > 24, 255, 0).astype(np.uint8), out=local)
        grow = max(3, min(14, int(round(font_size * 0.25))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        dilated_text = cv2.dilate(np.where(local == 255, np.uint8(255), np.uint8(0)), kernel)
        np.maximum(local, dilated_text, out=local)
        return cv2.GaussianBlur(local.astype(np.float32), (0, 0), 5.0)
    if framed_inner_box is not None:
        # Deliberate framed caption box: keep the frame, normalize the whole
        # interior with a uniform translucent fill (like the original's own
        # backing) instead of footprint-shaped haze. The soft value here stays
        # below the transfer curve's saturation knee so art shows through.
        fx1, fy1, fx2, fy2 = framed_inner_box
        fx1, fy1 = max(0, fx1), max(0, fy1)
        fx2, fy2 = min(width, fx2), min(height, fy2)
        if fx2 > fx1 and fy2 > fy1:
            local[fy1:fy2, fx1:fx2] = 130
        text_alpha = np.array(text_layer.getchannel("A"))
        np.maximum(local, np.where(text_alpha > 24, 255, 0).astype(np.uint8), out=local)
        grow = max(3, min(14, int(round(font_size * 0.25))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        dilated_text = cv2.dilate(np.where(local == 255, np.uint8(255), np.uint8(0)), kernel)
        np.maximum(local, dilated_text, out=local)
        return cv2.GaussianBlur(local.astype(np.float32), (0, 0), 4.0)
    if x2 > x1 and y2 > y1:
        pad = max(4, int(round(font_size * 0.25)))
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(width, x2 + pad), min(height, y2 + pad)
        if erase_footprint is not None and np.any(erase_footprint[cy1:cy2, cx1:cx2]):
            window = local[cy1:cy2, cx1:cx2]
            window[erase_footprint[cy1:cy2, cx1:cx2]] = 255
        else:
            # No footprint recorded: fall back to a slightly inset strip so the
            # feather still stays within the original text's own region.
            inset = max(2, int(min(x2 - x1, y2 - y1) * 0.08))
            local[y1 + inset:max(y1 + inset + 1, y2 - inset), x1 + inset:max(x1 + inset + 1, x2 - inset)] = 255
    text_alpha = np.array(text_layer.getchannel("A"))
    np.maximum(local, np.where(text_alpha > 24, 255, 0).astype(np.uint8), out=local)
    grow = max(3, min(18, int(round(font_size * 0.30))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    local = cv2.dilate(local, kernel)
    sigma = max(5.0, min(18.0, font_size * 0.30))
    return cv2.GaussianBlur(local.astype(np.float32), (0, 0), sigma)


def _composite_caption_haze(canvas: Image.Image, haze_acc: np.ndarray) -> None:
    """Composite the accumulated caption haze under the (later-composited)
    caption text. The transfer curve saturates the core to ~94% opacity so
    ghosting and residual glyphs cannot read through, while the skirt fades
    smoothly to zero -- overlapping captions merge into one cloud."""
    if not np.any(haze_acc > 1.0):
        return
    alpha = np.clip(haze_acc * 1.7, 0.0, 255.0) * 0.94
    haze = np.empty((canvas.height, canvas.width, 4), dtype=np.uint8)
    haze[:, :, 0] = 252
    haze[:, :, 1] = 250
    haze[:, :, 2] = 247
    haze[:, :, 3] = alpha.astype(np.uint8)
    canvas.alpha_composite(Image.fromarray(haze, mode="RGBA"))


def _single_dark_bubble_mask_for_layout(
    layout: dict,
    image_size: tuple[int, int],
    background: Image.Image | None,
) -> Image.Image | None:
    if background is None or layout.get("bubble_idx", -1) != -1:
        return None
    if not _floating_dialogue_layout(layout):
        return None

    red_box = _coerce_box(layout.get("red_box", layout["green_box"]), image_size)
    green_box = _coerce_box(layout["green_box"], image_size)
    rx1, ry1, rx2, ry2 = red_box
    if rx2 <= rx1 or ry2 <= ry1:
        return None

    gray = np.array(background.convert("L"))
    patch = gray[ry1:ry2, rx1:rx2]
    if patch.size == 0:
        return None
    median_luma = float(np.median(patch))
    p75_luma = float(np.percentile(patch, 75))
    dark_fraction = float(np.mean(patch < 118))
    if not (median_luma < 104.0 and p75_luma < 172.0 and dark_fraction >= 0.38):
        return None

    search_box = [
        min(green_box[0], red_box[0]),
        min(green_box[1], red_box[1]),
        max(green_box[2], red_box[2]),
        max(green_box[3], red_box[3]),
    ]
    search_w = max(1, search_box[2] - search_box[0])
    search_h = max(1, search_box[3] - search_box[1])
    pad_x = max(34, min(150, int(search_w * 1.20)))
    pad_y = max(32, min(190, int(search_h * 0.34)))
    sx1, sy1, sx2, sy2 = _coerce_box(
        [
            search_box[0] - pad_x,
            search_box[1] - pad_y,
            search_box[2] + pad_x,
            search_box[3] + pad_y,
        ],
        image_size,
    )
    local_gray = gray[sy1:sy2, sx1:sx2]
    if local_gray.size == 0:
        return None

    local_dark = (local_gray < 112).astype(np.uint8)
    local_dark = cv2.morphologyEx(
        local_dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(local_dark, connectivity=8)
    if component_count <= 1:
        return None

    lrx1, lry1 = max(0, rx1 - sx1), max(0, ry1 - sy1)
    lrx2, lry2 = min(labels.shape[1], rx2 - sx1), min(labels.shape[0], ry2 - sy1)
    if lrx2 <= lrx1 or lry2 <= lry1:
        return None
    roi_labels = labels[lry1:lry2, lrx1:lrx2]
    label_ids, counts = np.unique(roi_labels[roi_labels > 0], return_counts=True)
    if len(label_ids) == 0:
        return None
    label = int(label_ids[int(np.argmax(counts))])
    overlap = int(np.max(counts))
    red_area = max(1, (rx2 - rx1) * (ry2 - ry1))
    if overlap < max(18, int(red_area * 0.22)):
        return None

    local_area = max(1, local_gray.shape[0] * local_gray.shape[1])
    area = int(stats[label, cv2.CC_STAT_AREA])
    cx = int(stats[label, cv2.CC_STAT_LEFT] + sx1)
    cy = int(stats[label, cv2.CC_STAT_TOP] + sy1)
    cw = int(stats[label, cv2.CC_STAT_WIDTH])
    ch = int(stats[label, cv2.CC_STAT_HEIGHT])
    if area < 500 or area > int(local_area * 0.84) or cw < 26 or ch < 48:
        return None

    gx1, gy1, gx2, gy2 = green_box
    if cx > min(gx1, rx1) + 12 or cy > min(gy1, ry1) + 18:
        return None
    if cx + cw < max(gx2, rx2) - 12 or cy + ch < max(gy2, ry2) - 18:
        return None

    component_values = local_gray[labels == label]
    if component_values.size < 80 or float(np.percentile(component_values, 85)) > 126.0:
        return None

    allowed = np.zeros_like(gray, dtype=np.uint8)
    allowed[sy1:sy2, sx1:sx2] = (labels == label).astype(np.uint8) * 255
    eroded = cv2.erode(
        allowed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    if np.count_nonzero(eroded > 0) >= max(160, int(area * 0.38)):
        allowed = eroded
    return Image.fromarray(allowed, mode="L")


def _floating_reverse_dark_groups(
    layout_data: list[dict],
    trans_map: dict,
    background: Image.Image,
    image_size: tuple[int, int],
) -> tuple[list[dict], dict, set]:
    gray = np.array(background.convert("L"))
    dark = (gray < 82).astype(np.uint8)
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if component_count <= 1:
        return [], {}, set()

    page_area = max(1, image_size[0] * image_size[1])
    grouped: dict[int, list[dict]] = {}
    for layout in layout_data:
        tid = layout.get("id")
        trans_item = trans_map.get(tid)
        if not trans_item or not _has_alphabetical(str(trans_item.get("en_text", ""))):
            continue
        if not _floating_dialogue_layout(layout):
            continue

        red_box = _coerce_box(layout.get("red_box", layout["green_box"]), image_size)
        x1, y1, x2, y2 = red_box
        if x2 <= x1 or y2 <= y1:
            continue
        roi_labels = labels[y1:y2, x1:x2]
        if roi_labels.size == 0:
            continue
        label_ids, counts = np.unique(roi_labels[roi_labels > 0], return_counts=True)
        if len(label_ids) == 0:
            continue
        label = int(label_ids[int(np.argmax(counts))])
        overlap = int(np.max(counts))
        if overlap < max(10, int((x2 - x1) * (y2 - y1) * 0.08)):
            continue

        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 850 or area > int(page_area * 0.14):
            continue
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if cw < 28 or ch < 52:
            continue
        component_values = gray[labels == label]
        if component_values.size < 80:
            continue
        if float(np.percentile(component_values, 85)) > 96.0:
            continue

        grouped.setdefault(label, []).append(layout)

    synthetic_layouts: list[dict] = []
    synthetic_masks: dict = {}
    skip_ids: set = set()
    for label, members in grouped.items():
        if len(members) < 2:
            continue

        member_boxes = [_coerce_box(member.get("red_box", member["green_box"]), image_size) for member in members]
        total_member_area = sum(max(1, (box[2] - box[0]) * (box[3] - box[1])) for box in member_boxes)
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        component_area = max(1, int(stats[label, cv2.CC_STAT_AREA]))
        if total_member_area / float(max(1, component_area)) > 0.72:
            continue

        vertical_members = sum(
            1 for box in member_boxes
            if (box[3] - box[1]) >= max(1, box[2] - box[0]) * 1.45
        )
        if vertical_members >= max(1, len(members) // 2):
            ordered = sorted(
                members,
                key=lambda item: (
                    -_box_center(_coerce_box(item.get("red_box", item["green_box"]), image_size))[0],
                    _box_center(_coerce_box(item.get("red_box", item["green_box"]), image_size))[1],
                ),
            )
        else:
            ordered = sorted(
                members,
                key=lambda item: (
                    _box_center(_coerce_box(item.get("red_box", item["green_box"]), image_size))[1],
                    _box_center(_coerce_box(item.get("red_box", item["green_box"]), image_size))[0],
                ),
            )

        text_parts = [
            str(trans_map[item["id"]].get("en_text", "")).strip()
            for item in ordered
            if str(trans_map[item["id"]].get("en_text", "")).strip()
        ]
        if len(text_parts) < 2:
            continue

        group_id = "reverse_dark_" + "_".join(str(item["id"]) for item in ordered)
        union = [
            min(box[0] for box in member_boxes),
            min(box[1] for box in member_boxes),
            max(box[2] for box in member_boxes),
            max(box[3] for box in member_boxes),
        ]
        union_area = max(1, (union[2] - union[0]) * (union[3] - union[1]))
        component_bbox_area = max(1, cw * ch)
        if (
            component_bbox_area / float(union_area) > 1.85
            or cw > max(1, union[2] - union[0]) * 1.75
            or ch > max(1, union[3] - union[1]) * 1.55
        ):
            continue
        allowed = (labels == label).astype(np.uint8) * 255
        eroded = cv2.erode(
            allowed,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        if np.count_nonzero(eroded > 0) >= max(120, int(component_area * 0.45)):
            allowed = eroded
        allowed_mask = Image.fromarray(allowed, mode="L")
        synthetic_layouts.append({
            "id": group_id,
            "text": " ".join(str(item.get("text", "")) for item in ordered).strip(),
            "red_box": union,
            "green_box": [cx, cy, cx + cw, cy + ch],
            "green_polygon": [[cx, cy], [cx + cw, cy], [cx + cw, cy + ch], [cx, cy + ch]],
            "bubble_idx": -1,
            "mask_mode": "reverse_dark_bubble",
            "route": "reverse_bubble_dialogue",
            "semantic_role": "dialogue",
            "reverse_dark_bubble": True,
            "member_ids": [item["id"] for item in ordered],
            "member_boxes": member_boxes,
        })
        synthetic_masks[group_id] = allowed_mask
        trans_map[group_id] = {
            "id": group_id,
            "en_text": " ".join(text_parts),
            "translation_source": "grouped_reverse_dark_bubble",
        }
        skip_ids.update(item["id"] for item in ordered)

    eligible: list[dict] = []
    for layout in layout_data:
        tid = layout.get("id")
        if tid in skip_ids:
            continue
        trans_item = trans_map.get(tid)
        if not trans_item or not _has_alphabetical(str(trans_item.get("en_text", ""))):
            continue
        if not _floating_dialogue_layout(layout):
            continue
        red_box = _coerce_box(layout.get("red_box", layout["green_box"]), image_size)
        x1, y1, x2, y2 = red_box
        if x2 <= x1 or y2 <= y1:
            continue
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        median_luma = float(np.median(patch))
        p75_luma = float(np.percentile(patch, 75))
        dark_fraction = float(np.mean(patch < 118))
        vertical_source = (y2 - y1) >= max(1, x2 - x1) * 1.35
        if vertical_source and median_luma < 96.0 and p75_luma < 170.0 and dark_fraction >= 0.46:
            eligible.append(layout)

    parent = {item["id"]: item["id"] for item in eligible}

    def find(item_id):
        while parent[item_id] != item_id:
            parent[item_id] = parent[parent[item_id]]
            item_id = parent[item_id]
        return item_id

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for index, left_layout in enumerate(eligible):
        left_box = _coerce_box(left_layout.get("red_box", left_layout["green_box"]), image_size)
        for right_layout in eligible[index + 1:]:
            right_box = _coerce_box(right_layout.get("red_box", right_layout["green_box"]), image_size)
            y_overlap = min(left_box[3], right_box[3]) - max(left_box[1], right_box[1])
            if y_overlap <= 0:
                continue
            min_height = max(1, min(left_box[3] - left_box[1], right_box[3] - right_box[1]))
            overlap_ratio = y_overlap / float(min_height)
            x_gap = max(0, max(left_box[0], right_box[0]) - min(left_box[2], right_box[2]))
            max_width = max(left_box[2] - left_box[0], right_box[2] - right_box[0])
            if overlap_ratio >= 0.38 and x_gap <= max(18, min(88, int(max_width * 1.35))):
                union(left_layout["id"], right_layout["id"])

    clusters: dict = {}
    for item in eligible:
        clusters.setdefault(find(item["id"]), []).append(item)

    for members in clusters.values():
        if len(members) < 2:
            continue
        member_boxes = [_coerce_box(member.get("red_box", member["green_box"]), image_size) for member in members]
        union_box = [
            min(box[0] for box in member_boxes),
            min(box[1] for box in member_boxes),
            max(box[2] for box in member_boxes),
            max(box[3] for box in member_boxes),
        ]
        union_w = max(1, union_box[2] - union_box[0])
        union_h = max(1, union_box[3] - union_box[1])
        pad_x = max(34, min(96, int(union_w * 0.85)))
        pad_y = max(22, min(72, int(union_h * 0.18)))
        rx1 = max(0, union_box[0] - pad_x)
        ry1 = max(0, union_box[1] - pad_y)
        rx2 = min(image_size[0], union_box[2] + pad_x)
        ry2 = min(image_size[1], union_box[3] + pad_y)
        local_gray = gray[ry1:ry2, rx1:rx2]
        if local_gray.size == 0:
            continue
        local_dark = (local_gray < 116).astype(np.uint8)
        local_dark = cv2.morphologyEx(
            local_dark,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        local_count, local_labels, local_stats, _ = cv2.connectedComponentsWithStats(local_dark, connectivity=8)
        keep_labels: set[int] = set()
        for box in member_boxes:
            lx1, ly1 = max(0, box[0] - rx1), max(0, box[1] - ry1)
            lx2, ly2 = min(local_labels.shape[1], box[2] - rx1), min(local_labels.shape[0], box[3] - ry1)
            if lx2 <= lx1 or ly2 <= ly1:
                continue
            label_ids, counts = np.unique(local_labels[ly1:ly2, lx1:lx2], return_counts=True)
            for label_id, count in zip(label_ids.tolist(), counts.tolist()):
                if label_id > 0 and count >= max(8, int((lx2 - lx1) * (ly2 - ly1) * 0.05)):
                    keep_labels.add(int(label_id))
        if not keep_labels:
            continue

        local_mask = np.isin(local_labels, list(keep_labels)).astype(np.uint8) * 255
        if np.count_nonzero(local_mask > 0) < max(120, int(union_w * union_h * 0.45)):
            continue
        ys, xs = np.where(local_mask > 0)
        if len(xs) == 0:
            continue
        bx1, by1 = int(xs.min() + rx1), int(ys.min() + ry1)
        bx2, by2 = int(xs.max() + 1 + rx1), int(ys.max() + 1 + ry1)
        component_bbox_area = max(1, (bx2 - bx1) * (by2 - by1))
        union_area = max(1, union_w * union_h)
        if (
            component_bbox_area / float(union_area) > 3.40
            or (bx2 - bx1) > union_w * 2.85
            or (by2 - by1) > union_h * 1.70
        ):
            continue
        full_mask = np.zeros_like(gray, dtype=np.uint8)
        full_mask[ry1:ry2, rx1:rx2] = local_mask
        page_margin = max(8, min(22, int(union_w * 0.18)))
        if bx1 <= 2:
            full_mask[:, :page_margin] = 0
        if bx2 >= image_size[0] - 2:
            full_mask[:, image_size[0] - page_margin:] = 0
        eroded = cv2.erode(
            full_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        if np.count_nonzero(eroded > 0) >= int(np.count_nonzero(full_mask > 0) * 0.48):
            full_mask = eroded

        ordered = sorted(
            members,
            key=lambda item: (
                -_box_center(_coerce_box(item.get("red_box", item["green_box"]), image_size))[0],
                _box_center(_coerce_box(item.get("red_box", item["green_box"]), image_size))[1],
            ),
        )
        text_parts = [
            str(trans_map[item["id"]].get("en_text", "")).strip()
            for item in ordered
            if str(trans_map[item["id"]].get("en_text", "")).strip()
        ]
        if len(text_parts) < 2:
            continue
        group_id = "reverse_dark_" + "_".join(str(item["id"]) for item in ordered)
        if group_id in synthetic_masks:
            continue
        synthetic_layouts.append({
            "id": group_id,
            "text": " ".join(str(item.get("text", "")) for item in ordered).strip(),
            "red_box": union_box,
            "green_box": [bx1, by1, bx2, by2],
            "green_polygon": [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]],
            "bubble_idx": -1,
            "mask_mode": "reverse_dark_bubble",
            "route": "reverse_bubble_dialogue",
            "semantic_role": "dialogue",
            "reverse_dark_bubble": True,
            "member_ids": [item["id"] for item in ordered],
            "member_boxes": member_boxes,
        })
        synthetic_masks[group_id] = Image.fromarray(full_mask, mode="L")
        trans_map[group_id] = {
            "id": group_id,
            "en_text": " ".join(text_parts),
            "translation_source": "grouped_reverse_dark_bubble",
        }
        skip_ids.update(item["id"] for item in ordered)

    return synthetic_layouts, synthetic_masks, skip_ids


def _cover_reverse_dark_source_text(
    canvas: Image.Image,
    member_boxes: list[list[int]],
) -> Image.Image:
    arr = np.array(canvas.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    changed = False
    img_h, img_w = gray.shape[:2]
    for box in member_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        if x2 <= x1 or y2 <= y1:
            continue
        pad = max(4, min(12, int(max(x2 - x1, y2 - y1) * 0.06)))
        rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
        rx2, ry2 = min(img_w, x2 + pad), min(img_h, y2 + pad)
        local_gray = gray[ry1:ry2, rx1:rx2]
        local_sat = saturation[ry1:ry2, rx1:rx2]
        local_arr = arr[ry1:ry2, rx1:rx2]
        dark_background = (local_gray < 112) & (local_sat < 190)
        if float(np.mean(dark_background)) < 0.34:
            continue
        source_pixels = (local_gray > 206) & (local_sat < 170)
        source_pixels = cv2.dilate(
            source_pixels.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        ) > 0
        source_pixels &= ~dark_background
        if int(np.count_nonzero(source_pixels)) < 6:
            continue
        background_pixels = dark_background & ~source_pixels
        if int(np.count_nonzero(background_pixels)) < 20:
            continue
        fill = np.median(local_arr[background_pixels], axis=0).astype(np.uint8)
        local_arr[source_pixels] = fill
        arr[ry1:ry2, rx1:rx2] = local_arr
        changed = True
    if not changed:
        return canvas
    return Image.fromarray(arr, mode="RGB").convert("RGBA")


def _composite_overlay_badge(
    base: Image.Image,
    position: tuple[int, int],
    block_size: tuple[int, int],
    font_size: int,
    text_style: dict,
) -> Image.Image:
    if os.getenv("FMT_TYPESSET_BADGE_BACKGROUND", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return base

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


def _font_path_for_layout(layout: dict, text: str = "", selected_font_path: str | None = None) -> str:
    """One font for the whole page.

    `selected_font_path` is resolved ONCE per page (see _resolve_font_family) and passed
    down explicitly rather than read from a module global or env var: step 8 runs
    IN-PROCESS and the GPU scheduler admits several requests concurrently, so any
    process-wide font state would race between two pages that chose different fonts.

    The previous per-region role branching (peppercarrot_svg / dense / floating /
    tall-narrow-floating) is intentionally gone -- it was the mechanism that put two
    font families on 47% of pages. Region role still drives SIZE (fit-to-box) and
    styling like outline width, which is correct comic typesetting; it no longer drives
    font FAMILY.
    """
    if selected_font_path:
        return selected_font_path
    return FONT_PATH


def _render_scale_for_image(image_size: tuple[int, int]) -> int:
    width, _ = image_size
    if width >= LOW_RES_TARGET_WIDTH:
        return 1
    return min(LOW_RES_MAX_RENDER_SCALE, max(1, math.ceil(LOW_RES_TARGET_WIDTH / width)))


def _scale_box(box: list[int] | tuple[int, int, int, int], scale: int) -> list[int]:
    return [int(round(value * scale)) for value in box]


def _scale_erase_box(box: dict | list[int] | tuple[int, int, int, int], scale: int) -> dict | list[int]:
    if isinstance(box, dict):
        scaled = dict(box)
        for key in ("x1", "y1", "x2", "y2", "width", "height"):
            if key in scaled:
                scaled[key] = int(round(float(scaled[key]) * scale))
        return scaled
    return _scale_box(box, scale)


def _scale_polygon_points(points: list | tuple, scale: int) -> list[list[int]]:
    return [
        [int(round(point[0] * scale)), int(round(point[1] * scale))]
        for point in points
        if isinstance(point, (list, tuple)) and len(point) >= 2
    ]


def _scale_layout(layout: dict, scale: int) -> dict:
    if scale == 1:
        return dict(layout)

    scaled = dict(layout)
    scaled["red_box"] = _scale_box(layout["red_box"], scale)
    scaled["green_box"] = _scale_box(layout["green_box"], scale)
    polygon = layout.get("green_polygon") or []
    scaled["green_polygon"] = _scale_polygon_points(polygon, scale)
    scaled["green_polygons"] = [
        _scale_polygon_points(poly, scale)
        for poly in (layout.get("green_polygons") or [])
        if poly
    ]
    if layout.get("inferred_bubble_outline"):
        scaled["inferred_bubble_outline"] = _scale_polygon_points(
            layout.get("inferred_bubble_outline") or [],
            scale,
        )
    if layout.get("erase_boxes"):
        scaled["erase_boxes"] = [_scale_erase_box(box, scale) for box in layout.get("erase_boxes") or []]
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


def _layout_source_boxes(layout: dict, image_size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    boxes = []
    for raw_box in [layout.get("red_box"), *(layout.get("erase_boxes") or [])]:
        if isinstance(raw_box, dict):
            raw_box = [
                raw_box.get("x1", 0),
                raw_box.get("y1", 0),
                raw_box.get("x2", 0),
                raw_box.get("y2", 0),
            ]
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) < 4:
            continue
        box = _coerce_box(raw_box[:4], image_size)
        if box[2] > box[0] and box[3] > box[1] and box not in boxes:
            boxes.append(box)
    return boxes


def _source_text_mask_for_layout(
    layout: dict,
    image_size: tuple[int, int],
    detect_dir: Path,
) -> np.ndarray:
    source_mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    source_boxes = _layout_source_boxes(layout, image_size)
    seg_mask_path = detect_dir / "seg_mask.png"
    seg_mask = None
    if seg_mask_path.exists():
        seg_mask = Image.open(str(seg_mask_path)).convert("L")
        if seg_mask.size != image_size:
            seg_mask = seg_mask.resize(image_size, Image.Resampling.NEAREST)
        seg_mask = np.array(seg_mask)

    for x1, y1, x2, y2 in source_boxes:
        if seg_mask is not None:
            roi = seg_mask[y1:y2, x1:x2]
            if np.count_nonzero(roi > 0) >= 6:
                source_mask[y1:y2, x1:x2] = np.maximum(source_mask[y1:y2, x1:x2], roi)

    if np.count_nonzero(source_mask > 0) >= 6:
        source_mask = cv2.dilate(
            (source_mask > 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )
        return source_mask

    for x1, y1, x2, y2 in source_boxes:
        pad_x = max(2, min(12, int((x2 - x1) * 0.06)))
        pad_y = max(2, min(12, int((y2 - y1) * 0.04)))
        fx1 = max(0, x1 - pad_x)
        fy1 = max(0, y1 - pad_y)
        fx2 = min(image_size[0], x2 + pad_x)
        fy2 = min(image_size[1], y2 + pad_y)
        source_mask[fy1:fy2, fx1:fx2] = 255
    return source_mask


def _structural_art_mask(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, 45, 135) > 0
    dark_ink = gray < 124
    seed = (edges | dark_ink).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    # Vectorized equivalent of the original per-component loop (labels == component_idx over the
    # full page, once per component -- profiled at 96% of this function's time on a page with
    # thousands of components). Same predicate, applied to the whole stats array at once, then a
    # single label lookup (keep[labels]) instead of one full-page comparison per component.
    widths = stats[:, cv2.CC_STAT_WIDTH]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    areas = stats[:, cv2.CC_STAT_AREA]
    max_dim = np.maximum(widths, heights)
    min_dim = np.minimum(widths, heights)
    keep = (
        (areas >= 24)
        | (max_dim >= 18)
        | ((areas >= 10) & (min_dim <= 3) & (max_dim >= 12))
    )
    # Component 0 is the background label (cv2 convention) -- the original loop started at
    # range(1, component_count), never evaluating it. keep[0] must be forced False or the
    # background (which trivially has a huge area/max_dim) would flip the entire mask to True.
    keep[0] = False
    return keep[labels]


def _source_footprint_mask_for_layout(
    layout: dict,
    image_size: tuple[int, int],
    detect_dir: Path,
    source_mask: np.ndarray | None = None,
    full_source_boxes: bool = False,
) -> np.ndarray:
    footprint = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    source_boxes = _layout_source_boxes(layout, image_size)
    if full_source_boxes:
        for x1, y1, x2, y2 in source_boxes:
            pad_x = max(2, min(10, int(round((x2 - x1) * 0.035))))
            pad_y = max(2, min(10, int(round((y2 - y1) * 0.035))))
            fx1 = max(0, x1 - pad_x)
            fy1 = max(0, y1 - pad_y)
            fx2 = min(image_size[0], x2 + pad_x)
            fy2 = min(image_size[1], y2 + pad_y)
            footprint[fy1:fy2, fx1:fx2] = 255
        return footprint

    if source_mask is None:
        source_mask = _source_text_mask_for_layout(layout, image_size, detect_dir)
    source_bool = source_mask > 0
    if int(np.count_nonzero(source_bool)) < 6:
        return _source_footprint_mask_for_layout(
            layout,
            image_size,
            detect_dir,
            source_mask=source_mask,
            full_source_boxes=True,
        )

    green_box = _coerce_box(layout["green_box"], image_size)
    green_width = green_box[2] - green_box[0]
    green_height = green_box[3] - green_box[1]
    kernel_size = int(max(19, min(45, round(min(image_size) * 0.020))))
    if green_width >= green_height * 1.55:
        kernel_width = min(61, max(kernel_size, int(round(green_width * 0.18))))
        kernel_height = kernel_size
    elif green_height >= green_width * 1.55:
        kernel_width = kernel_size
        kernel_height = min(61, max(kernel_size, int(round(green_height * 0.12))))
    else:
        kernel_width = kernel_size
        kernel_height = kernel_size
    if kernel_width % 2 == 0:
        kernel_width += 1
    if kernel_height % 2 == 0:
        kernel_height += 1
    footprint = cv2.dilate(
        source_bool.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_width, kernel_height)),
        iterations=1,
    )
    return footprint


def _floating_art_context(
    allowed_np: np.ndarray,
    layout: dict,
    image_size: tuple[int, int],
    detect_dir: Path,
    analysis_image: Image.Image | None,
    source_mask: np.ndarray,
) -> tuple[np.ndarray | None, float, float]:
    if analysis_image is None:
        return None, 1.0, 1.0
    gray = np.array(analysis_image.convert("L"))
    if gray.shape != allowed_np.shape:
        return None, 1.0, 1.0
    structural = _structural_art_mask(gray)
    source_bool = source_mask > 0
    source_probe = _source_footprint_mask_for_layout(
        layout,
        image_size,
        detect_dir,
        source_mask=source_mask,
        full_source_boxes=False,
    ) > 0
    art_without_source = structural & ~source_bool
    allowed_area = max(1, int(np.count_nonzero(allowed_np)))
    probe_area = max(1, int(np.count_nonzero(source_probe)))
    allowed_ratio = float(np.count_nonzero(art_without_source & allowed_np)) / allowed_area
    source_ratio = float(np.count_nonzero(art_without_source & source_probe)) / probe_area
    return structural, allowed_ratio, source_ratio


def _exclude_panel_divider_lines(allowed: Image.Image, gray: np.ndarray) -> Image.Image:
    """Never let floating text cross a panel-divider/gutter line.

    A ruled panel border is dangerous regardless of how small its pixel area
    is relative to the whole allowed region -- a single row/column spanning
    the full local width still cuts straight through any text placed across
    it. This is independent of `_apply_floating_art_avoidance`'s overall
    structural-art ratio gate, which a thin full-span line can slip under
    (external_ja_1: 5.4% ratio, well under the 9% gate, yet the line still
    struck through "IT'S LIKE A")."""
    allowed_np = np.array(allowed) > 0
    ys, xs = np.where(allowed_np)
    if ys.size < 40:
        return allowed
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    if gray.shape[0] < y2 or gray.shape[1] < x2:
        return allowed
    window_gray = gray[y1:y2, x1:x2]
    window_allowed = allowed_np[y1:y2, x1:x2]
    dark = window_gray < 130
    h, w = window_gray.shape[:2]
    exclude = np.zeros((h, w), dtype=bool)
    if w >= 12:
        row_frac = np.mean(dark & window_allowed, axis=1) / np.maximum(
            1e-6, np.mean(window_allowed, axis=1)
        )
        for row in np.where(row_frac >= 0.85)[0]:
            exclude[max(0, row - 3) : row + 4, :] = True
    if h >= 12:
        col_frac = np.mean(dark & window_allowed, axis=0) / np.maximum(
            1e-6, np.mean(window_allowed, axis=0)
        )
        for col in np.where(col_frac >= 0.85)[0]:
            exclude[:, max(0, col - 3) : col + 4] = True
    if not exclude.any():
        return allowed
    refined = window_allowed & ~exclude
    if int(np.count_nonzero(refined)) < max(60, int(np.count_nonzero(window_allowed) * 0.15)):
        return allowed
    full = allowed_np.copy()
    full[y1:y2, x1:x2] = refined
    return Image.fromarray((full.astype(np.uint8) * 255), mode="L")


def _apply_floating_art_avoidance(
    allowed: Image.Image,
    layout: dict,
    detect_dir: Path,
    protection_image: Image.Image | None,
    source_image: Image.Image | None = None,
) -> Image.Image:
    if protection_image is None:
        return allowed
    allowed_np = np.array(allowed) > 0
    allowed_count = int(np.count_nonzero(allowed_np))
    if allowed_count < 40:
        return allowed

    image_size = allowed.size
    source_mask_raw = _source_text_mask_for_layout(layout, image_size, detect_dir)
    analysis_image = source_image or protection_image
    structural, art_allowed_ratio, art_source_ratio = _floating_art_context(
        allowed_np,
        layout,
        image_size,
        detect_dir,
        analysis_image,
        source_mask_raw,
    )
    if structural is None:
        return allowed
    if art_allowed_ratio < 0.09:
        return allowed

    source_footprint = _source_footprint_mask_for_layout(
        layout,
        image_size,
        detect_dir,
        source_mask=source_mask_raw,
        full_source_boxes=art_source_ratio >= 0.14,
    ) > 0
    protected_seed = structural & allowed_np & ~source_footprint
    if int(np.count_nonzero(protected_seed)) < 6:
        return allowed

    page_width, page_height = image_size
    kernel_size = int(max(9, min(27, round(min(page_width, page_height) * 0.018))))
    if kernel_size % 2 == 0:
        kernel_size += 1
    protected = cv2.dilate(
        protected_seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        iterations=1,
    ) > 0

    refined = (allowed_np & ~protected) | (allowed_np & source_footprint)
    refined_count = int(np.count_nonzero(refined))
    source_count = int(np.count_nonzero(allowed_np & source_footprint))

    min_required = max(90, int(allowed_count * 0.10), min(650, source_count))
    if refined_count < min_required:
        return allowed
    return Image.fromarray((refined.astype(np.uint8) * 255), mode="L")


def _allowed_mask_for_layout(
    layout: dict,
    image_size: tuple[int, int],
    detect_dir: Path,
    protection_image: Image.Image | None = None,
    source_image: Image.Image | None = None,
) -> Image.Image:
    allowed = Image.new("L", image_size, 0)
    allowed_draw = ImageDraw.Draw(allowed)
    polygons = layout.get("green_polygons") or []
    polygon = layout.get("green_polygon") or []
    bubble_idx = layout.get("bubble_idx", -1)
    inferred_bubble = _inferred_bubble_layout(layout)

    if bubble_idx == -1 and inferred_bubble and layout.get("inferred_bubble_outline"):
        points = [
            (int(round(point[0])), int(round(point[1])))
            for point in layout.get("inferred_bubble_outline") or []
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if len(points) >= 3:
            allowed_draw.polygon(points, fill=255)
        else:
            allowed_draw.rectangle(_coerce_box(layout["green_box"], image_size), fill=255)
    elif bubble_idx == -1 and layout.get("art_aware_routed") and polygons:
        for routed_polygon in polygons:
            if routed_polygon and len(routed_polygon) >= 3:
                points = [(int(round(point[0])), int(round(point[1]))) for point in routed_polygon]
                allowed_draw.polygon(points, fill=255)
    elif bubble_idx == -1 and layout.get("art_aware_routed") and len(polygon) >= 3:
        points = [(int(round(point[0])), int(round(point[1]))) for point in polygon]
        allowed_draw.polygon(points, fill=255)
    elif bubble_idx == -1:
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
            narrow_vertical = box_width <= page_width * 0.055 and box_height >= box_width * 2.6
            if narrow_vertical:
                expand_x = min(
                    max_expand_x,
                    max(12, int(round(box_width * 0.85)), int(round(box_height * 0.22))),
                )
            else:
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

    dark_bubble_mask = _single_dark_bubble_mask_for_layout(layout, image_size, protection_image)
    if dark_bubble_mask is not None and _bbox_from_mask(dark_bubble_mask) is not None:
        allowed = dark_bubble_mask

    if bubble_idx == -1 and not inferred_bubble and dark_bubble_mask is None:
        art_safe_allowed = _apply_floating_art_avoidance(
            allowed,
            layout,
            detect_dir,
            protection_image,
            source_image=source_image,
        )
        if _bbox_from_mask(art_safe_allowed) is not None:
            allowed = art_safe_allowed
        analysis_for_lines = source_image or protection_image
        if analysis_for_lines is not None:
            gray_for_lines = np.array(analysis_for_lines.convert("L"))
            if gray_for_lines.shape == (image_size[1], image_size[0]):
                line_safe_allowed = _exclude_panel_divider_lines(allowed, gray_for_lines)
                if _bbox_from_mask(line_safe_allowed) is not None:
                    allowed = line_safe_allowed

    if bubble_idx == -1 and not inferred_bubble and layout.get("erase_boxes") and protection_image is not None:
        allowed_np = np.array(allowed) > 0
        source_mask_raw = _source_text_mask_for_layout(layout, image_size, detect_dir)
        structural, art_allowed_ratio, art_source_ratio = _floating_art_context(
            allowed_np,
            layout,
            image_size,
            detect_dir,
            source_image or protection_image,
            source_mask_raw,
        )
        if art_allowed_ratio < 0.09:
            return allowed
        gray = np.array(protection_image.convert("L"))
        if gray.shape == allowed_np.shape:
            if structural is None:
                structural = _structural_art_mask(gray)
            dark_ink = structural & allowed_np
            source_footprint = _source_footprint_mask_for_layout(
                layout,
                image_size,
                detect_dir,
                source_mask=source_mask_raw,
                full_source_boxes=art_source_ratio >= 0.14,
            ) > 0
            erase_np = np.zeros_like(allowed_np, dtype=bool)
            red_box = layout.get("red_box")
            if isinstance(red_box, (list, tuple)) and len(red_box) >= 4:
                rx1, ry1, rx2, ry2 = _coerce_box(red_box[:4], image_size)
                rx1 = max(0, rx1 - 4)
                ry1 = max(0, ry1 - 4)
                rx2 = min(image_size[0], rx2 + 4)
                ry2 = min(image_size[1], ry2 + 4)
                if rx2 > rx1 and ry2 > ry1:
                    erase_np[ry1:ry2, rx1:rx2] = True
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
            # The dark source strokes inside erase_boxes are expected to be
            # cleaned by Step 4, so they must remain valid typesetting space.
            # Only protect unrelated dark artwork outside those text boxes.
            dark_ink &= ~(erase_np | source_footprint)
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


def _mask_bbox_from_np(mask_np: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_np)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _split_two_lobe_masks(allowed_mask: Image.Image) -> list[Image.Image] | None:
    mask_np = np.array(allowed_mask) > 0
    bounds = _mask_bbox_from_np(mask_np)
    if bounds is None:
        return None
    x1, y1, x2, y2 = bounds
    width = x2 - x1
    height = y2 - y1
    area = int(np.count_nonzero(mask_np))
    if width < 120 or height < 80 or area < 1800:
        return None

    row_centers = []
    row_counts = []
    for y in range(y1, y2):
        xs = np.where(mask_np[y, x1:x2])[0]
        if len(xs) < max(8, int(width * 0.08)):
            row_centers.append(np.nan)
            row_counts.append(0)
        else:
            row_centers.append(float(xs.mean() + x1))
            row_counts.append(int(len(xs)))

    valid_centers = np.array([c for c in row_centers if not np.isnan(c)], dtype=np.float32)
    if valid_centers.size < 20:
        return None
    top_sample = valid_centers[: max(5, valid_centers.size // 3)]
    bottom_sample = valid_centers[-max(5, valid_centers.size // 3):]
    center_delta = abs(float(np.median(top_sample)) - float(np.median(bottom_sample)))
    if center_delta < max(44, width * 0.22):
        return None

    # Score every candidate split row using bounded SLICES (views, no copy) instead of a full
    # image-sized array copy per candidate -- top_center/bottom_center only depend on the X-bounds
    # of the nonzero pixels in each half, and slicing rows leaves X-indices untouched, so this is
    # an exact equivalent of the original zeroed-copy approach for every value the score actually
    # uses (_mask_bbox_from_np's Y-bounds were computed but never read downstream either way).
    # Only the WINNING split_y needs a full-size mask constructed, once, after the search.
    best = None
    for split_y in range(y1 + int(height * 0.30), y1 + int(height * 0.72)):
        top_slice = mask_np[:split_y, :]
        bottom_slice = mask_np[split_y:, :]
        top_area = int(np.count_nonzero(top_slice))
        bottom_area = int(np.count_nonzero(bottom_slice))
        if top_area < max(450, int(area * 0.18)) or bottom_area < max(450, int(area * 0.18)):
            continue
        top_bounds = _mask_bbox_from_np(top_slice)
        bottom_bounds = _mask_bbox_from_np(bottom_slice)
        if top_bounds is None or bottom_bounds is None:
            continue
        tx1, _, tx2, _ = top_bounds
        bx1, _, bx2, _ = bottom_bounds
        top_center = (tx1 + tx2) / 2.0
        bottom_center = (bx1 + bx2) / 2.0
        split_band_count = row_counts[split_y - y1] if 0 <= split_y - y1 < len(row_counts) else width
        score = (
            abs(top_center - bottom_center) * 2.0
            + min(top_area, bottom_area) / max(1, area) * 180.0
            - split_band_count * 0.12
            - abs((top_area / max(1, bottom_area)) - 1.0) * 18.0
        )
        if best is None or score > best[0]:
            best = (score, split_y)

    if best is None:
        return None
    _, best_split_y = best
    top_np = mask_np.copy()
    bottom_np = mask_np.copy()
    top_np[best_split_y:, :] = False
    bottom_np[:best_split_y, :] = False
    masks = []
    for part_np in (top_np, bottom_np):
        if np.count_nonzero(part_np) < 450:
            return None
        masks.append(Image.fromarray((part_np.astype(np.uint8) * 255), mode="L"))
    return masks


def _clipped_pixels_for_fit(fit: dict, allowed_np: np.ndarray) -> int:
    if "parts" in fit:
        return sum(
            _count_clipped_pixels(part["alpha"], allowed_np, part["position"])
            for part in fit["parts"]
        )
    return _count_clipped_pixels(fit["alpha"], allowed_np, fit["position"])


def _find_two_region_layout(
    text: str,
    layout: dict,
    allowed_mask: Image.Image,
    text_style: dict,
    selected_font_path: str | None = None,
) -> dict | None:
    if layout.get("bubble_idx", -1) == -1:
        return None
    masks = _split_two_lobe_masks(allowed_mask)
    if not masks:
        return None
    words = text.split()
    if len(words) < 8:
        return None

    best = None
    best_score = -1e18
    low = max(3, int(len(words) * 0.35))
    high = min(len(words) - 3, int(len(words) * 0.68))
    if high < low:
        return None

    for split_index in range(low, high + 1):
        texts = [" ".join(words[:split_index]), " ".join(words[split_index:])]
        parts = []
        failed = False
        for part_text, part_mask in zip(texts, masks):
            part_bounds = _bbox_from_mask(part_mask)
            if part_bounds is None:
                failed = True
                break
            part_layout = dict(layout)
            part_layout["green_box"] = list(part_bounds)
            part_layout["red_box"] = list(part_bounds)
            fit = _find_mask_aware_layout(
                part_text, part_layout, part_mask, text_style,
                selected_font_path=selected_font_path,
            )
            if fit["status"] == "fallback_clipped":
                failed = True
                break
            fit = dict(fit)
            fit["text"] = part_text
            fit["allowed_mask"] = part_mask
            parts.append(fit)
        if failed or len(parts) != 2:
            continue
        font_floor = min(part["font_size"] for part in parts)
        clipped = sum(
            _count_clipped_pixels(part["alpha"], np.array(part["allowed_mask"]) > 0, part["position"])
            for part in parts
        )
        if clipped:
            continue
        area_usage = sum(
            (part["block"].width * part["block"].height)
            / max(1, np.count_nonzero(np.array(part["allowed_mask"]) > 0))
            for part in parts
        )
        balance_penalty = abs(parts[0]["font_size"] - parts[1]["font_size"]) * 10.0
        score = font_floor * 100.0 + area_usage * 80.0 - balance_penalty
        if score > best_score:
            all_lines = []
            for part in parts:
                all_lines.extend(part["lines"])
            best = {
                "parts": parts,
                "font_size": font_floor,
                "lines": all_lines,
                "status": "fit_multi_region",
                "clipped_pixels": 0,
                "metrics": {
                    "outline_width": max(part["metrics"]["outline_width"] for part in parts),
                },
            }
            best_score = score
    return best


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


def _independent_dialogue_max_size(layout: dict, image_size: tuple[int, int]) -> int:
    """Cheap proxy for the max_size _find_mask_aware_layout would compute on
    its own for this layout, using green_box height instead of the (more
    expensive) allowed-mask bounds. Used only for grouping/capping decisions,
    not for the actual fit -- close enough since mask_bounds falls back to
    green_box whenever the mask is empty, and is rarely much smaller."""
    green_box = _coerce_box(layout["green_box"], image_size)
    bounds_height = max(1, green_box[3] - green_box[1])
    is_floating = _typeset_as_floating(layout)
    inferred_bubble = _inferred_bubble_layout(layout)
    max_size = min(MAX_FONT_DIALOGUE, max(MIN_FONT_SIZE, int(bounds_height * 0.9)))
    if is_floating:
        max_size = min(max_size, max(36, min(80, int(bounds_height * 0.55))))
    elif inferred_bubble:
        max_size = min(max_size, max(36, min(72, int(bounds_height * 0.52))))
    return max_size


def _compute_group_font_caps(layout_data: list[dict], image_size: tuple[int, int]) -> dict:
    """Group dialogue-role layouts by spatial proximity (same panel/nearby
    balloons) and cap each group's font size to its smallest member's own
    independent max_size. Without this, two same-register dialogue regions
    with slightly different box heights get sized completely independently,
    producing visibly inconsistent lettering across a page even when the
    text length/register is similar -- each region is otherwise fit in
    total isolation from its neighbors."""
    eligible = [
        layout for layout in layout_data
        if _floating_dialogue_layout(layout) or layout.get("bubble_idx", -1) != -1 or _inferred_bubble_layout(layout)
    ]
    if len(eligible) < 2:
        return {}

    boxes = []
    for layout in eligible:
        x1, y1, x2, y2 = _coerce_box(layout["green_box"], image_size)
        boxes.append((layout["id"], x1, y1, x2, y2))

    # Union-find clustering: two boxes belong to the same group when their
    # padded bounds overlap. Padding is proportional to box size so it
    # scales with page resolution instead of a fixed pixel constant.
    parent = {box_id: box_id for box_id, *_ in boxes}

    def find(box_id):
        while parent[box_id] != box_id:
            parent[box_id] = parent[parent[box_id]]
            box_id = parent[box_id]
        return box_id

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(boxes)):
        id_a, ax1, ay1, ax2, ay2 = boxes[i]
        pad_a = max(20, int((ax2 - ax1 + ay2 - ay1) * 0.18))
        for j in range(i + 1, len(boxes)):
            id_b, bx1, by1, bx2, by2 = boxes[j]
            pad_b = max(20, int((bx2 - bx1 + by2 - by1) * 0.18))
            pad = max(pad_a, pad_b)
            overlap_x = max(ax1 - pad, bx1) < min(ax2 + pad, bx2)
            overlap_y = max(ay1 - pad, by1) < min(ay2 + pad, by2)
            if overlap_x and overlap_y:
                union(id_a, id_b)

    groups: dict = {}
    for box_id, *_ in boxes:
        groups.setdefault(find(box_id), []).append(box_id)

    caps: dict = {}
    layout_by_id = {layout["id"]: layout for layout in eligible}
    for members in groups.values():
        if len(members) < 2:
            continue
        member_max_sizes = [
            _independent_dialogue_max_size(layout_by_id[mid], image_size)
            for mid in members
        ]
        shared_cap = min(member_max_sizes)
        for mid in members:
            caps[mid] = shared_cap
    return caps


def _drop_duplicate_overlapping_layouts(
    layout_data: list[dict],
    trans_map: dict,
    image_size: tuple[int, int],
) -> list[dict]:
    """Duplicate detections of the SAME text region (>=70% overlap of the
    smaller box) each carry their own OCR/translation variant and would
    typeset on top of each other. Keep the layout with the longer translation
    (more content survives), drop the rest. Applies to bubbles and floating
    text alike; heavy-but-partial overlaps (<70%) are left for the caption
    merge pass, which joins continuation fragments instead."""
    dropped: set = set()
    items = [
        layout for layout in layout_data
        if str(trans_map.get(layout.get("id"), {}).get("en_text", "") or "").strip()
    ]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if a["id"] in dropped or b["id"] in dropped:
                continue
            # Step 6 already proved these two are SEPARATE utterances that
            # happen to share one traced touching-bubble container (their
            # bounding green_box rectangles can still overlap heavily even
            # after the per-cluster split, when one constraint's own outline
            # never crossed into its neighbor's space to begin with --
            # verified: new_sample_13's id6/id7). This box-only overlap
            # heuristic has no way to tell that apart from a genuine
            # same-region duplicate detection, so trust Step 6's explicit
            # sibling tag over the geometry here.
            # Step 6's writer tags both sides of a sibling pair symmetrically, so this
            # check is defensive, not a known asymmetry: never let a drop decision
            # depend on which of two writers happened to run first.
            if (b["id"] in (a.get("touching_container_siblings") or [])
                    or a["id"] in (b.get("touching_container_siblings") or [])):
                continue
            box_a = _coerce_box(a.get("green_box", a.get("red_box")), image_size)
            box_b = _coerce_box(b.get("green_box", b.get("red_box")), image_size)
            inter_w = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
            inter_h = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
            if inter_w <= 0 or inter_h <= 0:
                continue
            area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
            area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
            if (inter_w * inter_h) / float(min(area_a, area_b)) < 0.70:
                continue
            text_a = str(trans_map[a["id"]].get("en_text", "") or "")
            text_b = str(trans_map[b["id"]].get("en_text", "") or "")
            dropped.add(a["id"] if len(text_a) < len(text_b) else b["id"])
    if not dropped:
        return layout_data
    return [layout for layout in layout_data if layout.get("id") not in dropped]


def _merge_overlapping_floating_captions(
    layout_data: list[dict],
    trans_map: dict,
    image_size: tuple[int, int],
    excluded_ids: set | None = None,
) -> tuple[list[dict], set]:
    """Detection sometimes splits one caption column into overlapping layouts
    (a full-sentence box plus a continuation fragment). Typeset independently
    their fitted blocks collide in the shared area. Merge such layouts into a
    single synthetic caption: union box, text joined in reading order
    (right-to-left for vertical CJK columns, top-to-bottom otherwise)."""
    eligible = []
    for layout in layout_data:
        tid = layout.get("id")
        if excluded_ids and tid in excluded_ids:
            continue
        trans_item = trans_map.get(tid)
        if not trans_item or not _has_alphabetical(str(trans_item.get("en_text", ""))):
            continue
        if not _floating_dialogue_layout(layout) or _inferred_bubble_layout(layout):
            continue
        if layout.get("member_ids") or layout.get("reverse_dark_bubble"):
            continue
        eligible.append(layout)
    if len(eligible) < 2:
        return [], set()

    boxes = {
        layout["id"]: _coerce_box(layout["green_box"], image_size)
        for layout in eligible
    }
    parent = {layout["id"]: layout["id"] for layout in eligible}

    def find(item_id):
        while parent[item_id] != item_id:
            parent[item_id] = parent[parent[item_id]]
            item_id = parent[item_id]
        return item_id

    def union_ids(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    ids = [layout["id"] for layout in eligible]
    for i in range(len(ids)):
        ax1, ay1, ax2, ay2 = boxes[ids[i]]
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        for j in range(i + 1, len(ids)):
            bx1, by1, bx2, by2 = boxes[ids[j]]
            area_b = max(1, (bx2 - bx1) * (by2 - by1))
            inter_w = min(ax2, bx2) - max(ax1, bx1)
            inter_h = min(ay2, by2) - max(ay1, by1)
            if inter_w <= 0 or inter_h <= 0:
                continue
            # Same-column duplicates overlap heavily relative to the smaller
            # box; captions that merely touch stay independent.
            if (inter_w * inter_h) / float(min(area_a, area_b)) >= 0.45:
                union_ids(ids[i], ids[j])

    groups: dict = {}
    for layout in eligible:
        groups.setdefault(find(layout["id"]), []).append(layout)

    synthetic_layouts: list[dict] = []
    skip_ids: set = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        member_boxes = [boxes[member["id"]] for member in members]
        vertical_members = sum(
            1 for box in member_boxes
            if (box[3] - box[1]) >= max(1, box[2] - box[0]) * 1.35
        )
        if vertical_members >= max(1, len(members) // 2):
            ordered = sorted(members, key=lambda item: -_box_center(boxes[item["id"]])[0])
        else:
            ordered = sorted(
                members,
                key=lambda item: (_box_center(boxes[item["id"]])[1], _box_center(boxes[item["id"]])[0]),
            )
        text_parts = [
            str(trans_map[item["id"]].get("en_text", "")).strip()
            for item in ordered
            if str(trans_map[item["id"]].get("en_text", "")).strip()
        ]
        if len(text_parts) < 2:
            continue
        green_union = [
            min(box[0] for box in member_boxes),
            min(box[1] for box in member_boxes),
            max(box[2] for box in member_boxes),
            max(box[3] for box in member_boxes),
        ]
        red_boxes = [
            _coerce_box(member.get("red_box", member["green_box"]), image_size)
            for member in members
        ]
        red_union = [
            min(box[0] for box in red_boxes),
            min(box[1] for box in red_boxes),
            max(box[2] for box in red_boxes),
            max(box[3] for box in red_boxes),
        ]
        group_id = "merged_caption_" + "_".join(str(item["id"]) for item in ordered)
        synthetic_layouts.append({
            "id": group_id,
            "red_box": red_union,
            "green_box": green_union,
            "green_polygon": [
                [green_union[0], green_union[1]],
                [green_union[2], green_union[1]],
                [green_union[2], green_union[3]],
                [green_union[0], green_union[3]],
            ],
            "bubble_idx": -1,
            "route": "dialogue",
            "semantic_role": "dialogue",
            "member_ids": [item["id"] for item in ordered],
            "member_boxes": member_boxes,
        })
        trans_map[group_id] = {
            "id": group_id,
            "en_text": " ".join(text_parts),
            "translation_source": "merged_overlapping_captions",
        }
        skip_ids.update(item["id"] for item in ordered)
    return synthetic_layouts, skip_ids


def _find_mask_aware_layout(
    text: str,
    layout: dict,
    allowed_mask: Image.Image,
    text_style: dict,
    group_max_size_cap: int | None = None,
    selected_font_path: str | None = None,
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
    is_floating = _typeset_as_floating(layout)
    inferred_bubble = _inferred_bubble_layout(layout)
    median_luma = text_style.get("background_luma_median")
    dark_bubble_like = bool(
        is_floating
        and _floating_dialogue_layout(layout)
        and text_style.get("name") == "light_on_dark"
        and median_luma is not None
        and median_luma < 84.0
        and bounds_width >= 30
        and bounds_height >= 64
    )
    readable_floor = _readable_floor_for_bounds(bounds_width, bounds_height, is_floating)
    if dark_bubble_like:
        readable_floor = max(
            readable_floor,
            min(24, max(14, int(round(min(bounds_width, bounds_height) * 0.18)))),
        )
    tall_narrow_floating = bool(is_floating and not dark_bubble_like and bounds_height >= bounds_width * 1.35)
    modern_reference_bubble = (
        layout.get("fallback_source") == "peppercarrot_svg"
        and not is_floating
    )
    # One font for the whole page -- `tall_narrow_floating` still shapes SIZE and
    # wrapping below, it just no longer swaps the font family (that swap, to Arial
    # Narrow Bold, was one of the sources of same-page font mixing).
    font_path = _font_path_for_layout(layout, text, selected_font_path)

    max_size = min(MAX_FONT_DIALOGUE, max(MIN_FONT_SIZE, int(bounds_height * 0.9)))
    if is_floating:
        if layout.get("reverse_dark_bubble") or dark_bubble_like:
            max_size = min(max_size, max(34, min(50, int(bounds_height * 0.42))))
        else:
            max_size = min(max_size, 72 if text_style.get("source_cover") else max(36, min(80, int(bounds_height * 0.55))))
    elif inferred_bubble:
        max_size = min(max_size, max(36, min(72, int(bounds_height * 0.52))))
    if modern_reference_bubble:
        max_size = min(max_size, max(MIN_FONT_SIZE, int(bounds_height * 0.46)))
    if group_max_size_cap is not None:
        max_size = max(MIN_FONT_SIZE, min(max_size, group_max_size_cap))
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
            elif is_floating:
                if text_style.get("name") == "dark_on_light":
                    outline_width = max(outline_width, 3 if size <= 26 else 2)
                elif dark_bubble_like:
                    outline_width = max(outline_width, 3 if size >= 18 else 2)
                outline_width = min(
                    outline_width,
                    max(
                        int(text_style.get("floating_outline_cap", outline_width)),
                        3 if dark_bubble_like else 0,
                    ),
                )
            stroke_color = (
                text_style.get("floating_stroke_color", text_style["stroke_color"])
                if is_floating and not text_style.get("source_cover")
                else text_style["stroke_color"]
            )

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

                # Measure first (cheap: textbbox only) and reject oversized candidates before
                # paying for the FreeType render+stroke pass -- profiling showed that pass is
                # ~70% of step 8's wall time, and the vast majority of the ~800 candidates
                # tried per region fail this exact bounds check. block.width/height here are
                # identical to what the old single-call _render_text_block produced before
                # rendering, so this is the same decision, just made before instead of after
                # rasterizing.
                measurement = _measure_text_block(lines, font, size, outline_width)
                if (
                    measurement["block_width"] + 2 > bounds_width
                    or measurement["block_height"] + 2 > bounds_height
                ):
                    continue

                block, alpha, metrics = _render_text_block_from_measurement(
                    lines,
                    font,
                    measurement,
                    outline_width,
                    text_style["fill_color"],
                    stroke_color,
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
                    if is_floating:
                        if dark_bubble_like:
                            target_area_usage = 0.30
                            underuse_penalty = max(0.0, target_area_usage - area_usage) * 560.0
                            small_font_penalty = max(0, readable_floor - size) * 120.0
                            overfill_penalty = max(0.0, area_usage - 0.72) * 160.0
                            area_reward = 380.0
                        else:
                            target_area_usage = 0.18 if tall_narrow_floating else 0.20
                            underuse_penalty = max(0.0, target_area_usage - area_usage) * 220.0
                            small_font_penalty = max(0, readable_floor - size) * 95.0
                            overfill_penalty = max(0.0, area_usage - 0.68) * 120.0
                            area_reward = 260.0
                    else:
                        word_count = len(words)
                        if word_count <= 8:
                            target_area_usage = TARGET_BUBBLE_AREA_USAGE + 0.04
                        elif word_count <= 16:
                            target_area_usage = TARGET_BUBBLE_AREA_USAGE
                        else:
                            target_area_usage = TARGET_BUBBLE_AREA_USAGE - 0.06
                        if bounds_width < bounds_height * 0.75:
                            target_area_usage *= 0.82
                        underuse_penalty = max(0.0, target_area_usage - area_usage) * 680.0
                        small_font_penalty = max(0, readable_floor - size) * 110.0
                        overfill_penalty = max(0.0, area_usage - 0.66) * 220.0
                        area_reward = 360.0
                    score = (
                        size * 38.0
                        + area_usage * area_reward
                        - anchor_distance * 0.8
                        - aspect_penalty
                        - line_penalty
                        - short_line_penalty
                        - split_penalty
                        - underuse_penalty
                        - small_font_penalty
                        - overfill_penalty
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
            and (
                is_floating
                or (
                    (best_candidate["block"].width * best_candidate["block"].height)
                    / max(1, bounds_width * bounds_height)
                ) >= 0.20
                or best_candidate["font_size"] >= readable_floor + 2
            )
        ):
            return best_candidate

    if best_candidate is not None:
        return best_candidate

    # Fallback: text didn't fit at any tested size. Try to find the largest
    # size that minimizes clipping rather than dropping straight to MIN_FONT_SIZE.
    fallback_font_size = MIN_FONT_SIZE
    best_fallback_size = MIN_FONT_SIZE
    best_fallback_clipped = None
    for try_size in range(min(max_size, 48), MIN_FONT_SIZE - 1, -2):
        try_font = _load_font(try_size, font_path)
        try_outline = 0 if modern_reference_bubble else _outline_for_size(try_size, is_floating=is_floating)
        if text_style.get("source_cover"):
            try_outline = max(try_outline, min(5, int(round(try_size * 0.12))))
        elif is_floating:
            if text_style.get("name") == "dark_on_light":
                try_outline = max(try_outline, 3 if try_size <= 26 else 2)
            elif dark_bubble_like:
                try_outline = max(try_outline, 2)
            try_outline = min(
                try_outline,
                max(
                    int(text_style.get("floating_outline_cap", try_outline)),
                    3 if dark_bubble_like else 0,
                ),
            )
        try_stroke = (
            text_style.get("floating_stroke_color", text_style["stroke_color"])
            if is_floating and not text_style.get("source_cover")
            else text_style["stroke_color"]
        )
        try_lines = _wrap_standard(words, try_font, measure_draw, max(8.0, bounds_width * 0.9), allow_word_split=True, min_split_length=6)
        try_block, try_alpha, try_metrics = _render_text_block(
            try_lines or words, try_font, try_size, try_outline,
            text_style["fill_color"], try_stroke,
        )
        try_pos = _clamp_top_left(green_center, (try_block.width, try_block.height), mask_bounds) or (mask_bounds[0], mask_bounds[1])
        clipped = _count_clipped_pixels(try_alpha, allowed_np, try_pos)
        if clipped == 0:
            return {
                "font_size": try_size, "lines": try_lines, "block": try_block,
                "alpha": try_alpha, "position": try_pos, "metrics": try_metrics,
                "status": "fallback_fit", "clipped_pixels": 0,
            }
        if best_fallback_clipped is None or clipped < best_fallback_clipped:
            best_fallback_clipped = clipped
            best_fallback_size = try_size
            fallback_font_size = try_size
            fallback_font = try_font
            fallback_outline = try_outline
            fallback_lines = try_lines
            fallback_block = try_block
            fallback_alpha = try_alpha
            fallback_metrics = try_metrics
            fallback_position = try_pos

    # Last resort: every tested size still clipped even with the normal
    # (at most 2-piece) word split. Retry ONLY the single best-scoring size
    # found above, this time allowing a long word to break into more than 2
    # pieces (_multipiece_split_word) -- a very narrow bubble with a long
    # compound proper noun is exactly the case a 2-piece split can't rescue.
    # Only replaces the candidate on a strict improvement, so this can never
    # make an already-fitting or already-minimal-clipping render worse.
    used_multisplit_fallback = False
    if best_fallback_clipped is not None and best_fallback_clipped > 0:
        retry_stroke = (
            text_style.get("floating_stroke_color", text_style["stroke_color"])
            if is_floating and not text_style.get("source_cover")
            else text_style["stroke_color"]
        )
        retry_lines = _wrap_standard(
            words, fallback_font, measure_draw, max(8.0, bounds_width * 0.9),
            allow_word_split=True, min_split_length=6, allow_multipiece=True,
        )
        if retry_lines and retry_lines != fallback_lines:
            retry_block, retry_alpha, retry_metrics = _render_text_block(
                retry_lines, fallback_font, best_fallback_size, fallback_outline,
                text_style["fill_color"], retry_stroke,
            )
            retry_pos = _clamp_top_left(green_center, (retry_block.width, retry_block.height), mask_bounds) or (mask_bounds[0], mask_bounds[1])
            retry_clipped = _count_clipped_pixels(retry_alpha, allowed_np, retry_pos)
            if retry_clipped < best_fallback_clipped:
                fallback_font_size = best_fallback_size
                fallback_lines = retry_lines
                fallback_block = retry_block
                fallback_alpha = retry_alpha
                fallback_metrics = retry_metrics
                fallback_position = retry_pos
                best_fallback_clipped = retry_clipped
                # Only the quality gate's BAD_STEP8_STATUSES-escaping status
                # is reserved for a FULL fit -- a merely-smaller clip count
                # still needs the same "fallback_clipped" flag it always had
                # so the gate keeps surfacing it (clipped_pixels below is
                # still the improved, smaller count either way).
                used_multisplit_fallback = retry_clipped == 0

    # If no size was tried (max_size < MIN_FONT_SIZE), use MIN_FONT_SIZE
    if best_fallback_clipped is None:
        fallback_font_size = MIN_FONT_SIZE
        fallback_font = _load_font(fallback_font_size, font_path)
        fallback_outline = 0 if modern_reference_bubble else _outline_for_size(fallback_font_size, is_floating=is_floating)
        if text_style.get("source_cover"):
            fallback_outline = max(fallback_outline, min(5, int(round(fallback_font_size * 0.12))))
        elif is_floating:
            if text_style.get("name") == "dark_on_light":
                fallback_outline = max(fallback_outline, 3 if fallback_font_size <= 26 else 2)
            elif dark_bubble_like:
                fallback_outline = max(fallback_outline, 2)
            fallback_outline = min(
                fallback_outline,
                max(
                    int(text_style.get("floating_outline_cap", fallback_outline)),
                    3 if dark_bubble_like else 0,
                ),
            )
        fallback_stroke_color = (
            text_style.get("floating_stroke_color", text_style["stroke_color"])
            if is_floating and not text_style.get("source_cover")
            else text_style["stroke_color"]
        )
        fallback_lines = _wrap_standard(words, fallback_font, measure_draw, max(8.0, bounds_width * 0.9), allow_word_split=True, min_split_length=6)
        fallback_block, fallback_alpha, fallback_metrics = _render_text_block(
            fallback_lines or words, fallback_font, fallback_font_size, fallback_outline,
            text_style["fill_color"], fallback_stroke_color,
        )
        fallback_position = _clamp_top_left(green_center, (fallback_block.width, fallback_block.height), mask_bounds) or (mask_bounds[0], mask_bounds[1])
    return {
        "font_size": fallback_font_size,
        "lines": fallback_lines,
        "block": fallback_block,
        "alpha": fallback_alpha,
        "position": fallback_position,
        "metrics": fallback_metrics,
        "status": "fallback_fit_multisplit" if used_multisplit_fallback else "fallback_clipped",
        "clipped_pixels": _count_clipped_pixels(fallback_alpha, allowed_np, fallback_position),
    }


def run_step8_typeset(
    sample_map: dict[str, str] | None = None,
    samples_dir: Path | None = None,
    font_family: str | None = None,
):
    print("=" * 60)
    print("  Step 8 - Automated Typesetting (Mask-Aware Fit)")
    print("=" * 60)

    samples_dir = Path(samples_dir) if samples_dir is not None else sample_root_from_env(DEFAULT_SAMPLES_ROOT)
    sample_map = sample_map or SAMPLE_MAP

    # Resolve the page font ONCE, here, and pass it down explicitly. Not a module
    # global and not an env var: this function runs in-process and the GPU scheduler
    # admits multiple requests concurrently, so process-wide font state would race
    # between two pages that picked different fonts.
    selected_font_path, resolved_font_label = _resolve_font_family(font_family or DEFAULT_FONT_FAMILY)
    print(f"  [font] requested={font_family or DEFAULT_FONT_FAMILY!r} "
          f"resolved={resolved_font_label!r} file={os.path.basename(selected_font_path)}")

    for sample_name, img_file in sample_map.items():
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
        source_pil_img = None
        source_image_path = sample_path / img_file
        if source_image_path.exists():
            source_image = cv2.imread(str(source_image_path))
            if source_image is not None:
                if render_scale > 1:
                    source_image = cv2.resize(
                        source_image,
                        (native_size[0] * render_scale, native_size[1] * render_scale),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                source_pil_img = Image.fromarray(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)).convert("RGBA")
        image_size = pil_img.size
        # Caption haze state: contributions accumulate across the layout loop
        # and are composited once at the end, so overlapping captions merge
        # into a single smooth cloud and a later caption's backing can never
        # bury an earlier caption's text.
        erase_footprint = _load_erase_footprint(sample_path, image_size)
        caption_haze_acc = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        deferred_caption_layers: list[Image.Image] = []
        # Pixels already claimed by rendered text (slightly dilated). Floating
        # layouts subtract this from their allowed mask so adjacent fragments
        # can never typeset on top of each other (the "IS..A ITBIT" garble).
        occupied_np = np.zeros((image_size[1], image_size[0]), dtype=bool)
        occupied_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        report = []
        synthetic_layouts, synthetic_allowed_masks, grouped_skip_ids = _floating_reverse_dark_groups(
            layout_data,
            trans_map,
            pil_img,
            image_size,
        )
        if synthetic_layouts:
            layout_data = synthetic_layouts + [
                layout for layout in layout_data
                if layout.get("id") not in grouped_skip_ids
            ]

        sfx_preserved_ids = {
            int(key)
            for key, value in cleanup_status.items()
            if key.lstrip("-").isdigit()
            and isinstance(value, dict)
            and value.get("reason") == "sfx_preserved_artwork"
        }
        layout_data = _drop_duplicate_overlapping_layouts(
            layout_data, trans_map, image_size
        )
        merged_captions, merged_skip_ids = _merge_overlapping_floating_captions(
            layout_data,
            trans_map,
            image_size,
            excluded_ids=sfx_preserved_ids,
        )
        if merged_captions:
            layout_data = merged_captions + [
                layout for layout in layout_data
                if layout.get("id") not in merged_skip_ids
            ]

        group_font_caps = _compute_group_font_caps(layout_data, image_size)

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
            force_transparent_overlay = False
            caption_haze_box = None  # set when a busy-background floating caption gets an Ichigo-style feathered haze backing
            caption_frame_inner = None  # interior of a deliberate framed caption box, when the source has one
            caption_backing_mask = None  # silhouette of the artist's own translucent backing panel, when the source has one
            # This gate used to be scoped to `bubble_idx == -1` (floating
            # text only), so bubble dialogue was always typeset regardless
            # of whether Step 4 actually cleaned it. Now that Step 4 records
            # a real `cleaned` verdict for bubble regions too (residual-ink
            # check, not a hardcoded True), apply the same gate uniformly.
            status = cleanup_status.get(str(tid))
            if status is not None:
                cleanup_reason = status.get("reason", "unsafe_art_preserved")
            if status is not None and cleanup_reason in {
                "device_surface_preserved_overlay",
                "dark_device_surface_text_cleanup",
            }:
                force_transparent_overlay = True
            elif status is not None and not status.get("cleaned", True):
                if cleanup_reason == "device_surface_preserved_overlay":
                    force_transparent_overlay = True
                elif cleanup_reason == "device_overlay_required":
                    force_overlay_badge = True
                elif cleanup_reason == "source_cover_required":
                    force_source_cover = True
                else:
                    report.append({
                        "id": tid,
                        "text": en_text,
                        "status": "skipped_unsafe_floating_cleanup",
                        "reason": cleanup_reason,
                        "bubble_idx": layout.get("bubble_idx", -1),
                        "font_role": "floating" if _typeset_as_floating(layout) else "dialogue",
                        "font_file": os.path.basename(selected_font_path),
                        "font_family_resolved": resolved_font_label,
                    })
                    continue
            if force_overlay_badge:
                allowed_mask = _overlay_fallback_mask_for_layout(layout, image_size)
                text_style = _overlay_badge_style(pil_img, allowed_mask)
            elif force_source_cover:
                allowed_mask = _allowed_mask_for_layout(layout, image_size, detect_dir, None)
                text_style = _source_cover_text_style(pil_img, allowed_mask)
            elif force_transparent_overlay:
                allowed_mask = _device_surface_overlay_mask_for_layout(layout, image_size, pil_img)
                text_style = _text_style_for_layout(pil_img, allowed_mask)
            elif tid in synthetic_allowed_masks:
                pil_img = _cover_reverse_dark_source_text(
                    pil_img,
                    layout.get("member_boxes", []),
                )
                allowed_mask = synthetic_allowed_masks[tid]
                text_style = _text_style_for_layout(pil_img, allowed_mask)
            else:
                allowed_mask = _allowed_mask_for_layout(
                    layout,
                    image_size,
                    detect_dir,
                    pil_img,
                    source_image=source_pil_img,
                )
                text_style = _text_style_for_layout(pil_img, allowed_mask)
                # Text always renders TRANSPARENT (fill + stroke only) on the
                # reconstructed art — no haze, no synthetic backing, ever
                # (user rule, 2026-07-03: the framed-caption haze exception is
                # revoked). If the reconstruction under a caption still shows
                # damage, that is Step 4's defect to fix, not Step 8's to hide.
                if caption_haze_box is None and text_style.get("name") == "dark_on_light":
                    # Same color-identity rule for cleanly-typeset regions
                    # (framed caption boxes, bubbles with colored lettering):
                    # colored source glyphs keep their color, black stays black.
                    glyph_rgb = _source_caption_glyph_color(
                        source_pil_img,
                        _coerce_box(layout.get("red_box", layout.get("green_box")), image_size),
                        erase_footprint,
                    )
                    if glyph_rgb is not None:
                        text_style = {**text_style, "fill_color": (*glyph_rgb, 255)}
            if _typeset_as_floating(layout) and not force_overlay_badge and not force_source_cover and caption_haze_box is None:
                red_style_mask = Image.new("L", image_size, 0)
                ImageDraw.Draw(red_style_mask).rectangle(
                    _coerce_box(layout.get("red_box", layout["green_box"]), image_size),
                    fill=255,
                )
                red_style = _text_style_for_layout(pil_img, red_style_mask)
                if red_style["name"] == "light_on_dark":
                    red_median = red_style.get("background_luma_median")
                    red_p75 = red_style.get("background_luma_p75")
                    red_is_truly_dark = (
                        red_median is not None
                        and red_p75 is not None
                        and (red_median < 118 or (red_median < 132 and red_p75 < 142))
                    )
                    if red_is_truly_dark:
                        text_style = red_style
            if np.any(occupied_np):
                # Applies to bubbles too: on dense pages a bubble's own text
                # must not land on floating text already placed nearby.
                allowed_arr = np.array(allowed_mask, dtype=np.uint8)
                allowed_arr[occupied_np] = 0
                allowed_mask = Image.fromarray(allowed_arr, mode="L")
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
                fitted_candidates = []
                split_candidate = _find_two_region_layout(
                    candidate_text,
                    layout,
                    allowed_mask,
                    text_style,
                    selected_font_path=selected_font_path,
                )
                if split_candidate is not None:
                    fitted_candidates.append(split_candidate)
                fitted_candidates.append(
                    _find_mask_aware_layout(
                        candidate_text,
                        layout,
                        allowed_mask,
                        text_style,
                        group_max_size_cap=group_font_caps.get(tid),
                        selected_font_path=selected_font_path,
                    )
                )
                accepted = False
                for fitted_candidate in fitted_candidates:
                    clipped_candidate = _clipped_pixels_for_fit(fitted_candidate, allowed_np)
                    if selected_fit is None:
                        selected_text = candidate_text
                        selected_fit = fitted_candidate
                        selected_clipped_pixels = clipped_candidate
                    if clipped_candidate == 0 and fitted_candidate["status"] != "fallback_clipped":
                        selected_text = candidate_text
                        selected_fit = fitted_candidate
                        selected_clipped_pixels = 0
                        accepted = True
                        break
                if accepted:
                    break

            if selected_fit is None or selected_text is None:
                continue

            en_text = selected_text
            fitted = selected_fit
            # Hazed captions read best centered on the original text strip --
            # mask-aware placement can leave a short block hanging low in a
            # tall column. Recenter only when the centered position still fits.
            if caption_haze_box is not None and "parts" not in fitted:
                strip_cx = (caption_haze_box[0] + caption_haze_box[2]) / 2
                strip_cy = (caption_haze_box[1] + caption_haze_box[3]) / 2
                center_block = fitted["block"]
                centered_pos = (
                    max(0, min(image_size[0] - center_block.width, int(round(strip_cx - center_block.width / 2)))),
                    max(0, min(image_size[1] - center_block.height, int(round(strip_cy - center_block.height / 2)))),
                )
                if centered_pos != tuple(fitted["position"]) and _alpha_fits(
                    fitted["alpha"], allowed_np, centered_pos
                ):
                    fitted = {**fitted, "position": centered_pos}
                    selected_clipped_pixels = 0
            text_layer = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
            if "parts" in fitted:
                part_boxes = []
                for part in fitted["parts"]:
                    part_block = part["block"]
                    part_left, part_top = part["position"]
                    text_layer.alpha_composite(part_block, (part_left, part_top))
                    part_boxes.append([
                        part_left,
                        part_top,
                        part_left + part_block.width,
                        part_top + part_block.height,
                    ])
                left = min(box[0] for box in part_boxes)
                top = min(box[1] for box in part_boxes)
                right = max(box[2] for box in part_boxes)
                bottom = max(box[3] for box in part_boxes)
            else:
                text_block = fitted["block"]
                left, top = fitted["position"]
                text_layer.alpha_composite(text_block, (left, top))
                right = left + text_block.width
                bottom = top + text_block.height
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

            occupied_np |= cv2.dilate(
                (np.array(text_layer.getchannel("A")) > 0).astype(np.uint8),
                occupied_kernel,
            ).astype(bool)
            if caption_haze_box is not None:
                # Accumulate the feathered backing and defer the text: hazes
                # composite together after the loop, then all caption text
                # lands on top of the merged cloud.
                np.maximum(
                    caption_haze_acc,
                    _caption_haze_contribution(
                        image_size,
                        caption_haze_box,
                        erase_footprint,
                        text_layer,
                        fitted["font_size"],
                        framed_inner_box=caption_frame_inner,
                        backing_mask=caption_backing_mask,
                    ),
                    out=caption_haze_acc,
                )
                deferred_caption_layers.append(text_layer)
            else:
                pil_img = Image.alpha_composite(pil_img, text_layer)

            report.append({
                "id": tid,
                "text": en_text,
                # merged_from: the source ids a synthetic (merged/grouped) layout absorbed --
                # already computed at construction time (member_ids, set at each of the 3 sites
                # that build a synthetic layout: reverse_dark_bubble x2, merged_caption) but
                # dropped here before this fix. Without it, a gate checking "did every kept id
                # reach typeset_report" has no way to know an absorbed id's text still rendered
                # under a different id -- it looks identical to a real silent drop (measured:
                # 5 of G2's 6 "silent drops" were this, not real defects; see task #219).
                "merged_from": layout.get("member_ids") or [],
                "font_size": fitted["font_size"],
                "lines": fitted["lines"],
                "position": [left, top, right, bottom],
                "status": fitted["status"],
                "reason": cleanup_reason,
                "overlay_badge": force_overlay_badge,
                "source_cover": force_source_cover,
                "transparent_overlay": force_transparent_overlay,
                "clipped_pixels": clipped_pixels,
                "outline_width": fitted["metrics"]["outline_width"],
                "bubble_idx": layout.get("bubble_idx", -1),
                "render_scale": render_scale,
                "native_size": list(native_size),
                "output_size": [image_size[0], image_size[1]],
                "font_role": "floating" if _typeset_as_floating(layout) else "dialogue",
                "font_file": os.path.basename(selected_font_path),
                "font_family_resolved": resolved_font_label,
                "caption_backing": "feathered_haze" if caption_haze_box is not None else None,
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

        # Lay the merged caption haze, then all deferred caption text on top.
        if deferred_caption_layers:
            _composite_caption_haze(pil_img, caption_haze_acc)
            for caption_layer in deferred_caption_layers:
                pil_img = Image.alpha_composite(pil_img, caption_layer)

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
