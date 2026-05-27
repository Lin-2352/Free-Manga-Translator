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
import os
import re
import cv2
import numpy as np
from pathlib import Path
from ml_region_lib import SAMPLE_MAP, classify_text_by_content
from pipeline_paths import DEFAULT_SAMPLES_ROOT, sample_root_from_env


CREDIT_MARKERS = ("原作", "作画", "漫画", "監修", "キャラクターデザイン")


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


_SHORT_DIALOGUE_STEMS = (
    "\u3042\u3063\u305f",                          
    "\u3042\u3064",                            
    "\u3055\u3080",             
    "\u3044\u305f",              
    "\u3084\u3060",                   
    "\u3060\u3081",                
    "\u3044\u3084",                   
    "\u3044\u3044",                    
    "\u306d\u3048",            
    "\u307b\u3089",             
    "\u307e\u3063\u3066",         
    "\u3053\u308c",             
    "\u305d\u308c",             
    "\u306a\u3093",                           
    "\u3059\u3054",                
    "\u3053\u308f",              
    "\u304d\u3082",              
    "\u3046\u305d",                     
)


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
        "\u3061\u3085\u3071",         
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
    if not compact or _obvious_sfx_fragment(text):
        return False

    counts = _script_counts(text)
    cjk_total = counts["hangul"] + counts["kana"] + counts["han"]
    if cjk_total < 2 or len(compact) > 8:
        return False
    if counts["han"] >= 1 or counts["hangul"] >= 2:
        return True
    if not any(stem in compact for stem in _SHORT_DIALOGUE_STEMS):
        return False

    box = item.get("box", {})
    coords = (
        int(box.get("x1", 0)),
        int(box.get("y1", 0)),
        int(box.get("x2", 0)),
        int(box.get("y2", 0)),
    )
    pale_fraction = _pale_region_fraction(image, coords)
    dark_fraction = _dark_region_fraction(image, coords)
    return pale_fraction >= 0.32 or dark_fraction >= 0.18




def _recoverable_large_floating_dialogue(
    text: str,
    counts: dict[str, int],
    width: int,
    height: int,
    x1: int,
    y1: int,
    img_w: int,
    img_h: int,
) -> bool:
    cleaned = str(text or "").strip()
    compact_len = len("".join(c for c in cleaned if c.isalnum()))
    cjk_total = counts["hangul"] + counts["kana"] + counts["han"]
    if compact_len < 8 or cjk_total < 6:
        return False
    if not _strong_dialogue_candidate(cleaned, counts):
        return False

    area_ratio = (width * height) / max(1, img_w * img_h)
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

    if wide_title_like or top_title_like:
        return False
    return vertical_dialogue or compact_caption



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
    cls = classify_text_by_content(text)
    pale_fraction = _pale_region_fraction(image, (x1, y1, x2, y2))
    dark_fraction = _dark_region_fraction(image, (x1, y1, x2, y2))
    pale_fragment = pale_fraction >= 0.42
    top_pale_fragment = bool(img_h and y1 <= int(img_h * 0.26) and pale_fragment)
    safe_caption_fragment = pale_fraction >= 0.55 and dark_fraction <= 0.24

                                                                               
                                                                               
                                                                                
                                                                               
                             
    if cls in {"sfx", "noise", "english"}:
        return False

    if cls != "dialogue" and not (top_pale_fragment or safe_caption_fragment):
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
        same_vertical_band = v_overlap >= max(8, int(min(height, r_height) * 0.20))
        close_y = abs(center_y - ((ry1 + ry2) / 2.0)) <= max(92, int(max(height, r_height) * 0.90))
        close_x = h_gap <= max(96, int(min(height, r_height) * 1.15))
        if close_x and (same_vertical_band or (close_y and v_gap <= max(36, int(max(height, r_height) * 0.28)))):
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
    ocr_confidence = item.get("ocr_confidence")
    ocr_provider = str(item.get("ocr_provider", ""))
    external_ko_vertical_paddle = (
        sample_name.startswith("external_ko")
        and ocr_provider.startswith(("paddleocr_ko", "paddleocr_ch_mixed"))
        and (counts["hangul"] >= 2 or counts["han"] >= 2)
        and height >= width * 1.15
        and (not img_w or width <= max(380, int(img_w * 0.42)))
    )

    if _external_local_mode():
        if sample_name.startswith("external_ko"):
            if counts["hangul"] == 0 and not (ocr_provider.startswith("paddleocr") and counts["han"] >= 2):
                return "ocr_language_mismatch"
            min_korean_confidence = 0.65 if ocr_provider.startswith("paddleocr_ch_mixed") else 0.45 if ocr_provider.startswith("paddleocr_ko") else 0.55
            if ocr_confidence is not None and float(ocr_confidence) < min_korean_confidence:
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
            recover_large_dialogue = _recoverable_large_floating_dialogue(
                text,
                counts,
                width,
                height,
                x1,
                y1,
                img_w,
                img_h,
            )
            if (
                y1 <= int(img_h * 0.20)
                and (pale_fraction < 0.65 or dark_fraction > 0.24)
                and not external_ko_vertical_paddle
                and not strong_dialogue
            ):
                return "title_logo"
            if y1 <= int(img_h * 0.18) and not external_ko_vertical_paddle and not strong_dialogue:
                return "title_logo"
            if area_ratio > 0.025 and not external_ko_vertical_paddle and not recover_large_dialogue:
                return "floating_too_large"
            if width > int(img_w * 0.28) and not external_ko_vertical_paddle and not recover_large_dialogue:
                return "floating_too_wide"

    if bubble_idx != -1:
        return "dialogue"

    compact_text_len = len("".join(c for c in text if c.isalnum()))

    if any(marker in text for marker in CREDIT_MARKERS):
        return "credit"
    if img_h and y1 >= int(img_h * 0.90) and height <= max(80, int(img_h * 0.06)):
        return "credit"
    if img_h and y1 <= int(img_h * 0.08) and compact_text_len <= 4:
        return "title_logo"
    if bubble_idx == -1 and img_h and img_w:
        area_ratio = (width * height) / max(1, img_h * img_w)
        if (
            (area_ratio > 0.065 or width > int(img_w * 0.52))
            and counts["han"] == 0
            and compact_text_len < 10
        ):
            return "floating_too_large"
    return "dialogue"


def run_step6_layout():
    print("=" * 60)
    print("  Step 6 — Layout Export (v4: Strict 3-Color Debug)")
    print("=" * 60)

    samples_dir = sample_root_from_env(DEFAULT_SAMPLES_ROOT)

    for sample_name, img_file in SAMPLE_MAP.items():
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

                                                                          
                                                                      
                                                          
        constraints = []
        rejected = []
        for item in ocr_data:
            semantic_role = _semantic_role_for_item(item, image_shape, sample_name, image)
            if semantic_role in {
                "credit",
                "title_logo",
                "floating_too_large",
                "floating_too_wide",
                "ocr_language_mismatch",
                "ocr_low_confidence",
            }:
                rejected.append(_layout_rejection(item, semantic_role, semantic_role=semantic_role))
                continue

            cls = classify_text_by_content(item.get("text", ""))
            if (
                item.get("bubble_idx", -1) == -1
                and _floating_roi_has_dominant_sfx_art(item, image)
            ):
                rejected.append(_layout_rejection(item, "floating_sfx_art", semantic_role="sfx", classification=cls))
                continue
            keep_short_bubble = (
                cls == "sfx"
                and item.get("bubble_idx", -1) != -1
                and _has_usable_translation(item.get("text", ""))
            )
            if cls != "dialogue" and not keep_short_bubble:
                rejected.append(_layout_rejection(item, f"classification_{cls}", semantic_role=semantic_role, classification=cls))
                continue
            if cls == "dialogue" and not _has_usable_translation(item.get("text", "")):
                rejected.append(_layout_rejection(item, "no_usable_translation", semantic_role=semantic_role, classification=cls))
                continue

            rb = item["box"]
            gb = item["green_box"]
            constraints.append({
                "id":              item["id"],
                "text":            item["text"],
                "red_box":         [rb["x1"], rb["y1"], rb["x2"], rb["y2"]],
                "erase_boxes":      item.get("erase_boxes", []),
                "green_box":       [gb["x1"], gb["y1"], gb["x2"], gb["y2"]],
                "green_polygon":   item.get("green_polygon", []),
                "bubble_idx":      item.get("bubble_idx", -1),
                "mask_mode":       item.get("mask_mode", "stroke"),
                "route":           item.get("route", "floating_dialogue"),
                "semantic_role":   semantic_role,
                "fallback_source": item.get("fallback_source"),
                "force_bubble_cleanup": bool(item.get("force_bubble_cleanup", False)),
            })

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

                                                                          
        out_dir = sample_path / "step_6_layout"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "layout_constraints.json", "w", encoding="utf-8") as f:
            json.dump(constraints, f, indent=2, ensure_ascii=False)
        with open(out_dir / "rejected_layout_items.json", "w", encoding="utf-8") as f:
            json.dump(rejected, f, indent=2, ensure_ascii=False)

                                                                           
        if image is None:
            continue

                                           
        if detect_dir.exists():
            for i in range(100):
                bm_path = detect_dir / f"bubble_{i}.png"
                if not bm_path.exists():
                    break
                bmask = cv2.imread(str(bm_path), cv2.IMREAD_GRAYSCALE)
                if bmask is not None:
                    contours, _ = cv2.findContours(
                        bmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(image, contours, -1, (255, 0, 0), 2)

                                           
        for c in constraints:
            rb   = c["red_box"]
            gb   = c["green_box"]
            poly = c.get("green_polygon", [])

                                                 
            cv2.rectangle(
                image,
                (rb[0], rb[1]), (rb[2], rb[3]),
                (0, 0, 255), 2
            )

                                                                        
            if poly and len(poly) >= 3:
                pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    image, [pts], isClosed=True,
                    color=(0, 255, 0), thickness=1                             
                )
            else:
                cv2.rectangle(
                    image,
                    (gb[0], gb[1]), (gb[2], gb[3]),
                    (0, 255, 0), 1
                )

        cv2.imwrite(str(out_dir / "debug_layout_boxes.jpg"), image)
        print(f"  Saved {len(constraints)} constraints -> {out_dir}/debug_layout_boxes.jpg")


if __name__ == "__main__":
    run_step6_layout()
