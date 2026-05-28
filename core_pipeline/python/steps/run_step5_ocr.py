"""
Step 5 — OCR & Consolidation (v4: Self-Contained Detection)
==========================================================
1. Check for Step 1-3 results. If missing, RUN detection models.
2. Group and consolidate detections by bubble.
3. Run Japanese OCR on consolidated Red Box regions.
4. Save results for Layout and Translation.
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
import cv2
import numpy as np
from pathlib import Path
from pipeline_paths import DEFAULT_SAMPLES_ROOT, sample_root_from_env
from ml_region_lib import (
    MLConfig, load_ocr_model, load_text_model, load_bubble_model, load_semantic_model,
    detect_text, detect_bubbles, detect_semantic_text_regions,
    build_step2_routing_state, consolidate_by_bubble, Box, 
    SAMPLE_MAP, classify_text_by_content
)

_EASYOCR_READERS = {}
_PADDLEOCR_READERS = {}
_PADDLEOCR_UNAVAILABLE = set()
_MANGA_OCR_MODEL = None
_TEXT_HANDLE = None
_BUBBLE_MODEL = None
_BUBBLE_DEVICE = None
_SEMANTIC_HANDLE = None


def _local_cjk_mode() -> bool:
    return os.environ.get("LOCAL_NLLB_TRANSLATION", "").strip().lower() in {"1", "true", "yes", "on"}


def _sample_cjk_ocr_language(sample_name: str) -> str | None:
    lowered = sample_name.lower()
    if "_zh_" in lowered or lowered.startswith(("external_zh", "modern_zh", "runtime_zh")):
        return "ch_tra"
    if "_ko_" in lowered or lowered.startswith(("external_ko", "modern_ko", "runtime_ko")):
        return "ko"
    return None


def _easyocr_reader(language: str):
    if language not in _EASYOCR_READERS:
        import easyocr

        _EASYOCR_READERS[language] = easyocr.Reader([language, "en"], gpu=True, verbose=False)
        print(f"  [Model D2] EasyOCR {language}: LOADED")
    return _EASYOCR_READERS[language]


def _paddleocr_reader(language: str):
    if language not in {"ko", "ch"}:
        return None
    if language in _PADDLEOCR_UNAVAILABLE:
        return None
    if language not in _PADDLEOCR_READERS:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            from paddleocr import PaddleOCR
        except ModuleNotFoundError:
            _PADDLEOCR_UNAVAILABLE.add(language)
            print(f"  [PaddleOCR warn] {language}: paddleocr is not installed; using non-Paddle OCR fallbacks")
            return None

        paddle_lang = "korean" if language == "ko" else "ch"
        _PADDLEOCR_READERS[language] = PaddleOCR(
            lang=paddle_lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        print(f"  [Model D3] PaddleOCR {paddle_lang}: LOADED")
    return _PADDLEOCR_READERS[language]


def _script_count(text: str, script: str) -> int:
    import re

    patterns = {
        "hangul": r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]",
        "han": r"[\u3400-\u9fff]",
        "kana": r"[\u3040-\u30ff]",
    }
    return len(re.findall(patterns[script], text or ""))


def _combine_easyocr_results(results: list, language: str) -> dict:
    if not results:
        return {"text": "", "confidence": 0.0}

    min_confidence = 0.35 if language == "ch_tra" else 0.55
    kept = []
    for polygon, text, confidence in results:
        clean_text = str(text or "").strip()
        if not clean_text or float(confidence) < min_confidence:
            continue
        if language == "ko" and _script_count(clean_text, "hangul") == 0:
            continue
        if language == "ch_tra" and _script_count(clean_text, "han") == 0:
            continue
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        kept.append(
            {
                "text": clean_text,
                "confidence": float(confidence),
                "x": float(min(xs)),
                "y": float(min(ys)),
                "height": float(max(ys) - min(ys)),
            }
        )

    if not kept:
        return {"text": "", "confidence": 0.0}

    kept.sort(key=lambda item: (round(item["y"] / max(8.0, item["height"] * 0.7)), item["x"]))
    text = " ".join(item["text"] for item in kept).strip()
    weights = [max(1, len(item["text"])) for item in kept]
    confidence = sum(item["confidence"] * weight for item, weight in zip(kept, weights)) / sum(weights)
    return {"text": text, "confidence": confidence}


class LocalCjkOcr:
    def __init__(self, manga_ocr_model, language: str | None):
        self.manga_ocr_model = manga_ocr_model
        self.language = language

    def __call__(self, pil_image):
        return self.read_pil(pil_image)["text"]

    def read_pil(self, pil_image) -> dict:
        if not self.language:
            text = self.manga_ocr_model(pil_image)
            return {"text": text, "provider": "manga_ocr", "confidence": None}

        crop_rgb = np.array(pil_image.convert("RGB"))
        try:
            results = _easyocr_reader(self.language).readtext(
                crop_rgb,
                detail=1,
                paragraph=False,
                batch_size=8,
                contrast_ths=0.05,
                adjust_contrast=0.7,
                text_threshold=0.4,
                low_text=0.2,
                link_threshold=0.3,
            )
        except Exception as error:
            print(f"  [EasyOCR warn] {self.language}: {str(error)[:100]}")
            return {"text": "", "provider": f"easyocr_{self.language}", "confidence": 0.0}

        combined = _combine_easyocr_results(results, self.language)
        return {
            "text": combined["text"],
            "provider": f"easyocr_{self.language}",
            "confidence": round(float(combined["confidence"]), 4),
        }


def _read_ocr_crop(ocr_runtime, crop) -> dict:
    from PIL import Image

    if crop.size == 0:
        return {"text": "", "provider": "empty", "confidence": 0.0}
    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if hasattr(ocr_runtime, "read_pil"):
        return ocr_runtime.read_pil(pil_crop)
    text = ocr_runtime(pil_crop)
    return {"text": text, "provider": "manga_ocr", "confidence": None}


def _map_paddle_box_to_original(box, angle, img_w: int, img_h: int) -> Box:
    x1, y1, x2, y2 = [float(v) for v in box]
    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    try:
        angle = int(angle or 0) % 360
    except Exception:
        angle = 0

    mapped = []
    for x, y in points:
        if angle == 270:
            mapped.append((y, img_h - x))
        elif angle == 90:
            mapped.append((img_w - y, x))
        elif angle == 180:
            mapped.append((img_w - x, img_h - y))
        else:
            mapped.append((x, y))

    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    return Box(
        max(0, int(round(min(xs)))),
        max(0, int(round(min(ys)))),
        min(img_w, int(round(max(xs)))),
        min(img_h, int(round(max(ys)))),
    )


def _expanded_vertical_layout_box(red_box: Box, img_w: int, img_h: int) -> Box:
    width = max(1, red_box.width)
    height = max(1, red_box.height)
    if height >= width * 1.7:
        target_width = min(img_w, max(width + 48, 96, int(height * 0.45)))
        target_height = min(img_h, height + max(18, int(height * 0.08)))
    else:
        target_width = min(img_w, max(width + 32, int(width * 1.35)))
        target_height = min(img_h, height + 20)

    center_x = (red_box.x1 + red_box.x2) // 2
    center_y = (red_box.y1 + red_box.y2) // 2
    x1 = max(0, min(img_w - target_width, center_x - target_width // 2))
    y1 = max(0, min(img_h - target_height, center_y - target_height // 2))
    return Box(int(x1), int(y1), int(x1 + target_width), int(y1 + target_height))


def _paddle_result_payload(result) -> dict:
    payload = getattr(result, "json", None)
    if isinstance(payload, dict):
        return payload.get("res", payload)
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
        if isinstance(payload, dict):
            return payload.get("res", payload)
    return {}


def _boxes_overlap(left: Box, right: Box) -> float:
    inter_x1 = max(left.x1, right.x1)
    inter_y1 = max(left.y1, right.y1)
    inter_x2 = min(left.x2, right.x2)
    inter_y2 = min(left.y2, right.y2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    smaller = min(max(1, left.width * left.height), max(1, right.width * right.height))
    return intersection / smaller


def _expanded_xy(box: Box, pad_x: int, pad_y: int, img_w: int, img_h: int) -> Box:
    return Box(
        max(0, box.x1 - pad_x),
        max(0, box.y1 - pad_y),
        min(img_w, box.x2 + pad_x),
        min(img_h, box.y2 + pad_y),
    )


def _union_boxes(boxes: list[Box], img_w: int, img_h: int, pad: int = 0) -> Box:
    return Box(
        max(0, min(box.x1 for box in boxes) - pad),
        max(0, min(box.y1 for box in boxes) - pad),
        min(img_w, max(box.x2 for box in boxes) + pad),
        min(img_h, max(box.y2 for box in boxes) + pad),
    )


def _vertical_group_connected(left: Box, right: Box, img_w: int, img_h: int) -> bool:
    prospective = _union_boxes([left, right], img_w, img_h)
    if prospective.height > int(img_h * 0.72) and prospective.width > int(img_w * 0.24):
        return False

    center_gap_y = abs(((left.y1 + left.y2) / 2) - ((right.y1 + right.y2) / 2))
    if center_gap_y > img_h * 0.30:
        return False

    left_expanded = _expanded_xy(left, 8, 14, img_w, img_h)
    right_expanded = _expanded_xy(right, 8, 14, img_w, img_h)
    if _boxes_overlap(left_expanded, right_expanded) > 0:
        return True

    y_overlap = max(0, min(left.y2, right.y2) - max(left.y1, right.y1))
    y_ratio = y_overlap / max(1, min(left.height, right.height))
    x_gap = max(0, max(left.x1, right.x1) - min(left.x2, right.x2))
    max_column_gap = max(14, int(img_w * 0.018))
    if prospective.width > int(img_w * 0.30):
        max_column_gap = min(max_column_gap, 16)
    return y_ratio >= 0.45 and x_gap <= max_column_gap


def _is_vertical_page_column(box: Box, img_h: int) -> bool:
    return box.height >= max(80, int(img_h * 0.18)) and box.height >= box.width * 4.0


def _split_overwide_vertical_group(
    members: list[dict],
    member_boxes: list[Box],
    img_w: int,
    img_h: int,
) -> list[list[tuple[dict, Box]]]:
    if len(members) < 4:
        return [list(zip(members, member_boxes))]

    vertical_boxes = [box for box in member_boxes if _is_vertical_page_column(box, img_h)]
    if len(vertical_boxes) < 4:
        return [list(zip(members, member_boxes))]

    union_box = _union_boxes(member_boxes, img_w, img_h)
    median_width = sorted(box.width for box in vertical_boxes)[len(vertical_boxes) // 2]
    if union_box.width <= max(90, int(img_w * 0.11), int(median_width * 2.9)):
        return [list(zip(members, member_boxes))]

    ordered = sorted(zip(members, member_boxes), key=lambda pair: pair[1].x1)
    chunk_size = 2 if len(ordered) <= 5 else 3
    chunks: list[list[tuple[dict, Box]]] = []
    for index in range(0, len(ordered), chunk_size):
        chunk = ordered[index:index + chunk_size]
        if chunks and len(chunk) == 1:
            chunks[-1].extend(chunk)
        else:
            chunks.append(chunk)
    return chunks


def _group_paddle_page_items(items: list[dict], img_w: int, img_h: int) -> list[dict]:
    if not items:
        return []

    boxes = [
        Box(item["box"]["x1"], item["box"]["y1"], item["box"]["x2"], item["box"]["y2"])
        for item in items
    ]
    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index in range(len(items)):
        for right_index in range(left_index + 1, len(items)):
            if _vertical_group_connected(boxes[left_index], boxes[right_index], img_w, img_h):
                union(left_index, right_index)

    grouped_indices: dict[int, list[int]] = {}
    for index in range(len(items)):
        grouped_indices.setdefault(find(index), []).append(index)

    grouped_items = []
    for indices in grouped_indices.values():
        members = [items[index] for index in indices]
        member_boxes = [boxes[index] for index in indices]
        for chunk in _split_overwide_vertical_group(members, member_boxes, img_w, img_h):
            chunk_members = [member for member, _ in chunk]
            chunk_boxes = [box for _, box in chunk]
            if len(chunk_members) == 1:
                grouped_items.append(chunk_members[0])
                continue

            red_box = _union_boxes(chunk_boxes, img_w, img_h, pad=6)
            green_box = red_box.expanded(max(12, int(min(red_box.width, red_box.height) * 0.08)), img_w, img_h)
            ordered = sorted(chunk_members, key=lambda item: (item["box"]["x1"], item["box"]["y1"]))
            combined_text = " ".join(item["text"] for item in ordered).strip()
            confidence_values = [float(item.get("ocr_confidence") or 0.0) for item in chunk_members]
            confidence = sum(confidence_values) / max(1, len(confidence_values))
            erase_boxes = []
            for member in chunk_members:
                erase_boxes.extend(member.get("erase_boxes") or [member["box"]])
            providers = sorted({str(member.get("ocr_provider", "paddleocr")).removesuffix("_grouped") for member in chunk_members})
            grouped_provider = (
                f"{providers[0]}_grouped"
                if len(providers) == 1
                else "paddleocr_mixed_grouped"
            )
            grouped_items.append({
                "id": len(grouped_items),
                "text": combined_text,
                "ocr_provider": grouped_provider,
                "ocr_confidence": round(confidence, 4),
                "box": {k: int(v) for k, v in red_box.to_dict().items()},
                "erase_boxes": [
                    {k: int(v) for k, v in Box(box["x1"], box["y1"], box["x2"], box["y2"]).to_dict().items()}
                    for box in erase_boxes
                ],
                "green_box": {k: int(v) for k, v in green_box.to_dict().items()},
                "green_polygon": [
                    [green_box.x1, green_box.y1],
                    [green_box.x2, green_box.y1],
                    [green_box.x2, green_box.y2],
                    [green_box.x1, green_box.y2],
                ],
                "route": "floating_dialogue",
                "bubble_idx": -1,
                "mask_mode": "stroke",
                "fallback_source": "paddleocr_korean_grouped_page",
                "force_bubble_cleanup": False,
            })

    grouped_items.sort(key=lambda item: (item["box"]["y1"], item["box"]["x1"]))
    for idx, item in enumerate(grouped_items):
        item["id"] = idx
    return grouped_items


def _fallback_korean_from_paddle(image_path: Path, image) -> list[dict]:
    img_h, img_w = image.shape[:2]
    final_results = []

    ocr_passes = [
        ("ko", "paddleocr_ko", "hangul", 2, 0.50),
        ("ch", "paddleocr_ch_mixed", "han", 2, 0.65),
    ]
    for language, provider, script, min_script_count, min_score in ocr_passes:
        pass_results = []
        pass_seen = set()
        reader = _paddleocr_reader(language)
        if reader is None:
            continue
        try:
            results = reader.predict(str(image_path))
        except Exception as error:
            print(f"  [PaddleOCR warn] {language}: {str(error)[:120]}")
            continue

        for result in results:
            payload = _paddle_result_payload(result)
            angle = (payload.get("doc_preprocessor_res") or {}).get("angle", 0)
            texts = payload.get("rec_texts") or []
            scores = payload.get("rec_scores") or []
            boxes = payload.get("rec_boxes") or []

            for text, score, box in zip(texts, scores, boxes):
                clean_text = str(text or "").strip()
                if not clean_text:
                    continue
                score = float(score or 0.0)
                script_hits = _script_count(clean_text, script)
                korean_geometry_hint = (
                    language == "ko"
                    and script_hits == 1
                    and score >= 0.08
                )
                if script_hits < min_script_count and not korean_geometry_hint:
                    continue
                if score < min_score and not korean_geometry_hint:
                    continue

                red_box = _map_paddle_box_to_original(box, angle, img_w, img_h).expanded(4, img_w, img_h)
                if red_box.width < 8 or red_box.height < 8:
                    continue
                if any(_boxes_overlap(red_box, Box(item["box"]["x1"], item["box"]["y1"], item["box"]["x2"], item["box"]["y2"])) > 0.72 for item in pass_results):
                    continue
                key = (clean_text, red_box.x1 // 6, red_box.y1 // 6, red_box.x2 // 6, red_box.y2 // 6)
                if key in pass_seen:
                    continue
                pass_seen.add(key)

                green_box = _expanded_vertical_layout_box(red_box, img_w, img_h)
                pass_results.append({
                    "id": len(pass_results),
                    "text": clean_text,
                    "ocr_provider": provider,
                    "ocr_confidence": round(score, 4),
                    "box": {k: int(v) for k, v in red_box.to_dict().items()},
                    "erase_boxes": [{k: int(v) for k, v in red_box.to_dict().items()}],
                    "green_box": {k: int(v) for k, v in green_box.to_dict().items()},
                    "green_polygon": [
                        [green_box.x1, green_box.y1],
                        [green_box.x2, green_box.y1],
                        [green_box.x2, green_box.y2],
                        [green_box.x1, green_box.y2],
                    ],
                    "route": "floating_dialogue",
                    "bubble_idx": -1,
                    "mask_mode": "stroke",
                    "fallback_source": "paddleocr_korean_page",
                    "force_bubble_cleanup": False,
                })

        for item in _group_paddle_page_items(pass_results, img_w, img_h):
            item_box = Box(item["box"]["x1"], item["box"]["y1"], item["box"]["x2"], item["box"]["y2"])
            if any(
                _boxes_overlap(
                    item_box,
                    Box(existing["box"]["x1"], existing["box"]["y1"], existing["box"]["x2"], existing["box"]["y2"]),
                )
                > 0.72
                for existing in final_results
            ):
                continue
            final_results.append(item)

    final_results.sort(key=lambda item: (item["box"]["y1"], item["box"]["x1"]))
    for idx, item in enumerate(final_results):
        item["id"] = idx
    return final_results


def _bubble_overlap_ratio(box: Box, bubble_mask) -> float:
    if bubble_mask is None or box.area <= 0:
        return 0.0
    roi = bubble_mask[box.y1:box.y2, box.x1:box.x2]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi > 0)) / float(max(1, box.area))


def _best_bubble_for_line(box: Box, bubble_masks) -> int:
    best_idx = -1
    best_ratio = 0.0
    for idx, bubble_mask in enumerate(bubble_masks):
        ratio = _bubble_overlap_ratio(box, bubble_mask)
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = idx
    return best_idx if best_ratio >= 0.18 else -1


def _weighted_confidence(lines: list[dict]) -> float:
    weights = [max(1, len(line["text"])) for line in lines]
    return sum(float(line["score"]) * weight for line, weight in zip(lines, weights)) / max(1, sum(weights))


def _sort_horizontal_lines(lines: list[dict]) -> list[dict]:
    if not lines:
        return []
    median_height = float(np.median([line["box"].height for line in lines])) if lines else 16.0
    row_unit = max(10.0, median_height * 0.75)
    return sorted(lines, key=lambda line: (round(line["box"].y1 / row_unit), line["box"].x1))


def _horizontal_lines_belong_together(previous: Box, current: Box) -> bool:
    max_height = max(previous.height, current.height)
    gap_y = current.y1 - previous.y2
    if gap_y > max(32, int(max_height * 1.35)):
        return False

    overlap_x = max(0, min(previous.x2, current.x2) - max(previous.x1, current.x1))
    min_width = max(1, min(previous.width, current.width))
    max_width = max(previous.width, current.width)
    left_aligned = abs(previous.x1 - current.x1) <= max(22, int(max_width * 0.18))
    center_aligned = abs(((previous.x1 + previous.x2) / 2) - ((current.x1 + current.x2) / 2)) <= max(
        34,
        int(max_width * 0.32),
    )
    strong_overlap = overlap_x >= int(min_width * 0.35)

    if gap_y <= max(8, int(max_height * 0.40)) and (strong_overlap or left_aligned):
        return True
    return strong_overlap or left_aligned or center_aligned


def _cluster_horizontal_cjk_lines(lines: list[dict]) -> list[list[dict]]:
    ordered = _sort_horizontal_lines(lines)
    clusters: list[list[dict]] = []
    for line in ordered:
        if not clusters:
            clusters.append([line])
            continue
        previous = clusters[-1][-1]["box"]
        current = line["box"]
        if _horizontal_lines_belong_together(previous, current):
            clusters[-1].append(line)
        else:
            clusters.append([line])
    return clusters


def _safe_bubble_mask(bubble_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(bubble_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return bubble_mask.copy()
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    kernel_size = max(3, min(19, int(min(bw, bh) * 0.045)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    safe = cv2.erode(bubble_mask, kernel, iterations=1)
    return safe if np.count_nonzero(safe > 0) >= 60 else bubble_mask.copy()


def _polygon_from_mask(mask: np.ndarray, fallback_box: Box) -> list[list[int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        epsilon = max(1.5, cv2.arcLength(largest, True) * 0.006)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        points = [point[0].astype(int).tolist() for point in approx]
        if len(points) >= 3:
            return points
    return [
        [fallback_box.x1, fallback_box.y1],
        [fallback_box.x2, fallback_box.y1],
        [fallback_box.x2, fallback_box.y2],
        [fallback_box.x1, fallback_box.y2],
    ]


def _bubble_cluster_zones(cluster_boxes: list[Box], bubble_mask: np.ndarray) -> list[dict]:
    safe_bubble = _safe_bubble_mask(bubble_mask)
    h, w = safe_bubble.shape
    zones: list[dict] = []
    if not cluster_boxes:
        return zones

    if len(cluster_boxes) == 1:
        territory_masks = [safe_bubble]
    else:
        dist_maps = []
        for box in cluster_boxes:
            seed = np.ones((h, w), dtype=np.uint8) * 255
            cv2.rectangle(seed, (box.x1, box.y1), (box.x2, box.y2), 0, -1)
            dist_maps.append(cv2.distanceTransform(seed, cv2.DIST_L2, 5))
        assignments = np.argmin(np.stack(dist_maps), axis=0)
        territory_masks = [
            (((assignments == idx) & (safe_bubble > 0)).astype(np.uint8) * 255)
            for idx in range(len(cluster_boxes))
        ]

    for box, territory in zip(cluster_boxes, territory_masks):
        if np.count_nonzero(territory > 0) < 30:
            territory = np.zeros_like(safe_bubble)
            territory[box.y1:box.y2, box.x1:box.x2] = 255
            territory = cv2.bitwise_and(territory, safe_bubble)
        points = _polygon_from_mask(territory, box)
        xs = [int(point[0]) for point in points]
        ys = [int(point[1]) for point in points]
        green_box = Box(max(0, min(xs)), max(0, min(ys)), min(w, max(xs) + 1), min(h, max(ys) + 1))
        zones.append({"green_box": green_box, "green_polygon": points})
    return zones


def _paddle_chinese_page_lines(image_path: Path, image, bubble_masks) -> list[dict]:
    reader = _paddleocr_reader("ch")
    if reader is None:
        return []
    img_h, img_w = image.shape[:2]
    try:
        results = reader.predict(str(image_path))
    except Exception as error:
        print(f"  [PaddleOCR warn] ch page: {str(error)[:120]}")
        return []

    lines = []
    seen = set()
    for result in results:
        payload = _paddle_result_payload(result)
        angle = (payload.get("doc_preprocessor_res") or {}).get("angle", 0)
        texts = payload.get("rec_texts") or []
        scores = payload.get("rec_scores") or []
        boxes = payload.get("rec_boxes") or []
        for text, score, box in zip(texts, scores, boxes):
            clean_text = str(text or "").strip()
            if not clean_text:
                continue
            score = float(score or 0.0)
            if score < 0.42:
                continue
            han_count = _script_count(clean_text, "han")
            if han_count == 0:
                continue
            red_box = _map_paddle_box_to_original(box, angle, img_w, img_h).expanded(3, img_w, img_h)
            if red_box.width < 6 or red_box.height < 6 or red_box.area < 45:
                continue
            key = (clean_text, red_box.x1 // 5, red_box.y1 // 5, red_box.x2 // 5, red_box.y2 // 5)
            if key in seen:
                continue
            seen.add(key)
            lines.append({
                "text": clean_text,
                "score": score,
                "box": red_box,
                "bubble_idx": _best_bubble_for_line(red_box, bubble_masks),
            })
    return lines


def _item_from_chinese_cluster(
    item_id: int,
    lines: list[dict],
    img_w: int,
    img_h: int,
    bubble_idx: int,
    zone: dict | None,
) -> dict:
    ordered = _sort_horizontal_lines(lines)
    line_boxes = [line["box"] for line in ordered]
    red_box = _union_boxes(line_boxes, img_w, img_h, pad=4)
    text = " ".join(line["text"] for line in ordered).strip()
    confidence = _weighted_confidence(ordered)
    if bubble_idx != -1 and zone is not None:
        green_box = zone["green_box"]
        green_polygon = zone["green_polygon"]
        route = "bubble_dialogue"
        mask_mode = "bubble_interior"
    else:
        green_box = red_box.expanded(max(10, int(min(red_box.width, red_box.height) * 0.18)), img_w, img_h)
        green_polygon = [
            [green_box.x1, green_box.y1],
            [green_box.x2, green_box.y1],
            [green_box.x2, green_box.y2],
            [green_box.x1, green_box.y2],
        ]
        route = "floating_dialogue"
        mask_mode = "stroke"
    return {
        "id": item_id,
        "text": text,
        "ocr_provider": "paddleocr_ch_page_clustered",
        "ocr_confidence": round(float(confidence), 4),
        "box": {k: int(v) for k, v in red_box.to_dict().items()},
        "erase_boxes": [{k: int(v) for k, v in box.to_dict().items()} for box in line_boxes],
        "green_box": {k: int(v) for k, v in green_box.to_dict().items()},
        "green_polygon": green_polygon,
        "route": route,
        "bubble_idx": int(bubble_idx),
        "mask_mode": mask_mode,
        "fallback_source": "paddleocr_ch_page_clustered",
        "force_bubble_cleanup": False,
    }


def _fallback_chinese_from_paddle(image_path: Path, image, bubble_masks) -> list[dict]:
    lines = _paddle_chinese_page_lines(image_path, image, bubble_masks)
    if not lines:
        return []

    img_h, img_w = image.shape[:2]
    grouped: dict[int, list[dict]] = {}
    for line in lines:
        grouped.setdefault(int(line["bubble_idx"]), []).append(line)

    items: list[dict] = []
    next_id = 0
    for bubble_idx in sorted(idx for idx in grouped if idx != -1):
        clusters = _cluster_horizontal_cjk_lines(grouped[bubble_idx])
        cluster_boxes = [_union_boxes([line["box"] for line in cluster], img_w, img_h, pad=4) for cluster in clusters]
        zones = _bubble_cluster_zones(cluster_boxes, bubble_masks[bubble_idx])
        for cluster, zone in zip(clusters, zones):
            items.append(_item_from_chinese_cluster(next_id, cluster, img_w, img_h, bubble_idx, zone))
            next_id += 1

    floating_clusters = _cluster_horizontal_cjk_lines(grouped.get(-1, []))
    for cluster in floating_clusters:
        items.append(_item_from_chinese_cluster(next_id, cluster, img_w, img_h, -1, None))
        next_id += 1

    items.sort(key=lambda item: (item["box"]["y1"], item["box"]["x1"]))
    for idx, item in enumerate(items):
        item["id"] = idx
    return items


def _save_ocr_outputs(sample_path: Path, image: np.ndarray, final_results: list[dict]) -> None:
    out_dir = sample_path / "step_5_ocr"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    debug_img = image.copy()
    for res in final_results:
        rb, gb = res["box"], res["green_box"]
        poly = res.get("green_polygon", [])
        cv2.rectangle(debug_img, (rb["x1"], rb["y1"]), (rb["x2"], rb["y2"]), (0, 0, 255), 2)

        color = (0, 165, 255) if res.get("overlap_collision") else (0, 255, 0)
        if poly and len(poly) >= 3:
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug_img, [pts], isClosed=True, color=color, thickness=2)
        else:
            cv2.rectangle(debug_img, (gb["x1"], gb["y1"]), (gb["x2"], gb["y2"]), color, 2)
    cv2.imwrite(str(out_dir / "debug_ocr_boxes.jpg"), debug_img)


def _best_bubble_for_box(box: Box, bubble_masks) -> int:
    best_idx = -1
    best_overlap = 0
    for idx, bubble_mask in enumerate(bubble_masks):
        roi = bubble_mask[box.y1:box.y2, box.x1:box.x2]
        overlap = int(np.count_nonzero(roi > 0))
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx
    return best_idx


def _mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _fallback_ocr_from_raw_detections(image, text_boxes, bubble_masks, ocr_model):
    if not text_boxes:
        return None

    img_h, img_w = image.shape[:2]
    usable_boxes = [
        box for box in text_boxes
        if box.width * box.height >= 80 and box.width >= 5 and box.height >= 5
    ]
    if not usable_boxes:
        return None

    x1 = max(0, min(box.x1 for box in usable_boxes) - 14)
    y1 = max(0, min(box.y1 for box in usable_boxes) - 14)
    x2 = min(img_w, max(box.x2 for box in usable_boxes) + 14)
    y2 = min(img_h, max(box.y2 for box in usable_boxes) + 14)
    union_box = Box(x1, y1, x2, y2)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    ocr_meta = _read_ocr_crop(ocr_model, crop)
    ocr_text = ocr_meta["text"]
    if classify_text_by_content(ocr_text) != "dialogue":
        return None

    bubble_idx = _best_bubble_for_box(union_box, bubble_masks)
    erase_box = union_box
    green_box = union_box.expanded(18, img_w, img_h)
    if bubble_idx != -1:
        bounds = _mask_bounds(bubble_masks[bubble_idx])
        if bounds:
            bx1, by1, bx2, by2 = bounds
            inset_x = max(2, int((bx2 - bx1) * 0.04))
            inset_y = max(2, int((by2 - by1) * 0.04))
            green_box = Box(
                max(0, bx1 + inset_x),
                max(0, by1 + inset_y),
                min(img_w, bx2 - inset_x),
                min(img_h, by2 - inset_y),
            )
            erase_box = green_box
    else:
        margin_x = max(4, int(img_w * 0.03))
        margin_y = max(4, int(img_h * 0.03))
        green_box = Box(margin_x, margin_y, img_w - margin_x, img_h - margin_y)
        erase_box = green_box

    return {
        "text": ocr_text,
        "box": erase_box,
        "green_box": green_box,
        "green_polygon": [
            [green_box.x1, green_box.y1],
            [green_box.x2, green_box.y1],
            [green_box.x2, green_box.y2],
            [green_box.x1, green_box.y2],
        ],
        "route": "bubble_dialogue" if bubble_idx != -1 else "floating_dialogue",
        "bubble_idx": bubble_idx,
        "mask_mode": "bubble_interior" if bubble_idx != -1 else "stroke",
        "force_bubble_cleanup": True,
        "ocr_provider": ocr_meta.get("provider", "unknown"),
        "ocr_confidence": ocr_meta.get("confidence"),
    }

def run_step5_ocr():
    global _MANGA_OCR_MODEL, _TEXT_HANDLE, _BUBBLE_MODEL, _BUBBLE_DEVICE, _SEMANTIC_HANDLE

    print("=" * 60)
    print("  Step 5 — OCR & Consolidation (v4)")
    print("=" * 60)

    cfg = MLConfig()
    samples_dir = sample_root_from_env(DEFAULT_SAMPLES_ROOT)
    resume_existing = os.environ.get("PIPELINE_RESUME_EXISTING_OCR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    def get_ocr_runtime(language: str | None):
        global _MANGA_OCR_MODEL
        if language:
            return LocalCjkOcr(None, language)
        if _MANGA_OCR_MODEL is None:
            _MANGA_OCR_MODEL = load_ocr_model(force_cpu=False)
        return LocalCjkOcr(_MANGA_OCR_MODEL, None)
    

    text_handle = _TEXT_HANDLE
    bubble_model = _BUBBLE_MODEL
    bubble_device = _BUBBLE_DEVICE
    semantic_handle = _SEMANTIC_HANDLE

    for sample_name, img_file in SAMPLE_MAP.items():
        sample_path = samples_dir / sample_name
        img_path = sample_path / img_file
        if not img_path.exists(): continue
            
        print(f"\nProcessing {sample_name}")
        existing_ocr = sample_path / "step_5_ocr" / "ocr_results.json"
        if resume_existing and existing_ocr.exists():
            print(f"  [resume] existing Step 5 OCR kept: {existing_ocr}")
            continue
        sample_ocr_language = _sample_cjk_ocr_language(sample_name) if _local_cjk_mode() else None
        ocr_runtime = get_ocr_runtime(sample_ocr_language)
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        

        detect_dir = sample_path / "step_1_detect"
        step1_res_path = detect_dir / "detections.json"
        seg_mask_path = detect_dir / "seg_mask.png"
        
        if not step1_res_path.exists() or not seg_mask_path.exists():
            print(f"  Step 1 results missing. Running detection models...")
            if text_handle is None:
                text_handle = load_text_model(cfg.text_model_path)
                bubble_model, bubble_device = load_bubble_model(cfg.bubble_model_path)
                semantic_handle = load_semantic_model("magi")
                _TEXT_HANDLE = text_handle
                _BUBBLE_MODEL = bubble_model
                _BUBBLE_DEVICE = bubble_device
                _SEMANTIC_HANDLE = semantic_handle
            
            detect_dir.mkdir(parents=True, exist_ok=True)
            

            text_result = detect_text(text_handle, image, cfg)
            cv2.imwrite(str(seg_mask_path), text_result.seg_mask)
            with open(step1_res_path, 'w') as f:
                json.dump({"boxes": [{k: int(v) for k, v in b.to_dict().items()} for b in text_result.boxes]}, f)
                

            bubble_masks = detect_bubbles(bubble_model, bubble_device, image, cfg)
            for i, bm in enumerate(bubble_masks):
                cv2.imwrite(str(detect_dir / f"bubble_{i}.png"), bm)
            
            semantic_result = detect_semantic_text_regions(semantic_handle, image, cfg)
            with open(detect_dir / "semantic_detections.json", 'w') as f:
                json.dump({"regions": [{"box": {k: int(v) for k, v in r.box.to_dict().items()}, "class_id": int(r.class_id), 
                                     "raw_class_name": str(r.raw_class_name), "semantic_class": str(r.semantic_class),
                                     "action": str(r.action), "confidence": float(r.confidence)} for r in semantic_result.regions]}, f)
        else:

            from ml_region_lib import TextDetectionResult, SemanticDetectionResult, SemanticTextRegion
            seg_mask = cv2.imread(str(seg_mask_path), cv2.IMREAD_GRAYSCALE)
            with open(step1_res_path, 'r') as f:
                d1 = json.load(f)
                text_result = TextDetectionResult(boxes=[Box(b["x1"], b["y1"], b["x2"], b["y2"]) for b in d1["boxes"]], seg_mask=seg_mask)
            with open(detect_dir / "semantic_detections.json", 'r') as f:
                d2 = json.load(f)
                semantic_result = SemanticDetectionResult(regions=[SemanticTextRegion(box=Box(r["box"]["x1"], r["box"]["y1"], r["box"]["x2"], r["box"]["y2"]), **{k:v for k,v in r.items() if k!="box"}) for r in d2["regions"]])
            bubble_masks = []
            for i in range(100):
                bm_p = detect_dir / f"bubble_{i}.png"
                if bm_p.exists(): bubble_masks.append(cv2.imread(str(bm_p), cv2.IMREAD_GRAYSCALE))
                else: break

        if sample_ocr_language == "ch_tra":
            final_results = _fallback_chinese_from_paddle(img_path, image, bubble_masks)
            if final_results:
                print(f"  [paddle-ch] using {len(final_results)} clustered page OCR regions")
                _save_ocr_outputs(sample_path, image, final_results)
                continue
        

        routed = build_step2_routing_state(text_result, semantic_result, bubble_masks, cfg, w, h, ocr_runtime, image)
        

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        consolidated = consolidate_by_bubble(routed, text_result.seg_mask, bubble_masks, cfg, gray_img)
        

        final_results = []
        for idx, ct in enumerate(consolidated):
            if ct.route_state == "onomatopoeia": continue
            
            crop = image[ct.box.y1:ct.box.y2, ct.box.x1:ct.box.x2]
            if crop.size == 0: continue
            
            ocr_meta = _read_ocr_crop(ocr_runtime, crop)
            ocr_text = ocr_meta["text"]
            
            final_results.append({
                "id": idx,
                "text": ocr_text,
                "ocr_provider": ocr_meta.get("provider", "unknown"),
                "ocr_confidence": ocr_meta.get("confidence"),
                "box": {k: int(v) for k, v in ct.box.to_dict().items()},
                "green_box": {k: int(v) for k, v in ct.expanded_box.to_dict().items()},
                "green_polygon": getattr(ct, "green_polygon", []),
                "route": ct.route_state,
                "bubble_idx": int(ct.bubble_idx),
                "mask_mode": ct.mask_mode,
                "overlap_collision": getattr(ct, "overlap_collision", False)
            })

            print(f"  [{idx}] {ocr_text[:30]}...")

        if not final_results:
            fallback = _fallback_ocr_from_raw_detections(
                image, text_result.boxes, bubble_masks, ocr_runtime
            )
            if fallback:
                rb = fallback["box"]
                gb = fallback["green_box"]
                final_results.append({
                    "id": 0,
                    "text": fallback["text"],
                    "box": {k: int(v) for k, v in rb.to_dict().items()},
                    "green_box": {k: int(v) for k, v in gb.to_dict().items()},
                    "green_polygon": fallback["green_polygon"],
                    "route": fallback["route"],
                    "bubble_idx": int(fallback["bubble_idx"]),
                    "mask_mode": fallback["mask_mode"],
                    "overlap_collision": False,
                    "fallback_source": "raw_detection_union",
                    "force_bubble_cleanup": fallback.get("force_bubble_cleanup", False),
                    "ocr_provider": fallback.get("ocr_provider", "unknown"),
                    "ocr_confidence": fallback.get("ocr_confidence"),
                })
                print(f"  [fallback] {fallback['text'][:30]}...")

        if not final_results and sample_ocr_language == "ko":
            final_results = _fallback_korean_from_paddle(img_path, image)
            for item in final_results:
                print(f"  [paddle] {item['text'][:30]}...")

        _save_ocr_outputs(sample_path, image, final_results)

if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass
    run_step5_ocr()
