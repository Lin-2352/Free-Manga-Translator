from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline_paths import PROJECT_ROOT


SCHEMA_VERSION = "training-case-v1"


def export_mode() -> str:
    value = os.environ.get("FMT_TRAINING_DATA_EXPORT", "review").strip().lower()
    aliases = {
        "1": "review",
        "true": "review",
        "yes": "review",
        "on": "review",
        "0": "off",
        "false": "off",
        "no": "off",
    }
    return aliases.get(value, value if value in {"off", "failure", "review", "all"} else "review")


def export_root() -> Path:
    raw = os.environ.get("FMT_TRAINING_DATA_EXPORT_ROOT", "training_data/failure_cases").strip()
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _json_read(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback
    except Exception:
        return fallback


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _box_tuple(value: Any) -> tuple[int, int, int, int] | None:
    if isinstance(value, dict):
        try:
            return (
                int(value["x1"]),
                int(value["y1"]),
                int(value["x2"]),
                int(value["y2"]),
            )
        except Exception:
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return tuple(int(value[index]) for index in range(4))  # type: ignore[return-value]
        except Exception:
            return None
    return None


def _expand_box(box: tuple[int, int, int, int], width: int, height: int, margin: int = 12) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(width, x2 + margin),
        min(height, y2 + margin),
    )


def _union_boxes(boxes: list[tuple[int, int, int, int]], width: int, height: int) -> tuple[int, int, int, int] | None:
    valid = [box for box in boxes if box[2] > box[0] and box[3] > box[1]]
    if not valid:
        return None
    return _expand_box(
        (
            min(box[0] for box in valid),
            min(box[1] for box in valid),
            max(box[2] for box in valid),
            max(box[3] for box in valid),
        ),
        width,
        height,
    )


def _copy_if_exists(source: Path, target: Path) -> str:
    if not source.exists():
        return ""
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target.name


def _save_crop(image: Image.Image, box: tuple[int, int, int, int], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(box).save(path, quality=95)
    return path.name


def _region_kind(ocr: dict[str, Any], layout: dict[str, Any] | None, rejected: dict[str, Any] | None) -> str:
    source = layout or rejected or ocr
    joined = " ".join(str(source.get(key, "")) for key in ("route", "semantic_role", "classification", "fallback_source", "mask_mode")).lower()
    if "sfx" in joined:
        return "sfx"
    if "device" in joined or "phone" in joined or "smartphone" in joined:
        return "device"
    if "art" in joined:
        return "art"
    if layout and int(layout.get("bubble_idx", -1)) != -1:
        return "dialogue"
    if "floating" in joined:
        return "dialogue"
    return "unknown"


def _expected_layout_behavior(kind: str, layout: dict[str, Any] | None) -> str:
    if kind == "sfx":
        return "skip_or_preserve_original_sfx"
    if kind == "device":
        return "clean_source_text_only_inside_device_surface_and_typeset_on_device"
    if layout and int(layout.get("bubble_idx", -1)) != -1:
        return "fit_translation_inside_speech_bubble_without_overflow"
    return "keep_translation_near_original_text_and_avoid_unrelated_character_art"


def _translation_by_id(translations: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for item in translations:
        try:
            output[int(item["id"])] = item
        except Exception:
            continue
    return output


def _dict_by_id(items: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for item in items:
        try:
            output[int(item["id"])] = item
        except Exception:
            continue
    return output


def _region_reasons(
    item_id: int,
    layout: dict[str, Any] | None,
    rejected: dict[str, Any] | None,
    translation: dict[str, Any] | None,
    rendered_ids: set[int],
) -> list[str]:
    reasons: list[str] = []
    if rejected:
        reasons.append(f"layout_rejected:{rejected.get('reason') or rejected.get('classification') or 'unknown'}")
    if not layout:
        reasons.append("missing_layout_constraint")
    text = str((translation or {}).get("en_text", "")).strip()
    if not text:
        reasons.append("missing_translation")
    elif text.startswith("[TL:"):
        reasons.append("placeholder_translation")
    if layout and item_id not in rendered_ids:
        reasons.append("layout_not_rendered")
    return reasons


def _should_export(report: dict[str, Any], critical_errors: list[str] | None) -> bool:
    mode = export_mode()
    if mode == "off":
        return False
    if mode == "all":
        return True
    if critical_errors:
        return True
    warnings = list(report.get("reviewWarnings") or [])
    if mode == "review" and warnings:
        return True
    return False


def export_runtime_training_case(
    samples_root: Path,
    sample_name: str,
    language: str,
    report: dict[str, Any],
    critical_errors: list[str] | None = None,
) -> dict[str, Any] | None:
    if not _should_export(report, critical_errors):
        return None

    sample_path = samples_root / sample_name
    input_path = sample_path / "input.jpg"
    if not input_path.exists():
        return None

    image_hash = _sha_file(input_path)
    reason_payload = {
        "warnings": report.get("reviewWarnings", []),
        "criticalErrors": critical_errors or [],
        "outputSafety": report.get("outputSafety"),
    }
    reason_hash = hashlib.sha256(json.dumps(reason_payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    case_id = f"{language}_{image_hash[:12]}_{reason_hash}"
    case_dir = export_root() / case_id
    regions_dir = case_dir / "regions"
    artifacts_dir = case_dir / "artifacts"
    case_dir.mkdir(parents=True, exist_ok=True)
    regions_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ocr_items = _json_read(sample_path / "step_5_ocr" / "ocr_results.json", [])
    layout_items = _json_read(sample_path / "step_6_layout" / "layout_constraints.json", [])
    rejected_items = _json_read(sample_path / "step_6_layout" / "rejected_layout_items.json", [])
    translation_items = _json_read(sample_path / "step_7_translate" / "translation_results.json", [])
    typeset_items = _json_read(sample_path / "step_8_typeset" / "typeset_report.json", [])

    layout_by_id = _dict_by_id(layout_items)
    rejected_by_id = _dict_by_id(rejected_items)
    translations_by_id = _translation_by_id(translation_items)
    rendered_ids = {
        int(item["id"])
        for item in typeset_items
        if isinstance(item, dict) and "id" in item and str(item.get("status", "")).lower() not in {"missing", "empty"}
    }

    max_regions = max(1, int(os.environ.get("FMT_TRAINING_DATA_EXPORT_MAX_REGIONS", "48")))
    source_image = Image.open(input_path).convert("RGB")
    inpaint_path = sample_path / "step_4_final" / "inpainted_result.jpg"
    final_path = sample_path / "step_8_typeset" / "final_output.png"
    inpaint_image = Image.open(inpaint_path).convert("RGB") if inpaint_path.exists() else None
    final_image = Image.open(final_path).convert("RGB") if final_path.exists() else None
    width, height = source_image.size

    _copy_if_exists(input_path, artifacts_dir / "input.jpg")
    _copy_if_exists(sample_path / "step_6_layout" / "debug_layout_boxes.jpg", artifacts_dir / "debug_layout_boxes.jpg")
    _copy_if_exists(inpaint_path, artifacts_dir / "inpainted_result.jpg")
    _copy_if_exists(final_path, artifacts_dir / "final_output.png")

    region_records: list[dict[str, Any]] = []
    label_records: list[dict[str, Any]] = []
    for ocr in ocr_items[:max_regions]:
        try:
            item_id = int(ocr["id"])
        except Exception:
            continue
        layout = layout_by_id.get(item_id)
        rejected = rejected_by_id.get(item_id)
        translation = translations_by_id.get(item_id)
        reasons = _region_reasons(item_id, layout, rejected, translation, rendered_ids)
        if not reasons and export_mode() != "all":
            continue
        boxes = []
        for source in (ocr, layout or {}, rejected or {}):
            for key in ("box", "red_box", "green_box"):
                box = _box_tuple(source.get(key))
                if box:
                    boxes.append(box)
            erase_boxes = source.get("erase_boxes", [])
            if not isinstance(erase_boxes, list):
                erase_boxes = []
            for erase_box in erase_boxes:
                box = _box_tuple(erase_box)
                if box:
                    boxes.append(box)
        crop_box = _union_boxes(boxes, width, height)
        if crop_box is None:
            continue
        prefix = f"region_{item_id:03d}"
        input_crop = _save_crop(source_image, crop_box, regions_dir / f"{prefix}_input.jpg")
        inpaint_crop = _save_crop(inpaint_image, crop_box, regions_dir / f"{prefix}_inpainted.jpg") if inpaint_image else ""
        final_crop = _save_crop(final_image, crop_box, regions_dir / f"{prefix}_final.jpg") if final_image else ""
        kind = _region_kind(ocr, layout, rejected)
        expected_behavior = _expected_layout_behavior(kind, layout)
        region_records.append(
            {
                "id": item_id,
                "reasons": reasons,
                "inputCrop": f"regions/{input_crop}",
                "inpaintedCrop": f"regions/{inpaint_crop}" if inpaint_crop else "",
                "finalCrop": f"regions/{final_crop}" if final_crop else "",
                "sourceText": str(ocr.get("text", "")),
                "currentEnglish": str((translation or {}).get("en_text", "")),
                "translationSource": str((translation or {}).get("translation_source", "")),
                "regionKindGuess": kind,
                "expectedLayoutBehaviorGuess": expected_behavior,
                "boxes": {
                    "ocrBox": ocr.get("box"),
                    "redBox": (layout or rejected or {}).get("red_box"),
                    "eraseBoxes": (layout or rejected or {}).get("erase_boxes", []),
                    "greenBox": (layout or rejected or ocr).get("green_box"),
                    "greenPolygon": (layout or rejected or ocr).get("green_polygon", []),
                },
                "metadata": {
                    "ocrProvider": ocr.get("ocr_provider", ""),
                    "ocrConfidence": ocr.get("ocr_confidence"),
                    "route": (layout or rejected or ocr).get("route", ""),
                    "bubbleIdx": (layout or rejected or {}).get("bubble_idx", -1),
                    "maskMode": (layout or rejected or ocr).get("mask_mode", ""),
                    "layoutAdjustment": (layout or rejected or {}).get("layout_adjustment", ""),
                },
            }
        )
        label_records.append(
            {
                "id": item_id,
                "correctOcrText": "",
                "correctEnglishTranslation": "",
                "regionKind": kind,
                "regionKindOptions": ["dialogue", "sfx", "art", "device", "unknown"],
                "expectedLayoutBehavior": expected_behavior,
                "notes": "",
                "acceptedForTraining": False,
            }
        )

    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "caseId": case_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sampleName": sample_name,
        "sourceLanguage": language,
        "sourceImageHash": image_hash,
        "exportMode": export_mode(),
        "reason": reason_payload,
        "artifacts": {
            "inputImage": "artifacts/input.jpg",
            "debugLayoutBoxes": "artifacts/debug_layout_boxes.jpg",
            "inpaintedResult": "artifacts/inpainted_result.jpg" if inpaint_path.exists() else "",
            "finalOutput": "artifacts/final_output.png" if final_path.exists() else "",
        },
        "pipelineReport": report,
        "regions": region_records,
    }
    (case_dir / "case_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    labels_path = case_dir / "labels_pending.json"
    if not labels_path.exists():
        labels_path.write_text(json.dumps({"caseId": case_id, "labels": label_records}, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "caseId": case_id,
        "path": str(case_dir),
        "regions": len(region_records),
        "labels": str(labels_path),
    }
