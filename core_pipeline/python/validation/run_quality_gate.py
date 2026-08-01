"""
Quality gate for the local manga translation pipeline.

This script does not regenerate model outputs. It validates the current run
folders, summarizes Step 4 / Step 8 risks, and creates visual contact sheets
for fast human review.
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
import py_compile
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from ml_region_lib import SAMPLE_MAP
from pipeline_paths import DEFAULT_SAMPLES_ROOT, PROJECT_ROOT, sample_root_from_env


PIPELINE_SCRIPTS = [
    "python/common/ml_region_lib.py",
    "python/common/pipeline_paths.py",
    "python/reference/run_svg_reference_translation_pipeline.py",
    "python/steps/run_multi_step3.py",
    "python/steps/run_step4_inpaint.py",
    "python/steps/run_step5_ocr.py",
    "python/steps/run_step6_layout.py",
    "python/steps/run_step7_translate.py",
    "python/steps/run_step8_typeset.py",
    "python/runtime/run_extension_pipeline_server.py",
    "python/validation/run_test_run_pipeline.py",
    "python/validation/run_test_run_quality_audit.py",
]

BAD_STEP8_STATUSES = {"fallback_clipped", "emergency", "clipped", "missing", "empty"}
LOW_READABILITY_FONT_SIZE = 9
STEP4_BUBBLE_DARK_FRACTION_WARN = 0.12
STEP4_FLOATING_PATCH_MIN_PIXELS = 80
STEP4_FLOATING_TONE_DELTA_WARN = 32.0
STEP4_FLOATING_TEXTURE_RATIO_WARN = 0.36
STEP4_FLOATING_RING_PADDING = 18
CONTACT_THUMB_SIZE = (300, 420)


@dataclass
class QualityIssue:
    severity: str
    sample: str
    check: str
    message: str


@dataclass
class QualityState:
    issues: list[QualityIssue] = field(default_factory=list)
    sample_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    compiled_scripts: list[str] = field(default_factory=list)

    def add(self, severity: str, sample: str, check: str, message: str) -> None:
        self.issues.append(QualityIssue(severity, sample, check, message))

    @property
    def failures(self) -> list[QualityIssue]:
        return [issue for issue in self.issues if issue.severity == "fail"]

    @property
    def warnings(self) -> list[QualityIssue]:
        return [issue for issue in self.issues if issue.severity == "warn"]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _compile_scripts(state: QualityState, project_root: Path) -> None:
    for script_name in PIPELINE_SCRIPTS:
        script_path = project_root / script_name
        if not script_path.exists():
            state.add("fail", "global", "compile", f"Missing pipeline script: {script_name}")
            continue
        try:
            py_compile.compile(str(script_path), doraise=True)
            state.compiled_scripts.append(script_name)
        except py_compile.PyCompileError as error:
            state.add("fail", "global", "compile", f"{script_name}: {error.msg}")


def _is_placeholder_translation(text: str) -> bool:
    return text.strip().startswith("[TL:")


def _region_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _summarize_step8_report(
    state: QualityState,
    sample_name: str,
    report_path: Path,
    strict_translation: bool = False,
) -> dict[str, Any]:
    report = _read_json(report_path)
    status_counts = Counter(item.get("status", "unknown") for item in report)
    clipped_items = [item for item in report if int(item.get("clipped_pixels", 0)) > 0]
    bad_items = [item for item in report if item.get("status") in BAD_STEP8_STATUSES]
    low_font_items = [
        item
        for item in report
        if int(item.get("font_size", 999)) <= LOW_READABILITY_FONT_SIZE
    ]
    floating_low_outline = [
        item
        for item in report
        if item.get("font_role") == "floating" and int(item.get("outline_width", 0)) < 2
    ]

    for item in clipped_items:
        state.add(
            "fail",
            sample_name,
            "step8_clipping",
            f"Region {item.get('id')} clipped {item.get('clipped_pixels')} pixels.",
        )
    for item in bad_items:
        state.add(
            "fail",
            sample_name,
            "step8_status",
            f"Region {item.get('id')} has bad status {item.get('status')}.",
        )
    low_font_severity = "fail" if strict_translation else "warn"
    for item in low_font_items:
        state.add(
            low_font_severity,
            sample_name,
            "step8_readability",
            f"Region {item.get('id')} uses very small font size {item.get('font_size')}.",
        )
    for item in floating_low_outline:
        state.add(
            "fail",
            sample_name,
            "floating_outline",
            f"Floating region {item.get('id')} has outline width {item.get('outline_width')}.",
        )

    return {
        "regions": len(report),
        "statuses": dict(status_counts),
        "clipped_regions": len(clipped_items),
        "bad_status_regions": len(bad_items),
        "low_font_regions": len(low_font_items),
        "render_scales": sorted({item.get("render_scale", 1) for item in report}),
        "text_styles": dict(Counter(item.get("text_style", "unknown") for item in report)),
        "output_sizes": sorted({
            tuple(item.get("output_size", []))
            for item in report
            if item.get("output_size")
        }),
    }


def _check_step7_translations(
    state: QualityState,
    sample_name: str,
    translation_path: Path,
    layout_ids: set[str],
    rendered_ids: set[str],
    strict_translation: bool,
) -> dict[str, Any]:
    translations = _read_json(translation_path)
    layout_translations = [
        item
        for item in translations
        if _region_id(item.get("id")) in layout_ids
    ]
    placeholder_items = [
        item for item in layout_translations if _is_placeholder_translation(item.get("en_text", ""))
    ]
    untranslated_renderable = [
        item
        for item in placeholder_items
        if any(ch.isalpha() for ch in item.get("en_text", ""))
    ]
    translated_ids = {
        region_id
        for item in layout_translations
        if item.get("en_text")
        for region_id in [_region_id(item.get("id"))]
        if region_id is not None
    }
    missing_rendered_ids = sorted(translated_ids - rendered_ids)

    severity = "fail" if strict_translation else "warn"
    for item in untranslated_renderable:
        state.add(
            severity,
            sample_name,
            "translation_placeholder",
            f"Region {item.get('id')} still has placeholder translation.",
        )

    if missing_rendered_ids:
        state.add(
            "warn",
            sample_name,
            "step8_skipped_translation",
            f"Translated IDs not rendered by Step 8: {missing_rendered_ids}",
        )

    return {
        "translations": len(layout_translations),
        "placeholder_translations": len(placeholder_items),
        "translated_not_rendered": missing_rendered_ids,
    }


def _check_stale_outputs(
    state: QualityState,
    sample_name: str,
    final_output_path: Path,
    prerequisites: list[Path],
) -> None:
    if not final_output_path.exists():
        return
    final_mtime = final_output_path.stat().st_mtime
    stale_inputs = [
        path.name
        for path in prerequisites
        if path.exists() and path.stat().st_mtime > final_mtime
    ]
    if stale_inputs:
        state.add(
            "warn",
            sample_name,
            "stale_output",
            f"Step 8 output is older than prerequisites: {stale_inputs}",
        )


def _coerce_box(box: list[int], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    left = max(0, min(width, int(round(box[0]))))
    top = max(0, min(height, int(round(box[1]))))
    right = max(0, min(width, int(round(box[2]))))
    bottom = max(0, min(height, int(round(box[3]))))
    return left, top, right, bottom


def _check_step4_residue(
    state: QualityState,
    sample_name: str,
    step4_path: Path,
    step4_mask_path: Path,
    constraints: list[dict[str, Any]],
) -> dict[str, Any]:
    if not step4_path.exists() or not constraints:
        return {"bubble_residue_warnings": 0}

    try:
        image = Image.open(step4_path).convert("L")
    except OSError:
        state.add("fail", sample_name, "step4_load", f"Cannot read Step 4 image: {step4_path}")
        return {"bubble_residue_warnings": 0}

    step4_mask: Image.Image | None = None
    if step4_mask_path.exists():
        try:
            step4_mask = Image.open(step4_mask_path).convert("L")
            if step4_mask.size != image.size:
                step4_mask = step4_mask.resize(image.size, Image.Resampling.NEAREST)
        except OSError:
            state.add("warn", sample_name, "step4_mask_load", f"Cannot read Step 4 mask: {step4_mask_path}")
            step4_mask = None

    warning_count = 0
    image_size = image.size
    for constraint in constraints:
        if constraint.get("reference_restored"):
            continue
        if constraint.get("bubble_idx", -1) == -1:
            continue
        red_box = constraint.get("red_box")
        if not red_box:
            continue
        left, top, right, bottom = _coerce_box(red_box, image_size)
        if right <= left or bottom <= top:
            continue
        crop = image.crop((left, top, right, bottom))
        measured_area = "the source red box"
        if step4_mask is not None:
            mask_crop = step4_mask.crop((left, top, right, bottom)).filter(ImageFilter.MaxFilter(11))
            mask_values = list(mask_crop.getdata())
            image_values = list(crop.getdata())
            total_pixels = sum(1 for value in mask_values if value > 0)
            if total_pixels > 0:
                dark_pixels = sum(
                    1
                    for pixel, mask_value in zip(image_values, mask_values)
                    if mask_value > 0 and pixel < 120
                )
                dark_fraction = dark_pixels / total_pixels
                measured_area = "the actual Step 4 cleanup mask"
            else:
                histogram = crop.histogram()
                total_pixels = sum(histogram)
                if total_pixels == 0:
                    continue
                dark_fraction = sum(histogram[:120]) / total_pixels
        else:
            histogram = crop.histogram()
            total_pixels = sum(histogram)
            if total_pixels == 0:
                continue
            dark_fraction = sum(histogram[:120]) / total_pixels
        if dark_fraction > STEP4_BUBBLE_DARK_FRACTION_WARN:
            warning_count += 1
            state.add(
                "warn",
                sample_name,
                "step4_residue_risk",
                (
                    f"Bubble region {constraint.get('id')} has dark-pixel fraction "
                    f"{dark_fraction:.3f} inside {measured_area} after inpainting."
                ),
            )

    return {"bubble_residue_warnings": warning_count}


def _crop_masked_luma_stats(image: Image.Image, mask: Image.Image) -> tuple[int, float, float]:
    pixels = image.convert("RGB").getdata()
    mask_pixels = mask.getdata()
    values: list[float] = []
    for (red, green, blue), mask_value in zip(pixels, mask_pixels):
        if mask_value > 0:
            values.append(0.299 * red + 0.587 * green + 0.114 * blue)
    if not values:
        return 0, 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return len(values), mean, variance ** 0.5


def _box_with_padding(box: tuple[int, int, int, int], image_size: tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    width, height = image_size
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


def _check_step4_floating_patch_artifacts(
    state: QualityState,
    sample_name: str,
    step4_path: Path,
    step4_mask_path: Path,
    constraints: list[dict[str, Any]],
) -> dict[str, Any]:
    if not step4_path.exists() or not step4_mask_path.exists() or not constraints:
        return {
            "floating_patch_tone_warnings": 0,
            "floating_patch_texture_warnings": 0,
        }

    try:
        image = Image.open(step4_path).convert("RGB")
        cleanup_mask = Image.open(step4_mask_path).convert("L")
    except OSError as error:
        state.add("warn", sample_name, "step4_patch_load", f"Cannot inspect Step 4 patch artifacts: {error}")
        return {
            "floating_patch_tone_warnings": 0,
            "floating_patch_texture_warnings": 0,
        }

    if cleanup_mask.size != image.size:
        cleanup_mask = cleanup_mask.resize(image.size, Image.Resampling.NEAREST)

    tone_warnings = 0
    texture_warnings = 0
    image_size = image.size
    for constraint in constraints:
        if constraint.get("bubble_idx", -1) != -1:
            continue
        if constraint.get("semantic_role") == "sfx":
            continue
        source_box = constraint.get("green_box") or constraint.get("red_box")
        if not source_box:
            continue

        box = _coerce_box(source_box, image_size)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        crop = image.crop(box)
        mask_crop = cleanup_mask.crop(box).filter(ImageFilter.MaxFilter(5))
        mask_pixels, mask_mean, mask_std = _crop_masked_luma_stats(crop, mask_crop)
        if mask_pixels < STEP4_FLOATING_PATCH_MIN_PIXELS:
            continue

        padded_box = _box_with_padding(box, image_size, STEP4_FLOATING_RING_PADDING)
        padded_crop = image.crop(padded_box)
        ring_mask = Image.new("L", padded_crop.size, 255)
        inner = (
            box[0] - padded_box[0],
            box[1] - padded_box[1],
            box[2] - padded_box[0],
            box[3] - padded_box[1],
        )
        ImageDraw.Draw(ring_mask).rectangle(inner, fill=0)
        ring_pixels, ring_mean, ring_std = _crop_masked_luma_stats(padded_crop, ring_mask)
        if ring_pixels < STEP4_FLOATING_PATCH_MIN_PIXELS:
            continue

        tone_delta = abs(mask_mean - ring_mean)
        texture_ratio = mask_std / max(ring_std, 1.0)
        region_id = constraint.get("id")

        if tone_delta > STEP4_FLOATING_TONE_DELTA_WARN:
            tone_warnings += 1
            state.add(
                "warn",
                sample_name,
                "step4_floating_tone_mismatch",
                (
                    f"Floating region {region_id} cleanup luminance differs from nearby art "
                    f"by {tone_delta:.1f}; possible grey/white patch."
                ),
            )
        if ring_std >= 10.0 and texture_ratio < STEP4_FLOATING_TEXTURE_RATIO_WARN:
            texture_warnings += 1
            state.add(
                "warn",
                sample_name,
                "step4_floating_texture_loss",
                (
                    f"Floating region {region_id} cleanup texture std ratio is {texture_ratio:.2f}; "
                    "possible over-smoothing or lost line art."
                ),
            )

    return {
        "floating_patch_tone_warnings": tone_warnings,
        "floating_patch_texture_warnings": texture_warnings,
    }


def _validate_sample(
    state: QualityState,
    samples_root: Path,
    sample_name: str,
    image_file: str,
    strict_translation: bool,
) -> None:
    sample_path = samples_root / sample_name
    original_path = sample_path / image_file
    step4_path = sample_path / "step_4_final" / "inpainted_result.jpg"
    step4_mask_path = sample_path / "step_4_final" / "mask.png"
    step6_path = sample_path / "step_6_layout" / "layout_constraints.json"
    rejected_regions_path = sample_path / "step_6_layout" / "rejected_regions.json"
    step7_path = sample_path / "step_7_translate" / "translation_results.json"
    step8_dir = sample_path / "step_8_typeset"
    step8_png_path = step8_dir / "final_output.png"
    step8_jpg_path = step8_dir / "final_output.jpg"
    step8_report_path = step8_dir / "typeset_report.json"

    summary: dict[str, Any] = {}
    required_paths = {
        "original": original_path,
        "step4_image": step4_path,
        "step4_mask": step4_mask_path,
        "step6_layout": step6_path,
        "step7_translation": step7_path,
        "step8_png": step8_png_path,
        "step8_jpg": step8_jpg_path,
        "step8_report": step8_report_path,
    }
    for label, path in required_paths.items():
        if not path.exists():
            state.add("fail", sample_name, "missing_file", f"Missing {label}: {path}")

    constraints: list[dict[str, Any]] = []
    if step6_path.exists():
        constraints = _read_json(step6_path)
        layout_ids = {
            region_id
            for item in constraints
            if "id" in item
            for region_id in [_region_id(item.get("id"))]
            if region_id is not None
        }
        summary["step6"] = {"constraints": len(constraints)}
        if len(constraints) == 0 and original_path.exists():
            state.add(
                "fail",
                sample_name,
                "no_layout_constraints",
                "Step 6 produced no layout constraints; this is a no-op output, not a successful translation.",
            )
    else:
        layout_ids = set()

    if step8_report_path.exists():
        raw_step8_report = _read_json(step8_report_path)
        if isinstance(raw_step8_report, list) and len(raw_step8_report) == 0:
            state.add(
                "fail",
                sample_name,
                "no_rendered_regions",
                "Step 8 rendered no regions; this is a no-op output, not a successful translation.",
            )
        step8_summary = _summarize_step8_report(
            state,
            sample_name,
            step8_report_path,
            strict_translation=strict_translation,
        )
        summary["step8"] = step8_summary
        rendered_ids = {
            region_id
            for item in raw_step8_report
            if "id" in item
            for region_id in [_region_id(item.get("id"))]
            if region_id is not None
        }
    else:
        rendered_ids = set()

    if step7_path.exists():
        summary["step7"] = _check_step7_translations(
            state,
            sample_name,
            step7_path,
            layout_ids,
            rendered_ids,
            strict_translation,
        )

    if rejected_regions_path.exists():
        rejected_regions = _read_json(rejected_regions_path)
        if isinstance(rejected_regions, list) and rejected_regions:
            summary["rejected_regions"] = len(rejected_regions)
            severity = "fail" if strict_translation else "warn"
            state.add(
                severity,
                sample_name,
                "unsafe_regions_rejected",
                (
                    f"{len(rejected_regions)} cloud-detected regions were not rendered "
                    "because the placement was unsafe for automatic typesetting."
                ),
            )

    _check_stale_outputs(
        state,
        sample_name,
        step8_png_path,
        [step4_path, step6_path, step7_path],
    )
    summary["step4"] = _check_step4_residue(
        state,
        sample_name,
        step4_path,
        step4_mask_path,
        constraints,
    )
    summary["step4"].update(
        _check_step4_floating_patch_artifacts(
            state,
            sample_name,
            step4_path,
            step4_mask_path,
            constraints,
        )
    )
    state.sample_summaries[sample_name] = summary


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _make_labeled_thumb(path: Path | None, label: str) -> Image.Image:
    font = _load_font(18)
    canvas = Image.new("RGB", (CONTACT_THUMB_SIZE[0], CONTACT_THUMB_SIZE[1] + 34), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width - 1, canvas.height - 1), outline=(190, 190, 190), width=1)
    draw.text((8, 7), label, fill="black", font=font)

    if path is None or not path.exists():
        draw.text((8, 48), "missing", fill=(170, 0, 0), font=font)
        return canvas

    try:
        image = Image.open(path).convert("RGB")
    except OSError:
        draw.text((8, 48), "unreadable", fill=(170, 0, 0), font=font)
        return canvas

    thumb = ImageOps.contain(image, CONTACT_THUMB_SIZE, Image.Resampling.LANCZOS)
    offset_x = (CONTACT_THUMB_SIZE[0] - thumb.width) // 2
    offset_y = 34 + (CONTACT_THUMB_SIZE[1] - thumb.height) // 2
    canvas.paste(thumb, (offset_x, offset_y))
    return canvas


def _create_contact_sheet(
    samples_root: Path,
    output_dir: Path,
    sample_map: dict[str, str] | None = None,
    contact_sheet_name: str = "contact_sheet_pipeline.jpg",
    title: str = "Pipeline QA Contact Sheet: original / Step 4 / Step 8",
) -> dict[str, str]:
    sample_map = sample_map or SAMPLE_MAP
    output_dir.mkdir(parents=True, exist_ok=True)
    font = _load_font(20)
    columns = ["original", "step4", "step8"]
    cell_width = CONTACT_THUMB_SIZE[0]
    cell_height = CONTACT_THUMB_SIZE[1] + 34
    title_height = 36
    sheet_width = cell_width * len(columns)
    sheet_height = title_height + cell_height * len(sample_map)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), title, fill="black", font=font)

    for row_index, (sample_name, image_file) in enumerate(sample_map.items()):
        sample_path = samples_root / sample_name
        paths = [
            sample_path / image_file,
            sample_path / "step_4_final" / "inpainted_result.jpg",
            sample_path / "step_8_typeset" / "final_output.png",
        ]
        for column_index, path in enumerate(paths):
            label = f"{sample_name} - {columns[column_index]}"
            thumb = _make_labeled_thumb(path, label)
            sheet.paste(thumb, (column_index * cell_width, title_height + row_index * cell_height))

    output_path = output_dir / contact_sheet_name
    sheet.save(output_path, quality=92)
    return {"pipeline_contact_sheet": str(output_path)}


def _write_reports(state: QualityState, project_root: Path, output_dir: Path, assets: dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "fail" if state.failures else "pass",
        "failures": [issue.__dict__ for issue in state.failures],
        "warnings": [issue.__dict__ for issue in state.warnings],
        "compiled_scripts": state.compiled_scripts,
        "sample_summaries": state.sample_summaries,
        "assets": assets,
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=list),
        encoding="utf-8",
    )

    lines = [
        "# Pipeline Quality Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Failures: `{len(state.failures)}`",
        f"- Warnings: `{len(state.warnings)}`",
        f"- Contact sheet: `{assets.get('pipeline_contact_sheet', 'missing')}`",
        "",
        "## Failures",
    ]
    if state.failures:
        for issue in state.failures:
            lines.append(f"- `{issue.sample}` `{issue.check}`: {issue.message}")
    else:
        lines.append("- None")

    lines.extend(["", "## Warnings"])
    if state.warnings:
        for issue in state.warnings:
            lines.append(f"- `{issue.sample}` `{issue.check}`: {issue.message}")
    else:
        lines.append("- None")

    lines.extend(["", "## Sample Summaries"])
    for sample_name, summary in state.sample_summaries.items():
        lines.append(f"- `{sample_name}`: `{json.dumps(summary, ensure_ascii=False, default=list)}`")

    (output_dir / "quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_sample_map_json(sample_map_json: Path) -> dict[str, str]:
    data = _read_json(sample_map_json)
    if isinstance(data, dict) and "samples" in data:
        samples = data["samples"]
        if isinstance(samples, list):
            return {
                str(item["sample_name"]): str(item.get("input_file", item.get("image_file", "input.png")))
                for item in samples
                if isinstance(item, dict) and item.get("sample_name")
            }
    if isinstance(data, dict):
        return {str(key): str(value) for key, value in data.items()}
    raise ValueError(f"Unsupported sample map JSON structure: {sample_map_json}")


def run_quality_gate(
    strict_translation: bool = False,
    sample_map: dict[str, str] | None = None,
    samples_root: Path | None = None,
    output_dir: Path | None = None,
    contact_sheet_name: str = "contact_sheet_pipeline.jpg",
    contact_sheet_title: str = "Pipeline QA Contact Sheet: original / Step 4 / Step 8",
) -> int:
    project_root = PROJECT_ROOT
    samples_root = samples_root or sample_root_from_env(DEFAULT_SAMPLES_ROOT)
    sample_map = sample_map or SAMPLE_MAP
    output_dir = output_dir or (project_root / "quality_reports")
    state = QualityState()

    _compile_scripts(state, project_root)

    for sample_name, image_file in sample_map.items():
        _validate_sample(state, samples_root, sample_name, image_file, strict_translation)

    assets = _create_contact_sheet(
        samples_root,
        output_dir,
        sample_map=sample_map,
        contact_sheet_name=contact_sheet_name,
        title=contact_sheet_title,
    )
    _write_reports(state, project_root, output_dir, assets)

    print("=" * 60)
    print("Pipeline quality gate")
    print("=" * 60)
    print(f"Failures: {len(state.failures)}")
    print(f"Warnings: {len(state.warnings)}")
    print(f"Report: {output_dir / 'quality_report.md'}")
    print(f"Contact sheet: {assets['pipeline_contact_sheet']}")
    return 1 if state.failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate current manga pipeline outputs.")
    parser.add_argument(
        "--strict-translation",
        action="store_true",
        help="Treat placeholder translations as hard failures instead of warnings.",
    )
    parser.add_argument(
        "--sample-map-json",
        type=Path,
        help="Optional JSON sample map or downloader manifest for non-default samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory for the generated quality report.",
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        help="Optional samples root. Useful for app-level test-run categories.",
    )
    parser.add_argument(
        "--contact-sheet-name",
        default="contact_sheet_pipeline.jpg",
        help="Filename for the generated contact sheet.",
    )
    args = parser.parse_args()
    sample_map = _load_sample_map_json(args.sample_map_json) if args.sample_map_json else None
    return run_quality_gate(
        strict_translation=args.strict_translation,
        sample_map=sample_map,
        samples_root=args.samples_root,
        output_dir=args.output_dir,
        contact_sheet_name=args.contact_sheet_name,
        contact_sheet_title="External CJK Pipeline QA: original / Step 4 / Step 8" if sample_map else "Pipeline QA Contact Sheet: original / Step 4 / Step 8",
    )


if __name__ == "__main__":
    sys.exit(main())
