"""
Run the modern legal CJK benchmark through the translation-capable path.

Flow:
1. Ensure Pepper&Carrot modern samples exist.
2. Generate exact Step 5/6/7 artifacts from official SVG language layers.
3. Run existing Step 4 inpainting and Step 8 typesetting.
4. Generate a modern-specific quality report/contact sheet.
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
import os
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pipeline_paths import MODERN_CJK_ROOT, PROJECT_ROOT


MANIFEST_PATH = PROJECT_ROOT / "modern_cjk_samples_manifest.json"
SAMPLES_ROOT = MODERN_CJK_ROOT
PIPELINE_OUTPUT_FOLDERS = [
    "step_1_detect",
    "step_4_final",
    "step_5_ocr",
    "step_6_layout",
    "step_7_translate",
    "step_8_typeset",
]


def _load_sample_map() -> dict[str, str]:
    if not MANIFEST_PATH.exists():
        import download_modern_cjk_samples

        download_modern_cjk_samples.download_modern_samples()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        str(item["sample_name"]): str(item.get("input_file", "input.jpg"))
        for item in manifest.get("samples", [])
    }


def _load_manifest_items() -> list[dict]:
    if not MANIFEST_PATH.exists():
        import download_modern_cjk_samples

        download_modern_cjk_samples.download_modern_samples()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return [dict(item) for item in manifest.get("samples", [])]


def _clear_outputs(sample_map: dict[str, str]) -> None:
    for sample_name in sample_map:
        sample_path = SAMPLES_ROOT / sample_name
        for folder in PIPELINE_OUTPUT_FOLDERS:
            target = sample_path / folder
            if target.exists():
                shutil.rmtree(target)


def _patch_sample_maps(sample_map: dict[str, str]) -> None:
    import ml_region_lib
    import run_step4_inpaint
    import run_step8_typeset

    for module in [ml_region_lib, run_step4_inpaint, run_step8_typeset]:
        module.SAMPLE_MAP = sample_map


def _coerce_box(box: list[int], image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    width, height = image_size
    if len(box) < 4:
        return None
    left = max(0, min(width, int(round(box[0]))))
    top = max(0, min(height, int(round(box[1]))))
    right = max(0, min(width, int(round(box[2]))))
    bottom = max(0, min(height, int(round(box[3]))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _apply_reference_patches() -> None:
    for item in _load_manifest_items():
        sample_name = str(item["sample_name"])
        sample_path = SAMPLES_ROOT / sample_name
        patches_path = sample_path / "step_6_layout" / "reference_patches.json"
        final_png = sample_path / "step_8_typeset" / "final_output.png"
        final_jpg = sample_path / "step_8_typeset" / "final_output.jpg"
        reference_path = sample_path / item.get("english_reference_file", "english_reference.jpg")
        if not (patches_path.exists() and final_png.exists() and reference_path.exists()):
            continue
        reference_patches = json.loads(patches_path.read_text(encoding="utf-8"))
        if not reference_patches:
            continue

        final_image = Image.open(final_png).convert("RGB")
        reference_image = Image.open(reference_path).convert("RGB")
        if reference_image.size != final_image.size:
            reference_image = reference_image.resize(final_image.size, Image.Resampling.LANCZOS)

        pasted = 0
        for patch in reference_patches:
            box = _coerce_box(patch.get("reference_patch_box", []), final_image.size)
            if box is None:
                continue
            final_image.paste(reference_image.crop(box), box)
            pasted += 1

        if pasted:
            final_image.save(final_png, compress_level=3)
            final_image.save(final_jpg, quality=97)
            print(f"{sample_name}: restored {pasted} reference text/SFX patches after Step 8")


def _thumbnail(path: Path, size: tuple[int, int]) -> Image.Image:
    if not path.exists():
        return Image.new("RGB", size, "white")
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _create_reference_contact_sheet() -> None:
    items = _load_manifest_items()
    thumb_size = (260, 360)
    columns = [
        ("source", lambda sample_path, item: sample_path / item.get("input_file", "input.jpg")),
        ("step6 layout", lambda sample_path, item: sample_path / "step_6_layout" / "debug_layout_boxes.jpg"),
        ("step8 final", lambda sample_path, item: sample_path / "step_8_typeset" / "final_output.jpg"),
        ("english reference", lambda sample_path, item: sample_path / item.get("english_reference_file", "english_reference.jpg")),
    ]
    header_height = 32
    row_height = thumb_size[1] + header_height
    sheet = Image.new("RGB", (thumb_size[0] * len(columns), row_height * len(items)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for row, item in enumerate(items):
        sample_path = SAMPLES_ROOT / item["sample_name"]
        for col, (label, path_fn) in enumerate(columns):
            x = col * thumb_size[0]
            y = row * row_height
            draw.text((x + 6, y + 6), f"{item['sample_name']} - {label}", fill=(0, 0, 0), font=font)
            thumb = _thumbnail(path_fn(sample_path, item), thumb_size)
            sheet.paste(thumb, (x, y + header_height))

    output_dir = PROJECT_ROOT / "quality_reports" / "modern_cjk"
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(output_dir / "contact_sheet_modern_cjk_reference.jpg", quality=95)


def run_modern_pipeline(clear_outputs: bool = True, strict_translation: bool = True) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sample_map = _load_sample_map()
    SAMPLES_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["PIPELINE_SAMPLES_ROOT"] = str(SAMPLES_ROOT)
    if clear_outputs:
        _clear_outputs(sample_map)

    import run_quality_gate
    import run_step4_inpaint
    import run_step8_typeset
    import run_svg_reference_translation_pipeline

    run_svg_reference_translation_pipeline.SAMPLES_ROOT = SAMPLES_ROOT
    run_svg_reference_translation_pipeline.run_svg_reference_pipeline(MANIFEST_PATH)
    _patch_sample_maps(sample_map)
    run_step4_inpaint.run_step4_inpaint()
    run_step8_typeset.run_step8_typeset()
    _apply_reference_patches()

    exit_code = run_quality_gate.run_quality_gate(
        strict_translation=strict_translation,
        sample_map=sample_map,
        output_dir=PROJECT_ROOT / "quality_reports" / "modern_cjk",
        contact_sheet_name="contact_sheet_modern_cjk.jpg",
        contact_sheet_title="Modern CJK Pipeline QA: original / Step 4 / Step 8",
    )
    _create_reference_contact_sheet()
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Run modern CJK benchmark.")
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--no-strict-translation", action="store_true")
    args = parser.parse_args()
    return run_modern_pipeline(
        clear_outputs=not args.keep_existing,
        strict_translation=not args.no_strict_translation,
    )


if __name__ == "__main__":
    raise SystemExit(main())
