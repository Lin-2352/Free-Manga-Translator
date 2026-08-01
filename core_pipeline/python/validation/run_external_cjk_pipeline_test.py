"""
Run the external legal CJK sample set through the same local pipeline used by
the curated original samples.

This runner deliberately does not use cloud-detected geometry. Step 5 regenerates
local detection/OCR artifacts, Step 6 owns layout geometry, Step 7 applies the
local translator gate, and Step 4/Step 8 render only locally accepted regions.
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

from pipeline_paths import EXTERNAL_CJK_ROOT, PROJECT_ROOT


MANIFEST_PATH = PROJECT_ROOT / "external_cjk_samples_manifest.json"
SAMPLES_ROOT = EXTERNAL_CJK_ROOT
PIPELINE_OUTPUT_FOLDERS = [
    "step_1_detect",
    "step_4_final",
    "step_5_ocr",
    "step_6_layout",
    "step_7_translate",
    "step_8_typeset",
]


def _load_external_map(
    manifest_path: Path = MANIFEST_PATH,
    sample_names: set[str] | None = None,
) -> dict[str, str]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run download_external_cjk_samples.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_map = {
        str(item["sample_name"]): str(item.get("input_file", "input.jpg"))
        for item in manifest.get("samples", [])
    }
    if sample_names:
        missing = sorted(sample_names - set(sample_map))
        if missing:
            raise ValueError(f"Unknown external sample(s): {', '.join(missing)}")
        sample_map = {name: sample_map[name] for name in sample_map if name in sample_names}
    return sample_map


def _clear_outputs(
    sample_map: dict[str, str],
    keep_step1: bool = False,
    keep_step5: bool = False,
) -> None:
    for sample_name in sample_map:
        sample_path = SAMPLES_ROOT / sample_name
        for folder in PIPELINE_OUTPUT_FOLDERS:
            if keep_step1 and folder == "step_1_detect":
                continue
            if keep_step5 and folder == "step_5_ocr":
                continue
            target = sample_path / folder
            if target.exists():
                shutil.rmtree(target)


def _patch_sample_maps(sample_map: dict[str, str]) -> None:
    import ml_region_lib
    import run_step4_inpaint
    import run_step5_ocr
    import run_step6_layout
    import run_step7_translate
    import run_step8_typeset

    for module in [
        ml_region_lib,
        run_step4_inpaint,
        run_step5_ocr,
        run_step6_layout,
        run_step7_translate,
        run_step8_typeset,
    ]:
        module.SAMPLE_MAP = sample_map


def _run_local_pipeline_steps(
    resume_existing_ocr: bool = False,
    skip_step5: bool = False,
) -> None:
    import run_step4_inpaint
    import run_step5_ocr
    import run_step6_layout
    import run_step7_translate
    import run_step8_typeset

    previous_resume = os.environ.get("PIPELINE_RESUME_EXISTING_OCR")
    if resume_existing_ocr:
        os.environ["PIPELINE_RESUME_EXISTING_OCR"] = "1"
    else:
        os.environ.pop("PIPELINE_RESUME_EXISTING_OCR", None)
    try:
        if not skip_step5:
            run_step5_ocr.run_step5_ocr()
    finally:
        if previous_resume is None:
            os.environ.pop("PIPELINE_RESUME_EXISTING_OCR", None)
        else:
            os.environ["PIPELINE_RESUME_EXISTING_OCR"] = previous_resume
    run_step6_layout.run_step6_layout()
    run_step7_translate.run_step7_translate()
    run_step4_inpaint.run_step4_inpaint()
    run_step8_typeset.run_step8_typeset()


def _run_quality_report(sample_map: dict[str, str], strict_translation: bool) -> int:
    os.environ["PIPELINE_SAMPLES_ROOT"] = str(SAMPLES_ROOT)
    import run_quality_gate

    return run_quality_gate.run_quality_gate(
        strict_translation=strict_translation,
        sample_map=sample_map,
        output_dir=PROJECT_ROOT / "quality_reports" / "external_cjk",
        contact_sheet_name="contact_sheet_external_cjk.jpg",
        contact_sheet_title="External CJK Pipeline QA: original / Step 4 / Step 8",
    )


def run_external_cjk_pipeline(
    clear_outputs: bool = True,
    strict_translation: bool = True,
    sample_names: set[str] | None = None,
    resume_existing_ocr: bool = False,
    skip_step5: bool = False,
) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sample_map = _load_external_map(sample_names=sample_names)
    if not sample_map:
        raise RuntimeError("External sample manifest does not contain any samples.")
    SAMPLES_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["PIPELINE_SAMPLES_ROOT"] = str(SAMPLES_ROOT)
    os.environ["LOCAL_NLLB_TRANSLATION"] = "1"

    if clear_outputs:
        _clear_outputs(
            sample_map,
            keep_step1=resume_existing_ocr or skip_step5,
            keep_step5=resume_existing_ocr or skip_step5,
        )

    _patch_sample_maps(sample_map)
    _run_local_pipeline_steps(
        resume_existing_ocr=resume_existing_ocr,
        skip_step5=skip_step5,
    )
    return _run_quality_report(sample_map, strict_translation=strict_translation)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run external legal CJK pipeline validation.")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete existing external sample run folders before processing.",
    )
    parser.add_argument(
        "--strict-translation",
        action="store_true",
        help="Deprecated; external validation is strict by default.",
    )
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Do not fail the quality report for placeholder translations.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Run only one external sample. Repeat for multiple samples.",
    )
    parser.add_argument(
        "--resume-existing-ocr",
        action="store_true",
        help="Keep existing Step 5 OCR artifacts and only OCR samples without them.",
    )
    parser.add_argument(
        "--skip-step5",
        action="store_true",
        help="Run Step 6 onward from existing Step 5 OCR artifacts.",
    )
    args = parser.parse_args()
    return run_external_cjk_pipeline(
        clear_outputs=not args.keep_existing,
        strict_translation=not args.allow_placeholders,
        sample_names=set(args.sample) or None,
        resume_existing_ocr=args.resume_existing_ocr,
        skip_step5=args.skip_step5,
    )


if __name__ == "__main__":
    raise SystemExit(main())
