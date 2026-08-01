from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
for rel in (
    "python/common",
    "python/steps",
    "python/validation",
    "python/runtime",
    "python/downloaders",
    "python/reference",
    "python/diagnostics",
):
    path = str(PROJECT_ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

import run_step4_inpaint
import run_step5_ocr
import run_step6_layout
import run_step7_translate
import run_step8_typeset
from run_quality_gate import run_quality_gate


STEP_DIRS = (
    "step_1_detect",
    "step_4_final",
    "step_5_ocr",
    "step_6_layout",
    "step_7_translate",
    "step_8_typeset",
)
# Retired pipeline stages (pre art-safe-inpaint rework). No active code reads
# or writes these; they only linger on disk from old runs. Purged alongside
# the current STEP_DIRS so stale artifacts never get mistaken for live output.
LEGACY_STEP_DIRS = (
    "step_2_cleanup",
    "step_3_reconstruct",
)
PREFERRED_INPUTS = ("input.jpg", "input.jpeg", "input.png", "input.webp")


def _default_test_run_root() -> Path:
    return PROJECT_ROOT.parents[1] / "test run"


def _default_audit_output_dir() -> Path:
    return PROJECT_ROOT / "quality_reports" / "test_run"


def _discover_sample_map(category_dir: Path) -> dict[str, str]:
    sample_map: dict[str, str] = {}
    for sample_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
        for filename in PREFERRED_INPUTS:
            if (sample_dir / filename).exists():
                sample_map[sample_dir.name] = filename
                break
    return sample_map


def _clear_generated_outputs(category_dir: Path, sample_map: dict[str, str]) -> None:
    for sample_name in sample_map:
        sample_dir = category_dir / sample_name
        for step_dir in (*STEP_DIRS, *LEGACY_STEP_DIRS):
            target = sample_dir / step_dir
            if target.exists():
                shutil.rmtree(target)


def _run_steps(sample_map: dict[str, str], category_dir: Path) -> None:
    runners: tuple[Callable[[], None], ...] = (
        lambda: run_step5_ocr.run_step5_ocr(sample_map=sample_map, samples_dir=category_dir),
        lambda: run_step6_layout.run_step6_layout(sample_map=sample_map, samples_dir=category_dir),
        lambda: run_step7_translate.run_step7_translate(sample_map=sample_map, samples_dir=category_dir),
        lambda: run_step4_inpaint.run_step4_inpaint(sample_map=sample_map, samples_dir=category_dir),
        lambda: run_step8_typeset.run_step8_typeset(sample_map=sample_map, samples_dir=category_dir),
    )
    for runner in runners:
        runner()


def _complete_count(category_dir: Path, sample_map: dict[str, str]) -> tuple[int, list[str]]:
    missing = [
        sample_name
        for sample_name in sample_map
        if not (category_dir / sample_name / "step_8_typeset" / "final_output.jpg").exists()
    ]
    return len(sample_map) - len(missing), missing


def _run_category(
    category_dir: Path,
    clear_outputs: bool,
    audit_output_dir: Path,
    run_audit: bool,
    strict_translation: bool,
    allow_api: bool,
) -> int:
    sample_map = _discover_sample_map(category_dir)
    if not sample_map:
        print(f"[test-run] skip {category_dir.name}: no input images")
        return 0

    if clear_outputs:
        _clear_generated_outputs(category_dir, sample_map)

    os.environ["LOCAL_NLLB_TRANSLATION"] = "1"
    if not allow_api:
        os.environ["USE_API_TRANSLATION"] = "0"
        os.environ["USE_API_VISION_OCR"] = "0"

    start = time.perf_counter()
    api_mode = "live" if allow_api else "blocked"
    print(f"[test-run] category={category_dir.name} samples={len(sample_map)} api={api_mode}")
    _run_steps(sample_map, category_dir)
    complete, missing = _complete_count(category_dir, sample_map)
    elapsed = time.perf_counter() - start
    print(f"[test-run] category={category_dir.name} complete={complete}/{len(sample_map)} elapsed={elapsed:.2f}s missing={missing}")

    exit_code = 0 if not missing else 1
    if run_audit:
        audit_code = run_quality_gate(
            strict_translation=strict_translation,
            sample_map=sample_map,
            samples_root=category_dir,
            output_dir=audit_output_dir / category_dir.name,
            contact_sheet_name=f"{category_dir.name}_contact_sheet.jpg",
            contact_sheet_title=f"{category_dir.name}: input / Step 4 / Step 8",
        )
        exit_code = max(exit_code, audit_code)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate the app-level test run categories with dynamic sample maps.")
    parser.add_argument("--test-run-root", type=Path, default=_default_test_run_root())
    parser.add_argument("--category", action="append", help="Category folder to run. May be passed more than once.")
    parser.add_argument("--no-clear", action="store_true", help="Keep existing step folders instead of regenerating from scratch.")
    parser.add_argument("--no-audit", action="store_true", help="Do not run the quality gate after each category.")
    parser.add_argument("--strict-translation", action="store_true")
    parser.add_argument("--allow-api", action="store_true", help="Permit live API translation/OCR fallback during validation.")
    parser.add_argument("--audit-output-dir", type=Path, default=_default_audit_output_dir())
    args = parser.parse_args()

    test_run_root = args.test_run_root.resolve()
    if not test_run_root.exists():
        raise FileNotFoundError(f"Missing test run root: {test_run_root}")

    requested = set(args.category or [])
    category_dirs = [
        path
        for path in sorted(test_run_root.iterdir())
        if path.is_dir() and (not requested or path.name in requested)
    ]
    missing_categories = requested - {path.name for path in category_dirs}
    if missing_categories:
        raise FileNotFoundError(f"Missing requested categories: {sorted(missing_categories)}")

    start = time.perf_counter()
    exit_code = 0
    for category_dir in category_dirs:
        exit_code = max(
            exit_code,
            _run_category(
                category_dir=category_dir,
                clear_outputs=not args.no_clear,
                audit_output_dir=args.audit_output_dir,
                run_audit=not args.no_audit,
                strict_translation=args.strict_translation,
                allow_api=args.allow_api,
            ),
        )
    print(f"[test-run] total_elapsed={time.perf_counter() - start:.2f}s")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
