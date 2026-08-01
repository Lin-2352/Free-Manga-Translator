from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_quality_gate import run_quality_gate


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
PREFERRED_INPUTS = ("input.jpg", "input.jpeg", "input.png", "input.webp")


def _default_test_run_root() -> Path:
    core_root = Path(__file__).resolve().parents[2]
    return core_root.parents[1] / "test run"


def _find_input_file(sample_dir: Path) -> str | None:
    for filename in PREFERRED_INPUTS:
        if (sample_dir / filename).exists():
            return filename
    for path in sorted(sample_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path.name
    return None


def _discover_sample_map(category_dir: Path) -> dict[str, str]:
    sample_map: dict[str, str] = {}
    for sample_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
        input_file = _find_input_file(sample_dir)
        if input_file:
            sample_map[sample_dir.name] = input_file
    return sample_map


def _write_index_report(output_dir: Path, category_results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    status = "fail" if any(item["exit_code"] for item in category_results) else "pass"
    index = {
        "status": status,
        "categories": category_results,
    }
    (output_dir / "test_run_quality_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Test Run Quality Audit",
        "",
        f"- Status: `{status}`",
        f"- Categories: `{len(category_results)}`",
        "",
        "## Category Reports",
    ]
    for result in category_results:
        lines.append(
            f"- `{result['category']}`: exit `{result['exit_code']}`, "
            f"samples `{result['samples']}`, report `{result['report']}`"
        )
    (output_dir / "test_run_quality_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the quality gate across every category in the app-level test run folder.")
    parser.add_argument("--test-run-root", type=Path, default=_default_test_run_root())
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "quality_reports" / "test_run")
    parser.add_argument("--strict-translation", action="store_true")
    args = parser.parse_args()

    test_run_root = args.test_run_root.resolve()
    if not test_run_root.exists():
        raise FileNotFoundError(f"Test run root not found: {test_run_root}")

    category_results: list[dict[str, Any]] = []
    for category_dir in sorted(path for path in test_run_root.iterdir() if path.is_dir()):
        sample_map = _discover_sample_map(category_dir)
        if not sample_map:
            continue
        category_output_dir = args.output_dir / category_dir.name
        previous_root = os.environ.get("PIPELINE_SAMPLES_ROOT")
        os.environ["PIPELINE_SAMPLES_ROOT"] = str(category_dir)
        try:
            exit_code = run_quality_gate(
                strict_translation=args.strict_translation,
                sample_map=sample_map,
                output_dir=category_output_dir,
                contact_sheet_name=f"{category_dir.name}_contact_sheet.jpg",
                contact_sheet_title=f"{category_dir.name}: input / Step 4 / Step 8",
            )
        finally:
            if previous_root is None:
                os.environ.pop("PIPELINE_SAMPLES_ROOT", None)
            else:
                os.environ["PIPELINE_SAMPLES_ROOT"] = previous_root
        category_results.append(
            {
                "category": category_dir.name,
                "samples": len(sample_map),
                "exit_code": exit_code,
                "report": str(category_output_dir / "quality_report.md"),
                "contact_sheet": str(category_output_dir / f"{category_dir.name}_contact_sheet.jpg"),
            }
        )

    _write_index_report(args.output_dir, category_results)
    return 1 if any(item["exit_code"] for item in category_results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
