"""C-0 invariant for the bubble-interior damage plan (see the plan at
C:\\Users\\Harsh Raj\\.claude\\plans\\8-tasks-1-done-temporal-shell.md).

Binary, non-visual gate: every region step 7 translates AND step 6 KEPT must have >0% of its
pixels changed by step 4 inside its box.

SCOPE CORRECTED 2026-07-30: step 7 reads step_5_ocr/ocr_results.json directly
(run_step7_translate.py:1420), not step 6's kept constraints -- so step 7 translates every OCR
item, including ones step 6 rejected or silently dropped. Step 4 only ever cleans what step 6
KEPT. The earlier version of this gate asked "was every translated region cleaned?" over ALL of
step 7's output, which is unanswerable-by-construction for anything step 6 didn't keep -- of a
baseline's 12 violations under the old scope, 8 were step-6 silent drops (a real but SEPARATE
defect, task #191) and 3 were correctly-rejected floating_too_large items, leaving exactly 1
genuine violation. Scoping to step-6-kept ids makes this gate ask the question it can actually
answer, and the PRESERVED_ROLES allowlist below (which the kept-id restriction now makes mostly
redundant, not wrong) still holds for edge cases.

WHAT A SURVIVING VIOLATION MEANS (verified 2026-07-27 by classifying every violation against
step 6's kept/rejected sets and crop-zooming the step-8 output): step 7 produced a translation
for a region step 6 KEPT, but step 4 never cleaned it -- so step 8 has no clean interior to
render onto and drops it. The user-visible result is an UNTRANSLATED bubble, not English
stamped over Japanese ink; of the violations measured across the 34-sample suite, zero produced
overlapping double text.

Two consequences, so this script is not over-read:
  * It is a missed-translation detector, not a ghost-text detector.
  * It is structurally blind to fill-quality damage: a bubble whose wall or interior art is
    destroyed by the fill still shows >0% pixels changed and therefore PASSES. Use per-bubble
    crop-zoom for fill quality; use this only as a regression tripwire on rendering coverage.

Regions whose id appears in rejected_layout_items.json under an intentionally-preserved
semantic_role are allowlisted -- they are NEVER supposed to be erased (sfx_artwork,
classification_sfx, title_logo, credit; see run_step6_layout.py:3130-3139). Flagging those would
repeat the manga-semantic-detector-sfx-false-positives scar (new_sample_13_(chi) id=3 is exactly
this case and must never be flagged). This mostly can't fire anymore now that non-kept ids are
out of scope entirely, but is kept in case a kept constraint is ever retroactively tagged with a
preserved role.

Windows console safety: never print raw CJK -- str.encode('ascii', 'replace') for any
user-facing text field, per a documented project footgun.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_common import GateCoverage, is_renderable_translation, iter_baseline_samples, load_manifest  # noqa: E402

PRESERVED_ROLES = {"sfx_artwork", "classification_sfx", "title_logo", "credit"}

# Same "meaningfully changed" pixel threshold the earlier RC-1 measurement used.
CHANGE_THRESHOLD = 25

REQUIRED = (
    "step_4_final/inpainted_result.jpg",
    "step_6_layout/layout_constraints.json",
    "step_7_translate/translation_results.json",
)


def _safe(s: str | None) -> str:
    return (s or "").encode("ascii", "replace").decode("ascii")


def _box_from_translation_item(item: dict) -> tuple[int, int, int, int] | None:
    box = item.get("box")
    if not isinstance(box, dict):
        return None
    try:
        return int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
    except (KeyError, TypeError, ValueError):
        return None


def check_sample(sample_dir: Path) -> list[dict]:
    source_path = sample_dir / "input.jpg"
    if not source_path.exists():
        source_path = next(sample_dir.glob("input.*"), None)
    output_path = sample_dir / "step_4_final" / "inpainted_result.jpg"
    translations_path = sample_dir / "step_7_translate" / "translation_results.json"
    rejected_path = sample_dir / "step_6_layout" / "rejected_layout_items.json"
    constraints_path = sample_dir / "step_6_layout" / "layout_constraints.json"

    if source_path is None or not source_path.exists():
        return []

    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    output = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    if source is None or output is None:
        raise OSError(f"cv2.imread failed on required file(s) under {sample_dir}")
    h, w = source.shape[:2]

    preserved_ids: set[str] = set()
    if rejected_path.exists():
        rejected = json.loads(rejected_path.read_text(encoding="utf-8"))
        for r in rejected:
            # `reason` is the actual rejection cause (e.g. "classification_sfx"); `semantic_role`
            # is frequently just "dialogue" even for an SFX reject, so reason must be checked
            # first -- checking semantic_role first silently defeats the SFX/title/credit
            # allowlist (measured: 26 of 33 baseline "violations" were this bug, not real gaps).
            role = r.get("reason") or r.get("semantic_role")
            if role in PRESERVED_ROLES:
                preserved_ids.add(str(r.get("id")))

    constraints = json.loads(constraints_path.read_text(encoding="utf-8"))
    kept_ids: set[str] = set()
    for c in constraints:
        cid = str(c.get("id"))
        kept_ids.add(cid)
        if c.get("semantic_role") in PRESERVED_ROLES:
            preserved_ids.add(cid)

    translations = json.loads(translations_path.read_text(encoding="utf-8"))
    results = []
    for item in translations:
        en_text = (item.get("en_text") or "").strip()
        if not is_renderable_translation(en_text):
            # Not just "empty" -- run_step4_inpaint.py never cleans, and step 8 never
            # renders, a translation with zero alphanumeric content ("...") in the first
            # place (see is_renderable_translation's docstring). Flagging it here would
            # be re-discovering intentional pipeline behavior as a fresh violation.
            continue
        item_id = str(item.get("id"))
        if item_id not in kept_ids:
            # Out of scope: step 7 translates every step-5 OCR item regardless of what step 6
            # decided, so an id absent from layout_constraints.json was never something step 4
            # was asked to clean. Rejected-for-a-reason and silently-dropped ids are both here --
            # the silent-drop case is task #191, a real defect this gate cannot see or should.
            continue
        box = _box_from_translation_item(item)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue

        src_crop = source[y1:y2, x1:x2]
        out_crop = output[y1:y2, x1:x2]
        diff = cv2.absdiff(src_crop, out_crop).max(axis=2)
        changed_pct = float(np.mean(diff > CHANGE_THRESHOLD)) * 100.0

        allowlisted = item_id in preserved_ids
        violated = changed_pct <= 0.0 and not allowlisted

        results.append({
            "id": item_id,
            "box": [x1, y1, x2, y2],
            "changed_pct": round(changed_pct, 2),
            "allowlisted": allowlisted,
            "en_preview": _safe(en_text[:40]),
            "violated": violated,
        })
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("test_run_root", type=Path, nargs="?", default=None,
                     help="Legacy mode: a directory of group/sample/step_N_* trees "
                          "(e.g. 'test run latest pipeline').")
    ap.add_argument("--baseline-manifest", type=Path, default=None,
                     help="P0-3 mode: a baseline dir with manifest.json (see gate_common.py) "
                          "-- resolves each sample to its live runtime_samples/extension/ dir "
                          "instead of a fixed test-run tree.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if bool(args.test_run_root) == bool(args.baseline_manifest):
        ap.error("pass exactly one of test_run_root or --baseline-manifest")

    all_results: dict[str, list[dict]] = {}

    if args.test_run_root:
        root: Path = args.test_run_root
        sample_dirs: list[tuple[str, Path]] = []
        for group_dir in sorted(root.iterdir()):
            if not group_dir.is_dir() or group_dir.name.startswith("_"):
                continue
            for sample_dir in sorted(group_dir.iterdir()):
                if sample_dir.is_dir():
                    sample_dirs.append((f"{group_dir.name}/{sample_dir.name}", sample_dir))

        if not sample_dirs:
            # A GateCoverage(name, 0) trivially reports failed()=False (nothing declared,
            # nothing missing) -- correct for "this baseline legitimately has zero rows",
            # wrong here: an empty/bad test_run_root path is a usage error, not a clean run.
            print(f"ERROR: no group/sample directories found under {root}")
            return 1

        coverage = GateCoverage("G1 translated-regions-cleaned", len(sample_dirs))
        for group_sample, sample_dir in sample_dirs:
            missing = [rel for rel in REQUIRED if not (sample_dir / rel).exists()]
            if missing:
                coverage.record_incomplete(group_sample, missing)
                continue
            try:
                checked = check_sample(sample_dir)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  UNREADABLE {group_sample}: {e}")
                coverage.record_unreadable(group_sample)
                continue
            coverage.record_checked()
            if checked:
                all_results[group_sample] = checked
        out_path = args.out or (root / "translated_regions_cleaned_check.json")
    else:
        manifest_total = len(load_manifest(args.baseline_manifest))
        coverage = GateCoverage("G1 translated-regions-cleaned", manifest_total)
        for group_sample, entry, runtime_dir, missing in iter_baseline_samples(args.baseline_manifest, REQUIRED):
            if runtime_dir is None:
                coverage.record_missing(group_sample)
                continue
            if missing:
                coverage.record_incomplete(group_sample, missing)
                continue
            try:
                checked = check_sample(runtime_dir)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  UNREADABLE {group_sample}: {e}")
                coverage.record_unreadable(group_sample)
                continue
            coverage.record_checked()
            if checked:
                all_results[group_sample] = checked
        out_path = args.out or (args.baseline_manifest / "g1_translated_regions_cleaned.json")

    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    total = 0
    violations = 0
    allowlisted_count = 0
    print(f"G1 translated-regions-cleaned: results -> {out_path}")
    for sample, items in all_results.items():
        for it in items:
            total += 1
            if it["allowlisted"]:
                allowlisted_count += 1
            if it["violated"]:
                violations += 1
                print(f"  VIOLATION {sample} id={it['id']} changed_pct={it['changed_pct']} "
                      f"box={it['box']} en={it['en_preview']!r}")
    print(f"total translated regions checked: {total}")
    print(f"allowlisted (preserved art/SFX/title/credit): {allowlisted_count}")
    print(f"VIOLATIONS (translated but never cleaned): {violations}")
    coverage.report()
    return 1 if (violations > 0 or coverage.failed()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
