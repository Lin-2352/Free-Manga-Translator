"""G3 -- art-integrity gate (P0-3).

Catches the burst-bubble class of damage (the 1537px destroyed-art
regression this session diagnosed and fixed in 70be81b): cleanup/inpaint
spilling OUTSIDE every region step 8 was actually allowed to touch. Diffs
the final rendered page against the ORIGINAL posted bytes (not the runtime
tree's input.jpg, which is a requantized JPEG re-encode -- see
run_extension_pipeline_server.py:758 -- and would measure JPEG artifacts on
every pixel of every page, not real damage), outside the union of every
typeset region's box (plus a small margin for legitimate anti-aliasing at
box edges).

Uses the P0-2-style baseline's own archived rendered.png and the manifest's
group/sample mapping back to `test run latest pipeline/<group>/<sample>/
input.*` for the original bytes, since the live runtime tree's stage
folders get wiped by ANY subsequent /v1/cache/clear call (global, not
per-sample -- confirmed while building G4) and can't be trusted to still
hold this sample's artifacts by the time this gate runs.

Threshold is not asserted from theory -- it's set from the actual
distribution of outside-box changed-pixel percentage across the current
34-sample P0-2 baseline (see --calibrate). Measured 2026-07-28: range
0.02%-5.98%, median ~0.5%. The ceiling (original/sample2 at 5.98%,
external/external_ko_1 at 5.76%) correlates with ALREADY-TRACKED,
pre-existing residual wall/border erosion (task #206, not yet fixed) --
not a new defect this gate is discovering. The default limit is set with
headroom above that ceiling (8%) so this gate catches a genuinely new,
large-scale regression (the class of damage 70be81b fixed, where cleanup
spilled far outside its intended region) without permanently failing on
already-known-and-tracked minor residue. This is a coarse tripwire, not a
zero-tolerance check -- #206's fix should tighten this limit once landed,
not the other way around.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_common import GateCoverage, SAMPLES_ROOT, load_manifest, safe  # noqa: E402

CHANGE_THRESHOLD = 25  # same "meaningfully changed" per-pixel diff as G1 (check_translated_regions_cleaned.py)
BOX_MARGIN = 4  # px, in rendered-image space -- absorbs legitimate outline/anti-aliasing bleed at box edges
DEFAULT_OUTSIDE_BOX_LIMIT_PCT = 8.0  # set from --calibrate against the P0-2 baseline, see docstring

REQUIRED = ("rendered.png", "step_8_typeset/typeset_report.json")


def _typeset_boxes(typeset_report: list[dict]) -> list[tuple[int, int, int, int]]:
    boxes = []
    for item in typeset_report:
        pos = item.get("position")
        if isinstance(pos, list) and len(pos) == 4:
            boxes.append(tuple(int(v) for v in pos))
    return boxes


def score_sample(group: str, sample: str, baseline_dir: Path) -> dict:
    rendered_path = baseline_dir / group / sample / "rendered.png"
    typeset_path = baseline_dir / group / sample / "step_8_typeset" / "typeset_report.json"
    source_matches = glob.glob(str(SAMPLES_ROOT / group / sample / "input.*"))

    if not source_matches:
        raise FileNotFoundError(f"no source input.* found for {group}/{sample} under {SAMPLES_ROOT}")

    rendered = cv2.imread(str(rendered_path), cv2.IMREAD_COLOR)
    source = cv2.imread(source_matches[0], cv2.IMREAD_COLOR)
    if rendered is None or source is None:
        raise OSError(f"cv2.imread failed for {group}/{sample}")

    rh, rw = rendered.shape[:2]
    sh, sw = source.shape[:2]
    if (sh, sw) != (rh, rw):
        source = cv2.resize(source, (rw, rh), interpolation=cv2.INTER_LINEAR)

    typeset_report = json.loads(typeset_path.read_text(encoding="utf-8"))
    boxes = _typeset_boxes(typeset_report)

    outside_mask = np.ones((rh, rw), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, x1 - BOX_MARGIN)
        y1 = max(0, y1 - BOX_MARGIN)
        x2 = min(rw, x2 + BOX_MARGIN)
        y2 = min(rh, y2 + BOX_MARGIN)
        outside_mask[y1:y2, x1:x2] = False

    diff = cv2.absdiff(source, rendered).max(axis=2)
    changed = diff > CHANGE_THRESHOLD
    outside_changed = changed & outside_mask

    outside_pixel_count = int(np.count_nonzero(outside_mask))
    changed_pct = (
        float(np.count_nonzero(outside_changed)) / outside_pixel_count * 100.0
        if outside_pixel_count > 0 else 0.0
    )

    return {
        "outside_box_changed_pct": round(changed_pct, 3),
        "outside_box_pixel_count": outside_pixel_count,
        "typeset_box_count": len(boxes),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit-pct", type=float, default=DEFAULT_OUTSIDE_BOX_LIMIT_PCT)
    ap.add_argument("--calibrate", action="store_true",
                     help="Print the full distribution instead of gating, to set --limit-pct.")
    args = ap.parse_args()

    manifest = load_manifest(args.baseline_dir)
    coverage = GateCoverage("G3 art-integrity", len(manifest))
    all_results: dict[str, dict] = {}
    for group_sample in sorted(manifest):
        group, sample = group_sample.split("/", 1)
        sample_dir = args.baseline_dir / group / sample
        missing = [rel for rel in REQUIRED if not (sample_dir / rel).exists()]
        if not sample_dir.exists():
            coverage.record_missing(group_sample)
            continue
        if missing:
            coverage.record_incomplete(group_sample, missing)
            continue
        try:
            result = score_sample(group, sample, args.baseline_dir)
        except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
            print(f"  UNREADABLE {group_sample}: {e}")
            coverage.record_unreadable(group_sample)
            continue
        coverage.record_checked()
        all_results[group_sample] = result

    out_path = args.out or (args.baseline_dir / "g3_art_integrity.json")
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    pcts = sorted((v["outside_box_changed_pct"], k) for k, v in all_results.items())
    print(f"G3 art-integrity: results -> {out_path}")

    if args.calibrate:
        print("Full distribution (outside-box changed %), lowest to highest:")
        for pct, k in pcts:
            print(f"  {pct:7.3f}%  {k}")
        coverage.report()
        return 0

    violations = [(pct, k) for pct, k in pcts if pct > args.limit_pct]
    for pct, k in violations:
        print(f"  VIOLATION {k}: {pct}% of outside-box pixels changed (limit {args.limit_pct}%)")
    print(f"limit: {args.limit_pct}%, violations: {len(violations)}/{len(all_results)}")
    coverage.report()
    return 1 if (violations or coverage.failed()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
