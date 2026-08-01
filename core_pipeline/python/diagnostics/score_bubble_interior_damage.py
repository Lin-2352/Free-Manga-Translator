"""B-0 metric scorer for the bubble-interior inpainting bake-off (see the plan at
C:\\Users\\Harsh Raj\\.claude\\plans\\8-tasks-1-done-temporal-shell.md).

Scores each bubble/floating-dialogue constraint in a sample's step_6_layout output against
its step_4_final/inpainted_result.jpg, using ONLY the pristine input.jpg as ground truth for
what "clean" looks like. Three metrics per constraint:

  residual_ink_pct   -- how much dark ink is still inside the region after cleanup
  outline_integrity  -- edge-pixel retention along the bubble wall vs the same ring in source
  interior_smudge     -- luma_lift (median inside minus median of the SAME region in source)
                          plus a std-ratio check, adapted from step 8's own damage detector
                          _reconstruction_left_blob (run_step8_typeset.py ~:836-869)

Deliberately does NOT read green_polygon as the wall (it is a Voronoi slice, not the wall --
see step-6 exploration findings). Uses green_box eroded by a small margin as the interior
sample region, and a thin ring just inside the green_box edge as the "wall" proxy for
outline_integrity, since a faithful refined_bubble_outline is not present on the pages this
scorer was built to catch damage on.

Windows console safety: this script must never print raw CJK text -- str.encode('ascii',
'replace') is used for any user-facing text field, per a documented project footgun.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_common import GateCoverage, iter_baseline_samples, load_manifest  # noqa: E402

REQUIRED = (
    "input.jpg",
    "step_4_final/inpainted_result.jpg",
    "step_6_layout/layout_constraints.json",
)


def _safe(s: str | None) -> str:
    return (s or "").encode("ascii", "replace").decode("ascii")


def _load_gray(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


def score_constraint(source_bgr: np.ndarray, output_bgr: np.ndarray, constraint: dict) -> dict:
    h, w = source_bgr.shape[:2]
    gx1, gy1, gx2, gy2 = constraint.get("green_box") or constraint.get("red_box")
    gx1, gy1 = max(0, gx1), max(0, gy1)
    gx2, gy2 = min(w, gx2), min(h, gy2)
    if gx2 - gx1 < 6 or gy2 - gy1 < 6:
        return {"id": constraint.get("id"), "skipped": "region_too_small"}

    # Interior sample: erode the box inward by ~12% of the min side so we sample the true
    # interior, not the wall itself.
    margin = max(2, int(min(gx2 - gx1, gy2 - gy1) * 0.12))
    ix1, iy1 = gx1 + margin, gy1 + margin
    ix2, iy2 = gx2 - margin, gy2 - margin
    if ix2 - ix1 < 4 or iy2 - iy1 < 4:
        ix1, iy1, ix2, iy2 = gx1, gy1, gx2, gy2

    out_interior = output_bgr[iy1:iy2, ix1:ix2]
    out_gray = cv2.cvtColor(out_interior, cv2.COLOR_BGR2GRAY)

    # residual ink: dark pixels remaining in the OUTPUT interior
    residual_ink_pct = float(np.mean(out_gray < 100)) * 100.0

    # Interior smudge -- NOT a same-region source/output comparison. The source interior
    # legitimately contains ink (the original dialogue), so its std/median are not a valid
    # "clean" baseline -- ANY cleanup, good or bad, lowers std relative to a text-filled
    # source. The real bleed signature is texture leaking IN FROM OUTSIDE: compare the
    # OUTPUT interior against a ring of SOURCE pixels just OUTSIDE the green_box (the
    # surrounding art/screentone). If the output interior's edge density and std resemble
    # the exterior's texture rather than a flat fill, that is outside art bleeding inside.
    ext_pad = max(6, margin)
    ex1, ey1 = max(0, gx1 - ext_pad), max(0, gy1 - ext_pad)
    ex2, ey2 = min(w, gx2 + ext_pad), min(h, gy2 + ext_pad)
    ext_window = source_bgr[ey1:ey2, ex1:ex2]
    ext_mask = np.ones(ext_window.shape[:2], dtype=bool)
    # blank out the box itself (offset into the window's local coords)
    bx1, by1 = gx1 - ex1, gy1 - ey1
    bx2, by2 = gx2 - ex1, gy2 - ey1
    ext_mask[max(0, by1):max(0, by2), max(0, bx1):max(0, bx2)] = False
    ext_gray_full = cv2.cvtColor(ext_window, cv2.COLOR_BGR2GRAY)
    if int(np.count_nonzero(ext_mask)) < 30:
        exterior_std = None
        exterior_edge_density = None
        luma_lift = None
        smudge_flag = False
        std_ratio = None
    else:
        exterior_vals = ext_gray_full[ext_mask]
        exterior_std = float(np.std(exterior_vals))
        exterior_edges = cv2.Canny(ext_gray_full, 45, 135) > 0
        exterior_edge_density = float(np.mean(exterior_edges[ext_mask]))

        out_std = float(np.std(out_gray))
        out_edges = cv2.Canny(out_gray, 45, 135) > 0
        out_edge_density = float(np.mean(out_edges))

        # An untouched flat bubble interior should have low std (~0-15) and near-zero edge
        # density (a solid fill). Flag when the OUTPUT interior instead resembles the
        # exterior's own texture: comparable std (within 50%) AND non-trivial edge density,
        # while the exterior itself is genuinely textured (std >= 20 or edge_density >= 0.05
        # -- i.e. don't flag on a flat exterior, that can't be the source of a bleed).
        exterior_is_textured = exterior_std >= 20.0 or exterior_edge_density >= 0.05
        interior_resembles_exterior = (
            exterior_std > 0
            and 0.5 <= (out_std / exterior_std) <= 2.0
            and out_edge_density >= 0.04
        )
        luma_lift = abs(out_std - exterior_std)  # kept as a reported number, not the sole gate
        smudge_flag = bool(exterior_is_textured and interior_resembles_exterior)
        std_ratio = round(out_std / max(exterior_std, 1.0), 3)

    # outline integrity: thin ring just inside the green_box perimeter, edge-pixel retention
    ring_w = max(2, margin // 2)
    def _perimeter_ring(x1, y1, x2, y2, rw):
        m = np.zeros((y2 - y1, x2 - x1), dtype=bool)
        m[:rw, :] = True
        m[-rw:, :] = True
        m[:, :rw] = True
        m[:, -rw:] = True
        return m

    src_box = source_bgr[gy1:gy2, gx1:gx2]
    out_box = output_bgr[gy1:gy2, gx1:gx2]
    if src_box.size == 0 or out_box.size == 0:
        outline_integrity = None
    else:
        ring_mask = _perimeter_ring(gx1, gy1, gx2, gy2, ring_w)
        src_box_gray = cv2.cvtColor(src_box, cv2.COLOR_BGR2GRAY)
        out_box_gray = cv2.cvtColor(out_box, cv2.COLOR_BGR2GRAY)
        src_edges = cv2.Canny(src_box_gray, 45, 135) > 0
        out_edges = cv2.Canny(out_box_gray, 45, 135) > 0
        src_ring_edges = int(np.count_nonzero(src_edges & ring_mask))
        out_ring_edges = int(np.count_nonzero(out_edges & ring_mask))
        outline_integrity = (out_ring_edges / src_ring_edges) if src_ring_edges > 0 else None

    return {
        "id": constraint.get("id"),
        "bubble_idx": constraint.get("bubble_idx"),
        "mask_mode": constraint.get("mask_mode"),
        "route": constraint.get("route"),
        "green_box": [gx1, gy1, gx2, gy2],
        "residual_ink_pct": round(residual_ink_pct, 2),
        "luma_lift": round(luma_lift, 2),
        "std_ratio": round(std_ratio, 3),
        "smudge_flag": smudge_flag,
        "outline_integrity": None if outline_integrity is None else round(outline_integrity, 3),
        "text_preview": _safe((constraint.get("text") or "")[:20]),
    }


def score_sample(sample_dir: Path) -> list[dict]:
    source_path = sample_dir / "input.jpg"
    output_path = sample_dir / "step_4_final" / "inpainted_result.jpg"
    layout_path = sample_dir / "step_6_layout" / "layout_constraints.json"
    if not (source_path.exists() and output_path.exists() and layout_path.exists()):
        return []
    source_bgr = _load_gray(source_path)
    output_bgr = _load_gray(output_path)
    if source_bgr is None or output_bgr is None:
        return []
    constraints = json.loads(layout_path.read_text(encoding="utf-8"))
    results = []
    for c in constraints:
        try:
            results.append(score_constraint(source_bgr, output_bgr, c))
        except Exception as e:  # noqa: BLE001 -- scorer must never crash a full-suite run
            results.append({"id": c.get("id"), "error": _safe(str(e))})
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("test_run_root", type=Path, nargs="?", default=None,
                     help="Legacy mode: a directory of group/sample/step_N_* trees.")
    ap.add_argument("--baseline-manifest", type=Path, default=None,
                     help="Resolve each sample via a baseline manifest.json's runtimeSampleName "
                          "-- the archived baseline dir itself only has step_6_layout and "
                          "step_8_typeset (see build_baseline.py's ARCHIVED_ARTIFACTS), not "
                          "input.jpg or step_4_final, so this reads the LIVE runtime_samples/ "
                          "tree instead, same as check_translated_regions_cleaned.py's dual mode.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if bool(args.test_run_root) == bool(args.baseline_manifest):
        ap.error("pass exactly one of test_run_root or --baseline-manifest")

    all_results: dict[str, list[dict]] = {}

    if args.test_run_root:
        root: Path = args.test_run_root
        sample_dirs: list[tuple[str, Path]] = []
        for group_dir in sorted(root.iterdir()):
            if not group_dir.is_dir():
                continue
            for sample_dir in sorted(group_dir.iterdir()):
                if sample_dir.is_dir():
                    sample_dirs.append((f"{group_dir.name}/{sample_dir.name}", sample_dir))

        if not sample_dirs:
            print(f"ERROR: no group/sample directories found under {root}")
            return 1

        coverage = GateCoverage("bubble-interior-damage scorer", len(sample_dirs))
        for group_sample, sample_dir in sample_dirs:
            missing = [rel for rel in REQUIRED if not (sample_dir / rel).exists()]
            if missing:
                coverage.record_incomplete(group_sample, missing)
                continue
            scored = score_sample(sample_dir)
            coverage.record_checked()
            if scored:
                all_results[group_sample] = scored
        out_path = args.out or (root / "bubble_damage_scores.json")
    else:
        manifest_total = len(load_manifest(args.baseline_manifest))
        coverage = GateCoverage("bubble-interior-damage scorer", manifest_total)
        for group_sample, entry, runtime_dir, missing in iter_baseline_samples(args.baseline_manifest, REQUIRED):
            if runtime_dir is None:
                coverage.record_missing(group_sample)
                continue
            if missing:
                coverage.record_incomplete(group_sample, missing)
                continue
            scored = score_sample(runtime_dir)
            coverage.record_checked()
            if scored:
                all_results[group_sample] = scored
        out_path = args.out or (args.baseline_manifest / "bubble_damage_scores.json")

    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"scored {len(all_results)} samples -> {out_path}")
    coverage.report()

    # Summary of flagged constraints
    flagged = 0
    for sample, constraints in all_results.items():
        for c in constraints:
            if c.get("smudge_flag") or (c.get("outline_integrity") is not None and c["outline_integrity"] < 0.5):
                flagged += 1
                print(f"  FLAGGED {sample} id={c.get('id')} mode={c.get('mask_mode')} "
                      f"luma_lift={c.get('luma_lift')} std_ratio={c.get('std_ratio')} "
                      f"outline_integrity={c.get('outline_integrity')}")
    print(f"total flagged constraints: {flagged}")
    # coverage.failed() catches the same vacuous-pass shape Part A eliminated from the gates:
    # a mostly-stripped tree honestly prints "1/34 checked, MISSING (33)" above but, before this,
    # still exited 0 -- indistinguishable from a real 34/34 pass to anything reading the exit code
    # (a CI step, a `&&` chain, a future run_all_gates.py wiring). Found by /verify 2026-07-31.
    return 1 if coverage.failed() else 0


if __name__ == "__main__":
    raise SystemExit(main())
