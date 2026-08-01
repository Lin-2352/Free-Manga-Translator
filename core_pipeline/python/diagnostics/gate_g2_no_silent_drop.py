"""G2 -- no-silent-drop gate (P0-3, task #191).

Every step-6 kept constraint that step 7 gave a non-empty translation to must
either (a) reach step 8's typeset_report.json directly, (b) be absorbed into a
synthetic (merged/grouped) layout that DID reach typeset_report -- tracked via
its merged_from field, see below -- or (c) be intentionally excluded by step
8's own documented duplicate-overlap drop (which prefers the longer of two
overlapping translations -- see _drop_duplicate_overlapping_layouts in
run_step8_typeset.py). An id that fails all three is a silent drop: it was
going to be shown to the user and then vanished with no record.

PREMISE CORRECTED 2026-07-30: step 8 has THREE sites that merge multiple
kept ids into one synthetic layout (two reverse_dark_bubble groupings plus
merged_caption) -- the merged item renders under a NEW id
("reverse_dark_2_1", "merged_caption_3_4", etc.) while the absorbed ids
vanish from typeset_report with no trace. Before merged_from was threaded
through to the report (this commit), G2 had no way to tell that apart from
an actual drop: verified directly, 5 of a baseline's 6 "silent drops" were
merge absorptions with their text rendering fine under the group id, and
only 1 (new_sample_6 id=7) was real. This does not try to re-derive step 8's
overlap heuristic -- doing so would just duplicate the bug surface it might
have. Instead it flags every id not accounted for by (a)/(b)/(c) and reports
the sibling (touching_container_siblings / shared bubble_idx) context, so a
human can tell "known duplicate-drop, working as intended" from "actually
silent." Known positive: original/sample3 id=11 (shares bubble_idx=4 with
kept id=8, dropped by the duplicate-overlap pass, exactly the case
#213/P1-1 targeted and fixed).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_common import (  # noqa: E402
    GateCoverage, is_renderable_translation, iter_baseline_samples, load_manifest, safe,
)

REQUIRED = (
    "step_6_layout/layout_constraints.json",
    "step_7_translate/translation_results.json",
    "step_8_typeset/typeset_report.json",
)


def check_sample(runtime_dir: Path) -> list[dict]:
    constraints_path = runtime_dir / "step_6_layout" / "layout_constraints.json"
    rejected_path = runtime_dir / "step_6_layout" / "rejected_layout_items.json"
    typeset_path = runtime_dir / "step_8_typeset" / "typeset_report.json"
    translations_path = runtime_dir / "step_7_translate" / "translation_results.json"

    # Required-file presence is already guaranteed by resolve_sample's `required`
    # check before this is called; a JSON parse failure here is a genuine
    # UNREADABLE sample, not "nothing to report" -- let it raise.
    constraints = json.loads(constraints_path.read_text(encoding="utf-8"))
    typeset = json.loads(typeset_path.read_text(encoding="utf-8"))
    translations = json.loads(translations_path.read_text(encoding="utf-8"))
    rejected = json.loads(rejected_path.read_text(encoding="utf-8")) if rejected_path.exists() else []

    typeset_ids = {str(t.get("id")) for t in typeset}
    rejected_ids = {str(r.get("id")) for r in rejected}
    trans_map = {str(t.get("id")): t for t in translations}

    merged_ids: set[str] = set()
    for t in typeset:
        for member_id in (t.get("merged_from") or []):
            merged_ids.add(str(member_id))

    by_bubble: dict[int, list[str]] = {}
    for c in constraints:
        bidx = c.get("bubble_idx")
        if isinstance(bidx, int) and bidx >= 0:
            by_bubble.setdefault(bidx, []).append(str(c.get("id")))

    results = []
    for c in constraints:
        cid = str(c.get("id"))
        en_text = str(trans_map.get(cid, {}).get("en_text", "") or "").strip()
        if not is_renderable_translation(en_text):
            # Not just "empty" -- run_step4_inpaint.py never cleans, and step 8 never
            # renders, a translation with zero alphanumeric content ("...") in the first
            # place. Flagging it would re-discover intentional pipeline behavior as a
            # fresh "silent drop" (see is_renderable_translation's docstring).
            continue
        if cid in typeset_ids or cid in rejected_ids or cid in merged_ids:
            continue  # accounted for: rendered directly, merged into a rendered synthetic
            # item, or explicitly rejected with a reason

        bidx = c.get("bubble_idx")
        siblings = [sid for sid in by_bubble.get(bidx, []) if sid != cid] if isinstance(bidx, int) and bidx >= 0 else []
        results.append({
            "id": cid,
            "bubble_idx": bidx,
            "siblings_sharing_bubble": siblings,
            "en_preview": safe(en_text[:60]),
            "silent_drop": True,
        })
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    manifest_total = len(load_manifest(args.baseline_dir))
    coverage = GateCoverage("G2 no-silent-drop", manifest_total)
    all_results: dict[str, list[dict]] = {}

    for group_sample, entry, runtime_dir, missing in iter_baseline_samples(args.baseline_dir, REQUIRED):
        if runtime_dir is None:
            coverage.record_missing(group_sample)
            continue
        if missing:
            coverage.record_incomplete(group_sample, missing)
            continue
        try:
            found = check_sample(runtime_dir)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  UNREADABLE {group_sample}: {e}")
            coverage.record_unreadable(group_sample)
            continue
        coverage.record_checked()
        if found:
            all_results[group_sample] = found

    out_path = args.out or (args.baseline_dir / "g2_no_silent_drop.json")
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    total_drops = sum(len(v) for v in all_results.values())
    print(f"G2 no-silent-drop: results -> {out_path}")
    for sample, items in all_results.items():
        for it in items:
            print(f"  SILENT DROP {sample} id={it['id']} bubble_idx={it['bubble_idx']} "
                  f"siblings={it['siblings_sharing_bubble']} en={it['en_preview']!r}")
    coverage.report()
    print(f"total silent drops: {total_drops} across {len(all_results)} samples")
    return 1 if (total_drops > 0 or coverage.failed()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
