"""G4 -- determinism gate (P0-3).

Sends the SAME request N times through the real backend and asserts the
resulting layout_constraints.json / typeset_report.json are byte-identical
across runs. This is what makes every other gate's "verified" claim mean
anything: if the pipeline isn't deterministic, a single passing run of G1-G3
proves nothing about the next request for the same image.

Two things that make this gate meaningless if missed:

1. The runtime output cache. _has_reusable_runtime_output serves request 2
   for the same image+language straight from disk with zero re-inference,
   which would make this gate "pass" by comparing a file to itself. Every
   iteration MUST clear the runtime cache first and assert
   report.runtimeOutputCache == "miss" -- if that assertion doesn't hold,
   the run is invalid, reported as such, never silently treated as a pass.

2. A missing artifact is NOT evidence of determinism. An earlier version of
   this gate hashed an absent file to the literal string "MISSING", so two
   runs that BOTH failed to write layout_constraints.json hash-matched and
   reported "All targets deterministic". A run that didn't produce a
   required artifact is an invalid run, same as an HTTP error -- it is
   never compared.

Runs a small representative subset by default (not full-suite -- that's
Phase 3's job): external_ja_2 (the exact sample whose ~1px semantic-detector
jitter motivated P0-1), original/sample3 (the dropped-sibling case, to prove
determinism doesn't mask/unmask that bug run to run), and
new_test/new_sample_11_(chi) (largest sample, most surface area for
nondeterminism to hide in).

SCOPE, measured 2026-07-28: step 7's translation call is a remote LLM request
with no fixed seed/temperature, so its output text legitimately differs
between runs of the SAME image ("WHAT?!" vs "WHA-?!" on external_ja_2 id=3,
captured directly). That is not a pipeline bug and this gate does not try to
catch it -- see TYPESET_STRUCTURAL_FIELDS below. layout_constraints.json is
still compared in full: step 6 has no such remote-call excuse and P0-1's
claim is specifically that step 6 became fully deterministic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_common import RUNTIME_ROOT, clear_cache, drive_once  # noqa: E402

DEFAULT_TARGETS = [
    ("external", "external_ja_2", "ja"),
    ("original", "sample3", "ja"),
    ("new_test", "new_sample_11_(chi)", "zh"),
]

COMPARE_ARTIFACTS = [
    ("step_6_layout", "layout_constraints.json"),
    ("step_8_typeset", "typeset_report.json"),
]


# typeset_report.json fields to compare. Deliberately EXCLUDES text, lines, font_size,
# position, outline_width, status, clipped_pixels, and source_cover: step 7's translation
# call is a remote LLM request with no fixed seed/temperature, so re-running the same image
# legitimately gets different English phrasing every time ("WHAT?!" vs "WHA-?!", measured
# directly on external_ja_2 2026-07-28) -- and every excluded field is computed FROM that
# string (its length, whether it needed a word split to fit, how much of the box it covers),
# so they vary as a downstream consequence, not a bug. Confirmed by direct measurement: two
# fresh runs of external_ja_2 differed in `status` alone for id=4 ("fit_with_word_split" vs
# "fit") purely because one phrasing needed a word split on an otherwise-identical box and
# the other didn't. Re-adding any of these will make this gate fail forever on every sample.
# What's left is font_role/caption_backing/text_style/transparent_overlay -- step-6-derived
# classification fields that shouldn't depend on the translated string's wording -- plus id
# and bubble_idx, so the invariant this gate actually checks is: the same image produces the
# same SET of rendered ids, at the same bubble mapping, with the same semantic role/style
# classification, independent of what the translator happened to phrase.
TYPESET_STRUCTURAL_FIELDS = [
    "id", "bubble_idx", "transparent_overlay", "font_role", "caption_backing", "text_style",
]


def _typeset_structural(content: list[dict]) -> list[dict]:
    return [
        {k: item.get(k) for k in TYPESET_STRUCTURAL_FIELDS}
        for item in sorted(content, key=lambda it: str(it.get("id")))
    ]


def artifact_hashes(runtime_dir: Path) -> tuple[dict[str, str] | None, str | None]:
    """Returns (hashes, None) on success, or (None, reason) if any required artifact
    is missing -- a missing artifact is an invalid run, never a hash to compare."""
    hashes = {}
    for stage, fname in COMPARE_ARTIFACTS:
        p = runtime_dir / stage / fname
        if not p.exists():
            return None, f"run wrote no {stage}/{fname}"
        content = json.loads(p.read_text(encoding="utf-8"))
        # Hash the parsed-and-reserialized content, not raw bytes: key order in the
        # source JSON isn't a determinism signal, only the actual field values are.
        if fname == "typeset_report.json":
            content = _typeset_structural(content)
        canon = json.dumps(content, sort_keys=True, ensure_ascii=True).encode("utf-8")
        hashes[f"{stage}/{fname}"] = hashlib.sha256(canon).hexdigest()
    return hashes, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="8766")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.repeats < 2:
        ap.error("--repeats must be >= 2 -- a determinism gate that compares fewer than 2 runs "
                  "compares nothing")

    all_results: dict[str, dict] = {}
    invalid = []
    invalid_targets = []
    nondeterministic = []

    for group, sample, lang in DEFAULT_TARGETS:
        key = f"{group}/{sample}"
        print(f"\n=== {key} (lang={lang}), {args.repeats} runs ===", flush=True)
        run_hashes = []
        for i in range(args.repeats):
            clear_cache(args.port)
            result = drive_once(args.port, group, sample, lang)
            print(f"  run {i}: HTTP {result['status_code']} cache={result.get('cache')} "
                  f"sample={result.get('sampleName')} elapsed={result.get('elapsedSeconds')}s", flush=True)

            if result["status_code"] != 200:
                invalid.append(f"{key} run {i}: HTTP {result['status_code']} ({result.get('reason')})")
                continue
            if result.get("cache") != "miss":
                invalid.append(f"{key} run {i}: runtimeOutputCache={result.get('cache')!r}, expected "
                                f"'miss' -- this run does not prove determinism, it compares a file to itself")
                continue

            runtime_dir = RUNTIME_ROOT / result["sampleName"]
            hashes, reason = artifact_hashes(runtime_dir)
            if hashes is None:
                invalid.append(f"{key} run {i}: {reason}")
                continue
            run_hashes.append(hashes)
            print(f"    hashes: {hashes}", flush=True)

        all_results[key] = {"runs": run_hashes}
        if len(run_hashes) < 2:
            invalid_targets.append(f"{key}: only {len(run_hashes)}/{args.repeats} runs were valid "
                                    f"-- not enough to compare")
            continue
        first = run_hashes[0]
        for idx, h in enumerate(run_hashes[1:], 1):
            if h != first:
                nondeterministic.append(f"{key}: run 0 vs run {idx} differ: {first} vs {h}")

    out_path = args.out or (Path(__file__).resolve().parents[2] / "baselines" / "g4_determinism.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print(f"\nG4 determinism: results -> {out_path}")
    if invalid:
        print(f"INVALID RUNS ({len(invalid)}) -- did not get a clean cache-miss comparison:")
        for msg in invalid:
            print(f"  {msg}")
    if invalid_targets:
        print(f"INVALID TARGETS ({len(invalid_targets)}) -- too few valid runs to compare:")
        for msg in invalid_targets:
            print(f"  {msg}")
    if nondeterministic:
        print(f"NONDETERMINISTIC ({len(nondeterministic)}):")
        for msg in nondeterministic:
            print(f"  {msg}")
    if not invalid and not invalid_targets and not nondeterministic:
        print("All targets deterministic across all valid (cache-miss) runs.")

    return 1 if (invalid or invalid_targets or nondeterministic) else 0


if __name__ == "__main__":
    raise SystemExit(main())
