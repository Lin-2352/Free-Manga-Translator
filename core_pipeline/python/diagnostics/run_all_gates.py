"""P0-3 runner: executes all five gates against a P0-2-style baseline and
exits non-zero if any fails.

Order is driven by what each gate does to the runtime tree, not by gate
number -- an earlier version ran G1/G2 (read the tree) before G5 (which
POPULATES it) and G4 last (which DESTROYS it with 9 global cache clears),
so the documented canonical command hard-failed at G1/G2 on a cold tree,
minutes before the step that would have populated it:

  clear cache        one global /v1/cache/clear -> forces the G5 drive below to be a cache miss
  G5 surface health  live, drives all 34         -> POPULATES the tree with fresh artifacts
  G1 translated-regions-cleaned                  -> reads the tree G5 just wrote
  G2 no-silent-drop                              -> reads the same tree
  G3 art-integrity                               -> reads the archived baseline (independent)
  G4 determinism      live, 9 global cache clears -> DESTRUCTIVE, must run last

The leading clear is load-bearing, not tidiness: G5 does not clear the cache
itself (its docstring is explicit -- it uses whatever cache state the server
is already in), so on a WARM tree its 34 requests would be served from
cache, re-inference would never run, and G5 would populate nothing. Without
the leading clear, G1/G2 could silently read artifacts from whatever earlier
commit last populated the tree -- exactly the "verified against artifacts
that didn't come from the code under test" failure this whole gate suite
exists to prevent. G5 is called with --require-fresh to make the guarantee
checkable rather than assumed.

G4 runs last and leaves the tree holding only ONE sample -- not its 3
targets, a claim this docstring made until 2026-07-30 and a live /verify
proved wrong (G4's clear_cache() runs before EACH of its 9 drives, so only
the final drive survives; verified: 1/34 checked, 33 missing, 6 step_*
dirs on disk). An immediately following --skip-live run will correctly
fail on the other 33 until repopulated -- stated explicitly in the
summary below, not left for the next person to rediscover.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_common import assert_manifest_matches_suite, clear_cache, load_manifest  # noqa: E402

DIAG_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def run(name: str, cmd: list[str]) -> bool:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}", flush=True)
    result = subprocess.run(cmd, cwd=str(DIAG_DIR.parents[1]))
    passed = result.returncode == 0
    print(f"{name}: {'PASS' if passed else 'FAIL'} (exit {result.returncode})", flush=True)
    return passed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_dir", type=Path)
    ap.add_argument("--skip-live", action="store_true",
                     help="Skip the leading cache clear, G5, and G4 -- only G1/G2/G3 run, "
                          "against whatever is already in the runtime tree. Fast sanity pass "
                          "only; requires an already-populated tree; not a real gate pass.")
    ap.add_argument("--port", default="8766")
    args = ap.parse_args()

    baseline = str(args.baseline_dir)

    # Fatal before running anything: G1-G3 iterate the manifest, G5 iterates ALL_SAMPLES.
    # If those two lists disagree, "the full suite" means two different things to different
    # gates and every subsequent PASS/FAIL is answering an ambiguous question.
    manifest = load_manifest(args.baseline_dir)
    drift = assert_manifest_matches_suite(manifest)
    if drift:
        print("FATAL: baseline manifest does not match ALL_SAMPLES -- refusing to run any gate.")
        for line in drift:
            print(f"  {line}")
        return 1

    results: dict[str, bool] = {}

    if args.skip_live:
        print("\n--skip-live: leading cache clear, G5, and G4 NOT run. This is a fast sanity "
              "pass against whatever is already in the runtime tree, not a real gate pass.",
              flush=True)
    else:
        print(f"\n{'=' * 60}\nclear cache (forces the G5 drive below to be a fresh miss)\n{'=' * 60}",
              flush=True)
        clear_cache(args.port)
        results["G5 surface health"] = run(
            "G5 surface health",
            [PYTHON, str(DIAG_DIR / "gate_g5_surface_health.py"), "--port", args.port, "--require-fresh"],
        )

    results["G1 translated-regions-cleaned"] = run(
        "G1 translated-regions-cleaned",
        [PYTHON, str(DIAG_DIR / "check_translated_regions_cleaned.py"), "--baseline-manifest", baseline],
    )
    results["G2 no-silent-drop"] = run(
        "G2 no-silent-drop",
        [PYTHON, str(DIAG_DIR / "gate_g2_no_silent_drop.py"), baseline],
    )
    results["G3 art-integrity"] = run(
        "G3 art-integrity",
        [PYTHON, str(DIAG_DIR / "gate_g3_art_integrity.py"), baseline],
    )

    if not args.skip_live:
        results["G4 determinism"] = run(
            "G4 determinism",
            [PYTHON, str(DIAG_DIR / "gate_g4_determinism.py"), "--port", args.port],
        )

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    if args.skip_live:
        print("  SKIPPED  G5 surface health (--skip-live)")
        print("  SKIPPED  G4 determinism (--skip-live)")
    else:
        print("\nEnd state: G4 just ran last, clearing the cache before EACH of its 9 drives, "
              "so only its final drive's sample survives -- the tree now holds ONE sample, not "
              "the 3 targets G4 covers. A following --skip-live run will correctly FAIL on the "
              "other 33 samples until this repopulates the tree again.")

    failed = [name for name, passed in results.items() if not passed]
    if failed:
        print(f"\n{len(failed)} gate(s) failed: {', '.join(failed)}")
    else:
        print(f"\nAll {len(results)} gate(s) passed" + (" (live gates skipped)" if args.skip_live else "."))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
