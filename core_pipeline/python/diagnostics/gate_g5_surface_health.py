"""G5 -- surface health gate (P0-3).

Drives every sample in the suite through the REAL backend (POST /translate,
the same endpoint and headers the extension uses -- not a batch driver,
which is what let the step-6 500 and the runtime-only clip failure through
undetected all last session) and asserts HTTP 200 with a rendered image in
the response. This is a live check, not a read of a stored baseline: its
whole job is to catch a crash that a batch-driver run can't see.

Does not clear the runtime cache between requests by default -- surface
health doesn't care whether a request was served fresh or from cache, only
whether it comes back successfully. Uses whatever cache state the server is
already in.

--require-fresh asserts every response is a cache MISS (report.
runtimeOutputCache == "miss"). run_all_gates.py always passes it: when this
gate runs as part of the full suite, it is what REPOPULATES the tree G1/G2
read next, and a warm-cache hit would mean those gates read artifacts from
whatever earlier commit last populated the tree, not the one under test.
Standalone G5 keeps the cache-agnostic default, correct for a pure
surface-health check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_common import ALL_SAMPLES, drive_once  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="8766")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--require-fresh", action="store_true",
                     help="Fail any sample served from the runtime output cache instead of "
                          "freshly re-inferenced. Always passed by run_all_gates.py.")
    args = ap.parse_args()

    results = {}
    for i, (group, sample, lang) in enumerate(ALL_SAMPLES, 1):
        key = f"{group}/{sample}"
        print(f"[{i}/{len(ALL_SAMPLES)}] {key} (lang={lang})", flush=True)
        result = drive_once(args.port, group, sample, lang)

        if result["ok"] and args.require_fresh and result.get("cache") != "miss":
            result = {**result, "ok": False,
                       "reason": f"runtimeOutputCache={result.get('cache')!r}, --require-fresh needs 'miss'"}

        result.pop("payload", None)  # not JSON-summary-friendly, not needed once logged
        results[key] = result
        status = "OK" if result["ok"] else f"FAIL ({result.get('reason')})"
        print(f"  {status} cache={result.get('cache')} elapsed={result.get('elapsedSeconds')}s", flush=True)

    out_path = args.out or (Path(__file__).resolve().parents[2] / "baselines" / "g5_surface_health.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    failures = {k: v for k, v in results.items() if not v["ok"]}
    print(f"\nG5 surface health: results -> {out_path}")
    print(f"{len(results) - len(failures)}/{len(results)} passed")
    for k, v in failures.items():
        print(f"  FAIL {k}: {v.get('reason')}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
