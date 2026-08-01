"""Baseline producer for G1/G2/G3 (P0-3, item A4).

Drives all 34 ALL_SAMPLES through the real backend sequentially, archives
per-sample rendered image + step_6/step_8 artifacts, keyed by the backend's
own runtime_{lang}_{digest} identity, and writes manifest.json mapping
group/sample -> that runtime sample name.

This is the producer G1/G2/G3 depend on. It did not previously exist as a
committed script -- the baseline that shipped Phase 0 was built by a
throwaway scratchpad driver that was never committed, so a clean clone of
this repo had no way to regenerate the gates' input at all.

Fails loudly on partial capture: a baseline silently built while one sample
500'd would leave run_all_gates.py's manifest-vs-ALL_SAMPLES check
permanently red, so any capture failure aborts the whole run rather than
writing a partial manifest.json. manifest.json is now only written on full
success (2026-07-30 fix) -- an earlier version wrote it unconditionally
before checking for failures, so an aborted run left exactly the partial
manifest this docstring said it wouldn't.

Clears the runtime cache once at the start (2026-07-30 fix), the same
pattern run_all_gates.py already uses, so a sample driven manually minutes
earlier in the same session doesn't trip the mid-run cache-hit abort below.
Before this, a stale sample cost two full ~20-minute aborted runs in one
session -- the cache-hit check still fires mid-run as a safety net (e.g. a
concurrent process re-driving the same sample), it just no longer fires on
the FIRST request as a matter of course.
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import shutil
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_common import ALL_SAMPLES, RUNTIME_ROOT, SAMPLES_ROOT, clear_cache, drive_once  # noqa: E402

ARCHIVED_ARTIFACTS = [
    ("step_6_layout", "layout_constraints.json"),
    ("step_6_layout", "rejected_layout_items.json"),
    ("step_8_typeset", "typeset_report.json"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_dir", type=Path)
    ap.add_argument("--port", default="8766")
    args = ap.parse_args()

    args.baseline_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    failures: list[tuple[str, str, str]] = []

    print("Clearing runtime cache before starting (guarantees every drive below is a fresh "
          "miss, even if a sample was manually driven earlier in this session)...", flush=True)
    try:
        clear_cache(args.port)
    except (urllib.error.URLError, OSError) as e:
        # Before this, an unreachable backend surfaced as a raw URLError traceback -- the
        # ONE failure mode in this script that didn't get drive_once's clean
        # {"ok": False, "reason": ...} treatment, because this call sits outside the
        # per-sample loop. Found by /verify 2026-07-31.
        print(f"ERROR: could not reach backend at port {args.port} to clear cache -- "
              f"is it running? ({e})", flush=True)
        return 1

    for i, (group, sample, lang) in enumerate(ALL_SAMPLES, 1):
        key = f"{group}/{sample}"
        print(f"\n===== [{i}/{len(ALL_SAMPLES)}] {key} (lang={lang}) =====", flush=True)

        if not glob.glob(str(SAMPLES_ROOT / group / sample / "input.*")):
            print(f"  FAIL: no input file found for {key}", flush=True)
            failures.append((group, sample, "no input file"))
            continue

        result = drive_once(args.port, group, sample, lang)
        print(f"  HTTP {result['status_code']} cache={result.get('cache')} "
              f"elapsed={result.get('elapsedSeconds')}s", flush=True)

        if result.get("cache") == "hit":
            print("  FATAL: served from cache, not freshly re-inferenced -- ABORTING RUN", flush=True)
            failures.append((group, sample, "cache hit -- baseline invalid"))
            break

        if not result["ok"]:
            print(f"  FAIL: {result.get('reason')}", flush=True)
            failures.append((group, sample, result.get("reason") or "unknown failure"))
            continue

        payload = result["payload"]
        runtime_sample_name = result.get("sampleName")
        if not runtime_sample_name:
            print(f"  FAIL: response carried no report.sampleName for {key}", flush=True)
            failures.append((group, sample, "no sampleName in response"))
            continue

        dest = args.baseline_dir / group / sample
        dest.mkdir(parents=True, exist_ok=True)

        img_field = payload.get("translatedImageDataUrl") or payload.get("imageDataUrl")
        if isinstance(img_field, str) and img_field.startswith("data:"):
            (dest / "rendered.png").write_bytes(base64.b64decode(img_field.split(",", 1)[1]))
        else:
            print(f"  FAIL: no rendered image in response for {key}", flush=True)
            failures.append((group, sample, "no rendered image in response"))
            continue

        slim = {k: v for k, v in payload.items() if k not in ("translatedImageDataUrl", "imageDataUrl")}
        (dest / "response.json").write_text(json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8")

        runtime_dir = RUNTIME_ROOT / runtime_sample_name
        for stage, fname in ARCHIVED_ARTIFACTS:
            src = runtime_dir / stage / fname
            if src.exists():
                (dest / stage).mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / stage / fname)

        manifest[key] = {
            "lang": lang,
            "runtimeSampleName": runtime_sample_name,
            "httpStatus": result["status_code"],
            "elapsedSeconds": result.get("elapsedSeconds"),
            "cache": result.get("cache"),
        }

    print(f"\n===== DONE: {len(manifest)}/{len(ALL_SAMPLES)} succeeded, {len(failures)} failed/skipped =====",
          flush=True)
    if failures:
        for g, s, reason in failures:
            print(f"  FAILED: {g}/{s}: {reason}", flush=True)
        # A partial baseline is worse than no baseline: run_all_gates.py's manifest-vs-
        # ALL_SAMPLES check would otherwise be permanently red for a reason nobody chose.
        # So manifest.json is written ONLY here, on full success -- an earlier version wrote
        # it unconditionally before this check, leaving exactly the partial artifact this
        # comment always said was worse than none.
        print("baseline NOT written -- rebuild required before gating against this dir.", flush=True)
        return 1

    (args.baseline_dir / "manifest.json").write_text(
        json.dumps({"manifest": manifest, "failures": failures}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
