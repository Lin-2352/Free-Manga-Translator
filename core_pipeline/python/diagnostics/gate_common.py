"""Shared helpers for the P0-3 gates (G1-G5): read a P0-2-style baseline
manifest (group/sample -> runtime_{lang}_{digest}) and resolve each entry to
its actual artifact directory in runtime_samples/extension/.

Gates key off the manifest rather than globbing runtime_samples/extension/
directly so a sample can never be silently skipped because its runtime dir
happens to be missing -- the manifest is the list of what SHOULD exist.

GateCoverage + resolve_sample/resolve_baseline_sample exist because a gate
that silently checks zero samples and a gate that checks N samples and finds
zero violations produce THE SAME "PASS" -- and that ambiguity is a real,
previously-shipped bug (see 5040ea6, then the follow-up that found the same
class in G3 and in two more granularities: G3 had no guard at all, and
G1/G2's guard only checked for ANY step_* folder, so a tree where the
pipeline crashed partway through (e.g. only step_1..step_5 present) still
counted every sample as "checked"). Every gate that reads this tree routes
through GateCoverage so "checked nothing" is a distinct, loud, reported
state -- never silently indistinguishable from "checked everything, found
nothing wrong."
"""
from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path

RUNTIME_ROOT = (
    Path(__file__).resolve().parents[2] / "runtime_samples" / "extension"
)
SAMPLES_ROOT = Path(r"D:/Desktop/translator D/app/test run latest pipeline")

# The full 34-sample suite, (group, sample, sourceLanguage). Language values verified
# 2026-07-28 against each sample's own step_5_ocr/ocr_results.json ocr_provider field
# (manga_ocr => ja) for the 12 samples whose folder name doesn't encode a language.
ALL_SAMPLES = [
    ("external", "external_ja_1", "ja"),
    ("external", "external_ja_2", "ja"),
    ("external", "external_ko_1", "ko"),
    ("external", "external_ko_2", "ko"),
    ("external", "external_zh_1", "zh"),
    ("external", "external_zh_2", "zh"),
    ("modern", "modern_ja_1", "ja"),
    ("modern", "modern_ja_2", "ja"),
    ("modern", "modern_ko_1", "ko"),
    ("modern", "modern_ko_2", "ko"),
    ("modern", "modern_zh_1", "zh"),
    ("modern", "modern_zh_2", "zh"),
    ("new_test", "new_sample_1", "ja"),
    ("new_test", "new_sample_2", "ja"),
    ("new_test", "new_sample_3", "ja"),
    ("new_test", "new_sample_4", "ja"),
    ("new_test", "new_sample_5", "ja"),
    ("new_test", "new_sample_6", "ja"),
    ("new_test", "new_sample_7_(ko)", "ko"),
    ("new_test", "new_sample_8_(ja)", "ja"),
    ("new_test", "new_sample_9_(chi)", "zh"),
    ("new_test", "new_sample_10_(chi)", "zh"),
    ("new_test", "new_sample_11_(chi)", "zh"),
    ("new_test", "new_sample_12_(ko)", "ko"),
    ("new_test", "new_sample_13_(chi)", "zh"),
    ("new_test", "new_sample_14_(ko)", "ko"),
    ("new_test", "new_sample_15_(ja)", "ja"),
    ("new_test", "new_sample_16_(ja)", "ja"),
    ("original", "sample1", "ja"),
    ("original", "sample2", "ja"),
    ("original", "sample3", "ja"),
    ("original", "sample4", "ja"),
    ("original", "sample5", "ja"),
    ("original", "sample6", "ja"),
]


def load_manifest(baseline_dir: Path) -> dict[str, dict]:
    manifest_path = baseline_dir / "manifest.json"
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return data["manifest"]
    except FileNotFoundError:
        print(f"ERROR: no manifest.json at {manifest_path}")
        raise SystemExit(2)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"ERROR: malformed manifest.json at {manifest_path}: {e}")
        raise SystemExit(2)


def assert_manifest_matches_suite(manifest: dict) -> list[str]:
    """Symmetric diff between the manifest's rows and ALL_SAMPLES. A drift here means
    G1-G3 (manifest-driven) and G5 (ALL_SAMPLES-driven) silently disagree about what
    "the full suite" means. Returns human-readable diff lines; empty if they match."""
    manifest_keys = set(manifest.keys())
    suite_keys = {f"{g}/{s}" for g, s, _ in ALL_SAMPLES}
    lines = []
    only_manifest = sorted(manifest_keys - suite_keys)
    only_suite = sorted(suite_keys - manifest_keys)
    if only_manifest:
        lines.append(f"in manifest but not ALL_SAMPLES: {only_manifest}")
    if only_suite:
        lines.append(f"in ALL_SAMPLES but not manifest: {only_suite}")
    return lines


def resolve_sample(entry: dict, required: tuple[str, ...] = ()) -> tuple[Path | None, list[str]]:
    """Resolve a manifest entry to its runtime_samples/extension/ dir.

    Returns (dir, missing_required_artifacts). dir is None when the sample cannot be
    used at all (no recorded runtime name, dir absent, or dir stripped of every
    step_* folder -- the POST /v1/cache/clear signature: it removes each sample's
    stage folders but LEAVES the sample dir itself holding input.jpg +
    source_upload.bin, so a bare Path.exists() silently accepted a stripped sample as
    usable -- measured directly: the same run_all_gates.py invocation on the same
    baseline went from FAIL (real violations) to "All gate(s) passed" with no code
    change, only a cache clear).

    When dir is not None, `missing` lists any of `required` (relative paths from the
    sample dir) that don't exist -- e.g. a tree where the pipeline crashed at step 6
    still has step_1..step_5 folders (passing the check above) but is missing
    step_6_layout/layout_constraints.json, which every caller must treat as
    unusable, not silently skip.
    """
    name = entry.get("runtimeSampleName")
    if not name:
        return None, list(required)
    d = RUNTIME_ROOT / name
    if not d.exists() or not any(d.glob("step_*")):
        return None, list(required)
    missing = [rel for rel in required if not (d / rel).exists()]
    return d, missing


def resolve_baseline_sample(
    baseline_dir: Path, group: str, sample: str, required: tuple[str, ...] = ()
) -> tuple[Path | None, list[str]]:
    """Same contract as resolve_sample, for gates (G3) that read the archived
    baseline tree (baseline_dir/<group>/<sample>/...) instead of the live runtime
    tree."""
    d = baseline_dir / group / sample
    if not d.exists():
        return None, list(required)
    missing = [rel for rel in required if not (d / rel).exists()]
    return d, missing


class GateCoverage:
    """Accumulates what a gate actually managed to check, so 'checked nothing' can
    never be silently reported as 'passed'."""

    def __init__(self, gate_name: str, manifest_total: int):
        self.gate_name = gate_name
        self.manifest_total = manifest_total
        self.checked = 0
        self.missing: list[str] = []
        self.incomplete: dict[str, list[str]] = {}
        self.unreadable: list[str] = []

    def record_missing(self, group_sample: str) -> None:
        self.missing.append(group_sample)

    def record_incomplete(self, group_sample: str, missing_artifacts: list[str]) -> None:
        self.incomplete[group_sample] = missing_artifacts

    def record_unreadable(self, group_sample: str) -> None:
        self.unreadable.append(group_sample)

    def record_checked(self) -> None:
        self.checked += 1

    def report(self) -> None:
        print(f"{self.gate_name}: {self.checked}/{self.manifest_total} manifest samples checked")
        if self.missing:
            print(f"  MISSING runtime dir ({len(self.missing)}): {self.missing}")
        if self.incomplete:
            for gs, arts in self.incomplete.items():
                print(f"  INCOMPLETE {gs}: missing {arts}")
        if self.unreadable:
            print(f"  UNREADABLE ({len(self.unreadable)}): {self.unreadable}")

    def failed(self) -> bool:
        if self.missing or self.incomplete or self.unreadable:
            return True
        if self.manifest_total > 0 and self.checked == 0:
            return True
        return False


def iter_baseline_samples(baseline_dir: Path, required: tuple[str, ...] = ()):
    """Yields (group_sample, entry, runtime_dir_or_None, missing_required_artifacts)
    for every manifest row, resolved via resolve_sample."""
    manifest = load_manifest(baseline_dir)
    for group_sample, entry in sorted(manifest.items()):
        runtime_dir, missing = resolve_sample(entry, required)
        yield group_sample, entry, runtime_dir, missing


def clear_cache(port: str = "8766") -> None:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/cache/clear",
        data=b"{}",
        headers={"Content-Type": "application/json", "X-Fmt-Client": "free-manga-translator-extension"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()


def drive_once(port: str, group: str, sample: str, lang: str, timeout: int = 900) -> dict:
    """POST one sample through the real backend /translate endpoint, exactly as the
    extension does. Shared by G4 (determinism) and G5 (surface health) -- previously
    two near-identical copies of this request-building logic."""
    import glob

    matches = glob.glob(str(SAMPLES_ROOT / group / sample / "input.*"))
    if not matches:
        return {"ok": False, "status_code": None, "reason": "no input file found"}
    img_path = matches[0]
    mime = mimetypes.guess_type(img_path)[0] or "image/jpeg"
    with open(img_path, "rb") as f:
        raw = f.read()
    data_url = f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")

    body = json.dumps({
        "imageData": data_url,
        "sourceLanguage": lang,
        "targetLanguage": "en",
        "qualityProfile": "strict",
        "requestedOutput": "translatedImageDataUrl",
        "clientRequestId": f"gate-drive-{sample}",
    }).encode("utf-8")

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/translate",
        data=body,
        headers={"Content-Type": "application/json", "X-Fmt-Client": "free-manga-translator-extension"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            status_code = resp.status
    except urllib.error.HTTPError as e:
        return {"ok": False, "status_code": e.code, "reason": f"HTTP {e.code}",
                "elapsedSeconds": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001 -- a network/timeout failure is what these gates exist to catch
        return {"ok": False, "status_code": None, "reason": f"request error: {e}",
                "elapsedSeconds": round(time.time() - t0, 1)}

    elapsed = round(time.time() - t0, 1)
    report = payload.get("report") if isinstance(payload.get("report"), dict) else {}
    img_field = payload.get("translatedImageDataUrl") or payload.get("imageDataUrl")

    return {
        "ok": status_code == 200 and isinstance(img_field, str) and img_field.startswith("data:"),
        "status_code": status_code,
        "reason": None if status_code == 200 else f"HTTP {status_code}",
        "elapsedSeconds": elapsed,
        "cache": report.get("runtimeOutputCache"),
        "sampleName": report.get("sampleName"),
        "payload": payload,
    }


def safe(s: str | None) -> str:
    return (s or "").encode("ascii", "replace").decode("ascii")


def is_renderable_translation(text: str) -> bool:
    """Mirrors run_step4_inpaint.py's _is_renderable_translation exactly (not a fresh
    heuristic -- see task #219/Part B, 2026-07-30): a translation with zero alphanumeric
    characters (e.g. "...") is intentionally never cleaned by step 4 or rendered by step 8,
    which is correct pipeline behavior, not a bug -- there is nothing to typeset. Before
    this, that decision was only ever logged as an aggregate console count
    ("skipped N without renderable English"), with no per-id record on disk, so G1/G2 had
    no way to tell it apart from a genuine silent drop. Gates must use this exact
    predicate, not en_text truthiness alone, or they will re-flag every ellipsis-only
    translation as a fresh violation."""
    if not text or text.startswith("[TL:"):
        return False
    return any(char.isalnum() for char in text)
