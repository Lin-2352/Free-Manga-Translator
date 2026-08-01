from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_BOOTSTRAP_FILE = Path(__file__).resolve()
for _candidate in _BOOTSTRAP_FILE.parents:
    if (_candidate / "samples").exists() and (_candidate / "python").exists():
        _PROJECT_ROOT_FOR_IMPORTS = _candidate
        break
else:
    _PROJECT_ROOT_FOR_IMPORTS = _BOOTSTRAP_FILE.parents[2]
for _rel in ("python/common", "python/steps", "python/runtime"):
    _path = str(_PROJECT_ROOT_FOR_IMPORTS / _rel)
    if _path not in sys.path:
        sys.path.insert(0, _path)
del _BOOTSTRAP_FILE, _candidate, _PROJECT_ROOT_FOR_IMPORTS, _rel, _path

import run_extension_pipeline_server as server


def test_collect_runtime_report_survives_a_truncated_artifact(tmp_path: Path) -> None:
    """A cache HIT calls _collect_runtime_report() directly, without ever re-running the
    pipeline. If a stage artifact (ocr_results.json, layout_constraints.json, etc.) is
    truncated -- a crash or power loss mid-write is the realistic cause -- json.loads()
    previously raised straight out of this function, uncaught. Because a cache hit never
    regenerates the artifact, that 500 would repeat on every subsequent request for the same
    cached image forever, with no self-heal, unlike a fresh pipeline run which would just
    write a fresh, valid file.

    This writes a real truncated (invalid) JSON file at the ocr_results.json path a real
    sample would use, and asserts _collect_runtime_report() degrades to treating it as absent
    (empty list) instead of raising."""
    sample_name = "test_corrupt_artifact_sample"
    sample_dir = server.SAMPLES_ROOT / sample_name
    ocr_dir = sample_dir / "step_5_ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    corrupt_path = ocr_dir / "ocr_results.json"
    corrupt_path.write_text('[{"id": 1, "text": "truncated mid-wri', encoding="utf-8")

    try:
        report = server._collect_runtime_report(sample_name, "ja")
    except Exception as error:
        raise AssertionError(
            f"_collect_runtime_report() must survive a truncated JSON artifact, not raise -- "
            f"got {type(error).__name__}: {error}"
        ) from error
    finally:
        corrupt_path.unlink(missing_ok=True)
        try:
            ocr_dir.rmdir()
            sample_dir.rmdir()
        except OSError:
            pass

    assert report.get("ocrItems") == 0, (
        f"a corrupt ocr_results.json should degrade to a count of 0, not fabricate data or "
        f"leave the field missing -- got report[\"ocrItems\"]={report.get('ocrItems')!r}"
    )


def main() -> int:
    with tempfile.TemporaryDirectory():
        test_collect_runtime_report_survives_a_truncated_artifact(Path("."))
    print("corrupt_artifact_cache_hit=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
