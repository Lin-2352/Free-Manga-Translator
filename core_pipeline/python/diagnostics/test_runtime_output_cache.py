from __future__ import annotations

import base64
import io
import shutil
import sys
from pathlib import Path

from PIL import Image

_BOOTSTRAP_FILE = Path(__file__).resolve()
for _candidate in _BOOTSTRAP_FILE.parents:
    if (_candidate / "samples").exists() and (_candidate / "python").exists():
        _PROJECT_ROOT_FOR_IMPORTS = _candidate
        break
else:
    _PROJECT_ROOT_FOR_IMPORTS = _BOOTSTRAP_FILE.parents[2]
for _rel in ("", "python/common", "python/runtime"):
    _path = str(_PROJECT_ROOT_FOR_IMPORTS / _rel)
    if _path not in sys.path:
        sys.path.insert(0, _path)
del _BOOTSTRAP_FILE, _candidate, _PROJECT_ROOT_FOR_IMPORTS, _rel, _path

from backend_api.app import pipeline_bridge
import run_extension_pipeline_server as runtime_bridge


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(output, format="PNG")
    return output.getvalue()


def _write_minimal_pass_artifacts(sample_name: str, language: str) -> None:
    sample_path = runtime_bridge.SAMPLES_ROOT / sample_name
    (sample_path / "step_1_detect").mkdir(parents=True, exist_ok=True)
    (sample_path / "step_4_final").mkdir(parents=True, exist_ok=True)
    (sample_path / "step_5_ocr").mkdir(parents=True, exist_ok=True)
    (sample_path / "step_6_layout").mkdir(parents=True, exist_ok=True)
    (sample_path / "step_7_translate").mkdir(parents=True, exist_ok=True)
    (sample_path / "step_8_typeset").mkdir(parents=True, exist_ok=True)

    (sample_path / "step_1_detect" / "detections.json").write_text("[]", encoding="utf-8")
    (sample_path / "step_1_detect" / "semantic_detections.json").write_text("[]", encoding="utf-8")
    (sample_path / "step_5_ocr" / "ocr_results.json").write_text("[]", encoding="utf-8")
    (sample_path / "step_6_layout" / "layout_constraints.json").write_text('[{"id": 0}]', encoding="utf-8")
    (sample_path / "step_6_layout" / "rejected_layout_items.json").write_text("[]", encoding="utf-8")
    (sample_path / "step_7_translate" / "translation_results.json").write_text(
        '[{"id": 0, "en_text": "HELLO"}]',
        encoding="utf-8",
    )
    (sample_path / "step_8_typeset" / "typeset_report.json").write_text(
        '[{"id": 0, "status": "fit"}]',
        encoding="utf-8",
    )
    Image.new("RGB", (32, 32), (255, 255, 255)).save(sample_path / "step_4_final" / "inpainted_result.jpg")
    Image.new("RGB", (32, 32), (255, 255, 255)).save(sample_path / "step_8_typeset" / "final_output.png")
    report = runtime_bridge._collect_runtime_report(sample_name, language)
    runtime_bridge._assert_runtime_report_safe(report)
    runtime_bridge._write_runtime_cache_meta(sample_name, language, report)


def _write_no_renderable_artifacts(sample_name: str) -> None:
    sample_path = runtime_bridge.SAMPLES_ROOT / sample_name
    (sample_path / "step_5_ocr").mkdir(parents=True, exist_ok=True)
    (sample_path / "step_6_layout").mkdir(parents=True, exist_ok=True)
    (sample_path / "step_7_translate").mkdir(parents=True, exist_ok=True)
    (sample_path / "step_8_typeset").mkdir(parents=True, exist_ok=True)

    (sample_path / "step_5_ocr" / "ocr_results.json").write_text("[]", encoding="utf-8")
    (sample_path / "step_6_layout" / "layout_constraints.json").write_text("[]", encoding="utf-8")
    (sample_path / "step_6_layout" / "rejected_layout_items.json").write_text("[]", encoding="utf-8")
    (sample_path / "step_7_translate" / "translation_results.json").write_text("[]", encoding="utf-8")
    (sample_path / "step_8_typeset" / "typeset_report.json").write_text("[]", encoding="utf-8")
    Image.new("RGB", (32, 32), (240, 240, 240)).save(sample_path / "step_8_typeset" / "final_output.png")


def _assert_active_cache_clear_is_guarded() -> None:
    original_root = runtime_bridge.SAMPLES_ROOT
    temp_root = original_root.parent / "_diagnostics_runtime_cache_guard"
    sample_name = "runtime_cache_guard_active"
    runtime_bridge.SAMPLES_ROOT = temp_root
    shutil.rmtree(temp_root, ignore_errors=True)
    try:
        _write_minimal_pass_artifacts(sample_name, "ja")
        runtime_bridge._mark_sample_active(sample_name)
        result = runtime_bridge.clear_runtime_output_cache()
        assert sample_name in result["skippedActiveSamples"], result
        sample_path = runtime_bridge.SAMPLES_ROOT / sample_name
        assert (sample_path / "step_6_layout" / "layout_constraints.json").exists()
        assert (sample_path / "step_8_typeset" / "final_output.png").exists()
    finally:
        runtime_bridge._mark_sample_inactive(sample_name)
        runtime_bridge.SAMPLES_ROOT = original_root
        shutil.rmtree(temp_root, ignore_errors=True)


def main() -> int:
    image_bytes = _png_bytes((12, 34, 56))
    language = "ja"
    sample_name, _ = runtime_bridge._write_runtime_sample(image_bytes, language, preserve_reusable_output=False)
    _write_minimal_pass_artifacts(sample_name, language)

    original_runner = runtime_bridge._run_runtime_pipeline

    def _unexpected_runner(*_args, **_kwargs):
        raise AssertionError("Runtime pipeline should not run when cached Step 8 output is reusable")

    runtime_bridge._run_runtime_pipeline = _unexpected_runner
    try:
        payload = {
            "imageData": f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}",
            "sourceLanguage": language,
            "targetLanguage": "en",
            "qualityProfile": "strict",
            "metadata": {"source": "runtime-output-cache-test"},
        }
        result = pipeline_bridge.run_pipeline_payload(payload)
        assert result["report"]["runtimeOutputCache"] == "hit"
        assert result["translatedImageDataUrl"].startswith("data:image/png;base64,")

        _assert_active_cache_clear_is_guarded()

        no_text_sample = f"{sample_name}_no_text"
        no_text_path = runtime_bridge.SAMPLES_ROOT / no_text_sample
        no_text_path.mkdir(parents=True, exist_ok=True)
        _write_no_renderable_artifacts(no_text_sample)
        no_text_report = runtime_bridge._collect_runtime_report(no_text_sample, language)
        runtime_bridge._assert_runtime_report_safe(no_text_report)
        assert no_text_report["outputSafety"] == "no_renderable_text", no_text_report

        # A blank/no-text outcome must never be served from cache on a later
        # request for the same image bytes -- otherwise a page that failed
        # once (e.g. before a vision-rescue fix landed, or before keys were
        # rotated) would be stuck blank forever even after the underlying
        # cause is fixed. _write_runtime_cache_meta must write a status other
        # than "pass" for it, and _has_reusable_runtime_output must decline.
        runtime_bridge._write_runtime_cache_meta(no_text_sample, language, no_text_report)
        assert runtime_bridge._has_reusable_runtime_output(no_text_sample, language) is False, (
            "a no_renderable_text result must not be reusable from cache"
        )
        # A genuine pass result, by contrast, must still be reusable -- this
        # guards against the fix accidentally becoming "never cache anything".
        assert runtime_bridge._has_reusable_runtime_output(sample_name, language) is True, (
            "a genuine pass result must remain reusable from cache"
        )
        shutil.rmtree(no_text_path, ignore_errors=True)

        print("runtime_output_cache_contract=pass")
        return 0
    finally:
        runtime_bridge._run_runtime_pipeline = original_runner
        shutil.rmtree(runtime_bridge.SAMPLES_ROOT / sample_name, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
