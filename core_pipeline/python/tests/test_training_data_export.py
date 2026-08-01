from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_ROOT = PROJECT_ROOT / "python" / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

from training_data_export import export_runtime_training_case


class TrainingDataExportTests(unittest.TestCase):
    def test_exports_review_case_with_label_template_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            samples_root = root / "samples"
            sample = samples_root / "sample_a"
            (sample / "step_5_ocr").mkdir(parents=True)
            (sample / "step_6_layout").mkdir()
            (sample / "step_7_translate").mkdir()
            (sample / "step_8_typeset").mkdir()
            (sample / "step_4_final").mkdir()

            Image.new("RGB", (120, 120), "white").save(sample / "input.jpg")
            Image.new("RGB", (120, 120), "white").save(sample / "step_4_final" / "inpainted_result.jpg")
            Image.new("RGB", (120, 120), "white").save(sample / "step_8_typeset" / "final_output.png")

            (sample / "step_5_ocr" / "ocr_results.json").write_text(
                json.dumps([
                    {
                        "id": 0,
                        "text": "테스트",
                        "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 40},
                        "green_box": {"x1": 8, "y1": 8, "x2": 55, "y2": 48},
                        "ocr_provider": "test",
                    }
                ]),
                encoding="utf-8",
            )
            (sample / "step_6_layout" / "layout_constraints.json").write_text(
                json.dumps([
                    {
                        "id": 0,
                        "text": "테스트",
                        "red_box": [10, 10, 50, 40],
                        "green_box": [8, 8, 55, 48],
                        "erase_boxes": [],
                        "bubble_idx": -1,
                        "route": "floating_dialogue",
                    }
                ]),
                encoding="utf-8",
            )
            (sample / "step_6_layout" / "rejected_layout_items.json").write_text("[]", encoding="utf-8")
            (sample / "step_7_translate" / "translation_results.json").write_text(
                json.dumps([
                    {
                        "id": 0,
                        "jp_text": "테스트",
                        "en_text": "",
                        "translation_source": "fallback",
                    }
                ]),
                encoding="utf-8",
            )
            (sample / "step_8_typeset" / "typeset_report.json").write_text("[]", encoding="utf-8")

            report = {
                "sampleName": "sample_a",
                "sourceLanguage": "ko",
                "reviewWarnings": ["translations_below_layout_constraints=0/1"],
                "outputSafety": "review",
            }
            with patch.dict(
                os.environ,
                {
                    "FMT_TRAINING_DATA_EXPORT": "review",
                    "FMT_TRAINING_DATA_EXPORT_ROOT": str(root / "training"),
                },
                clear=False,
            ):
                exported = export_runtime_training_case(samples_root, "sample_a", "ko", report)

            self.assertIsNotNone(exported)
            assert exported is not None
            case_dir = Path(exported["path"])
            self.assertTrue((case_dir / "case_manifest.json").exists())
            self.assertTrue((case_dir / "labels_pending.json").exists())
            labels = json.loads((case_dir / "labels_pending.json").read_text(encoding="utf-8"))
            self.assertEqual(labels["labels"][0]["correctOcrText"], "")
            self.assertEqual(labels["labels"][0]["regionKind"], "dialogue")
            self.assertTrue((case_dir / "regions" / "region_000_input.jpg").exists())


if __name__ == "__main__":
    unittest.main()
