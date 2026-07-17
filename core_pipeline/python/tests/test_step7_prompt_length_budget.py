from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _rel in ("python/common", "python/steps"):
    _path = str(PROJECT_ROOT / _rel)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_step7_translate as step7  # noqa: E402


SAMPLE_ITEMS = [
    {"id": 0, "text": "こんにちは世界", "box": {"width": 120, "height": 60}},
    {"id": 1, "text": "はあ?", "box": {"width": 30, "height": 20}, "short_fragment": True},
]


class PromptLengthBudgetFlagTests(unittest.TestCase):
    def test_flag_off_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STEP7_PROMPT_LENGTH_BUDGET", None)
            self.assertFalse(step7._prompt_length_budgets_enabled())

    def test_flag_off_prompt_has_no_max_chars_or_budget_instruction(self) -> None:
        with patch.dict(os.environ, {"STEP7_PROMPT_LENGTH_BUDGET": "0"}, clear=False):
            prompt = step7._translation_prompt(SAMPLE_ITEMS, "sample_test")
        self.assertNotIn("max_chars", prompt)
        self.assertNotIn("approximate character budget", prompt)
        items_json = json.loads(prompt.split("Items: ", 1)[1])
        for item in items_json:
            self.assertNotIn("max_chars", item)

    def test_flag_on_adds_max_chars_matching_box_budget(self) -> None:
        with patch.dict(os.environ, {"STEP7_PROMPT_LENGTH_BUDGET": "1"}, clear=False):
            prompt = step7._translation_prompt(SAMPLE_ITEMS, "sample_test")
        self.assertIn("approximate character budget", prompt)
        items_json = json.loads(prompt.split("Items: ", 1)[1])
        by_id = {item["id"]: item for item in items_json}
        self.assertEqual(by_id[0]["max_chars"], step7._translation_box_budget(SAMPLE_ITEMS[0]))
        self.assertEqual(by_id[1]["max_chars"], step7._translation_box_budget(SAMPLE_ITEMS[1]))

    def test_flag_off_prompt_byte_identical_regardless_of_source_case(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STEP7_PROMPT_LENGTH_BUDGET", None)
            prompt_a = step7._translation_prompt(SAMPLE_ITEMS, "sample_test")
            os.environ["STEP7_PROMPT_LENGTH_BUDGET"] = "off"
            prompt_b = step7._translation_prompt(SAMPLE_ITEMS, "sample_test")
        self.assertEqual(prompt_a, prompt_b)


if __name__ == "__main__":
    unittest.main()
