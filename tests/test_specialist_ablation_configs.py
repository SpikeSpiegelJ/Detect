# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for specialist ablation label classification."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_specialist_ablation_configs", ROOT / "dataset/prepare_specialist_ablation_configs.py"
)
ABLATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ABLATION)


class SpecialistAblationConfigTest(unittest.TestCase):
    def test_label_classes_distinguishes_positive_and_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            positive = Path(directory) / "positive.txt"
            negative = Path(directory) / "negative.txt"
            positive.write_text("0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n", encoding="utf-8")
            negative.write_text("", encoding="utf-8")
            self.assertEqual(ABLATION.label_classes(positive), [0, 1])
            self.assertEqual(ABLATION.label_classes(negative), [])


if __name__ == "__main__":
    unittest.main()
