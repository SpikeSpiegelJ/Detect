# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the read-only dataset split integrity audit."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("audit_split_integrity", ROOT / "dataset/audit_split_integrity.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class DatasetSplitIntegrityTest(unittest.TestCase):
    def test_identity_uses_embedded_source_hash(self):
        first = Path("train__base_0123456789abcdefabcd.jpg")
        second = Path("crop_cut_base_0123456789abcdefabcd_00.jpg")
        self.assertEqual(AUDIT.canonical_identity(first), "0123456789abcdefabcd")
        self.assertEqual(AUDIT.canonical_identity(first), AUDIT.canonical_identity(second))

    def test_cross_split_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.full((40, 60, 3), 127, dtype=np.uint8)
            first, second = root / "a.jpg", root / "b.jpg"
            cv2.imwrite(str(first), image)
            cv2.imwrite(str(second), image)
            samples = [
                {"split": "train", "image": first, "signature": AUDIT.image_signature(first)},
                {"split": "test", "image": second, "signature": AUDIT.image_signature(second)},
            ]
            exact, near = AUDIT.cross_split_pairs(samples, 4, 12.0)
            self.assertEqual(len(exact), 1)
            self.assertFalse(near)

    def test_labels_report_invalid_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "sample.txt"
            label.write_text("0 0.5 0.5 0.2 0.2\n1 1.2 0.5 0.2 0.2\n", encoding="utf-8")
            counts, errors = AUDIT.read_labels(label, 2)
            self.assertEqual(counts, {0: 1, 1: 1})
            self.assertTrue(any("invalid normalized coordinates" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
