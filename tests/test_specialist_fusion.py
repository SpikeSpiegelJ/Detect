# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for mapping two-class specialist predictions into the nine-class detection space."""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("val_specialist_fusion", ROOT / "val_specialist_fusion.py")
FUSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FUSION)


class DummyModel(torch.nn.Module):
    """Return fixed predictions and record whether augmented inference was requested."""

    def __init__(self, prediction, names):
        super().__init__()
        self.prediction = prediction
        self.names = names
        self.stride = torch.tensor([32])
        self.yaml = {}
        self.augment = None

    def forward(self, _images, augment=False, **_kwargs):
        self.augment = augment
        return self.prediction


class SpecialistFusionTest(unittest.TestCase):
    def test_maps_scores_and_preserves_inference_modes(self):
        names = {
            0: "coupler",
            1: "antenna_s",
            2: "RRU",
            3: "Feeder_RRU",
            4: "antenna_b",
            5: "cut",
            6: "Feeder_antenna",
            7: "POWER",
            8: "BOX",
        }
        general_prediction = torch.zeros((1, 13, 2))
        specialist_prediction = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [0.8], [0.6]]])
        general = DummyModel(general_prediction, names)
        specialist = DummyModel(specialist_prediction, {0: "cut", 1: "Feeder_antenna"})

        fused, training_output = FUSION.SpecialistFusion(general, specialist, 0.5)(torch.zeros((1, 3, 32, 32)))

        self.assertIsNone(training_output)
        self.assertEqual(fused.shape, (1, 13, 3))
        torch.testing.assert_close(fused[:, :4, 2:], specialist_prediction[:, :4])
        self.assertAlmostEqual(fused[0, 4 + 5, 2].item(), 0.4)
        self.assertAlmostEqual(fused[0, 4 + 6, 2].item(), 0.3)
        self.assertTrue(general.augment)
        self.assertFalse(specialist.augment)


if __name__ == "__main__":
    unittest.main()
