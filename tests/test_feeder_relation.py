# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for endpoint-device relation calibration."""

import torch

from val_feeder_relation import FeederRelationCalibration, SpatialFeederRelationCalibration


class FakeGeneral(torch.nn.Module):
    """Minimal detector metadata and prediction output for calibration tests."""

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
    stride = torch.tensor([32])
    yaml = {}


def test_relation_calibration_favors_feeder_near_supported_endpoint_class():
    """RRU-dominant evidence must raise RRU-side and lower antenna-side feeder scores."""
    model = FeederRelationCalibration(FakeGeneral(), strength=1.0)
    prediction = torch.zeros(1, 13, 3)
    prediction[:, 4 + 2, 0] = 0.9
    prediction[:, 4 + 4, 1] = 0.1
    prediction[:, 4 + 3, 2] = 0.4
    prediction[:, 4 + 6, 2] = 0.4

    calibrated = model.calibrate(prediction)

    assert calibrated[0, 4 + 3, 2] > 0.4
    assert calibrated[0, 4 + 6, 2] < 0.4
    assert torch.equal(calibrated[:, :4], prediction[:, :4])


def test_zero_strength_returns_identical_prediction():
    """The zero-strength ablation must exactly reproduce the uncalibrated tensor."""
    model = FeederRelationCalibration(FakeGeneral(), strength=0.0)
    prediction = torch.rand(2, 13, 5)

    assert model.calibrate(prediction) is prediction


def test_spatial_relation_uses_candidate_endpoint_proximity():
    """Equal global support must favor the endpoint located next to the feeder candidate."""
    model = SpatialFeederRelationCalibration(FakeGeneral(), strength=0.5, temperature=0.05, topk=2)
    prediction = torch.zeros(1, 13, 3)
    prediction[:, :4, 0] = torch.tensor([100.0, 100.0, 40.0, 40.0])
    prediction[:, :4, 1] = torch.tensor([120.0, 100.0, 40.0, 40.0])
    prediction[:, :4, 2] = torch.tensor([900.0, 900.0, 40.0, 40.0])
    prediction[:, 4 + 3, 0] = 0.4
    prediction[:, 4 + 6, 0] = 0.4
    prediction[:, 4 + 2, 1] = 0.9
    prediction[:, 4 + 4, 2] = 0.9

    calibrated = model.calibrate(prediction, image_size=(1000, 1000))

    assert calibrated[0, 4 + 3, 0] > 0.4
    assert calibrated[0, 4 + 6, 0] < 0.4
