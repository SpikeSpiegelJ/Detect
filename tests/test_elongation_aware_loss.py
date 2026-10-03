# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for normalized elongation-aware box regression weighting."""

import pytest
import torch

from ultralytics.utils.loss import BboxLoss


def sample_targets():
    """Return one square and one 8:1 target with equal quality scores."""
    boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 8.0, 1.0]]])
    scores = torch.ones(1, 2, 1)
    foreground = torch.ones(1, 2, dtype=torch.bool)
    return boxes, scores, foreground


def test_zero_gain_preserves_original_regression_weights():
    """The default setting must be exactly backward compatible."""
    boxes, scores, foreground = sample_targets()

    weights = BboxLoss(reg_max=1).regression_weight(boxes, scores, foreground)

    assert torch.equal(weights, torch.ones(2, 1))


def test_elongation_emphasis_preserves_total_weight():
    """Elongated targets receive more relative weight without increasing the batch total."""
    boxes, scores, foreground = sample_targets()

    weights = BboxLoss(reg_max=1, elongation_gain=1.0, elongation_threshold=4.0).regression_weight(
        boxes, scores, foreground
    )

    assert weights[1] > weights[0]
    assert weights.sum() == pytest.approx(2.0)


@pytest.mark.parametrize("gain,threshold", [(-1.0, 4.0), (0.0, 1.0)])
def test_invalid_elongation_settings_fail_early(gain, threshold):
    """Invalid weighting settings must fail before training starts."""
    with pytest.raises(ValueError, match="elongation"):
        BboxLoss(reg_max=1, elongation_gain=gain, elongation_threshold=threshold)
