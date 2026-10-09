# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for scale-adaptive NWD localization and hardest-negative class separation."""

from copy import deepcopy

import pytest
import torch

from ultralytics.nn.modules import ContextGate
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss


def test_scale_adaptive_nwd_only_targets_small_boxes():
    """NWD mixing should decrease continuously to zero at the configured scale threshold."""
    loss = BboxLoss(reg_max=1, nwd_gain=0.5, nwd_threshold=0.04)
    boxes = torch.tensor([[0.0, 0.0, 16.0, 16.0], [0.0, 0.0, 64.0, 64.0]])

    weights = loss.small_target_weight(boxes, torch.tensor([640.0, 640.0]))

    assert weights[0] == pytest.approx(0.1875)
    assert weights[1] == 0


def test_zero_nwd_gain_preserves_iou_only_default():
    """The default setting must not mix NWD into box localization."""
    boxes = torch.tensor([[0.0, 0.0, 16.0, 16.0]])

    weights = BboxLoss(reg_max=1).small_target_weight(boxes, torch.tensor([640.0, 640.0]))

    assert torch.equal(weights, torch.zeros(1))


def test_nwd_loss_increases_with_center_error():
    """An identical box should have lower NWD loss than a shifted box."""
    target = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    shifted = target + torch.tensor([[5.0, 0.0, 5.0, 0.0]])

    identical_loss = BboxLoss.normalized_wasserstein_loss(target, target, 12.8)
    shifted_loss = BboxLoss.normalized_wasserstein_loss(shifted, target, 12.8)

    assert identical_loss.item() < 1e-4
    assert shifted_loss > identical_loss


@pytest.mark.parametrize(
    ("gain", "threshold", "constant"),
    [(-0.1, 0.04, 12.8), (1.1, 0.04, 12.8), (0.5, 0.0, 12.8), (0.5, 0.04, 0.0)],
)
def test_invalid_nwd_settings_fail_early(gain, threshold, constant):
    """Invalid NWD settings should fail before training starts."""
    with pytest.raises(ValueError, match="nwd"):
        BboxLoss(reg_max=1, nwd_gain=gain, nwd_threshold=threshold, nwd_constant=constant)


def test_class_margin_penalizes_only_ambiguous_foreground():
    """The margin term should ignore separated logits and penalize a close hardest negative."""
    criterion = object.__new__(v8DetectionLoss)
    criterion.nc = 3
    criterion.class_margin = 0.5
    targets = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    foreground = torch.tensor([[True, True]])

    separated = torch.tensor([[[2.0, 0.0, -1.0], [2.0, 0.0, -1.0]]])
    ambiguous = torch.tensor([[[2.0, 0.0, -1.0], [0.2, 0.1, -1.0]]])

    assert criterion.class_margin_loss(separated, targets, foreground) == 0
    assert criterion.class_margin_loss(ambiguous, targets, foreground) == pytest.approx(0.2)


def test_class_margin_handles_empty_foreground():
    """Images without assigned positives should return a differentiable zero."""
    criterion = object.__new__(v8DetectionLoss)
    criterion.nc = 3
    criterion.class_margin = 0.5
    predictions = torch.randn(1, 2, 3, requires_grad=True)

    result = criterion.class_margin_loss(
        predictions, torch.zeros_like(predictions), torch.zeros(1, 2, dtype=torch.bool)
    )
    result.backward()

    assert result == 0
    assert predictions.grad is not None


def test_context_gate_starts_as_identity_and_retains_logits():
    """Zero-initialized gate logits should preserve the pretrained fine feature at initialization."""
    gate = ContextGate([8, 16])
    gate.capture_attention = True
    fine = torch.randn(2, 8, 8, 8)
    context = torch.randn(2, 16, 4, 4)

    output = gate([fine, context])

    assert torch.equal(output, fine)
    assert gate.gate_logits.shape == (2, 1, 8, 8)


def test_context_gate_does_not_retain_graph_without_supervision():
    """Ordinary inference must leave the module safe for EMA deepcopy."""
    gate = ContextGate([8, 16])

    gate([torch.randn(2, 8, 8, 8), torch.randn(2, 16, 4, 4)])

    assert gate.gate_logits is None
    deepcopy(gate)


def test_context_foreground_supervision_updates_gate():
    """Box-derived foreground supervision should produce finite gradients for the spatial gate."""
    gate = ContextGate([8, 16])
    gate.capture_attention = True
    gate([torch.randn(2, 8, 8, 8), torch.randn(2, 16, 4, 4)])
    criterion = object.__new__(v8DetectionLoss)
    criterion.context_gates = [gate]
    batch = {
        "batch_idx": torch.tensor([0, 1]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25], [0.25, 0.25, 0.125, 0.125]]),
    }

    loss = criterion.context_foreground_loss(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert gate.gate.weight.grad is not None
    assert torch.isfinite(gate.gate.weight.grad).all()


def test_csl_model_yaml_builds_context_gate():
    """The CSL model definition should parse the two-input gate and preserve three detection scales."""
    model = DetectionModel("ultralytics/cfg/models/26/yolo26s-csl-o2m.yaml", nc=9, verbose=False)

    assert isinstance(model.model[-2], ContextGate)
    assert model.model[-1].nl == 3
