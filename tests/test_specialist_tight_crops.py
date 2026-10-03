# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for deterministic target-tight crop geometry."""

import pytest

from dataset.prepare_specialist_tight_crops import remap_box, tight_bounds


def test_tight_crop_contains_target_with_requested_margin():
    """A centered target must retain its box and symmetric margin after remapping."""
    box = [1, 0.5, 0.5, 0.2, 0.4]
    bounds = tight_bounds(box, width=1000, height=500, margin=0.25)
    mapped = remap_box(box, bounds, width=1000, height=500)

    assert bounds == (350, 100, 650, 400)
    assert mapped[1:3] == pytest.approx([0.5, 0.5])
    assert mapped[3:] == pytest.approx([2 / 3, 2 / 3])


def test_tight_crop_clips_at_image_boundary():
    """Boundary targets must produce valid clipped crop coordinates."""
    bounds = tight_bounds([0, 0.02, 0.03, 0.08, 0.10], width=100, height=100, margin=0.15)

    assert bounds[0] == 0
    assert bounds[1] == 0
    assert bounds[2] <= 100
    assert bounds[3] <= 100
