"""Tests for geometry-aware weak-class error review."""

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset"))
spec = importlib.util.spec_from_file_location("weak_error_review", ROOT / "dataset/review_v8_weak_class_errors.py")
weak_error_review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weak_error_review)


def test_geometry_and_endpoint_profiles():
    """Geometry is input-scaled and feeder identity uses its defining endpoint."""
    box = np.array([100, 100, 500, 150])
    geometry = weak_error_review.geometry_profile(box, width=1000, height=500, imgsz=1000)
    assert geometry["gt_aspect_ratio"] == 8
    assert geometry["gt_short_side_px_at_imgsz"] == 50
    assert geometry["gt_elongated"]
    assert not geometry["gt_small_at_imgsz"]

    gt_classes = np.array([3, 2])
    gt_boxes = np.array([[100, 100, 500, 150], [50, 50, 250, 300]])
    names = {2: "RRU", 3: "Feeder_RRU"}
    endpoint = weak_error_review.endpoint_profile(
        "Feeder_RRU", box, gt_classes, gt_boxes, names, width=1000, height=500
    )
    assert endpoint["endpoint_visible"]
    assert endpoint["endpoint_min_center_distance_normalized"] > 0
    assert weak_error_review.endpoint_profile("cut", box, gt_classes, gt_boxes, names, 1000, 500) == {}
    assert weak_error_review.distribution_summary([1, 2, 3]) == {"q25": 1.5, "q50": 2.0, "q75": 2.5}
