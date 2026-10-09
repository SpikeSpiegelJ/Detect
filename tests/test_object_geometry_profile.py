# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for input-scale object geometry profiling."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset"))
SPEC = importlib.util.spec_from_file_location("profile_object_geometry", ROOT / "dataset/profile_object_geometry.py")
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


def test_label_geometry_respects_letterbox_scale():
    """A normalized box must be measured after uniform long-side resizing."""
    class_id, row = PROFILE.label_geometry("5 0.5 0.5 0.1 0.2", (1000, 2000), 1000)

    assert class_id == 5
    assert row["width_px"] == 100
    assert row["height_px"] == 100
    assert row["area_px2"] == 10000
    assert row["aspect_ratio"] == 1


def test_summary_reports_small_and_elongated_counts():
    """Fixed geometry bins must remain stable for paper comparisons."""
    rows = [
        PROFILE.label_geometry("0 0.5 0.5 0.01 0.08", (1000, 1000), 1000)[1],
        PROFILE.label_geometry("0 0.5 0.5 0.20 0.20", (1000, 1000), 1000)[1],
    ]

    summary = PROFILE.summarize(rows)

    assert summary["size_bins"]["small_lt_32sq"] == 1
    assert summary["size_bins"]["large_ge_96sq"] == 1
    assert summary["elongation_bins"]["aspect_ge_8"] == 1
