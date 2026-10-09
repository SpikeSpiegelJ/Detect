from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("merge_reviewed_corrections", ROOT / "dataset" / "merge_reviewed_corrections.py")
MERGE = module_from_spec(SPEC)
SPEC.loader.exec_module(MERGE)


def write_label(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(" ".join(map(str, row)) + "\n" for row in rows), encoding="utf-8")


def test_merge_label_replaces_unchanged_original(tmp_path):
    original, reviewed, current, destination = (tmp_path / name for name in ("o.txt", "r.txt", "c.txt", "d.txt"))
    base = [(6, 0.5, 0.5, 0.2, 0.2)]
    write_label(original, base)
    write_label(reviewed, [*base, (5, 0.2, 0.2, 0.1, 0.1)])
    write_label(current, base)
    write_label(destination, base)
    assert MERGE.merge_label(original, reviewed, current, destination) == "applied_reviewed_label"
    assert MERGE.labels_equal(reviewed, destination)


def test_merge_label_preserves_equivalent_independent_review(tmp_path):
    original, reviewed, current, destination = (tmp_path / name for name in ("o.txt", "r.txt", "c.txt", "d.txt"))
    base = [(4, 0.5, 0.5, 0.2, 0.2)]
    write_label(original, base)
    write_label(reviewed, [*base, (5, 0.30, 0.70, 0.10, 0.20)])
    write_label(current, [*base, (5, 0.305, 0.70, 0.10, 0.20)])
    write_label(destination, [*base, (5, 0.305, 0.70, 0.10, 0.20)])
    assert MERGE.merge_label(original, reviewed, current, destination) == "already_equivalent"
    assert MERGE.labels_equal(current, destination)


def test_merge_label_rejects_unrelated_current_change(tmp_path):
    original, reviewed, current, destination = (tmp_path / name for name in ("o.txt", "r.txt", "c.txt", "d.txt"))
    write_label(original, [(4, 0.5, 0.5, 0.2, 0.2)])
    write_label(reviewed, [(5, 0.5, 0.5, 0.2, 0.2)])
    write_label(current, [(6, 0.5, 0.5, 0.2, 0.2)])
    write_label(destination, [(6, 0.5, 0.5, 0.2, 0.2)])
    with pytest.raises(RuntimeError, match="Conflicting current label"):
        MERGE.merge_label(original, reviewed, current, destination)
