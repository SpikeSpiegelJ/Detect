from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("audit_annotation_provenance", ROOT / "dataset" / "audit_annotation_provenance.py")
AUDIT = module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def write_label(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(" ".join(map(str, row)) + "\n" for row in rows), encoding="utf-8")


def test_labels_equal_ignores_order_and_formatting(tmp_path):
    left, right = tmp_path / "left.txt", tmp_path / "right.txt"
    write_label(left, [(5, 0.1, 0.2, 0.3, 0.4), (6, 0.5, 0.6, 0.2, 0.1)])
    write_label(right, [(6, 0.500000, 0.600000, 0.200000, 0.100000), (5, 0.1, 0.2, 0.3, 0.4)])
    assert AUDIT.labels_equal(left, right)


def test_label_index_rejects_duplicate_base_names(tmp_path):
    write_label(tmp_path / "labels" / "train" / "same.txt", [(5, 0.5, 0.5, 0.2, 0.2)])
    write_label(tmp_path / "labels" / "val" / "same.txt", [(5, 0.5, 0.5, 0.2, 0.2)])
    try:
        AUDIT.label_index(tmp_path)
    except ValueError as error:
        assert "repeated base label names" in str(error)
    else:
        raise AssertionError("duplicate base label names were accepted")


def test_equivalent_delta_accepts_reconfirmed_box_and_extra_change(tmp_path):
    original, reviewed, current = (tmp_path / name for name in ("original.txt", "reviewed.txt", "current.txt"))
    base = (4, 0.5, 0.5, 0.2, 0.2)
    write_label(original, [base])
    write_label(reviewed, [base, (5, 0.30, 0.70, 0.10, 0.20)])
    write_label(current, [base, (5, 0.305, 0.70, 0.10, 0.20), (5, 0.8, 0.8, 0.05, 0.05)])
    assert AUDIT.equivalent_delta(original, reviewed, current)
