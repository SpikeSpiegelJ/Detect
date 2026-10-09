"""Audit whether reviewed OOF label corrections are present in a later dataset copy."""

from __future__ import annotations

import csv
import json
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    """Parse annotation-provenance audit arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--oof-report", type=Path, required=True)
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--current-report", type=Path)
    parser.add_argument("--selection-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def labels_equal(left: Path, right: Path, tolerance: float = 1e-6) -> bool:
    """Return whether two YOLO label files contain the same boxes, independent of row order and formatting."""
    left_rows = sorted(tuple(map(float, line.split())) for line in left.read_text(encoding="utf-8").splitlines())
    right_rows = sorted(tuple(map(float, line.split())) for line in right.read_text(encoding="utf-8").splitlines())
    return len(left_rows) == len(right_rows) and bool(
        np.allclose(np.asarray(left_rows, dtype=float), np.asarray(right_rows, dtype=float), atol=tolerance, rtol=0)
    )


def read_rows(path: Path) -> list[tuple[float, ...]]:
    """Read numeric YOLO rows."""
    return [tuple(map(float, line.split())) for line in path.read_text(encoding="utf-8").splitlines()]


def unmatched_rows(left: list[tuple[float, ...]], right: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
    """Return rows in ``left`` that have no exact semantic match in ``right``."""
    remaining = list(right)
    unmatched = []
    for row in left:
        match = next(
            (
                index
                for index, candidate in enumerate(remaining)
                if np.allclose(row, candidate, atol=1e-6, rtol=0)
            ),
            None,
        )
        if match is None:
            unmatched.append(row)
        else:
            remaining.pop(match)
    return unmatched


def box_iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Return IoU for two normalized YOLO rows with an equal class id."""
    if int(left[0]) != int(right[0]):
        return 0.0
    _, left_x, left_y, left_w, left_h = left
    _, right_x, right_y, right_w, right_h = right
    left_box = (left_x - left_w / 2, left_y - left_h / 2, left_x + left_w / 2, left_y + left_h / 2)
    right_box = (right_x - right_w / 2, right_y - right_h / 2, right_x + right_w / 2, right_y + right_h / 2)
    intersection = max(0.0, min(left_box[2], right_box[2]) - max(left_box[0], right_box[0])) * max(
        0.0, min(left_box[3], right_box[3]) - max(left_box[1], right_box[1])
    )
    union = left_w * left_h + right_w * right_h - intersection
    return intersection / union if union else 0.0


def equivalent_delta(original: Path, reviewed: Path, current: Path, iou_threshold: float = 0.85) -> bool:
    """Return whether current labels contain an equivalent reviewed delta plus optional independent changes."""
    original_rows, reviewed_rows, current_rows = map(read_rows, (original, reviewed, current))
    reviewed_added = unmatched_rows(reviewed_rows, original_rows)
    reviewed_removed = unmatched_rows(original_rows, reviewed_rows)
    current_added = unmatched_rows(current_rows, original_rows)
    if not reviewed_added and not reviewed_removed:
        return False
    for row in reviewed_added:
        match = max((box_iou(row, candidate) for candidate in current_added), default=0.0)
        if match < iou_threshold:
            return False
    return not any(
        np.allclose(removed, candidate, atol=1e-6, rtol=0)
        for removed in reviewed_removed
        for candidate in current_rows
    )


def label_index(root: Path) -> dict[str, Path]:
    """Index unique base train/validation label files by filename."""
    indexed = defaultdict(list)
    for split in ("train", "val"):
        for path in sorted((root / "labels" / split).glob("*.txt")):
            indexed[path.name].append(path)
    repeated = {name: paths for name, paths in indexed.items() if len(paths) != 1}
    if repeated:
        raise ValueError(f"Current dataset has repeated base label names: {sorted(repeated)[:5]}")
    return {name: paths[0] for name, paths in indexed.items()}


def read_review_counts(path: Path) -> dict:
    """Count normalized review rows and unique source images without treating the selection as random."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    decisions = Counter(row["decision"] for row in rows)
    images = defaultdict(set)
    for row in rows:
        images[row["decision"]].add(row["source_image"])
    return {
        "rows": len(rows),
        "unique_images": len({row["source_image"] for row in rows}),
        "rows_by_decision": dict(sorted(decisions.items())),
        "unique_images_by_decision": {key: len(value) for key, value in sorted(images.items())},
    }


def main():
    """Compare original, reviewed, and current labels and write a machine-readable provenance report."""
    args = parse_args()
    oof_report_path = args.oof_report.resolve()
    oof_report = json.loads(oof_report_path.read_text(encoding="utf-8"))
    original_root = Path(oof_report["source"]).resolve()
    reviewed_root = Path(oof_report["output"]).resolve()
    current_root = args.current_root.resolve()
    current_labels = label_index(current_root)

    records = []
    status_counts = Counter()
    transition_counts = Counter()
    added_class_counts = Counter()
    corrected_class_counts = Counter()
    no_op_files = []
    missing_files = []
    for reviewed_name, changes in sorted(oof_report["changes"].items()):
        reviewed_path = Path(reviewed_name).resolve()
        relative = reviewed_path.relative_to(reviewed_root)
        original_path = original_root / relative
        current_path = current_labels.get(reviewed_path.name)
        if current_path is None:
            missing_files.append(reviewed_path.name)
            continue
        original_equals_reviewed = labels_equal(original_path, reviewed_path)
        current_equals_original = labels_equal(current_path, original_path)
        current_equals_reviewed = labels_equal(current_path, reviewed_path)
        if original_equals_reviewed:
            status = "reviewed_no_semantic_change"
            no_op_files.append(reviewed_path.name)
        elif current_equals_reviewed:
            status = "inherited_reviewed_correction"
        elif equivalent_delta(original_path, reviewed_path, current_path):
            status = "equivalent_current_correction"
        elif current_equals_original:
            status = "still_original"
        else:
            status = "different_from_both"
        status_counts[status] += 1
        records.append(
            {
                "label": reviewed_path.name,
                "original_split": relative.parts[1],
                "current_split": current_path.parent.name,
                "status": status,
                "changes": changes,
            }
        )
        for change in changes:
            if change["action"] == "add_pred_box":
                added_class_counts[change["class"]] += 1
                corrected_class_counts[change["class"]] += 1
            else:
                transition_counts[f'{change["from"]}->{change["to"]}'] += 1
                corrected_class_counts[change["to"]] += 1

    current_report = None
    current_changed_names = set()
    if args.current_report:
        current_report = json.loads(args.current_report.resolve().read_text(encoding="utf-8"))
        current_changed_names = {Path(path).name for path in current_report.get("changed", {})}
    overlap = sorted(current_changed_names & {Path(path).name for path in oof_report["changes"]})

    selection = None
    if args.selection_summary:
        selection_raw = json.loads(args.selection_summary.resolve().read_text(encoding="utf-8"))
        selection = {
            key: selection_raw.get(key)
            for key in (
                "method",
                "development_images",
                "development_unique_images",
                "independent_test_images_checked_only_for_leakage",
                "folds",
                "all_error_images",
                "selected_review_images",
                "selected_review_rows",
                "all_errors",
                "selected_errors",
            )
            if key in selection_raw
        }

    result = {
        "scope": {
            "review_sampling": (
                "model-error-mined OOF candidates; not a random sample and not a dataset-wide error-rate estimate"
            ),
            "original_root": str(original_root),
            "reviewed_root": str(reviewed_root),
            "current_root": str(current_root),
            "test_labels_changed_by_oof_review": not bool(oof_report["test_label_hashes_equal"]),
        },
        "oof_review": {
            "decisions": oof_report["decisions"],
            "actions": oof_report["actions"],
            "reviewed_corrections": oof_report["corrections"],
            "applied_corrections": oof_report["applied_corrections"],
            "changed_labels": oof_report["changed_labels"],
            "normalized_review": read_review_counts(Path(oof_report["normalized_review"])),
            "selection_summary": selection,
            "transition_counts": dict(sorted(transition_counts.items())),
            "added_class_counts": dict(sorted(added_class_counts.items())),
            "corrected_class_counts": dict(sorted(corrected_class_counts.items())),
        },
        "current_dataset_comparison": {
            "reviewed_changed_files_checked": len(records),
            "status_counts": dict(sorted(status_counts.items())),
            "missing_files": missing_files,
            "semantic_no_op_files": no_op_files,
            "current_review_changed_files": len(current_changed_names),
            "overlap_with_oof_review": overlap,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"output": str(args.output.resolve()), **result["current_dataset_comparison"]}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
