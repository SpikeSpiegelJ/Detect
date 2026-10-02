# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validate a prepared LabelMe correction package and apply it to its mapped YOLO labels."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from LabelMeToYOLO import convert_labelme_to_yolo, serialize


ROOT = Path(__file__).resolve().parent
DEFAULT_RELABEL = ROOT / "relabel_v8_annotation_errors"


def parse_args():
    """Parse correction package and apply mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relabel", type=Path, default=DEFAULT_RELABEL, help="Prepared LabelMe correction package.")
    parser.add_argument("--images", nargs="+", help="Apply only these image names or stems from the package.")
    parser.add_argument("--apply", action="store_true", help="Back up and replace the mapped dataset labels.")
    return parser.parse_args()


def parse_yolo(text, label_path):
    """Validate existing YOLO labels and return their class IDs."""
    classes = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_number} must contain five values")
        class_id = int(values[0])
        coordinates = [float(value) for value in values[1:]]
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"{label_path}:{line_number} contains coordinates outside [0, 1]")
        classes.append(class_id)
    return classes


def load_changes(relabel, image_names=None):
    """Validate every mapping record before any dataset label is changed."""
    mapping_path = relabel / "mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, list) or not mapping:
        raise ValueError(f"No correction records found in {mapping_path}")
    if image_names:
        requested = {Path(name).stem for name in image_names}
        mapping = [record for record in mapping if Path(record["dataset_image"]).stem in requested]
        found = {Path(record["dataset_image"]).stem for record in mapping}
        if missing := requested - found:
            raise FileNotFoundError(f"Images are absent from the correction package: {', '.join(sorted(missing))}")

    changes, targets = [], set()
    for record in mapping:
        json_path = Path(record["labelme_json"]).resolve()
        copied_image = Path(record["image"]).resolve()
        dataset_image = Path(record["dataset_image"]).resolve()
        dataset_label = Path(record["dataset_label"]).resolve()
        if dataset_label in targets:
            raise ValueError(f"Duplicate dataset label in mapping: {dataset_label}")
        targets.add(dataset_label)
        for path in (json_path, copied_image, dataset_image, dataset_label):
            if not path.is_file():
                raise FileNotFoundError(path)

        converted_image, rows = convert_labelme_to_yolo(json_path)
        if converted_image.resolve() != copied_image:
            raise ValueError(f"{json_path} points to unexpected image {converted_image}")
        old_text = dataset_label.read_text(encoding="utf-8")
        old_classes = parse_yolo(old_text, dataset_label)
        new_text = serialize(rows)
        changes.append(
            {
                "record": record,
                "dataset_label": dataset_label,
                "old_text": old_text,
                "new_text": new_text,
                "changed": old_text != new_text,
                "old_classes": dict(sorted(Counter(old_classes).items())),
                "new_classes": dict(sorted(Counter(row[0] for row in rows).items())),
            }
        )
    return changes


def apply_changes(relabel, changes):
    """Back up original labels and atomically replace them with validated corrections."""
    backup = relabel / "applied_label_backup"
    backup_paths = [backup / change["record"]["split"] / change["dataset_label"].name for change in changes]
    if conflicts := [path for path in backup_paths if path.exists()]:
        raise FileExistsError(f"Labels were already applied: {', '.join(path.name for path in conflicts)}")

    for change, destination in zip(changes, backup_paths):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(change["dataset_label"], destination)

    for change in changes:
        temporary = change["dataset_label"].with_suffix(".txt.relabeling")
        temporary.write_text(change["new_text"], encoding="utf-8")
        temporary.replace(change["dataset_label"])

    report = [
        {
            "dataset_label": str(change["dataset_label"]),
            "split": change["record"]["split"],
            "changed": change["changed"],
            "old_classes": change["old_classes"],
            "new_classes": change["new_classes"],
        }
        for change in changes
    ]
    (relabel / "apply_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return backup


def main():
    """Validate the corrections, show their changes, and optionally apply them."""
    args = parse_args()
    relabel = args.relabel.resolve()
    changes = load_changes(relabel, args.images)
    print(
        json.dumps(
            [
                {
                    "label": change["dataset_label"].name,
                    "changed": change["changed"],
                    "old_classes": change["old_classes"],
                    "new_classes": change["new_classes"],
                }
                for change in changes
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    if not args.apply:
        print(f"Validated {len(changes)} corrections. Re-run with --apply to write them.")
        return
    backup = apply_changes(relabel, changes)
    print(f"Applied {len(changes)} corrected labels. Backup: {backup}")


if __name__ == "__main__":
    main()
