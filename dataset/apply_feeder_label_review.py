# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Apply reviewed Feeder predictions and manually corrected LabelMe annotations to the V7 train labels."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.loaders import LoadImagesAndVideos, SourceTypes  # noqa: E402

from build_focus_hardneg_manifest import box_iou, label_path, load_labels, xywhn_to_xyxy  # noqa: E402


DATASET_ROOT = PROJECT_ROOT / "dataset" / "repartition_v7_dataset8_9c"
DEFAULT_REVIEW = PROJECT_ROOT / "runs" / "audit" / "feeder_hard_negatives_control" / "review.csv"
DEFAULT_AUDIT = DATASET_ROOT / "train_feeder_errors_audit.json"
DEFAULT_RELABEL = PROJECT_ROOT / "dataset" / "relabel_feeder_annotation_errors"
DECISIONS = {"missing_label", "true_negative", "wrong_class", "uncertain"}


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--relabel", type=Path, default=DEFAULT_RELABEL)
    parser.add_argument("--device", default="0")
    parser.add_argument("--apply", action="store_true", help="Write validated corrections and normalize review.csv.")
    return parser.parse_args()


def load_names():
    """Load class names and their IDs from the V7 dataset descriptor."""
    path = PROJECT_ROOT / "dataset" / "data_repartition_v7_dataset8_9c.yaml"
    with path.open(encoding="utf-8") as stream:
        names = yaml.safe_load(stream)["names"]
    names = {int(class_id): name for class_id, name in names.items()}
    return names, {name: class_id for class_id, name in names.items()}


def load_review(path):
    """Load review rows and accept decisions accidentally entered in correct_class."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if not rows:
            raise ValueError(f"Review file is empty: {path}")
        first_column = reader.fieldnames[0]

    for row in rows:
        row["index"] = row.pop(first_column)
        decision = row["decision"].strip()
        correct_class = row["correct_class"].strip()
        if decision.upper() == "TODO" and correct_class in DECISIONS:
            row["decision"] = correct_class
            row["correct_class"] = ""
        if row["decision"] not in DECISIONS:
            raise ValueError(f"Row {row['index']} has invalid decision: {row['decision']!r}")
        if row["decision"] == "missing_label" and not row["correct_class"]:
            row["correct_class"] = row["flagged_classes"]
    return rows


def parse_yolo(path):
    """Read a YOLO detection label into numeric rows."""
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{path}:{line_number} must contain five values")
        rows.append((int(values[0]), *(float(value) for value in values[1:])))
    return rows


def labelme_to_yolo(path, name_to_id):
    """Convert LabelMe rectangles or polygons to YOLO detection boxes."""
    data = json.loads(path.read_text(encoding="utf-8"))
    width, height = int(data["imageWidth"]), int(data["imageHeight"])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions in {path}")
    boxes = []
    for index, shape in enumerate(data["shapes"], 1):
        label = shape["label"]
        if label not in name_to_id:
            raise ValueError(f"{path}: shape {index} has unknown class {label!r}")
        points = shape.get("points", [])
        if shape.get("shape_type") not in {"rectangle", "polygon"} or len(points) < 2:
            raise ValueError(f"{path}: shape {index} must be a rectangle or polygon")
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        x1, y1 = max(0.0, min(xs)), max(0.0, min(ys))
        x2, y2 = min(float(width), max(xs)), min(float(height), max(ys))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"{path}: shape {index} has an empty bounding box")
        boxes.append(
            (
                name_to_id[label],
                (x1 + x2) / (2 * width),
                (y1 + y2) / (2 * height),
                (x2 - x1) / width,
                (y2 - y1) / height,
            )
        )
    return boxes


def load_manual_relabels(folder, rows, name_to_id):
    """Load manual replacements and align their review decisions with the corrected labels."""
    mapping = json.loads((folder / "mapping.json").read_text(encoding="utf-8"))
    row_by_stem = {Path(row["source_image"]).stem: row for row in rows}
    replacements, changes = {}, {}
    for record in mapping:
        dataset_label = Path(record["dataset_label"]).resolve()
        json_path = Path(record["labelme_json"]).resolve()
        old_boxes = parse_yolo(dataset_label)
        new_boxes = labelme_to_yolo(json_path, name_to_id)
        stem = dataset_label.stem
        if stem not in row_by_stem:
            raise ValueError(f"Manual correction is absent from review.csv: {stem}")
        old_counts = Counter(box[0] for box in old_boxes)
        new_counts = Counter(box[0] for box in new_boxes)
        added = new_counts - old_counts
        removed = old_counts - new_counts
        row = row_by_stem[stem]
        row["decision"] = "wrong_class" if added and removed else "missing_label"
        row["correct_class"] = "; ".join(
            name for name, class_id in name_to_id.items() if added[class_id]
        )
        replacements[dataset_label] = new_boxes
        changes[stem] = {"added": dict(added), "removed": dict(removed)}
    return replacements, changes


def expected_prediction_counts(row, name_to_id):
    """Count the red predictions recorded in one review row by class."""
    counts = Counter()
    for prediction in row["predictions"].split(";"):
        name, _ = prediction.strip().rsplit(":", 1)
        counts[name_to_id[name]] += 1
    return counts


def reproduce_approved_boxes(rows, audit, name_to_id, device, skip_stems):
    """Reproduce model boxes accepted as missing labels and verify them against review.csv."""
    selected = [
        row
        for row in rows
        if row["decision"] == "missing_label" and Path(row["source_image"]).stem not in skip_stems
    ]
    model = YOLO(Path(audit["weights"]))
    parameters = audit["parameters"]
    source = LoadImagesAndVideos([row["source_image"] for row in selected], batch=parameters["batch"])
    source.source_type = SourceTypes()
    results = model.predict(
        source=source,
        imgsz=parameters["imgsz"],
        batch=parameters["batch"],
        device=device,
        conf=parameters["candidate_conf"],
        iou=0.7,
        max_det=300,
        stream=True,
        verbose=False,
    )
    additions = {}
    for row, result in zip(selected, results, strict=True):
        image_path = Path(row["source_image"]).resolve()
        if Path(result.path).resolve() != image_path:
            raise RuntimeError(f"Inference order changed: expected {image_path}, got {result.path}")
        height, width = result.orig_shape
        gt_classes, gt_boxes_n = load_labels(label_path(image_path))
        gt_boxes = xywhn_to_xyxy(gt_boxes_n, width, height)
        predictions = result.boxes.cpu().numpy()
        approved = []
        approved_counts = Counter()
        for class_name in (name.strip() for name in row["correct_class"].split(";")):
            class_id = name_to_id[class_name]
            indices = np.flatnonzero(
                (predictions.cls.astype(int) == class_id)
                & (predictions.conf >= parameters["hard_negative_conf"])
            )
            same_gt = gt_boxes[gt_classes == class_id]
            for prediction_index in indices:
                xyxy = predictions.xyxy[prediction_index]
                maximum_iou = float(box_iou(xyxy[None, :], same_gt).max()) if len(same_gt) else 0.0
                if maximum_iou >= parameters["negative_iou"]:
                    continue
                approved.append((class_id, *map(float, predictions.xywhn[prediction_index])))
                approved_counts[class_id] += 1
        expected = expected_prediction_counts(row, name_to_id)
        if approved_counts != expected:
            raise RuntimeError(
                f"Row {row['index']} prediction count changed: expected {dict(expected)}, got {dict(approved_counts)}"
            )
        additions[label_path(image_path).resolve()] = approved
    return additions


def serialize_boxes(boxes):
    """Serialize numeric detection boxes in normalized YOLO format."""
    return "".join(f"{class_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n" for class_id, x, y, width, height in boxes)


def write_review(path, rows):
    """Write normalized review decisions with a valid index header."""
    fieldnames = [
        "index",
        "review_image",
        "source_image",
        "flagged_classes",
        "predictions",
        "max_confidence",
        "decision",
        "correct_class",
        "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def apply_changes(review_path, rows, additions, replacements, manual_changes, names):
    """Back up source files, apply label changes, and save an audit report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = PROJECT_ROOT / "dataset" / "label_corrections" / f"{timestamp}_feeder_review"
    labels_backup = backup / "labels"
    labels_backup.mkdir(parents=True)
    shutil.copy2(review_path, backup / "review.original.csv")

    changed = set(additions) | set(replacements)
    for path in sorted(changed):
        shutil.copy2(path, labels_backup / path.name)
        boxes = replacements.get(path, parse_yolo(path) + additions.get(path, []))
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(serialize_boxes(boxes), encoding="utf-8")
        temporary.replace(path)
    write_review(review_path, rows)

    cache = DATASET_ROOT / "labels" / "train.cache"
    if cache.exists():
        shutil.copy2(cache, backup / cache.name)
        cache.unlink()
    added_counts = Counter(box[0] for boxes in additions.values() for box in boxes)
    report = {
        "review": str(review_path),
        "changed_labels": len(changed),
        "prediction_labels_changed": len(additions),
        "manual_labels_replaced": len(replacements),
        "approved_boxes_added": {names[class_id]: count for class_id, count in sorted(added_counts.items())},
        "manual_class_count_changes": manual_changes,
        "backup": str(backup),
    }
    (backup / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    """Validate the completed review and optionally write all corrections."""
    args = parse_args()
    names, name_to_id = load_names()
    rows = load_review(args.review.resolve())
    replacements, manual_changes = load_manual_relabels(args.relabel.resolve(), rows, name_to_id)
    audit = json.loads(args.audit.resolve().read_text(encoding="utf-8"))
    additions = reproduce_approved_boxes(rows, audit, name_to_id, args.device, {path.stem for path in replacements})
    counts = Counter(row["decision"] for row in rows)
    added_counts = Counter(box[0] for boxes in additions.values() for box in boxes)
    print(f"Validated review decisions: {dict(counts)}")
    print(f"Approved prediction boxes: {dict((names[key], value) for key, value in added_counts.items())}")
    print(f"Labels to update: {len(set(additions) | set(replacements))}")
    if not args.apply:
        print("Dry run complete. Pass --apply to back up and write the validated corrections.")
        return
    report = apply_changes(args.review.resolve(), rows, additions, replacements, manual_changes, names)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
