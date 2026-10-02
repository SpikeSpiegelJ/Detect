# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Apply the reviewed OOF cut/Feeder_antenna corrections to an isolated dataset copy."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from build_focus_hardneg_manifest import box_iou, xywhn_to_xyxy  # noqa: E402
from apply_weak_class_review import (  # noqa: E402
    copy_dataset,
    parse_labels,
    serialize,
    target_label,
    xyxy_to_xywhn,
)


DEFAULT_AUDIT = PROJECT_ROOT / "runs" / "audit" / "v12_oof_cut_feeder_errors"
DEFAULT_SOURCE = PROJECT_ROOT / "dataset" / "repartition_v9_trainval_9c"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "repartition_v15_oof_reviewed"
DEFAULT_YAML = PROJECT_ROOT / "dataset" / "data_repartition_v15_oof_reviewed_9c.yaml"
DECISIONS = {"model_error", "annotation_error", "acceptable", "uncertain"}
ACTIONS = {"add_pred_box", "reclass_gt", "replace_gt_with_pred", "delete_gt"}


def parse_args():
    """Parse review-application settings."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_review(path: Path) -> list[dict[str, str]]:
    """Read the Excel-saved review CSV using its detected encoding."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeError(f"Unable to decode {path}")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError(f"Empty review: {path}")
    return rows


def normalize_review(rows: list[dict[str, str]], names: dict[int, str]):
    """Normalize deterministic omissions and the two explicitly resolved no-op entries."""
    valid_classes = set(names.values())
    changes = []
    for row in rows:
        index = int(row["index"])
        for key in ("decision", "correct_class", "notes"):
            row[key] = row[key].strip()
        before = (row["decision"], row["correct_class"], row["notes"])
        if index == 20:
            # The user reconfirmed Feeder_antenna, which is already the GT class; reclassification is a no-op.
            row["decision"], row["correct_class"], row["notes"] = "model_error", "", ""
        elif row["decision"] == "annotation_error" and row["correct_class"] == "model_error":
            row["decision"], row["correct_class"], row["notes"] = "model_error", "", ""
        elif row["decision"] == "annotation_error":
            if "add_pred_box" in row["notes"]:
                row["notes"] = "add_pred_box"
            elif row["notes"] == "replace_gt_with_pred" and not row["pred_box_xyxy"].strip():
                # A missed_gt has no prediction to copy; preserve its reviewed GT geometry and change only its class.
                row["notes"] = "reclass_gt"
            elif not row["notes"]:
                if row["error_type"] == "false_positive":
                    row["notes"] = "add_pred_box"
                elif row["error_type"] in {"class_confusion", "class_confusion_prediction"}:
                    row["notes"] = "reclass_gt"
        after = (row["decision"], row["correct_class"], row["notes"])
        if before != after:
            changes.append({"index": index, "before": before, "after": after})
        if row["decision"] not in DECISIONS:
            raise ValueError(f"Row {index} has invalid decision {row['decision']!r}")
        if row["decision"] == "annotation_error":
            if row["notes"] not in ACTIONS:
                raise ValueError(f"Row {index} has invalid action {row['notes']!r}")
            if row["notes"] != "delete_gt" and row["correct_class"] not in valid_classes:
                raise ValueError(f"Row {index} has invalid correct_class {row['correct_class']!r}")
    return changes


def parse_box(value: str, row_index: int, field: str) -> np.ndarray:
    """Parse a four-value xyxy spreadsheet cell."""
    try:
        box = np.asarray([float(item) for item in value.split(",")], dtype=float)
    except ValueError as error:
        raise ValueError(f"Row {row_index} has invalid {field}: {value!r}") from error
    if box.shape != (4,) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"Row {row_index} has invalid {field}: {value!r}")
    return box


def image_shape(path: Path) -> tuple[int, int]:
    """Read image dimensions while supporting non-ASCII Windows paths."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Unable to read image: {path}")
    return image.shape[:2]


def find_gt(labels, target_box, width, height, row_index) -> int:
    """Find the exact reviewed GT in the current label list."""
    if not labels:
        raise RuntimeError(f"Row {row_index} cannot find a GT in an empty label file")
    boxes = xywhn_to_xyxy(np.asarray([label[1:] for label in labels], dtype=float), width, height)
    overlaps = box_iou(target_box[None, :], boxes)[0]
    index = int(overlaps.argmax())
    if overlaps[index] < 0.995:
        raise RuntimeError(f"Row {row_index} cannot identify its reviewed GT; best IoU={overlaps[index]:.6f}")
    return index


def validate_no_conflicts(rows):
    """Reject multiple incompatible changes to the same reviewed GT."""
    seen = {}
    for row in rows:
        if row["decision"] != "annotation_error" or row["notes"] == "add_pred_box":
            continue
        key = (str(Path(row["source_label"]).resolve()).lower(), row["gt_box_xyxy"])
        action = (row["notes"], row["correct_class"])
        if key in seen and seen[key] != action:
            raise ValueError(f"Conflicting corrections for the same GT: {seen[key]} vs {action}")
        seen[key] = action


def write_normalized(path: Path, rows):
    """Write the normalized review without changing the user's original CSV."""
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def validate_labels(root: Path, class_count: int):
    """Validate YOLO label syntax, ranges, and image/label pairing."""
    split_counts = {}
    suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    for split in ("train", "val", "test"):
        images = sorted(path for path in (root / "images" / split).iterdir() if path.suffix.lower() in suffixes)
        labels = sorted((root / "labels" / split).glob("*.txt"))
        if {path.stem for path in images} != {path.stem for path in labels}:
            raise RuntimeError(f"Image/label pairing differs in {split}")
        instances = 0
        for path in labels:
            for class_id, x, y, width, height in parse_labels(path):
                if class_id not in range(class_count):
                    raise ValueError(f"Invalid class {class_id} in {path}")
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    raise ValueError(f"Invalid normalized box in {path}")
                instances += 1
        split_counts[split] = {"images": len(images), "labels": len(labels), "instances": instances}
    return split_counts


def main():
    """Validate the completed review and optionally create the corrected dataset."""
    args = parse_args()
    audit, source, output, yaml_path = map(Path.resolve, (args.audit, args.source, args.output, args.yaml))
    review_path = audit / "review.csv"
    source_yaml = PROJECT_ROOT / "dataset" / "data_repartition_v9_trainval_9c.yaml"
    descriptor = yaml.safe_load(source_yaml.read_text(encoding="utf-8-sig"))
    names = {int(key): value for key, value in descriptor["names"].items()}
    name_to_id = {name: class_id for class_id, name in names.items()}
    rows = read_review(review_path)
    normalizations = normalize_review(rows, names)
    validate_no_conflicts(rows)
    corrections = [row for row in rows if row["decision"] == "annotation_error"]
    for row in corrections:
        source_image = Path(row["source_image"]).resolve()
        source_label = Path(row["source_label"]).resolve()
        if source not in source_image.parents or source not in source_label.parents:
            raise ValueError(f"Row {row['index']} points outside the source dataset")
        if "test" in source_image.relative_to(source).parts or "test" in source_label.relative_to(source).parts:
            raise ValueError(f"Row {row['index']} attempts to change the independent test split")
        if not source_image.is_file() or not source_label.is_file():
            raise FileNotFoundError(f"Row {row['index']} source image/label is missing")
        row_index = int(row["index"])
        if row["notes"] in {"add_pred_box", "replace_gt_with_pred"}:
            parse_box(row["pred_box_xyxy"], row_index, "pred_box_xyxy")
        if row["notes"] != "add_pred_box":
            parse_box(row["gt_box_xyxy"], row_index, "gt_box_xyxy")

    report = {
        "decisions": dict(Counter(row["decision"] for row in rows)),
        "actions": dict(Counter(row["notes"] for row in corrections)),
        "normalizations": normalizations,
        "corrections": len(corrections),
        "changed_source_labels": len({row["source_label"] for row in corrections}),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.apply:
        print("Dry run complete. Pass --apply to create the corrected dataset.")
        return
    if yaml_path.exists():
        raise FileExistsError(f"Output YAML already exists: {yaml_path}")
    if output.exists():
        expected = (PROJECT_ROOT / "dataset" / "repartition_v15_oof_reviewed").resolve()
        if output != expected or (output / "review_corrections.json").exists():
            raise FileExistsError(f"Refusing to replace an existing completed or non-default output: {output}")
        shutil.rmtree(output)

    normalized_path = audit / "review_normalized.csv"
    write_normalized(normalized_path, rows)
    source_hashes = {Path(path).resolve(): sha256(Path(path).resolve()) for path in {row["source_label"] for row in corrections}}
    source_test_hashes = {path.name: sha256(path) for path in sorted((source / "labels" / "test").glob("*.txt"))}
    copy_dataset(source, output)
    changed = defaultdict(list)
    skipped_duplicates = []
    for row in corrections:
        row_index = int(row["index"])
        source_image = Path(row["source_image"]).resolve()
        destination = target_label(source_image, source, output)
        labels = parse_labels(destination)
        height, width = image_shape(source_image)
        action = row["notes"]
        class_name = row["correct_class"]
        if action == "add_pred_box":
            pred_box = parse_box(row["pred_box_xyxy"], row_index, "pred_box_xyxy")
            candidate = (name_to_id[class_name], *xyxy_to_xywhn(pred_box, width, height))
            same_class = [index for index, label in enumerate(labels) if label[0] == candidate[0]]
            if same_class:
                boxes = xywhn_to_xyxy(np.asarray([labels[index][1:] for index in same_class]), width, height)
                maximum = float(box_iou(pred_box[None, :], boxes)[0].max())
                if maximum >= 0.90:
                    skipped_duplicates.append({"row": row_index, "same_class_iou": maximum})
                    continue
            labels.append(candidate)
            changed[str(destination)].append({"row": row_index, "action": action, "class": class_name})
        else:
            gt_box = parse_box(row["gt_box_xyxy"], row_index, "gt_box_xyxy")
            index = find_gt(labels, gt_box, width, height, row_index)
            old_class = names[labels[index][0]]
            if action == "reclass_gt":
                labels[index] = (name_to_id[class_name], *labels[index][1:])
            elif action == "replace_gt_with_pred":
                pred_box = parse_box(row["pred_box_xyxy"], row_index, "pred_box_xyxy")
                labels[index] = (name_to_id[class_name], *xyxy_to_xywhn(pred_box, width, height))
            else:
                labels.pop(index)
            changed[str(destination)].append(
                {"row": row_index, "action": action, "from": old_class, "to": class_name}
            )
        destination.write_text(serialize(labels), encoding="utf-8")

    for path, digest in source_hashes.items():
        if sha256(path) != digest:
            raise RuntimeError(f"Source label was modified: {path}")
    output_test_hashes = {path.name: sha256(path) for path in sorted((output / "labels" / "test").glob("*.txt"))}
    if source_test_hashes != output_test_hashes:
        raise RuntimeError("Independent test labels changed")
    split_counts = validate_labels(output, len(names))
    descriptor["path"] = output.as_posix()
    yaml_path.write_text(yaml.safe_dump(descriptor, sort_keys=False, allow_unicode=True), encoding="utf-8")
    report.update(
        {
            "source": str(source),
            "output": str(output),
            "yaml": str(yaml_path),
            "copy_mode": "independent_files",
            "normalized_review": str(normalized_path),
            "applied_corrections": sum(map(len, changed.values())),
            "changed_labels": len(changed),
            "skipped_duplicate_additions": skipped_duplicates,
            "test_label_hashes_equal": True,
            "split_counts": split_counts,
            "changes": changed,
        }
    )
    (output / "review_corrections.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "yaml": str(yaml_path),
                "applied_corrections": report["applied_corrections"],
                "changed_labels": report["changed_labels"],
                "test_label_hashes_equal": True,
                "split_counts": split_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
