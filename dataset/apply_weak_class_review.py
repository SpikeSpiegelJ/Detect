# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Reproduce reviewed detector errors and apply approved label corrections to a dataset copy."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
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
from review_v8_weak_class_errors import analyze_image, collect_images  # noqa: E402


DEFAULT_REVIEW = PROJECT_ROOT / "runs" / "audit" / "v13_hard_val_baseline_errors" / "review.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "runs" / "audit" / "v13_hard_val_baseline_errors" / "summary.json"
DEFAULT_SOURCE = PROJECT_ROOT / "dataset" / "repartition_v13_cut_feeder_hard"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "repartition_v13_cut_feeder_hard_corrected"
DECISIONS = {"model_error", "annotation_error", "acceptable", "uncertain"}


def parse_args():
    """Parse correction options."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--apply", action="store_true", help="Create the corrected dataset copy after validation.")
    return parser.parse_args()


def read_review(path: Path) -> list[dict[str, str]]:
    """Load and validate a completed review CSV."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("gb18030")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError(f"Empty review: {path}")
    for row in rows:
        row["decision"] = row["decision"].strip()
        row["correct_class"] = row["correct_class"].strip()
        if row["decision"] not in DECISIONS:
            raise ValueError(f"Row {row['index']} has invalid decision {row['decision']!r}")
        if row["decision"] == "annotation_error" and row["error_type"] not in {
            "false_positive",
            "class_confusion",
            "class_confusion_prediction",
            "missed_gt",
            "localization",
        }:
            raise ValueError(f"Row {row['index']} has unsupported correction type {row['error_type']}")
    return rows


def parse_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read numeric YOLO labels."""
    return [
        (int(float(cls)), float(x), float(y), float(width), float(height))
        for cls, x, y, width, height in (line.split() for line in path.read_text(encoding="utf-8").splitlines())
    ]


def serialize(labels: list[tuple[int, float, float, float, float]]) -> str:
    """Serialize YOLO labels deterministically."""
    return "".join(f"{cls} {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n" for cls, x, y, width, height in labels)


def xyxy_to_xywhn(box, width: int, height: int) -> tuple[float, float, float, float]:
    """Convert a pixel xyxy box to normalized YOLO xywh."""
    x1, y1, x2, y2 = map(float, box)
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height)


def reproduce(rows: list[dict[str, str]], summary: dict, names: dict[int, str], device: str) -> dict[int, dict]:
    """Reproduce each reviewed error and verify stable inference ordering and values."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[Path(row["source_image"]).resolve()].append(row)
    parameters = summary["parameters"]
    model = YOLO(summary["weights"])
    reproduced = {}
    images, _ = collect_images(Path(summary["data"]), summary["split"])
    source = LoadImagesAndVideos([str(path) for path in images], batch=4)
    source.source_type = SourceTypes()
    results = model.predict(
            source=source,
            imgsz=1280,
            batch=4,
            device=device,
            conf=parameters["candidate_conf"],
            iou=0.7,
            max_det=300,
            stream=True,
            verbose=False,
        )
    for result in results:
        image_path = Path(result.path).resolve()
        image_rows = grouped.get(image_path)
        if not image_rows:
            continue
        height, width = result.orig_shape
        gt_classes, gt_boxes_n = load_labels(label_path(image_path))
        gt_boxes = xywhn_to_xyxy(gt_boxes_n, width, height)
        errors = analyze_image(
            result,
            gt_classes,
            gt_boxes,
            type("Args", (), parameters),
            names,
        )
        for row in image_rows:
            error = errors[int(row["error_index"]) - 1]
            if error["class"] != row["class"] or error["error_type"] != row["error_type"]:
                raise RuntimeError(f"Row {row['index']} no longer reproduces: {error}")
            if abs(error["confidence"] - float(row["confidence"])) > 0.002:
                raise RuntimeError(f"Row {row['index']} confidence changed")
            reproduced[int(row["index"])] = {"error": error, "shape": (height, width)}
    return reproduced


def copy_dataset(source: Path, output: Path) -> None:
    """Copy a dataset so later changes cannot mutate any source file."""
    if output.exists():
        raise FileExistsError(output)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        destination = output / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.suffix == ".cache":
            continue
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def target_label(source_image: Path, source_root: Path, output_root: Path) -> Path:
    """Map a reviewed source image to the corrected copy label."""
    relative = source_image.resolve().relative_to(source_root.resolve())
    parts = list(relative.parts)
    parts[parts.index("images")] = "labels"
    return (output_root / Path(*parts)).with_suffix(".txt")


def find_gt(labels, error, width, height) -> int:
    """Locate the source label corresponding to an audited GT box."""
    normalized = np.asarray([row[1:] for row in labels], dtype=float)
    boxes = xywhn_to_xyxy(normalized, width, height)
    overlaps = box_iou(np.asarray(error["gt_box"], dtype=float)[None, :], boxes)[0]
    index = int(overlaps.argmax())
    if overlaps[index] < 0.999:
        raise RuntimeError(f"Unable to identify reviewed GT box; best IoU={overlaps[index]:.6f}")
    return index


def main():
    """Validate review decisions and optionally write corrections to a new dataset."""
    args = parse_args()
    rows = read_review(args.review.resolve())
    summary = json.loads(args.summary.resolve().read_text(encoding="utf-8"))
    descriptor = yaml.safe_load(Path(summary["data"]).read_text(encoding="utf-8-sig"))
    names = {int(key): value for key, value in descriptor["names"].items()}
    name_to_id = {value: key for key, value in names.items()}
    reproduced = reproduce(rows, summary, names, args.device)
    correction_rows = [row for row in rows if row["decision"] == "annotation_error"]
    actions = Counter()
    for row in correction_rows:
        kind = row["error_type"]
        actions["add_box" if kind == "false_positive" else "change_existing_box"] += 1
        if kind != "false_positive" and not row["correct_class"]:
            raise ValueError(f"Row {row['index']} needs correct_class to modify an existing annotation")
    report = {
        "decisions": dict(Counter(row["decision"] for row in rows)),
        "validated_rows": len(reproduced),
        "corrections": len(correction_rows),
        "actions": dict(actions),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.apply:
        print("Dry run complete. Pass --apply to create the corrected dataset copy.")
        return

    source_root, output_root = args.source.resolve(), args.output.resolve()
    copy_dataset(source_root, output_root)
    changed = defaultdict(list)
    for row in correction_rows:
        source_image = Path(row["source_image"]).resolve()
        path = target_label(source_image, source_root, output_root)
        labels = parse_labels(path)
        reproduced_row = reproduced[int(row["index"])]
        error = reproduced_row["error"]
        height, width = reproduced_row["shape"]
        if row["error_type"] == "false_positive":
            class_name = row["correct_class"] or row["class"]
            candidate = (name_to_id[class_name], *xyxy_to_xywhn(error["pred_box"], width, height))
            labels.append(candidate)
            changed[str(path)].append({"row": row["index"], "action": "add", "class": class_name})
        else:
            index = find_gt(labels, error, width, height)
            old_class = names[labels[index][0]]
            class_name = row["correct_class"]
            labels[index] = (name_to_id[class_name], *labels[index][1:])
            changed[str(path)].append(
                {"row": row["index"], "action": "reclassify", "from": old_class, "to": class_name}
            )
        path.write_text(serialize(labels), encoding="utf-8")

    corrected_yaml = PROJECT_ROOT / "dataset" / "data_repartition_v13_cut_feeder_hard_corrected_9c.yaml"
    descriptor["path"] = output_root.as_posix()
    corrected_yaml.write_text(yaml.safe_dump(descriptor, sort_keys=False, allow_unicode=True), encoding="utf-8")
    report.update({"output": str(output_root), "yaml": str(corrected_yaml), "changed": changed})
    (output_root / "review_corrections.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output_root), "yaml": str(corrected_yaml), "changed_labels": len(changed)}, indent=2))


if __name__ == "__main__":
    main()
