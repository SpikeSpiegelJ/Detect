# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Mine out-of-fold cut and Feeder_antenna errors for manual training-label review."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.loaders import LoadImagesAndVideos, SourceTypes  # noqa: E402

from build_focus_hardneg_manifest import label_path, load_labels, xywhn_to_xyxy  # noqa: E402
from prepare_grouped_kfold import canonical_id  # noqa: E402
import review_v8_weak_class_errors as review  # noqa: E402
from review_feeder_hard_negatives import make_tile, resize_max_side, save_contact_sheets  # noqa: E402


FOLD_ROOT = PROJECT_ROOT / "dataset" / "repartition_v12_grouped_3fold"
DEFAULT_WEIGHTS = [
    PROJECT_ROOT / "runs" / "train" / f"yolo26s_v12_cv_fold{fold}" / "weights" / "best.pt"
    for fold in range(1, 4)
]
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "audit" / "v12_oof_cut_feeder_errors"
TARGET_CLASSES = {5: "cut", 6: "Feeder_antenna"}
ERROR_PRIORITY = {
    "class_confusion": 5.0,
    "class_confusion_prediction": 5.0,
    "missed_gt": 4.2,
    "false_positive": 4.0,
    "localization": 2.0,
    "duplicate_prediction": 1.0,
}


def parse_args():
    """Parse OOF audit settings."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, default=FOLD_ROOT)
    parser.add_argument("--weights", type=Path, nargs=3, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--candidate-conf", type=float, default=0.01)
    parser.add_argument("--false-positive-conf", type=float, default=0.40)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--localization-iou", type=float, default=0.10)
    parser.add_argument("--confusion-iou", type=float, default=0.30)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--sheet-columns", type=int, default=3)
    parser.add_argument("--sheet-rows", type=int, default=3)
    return parser.parse_args()


def read_image_list(path):
    """Read an absolute-path image list, rejecting duplicates."""
    images = [Path(line.strip()).resolve() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(images) != len(set(images)):
        raise ValueError(f"Duplicate image paths in {path}")
    missing = [image for image in images if not image.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images from {path}, e.g. {missing[:3]}")
    return images


def resolve_yaml_path(yaml_path, value):
    """Resolve a dataset YAML path entry."""
    value = Path(value)
    if value.is_absolute():
        return value.resolve()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
    root = Path(data.get("path", yaml_path.parent))
    root = root if root.is_absolute() else (yaml_path.parent / root)
    return (root / value).resolve()


def collect_split_images(path):
    """Collect images from either a directory or a text manifest."""
    if path.suffix.lower() == ".txt":
        return read_image_list(path)
    suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    return sorted(item.resolve() for item in path.rglob("*") if item.is_file() and item.suffix.lower() in suffixes)


def validate_folds(fold_root):
    """Verify mutually exclusive OOF folds and zero development/test identity overlap."""
    yaml_paths = [fold_root / f"fold{fold}.yaml" for fold in range(1, 4)]
    val_paths = [fold_root / f"fold{fold}_val.txt" for fold in range(1, 4)]
    for required in [*yaml_paths, *val_paths]:
        if not required.is_file():
            raise FileNotFoundError(required)
    folds = [read_image_list(path) for path in val_paths]
    for left in range(len(folds)):
        for right in range(left + 1, len(folds)):
            overlap = set(folds[left]) & set(folds[right])
            if overlap:
                raise ValueError(f"Fold {left + 1}/{right + 1} path overlap: {len(overlap)}")
    development = {image for fold in folds for image in fold}
    development_ids = {canonical_id(image) for image in development}
    first_data = yaml.safe_load(yaml_paths[0].read_text(encoding="utf-8-sig"))
    test_path = resolve_yaml_path(yaml_paths[0], first_data["test"])
    test_images = collect_split_images(test_path)
    test_ids = {canonical_id(image) for image in test_images}
    identity_overlap = development_ids & test_ids
    if identity_overlap:
        raise ValueError(f"Development/test source identity overlap: {len(identity_overlap)}")
    return yaml_paths, folds, test_images


def box_text(box):
    """Serialize an optional xyxy box for a spreadsheet cell."""
    if box is None:
        return ""
    return ",".join(f"{float(value):.2f}" for value in box)


def error_score(error):
    """Rank errors by review value without consulting the independent test set."""
    priority = ERROR_PRIORITY.get(error["error_type"], 0.0)
    confidence = float(error["confidence"])
    overlap = float(error["best_iou"])
    if error["error_type"] == "missed_gt":
        return priority + (1.0 - overlap)
    if error["error_type"] == "false_positive":
        return priority + confidence
    return priority + 0.5 * confidence + 0.5 * overlap


def image_score(record, class_name=None):
    """Return the review score for one image, optionally for one target class."""
    errors = [error for error in record["errors"] if class_name is None or error["class"] == class_name]
    if not errors:
        return -1.0
    return max(map(error_score, errors)) + 0.05 * min(len(errors) - 1, 10)


def select_balanced(records, maximum):
    """Select a class-balanced set of high-value review images."""
    per_class = max(1, maximum // len(TARGET_CLASSES))
    selected = {}
    for class_name in TARGET_CLASSES.values():
        candidates = [record for record in records if image_score(record, class_name) >= 0]
        candidates.sort(key=lambda record: (-image_score(record, class_name), str(record["image"])))
        for record in candidates[:per_class]:
            selected[record["image"]] = record
    if len(selected) < maximum:
        remaining = [record for record in records if record["image"] not in selected]
        remaining.sort(key=lambda record: (-image_score(record), str(record["image"])))
        for record in remaining[: maximum - len(selected)]:
            selected[record["image"]] = record
    ordered = list(selected.values())
    ordered.sort(key=lambda record: (-image_score(record), str(record["image"])))
    return ordered[:maximum]


def load_image(path):
    """Read an image while supporting non-ASCII Windows paths."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Failed to read image: {path}")
    return image


def main():
    """Generate a balanced OOF error-review package."""
    args = parse_args()
    if args.max_images < 1:
        raise ValueError("--max-images must be positive")
    if not 0 < args.candidate_conf <= args.false_positive_conf < 1:
        raise ValueError("Require 0 < candidate-conf <= false-positive-conf < 1")
    if not 0 <= args.localization_iou < args.confusion_iou < args.match_iou <= 1:
        raise ValueError("Require localization-iou < confusion-iou < match-iou")
    fold_root = args.fold_root.resolve()
    output = args.output.resolve()
    weights = [path.resolve() for path in args.weights]
    for required in weights:
        if not required.is_file():
            raise FileNotFoundError(required)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    yaml_paths, fold_images, test_images = validate_folds(fold_root)
    audit_args = SimpleNamespace(
        match_iou=args.match_iou,
        localization_iou=args.localization_iou,
        confusion_iou=args.confusion_iou,
        false_positive_conf=args.false_positive_conf,
        include_duplicates=False,
    )
    review.TARGET_CLASSES = TARGET_CLASSES
    records = []
    full_counts = Counter()
    fold_stats = []
    expected_names = None
    for fold_index, (yaml_path, images, checkpoint) in enumerate(zip(yaml_paths, fold_images, weights), 1):
        descriptor = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
        names = {int(class_id): name for class_id, name in descriptor["names"].items()}
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError("Class names differ among fold YAML files")
        model = YOLO(checkpoint)
        if any(model.names.get(class_id) != name for class_id, name in names.items()):
            raise ValueError(f"Checkpoint and dataset class names differ for fold {fold_index}")
        source = LoadImagesAndVideos([str(path) for path in images], batch=args.batch)
        source.source_type = SourceTypes()
        results = model.predict(
            source=source,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=args.candidate_conf,
            iou=0.7,
            max_det=300,
            stream=True,
            verbose=False,
        )
        visited = set()
        fold_error_images = 0
        for result in results:
            image_path = Path(result.path).resolve()
            if image_path not in set(images) or image_path in visited:
                raise RuntimeError(f"Unexpected or duplicate fold-{fold_index} image: {image_path}")
            visited.add(image_path)
            height, width = result.orig_shape
            source_label = label_path(image_path)
            gt_classes, gt_boxes_n = load_labels(source_label)
            gt_boxes = xywhn_to_xyxy(gt_boxes_n, width, height)
            errors = review.analyze_image(result, gt_classes, gt_boxes, audit_args, names)
            errors = [error for error in errors if error["class"] in TARGET_CLASSES.values()]
            if not errors:
                continue
            fold_error_images += 1
            for error in errors:
                full_counts[(error["class"], error["error_type"])] += 1
            records.append(
                {
                    "image": image_path,
                    "label": source_label,
                    "fold": fold_index,
                    "gt_classes": gt_classes,
                    "gt_boxes": gt_boxes,
                    "errors": errors,
                }
            )
        expected = set(images)
        if visited != expected:
            raise RuntimeError(f"Fold {fold_index} inference missed {len(expected - visited)} images")
        fold_stats.append({"fold": fold_index, "images": len(images), "error_images": fold_error_images})

    selected = select_balanced(records, args.max_images)
    if not selected:
        raise RuntimeError("No target-class OOF errors found")
    images_output = output / "images"
    images_output.mkdir(parents=True)
    rows, tiles, selected_counts = [], [], Counter()
    for review_index, record in enumerate(selected, 1):
        image_path = record["image"]
        rendered = load_image(image_path)
        review.render_errors(rendered, record["gt_classes"], record["gt_boxes"], record["errors"], expected_names)
        saved = resize_max_side(rendered, args.max_side)
        output_name = f"{review_index:03d}_f{record['fold']}_{image_path.stem}.jpg"
        if not cv2.imwrite(str(images_output / output_name), saved):
            raise OSError(f"Failed to save {output_name}")
        counts = Counter((error["class"], error["error_type"]) for error in record["errors"])
        caption = " | ".join(f"{name}:{kind} x{count}" for (name, kind), count in counts.items())
        tiles.append(make_tile(saved, f"{review_index:03d} F{record['fold']}  {caption}"))
        for error_index, error in enumerate(record["errors"], 1):
            selected_counts[(error["class"], error["error_type"])] += 1
            rows.append(
                {
                    "index": len(rows) + 1,
                    "review_image": f"images/{output_name}",
                    "source_image": str(image_path),
                    "source_label": str(record["label"]),
                    "oof_fold": record["fold"],
                    "error_index": error_index,
                    "class": error["class"],
                    "error_type": error["error_type"],
                    "confidence": f"{error['confidence']:.4f}",
                    "best_iou": f"{error['best_iou']:.4f}",
                    "predicted_class": error["predicted_class"],
                    "gt_box_xyxy": box_text(error["gt_box"]),
                    "pred_box_xyxy": box_text(error["pred_box"]),
                    "decision": "TODO",
                    "correct_class": "",
                    "notes": "",
                }
            )
    with (output / "review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_contact_sheets(tiles, output, args.sheet_columns, args.sheet_rows)
    summary = {
        "method": "three-fold out-of-fold error mining",
        "weights": list(map(str, weights)),
        "fold_root": str(fold_root),
        "development_images": sum(len(images) for images in fold_images),
        "development_unique_images": len({image for images in fold_images for image in images}),
        "independent_test_images_checked_only_for_leakage": len(test_images),
        "development_test_path_overlap": 0,
        "development_test_identity_overlap": 0,
        "folds": fold_stats,
        "all_error_images": len(records),
        "selected_review_images": len(selected),
        "selected_review_rows": len(rows),
        "all_errors": {f"{name}/{kind}": count for (name, kind), count in sorted(full_counts.items())},
        "selected_errors": {
            f"{name}/{kind}": count for (name, kind), count in sorted(selected_counts.items())
        },
        "parameters": {
            "imgsz": args.imgsz,
            "candidate_conf": args.candidate_conf,
            "false_positive_conf": args.false_positive_conf,
            "localization_iou": args.localization_iou,
            "confusion_iou": args.confusion_iou,
            "match_iou": args.match_iou,
            "max_images": args.max_images,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.txt").write_text(
        "这是三折 OOF 错误审核包：每张图只由训练时未见过它的折模型预测，独立 test 未参与候选筛选。\n"
        "颜色：绿色=现有 GT；橙色=涉及错误的 GT；红色=误检或类别混淆；紫色=定位误差。\n"
        "review.csv 每行是一个候选。同图多行时分别审核，不要仅按整图给结论。\n\n"
        "decision 只填以下四种：\n"
        "annotation_error：训练标注确有缺失、类别错误或框明显错误；correct_class/notes 写修正信息。\n"
        "model_error：现有标注正确，模型漏检、误检、混淆或定位失败。\n"
        "acceptable：预测与标注差异可接受，无需改标签。\n"
        "uncertain：无法确定，暂不改标签。\n\n"
        "error_type：missed_gt=漏检已有 GT；false_positive=高置信预测无匹配 GT；"
        "class_confusion=已有 GT 被预测为其他类；class_confusion_prediction=目标类预测覆盖其他类 GT；"
        "localization=类别正确但 IoU<0.5。\n"
        "若 decision=annotation_error，notes 使用以下动作码：add_pred_box（补入红/紫预测框）、"
        "reclass_gt（把橙色 GT 改为 correct_class）、replace_gt_with_pred（用预测框替换橙色 GT）、"
        "delete_gt（删除橙色错误 GT）。动作不适合时填 manual，并说明精确修改。\n"
        "审核完成前不要直接修改原标签；后续脚本会先复制数据并仅应用 annotation_error。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Review package: {output}")


if __name__ == "__main__":
    main()
