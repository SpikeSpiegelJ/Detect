# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Render weak-class detection errors from a dataset split for manual label review."""

from __future__ import annotations

import csv
import json
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from build_focus_hardneg_manifest import box_iou, label_path, load_labels, xywhn_to_xyxy
from review_feeder_hard_negatives import (
    GT_COLOR,
    PRED_COLOR,
    draw_box,
    make_tile,
    resize_max_side,
    save_contact_sheets,
)

from ultralytics import YOLO
from ultralytics.data.loaders import LoadImagesAndVideos, SourceTypes

DEFAULT_DATA = PROJECT_ROOT / "dataset" / "data_repartition_v8_dataset9_9c.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "train" / "yolo26s_v8_dataset9_relabel_ft" / "weights" / "best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "audit" / "yolo26s_v8_train_weak_class_errors"
TARGET_CLASSES = {1: "antenna_s", 3: "Feeder_RRU", 5: "cut", 6: "Feeder_antenna"}
FEEDER_ENDPOINTS = {"Feeder_RRU": "RRU", "Feeder_antenna": "antenna_b"}
MISS_COLOR = (0, 150, 255)
LOCALIZATION_COLOR = (210, 70, 180)


def parse_args():
    """Parse error-audit settings."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, help="Optional second path for the machine-readable summary")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--candidate-conf", type=float, default=0.01)
    parser.add_argument("--false-positive-conf", type=float, default=0.4)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--localization-iou", type=float, default=0.1)
    parser.add_argument("--confusion-iou", type=float, default=0.3)
    parser.add_argument("--nwd-threshold", type=float, default=0.04)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--sheet-columns", type=int, default=3)
    parser.add_argument("--sheet-rows", type=int, default=3)
    parser.add_argument("--include-duplicates", action="store_true")
    return parser.parse_args()


def collect_images(data_path, split):
    """Resolve one image split from the dataset YAML."""
    descriptor = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
    root = Path(descriptor["path"]).resolve()
    image_dir = Path(descriptor[split])
    image_dir = image_dir if image_dir.is_absolute() else root / image_dir
    suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    images = sorted(path.resolve() for path in image_dir.rglob("*") if path.suffix.lower() in suffixes)
    if not images:
        raise FileNotFoundError(image_dir)
    return images, descriptor["names"]


def greedy_matches(gt_classes, gt_boxes, pred_classes, pred_boxes, threshold):
    """Return one-to-one same-class matches in descending IoU order."""
    overlaps = box_iou(gt_boxes, pred_boxes)
    candidates = np.argwhere((gt_classes[:, None] == pred_classes[None, :]) & (overlaps >= threshold))
    matched_gt, matched_predictions = set(), set()
    for gt_index, pred_index in sorted(candidates.tolist(), key=lambda pair: -overlaps[tuple(pair)]):
        if gt_index in matched_gt or pred_index in matched_predictions:
            continue
        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)
    return matched_gt, matched_predictions, overlaps


def best_candidate(indices, overlaps, confidences):
    """Choose a candidate by IoU first and confidence second."""
    if not len(indices):
        return None
    return max(indices, key=lambda index: (float(overlaps[index]), float(confidences[index])))


def geometry_profile(box, width, height, imgsz, nwd_threshold=0.04):
    """Return detector-input geometry for one xyxy box."""
    if box is None:
        return {}
    box_width, box_height = max(float(box[2] - box[0]), 0.0), max(float(box[3] - box[1]), 0.0)
    scale = min(imgsz / width, imgsz / height)
    short_side = min(box_width, box_height) * scale
    aspect_ratio = max(box_width, box_height) / max(min(box_width, box_height), 1e-9)
    normalized_sqrt_area = float(np.sqrt(box_width * box_height) * scale / imgsz)
    return {
        "gt_aspect_ratio": aspect_ratio,
        "gt_short_side_px_at_imgsz": short_side,
        "gt_area_fraction": box_width * box_height / (width * height),
        "gt_normalized_sqrt_area": normalized_sqrt_area,
        "gt_elongated": aspect_ratio >= 4,
        "gt_small_at_imgsz": box_width * box_height * scale**2 < 32**2,
        "gt_nwd_active": bool(normalized_sqrt_area < nwd_threshold),
    }


def endpoint_profile(class_name, box, gt_classes, gt_boxes, names, width, height):
    """Describe whether the class-defining feeder endpoint is visible and nearby."""
    endpoint_name = FEEDER_ENDPOINTS.get(class_name)
    if endpoint_name is None:
        return {}
    endpoint_ids = [class_id for class_id, name in names.items() if name == endpoint_name]
    endpoint_indices = np.flatnonzero(np.isin(gt_classes, endpoint_ids))
    profile = {"endpoint_class": endpoint_name, "endpoint_visible": bool(len(endpoint_indices))}
    if box is None or not len(endpoint_indices):
        return profile
    center = (np.asarray(box[:2]) + np.asarray(box[2:])) / 2
    endpoint_centers = (gt_boxes[endpoint_indices, :2] + gt_boxes[endpoint_indices, 2:]) / 2
    distances = np.linalg.norm(endpoint_centers - center, axis=1) / np.hypot(width, height)
    profile["endpoint_min_center_distance_normalized"] = float(distances.min())
    return profile


def distribution_summary(values):
    """Return compact quartiles for a numeric sequence."""
    if not values:
        return {}
    return {key: float(value) for key, value in zip(("q25", "q50", "q75"), np.quantile(values, (0.25, 0.5, 0.75)))}


def analyze_image(result, gt_classes, gt_boxes, args, names):
    """Classify missed GTs, localization errors, class confusion, and false positives."""
    predictions = result.boxes.cpu().numpy()
    pred_classes = predictions.cls.astype(int)
    pred_boxes = predictions.xyxy
    confidences = predictions.conf
    matched_gt, matched_predictions, overlaps = greedy_matches(
        gt_classes, gt_boxes, pred_classes, pred_boxes, args.match_iou
    )
    errors = []
    consumed_predictions = set(matched_predictions)

    for gt_index in np.flatnonzero(np.isin(gt_classes, tuple(TARGET_CLASSES))):
        if int(gt_index) in matched_gt:
            continue
        class_id = int(gt_classes[gt_index])
        remaining_same = np.flatnonzero(
            (pred_classes == class_id) & ~np.isin(np.arange(len(pred_classes)), list(consumed_predictions))
        )
        same_index = best_candidate(remaining_same, overlaps[gt_index], confidences)
        same_iou = float(overlaps[gt_index, same_index]) if same_index is not None else 0.0
        if same_index is not None and same_iou >= args.localization_iou:
            consumed_predictions.add(int(same_index))
            errors.append(
                {
                    "class": names[class_id],
                    "error_type": "localization",
                    "origin": "ground_truth",
                    "confidence": float(confidences[same_index]),
                    "best_iou": same_iou,
                    "ground_truth_class": names[class_id],
                    "predicted_class": names[class_id],
                    "gt_index": int(gt_index),
                    "gt_box": gt_boxes[gt_index],
                    "pred_box": pred_boxes[same_index],
                }
            )
            continue

        other_indices = np.flatnonzero(
            (pred_classes != class_id) & ~np.isin(np.arange(len(pred_classes)), list(consumed_predictions))
        )
        other_index = best_candidate(other_indices, overlaps[gt_index], confidences)
        other_iou = float(overlaps[gt_index, other_index]) if other_index is not None else 0.0
        if other_index is not None and other_iou >= args.confusion_iou:
            consumed_predictions.add(int(other_index))
            errors.append(
                {
                    "class": names[class_id],
                    "error_type": "class_confusion",
                    "origin": "ground_truth",
                    "confidence": float(confidences[other_index]),
                    "best_iou": other_iou,
                    "ground_truth_class": names[class_id],
                    "predicted_class": names[int(pred_classes[other_index])],
                    "gt_index": int(gt_index),
                    "gt_box": gt_boxes[gt_index],
                    "pred_box": pred_boxes[other_index],
                }
            )
        else:
            errors.append(
                {
                    "class": names[class_id],
                    "error_type": "missed_gt",
                    "origin": "ground_truth",
                    "confidence": float(confidences[same_index]) if same_index is not None else 0.0,
                    "best_iou": same_iou,
                    "ground_truth_class": names[class_id],
                    "predicted_class": "",
                    "gt_index": int(gt_index),
                    "gt_box": gt_boxes[gt_index],
                    "pred_box": None,
                }
            )

    target_predictions = np.isin(pred_classes, tuple(TARGET_CLASSES)) & (confidences >= args.false_positive_conf)
    for pred_index in np.flatnonzero(target_predictions):
        if int(pred_index) in consumed_predictions:
            continue
        class_id = int(pred_classes[pred_index])
        best_gt = int(np.argmax(overlaps[:, pred_index])) if len(gt_boxes) else None
        maximum_iou = float(overlaps[best_gt, pred_index]) if best_gt is not None else 0.0
        gt_class = int(gt_classes[best_gt]) if best_gt is not None else None
        if gt_class == class_id and maximum_iou >= args.localization_iou:
            error_type = "duplicate_prediction"
        elif gt_class is not None and gt_class != class_id and maximum_iou >= args.confusion_iou:
            error_type = "class_confusion_prediction"
        else:
            error_type = "false_positive"
        if error_type == "duplicate_prediction" and not args.include_duplicates:
            continue
        actual_class = names[gt_class] if error_type == "class_confusion_prediction" else ""
        errors.append(
            {
                "class": names[class_id],
                "error_type": error_type,
                "origin": "prediction",
                "confidence": float(confidences[pred_index]),
                "best_iou": maximum_iou,
                "ground_truth_class": actual_class,
                "predicted_class": names[class_id],
                "gt_index": best_gt if error_type != "false_positive" else None,
                "gt_box": gt_boxes[best_gt] if error_type != "false_positive" else None,
                "pred_box": pred_boxes[pred_index],
            }
        )
    return errors


def render_errors(image, gt_classes, gt_boxes, errors, names):
    """Draw all GTs and emphasize the boxes involved in errors."""
    for class_id, box in zip(gt_classes, gt_boxes):
        draw_box(image, box, f"GT {names[int(class_id)]}", GT_COLOR)
    for error in errors:
        if error["gt_box"] is not None:
            draw_box(image, error["gt_box"], f"ERROR GT {error['class']}", MISS_COLOR)
        if error["pred_box"] is not None:
            color = LOCALIZATION_COLOR if error["error_type"] == "localization" else PRED_COLOR
            label = f"{error['error_type']} {error['class']} {error['confidence']:.2f}"
            draw_box(image, error["pred_box"], label, color)


def main():
    """Generate the weak-class error review package."""
    args = parse_args()
    if not 0 < args.candidate_conf <= args.false_positive_conf < 1:
        raise ValueError("Require 0 < candidate-conf <= false-positive-conf < 1")
    if not 0 <= args.localization_iou < args.confusion_iou < args.match_iou <= 1:
        raise ValueError("Require localization-iou < confusion-iou < match-iou")
    if args.nwd_threshold <= 0:
        raise ValueError("nwd-threshold must be positive")
    data_path, weights, output = args.data.resolve(), args.weights.resolve(), args.output.resolve()
    for required in (data_path, weights):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    summary_output = args.summary_output.resolve() if args.summary_output else None
    if summary_output and summary_output.exists():
        raise FileExistsError(f"Refusing to overwrite existing summary: {summary_output}")
    images_output = output / "images"
    images_output.mkdir(parents=True)

    images, yaml_names = collect_images(data_path, args.split)
    names = {int(class_id): name for class_id, name in yaml_names.items()}
    model = YOLO(weights)
    if any(model.names.get(class_id) != name for class_id, name in names.items()):
        raise ValueError("Checkpoint and dataset class names differ")
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

    rows, tiles, error_counts, gt_error_counts, geometry_counts, endpoint_counts, target_gt_counts, visited = (
        [],
        [],
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        set(),
    )
    target_geometry, error_geometry = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for result in results:
        image_path = Path(result.path).resolve()
        if image_path not in images or image_path in visited:
            raise RuntimeError(f"Unexpected or duplicate inference image: {image_path}")
        visited.add(image_path)
        height, width = result.orig_shape
        gt_classes, gt_boxes_n = load_labels(label_path(image_path))
        gt_boxes = xywhn_to_xyxy(gt_boxes_n, width, height)
        for gt_class, gt_box in zip(gt_classes, gt_boxes):
            class_name = names[int(gt_class)]
            if int(gt_class) not in TARGET_CLASSES:
                continue
            geometry = geometry_profile(gt_box, width, height, args.imgsz, args.nwd_threshold)
            endpoint = endpoint_profile(class_name, gt_box, gt_classes, gt_boxes, names, width, height)
            target_gt_counts[(class_name, "all")] += 1
            target_gt_counts[(class_name, "elongated" if geometry["gt_elongated"] else "not_elongated")] += 1
            target_gt_counts[(class_name, f"nwd_active={geometry['gt_nwd_active']}")] += 1
            target_geometry[class_name]["short_side_px_at_imgsz"].append(geometry["gt_short_side_px_at_imgsz"])
            target_geometry[class_name]["aspect_ratio"].append(geometry["gt_aspect_ratio"])
            if endpoint:
                target_gt_counts[(class_name, f"endpoint_visible={endpoint['endpoint_visible']}")] += 1
        errors = analyze_image(result, gt_classes, gt_boxes, args, names)
        if not errors:
            continue
        review_index = len(tiles) + 1
        rendered = result.orig_img.copy()
        render_errors(rendered, gt_classes, gt_boxes, errors, names)
        saved = resize_max_side(rendered, args.max_side)
        output_name = f"{review_index:03d}_{image_path.stem}.jpg"
        if not cv2.imwrite(str(images_output / output_name), saved):
            raise OSError(f"Failed to save {output_name}")
        summary = Counter((error["class"], error["error_type"]) for error in errors)
        caption = " | ".join(f"{name}:{kind} x{count}" for (name, kind), count in summary.items())
        tiles.append(make_tile(saved, f"{review_index:03d}  {caption}"))
        for error_index, error in enumerate(errors, 1):
            error_counts[(error["class"], error["error_type"])] += 1
            if error["origin"] == "ground_truth":
                gt_error_counts[(error["ground_truth_class"], error["error_type"])] += 1
            geometry = geometry_profile(error["gt_box"], width, height, args.imgsz, args.nwd_threshold)
            profile_class = error["ground_truth_class"] or error["class"]
            endpoint = endpoint_profile(
                profile_class,
                error["gt_box"] if error["gt_box"] is not None else error["pred_box"],
                gt_classes,
                gt_boxes,
                names,
                width,
                height,
            )
            if geometry:
                geometry_counts[(profile_class, error["error_type"], geometry["gt_elongated"])] += 1
                if error["origin"] == "ground_truth":
                    error_geometry[profile_class]["short_side_px_at_imgsz"].append(
                        geometry["gt_short_side_px_at_imgsz"]
                    )
                    error_geometry[profile_class]["aspect_ratio"].append(geometry["gt_aspect_ratio"])
            if endpoint:
                endpoint_counts[(profile_class, error["error_type"], endpoint["endpoint_visible"])] += 1
            rows.append(
                {
                    "index": len(rows) + 1,
                    "review_image": f"images/{output_name}",
                    "source_image": str(image_path),
                    "error_index": error_index,
                    "class": error["class"],
                    "error_type": error["error_type"],
                    "origin": error["origin"],
                    "confidence": f"{error['confidence']:.4f}",
                    "best_iou": f"{error['best_iou']:.4f}",
                    "ground_truth_class": error["ground_truth_class"],
                    "predicted_class": error["predicted_class"],
                    "gt_aspect_ratio": f"{geometry.get('gt_aspect_ratio', ''):.4f}" if geometry else "",
                    "gt_short_side_px_at_imgsz": (
                        f"{geometry.get('gt_short_side_px_at_imgsz', ''):.2f}" if geometry else ""
                    ),
                    "gt_area_fraction": f"{geometry.get('gt_area_fraction', ''):.8f}" if geometry else "",
                    "gt_normalized_sqrt_area": (
                        f"{geometry.get('gt_normalized_sqrt_area', ''):.6f}" if geometry else ""
                    ),
                    "gt_elongated": geometry.get("gt_elongated", ""),
                    "gt_small_at_imgsz": geometry.get("gt_small_at_imgsz", ""),
                    "gt_nwd_active": geometry.get("gt_nwd_active", ""),
                    "endpoint_class": endpoint.get("endpoint_class", ""),
                    "endpoint_visible": endpoint.get("endpoint_visible", ""),
                    "endpoint_min_center_distance_normalized": (
                        f"{endpoint['endpoint_min_center_distance_normalized']:.4f}"
                        if "endpoint_min_center_distance_normalized" in endpoint
                        else ""
                    ),
                    "decision": "TODO",
                    "correct_class": "",
                    "notes": "",
                }
            )
    if visited != set(images):
        raise RuntimeError(f"Inference missed {len(set(images) - visited)} {args.split} images")
    if not rows:
        raise RuntimeError("No target-class errors found")

    with (output / "review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    save_contact_sheets(tiles, output, args.sheet_columns, args.sheet_rows)
    summary = {
        "weights": str(weights),
        "data": str(data_path),
        "split": args.split,
        "split_images": len(images),
        "review_images": len(tiles),
        "review_rows": len(rows),
        "errors": {f"{name}/{kind}": count for (name, kind), count in sorted(error_counts.items())},
        "ground_truth_errors": {f"{name}/{kind}": count for (name, kind), count in sorted(gt_error_counts.items())},
        "class_confusion_pairs": dict(
            sorted(
                Counter(
                    f"{row['ground_truth_class']}->{row['predicted_class']}"
                    for row in rows
                    if row["ground_truth_class"]
                    and row["predicted_class"]
                    and row["ground_truth_class"] != row["predicted_class"]
                ).items()
            )
        ),
        "error_geometry": {
            f"{name}/{kind}/elongated={elongated}": count
            for (name, kind, elongated), count in sorted(geometry_counts.items())
        },
        "error_endpoint_visibility": {
            f"{name}/{kind}/endpoint_visible={visible}": count
            for (name, kind, visible), count in sorted(endpoint_counts.items())
        },
        "target_gt_profiles": {
            f"{name}/{profile}": count for (name, profile), count in sorted(target_gt_counts.items())
        },
        "ground_truth_error_rates": {
            name: {
                "errors": sum(count for (error_name, _), count in gt_error_counts.items() if error_name == name),
                "targets": target_gt_counts[(name, "all")],
                "rate": sum(count for (error_name, _), count in gt_error_counts.items() if error_name == name)
                / target_gt_counts[(name, "all")],
            }
            for name in TARGET_CLASSES.values()
        },
        "nwd_scale_error_rates": {
            name: {
                f"active={active}": {
                    "errors": sum(
                        row["origin"] == "ground_truth"
                        and row["ground_truth_class"] == name
                        and row["gt_nwd_active"] is active
                        for row in rows
                    ),
                    "targets": target_gt_counts[(name, f"nwd_active={active}")],
                    "rate": (
                        sum(
                            row["origin"] == "ground_truth"
                            and row["ground_truth_class"] == name
                            and row["gt_nwd_active"] is active
                            for row in rows
                        )
                        / target_gt_counts[(name, f"nwd_active={active}")]
                        if target_gt_counts[(name, f"nwd_active={active}")]
                        else None
                    ),
                }
                for active in (True, False)
            }
            for name in TARGET_CLASSES.values()
        },
        "geometry_distributions": {
            name: {
                "all_targets": {metric: distribution_summary(values) for metric, values in metrics.items()},
                "ground_truth_errors": {
                    metric: distribution_summary(error_geometry[name][metric]) for metric in metrics
                },
            }
            for name, metrics in target_geometry.items()
        },
        "parameters": {
            "candidate_conf": args.candidate_conf,
            "false_positive_conf": args.false_positive_conf,
            "localization_iou": args.localization_iou,
            "confusion_iou": args.confusion_iou,
            "match_iou": args.match_iou,
            "include_duplicates": args.include_duplicates,
            "nwd_scale_threshold": args.nwd_threshold,
        },
    }
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    (output / "summary.json").write_text(summary_text, encoding="utf-8")
    if summary_output:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(summary_text, encoding="utf-8")
    (output / "README.txt").write_text(
        "颜色说明：绿色是现有标注；橙色是涉及错误的标注；红色是误检、重复预测或类别混淆；"
        "紫色是 IoU 低于 0.5 的同类预测。\n"
        "review.csv 每一行对应一个错误候选，同一张图片可能有多行。\n"
        "decision 填写 model_error、annotation_error、acceptable 或 uncertain。\n"
        "标注类别错误时，在 correct_class 填写正确类别。完成审核前不要修改数据集标签。\n"
        "默认不输出重复预测，以减少训练集审查量；需要时可使用 --include-duplicates。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Review package: {output}")


if __name__ == "__main__":
    main()
