# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Render mined Feeder hard negatives for manual label review."""

from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.loaders import LoadImagesAndVideos, SourceTypes  # noqa: E402

from build_focus_hardneg_manifest import FOCUS_CLASSES, box_iou, label_path, load_labels, xywhn_to_xyxy  # noqa: E402


DEFAULT_AUDIT = PROJECT_ROOT / "dataset" / "repartition_v7_dataset8_9c" / "train_feeder_errors_audit.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "audit" / "feeder_hard_negatives_control"
GT_COLOR = (40, 190, 40)
PRED_COLOR = (30, 30, 230)


def parse_args():
    """Parse review rendering options."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-side", type=int, default=1600, help="Maximum side of each saved review image.")
    parser.add_argument("--sheet-columns", type=int, default=3)
    parser.add_argument("--sheet-rows", type=int, default=3)
    return parser.parse_args()


def draw_box(image, box, label, color):
    """Draw one labeled xyxy box on a BGR image."""
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    thickness = max(round((image.shape[0] + image.shape[1]) / 900), 2)
    font_scale = max(thickness / 3, 0.55)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    top = max(y1, height + 8)
    cv2.rectangle(image, (x1, top - height - 8), (x1 + width + 6, top), color, -1)
    cv2.putText(
        image,
        label,
        (x1 + 3, top - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def resize_max_side(image, maximum):
    """Resize an image without exceeding the requested maximum side."""
    scale = min(maximum / max(image.shape[:2]), 1.0)
    if scale == 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def make_tile(image, caption, width=520, height=390):
    """Letterbox one annotated image into a captioned contact-sheet tile."""
    caption_height = 42
    available_height = height - caption_height
    scale = min(width / image.shape[1], available_height / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    tile = np.full((height, width, 3), 235, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = caption_height + (available_height - resized.shape[0]) // 2
    tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.rectangle(tile, (0, 0), (width, caption_height), (35, 35, 35), -1)
    cv2.putText(tile, caption, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def save_contact_sheets(tiles, output, columns, rows):
    """Save annotated thumbnails in fixed-size contact sheets."""
    per_sheet = columns * rows
    blank = np.full_like(tiles[0], 245)
    for sheet_index, start in enumerate(range(0, len(tiles), per_sheet), 1):
        page = tiles[start : start + per_sheet]
        page.extend([blank] * (per_sheet - len(page)))
        sheet = np.vstack([np.hstack(page[row * columns : (row + 1) * columns]) for row in range(rows)])
        cv2.imwrite(str(output / f"contact_sheet_{sheet_index:02d}.jpg"), sheet)


def main():
    """Render every selected hard negative and create a review worksheet."""
    args = parse_args()
    audit_path = args.audit.resolve()
    output = args.output.resolve()
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    if args.batch < 1 or args.max_side < 256 or args.sheet_columns < 1 or args.sheet_rows < 1:
        raise ValueError("Batch, sheet dimensions, and max-side must be positive.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite review output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    images_output = output / "images"
    images_output.mkdir()

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    root = Path(audit["dataset_root"]).resolve()
    weights = Path(audit["weights"]).resolve()
    parameters = audit["parameters"]
    selected = {
        (root / relative).resolve(): reason["hard_negative"]
        for relative, reason in audit["focus_extra_reasons"].items()
        if reason["hard_negative"]
    }
    if not weights.is_file() or not selected:
        raise FileNotFoundError(f"Missing weights or hard-negative selections: {weights}")

    id_by_name = {name: class_id for class_id, name in FOCUS_CLASSES.items()}
    model = YOLO(weights)
    source = LoadImagesAndVideos([str(path) for path in sorted(selected)], batch=args.batch)
    source.source_type = SourceTypes()
    results = model.predict(
        source=source,
        imgsz=parameters["imgsz"],
        batch=args.batch,
        device=args.device,
        conf=parameters["candidate_conf"],
        iou=0.7,
        max_det=300,
        stream=True,
        verbose=False,
    )

    rows, tiles, visited = [], [], set()
    for index, result in enumerate(results, 1):
        image_path = Path(result.path).resolve()
        if image_path not in selected or image_path in visited:
            raise RuntimeError(f"Unexpected or duplicate inference image: {image_path}")
        visited.add(image_path)
        image = result.orig_img.copy()
        height, width = result.orig_shape
        gt_classes, gt_boxes_n = load_labels(label_path(image_path))
        gt_boxes = xywhn_to_xyxy(gt_boxes_n, width, height)
        for class_id, box in zip(gt_classes, gt_boxes):
            draw_box(image, box, f"GT {model.names[int(class_id)]}", GT_COLOR)

        predictions = result.boxes.cpu().numpy()
        review_predictions = []
        for class_name in selected[image_path]:
            class_id = id_by_name[class_name]
            candidate_indices = np.flatnonzero(
                (predictions.cls.astype(int) == class_id) & (predictions.conf >= parameters["hard_negative_conf"])
            )
            same_gt = gt_boxes[gt_classes == class_id]
            for prediction_index in candidate_indices:
                box = predictions.xyxy[prediction_index]
                maximum_iou = float(box_iou(box[None, :], same_gt).max()) if len(same_gt) else 0.0
                if maximum_iou >= parameters["negative_iou"]:
                    continue
                confidence = float(predictions.conf[prediction_index])
                review_predictions.append((class_name, confidence, maximum_iou))
                draw_box(image, box, f"PRED {class_name} {confidence:.2f}", PRED_COLOR)
        if not review_predictions:
            raise RuntimeError(f"Could not reproduce the mined prediction for {image_path}")

        saved = resize_max_side(image, args.max_side)
        output_name = f"{index:03d}_{image_path.stem}.jpg"
        if not cv2.imwrite(str(images_output / output_name), saved):
            raise OSError(f"Failed to save {output_name}")
        predicted = "; ".join(f"{name}:{confidence:.3f}" for name, confidence, _ in review_predictions)
        maximum_confidence = max(confidence for _, confidence, _ in review_predictions)
        flags = "; ".join(selected[image_path])
        rows.append(
            {
                "index": index,
                "review_image": f"images/{output_name}",
                "source_image": str(image_path),
                "flagged_classes": flags,
                "predictions": predicted,
                "max_confidence": f"{maximum_confidence:.4f}",
                "decision": "TODO",
                "correct_class": "",
                "notes": "",
            }
        )
        tiles.append(make_tile(saved, f"{index:03d}  {flags}  max={maximum_confidence:.2f}"))

    if visited != set(selected):
        raise RuntimeError(f"Inference missed {len(set(selected) - visited)} selected images")
    with (output / "review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    save_contact_sheets(tiles, output, args.sheet_columns, args.sheet_rows)
    (output / "README.txt").write_text(
        "Green boxes are ground truth; red boxes are control-model Feeder predictions.\n"
        "Fill decision in review.csv with: missing_label, true_negative, wrong_class, or uncertain.\n"
        "If missing_label, fill correct_class. Do not edit the source dataset until the review is complete.\n",
        encoding="utf-8",
    )
    print(f"Rendered {len(rows)} hard-negative review images to {output}")


if __name__ == "__main__":
    main()
