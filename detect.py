"""Run a general detector and an optional tiled fault-specialist detector."""

from argparse import ArgumentParser
from collections import defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.utils.nms import TorchNMS


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FAULT_CLASSES = {"cut"}
FAULT_RULES = {
    "Cut": {"cut"},
    "Coupler issue": {"coupler", "antenna_s"},
    "Equipment blockage": {"RRU", "antenna_s"},
    "Equipment normal": {"RRU", "Feeder_RRU"},
    "Antenna normal": {"antenna_b", "Feeder_antenna"},
    "Antenna damage": {"antenna_b"},
}


@dataclass
class Detection:
    """One original-image detection produced by the general or fault model."""

    xyxy: np.ndarray
    confidence: float
    label: str
    source: str


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=(
            ROOT
            / "runs"
            / "train"
            / "yolo26s_v8_dataset9_relabel_ft"
            / "weights"
            / "best.pt"
        ),
        help="Path to the general 9-class best.pt.",
    )
    parser.add_argument("--fault-model", type=Path, help="Optional one-class cut-specialist best.pt.")
    parser.add_argument(
        "--source", type=Path, default=ROOT / "dataset" / "repartition_v8_dataset9_9c" / "images" / "test"
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25, help="General-model confidence threshold.")
    parser.add_argument("--fault-conf", type=float, default=0.05, help="Fault-model confidence threshold.")
    parser.add_argument(
        "--tile", type=int, default=1280, help="Fault-model sliding-window size; set 0 to disable tiling."
    )
    parser.add_argument("--overlap", type=float, default=0.25, help="Fault-tile overlap fraction in [0, 1).")
    parser.add_argument("--iou", type=float, default=0.45, help="Class-aware merge NMS IoU threshold.")
    parser.add_argument("--name", default="yolo26s_v8_dataset9_relabel_ft_one_to_many_result")
    return parser.parse_args()


def judge_fault(labels):
    """Return the first applicable image-level telecom rule."""
    return next((name for name, rule in FAULT_RULES.items() if rule.issubset(labels)), "No matching fault rule")


def image_paths(source):
    """Return supported image paths for either an image path or a directory."""
    source = source.resolve()
    if source.is_file():
        return [source]
    return sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def tile_origins(length, tile, overlap):
    """Return tile start positions while always covering the image boundary."""
    if length <= tile:
        return [0]
    stride = max(1, round(tile * (1 - overlap)))
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def result_detections(result, names, source, offset=(0, 0)):
    """Map an Ultralytics result to original-image named detections."""
    x_offset, y_offset = offset
    detections = []
    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(np.float32)
        xyxy[[0, 2]] += x_offset
        xyxy[[1, 3]] += y_offset
        detections.append(Detection(xyxy, float(box.conf.item()), names[int(box.cls.item())], source))
    return detections


def nms_named(detections, iou):
    """Apply class-aware NMS to named detections from either model or overlapping tiles."""
    grouped = defaultdict(list)
    for detection in detections:
        grouped[detection.label].append(detection)
    merged = []
    for candidates in grouped.values():
        boxes = torch.from_numpy(np.stack([candidate.xyxy for candidate in candidates]))
        scores = torch.tensor([candidate.confidence for candidate in candidates])
        for index in TorchNMS.nms(boxes, scores, iou).tolist():
            merged.append(candidates[index])
    return sorted(merged, key=lambda detection: detection.confidence, reverse=True)


def predict_full_image(model, image, imgsz, conf, device, source):
    """Run one detector on the complete image."""
    result = model.predict(image, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    return result_detections(result, model.names, source)


def predict_fault_tiles(model, image, args, device):
    """Run the fault model on native-resolution overlapping tiles and merge its outputs."""
    if not args.tile:
        return predict_full_image(model, image, args.imgsz, args.fault_conf, device, "fault")
    height, width = image.shape[:2]
    detections = []
    for y in tile_origins(height, args.tile, args.overlap):
        for x in tile_origins(width, args.tile, args.overlap):
            tile = image[y : min(y + args.tile, height), x : min(x + args.tile, width)]
            result = model.predict(tile, imgsz=args.imgsz, conf=args.fault_conf, device=device, verbose=False)[0]
            detections.extend(result_detections(result, model.names, "fault", (x, y)))
    return nms_named(detections, args.iou)


def draw_detections(image, detections):
    """Render normal equipment in green and fault-specialist results in red."""
    for detection in detections:
        x1, y1, x2, y2 = (round(value) for value in detection.xyxy)
        color = (0, 0, 255) if detection.label in FAULT_CLASSES else (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{detection.label} {detection.confidence:.2f}",
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return image


def save_result(output_dir, path, image, detections):
    """Save annotated image and original-pixel detection records."""
    encoded, buffer = cv2.imencode(path.suffix, draw_detections(image.copy(), detections))
    if not encoded:
        raise ValueError(f"Unable to encode output image: {path.name}")
    buffer.tofile(output_dir / path.name)
    with (output_dir / f"{path.stem}.txt").open("w", encoding="utf-8") as file:
        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy
            file.write(
                f"{detection.label} {detection.confidence:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {detection.source}\n"
            )


def main():
    args = parse_args()
    if not 0 <= args.conf <= 1 or not 0 <= args.fault_conf <= 1:
        raise ValueError("--conf and --fault-conf must be in [0, 1].")
    if not 0 <= args.overlap < 1:
        raise ValueError("--overlap must be in [0, 1).")
    if args.tile and args.tile < 64:
        raise ValueError("--tile must be at least 64 pixels.")
    if not args.model.is_file():
        raise FileNotFoundError(f"General detector weights not found: {args.model}")
    paths = image_paths(args.source)
    if not paths:
        raise FileNotFoundError(f"No supported images found in {args.source}")

    general_model = YOLO(args.model)
    general_model.model.end2end = False  # Use the validated one-to-many head with NMS.
    fault_model = YOLO(args.fault_model) if args.fault_model else None
    if fault_model and set(fault_model.names.values()) != FAULT_CLASSES:
        raise ValueError(
            "--fault-model must use exactly these classes: "
            f"{sorted(FAULT_CLASSES)}, received {fault_model.names}."
        )
    device = 0 if torch.cuda.is_available() else "cpu"
    print("General detection head: one-to-many")
    output_dir = ROOT / "runs" / "detect" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in paths:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {path}")
        detections = predict_full_image(general_model, image, args.imgsz, args.conf, device, "general")
        if fault_model:
            # The specialist owns these labels, so stale general-model fault candidates cannot override it.
            detections = [detection for detection in detections if detection.label not in FAULT_CLASSES]
            detections.extend(predict_fault_tiles(fault_model, image, args, device))
        detections = nms_named(detections, args.iou)
        save_result(output_dir, path, image, detections)
        labels = {detection.label for detection in detections}
        summary.append((path.name, labels, detections))

    with (output_dir / "detection_results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Image", "Detected classes", "Fault decision", "Fault detections"])
        for index, (name, labels, detections) in enumerate(summary, 1):
            faults = [
                f"{detection.label} ({detection.confidence:.2f})"
                for detection in detections
                if detection.label in FAULT_CLASSES
            ]
            writer.writerow([index, name, ", ".join(sorted(labels)), judge_fault(labels), ", ".join(faults)])
    print(f"Saved annotations and report to {output_dir}")


if __name__ == "__main__":
    main()
