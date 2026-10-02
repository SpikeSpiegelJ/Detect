"""Build a train-only manifest that replays reviewed model errors or mined difficult samples."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "repartition_v7_dataset8_9c"
WEIGHTS = PROJECT_ROOT / "runs" / "train" / "yolo26s_v7_control_ft-2" / "weights" / "best.pt"
SOURCE_DATA = PROJECT_ROOT / "dataset" / "data_repartition_v7_dataset8_9c.yaml"
FOCUS_CLASSES = {3: "Feeder_RRU", 6: "Feeder_antenna"}
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    """Parse the train-only replay configuration."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--source-data", type=Path, default=SOURCE_DATA)
    parser.add_argument("--manifest-name", default="train_feeder_errors.txt")
    parser.add_argument("--data-name", default="data_v7_feeder_errors.yaml")
    parser.add_argument("--audit-name", default="train_feeder_errors_audit.json")
    parser.add_argument(
        "--review", type=Path, help="Completed review.csv; replay confirmed model_error images instead of mining."
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hard-negative-conf", type=float, default=0.15)
    parser.add_argument("--negative-iou", type=float, default=0.1)
    parser.add_argument("--hard-negative-cap", type=int, default=150)
    parser.add_argument("--candidate-conf", type=float, default=0.001)
    parser.add_argument("--match-iou", type=float, default=0.5)
    return parser.parse_args()


def label_path(image_path: Path) -> Path:
    """Map one train image path to its paired YOLO label path."""
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError as error:
        raise ValueError(f"Image path is not inside an images directory: {image_path}") from error
    return Path(*parts[:index], "labels", *parts[index + 1 :]).with_suffix(".txt")


def load_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load classes and normalized xywh boxes from one YOLO label file."""
    classes, boxes = [], []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number} must contain class plus four xywh values")
        class_id = int(fields[0])
        box = [float(value) for value in fields[1:]]
        if class_id not in range(9) or not all(0.0 <= value <= 1.0 for value in box):
            raise ValueError(f"{path}:{line_number} contains an invalid class or normalized box")
        classes.append(class_id)
        boxes.append(box)
    return np.asarray(classes, dtype=np.int64), np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def xywhn_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized xywh boxes to clipped pixel xyxy boxes."""
    if not len(boxes):
        return np.empty((0, 4), dtype=np.float32)
    x, y, w, h = boxes.T
    output = np.column_stack(((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height))
    output[:, [0, 2]] = output[:, [0, 2]].clip(0, width)
    output[:, [1, 3]] = output[:, [1, 3]].clip(0, height)
    return output.astype(np.float32)


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """Compute IoU for two xyxy box collections without importing model internals."""
    if not len(box_a) or not len(box_b):
        return np.zeros((len(box_a), len(box_b)), dtype=np.float32)
    top_left = np.maximum(box_a[:, None, :2], box_b[None, :, :2])
    bottom_right = np.minimum(box_a[:, None, 2:], box_b[None, :, 2:])
    intersection = np.prod(np.clip(bottom_right - top_left, 0, None), axis=2)
    area_a = np.prod(np.clip(box_a[:, 2:] - box_a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(box_b[:, 2:] - box_b[:, :2], 0, None), axis=1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, np.finfo(np.float32).eps)


def collect_train_images(root: Path) -> list[Path]:
    """Return sorted train images and require one label file for every image."""
    images = sorted(
        path.resolve() for path in (root / "images" / "train").rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(root / "images" / "train")
    missing = [path for path in images if not label_path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing labels for {len(missing)} train images, first: {missing[0]}")
    return images


def mine_difficult_samples(
    images: list[Path],
    labels: dict[Path, tuple[np.ndarray, np.ndarray]],
    weights: Path,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    confidence: float,
    negative_iou: float,
    candidate_conf: float,
    match_iou: float,
) -> tuple[dict[int, dict[Path, float]], dict[Path, list[dict]]]:
    """Find difficult GTs and confident unmatched predictions using training images only."""
    sys.path.insert(0, str(PROJECT_ROOT))
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from ultralytics.data.loaders import LoadImagesAndVideos, SourceTypes
    from ultralytics import YOLO

    model = YOLO(weights)
    if any(model.names.get(k) != v for k, v in FOCUS_CLASSES.items()):
        raise ValueError("Checkpoint focus class mapping does not match the dataset")
    mined = {class_id: {} for class_id in FOCUS_CLASSES}
    difficult = {}
    visited = set()
    source = LoadImagesAndVideos([str(path) for path in images], batch=batch)
    # ``model.predict()`` treats a pre-built loader as an in-memory source and
    # expects it to expose the same source classification used by native loaders.
    source.source_type = SourceTypes()
    results = model.predict(
        source=source,
        imgsz=imgsz,
        device=device,
        workers=workers,
        conf=candidate_conf,
        iou=0.7,
        max_det=300,
        stream=True,
        verbose=False,
    )
    for result in results:
        image = Path(result.path).resolve()
        if image in visited or image not in labels:
            raise ValueError(f"Unexpected or duplicate inference image: {image}")
        visited.add(image)
        gt_classes, gt_boxes_n = labels[image]
        height, width = result.orig_shape
        gt_boxes = xywhn_to_xyxy(gt_boxes_n, width, height)
        predictions = result.boxes.cpu().numpy()
        for class_id in FOCUS_CLASSES:
            candidate_indices = np.flatnonzero(predictions.cls.astype(int) == class_id)
            same_gt = gt_boxes[gt_classes == class_id]
            overlaps = box_iou(same_gt, predictions.xyxy[candidate_indices])
            scores = predictions.conf[candidate_indices]
            # One-to-one assignment avoids treating one prediction as several successful GTs.
            pairs = np.argwhere(overlaps >= match_iou)
            found, used = set(), set()
            for g, p in sorted(pairs.tolist(), key=lambda pair: -overlaps[tuple(pair)]):
                if g not in found and p not in used:
                    found.add(g)
                    used.add(p)
            for g in range(len(same_gt)):
                if g in found:
                    continue
                difficult.setdefault(image, []).append(
                    {
                        "class": FOCUS_CLASSES[class_id],
                        "kind": "unmatched_or_localization",
                        "best_same_class_iou": float(overlaps[g].max()) if len(candidate_indices) else 0.0,
                    }
                )
            for index in candidate_indices:
                if predictions.conf[index] < confidence:
                    continue
                prediction_box = predictions.xyxy[index : index + 1]
                maximum_iou = float(box_iou(prediction_box, same_gt).max()) if len(same_gt) else 0.0
                if maximum_iou < negative_iou:
                    mined[class_id][image] = max(mined[class_id].get(image, 0.0), float(predictions.conf[index]))
    if visited != set(images):
        raise RuntimeError("Mining did not visit every training image")
    return mined, difficult


def select_hard_negatives(mined: dict[int, dict[Path, float]], cap: int) -> dict[Path, list[str]]:
    """Retain the highest-confidence mined images per focus class, retaining reasons after deduplication."""
    selected: dict[Path, list[str]] = {}
    for class_id, by_image in mined.items():
        ranked = sorted(by_image.items(), key=lambda item: (-item[1], str(item[0])))[:cap]
        for image, _ in ranked:
            selected.setdefault(image, []).append(FOCUS_CLASSES[class_id])
    return selected


def sha256_file(path: Path) -> str:
    """Return the SHA-256 for an output manifest or descriptor."""
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    """Build and audit a reversible train-only sampling configuration."""
    args = parse_args()
    if (
        not 0 < args.candidate_conf <= args.hard_negative_conf < 1
        or not 0 <= args.negative_iou < args.match_iou <= 1
        or args.hard_negative_cap < 0
    ):
        raise ValueError("Require 0<hard-negative-conf<1, 0<=negative-iou<1 and a non-negative cap")
    root = args.dataset_root.resolve()
    weights = args.weights.resolve()
    source_data = args.source_data.resolve()
    review = args.review.resolve() if args.review else None
    manifest = root / args.manifest_name
    data_output = source_data.parent / args.data_name
    audit_output = root / args.audit_name
    required_paths = [weights, source_data, root / "images" / "train", root / "labels" / "train"]
    if review:
        required_paths.append(review)
    for required in required_paths:
        if not required.exists():
            raise FileNotFoundError(required)
    for output in (manifest, data_output, audit_output):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    images = collect_train_images(root)
    descriptor = yaml.safe_load(source_data.read_text(encoding="utf-8-sig"))
    if Path(descriptor["path"]).resolve() != root or any(
        descriptor[s] != f"images/{s}" for s in ("train", "val", "test")
    ):
        raise ValueError("Expected the unchanged original train/val/test descriptor")
    if any(descriptor["names"].get(k) != v for k, v in FOCUS_CLASSES.items()):
        raise ValueError("Dataset focus class mapping is incorrect")
    source_files = [
        p
        for kind in ("images", "labels")
        for split in ("train", "val", "test")
        for p in (root / kind / split).rglob("*")
        if p.is_file()
    ] + [source_data, weights]
    if review:
        source_files.append(review)
    fingerprints = {str(p): sha256_file(p) for p in source_files}
    labels = {image: load_labels(label_path(image)) for image in images}
    reviewed = {}
    if review:
        with review.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or not {"source_image", "decision", "class", "error_type"}.issubset(rows[0]):
            raise ValueError("Review must contain source_image, decision, class and error_type columns")
        if any(row["decision"] not in {"acceptable", "annotation_error", "model_error"} for row in rows):
            raise ValueError("Complete every review decision before building the sampling manifest")
        train_images = set(images)
        for row in rows:
            if row["decision"] != "model_error":
                continue
            image = Path(row["source_image"]).resolve()
            if image not in train_images:
                raise ValueError(f"Reviewed model error is not an existing train image: {image}")
            reviewed.setdefault(image, []).append({"class": row["class"], "kind": row["error_type"]})
        if not reviewed:
            raise ValueError("Review contains no confirmed model_error images")
        mined, focus_positive = {class_id: {} for class_id in FOCUS_CLASSES}, {}
    else:
        print(f"Mining train-only hard negatives from {len(images)} images at imgsz={args.imgsz}...", flush=True)
        mined, focus_positive = mine_difficult_samples(
            images,
            labels,
            weights,
            args.imgsz,
            args.batch,
            args.device,
            args.workers,
            args.hard_negative_conf,
            args.negative_iou,
            args.candidate_conf,
            args.match_iou,
        )
    hard_negatives = select_hard_negatives(mined, args.hard_negative_cap)
    extras = {
        image: {
            "focus_positive": focus_positive.get(image, []),
            "hard_negative": hard_negatives.get(image, []),
            "reviewed_model_error": reviewed.get(image, []),
        }
        for image in images
    }
    extras = {image: reasons for image, reasons in extras.items() if any(reasons.values())}

    # Each original remains once. An image receives at most one additional appearance, even if it has several reasons.
    lines = ["./" + image.relative_to(root).as_posix() for image in images]
    lines.extend("./" + image.relative_to(root).as_posix() for image in sorted(extras))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    descriptor["path"] = root.as_posix()
    descriptor["train"] = args.manifest_name
    descriptor["val"] = "images/val"
    descriptor["test"] = "images/test"
    data_output.write_text(yaml.safe_dump(descriptor, allow_unicode=True, sort_keys=False), encoding="utf-8")

    manifest_paths = [root / line for line in lines]
    if any(
        not path.is_file() or label_path(path).resolve().parent != (root / "labels" / "train").resolve()
        for path in manifest_paths
    ):
        raise RuntimeError("Manifest contains a missing image or a path outside labels/train")
    train_root = (root / "images" / "train").resolve()
    if any(train_root not in path.resolve().parents for path in manifest_paths):
        raise RuntimeError("Manifest contains a non-train image")
    base_counts = Counter(int(class_id) for classes, _ in labels.values() for class_id in classes)
    repeated_labels = [labels[image] for image in extras]
    replay_counts = Counter(int(class_id) for classes, _ in repeated_labels for class_id in classes)
    if any(sha256_file(Path(p)) != h for p, h in fingerprints.items()):
        raise RuntimeError("Source data or weights changed during mining")
    audit = {
        "purpose": "Train-only 2x replay of reviewed model errors or mined difficult samples; source data is unchanged.",
        "selection": "completed_review" if review else "model_mining",
        "review": str(review) if review else None,
        "dataset_root": str(root),
        "weights": str(weights),
        "source_data": str(source_data),
        "manifest": str(manifest),
        "data": str(data_output),
        "parameters": (
            {"decision": "model_error", "repeat_factor": 2}
            if review
            else {
                "imgsz": args.imgsz,
                "batch": args.batch,
                "hard_negative_conf": args.hard_negative_conf,
                "negative_iou": args.negative_iou,
                "hard_negative_cap_per_class": args.hard_negative_cap,
                "candidate_conf": args.candidate_conf,
                "match_iou": args.match_iou,
            }
        ),
        "images": {
            "unique_train_images": len(images),
            "manifest_entries": len(lines),
            "extra_entries": len(extras),
            "focus_positive_extra_entries": sum(bool(reason["focus_positive"]) for reason in extras.values()),
            "hard_negative_extra_entries": sum(bool(reason["hard_negative"]) for reason in extras.values()),
            "reviewed_model_error_extra_entries": len(reviewed),
            "max_appearances_per_image": 2,
        },
        "hard_negatives_mined_before_cap": {FOCUS_CLASSES[key]: len(value) for key, value in mined.items()},
        "hard_negatives_selected_after_cap": {
            FOCUS_CLASSES[key]: sum(FOCUS_CLASSES[key] in classes for classes in hard_negatives.values())
            for key in FOCUS_CLASSES
        },
        "instances": {
            "base": {str(key): base_counts[key] for key in range(9)},
            "added_replay": {str(key): replay_counts[key] for key in range(9)},
            "effective_manifest": {str(key): base_counts[key] + replay_counts[key] for key in range(9)},
        },
        "focus_extra_reasons": {
            image.relative_to(root).as_posix(): reasons for image, reasons in sorted(extras.items())
        },
        "verification": {
            "source_files_unchanged": True,
            "source_sha256": fingerprints,
            "all_entries_are_train_images": True,
            "all_entries_have_train_labels": True,
            "val_or_test_entries": 0,
            "manifest_sha256": sha256_file(manifest),
            "data_sha256": sha256_file(data_output),
        },
    }
    audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest_entries": len(lines),
                "extra_entries": len(extras),
                "positive_extras": audit["images"]["focus_positive_extra_entries"],
                "hard_negative_extras": audit["images"]["hard_negative_extra_entries"],
                "reviewed_model_error_extras": len(reviewed),
                "manifest": str(manifest),
                "data": str(data_output),
                "audit": str(audit_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
