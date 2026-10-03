# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Audit YOLO split integrity without modifying images, labels, or dataset descriptors."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data_repartition_v13_cut_feeder_hard_corrected_9c.yaml"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IDENTITY_RE = re.compile(r"(?:^|_)([0-9a-f]{20})(?=_|$)", re.IGNORECASE)
SPLITS = ("train", "val", "test")


def parse_args():
    """Parse read-only dataset audit arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument("--phash-distance", type=int, default=4)
    parser.add_argument("--thumbnail-mae", type=float, default=12.0)
    return parser.parse_args()


def resolve_entry(data_file: Path, descriptor: dict, entry: str) -> Path:
    """Resolve one dataset entry against the descriptor root."""
    root = Path(descriptor.get("path", data_file.parent))
    if not root.is_absolute():
        root = (data_file.parent / root).resolve()
    path = Path(entry)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def split_images(data_file: Path, descriptor: dict, split: str) -> list[Path]:
    """Expand directory, image-list, or multi-directory split entries."""
    entries = descriptor[split]
    entries = entries if isinstance(entries, list) else [entries]
    images = []
    for entry in entries:
        path = resolve_entry(data_file, descriptor, entry)
        if path.is_file() and path.suffix.lower() == ".txt":
            images.extend(Path(line.strip()).resolve() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
        elif path.is_dir():
            images.extend(item.resolve() for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
        else:
            raise FileNotFoundError(f"Dataset split entry does not exist: {path}")
    return sorted(set(images))


def paired_label(image: Path) -> Path:
    """Resolve the conventional YOLO label path paired with an image."""
    parts = list(image.parts)
    positions = [index for index, part in enumerate(parts) if part == "images"]
    if not positions:
        raise ValueError(f"Image path has no 'images' component: {image}")
    parts[positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def canonical_identity(image: Path) -> str:
    """Return the embedded source-image identity, falling back to the filename stem."""
    match = IDENTITY_RE.search(image.stem)
    return match.group(1).lower() if match else image.stem.lower()


def read_labels(path: Path, class_count: int) -> tuple[Counter, list[str]]:
    """Validate one YOLO label file and return class counts and errors."""
    counts, errors = Counter(), []
    if not path.is_file():
        return counts, [f"Missing label: {path}"]
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        values = line.split()
        if len(values) != 5:
            errors.append(f"{path}:{number} expected 5 values")
            continue
        try:
            class_id = int(float(values[0]))
            x, y, width, height = map(float, values[1:])
        except ValueError:
            errors.append(f"{path}:{number} contains non-numeric values")
            continue
        if not 0 <= class_id < class_count:
            errors.append(f"{path}:{number} class {class_id} is outside 0-{class_count - 1}")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            errors.append(f"{path}:{number} has invalid normalized coordinates")
        if x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6 or y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6:
            errors.append(f"{path}:{number} box exceeds image bounds")
        counts[class_id] += 1
    return counts, errors


def image_signature(path: Path) -> dict:
    """Compute exact decoded-pixel and conservative perceptual signatures."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thumbnail = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    low_frequency = cv2.dct(thumbnail.astype(np.float32))[:8, :8].flatten()[1:]
    median = float(np.median(low_frequency))
    phash = sum(int(value > median) << index for index, value in enumerate(low_frequency))
    pixels = sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
    return {"pixels": pixels, "phash": phash, "aspect": width / height, "thumbnail": thumbnail}


def cross_split_pairs(samples: list[dict], phash_distance: int, thumbnail_mae: float) -> tuple[list[dict], list[dict]]:
    """Return exact decoded-pixel duplicates and conservative near-duplicate candidates."""
    exact, near = [], []
    for index, left in enumerate(samples):
        for right in samples[index + 1 :]:
            if left["split"] == right["split"]:
                continue
            if left["signature"]["pixels"] == right["signature"]["pixels"]:
                exact.append({"a": str(left["image"]), "a_split": left["split"], "b": str(right["image"]), "b_split": right["split"]})
                continue
            distance = (left["signature"]["phash"] ^ right["signature"]["phash"]).bit_count()
            aspect_ratio = left["signature"]["aspect"] / right["signature"]["aspect"]
            if distance > phash_distance or not 0.95 <= aspect_ratio <= 1.05:
                continue
            mae = float(np.abs(left["signature"]["thumbnail"].astype(float) - right["signature"]["thumbnail"].astype(float)).mean())
            if mae <= thumbnail_mae:
                near.append({
                    "a": str(left["image"]),
                    "a_split": left["split"],
                    "b": str(right["image"]),
                    "b_split": right["split"],
                    "phash_distance": distance,
                    "thumbnail_mae": mae,
                })
    return exact, near


def audit(data_file: Path, phash_distance: int = 4, thumbnail_mae: float = 12.0) -> dict:
    """Audit split membership, labels, and cross-split duplicate evidence."""
    data_file = data_file.resolve()
    descriptor = yaml.safe_load(data_file.read_text(encoding="utf-8-sig"))
    class_count = int(descriptor["nc"])
    samples, errors, split_summary = [], [], {}
    identities = {split: set() for split in SPLITS}
    for split in SPLITS:
        counts = Counter()
        images = split_images(data_file, descriptor, split)
        for image in images:
            label_counts, label_errors = read_labels(paired_label(image), class_count)
            counts.update(label_counts)
            errors.extend(label_errors)
            identities[split].add(canonical_identity(image))
            try:
                signature = image_signature(image)
            except ValueError as error:
                errors.append(str(error))
                continue
            samples.append({"split": split, "image": image, "signature": signature})
        split_summary[split] = {"images": len(images), "instances": [counts[index] for index in range(class_count)]}
    identity_overlap = []
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = sorted(identities[left] & identities[right])
            identity_overlap.append({"splits": [left, right], "count": len(overlap), "identities": overlap})
    exact, near = cross_split_pairs(samples, phash_distance, thumbnail_mae)
    return {
        "data": str(data_file),
        "classes": descriptor["names"],
        "splits": split_summary,
        "label_or_image_errors": errors,
        "identity_overlap": identity_overlap,
        "exact_pixel_duplicates": exact,
        "near_duplicate_candidates": near,
        "thresholds": {"phash_distance": phash_distance, "thumbnail_mae": thumbnail_mae, "aspect_ratio_tolerance": 0.05},
        "note": "Near-duplicate candidates are a conservative screen and require visual review before exclusion.",
    }


def main():
    """Run the audit and optionally persist its JSON report."""
    args = parse_args()
    if args.phash_distance < 0 or args.thumbnail_mae < 0:
        raise ValueError("Similarity thresholds must be non-negative")
    report = audit(args.data, args.phash_distance, args.thumbnail_mae)
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "splits": report["splits"],
        "label_or_image_errors": len(report["label_or_image_errors"]),
        "identity_overlap": [{"splits": row["splits"], "count": row["count"]} for row in report["identity_overlap"]],
        "exact_pixel_duplicates": len(report["exact_pixel_duplicates"]),
        "near_duplicate_candidates": len(report["near_duplicate_candidates"]),
        "output": str(args.output.resolve()) if args.output else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
