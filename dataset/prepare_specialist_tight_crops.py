# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Create target-tight specialist crops and a full-image-plus-crop training manifest."""

from __future__ import annotations

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from dataset.prepare_specialist_ablation_configs import images, label_classes, paired_label


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path("E:/Python-program/Detect/ultralytics/dataset/repartition_v14_cut_feeder_specialist")
DEFAULT_OUTPUT = ROOT.parent / "runs" / "datasets" / "specialist_tight_crops"
DEFAULT_MANIFEST = ROOT / "specialist_ablation_manifests" / "full_positive_tight.txt"
DEFAULT_DATA = ROOT / "specialist_ablation_manifests" / "full_positive_tight.yaml"


def parse_args():
    """Parse tight-crop generation settings."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--margin", type=float, default=0.15)
    return parser.parse_args()


def tight_bounds(box: list[float], width: int, height: int, margin: float) -> tuple[int, int, int, int]:
    """Return clipped pixel bounds around one normalized xywh target."""
    _, x, y, box_width, box_height = box
    half_width = box_width * width * (0.5 + margin)
    half_height = box_height * height * (0.5 + margin)
    x1 = max(0, int(np.floor(x * width - half_width)))
    y1 = max(0, int(np.floor(y * height - half_height)))
    x2 = min(width, int(np.ceil(x * width + half_width)))
    y2 = min(height, int(np.ceil(y * height + half_height)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("Tight crop is smaller than two pixels")
    return x1, y1, x2, y2


def remap_box(box: list[float], bounds: tuple[int, int, int, int], width: int, height: int) -> list[float]:
    """Map one normalized source box into normalized crop coordinates."""
    class_id, x, y, box_width, box_height = box
    x1, y1, x2, y2 = bounds
    crop_width, crop_height = x2 - x1, y2 - y1
    return [
        class_id,
        (x * width - x1) / crop_width,
        (y * height - y1) / crop_height,
        box_width * width / crop_width,
        box_height * height / crop_height,
    ]


def read_boxes(path: Path) -> list[list[float]]:
    """Read a two-class YOLO label file."""
    boxes = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{path}:{number} expected five values")
        class_id = int(values[0])
        box = [class_id, *(float(value) for value in values[1:])]
        if class_id not in (0, 1):
            raise ValueError(f"{path}:{number} has unexpected class {class_id}")
        boxes.append(box)
    return boxes


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a reproducibility artifact."""
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    """Materialize tight crops without changing source images or labels."""
    args = parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not 0 <= args.margin <= 1:
        raise ValueError("--margin must be between 0 and 1")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    positive = [image for image in images(source / "images" / "train") if label_classes(paired_label(image))]
    if not positive:
        raise RuntimeError("No positive specialist images found")
    image_output, label_output = output / "images" / "train", output / "labels" / "train"
    image_output.mkdir(parents=True)
    label_output.mkdir(parents=True)
    crops, instances = [], [0, 0]
    for image_path in positive:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable image: {image_path}")
        boxes = read_boxes(paired_label(image_path))
        if len(boxes) != 1:
            raise ValueError(f"Expected one specialist target in {image_path}, found {len(boxes)}")
        height, width = image.shape[:2]
        bounds = tight_bounds(boxes[0], width, height, args.margin)
        x1, y1, x2, y2 = bounds
        crop = image[y1:y2, x1:x2]
        stem = f"{image_path.stem}_tight_m{int(round(args.margin * 100)):02d}"
        crop_path = image_output / f"{stem}.jpg"
        encoded, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 98])
        if not encoded:
            raise OSError(f"Failed to encode crop for {image_path}")
        buffer.tofile(crop_path)
        mapped = remap_box(boxes[0], bounds, width, height)
        (label_output / f"{stem}.txt").write_text(
            f"{int(mapped[0])} {mapped[1]:.6f} {mapped[2]:.6f} {mapped[3]:.6f} {mapped[4]:.6f}\n",
            encoding="utf-8",
        )
        instances[int(mapped[0])] += 1
        crops.append(crop_path)
    manifest = args.manifest.resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(f"{path.resolve()}\n" for path in [*positive, *crops]), encoding="utf-8")
    descriptor = {
        "path": source.as_posix(),
        "train": manifest.as_posix(),
        "val": (source / "images" / "val").as_posix(),
        "test": (source / "images" / "test").as_posix(),
        "nc": 2,
        "names": {0: "cut", 1: "Feeder_antenna"},
    }
    data = args.data.resolve()
    data.write_text(yaml.safe_dump(descriptor, sort_keys=False, allow_unicode=True), encoding="utf-8")
    audit = {
        "source": str(source),
        "output": str(output),
        "margin": args.margin,
        "full_positive_images": len(positive),
        "tight_crop_images": len(crops),
        "tight_crop_instances": instances,
        "train_images": len(positive) + len(crops),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "data": str(data),
        "data_sha256": sha256_file(data),
        "context_excluded": "Each crop label contains only its single target; surrounding device labels are removed.",
    }
    (data.with_suffix(".audit.json")).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
