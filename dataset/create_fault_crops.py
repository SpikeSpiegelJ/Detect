"""Create train-only multi-scale crops for a selected telecom target class."""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "repartition_v3"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
DEFAULT_TARGET_CLASS_ID = 5


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Root of the repartitioned YOLO dataset containing images/ and labels/.",
    )
    parser.add_argument("--class-id", type=int, default=DEFAULT_TARGET_CLASS_ID, help="YOLO class ID to crop from train.")
    parser.add_argument("--copies", type=int, default=2, help="Crops produced for every target instance.")
    parser.add_argument("--seed", type=int, default=260809)
    parser.add_argument("--output", default="train_cut_context_v3")
    parser.add_argument(
        "--band",
        action="append",
        nargs=4,
        metavar=("OUTPUT", "MIN", "MAX", "COPIES"),
        help="Repeatable multi-scale crop band. Overrides --output, --target-fraction, and --copies when supplied.",
    )
    parser.add_argument(
        "--target-fraction",
        type=float,
        nargs=2,
        default=(0.06, 0.20),
        metavar=("MIN", "MAX"),
        help="Target width or height as a fraction of the crop; use 0.06 0.20 for real-scale fault crops.",
    )
    return parser.parse_args()


def read_labels(path):
    return [[int(c), float(x), float(y), float(w), float(h)] for c, x, y, w, h in (line.split() for line in path.read_text(encoding="utf-8").splitlines())]


def find_image(dataset_root, stem):
    for suffix in IMAGE_SUFFIXES:
        path = dataset_root / "images" / "train" / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def crop_box(label, image_width, image_height, rng, target_fraction):
    """Return a square crop with random context, placing the target at a useful training scale."""
    _, x, y, width, height = label
    target = max(width * image_width, height * image_height)
    fraction = rng.uniform(*target_fraction)
    side = int(np.clip(target / fraction, 256, min(image_width, image_height)))
    cx = x * image_width + rng.uniform(-0.22, 0.22) * side
    cy = y * image_height + rng.uniform(-0.22, 0.22) * side
    x1 = int(np.clip(round(cx - side / 2), 0, image_width - side))
    y1 = int(np.clip(round(cy - side / 2), 0, image_height - side))
    return x1, y1, side


def labels_in_crop(labels, x1, y1, side, image_width, image_height):
    converted = []
    for cls, x, y, width, height in labels:
        left = (x - width / 2) * image_width
        top = (y - height / 2) * image_height
        right = (x + width / 2) * image_width
        bottom = (y + height / 2) * image_height
        left, top = max(left, x1), max(top, y1)
        right, bottom = min(right, x1 + side), min(bottom, y1 + side)
        if right - left < 2 or bottom - top < 2:
            continue
        converted.append(
            [cls, (left + right - 2 * x1) / (2 * side), (top + bottom - 2 * y1) / (2 * side), (right - left) / side, (bottom - top) / side]
        )
    return converted


def write_labels(path, labels):
    path.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in labels), encoding="utf-8")


def create_band(dataset_root, output, target_fraction, copies, target_class_id, rng):
    """Create one scale band of train-only crops for a selected class."""
    image_output = dataset_root / "images" / output
    label_output = dataset_root / "labels" / output
    if any(image_output.glob("*")) or any(label_output.glob("*")):
        raise FileExistsError(f"Crop output already contains files: {output}")
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    created = 0
    for label_path in sorted((dataset_root / "labels" / "train").glob("*.txt")):
        labels = read_labels(label_path)
        targets = [label for label in labels if label[0] == target_class_id]
        if not targets:
            continue
        image_path = find_image(dataset_root, label_path.stem)
        if image_path is None:
            continue
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image_height, image_width = image.shape[:2]
        for target_index, target in enumerate(targets):
            for copy in range(copies):
                x1, y1, side = crop_box(target, image_width, image_height, rng, target_fraction)
                crop = image[y1 : y1 + side, x1 : x1 + side]
                crop_labels = labels_in_crop(labels, x1, y1, side, image_width, image_height)
                if not any(label[0] == target_class_id for label in crop_labels):
                    continue
                stem = f"{label_path.stem}_class{target_class_id}crop{target_index:02d}_{copy:02d}"
                encoded, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 98])
                if not encoded:
                    continue
                buffer.tofile(image_output / f"{stem}.jpg")
                write_labels(label_output / f"{stem}.txt", crop_labels)
                created += 1
    print(f"{output}: created {created} crops for class {target_class_id}")


def crop_bands(args):
    """Return compatible single-band or repeatable multi-band crop settings."""
    if not args.band:
        return [(args.output, args.target_fraction, args.copies)]
    bands = []
    for output, minimum, maximum, copies in args.band:
        bands.append((output, (float(minimum), float(maximum)), int(copies)))
    return bands


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not (dataset_root / "images" / "train").is_dir() or not (dataset_root / "labels" / "train").is_dir():
        raise FileNotFoundError(f"Missing repartitioned train split under {dataset_root}")
    bands = crop_bands(args)
    if args.class_id < 0:
        raise ValueError("--class-id must be non-negative.")
    names = [name for name, _, _ in bands]
    if len(names) != len(set(names)):
        raise ValueError("Each --band output name must be unique.")
    rng = np.random.default_rng(args.seed)
    for output, target_fraction, copies in bands:
        if copies < 1:
            raise ValueError("Crop copies must be at least 1.")
        if not 0 < target_fraction[0] < target_fraction[1] < 1:
            raise ValueError("Each crop band must satisfy 0 < MIN < MAX < 1.")
        create_band(dataset_root, output, target_fraction, copies, args.class_id, rng)


if __name__ == "__main__":
    main()
