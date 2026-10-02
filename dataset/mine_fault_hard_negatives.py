"""Mine difficult cut-free crops for the one-class cut specialist.

The script only reads original training images without ``cut`` annotations. It saves crops around the specialist's high-confidence
false-positive candidates with empty YOLO labels. These hard negatives teach
the second-stage detector not to mistake ordinary equipment, rails, cables,
and complex rooftops for a fault.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from detect import tile_origins
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "repartition_v3"
SPECIALIST_ROOT = DATASET_ROOT / "cut_specialist"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FAULT_CLASS_IDS = {5}  # General-detector class ID: cut.
FAULT_CLASS_NAMES = {"cut"}


def parse_args():
    """Parse hard-negative mining options."""
    parser = ArgumentParser(description="Mine tiled false positives as fault-free crops.")
    parser.add_argument("--model", type=Path, required=True, help="Trained one-class cut-specialist weights.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Root of repartition_v3 used as the fault-free image source.",
    )
    parser.add_argument(
        "--specialist-root",
        type=Path,
        default=SPECIALIST_ROOT,
        help="Existing fault-specialist dataset that receives hard negatives.",
    )
    parser.add_argument("--source", default="train", choices=("train",), help="Original dataset split to mine.")
    parser.add_argument("--output", default="train_hard_negatives", help="Output split under cut_specialist.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Specialist inference size.")
    parser.add_argument("--tile", type=int, default=1280, help="Square tile size in source pixels.")
    parser.add_argument("--overlap", type=float, default=0.25, help="Adjacent-tile overlap ratio.")
    parser.add_argument("--conf", type=float, default=0.01, help="Low candidate threshold before ranking.")
    parser.add_argument("--crop-size", type=int, default=1280, help="Square hard-negative crop size.")
    parser.add_argument("--max-per-image", type=int, default=2, help="Maximum crops retained from one image.")
    parser.add_argument("--max-images", type=int, default=600, help="Maximum hard-negative crops to create.")
    return parser.parse_args()


def has_fault_label(label_path: Path) -> bool:
    """Return whether a source image already contains a cut target."""
    if not label_path.exists():
        return False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and int(float(fields[0])) in FAULT_CLASS_IDS:
            return True
    return False


def crop_centered(image: np.ndarray, center_x: float, center_y: float, size: int) -> np.ndarray:
    """Return a padded square crop centred on a candidate location."""
    height, width = image.shape[:2]
    half = size // 2
    left = int(round(center_x)) - half
    top = int(round(center_y)) - half
    right, bottom = left + size, top + size
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - width), max(0, bottom - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
        left += pad_left
        top += pad_top
    return image[top : top + size, left : left + size]


def candidate_centers(model: YOLO, image: np.ndarray, args) -> list[tuple[float, float, float]]:
    """Collect and rank tiled specialist candidates in original-image coordinates."""
    height, width = image.shape[:2]
    candidates = []
    for top in tile_origins(height, args.tile, args.overlap):
        for left in tile_origins(width, args.tile, args.overlap):
            tile = image[top : min(top + args.tile, height), left : min(left + args.tile, width)]
            result = model.predict(tile, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            if result.boxes is None or not len(result.boxes):
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), score in zip(boxes, scores):
                candidates.append((float(score), left + float(x1 + x2) / 2, top + float(y1 + y2) / 2))
    return sorted(candidates, reverse=True)


def save_image(path: Path, image: np.ndarray) -> None:
    """Save an image reliably when the output path contains non-ASCII characters."""
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError(f"Unable to encode {path}")
    encoded.tofile(str(path))


def main():
    """Mine fault-free hard-negative crops from original training images."""
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Specialist weights not found: {args.model}")
    if not 0 <= args.overlap < 1:
        raise ValueError("--overlap must be in [0, 1).")

    dataset_root = args.dataset_root.resolve()
    source_images = dataset_root / "images" / args.source
    source_labels = dataset_root / "labels" / args.source
    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError(f"Missing source split '{args.source}' under {dataset_root}")
    model = YOLO(args.model)
    if set(model.names.values()) != FAULT_CLASS_NAMES:
        raise ValueError(
            "--model must be the one-class cut-specialist model with classes "
            f"{sorted(FAULT_CLASS_NAMES)}, received {model.names}."
        )
    target_root = args.specialist_root.resolve()
    target_images = target_root / "images" / args.output
    target_labels = target_root / "labels" / args.output
    if (target_images.exists() and any(target_images.iterdir())) or (target_labels.exists() and any(target_labels.iterdir())):
        raise FileExistsError(f"Hard-negative output already contains files: {args.output}")
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)
    saved = 0
    scanned = 0
    for image_path in sorted(source_images.iterdir()):
        if saved >= args.max_images:
            break
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = source_labels / f"{image_path.stem}.txt"
        if has_fault_label(label_path):
            continue
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        scanned += 1
        for rank, (_, center_x, center_y) in enumerate(candidate_centers(model, image, args)[: args.max_per_image]):
            if saved >= args.max_images:
                break
            stem = f"hardneg_{image_path.stem}_{rank:02d}"
            save_image(target_images / f"{stem}.jpg", crop_centered(image, center_x, center_y, args.crop_size))
            (target_labels / f"{stem}.txt").write_text("", encoding="utf-8")
            saved += 1

    print(f"Scanned {scanned} fault-free images and created {saved} hard-negative crops in {target_images}.")


if __name__ == "__main__":
    main()
