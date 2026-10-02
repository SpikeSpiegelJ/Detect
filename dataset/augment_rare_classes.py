"""Build a train-only augmented view of an existing YOLO detection dataset."""

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_class_limits(value):
    """Parse ``class_id:image_count`` pairs."""
    limits = {}
    for item in value.split(","):
        class_id, count = item.split(":", 1)
        limits[int(class_id)] = int(count)
    if not limits or any(class_id < 0 or count <= 0 for class_id, count in limits.items()):
        raise ValueError("Class limits must contain positive counts, for example 1:160,3:240.")
    return limits


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--source-yaml", type=Path, default=ROOT / "data_repartition_v9_trainval_9c.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "repartition_v10_augmented_9c")
    parser.add_argument("--output-yaml", type=Path, default=ROOT / "data_repartition_v10_augmented_9c.yaml")
    parser.add_argument("--class-limits", type=parse_class_limits, default="1:160,3:240,5:200,6:240")
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--seed", type=int, default=260925)
    parser.add_argument("--mode", choices=("full", "target-crop"), default="full")
    parser.add_argument("--occlusion-prob", type=float, default=0.35)
    return parser.parse_args()


def read_labels(path):
    """Read one YOLO label file."""
    labels = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if not values:
            continue
        if len(values) != 5:
            raise ValueError(f"Expected five columns in {path}:{line_number}, found {len(values)}.")
        labels.append([int(values[0]), *(float(value) for value in values[1:])])
    return labels


def write_labels(path, labels):
    """Write YOLO labels using stable numeric formatting."""
    path.write_text(
        "".join(f"{int(cls)} {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n" for cls, x, y, width, height in labels),
        encoding="utf-8",
    )


def resolve_dataset_root(source_yaml, data):
    """Resolve a dataset root relative to its YAML file."""
    root = Path(data.get("path", source_yaml.parent))
    return root if root.is_absolute() else (source_yaml.parent / root).resolve()


def find_image(image_dir, stem):
    """Find the image paired with a label stem."""
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def transform(image, labels, rng):
    """Apply conservative photometric and affine transforms to an image and its boxes."""
    height, width = image.shape[:2]
    image = cv2.convertScaleAbs(image, alpha=rng.uniform(0.85, 1.15), beta=rng.uniform(-15, 15))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-3, 3)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.85, 1.15), 0, 255)
    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    labels = [label.copy() for label in labels]
    if rng.random() < 0.5:
        image = cv2.flip(image, 1)
        for label in labels:
            label[1] = 1 - label[1]

    matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), rng.uniform(-2.0, 2.0), rng.uniform(0.97, 1.04)
    )
    matrix[:, 2] += rng.uniform(-0.02, 0.02) * width, rng.uniform(-0.02, 0.02) * height
    image = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)

    transformed = []
    for cls, x, y, box_width, box_height in labels:
        corners = np.array(
            [
                [x - box_width / 2, y - box_height / 2, 1],
                [x + box_width / 2, y - box_height / 2, 1],
                [x + box_width / 2, y + box_height / 2, 1],
                [x - box_width / 2, y + box_height / 2, 1],
            ],
            dtype=np.float32,
        )
        corners[:, 0] *= width
        corners[:, 1] *= height
        warped = corners @ matrix.T
        x1, y1 = np.clip(warped.min(axis=0), (0, 0), (width, height))
        x2, y2 = np.clip(warped.max(axis=0), (0, 0), (width, height))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            transformed.append(
                [cls, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height]
            )

    if rng.random() < 0.20:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    if rng.random() < 0.25:
        noise = rng.normal(0, rng.uniform(1.0, 4.0), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image, transformed


def transform_target_crop(image, labels, focus_class, rng, occlusion_prob):
    """Crop around one weak-class target while retaining surrounding connection context."""
    height, width = image.shape[:2]
    focus_indices = [index for index, label in enumerate(labels) if int(label[0]) == focus_class]
    focus_index = int(rng.choice(focus_indices))
    _, x, y, box_width, box_height = labels[focus_index]
    crop_width = min(
        width,
        max(box_width * width * rng.uniform(2.0, 3.5), width * rng.uniform(0.45, 0.65)),
    )
    crop_height = min(
        height,
        max(box_height * height * rng.uniform(1.6, 2.8), height * rng.uniform(0.45, 0.65)),
    )
    center_x = x * width + rng.uniform(-0.06, 0.06) * crop_width
    center_y = y * height + rng.uniform(-0.06, 0.06) * crop_height
    left = int(np.clip(center_x - crop_width / 2, 0, width - crop_width))
    top = int(np.clip(center_y - crop_height / 2, 0, height - crop_height))
    right = int(round(left + crop_width))
    bottom = int(round(top + crop_height))
    crop_width, crop_height = right - left, bottom - top

    transformed = []
    transformed_focus = None
    for index, (cls, bx, by, bw, bh) in enumerate(labels):
        x1, y1 = (bx - bw / 2) * width, (by - bh / 2) * height
        x2, y2 = (bx + bw / 2) * width, (by + bh / 2) * height
        clipped_x1, clipped_y1 = max(x1, left), max(y1, top)
        clipped_x2, clipped_y2 = min(x2, right), min(y2, bottom)
        intersection = max(0, clipped_x2 - clipped_x1) * max(0, clipped_y2 - clipped_y1)
        if intersection / max((x2 - x1) * (y2 - y1), 1) < 0.55:
            continue
        new_label = [
            cls,
            ((clipped_x1 + clipped_x2) / 2 - left) / crop_width,
            ((clipped_y1 + clipped_y2) / 2 - top) / crop_height,
            (clipped_x2 - clipped_x1) / crop_width,
            (clipped_y2 - clipped_y1) / crop_height,
        ]
        transformed.append(new_label)
        if index == focus_index:
            transformed_focus = new_label
    if transformed_focus is None:
        raise RuntimeError("Target crop removed its focus object.")

    cropped = image[top:bottom, left:right]
    cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    cropped = cv2.convertScaleAbs(cropped, alpha=rng.uniform(0.90, 1.10), beta=rng.uniform(-10, 10))
    if rng.random() < 0.5:
        cropped = cv2.flip(cropped, 1)
        for label in transformed:
            label[1] = 1 - label[1]

    if rng.random() < occlusion_prob:
        _, fx, fy, fw, fh = transformed_focus
        target_width, target_height = max(3, int(fw * width)), max(3, int(fh * height))
        occlusion_width = max(3, int(target_width * rng.uniform(0.12, 0.25)))
        occlusion_height = max(3, int(target_height * rng.uniform(0.12, 0.25)))
        target_x1, target_y1 = int((fx - fw / 2) * width), int((fy - fh / 2) * height)
        target_x2, target_y2 = int((fx + fw / 2) * width), int((fy + fh / 2) * height)
        occlusion_x = int(rng.choice((target_x1, max(target_x1, target_x2 - occlusion_width))))
        occlusion_y = int(rng.choice((target_y1, max(target_y1, target_y2 - occlusion_height))))
        occlusion_x = int(np.clip(occlusion_x, 0, width - occlusion_width))
        occlusion_y = int(np.clip(occlusion_y, 0, height - occlusion_height))
        patch = cropped[
            occlusion_y : occlusion_y + occlusion_height,
            occlusion_x : occlusion_x + occlusion_width,
        ]
        cropped[
            occlusion_y : occlusion_y + occlusion_height,
            occlusion_x : occlusion_x + occlusion_width,
        ] = cv2.GaussianBlur(patch, (0, 0), sigmaX=max(2, min(occlusion_width, occlusion_height) / 3))
    return cropped, transformed


def select_labels(label_dir, class_limits, rng):
    """Select a deterministic union of training images for each target class."""
    labels_by_path = {}
    paths_by_class = {class_id: [] for class_id in class_limits}
    for path in sorted(label_dir.glob("*.txt")):
        labels = read_labels(path)
        labels_by_path[path] = labels
        classes = {int(label[0]) for label in labels}
        for class_id in classes & class_limits.keys():
            paths_by_class[class_id].append(path)

    selected = set()
    for class_id, limit in class_limits.items():
        candidates = paths_by_class[class_id].copy()
        rng.shuffle(candidates)
        selected.update(candidates[:limit])
    return sorted(selected), labels_by_path


def select_targets(label_dir, class_limits, rng):
    """Select class-balanced image/class pairs for target-centered augmentation."""
    labels_by_path = {}
    paths_by_class = {class_id: [] for class_id in class_limits}
    for path in sorted(label_dir.glob("*.txt")):
        labels = read_labels(path)
        labels_by_path[path] = labels
        classes = {int(label[0]) for label in labels}
        for class_id in classes & class_limits.keys():
            paths_by_class[class_id].append(path)
    selected = []
    for class_id, limit in class_limits.items():
        candidates = paths_by_class[class_id].copy()
        rng.shuffle(candidates)
        selected.extend((path, class_id) for path in candidates[:limit])
    return selected, labels_by_path


def main():
    args = parse_args()
    if args.variants <= 0:
        raise ValueError("--variants must be positive.")
    source_yaml = args.source_yaml.resolve()
    data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    source_root = resolve_dataset_root(source_yaml, data)
    image_dir = source_root / "images" / "train"
    label_dir = source_root / "labels" / "train"
    output_root = args.output.resolve()
    output_images = output_root / "images" / "train"
    output_labels = output_root / "labels" / "train"
    if output_images.exists() and any(output_images.iterdir()):
        raise FileExistsError(f"Output already contains images: {output_images}")
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    if not 0 <= args.occlusion_prob <= 1:
        raise ValueError("--occlusion-prob must be between zero and one.")
    if args.mode == "target-crop":
        selected, labels_by_path = select_targets(label_dir, args.class_limits, rng)
    else:
        paths, labels_by_path = select_labels(label_dir, args.class_limits, rng)
        selected = [(path, None) for path in paths]
    created_instances = Counter()
    missing_images = []
    for label_path, focus_class in selected:
        image_path = find_image(image_dir, label_path.stem)
        if image_path is None:
            missing_images.append(label_path.name)
            continue
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode {image_path}")
        labels = labels_by_path[label_path]
        for variant in range(args.variants):
            if args.mode == "target-crop":
                augmented, transformed_labels = transform_target_crop(
                    image, labels, focus_class, rng, args.occlusion_prob
                )
                name = f"crop_c{focus_class}_{image_path.stem}_{variant:02d}.jpg"
            else:
                augmented, transformed_labels = transform(image, labels, rng)
                name = f"aug_{image_path.stem}_{variant:02d}.jpg"
            encoded, buffer = cv2.imencode(".jpg", augmented, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(88, 98))])
            if not encoded:
                raise ValueError(f"Unable to encode augmented image from {image_path}")
            buffer.tofile(output_images / name)
            write_labels(output_labels / f"{Path(name).stem}.txt", transformed_labels)
            created_instances.update(int(label[0]) for label in transformed_labels)

    output_data = {
        "train": [str((source_root / "images" / "train").resolve()), str(output_images.resolve())],
        "val": str((source_root / "images" / "val").resolve()),
        "test": str((source_root / "images" / "test").resolve()),
        "nc": data["nc"],
        "names": data["names"],
    }
    args.output_yaml.resolve().write_text(
        yaml.safe_dump(output_data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    manifest = {
        "source_yaml": str(source_yaml),
        "seed": args.seed,
        "class_limits": args.class_limits,
        "mode": args.mode,
        "occlusion_prob": args.occlusion_prob if args.mode == "target-crop" else 0.0,
        "variants": args.variants,
        "selected_source_images": len(selected),
        "created_images": len(list(output_images.glob("*.jpg"))),
        "created_instances": dict(sorted(created_instances.items())),
        "missing_images": missing_images,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Dataset YAML: {args.output_yaml.resolve()}")


if __name__ == "__main__":
    main()
