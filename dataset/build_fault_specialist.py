"""Build a one-class cut dataset from the telecom detector training corpus."""

from argparse import ArgumentParser
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "repartition_v3"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CUT_CLASS_ID = 5
FAULT_CLASSES = {CUT_CLASS_ID: 0}  # General-detector class ID -> cut-specialist class ID.
DEFAULT_TRAIN_SOURCES = ("train",)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Root of repartition_v3 containing images/ and labels/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_ROOT / "cut_specialist",
        help="Output directory for the one-class cut dataset.",
    )
    parser.add_argument(
        "--train-sources",
        nargs="+",
        default=DEFAULT_TRAIN_SOURCES,
        help="Training image/label subdirectories under --dataset-root.",
    )
    parser.add_argument("--replace", action="store_true", help="Replace an existing generated output directory.")
    return parser.parse_args()


def read_fault_labels(path):
    """Load only cut labels, remapped to the one-class dataset."""
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cls, x, y, width, height = line.split()
        cls = int(cls)
        if cls in FAULT_CLASSES:
            labels.append((FAULT_CLASSES[cls], float(x), float(y), float(width), float(height)))
    return labels


def write_labels(path, labels):
    path.write_text(
        "".join(f"{cls} {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n" for cls, x, y, width, height in labels),
        encoding="utf-8",
    )


def link_or_copy(source, destination):
    """Avoid duplicating source images when the filesystem supports hard links."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_split(dataset_root, output, split, sources):
    image_output = output / "images" / split
    label_output = output / "labels" / split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    counts = {0: 0, 1: 0}
    images = 0
    for source in sources:
        image_dir = dataset_root / "images" / source
        label_dir = dataset_root / "labels" / source
        for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            labels = read_fault_labels(label_dir / f"{image_path.stem}.txt")
            name = f"{source}__{image_path.stem}"
            destination = image_output / f"{name}{image_path.suffix.lower()}"
            link_or_copy(image_path, destination)
            write_labels(label_output / f"{name}.txt", labels)
            images += 1
            for cls, *_ in labels:
                counts[cls] += 1
    return images, counts


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve()
    source_groups = {"train": args.train_sources, "val": ("val",), "test": ("test",)}
    for sources in source_groups.values():
        for source in sources:
            if not (dataset_root / "images" / source).is_dir() or not (dataset_root / "labels" / source).is_dir():
                raise FileNotFoundError(f"Missing image or label directory for source '{source}' under {dataset_root}")
    if output.exists():
        if not args.replace:
            raise FileExistsError(f"{output} already exists. Use --replace to rebuild it.")
        shutil.rmtree(output)
    summary = {
        "train": build_split(dataset_root, output, "train", args.train_sources),
        "val": build_split(dataset_root, output, "val", ("val",)),
        "test": build_split(dataset_root, output, "test", ("test",)),
    }
    for split, (images, counts) in summary.items():
        print(f"{split}: images={images}, cut={counts[0]}")


if __name__ == "__main__":
    main()
