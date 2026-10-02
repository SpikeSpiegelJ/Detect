# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build a leakage-free hard validation split for cut and Feeder_antenna experiments."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import re
import shutil

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data_repartition_v9_trainval_9c.yaml"
DEFAULT_OUTPUT = ROOT / "repartition_v13_cut_feeder_hard"
TARGET_CLASSES = (5, 6)
CONFUSER_CLASSES = {1, 3, 4}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IDENTITY_RE = re.compile(r"([0-9a-f]{20})(?:\.[^.]+)?$", re.IGNORECASE)


def parse_args():
    """Parse dataset preparation arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--target-images", type=int, default=140, help="Minimum validation images containing each target.")
    parser.add_argument("--seed", type=int, default=260927)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def identity(path: Path) -> str:
    """Return the stable source identity embedded in generated filenames."""
    match = IDENTITY_RE.search(path.stem)
    return match.group(1).lower() if match else path.stem.lower()


def label_path(image: Path) -> Path:
    """Resolve the YOLO label paired with an image."""
    parts = list(image.parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read one YOLO label file."""
    return [
        (int(float(cls)), float(x), float(y), float(width), float(height))
        for cls, x, y, width, height in (line.split() for line in path.read_text(encoding="utf-8").splitlines())
    ]


def difficulty(labels: list[tuple[int, float, float, float, float]]) -> float:
    """Rank small target objects, target/confuser co-occurrence, and dense scenes as difficult."""
    target = [row for row in labels if row[0] in TARGET_CLASSES]
    score = 5.0 * len(target) + min(len(labels), 8) / 8
    if target:
        smallest_area = min(width * height for _, _, _, width, height in target)
        score += min(4.0, max(0.0, -math.log10(max(smallest_area, 1e-8)) - 1.0))
        score += 1.5 * bool({row[0] for row in labels} & CONFUSER_CLASSES)
    return score


def collect_images(root: Path, split: str) -> list[Path]:
    """Collect images from one materialized split."""
    directory = root / "images" / split
    return sorted(path.resolve() for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def link_or_copy(source: Path, destination: Path) -> None:
    """Create a space-efficient hard link, falling back to a file copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def materialize(records: list[dict], output: Path, split: str) -> None:
    """Materialize one split with paired image and label files."""
    for record in records:
        image = record["image"]
        label = record["label"]
        link_or_copy(image, output / "images" / split / image.name)
        link_or_copy(label, output / "labels" / split / f"{image.stem}.txt")


def class_counts(records: list[dict]) -> list[int]:
    """Count instances for all nine detector classes."""
    counts = Counter(row[0] for record in records for row in record["labels"])
    return [counts[index] for index in range(9)]


def main():
    """Create the fixed hard validation split and its training complement."""
    args = parse_args()
    descriptor = yaml.safe_load(args.data.resolve().read_text(encoding="utf-8-sig"))
    source_root = Path(descriptor["path"]).resolve()
    output = args.output.resolve()
    if output.exists():
        if not args.replace:
            raise FileExistsError(f"{output} exists; pass --replace to rebuild it")
        shutil.rmtree(output)

    records = []
    for split in ("train", "val"):
        for image in collect_images(source_root, split):
            label = label_path(image)
            labels = read_labels(label)
            records.append(
                {
                    "image": image,
                    "label": label,
                    "labels": labels,
                    "identity": identity(image),
                    "difficulty": difficulty(labels),
                }
            )
    if len({record["identity"] for record in records}) != len(records):
        raise RuntimeError("Development images contain repeated source identities; group them before splitting")
    if not 0 < args.val_size < len(records):
        raise ValueError("--val-size must leave at least one training image")

    rng = random.Random(args.seed)
    rng.shuffle(records)
    selected: dict[str, dict] = {}
    for class_id in TARGET_CLASSES:
        candidates = [record for record in records if any(row[0] == class_id for row in record["labels"])]
        candidates.sort(key=lambda record: record["difficulty"], reverse=True)
        for record in candidates:
            if sum(any(row[0] == class_id for row in item["labels"]) for item in selected.values()) >= args.target_images:
                break
            selected[record["identity"]] = record
    non_target = [
        record for record in records if not any(row[0] in TARGET_CLASSES for row in record["labels"])
    ]
    for record in sorted(non_target, key=lambda item: item["difficulty"], reverse=True):
        if len(selected) >= args.val_size:
            break
        selected[record["identity"]] = record
    for record in sorted(records, key=lambda item: item["difficulty"], reverse=True):
        if len(selected) >= args.val_size:
            break
        selected[record["identity"]] = record
    if len(selected) != args.val_size:
        raise RuntimeError(f"Selected {len(selected)} validation images, expected {args.val_size}")

    validation = sorted(selected.values(), key=lambda record: record["image"].name)
    training = sorted(
        (record for record in records if record["identity"] not in selected), key=lambda record: record["image"].name
    )
    test = collect_images(source_root, "test")
    test_identities = {identity(image) for image in test}
    overlap = test_identities & {record["identity"] for record in records}
    if overlap:
        raise RuntimeError(f"Frozen test overlaps development identities: {sorted(overlap)[:5]}")

    materialize(training, output, "train")
    materialize(validation, output, "val")
    yaml_path = ROOT / "data_repartition_v13_cut_feeder_hard_9c.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": output.as_posix(),
                "train": ["images/train", "images/train_cut_context", "images/train_feeder_antenna_context"],
                "val": "images/val",
                "test": (source_root / "images" / "test").as_posix(),
                "nc": 9,
                "names": descriptor["names"],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    summary = {
        "source_yaml": str(args.data.resolve()),
        "yaml": str(yaml_path.resolve()),
        "seed": args.seed,
        "train_images": len(training),
        "val_images": len(validation),
        "test_images": len(test),
        "train_instances": class_counts(training),
        "val_instances": class_counts(validation),
        "val_target_images": {
            str(class_id): sum(any(row[0] == class_id for row in record["labels"]) for record in validation)
            for class_id in TARGET_CLASSES
        },
        "train_val_identity_overlap": sorted(
            {record["identity"] for record in training} & {record["identity"] for record in validation}
        ),
        "development_test_identity_overlap": sorted(overlap),
        "val": [str(record["image"]) for record in validation],
    }
    (output / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "val"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
