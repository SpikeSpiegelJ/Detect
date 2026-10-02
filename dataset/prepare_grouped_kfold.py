"""Create grouped, class-balanced YOLO cross-validation folds without copying images."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HASH_PATTERN = re.compile(r"([0-9a-f]{20})$", re.IGNORECASE)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--source-yaml", type=Path, default=ROOT / "data_repartition_v9_trainval_9c.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "repartition_v12_grouped_3fold")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=260927)
    return parser.parse_args()


def canonical_id(path):
    """Return the source-image identity encoded in the normalized filename."""
    match = HASH_PATTERN.search(path.stem)
    return match.group(1).lower() if match else path.stem.lower()


def read_classes(label_path, class_count):
    """Count instances by class in one YOLO label file."""
    counts = np.zeros(class_count, dtype=np.int64)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            counts[int(line.split()[0])] += 1
    return counts


def paired_label(image_path):
    """Resolve the standard YOLO label path paired with an image."""
    parts = list(image_path.parts)
    index = max(i for i, part in enumerate(parts) if part == "images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def resolve_split(source_yaml, data, split):
    """Resolve one image-directory split from a dataset YAML."""
    root = Path(data.get("path", source_yaml.parent))
    if not root.is_absolute():
        root = (source_yaml.parent / root).resolve()
    path = Path(data[split])
    return path if path.is_absolute() else root / path


def assign_groups(groups, folds, class_count, seed):
    """Greedily balance grouped images by class instances and image count."""
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    totals = np.sum([item[1]["counts"] for item in items], axis=0)
    rarity = 1 / np.maximum(totals, 1)
    items.sort(key=lambda item: float(np.sum(item[1]["counts"] * rarity)), reverse=True)
    fold_counts = np.zeros((folds, class_count), dtype=np.float64)
    fold_images = np.zeros(folds, dtype=np.float64)
    assignments = {}
    target_counts = totals / folds
    total_images = sum(len(item[1]["images"]) for item in items)
    capacities = np.full(folds, total_images // folds, dtype=np.int64)
    capacities[: total_images % folds] += 1
    for group_id, group in items:
        group_size = len(group["images"])
        candidates = [fold for fold in range(folds) if fold_images[fold] + group_size <= capacities[fold]]
        if not candidates:
            candidates = list(range(folds))
        deficits = np.maximum(target_counts - fold_counts, 0) / np.maximum(target_counts, 1)
        chosen = max(
            candidates,
            key=lambda fold: (
                float(np.sum(group["counts"] * deficits[fold])),
                -fold_images[fold] / capacities[fold],
                -fold,
            ),
        )
        assignments[group_id] = chosen
        fold_counts[chosen] += group["counts"]
        fold_images[chosen] += len(group["images"])
    return assignments


def main():
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least two.")
    source_yaml = args.source_yaml.resolve()
    data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    class_count = int(data["nc"])
    development_images = []
    for split in ("train", "val"):
        split_dir = resolve_split(source_yaml, data, split)
        development_images.extend(
            path.resolve() for path in split_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    test_dir = resolve_split(source_yaml, data, "test")
    test_images = [
        path.resolve() for path in test_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]

    groups = defaultdict(lambda: {"images": [], "counts": np.zeros(class_count, dtype=np.int64)})
    for image_path in sorted(development_images):
        label_path = paired_label(image_path)
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
        group = groups[canonical_id(image_path)]
        group["images"].append(image_path)
        group["counts"] += read_classes(label_path, class_count)

    test_ids = {canonical_id(path) for path in test_images}
    overlap = sorted(test_ids & groups.keys())
    if overlap:
        raise ValueError(f"Found {len(overlap)} source identities in both development and test, e.g. {overlap[:5]}")
    assignments = assign_groups(groups, args.folds, class_count, args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {"source_yaml": str(source_yaml), "seed": args.seed, "folds": []}
    all_group_ids = set(groups)
    for fold in range(args.folds):
        val_ids = {group_id for group_id, assigned in assignments.items() if assigned == fold}
        train_ids = all_group_ids - val_ids
        if train_ids & val_ids:
            raise AssertionError("A source identity appears in both train and val.")
        train_images = sorted(path for group_id in train_ids for path in groups[group_id]["images"])
        val_images = sorted(path for group_id in val_ids for path in groups[group_id]["images"])
        train_txt = output / f"fold{fold + 1}_train.txt"
        val_txt = output / f"fold{fold + 1}_val.txt"
        train_txt.write_text("\n".join(map(str, train_images)) + "\n", encoding="utf-8")
        val_txt.write_text("\n".join(map(str, val_images)) + "\n", encoding="utf-8")
        fold_yaml = output / f"fold{fold + 1}.yaml"
        fold_data = {
            "train": str(train_txt),
            "val": str(val_txt),
            "test": str(test_dir.resolve()),
            "nc": class_count,
            "names": data["names"],
        }
        fold_yaml.write_text(yaml.safe_dump(fold_data, sort_keys=False), encoding="utf-8")
        val_counts = np.sum([groups[group_id]["counts"] for group_id in val_ids], axis=0)
        summary["folds"].append(
            {
                "fold": fold + 1,
                "train_images": len(train_images),
                "val_images": len(val_images),
                "train_groups": len(train_ids),
                "val_groups": len(val_ids),
                "val_instances": val_counts.tolist(),
                "yaml": str(fold_yaml),
            }
        )
    summary["development_images"] = len(development_images)
    summary["development_groups"] = len(groups)
    summary["test_images"] = len(test_images)
    summary["test_identity_overlap"] = overlap
    (output / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
