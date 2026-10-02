"""Audit and safely repartition a base YOLO dataset with an external YOLO folder."""

from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from hashlib import blake2b
import json
from pathlib import Path
import random
import shutil

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CLASS_COUNT = 9
SPLITS = ("train", "val", "test")
BOUND_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Sample:
    """One valid YOLO image and label pair."""

    image: Path
    label: Path
    origin: str
    classes: tuple[int, ...]


def parse_args():
    """Parse audit and repartition options."""
    parser = ArgumentParser(description="Audit an external YOLO dataset and create a clean merged split.")
    parser.add_argument("--base-root", type=Path, default=ROOT / "repartition_v3", help="Existing split dataset root.")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "dataset_3", help="External dataset folder.")
    parser.add_argument("--output", default="repartition_v3_merged", help="New output directory under dataset/.")
    parser.add_argument("--seed", type=int, default=260830, help="Deterministic split seed.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--preserve-base-splits",
        action="store_true",
        help="Keep the current base train/val/test assignments and add unique external samples to train only.",
    )
    parser.add_argument("--apply", action="store_true", help="Create the merged repartition after the audit succeeds.")
    return parser.parse_args()


def image_map(directory: Path) -> tuple[dict[str, Path], list[str]]:
    """Return image paths keyed by stem and report ambiguous duplicate stems."""
    images = {}
    suffix_priority = {".jpg": 0, ".jpeg": 1, ".png": 2}
    if not directory.is_dir():
        return images, [f"Missing image directory: {directory}"]
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem not in images or suffix_priority[path.suffix.lower()] < suffix_priority[images[path.stem].suffix.lower()]:
            images[path.stem] = path
    return images, []


def label_map(directory: Path) -> tuple[dict[str, Path], list[str]]:
    """Return label paths keyed by stem and report invalid directory state."""
    if not directory.is_dir():
        return {}, [f"Missing label directory: {directory}"]
    return {path.stem: path for path in sorted(directory.glob("*.txt"))}, []


def parse_label(path: Path) -> tuple[tuple[int, ...], list[str]]:
    """Validate YOLO detection labels and return their class IDs."""
    classes = []
    errors = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    for number, line in enumerate(lines, 1):
        values = line.split()
        if len(values) != 5:
            errors.append(f"{path.name}:{number} must contain 5 YOLO values")
            continue
        try:
            cls = int(values[0])
            x, y, width, height = map(float, values[1:])
        except ValueError:
            errors.append(f"{path.name}:{number} contains non-numeric values")
            continue
        if not 0 <= cls < CLASS_COUNT:
            errors.append(f"{path.name}:{number} class {cls} is outside 0-{CLASS_COUNT - 1}")
        if not 0 < width <= 1 or not 0 < height <= 1 or not 0 <= x <= 1 or not 0 <= y <= 1:
            errors.append(f"{path.name}:{number} has invalid normalized coordinates")
        if (
            x - width / 2 < -BOUND_TOLERANCE
            or x + width / 2 > 1 + BOUND_TOLERANCE
            or y - height / 2 < -BOUND_TOLERANCE
            or y + height / 2 > 1 + BOUND_TOLERANCE
        ):
            errors.append(f"{path.name}:{number} box exceeds image bounds")
        classes.append(cls)
    return tuple(classes), errors


def readable_image(path: Path) -> bool:
    """Read an image without Unicode path limitations in OpenCV on Windows."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return image is not None and image.size > 0


def audit_pair_directories(image_dir: Path, label_dir: Path, origin: str):
    """Audit a single image/label directory pair."""
    images, issues = image_map(image_dir)
    labels, label_issues = label_map(label_dir)
    issues.extend(label_issues)
    only_images = sorted(set(images) - set(labels))
    only_labels = sorted(set(labels) - set(images))
    samples = []
    for stem in sorted(set(images) & set(labels)):
        classes, label_errors = parse_label(labels[stem])
        issues.extend(label_errors)
        if label_errors:
            continue
        if not readable_image(images[stem]):
            issues.append(f"Unreadable image: {images[stem]}")
            continue
        samples.append(Sample(images[stem], labels[stem], origin, classes))
    return samples, issues, only_images, only_labels


def external_label_dir(source: Path) -> Path:
    """Return the conventional label directory used by an external YOLO folder."""
    return source / "labels" if (source / "labels").is_dir() else source / "label"


def audit_sources(base_root: Path, source: Path):
    """Audit current base splits and the external dataset before any writes."""
    existing = []
    issues = []
    unmatched_images = []
    unmatched_labels = []
    for split in SPLITS:
        samples, split_issues, only_images, only_labels = audit_pair_directories(
            base_root / "images" / split, base_root / "labels" / split, f"base_{split}"
        )
        existing.extend(samples)
        issues.extend(split_issues)
        unmatched_images.extend((f"current_{split}", stem) for stem in only_images)
        unmatched_labels.extend((f"current_{split}", stem) for stem in only_labels)

    external, external_issues, only_images, only_labels = audit_pair_directories(
        source / "images", external_label_dir(source), source.name
    )
    issues.extend(external_issues)
    unmatched_images.extend((source.name, stem) for stem in only_images)
    unmatched_labels.extend((source.name, stem) for stem in only_labels)
    return existing, external, issues, unmatched_images, unmatched_labels


def image_hash(path: Path) -> str:
    """Return a compact content hash for exact-image deduplication."""
    digest = blake2b(digest_size=16)
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def unique_samples(samples: list[Sample]):
    """Keep the first sample for each exact image and report discarded duplicates."""
    by_hash = {}
    duplicates = []
    for sample in samples:
        digest = image_hash(sample.image)
        if digest in by_hash:
            duplicates.append((sample, by_hash[digest]))
        else:
            by_hash[digest] = sample
    return list(by_hash.values()), duplicates


def class_counts(samples: list[Sample]) -> Counter:
    """Count annotated instances by class."""
    return Counter(cls for sample in samples for cls in sample.classes)


def print_audit(base_root, source, existing, external, merged, issues, unmatched_images, unmatched_labels, duplicates):
    """Print a concise, actionable dataset-quality report."""
    print(f"Base dataset ({base_root.name}): {len(existing)} valid image/label pairs")
    print(f"External dataset ({source.name}): {len(external)} valid image/label pairs")
    print(f"Merged unique candidates: {len(merged)}")
    print(f"{source.name} unmatched images (excluded): {sum(origin == source.name for origin, _ in unmatched_images)}")
    print(f"{source.name} unmatched labels (excluded): {sum(origin == source.name for origin, _ in unmatched_labels)}")
    print(f"Exact duplicate images discarded: {len(duplicates)}")
    print(f"Validation issues: {len(issues)}")
    print(f"{source.name} class instances:", dict(sorted(class_counts(external).items())))
    print("merged class instances:", dict(sorted(class_counts(merged).items())))
    for issue in issues[:10]:
        print(f"ISSUE: {issue}")
    for origin, stem in unmatched_images[:10]:
        print(f"UNMATCHED IMAGE: {origin}/{stem}")
    for origin, stem in unmatched_labels[:10]:
        print(f"UNMATCHED LABEL: {origin}/{stem}")


def split_sizes(total: int, ratios: dict[str, float]) -> dict[str, int]:
    """Allocate exact split sizes according to ratios."""
    raw = {split: total * ratio for split, ratio in ratios.items()}
    sizes = {split: int(value) for split, value in raw.items()}
    for split in sorted(SPLITS, key=lambda item: raw[item] - sizes[item], reverse=True)[: total - sum(sizes.values())]:
        sizes[split] += 1
    return sizes


def stratified_split(samples: list[Sample], ratios: dict[str, float], seed: int):
    """Split multi-label samples while keeping rare classes represented in each split."""
    rng = random.Random(seed)
    totals = class_counts(samples)
    capacities = split_sizes(len(samples), ratios)
    targets = {split: {cls: totals[cls] * ratios[split] for cls in totals} for split in SPLITS}
    allocated = {split: [] for split in SPLITS}
    counts = {split: Counter() for split in SPLITS}
    shuffled = samples[:]
    rng.shuffle(shuffled)
    ordered = sorted(
        shuffled,
        key=lambda sample: (min((totals[cls] for cls in set(sample.classes)), default=10**9), -len(sample.classes)),
    )
    for sample in ordered:
        eligible = [split for split in SPLITS if len(allocated[split]) < capacities[split]]
        best = max(
            eligible,
            key=lambda split: (
                sum(max(targets[split][cls] - counts[split][cls], 0) / max(targets[split][cls], 1) for cls in set(sample.classes)),
                (capacities[split] - len(allocated[split])) / capacities[split],
            ),
        )
        allocated[best].append(sample)
        counts[best].update(sample.classes)
    return allocated


def preserved_base_split(existing: list[Sample], merged: list[Sample], source_name: str):
    """Keep base split membership unchanged and append unique external samples to training."""
    allocated = {split: [sample for sample in existing if sample.origin == f"base_{split}"] for split in SPLITS}
    allocated["train"].extend(sample for sample in merged if sample.origin == source_name)
    return allocated


def link_or_copy(source: Path, destination: Path):
    """Hard-link images where possible, falling back to a metadata-preserving copy."""
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def output_stem(sample: Sample) -> str:
    """Create collision-free names that retain a source trace."""
    return f"{sample.origin}__{sample.image.stem}"


def write_split(output: Path, allocated: dict[str, list[Sample]], seed: int):
    """Write a new split and a manifest without altering source datasets."""
    staging = output.parent / f".{output.name}.building"
    if output.exists() or staging.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --output name.")
    manifest = {"seed": seed, "splits": {}}
    for split, samples in allocated.items():
        image_dir = staging / "images" / split
        label_dir = staging / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        manifest["splits"][split] = []
        for sample in samples:
            stem = output_stem(sample)
            image_destination = image_dir / f"{stem}{sample.image.suffix.lower()}"
            label_destination = label_dir / f"{stem}.txt"
            link_or_copy(sample.image, image_destination)
            shutil.copy2(sample.label, label_destination)
            manifest["splits"][split].append(
                {"image": str(sample.image), "label": str(sample.label), "origin": sample.origin, "name": image_destination.name}
            )
    (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    staging.rename(output)


def main():
    """Audit external data and optionally create a safe merged repartition."""
    args = parse_args()
    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    if any(ratio <= 0 for ratio in ratios.values()) or not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError("Split ratios must be positive and sum to 1.0.")
    base_root = args.base_root.resolve()
    source = args.source.resolve()
    existing, external, issues, unmatched_images, unmatched_labels = audit_sources(base_root, source)
    merged, duplicates = unique_samples(existing + external)
    print_audit(base_root, source, existing, external, merged, issues, unmatched_images, unmatched_labels, duplicates)
    if not args.apply:
        print("Audit only. Re-run with --apply after reviewing this report.")
        return
    if issues:
        raise ValueError("Dataset contains validation issues. Fix them before applying a repartition.")
    if not external:
        raise ValueError("No valid external image/label pairs are available for merging.")
    allocated = (
        preserved_base_split(existing, merged, source.name)
        if args.preserve_base_splits
        else stratified_split(merged, ratios, args.seed)
    )
    output = ROOT / args.output
    write_split(output, allocated, args.seed)
    print(f"Created {output}")
    for split in SPLITS:
        print(f"{split}: images={len(allocated[split])}, class_instances={dict(sorted(class_counts(allocated[split]).items()))}")


if __name__ == "__main__":
    main()
