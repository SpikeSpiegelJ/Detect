"""Remove one YOLO class from a split dataset while preserving its split allocation."""

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")


def parse_args():
    """Parse class-removal options."""
    parser = ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT / "repartition_v3")
    parser.add_argument("--output", type=Path, default=ROOT / "repartition_v3_without_feeder_damage")
    parser.add_argument("--class-id", type=int, default=0, help="Original class ID to remove.")
    parser.add_argument("--class-count", type=int, default=10, help="Number of source classes before removal.")
    parser.add_argument("--apply", action="store_true", help="Write the transformed dataset after a successful audit.")
    return parser.parse_args()


def image_paths(directory: Path) -> dict[str, Path]:
    """Return supported images keyed by filename stem."""
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def label_paths(directory: Path) -> dict[str, Path]:
    """Return YOLO labels keyed by filename stem."""
    return {path.stem: path for path in sorted(directory.glob("*.txt"))}


def transform_label(label_path: Path, removed_class: int, class_count: int) -> tuple[list[str], Counter, int]:
    """Remove the selected class and shift greater class IDs down by one."""
    transformed = []
    counts = Counter()
    removed = 0
    for number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{number} must contain 5 YOLO values")
        try:
            class_id = int(values[0])
            coordinates = [float(value) for value in values[1:]]
        except ValueError as error:
            raise ValueError(f"{label_path}:{number} contains non-numeric values") from error
        if not 0 <= class_id < class_count:
            raise ValueError(f"{label_path}:{number} class {class_id} is outside 0-{class_count - 1}")
        if class_id == removed_class:
            removed += 1
            continue
        new_class = class_id - 1 if class_id > removed_class else class_id
        transformed.append(f"{new_class} {' '.join(values[1:])}\n")
        counts[new_class] += 1
        if not all(0 <= coordinate <= 1 for coordinate in coordinates):
            raise ValueError(f"{label_path}:{number} has coordinates outside [0, 1]")
    return transformed, counts, removed


def link_or_copy(source: Path, destination: Path):
    """Hard-link images where possible, falling back to a metadata-preserving copy."""
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def audit(source_root: Path, removed_class: int, class_count: int):
    """Validate paired files and calculate transformed class counts without writing data."""
    summary = {"splits": {}, "removed_instances": 0, "class_instances": Counter()}
    for split in SPLITS:
        image_dir = source_root / "images" / split
        label_dir = source_root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing image or label split: {split}")
        images = image_paths(image_dir)
        labels = label_paths(label_dir)
        if images.keys() != labels.keys():
            missing_labels = sorted(images.keys() - labels.keys())
            missing_images = sorted(labels.keys() - images.keys())
            raise ValueError(f"{split} image/label mismatch: missing_labels={missing_labels[:3]}, missing_images={missing_images[:3]}")
        split_counts = Counter()
        split_removed = 0
        for stem in images:
            _, counts, removed = transform_label(labels[stem], removed_class, class_count)
            split_counts.update(counts)
            split_removed += removed
        summary["splits"][split] = {"images": len(images), "instances": split_counts, "removed_instances": split_removed}
        summary["class_instances"].update(split_counts)
        summary["removed_instances"] += split_removed
    return summary


def write_dataset(source_root: Path, output: Path, removed_class: int, class_count: int, summary):
    """Write the transformed dataset atomically after validation succeeds."""
    staging = output.parent / f".{output.name}.building"
    if output.exists() or staging.exists():
        raise FileExistsError(f"Output already exists: {output}")
    manifest = {
        "source": str(source_root),
        "removed_class": removed_class,
        "class_count": class_count - 1,
        "removed_instances": summary["removed_instances"],
        "class_instances": dict(sorted(summary["class_instances"].items())),
    }
    for split in SPLITS:
        image_output = staging / "images" / split
        label_output = staging / "labels" / split
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        for stem, image_path in image_paths(source_root / "images" / split).items():
            transformed, _, _ = transform_label(source_root / "labels" / split / f"{stem}.txt", removed_class, class_count)
            link_or_copy(image_path, image_output / image_path.name)
            (label_output / f"{stem}.txt").write_text("".join(transformed), encoding="utf-8")
    (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    staging.rename(output)


def main():
    """Audit a class removal and optionally write the transformed data."""
    args = parse_args()
    if not 0 <= args.class_id < args.class_count:
        raise ValueError("--class-id must be in the source class range.")
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    summary = audit(source_root, args.class_id, args.class_count)
    print(f"Source dataset: {source_root}")
    print(f"Removed class {args.class_id}: {summary['removed_instances']} instances")
    for split, details in summary["splits"].items():
        print(
            f"{split}: images={details['images']}, removed={details['removed_instances']}, "
            f"class_instances={dict(sorted(details['instances'].items()))}"
        )
    if not args.apply:
        print("Audit only. Re-run with --apply to write the transformed dataset.")
        return
    write_dataset(source_root, output, args.class_id, args.class_count, summary)
    print(f"Created {output}")


if __name__ == "__main__":
    main()
