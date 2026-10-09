"""Merge reviewed OOF corrections into a later train/validation repartition without changing its split."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.audit_annotation_provenance import equivalent_delta, label_index, labels_equal

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    """Parse correction-merge arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--oof-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    """Hard-link an immutable source file, falling back to copying across volumes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def materialize_base_dataset(source: Path, output: Path) -> None:
    """Materialize only base train/validation images and labels, excluding derived context crops."""
    if output.exists():
        raise FileExistsError(output)
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            directory = source / kind / split
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix != ".cache":
                    link_or_copy(path, output / kind / split / path.name)


def merge_label(original: Path, reviewed: Path, current: Path, destination: Path) -> str:
    """Apply one reviewed semantic delta or preserve an equivalent current correction."""
    if labels_equal(original, reviewed):
        return "reviewed_no_semantic_change"
    if labels_equal(current, reviewed):
        return "already_exact"
    if equivalent_delta(original, reviewed, current):
        return "already_equivalent"
    if not labels_equal(current, original):
        raise RuntimeError(f"Conflicting current label: {current}")
    destination.unlink()
    shutil.copy2(reviewed, destination)
    return "applied_reviewed_label"


def validate_dataset(root: Path) -> dict:
    """Validate image-label pairing and basic YOLO label syntax."""
    result = {}
    for split in ("train", "val"):
        images = sorted(
            path for path in (root / "images" / split).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        labels = sorted((root / "labels" / split).glob("*.txt"))
        if {path.stem for path in images} != {path.stem for path in labels}:
            raise RuntimeError(f"Image-label pairing differs in {split}")
        instances = 0
        for path in labels:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                values = list(map(float, line.split()))
                if len(values) != 5 or int(values[0]) not in range(9):
                    raise ValueError(f"Invalid label row {path}:{line_number}")
                if not all(0 <= value <= 1 for value in values[1:]) or values[3] <= 0 or values[4] <= 0:
                    raise ValueError(f"Invalid normalized box {path}:{line_number}")
                instances += 1
        result[split] = {"images": len(images), "labels": len(labels), "instances": instances}
    return result


def main():
    """Merge reviewed labels, validate the isolated output, and write its immutable manifest."""
    args = parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    report = json.loads(args.oof_report.resolve().read_text(encoding="utf-8"))
    original_root, reviewed_root = Path(report["source"]).resolve(), Path(report["output"]).resolve()
    current_labels = label_index(source)
    protected_hashes = {path: sha256(path) for path in current_labels.values()}
    materialize_base_dataset(source, output)
    output_labels = label_index(output)

    records = []
    statuses = Counter()
    for reviewed_name, changes in sorted(report["changes"].items()):
        reviewed = Path(reviewed_name).resolve()
        relative = reviewed.relative_to(reviewed_root)
        original = original_root / relative
        current = current_labels.get(reviewed.name)
        destination = output_labels.get(reviewed.name)
        if current is None or destination is None:
            raise FileNotFoundError(f"Unable to map reviewed label {reviewed.name} into current split")
        status = merge_label(original, reviewed, current, destination)
        statuses[status] += 1
        records.append(
            {
                "label": reviewed.name,
                "current_split": current.parent.name,
                "status": status,
                "changes": changes,
                "output_sha256": sha256(destination),
            }
        )

    changed_sources = [str(path) for path, digest in protected_hashes.items() if sha256(path) != digest]
    if changed_sources:
        raise RuntimeError(f"Source labels changed: {changed_sources[:5]}")
    manifest = {
        "source": str(source),
        "oof_report": str(args.oof_report.resolve()),
        "output": str(output),
        "copy_mode": "hard_links_with_independent_reviewed_labels",
        "derived_context_crops_included": False,
        "status_counts": dict(sorted(statuses.items())),
        "source_labels_unchanged": True,
        "split_counts": validate_dataset(output),
        "records": records,
    }
    manifest_path = output / "merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"manifest": str(manifest_path), **manifest["status_counts"], **manifest["split_counts"]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
