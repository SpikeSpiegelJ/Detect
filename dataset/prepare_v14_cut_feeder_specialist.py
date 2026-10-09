# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build a two-class cut and Feeder_antenna specialist dataset without test leakage."""

from __future__ import annotations

import json
import os
import random
import shutil
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "repartition_v13_cut_feeder_hard_corrected"
DEFAULT_OUTPUT = ROOT / "repartition_v14_cut_feeder_specialist"
SOURCE_TEST = ROOT / "repartition_v9_trainval_9c" / "images" / "test"
CLASS_MAP = {5: 0, 6: 1}
CONFUSERS = {1, 3, 4}
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--negative-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=260927)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def paired_label(image: Path) -> Path:
    parts = list(image.parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(path: Path):
    return [
        (int(float(cls)), float(x), float(y), float(width), float(height))
        for cls, x, y, width, height in (line.split() for line in path.read_text(encoding="utf-8").splitlines())
    ]


def remap(labels):
    return [(CLASS_MAP[cls], x, y, width, height) for cls, x, y, width, height in labels if cls in CLASS_MAP]


def write_labels(path: Path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{cls} {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n" for cls, x, y, width, height in labels),
        encoding="utf-8",
    )


def link(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def images(directory: Path):
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in SUFFIXES)


def materialize(records, output: Path, split: str):
    counts = Counter()
    for image, labels in records:
        link(image, output / "images" / split / image.name)
        mapped = remap(labels)
        write_labels(output / "labels" / split / f"{image.stem}.txt", mapped)
        counts.update(row[0] for row in mapped)
    return {"images": len(records), "instances": [counts[0], counts[1]]}


def main():
    args = parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        if not args.replace:
            raise FileExistsError(f"{output} exists; pass --replace")
        shutil.rmtree(output)
    rng = random.Random(args.seed)

    original = [(image, read_labels(paired_label(image))) for image in images(source / "images" / "train")]
    positives = [record for record in original if remap(record[1])]
    negatives = [record for record in original if not remap(record[1]) and {row[0] for row in record[1]} & CONFUSERS]
    rng.shuffle(negatives)
    selected_negatives = negatives[: round(len(positives) * args.negative_ratio)]
    train = positives + selected_negatives
    rng.shuffle(train)

    crop_records = []
    for split in ("train_cut_context", "train_feeder_antenna_context"):
        crop_records.extend((image, read_labels(paired_label(image))) for image in images(source / "images" / split))
    val = [(image, read_labels(paired_label(image))) for image in images(source / "images" / "val")]
    test = [(image, read_labels(paired_label(image))) for image in images(SOURCE_TEST)]

    summary = {
        "source": str(source),
        "seed": args.seed,
        "negative_ratio": args.negative_ratio,
        "train": materialize(train, output, "train"),
        "crops": materialize(crop_records, output, "train_context"),
        "val": materialize(val, output, "val"),
        "test": materialize(test, output, "test"),
        "positive_full_images": len(positives),
        "confuser_negative_images": len(selected_negatives),
        "negative_selection": "fixed-seed random sample without specialist targets and with classes 1, 3, or 4",
    }
    descriptor = {
        "path": output.as_posix(),
        "train": ["images/train", "images/train_context"],
        "val": "images/val",
        "test": "images/test",
        "nc": 2,
        "names": {0: "cut", 1: "Feeder_antenna"},
    }
    yaml_path = ROOT / "data_repartition_v14_cut_feeder_specialist.yaml"
    yaml_path.write_text(yaml.safe_dump(descriptor, sort_keys=False, allow_unicode=True), encoding="utf-8")
    summary["yaml"] = str(yaml_path.resolve())
    (output / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
