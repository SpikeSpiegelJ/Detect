# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Create reversible data manifests for specialist context and confuser-negative ablations."""

from __future__ import annotations

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data_repartition_v14_cut_feeder_specialist.yaml"
DEFAULT_OUTPUT = ROOT / "specialist_ablation_manifests"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    """Parse specialist ablation manifest arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def paired_label(image: Path) -> Path:
    """Return the YOLO label paired with an image path."""
    parts = list(image.parts)
    index = parts.index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def label_classes(path: Path) -> list[int]:
    """Read and validate the two-class specialist labels."""
    classes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{path}:{line_number} expected 5 values")
        class_id = int(float(values[0]))
        if class_id not in (0, 1):
            raise ValueError(f"{path}:{line_number} has invalid specialist class {class_id}")
        classes.append(class_id)
    return classes


def images(directory: Path) -> list[Path]:
    """Return the image files directly inside one materialized split."""
    return sorted(path.resolve() for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a generated manifest or descriptor."""
    return sha256(path.read_bytes()).hexdigest()


def main():
    """Write four data views that independently toggle context and confuser-negative samples."""
    args = parse_args()
    data_file, output = args.data.resolve(), args.output.resolve()
    descriptor = yaml.safe_load(data_file.read_text(encoding="utf-8-sig"))
    if descriptor["names"] != {0: "cut", 1: "Feeder_antenna"}:
        raise ValueError("Expected the cut and Feeder_antenna specialist class mapping")
    root = Path(descriptor["path"]).resolve()
    train_images = images(root / "images" / "train")
    context_images = images(root / "images" / "train_context")
    for image in [*train_images, *context_images]:
        if not paired_label(image).is_file():
            raise FileNotFoundError(paired_label(image))
    positive = [image for image in train_images if label_classes(paired_label(image))]
    confuser_negative = [image for image in train_images if not label_classes(paired_label(image))]
    if any(not label_classes(paired_label(image)) for image in context_images):
        raise ValueError("Every context crop must contain a specialist target")

    variants = {
        "full_positive": positive,
        "full_positive_confuser": [*positive, *confuser_negative],
        "full_positive_context": [*positive, *context_images],
        "full_context_confuser": [*positive, *context_images, *confuser_negative],
    }
    expected = [output / f"{name}.{suffix}" for name in variants for suffix in ("txt", "yaml")]
    expected.append(output / "audit.json")
    if not args.replace and any(path.exists() for path in expected):
        raise FileExistsError("Ablation outputs already exist; pass --replace to regenerate them")
    output.mkdir(parents=True, exist_ok=True)

    generated = {}
    for name, selected in variants.items():
        manifest = output / f"{name}.txt"
        manifest.write_text("\n".join(str(path) for path in selected) + "\n", encoding="utf-8")
        variant_data = {
            "path": root.as_posix(),
            "train": manifest.as_posix(),
            "val": (root / "images" / "val").as_posix(),
            "test": (root / "images" / "test").as_posix(),
            "nc": 2,
            "names": descriptor["names"],
        }
        yaml_path = output / f"{name}.yaml"
        yaml_path.write_text(yaml.safe_dump(variant_data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        generated[name] = {
            "images": len(selected),
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "data": str(yaml_path),
            "data_sha256": sha256_file(yaml_path),
        }
    audit = {
        "source_data": str(data_file),
        "source_root": str(root),
        "groups": {
            "positive_full_images": len(positive),
            "context_images": len(context_images),
            "confuser_negative_images": len(confuser_negative),
        },
        "negative_definition": "empty specialist labels in the materialized train split",
        "variants": generated,
        "validation_and_test_unchanged": True,
    }
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
