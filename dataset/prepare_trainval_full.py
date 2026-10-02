"""Repartition the existing train and validation samples while preserving the independent test split."""

from pathlib import Path

import yaml

from split_data import SPLITS, audit_pair_directories, class_counts, stratified_split, write_split


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "repartition_v8_dataset9_9c"
OUTPUT = ROOT / "repartition_v9_trainval_9c"
DESCRIPTOR = ROOT / "data_repartition_v9_trainval_9c.yaml"
SEED = 260924


def load_split(split):
    """Audit and return one source split."""
    samples, issues, only_images, only_labels = audit_pair_directories(
        SOURCE / "images" / split, SOURCE / "labels" / split, f"v8_{split}"
    )
    if issues or only_images or only_labels:
        raise ValueError(
            f"Invalid {split} split: {len(issues)} issues, {len(only_images)} unmatched images, "
            f"{len(only_labels)} unmatched labels"
        )
    return samples


def main():
    """Combine train+val, make a new 90/10 development split, and retain test unchanged."""
    development = load_split("train") + load_split("val")
    allocated = stratified_split(development, {"train": 0.9, "val": 0.1, "test": 0.0}, SEED)
    allocated["test"] = load_split("test")
    write_split(OUTPUT, allocated, SEED)

    source_descriptor = yaml.safe_load((ROOT / "data_repartition_v8_dataset9_9c.yaml").read_text(encoding="utf-8"))
    descriptor = {
        "path": str(OUTPUT).replace("\\", "/"),
        **{split: f"images/{split}" for split in SPLITS},
        "nc": source_descriptor["nc"],
        "names": source_descriptor["names"],
    }
    DESCRIPTOR.write_text(yaml.safe_dump(descriptor, sort_keys=False, allow_unicode=True), encoding="utf-8")
    for split in SPLITS:
        print(f"{split}: {len(allocated[split])} images, classes={dict(sorted(class_counts(allocated[split]).items()))}")
    print(f"Dataset: {OUTPUT}")
    print(f"Descriptor: {DESCRIPTOR}")


if __name__ == "__main__":
    main()
