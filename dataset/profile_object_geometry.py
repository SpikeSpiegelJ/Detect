# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Profile object scale and elongation at the detector input resolution."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from audit_split_integrity import paired_label, split_images

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data_repartition_v9_trainval_9c.yaml"


def parse_args():
    """Parse geometry-profile arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def label_geometry(label: str, image_shape: tuple[int, int], imgsz: int) -> tuple[int, dict]:
    """Convert one normalized YOLO label to input-scale geometry."""
    values = label.split()
    if len(values) != 5:
        raise ValueError(f"Expected 5 label values, received {len(values)}")
    class_id = int(float(values[0]))
    width, height = map(float, values[3:])
    image_height, image_width = image_shape
    scale = imgsz / max(image_height, image_width)
    width_px, height_px = width * image_width * scale, height * image_height * scale
    short_side, long_side = sorted((width_px, height_px))
    return class_id, {
        "width_px": width_px,
        "height_px": height_px,
        "short_side_px": short_side,
        "long_side_px": long_side,
        "area_px2": width_px * height_px,
        "aspect_ratio": long_side / max(short_side, 1e-9),
    }


def summarize(rows: list[dict]) -> dict:
    """Summarize one geometry group with fixed bins and quantiles."""
    quantiles = (0.10, 0.25, 0.50, 0.75, 0.90)
    summary = {"instances": len(rows)}
    for key in ("width_px", "height_px", "short_side_px", "long_side_px", "area_px2", "aspect_ratio"):
        values = np.asarray([row[key] for row in rows], dtype=float)
        summary[key] = {f"q{int(q * 100):02d}": float(np.quantile(values, q)) for q in quantiles}
    areas = np.asarray([row["area_px2"] for row in rows], dtype=float)
    ratios = np.asarray([row["aspect_ratio"] for row in rows], dtype=float)
    summary["size_bins"] = {
        "small_lt_32sq": int((areas < 32**2).sum()),
        "medium_32sq_to_96sq": int(((areas >= 32**2) & (areas < 96**2)).sum()),
        "large_ge_96sq": int((areas >= 96**2).sum()),
    }
    summary["elongation_bins"] = {
        "aspect_ge_4": int((ratios >= 4).sum()),
        "aspect_ge_8": int((ratios >= 8).sum()),
    }
    for group in ("size_bins", "elongation_bins"):
        summary[f"{group}_fraction"] = {key: value / len(rows) for key, value in summary[group].items()}
    return summary


def profile(data_file: Path, imgsz: int, splits: list[str]) -> dict:
    """Profile all labeled objects in selected splits."""
    data_file = data_file.resolve()
    descriptor = yaml.safe_load(data_file.read_text(encoding="utf-8-sig"))
    names = {int(key): value for key, value in descriptor["names"].items()}
    report = {"data": str(data_file), "imgsz": imgsz, "splits": {}}
    for split in splits:
        by_class, all_rows = defaultdict(list), []
        images = split_images(data_file, descriptor, split)
        for image in images:
            decoded = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError(f"Unreadable image: {image}")
            label_path = paired_label(image)
            for line in label_path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                class_id, row = label_geometry(line, decoded.shape[:2], imgsz)
                by_class[class_id].append(row)
                all_rows.append(row)
        report["splits"][split] = {
            "images": len(images),
            "overall": summarize(all_rows),
            "per_class": {names[class_id]: summarize(by_class[class_id]) for class_id in sorted(by_class)},
        }
    return report


def main():
    """Generate the machine-readable geometry report."""
    args = parse_args()
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")
    report = profile(args.data, args.imgsz, args.splits)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "splits": {
                    split: {
                        "images": values["images"],
                        "instances": values["overall"]["instances"],
                    }
                    for split, values in report["splits"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
