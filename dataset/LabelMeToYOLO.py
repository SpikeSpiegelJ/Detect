# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert a directory of LabelMe detection annotations to validated YOLO labels."""

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import shutil

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "dataset_10" / "images"
DEFAULT_OUTPUT = ROOT / "dataset_10" / "labels"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASS_MAPPING = {
    "coupler": 0,
    "antenna_s": 1,
    "RRU": 2,
    "Feeder_RRU": 3,
    "antenna_b": 4,
    "cut": 5,
    "Feeder_antenna": 6,
    "POWER": 7,
    "BOX": 8,
}


def parse_args():
    """Parse conversion paths."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE, help="Directory containing images and JSON files."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for converted YOLO labels.")
    return parser.parse_args()


def find_image(json_path, image_path):
    """Resolve the image recorded by LabelMe, with a same-stem fallback."""
    recorded = json_path.parent / image_path
    if recorded.is_file():
        return recorded
    matches = [json_path.with_suffix(suffix) for suffix in IMAGE_SUFFIXES if json_path.with_suffix(suffix).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one image for {json_path.name}, found {len(matches)}")
    return matches[0]


def shape_box(shape, json_path, index, width, height):
    """Validate one LabelMe shape and return its clipped xyxy box."""
    shape_type = shape.get("shape_type")
    points = shape.get("points", [])
    minimum_points = 2 if shape_type == "rectangle" else 3
    if shape_type not in {"rectangle", "polygon"} or len(points) < minimum_points:
        raise ValueError(f"{json_path.name}: shape {index} must be a rectangle or polygon")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x1, y1 = max(0.0, min(xs)), max(0.0, min(ys))
    x2, y2 = min(float(width), max(xs)), min(float(height), max(ys))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{json_path.name}: shape {index} has an empty box")
    return x1, y1, x2, y2


def convert_labelme_to_yolo(json_path, class_mapping=CLASS_MAPPING):
    """Convert one LabelMe JSON document to validated YOLO rows."""
    data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    image_path = find_image(json_path, data.get("imagePath", ""))
    with Image.open(image_path) as image:
        width, height = image.size
    declared = data.get("imageWidth"), data.get("imageHeight")
    if declared != (width, height):
        raise ValueError(f"{json_path.name}: JSON dimensions {declared} do not match image {(width, height)}")

    rows = []
    for index, shape in enumerate(data.get("shapes", []), 1):
        label = shape.get("label")
        if label not in class_mapping:
            raise ValueError(f"{json_path.name}: shape {index} has unknown class {label!r}")
        x1, y1, x2, y2 = shape_box(shape, json_path, index, width, height)
        rows.append(
            (
                class_mapping[label],
                (x1 + x2) / (2 * width),
                (y1 + y2) / (2 * height),
                (x2 - x1) / width,
                (y2 - y1) / height,
            )
        )
    return image_path, rows


def serialize(rows):
    """Serialize normalized detection rows."""
    return "".join(f"{class_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n" for class_id, x, y, width, height in rows)


def batch_convert_labelme_to_yolo(source, output, class_mapping=CLASS_MAPPING):
    """Validate all LabelMe files, then atomically create their YOLO label directory."""
    source, output = source.resolve(), output.resolve()
    json_paths = sorted(source.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No LabelMe JSON files found in {source}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")

    converted = []
    class_counts = Counter()
    image_stems = set()
    for json_path in json_paths:
        image_path, rows = convert_labelme_to_yolo(json_path, class_mapping)
        if image_path.stem in image_stems:
            raise ValueError(f"Duplicate image stem: {image_path.stem}")
        image_stems.add(image_path.stem)
        converted.append((json_path.stem, rows))
        class_counts.update(row[0] for row in rows)

    staging = output.parent / f".{output.name}.building"
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        for stem, rows in converted:
            (staging / f"{stem}.txt").write_text(serialize(rows), encoding="utf-8")
        if output.exists():
            output.rmdir()
        staging.rename(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(f"Converted {len(converted)} LabelMe files to {output}")
    print(f"Class instances: {dict(sorted(class_counts.items()))}")


def main():
    """Run the LabelMe-to-YOLO conversion."""
    args = parse_args()
    batch_convert_labelme_to_yolo(args.source, args.output)


if __name__ == "__main__":
    main()
