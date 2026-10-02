# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Prepare selected YOLO samples for correction in LabelMe without changing the training dataset."""

import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data_repartition_v8_dataset9_9c.yaml"
DEFAULT_REVIEW = ROOT.parent / "runs" / "audit" / "yolo26s_v8_train_weak_class_errors" / "review.csv"
DEFAULT_OUTPUT = ROOT / "relabel_v8_train_annotation_errors"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset YAML containing path and names.")
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW, help="Reviewed error CSV.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for the LabelMe correction package."
    )
    parser.add_argument(
        "--images",
        nargs="+",
        help="Dataset image names to prepare. By default, use unique annotation_error rows from review.csv.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing generated correction package.")
    return parser.parse_args()


def load_dataset(data_path):
    """Load the dataset directory and class names from a dataset YAML."""
    with data_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    dataset_dir = Path(config["path"])
    if not dataset_dir.is_absolute():
        dataset_dir = data_path.parent / dataset_dir
    raw_names = config["names"]
    names = enumerate(raw_names) if isinstance(raw_names, list) else raw_names.items()
    return dataset_dir.resolve(), {int(index): name for index, name in names}


def load_review_images(review_path):
    """Return unique source image names confirmed as annotation errors."""
    with review_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"source_image", "decision"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{review_path} must contain source_image and decision columns")
    images = sorted(
        {Path(row["source_image"]).name for row in rows if row["decision"].strip().lower() == "annotation_error"}
    )
    if not images:
        raise ValueError(f"No annotation_error rows found in {review_path}")
    return images


def yolo_to_labelme(label_path, width, height, names):
    """Convert normalized YOLO boxes to LabelMe rectangle shapes."""
    shapes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_number} must contain class_id x_center y_center width height")
        class_id = int(values[0])
        if class_id not in names:
            raise ValueError(f"{label_path}:{line_number} contains unknown class ID {class_id}")
        x_center, y_center, box_width, box_height = map(float, values[1:])
        x1 = max(0.0, (x_center - box_width / 2) * width)
        y1 = max(0.0, (y_center - box_height / 2) * height)
        x2 = min(float(width), (x_center + box_width / 2) * width)
        y2 = min(float(height), (y_center + box_height / 2) * height)
        shapes.append(
            {
                "label": names[class_id],
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None,
            }
        )
    return shapes


def find_manifest_entries(dataset_dir, image_names):
    """Return dataset split and manifest record for each requested image."""
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {
        record["name"]: (split, record)
        for split in ("train", "val", "test")
        for record in manifest["splits"][split]
    }
    missing = sorted(set(image_names) - records.keys())
    if missing:
        raise FileNotFoundError(f"Images are absent from the dataset manifest: {', '.join(missing)}")
    return records


def prepare_sample(dataset_dir, image_name, output, names, split, manifest_record):
    """Copy one image and create its editable LabelMe JSON and backup label."""
    image_path = dataset_dir / "images" / split / image_name
    label_path = dataset_dir / "labels" / split / f"{Path(image_name).stem}.txt"
    if not image_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"Missing dataset pair: {image_path} / {label_path}")

    output_image = output / image_name
    shutil.copy2(image_path, output_image)
    shutil.copy2(label_path, output / "original_labels" / label_path.name)
    with Image.open(image_path) as image:
        width, height = image.size

    labelme = {
        "version": "5.1.0",
        "flags": {},
        "shapes": yolo_to_labelme(label_path, width, height, names),
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    json_path = output / f"{Path(image_name).stem}.json"
    json_path.write_text(json.dumps(labelme, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "image": str(output_image.resolve()),
        "labelme_json": str(json_path.resolve()),
        "dataset_image": str(image_path.resolve()),
        "dataset_label": str(label_path.resolve()),
        "split": split,
        "source_image": manifest_record["image"],
        "source_label": manifest_record["label"],
    }


def main():
    """Create a self-contained LabelMe correction package for the selected images."""
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {output}. Use --force to replace it.")
        shutil.rmtree(output)
    (output / "original_labels").mkdir(parents=True)

    data_path = args.data.resolve()
    dataset_dir, names = load_dataset(data_path)
    image_names = sorted(args.images) if args.images else load_review_images(args.review.resolve())
    records = find_manifest_entries(dataset_dir, image_names)
    mapping = [
        prepare_sample(dataset_dir, name, output, names, *records[name])
        for name in image_names
    ]
    (output / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = f"""LabelMe 重标注材料

图片数：{len(mapping)}
数据集：{dataset_dir}
类别：{', '.join(f'{index}={name}' for index, name in names.items())}

操作步骤：
1. 使用 LabelMe 打开本目录。
2. 删除错误框，修改错误类别，重新绘制不准确或缺失的目标。
3. 只使用 rectangle，并严格使用上面列出的类别名称。
4. 保存到图片旁边已有的同名 JSON 文件。
5. 不要修改 original_labels，里面是修改前的 YOLO 标签备份。

当前数据集尚未修改。mapping.json 保存了每张图片、标签、分区及原始来源的路径，供完成标注后校验并回写。
"""
    (output / "README.txt").write_text(readme, encoding="utf-8")
    print(f"Prepared {len(mapping)} images for correction: {output}")
    print("Open the directory in LabelMe, edit every JSON annotation, and save it in place.")


if __name__ == "__main__":
    main()
