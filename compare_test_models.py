# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Compare the selected V8 model and corrected-label full-training candidate on the held-out test split."""

import csv
from pathlib import Path

import torch

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "data_repartition_v8_dataset9_9c.yaml"
OUTPUT = ROOT / "runs" / "test" / "yolo26s_v8_corrected_full_comparison"
MODELS = {
    "previous_best": ROOT / "runs" / "train" / "yolo26s_v8_dataset9_relabel_ft" / "weights" / "best.pt",
    "corrected_full": ROOT / "runs" / "train" / "yolo26s_v8_corrected_full" / "weights" / "best.pt",
}


def metric_rows(model_name, metrics):
    """Return overall and per-class detection metrics as CSV rows."""
    box = metrics.box
    rows = [
        {
            "model": model_name,
            "class_id": "all",
            "class": "all",
            "precision": box.mp,
            "recall": box.mr,
            "mAP50": box.map50,
            "mAP50-95": box.map,
        }
    ]
    for position, class_id in enumerate(box.ap_class_index):
        precision, recall, map50, map50_95 = box.class_result(position)
        rows.append(
            {
                "model": model_name,
                "class_id": int(class_id),
                "class": metrics.names[int(class_id)],
                "precision": precision,
                "recall": recall,
                "mAP50": map50,
                "mAP50-95": map50_95,
            }
        )
    return rows


def print_comparison(rows):
    """Print the candidate-minus-previous metric differences."""
    by_model = {(row["model"], row["class"]): row for row in rows}
    print("\nCorrected-label full training minus previous best:")
    print(f"{'Class':<18}{'Precision':>12}{'Recall':>12}{'mAP50':>12}{'mAP50-95':>12}")
    classes = [row["class"] for row in rows if row["model"] == "previous_best"]
    for class_name in classes:
        previous = by_model[("previous_best", class_name)]
        candidate = by_model[("corrected_full", class_name)]
        differences = [candidate[key] - previous[key] for key in ("precision", "recall", "mAP50", "mAP50-95")]
        print(f"{class_name:<18}" + "".join(f"{difference:>+12.4f}" for difference in differences))


def main():
    """Evaluate both checkpoints once with identical test settings and save their metrics."""
    for path in (DATA, *MODELS.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = 0 if torch.cuda.is_available() else "cpu"
    batch = 4 if device != "cpu" else 1
    workers = 4 if device != "cpu" else 0
    OUTPUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, weights in MODELS.items():
        print(f"\nEvaluating {model_name}: {weights}")
        metrics = YOLO(weights).val(
            data=DATA,
            split="test",
            imgsz=1280,
            batch=batch,
            device=device,
            workers=workers,
            project=OUTPUT,
            name=model_name,
            exist_ok=True,
            plots=True,
        )
        rows.extend(metric_rows(model_name, metrics))

    output_csv = OUTPUT / "comparison.csv"
    with output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print_comparison(rows)
    print(f"\nComparison saved to {output_csv}")


if __name__ == "__main__":
    main()
