# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate specialist data-strategy ablations on unchanged validation and test splits."""

from __future__ import annotations

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import torch

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS = ROOT / "runs" / "train"
DEFAULT_DATA = ROOT / "dataset" / "specialist_ablation_manifests" / "full_context_confuser.yaml"
VARIANTS = (
    "full_positive",
    "full_positive_confuser",
    "full_positive_context",
    "full_context_confuser",
    "full_positive_tight",
)
DEFAULT_VARIANTS = VARIANTS[:-1]


def parse_args():
    """Parse specialist ablation evaluation arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=DEFAULT_VARIANTS)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics_record(metrics, split: str, weight: Path) -> dict:
    """Convert Ultralytics detection metrics into stable JSON primitives."""
    per_class = []
    for result_index, class_id in enumerate(metrics.ap_class_index):
        precision, recall, map50, map50_95 = metrics.class_result(result_index)
        per_class.append(
            {
                "class_id": int(class_id),
                "class_name": metrics.names[int(class_id)],
                "instances": int(metrics.nt_per_class[int(class_id)]),
                "precision": float(precision),
                "recall": float(recall),
                "mAP50": float(map50),
                "mAP50-95": float(map50_95),
            }
        )
    return {
        "split": split,
        "weight": str(weight),
        "weight_sha256": sha256_file(weight),
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "per_class": per_class,
    }


def main():
    """Evaluate every ablation checkpoint and persist results after each run."""
    args = parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    if not data.is_file():
        raise FileNotFoundError(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"data": str(data), "imgsz": args.imgsz, "batch": args.batch, "variants": {}}
    device = 0 if torch.cuda.is_available() else "cpu"
    for variant in args.variants:
        weight = args.runs.resolve() / f"paper_specialist_ablation_{variant}_v9seed0" / "weights" / "best.pt"
        if not weight.is_file():
            raise FileNotFoundError(weight)
        report["variants"][variant] = []
        model = YOLO(weight)
        for split in ("val", "test"):
            metrics = model.val(
                data=data,
                split=split,
                imgsz=args.imgsz,
                batch=args.batch if device != "cpu" else 1,
                device=device,
                workers=4 if device != "cpu" else 0,
                plots=False,
                project=ROOT / "runs" / "val" / "specialist_ablation_2026-10-03",
                name=f"{variant}_{split}",
                verbose=False,
            )
            report["variants"][variant].append(metrics_record(metrics, split, weight))
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(
                f"{variant} {split}: mAP50={metrics.box.map50:.4f}, "
                f"mAP50-95={metrics.box.map:.4f}"
            )


if __name__ == "__main__":
    main()
