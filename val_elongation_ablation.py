# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate named frozen checkpoints with an identical protocol."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from val_specialist_ablation import metrics_record


ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS = ROOT / "runs" / "train"
DEFAULT_DATA = ROOT / "dataset" / "data_repartition_v13_cut_feeder_hard_corrected_9c.yaml"
DEFAULT_VARIANTS = {
    "control": "paper_yolo26s_elongation_control_seed0",
    "elongation_loss": "paper_yolo26s_elongation_loss_seed0",
}


def parse_args():
    """Parse paired-evaluation arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val"])
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--variant",
        action="append",
        metavar="LABEL=RUN_NAME",
        help="Named run under --runs; repeat for a custom comparison.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    """Evaluate both frozen checkpoints and persist after every run."""
    args = parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    if not data.is_file():
        raise FileNotFoundError(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "data": str(data),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "nms_iou": args.iou,
        "augment": False,
        "variants": {},
    }
    device = 0 if torch.cuda.is_available() else "cpu"
    variants = DEFAULT_VARIANTS
    if args.variant:
        if any("=" not in item for item in args.variant):
            raise ValueError("Each --variant must use LABEL=RUN_NAME")
        variants = dict(item.split("=", 1) for item in args.variant)
        if any(not label or not run_name for label, run_name in variants.items()):
            raise ValueError("Variant labels and run names must not be empty")
    for variant, run_name in variants.items():
        weight = args.runs.resolve() / run_name / "weights" / "best.pt"
        if not weight.is_file():
            raise FileNotFoundError(weight)
        report["variants"][variant] = []
        model = YOLO(weight)
        for split in args.splits:
            metrics = model.val(
                data=data,
                split=split,
                imgsz=args.imgsz,
                batch=args.batch if device != "cpu" else 1,
                device=device,
                workers=4 if device != "cpu" else 0,
                iou=args.iou,
                augment=False,
                plots=False,
                project=ROOT / "runs" / "val" / "paired_ablation_2026-10-03",
                name=f"{variant}_{split}",
                verbose=False,
            )
            report["variants"][variant].append(metrics_record(metrics, split, weight))
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{variant} {split}: mAP50={metrics.box.map50:.5f}, mAP50-95={metrics.box.map:.5f}")


if __name__ == "__main__":
    main()
