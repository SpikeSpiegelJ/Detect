"""Evaluate a trained detector using the same resolution as training."""

from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import Ensemble


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "dataset" / "data_repartition_v15_oof_reviewed_9c.yaml"
DEFAULT_MODEL = ROOT / "runs" / "train" / "yolo26s_v15_oof_reviewed_ft" / "weights" / "best.pt"


def parse_weights(value):
    """Parse one model's scalar or comma-separated per-class ensemble weights."""
    return tuple(float(weight) for weight in value.split(","))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model", type=Path, nargs="+", default=[DEFAULT_MODEL], help="One checkpoint, or several for NMS ensembling."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset YAML.")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--head",
        choices=("one-to-many", "one-to-one"),
        default="one-to-many",
        help="YOLO26 detection head. The one-to-many head applies NMS.",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument(
        "--ensemble-weights",
        type=parse_weights,
        nargs="+",
        help="One scalar or comma-separated class-weight vector per ensemble member.",
    )
    parser.add_argument(
        "--augment", action=BooleanOptionalAction, default=False, help="Use test-time augmentation during validation."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_paths = [path.resolve() for path in args.model]
    data_path = args.data.resolve()
    for model_path in model_paths:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"No evaluation weights found at {model_path}. Train the selected model first, or pass --model explicitly."
            )
    device = 0 if torch.cuda.is_available() else "cpu"
    end2end = args.head == "one-to-one"
    if len(model_paths) > 1 and end2end:
        raise ValueError("Checkpoint ensembling requires --head one-to-many so NMS can merge all predictions.")
    if args.ensemble_weights is not None and len(args.ensemble_weights) != len(model_paths):
        raise ValueError("--ensemble-weights must provide one scalar or class-weight vector per model.")
    print("Evaluating model(s):")
    for model_path in model_paths:
        print(f"  {model_path}")
    print(f"Dataset config: {data_path}")
    print(f"Dataset split: {args.split}")
    print(f"Detection head: {args.head}")
    print(f"Input size: {args.imgsz}, batch: {args.batch}, TTA: {args.augment}, NMS IoU: {args.iou}")
    mode = "tta" if args.augment else "single_scale"
    run_prefix = "__".join(path.parent.parent.name for path in model_paths)
    iou_tag = str(args.iou).replace(".", "p")
    run_name = f"{run_prefix}_{args.head.replace('-', '_')}_{args.imgsz}_{mode}_iou{iou_tag}_{args.split}"
    val_args = {
        "data": data_path,
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch if device != "cpu" else 1,
        "device": device,
        "workers": 4 if device != "cpu" else 0,
        "end2end": end2end,
        "augment": args.augment,
        "iou": args.iou,
        "plots": True,
    }
    if len(model_paths) == 1:
        metrics = YOLO(model_paths[0]).val(
            **val_args,
            project=ROOT / "runs" / "val",
            name=run_name,
        )
    else:
        ensemble = Ensemble(args.ensemble_weights)
        for model_path in model_paths:
            member = YOLO(model_path).model
            member.end2end = False
            if ensemble and member.names != ensemble.names:
                raise ValueError(f"Class names differ between ensemble members: {model_path}")
            if args.ensemble_weights is not None and len(args.ensemble_weights[len(ensemble)]) not in {1, len(member.names)}:
                raise ValueError("Each ensemble weight must be one scalar or contain one value per class.")
            ensemble.append(member)
            ensemble.names = member.names
        ensemble.stride = ensemble[0].stride
        ensemble.yaml = ensemble[0].yaml
        ensemble.end2end = False
        validator = DetectionValidator(
            args={"model": "ensemble.pt", **val_args}, save_dir=ROOT / "runs" / "val" / run_name
        )
        validator(model=ensemble)
        metrics = validator.metrics
    print(f"mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
