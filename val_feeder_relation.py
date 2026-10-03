# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validate feeder-class score calibration from detected endpoint-device relations."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import torch

from ultralytics.models.yolo.detect import DetectionValidator

from val_specialist_fusion import DEFAULT_DATA, DEFAULT_GENERAL, load_general_ensemble


ROOT = Path(__file__).resolve().parent


class FeederRelationCalibration(torch.nn.Module):
    """Calibrate feeder scores using image-level RRU and antenna-b evidence."""

    def __init__(self, general: torch.nn.Module, strength: float = 0.0):
        super().__init__()
        if strength < 0:
            raise ValueError("strength must be non-negative")
        self.general = general
        self.strength = strength
        self.names = general.names
        self.stride = general.stride
        self.yaml = general.yaml
        self.end2end = False
        name_to_id = {name: class_id for class_id, name in self.names.items()}
        required = {"RRU", "Feeder_RRU", "antenna_b", "Feeder_antenna"}
        if missing := required - name_to_id.keys():
            raise ValueError(f"Missing relation classes: {sorted(missing)}")
        self.rru_id = name_to_id["RRU"]
        self.feeder_rru_id = name_to_id["Feeder_RRU"]
        self.antenna_id = name_to_id["antenna_b"]
        self.feeder_antenna_id = name_to_id["Feeder_antenna"]

    @staticmethod
    def predictions(output):
        """Extract inference predictions from a model output tuple."""
        return output[0] if isinstance(output, (tuple, list)) else output

    def calibrate(self, prediction: torch.Tensor) -> torch.Tensor:
        """Apply symmetric score gains from relative endpoint-device evidence."""
        if self.strength == 0:
            return prediction
        prediction = prediction.clone()
        scores = prediction[:, 4:]
        rru_support = scores[:, self.rru_id].amax(dim=1)
        antenna_support = scores[:, self.antenna_id].amax(dim=1)
        evidence = (rru_support - antenna_support) / (rru_support + antenna_support).clamp_min(1e-6)
        rru_gain = torch.exp(self.strength * evidence).unsqueeze(1)
        antenna_gain = torch.exp(-self.strength * evidence).unsqueeze(1)
        scores[:, self.feeder_rru_id] = (scores[:, self.feeder_rru_id] * rru_gain).clamp_max(1)
        scores[:, self.feeder_antenna_id] = (scores[:, self.feeder_antenna_id] * antenna_gain).clamp_max(1)
        return prediction

    def forward(self, x, augment=False, profile=False, visualize=False, embed=None):
        """Run the general TTA ensemble and calibrate its feeder scores before NMS."""
        prediction = self.predictions(
            self.general(x, augment=True, profile=profile, visualize=visualize, embed=embed)
        )
        return self.calibrate(prediction), None


class SpatialFeederRelationCalibration(FeederRelationCalibration):
    """Calibrate each feeder candidate from its proximity to endpoint-device candidates."""

    def __init__(self, general: torch.nn.Module, strength: float = 0.0, temperature: float = 0.05, topk: int = 20):
        super().__init__(general, strength)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if topk <= 0:
            raise ValueError("topk must be positive")
        self.temperature = temperature
        self.topk = topk

    def nearby_support(
        self, boxes: torch.Tensor, anchor_scores: torch.Tensor, image_size: tuple[int, int]
    ) -> torch.Tensor:
        """Return distance-decayed support from the strongest endpoint candidates."""
        count = min(self.topk, anchor_scores.shape[1])
        scores, indices = anchor_scores.topk(count, dim=1)
        anchors = boxes.gather(1, indices.unsqueeze(-1).expand(-1, -1, 4))
        feeder, anchors = boxes.unsqueeze(2), anchors.unsqueeze(1)
        dx = (
            (feeder[..., 0] - anchors[..., 0]).abs() - (feeder[..., 2] + anchors[..., 2]) / 2
        ).clamp_min(0)
        dy = (
            (feeder[..., 1] - anchors[..., 1]).abs() - (feeder[..., 3] + anchors[..., 3]) / 2
        ).clamp_min(0)
        height, width = image_size
        distance = torch.sqrt((dx / width).square() + (dy / height).square())
        return (scores.unsqueeze(1) * torch.exp(-distance / self.temperature)).amax(dim=2)

    def calibrate(self, prediction: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        """Apply candidate-specific gains from nearby endpoint predictions."""
        if self.strength == 0:
            return prediction
        prediction = prediction.clone()
        boxes, scores = prediction[:, :4].transpose(1, 2), prediction[:, 4:]
        rru_support = self.nearby_support(boxes, scores[:, self.rru_id], image_size)
        antenna_support = self.nearby_support(boxes, scores[:, self.antenna_id], image_size)
        evidence = (rru_support - antenna_support) / (rru_support + antenna_support).clamp_min(1e-6)
        scores[:, self.feeder_rru_id] = (
            scores[:, self.feeder_rru_id] * torch.exp(self.strength * evidence)
        ).clamp_max(1)
        scores[:, self.feeder_antenna_id] = (
            scores[:, self.feeder_antenna_id] * torch.exp(-self.strength * evidence)
        ).clamp_max(1)
        return prediction

    def forward(self, x, augment=False, profile=False, visualize=False, embed=None):
        """Run general TTA and candidate-level spatial calibration before NMS."""
        prediction = self.predictions(
            self.general(x, augment=True, profile=profile, visualize=visualize, embed=embed)
        )
        return self.calibrate(prediction, x.shape[-2:]), None


def parse_args():
    """Parse relation-calibration validation settings."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--general", type=Path, nargs="+", default=DEFAULT_GENERAL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--strength", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--relation-mode", choices=("global", "spatial"), default="global")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "val" / "feeder_relation")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_setting(args, general, strength: float) -> dict:
    """Validate one frozen relation strength and return stable metrics."""
    model = (
        FeederRelationCalibration(general, strength)
        if args.relation_mode == "global"
        else SpatialFeederRelationCalibration(general, strength, args.temperature, args.topk)
    )
    device = 0 if torch.cuda.is_available() else "cpu"
    validator = DetectionValidator(
        args={
            "model": "feeder_relation.pt",
            "data": args.data.resolve(),
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch if device != "cpu" else 1,
            "device": device,
            "workers": 4 if device != "cpu" else 0,
            "end2end": False,
            "augment": True,
            "iou": args.iou,
            "plots": False,
        },
        save_dir=args.project.resolve()
        / f"{args.relation_mode}_strength_{str(strength).replace('.', 'p')}_{args.split}",
    )
    started = perf_counter()
    validator(model=model)
    metrics = validator.metrics
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
    result = {
        "strength": strength,
        "split": args.split,
        "images": int(validator.seen),
        "elapsed_seconds": perf_counter() - started,
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "per_class": per_class,
    }
    print(f"strength={strength:g}, mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}")
    return result


def main():
    """Run the validation-only strength sweep and save every setting."""
    args = parse_args()
    required = [*args.general, args.data]
    for path in required:
        if not path.resolve().is_file():
            raise FileNotFoundError(path.resolve())
    if any(strength < 0 for strength in args.strength):
        raise ValueError("--strength must be non-negative")
    if not 0 < args.iou <= 1:
        raise ValueError("--iou must be in (0, 1]")
    if args.temperature <= 0 or args.topk <= 0:
        raise ValueError("--temperature and --topk must be positive")
    general = load_general_ensemble(args.general)
    report = {
        "general_weights": [str(path.resolve()) for path in args.general],
        "data": str(args.data.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "nms_iou": args.iou,
        "relation_mode": args.relation_mode,
        "temperature": args.temperature,
        "topk": args.topk,
        "results": [],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    for strength in args.strength:
        report["results"].append(validate_setting(args, general, strength))
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
