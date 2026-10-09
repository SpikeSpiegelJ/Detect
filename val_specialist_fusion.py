# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validate a nine-class TTA ensemble augmented by a single-scale two-class specialist."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import Ensemble

ROOT = Path(__file__).resolve().parent
DEFAULT_GENERAL = [
    ROOT / "runs" / "train" / "yolo26s_v9_trainval_o2m_dfl15_ft" / "weights" / "best.pt",
    ROOT / "runs" / "train" / "yolo26m_v9_trainval_ft" / "weights" / "best.pt",
]
DEFAULT_SPECIALIST = ROOT / "runs" / "train" / "yolo26s_v14_cut_feeder_specialist-2" / "weights" / "best.pt"
DEFAULT_DATA = ROOT / "dataset" / "data_repartition_v13_cut_feeder_hard_corrected_9c.yaml"


class SpecialistFusion(torch.nn.Module):
    """Concatenate mapped specialist scores with general-detector predictions before NMS."""

    def __init__(self, general: torch.nn.Module, specialist: torch.nn.Module, specialist_weight: float = 1.0):
        super().__init__()
        self.general = general
        self.specialist = specialist
        self.specialist_weight = specialist_weight
        self.names = general.names
        self.stride = general.stride
        self.yaml = general.yaml
        self.end2end = False
        expected = {0: "cut", 1: "Feeder_antenna"}
        if specialist.names != expected:
            raise ValueError(f"Expected specialist classes {expected}, received {specialist.names}")
        name_to_id = {name: class_id for class_id, name in self.names.items()}
        self.class_map = (name_to_id["cut"], name_to_id["Feeder_antenna"])

    @staticmethod
    def predictions(output):
        """Extract inference predictions from a model output tuple."""
        return output[0] if isinstance(output, (tuple, list)) else output

    def forward(self, x, augment=False, profile=False, visualize=False, embed=None):
        """Run general TTA and specialist single-scale inference, then map two scores into nine classes."""
        general = self.predictions(self.general(x, augment=True, profile=profile, visualize=visualize, embed=embed))
        specialist = self.predictions(
            self.specialist(x, augment=False, profile=profile, visualize=visualize, embed=embed)
        )
        mapped = specialist.new_zeros((specialist.shape[0], 4 + len(self.names), specialist.shape[2]))
        mapped[:, :4] = specialist[:, :4]
        for source_id, target_id in enumerate(self.class_map):
            mapped[:, 4 + target_id] = specialist[:, 4 + source_id] * self.specialist_weight
        return torch.cat((general, mapped), dim=2), None


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--general", type=Path, nargs="+", default=DEFAULT_GENERAL)
    parser.add_argument("--specialist", type=Path, default=DEFAULT_SPECIALIST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--iou", type=float, nargs="+", default=[0.50])
    parser.add_argument("--specialist-weight", type=float, nargs="+", default=[1.0])
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "val")
    parser.add_argument("--output", type=Path, help="Optional JSON summary for all evaluated settings.")
    parser.add_argument("--no-plots", action="store_false", dest="plots", help="Skip validation plots.")
    parser.set_defaults(plots=True)
    return parser.parse_args()


def load_general_ensemble(general_paths: list[Path]) -> Ensemble:
    """Load compatible general detectors into one reusable ensemble."""
    general = Ensemble()
    for path in general_paths:
        member = YOLO(path.resolve()).model
        member.end2end = False
        if general and member.names != general.names:
            raise ValueError(f"General model class names differ: {path}")
        general.append(member)
        general.names = member.names
    general.stride = general[0].stride
    general.yaml = general[0].yaml
    general.end2end = False
    return general


def load_models(general_paths: list[Path], specialist_path: Path) -> tuple[Ensemble, torch.nn.Module]:
    """Load the general ensemble and two-class specialist once for a validation sweep."""
    general = load_general_ensemble(general_paths)
    specialist = YOLO(specialist_path.resolve()).model
    specialist.end2end = False
    return general, specialist


def validate_setting(args, general: Ensemble, specialist: torch.nn.Module, weight: float, iou: float) -> dict:
    """Validate one specialist-weight and NMS-IoU setting and return machine-readable metrics."""
    model = SpecialistFusion(general, specialist, weight)

    device = 0 if torch.cuda.is_available() else "cpu"
    run_name = (
        f"v14_specialist_fusion_w{str(weight).replace('.', 'p')}_"
        f"tta_general_single_specialist_iou{str(iou).replace('.', 'p')}_{args.split}"
    )
    validator = DetectionValidator(
        args={
            "model": "specialist_fusion.pt",
            "data": args.data.resolve(),
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch if device != "cpu" else 1,
            "device": device,
            "workers": 4 if device != "cpu" else 0,
            "end2end": False,
            "augment": True,
            "iou": iou,
            "plots": args.plots,
        },
        save_dir=args.project.resolve() / run_name,
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
        "specialist_weight": weight,
        "nms_iou": iou,
        "split": args.split,
        "images": int(validator.seen),
        "elapsed_seconds": perf_counter() - started,
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "per_class": per_class,
        "save_dir": str(validator.save_dir),
    }
    print(f"weight={weight:g}, iou={iou:g}, mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}")
    return result


def main():
    args = parse_args()
    paths = [path.resolve() for path in [*args.general, args.specialist, args.data]]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if any(weight < 0 for weight in args.specialist_weight):
        raise ValueError("--specialist-weight must be non-negative")
    if any(not 0 < iou <= 1 for iou in args.iou):
        raise ValueError("--iou must be in (0, 1]")

    general, specialist = load_models(args.general, args.specialist)
    results = [
        validate_setting(args, general, specialist, weight, iou)
        for weight in args.specialist_weight
        for iou in args.iou
    ]
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "general_weights": [str(path.resolve()) for path in args.general],
            "specialist_weight_file": str(args.specialist.resolve()),
            "data": str(args.data.resolve()),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "results": results,
        }
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
