"""Train the local YOLO detector."""

from argparse import ArgumentParser, BooleanOptionalAction
import os
from pathlib import Path

# This Conda environment loads both Intel OpenMP runtimes. Set before importing PyTorch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
V6_WEIGHTS = ROOT / "runs" / "train" / "yolo12s_p2_repartition_v6_dataset7_9c" / "weights" / "best.pt"
V6_DATA = ROOT / "dataset" / "data_rep     artition_v6_dataset7_9c.yaml"
V7_DATA = ROOT / "dataset" / "data_repartition_v7_dataset8_9c.yaml"
V8_DATA = ROOT / "dataset" / "data_repartition_v8_dataset9_9c.yaml"
V9_DATA = ROOT / "dataset" / "data_repartition_v9_trainval_9c.yaml"
V10_AUGMENTED_DATA = ROOT / "dataset" / "data_repartition_v10_augmented_9c.yaml"
V11_TARGET_CROP_DATA = ROOT / "dataset" / "data_repartition_v11_target_crop_9c.yaml"
V13_CUT_FEEDER_HARD_DATA = ROOT / "dataset" / "data_repartition_v13_cut_feeder_hard_9c.yaml"
V14_CUT_FEEDER_SPECIALIST_DATA = ROOT / "dataset" / "data_repartition_v14_cut_feeder_specialist.yaml"
V15_OOF_REVIEWED_DATA = ROOT / "dataset" / "data_repartition_v15_oof_reviewed_9c.yaml"
V15_ANNOTATION_CONTROL_DATA = ROOT / "dataset" / "data_repartition_v15_annotation_control_clean_val_9c.yaml"
V8_REVIEWED_HARD_DATA = ROOT / "dataset" / "data_repartition_v8_reviewed_hard_2x.yaml"
FOCUS_HARDNEG_DATA = ROOT / "dataset" / "data_repartition_v6_focus_hardneg.yaml"
YOLO26S_WEIGHTS = ROOT / "yolo26s.pt"
YOLO26M_WEIGHTS = ROOT / "yolo26m.pt"
YOLO26S_O2M_CFG = ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26s-o2m.yaml"
YOLO26S_CBAM_P3_O2M_CFG = ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26s-cbam-p3-o2m.yaml"
YOLO26S_P2_O2M_CFG = ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26s-p2-o2m.yaml"
YOLO26M_O2M_CFG = ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26m-o2m.yaml"
V8_WEIGHTS = ROOT / "runs" / "train" / "yolo26s_v8_dataset9_9c" / "weights" / "best.pt"
V8_RELABEL_WEIGHTS = ROOT / "runs" / "train" / "yolo26s_v8_dataset9_relabel_ft" / "weights" / "best.pt"
V8_YOLO26M_WEIGHTS = ROOT / "runs" / "train" / "yolo26m_v8_corrected_full" / "weights" / "best.pt"
V9_YOLO26S_WEIGHTS = ROOT / "runs" / "train" / "yolo26s_v9_trainval_o2m_dfl15_ft" / "weights" / "best.pt"
DEFAULT_PRESET = "v15_oof_reviewed"


PRESETS = {
    "focus_hardneg": {
        "model": V6_WEIGHTS,
        "data": FOCUS_HARDNEG_DATA,
        "name": "yolo12s_p2_v6_focus_hardneg_ft",
        "imgsz": 1280,
        "epochs": 30,
        "batch": 2,
        "lr0": 0.00005,
        "lrf": 0.01,
        "patience": 10,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "degrees": 0.5,
        "translate": 0.02,
        "scale": 0.10,
        "cls_pw": 0.0,
    },
    "v6_nomosaic": {
        "model": V6_WEIGHTS,
        "data": V6_DATA,
        "name": "yolo12s_p2_v6_nomosaic_ft",
        "imgsz": 1280,
        "epochs": 30,
        "batch": 2,
        "lr0": 0.00005,
        "lrf": 0.01,
        "patience": 10,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "degrees": 0.5,
        "translate": 0.02,
        "scale": 0.10,
        "cls_pw": 0.0,
    },
}

PRESETS["v6_cbam"] = {
    **PRESETS["v6_nomosaic"],
    "model": ROOT / "ultralytics/cfg/models/12/yolo12s-p2-cbam.yaml",
    "name": "yolo12s_p2_v6_cbam_ft",
}

PRESETS["yolo26s"] = {
    "model": YOLO26S_WEIGHTS,
    "data": V7_DATA,
    "name": "yolo26s_repartition_v7_dataset8_9c",
    "imgsz": 1280,
    "epochs": 150,
    "batch": 2,
    "optimizer": "MuSGD",
    "lr0": 0.00038,
    "lrf": 0.88219,
    "momentum": 0.94751,
    "weight_decay": 0.00027,
    "warmup_epochs": 1.0,
    "cos_lr": False,
    "box": 9.83241,
    "cls": 0.64896,
    "dfl": 0.95824,
    "patience": 30,
    "mosaic": 0.5,
    "close_mosaic": 10,
    "degrees": 0.0,
    "translate": 0.08,
    "scale": 0.35,
    "fliplr": 0.30393,
    "hsv_h": 0.01315,
    "hsv_s": 0.35348,
    "hsv_v": 0.19383,
    "cls_pw": 0.0,
}

PRESETS["v7_control"] = {
    **PRESETS["yolo26s"],
    "model": ROOT / "runs/train/yolo26s_repartition_v7_dataset8_9c/weights/best.pt",
    "name": "yolo26s_v7_control_ft",
    "epochs": 40,
    "patience": 15,
    "lr0": 0.00005,
    "lrf": 0.1,
    "cos_lr": True,
    "mosaic": 0.0,
    "close_mosaic": 0,
    "scale": 0.15,
    "translate": 0.03,
}
PRESETS["v7_corrected"] = {
    **PRESETS["v7_control"],
    "model": ROOT / "runs/train/yolo26s_v7_control_ft-2/weights/best.pt",
    "data": V7_DATA,
    "name": "yolo26s_v7_corrected_ft",
}
PRESETS["v8_dataset9"] = {
    **PRESETS["yolo26s"],
    "data": V8_DATA,
    "name": "yolo26s_v8_dataset9_9c",
}
PRESETS["v8_corrected_full"] = {
    **PRESETS["v8_dataset9"],
    "name": "yolo26s_v8_corrected_full",
    "fl_gamma": 0.0,
}
PRESETS["v8_yolo26m"] = {
    **PRESETS["v8_corrected_full"],
    "model": YOLO26M_WEIGHTS,
    "name": "yolo26m_v8_corrected_full",
}
PRESETS["v8_reviewed_hard_2x"] = {
    **PRESETS["v8_corrected_full"],
    "data": V8_REVIEWED_HARD_DATA,
    "name": "yolo26s_v8_reviewed_hard_2x",
}
PRESETS["v8_corrected_focal"] = {
    **PRESETS["v8_corrected_full"],
    "name": "yolo26s_v8_corrected_focal",
    "fl_gamma": 1.5,
}
PRESETS["v8_relabel_ft"] = {
    **PRESETS["v8_dataset9"],
    "model": V8_WEIGHTS,
    "name": "yolo26s_v8_dataset9_relabel_ft",
    "epochs": 30,
    "patience": 12,
    "lr0": 0.00005,
    "lrf": 0.1,
    "cos_lr": True,
    "mosaic": 0.0,
    "close_mosaic": 0,
    "translate": 0.03,
    "scale": 0.15,
}
PRESETS["v8_train_relabel_ft"] = {
    **PRESETS["v8_relabel_ft"],
    "model": V8_RELABEL_WEIGHTS,
    "name": "yolo26s_v8_train_relabel_ft",
    "epochs": 25,
    "patience": 10,
    "lr0": 0.00003,
    "translate": 0.02,
    "scale": 0.10,
}
PRESETS["v8_yolo26s_o2m_ft"] = {
    **PRESETS["v8_train_relabel_ft"],
    "model": YOLO26S_O2M_CFG,
    "weights": V8_RELABEL_WEIGHTS,
    "name": "yolo26s_v8_o2m_ft",
    "save_period": 1,
}
PRESETS["v8_yolo26s_p2_o2m_ft"] = {
    **PRESETS["v8_yolo26s_o2m_ft"],
    "model": YOLO26S_P2_O2M_CFG,
    "name": "yolo26s_p2_v8_o2m_dfl15_headtransfer_ft",
    "batch": 1,
    "dfl": 1.5,
}
PRESETS["v8_yolo26s_o2m_topk13_ft"] = {
    **PRESETS["v8_yolo26s_o2m_ft"],
    "name": "yolo26s_v8_o2m_dfl15_topk13_ft",
    "dfl": 1.5,
    "tal_topk": 13,
}
PRESETS["v8_yolo26s_cbam_p3_o2m_ft"] = {
    **PRESETS["v8_yolo26s_o2m_ft"],
    "model": YOLO26S_CBAM_P3_O2M_CFG,
    "name": "yolo26s_cbam_p3_v8_o2m_dfl15_ft",
    "dfl": 1.5,
}
PRESETS["v8_yolo26m_ft"] = {
    **PRESETS["v8_yolo26m"],
    "model": V8_YOLO26M_WEIGHTS,
    "name": "yolo26m_v8_corrected_full_ft",
    "epochs": 30,
    "patience": 10,
    "lr0": 0.00002,
    "lrf": 0.1,
    "cos_lr": True,
    "mosaic": 0.0,
    "close_mosaic": 0,
    "translate": 0.02,
    "scale": 0.10,
}
PRESETS["v8_yolo26m_o2m_ft"] = {
    **PRESETS["v8_yolo26m_ft"],
    "model": YOLO26M_O2M_CFG,
    "weights": V8_YOLO26M_WEIGHTS,
    "name": "yolo26m_v8_o2m_dfl15_ft",
    "epochs": 25,
    "dfl": 1.5,
    "save_period": 1,
}
PRESETS["v10_augmented_yolo26s_ft"] = {
    **PRESETS["v8_yolo26s_o2m_ft"],
    "model": V9_YOLO26S_WEIGHTS,
    "data": V10_AUGMENTED_DATA,
    "name": "yolo26s_v10_augmented_ft",
    "epochs": 30,
    "patience": 10,
    "lr0": 0.00003,
    "lrf": 0.1,
    "cos_lr": True,
    "dfl": 1.5,
    "mosaic": 0.0,
    "close_mosaic": 0,
    "degrees": 0.0,
    "translate": 0.02,
    "scale": 0.10,
}
PRESETS["v11_target_crop_yolo26s_ft"] = {
    **PRESETS["v10_augmented_yolo26s_ft"],
    "data": V11_TARGET_CROP_DATA,
    "name": "yolo26s_v11_target_crop_ft",
}
PRESETS["v13_cut_feeder_hard_yolo26s_ft"] = {
    **PRESETS["v11_target_crop_yolo26s_ft"],
    "data": V13_CUT_FEEDER_HARD_DATA,
    "name": "yolo26s_v13_cut_feeder_hard_ft",
}
PRESETS["v14_cut_feeder_specialist"] = {
    **PRESETS["v11_target_crop_yolo26s_ft"],
    "model": V9_YOLO26S_WEIGHTS,
    "weights": None,
    "data": V14_CUT_FEEDER_SPECIALIST_DATA,
    "name": "yolo26s_v14_cut_feeder_specialist",
    "epochs": 40,
    "patience": 12,
}
PRESETS["v15_oof_reviewed"] = {
    **PRESETS["v10_augmented_yolo26s_ft"],
    "model": V9_YOLO26S_WEIGHTS,
    "weights": None,
    "data": V15_OOF_REVIEWED_DATA,
    "name": "yolo26s_v15_oof_reviewed_ft",
    "epochs": 30,
    "patience": 10,
    "save_period": -1,
}
PRESETS["paper_elongation_control"] = {
    **PRESETS["v10_augmented_yolo26s_ft"],
    "model": V9_YOLO26S_WEIGHTS,
    "weights": None,
    "data": V9_DATA,
    "name": "paper_yolo26s_elongation_control_seed0",
    "epochs": 15,
    "patience": 6,
    "save_period": -1,
    "elongation_gain": 0.0,
    "elongation_threshold": 4.0,
}
PRESETS["paper_elongation_loss"] = {
    **PRESETS["paper_elongation_control"],
    "name": "paper_yolo26s_elongation_loss_seed0",
    "elongation_gain": 1.0,
}
PRESETS["paper_p2_baseline"] = {
    **PRESETS["paper_elongation_control"],
    "model": YOLO26S_P2_O2M_CFG,
    "weights": V9_YOLO26S_WEIGHTS,
    "name": "paper_yolo26s_p2_baseline_seed0",
}
PRESETS["paper_annotation_control"] = {
    **PRESETS["v15_oof_reviewed"],
    "data": V15_ANNOTATION_CONTROL_DATA,
    "name": "paper_yolo26s_v15_annotation_control_seed0",
}


def build_model(args):
    """Build the candidate and explicitly transfer the unchanged V6 layers."""
    model = YOLO(args.model)
    if args.preset == "v6_cbam":
        source = YOLO(args.weights or V6_WEIGHTS).model.float()
        target = model.model
        assert len(source.model) == 28 and len(target.model) == 29
        for index in range(27):
            target.model[index].load_state_dict(source.model[index].state_dict(), strict=True)
        target.model[28].load_state_dict(source.model[27].state_dict(), strict=True)
        target.names = source.names.copy()
        # Model.train rebuilds from YAML and transfers this initialized model only when ckpt is set.
        model.ckpt = {"model": target}
        print("Transferred all 28 V6 layers; only P2 CBAM is newly initialized.")
    elif args.preset in {"v8_yolo26s_p2_o2m_ft", "paper_p2_baseline"}:
        source = YOLO(args.weights or V8_RELABEL_WEIGHTS).model.float()
        target = model.model
        assert len(source.model) == 24 and len(target.model) == 30
        for index in range(17):
            target.model[index].load_state_dict(source.model[index].state_dict(), strict=True)
        for source_index, target_index in zip(range(17, 23), range(23, 29)):
            target.model[target_index].load_state_dict(source.model[source_index].state_dict(), strict=True)
        for branch in ("cv2", "cv3"):
            source_layers = getattr(source.model[23], branch)
            target_layers = getattr(target.model[29], branch)
            for source_index, target_index in enumerate(range(1, 4)):
                target_layers[target_index].load_state_dict(source_layers[source_index].state_dict(), strict=True)
        target.names = source.names.copy()
        model.ckpt = {"model": target}
        print("Transferred the backbone plus reusable P3-P5 neck and Detect branches; only P2 is initialized.")
    elif args.preset == "v8_yolo26s_cbam_p3_o2m_ft":
        source = YOLO(args.weights or V8_RELABEL_WEIGHTS).model.float()
        target = model.model
        assert len(source.model) == 24 and len(target.model) == 25
        for index in range(23):
            target.model[index].load_state_dict(source.model[index].state_dict(), strict=True)
        for branch in ("cv2", "cv3"):
            getattr(target.model[24], branch).load_state_dict(
                getattr(source.model[23], branch).state_dict(), strict=True
            )
        target.names = source.names.copy()
        model.ckpt = {"model": target}
        print("Transferred all backbone, neck, and one-to-many Detect weights; only P3 CBAM is initialized.")
    elif args.weights is not None:
        model.load(args.weights)
    return model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--preset", choices=tuple(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--seed", type=int, default=0, help="Use the same seed for both comparison runs.")
    parser.add_argument("--model", type=Path, help="Checkpoint or model YAML. Defaults to the selected preset.")
    parser.add_argument("--weights", type=Path, help="Optional initialization weights when --model is a YAML.")
    parser.add_argument("--data", type=Path, help="Dataset YAML. Defaults to the selected preset.")
    parser.add_argument("--imgsz", type=int, help="Input size. Defaults to the selected preset.")
    parser.add_argument("--epochs", type=int, help="Training epochs. Defaults to the selected preset.")
    parser.add_argument("--batch", type=int, help="Fixed GPU batch size. Defaults to the selected preset.")
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0. Defaults to CUDA when available.")
    parser.add_argument("--name", help="Run name. Defaults to the selected preset.")
    parser.add_argument("--lr0", type=float, help="Initial learning rate. Defaults to the selected preset.")
    parser.add_argument("--lrf", type=float, help="Final learning rate fraction. Defaults to the selected preset.")
    parser.add_argument("--optimizer", help="Optimizer name. Defaults to the selected preset or AdamW.")
    parser.add_argument("--momentum", type=float, help="Optimizer momentum. Defaults to the framework setting.")
    parser.add_argument("--weight-decay", type=float, help="Weight decay. Defaults to the framework setting.")
    parser.add_argument("--warmup-epochs", type=float, help="Warmup length. Defaults to the framework setting.")
    parser.add_argument("--cos-lr", action=BooleanOptionalAction, default=None)
    parser.add_argument("--box", type=float, help="Box loss gain. Defaults to the framework setting.")
    parser.add_argument("--cls", type=float, help="Classification loss gain. Defaults to the framework setting.")
    parser.add_argument("--dfl", type=float, help="Distance regression loss gain. Defaults to the framework setting.")
    parser.add_argument("--elongation-gain", type=float, help="Normalized regression emphasis for elongated boxes.")
    parser.add_argument("--elongation-threshold", type=float, help="Aspect ratio that starts elongated-box emphasis.")
    parser.add_argument("--tal-topk", type=int, help="Task-aligned assigner candidates per ground-truth box.")
    parser.add_argument("--patience", type=int, help="Early-stopping patience. Defaults to the selected preset.")
    parser.add_argument("--save-period", type=int, help="Save every N epochs. Negative values disable periodic saves.")
    parser.add_argument("--mosaic", type=float, help="Mosaic probability. Defaults to the selected preset.")
    parser.add_argument(
        "--close-mosaic", type=int, help="Final epochs without mosaic. Defaults to the selected preset."
    )
    parser.add_argument("--degrees", type=float, help="Rotation range. Defaults to the selected preset.")
    parser.add_argument("--translate", type=float, help="Translation range. Defaults to the selected preset.")
    parser.add_argument("--scale", type=float, help="Scale range. Defaults to the selected preset.")
    parser.add_argument("--fliplr", type=float, help="Horizontal flip probability. Defaults to the framework setting.")
    parser.add_argument("--hsv-h", type=float, help="HSV hue augmentation. Defaults to the framework setting.")
    parser.add_argument("--hsv-s", type=float, help="HSV saturation augmentation. Defaults to the framework setting.")
    parser.add_argument("--hsv-v", type=float, help="HSV value augmentation. Defaults to the framework setting.")
    parser.add_argument(
        "--cls-pw", type=float, help="Inverse-frequency class-weight power. Defaults to the selected preset."
    )
    parser.add_argument("--fl-gamma", type=float, help="Focal classification loss gamma. Zero disables it.")
    parser.add_argument("--distill-model", type=Path, help="Optional teacher checkpoint for feature distillation.")
    parser.add_argument("--dis", type=float, default=6.0, help="Feature-distillation loss gain.")
    return parser.parse_args()


def main():
    args = parse_args()
    preset = PRESETS[args.preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.model = Path(args.model)
    args.data = Path(args.data)
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset descriptor not found: {args.data}")
    if args.weights is not None and not args.weights.is_file():
        raise FileNotFoundError(f"Initialization weights not found: {args.weights}")
    cuda = torch.cuda.is_available()
    device = args.device if args.device is not None else (0 if cuda else "cpu")
    using_cuda = cuda and str(device).lower() != "cpu"
    batch = args.batch if using_cuda else 1  # Automatic batching is CUDA-only.
    workers = 4 if using_cuda else 0
    device_name = torch.cuda.get_device_name(int(str(device).split(",")[0])) if using_cuda else "CPU"
    print(f"Device: {device} ({device_name})")
    print(f"Preset: {args.preset}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Focal gamma: {args.fl_gamma if args.fl_gamma is not None else 0.0}")

    model = build_model(args)
    model.train(
        data=args.data.resolve(),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=batch,
        device=device,
        workers=workers,
        optimizer=args.optimizer or "AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum if args.momentum is not None else 0.937,
        weight_decay=args.weight_decay if args.weight_decay is not None else 0.0005,
        warmup_epochs=args.warmup_epochs if args.warmup_epochs is not None else 3.0,
        cos_lr=args.cos_lr if args.cos_lr is not None else True,
        box=args.box if args.box is not None else 7.5,
        cls=args.cls if args.cls is not None else 0.5,
        dfl=args.dfl if args.dfl is not None else 1.5,
        elongation_gain=args.elongation_gain if args.elongation_gain is not None else 0.0,
        elongation_threshold=args.elongation_threshold if args.elongation_threshold is not None else 4.0,
        tal_topk=args.tal_topk if args.tal_topk is not None else 10,
        patience=args.patience,
        save_period=args.save_period if args.save_period is not None else -1,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        cls_pw=args.cls_pw,
        fl_gamma=args.fl_gamma if args.fl_gamma is not None else 0.0,
        distill_model=args.distill_model.resolve() if args.distill_model is not None else None,
        dis=args.dis,
        fliplr=args.fliplr if args.fliplr is not None else 0.5,
        hsv_h=args.hsv_h if args.hsv_h is not None else 0.015,
        hsv_s=args.hsv_s if args.hsv_s is not None else 0.7,
        hsv_v=args.hsv_v if args.hsv_v is not None else 0.4,
        mixup=0.0,
        erasing=0.0,
        seed=args.seed,
        deterministic=True,
        resume=False,
        project=ROOT / "runs" / "train",
        name=args.name,
    )


if __name__ == "__main__":
    main()
