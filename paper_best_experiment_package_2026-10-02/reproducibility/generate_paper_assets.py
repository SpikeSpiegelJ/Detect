from __future__ import annotations

import csv
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[1]
TABLES = PACKAGE / "tables"
FIGURES = PACKAGE / "figures"
SOURCES = PACKAGE / "source_records"

COLORS = {
    "blue": "#2F5597",
    "orange": "#ED7D31",
    "green": "#70AD47",
    "red": "#C00000",
    "gray": "#7F7F7F",
    "light": "#D9E2F3",
}

BEST_TEST = [
    ["all", 535, 1178, 0.855, 0.840, 0.8841, 0.5202],
    ["coupler", 84, 109, 0.873, 0.817, 0.916, 0.505],
    ["antenna_s", 56, 71, 0.860, 0.777, 0.812, 0.371],
    ["RRU", 192, 208, 0.976, 0.960, 0.989, 0.684],
    ["Feeder_RRU", 150, 161, 0.838, 0.789, 0.854, 0.373],
    ["antenna_b", 219, 263, 0.919, 0.953, 0.958, 0.553],
    ["cut", 105, 121, 0.777, 0.702, 0.802, 0.493],
    ["Feeder_antenna", 120, 122, 0.835, 0.706, 0.781, 0.464],
    ["POWER", 67, 68, 0.849, 0.910, 0.906, 0.564],
    ["BOX", 51, 55, 0.771, 0.945, 0.939, 0.674],
]

TEST_ABLATION = [
    ["YOLO26s v15 reviewed (single model)", 0.873, 0.822, 0.8775, 0.5100, 46.1],
    ["YOLO26s v9 + YOLO26m v9 (general ensemble)", None, None, 0.8831, 0.5192, None],
    ["General ensemble + cut/Feeder_antenna specialist (proposed)", 0.855, 0.840, 0.8841, 0.5202, 200.7],
]

VAL_GENERAL = [
    ["all", 0.909, 0.912, 0.9487, 0.5951],
    ["coupler", 0.947, 0.949, 0.978, 0.585],
    ["antenna_s", 0.927, 0.953, 0.960, 0.485],
    ["RRU", 0.966, 0.987, 0.992, 0.714],
    ["Feeder_RRU", 0.884, 0.861, 0.911, 0.468],
    ["antenna_b", 0.926, 0.977, 0.979, 0.657],
    ["cut", 0.892, 0.886, 0.922, 0.631],
    ["Feeder_antenna", 0.864, 0.703, 0.849, 0.490],
    ["POWER", 0.896, 0.972, 0.980, 0.685],
    ["BOX", 0.877, 0.917, 0.968, 0.640],
]

VAL_PROPOSED = [
    ["all", 0.920, 0.905, 0.9508, 0.5992],
    ["coupler", 0.958, 0.943, 0.978, 0.585],
    ["antenna_s", 0.931, 0.951, 0.960, 0.485],
    ["RRU", 0.966, 0.986, 0.992, 0.714],
    ["Feeder_RRU", 0.902, 0.837, 0.911, 0.468],
    ["antenna_b", 0.926, 0.971, 0.979, 0.657],
    ["cut", 0.895, 0.889, 0.924, 0.645],
    ["Feeder_antenna", 0.877, 0.700, 0.866, 0.513],
    ["POWER", 0.920, 0.963, 0.980, 0.685],
    ["BOX", 0.909, 0.908, 0.968, 0.640],
]


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def overall_comparison() -> None:
    labels = ["YOLO26s v15", "S+M ensemble", "Proposed fusion"]
    map50 = [row[3] for row in TEST_ABLATION]
    map5095 = [row[4] for row in TEST_ABLATION]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars1 = ax.bar(x - width / 2, map50, width, label="mAP@0.50", color=COLORS["blue"])
    bars2 = ax.bar(x + width / 2, map5095, width, label="mAP@0.50:0.95", color=COLORS["orange"])
    ax.set_ylabel("Average precision")
    ax.set_title("Independent test performance")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", ncols=2)
    ax.bar_label(bars1, fmt="%.4f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.4f", padding=3, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "Fig1_test_overall_comparison")


def per_class_performance() -> None:
    rows = BEST_TEST[1:]
    labels = [row[0] for row in rows][::-1]
    map50 = np.array([row[5] for row in rows][::-1])
    map5095 = np.array([row[6] for row in rows][::-1])
    y = np.arange(len(labels))
    height = 0.35
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.barh(y + height / 2, map50, height, label="mAP@0.50", color=COLORS["blue"])
    ax.barh(y - height / 2, map5095, height, label="mAP@0.50:0.95", color=COLORS["orange"])
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Average precision")
    ax.set_title("Per-class performance of the proposed fusion on the independent test set")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=2)
    fig.tight_layout()
    save_figure(fig, "Fig2_test_per_class_ap")


def precision_recall() -> None:
    rows = BEST_TEST[1:]
    labels = [row[0] for row in rows]
    precision = np.array([row[3] for row in rows])
    recall = np.array([row[4] for row in rows])
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - width / 2, precision, width, label="Precision", color=COLORS["blue"])
    ax.bar(x + width / 2, recall, width, label="Recall", color=COLORS["green"])
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Precision and recall by class on the independent test set")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "Fig3_test_precision_recall")


def specialist_ablation() -> None:
    general = {row[0]: row for row in VAL_GENERAL}
    proposed = {row[0]: row for row in VAL_PROPOSED}
    labels = ["Overall", "cut", "Feeder_antenna"]
    keys = ["all", "cut", "Feeder_antenna"]
    delta50 = [(proposed[key][3] - general[key][3]) * 100 for key in keys]
    delta95 = [(proposed[key][4] - general[key][4]) * 100 for key in keys]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars1 = ax.bar(x - width / 2, delta50, width, label="mAP@0.50", color=COLORS["blue"])
    bars2 = ax.bar(x + width / 2, delta95, width, label="mAP@0.50:0.95", color=COLORS["orange"])
    ax.axhline(0, color="#404040", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Absolute improvement (percentage points)")
    ax.set_title("Contribution of the weak-class specialist on the development validation set")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.bar_label(bars1, fmt="%+.2f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%+.2f", padding=3, fontsize=9)
    fig.tight_layout()
    save_figure(fig, "Fig4_specialist_ablation_val")


def generalization_gap() -> None:
    labels = ["Development validation", "Independent test"]
    map50 = [0.9508, 0.8841]
    map5095 = [0.5992, 0.5202]
    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars1 = ax.bar(x - width / 2, map50, width, label="mAP@0.50", color=COLORS["blue"])
    bars2 = ax.bar(x + width / 2, map5095, width, label="mAP@0.50:0.95", color=COLORS["orange"])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Average precision")
    ax.set_title("Generalization gap of the proposed fusion")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", ncols=2)
    ax.bar_label(bars1, fmt="%.4f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.4f", padding=3, fontsize=9)
    fig.tight_layout()
    save_figure(fig, "Fig5_generalization_gap")


def method_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        patch = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#404040", linewidth=1.0)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#404040"})

    box(0.25, 2.0, 1.35, 1.0, "Input image\n1280 px", "#F2F2F2")
    box(2.2, 3.25, 1.65, 1.0, "YOLO26s\ngeneralist + TTA", "#D9EAF7")
    box(2.2, 1.85, 1.65, 1.0, "YOLO26m\ngeneralist + TTA", "#D9EAF7")
    box(2.2, 0.45, 1.65, 1.0, "YOLO26s specialist\nsingle scale", "#FCE4D6")
    box(4.65, 2.55, 1.7, 1.0, "General prediction\nconcatenation", "#E2F0D9")
    box(4.65, 0.45, 1.7, 1.0, "2-to-9 class map\ncut / Feeder_antenna", "#FFF2CC")
    box(7.15, 1.5, 1.7, 1.0, "Prediction-level\nfusion", "#E4DFEC")
    box(9.55, 1.5, 1.55, 1.0, "Class-aware NMS\nIoU = 0.50", "#DDEBF7")
    box(11.35, 1.5, 0.5, 1.0, "9-class\noutput", "#E2F0D9")
    for y in (3.75, 2.35, 0.95):
        arrow(1.6, 2.5, 2.2, y)
    arrow(3.85, 3.75, 4.65, 3.05)
    arrow(3.85, 2.35, 4.65, 3.05)
    arrow(3.85, 0.95, 4.65, 0.95)
    arrow(6.35, 3.05, 7.15, 2.0)
    arrow(6.35, 0.95, 7.15, 2.0)
    arrow(8.85, 2.0, 9.55, 2.0)
    arrow(11.1, 2.0, 11.35, 2.0)
    ax.set_title("Generalist-specialist prediction fusion", fontsize=13, pad=10)
    fig.tight_layout()
    save_figure(fig, "Fig6_method_pipeline")


def copy_source_records() -> None:
    mappings = {
        ROOT / "val_specialist_fusion.py": SOURCES / "val_specialist_fusion.py",
        ROOT / "dataset" / "prepare_v14_cut_feeder_specialist.py": SOURCES / "prepare_v14_cut_feeder_specialist.py",
        ROOT / "dataset" / "data_repartition_v14_cut_feeder_specialist.yaml": SOURCES / "data_repartition_v14_cut_feeder_specialist.yaml",
        ROOT / "dataset" / "repartition_v14_cut_feeder_specialist" / "manifest.json": SOURCES / "v14_specialist_manifest.json",
        ROOT / "dataset" / "repartition_v13_cut_feeder_hard_corrected" / "review_corrections.json": SOURCES / "v13_review_corrections.json",
        ROOT / "runs" / "train" / "yolo26s_v14_cut_feeder_specialist-2" / "args.yaml": SOURCES / "specialist_train_args.yaml",
        ROOT / "runs" / "train" / "yolo26s_v9_trainval_o2m_dfl15_ft" / "args.yaml": SOURCES / "yolo26s_general_train_args.yaml",
        ROOT / "runs" / "train" / "yolo26m_v9_trainval_ft" / "args.yaml": SOURCES / "yolo26m_general_train_args.yaml",
    }
    best_run = ROOT / "runs" / "val" / "v14_specialist_fusion_w1p0_tta_general_single_specialist_iou0p5_test"
    for name in (
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxPR_curve.png",
        "BoxR_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
    ):
        mappings[best_run / name] = FIGURES / f"BestFusion_{name}"
    for source, target in mappings.items():
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def main() -> None:
    for directory in (TABLES, FIGURES, SOURCES):
        directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.unicode_minus": False})

    write_csv(TABLES / "Table1_best_test_overall.csv", ["method", "images", "instances", "precision", "recall", "mAP50", "mAP50-95", "inference_ms_per_image"], [["Proposed generalist-specialist fusion", 535, 1178, 0.855, 0.840, 0.8841, 0.5202, 200.7]])
    write_csv(TABLES / "Table2_best_test_per_class.csv", ["class", "images", "instances", "precision", "recall", "mAP50", "mAP50-95"], BEST_TEST)
    write_csv(TABLES / "Table3_test_ablation.csv", ["method", "precision", "recall", "mAP50", "mAP50-95", "inference_ms_per_image"], TEST_ABLATION)
    comparison = []
    for general, proposed in zip(VAL_GENERAL, VAL_PROPOSED, strict=True):
        comparison.append([general[0], *general[1:], *proposed[1:], proposed[3] - general[3], proposed[4] - general[4]])
    write_csv(TABLES / "Table4_specialist_ablation_val.csv", ["class", "general_P", "general_R", "general_mAP50", "general_mAP50-95", "proposed_P", "proposed_R", "proposed_mAP50", "proposed_mAP50-95", "delta_mAP50", "delta_mAP50-95"], comparison)
    write_csv(TABLES / "Table5_protocol.csv", ["item", "value"], [
        ["general models", "YOLO26s v9 + YOLO26m v9"],
        ["specialist", "YOLO26s two-class specialist: cut and Feeder_antenna"],
        ["general inference", "TTA"],
        ["specialist inference", "single scale"],
        ["input size", 1280],
        ["batch size", 2],
        ["NMS IoU", 0.50],
        ["specialist weight", 1.0],
        ["test images", 535],
        ["test instances", 1178],
        ["hardware", "NVIDIA GeForce RTX 3060 12 GB"],
        ["software", "Ultralytics 8.4.104; Python 3.13.14; torch 2.12.1+cu126"],
    ])

    overall_comparison()
    per_class_performance()
    precision_recall()
    specialist_ablation()
    generalization_gap()
    method_pipeline()
    copy_source_records()


if __name__ == "__main__":
    main()
