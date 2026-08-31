#!/usr/bin/env python3
"""Draw the four figures used by the current scheme overview.

The values below mirror the formal tables in
``docs/paper/现有方案介绍.md``.  This small plotting entry point keeps the
overview figures aligned with the selected main baselines and intentionally
does not add a second, grey explanatory caption below the legend.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_paper_experiment_figures import configure_axes, save_figure


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "analysis_outputs/20260810_paper_experiment_figures_cited"

METHODS = (
    "ed",
    "scse",
    "iad_per",
    "udma",
    "gretel",
    "tfam",
    "patchcore",
    "spade",
    "ours",
)
LABELS = {
    "ed": "ED",
    "scse": "SCSE",
    "iad_per": "IAD-PER",
    "udma": "UDMA-ResNet18",
    "gretel": "GRETEL",
    "tfam": "TFAM-AAE",
    "patchcore": "PatchCore",
    "spade": "SPADE",
    "ours": "SpectraMemAD",
}
COLORS = {
    "ed": "#8C6D31",
    "scse": "#B22222",
    "iad_per": "#4E79A7",
    "udma": "#B07AA1",
    "gretel": "#6F6F6F",
    "tfam": "#59A14F",
    "patchcore": "#006D77",
    "spade": "#7F3C8D",
    "ours": "#D55E00",
}
MARKERS = {
    "ed": "v",
    "scse": "*",
    "iad_per": "D",
    "udma": "8",
    "gretel": "^",
    "tfam": "P",
    "patchcore": "s",
    "spade": "X",
    "ours": "o",
}


def metric(auroc: float, auprc: float, fpr95: float) -> dict[str, float]:
    return {"auroc": auroc, "auprc": auprc, "fpr95": fpr95}


def scaling(
    auroc: list[float],
    auprc: list[float],
    fpr95: list[float],
) -> dict[str, list[float]]:
    return {"auroc": auroc, "auprc": auprc, "fpr95": fpr95}


INHOUSE = {
    "ed": metric(44.47, 29.00, 91.63),
    "scse": metric(74.89, 56.99, 65.63),
    "iad_per": metric(53.00, 35.72, 79.48),
    "udma": metric(57.95, 38.77, 75.76),
    "gretel": metric(63.96, 43.80, 68.61),
    "tfam": metric(59.50, 40.16, 73.60),
    "patchcore": metric(84.38, 70.17, 55.94),
    "spade": metric(71.32, 53.16, 66.73),
    "ours": metric(91.97, 81.80, 22.00),
}

PUBLIC_RF = {
    "ed": scaling([49.39, 49.39, 49.39], [11.27, 11.27, 11.27], [96.92, 96.92, 96.92]),
    "scse": scaling([60.05, 61.16, 61.34], [9.06, 9.23, 9.30], [85.05, 82.11, 83.58]),
    "iad_per": scaling([52.89, 52.70, 51.80], [9.11, 8.37, 8.08], [90.80, 91.26, 91.46]),
    "udma": scaling([52.15, 52.20, 51.91], [9.06, 9.03, 9.56], [92.52, 92.73, 91.99]),
    "gretel": scaling([57.30, 57.89, 57.27], [12.78, 13.06, 13.72], [88.90, 88.46, 87.57]),
    "tfam": scaling([54.31, 53.70, 54.73], [9.15, 7.94, 7.55], [89.03, 88.22, 88.42]),
    "patchcore": scaling([70.83, 74.17, 78.70], [25.13, 30.50, 33.18], [66.90, 62.85, 56.78]),
    "spade": scaling([66.07, 65.78, 67.35], [16.72, 17.45, 17.96], [77.71, 75.72, 77.64]),
    "ours": scaling([81.74, 82.97, 83.74], [44.43, 44.43, 45.39], [45.69, 44.27, 44.12]),
}

OFDMA = {
    "ed": scaling([62.92, 62.92, 62.92], [68.83, 68.83, 68.83], [90.60, 90.60, 90.60]),
    "scse": scaling([69.70, 68.76, 68.46], [74.77, 72.85, 72.41], [87.20, 86.77, 86.93]),
    "iad_per": scaling([53.08, 53.89, 55.48], [56.44, 57.53, 59.23], [93.83, 93.20, 92.53]),
    "udma": scaling([57.77, 57.43, 57.69], [62.85, 62.78, 63.23], [92.50, 92.90, 93.23]),
    "gretel": scaling([63.58, 63.92, 63.39], [69.71, 69.91, 69.60], [90.81, 90.89, 90.78]),
    "tfam": scaling([57.99, 57.74, 57.60], [79.97, 79.84, 79.65], [93.70, 93.70, 94.20]),
    "patchcore": scaling([70.14, 75.83, 81.81], [76.78, 81.29, 86.14], [86.03, 81.37, 72.50]),
    "spade": scaling([56.70, 58.50, 58.33], [76.80, 78.37, 77.72], [92.50, 92.16, 92.26]),
    "ours": scaling([85.30, 89.53, 92.75], [88.80, 92.01, 94.55], [69.77, 57.63, 45.90]),
}

FEDJAM = {
    "ed": scaling([64.93, 64.93, 64.93], [84.64, 84.64, 84.64], [85.72, 85.72, 85.72]),
    "scse": scaling([63.51, 68.79, 76.52], [84.96, 87.38, 90.30], [100.00, 86.89, 74.89]),
    "iad_per": scaling([48.77, 42.77, 50.51], [73.24, 69.87, 74.35], [94.22, 96.61, 94.50]),
    "udma": scaling([50.45, 46.68, 47.68], [73.16, 71.29, 71.78], [90.28, 92.33, 91.61]),
    "gretel": scaling([45.66, 43.14, 42.28], [71.78, 70.55, 70.33], [95.46, 97.04, 96.76]),
    "tfam": scaling([46.43, 43.90, 41.83], [72.35, 70.78, 69.93], [94.67, 94.33, 96.00]),
    "patchcore": scaling([81.27, 83.71, 83.54], [93.45, 94.12, 94.13], [74.22, 64.06, 67.33]),
    "spade": scaling([65.71, 62.88, 64.68], [86.65, 84.15, 85.23], [88.89, 88.72, 86.50]),
    "ours": scaling([83.71, 84.48, 84.80], [94.62, 94.76, 94.75], [78.50, 73.22, 68.83]),
}


def plot_main_auroc(data: dict[str, dict[str, float]], output: Path) -> None:
    methods = list(reversed(METHODS))
    fig, ax = plt.subplots(figsize=(6.25, 3.85))
    y = np.arange(len(methods))
    for method, y_pos in zip(methods, y):
        value = data[method]["auroc"]
        highlight = method in {"patchcore", "spade", "ours"}
        ax.scatter(
            value,
            y_pos,
            s=50 if method == "ours" else (42 if highlight else 31),
            marker=MARKERS[method],
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax.text(value + 0.55, y_pos, f"{value:.2f}", va="center", fontsize=8)
    ax.set_yticks(y, [LABELS[method] for method in methods])
    ax.invert_yaxis()
    ax.set_xlim(40, 100)
    ax.set_xticks(np.arange(40, 101, 10))
    ax.set_xlabel("Macro-averaged AUROC (%)")
    ax.set_title("(a) In-house RF", loc="left", fontweight="semibold")
    configure_axes(ax, grid_axis="x")
    fig.subplots_adjust(left=0.29, right=0.98, bottom=0.13, top=0.92)
    save_figure(fig, output)


def plot_scaling(
    data: dict[str, dict[str, list[float]]],
    dataset_title: str,
    output: Path,
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> None:
    shots = [1, 2, 4]
    metric_info = (
        ("auroc", "AUROC (%) ↑"),
        ("auprc", "AUPRC (%) ↑"),
        ("fpr95", "FPR@95%TPR (%) ↓"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.15), sharex=True)
    for index, (ax, (metric_name, ylabel), ylim) in enumerate(zip(axes, metric_info, limits)):
        for method in METHODS:
            highlight = method == "ours"
            ax.plot(
                shots,
                data[method][metric_name],
                marker=MARKERS[method],
                color=COLORS[method],
                linewidth=2.2 if highlight else (1.65 if method in {"patchcore", "spade"} else 1.1),
                markersize=5.4 if highlight else 4.0,
                alpha=1.0 if highlight or method in {"patchcore", "spade"} else 0.82,
                label=LABELS[method],
            )
        ax.set_xticks(shots, ["1-shot", "2-shot", "4-shot"])
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(f"({'abc'[index]}) {dataset_title}", loc="left", fontweight="semibold")
        configure_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=3,
        columnspacing=1.05,
        handlelength=1.8,
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.29, top=0.89, wspace=0.30)
    save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    plot_main_auroc(INHOUSE, output / "fig_main_auroc_inhouse_rf.png")
    plot_scaling(
        PUBLIC_RF,
        "Public RF",
        output / "fig_public_rf_k_scaling.png",
        ((45, 90), (0, 50), (35, 100)),
    )
    plot_scaling(
        OFDMA,
        "OFDMA",
        output / "fig_ofdma_shot_scaling.png",
        ((40, 100), (45, 100), (25, 100)),
    )
    plot_scaling(
        FEDJAM,
        "FedJam",
        output / "fig_fedjam_shot_scaling.png",
        ((25, 90), (60, 100), (25, 100)),
    )
    print(f"Wrote current scheme figures to {output}")


if __name__ == "__main__":
    main()
