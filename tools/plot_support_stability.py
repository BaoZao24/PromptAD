#!/usr/bin/env python3
"""Plot support-repetition means with 95% confidence intervals."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("ViT-only", "CNN-only", "Ungated OR", "Ours")
METHOD_LABELS = {
    "ViT-only": "ViT-only (this work)",
    "CNN-only": "CNN-only (this work)",
    "Ungated OR": "Ungated OR (this work)",
    "Ours": "Confidence-gated OR (ours)",
}
COLORS = {
    "ViT-only": "#4C78A8",
    "CNN-only": "#F58518",
    "Ungated OR": "#7F7F7F",
    "Ours": "#54A24B",
}
DATASET_LABELS = {
    "self_rf": "In-house RF",
    "public_rf": "Public RF",
    "ofdma": "OFDMA",
}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_signal_summary(rows: list[dict], output: Path, metric: str) -> None:
    rows = [row for row in rows if row["metric"] == metric]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["signal"])].append(row)
    if not groups:
        return
    datasets = ["self_rf", "public_rf"]
    signals = sorted({signal for _, signal in groups})
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 3.5), squeeze=False)
    x = np.arange(len(signals), dtype=float)
    for axis, dataset in zip(axes.flat, datasets):
        for method in METHODS:
            means, lows, highs = [], [], []
            for signal in signals:
                row = next((item for item in groups.get((dataset, signal), []) if item["method"] == method), None)
                if row is None:
                    means.append(np.nan); lows.append(np.nan); highs.append(np.nan)
                else:
                    means.append(float(row["mean"]))
                    lows.append(float(row["ci95_low"]))
                    highs.append(float(row["ci95_high"]))
            means = np.asarray(means)
            lows = np.asarray(lows)
            highs = np.asarray(highs)
            axis.errorbar(
                x,
                means,
                yerr=np.vstack([means - lows, highs - means]),
                marker="o",
                linewidth=1.3,
                markersize=4.5,
                capsize=2.5,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_xticks(x, [signal.replace("_", " ").title() for signal in signals], rotation=22, ha="right")
        axis.set_title(DATASET_LABELS.get(dataset, dataset))
        ylabel = "FPR@95%TPR" if metric == "fpr95" else metric.upper()
        axis.set_ylabel(ylabel + " (%)")
        axis.grid(axis="y", alpha=0.25, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fusion_summary(rows: list[dict], output: Path, metric: str) -> None:
    """Plot the branch/fusion ablation with repeated-support 95% CIs."""

    rows = [
        row for row in rows
        if row["metric"] == metric and row["method"] in
        {"ViT-only", "CNN-only", "Ungated OR", "Ours"}
    ]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["shot"])].append(row)
    if not groups:
        return
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.4), squeeze=False)
    methods = ("ViT-only", "CNN-only", "Ungated OR", "Ours")
    for axis, ((dataset, shot), values) in zip(axes.flat, sorted(groups.items())):
        by_method = {row["method"]: row for row in values}
        available = [method for method in methods if method in by_method]
        x = np.arange(len(available), dtype=float)
        means = np.asarray([float(by_method[method]["mean"]) for method in available])
        low = np.asarray([float(by_method[method]["ci95_low"]) for method in available])
        high = np.asarray([float(by_method[method]["ci95_high"]) for method in available])
        axis.errorbar(
            x,
            means,
            yerr=np.vstack([means - low, high - means]),
            fmt="none",
            ecolor="#444444",
            capsize=4,
            linewidth=1.2,
        )
        for xpos, method, mean in zip(x, available, means):
            axis.scatter([xpos], [mean], s=45, color=COLORS[method], zorder=3, label=METHOD_LABELS[method])
        axis.set_xticks(x, [METHOD_LABELS[method] for method in available], rotation=20, ha="right")
        axis.set_title(DATASET_LABELS.get(dataset, dataset) + (f" ({shot}-shot)" if shot != "all" else ""))
        ylabel = "FPR@95%TPR" if metric == "fpr95" else metric.upper()
        axis.set_ylabel(ylabel + " (%)")
        axis.grid(axis="y", alpha=0.25, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(handles), loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_jammer_summary(rows: list[dict], output: Path, metric: str) -> None:
    """Plot repeated-support OFDMA jammer metrics with replicate error bars."""

    rows = [row for row in rows if row["metric"] == metric and row["shot"] == "4"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["jammer"]].append(row)
    if not groups:
        return
    methods = [method for method in METHODS if any(row["method"] == method for row in rows)]
    jammers = sorted(groups)
    x = np.arange(len(jammers), dtype=float)
    width = 0.8 / max(1, len(methods))
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axis = plt.subplots(figsize=(6.4, 3.4))
    for index, method in enumerate(methods):
        values = []
        errors_low = []
        errors_high = []
        for jammer in jammers:
            row = next(item for item in groups[jammer] if item["method"] == method)
            mean = float(row["mean"])
            values.append(mean)
            # Use the same 95% normal CI convention as the main stability
            # figures; asymmetric errors are possible for small replicate sets.
            errors_low.append(mean - float(row["ci95_low"]))
            errors_high.append(float(row["ci95_high"]) - mean)
        positions = x + (index - (len(methods) - 1) / 2.0) * width
        axis.bar(
            positions,
            values,
            width=width * 0.88,
            yerr=np.vstack([errors_low, errors_high]),
            capsize=2.2,
            color=COLORS[method],
            edgecolor="none",
            label=METHOD_LABELS[method],
        )
    axis.set_xticks(x, [jammer.replace("_", " ").title() for jammer in jammers], rotation=18, ha="right")
    axis.set_ylabel("FPR@95%TPR (%)" if metric == "fpr95" else metric.upper() + " (%)")
    axis.set_title("OFDMA 4-shot jammer breakdown")
    axis.grid(axis="y", alpha=0.25, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, ncol=len(methods), loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ofdma_shot_scaling(
    rows: list[dict],
    output: Path,
    metric: str,
    scene_rows: list[dict] | None = None,
) -> None:
    """Plot repeated-support OFDMA shot scaling with scene-bootstrap intervals."""

    rows = [row for row in rows if row["dataset"] == "ofdma" and row["metric"] == metric]
    if not rows:
        return
    scene_rows = scene_rows or []
    scene_rows = [row for row in scene_rows if row["shot"] in {"1", "2", "4"} and row["metric"] == metric]
    scene_map = {
        (row["shot"], row["method"]): row
        for row in scene_rows
    }
    methods = [method for method in METHODS if any(row["method"] == method for row in rows)]
    shots = ["1", "2", "4"]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axis = plt.subplots(figsize=(5.6, 3.4))
    for method in methods:
        means, lows, highs = [], [], []
        for shot in shots:
            row = next(item for item in rows if item["shot"] == shot and item["method"] == method)
            scene = scene_map.get((shot, method))
            if scene is None:
                means.append(float(row["mean"]))
                lows.append(float(row["ci95_low"]))
                highs.append(float(row["ci95_high"]))
            else:
                means.append(float(scene["scene_macro_mean"]))
                lows.append(float(scene["scene_bootstrap_ci95_low"]))
                highs.append(float(scene["scene_bootstrap_ci95_high"]))
        means = np.asarray(means)
        yerr = np.vstack([means - np.asarray(lows), np.asarray(highs) - means])
        axis.errorbar(
            np.arange(len(shots)),
            means,
            yerr=yerr,
            marker="o",
            markersize=4.8,
            linewidth=1.35 if method == "Ours" else 1.0,
            capsize=3,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.set_xticks(np.arange(len(shots)), [f"{shot}-shot" for shot in shots])
    axis.set_ylabel("FPR@95%TPR (%)" if metric == "fpr95" else metric.upper() + " (%)")
    axis.set_title("OFDMA support stability across shots")
    axis.grid(axis="y", alpha=0.25, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, ncol=min(len(methods), 4), loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metric", choices=["auroc", "auprc", "fpr95"], default="auroc")
    parser.add_argument("--signal-summary-csv", type=Path, default=None)
    parser.add_argument("--jammer-summary-csv", type=Path, default=None)
    parser.add_argument("--ofdma-scene-summary-csv", type=Path, default=None)
    args = parser.parse_args()

    rows = [row for row in read_rows(args.summary_csv) if row["metric"] == args.metric]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["shot"])].append(row)
    if not groups:
        raise RuntimeError("No matching summary rows")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.4), squeeze=False)
    for axis, ((dataset, shot), values) in zip(axes.flat, sorted(groups.items())):
        by_method = {row["method"]: row for row in values}
        methods = [method for method in METHODS if method in by_method]
        x = np.arange(len(methods), dtype=float)
        means = np.asarray([float(by_method[m]["mean"]) for m in methods])
        low = np.asarray([float(by_method[m]["ci95_low"]) for m in methods])
        high = np.asarray([float(by_method[m]["ci95_high"]) for m in methods])
        yerr = np.vstack([means - low, high - means])
        axis.errorbar(
            x,
            means,
            yerr=yerr,
            fmt="o",
            capsize=4,
            linewidth=1.4,
            markersize=6,
            color="#333333",
        )
        for xpos, method, mean in zip(x, methods, means):
            axis.scatter([xpos], [mean], s=42, color=COLORS[method], zorder=3)
        axis.set_xticks(x, [METHOD_LABELS[method] for method in methods], rotation=20, ha="right")
        title = DATASET_LABELS.get(dataset, dataset)
        axis.set_title(title + (f" ({shot}-shot)" if shot != "all" else ""))
        ylabel = "FPR@95%TPR" if args.metric == "fpr95" else args.metric.upper()
        axis.set_ylabel(ylabel + " (%)")
        axis.grid(axis="y", alpha=0.25, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    args.output_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_root / f"support_stability_{args.metric}.pdf", bbox_inches="tight")
    fig.savefig(args.output_root / f"support_stability_{args.metric}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    if args.signal_summary_csv is not None:
        plot_signal_summary(
            read_rows(args.signal_summary_csv),
            args.output_root / f"support_stability_signal_{args.metric}",
            args.metric,
        )
    plot_fusion_summary(
        read_rows(args.summary_csv),
        args.output_root / f"support_stability_fusion_{args.metric}",
        args.metric,
    )
    if args.jammer_summary_csv is not None:
        plot_jammer_summary(
            read_rows(args.jammer_summary_csv),
            args.output_root / f"support_stability_ofdma_jammer_{args.metric}",
            args.metric,
        )
    if args.ofdma_scene_summary_csv is not None:
        plot_ofdma_shot_scaling(
            rows,
            args.output_root / f"support_stability_ofdma_shots_{args.metric}",
            args.metric,
            read_rows(args.ofdma_scene_summary_csv),
        )


if __name__ == "__main__":
    main()
