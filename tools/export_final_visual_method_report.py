#!/usr/bin/env python
"""Export final pure-visual method metrics, ROC curves, and ablation figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


METHOD = "confidence_gated_dual_visual"


def metric_row(labels, scores):
    labels, scores = np.asarray(labels, dtype=np.int32), np.asarray(scores, dtype=np.float64)
    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        "image_auroc": float(roc_auc_score(labels, scores) * 100.0),
        "image_auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr_at_95_tpr": float(fpr[np.flatnonzero(tpr >= 0.95)[0]] * 100.0),
    }


def parse_cell(path: Path, dataset: str):
    stem = path.stem.removesuffix("-scores")
    parts = stem.split("-")
    if dataset == "self_rf":
        return parts[0], parts[-2], parts[-1]
    return parts[0], "RF_SPE_PNG_public", parts[-1]


def load_dataset(score_dir: Path, dataset: str):
    cells = []
    self_prefixes = ("burst_signal-", "chirp_signal-", "dsss_signal-", "pulse_signal-", "wideband_pulse-")
    public_prefixes = ("burst-", "chirp-", "dsss-", "pulse-", "wideband_pulse-")
    for path in sorted(score_dir.glob("*-scores.npz")):
        is_self_cell = path.stem.startswith(self_prefixes)
        is_public_cell = path.stem.startswith(public_prefixes)
        if dataset == "self_rf" and not is_self_cell:
            continue
        if dataset == "public_rf" and not is_public_cell:
            continue
        data = np.load(path, allow_pickle=True)
        labels = data["labels"].astype(np.int32)
        required = ("vit_score", "cnn_score", "confidence_gated_score")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise RuntimeError(
                f"{path} is not a confidence-gate output; missing={missing}"
            )
        vit = data["vit_score"].astype(np.float64)
        cnn = data["cnn_score"].astype(np.float64)
        final = data["confidence_gated_score"].astype(np.float64)
        signal, scene, jsr = parse_cell(path, dataset)
        cells.append({
            "dataset": dataset,
            "signal": signal,
            "scene": scene,
            "jsr": jsr,
            "labels": labels,
            "vit": vit,
            "cnn": cnn,
            "final": final,
            "source": path,
        })
    return cells


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_roc(cells, dataset, output):
    plt.figure(figsize=(8, 6), dpi=180)
    by_signal = {}
    for cell in cells:
        by_signal.setdefault(cell["signal"], []).append(cell)
    all_labels, all_scores = [], []
    for signal, items in sorted(by_signal.items()):
        labels = np.concatenate([item["labels"] for item in items])
        scores = np.concatenate([item["final"] for item in items])
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores) * 100.0
        plt.plot(fpr, tpr, linewidth=2, label=f"{signal} ({auc:.2f})")
        all_labels.append(labels)
        all_scores.append(scores)
    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.plot(fpr, tpr, "k--", linewidth=2.5, label=f"pooled ({roc_auc_score(labels, scores) * 100.0:.2f})")
    plt.plot([0, 1], [0, 1], color="#94a3b8", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{dataset}: final visual method ROC")
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def plot_ablation(ablation_rows, output_root):
    datasets = ["self_rf", "public_rf"]
    labels = ["ViT only", "CNN only", "ViT + gated CNN"]
    keys = ["vit_only", "cnn_only", "full"]
    x = np.arange(len(labels))
    width = 0.34
    values = {row["dataset"]: row for row in ablation_rows}
    plt.figure(figsize=(8.5, 5), dpi=180)
    for index, dataset in enumerate(datasets):
        scores = [values[dataset][key] for key in keys]
        plt.bar(x + (index - 0.5) * width, scores, width, label=dataset)
    plt.xticks(x, labels)
    plt.ylabel("Macro Image-AUROC")
    plt.ylim(65, 100)
    plt.legend(frameon=False)
    plt.title("Component ablation")
    plt.tight_layout()
    plt.savefig(output_root / "component_ablation.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6.5, 5), dpi=180)
    labels = ["No TTA", "Paired TTA"]
    x = np.arange(2)
    for index, dataset in enumerate(datasets):
        row = values[dataset]
        plt.bar(x + (index - 0.5) * width, [row["no_tta"], row["full"]], width, label=dataset)
    plt.xticks(x, labels)
    plt.ylabel("Macro Image-AUROC")
    plt.ylim(80, 94)
    plt.legend(frameon=False)
    plt.title("Paired TTA ablation")
    plt.tight_layout()
    plt.savefig(output_root / "tta_ablation.png", bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-full", required=True)
    parser.add_argument("--self-no-tta")
    parser.add_argument("--public-full", required=True)
    parser.add_argument("--public-no-tta")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    score_root = output_root / "scores"
    score_root.mkdir(parents=True, exist_ok=True)
    full_sets = {
        "self_rf": load_dataset(Path(args.self_full), "self_rf"),
        "public_rf": load_dataset(Path(args.public_full), "public_rf"),
    }
    no_tta_sets = None
    if args.self_no_tta or args.public_no_tta:
        if not (args.self_no_tta and args.public_no_tta):
            parser.error("--self-no-tta and --public-no-tta must be provided together")
        no_tta_sets = {
            "self_rf": load_dataset(Path(args.self_no_tta), "self_rf"),
            "public_rf": load_dataset(Path(args.public_no_tta), "public_rf"),
        }

    per_cell = []
    for dataset, cells in full_sets.items():
        for cell in cells:
            metrics = metric_row(cell["labels"], cell["final"])
            per_cell.append({
                "dataset": dataset,
                "signal": cell["signal"],
                "scene": cell["scene"],
                "jsr": cell["jsr"],
                "n": len(cell["labels"]),
                **metrics,
            })
            np.savez_compressed(
                score_root / cell["source"].name,
                labels=cell["labels"],
                vit_scores=cell["vit"],
                cnn_scores=cell["cnn"],
                final_scores=cell["final"],
            )
    write_csv(output_root / "metrics_per_cell.csv", per_cell)
    for dataset in full_sets:
        write_csv(
            output_root / f"metrics_per_cell_{dataset}.csv",
            [row for row in per_cell if row["dataset"] == dataset],
        )

    def group_rows(key):
        rows = []
        for dataset, cells in full_sets.items():
            groups = {}
            for cell in cells:
                groups.setdefault(cell[key], []).append(cell)
            for value, items in sorted(groups.items()):
                labels = np.concatenate([item["labels"] for item in items])
                scores = np.concatenate([item["final"] for item in items])
                rows.append({"dataset": dataset, key: value, "n_cells": len(items), **metric_row(labels, scores)})
        return rows

    signal_rows = group_rows("signal")
    jsr_rows = group_rows("jsr")
    write_csv(output_root / "metrics_by_signal.csv", signal_rows)
    write_csv(output_root / "metrics_by_jsr.csv", jsr_rows)
    for dataset in full_sets:
        write_csv(
            output_root / f"metrics_by_signal_{dataset}.csv",
            [row for row in signal_rows if row["dataset"] == dataset],
        )
        write_csv(
            output_root / f"metrics_by_jsr_{dataset}.csv",
            [row for row in jsr_rows if row["dataset"] == dataset],
        )

    dataset_summary = []
    for dataset, cells in full_sets.items():
        cell_metrics = [metric_row(cell["labels"], cell["final"]) for cell in cells]
        dataset_summary.append({
            "dataset": dataset,
            "n_cells": len(cells),
            "macro_image_auroc": float(np.mean([row["image_auroc"] for row in cell_metrics])),
            "macro_image_auprc": float(np.mean([row["image_auprc"] for row in cell_metrics])),
            "macro_fpr_at_95_tpr": float(np.mean([row["fpr_at_95_tpr"] for row in cell_metrics])),
            "roc_curve": f"roc_{dataset}.png",
        })
    write_csv(output_root / "dataset_summary.csv", dataset_summary)
    for row in dataset_summary:
        write_csv(output_root / f"dataset_summary_{row['dataset']}.csv", [row])

    if no_tta_sets is not None:
        ablation_rows = []
        for dataset, cells in full_sets.items():
            no_tta_cells = no_tta_sets[dataset]

            def macro(items, key):
                values = []
                for item in items:
                    values.append(metric_row(item["labels"], item[key])["image_auroc"])
                return float(np.mean(values))

            ablation_rows.append({
                "dataset": dataset,
                "vit_only": macro(cells, "vit"),
                "cnn_only": macro(cells, "cnn"),
                "full": macro(cells, "final"),
                "no_tta": macro(no_tta_cells, "final"),
            })
        write_csv(output_root / "ablation_macro.csv", ablation_rows)
        plot_ablation(ablation_rows, output_root)
    for dataset, cells in full_sets.items():
        plot_roc(cells, dataset, output_root / f"roc_{dataset}.png")

    lines = [
        "# Final Pure-Visual Method Report",
        "",
        "Protocol: per-frequency normal-only support; paired TTA with 3x3 blur and +-4 px time shifts; label-free batch-level CNN confidence gate.",
        "",
        "## Deliverables",
        "",
        "- `dataset_summary_self_rf.csv`, `dataset_summary_public_rf.csv`: one summary table per dataset",
        "- `metrics_per_cell.csv`: per anomaly, scene, and JSR metrics",
        "- `metrics_per_cell_self_rf.csv`, `metrics_per_cell_public_rf.csv`: per-cell tables split by dataset",
        "- `metrics_by_signal.csv`: pooled ROC/AUPRC/FPR@95 by anomaly type",
        "- `metrics_by_signal_self_rf.csv`, `metrics_by_signal_public_rf.csv`: anomaly-type tables split by dataset",
        "- `metrics_by_jsr.csv`: pooled ROC/AUPRC/FPR@95 by JSR",
        "- `metrics_by_jsr_self_rf.csv`, `metrics_by_jsr_public_rf.csv`: JSR tables split by dataset",
        "- `roc_self_rf.png`, `roc_public_rf.png`: final-method ROC figures",
    ]
    if no_tta_sets is not None:
        lines.extend([
            "- `ablation_macro.csv`: component and TTA ablation",
            "- `component_ablation.png`, `tta_ablation.png`: ablation figures",
        ])
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
