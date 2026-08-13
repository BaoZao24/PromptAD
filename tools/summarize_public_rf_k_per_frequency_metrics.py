#!/usr/bin/env python
"""Summarize all full-test Public RF k-per-frequency image metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


KS = (1, 2, 4)
TRADITIONAL_ORDER = (
    "energy_detector",
    "spectral_entropy",
    "spectral_flatness",
    "spectral_kurtosis",
    "ca_cfar",
    "statistical_fusion",
)
INFORMATION_THEORETIC_ORDER = ("kld_reference", "ica_frozen")
DISPLAY_NAMES = {
    "energy_detector": "ED",
    "spectral_entropy": "Spectral entropy",
    "spectral_flatness": "Spectral flatness",
    "spectral_kurtosis": "Spectral kurtosis",
    "ca_cfar": "CA-CFAR",
    "statistical_fusion": "SCSE (Support-Calibrated Spectral Ensemble)",
    "vae_reconstruction": "VAE-MSE",
    "iad_per": "IAD-PER",
    "saife_reconstruction": "SAIFE",
    "kld_reference": "KLD-Ref",
    "ica_frozen": "ICA-Frozen",
    "udma_reimplementation": "UDMA-ResNet18",
    "deep_svdd": "Deep SVDD",
    "padim_diag_resnet18": "PaDiM",
    "stfpm_resnet18": "STFPM",
    "confidence_gated_dual_visual": "Ours: support-only confidence fusion",
}
SCORE_METHODS = {
    "vae_reconstruction": "vae_full_k{k}",
    "deep_svdd": "deepsvdd_full_k{k}",
    "padim_diag_resnet18": "padim_full_k{k}",
    "stfpm_resnet18": "stfpm_full_k{k}",
}
PUBLISHED_SCORE_METHODS = {
    "iad_per": "20260811_iad_per_public_rf_k{k}_formal",
    "saife_reconstruction": "20260812_public_rf_k_per_frequency/saife_k{k}",
    "udma_reimplementation": "20260811_udma_resnet18_public_rf_k{k}_formal",
}
INFORMATION_THEORETIC_DIRECTORY = "20260811_kld_ica_public_rf_k{k}_formal"
MAIN_METHOD_ORDER = (
    *TRADITIONAL_ORDER,
    *INFORMATION_THEORETIC_ORDER,
    "vae_reconstruction",
    "iad_per",
    "saife_reconstruction",
    "udma_reimplementation",
    "deep_svdd",
    "padim_diag_resnet18",
    "stfpm_resnet18",
    "confidence_gated_dual_visual",
    "patchcore_resnet50",
    "winclip_fewshot",
)
FULL_VISUAL_METHODS = {
    "patchcore_resnet50": "patchcore_full_k{k}",
    "winclip_fewshot": "winclip_full_k{k}",
}
CONTROLLED_METHODS = {
    "patchcore_resnet50": "patchcore_controlled_k{k}",
    "winclip_fewshot": "winclip_controlled_k{k}",
}
DISPLAY_NAMES.update(
    {
        "patchcore_resnet50": "PatchCore",
        "winclip_fewshot": "WinCLIP",
    }
)


def score_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels, scores) * 100.0),
        "auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if reached.size else 100.0,
    }


def require_file(path: Path, *, context: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{context}: missing completed result file: {path}")
    return path


def mean_score_directory(path: Path, *, method: str, k: int) -> dict[str, float]:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Public RF k={k} {DISPLAY_NAMES[method]}: missing result directory: {path}"
        )
    files = sorted((path / "scores").glob("*.npz"))
    if len(files) != 15:
        raise RuntimeError(
            f"Public RF k={k} {DISPLAY_NAMES[method]} is incomplete: expected "
            f"15 score files under {path / 'scores'}, found {len(files)}"
        )
    rows = []
    for score_file in files:
        with np.load(score_file, allow_pickle=True) as data:
            missing = {"labels", "scores"} - set(data.files)
            if missing:
                raise KeyError(
                    f"Public RF k={k} {DISPLAY_NAMES[method]}: {score_file} "
                    f"is missing arrays {sorted(missing)}"
                )
            rows.append(score_metrics(data["labels"], data["scores"]))
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("auroc", "auprc", "fpr95")
    }


def validate_published_score_run(path: Path, *, method: str, k: int) -> None:
    summary_path = require_file(
        path / "summary.json", context=f"Public RF k={k} {DISPLAY_NAMES[method]}"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "method": method,
        "protocol": "public_rf",
        "num_jobs": 15,
        "per_frequency_k": k,
    }
    mismatches = {
        key: (summary.get(key), value)
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if method == "udma_reimplementation" and summary.get("reference_extractor") != "resnet18_imagenet":
        mismatches["reference_extractor"] = (
            summary.get("reference_extractor"),
            "resnet18_imagenet",
        )
    if mismatches:
        raise RuntimeError(
            f"Public RF k={k} {DISPLAY_NAMES[method]} has an unexpected formal "
            f"protocol in {summary_path}: {mismatches}"
        )


def read_traditional(path: Path) -> dict[str, dict[str, float]]:
    metrics_path = require_file(
        path / "metrics_macro.csv", context="Public RF traditional baselines"
    )
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    missing = set(TRADITIONAL_ORDER) - set(rows)
    if missing:
        raise RuntimeError(
            f"Public RF traditional result is incomplete at {metrics_path}: "
            f"missing methods {sorted(missing)}"
        )
    return {
        method: {key: float(rows[method][key]) for key in ("auroc", "auprc", "fpr95")}
        for method in TRADITIONAL_ORDER
    }


def read_ours(path: Path) -> dict[str, float]:
    summary_path = require_file(
        path / "summary.json", context="Public RF confidence fusion"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["num_cells"] != 15 or summary["gate_protocol"] != "support_only":
        raise RuntimeError(f"Unexpected formal fusion protocol in {path}")
    macro = summary["macro"]
    return {
        "auroc": float(macro["confidence_gated_auc"]),
        "auprc": float(macro["confidence_gated_auprc"]),
        "fpr95": float(macro["confidence_gated_fpr95"]),
    }


def read_information_theoretic(path: Path, *, method: str, k: int) -> dict[str, float]:
    summary_path = require_file(
        path / "summary.json",
        context=f"Public RF k={k} {DISPLAY_NAMES[method]}",
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "complete"
        or summary.get("formal_full_test") is not True
        or summary.get("num_cell_rows") != 30
        or method not in summary.get("methods", [])
    ):
        raise RuntimeError(
            f"Public RF k={k} {DISPLAY_NAMES[method]} is not a complete formal "
            f"15-cell run according to {summary_path}"
        )
    metrics_path = require_file(
        path / "metrics_macro.csv",
        context=f"Public RF k={k} {DISPLAY_NAMES[method]}",
    )
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        selected = [
            row
            for row in csv.DictReader(handle)
            if row.get("row_type") == "macro"
            and row.get("scope") == "overall"
            and row.get("method") == method
            and row.get("shot") == f"k{k}_per_frequency"
        ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Public RF k={k} {DISPLAY_NAMES[method]} is incomplete or ambiguous: "
            f"expected one overall macro row in {metrics_path}, found {len(selected)}"
        )
    return {key: float(selected[0][key]) for key in ("auroc", "auprc", "fpr95")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="analysis_outputs/20260810_public_rf_k_per_frequency",
    )
    parser.add_argument(
        "--new-results-root",
        type=Path,
        default=None,
        help="Directory containing the 20260811 IAD-PER, KLD/ICA, UDMA, and 20260812 SAIFE runs "
        "(default: parent of --root).",
    )
    parser.add_argument("--output", default="full_test_metrics_summary.csv")
    args = parser.parse_args()
    root = Path(args.root)
    new_results_root = args.new_results_root or root.parent

    by_method: dict[str, dict[int, dict[str, float]]] = {
        method: {} for method in MAIN_METHOD_ORDER
    }

    for k in KS:
        for method, metrics in read_traditional(root / f"traditional_full_k{k}").items():
            by_method[method][k] = metrics
        for method, directory in SCORE_METHODS.items():
            by_method[method][k] = mean_score_directory(
                root / directory.format(k=k), method=method, k=k
            )
        for method, directory in FULL_VISUAL_METHODS.items():
            by_method[method][k] = mean_score_directory(
                root / directory.format(k=k), method=method, k=k
            )
        for method, directory in PUBLISHED_SCORE_METHODS.items():
            published_root = new_results_root / directory.format(k=k)
            validate_published_score_run(published_root, method=method, k=k)
            by_method[method][k] = mean_score_directory(
                published_root, method=method, k=k
            )
        information_root = new_results_root / INFORMATION_THEORETIC_DIRECTORY.format(k=k)
        for method in INFORMATION_THEORETIC_ORDER:
            by_method[method][k] = read_information_theoretic(
                information_root, method=method, k=k
            )
        by_method["confidence_gated_dual_visual"][k] = read_ours(
            root / f"confidence_fusion_full_k{k}"
        )

    rows = []
    for method in MAIN_METHOD_ORDER:
        values = by_method[method]
        row = {"method": method, "method_display": DISPLAY_NAMES[method]}
        for k in KS:
            for metric in ("auroc", "auprc", "fpr95"):
                row[f"k{k}_{metric}"] = values[k][metric]
        rows.append(row)

    output = root / args.output
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)

    controlled_rows = []
    for method, directory in CONTROLLED_METHODS.items():
        row = {"method": method, "method_display": DISPLAY_NAMES[method]}
        for k in KS:
            metrics = mean_score_directory(
                root / directory.format(k=k), method=method, k=k
            )
            for metric in ("auroc", "auprc", "fpr95"):
                row[f"k{k}_{metric}"] = metrics[metric]
        controlled_rows.append(row)
    controlled_output = root / "controlled_test_metrics_summary.csv"
    with controlled_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(controlled_rows[0]))
        writer.writeheader()
        writer.writerows(controlled_rows)
    print(controlled_output)


if __name__ == "__main__":
    main()
