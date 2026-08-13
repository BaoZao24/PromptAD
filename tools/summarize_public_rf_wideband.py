#!/usr/bin/env python
"""Summarize the public-RF wideband-pulse supplement.

The script reads the fixed wideband score directories produced by the formal
evaluation commands and reports per-JSR and macro image-level metrics using the
same definitions as the paper tables.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


METHODS = {
    "VAE": ("vae", "scores", "public_rf-wideband_pulse-RF_SPE_PNG_public-"),
    "SAIFE": ("saife", "scores", "public_rf-wideband_pulse-RF_SPE_PNG_public-"),
    "Deep SVDD": ("deepsvdd", "scores", "public_rf-wideband_pulse-RF_SPE_PNG_public-"),
    "PaDiM": ("padim", "scores", "public_rf-wideband_pulse-RF_SPE_PNG_public-"),
    "STFPM": ("stfpm", "scores", "public_rf-wideband_pulse-RF_SPE_PNG_public-"),
    "WinCLIP": ("winclip", "scores", "public_rf-wideband_pulse-RF_SPE_PNG_public-"),
    "PatchCore": ("patchcore", "scores", "public_rf-wideband_pulse-RF_SPE_PNG_public-"),
    "PromptAD": ("vit", "scores", "wideband_pulse-"),
    "Ours": ("fusion", "scores", "wideband_pulse-"),
}

# Macro values of the already reported 12-cell public-RF protocol.  The new
# wideband cells are equal-weighted with these existing cells below.
ORIGINAL_PUBLIC_MACRO = {
    "VAE": (51.16, 11.16, 91.81),
    "SAIFE": (51.28, 12.83, 91.76),
    "Deep SVDD": (63.54, 18.55, 76.89),
    "PaDiM": (72.69, 20.42, 65.57),
    "STFPM": (54.17, 11.42, 84.51),
    "WinCLIP": (66.28, 16.37, 78.18),
    "PatchCore": (80.11, 46.96, 58.00),
    "PromptAD": (87.62, 54.67, 42.53),
    "Ours": (88.92, 55.76, 36.10),
}


def score_key(method: str) -> str:
    if method == "PromptAD":
        return "text_vit_patchcore_max"
    if method == "Ours":
        return "confidence_gated_score"
    return "scores"


def metric_row(labels, scores) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[reached[0]] * 100.0) if len(reached) else 100.0
    return {
        "image_auroc": float(roc_auc_score(labels, scores) * 100.0),
        "image_ap": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": fpr95,
    }


def read_method(root: Path, method: str) -> list[dict]:
    folder, _, prefix = METHODS[method]
    score_dir = root / folder / "scores"
    rows = []
    key = score_key(method)
    for path in sorted(score_dir.glob("*-scores.npz")):
        if not path.name.startswith(prefix):
            continue
        jsr = path.name.removesuffix("-scores.npz").split("-")[-1]
        with np.load(path, allow_pickle=True) as data:
            if key not in data.files:
                raise KeyError(f"{path} does not contain {key!r}")
            metrics = metric_row(data["labels"], data[key])
            n = int(np.asarray(data["labels"]).size)
        rows.append({"method": method, "signal": "wideband_pulse", "jsr": jsr, "n": n, **metrics})
    if len(rows) != 3:
        raise RuntimeError(f"{method}: expected 3 wideband cells under {score_dir}, found {len(rows)}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="wideband experiment output root")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    output_root = Path(args.output_root)
    per_cell = []
    summary = []
    for method in METHODS:
        rows = read_method(root, method)
        per_cell.extend(rows)
        wideband_macro = {
            "image_auroc": float(np.mean([row["image_auroc"] for row in rows])),
            "image_ap": float(np.mean([row["image_ap"] for row in rows])),
            "fpr95": float(np.mean([row["fpr95"] for row in rows])),
        }
        summary.append({
            "method": method,
            "signal": "wideband_pulse",
            "n_cells": len(rows),
            **wideband_macro,
        })
    write_csv(output_root / "metrics_per_cell.csv", per_cell)
    write_csv(output_root / "metrics_macro.csv", summary)
    five_type = []
    for row in summary:
        old = ORIGINAL_PUBLIC_MACRO[row["method"]]
        five_type.append({
            "method": row["method"],
            "n_cells": 15,
            "image_auroc": round((12.0 * old[0] + 3.0 * row["image_auroc"]) / 15.0, 2),
            "image_ap": round((12.0 * old[1] + 3.0 * row["image_ap"]) / 15.0, 2),
            "fpr95": round((12.0 * old[2] + 3.0 * row["fpr95"]) / 15.0, 2),
        })
    write_csv(output_root / "public_rf_five_type_macro.csv", five_type)
    print(f"wrote {output_root / 'metrics_per_cell.csv'}")
    print(f"wrote {output_root / 'metrics_macro.csv'}")


if __name__ == "__main__":
    main()
