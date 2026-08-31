#!/usr/bin/env python
"""Compute AUROC / AUPRC / FPR@95%TPR for saved SPADE cell scores.

Reads every scores/*.npz under an output root, computes the three image-level
metrics per cell, and writes a CSV plus a summary JSON.  The metric
conventions match the main-table protocol (roc_curve at tpr >= 0.95).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def metric(labels, scores):
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "fpr95": float("nan")}
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels, scores) * 100.0),
        "auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if reached.size else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="output root with scores/ dir")
    parser.add_argument("--output", required=True, help="output CSV path")
    args = parser.parse_args()

    root = Path(args.root)
    score_dir = root / "scores"
    if not score_dir.is_dir():
        raise SystemExit(f"No scores dir under {root}")

    rows = []
    for npz_path in sorted(score_dir.glob("*.npz")):
        with np.load(npz_path, allow_pickle=True) as loaded:
            labels = np.asarray(loaded["labels"], dtype=np.int32)
            scores = np.asarray(loaded["scores"], dtype=np.float64)
        values = metric(labels, scores)
        rows.append({"cell": npz_path.stem, **values})

    if not rows:
        raise SystemExit(f"No npz files under {score_dir}")

    # macro over non-nan auroc cells
    mean = {key: float(np.nanmean([r[key] for r in rows])) for key in ("auroc", "auprc", "fpr95")}
    rows.append({"cell": "macro", **mean})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cell", "auroc", "auprc", "fpr95"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")
    print(json.dumps(mean, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()