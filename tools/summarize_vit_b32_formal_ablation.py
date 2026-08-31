#!/usr/bin/env python
"""Summarize the ViT-only rows from the ViT-B-32 formal rerun."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


METRICS = ("auroc", "auprc", "fpr95")


def score_metrics(labels, scores) -> dict[str, float]:
    labels = np.asarray(labels).reshape(-1)
    binary = (labels != 0).astype(np.int32)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if np.unique(binary).size < 2:
        return {key: float("nan") for key in METRICS}
    fpr, tpr, _ = roc_curve(binary, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(binary, scores) * 100.0),
        "auprc": float(average_precision_score(binary, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if reached.size else 100.0,
    }


def summarize_npz_dir(path: Path, dataset: str) -> tuple[list[dict], dict]:
    rows = []
    for score_path in sorted(path.glob("*.npz")):
        with np.load(score_path) as data:
            metrics = score_metrics(data["labels"], data["vit_patchcore_max_scores"])
        rows.append({"dataset": dataset, "cell": score_path.stem, **metrics})
    if not rows:
        raise FileNotFoundError(f"No ViT score files found under {path}")
    return rows, macro(rows)


def macro(rows: list[dict]) -> dict:
    return {
        key: float(np.nanmean([float(row[key]) for row in rows]))
        for key in METRICS
    }


def read_ofdma(path: Path) -> tuple[list[dict], dict]:
    rows = []
    with (path / "branch_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("row_type") == "scene_macro"
                and row.get("scope") == "overall"
                and row.get("method") == "ours_vit"
            ):
                rows.append(
                    {
                        "dataset": "ofdma",
                        "shot": int(row["shot"]),
                        "auroc": float(row["auroc"]),
                        "auprc": float(row["auprc"]),
                        "fpr95": float(row["fpr95"]),
                    }
                )
    if not rows:
        raise FileNotFoundError(f"No OFDMA ours_vit scene macro rows under {path}")
    return rows, {
        str(shot): macro([row for row in rows if row["shot"] == shot])
        for shot in sorted({row["shot"] for row in rows})
    }


def read_fedjam(path: Path) -> tuple[list[dict], dict]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    rows = []
    for shot in (1, 2, 4):
        values = summary[f"shot_{shot}"]["vit_patchcore"]
        rows.append({"dataset": "fedjam", "shot": shot, **{
            key: float(values[key]) for key in METRICS
        }})
    return rows, {
        str(row["shot"]): {key: row[key] for key in METRICS}
        for row in rows
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="analysis_outputs/20260824_vit_b32_formal_ablation",
    )
    args = parser.parse_args()
    root = Path(args.output_root).resolve()

    cell_rows = []
    macro_rows = []
    for name, dataset in (("inhouse_rf", "inhouse_rf"),):
        rows, values = summarize_npz_dir(root / name / "scores", dataset)
        cell_rows.extend(rows)
        macro_rows.append({"dataset": dataset, "shot": "per_frequency", **values})
    for shot in (1, 2, 4):
        rows, values = summarize_npz_dir(root / f"public_rf_k{shot}" / "scores", f"public_rf_k{shot}")
        cell_rows.extend(rows)
        macro_rows.append({"dataset": f"public_rf_k{shot}", "shot": shot, **values})

    ofdma_rows, ofdma_summary = read_ofdma(root / "ofdma")
    fedjam_rows, fedjam_summary = read_fedjam(root / "fedjam")
    macro_rows.extend(ofdma_rows)
    macro_rows.extend(fedjam_rows)

    write_csv(root / "vit_only_metrics_by_cell.csv", cell_rows)
    write_csv(root / "vit_only_metrics_macro.csv", macro_rows)
    summary = {
        "backbone": "ViT-B-32",
        "score_branch": "ViT-only",
        "aggregation": {
            "inhouse_rf": "equal-weighted mean of 60 cell metrics",
            "public_rf": "equal-weighted mean of 15 cell metrics per k",
            "ofdma": "30-scene macro rows from branch_diagnostics.csv",
            "fedjam": "full-test overall metrics per shot",
        },
        "protocol": {
            "support_only": True,
            "coreset": "0.5 farthest-first",
            "rf_fedjam_nn": "5-NN mean",
            "ofdma_nn": "1-NN, third-highest patch, max over 21 SUs",
            "vit_tta": "identity, time-axis +/-4 px, frequency-response drift strength 3",
        },
        "ofdma": ofdma_summary,
        "fedjam": fedjam_summary,
        "source_files": {
            "inhouse_rf": str((root / "inhouse_rf").resolve()),
            "public_rf": str((root / "public_rf_k1").resolve()),
            "ofdma": str((root / "ofdma" / "branch_diagnostics.csv").resolve()),
            "fedjam": str((root / "fedjam" / "summary.json").resolve()),
        },
    }
    (root / "vit_only_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {root / 'vit_only_metrics_macro.csv'}")
    print(f"wrote {root / 'vit_only_summary.json'}")


if __name__ == "__main__":
    main()
