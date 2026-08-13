#!/usr/bin/env python3
"""Evaluate an inductive support-only confidence gate from frozen RF scores.

The formal test scores are reused, but the gate calibration is rebuilt from
normal support reference scores produced by the ``--support-reference-only``
branches.  No other test sample is used to normalize or rank a test sample.
The safe support-only result is the formal gate; the transductive result is
retained only as a protocol comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.confidence_gate import safe_support_only_gate


METRICS = ("auroc", "auprc", "fpr95")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels, scores) * 100.0),
        "auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if len(reached) else 100.0,
    }


def support_minmax(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    reference = reference[np.isfinite(reference)]
    if reference.size == 0:
        raise ValueError("Support reference is empty")
    low, high = float(reference.min()), float(reference.max())
    if high - low < 1e-12:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def empirical_support_rank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map values to an empirical CDF using support-only mid-ranks."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    reference = np.sort(np.asarray(reference, dtype=np.float64).reshape(-1))
    reference = reference[np.isfinite(reference)]
    if reference.size <= 1:
        return np.zeros_like(values)
    left = np.searchsorted(reference, values, side="left").astype(np.float64)
    right = np.searchsorted(reference, values, side="right").astype(np.float64)
    # The external value occupies the average ordinal position between the
    # reference values on either side. Clip tails to the support rank range.
    ordinal = (left + right - 1.0) / 2.0
    return np.clip(ordinal / float(reference.size - 1), 0.0, 1.0)


def support_only_gate(
    vit_scores: np.ndarray,
    cnn_scores: np.ndarray,
    vit_reference: np.ndarray,
    cnn_reference: np.ndarray,
    quantile: float = 0.5,
) -> dict[str, np.ndarray]:
    vit = support_minmax(vit_scores, vit_reference)
    cnn = support_minmax(cnn_scores, cnn_reference)
    cnn_rank = empirical_support_rank(cnn_scores, cnn_reference)
    rank_confidence = np.clip(
        (cnn_rank - float(quantile)) / (1.0 - float(quantile)),
        0.0,
        1.0,
    )
    cnn_advantage = np.clip(cnn - vit, 0.0, 1.0)
    gate = rank_confidence * cnn_advantage
    score = 1.0 - (1.0 - vit) * (1.0 - gate * cnn)
    return {
        "score": score,
        "vit_normalized": vit,
        "cnn_normalized": cnn,
        "cnn_rank": cnn_rank,
        "gate": gate,
    }


def cell_key(path: Path, dataset: str) -> tuple[str, str, str]:
    stem = path.stem
    if dataset == "self_rf":
        prefix = "in_house_rf-"
        if not stem.startswith(prefix):
            raise ValueError(f"Unexpected In-house RF score name: {path.name}")
        signal, scene, jsr = stem[len(prefix) :].split("-", 2)
        return signal, scene, jsr
    prefix = "public_rf-"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected Public RF score name: {path.name}")
    signal, scene, jsr = stem[len(prefix) :].split("-", 2)
    return signal, scene, jsr


def load_reference(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if key not in data.files:
            raise KeyError(f"{path} does not contain {key!r}")
        return np.asarray(data[key], dtype=np.float64)


def macro_rows(rows: list[dict]) -> list[dict]:
    output = []
    for dataset in ("self_rf", "public_rf"):
        subset = [row for row in rows if row["dataset"] == dataset]
        for method, prefix in (
            ("support_only_naive", "support"),
            ("support_only_safe", "safe"),
            ("test_batch", "batch"),
        ):
            output.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n_cells": len(subset),
                    "auroc": float(np.mean([row[f"{prefix}_auroc"] for row in subset])),
                    "auprc": float(np.mean([row[f"{prefix}_auprc"] for row in subset])),
                    "fpr95": float(np.mean([row[f"{prefix}_fpr95"] for row in subset])),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-score-root", type=Path, required=True)
    parser.add_argument("--vit-reference-root", type=Path, required=True)
    parser.add_argument("--cnn-reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.5)
    args = parser.parse_args()

    if not 0.0 <= args.quantile < 1.0:
        raise ValueError("--quantile must be in [0, 1)")

    reference_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def refs(dataset: str, scene: str) -> tuple[np.ndarray, np.ndarray]:
        key = (dataset, scene)
        if key in reference_cache:
            return reference_cache[key]
        if dataset == "self_rf":
            vit_path = args.vit_reference_root / "support_reference" / f"{scene}.npz"
            cnn_path = args.cnn_reference_root / "support_reference" / f"{scene}.npz"
        else:
            vit_path = args.vit_reference_root / "support_reference" / "public_rf.npz"
            cnn_path = args.cnn_reference_root / "support_reference" / "public_rf.npz"
        value = (
            load_reference(vit_path, "vit_scores"),
            load_reference(cnn_path, "cnn_scores"),
        )
        reference_cache[key] = value
        return value

    rows = []
    output_scores = args.output_root / "scores"
    output_scores.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.test_score_root.glob("*.npz")):
        dataset = "self_rf" if path.name.startswith("in_house_rf-") else "public_rf"
        signal, scene, jsr = cell_key(path, dataset)
        with np.load(path, allow_pickle=True) as data:
            labels = np.asarray(data["labels"], dtype=np.int32)
            vit_scores = np.asarray(data["vit_scores"], dtype=np.float64)
            cnn_scores = np.asarray(data["cnn_scores"], dtype=np.float64)
            batch_scores = np.asarray(data["vit_cnn_current_gate"], dtype=np.float64)
        vit_reference, cnn_reference = refs(dataset, scene)
        result = support_only_gate(
            vit_scores,
            cnn_scores,
            vit_reference,
            cnn_reference,
            quantile=args.quantile,
        )
        safe_result = safe_support_only_gate(
            vit_scores,
            cnn_scores,
            vit_reference,
            cnn_reference,
        )
        support_metrics = score_metric(labels, result["score"])
        safe_metrics = score_metric(labels, safe_result["score"])
        batch_metrics = score_metric(labels, batch_scores)
        rows.append(
            {
                "dataset": dataset,
                "signal": signal,
                "scene": scene,
                "jsr": jsr,
                "n": len(labels),
                "support_auroc": support_metrics["auroc"],
                "support_auprc": support_metrics["auprc"],
                "support_fpr95": support_metrics["fpr95"],
                "safe_auroc": safe_metrics["auroc"],
                "safe_auprc": safe_metrics["auprc"],
                "safe_fpr95": safe_metrics["fpr95"],
                "batch_auroc": batch_metrics["auroc"],
                "batch_auprc": batch_metrics["auprc"],
                "batch_fpr95": batch_metrics["fpr95"],
                "support_gate_active_rate": float(np.mean(result["gate"] > 0.0)),
                "safe_gate_active_rate": float(np.mean(safe_result["gate"] > 0.0)),
            }
        )
        np.savez_compressed(
            output_scores / path.name,
            labels=labels,
            vit_scores=vit_scores,
            cnn_scores=cnn_scores,
            vit_only=vit_scores.astype(np.float32),
            cnn_only=cnn_scores.astype(np.float32),
            support_only_gate=result["score"].astype(np.float32),
            support_only_safe=safe_result["score"].astype(np.float32),
            confidence_gated_score=safe_result["score"].astype(np.float32),
            test_batch_gate=batch_scores.astype(np.float32),
            support_cnn_rank=result["cnn_rank"].astype(np.float32),
            support_gate=result["gate"].astype(np.float32),
            safe_cnn_rank=safe_result["cnn_rank"].astype(np.float32),
            safe_gate=safe_result["gate"].astype(np.float32),
        )

    if len(rows) != 75:
        raise RuntimeError(f"Expected 75 RF cells, found {len(rows)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "metrics_per_cell.csv", rows)
    write_csv(args.output_root / "metrics_macro.csv", macro_rows(rows))

    by_signal = []
    for dataset in ("self_rf", "public_rf"):
        for signal in sorted({row["signal"] for row in rows if row["dataset"] == dataset}):
            subset = [row for row in rows if row["dataset"] == dataset and row["signal"] == signal]
            by_signal.append(
                {
                    "dataset": dataset,
                    "signal": signal,
                    "n_cells": len(subset),
                    "support_auroc": float(np.mean([row["support_auroc"] for row in subset])),
                    "batch_auroc": float(np.mean([row["batch_auroc"] for row in subset])),
                    "support_auprc": float(np.mean([row["support_auprc"] for row in subset])),
                    "batch_auprc": float(np.mean([row["batch_auprc"] for row in subset])),
                    "support_fpr95": float(np.mean([row["support_fpr95"] for row in subset])),
                    "safe_auroc": float(np.mean([row["safe_auroc"] for row in subset])),
                    "safe_auprc": float(np.mean([row["safe_auprc"] for row in subset])),
                    "safe_fpr95": float(np.mean([row["safe_fpr95"] for row in subset])),
                    "batch_fpr95": float(np.mean([row["batch_fpr95"] for row in subset])),
                }
            )
    write_csv(args.output_root / "metrics_by_signal.csv", by_signal)

    protocol = {
        "method": "support_only_confidence_gated_rf",
        "formal_method": "safe_support_only",
        "test_score_root": str(args.test_score_root),
        "vit_reference_root": str(args.vit_reference_root),
        "cnn_reference_root": str(args.cnn_reference_root),
        "support_only": True,
        "uses_test_batch_statistics": False,
        "uses_test_labels_for_scoring": False,
        "reference_views": {
            "vit": ["time_shift_up_large", "time_shift_down_large"],
            "cnn": ["time_shift_up", "time_shift_down", "blur"],
        },
        "rank": "empirical support CDF with mid-ranks",
        "minmax": "support-reference min/max, clipped to [0,1]; constant reference maps to zero",
        "rank_quantile": args.quantile,
        "safe_rank_quantile": 0.8,
        "safe_temperature": 2.5,
        "safe_alpha": 2.5,
        "num_cells": len(rows),
    }
    (args.output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {row["dataset"]: {} for row in macro_rows(rows)}
    for row in macro_rows(rows):
        summary[row["dataset"]][row["method"]] = {
            metric: row[metric] for metric in METRICS
        }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = """# RF support-only gate evaluation

The formal RF gate is the safe support-only rule. Frozen ViT/CNN branch scores
are reused, while all calibration statistics are computed only from normal
support reference views. No test sample is used to construct the reference
distribution. The transductive test-batch gate is retained only as a protocol
comparison in the output files.

Command:

```bash
python tools/eval_support_only_rf_gate.py \\
  --test-score-root analysis_outputs/20260731_rf_dual_visual_scores/scores \\
  --vit-reference-root analysis_outputs/20260801_rf_support_only_vit_combined \\
  --cnn-reference-root analysis_outputs/20260801_rf_support_only_cnn_combined \\
  --output-root analysis_outputs/20260801_rf_support_only_probe
```

The exact protocol is recorded in `protocol.json`; per-cell and macro metrics
are in `metrics_per_cell.csv` and `metrics_macro.csv`. The formal score key in
each output NPZ is `confidence_gated_score`.
"""
    (args.output_root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
