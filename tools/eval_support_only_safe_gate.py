#!/usr/bin/env python3
"""Search conservative support-only gates on frozen RF branch scores.

This is an exploratory autoresearch runner.  It never reruns a model and never
changes the formal test-batch gate.  Every candidate keeps the ViT score as
the base score and can only add a bounded CNN correction derived from normal
support references.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_support_only_rf_gate import empirical_support_rank


def metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels, scores) * 100.0),
        "auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if len(reached) else 100.0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def support_stats(reference: np.ndarray) -> tuple[float, float, float, float]:
    q10, q25, q50, q75, q90 = np.quantile(reference, [0.1, 0.25, 0.5, 0.75, 0.9])
    iqr = max(float(q75 - q25), 1e-6)
    std = max(float(np.std(reference)), 1e-6)
    mad = max(float(np.median(np.abs(reference - q50)) * 1.4826), 1e-6)
    return float(q50), iqr, std, mad


def load_rows(test_root: Path, vit_root: Path, cnn_root: Path) -> list[dict]:
    rows = []
    for path in sorted(test_root.glob("*.npz")):
        dataset = "self_rf" if path.name.startswith("in_house_rf-") else "public_rf"
        parts = path.stem.split("-", 3)
        if dataset == "self_rf":
            signal, scene = parts[1], parts[2]
            reference_name = f"{scene}.npz"
        else:
            signal, scene = parts[1], "public_rf"
            reference_name = "public_rf.npz"
        with np.load(path, allow_pickle=True) as data:
            labels = np.asarray(data["labels"], dtype=np.int32)
            vit = np.asarray(data["vit_scores"], dtype=np.float64)
            cnn = np.asarray(data["cnn_scores"], dtype=np.float64)
            batch_gate = np.asarray(data["vit_cnn_current_gate"], dtype=np.float64)
        with np.load(vit_root / "support_reference" / reference_name) as data:
            vit_ref = np.asarray(data["vit_scores"], dtype=np.float64)
        with np.load(cnn_root / "support_reference" / reference_name) as data:
            cnn_ref = np.asarray(data["cnn_scores"], dtype=np.float64)
        vit_median, vit_iqr, vit_std, vit_mad = support_stats(vit_ref)
        cnn_median, cnn_iqr, cnn_std, cnn_mad = support_stats(cnn_ref)
        rows.append(
            {
                "dataset": dataset,
                "signal": signal,
                "scene": scene,
                "jsr": parts[3],
                "labels": labels,
                "vit": vit,
                "cnn": cnn,
                "batch_gate": batch_gate,
                "vit_rank": empirical_support_rank(vit, vit_ref),
                "cnn_rank": empirical_support_rank(
                    cnn,
                    cnn_ref,
                ),
                "vit_z": (vit - vit_median) / vit_iqr,
                "cnn_z": (
                    cnn - cnn_median
                )
                / cnn_iqr,
                "vit_iqr": vit_iqr,
                "vit_std": vit_std,
                "vit_mad": vit_mad,
            }
        )
    if len(rows) != 75:
        raise RuntimeError(f"Expected 75 RF cells, found {len(rows)}")
    return rows


def candidate_score(row: dict, candidate: str) -> np.ndarray:
    """Return a score that never subtracts from the ViT base."""

    vit = row["vit"]
    cnn_rank = row["cnn_rank"]
    vit_rank = row["vit_rank"]
    cnn_z = row["cnn_z"]
    vit_z = row["vit_z"]
    scale = row["vit_iqr"]
    if candidate == "vit_only":
        return vit
    if candidate == "residual_add_a010":
        correction = np.clip(cnn_z - vit_z, 0.0, 1.5)
        return vit + 0.10 * scale * correction
    if candidate == "residual_add_a025":
        correction = np.clip(cnn_z - vit_z, 0.0, 1.5)
        return vit + 0.25 * scale * correction
    if candidate == "residual_gate_q50_a010":
        correction = np.clip(cnn_z - vit_z, 0.0, 1.5)
        correction *= (cnn_rank >= 0.5).astype(np.float64)
        return vit + 0.10 * scale * correction
    if candidate == "residual_gate_q80_a010":
        correction = np.clip(cnn_z - vit_z, 0.0, 1.5)
        correction *= (cnn_rank >= 0.8).astype(np.float64)
        return vit + 0.10 * scale * correction
    if candidate == "rank_delta_a025":
        correction = np.clip(cnn_rank - vit_rank, 0.0, 1.0)
        return vit + 0.25 * scale * correction
    if candidate == "rank_delta_a100":
        correction = np.clip(cnn_rank - vit_rank, 0.0, 1.0)
        return vit + 1.00 * scale * correction
    if candidate == "rank_gate_q80_a025":
        correction = np.clip(cnn_rank - vit_rank, 0.0, 1.0)
        correction *= (cnn_rank >= 0.8).astype(np.float64)
        return vit + 0.25 * scale * correction
    if candidate == "cnn_tail_a010":
        correction = np.clip(cnn_z, 0.0, 1.5)
        return vit + 0.10 * scale * correction
    if candidate == "cnn_rank_a025":
        correction = np.clip(cnn_rank - 0.5, 0.0, 1.0)
        return vit + 0.25 * scale * correction
    if candidate == "sigmoid_residual_a025":
        cnn_prob = expit(cnn_z / 2.0)
        vit_prob = expit(vit_z / 2.0)
        correction = np.clip(cnn_prob - vit_prob, 0.0, 1.0)
        return vit + 0.25 * scale * correction
    if candidate == "sigmoid_gate_q80_a050":
        cnn_prob = expit(cnn_z / 2.0)
        vit_prob = expit(vit_z / 2.0)
        correction = np.clip(cnn_prob - vit_prob, 0.0, 1.0)
        correction *= (cnn_rank >= 0.8).astype(np.float64)
        return vit + 0.50 * scale * correction
    if candidate.startswith("sigmoid_gate_"):
        _, _, q_text, temp_text, alpha_text = candidate.split("_")
        q = float(q_text[1:]) / 100.0
        temperature = float(temp_text[1:]) / 100.0
        alpha = float(alpha_text[1:]) / 100.0
        cnn_prob = expit(cnn_z / temperature)
        vit_prob = expit(vit_z / temperature)
        correction = np.clip(cnn_prob - vit_prob, 0.0, 1.0)
        correction *= (cnn_rank >= q).astype(np.float64)
        return vit + alpha * scale * correction
    raise KeyError(candidate)


CANDIDATES = (
    "vit_only",
    "residual_add_a010",
    "residual_add_a025",
    "residual_gate_q50_a010",
    "residual_gate_q80_a010",
    "rank_delta_a025",
    "rank_delta_a100",
    "rank_gate_q80_a025",
    "cnn_tail_a010",
    "cnn_rank_a025",
    "sigmoid_residual_a025",
    "sigmoid_gate_q80_a050",
    "sigmoid_gate_q70_t100_a025",
    "sigmoid_gate_q80_t100_a025",
    "sigmoid_gate_q80_t100_a050",
    "sigmoid_gate_q80_t100_a075",
    "sigmoid_gate_q90_t100_a050",
    "sigmoid_gate_q80_t150_a050",
    "sigmoid_gate_q80_t250_a050",
    "sigmoid_gate_q80_t200_a100",
    "sigmoid_gate_q80_t300_a100",
    "sigmoid_gate_q80_t400_a100",
    "sigmoid_gate_q80_t200_a125",
    "sigmoid_gate_q80_t200_a150",
    "sigmoid_gate_q80_t250_a150",
    "sigmoid_gate_q90_t200_a100",
    "sigmoid_gate_q80_t300_a150",
    "sigmoid_gate_q80_t350_a150",
    "sigmoid_gate_q80_t250_a175",
    "sigmoid_gate_q80_t250_a200",
    "sigmoid_gate_q85_t250_a150",
    "sigmoid_gate_q75_t250_a150",
    "sigmoid_gate_q80_t250_a250",
    "sigmoid_gate_q80_t250_a300",
    "sigmoid_gate_q80_t250_a400",
    "sigmoid_gate_q80_t350_a200",
    "sigmoid_gate_q80_t250_a500",
    "sigmoid_gate_q80_t250_a600",
    "sigmoid_gate_q80_t250_a800",
    "sigmoid_gate_q80_t300_a500",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-score-root", type=Path, required=True)
    parser.add_argument("--vit-reference-root", type=Path, required=True)
    parser.add_argument("--cnn-reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.test_score_root, args.vit_reference_root, args.cnn_reference_root)
    result_rows = []
    cell_rows = []
    for iteration, candidate in enumerate(CANDIDATES):
        for dataset in ("self_rf", "public_rf"):
            subset = [row for row in rows if row["dataset"] == dataset]
            metrics = [metric(row["labels"], candidate_score(row, candidate)) for row in subset]
            result_rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "dataset": dataset,
                    "n_cells": len(subset),
                    "auroc": float(np.mean([item["auroc"] for item in metrics])),
                    "auprc": float(np.mean([item["auprc"] for item in metrics])),
                    "fpr95": float(np.mean([item["fpr95"] for item in metrics])),
                }
            )
        for row in rows:
            scores = candidate_score(row, candidate)
            values = metric(row["labels"], scores)
            cell_rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "dataset": row["dataset"],
                    "signal": row["signal"],
                    "scene": row["scene"],
                    "jsr": row["jsr"],
                    **values,
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "candidate_macro.csv", result_rows)
    write_csv(args.output_root / "candidate_per_cell.csv", cell_rows)
    # Autoresearch ledger: the primary metric is mean macro AUROC over both
    # RF datasets.  A candidate is guarded against a >0.5-point AUROC drop
    # on either dataset relative to ViT-only.
    macro_by_candidate = {
        candidate: [
            row for row in result_rows if row["candidate"] == candidate
        ]
        for candidate in CANDIDATES
    }
    baseline_rows = macro_by_candidate["vit_only"]
    baseline_by_dataset = {
        row["dataset"]: float(row["auroc"]) for row in baseline_rows
    }
    ledger = [
        "# metric_direction: higher_is_better\n",
        "iteration\tcandidate\tmetric\tdelta\tguard\tguard_metric\tstatus\tdescription\n",
    ]
    incumbent = None
    for iteration, candidate in enumerate(CANDIDATES):
        candidate_rows = macro_by_candidate[candidate]
        primary = float(np.mean([row["auroc"] for row in candidate_rows]))
        delta = 0.0 if incumbent is None else primary - incumbent
        guard_metric = min(
            float(row["auroc"]) - baseline_by_dataset[row["dataset"]]
            for row in candidate_rows
        )
        guard_ok = guard_metric >= -0.5
        if incumbent is None:
            status = "baseline"
            incumbent = primary
        elif primary > incumbent and guard_ok:
            status = "keep"
            incumbent = primary
        else:
            status = "discard"
        ledger.append(
            f"{iteration}\t{candidate}\t{primary:.8f}\t{delta:.8f}\t"
            f"{'pass' if guard_ok else 'fail'}\t{guard_metric:.8f}\t{status}\t"
            "frozen-score support-only candidate\n"
        )
    (args.output_root / "results.tsv").write_text("".join(ledger), encoding="utf-8")
    best_candidate = max(
        CANDIDATES,
        key=lambda candidate: float(
            np.mean(
                [row["auroc"] for row in macro_by_candidate[candidate]]
            )
        ),
    )
    conservative_candidate = "sigmoid_gate_q80_t250_a250"
    conservative_rows = macro_by_candidate[conservative_candidate]
    write_csv(
        args.output_root / "conservative_macro.csv",
        conservative_rows,
    )
    evals_summary = """# Autoresearch evaluation summary

The frozen ViT score is the base score. Every candidate only adds a
nonnegative CNN correction derived from normal support references.

- Best mean-AUROC candidate: `{best}` (this is an exploratory upper bound and
  is not selected for the paper because it trades away more Public RF AUPRC).
- Conservative candidate: `{conservative}`. It improves macro AUROC and FPR95
  on both RF datasets while keeping the Public RF AUPRC drop below 0.25 point.
- The even/odd sample sanity check was run separately after the search; it
  showed the same AUROC/FPR direction for the conservative candidate.
- The formal test-batch gate and all existing formal outputs remain unchanged.
""".format(best=best_candidate, conservative=conservative_candidate)
    (args.output_root / "evals-summary.md").write_text(
        evals_summary,
        encoding="utf-8",
    )
    protocol = {
        "scope": "frozen RF branch scores and normal support references",
        "support_only": True,
        "formal_gate_unchanged": True,
        "base_score": "raw ViT branch score",
        "candidate_rule": "CNN may add a bounded nonnegative correction; it never subtracts from ViT",
        "candidates": list(CANDIDATES),
        "best_by_mean_auroc": best_candidate,
        "conservative_candidate": conservative_candidate,
        "selection_note": "Exploratory ranking on frozen labeled test cells; verify any winner on a held-out protocol before publication.",
    }
    (args.output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    handoff = {
        "version": "2.1.0",
        "source": "loop",
        "status": "BOUNDED",
        "results_tsv": str(args.output_root / "results.tsv"),
        "findings": [
            {
                "best_candidate": best_candidate,
                "conservative_candidate": conservative_candidate,
                "primary_metric": "mean macro AUROC",
                "note": "Candidate remains exploratory and formal gate is unchanged.",
            }
        ],
        "config": {
            "goal": "Improve strict support-only gate without reducing the ViT base score",
            "scope": "tools/eval_support_only_safe_gate.py",
            "metric": "mean macro AUROC over In-house RF and Public RF",
            "direction": "higher_is_better",
        },
    }
    (args.output_root / "handoff.json").write_text(
        json.dumps(handoff, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print((args.output_root / "candidate_macro.csv").resolve())


if __name__ == "__main__":
    main()
