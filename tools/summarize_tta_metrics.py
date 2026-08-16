"""Summarize AUROC, AUPRC, and FPR@95%TPR for existing TTA runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[1] / "analysis_outputs/exploratory"


def score_metrics(labels: np.ndarray, scores: np.ndarray, *, nonzero_is_anomaly: bool = False) -> dict[str, float]:
    labels = np.asarray(labels).astype(int)
    if nonzero_is_anomaly:
        labels = (labels != 0).astype(int)
    if np.unique(labels).size != 2:
        raise ValueError(f"Expected binary labels, got {np.unique(labels)}")
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels, scores) * 100.0),
        "auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if len(reached) else 100.0,
    }


def summarize_score_dir(root: Path, score_key: str, *, nonzero_is_anomaly: bool = False) -> dict[str, float]:
    rows = []
    for path in sorted((root / "scores").glob("*.npz")):
        data = np.load(path)
        rows.append(
            score_metrics(
                data["labels"],
                data[score_key],
                nonzero_is_anomaly=nonzero_is_anomaly,
            )
        )
    if not rows:
        raise FileNotFoundError(f"No score files under {root / 'scores'}")
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def summarize_single_score_file(path: Path, score_key: str, *, nonzero_is_anomaly: bool = False) -> dict[str, float]:
    data = np.load(path)
    return score_metrics(data["labels"], data[score_key], nonzero_is_anomaly=nonzero_is_anomaly)


def summarize_ofdma(root: Path, shot: int) -> dict[str, float]:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    rows = [
        row
        for row in summary["overall_scene_macro"]
        if int(row["shot"]) == int(shot)
        and row["method"] == "confidence_gated_dual_visual"
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one OFDMA overall row for shot={shot}, got {len(rows)}")
    row = rows[0]
    return {"auroc": float(row["auroc"]), "auprc": float(row["auprc"]), "fpr95": float(row["fpr95"])}


def add_pair(rows: list[dict], name: str, baseline: dict[str, float], tta: dict[str, float]) -> None:
    rows.append(
        {
            "dataset": name,
            **{f"baseline_{key}": baseline[key] for key in baseline},
            **{f"tta_{key}": tta[key] for key in tta},
            **{f"delta_{key}": tta[key] - baseline[key] for key in baseline},
        }
    )


def main() -> None:
    rows: list[dict] = []

    rf_pairs = [
        (
            "In-house RF",
            "20260813_frequency_response_memory_rf_full/no_tta_merged",
            "20260813_frequency_response_memory_rf_full/response_s3_merged",
        ),
        (
            "Public RF k=1",
            "20260813_frequency_response_memory_public_rf_k1/no_tta_merged",
            "20260813_frequency_response_memory_public_rf_k1/response_s3_merged",
        ),
        (
            "Public RF k=2",
            "20260814_frequency_response_memory_public_rf_k2/no_tta_merged",
            "20260814_frequency_response_memory_public_rf_k2/response_s3_merged",
        ),
        (
            "Public RF k=4",
            "20260814_frequency_response_memory_public_rf_k4/no_tta_merged",
            "20260814_frequency_response_memory_public_rf_k4/response_s3_merged",
        ),
    ]
    for name, baseline_root, tta_root in rf_pairs:
        add_pair(
            rows,
            name,
            summarize_score_dir(ROOT / baseline_root, "vit_patchcore_max_scores"),
            summarize_score_dir(ROOT / tta_root, "vit_patchcore_max_scores"),
        )

    extra_rf_pairs = [
        (
            "In-house RF response strength=5",
            "20260813_frequency_response_memory_rf_full/no_tta_merged",
            "20260813_frequency_response_memory_rf_full/s5",
        ),
        (
            "Public RF k=1 response strength=5",
            "20260813_frequency_response_memory_public_rf_k1/no_tta_merged",
            "20260813_frequency_response_memory_public_rf_k1/response_s5_merged",
        ),
    ]
    for name, baseline_root, tta_root in extra_rf_pairs:
        add_pair(
            rows,
            name,
            summarize_score_dir(ROOT / baseline_root, "vit_patchcore_max_scores"),
            summarize_score_dir(ROOT / tta_root, "vit_patchcore_max_scores"),
        )

    fedjam_root = ROOT / "20260813_frequency_response_memory_fedjam_full"
    for shot in (1, 2, 4):
        key = f"fedjam_{shot}shot_scores.npz"
        baseline = summarize_single_score_file(
            fedjam_root / "no_tta_merged/scores" / key,
            "confidence_gated_score",
            nonzero_is_anomaly=True,
        )
        for strength in (1, 3, 5):
            add_pair(
                rows,
                f"FedJam {shot}-shot response strength={strength}",
                baseline,
                summarize_single_score_file(
                    fedjam_root / f"response_s{strength}_merged/scores" / key,
                    "confidence_gated_score",
                    nonzero_is_anomaly=True,
                ),
            )

    ofdma_root = ROOT / "20260813_ofdma_spectral_response_tta"
    for shot in (1, 2, 4):
        add_pair(
            rows,
            f"OFDMA {shot}-shot",
            summarize_ofdma(ofdma_root / "full_no_tta", shot),
            summarize_ofdma(ofdma_root / "full_spectral_response", shot),
        )

    output = ROOT / "20260815_tta_three_metrics.json"
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print("dataset | AUROC delta | AUPRC delta | FPR95 delta")
    for row in rows:
        print(
            f"{row['dataset']} | {row['delta_auroc']:+.4f} | "
            f"{row['delta_auprc']:+.4f} | {row['delta_fpr95']:+.4f}"
        )


if __name__ == "__main__":
    main()
