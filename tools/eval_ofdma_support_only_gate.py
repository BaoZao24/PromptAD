#!/usr/bin/env python3
"""Build the formal OFDMA safe support-only score bundle.

Frozen OFDMA ViT/CNN branch outputs are reused.  Only the final calibration /
fusion is recomputed, so this entry does not retrain or recompute the visual
branches.  The safe support-only gate is the formal score; naive and
transductive variants remain protocol comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_support_only_rf_gate import empirical_support_rank, support_minmax


VIT = "ours_vit"
CNN = "ours_cnn"
FORMAL = "confidence_gated_dual_visual"
NAIVE = "support_only_naive"
SAFE = "support_only_safe"
METHODS = ("vit_only", "cnn_only", FORMAL, NAIVE, SAFE)
JAMMER_TYPES = ("barrage", "deceptive", "pilot", "sweep", "random_hop")


def metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels, scores) * 100.0),
        "auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if len(reached) else 100.0,
    }


def safe_support_gate(
    vit: np.ndarray,
    cnn: np.ndarray,
    vit_ref: np.ndarray,
    cnn_ref: np.ndarray,
) -> np.ndarray:
    """CNN-only positive correction from the RF conservative candidate.

    The ViT raw score is retained as the base.  Support median/IQR establish
    each branch's scale; CNN contributes only when its support CDF rank is at
    least 0.8 and its smooth evidence exceeds ViT's.
    """

    def robust_stats(reference: np.ndarray) -> tuple[float, float]:
        q25, q50, q75 = np.quantile(reference, [0.25, 0.5, 0.75])
        return float(q50), max(float(q75 - q25), 1e-6)

    vit_median, vit_iqr = robust_stats(vit_ref)
    cnn_median, cnn_iqr = robust_stats(cnn_ref)
    vit_z = (np.asarray(vit, dtype=np.float64) - vit_median) / vit_iqr
    cnn_z = (np.asarray(cnn, dtype=np.float64) - cnn_median) / cnn_iqr
    vit_prob = expit(vit_z / 2.5)
    cnn_prob = expit(cnn_z / 2.5)
    correction = np.clip(cnn_prob - vit_prob, 0.0, 1.0)
    correction *= (
        empirical_support_rank(cnn, cnn_ref) >= 0.8
    ).astype(np.float64)
    return np.asarray(vit, dtype=np.float64) + 2.5 * vit_iqr * correction


def naive_support_gate(
    vit: np.ndarray,
    cnn: np.ndarray,
    vit_ref: np.ndarray,
    cnn_ref: np.ndarray,
) -> np.ndarray:
    vit_norm = support_minmax(vit, vit_ref)
    cnn_norm = support_minmax(cnn, cnn_ref)
    cnn_rank = empirical_support_rank(cnn, cnn_ref)
    rank_conf = np.clip((cnn_rank - 0.5) / 0.5, 0.0, 1.0)
    advantage = np.clip(cnn_norm - vit_norm, 0.0, 1.0)
    gate = rank_conf * advantage
    return 1.0 - (1.0 - vit_norm) * (1.0 - gate * cnn_norm)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_name(path: Path) -> tuple[str, int]:
    match = re.fullmatch(r"(test_\d+)_(\d+)shot_observation_scores", path.stem)
    if match is None:
        raise ValueError(f"Unexpected OFDMA score name: {path.name}")
    return match.group(1), int(match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-score-root", type=Path, required=True)
    parser.add_argument("--support-reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    per_scene = []
    output_scores = args.output_root / "scores"
    output_scores.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.test_score_root.glob("test_*shot_observation_scores.npz")):
        scene, shot = parse_name(path)
        reference_path = args.support_reference_root / f"{scene}_{shot}shot.npz"
        if not reference_path.is_file():
            raise FileNotFoundError(f"Missing support reference: {reference_path}")
        with np.load(path, allow_pickle=True) as data:
            labels = np.asarray(data["labels"], dtype=np.int32)
            jammer_types = np.asarray(data["jammer_types"])
            vit = np.asarray(data[VIT], dtype=np.float64)
            cnn = np.asarray(data[CNN], dtype=np.float64)
            formal = np.asarray(data[FORMAL], dtype=np.float64)
            target_scene_ids = np.asarray(data["target_scene_ids"])
            observation_ids = np.asarray(data["observation_ids"])
        with np.load(reference_path, allow_pickle=True) as reference:
            vit_ref = np.asarray(reference[VIT], dtype=np.float64)
            cnn_ref = np.asarray(reference[CNN], dtype=np.float64)
        scores = {
            "vit_only": vit,
            "cnn_only": cnn,
            FORMAL: formal,
            NAIVE: naive_support_gate(vit, cnn, vit_ref, cnn_ref),
            SAFE: safe_support_gate(vit, cnn, vit_ref, cnn_ref),
        }
        np.savez_compressed(
            output_scores / path.name,
            target_scene_ids=target_scene_ids,
            observation_ids=observation_ids,
            labels=labels,
            jammer_types=jammer_types,
            **scores,
            confidence_gated_score=scores[SAFE].astype(np.float32),
        )
        for method, values in scores.items():
            for scope in ("overall", *JAMMER_TYPES):
                mask = (
                    np.ones(len(labels), dtype=bool)
                    if scope == "overall"
                    else (jammer_types == "no jammer") | (jammer_types == scope)
                )
                per_scene.append(
                    {
                        "row_type": "target_scene",
                        "target_scene_id": scene,
                        "shot": shot,
                        "scope": scope,
                        "method": method,
                        "num_observations": int(mask.sum()),
                        **metric(labels[mask], values[mask]),
                    }
                )

    if not per_scene:
        raise RuntimeError("No OFDMA observation score files found")
    macro = []
    for shot in sorted({row["shot"] for row in per_scene}):
        for scope in ("overall", *JAMMER_TYPES):
            for method in METHODS:
                selected = [
                    row
                    for row in per_scene
                    if row["shot"] == shot
                    and row["scope"] == scope
                    and row["method"] == method
                ]
                macro.append(
                    {
                        "row_type": "scene_macro",
                        "target_scene_id": "ALL",
                        "shot": shot,
                        "scope": scope,
                        "method": method,
                        "num_scenes": len(selected),
                        **{
                            key: float(np.mean([row[key] for row in selected]))
                            for key in ("auroc", "auprc", "fpr95")
                        },
                    }
                )

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "results_per_scene.csv", per_scene)
    write_csv(args.output_root / "results_macro.csv", macro)
    formal_rows = []
    for row in per_scene + macro:
        if row["method"] != SAFE:
            continue
        formal_row = dict(row)
        formal_row["method"] = FORMAL
        if formal_row["row_type"] == "scene_macro":
            formal_row["num_observations"] = int(
                sum(
                    item["num_observations"]
                    for item in per_scene
                    if item["method"] == SAFE
                    and item["shot"] == row["shot"]
                    and item["scope"] == row["scope"]
                )
            )
            formal_row.pop("num_scenes", None)
        formal_rows.append(formal_row)
    write_csv(args.output_root / "results.csv", formal_rows)
    summary = {
        str(shot): {
            row["method"]: {
                key: row[key]
                for key in ("auroc", "auprc", "fpr95")
            }
            for row in macro
            if row["shot"] == shot and row["scope"] == "overall"
        }
        for shot in sorted({row["shot"] for row in macro})
    }
    protocol = {
        "method": "confidence_gated_dual_visual",
        "formal_method": "safe_support_only",
        "test_score_root": str(args.test_score_root),
        "support_reference_root": str(args.support_reference_root),
        "support_only": True,
        "uses_test_batch_statistics": False,
        "uses_test_labels_for_scoring": False,
        "reference_views": [
            "time_shift_up",
            "time_shift_down",
            "time_shift_up_large",
            "time_shift_down_large",
        ],
        "support_reference_level": "observation after maximum over 21 SUs",
        "naive_gate": "support min-max plus support empirical CNN CDF, q=0.5",
        "safe_gate": "raw ViT base; support median/IQR sigmoid residual, CNN CDF >=0.8, temperature=2.5, alpha=2.5",
        "formal_gate_reused": False,
        "num_score_files": len({(row["target_scene_id"], row["shot"]) for row in per_scene}),
    }
    (args.output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "README.md").write_text(
        "# OFDMA support-only gate evaluation\n\n"
        "Frozen OFDMA ViT/CNN observation scores are reused. The formal Ours score is the safe support-only gate, calibrated only from held-out normal support views; no test batch statistics are used. The transductive gate is retained only for protocol comparison.\n\n"
        "Reference generation:\n\n"
        "```bash\n"
        "python tools/eval_cls_ofdma_target_scene_ours.py \\\n"
        "  --dataset-root /mnt/data/wangbei/data/ofdma-target-scene-coldstart-v2-realistic \\\n"
        "  --split test --shots 1 2 4 --support-reference-only \\\n"
        "  --output-root <support-reference-output>\n"
        "```\n\n"
        "Re-scoring:\n\n"
        "```bash\n"
        "python tools/eval_ofdma_support_only_gate.py \\\n"
        "  --test-score-root analysis_outputs/20260731_ofdma_target_scene_ours_dual_visual_formal/scores \\\n"
        "  --support-reference-root analysis_outputs/20260801_ofdma_support_only_refs_combined/support_reference \\\n"
        "  --output-root analysis_outputs/20260801_ofdma_support_only_probe\n"
        "```\n\n"
        "The formal bundle is `results.csv`; protocol comparison rows are in `results_macro.csv` and `results_per_scene.csv`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
