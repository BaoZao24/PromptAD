#!/usr/bin/env python
"""Evaluate learned text evidence fused with ViT and CNN normal memories.

This is an ablation tool for the full-shot RF protocol.  It consumes aligned
per-image score files and reports fixed, weight-free fusion rules.  It never
uses test labels to normalize or gate scores; labels are used only for AUROC.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores) * 100.0)


def minmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def name_to_index(names: np.ndarray) -> dict[str, int]:
    return {str(name): idx for idx, name in enumerate(names)}


def align(vit_npz, cnn_npz):
    vit_names = np.asarray(vit_npz["names"]).astype(str)
    cnn_names = np.asarray(cnn_npz["names"]).astype(str)
    vit_by_name = name_to_index(vit_names)
    cnn_idx = []
    vit_idx = []
    for index, name in enumerate(cnn_names):
        if name not in vit_by_name:
            raise RuntimeError(f"Cannot align sample {name}")
        cnn_idx.append(index)
        vit_idx.append(vit_by_name[name])
    vit_idx = np.asarray(vit_idx, dtype=np.int64)
    cnn_idx = np.asarray(cnn_idx, dtype=np.int64)
    labels = np.asarray(vit_npz["labels"])[vit_idx].astype(np.int32)
    if not np.array_equal(labels, np.asarray(cnn_npz["labels"])[cnn_idx].astype(np.int32)):
        raise RuntimeError("Label mismatch after score alignment")
    return vit_idx, cnn_idx, labels


def parse_cell(path: Path) -> tuple[str, str, str]:
    parts = path.name.removesuffix("-scores.npz").split("-", 2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected RF score filename: {path.name}")
    return tuple(parts)  # type: ignore[return-value]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args) -> list[dict]:
    vit_dir = Path(args.vit_score_dir)
    cnn_dir = Path(args.cnn_score_dir)
    output_dir = Path(args.output_root)
    score_dir = output_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for vit_path in sorted(vit_dir.glob("*-scores.npz")):
        cnn_path = cnn_dir / vit_path.name
        if not cnn_path.exists():
            raise FileNotFoundError(f"Missing CNN scores for {vit_path.name}")
        vit_npz = np.load(vit_path, allow_pickle=True)
        cnn_npz = np.load(cnn_path, allow_pickle=True)
        vit_idx, cnn_idx, labels = align(vit_npz, cnn_npz)

        text = np.asarray(vit_npz[args.text_key], dtype=np.float64)[vit_idx]
        vit = np.asarray(vit_npz[args.vit_key], dtype=np.float64)[vit_idx]
        cnn = np.asarray(cnn_npz[args.cnn_key], dtype=np.float64)[cnn_idx]

        # Grouped mean-max already produces an abnormal probability in [0, 1].
        # Keep that calibrated text probability intact; only distance scores
        # need per-cell scale alignment for this exploratory fusion.
        text_probability = np.clip(text, 0.0, 1.0)
        vit_n, cnn_n = minmax(vit), minmax(cnn)
        visual_max = np.maximum(vit_n, cnn_n)
        visual_or = 1.0 - (1.0 - vit_n) * (1.0 - cnn_n)
        text_visual_max = np.maximum(text_probability, visual_max)
        text_visual_or = 1.0 - (1.0 - text_probability) * (1.0 - visual_or)

        # Preserve confident learned text decisions.  Visual evidence only
        # contributes strongly when text is close to its decision boundary.
        text_confidence = 2.0 * np.abs(text_probability - 0.5)
        visual_gate = 1.0 - np.clip(text_confidence, 0.0, 1.0)
        text_gated_visual_or = 1.0 - (1.0 - text_probability) * (1.0 - visual_or * visual_gate)

        signal, scene, jsr = parse_cell(vit_path)
        score_bank = {
            "text_auc": safe_auc(labels, text),
            "vit_auc": safe_auc(labels, vit),
            "cnn_auc": safe_auc(labels, cnn),
            "visual_max_auc": safe_auc(labels, visual_max),
            "visual_or_auc": safe_auc(labels, visual_or),
            "text_visual_max_auc": safe_auc(labels, text_visual_max),
            "text_visual_or_auc": safe_auc(labels, text_visual_or),
            "text_gated_visual_or_auc": safe_auc(labels, text_gated_visual_or),
        }
        rows.append({
            "dataset": signal,
            "scene": scene,
            "jsr": jsr,
            "n": int(len(labels)),
            **score_bank,
        })
        np.savez_compressed(
            score_dir / vit_path.name,
            names=np.asarray(vit_npz["names"])[vit_idx],
            labels=labels,
            text_scores=text.astype(np.float32),
            text_confidence=text_confidence.astype(np.float32),
            vit_scores=vit.astype(np.float32),
            cnn_scores=cnn.astype(np.float32),
            text_visual_max=text_visual_max.astype(np.float32),
            text_visual_or=text_visual_or.astype(np.float32),
            text_gated_visual_or=text_gated_visual_or.astype(np.float32),
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vit-score-dir", required=True)
    parser.add_argument("--cnn-score-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--text-key", default="text_scores")
    parser.add_argument("--vit-key", default="vit_patchcore_max_scores")
    parser.add_argument("--cnn-key", default="resnet18_layer3_top0.1_scores")
    args = parser.parse_args()

    rows = evaluate(args)
    if not rows:
        raise RuntimeError("No score files found")
    output_root = Path(args.output_root)
    write_csv(output_root / "per_cell_text_dual_visual_fusion.csv", rows)
    metric_keys = [key for key in rows[0] if key.endswith("_auc")]
    macro = {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}
    summary = {
        "method": "learned_text_dual_visual_fusion",
        "num_cells": len(rows),
        "macro": macro,
        "score_keys": {
            "text": args.text_key,
            "vit": args.vit_key,
            "cnn": args.cnn_key,
        },
        "note": "Fusion uses no test labels. Scores are min-max aligned per test cell; this is exploratory, not support-calibrated final scoring.",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Learned Text + Dual Visual Fusion",
        "",
        "Per-cell min-max alignment is label-free but transductive; this is an exploratory fusion result.",
        "",
        "| score | macro Image-AUROC |",
        "|---|---:|",
    ]
    for key, value in macro.items():
        lines.append(f"| {key.removesuffix('_auc')} | {value:.4f} |")
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
