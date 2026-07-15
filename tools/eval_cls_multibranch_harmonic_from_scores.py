#!/usr/bin/env python
"""Offline CLS harmonic fusion from saved text, ViT-gallery, and CNN-gallery scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def safe_auc(labels, scores) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores) * 100.0)


def harmonic(*scores, eps=1e-12):
    clipped = [np.clip(np.asarray(score, dtype=np.float32), eps, None) for score in scores]
    denom = np.zeros_like(clipped[0], dtype=np.float32)
    for score in clipped:
        denom += 1.0 / score
    return 1.0 / denom


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def attach_formal_baseline(rows, baseline_csv):
    if not baseline_csv or not Path(baseline_csv).exists():
        return rows
    base = pd.read_csv(baseline_csv).rename(columns={"i_roc": "formal_baseline_auc"})
    merged = pd.DataFrame(rows).merge(
        base[["dataset", "scene", "jsr", "formal_baseline_auc"]],
        on=["dataset", "scene", "jsr"],
        how="left",
    )
    for col in [c for c in merged.columns if c.endswith("_auc") and c != "formal_baseline_auc"]:
        merged[f"delta_{col}_vs_formal"] = merged[col] - merged["formal_baseline_auc"]
    return merged.to_dict("records")


def split_score_name(path: Path):
    stem = path.name.removesuffix("-scores.npz")
    dataset, scene, jsr = stem.split("-", 2)
    return dataset, scene, jsr


def evaluate(args):
    vit_dir = Path(args.vit_score_dir)
    cnn_dir = Path(args.cnn_score_dir)
    rows = []
    score_root = Path(args.output_root) / "scores"
    score_root.mkdir(parents=True, exist_ok=True)

    for vit_path in sorted(vit_dir.glob("*-scores.npz")):
        dataset, scene, jsr = split_score_name(vit_path)
        cnn_path = cnn_dir / vit_path.name
        if not cnn_path.exists():
            raise FileNotFoundError(f"Missing CNN score file for {vit_path.name}: {cnn_path}")

        vit = np.load(vit_path)
        cnn = np.load(cnn_path)
        labels = vit["labels"].astype(np.int32)
        text = vit["text_scores"].astype(np.float32)
        vit_max = vit["vit_map_max_scores"].astype(np.float32)
        vit_top001 = vit["vit_map_top0p01_scores"].astype(np.float32)
        formal = vit["harmonic_text_vit_max"].astype(np.float32)
        cnn_l2_max = cnn["resnet18_layer2_max_scores"].astype(np.float32)
        cnn_l2_top001 = cnn["resnet18_layer2_top0.01_scores"].astype(np.float32)
        cnn_l3_top01 = cnn["resnet18_layer3_top0.1_scores"].astype(np.float32)

        score_bank = {
            "text_auc": text,
            "vit_max_auc": vit_max,
            "vit_top0p01_auc": vit_top001,
            "formal_text_vit_max_auc": formal,
            "cnn_layer2_max_auc": cnn_l2_max,
            "cnn_layer2_top0p01_auc": cnn_l2_top001,
            "cnn_layer3_top0p1_auc": cnn_l3_top01,
            "harmonic_text_vitmax_cnn_l2max_auc": harmonic(text, vit_max, cnn_l2_max),
            "harmonic_text_vitmax_cnn_l2top0p01_auc": harmonic(text, vit_max, cnn_l2_top001),
            "harmonic_text_vittop0p01_cnn_l2top0p01_auc": harmonic(text, vit_top001, cnn_l2_top001),
            "harmonic_formal_cnn_l2max_auc": harmonic(formal, cnn_l2_max),
            "harmonic_formal_cnn_l2top0p01_auc": harmonic(formal, cnn_l2_top001),
            "harmonic_formal_cnn_l3top0p1_auc": harmonic(formal, cnn_l3_top01),
        }
        row = {
            "method": "cls_multibranch_harmonic_from_scores",
            "task": "cls",
            "dataset": dataset,
            "scene": scene,
            "jsr": jsr,
        }
        payload = {"labels": labels, "names": vit["names"]}
        for key, values in score_bank.items():
            row[key] = safe_auc(labels, values)
            payload[key.removesuffix("_auc")] = np.asarray(values, dtype=np.float32)

        rows.append(row)
        np.savez_compressed(score_root / vit_path.name, **payload)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="analysis_outputs/20260703_cls_multibranch_harmonic_global4")
    parser.add_argument("--vit-score-dir", required=True)
    parser.add_argument("--cnn-score-dir", required=True)
    parser.add_argument("--formal-baseline-csv", default="")
    args = parser.parse_args()

    rows = evaluate(args)
    rows = attach_formal_baseline(rows, args.formal_baseline_csv)
    out_root = Path(args.output_root)
    result_path = out_root / "results_cls_multibranch_harmonic.csv"
    write_csv(result_path, rows)

    df = pd.DataFrame(rows)
    summary = {
        "method": "cls_multibranch_harmonic_from_scores",
        "vit_score_dir": args.vit_score_dir,
        "cnn_score_dir": args.cnn_score_dir,
        "formal_baseline_csv": args.formal_baseline_csv,
    }
    for col in [c for c in df.columns if c.endswith("_auc")]:
        summary[f"{col}_macro"] = float(df[col].mean())
    for col in [c for c in df.columns if c.startswith("delta_")]:
        summary[f"{col}_macro"] = float(df[col].mean())

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {result_path}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
