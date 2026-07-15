#!/usr/bin/env python
"""Formal target-normal-gallery image-level evaluation for pooled RF.

This is a no-training evaluation. It builds a target normal gallery from the
same pooled train-normal split used by train_rf_target_pooled_universal.py,
then evaluates image-level AUROC on the same 48 target test cells.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PromptAD import PromptAD
from train_rf_target_pooled_universal import (
    JSR_BY_SIGNAL,
    SIGNALS,
    build_eval_loaders,
    build_train_loader,
    to_model_input,
)
from utils.training_utils import setup_seed


def safe_auc(labels, scores) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores) * 100.0)


@torch.no_grad()
def encode_rf_loader_features(model, loader, device, desc, rgb_from_bgr):
    features, labels = [], []
    model.eval_mode()
    for imgs, mask, label, name, img_type in tqdm(loader, desc=desc, leave=False):
        batch = to_model_input(model, imgs, device, rgb_from_bgr=rgb_from_bgr)
        cls_feature = model.encode_image(batch)[0]
        features.append(F.normalize(cls_feature.float(), dim=-1).cpu())
        labels.extend([int(x) for x in label.numpy().tolist()])
    if not features:
        return torch.empty(0), labels
    return torch.cat(features, dim=0), labels


def gallery_scores(features, normal_gallery, topk_values):
    sim = features @ normal_gallery.T
    out = {}
    for k in topk_values:
        kk = max(1, min(int(k), normal_gallery.shape[0]))
        mean_topk_sim = torch.topk(sim, k=kk, dim=1).values.mean(dim=1).numpy()
        out[f"normal_dist_top{k}"] = 1.0 - mean_topk_sim
    max_sim = sim.max(dim=1).values.numpy()
    out["normal_dist_nn"] = 1.0 - max_sim
    mean_sim = sim.mean(dim=1).numpy()
    out["normal_dist_mean"] = 1.0 - mean_sim
    return out


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def attach_formal_baseline(rows, baseline_csv):
    if not baseline_csv or not Path(baseline_csv).exists():
        return rows
    base = pd.read_csv(baseline_csv)
    base = base.rename(columns={"i_roc": "formal_baseline_auc"})
    key_cols = ["dataset", "scene", "jsr"]
    merged = pd.DataFrame(rows).merge(base[key_cols + ["formal_baseline_auc"]], on=key_cols, how="left")
    for col in [c for c in merged.columns if c.endswith("_auc") and c != "formal_baseline_auc"]:
        merged[f"delta_{col}_vs_formal"] = merged[col] - merged["formal_baseline_auc"]
    return merged.to_dict("records")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="analysis_outputs/20260628_target_normal_gallery_cls")
    parser.add_argument("--formal-baseline-csv", default="analysis_outputs/20260627_method_funnel/results_cls_pooled_rf_rgb.csv")
    parser.add_argument("--signals", nargs="+", default=SIGNALS, choices=SIGNALS)
    parser.add_argument("--normal-topk", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--normal-sampling", choices=["all", "frequency_one_per_band"], default="all")
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument("--img-resize", type=int, default=240)
    parser.add_argument("--img-cropsize", type=int, default=240)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--backbone", default="ViT-B-16-plus-240")
    parser.add_argument("--pretrained_dataset", default="laion400m_e32")
    parser.add_argument("--prompt-mode", default="rf")
    parser.add_argument("--input-mode", default="rgb")
    parser.add_argument("--text-prototype-mode", default="single")
    parser.add_argument("--cls-score-mode", default="text_only")
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=3)
    parser.add_argument("--n_pro_ab", type=int, default=4)
    parser.add_argument("--use-cpu", type=int, default=0)
    args = parser.parse_args()

    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cpu" if args.use_cpu else "cuda:0"
    out_root = Path(args.output_root)

    model_kwargs = vars(args).copy()
    model_kwargs.update({
        "dataset": "rf_target_test_pool",
        "class_name": "signal",
        "device": device,
        "out_size_h": args.resolution,
        "out_size_w": args.resolution,
    })
    model = PromptAD(**model_kwargs).to(device)

    train_loader = build_train_loader(args)
    normal_gallery, _ = encode_rf_loader_features(
        model, train_loader, device, "Encode target normal gallery", rgb_from_bgr=True
    )
    normal_gallery = F.normalize(normal_gallery, dim=-1)
    print(f"[target-normal] gallery={normal_gallery.shape[0]}")

    rows = []
    for signal, scene, jsr, loader in build_eval_loaders(args):
        features, labels = encode_rf_loader_features(
            model, loader, device, f"Eval normal gallery cls {signal}/{scene}/{jsr}", rgb_from_bgr=False
        )
        score_dict = gallery_scores(features, normal_gallery, args.normal_topk)
        row = {
            "method": "target_normal_gallery",
            "task": "cls",
            "dataset": signal,
            "scene": scene,
            "jsr": jsr,
        }
        for name, values in score_dict.items():
            row[f"{name}_auc"] = safe_auc(labels, values)
        rows.append(row)

    rows = attach_formal_baseline(rows, args.formal_baseline_csv)
    fieldnames = list(rows[0].keys())
    result_path = out_root / "results_cls_target_normal_gallery.csv"
    write_csv(result_path, rows, fieldnames)

    df = pd.DataFrame(rows)
    auc_cols = [c for c in df.columns if c.endswith("_auc")]
    summary = {
        "method": "target_normal_gallery",
        "target_normal_count": int(normal_gallery.shape[0]),
        "formal_baseline_csv": args.formal_baseline_csv,
    }
    for col in auc_cols + ["formal_baseline_auc"]:
        if col in df.columns:
            summary[f"{col}_macro"] = float(df[col].mean())
    for col in [c for c in df.columns if c.startswith("delta_")]:
        summary[f"{col}_macro"] = float(df[col].mean())

    np.savez_compressed(out_root / "target_normal_gallery_features.npz", features=normal_gallery.numpy())
    summary_path = out_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {result_path}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
