#!/usr/bin/env python
"""Formal PromptAD + target-normal-gallery image-score fusion evaluation.

This is a no-training evaluation. It loads the formal pooled RF PromptAD
checkpoint, exports per-image PromptAD image scores, computes target-normal
gallery distance scores on the same samples, and evaluates their fusion.
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
    build_gallery,
    build_train_loader,
    to_model_input,
)
from utils.training_utils import setup_seed


def safe_auc(labels, scores) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores) * 100.0)


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    std = float(np.std(x))
    if std < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - float(np.mean(x))) / std


@torch.no_grad()
def encode_target_normal_gallery(model, train_loader, device):
    features = []
    model.eval_mode()
    for imgs, mask, label, name, img_type in tqdm(train_loader, desc="Encode target normal gallery", leave=False):
        batch = to_model_input(model, imgs, device, rgb_from_bgr=True)
        cls_feature = model.encode_image(batch)[0]
        features.append(F.normalize(cls_feature.float(), dim=-1).cpu())
    if not features:
        return torch.empty(0)
    return F.normalize(torch.cat(features, dim=0), dim=-1)


def topk_distance(cls_features: torch.Tensor, normal_gallery: torch.Tensor, k: int) -> np.ndarray:
    sim = cls_features @ normal_gallery.T
    kk = max(1, min(int(k), normal_gallery.shape[0]))
    topk_sim = torch.topk(sim, k=kk, dim=1).values.mean(dim=1)
    return (1.0 - topk_sim).cpu().numpy()


def fuse_scores(promptad_scores: np.ndarray, normal_scores: np.ndarray, lambdas):
    out = {}
    promptad_scores = np.asarray(promptad_scores, dtype=np.float32)
    normal_scores = np.asarray(normal_scores, dtype=np.float32)
    for lam in lambdas:
        out[f"fusion_raw_lam{lam:g}"] = promptad_scores + float(lam) * normal_scores
        out[f"fusion_minmax_lam{lam:g}"] = minmax(promptad_scores) + float(lam) * minmax(normal_scores)
        out[f"fusion_z_lam{lam:g}"] = zscore(promptad_scores) + float(lam) * zscore(normal_scores)
    return out


def load_checkpoint(model, checkpoint):
    state = torch.load(checkpoint, map_location="cpu")
    current = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            filtered[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[checkpoint] loaded {checkpoint}")
    print(f"[checkpoint] loaded_keys={len(filtered)} skipped_shape_mismatch={skipped}")
    print(f"[checkpoint] missing={len(missing)} unexpected={len(unexpected)}")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def attach_formal_baseline(rows, baseline_csv):
    if not baseline_csv or not Path(baseline_csv).exists():
        return rows
    base = pd.read_csv(baseline_csv).rename(columns={"i_roc": "formal_baseline_auc"})
    key_cols = ["dataset", "scene", "jsr"]
    merged = pd.DataFrame(rows).merge(base[key_cols + ["formal_baseline_auc"]], on=key_cols, how="left")
    for col in [c for c in merged.columns if c.endswith("_auc") and c != "formal_baseline_auc"]:
        merged[f"delta_{col}_vs_formal"] = merged[col] - merged["formal_baseline_auc"]
    return merged.to_dict("records")


@torch.no_grad()
def evaluate(model, eval_loaders, normal_gallery, args, device):
    model.eval_mode()
    # Do not rebuild text features here. The formal baseline checkpoint stores
    # the selected text feature gallery; rebuilding from the current prompt text
    # would silently change old-checkpoint scoring when prompt templates evolved.
    rows = []
    score_root = Path(args.output_root) / "scores"
    score_root.mkdir(parents=True, exist_ok=True)

    for signal, scene, jsr, loader in eval_loaders:
        labels = []
        names = []
        promptad_scores = []
        normal_scores = []

        for data, mask, label, name, img_type in tqdm(loader, desc=f"Eval fusion cls {signal}/{scene}/{jsr}", leave=False):
            data_t = to_model_input(model, data, device, rgb_from_bgr=False)
            visual_features = model.encode_image(data_t)
            score_img, _score_map = model.score_cached(visual_features, "cls")
            cls_features = F.normalize(visual_features[0].float(), dim=-1).cpu()
            promptad_scores.extend([float(x) for x in score_img])
            normal_scores.extend([float(x) for x in topk_distance(cls_features, normal_gallery, args.normal_topk)])
            labels.extend([int(x) for x in label.numpy().tolist()])
            names.extend(list(name))

        promptad_scores_np = np.asarray(promptad_scores, dtype=np.float32)
        normal_scores_np = np.asarray(normal_scores, dtype=np.float32)
        labels_np = np.asarray(labels, dtype=np.int32)
        fused = fuse_scores(promptad_scores_np, normal_scores_np, args.lambdas)

        row = {
            "method": "promptad_normal_gallery_fusion",
            "task": "cls",
            "dataset": signal,
            "scene": scene,
            "jsr": jsr,
            "promptad_rescore_auc": safe_auc(labels, promptad_scores_np),
            f"normal_dist_top{args.normal_topk}_auc": safe_auc(labels, normal_scores_np),
        }
        for key, values in fused.items():
            row[f"{key}_auc"] = safe_auc(labels, values)
        rows.append(row)

        np.savez_compressed(
            score_root / f"{signal}-{scene}-{jsr}-scores.npz",
            names=np.asarray(names),
            labels=labels_np,
            promptad_scores=promptad_scores_np,
            normal_dist_scores=normal_scores_np,
            **{k: np.asarray(v, dtype=np.float32) for k, v in fused.items()},
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="analysis_outputs/20260628_promptad_normal_gallery_fusion_cls")
    parser.add_argument("--formal-baseline-csv", default="analysis_outputs/20260627_method_funnel/results_cls_pooled_rf_rgb.csv")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--signals", nargs="+", default=SIGNALS, choices=SIGNALS)
    parser.add_argument("--normal-topk", type=int, default=50)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.4])
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--normal-train-ratio", type=float, default=0.75)
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

    if args.checkpoint is None:
        base = pd.read_csv(args.formal_baseline_csv)
        checkpoints = sorted(set(str(x) for x in base["checkpoint"].dropna()))
        if len(checkpoints) != 1:
            raise ValueError(f"Expected exactly one checkpoint in baseline CSV, got {checkpoints}")
        args.checkpoint = checkpoints[0]

    kwargs = vars(args).copy()
    kwargs.update({
        "dataset": "rf_target_test_pool",
        "class_name": "signal",
        "device": device,
        "out_size_h": args.resolution,
        "out_size_w": args.resolution,
    })
    model = PromptAD(**kwargs).to(device)

    train_loader = build_train_loader(args)
    build_gallery(model, train_loader, device)
    load_checkpoint(model, args.checkpoint)
    normal_gallery = encode_target_normal_gallery(model, train_loader, device)
    print(f"[target-normal] gallery={normal_gallery.shape[0]} topk={args.normal_topk}")

    rows = evaluate(model, build_eval_loaders(args), normal_gallery, args, device)
    rows = attach_formal_baseline(rows, args.formal_baseline_csv)

    out_root = Path(args.output_root)
    result_path = out_root / "results_cls_promptad_normal_gallery_fusion.csv"
    write_csv(result_path, rows)

    df = pd.DataFrame(rows)
    summary = {
        "method": "promptad_normal_gallery_fusion",
        "checkpoint": args.checkpoint,
        "formal_baseline_csv": args.formal_baseline_csv,
        "normal_topk": args.normal_topk,
        "target_normal_count": int(normal_gallery.shape[0]),
        "lambdas": args.lambdas,
    }
    for col in [c for c in df.columns if c.endswith("_auc")]:
        summary[f"{col}_macro"] = float(df[col].mean())
    for col in [c for c in df.columns if c.startswith("delta_")]:
        summary[f"{col}_macro"] = float(df[col].mean())

    summary_path = out_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {result_path}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
