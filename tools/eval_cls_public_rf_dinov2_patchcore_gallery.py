#!/usr/bin/env python
"""Evaluate public RF CLS with DINOv2 patch nearest-neighbour memory."""

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
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_cls_dinov2_patchcore_gallery import (
    build_gallery,
    build_transform,
    create_dino_model,
    extract_patch_features,
    robust_cosine_distance_chunked,
    to_dino_input,
)
from tools.eval_cls_public_rf_dual_gallery import (
    PUBLIC_SIGNALS,
    build_eval_loaders,
    build_train_loader,
    parse_jsrs_by_signal,
)
from tools.eval_cls_resnet_gallery_fusion import safe_auc
from tools.eval_cls_vit_patch_gallery import top_ratio_score
from utils.rf_frequency_sampling import NORMAL_SAMPLING_CHOICES
from utils.training_utils import setup_seed


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(model, gallery, eval_loaders, transform, args, device):
    rows = []
    score_root = Path(args.output_root) / "scores"
    score_root.mkdir(parents=True, exist_ok=True)

    for signal, scene, jsr, loader in eval_loaders:
        labels, names = [], []
        max_scores = []
        ratio_scores = {ratio: [] for ratio in args.map_top_ratios}
        for data, mask, label, name, img_type in tqdm(loader, desc=f"Eval public DINOv2 {signal}/{jsr}", leave=False):
            images = to_dino_input(data, transform, device, rgb_from_bgr=True)
            patch_features = extract_patch_features(model, images)
            bsz, num_patches, _ = patch_features.shape
            grid_h = grid_w = int(np.sqrt(num_patches))
            if grid_h * grid_w != num_patches:
                raise RuntimeError(f"DINOv2 patch count is not square: {num_patches}")
            patch_scores = robust_cosine_distance_chunked(
                patch_features.reshape(-1, patch_features.shape[-1]),
                gallery,
                args.gallery_chunk_size,
                args,
            ).reshape(bsz, grid_h, grid_w)
            map_np = patch_scores.detach().cpu().numpy().astype(np.float32)
            max_scores.extend(float(x) for x in map_np.reshape(map_np.shape[0], -1).max(axis=1))
            for ratio in args.map_top_ratios:
                ratio_scores[ratio].extend(float(x) for x in top_ratio_score(map_np, ratio))
            labels.extend(int(x) for x in label.numpy().tolist())
            names.extend(list(name))

        labels_np = np.asarray(labels, dtype=np.int32)
        max_np = np.asarray(max_scores, dtype=np.float32)
        row = {
            "method": "public_rf_dinov2_patchcore_gallery_cls",
            "task": "cls",
            "dataset": signal,
            "scene": scene,
            "jsr": jsr,
            "dino_model": args.dino_model,
            "dino_image_size": args.dino_image_size,
            "coreset_ratio": args.coreset_ratio,
            "coreset_method": args.coreset_method,
            "nn_topk": args.nn_topk,
            "nn_agg": args.nn_agg,
            "num_normal": int((labels_np == 0).sum()),
            "num_abnormal": int((labels_np == 1).sum()),
            "dino_patchcore_max_auc": safe_auc(labels, max_np),
        }
        payload = {
            "names": np.asarray(names),
            "labels": labels_np,
            "dino_patchcore_max_scores": max_np,
        }
        for ratio in args.map_top_ratios:
            key = f"{ratio:g}".replace(".", "p")
            values = np.asarray(ratio_scores[ratio], dtype=np.float32)
            row[f"dino_patchcore_top{key}_auc"] = safe_auc(labels, values)
            payload[f"dino_patchcore_top{key}_scores"] = values
        rows.append(row)
        np.savez_compressed(score_root / f"{signal}-{jsr}-scores.npz", **payload)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Public RF DINOv2 patch memory CLS")
    parser.add_argument("--output-root", default="analysis_outputs/20260705_public_rf_dinov2_patchcore_gallery")
    parser.add_argument("--signals", nargs="+", default=list(PUBLIC_SIGNALS), choices=list(PUBLIC_SIGNALS))
    parser.add_argument("--jsrs-by-signal", default=None)
    parser.add_argument("--normal-sampling", choices=NORMAL_SAMPLING_CHOICES, default="per_frequency")
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--dino-model", default="vit_small_patch14_dinov2")
    parser.add_argument("--dino-image-size", type=int, default=224)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--coreset-ratio", type=float, default=0.5)
    parser.add_argument("--coreset-method", choices=["random", "farthest"], default="farthest")
    parser.add_argument("--gallery-chunk-size", type=int, default=4096)
    parser.add_argument("--nn-topk", type=int, default=5)
    parser.add_argument("--nn-agg", choices=["mean", "weighted", "adaptive"], default="mean")
    parser.add_argument("--nn-weight-temp", type=float, default=0.05)
    parser.add_argument("--adaptive-sim-margin", type=float, default=0.02)
    parser.add_argument("--map-top-ratios", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--k-shot", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0.0 < args.coreset_ratio <= 1.0):
        raise ValueError("--coreset-ratio must be in (0, 1]")
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    transform = build_transform(args.dino_image_size)
    model = create_dino_model(args, device)
    train_loader = build_train_loader(args)
    gallery = build_gallery(model, train_loader, transform, args, device)
    rows = evaluate(model, gallery, build_eval_loaders(args), transform, args, device)

    out_root = Path(args.output_root)
    result_path = out_root / "results_cls_public_rf_dinov2_patchcore_gallery.csv"
    write_csv(result_path, rows)
    df = pd.DataFrame(rows)
    by_signal = df.groupby("dataset")[[c for c in df.columns if c.endswith("_auc")]].mean().reset_index()
    by_signal.to_csv(out_root / "by_signal_public_rf_dinov2_patchcore_gallery.csv", index=False)
    summary = {
        "method": "public_rf_dinov2_patchcore_gallery_cls",
        "normal_sampling": args.normal_sampling,
        "jsrs_by_signal": parse_jsrs_by_signal(args.jsrs_by_signal),
        "dino_model": args.dino_model,
        "dino_image_size": args.dino_image_size,
        "gallery_patch_count": int(gallery.shape[0]),
        "gallery_feature_dim": int(gallery.shape[1]),
        "coreset_ratio": args.coreset_ratio,
        "coreset_method": args.coreset_method,
        "nn_topk": args.nn_topk,
        "nn_agg": args.nn_agg,
        "num_cells": len(rows),
    }
    for col in [c for c in df.columns if c.endswith("_auc")]:
        summary[f"{col}_macro"] = float(df[col].mean())
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Public RF DINOv2 Patch Memory CLS",
        "",
        "Vision-only nearest-neighbour normal patch memory using DINOv2 patch features.",
        "",
        f"- normal sampling: `{args.normal_sampling}`",
        f"- DINOv2 model: `{args.dino_model}`",
        f"- coreset: `{args.coreset_method}` `{args.coreset_ratio}`",
        f"- nn top-k: `{args.nn_topk}`",
        f"- gallery patches: `{gallery.shape[0]}`",
        f"- cells: `{len(rows)}`",
        "",
        "## Macro AUROC",
        "",
    ]
    for key in sorted(k for k in summary if k.endswith("_auc_macro")):
        lines.append(f"- `{key}`: {summary[key]:.4f}")
    lines.extend(["", "## By Signal", "", by_signal.to_markdown(index=False)])
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
