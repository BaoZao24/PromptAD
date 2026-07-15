#!/usr/bin/env python
"""Evaluate DINOv2 patch nearest-neighbour memory for RF few-shot CLS.

This is a vision-only baseline candidate inspired by AnomalyDINO/UniVAD:
few-shot normal images are converted into a patch-feature memory bank, and each
test patch is scored by its distance to the nearest/top-k normal patches.
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
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_cls_vit_patch_gallery import build_selected_train_loader, top_ratio_score
from tools.eval_cls_vit_patchcore_gallery import (
    aggregate_topk_similarity,
    select_gallery_subset,
)
from train_rf_target_pooled_universal import SIGNALS, build_eval_loaders
from utils.training_utils import setup_seed


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def safe_auc(labels, scores) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores) * 100.0)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def to_dino_input(raw_batch, transform, device, rgb_from_bgr=True):
    imgs = []
    for arr in raw_batch:
        arr = arr.numpy()
        if rgb_from_bgr:
            arr = arr[:, :, ::-1]
        imgs.append(transform(Image.fromarray(arr)))
    return torch.stack(imgs, dim=0).to(device)


def create_dino_model(args, device):
    model = timm.create_model(
        args.dino_model,
        pretrained=not args.no_pretrained,
        num_classes=0,
        img_size=args.dino_image_size,
        dynamic_img_size=True,
    )
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def extract_patch_features(model, images):
    tokens = model.forward_features(images)
    if isinstance(tokens, dict):
        if "x_norm_patchtokens" in tokens:
            tokens = tokens["x_norm_patchtokens"]
        elif "x_prenorm" in tokens:
            tokens = tokens["x_prenorm"][:, 1:]
        else:
            raise RuntimeError(f"Unsupported DINOv2 feature dict keys: {list(tokens.keys())}")
    else:
        # timm ViT returns [CLS + patch tokens] for non-register DINOv2 models.
        tokens = tokens[:, 1:]
    return F.normalize(tokens.float(), dim=-1)


def robust_cosine_distance_chunked(probe, gallery, chunk_size, args):
    chunk_size = max(1, int(chunk_size))
    k = max(1, min(int(args.nn_topk), int(gallery.shape[0])))
    best = None
    for start in range(0, gallery.shape[0], chunk_size):
        chunk = gallery[start:start + chunk_size]
        sim = probe @ chunk.t()
        local = torch.topk(sim, k=min(k, sim.shape[1]), dim=-1).values
        best = local if best is None else torch.topk(torch.cat([best, local], dim=-1), k=k, dim=-1).values
    sim_score = aggregate_topk_similarity(best, args)
    return (1.0 - sim_score) / 2.0


@torch.no_grad()
def build_gallery(model, train_loader, transform, args, device):
    patches = []
    for data, mask, label, name, img_type in tqdm(train_loader, desc="Build DINOv2 gallery", leave=False):
        images = to_dino_input(data, transform, device, rgb_from_bgr=True)
        patch_features = extract_patch_features(model, images)
        patches.append(patch_features.reshape(-1, patch_features.shape[-1]))
    gallery = torch.cat(patches, dim=0)
    gallery = F.normalize(gallery.float(), dim=-1)
    if args.coreset_ratio < 1.0:
        idx, _ = select_gallery_subset(gallery, args)
        gallery = gallery[idx]
    return gallery.contiguous()


@torch.no_grad()
def evaluate(model, gallery, eval_loaders, transform, args, device):
    rows = []
    score_dir = Path(args.output_root) / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    for signal, scene, jsr, loader in eval_loaders:
        labels, names = [], []
        max_scores = []
        ratio_scores = {ratio: [] for ratio in args.map_top_ratios}

        for data, mask, label, name, img_type in tqdm(loader, desc=f"Eval DINOv2 {signal}/{scene}/{jsr}", leave=False):
            images = to_dino_input(data, transform, device, rgb_from_bgr=False)
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
            "method": "dinov2_patchcore_gallery_cls",
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
        np.savez_compressed(score_dir / f"{signal}-{scene}-{jsr}-scores.npz", **payload)

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv2 patch memory RF few-shot CLS")
    parser.add_argument("--output-root", default="analysis_outputs/20260705_dinov2_patchcore_gallery_self")
    parser.add_argument("--signals", nargs="+", default=SIGNALS, choices=SIGNALS)
    parser.add_argument("--normal-sampling", choices=["all", "first", "frequency_one_per_band"], default="frequency_one_per_band")
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
    parser.add_argument("--batch-size", type=int, default=64)
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
    train_loader, train_samples = build_selected_train_loader(args)
    gallery = build_gallery(model, train_loader, transform, args, device)
    rows = evaluate(model, gallery, build_eval_loaders(args), transform, args, device)

    out_root = Path(args.output_root)
    write_csv(out_root / "results_cls_dinov2_patchcore_gallery.csv", rows)
    df = pd.DataFrame(rows)
    summary = {
        "method": "dinov2_patchcore_gallery_cls",
        "dino_model": args.dino_model,
        "dino_image_size": args.dino_image_size,
        "normal_sampling": args.normal_sampling,
        "selected_normal_count": len(train_samples),
        "selected_normals": [sample[0] for sample in train_samples],
        "gallery_patch_count": int(gallery.shape[0]),
        "gallery_feature_dim": int(gallery.shape[1]),
        "coreset_ratio": args.coreset_ratio,
        "coreset_method": args.coreset_method,
        "nn_topk": args.nn_topk,
        "nn_agg": args.nn_agg,
        "map_top_ratios": args.map_top_ratios,
        "num_cells": len(rows),
    }
    for col in [c for c in df.columns if c.endswith("_auc")]:
        summary[f"{col}_macro"] = float(df[col].mean())

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    lines = [
        "# DINOv2 Patch Memory CLS",
        "",
        "Vision-only nearest-neighbour normal patch memory using DINOv2 patch features.",
        "",
        f"- normal sampling: `{args.normal_sampling}`",
        f"- selected normal images: `{len(train_samples)}`",
        f"- DINOv2 model: `{args.dino_model}`",
        f"- DINOv2 image size: `{args.dino_image_size}`",
        f"- coreset: `{args.coreset_method}` `{args.coreset_ratio}`",
        f"- nn top-k: `{args.nn_topk}`",
        f"- gallery patches: `{gallery.shape[0]}`",
        "",
        "## Macro AUROC",
        "",
    ]
    for key in sorted(k for k in summary if k.endswith("_auc_macro")):
        lines.append(f"- `{key}`: {summary[key]:.4f}")
    lines.extend(["", "## By Signal", ""])
    by_signal = df.groupby("dataset")[[c for c in df.columns if c.endswith("_auc")]].mean().reset_index()
    lines.append(by_signal.to_markdown(index=False))
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

