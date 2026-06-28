#!/usr/bin/env python
"""Evaluate CLS scores fused with a ResNet normal patch-feature gallery."""

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
from tools.eval_seg_resnet_gallery_fusion import (
    ResNet18LocalEncoder,
    build_resnet_gallery,
    flatten_feature_map,
    load_checkpoint,
)
from train_rf_target_pooled_universal import (
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


@torch.no_grad()
def resnet_patch_image_scores(feature_map: torch.Tensor, gallery: torch.Tensor, chunk_size: int, top_ratio: float):
    n, c, h, w = feature_map.shape
    probes = flatten_feature_map(F.normalize(feature_map.float(), dim=1))
    scores = []
    for start in range(0, probes.shape[0], chunk_size):
        chunk = probes[start : start + chunk_size]
        sim = chunk @ gallery.t()
        scores.append((1.0 - sim.max(dim=1).values).cpu())
    patch_scores = torch.cat(scores, dim=0).reshape(n, h * w)
    k = max(1, int(round(patch_scores.shape[1] * float(top_ratio))))
    return patch_scores.topk(k, dim=1).values.mean(dim=1).numpy()


def fuse_scores(promptad_scores: np.ndarray, visual_scores: np.ndarray, lambdas):
    out = {}
    promptad_scores = np.asarray(promptad_scores, dtype=np.float32)
    visual_scores = np.asarray(visual_scores, dtype=np.float32)
    for lam in lambdas:
        out[f"fusion_raw_lam{lam:g}"] = promptad_scores + float(lam) * visual_scores
        out[f"fusion_minmax_lam{lam:g}"] = minmax(promptad_scores) + float(lam) * minmax(visual_scores)
        out[f"fusion_z_lam{lam:g}"] = zscore(promptad_scores) + float(lam) * zscore(visual_scores)
    return out


@torch.no_grad()
def evaluate(model, encoder, galleries, eval_loaders, args, device):
    model.eval_mode()
    encoder.eval()
    rows = []
    score_root = Path(args.output_root) / "scores"
    score_root.mkdir(parents=True, exist_ok=True)

    for signal, scene, jsr, loader in eval_loaders:
        labels = []
        names = []
        promptad_scores = []
        layer_scores = {layer: [] for layer in args.resnet_layers}

        for data, mask, label, name, img_type in tqdm(loader, desc=f"Eval ResNet gallery cls {signal}/{scene}/{jsr}", leave=False):
            data_t = to_model_input(model, data, device, rgb_from_bgr=False)
            visual_features = model.encode_image(data_t)
            score_img, _score_map = model.score_cached(visual_features, "cls")
            promptad_scores.extend([float(x) for x in score_img])

            raw = data.permute(0, 3, 1, 2).to(device, non_blocking=True)
            cnn_features = encoder(raw)
            for layer in args.resnet_layers:
                values = resnet_patch_image_scores(
                    cnn_features[layer],
                    galleries[layer],
                    args.distance_chunk_size,
                    args.image_top_ratio,
                )
                layer_scores[layer].extend([float(x) for x in values])

            labels.extend([int(x) for x in label.numpy().tolist()])
            names.extend(list(name))

        labels_np = np.asarray(labels, dtype=np.int32)
        promptad_np = np.asarray(promptad_scores, dtype=np.float32)
        row = {
            "method": "cls_resnet_gallery_fusion",
            "task": "cls",
            "dataset": signal,
            "scene": scene,
            "jsr": jsr,
            "promptad_rescore_auc": safe_auc(labels, promptad_np),
        }
        npz_payload = {
            "names": np.asarray(names),
            "labels": labels_np,
            "promptad_scores": promptad_np,
        }

        for layer in args.resnet_layers:
            visual_np = np.asarray(layer_scores[layer], dtype=np.float32)
            row[f"resnet18_{layer}_gallery_top{args.image_top_ratio:g}_auc"] = safe_auc(labels, visual_np)
            npz_payload[f"resnet18_{layer}_scores"] = visual_np
            for key, values in fuse_scores(promptad_np, visual_np, args.lambdas).items():
                row[f"resnet18_{layer}_{key}_auc"] = safe_auc(labels, values)
                npz_payload[f"resnet18_{layer}_{key}"] = np.asarray(values, dtype=np.float32)

        rows.append(row)
        np.savez_compressed(score_root / f"{signal}-{scene}-{jsr}-scores.npz", **npz_payload)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="analysis_outputs/20260629_cls_resnet_gallery_fusion")
    parser.add_argument("--formal-baseline-csv", default="analysis_outputs/20260627_method_funnel/results_cls_pooled_rf_rgb.csv")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--signals", nargs="+", default=SIGNALS, choices=SIGNALS)
    parser.add_argument("--resnet-layers", nargs="+", default=["layer2", "layer3"], choices=["layer2", "layer3"])
    parser.add_argument("--image-top-ratio", type=float, default=0.1)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.2, 0.5, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--max-gallery-patches", type=int, default=50000)
    parser.add_argument("--distance-chunk-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
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

    encoder = ResNet18LocalEncoder().to(device)
    galleries = build_resnet_gallery(encoder, train_loader, args, device)

    rows = evaluate(model, encoder, galleries, build_eval_loaders(args), args, device)
    rows = attach_formal_baseline(rows, args.formal_baseline_csv)

    out_root = Path(args.output_root)
    result_path = out_root / "results_cls_resnet_gallery_fusion.csv"
    write_csv(result_path, rows)

    df = pd.DataFrame(rows)
    summary = {
        "method": "cls_resnet_gallery_fusion",
        "checkpoint": args.checkpoint,
        "formal_baseline_csv": args.formal_baseline_csv,
        "resnet": "resnet18_imagenet1k_v1",
        "resnet_layers": args.resnet_layers,
        "image_top_ratio": args.image_top_ratio,
        "lambdas": args.lambdas,
        "max_gallery_patches": args.max_gallery_patches,
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
