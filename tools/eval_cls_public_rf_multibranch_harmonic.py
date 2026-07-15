#!/usr/bin/env python
"""Evaluate public RF CLS with text, ViT patch-gallery, CNN-gallery, and harmonic fusion."""

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
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PromptAD import PromptAD
from tools.eval_cls_public_rf_dual_gallery import (
    PUBLIC_SIGNALS,
    PublicRFPathDataset,
    build_eval_loaders,
    build_train_loader,
    parse_jsrs_by_signal,
)
from tools.eval_cls_resnet_gallery_fusion import (
    resnet_patch_score_bank,
    safe_auc,
)
from tools.eval_seg_resnet_gallery_fusion import (
    ResNet18LocalEncoder,
    build_resnet_gallery,
    load_checkpoint,
)
from train_rf_target_pooled_universal import build_gallery, to_model_input
from utils.training_utils import setup_seed


def harmonic(*scores, eps=1e-12):
    clipped = [np.clip(np.asarray(score, dtype=np.float32), eps, None) for score in scores]
    denom = np.zeros_like(clipped[0], dtype=np.float32)
    for score in clipped:
        denom += 1.0 / score
    return 1.0 / denom


def top_ratio_score(maps, ratio):
    flat = maps.reshape(maps.shape[0], -1)
    k = max(1, int(flat.shape[1] * float(ratio)))
    part = np.partition(flat, flat.shape[1] - k, axis=1)[:, -k:]
    return part.mean(axis=1)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(model, encoder, resnet_galleries, eval_loaders, args, device):
    model.eval_mode()
    model.build_text_feature_gallery()
    encoder.eval()
    rows = []
    score_root = Path(args.output_root) / "scores"
    score_root.mkdir(parents=True, exist_ok=True)

    for signal, scene, jsr, loader in eval_loaders:
        labels, names = [], []
        text_scores, vit_max_scores, vit_top001_scores = [], [], []
        cnn_layer2_top001_scores, cnn_layer2_max_scores = [], []

        for data, mask, label, name, img_type in tqdm(loader, desc=f"Eval public RF harmonic {signal}/{jsr}", leave=False):
            data_t = to_model_input(model, data, device, rgb_from_bgr=True)
            visual_features = model.encode_image(data_t)
            score_img, score_map = model.score_cached(visual_features, "cls")
            map_np = np.asarray(score_map, dtype=np.float32)
            text_np = np.asarray(score_img, dtype=np.float32)
            text_scores.extend(float(x) for x in text_np)
            vit_max_scores.extend(float(x) for x in map_np.reshape(map_np.shape[0], -1).max(axis=1))
            vit_top001_scores.extend(float(x) for x in top_ratio_score(map_np, 0.01))

            raw = data.permute(0, 3, 1, 2).to(device, non_blocking=True)
            cnn_features = encoder(raw)
            layer2_scores = resnet_patch_score_bank(
                cnn_features["layer2"],
                resnet_galleries["layer2"],
                args.distance_chunk_size,
                [0.01],
            )
            cnn_layer2_max_scores.extend(float(x) for x in layer2_scores["max"])
            cnn_layer2_top001_scores.extend(float(x) for x in layer2_scores["top0.01"])

            labels.extend(int(x) for x in label.numpy().tolist())
            names.extend(list(name))

        labels_np = np.asarray(labels, dtype=np.int32)
        text_np = np.asarray(text_scores, dtype=np.float32)
        vit_max_np = np.asarray(vit_max_scores, dtype=np.float32)
        vit_top001_np = np.asarray(vit_top001_scores, dtype=np.float32)
        cnn_max_np = np.asarray(cnn_layer2_max_scores, dtype=np.float32)
        cnn_top001_np = np.asarray(cnn_layer2_top001_scores, dtype=np.float32)
        formal_np = harmonic(text_np, vit_max_np)
        row = {
            "method": "public_rf_cls_multibranch_harmonic",
            "task": "cls",
            "dataset": signal,
            "scene": scene,
            "jsr": jsr,
            "num_normal": int((labels_np == 0).sum()),
            "num_abnormal": int((labels_np == 1).sum()),
            "text_auc": safe_auc(labels, text_np),
            "vit_max_auc": safe_auc(labels, vit_max_np),
            "vit_top0p01_auc": safe_auc(labels, vit_top001_np),
            "formal_text_vit_max_auc": safe_auc(labels, formal_np),
            "cnn_layer2_max_auc": safe_auc(labels, cnn_max_np),
            "cnn_layer2_top0p01_auc": safe_auc(labels, cnn_top001_np),
            "harmonic_formal_cnn_l2max_auc": safe_auc(labels, harmonic(formal_np, cnn_max_np)),
            "harmonic_formal_cnn_l2top0p01_auc": safe_auc(labels, harmonic(formal_np, cnn_top001_np)),
            "harmonic_text_vitmax_cnn_l2top0p01_auc": safe_auc(labels, harmonic(text_np, vit_max_np, cnn_top001_np)),
            "harmonic_text_vittop0p01_cnn_l2top0p01_auc": safe_auc(labels, harmonic(text_np, vit_top001_np, cnn_top001_np)),
        }
        rows.append(row)
        np.savez_compressed(
            score_root / f"{signal}-{jsr}-scores.npz",
            names=np.asarray(names),
            labels=labels_np,
            text_scores=text_np,
            vit_max_scores=vit_max_np,
            vit_top0p01_scores=vit_top001_np,
            formal_text_vit_max=formal_np,
            cnn_layer2_max_scores=cnn_max_np,
            cnn_layer2_top0p01_scores=cnn_top001_np,
            harmonic_formal_cnn_l2top0p01=harmonic(formal_np, cnn_top001_np),
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="analysis_outputs/20260703_public_rf_cls_multibranch_harmonic")
    parser.add_argument("--checkpoint", default="analysis_outputs/20260627_method_funnel/runs/pooled_rf_rgb/cls/checkpoint/overall-best.pt")
    parser.add_argument("--signals", nargs="+", default=list(PUBLIC_SIGNALS), choices=list(PUBLIC_SIGNALS))
    parser.add_argument("--jsrs-by-signal", default=None)
    parser.add_argument("--max-gallery-patches", type=int, default=50000)
    parser.add_argument("--distance-chunk-size", type=int, default=1024)
    parser.add_argument("--normal-sampling", choices=["all", "frequency_one_per_band"], default="all")
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)
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

    kwargs = vars(args).copy()
    kwargs.update({
        "dataset": "rf_spe_png",
        "class_name": "radio frequency spectrogram",
        "device": device,
        "out_size_h": args.resolution,
        "out_size_w": args.resolution,
    })
    model = PromptAD(**kwargs).to(device)
    load_checkpoint(model, args.checkpoint)

    train_loader = build_train_loader(args)
    build_gallery(model, train_loader, device)

    args.resnet_layers = ["layer2"]
    encoder = ResNet18LocalEncoder().to(device)
    resnet_galleries = build_resnet_gallery(encoder, train_loader, args, device)
    rows = evaluate(model, encoder, resnet_galleries, build_eval_loaders(args), args, device)

    out_root = Path(args.output_root)
    result_path = out_root / "results_cls_public_rf_multibranch_harmonic.csv"
    write_csv(result_path, rows)
    df = pd.DataFrame(rows)
    summary = {
        "method": "public_rf_cls_multibranch_harmonic",
        "checkpoint": args.checkpoint,
        "normal_sampling": args.normal_sampling,
        "jsrs_by_signal": parse_jsrs_by_signal(args.jsrs_by_signal),
        "max_gallery_patches": args.max_gallery_patches,
    }
    for col in [c for c in df.columns if c.endswith("_auc")]:
        summary[f"{col}_macro"] = float(df[col].mean())
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {result_path}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
