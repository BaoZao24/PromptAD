#!/usr/bin/env python
"""Evaluate CLS scores where ViT anomaly regions guide CNN local scoring."""

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
    cnn_distance_map,
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


def harmonic(*scores, eps=1e-12):
    clipped = [np.clip(np.asarray(score, dtype=np.float32), eps, None) for score in scores]
    denom = np.zeros_like(clipped[0], dtype=np.float32)
    for score in clipped:
        denom += 1.0 / score
    return 1.0 / denom


def minmax(x):
    x = np.asarray(x, dtype=np.float32)
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def top_ratio_score(maps, ratio):
    flat = maps.reshape(maps.shape[0], -1)
    k = max(1, int(flat.shape[1] * float(ratio)))
    part = np.partition(flat, flat.shape[1] - k, axis=1)[:, -k:]
    return part.mean(axis=1)


def vit_guided_cnn_score(vit_maps: np.ndarray, cnn_maps: np.ndarray, ratio: float):
    n = vit_maps.shape[0]
    vit_flat = vit_maps.reshape(n, -1)
    cnn_flat = cnn_maps.reshape(n, -1)
    k = max(1, int(vit_flat.shape[1] * float(ratio)))
    top_idx = np.argpartition(vit_flat, vit_flat.shape[1] - k, axis=1)[:, -k:]
    return np.take_along_axis(cnn_flat, top_idx, axis=1).mean(axis=1)


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
def evaluate(model, encoder, resnet_galleries, eval_loaders, args, device):
    model.eval_mode()
    model.build_text_feature_gallery()
    encoder.eval()
    rows = []
    score_root = Path(args.output_root) / "scores"
    score_root.mkdir(parents=True, exist_ok=True)

    for signal, scene, jsr, loader in eval_loaders:
        labels, names = [], []
        score_bank = {
            "text": [],
            "vit_max": [],
            "vit_top0p01": [],
            "formal_text_vit_max": [],
        }
        for layer in args.resnet_layers:
            for ratio in args.guided_ratios:
                key = f"cnn_{layer}_guided_by_vit_top{ratio:g}".replace(".", "p")
                score_bank[key] = []

        for data, mask, label, name, img_type in tqdm(loader, desc=f"Eval ViT-guided CNN {signal}/{scene}/{jsr}", leave=False):
            data_t = to_model_input(model, data, device, rgb_from_bgr=False)
            visual_features = model.encode_image(data_t)
            text_score, vit_map = model.score_cached(visual_features, "cls")
            vit_np = np.asarray(vit_map, dtype=np.float32)
            text_np = np.asarray(text_score, dtype=np.float32)
            vit_max = vit_np.reshape(vit_np.shape[0], -1).max(axis=1)

            score_bank["text"].extend(float(x) for x in text_np)
            score_bank["vit_max"].extend(float(x) for x in vit_max)
            score_bank["vit_top0p01"].extend(float(x) for x in top_ratio_score(vit_np, 0.01))
            score_bank["formal_text_vit_max"].extend(float(x) for x in harmonic(text_np, vit_max))

            raw = data.permute(0, 3, 1, 2).to(device, non_blocking=True)
            cnn_features = encoder(raw)
            vit_t = torch.as_tensor(vit_np, device=device, dtype=torch.float32)
            for layer in args.resnet_layers:
                cnn_map = cnn_distance_map(cnn_features[layer], resnet_galleries[layer], args.distance_chunk_size).to(device)
                cnn_map = F.interpolate(cnn_map, size=vit_t.shape[-2:], mode="bilinear", align_corners=False)
                cnn_np = cnn_map.detach().cpu().numpy().astype(np.float32)
                for ratio in args.guided_ratios:
                    key = f"cnn_{layer}_guided_by_vit_top{ratio:g}".replace(".", "p")
                    values = vit_guided_cnn_score(vit_np, cnn_np, ratio)
                    score_bank[key].extend(float(x) for x in values)

            labels.extend(int(x) for x in label.numpy().tolist())
            names.extend(list(name))

        labels_np = np.asarray(labels, dtype=np.int32)
        arrays = {key: np.asarray(values, dtype=np.float32) for key, values in score_bank.items()}
        row = {
            "method": "cls_vit_guided_cnn_gallery",
            "task": "cls",
            "dataset": signal,
            "scene": scene,
            "jsr": jsr,
        }
        payload = {
            "names": np.asarray(names),
            "labels": labels_np,
        }
        for key, values in arrays.items():
            row[f"{key}_auc"] = safe_auc(labels, values)
            payload[key] = values

        formal = arrays["formal_text_vit_max"]
        for key, values in arrays.items():
            if not key.startswith("cnn_"):
                continue
            guided_norm = minmax(values)
            for beta in args.betas:
                boosted = formal * (1.0 + float(beta) * guided_norm)
                added = minmax(formal) + float(beta) * guided_norm
                row[f"boost_formal_{key}_beta{beta:g}_auc"] = safe_auc(labels, boosted)
                row[f"add_minmax_formal_{key}_beta{beta:g}_auc"] = safe_auc(labels, added)
                payload[f"boost_formal_{key}_beta{beta:g}"] = boosted.astype(np.float32)
                payload[f"add_minmax_formal_{key}_beta{beta:g}"] = added.astype(np.float32)

        rows.append(row)
        np.savez_compressed(score_root / f"{signal}-{scene}-{jsr}-scores.npz", **payload)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="analysis_outputs/20260703_rf_freqband_vit_guided_cnn_cls")
    parser.add_argument("--formal-baseline-csv", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--signals", nargs="+", default=SIGNALS, choices=SIGNALS)
    parser.add_argument("--resnet-layers", nargs="+", default=["layer2"], choices=["layer2", "layer3"])
    parser.add_argument("--guided-ratios", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--max-gallery-patches", type=int, default=50000)
    parser.add_argument("--distance-chunk-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--normal-sampling", choices=["all", "frequency_one_per_band"], default="frequency_one_per_band")
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
        "dataset": "rf_target_test_pool",
        "class_name": "signal",
        "device": device,
        "out_size_h": args.resolution,
        "out_size_w": args.resolution,
    })
    model = PromptAD(**kwargs).to(device)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)

    train_loader = build_train_loader(args)
    build_gallery(model, train_loader, device)

    encoder = ResNet18LocalEncoder().to(device)
    resnet_galleries = build_resnet_gallery(encoder, train_loader, args, device)
    rows = evaluate(model, encoder, resnet_galleries, build_eval_loaders(args), args, device)
    rows = attach_formal_baseline(rows, args.formal_baseline_csv)

    out_root = Path(args.output_root)
    result_path = out_root / "results_cls_vit_guided_cnn_gallery.csv"
    write_csv(result_path, rows)
    df = pd.DataFrame(rows)
    summary = {
        "method": "cls_vit_guided_cnn_gallery",
        "checkpoint": args.checkpoint,
        "formal_baseline_csv": args.formal_baseline_csv,
        "normal_sampling": args.normal_sampling,
        "resnet_layers": args.resnet_layers,
        "guided_ratios": args.guided_ratios,
        "betas": args.betas,
        "max_gallery_patches": args.max_gallery_patches,
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
