#!/usr/bin/env python
"""Evaluate CLS scoring rules based on the PromptAD ViT patch gallery."""

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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PromptAD import PromptAD
from tools.eval_seg_resnet_gallery_fusion import load_checkpoint
from train_rf_target_pooled_universal import (
    JSR_BY_SIGNAL,
    RFPathDataset,
    SCENES,
    SIGNALS,
    build_eval_loaders,
    build_gallery,
    collect_samples,
    to_model_input,
)
from utils.rf_frequency_sampling import NORMAL_SAMPLING_CHOICES, frequency_band_key, maybe_select_one_per_frequency_band
from utils.training_utils import setup_seed


def safe_auc(labels, scores) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores) * 100.0)


def harmonic(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return 1.0 / (1.0 / np.clip(a, eps, None) + 1.0 / np.clip(b, eps, None))


def top_ratio_score(maps, ratio):
    flat = maps.reshape(maps.shape[0], -1)
    k = max(1, int(flat.shape[1] * float(ratio)))
    part = np.partition(flat, flat.shape[1] - k, axis=1)[:, -k:]
    return part.mean(axis=1)


def collect_pooled_train_samples(args):
    samples = []
    for signal in args.signals:
        for scene in SCENES:
            for jsr in JSR_BY_SIGNAL[signal]:
                samples.extend(collect_samples(signal, scene, jsr, "train", k_shot=0))
    mode = "all" if args.normal_sampling == "first" else args.normal_sampling
    selected = maybe_select_one_per_frequency_band(samples, mode, path_getter=lambda sample: sample[0])

    if not selected:
        raise RuntimeError("No normal samples selected for ViT patch gallery")
    return selected


def build_selected_train_loader(args):
    samples = collect_pooled_train_samples(args)
    args.support_paths = {sample[0] for sample in samples}
    return DataLoader(
        RFPathDataset(samples),
        batch_size=min(args.batch_size, len(samples)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    ), samples


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
    merged = pd.DataFrame(rows).merge(
        base[["dataset", "scene", "jsr", "formal_baseline_auc"]],
        on=["dataset", "scene", "jsr"],
        how="left",
    )
    for col in [c for c in merged.columns if c.endswith("_auc") and c != "formal_baseline_auc"]:
        merged[f"delta_{col}_vs_formal"] = merged[col] - merged["formal_baseline_auc"]
    return merged.to_dict("records")


@torch.no_grad()
def evaluate(model, eval_loaders, args, device):
    model.eval_mode()
    model.build_text_feature_gallery()
    rows = []
    score_dir = Path(args.output_root) / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    for signal, scene, jsr, loader in eval_loaders:
        labels, names = [], []
        text_scores, map_max_scores = [], []
        ratio_scores = {ratio: [] for ratio in args.map_top_ratios}

        for data, mask, label, name, img_type in tqdm(loader, desc=f"Eval ViT patch gallery {signal}/{scene}/{jsr}", leave=False):
            data_t = to_model_input(model, data, device, rgb_from_bgr=False)
            visual_features = model.encode_image(data_t)
            score_img, score_map = model.score_cached(visual_features, "cls")
            map_np = np.asarray(score_map, dtype=np.float32)
            text_np = np.asarray(score_img, dtype=np.float32)
            text_scores.extend(float(x) for x in text_np)
            map_max_scores.extend(float(x) for x in map_np.reshape(map_np.shape[0], -1).max(axis=1))
            for ratio in args.map_top_ratios:
                ratio_scores[ratio].extend(float(x) for x in top_ratio_score(map_np, ratio))
            labels.extend(int(x) for x in label.numpy().tolist())
            names.extend(list(name))

        labels_np = np.asarray(labels, dtype=np.int32)
        text_np = np.asarray(text_scores, dtype=np.float32)
        map_max_np = np.asarray(map_max_scores, dtype=np.float32)
        row = {
            "method": "vit_patch_gallery_cls",
            "task": "cls",
            "dataset": signal,
            "scene": scene,
            "jsr": jsr,
            "text_auc": safe_auc(labels, text_np),
            "vit_map_max_auc": safe_auc(labels, map_max_np),
            "harmonic_text_vit_max_auc": safe_auc(labels, harmonic(text_np, map_max_np)),
        }
        payload = {
            "names": np.asarray(names),
            "labels": labels_np,
            "text_scores": text_np,
            "vit_map_max_scores": map_max_np,
            "harmonic_text_vit_max": harmonic(text_np, map_max_np),
        }
        for ratio in args.map_top_ratios:
            key = f"{ratio:g}".replace(".", "p")
            values = np.asarray(ratio_scores[ratio], dtype=np.float32)
            row[f"vit_map_top{key}_auc"] = safe_auc(labels, values)
            row[f"harmonic_text_vit_top{key}_auc"] = safe_auc(labels, harmonic(text_np, values))
            payload[f"vit_map_top{key}_scores"] = values
            payload[f"harmonic_text_vit_top{key}"] = harmonic(text_np, values)

        rows.append(row)
        np.savez_compressed(score_dir / f"{signal}-{scene}-{jsr}-scores.npz", **payload)

    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="analysis_outputs/20260703_vit_patch_gallery_cls")
    parser.add_argument("--formal-baseline-csv", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--signals", nargs="+", default=SIGNALS, choices=SIGNALS)
    parser.add_argument("--normal-sampling", choices=["first", *NORMAL_SAMPLING_CHOICES], default="per_frequency")
    parser.add_argument("--map-top-ratios", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--batch-size", type=int, default=64)
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
    return parser.parse_args()


def main():
    args = parse_args()
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

    train_loader, train_samples = build_selected_train_loader(args)
    build_gallery(model, train_loader, device)

    rows = evaluate(model, build_eval_loaders(args), args, device)
    rows = attach_formal_baseline(rows, args.formal_baseline_csv)

    out_root = Path(args.output_root)
    result_path = out_root / "results_cls_vit_patch_gallery.csv"
    write_csv(result_path, rows)

    df = pd.DataFrame(rows)
    summary = {
        "method": "vit_patch_gallery_cls",
        "checkpoint": args.checkpoint,
        "normal_sampling": args.normal_sampling,
        "selected_normal_count": len(train_samples),
        "selected_normals": [sample[0] for sample in train_samples],
        "map_top_ratios": args.map_top_ratios,
        "formal_baseline_csv": args.formal_baseline_csv,
    }
    for col in [c for c in df.columns if c.endswith("_auc")]:
        summary[f"{col}_macro"] = float(df[col].mean())
    for col in [c for c in df.columns if c.startswith("delta_")]:
        summary[f"{col}_macro"] = float(df[col].mean())

    summary_path = out_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(f"wrote {result_path}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
