#!/usr/bin/env python
"""Evaluate spectrum CLS with CLIP-ViT patch nearest-neighbour galleries."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PromptAD import PromptAD
from tools.eval_cls_vit_patch_gallery import harmonic, top_ratio_score
from tools.eval_cls_vit_patchcore_gallery import (
    _paired_tta_modes,
    build_frequency_row_stats,
    build_row_prototypes,
    build_vit_nn_gallery_with_rows,
    min_cosine_distance_chunked,
    paired_tta_batch,
    prepare_patch_features,
    row_prototype_scores,
    spectrogram_nn_scores,
)
from tools.eval_seg_resnet_gallery_fusion import load_checkpoint
from train_rf_target_pooled_universal import build_gallery, to_model_input
from utils.rf_frequency_sampling import NORMAL_SAMPLING_CHOICES, maybe_select_one_per_frequency_band
from utils.training_utils import setup_seed


SPECTRUM_ROOT = Path("datasets/spectrum")
SPECTRUM_CLASSES = ("16QAM", "CHIRP", "GMSK", "QPSK")


class SpectrumPathDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, sample_type, name_prefix = self.samples[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        h, w = img.shape[:2]
        gt = np.zeros((h, w), dtype=np.uint8)
        name = f"{name_prefix}-{sample_type}-{Path(img_path).stem}"
        return img, gt, int(label), name, sample_type


def safe_auc(labels, scores) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores) * 100.0)


def collect_train_samples(category: str, args):
    paths = sorted((Path(args.spectrum_root) / category / "train" / "good").glob("*.png"))
    paths = maybe_select_one_per_frequency_band(paths, args.normal_sampling)
    if args.max_train_normals > 0:
        paths = paths[: args.max_train_normals]
    return [(p, 0, f"{category}_train_good", f"spectrum-{category}") for p in paths]


def collect_eval_samples(category: str, args):
    root = Path(args.spectrum_root) / category / "test"
    good_paths = sorted((root / "good").glob("*.png"))
    bad_paths = sorted((root / "bad").glob("*.png"))
    if args.max_test_normals > 0:
        good_paths = good_paths[: args.max_test_normals]
    if args.max_abnormals > 0:
        bad_paths = bad_paths[: args.max_abnormals]
    samples = [(p, 0, f"{category}_test_good", f"spectrum-{category}") for p in good_paths]
    samples.extend((p, 1, f"{category}_test_bad", f"spectrum-{category}") for p in bad_paths)
    return samples


def make_loader(samples, args):
    return DataLoader(
        SpectrumPathDataset(samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def compute_spectrum_patch_map(model, raw_batch, gallery, gallery_rows, gallery_cols, args, device, row_stats=None, row_prototypes=None, paired_tta_mode="identity"):
    data_variant = paired_tta_batch(raw_batch, paired_tta_mode, args)
    data_t = to_model_input(model, data_variant, device, rgb_from_bgr=True)
    visual_features = model.encode_image(data_t)
    patch_features = prepare_patch_features(visual_features, args.patch_layer)
    bsz, num_patches, _ = patch_features.shape
    grid_h, grid_w = model.grid_size
    if args.memory_mode == "row_proto":
        return row_prototype_scores(
            patch_features,
            row_prototypes[0],
            row_prototypes[1],
            args,
            grid_h,
            grid_w,
        )
    if args.nn_topk == 1 and args.freq_window < 0 and row_stats is None:
        patch_scores = min_cosine_distance_chunked(
            patch_features.reshape(-1, patch_features.shape[-1]),
            gallery,
            args.gallery_chunk_size,
        ).reshape(bsz, num_patches)
        return patch_scores.reshape(bsz, grid_h, grid_w)
    return spectrogram_nn_scores(
        patch_features,
        gallery,
        gallery_rows,
        args,
        grid_h,
        grid_w,
        row_stats,
        gallery_cols,
    )


@torch.no_grad()
def evaluate_category(model, category, train_loader, eval_loader, args, device):
    build_gallery(model, train_loader, device)
    model.build_text_feature_gallery()
    paired_tta_galleries = None
    if args.paired_tta == "none":
        vit_nn_gallery, gallery_rows, gallery_cols = build_vit_nn_gallery_with_rows(model, train_loader, args, device)
    else:
        paired_tta_galleries = []
        for mode in _paired_tta_modes(args):
            tta_gallery, tta_rows, tta_cols = build_vit_nn_gallery_with_rows(model, train_loader, args, device, paired_tta_mode=mode)
            paired_tta_galleries.append((mode, tta_gallery, tta_rows, tta_cols))
        vit_nn_gallery, gallery_rows, gallery_cols = paired_tta_galleries[0][1:]
    if args.memory_mode == "row_nn" and args.freq_window < 0:
        args.freq_window = 0
    grid_h, grid_w = model.grid_size
    row_stats = build_frequency_row_stats(vit_nn_gallery, gallery_rows, args, grid_h) if args.freq_zscore else None
    row_prototypes = build_row_prototypes(vit_nn_gallery, gallery_rows, grid_h) if args.memory_mode == "row_proto" else None

    labels, names = [], []
    text_scores, vit_max_scores = [], []
    nn_max_scores = []
    nn_ratio_scores = {ratio: [] for ratio in args.map_top_ratios}

    model.eval_mode()
    for data, mask, label, name, img_type in tqdm(eval_loader, desc=f"Eval spectrum CLIP-ViT NN {category}", leave=False):
        data_t = to_model_input(model, data, device, rgb_from_bgr=True)
        visual_features = model.encode_image(data_t)
        text_np = np.asarray(model.calculate_textual_anomaly_score(visual_features, "cls"), dtype=np.float32)
        _, vit_map = model.score_cached(visual_features, "cls")
        vit_map = np.asarray(vit_map, dtype=np.float32)
        vit_max = vit_map.reshape(vit_map.shape[0], -1).max(axis=1)

        if paired_tta_galleries:
            maps = []
            for mode, tta_gallery, tta_rows, tta_cols in paired_tta_galleries:
                maps.append(compute_spectrum_patch_map(
                    model,
                    data,
                    tta_gallery,
                    tta_rows,
                    tta_cols,
                    args,
                    device,
                    row_stats=None,
                    row_prototypes=None,
                    paired_tta_mode=mode,
                ))
            stacked_maps = torch.stack(maps, dim=0)
            if args.paired_tta_fusion == "mean":
                nn_t = stacked_maps.mean(dim=0)
            elif args.paired_tta_fusion == "max":
                nn_t = stacked_maps.max(dim=0).values
            else:
                raise ValueError(f"Unsupported paired TTA fusion: {args.paired_tta_fusion}")
        else:
            nn_t = compute_spectrum_patch_map(
                model,
                data,
                vit_nn_gallery,
                gallery_rows,
                gallery_cols,
                args,
                device,
                row_stats,
                row_prototypes,
                paired_tta_mode="identity",
            )
        nn_map = nn_t.detach().cpu().numpy().astype(np.float32)

        text_scores.extend(float(x) for x in text_np)
        vit_max_scores.extend(float(x) for x in vit_max)
        nn_max_scores.extend(float(x) for x in nn_map.reshape(nn_map.shape[0], -1).max(axis=1))
        for ratio in args.map_top_ratios:
            nn_ratio_scores[ratio].extend(float(x) for x in top_ratio_score(nn_map, ratio))
        labels.extend(int(x) for x in label.numpy().tolist())
        names.extend(list(name))

    labels_np = np.asarray(labels, dtype=np.int32)
    text_np = np.asarray(text_scores, dtype=np.float32)
    vit_max_np = np.asarray(vit_max_scores, dtype=np.float32)
    nn_max_np = np.asarray(nn_max_scores, dtype=np.float32)
    text_vit = harmonic(text_np, vit_max_np)
    text_nn = harmonic(text_np, nn_max_np)
    row = {
        "method": "spectrum_clip_vit_nn_gallery_cls",
        "task": "cls",
        "dataset": "spectrum",
        "category": category,
        "patch_layer": args.patch_layer,
        "coreset_ratio": args.coreset_ratio,
        "freq_window": args.freq_window,
        "nn_topk": args.nn_topk,
        "nn_agg": args.nn_agg,
        "coreset_method": args.coreset_method,
        "rowwise_coreset": int(args.rowwise_coreset),
        "memory_mode": args.memory_mode,
        "freq_zscore": int(args.freq_zscore),
        "row_proto_zscore": int(args.row_proto_zscore),
        "paired_tta": args.paired_tta,
        "paired_tta_fusion": args.paired_tta_fusion,
        "num_normal": int((labels_np == 0).sum()),
        "num_abnormal": int((labels_np == 1).sum()),
        "text_auc": safe_auc(labels, text_np),
        "vit_max_auc": safe_auc(labels, vit_max_np),
        "text_vit_max_auc": safe_auc(labels, text_vit),
        "clip_vit_nn_max_auc": safe_auc(labels, nn_max_np),
        "text_clip_vit_nn_max_auc": safe_auc(labels, text_nn),
    }
    payload = {
        "names": np.asarray(names),
        "labels": labels_np,
        "text_scores": text_np,
        "vit_max_scores": vit_max_np,
        "text_vit_max": text_vit,
        "clip_vit_nn_max_scores": nn_max_np,
        "vit_patchcore_max_scores": nn_max_np,
        "text_clip_vit_nn_max": text_nn,
    }
    for ratio in args.map_top_ratios:
        key = f"{ratio:g}".replace(".", "p")
        values = np.asarray(nn_ratio_scores[ratio], dtype=np.float32)
        fused = harmonic(text_np, values)
        row[f"clip_vit_nn_top{key}_auc"] = safe_auc(labels, values)
        row[f"text_clip_vit_nn_top{key}_auc"] = safe_auc(labels, fused)
        payload[f"clip_vit_nn_top{key}_scores"] = values
        payload[f"vit_patchcore_top{key}_scores"] = values
        payload[f"text_clip_vit_nn_top{key}"] = fused

    score_root = Path(args.output_root) / "scores"
    score_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(score_root / f"{category}-scores.npz", **payload)
    return row, int(vit_nn_gallery.shape[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="analysis_outputs/20260705_spectrum_clip_vit_nn_gallery_cls")
    parser.add_argument("--spectrum-root", default=str(SPECTRUM_ROOT))
    parser.add_argument(
        "--checkpoint",
        default="analysis_outputs/90_rejected_or_aborted/20260706_cleanup_old_results/20260627_method_funnel/runs/pooled_rf_rgb/cls/checkpoint/overall-best.pt",
    )
    parser.add_argument("--categories", nargs="+", default=list(SPECTRUM_CLASSES), choices=list(SPECTRUM_CLASSES))
    parser.add_argument("--normal-sampling", choices=NORMAL_SAMPLING_CHOICES, default="per_frequency")
    parser.add_argument("--patch-layer", choices=["layer1", "layer2", "concat"], default="concat")
    parser.add_argument("--coreset-ratio", type=float, default=1.0)
    parser.add_argument("--coreset-method", choices=["random", "farthest"], default="random")
    parser.add_argument("--rowwise-coreset", action="store_true")
    parser.add_argument("--memory-mode", choices=["global_nn", "row_nn", "row_proto"], default="global_nn")
    parser.add_argument("--row-proto-zscore", action="store_true")
    parser.add_argument("--gallery-chunk-size", type=int, default=4096)
    parser.add_argument("--freq-window", type=int, default=-1)
    parser.add_argument("--position-soft-axis", choices=["frequency", "time"], default="frequency")
    parser.add_argument("--position-soft-weight", type=float, default=0.0)
    parser.add_argument("--nn-topk", type=int, default=1)
    parser.add_argument("--nn-agg", choices=["mean", "weighted", "adaptive"], default="mean")
    parser.add_argument("--nn-weight-temp", type=float, default=0.05)
    parser.add_argument("--adaptive-sim-margin", type=float, default=0.02)
    parser.add_argument("--freq-zscore", action="store_true")
    parser.add_argument("--map-top-ratios", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--paired-tta", choices=["none", "visionad_safe", "stft_shift_blur", "stft_time_shift_v2"], default="none")
    parser.add_argument("--paired-tta-fusion", choices=["mean", "max"], default="mean")
    parser.add_argument("--paired-tta-contrast", type=float, default=1.04)
    parser.add_argument("--paired-tta-brightness", type=float, default=1.0)
    parser.add_argument("--paired-tta-crop-ratio", type=float, default=0.02)
    parser.add_argument("--paired-tta-shift-px", type=int, default=4)
    parser.add_argument("--paired-tta-blur-ksize", type=int, default=3)
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--max-train-normals", type=int, default=0)
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
        "dataset": "spectrum",
        "class_name": "radio frequency spectrogram",
        "device": device,
        "out_size_h": args.resolution,
        "out_size_w": args.resolution,
    })
    model = PromptAD(**kwargs).to(device)
    load_checkpoint(model, args.checkpoint)

    rows = []
    gallery_counts = {}
    for category in args.categories:
        train_samples = collect_train_samples(category, args)
        eval_samples = collect_eval_samples(category, args)
        if not train_samples:
            print(f"[skip] no train/good samples for {category}")
            continue
        if not any(int(sample[1]) == 1 for sample in eval_samples):
            print(f"[skip] no test/bad samples for {category}")
            continue
        row, gallery_count = evaluate_category(
            model,
            category,
            make_loader(train_samples, args),
            make_loader(eval_samples, args),
            args,
            device,
        )
        rows.append(row)
        gallery_counts[category] = gallery_count

    out_root = Path(args.output_root)
    result_path = out_root / "results_cls_spectrum_clip_vit_nn_gallery.csv"
    write_csv(result_path, rows)
    df = pd.DataFrame(rows)
    summary = {
        "method": "spectrum_clip_vit_nn_gallery_cls",
        "checkpoint": args.checkpoint,
        "spectrum_root": args.spectrum_root,
        "categories": args.categories,
        "normal_sampling": args.normal_sampling,
        "max_train_normals": args.max_train_normals,
        "patch_layer": args.patch_layer,
        "coreset_ratio": args.coreset_ratio,
        "coreset_method": args.coreset_method,
        "rowwise_coreset": bool(args.rowwise_coreset),
        "memory_mode": args.memory_mode,
        "freq_window": args.freq_window,
        "nn_topk": args.nn_topk,
        "nn_agg": args.nn_agg,
        "nn_weight_temp": args.nn_weight_temp,
        "adaptive_sim_margin": args.adaptive_sim_margin,
        "freq_zscore": bool(args.freq_zscore),
        "row_proto_zscore": bool(args.row_proto_zscore),
        "paired_tta": args.paired_tta,
        "paired_tta_fusion": args.paired_tta_fusion,
        "paired_tta_modes": _paired_tta_modes(args),
        "paired_tta_contrast": args.paired_tta_contrast,
        "paired_tta_brightness": args.paired_tta_brightness,
        "paired_tta_crop_ratio": args.paired_tta_crop_ratio,
        "paired_tta_shift_px": args.paired_tta_shift_px,
        "paired_tta_blur_ksize": args.paired_tta_blur_ksize,
        "gallery_patch_counts": gallery_counts,
        "num_cells": len(rows),
    }
    for col in [c for c in df.columns if c.endswith("_auc")]:
        summary[f"{col}_macro"] = float(df[col].mean())
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Spectrum CLIP-ViT NN Gallery CLS",
        "",
        "Nearest-neighbour gallery scoring over PromptAD/CLIP ViT patch features.",
        "",
        f"- normal sampling: `{args.normal_sampling}`",
        f"- patch layer: `{args.patch_layer}`",
        f"- paired TTA: `{args.paired_tta}`",
        f"- paired TTA fusion: `{args.paired_tta_fusion}`",
        f"- categories: `{len(rows)}`",
        "",
        "## Macro AUROC",
        "",
    ]
    for key, value in sorted((k, v) for k, v in summary.items() if k.endswith("_auc_macro")):
        lines.append(f"- `{key}`: {value:.4f}")
    by_cat = df[["category", "text_vit_max_auc", "clip_vit_nn_max_auc", "text_clip_vit_nn_max_auc"]].copy()
    by_cat.to_csv(out_root / "by_category_spectrum_clip_vit_nn_gallery.csv", index=False)
    lines.extend(["", "## By Category", "", "| category | text+ViT | CLIP-ViT NN | text+CLIP-ViT NN |", "|---|---:|---:|---:|"])
    for _, row in by_cat.iterrows():
        lines.append(
            f"| {row['category']} | {row['text_vit_max_auc']:.4f} | "
            f"{row['clip_vit_nn_max_auc']:.4f} | {row['text_clip_vit_nn_max_auc']:.4f} |"
        )
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {result_path}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
