#!/usr/bin/env python
"""Evaluate public RF CLS with a CLIP-ViT patch nearest-neighbour gallery.

This is not the official PatchCore CNN/ResNet pipeline. It only reuses the
nearest-neighbour memory-bank scoring idea on PromptAD/CLIP ViT patch features.
Legacy output keys still use ``vit_patchcore`` for compatibility with previous
CSV/NPZ files.
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
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PromptAD import PromptAD
from tools.eval_cls_public_rf_dual_gallery import (
    PUBLIC_SIGNALS,
    build_eval_loaders,
    build_train_loader,
    parse_jsrs_by_signal,
)
from tools.eval_cls_resnet_gallery_fusion import safe_auc
from tools.eval_cls_vit_patch_gallery import harmonic, top_ratio_score
from tools.eval_cls_vit_patchcore_gallery import (
    _paired_tta_modes,
    build_frequency_row_stats,
    build_row_prototypes,
    build_vit_nn_gallery_with_rows,
    connected_region_score,
    min_cosine_distance_chunked,
    paired_tta_batch,
    prepare_patch_features,
    row_prototype_scores,
    spectrogram_nn_scores,
)
from tools.eval_seg_resnet_gallery_fusion import load_checkpoint
from train_rf_target_pooled_universal import build_gallery, to_model_input
from utils.rf_frequency_sampling import NORMAL_SAMPLING_CHOICES
from utils.training_utils import setup_seed


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def build_patchcore_gallery(model, train_loader, args, device):
    model.eval_mode()
    patches = []
    for data, mask, label, name, img_type in tqdm(train_loader, desc="Build public CLIP-ViT NN gallery", leave=False):
        data_t = to_model_input(model, data, device, rgb_from_bgr=True)
        visual_features = model.encode_image(data_t)
        patch_features = prepare_patch_features(visual_features, args.patch_layer)
        patches.append(patch_features.reshape(-1, patch_features.shape[-1]))
    gallery = torch.cat(patches, dim=0)
    if args.coreset_ratio < 1.0:
        keep = max(1, int(round(gallery.shape[0] * float(args.coreset_ratio))))
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        idx = torch.randperm(gallery.shape[0], generator=generator, device="cpu")[:keep].to(gallery.device)
        gallery = gallery[idx]
    return F.normalize(gallery.float(), dim=-1).contiguous()


@torch.no_grad()
def compute_public_patch_map(model, raw_batch, gallery, gallery_rows, gallery_cols, args, device, row_stats=None, row_prototypes=None, paired_tta_mode="identity"):
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
    if gallery_rows is None and args.nn_topk == 1 and args.freq_window < 0 and row_stats is None:
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
def evaluate(model, patchcore_gallery, eval_loaders, args, device, gallery_rows=None, gallery_cols=None, row_stats=None, row_prototypes=None, paired_tta_galleries=None):
    model.eval_mode()
    model.build_text_feature_gallery()
    rows = []
    score_root = Path(args.output_root) / "scores"
    score_root.mkdir(parents=True, exist_ok=True)

    for signal, scene, jsr, loader in eval_loaders:
        labels, names = [], []
        text_scores, vit_max_scores, vit_top001_scores = [], [], []
        pc_max_scores = []
        pc_coherent_scores = []
        pc_ratio_scores = {ratio: [] for ratio in args.map_top_ratios}

        for data, mask, label, name, img_type in tqdm(loader, desc=f"Eval public CLIP-ViT NN {signal}/{jsr}", leave=False):
            data_t = to_model_input(model, data, device, rgb_from_bgr=True)
            visual_features = model.encode_image(data_t)
            text_np = np.asarray(model.calculate_textual_anomaly_score(visual_features, "cls"), dtype=np.float32)
            _, score_map = model.score_cached(visual_features, "cls")
            vit_map = np.asarray(score_map, dtype=np.float32)
            vit_max = vit_map.reshape(vit_map.shape[0], -1).max(axis=1)
            vit_top001 = top_ratio_score(vit_map, 0.01)

            if paired_tta_galleries:
                maps = []
                for mode, tta_gallery, tta_rows, tta_cols in paired_tta_galleries:
                    maps.append(compute_public_patch_map(
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
                    pc_t = stacked_maps.mean(dim=0)
                elif args.paired_tta_fusion == "max":
                    pc_t = stacked_maps.max(dim=0).values
                else:
                    raise ValueError(f"Unsupported paired TTA fusion: {args.paired_tta_fusion}")
            else:
                pc_t = compute_public_patch_map(
                    model,
                    data,
                    patchcore_gallery,
                    gallery_rows,
                    gallery_cols,
                    args,
                    device,
                    row_stats,
                    row_prototypes,
                    paired_tta_mode="identity",
                )
            pc_map = pc_t.detach().cpu().numpy().astype(np.float32)

            text_scores.extend(float(x) for x in text_np)
            vit_max_scores.extend(float(x) for x in vit_max)
            vit_top001_scores.extend(float(x) for x in vit_top001)
            pc_max_scores.extend(float(x) for x in pc_map.reshape(pc_map.shape[0], -1).max(axis=1))
            if float(args.coherence_alpha) > 0:
                pc_coherent_scores.extend(float(x) for x in connected_region_score(
                    pc_map,
                    top_ratio=float(args.coherence_top_ratio),
                    alpha=float(args.coherence_alpha),
                ))
            for ratio in args.map_top_ratios:
                pc_ratio_scores[ratio].extend(float(x) for x in top_ratio_score(pc_map, ratio))
            labels.extend(int(x) for x in label.numpy().tolist())
            names.extend(list(name))

        labels_np = np.asarray(labels, dtype=np.int32)
        text_np = np.asarray(text_scores, dtype=np.float32)
        vit_max_np = np.asarray(vit_max_scores, dtype=np.float32)
        vit_top001_np = np.asarray(vit_top001_scores, dtype=np.float32)
        pc_max_np = np.asarray(pc_max_scores, dtype=np.float32)
        pc_coherent_np = np.asarray(pc_coherent_scores, dtype=np.float32) if pc_coherent_scores else None
        text_vit_max = harmonic(text_np, vit_max_np)
        text_pc_max = harmonic(text_np, pc_max_np)

        row = {
            "method": "public_rf_vit_patchcore_gallery_cls",
            "task": "cls",
            "dataset": signal,
            "scene": scene,
            "jsr": jsr,
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
            "position_soft_axis": args.position_soft_axis,
            "position_soft_weight": args.position_soft_weight,
            "coherence_alpha": args.coherence_alpha,
            "coherence_top_ratio": args.coherence_top_ratio,
            "paired_tta": args.paired_tta,
            "paired_tta_fusion": args.paired_tta_fusion,
            "num_normal": int((labels_np == 0).sum()),
            "num_abnormal": int((labels_np == 1).sum()),
            "text_auc": safe_auc(labels, text_np),
            "vit_max_auc": safe_auc(labels, vit_max_np),
            "vit_top0p01_auc": safe_auc(labels, vit_top001_np),
            "text_vit_max_auc": safe_auc(labels, text_vit_max),
            "vit_patchcore_max_auc": safe_auc(labels, pc_max_np),
            "text_vit_patchcore_max_auc": safe_auc(labels, text_pc_max),
        }
        payload = {
            "names": np.asarray(names),
            "labels": labels_np,
            "text_scores": text_np,
            "vit_max_scores": vit_max_np,
            "vit_top0p01_scores": vit_top001_np,
            "text_vit_max": text_vit_max,
            "vit_patchcore_max_scores": pc_max_np,
            "text_vit_patchcore_max": text_pc_max,
        }
        if pc_coherent_np is not None:
            text_pc_coherent = harmonic(text_np, pc_coherent_np)
            row["vit_patchcore_coherent_auc"] = safe_auc(labels, pc_coherent_np)
            row["text_vit_patchcore_coherent_auc"] = safe_auc(labels, text_pc_coherent)
            payload["vit_patchcore_coherent_scores"] = pc_coherent_np
            payload["text_vit_patchcore_coherent"] = text_pc_coherent
        for ratio in args.map_top_ratios:
            key = f"{ratio:g}".replace(".", "p")
            values = np.asarray(pc_ratio_scores[ratio], dtype=np.float32)
            fused = harmonic(text_np, values)
            row[f"vit_patchcore_top{key}_auc"] = safe_auc(labels, values)
            row[f"text_vit_patchcore_top{key}_auc"] = safe_auc(labels, fused)
            payload[f"vit_patchcore_top{key}_scores"] = values
            payload[f"text_vit_patchcore_top{key}"] = fused

        rows.append(row)
        np.savez_compressed(score_root / f"{signal}-{jsr}-scores.npz", **payload)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="analysis_outputs/20260704_public_rf_vit_patchcore_gallery_cls")
    parser.add_argument(
        "--checkpoint",
        default="analysis_outputs/90_rejected_or_aborted/20260706_cleanup_old_results/20260627_method_funnel/runs/pooled_rf_rgb/cls/checkpoint/overall-best.pt",
    )
    parser.add_argument("--signals", nargs="+", default=list(PUBLIC_SIGNALS), choices=list(PUBLIC_SIGNALS))
    parser.add_argument("--jsrs-by-signal", default=None)
    parser.add_argument("--normal-sampling", choices=NORMAL_SAMPLING_CHOICES, default="per_frequency")
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--patch-layer", choices=["layer1", "layer2", "concat"], default="concat")
    parser.add_argument("--coreset-ratio", type=float, default=1.0)
    parser.add_argument("--coreset-method", choices=["random", "farthest"], default="random")
    parser.add_argument("--rowwise-coreset", action="store_true")
    parser.add_argument("--memory-mode", choices=["global_nn", "row_nn", "row_proto"], default="global_nn")
    parser.add_argument("--row-proto-zscore", action="store_true")
    parser.add_argument("--support-augment", choices=["none", "rf_weak", "time_shift"], default="none")
    parser.add_argument("--support-shift-px", type=int, default=3)
    parser.add_argument("--support-crop-ratio", type=float, default=0.03)
    parser.add_argument("--gallery-chunk-size", type=int, default=4096)
    parser.add_argument("--freq-window", type=int, default=-1)
    parser.add_argument("--nn-topk", type=int, default=1)
    parser.add_argument("--nn-agg", choices=["mean", "weighted", "adaptive"], default="mean")
    parser.add_argument("--nn-weight-temp", type=float, default=0.05)
    parser.add_argument("--adaptive-sim-margin", type=float, default=0.02)
    parser.add_argument("--freq-zscore", action="store_true")
    parser.add_argument("--position-soft-axis", choices=["frequency", "time"], default="frequency")
    parser.add_argument("--position-soft-weight", type=float, default=0.0)
    parser.add_argument("--coherence-alpha", type=float, default=0.0)
    parser.add_argument("--coherence-top-ratio", type=float, default=0.05)
    parser.add_argument("--map-top-ratios", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--paired-tta", choices=["none", "visionad_safe", "stft_shift_blur", "stft_time_shift_v2"], default="none")
    parser.add_argument("--paired-tta-fusion", choices=["mean", "max"], default="mean")
    parser.add_argument("--paired-tta-contrast", type=float, default=1.04)
    parser.add_argument("--paired-tta-brightness", type=float, default=1.0)
    parser.add_argument("--paired-tta-crop-ratio", type=float, default=0.02)
    parser.add_argument("--paired-tta-shift-px", type=int, default=4)
    parser.add_argument("--paired-tta-blur-ksize", type=int, default=3)
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

    if not (0.0 < args.coreset_ratio <= 1.0):
        raise ValueError("--coreset-ratio must be in (0, 1]")
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
    paired_tta_galleries = None
    if args.paired_tta == "none":
        patchcore_gallery, gallery_rows, gallery_cols = build_vit_nn_gallery_with_rows(model, train_loader, args, device)
    else:
        if args.support_augment != "none":
            raise ValueError("--paired-tta and --support-augment should not be enabled together")
        paired_tta_galleries = []
        for mode in _paired_tta_modes(args):
            tta_gallery, tta_rows, tta_cols = build_vit_nn_gallery_with_rows(model, train_loader, args, device, paired_tta_mode=mode)
            paired_tta_galleries.append((mode, tta_gallery, tta_rows, tta_cols))
        patchcore_gallery, gallery_rows, gallery_cols = paired_tta_galleries[0][1:]
    if args.memory_mode == "row_nn" and args.freq_window < 0:
        args.freq_window = 0
    row_stats = build_frequency_row_stats(patchcore_gallery, gallery_rows, args, model.grid_size[0]) if args.freq_zscore else None
    row_prototypes = build_row_prototypes(patchcore_gallery, gallery_rows, model.grid_size[0]) if args.memory_mode == "row_proto" else None
    rows = evaluate(
        model,
        patchcore_gallery,
        build_eval_loaders(args),
        args,
        device,
        gallery_rows,
        gallery_cols,
        row_stats,
        row_prototypes,
        paired_tta_galleries=paired_tta_galleries,
    )

    out_root = Path(args.output_root)
    result_path = out_root / "results_cls_public_rf_vit_patchcore_gallery.csv"
    write_csv(result_path, rows)
    df = pd.DataFrame(rows)
    summary = {
        "method": "public_rf_vit_patchcore_gallery_cls",
        "checkpoint": args.checkpoint,
        "normal_sampling": args.normal_sampling,
        "jsrs_by_signal": parse_jsrs_by_signal(args.jsrs_by_signal),
        "patch_layer": args.patch_layer,
        "coreset_ratio": args.coreset_ratio,
        "coreset_method": args.coreset_method,
        "rowwise_coreset": bool(args.rowwise_coreset),
        "memory_mode": args.memory_mode,
        "support_augment": args.support_augment,
        "support_shift_px": args.support_shift_px,
        "support_crop_ratio": args.support_crop_ratio,
        "freq_window": args.freq_window,
        "nn_topk": args.nn_topk,
        "nn_agg": args.nn_agg,
        "nn_weight_temp": args.nn_weight_temp,
        "adaptive_sim_margin": args.adaptive_sim_margin,
        "freq_zscore": bool(args.freq_zscore),
        "row_proto_zscore": bool(args.row_proto_zscore),
        "position_soft_axis": args.position_soft_axis,
        "position_soft_weight": args.position_soft_weight,
        "coherence_alpha": args.coherence_alpha,
        "coherence_top_ratio": args.coherence_top_ratio,
        "paired_tta": args.paired_tta,
        "paired_tta_fusion": args.paired_tta_fusion,
        "paired_tta_modes": _paired_tta_modes(args),
        "paired_tta_contrast": args.paired_tta_contrast,
        "paired_tta_brightness": args.paired_tta_brightness,
        "paired_tta_crop_ratio": args.paired_tta_crop_ratio,
        "paired_tta_shift_px": args.paired_tta_shift_px,
        "paired_tta_blur_ksize": args.paired_tta_blur_ksize,
        "selected_normal_count": len(train_loader.dataset),
        "gallery_patch_count": int(patchcore_gallery.shape[0]),
        "gallery_feature_dim": int(patchcore_gallery.shape[1]),
        "num_cells": len(rows),
    }
    for col in [c for c in df.columns if c.endswith("_auc")]:
        summary[f"{col}_macro"] = float(df[col].mean())
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Public RF CLIP-ViT NN Gallery CLS",
        "",
        "This is a CLIP-ViT patch nearest-neighbour gallery, not the official CNN/ResNet PatchCore baseline.",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- normal sampling: `{args.normal_sampling}`",
        f"- patch layer: `{args.patch_layer}`",
        f"- support augment: `{args.support_augment}`",
        f"- paired TTA: `{args.paired_tta}`",
        f"- paired TTA fusion: `{args.paired_tta_fusion}`",
        f"- gallery patches: `{int(patchcore_gallery.shape[0])}`",
        f"- cells: `{len(rows)}`",
        "",
        "## Macro AUROC",
        "",
    ]
    for key, value in sorted((k, v) for k, v in summary.items() if k.endswith("_auc_macro")):
        lines.append(f"- `{key}`: {value:.4f}")
    by_signal = df.groupby("dataset")[[c for c in df.columns if c.endswith("_auc")]].mean().reset_index()
    by_signal.to_csv(out_root / "by_signal_public_rf_vit_patchcore_gallery.csv", index=False)
    lines.extend(["", "## By Signal", "", "| signal | text+ViT | text+CLIP-ViT-NN max | delta |", "|---|---:|---:|---:|"])
    for _, row in by_signal.iterrows():
        base = float(row["text_vit_max_auc"])
        new = float(row["text_vit_patchcore_max_auc"])
        lines.append(f"| {row['dataset']} | {base:.4f} | {new:.4f} | {new - base:+.4f} |")
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {result_path}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
