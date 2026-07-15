#!/usr/bin/env python
"""Evaluate SEG maps with both CLIP and ResNet normal-gallery calibration."""

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
from tools.eval_seg_resnet_gallery_fusion import (
    ResNet18LocalEncoder,
    build_resnet_gallery,
    cnn_distance_map,
    load_checkpoint,
    minmax_map,
    smooth_resize_maps,
)
from train_rf_target_pooled_universal import (
    SIGNALS,
    build_eval_loaders,
    build_gallery,
    build_train_loader,
    to_model_input,
)
from utils.eval_utils import specify_resolution
from utils.metrics import metric_cal_pix
from utils.training_utils import setup_seed


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def attach_formal_baseline(rows, baseline_csv):
    if not baseline_csv or not Path(baseline_csv).exists():
        return rows
    base = pd.read_csv(baseline_csv).rename(columns={"p_roc": "formal_baseline_p_roc"})
    merged = pd.DataFrame(rows).merge(
        base[["dataset", "scene", "jsr", "formal_baseline_p_roc"]],
        on=["dataset", "scene", "jsr"],
        how="left",
    )
    for col in [c for c in merged.columns if c.endswith("_p_roc") and c != "formal_baseline_p_roc"]:
        merged[f"delta_{col}_vs_formal"] = merged[col] - merged["formal_baseline_p_roc"]
    return merged.to_dict("records")


@torch.no_grad()
def evaluate(model, encoder, resnet_galleries, eval_loaders, args, device):
    model.eval_mode()
    encoder.eval()
    rows = []
    score_root = Path(args.output_root) / "scores"
    score_root.mkdir(parents=True, exist_ok=True)

    for signal, scene, jsr, loader in eval_loaders:
        gt_mask_list = []
        map_bank = {
            "textual_only": [],
            "clip_gallery_only": [],
        }
        for beta_clip in args.clip_betas:
            map_bank[f"gate_clip_beta{beta_clip:g}"] = []
        for layer in args.resnet_layers:
            map_bank[f"resnet18_{layer}_gallery_only"] = []
            for beta_resnet in args.resnet_betas:
                map_bank[f"gate_resnet18_{layer}_beta{beta_resnet:g}"] = []
            for beta_clip in args.clip_betas:
                for beta_resnet in args.resnet_betas:
                    map_bank[f"dual_gate_resnet18_{layer}_clip{beta_clip:g}_resnet{beta_resnet:g}"] = []

        for data, mask, label, name, img_type in tqdm(loader, desc=f"Eval dual gallery seg {signal}/{scene}/{jsr}", leave=False):
            data_t = to_model_input(model, data, device, rgb_from_bgr=False)
            visual_features = model.encode_image(data_t)
            textual = model.calculate_textual_anomaly_score(visual_features, "seg").to(device)
            clip_map = model.calculate_visual_anomaly_score(visual_features).to(device)
            clip_norm = minmax_map(clip_map)

            raw = data.permute(0, 3, 1, 2).to(device, non_blocking=True)
            resnet_features = encoder(raw)

            batch_maps = {
                "textual_only": textual,
                "clip_gallery_only": clip_map,
            }
            for beta_clip in args.clip_betas:
                batch_maps[f"gate_clip_beta{beta_clip:g}"] = textual * (1.0 + float(beta_clip) * clip_norm)

            for layer in args.resnet_layers:
                resnet_map = cnn_distance_map(
                    resnet_features[layer],
                    resnet_galleries[layer],
                    args.distance_chunk_size,
                ).to(device)
                resnet_map = F.interpolate(resnet_map, size=textual.shape[-2:], mode="bilinear", align_corners=False)
                resnet_norm = minmax_map(resnet_map)
                batch_maps[f"resnet18_{layer}_gallery_only"] = resnet_map
                for beta_resnet in args.resnet_betas:
                    batch_maps[f"gate_resnet18_{layer}_beta{beta_resnet:g}"] = textual * (
                        1.0 + float(beta_resnet) * resnet_norm
                    )
                for beta_clip in args.clip_betas:
                    for beta_resnet in args.resnet_betas:
                        batch_maps[f"dual_gate_resnet18_{layer}_clip{beta_clip:g}_resnet{beta_resnet:g}"] = (
                            textual
                            * (1.0 + float(beta_clip) * clip_norm)
                            * (1.0 + float(beta_resnet) * resnet_norm)
                        )

            for key, maps in batch_maps.items():
                map_bank[key].extend(smooth_resize_maps(maps, args.resolution))

            for m in mask:
                m = m.numpy()
                m[m > 0] = 1
                gt_mask_list.append(m)

        dummy_imgs = [np.zeros((args.resolution, args.resolution, 3), dtype=np.uint8) for _ in gt_mask_list]
        row = {
            "method": "seg_dual_gallery_fusion",
            "task": "seg",
            "dataset": signal,
            "scene": scene,
            "jsr": jsr,
        }
        npz_payload = {}
        resized_masks = None
        for key, score_maps in map_bank.items():
            _, score_maps_r, masks_r = specify_resolution(dummy_imgs, score_maps, gt_mask_list, resolution=(args.resolution, args.resolution))
            if resized_masks is None:
                resized_masks = np.asarray(masks_r, dtype=np.uint8)
            score_arr = np.asarray(score_maps_r, dtype=np.float32)
            row[f"{key}_p_roc"] = metric_cal_pix(score_arr, masks_r)["p_roc"]
            if args.save_maps:
                npz_payload[key] = score_arr

        rows.append(row)
        if args.save_maps:
            np.savez_compressed(
                score_root / f"{signal}-{scene}-{jsr}-dual_gallery_seg_maps.npz",
                masks=resized_masks,
                **npz_payload,
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="analysis_outputs/20260629_seg_dual_gallery_fusion")
    parser.add_argument("--formal-baseline-csv", default="analysis_outputs/20260627_method_funnel/results_seg_pooled_rf_rgb.csv")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--signals", nargs="+", default=SIGNALS, choices=SIGNALS)
    parser.add_argument("--resnet-layers", nargs="+", default=["layer2", "layer3"], choices=["layer2", "layer3"])
    parser.add_argument("--clip-betas", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--resnet-betas", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--max-gallery-patches", type=int, default=50000)
    parser.add_argument("--distance-chunk-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
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
    parser.add_argument("--save-maps", action="store_true", default=False)
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
    resnet_galleries = build_resnet_gallery(encoder, train_loader, args, device)

    rows = evaluate(model, encoder, resnet_galleries, build_eval_loaders(args), args, device)
    rows = attach_formal_baseline(rows, args.formal_baseline_csv)

    out_root = Path(args.output_root)
    result_path = out_root / "results_seg_dual_gallery_fusion.csv"
    write_csv(result_path, rows)

    df = pd.DataFrame(rows)
    summary = {
        "method": "seg_dual_gallery_fusion",
        "checkpoint": args.checkpoint,
        "formal_baseline_csv": args.formal_baseline_csv,
        "clip_betas": args.clip_betas,
        "resnet": "resnet18_imagenet1k_v1",
        "resnet_layers": args.resnet_layers,
        "resnet_betas": args.resnet_betas,
        "max_gallery_patches": args.max_gallery_patches,
    }
    for col in [c for c in df.columns if c.endswith("_p_roc")]:
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
