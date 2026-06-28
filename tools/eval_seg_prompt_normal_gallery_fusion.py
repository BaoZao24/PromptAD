#!/usr/bin/env python
"""Evaluate PromptAD SEG maps fused with target-normal patch distance maps.

No training is performed. The script loads a pooled RF segmentation checkpoint,
computes textual anomaly maps, visual normal-patch-distance maps, and several
fusion maps on the same target test cells.
"""

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
from scipy.ndimage import gaussian_filter
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
from utils.eval_utils import specify_resolution
from utils.metrics import metric_cal_pix
from utils.training_utils import setup_seed


def minmax_map(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(1)
    lo = flat.min(dim=1).values[:, None, None, None]
    hi = flat.max(dim=1).values[:, None, None, None]
    return (x - lo) / (hi - lo).clamp_min(1e-6)


def smooth_resize_maps(maps: torch.Tensor, resolution: int):
    maps = F.interpolate(maps, size=(resolution, resolution), mode="nearest")
    arr = maps.detach().squeeze(1).cpu().numpy()
    out = []
    for i in range(arr.shape[0]):
        out.append(gaussian_filter(arr[i], sigma=4))
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
def evaluate(model, eval_loaders, args, device):
    model.eval_mode()
    rows = []
    score_root = Path(args.output_root) / "scores"
    score_root.mkdir(parents=True, exist_ok=True)

    for signal, scene, jsr, loader in eval_loaders:
        gt_mask_list = []
        map_bank = {
            "textual_only": [],
            "normal_patch_only": [],
            "current_harmonic": [],
        }
        for beta in args.betas:
            map_bank[f"gate_norm_beta{beta:g}"] = []
            map_bank[f"add_norm_beta{beta:g}"] = []

        for data, mask, label, name, img_type in tqdm(loader, desc=f"Eval seg fusion {signal}/{scene}/{jsr}", leave=False):
            data_t = to_model_input(model, data, device, rgb_from_bgr=False)
            visual_features = model.encode_image(data_t)
            textual = model.calculate_textual_anomaly_score(visual_features, "seg").to(device)
            normal_patch = model.calculate_visual_anomaly_score(visual_features).to(device)
            normal_norm = minmax_map(normal_patch)
            harmonic = 1.0 / (1.0 / textual.clamp_min(1e-6) + 1.0 / normal_patch.clamp_min(1e-6))

            batch_maps = {
                "textual_only": textual,
                "normal_patch_only": normal_patch,
                "current_harmonic": harmonic,
            }
            for beta in args.betas:
                batch_maps[f"gate_norm_beta{beta:g}"] = textual * (1.0 + float(beta) * normal_norm)
                batch_maps[f"add_norm_beta{beta:g}"] = textual + float(beta) * normal_norm

            for key, maps in batch_maps.items():
                map_bank[key].extend(smooth_resize_maps(maps, args.resolution))

            for m in mask:
                m = m.numpy()
                m[m > 0] = 1
                gt_mask_list.append(m)

        dummy_imgs = [np.zeros((args.resolution, args.resolution, 3), dtype=np.uint8) for _ in gt_mask_list]
        resized_masks = None
        row = {
            "method": "seg_prompt_normal_gallery_fusion",
            "task": "seg",
            "dataset": signal,
            "scene": scene,
            "jsr": jsr,
        }
        npz_payload = {}
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
                score_root / f"{signal}-{scene}-{jsr}-seg_maps.npz",
                masks=resized_masks,
                **npz_payload,
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="analysis_outputs/20260628_seg_prompt_normal_gallery_fusion")
    parser.add_argument("--formal-baseline-csv", default="analysis_outputs/20260627_method_funnel/results_seg_pooled_rf_rgb.csv")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--signals", nargs="+", default=SIGNALS, choices=SIGNALS)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
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

    rows = evaluate(model, build_eval_loaders(args), args, device)
    rows = attach_formal_baseline(rows, args.formal_baseline_csv)

    out_root = Path(args.output_root)
    result_path = out_root / "results_seg_prompt_normal_gallery_fusion.csv"
    write_csv(result_path, rows)

    df = pd.DataFrame(rows)
    summary = {
        "method": "seg_prompt_normal_gallery_fusion",
        "checkpoint": args.checkpoint,
        "formal_baseline_csv": args.formal_baseline_csv,
        "betas": args.betas,
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
