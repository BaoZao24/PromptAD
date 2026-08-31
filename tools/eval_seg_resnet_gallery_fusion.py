#!/usr/bin/env python
"""Evaluate SEG maps fused with a ResNet normal-feature gallery.

This is an evaluation-only experiment. PromptAD's textual anomaly map is kept
unchanged. A frozen ImageNet ResNet18 extracts local CNN features from target
normal images to build a normal gallery; test patches are scored by nearest
normal CNN feature distance and fused with the textual map.
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
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torchvision.models import (
    ResNet18_Weights,
    Wide_ResNet50_2_Weights,
    resnet18,
    wide_resnet50_2,
)
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
    return [gaussian_filter(arr[i], sigma=4) for i in range(arr.shape[0])]


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


class ResNet18LocalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        net = resnet18(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.register_buffer(
            "mean",
            torch.tensor(weights.transforms().mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(weights.transforms().std).view(1, 3, 1, 1),
            persistent=False,
        )
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, images_bgr_uint8: torch.Tensor):
        x = images_bgr_uint8.float() / 255.0
        x = x[:, [2, 1, 0], :, :]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        x = self.stem(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return {
            "layer1": F.normalize(layer1.float(), dim=1),
            "layer2": F.normalize(layer2.float(), dim=1),
            "layer3": F.normalize(layer3.float(), dim=1),
            "layer4": F.normalize(layer4.float(), dim=1),
        }


class WideResNet50LocalEncoder(nn.Module):
    """Same interface as ResNet18LocalEncoder but with a WideResNet50-2 trunk."""

    def __init__(self):
        super().__init__()
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
        net = wide_resnet50_2(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.register_buffer(
            "mean",
            torch.tensor(weights.transforms().mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(weights.transforms().std).view(1, 3, 1, 1),
            persistent=False,
        )
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, images_bgr_uint8: torch.Tensor):
        x = images_bgr_uint8.float() / 255.0
        x = x[:, [2, 1, 0], :, :]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        x = self.stem(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return {
            "layer1": F.normalize(layer1.float(), dim=1),
            "layer2": F.normalize(layer2.float(), dim=1),
            "layer3": F.normalize(layer3.float(), dim=1),
            "layer4": F.normalize(layer4.float(), dim=1),
        }


def flatten_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    return feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1]).contiguous()


def cnn_tta_modes(args):
    mode = getattr(args, "cnn_paired_tta", "none")
    if mode == "none":
        return ["identity"]
    if mode == "stft_shift_blur":
        return ["identity", "blur", "time_shift_up", "time_shift_down"]
    raise ValueError(f"Unsupported CNN paired TTA mode: {mode}")


def normalize_cnn_input_batch(raw_batch: torch.Tensor, args) -> torch.Tensor:
    """Optional per-image normalization for experimental CNN-only evidence."""
    mode = getattr(args, "cnn_input_normalization", "raw")
    if mode == "raw":
        return raw_batch
    if mode != "robust_global":
        raise ValueError(f"Unsupported CNN input normalization: {mode}")
    gray = raw_batch.float().mean(dim=-1)
    flat = gray.flatten(1)
    median = flat.median(dim=1).values.view(-1, 1, 1)
    mad = (gray - median).abs().flatten(1).median(dim=1).values.view(-1, 1, 1)
    robust_z = (gray - median) / torch.clamp(1.4826 * mad, min=1.0)
    normalized = ((robust_z.clamp(-3.0, 3.0) + 3.0) / 6.0 * 255.0).round().to(torch.uint8)
    return normalized.unsqueeze(-1).repeat(1, 1, 1, 3)


def cnn_tta_batch(raw_batch: torch.Tensor, mode: str, args) -> torch.Tensor:
    if mode == "identity":
        return normalize_cnn_input_batch(raw_batch, args)
    variants = []
    shift = int(getattr(args, "cnn_paired_tta_shift_px", 4))
    blur_size = int(getattr(args, "cnn_paired_tta_blur_ksize", 3))
    if blur_size < 3 or blur_size % 2 == 0:
        raise ValueError("cnn_paired_tta_blur_ksize must be an odd integer >= 3")
    for arr_tensor in raw_batch:
        arr = arr_tensor.numpy()
        if mode == "blur":
            variant = cv2.GaussianBlur(arr, (blur_size, blur_size), 0)
        elif mode in {"time_shift_up", "time_shift_down"}:
            dy = -shift if mode == "time_shift_up" else shift
            matrix = np.float32([[1, 0, 0], [0, 1, dy]])
            h, w = arr.shape[:2]
            variant = cv2.warpAffine(
                arr, matrix, (w, h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        else:
            raise ValueError(f"Unsupported CNN paired TTA view: {mode}")
        variants.append(variant)
    return normalize_cnn_input_batch(torch.as_tensor(np.stack(variants, axis=0)), args)


def downsample_gallery(gallery: torch.Tensor, max_patches: int, seed: int) -> torch.Tensor:
    if max_patches <= 0 or gallery.shape[0] <= max_patches:
        return gallery
    gen = torch.Generator(device=gallery.device)
    gen.manual_seed(seed)
    idx = torch.randperm(gallery.shape[0], generator=gen, device=gallery.device)[:max_patches]
    return gallery[idx].contiguous()


@torch.no_grad()
def farthest_first_coreset(gallery: torch.Tensor, coreset_size: int, seed: int) -> torch.Tensor:
    """Keep diverse normal patches without changing the default full-memory path."""
    if coreset_size <= 0 or gallery.shape[0] <= coreset_size:
        return gallery

    count = gallery.shape[0]
    selected = torch.empty(coreset_size, dtype=torch.long, device=gallery.device)
    selected[0] = int(seed) % count
    # Features are L2-normalized, so maximizing cosine distance is equivalent
    # to minimizing cosine similarity to the selected set.
    closest_similarity = gallery @ gallery[selected[0]].unsqueeze(1)
    closest_similarity = closest_similarity.squeeze(1)
    for index in range(1, coreset_size):
        next_index = torch.argmin(closest_similarity)
        selected[index] = next_index
        similarity = gallery @ gallery[next_index].unsqueeze(1)
        closest_similarity = torch.maximum(closest_similarity, similarity.squeeze(1))
    return gallery[selected].contiguous()


@torch.no_grad()
def build_resnet_gallery(encoder, train_loader, args, device, tta_mode="identity"):
    encoder.eval()
    galleries = {layer: [] for layer in args.resnet_layers}
    for data, mask, label, name, img_type in tqdm(train_loader, desc="Build ResNet normal gallery", leave=False):
        data_variant = cnn_tta_batch(data, tta_mode, args)
        raw = data_variant.permute(0, 3, 1, 2).to(device, non_blocking=True)
        features = encoder(raw)
        for layer in args.resnet_layers:
            galleries[layer].append(flatten_feature_map(features[layer]).cpu())

    out = {}
    for layer, chunks in galleries.items():
        gallery = torch.cat(chunks, dim=0).to(device)
        gallery = downsample_gallery(gallery, args.max_gallery_patches, args.seed + len(layer))
        gallery = farthest_first_coreset(
            gallery,
            int(getattr(args, "cnn_coreset_size", 0)),
            args.seed + len(layer),
        )
        out[layer] = F.normalize(gallery, dim=1)
        print(
            f"[resnet_gallery] mode={tta_mode} {layer}: patches={out[layer].shape[0]} "
            f"dim={out[layer].shape[1]} coreset_size={getattr(args, 'cnn_coreset_size', 0)}"
        )
    return out


@torch.no_grad()
def cnn_distance_map(feature_map: torch.Tensor, gallery: torch.Tensor, chunk_size: int) -> torch.Tensor:
    n, c, h, w = feature_map.shape
    probes = flatten_feature_map(F.normalize(feature_map.float(), dim=1))
    scores = []
    for start in range(0, probes.shape[0], chunk_size):
        chunk = probes[start : start + chunk_size]
        sim = chunk @ gallery.t()
        scores.append((1.0 - sim.max(dim=1).values).cpu())
    score = torch.cat(scores, dim=0).reshape(n, h, w).unsqueeze(1)
    return score


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
            "gate_clip_beta0.25": [],
        }
        for layer in args.resnet_layers:
            map_bank[f"resnet18_{layer}_gallery_only"] = []
            for beta in args.betas:
                map_bank[f"gate_resnet18_{layer}_beta{beta:g}"] = []

        for data, mask, label, name, img_type in tqdm(loader, desc=f"Eval ResNet gallery {signal}/{scene}/{jsr}", leave=False):
            data_t = to_model_input(model, data, device, rgb_from_bgr=False)
            visual_features = model.encode_image(data_t)
            textual = model.calculate_textual_anomaly_score(visual_features, "seg").to(device)
            clip_gallery = model.calculate_visual_anomaly_score(visual_features).to(device)
            clip_norm = minmax_map(clip_gallery)

            raw = data.permute(0, 3, 1, 2).to(device, non_blocking=True)
            resnet_features = encoder(raw)
            batch_maps = {
                "textual_only": textual,
                "clip_gallery_only": clip_gallery,
                "gate_clip_beta0.25": textual * (1.0 + 0.25 * clip_norm),
            }
            for layer in args.resnet_layers:
                dist = cnn_distance_map(resnet_features[layer], resnet_galleries[layer], args.distance_chunk_size).to(device)
                dist = F.interpolate(dist, size=textual.shape[-2:], mode="bilinear", align_corners=False)
                dist_norm = minmax_map(dist)
                batch_maps[f"resnet18_{layer}_gallery_only"] = dist
                for beta in args.betas:
                    batch_maps[f"gate_resnet18_{layer}_beta{beta:g}"] = textual * (1.0 + float(beta) * dist_norm)

            for key, maps in batch_maps.items():
                map_bank[key].extend(smooth_resize_maps(maps, args.resolution))

            for m in mask:
                m = m.numpy()
                m[m > 0] = 1
                gt_mask_list.append(m)

        dummy_imgs = [np.zeros((args.resolution, args.resolution, 3), dtype=np.uint8) for _ in gt_mask_list]
        row = {
            "method": "seg_resnet_gallery_fusion",
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
                score_root / f"{signal}-{scene}-{jsr}-resnet_gallery_seg_maps.npz",
                masks=resized_masks,
                **npz_payload,
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="analysis_outputs/20260629_seg_resnet_gallery_fusion")
    parser.add_argument("--formal-baseline-csv", default="analysis_outputs/20260627_method_funnel/results_seg_pooled_rf_rgb.csv")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--signals", nargs="+", default=SIGNALS, choices=SIGNALS)
    parser.add_argument("--resnet-layers", nargs="+", default=["layer2", "layer3"], choices=["layer2", "layer3"])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
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
    result_path = out_root / "results_seg_resnet_gallery_fusion.csv"
    write_csv(result_path, rows)

    df = pd.DataFrame(rows)
    summary = {
        "method": "seg_resnet_gallery_fusion",
        "checkpoint": args.checkpoint,
        "formal_baseline_csv": args.formal_baseline_csv,
        "resnet": "resnet18_imagenet1k_v1",
        "resnet_layers": args.resnet_layers,
        "betas": args.betas,
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
