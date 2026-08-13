#!/usr/bin/env python
"""Evaluate a CLIP-ViT patch nearest-neighbour gallery.

This is not the official PatchCore CNN/ResNet pipeline. It only reuses the
nearest-neighbour memory-bank scoring idea on PromptAD/CLIP ViT patch features.
The ``vit_patchcore`` score name identifies this ViT memory-bank branch.
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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PromptAD import PromptAD
from datasets.rf_target import (
    RF_JSR_BY_SIGNAL as JSR_BY_SIGNAL,
    RF_SCENES as SCENES,
    RF_TARGET_SIGNALS as SIGNALS,
    collect_rf_target_samples as collect_samples,
)
from tools.eval_cls_vit_patch_gallery import (
    attach_formal_baseline,
    harmonic,
    top_ratio_score,
)
from tools.eval_seg_resnet_gallery_fusion import load_checkpoint
from train_rf_target_pooled_universal import (
    RFPathDataset,
    to_model_input,
)
from utils.rf_scene_support import (
    cell_entry,
    ensure_rf_target_scene_manifest,
    scene_support_entry,
)
from utils.spectral_tta import SPECTRAL_TTA_BUNDLES, augment_spectrogram
from utils.training_utils import setup_seed


def safe_auc(labels, scores) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores) * 100.0)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_patch_features(visual_features):
    """Return the fixed layer1 + layer2 feature used by the formal method."""

    layer1 = F.normalize(visual_features[2].float(), dim=-1)
    layer2 = F.normalize(visual_features[3].float(), dim=-1)
    return F.normalize(torch.cat([layer1, layer2], dim=-1), dim=-1)


def _shift_image(arr, dx=0, dy=0):
    h, w = arr.shape[:2]
    mat = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(arr, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _center_crop_resize(arr, crop_ratio=0.03):
    h, w = arr.shape[:2]
    mx = max(1, int(round(w * crop_ratio)))
    my = max(1, int(round(h * crop_ratio)))
    cropped = arr[my:h - my, mx:w - mx]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def _paired_tta_modes(args):
    mode = getattr(args, "paired_tta", "none")
    if mode == "none":
        return ["identity"]
    if mode == "visionad_safe":
        return ["identity", "contrast", "blur", "crop"]
    if mode == "stft_shift_blur":
        return ["identity", "blur", "time_shift_up", "time_shift_down"]
    if mode == "stft_time_shift_v2":
        return ["identity", "time_shift_up", "time_shift_down", "time_shift_up_large", "time_shift_down_large"]
    if mode == "ofdma_time_shift_blur":
        return ["identity", "blur", "time_shift_left", "time_shift_right"]
    if mode in SPECTRAL_TTA_BUNDLES:
        return list(SPECTRAL_TTA_BUNDLES[mode])
    raise ValueError(f"Unsupported paired TTA mode: {mode}")


def _augment_array(arr, mode, args):
    if mode == "identity":
        return arr
    if mode == "contrast":
        return np.clip(arr.astype(np.float32) * float(args.paired_tta_contrast) + float(args.paired_tta_brightness), 0, 255).astype(np.uint8)
    if mode == "blur":
        k = int(getattr(args, "paired_tta_blur_ksize", 3))
        if k < 3 or k % 2 == 0:
            raise ValueError("paired_tta_blur_ksize must be an odd integer >= 3")
        return cv2.GaussianBlur(arr, (k, k), 0)
    if mode == "crop":
        return _center_crop_resize(arr, crop_ratio=float(args.paired_tta_crop_ratio))
    if mode == "time_shift":
        return _shift_image(arr, dx=0, dy=int(args.paired_tta_shift_px))
    if mode == "time_shift_up":
        return _shift_image(arr, dx=0, dy=-int(args.paired_tta_shift_px))
    if mode == "time_shift_down":
        return _shift_image(arr, dx=0, dy=int(args.paired_tta_shift_px))
    if mode == "time_shift_up_large":
        return _shift_image(arr, dx=0, dy=-2 * int(args.paired_tta_shift_px))
    if mode == "time_shift_down_large":
        return _shift_image(arr, dx=0, dy=2 * int(args.paired_tta_shift_px))
    if mode == "time_shift_left":
        return _shift_image(arr, dx=-int(args.paired_tta_shift_px), dy=0)
    if mode == "time_shift_right":
        return _shift_image(arr, dx=int(args.paired_tta_shift_px), dy=0)
    if mode in {
        transform
        for bundle in SPECTRAL_TTA_BUNDLES.values()
        for transform in bundle
    }:
        return augment_spectrogram(
            arr,
            mode,
            shift_px=int(getattr(args, "paired_tta_shift_px", 2)),
            blur_ksize=int(getattr(args, "paired_tta_blur_ksize", 3)),
            background_noise_strength=float(getattr(args, "paired_tta_background_noise_strength", 3.0)),
            frequency_response_strength=float(getattr(args, "paired_tta_frequency_response_strength", 3.0)),
        )
    raise ValueError(f"Unsupported paired TTA transform: {mode}")


def paired_tta_batch(raw_batch, mode, args):
    if mode == "identity":
        return raw_batch
    variants = []
    for arr_t in raw_batch:
        variants.append(_augment_array(arr_t.numpy(), mode, args))
    return torch.as_tensor(np.stack(variants, axis=0))


def support_augmented_batches(raw_batch, args):
    """Return deterministic weak normal-support augmentations.

    Augmentation is only used while building the normal gallery. It never
    changes test images.
    """
    if getattr(args, "support_augment", "none") == "none":
        return [raw_batch]
    if args.support_augment not in {"rf_weak", "time_shift"}:
        raise ValueError(f"Unsupported support augmentation: {args.support_augment}")

    shift = int(args.support_shift_px)
    if args.support_augment == "time_shift":
        variants = [[], []]
        for arr_t in raw_batch:
            arr = arr_t.numpy()
            variants[0].append(_shift_image(arr, dx=0, dy=shift))
            variants[1].append(_shift_image(arr, dx=0, dy=-shift))
        out = [raw_batch]
        out.extend(torch.as_tensor(np.stack(batch, axis=0)) for batch in variants)
        return out

    variants = [[] for _ in range(5)]
    for arr_t in raw_batch:
        arr = arr_t.numpy()
        variants[0].append(np.clip(arr.astype(np.float32) * 1.06 + 2.0, 0, 255).astype(np.uint8))
        variants[1].append(cv2.GaussianBlur(arr, (3, 3), 0))
        variants[2].append(_shift_image(arr, dx=shift, dy=0))
        variants[3].append(_shift_image(arr, dx=0, dy=max(1, shift // 2)))
        variants[4].append(_center_crop_resize(arr, crop_ratio=float(args.support_crop_ratio)))
    out = [raw_batch]
    out.extend(torch.as_tensor(np.stack(batch, axis=0)) for batch in variants)
    return out


def select_gallery_subset(gallery, args):
    if args.coreset_ratio >= 1.0:
        return gallery, None
    keep = max(1, int(round(gallery.shape[0] * float(args.coreset_ratio))))
    if args.coreset_method == "random":
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        return torch.randperm(gallery.shape[0], generator=generator, device="cpu")[:keep].to(gallery.device), None
    if args.coreset_method == "farthest":
        selected = torch.empty(keep, device=gallery.device, dtype=torch.long)
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        first = int(torch.randint(gallery.shape[0], (1,), generator=generator).item())
        selected[0] = first
        min_dist = torch.full((gallery.shape[0],), float("inf"), device=gallery.device)
        for i in range(1, keep):
            sim = gallery @ gallery[selected[i - 1]].unsqueeze(-1)
            dist = (1.0 - sim.squeeze(-1)) / 2.0
            min_dist = torch.minimum(min_dist, dist)
            selected[i] = torch.argmax(min_dist)
        return selected, None
    raise ValueError(f"Unsupported coreset method: {args.coreset_method}")


def select_gallery_subset_by_rows(gallery, rows, args, cols=None):
    if args.coreset_ratio >= 1.0:
        return gallery, rows, cols
    selected = []
    for row in torch.unique(rows).tolist():
        row_idx = torch.nonzero(rows == int(row), as_tuple=False).flatten()
        row_gallery = gallery[row_idx]
        keep = max(1, int(round(row_gallery.shape[0] * float(args.coreset_ratio))))
        if args.coreset_method == "random":
            generator = torch.Generator(device="cpu").manual_seed(args.seed + int(row))
            local = torch.randperm(row_gallery.shape[0], generator=generator, device="cpu")[:keep].to(gallery.device)
        elif args.coreset_method == "farthest":
            local = torch.empty(keep, device=gallery.device, dtype=torch.long)
            generator = torch.Generator(device="cpu").manual_seed(args.seed + int(row))
            local[0] = int(torch.randint(row_gallery.shape[0], (1,), generator=generator).item())
            min_dist = torch.full((row_gallery.shape[0],), float("inf"), device=gallery.device)
            for i in range(1, keep):
                sim = row_gallery @ row_gallery[local[i - 1]].unsqueeze(-1)
                dist = (1.0 - sim.squeeze(-1)) / 2.0
                min_dist = torch.minimum(min_dist, dist)
                local[i] = torch.argmax(min_dist)
        else:
            raise ValueError(f"Unsupported coreset method: {args.coreset_method}")
        selected.append(row_idx[local])
    idx = torch.cat(selected, dim=0)
    out_cols = cols[idx] if cols is not None else None
    return gallery[idx], rows[idx], out_cols


@torch.no_grad()
def build_vit_patchcore_gallery(model, train_loader, args, device):
    model.eval_mode()
    patches = []
    for data, mask, label, name, img_type in tqdm(train_loader, desc="Build CLIP-ViT NN gallery", leave=False):
        data_t = to_model_input(model, data, device, rgb_from_bgr=True)
        visual_features = model.encode_image(data_t)
        patch_features = prepare_patch_features(visual_features)
        patches.append(patch_features.reshape(-1, patch_features.shape[-1]))
    gallery = torch.cat(patches, dim=0)
    if args.coreset_ratio < 1.0:
        idx, _ = select_gallery_subset(F.normalize(gallery.float(), dim=-1), args)
        gallery = gallery[idx]
    return F.normalize(gallery.float(), dim=-1).contiguous()


def min_cosine_distance_chunked(probe, gallery, chunk_size):
    chunk_size = max(1, int(chunk_size))
    best_sim = None
    for start in range(0, gallery.shape[0], chunk_size):
        chunk = gallery[start:start + chunk_size]
        sim, _ = (probe @ chunk.t()).max(dim=-1)
        best_sim = sim if best_sim is None else torch.maximum(best_sim, sim)
    return (1.0 - best_sim) / 2.0


def topk_cosine_distance_chunked(probe, gallery, chunk_size, k):
    chunk_size = max(1, int(chunk_size))
    k = max(1, min(int(k), int(gallery.shape[0])))
    best = None
    for start in range(0, gallery.shape[0], chunk_size):
        chunk = gallery[start:start + chunk_size]
        sim = probe @ chunk.t()
        local = torch.topk(sim, k=min(k, sim.shape[1]), dim=-1).values
        best = local if best is None else torch.topk(torch.cat([best, local], dim=-1), k=k, dim=-1).values
    return ((1.0 - best) / 2.0).mean(dim=-1)


def aggregate_topk_similarity(top_sims, args):
    if args.nn_agg == "mean":
        return top_sims.mean(dim=-1)
    if args.nn_agg == "weighted":
        weights = torch.softmax(top_sims / max(float(args.nn_weight_temp), 1e-6), dim=-1)
        return (weights * top_sims).sum(dim=-1)
    if args.nn_agg == "adaptive":
        best = top_sims[:, :1]
        keep = top_sims >= (best - float(args.adaptive_sim_margin))
        counts = keep.sum(dim=-1).clamp(min=1)
        return (top_sims * keep.float()).sum(dim=-1) / counts.float()
    raise ValueError(f"Unsupported nn aggregation: {args.nn_agg}")


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


def soft_position_cosine_distance_chunked(probe, probe_pos, gallery, gallery_pos, chunk_size, args, max_pos):
    """Nearest-neighbour distance with a soft spectrogram-position prior.

    The feature match is still global, but matches farther away on the selected
    spectrogram axis receive a small similarity penalty instead of being
    discarded.
    """
    chunk_size = max(1, int(chunk_size))
    k = max(1, min(int(args.nn_topk), int(gallery.shape[0])))
    denom = max(1.0, float(max_pos))
    best = None
    probe_pos = probe_pos.float().view(-1, 1)
    weight = float(args.position_soft_weight)
    for start in range(0, gallery.shape[0], chunk_size):
        chunk = gallery[start:start + chunk_size]
        chunk_pos = gallery_pos[start:start + chunk_size].float().view(1, -1)
        sim = probe @ chunk.t()
        penalty = weight * (probe_pos - chunk_pos).abs() / denom
        adjusted = sim - penalty
        local = torch.topk(adjusted, k=min(k, adjusted.shape[1]), dim=-1).values
        best = local if best is None else torch.topk(torch.cat([best, local], dim=-1), k=k, dim=-1).values
    sim_score = aggregate_topk_similarity(best, args)
    return (1.0 - sim_score) / 2.0


def patch_row_indices(num_images, grid_h, grid_w, device):
    rows = torch.arange(grid_h, device=device).repeat_interleave(grid_w)
    return rows.repeat(int(num_images))


def patch_col_indices(num_images, grid_h, grid_w, device):
    cols = torch.arange(grid_w, device=device).repeat(grid_h)
    return cols.repeat(int(num_images))


def finalize_vit_nn_gallery(gallery, rows, cols, args):
    """Normalize and coreset a gallery after all intended views are present."""
    gallery = F.normalize(gallery.float(), dim=-1)
    if args.coreset_ratio < 1.0:
        if args.rowwise_coreset:
            gallery, rows, cols = select_gallery_subset_by_rows(gallery, rows, args, cols)
        else:
            idx, _ = select_gallery_subset(gallery, args)
            gallery = gallery[idx]
            rows = rows[idx]
            cols = cols[idx]
    return gallery.contiguous(), rows.contiguous(), cols.contiguous()


@torch.no_grad()
def build_vit_nn_gallery_with_rows(
    model,
    train_loader,
    args,
    device,
    paired_tta_mode=None,
    apply_coreset=True,
):
    model.eval_mode()
    patches = []
    row_ids = []
    col_ids = []
    grid_h, grid_w = model.grid_size
    for data, mask, label, name, img_type in tqdm(train_loader, desc="Build CLIP-ViT NN gallery", leave=False):
        if paired_tta_mode is None:
            data_variants = support_augmented_batches(data, args)
        else:
            data_variants = [paired_tta_batch(data, paired_tta_mode, args)]
        for data_variant in data_variants:
            data_t = to_model_input(model, data_variant, device, rgb_from_bgr=True)
            visual_features = model.encode_image(data_t)
            patch_features = prepare_patch_features(visual_features)
            patches.append(patch_features.reshape(-1, patch_features.shape[-1]))
            row_ids.append(patch_row_indices(patch_features.shape[0], grid_h, grid_w, device))
            col_ids.append(patch_col_indices(patch_features.shape[0], grid_h, grid_w, device))
    gallery = torch.cat(patches, dim=0)
    rows = torch.cat(row_ids, dim=0)
    cols = torch.cat(col_ids, dim=0)
    if apply_coreset:
        return finalize_vit_nn_gallery(gallery, rows, cols, args)
    return F.normalize(gallery.float(), dim=-1).contiguous(), rows.contiguous(), cols.contiguous()


def constrained_candidates(gallery, gallery_rows, row, freq_window):
    if freq_window < 0:
        return gallery
    mask = (gallery_rows - int(row)).abs() <= int(freq_window)
    if not torch.any(mask):
        return gallery
    return gallery[mask]


def spectrogram_nn_scores(patch_features, gallery, gallery_rows, args, grid_h, grid_w, row_stats=None, gallery_cols=None):
    flat = patch_features.reshape(-1, patch_features.shape[-1])
    if float(args.position_soft_weight) > 0:
        if args.position_soft_axis == "frequency":
            probe_pos = patch_col_indices(patch_features.shape[0], grid_h, grid_w, flat.device)
            gallery_pos = gallery_cols
            max_pos = grid_w - 1
        else:
            probe_pos = patch_row_indices(patch_features.shape[0], grid_h, grid_w, flat.device)
            gallery_pos = gallery_rows
            max_pos = grid_h - 1
        out = soft_position_cosine_distance_chunked(
            flat,
            probe_pos,
            gallery,
            gallery_pos,
            args.gallery_chunk_size,
            args,
            max_pos,
        )
        return out.reshape(patch_features.shape[0], grid_h, grid_w)

    out = torch.empty(flat.shape[0], device=flat.device, dtype=torch.float32)
    probe_rows = patch_row_indices(patch_features.shape[0], grid_h, grid_w, flat.device)
    for row in range(grid_h):
        probe_mask = probe_rows == row
        if not torch.any(probe_mask):
            continue
        candidates = constrained_candidates(gallery, gallery_rows, row, args.freq_window)
        out[probe_mask] = robust_cosine_distance_chunked(
            flat[probe_mask],
            candidates,
            args.gallery_chunk_size,
            args,
        )
    if row_stats is not None:
        mu, sigma = row_stats
        out = (out - mu[probe_rows]) / (sigma[probe_rows] + 1e-6)
    return out.reshape(patch_features.shape[0], grid_h, grid_w)


def connected_region_score(map_np, top_ratio=0.05, alpha=0.5):
    scores = []
    for one in map_np:
        flat = one.reshape(-1)
        keep = max(1, int(round(flat.size * float(top_ratio))))
        threshold = np.partition(flat, flat.size - keep)[flat.size - keep]
        binary = (one >= threshold).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            scores.append(float(flat.max()))
            continue
        component_scores = []
        for comp_id in range(1, num_labels):
            area = float(stats[comp_id, cv2.CC_STAT_AREA])
            comp_mask = labels == comp_id
            comp_mean = float(one[comp_mask].mean())
            area_bonus = np.sqrt(area / float(flat.size))
            component_scores.append(comp_mean * (1.0 + float(alpha) * area_bonus))
        scores.append(float(max(component_scores)))
    return np.asarray(scores, dtype=np.float32)


def compute_patch_map(
    model,
    raw_batch,
    gallery,
    gallery_rows,
    gallery_cols,
    args,
    device,
    row_stats=None,
    row_prototypes=None,
    paired_tta_mode="identity",
    visual_features=None,
):
    if visual_features is None or paired_tta_mode != "identity":
        data_variant = paired_tta_batch(raw_batch, paired_tta_mode, args)
        data_t = to_model_input(model, data_variant, device, rgb_from_bgr=False)
        visual_features = model.encode_image(data_t)
    patch_features = prepare_patch_features(visual_features)
    bsz, num_patches, _ = patch_features.shape
    grid_h, grid_w = model.grid_size
    if args.memory_mode == "row_proto":
        map_t = row_prototype_scores(
            patch_features,
            row_prototypes[0],
            row_prototypes[1],
            args,
            grid_h,
            grid_w,
        )
    elif gallery_rows is None and args.nn_topk == 1 and args.freq_window < 0 and row_stats is None:
        patch_scores = min_cosine_distance_chunked(
            patch_features.reshape(-1, patch_features.shape[-1]),
            gallery,
            args.gallery_chunk_size,
        ).reshape(bsz, num_patches)
        map_t = patch_scores.reshape(bsz, grid_h, grid_w)
    else:
        map_t = spectrogram_nn_scores(
            patch_features,
            gallery,
            gallery_rows,
            args,
            grid_h,
            grid_w,
            row_stats,
            gallery_cols,
        )
    return map_t


@torch.no_grad()
def build_row_prototypes(gallery, gallery_rows, grid_h):
    dim = gallery.shape[-1]
    protos = torch.zeros(grid_h, dim, device=gallery.device, dtype=torch.float32)
    scales = torch.ones(grid_h, device=gallery.device, dtype=torch.float32)
    for row in range(grid_h):
        row_gallery = gallery[gallery_rows == row]
        if row_gallery.numel() == 0:
            continue
        proto = F.normalize(row_gallery.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        protos[row] = proto
        dists = (1.0 - (row_gallery @ proto)) / 2.0
        scales[row] = torch.clamp(dists.std(unbiased=False), min=1e-6)
    return protos.contiguous(), scales.contiguous()


def row_prototype_scores(patch_features, row_prototypes, row_scales, args, grid_h, grid_w):
    bsz = patch_features.shape[0]
    maps = []
    for row in range(grid_h):
        start = row * grid_w
        end = start + grid_w
        row_feat = patch_features[:, start:end, :]
        sim = row_feat @ row_prototypes[row].view(-1, 1)
        score = (1.0 - sim.squeeze(-1)) / 2.0
        if args.row_proto_zscore:
            score = score / (row_scales[row] + 1e-6)
        maps.append(score)
    return torch.stack(maps, dim=1).reshape(bsz, grid_h, grid_w)


@torch.no_grad()
def build_frequency_row_stats(gallery, gallery_rows, args, grid_h):
    means = torch.zeros(grid_h, device=gallery.device, dtype=torch.float32)
    stds = torch.ones(grid_h, device=gallery.device, dtype=torch.float32)
    for row in range(grid_h):
        probe_mask = gallery_rows == row
        if not torch.any(probe_mask):
            continue
        candidates = constrained_candidates(gallery, gallery_rows, row, args.freq_window)
        k = max(1, min(int(args.nn_topk) + 1, int(candidates.shape[0])))
        sims = gallery[probe_mask] @ candidates.t()
        vals = torch.topk(sims, k=k, dim=-1).values
        if vals.shape[1] > 1:
            vals = vals[:, 1:]
        dists = ((1.0 - vals) / 2.0).mean(dim=-1)
        means[row] = dists.mean()
        std = dists.std(unbiased=False)
        stds[row] = torch.clamp(std, min=1e-6)
    return means, stds


def build_target_scene_support_loader(manifest, scene, args):
    support = scene_support_entry(manifest, scene)["support"]
    samples = [
        (
            item["path"],
            0,
            0,
            f"{scene}_normal",
            f"scene-support-{scene}",
        )
        for item in support
    ]
    return (
        DataLoader(
            RFPathDataset(samples),
            batch_size=min(args.batch_size, len(samples)),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
        samples,
    )


def build_target_scene_eval_loaders(manifest, scene, args):
    loaders = []
    for signal in args.signals:
        for jsr in JSR_BY_SIGNAL[signal]:
            allowed_normal_paths = {
                item["path"]
                for item in cell_entry(manifest, signal, scene, jsr)["test_normals"]
            }
            allowed_abnormal_paths = {
                item["path"]
                for item in cell_entry(manifest, signal, scene, jsr)["test_abnormals"]
            }
            samples = [
                sample
                for sample in collect_samples(signal, scene, jsr, "test", k_shot=1)
                if (
                    (int(sample[2]) == 0 and str(sample[0]) in allowed_normal_paths)
                    or (
                        int(sample[2]) == 1
                        and str(sample[0]) in allowed_abnormal_paths
                    )
                )
            ]
            if args.max_test_normals > 0:
                normals = [sample for sample in samples if int(sample[2]) == 0][
                    : args.max_test_normals
                ]
                abnormals = [sample for sample in samples if int(sample[2]) == 1]
                samples = normals + abnormals
            if args.max_abnormals > 0:
                normals = [sample for sample in samples if int(sample[2]) == 0]
                abnormals = [sample for sample in samples if int(sample[2]) == 1][
                    : args.max_abnormals
                ]
                samples = normals + abnormals
            loaders.append(
                (
                    signal,
                    scene,
                    jsr,
                    DataLoader(
                        RFPathDataset(samples),
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                    ),
                )
            )
    return loaders


def prepare_target_scene_manifest(args):
    manifest_path = (
        Path(args.support_manifest)
        if args.support_manifest
        else Path(args.output_root) / "support_manifest.json"
    )
    manifest = ensure_rf_target_scene_manifest(
        manifest_path,
        collect_samples=collect_samples,
        signals=args.signals,
        scenes=args.scenes,
        jsr_by_signal=JSR_BY_SIGNAL,
        normal_sampling=args.normal_sampling,
        seed=(args.support_seed if args.support_seed is not None else args.seed),
    )
    args.support_manifest = str(manifest_path)
    args.support_manifest_sha256 = manifest["manifest_sha256"]
    return manifest


def build_gallery_state(model, train_loader, args, device):
    paired_tta_galleries = None
    if args.paired_tta == "none":
        gallery, gallery_rows, gallery_cols = build_vit_nn_gallery_with_rows(
            model,
            train_loader,
            args,
            device,
        )
    elif args.paired_tta_memory_layout == "separate":
        paired_tta_galleries = []
        for mode in _paired_tta_modes(args):
            tta_gallery, tta_rows, tta_cols = build_vit_nn_gallery_with_rows(
                model,
                train_loader,
                args,
                device,
                paired_tta_mode=mode,
            )
            paired_tta_galleries.append(
                (mode, tta_gallery, tta_rows, tta_cols)
            )
        gallery, gallery_rows, gallery_cols = paired_tta_galleries[0][1:]
    else:
        # The merged layout is the support-augmentation alternative: all four
        # normal views share one coreset, and each test image is encoded once.
        gallery_parts = []
        row_parts = []
        col_parts = []
        for mode in _paired_tta_modes(args):
            tta_gallery, tta_rows, tta_cols = build_vit_nn_gallery_with_rows(
                model,
                train_loader,
                args,
                device,
                paired_tta_mode=mode,
                apply_coreset=False,
            )
            gallery_parts.append(tta_gallery)
            row_parts.append(tta_rows)
            col_parts.append(tta_cols)
        gallery, gallery_rows, gallery_cols = finalize_vit_nn_gallery(
            torch.cat(gallery_parts, dim=0),
            torch.cat(row_parts, dim=0),
            torch.cat(col_parts, dim=0),
            args,
        )
    row_stats = (
        build_frequency_row_stats(
            gallery,
            gallery_rows,
            args,
            model.grid_size[0],
        )
        if args.freq_zscore
        else None
    )
    row_prototypes = (
        build_row_prototypes(gallery, gallery_rows, model.grid_size[0])
        if args.memory_mode == "row_proto"
        else None
    )
    return {
        "gallery": gallery,
        "gallery_rows": gallery_rows,
        "gallery_cols": gallery_cols,
        "row_stats": row_stats,
        "row_prototypes": row_prototypes,
        "paired_tta_galleries": paired_tta_galleries,
    }


def gallery_patch_count(state):
    """Count stored features across all paired memories, not just identity."""
    paired = state.get("paired_tta_galleries")
    if paired:
        return int(sum(view_gallery.shape[0] for _mode, view_gallery, _rows, _cols in paired))
    return int(state["gallery"].shape[0])


@torch.no_grad()
def compute_fused_patch_map(
    model,
    raw_batch,
    gallery,
    gallery_rows,
    gallery_cols,
    args,
    device,
    row_stats=None,
    row_prototypes=None,
    paired_tta_galleries=None,
    visual_features=None,
):
    if paired_tta_galleries:
        maps = [
            compute_patch_map(
                model,
                raw_batch,
                tta_gallery,
                tta_rows,
                tta_cols,
                args,
                device,
                row_stats=None,
                row_prototypes=None,
                paired_tta_mode=mode,
                visual_features=(visual_features if mode == "identity" else None),
            )
            for mode, tta_gallery, tta_rows, tta_cols in paired_tta_galleries
        ]
        stacked = torch.stack(maps, dim=0)
        if args.paired_tta_fusion == "mean":
            return stacked.mean(dim=0)
        if args.paired_tta_fusion == "max":
            return stacked.max(dim=0).values
        raise ValueError(f"Unsupported paired TTA fusion: {args.paired_tta_fusion}")
    return compute_patch_map(
        model,
        raw_batch,
        gallery,
        gallery_rows,
        gallery_cols,
        args,
        device,
        row_stats=row_stats,
        row_prototypes=row_prototypes,
        paired_tta_mode="identity",
        visual_features=visual_features,
    )


@torch.no_grad()
def evaluate(
    model,
    gallery,
    eval_loaders,
    args,
    device,
    gallery_rows=None,
    gallery_cols=None,
    row_stats=None,
    row_prototypes=None,
    paired_tta_galleries=None,
):
    model.eval_mode()
    model.build_text_feature_gallery()
    rows = []
    score_dir = Path(args.output_root) / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    for signal, scene, jsr, loader in eval_loaders:
        labels, names = [], []
        text_scores, max_scores = [], []
        coherent_scores = []
        ratio_scores = {ratio: [] for ratio in args.map_top_ratios}

        for data, mask, label, name, img_type in tqdm(loader, desc=f"Eval CLIP-ViT NN {signal}/{scene}/{jsr}", leave=False):
            data_t = to_model_input(model, data, device, rgb_from_bgr=False)
            visual_features = model.encode_image(data_t)
            text_np = np.asarray(model.calculate_textual_anomaly_score(visual_features, "cls"), dtype=np.float32)

            map_t = compute_fused_patch_map(
                model,
                data,
                gallery,
                gallery_rows,
                gallery_cols,
                args,
                device,
                row_stats,
                row_prototypes,
                paired_tta_galleries,
                visual_features=visual_features,
            )
            map_np = map_t.detach().cpu().numpy().astype(np.float32)

            text_scores.extend(float(x) for x in text_np)
            max_scores.extend(float(x) for x in map_np.reshape(map_np.shape[0], -1).max(axis=1))
            if float(args.coherence_alpha) > 0:
                coherent_scores.extend(float(x) for x in connected_region_score(
                    map_np,
                    top_ratio=float(args.coherence_top_ratio),
                    alpha=float(args.coherence_alpha),
                ))
            for ratio in args.map_top_ratios:
                ratio_scores[ratio].extend(float(x) for x in top_ratio_score(map_np, ratio))
            labels.extend(int(x) for x in label.numpy().tolist())
            names.extend(list(name))

        labels_np = np.asarray(labels, dtype=np.int32)
        text_np = np.asarray(text_scores, dtype=np.float32)
        max_np = np.asarray(max_scores, dtype=np.float32)
        coherent_np = np.asarray(coherent_scores, dtype=np.float32) if coherent_scores else None
        row = {
            "method": "vit_patchcore_gallery_cls",
            "task": "cls",
            "dataset": signal,
            "scene": scene,
            "jsr": jsr,
            "patch_layer": "concat",
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
            "text_auc": safe_auc(labels, text_np),
            "vit_patchcore_max_auc": safe_auc(labels, max_np),
            "harmonic_text_vit_patchcore_max_auc": safe_auc(labels, harmonic(text_np, max_np)),
        }
        payload = {
            "names": np.asarray(names),
            "labels": labels_np,
            "text_scores": text_np,
            "vit_patchcore_max_scores": max_np,
            "harmonic_text_vit_patchcore_max": harmonic(text_np, max_np),
        }
        if coherent_np is not None:
            row["vit_patchcore_coherent_auc"] = safe_auc(labels, coherent_np)
            row["harmonic_text_vit_patchcore_coherent_auc"] = safe_auc(labels, harmonic(text_np, coherent_np))
            payload["vit_patchcore_coherent_scores"] = coherent_np
            payload["harmonic_text_vit_patchcore_coherent"] = harmonic(text_np, coherent_np)
        for ratio in args.map_top_ratios:
            key = f"{ratio:g}".replace(".", "p")
            values = np.asarray(ratio_scores[ratio], dtype=np.float32)
            row[f"vit_patchcore_top{key}_auc"] = safe_auc(labels, values)
            row[f"harmonic_text_vit_patchcore_top{key}_auc"] = safe_auc(labels, harmonic(text_np, values))
            payload[f"vit_patchcore_top{key}_scores"] = values
            payload[f"harmonic_text_vit_patchcore_top{key}"] = harmonic(text_np, values)
        if getattr(args, "support_manifest_sha256", ""):
            payload["support_manifest_sha256"] = np.asarray(
                args.support_manifest_sha256
            )

        rows.append(row)
        np.savez_compressed(score_dir / f"{signal}-{scene}-{jsr}-scores.npz", **payload)

    return rows


@torch.no_grad()
def support_reference_scores(
    model,
    support_loader,
    state,
    args,
    device,
) -> tuple[np.ndarray, np.ndarray]:
    """Score normal support under held-out shifts for an inductive gate.

    The formal gallery contains identity/blur/\u00b14-pixel views.  The reference
    distribution uses deterministic \u00b18-pixel shifts of the same normal
    support, so it contains no test samples and does not reuse an exact
    gallery image.  This is an exploratory support-only calibration, kept
    separate from the default test-batch evaluator.
    """

    scores, names = [], []
    for data, _mask, _label, name, _img_type in tqdm(
        support_loader,
        desc="Score support-only ViT reference",
        leave=False,
    ):
        view_scores = []
        for mode in ("time_shift_up_large", "time_shift_down_large"):
            shifted = paired_tta_batch(data, mode, args)
            map_t = compute_fused_patch_map(
                model,
                shifted,
                state["gallery"],
                state["gallery_rows"],
                state["gallery_cols"],
                args,
                device,
                state["row_stats"],
                state["row_prototypes"],
                state["paired_tta_galleries"],
            )
            map_np = map_t.detach().cpu().numpy().astype(np.float32)
            view_scores.extend(
                map_np.reshape(map_np.shape[0], -1).max(axis=1).tolist()
            )
        # Keep the two held-out views as separate reference observations.
        scores.extend(view_scores)
        names.extend(str(value) for value in name)
        names.extend(str(value) for value in name)
    return np.asarray(scores, dtype=np.float32), np.asarray(names)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="analysis_outputs/20260704_vit_patchcore_gallery_cls")
    parser.add_argument("--formal-baseline-csv", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--signals", nargs="+", default=SIGNALS, choices=SIGNALS)
    parser.add_argument("--scenes", nargs="+", default=SCENES, choices=SCENES)
    parser.add_argument(
        "--normal-sampling",
        choices=("per_frequency", "1shot", "2shot", "4shot"),
        default="per_frequency",
    )
    parser.add_argument(
        "--support-manifest",
        default="",
        help="Shared target-scene support manifest. Created under output-root when omitted.",
    )
    parser.add_argument("--coreset-ratio", type=float, default=0.5)
    parser.add_argument("--coreset-method", choices=["random", "farthest"], default="farthest")
    parser.add_argument("--rowwise-coreset", action="store_true")
    parser.add_argument("--memory-mode", choices=["global_nn", "row_nn", "row_proto"], default="global_nn")
    parser.add_argument("--row-proto-zscore", action="store_true")
    parser.add_argument("--support-augment", choices=["none", "rf_weak", "time_shift"], default="none")
    parser.add_argument("--support-shift-px", type=int, default=3)
    parser.add_argument("--support-crop-ratio", type=float, default=0.03)
    parser.add_argument(
        "--paired-tta",
        choices=[
            "none",
            "visionad_safe",
            "stft_shift_blur",
            "stft_time_shift_v2",
            "ofdma_time_shift_blur",
            "rf_spectral_structure_v1",
            "rf_spectral_background_v1",
            "rf_spectral_time_background_v1",
            "rf_time_alignment_v1",
            "rf_frequency_response_only_v1",
            "rf_spectral_response_v1",
            "rf_spectral_physics_v1",
            "rf_spectral_physics_v2",
            "stft_physical_time_v1",
            "stft_physical_cfo_v1",
            "stft_physical_nuisance_v1",
            "ofdma_spectral_structure_v1",
            "ofdma_spectral_background_v1",
            "ofdma_spectral_time_background_v1",
            "ofdma_time_alignment_v1",
            "ofdma_frequency_response_only_v1",
            "ofdma_spectral_response_v1",
            "ofdma_spectral_physics_v1",
        ],
        default="stft_shift_blur",
    )
    parser.add_argument("--paired-tta-fusion", choices=["mean", "max"], default="max")
    parser.add_argument(
        "--paired-tta-memory-layout",
        choices=["separate", "merged"],
        default="separate",
        help=(
            "separate: match each augmented query with its own normal memory; "
            "merged: combine all augmented normal features into one memory and query the original once."
        ),
    )
    parser.add_argument("--paired-tta-contrast", type=float, default=1.04)
    parser.add_argument("--paired-tta-brightness", type=float, default=1.0)
    parser.add_argument("--paired-tta-crop-ratio", type=float, default=0.02)
    parser.add_argument("--paired-tta-shift-px", type=int, default=4)
    parser.add_argument("--paired-tta-blur-ksize", type=int, default=3)
    parser.add_argument("--paired-tta-background-noise-strength", type=float, default=3.0)
    parser.add_argument(
        "--paired-tta-frequency-response-strength",
        type=float,
        default=3.0,
        help="RMS intensity of the smooth frequency-response calibration view.",
    )
    parser.add_argument("--gallery-chunk-size", type=int, default=4096)
    parser.add_argument("--freq-window", type=int, default=-1, help="-1 disables frequency-position constraint; 0 uses the same row only.")
    parser.add_argument("--nn-topk", type=int, default=5)
    parser.add_argument("--nn-agg", choices=["mean", "weighted", "adaptive"], default="mean")
    parser.add_argument("--nn-weight-temp", type=float, default=0.05)
    parser.add_argument("--adaptive-sim-margin", type=float, default=0.02)
    parser.add_argument("--freq-zscore", action="store_true")
    parser.add_argument("--position-soft-axis", choices=["frequency", "time"], default="frequency")
    parser.add_argument("--position-soft-weight", type=float, default=0.0)
    parser.add_argument("--coherence-alpha", type=float, default=0.0)
    parser.add_argument("--coherence-top-ratio", type=float, default=0.05)
    parser.add_argument("--map-top-ratios", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument(
        "--support-seed",
        type=int,
        default=None,
        help="Independent target-scene support-selection seed; model/coreset seed remains --seed.",
    )
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
    parser.add_argument(
        "--support-reference-only",
        action="store_true",
        help="Build only the support-only calibration reference and skip test scoring.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0.0 < args.coreset_ratio <= 1.0):
        raise ValueError("--coreset-ratio must be in (0, 1]")
    if args.paired_tta != "none" and args.support_augment != "none":
        raise ValueError("--paired-tta already augments normal support; use it with --support-augment none.")
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

    if args.memory_mode == "row_nn" and args.freq_window < 0:
        args.freq_window = 0
    scene_gallery_patch_counts = {}
    manifest = prepare_target_scene_manifest(args)
    rows = []
    train_samples = []
    gallery_feature_dim = None
    for scene in args.scenes:
        train_loader, scene_train_samples = build_target_scene_support_loader(
            manifest,
            scene,
            args,
        )
        state = build_gallery_state(model, train_loader, args, device)
        if args.support_reference_only:
            reference, names = support_reference_scores(
                model,
                train_loader,
                state,
                args,
                device,
            )
            reference_root = Path(args.output_root) / "support_reference"
            reference_root.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                reference_root / f"{scene}.npz",
                vit_scores=reference,
                names=names,
                support_manifest_sha256=np.asarray(args.support_manifest_sha256),
            )
        else:
            rows.extend(
                evaluate(
                    model,
                    state["gallery"],
                    build_target_scene_eval_loaders(
                        manifest,
                        scene,
                        args,
                    ),
                    args,
                    device,
                    state["gallery_rows"],
                    state["gallery_cols"],
                    state["row_stats"],
                    state["row_prototypes"],
                    paired_tta_galleries=state["paired_tta_galleries"],
                )
            )
        train_samples.extend(scene_train_samples)
        scene_gallery_patch_counts[scene] = gallery_patch_count(state)
        gallery_feature_dim = int(state["gallery"].shape[1])
    if args.support_reference_only:
        out_root = Path(args.output_root)
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "support_reference_protocol.json").write_text(
            json.dumps(
                {
                    "method": "vit_patchcore_gallery_cls",
                    "support_only": True,
                    "reference_views": ["time_shift_up_large", "time_shift_down_large"],
                    "support_manifest": args.support_manifest,
                    "support_manifest_sha256": args.support_manifest_sha256,
                    "num_scenes": len(args.scenes),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote support references under {out_root / 'support_reference'}")
        return

    rows = attach_formal_baseline(rows, args.formal_baseline_csv)

    out_root = Path(args.output_root)
    result_path = out_root / "results_cls_vit_patchcore_gallery.csv"
    write_csv(result_path, rows)

    df = pd.DataFrame(rows)
    summary = {
        "method": "vit_patchcore_gallery_cls",
        "checkpoint": args.checkpoint,
        "support_protocol": "target_scene",
        "support_manifest": args.support_manifest or None,
        "support_manifest_sha256": getattr(
            args,
            "support_manifest_sha256",
            None,
        )
        or None,
        "normal_sampling": args.normal_sampling,
        "patch_layer": "concat",
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
        "position_soft_axis": args.position_soft_axis,
        "position_soft_weight": args.position_soft_weight,
        "coherence_alpha": args.coherence_alpha,
        "coherence_top_ratio": args.coherence_top_ratio,
        "support_augment": args.support_augment,
        "support_shift_px": args.support_shift_px,
        "support_crop_ratio": args.support_crop_ratio,
        "paired_tta": args.paired_tta,
        "paired_tta_fusion": args.paired_tta_fusion,
        "paired_tta_memory_layout": args.paired_tta_memory_layout,
        "paired_tta_modes": _paired_tta_modes(args),
        "paired_tta_contrast": args.paired_tta_contrast,
        "paired_tta_brightness": args.paired_tta_brightness,
        "paired_tta_crop_ratio": args.paired_tta_crop_ratio,
        "paired_tta_shift_px": args.paired_tta_shift_px,
        "paired_tta_blur_ksize": args.paired_tta_blur_ksize,
        "paired_tta_background_noise_strength": args.paired_tta_background_noise_strength,
        "paired_tta_frequency_response_strength": args.paired_tta_frequency_response_strength,
        "gallery_patch_count": int(sum(scene_gallery_patch_counts.values())),
        "scene_gallery_patch_counts": scene_gallery_patch_counts,
        "gallery_feature_dim": gallery_feature_dim,
        "selected_normal_count": len(train_samples),
        "selected_normals": [sample[0] for sample in train_samples],
        "num_cells": len(rows),
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
    readme = [
        "# CLIP-ViT NN Gallery CLS",
        "",
        "Nearest-neighbour gallery scoring over PromptAD/CLIP ViT patch features. This is not the official CNN/ResNet PatchCore baseline.",
        "",
        "- support protocol: `target_scene`",
        f"- normal sampling: `{args.normal_sampling}`",
        f"- selected normal images: `{len(train_samples)}`",
        "- patch layer: `concat`",
        f"- coreset ratio: `{args.coreset_ratio}`",
        f"- support augment: `{args.support_augment}`",
        f"- paired TTA: `{args.paired_tta}`",
        f"- TTA memory layout: `{args.paired_tta_memory_layout}`",
        f"- gallery patches across memories: `{int(sum(scene_gallery_patch_counts.values()))}`",
        "",
        "## Macro AUROC",
        "",
    ]
    for key, value in sorted((k, v) for k, v in summary.items() if k.endswith("_auc_macro")):
        readme.append(f"- `{key}`: {value:.4f}")
    (out_root / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"wrote {result_path}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
