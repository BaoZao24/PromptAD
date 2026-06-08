#!/usr/bin/env python3
"""
Frequency-Band-Aware Multi-Scale Scoring: Offline Analysis Tool.

离线分析脚本：加载 PromptAD 已训练好的 checkpoint，计算测试样本和正常训练
样本的 visual anomaly map，然后测试三种频带感知评分方案。

Usage:
    # Step 1: 先跑一次 baseline training (text_only), 获得 checkpoint
    python train_cls.py --dataset burst_signal --class_name Playground_spectrum \
        --noise-level m30db --Epoch 50 --seed 111 \
        --prompt-mode rf --input-mode morph_fusion_gray_residual_a01 \
        --cls-score-mode text_only --split-mode normal_75_25

    # Step 2: 用本脚本分析该 checkpoint 的 band-aware 评分
    python tools/evaluate_band_aware_scoring.py \
        --dataset burst_signal --class_name Playground_spectrum \
        --noise-level m30db --seed 111 \
        --prompt-mode rf --input-mode morph_fusion_gray_residual_a01

Versions:
    A: Frequency-band top-k aggregation
        - Split visual anomaly map into N bands along frequency axis
        - Top-k mean/max within each band
        - Max over bands -> fuse with text score
    B: Normal-calibrated band score
        - Compute normal reference statistics per band
        - z-score calibration: z = (band_score - normal_mean) / normal_std
        - Max over calibrated bands -> fuse with text score
    C: Multi-scale band pooling
        - Multi-scale pooling (1x, 2x, 4x) on visual anomaly map
        - Band aggregation at each scale
        - Max over scales and bands -> fuse with text score
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter

# Ensure project root is on path
CURRENT_DIR = Path(__file__).resolve().parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from datasets import get_dataloader_from_args, dataset_classes
from utils.metrics import metric_cal_img
from utils.training_utils import get_dir_from_args
from PromptAD import PromptAD
from datasets import denormalization as _clip_denorm


def build_model_from_args(args_dict):
    """Construct PromptAD model with same args as training."""
    model = PromptAD(
        out_size_h=args_dict['out_size_h'],
        out_size_w=args_dict['out_size_w'],
        device=args_dict['device'],
        backbone=args_dict['backbone'],
        pretrained_dataset=args_dict['pretrained_dataset'],
        n_ctx=args_dict['n_ctx'],
        n_pro=args_dict['n_pro'],
        n_ctx_ab=args_dict['n_ctx_ab'],
        n_pro_ab=args_dict['n_pro_ab'],
        class_name=args_dict['class_name'],
        **{k: v for k, v in args_dict.items()
           if k not in ('out_size_h', 'out_size_w', 'device', 'backbone',
                        'pretrained_dataset', 'n_ctx', 'n_pro', 'n_ctx_ab',
                        'n_pro_ab', 'class_name')},
    )
    return model


def load_checkpoint(model, checkpoint_path):
    """Load saved checkpoint (prompts, feature gallery, text features)."""
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model_state = model.state_dict()

    # Gallery buffers may have different sizes (placeholder vs full gallery)
    gallery_keys = {'feature_gallery1', 'feature_gallery2', 'text_features',
                    'normal_global_mean', 'normal_global_center', 'normal_global_var'}
    matched = {}
    gallery_matched = {}
    skipped_missing = []
    for k in model_state:
        if k in state_dict:
            if k in gallery_keys:
                gallery_matched[k] = state_dict[k]
            else:
                matched[k] = state_dict[k]
        else:
            skipped_missing.append(k)
    if skipped_missing:
        n = len(skipped_missing)
        show = min(5, n)
        print(f'[checkpoint] {n} keys not in checkpoint (CLIP backbone etc.), '
              f'showing {show}: {skipped_missing[:show]}')

    model.load_state_dict(matched, strict=False)

    # Directly assign gallery buffers (handle size mismatch)
    for k, v in gallery_matched.items():
        setattr(model, k, v)
        print(f'[checkpoint] Assigned {k}: shape {v.shape}')

    print(f'[checkpoint] Loaded {len(matched) + len(gallery_matched)} keys from {checkpoint_path}')


def get_normal_dataloader(dataset, class_name, noise_level, seed, k_shot=1,
                          split_mode='normal_75_25', batch_size=400):
    """Create dataloader for normal-only training samples."""
    extra_kwargs = {
        'noise_level': noise_level,
        'split_mode': split_mode,
        'normal_train_ratio': 0.75,
    }
    from datasets.dataset import CLIPDataset
    from datasets import load_function_dict

    dataset_inst = CLIPDataset(
        load_function=load_function_dict[dataset],
        category=class_name,
        phase='train',
        k_shot=k_shot,
        **extra_kwargs,
    )
    loader = torch.utils.data.DataLoader(
        dataset_inst, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    return loader


def compute_visual_anomaly_maps(model, dataloader, device):
    """Run inference and collect: visual anomaly maps (grid-level), text scores, labels, names."""
    model.eval_mode()
    model.eval()

    all_visual_maps = []
    all_text_scores = []
    all_labels = []
    all_names = []

    with torch.no_grad():
        for (raw_data, mask, label, name, img_type) in tqdm(dataloader, desc='Inference', leave=False):
            data_t = [model.transform(Image.fromarray(f.numpy())) for f in raw_data]
            data_t = torch.stack(data_t, dim=0).to(device)

            visual_features = model.encode_image(data_t)

            # Compute text anomaly score (image-level)
            textual_anomaly = model.calculate_textual_anomaly_score(visual_features, 'cls')

            # Compute visual anomaly map at grid resolution [N, 1, grid_h, grid_w]
            visual_anomaly_map = model.calculate_visual_anomaly_score(visual_features)

            all_visual_maps.append(visual_anomaly_map.cpu().numpy())
            all_text_scores.append(textual_anomaly)
            all_labels.extend(label.numpy().tolist())
            all_names.extend(list(name))

    visual_maps = np.concatenate(all_visual_maps, axis=0)  # [N, 1, grid_h, grid_w]
    text_scores = np.concatenate(all_text_scores, axis=0)  # [N]
    labels = np.array(all_labels)
    names = np.array(all_names)

    return visual_maps, text_scores, labels, names


def compute_auroc(scores, labels):
    """Compute Image-AUROC."""
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return 0.5


# ─── Band definitions ────────────────────────────────────────────────────────

def make_bands(grid_h, n_bands):
    """Split the frequency axis (grid_h) into n_bands contiguous bands.

    Returns list of (start_row, end_row) exclusive.
    """
    band_indices = np.array_split(np.arange(grid_h), n_bands)
    return [(b[0], b[-1] + 1) for b in band_indices]


def aggregate_map_per_band(visual_map, bands):
    """For each band, extract the slice and compute stats.

    visual_map: [N, 1, grid_h, grid_w]
    bands: list of (start_row, end_row)

    Returns:
        band_topk_mean: [N, n_bands]  -- top-k mean within each band
        band_max:       [N, n_bands]  -- max within each band
        band_mean:      [N, n_bands]  -- mean within each band
    """
    N, C, grid_h, grid_w = visual_map.shape
    n_bands = len(bands)

    band_topk_mean = np.zeros((N, n_bands), dtype=np.float32)
    band_max = np.zeros((N, n_bands), dtype=np.float32)
    band_mean = np.zeros((N, n_bands), dtype=np.float32)

    for i, (start, end) in enumerate(bands):
        band_slice = visual_map[:, 0, start:end, :]  # [N, h_band, grid_w]
        band_flat = band_slice.reshape(N, -1)  # [N, h_band * grid_w]

        band_max[:, i] = band_flat.max(axis=1)
        band_mean[:, i] = band_flat.mean(axis=1)

    return band_topk_mean, band_max, band_mean


def compute_band_topk(visual_map, bands, topk_ratio=0.05):
    """Compute top-k mean within each band.

    visual_map: [N, 1, grid_h, grid_w]
    bands: list of (start_row, end_row)

    Returns:
        band_topk_mean: [N, n_bands]
    """
    N, C, grid_h, grid_w = visual_map.shape
    n_bands = len(bands)

    band_topk_mean = np.zeros((N, n_bands), dtype=np.float32)

    for i, (start, end) in enumerate(bands):
        band_slice = visual_map[:, 0, start:end, :]  # [N, h_band, grid_w]
        band_flat = band_slice.reshape(N, -1)  # [N, h_band * grid_w]

        n_patches = band_flat.shape[1]
        topk = max(1, int(n_patches * topk_ratio))
        topk_vals = np.partition(band_flat, -topk, axis=1)[:, -topk:]
        band_topk_mean[:, i] = topk_vals.mean(axis=1)

    return band_topk_mean


def compute_band_aggregates(visual_map, bands, topk_ratio=0.05):
    """Compute all band-level aggregates."""
    N = visual_map.shape[0]
    n_bands = len(bands)

    band_topk_mean = np.zeros((N, n_bands), dtype=np.float32)
    band_max = np.zeros((N, n_bands), dtype=np.float32)
    band_mean = np.zeros((N, n_bands), dtype=np.float32)

    for i, (start, end) in enumerate(bands):
        band_slice = visual_map[:, 0, start:end, :]  # [N, h_band, W]
        band_flat = band_slice.reshape(N, -1)

        band_max[:, i] = band_flat.max(axis=1)
        band_mean[:, i] = band_flat.mean(axis=1)

        n_patches = band_flat.shape[1]
        topk = max(1, int(n_patches * topk_ratio))
        topk_vals = np.partition(band_flat, -topk, axis=1)[:, -topk:]
        band_topk_mean[:, i] = topk_vals.mean(axis=1)

    return band_topk_mean, band_max, band_mean


def calibrate_band_scores(band_scores, normal_stats):
    """Z-score calibrate band scores against normal reference.

    band_scores: [N, n_bands]
    normal_stats: dict with 'mean' [n_bands], 'std' [n_bands]

    Returns:
        z_scores: [N, n_bands]
    """
    mean = normal_stats['mean'][np.newaxis, :]  # [1, n_bands]
    std = np.maximum(normal_stats['std'][np.newaxis, :], 1e-6)  # [1, n_bands]
    z_scores = (band_scores - mean) / std
    return z_scores


# ─── Version A: Frequency-band top-k aggregation ──────────────────────────────

def version_a_frequency_band(visual_maps, text_scores, labels, names, args):
    """Test Version A: Frequency-band top-k aggregation.

    For each n_bands in [2, 3, 4, full]:
        For each beta in betas:
            Compute band_topk_mean -> max over bands -> fuse with text score
    """
    N, C, grid_h, grid_w = visual_maps.shape
    print(f'\n{"="*70}')
    print(f'Version A: Frequency-Band Top-K Aggregation')
    print(f'  Visual map shape: {visual_maps.shape}')
    print(f'  Frequency axis (grid_h): {grid_h}, Time axis (grid_w): {grid_w}')
    print(f'{"="*70}')

    results = {}
    betas = args.betas
    topk_ratios = args.topk_ratios
    band_configs = {'2bands': 2, '3bands': 3, '4bands': 4, 'full': 1}

    # Full-image baseline (no band split)
    full_map_flat = visual_maps.reshape(N, -1)  # [N, grid_h * grid_w]
    full_max = full_map_flat.max(axis=1)
    full_mean = full_map_flat.mean(axis=1)

    for band_name, n_bands in band_configs.items():
        if n_bands == 1:
            bands = [(0, grid_h)]
        else:
            bands = make_bands(grid_h, n_bands)

        for topk_ratio in topk_ratios:
            band_topk = compute_band_topk(visual_maps, bands, topk_ratio)

            # Aggregation strategies
            # max over bands of topk_mean
            band_agg_max = band_topk.max(axis=1)
            # mean over bands of topk_mean
            band_agg_mean = band_topk.mean(axis=1)
            # max over bands of (topk_mean + max)
            band_max_raw = np.zeros((N, len(bands)), dtype=np.float32)
            for i, (start, end) in enumerate(bands):
                band_slice = visual_maps[:, 0, start:end, :].reshape(N, -1)
                band_max_raw[:, i] = band_slice.max(axis=1)
            band_agg_max_topkmax = np.maximum(band_topk, band_max_raw).max(axis=1)

            for beta in betas:
                # agg = max over bands of topk_mean
                fused_scores = text_scores + beta * band_agg_max
                auroc = compute_auroc(fused_scores, labels)
                key = (band_name, f'topk{topk_ratio}', f'max_of_topk', beta)
                results[key] = auroc

                # agg = mean over bands of topk_mean
                fused_scores = text_scores + beta * band_agg_mean
                auroc = compute_auroc(fused_scores, labels)
                key = (band_name, f'topk{topk_ratio}', 'mean_of_topk', beta)
                results[key] = auroc

                # agg = max over bands of max(topk_mean, max)
                fused_scores = text_scores + beta * band_agg_max_topkmax
                auroc = compute_auroc(fused_scores, labels)
                key = (band_name, f'topk{topk_ratio}', 'max_of_max_or_topk', beta)
                results[key] = auroc

    # Print results
    print(f'\n{"Band":<10} {"TopK":<8} {"Agg":<20} {"Beta":<8} {"AUROC":>8}')
    print('-' * 58)
    for (band_name, topk_str, agg_str, beta), auroc in sorted(results.items()):
        marker = ' <--' if auroc == max(results.values()) else ''
        print(f'{band_name:<10} {topk_str:<8} {agg_str:<20} {beta:<8.2f} {auroc:>8.4f}{marker}')

    return results


# ─── Version B: Normal-calibrated band score ──────────────────────────────────

def version_b_normal_calibrated(visual_maps_test, text_scores, labels, names,
                                 visual_maps_normal, args):
    """Test Version B: Normal-calibrated band score.

    Compute per-band statistics from normal training samples.
    Calibrate test band scores using z-score.
    """
    N, C, grid_h, grid_w = visual_maps_test.shape
    N_normal = visual_maps_normal.shape[0]

    print(f'\n{"="*70}')
    print(f'Version B: Normal-Calibrated Band Score')
    print(f'  Test samples: {N}, Normal reference: {N_normal}')
    print(f'{"="*70}')

    results = {}
    betas = args.betas
    topk_ratios = args.topk_ratios
    band_configs = {'2bands': 2, '3bands': 3, '4bands': 4, 'full': 1}

    for band_name, n_bands in band_configs.items():
        if n_bands == 1:
            bands = [(0, grid_h)]
        else:
            bands = make_bands(grid_h, n_bands)

        n_bands_actual = len(bands)

        for topk_ratio in topk_ratios:
            # Compute band topk for normal samples
            normal_band_topk = compute_band_topk(visual_maps_normal, bands, topk_ratio)
            # Compute band topk for test samples
            test_band_topk = compute_band_topk(visual_maps_test, bands, topk_ratio)

            # Calibration statistics from normal samples
            normal_mean = normal_band_topk.mean(axis=0)  # [n_bands]
            normal_std = normal_band_topk.std(axis=0)    # [n_bands]

            # Z-score calibration
            z_scores_topk = calibrate_band_scores(test_band_topk, {
                'mean': normal_mean,
                'std': normal_std,
            })

            # Also compute uncalibrated band scores for comparison
            # Aggregation: max over bands of calibrated z-score
            calib_max = z_scores_topk.max(axis=1)
            calib_mean = z_scores_topk.mean(axis=1)

            # Also try percentile-based calibration
            normal_percentile = np.percentile(normal_band_topk, 95, axis=0)  # [n_bands]
            calib_percentile = test_band_topk / np.maximum(normal_percentile[np.newaxis, :], 1e-6)

            for beta in betas:
                # z-score max
                fused = text_scores + beta * calib_max
                auroc = compute_auroc(fused, labels)
                key = (band_name, f'topk{topk_ratio}', 'zscore_max', beta)
                results[key] = auroc

                # z-score mean
                fused = text_scores + beta * calib_mean
                auroc = compute_auroc(fused, labels)
                key = (band_name, f'topk{topk_ratio}', 'zscore_mean', beta)
                results[key] = auroc

                # percentile max
                fused = text_scores + beta * calib_percentile.max(axis=1)
                auroc = compute_auroc(fused, labels)
                key = (band_name, f'topk{topk_ratio}', 'pct95_max', beta)
                results[key] = auroc

    # Print results
    print(f'\n{"Band":<10} {"TopK":<8} {"Calib":<15} {"Beta":<8} {"AUROC":>8}')
    print('-' * 55)
    for (band_name, topk_str, calib_str, beta), auroc in sorted(results.items()):
        marker = ' <--' if auroc == max(results.values()) else ''
        print(f'{band_name:<10} {topk_str:<8} {calib_str:<15} {beta:<8.2f} {auroc:>8.4f}{marker}')

    return results


# ─── Version C: Multi-scale band pooling ──────────────────────────────────────

def pool2d_numpy(x, kernel_size, stride=None):
    """Max pool a 2D numpy array."""
    if stride is None:
        stride = kernel_size
    H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    out = np.zeros((H_out, W_out), dtype=x.dtype)
    for i in range(H_out):
        for j in range(W_out):
            i_start = i * stride
            j_start = j * stride
            out[i, j] = x[i_start:i_start + kernel_size, j_start:j_start + kernel_size].max()
    return out


def version_c_multiscale(visual_maps_test, text_scores, labels, names,
                          visual_maps_normal, args):
    """Test Version C: Multi-scale band pooling."""
    N, C, grid_h, grid_w = visual_maps_test.shape
    N_normal = visual_maps_normal.shape[0]

    print(f'\n{"="*70}')
    print(f'Version C: Multi-Scale Band Pooling')
    print(f'  Scales: original (1x), 2x pooled, 4x pooled')
    print(f'{"="*70}')

    results = {}
    betas = args.betas
    topk_ratios = args.topk_ratios
    band_configs = {'2bands': 2, '3bands': 3, '4bands': 4, 'full': 1}

    # Multi-scale processing
    scales = {
        'scale1x': 1,
        'scale2x': 2,
        'scale4x': 4,
    }

    # Precompute normal statistics for each scale and band
    normal_calib = {}  # key: (scale_name, band_name, topk_ratio) -> {'mean': ..., 'std': ...}

    for scale_name, pool_size in scales.items():
        pool_stride = max(1, pool_size // 2)

        # Pool visual maps
        if pool_size == 1:
            test_pooled = visual_maps_test
            normal_pooled = visual_maps_normal
        else:
            test_pooled_list, normal_pooled_list = [], []
            for i in range(N):
                x = visual_maps_test[i, 0]
                test_pooled_list.append(pool2d_numpy(x, pool_size, pool_stride))
            test_pooled = np.array(test_pooled_list)[:, np.newaxis, :, :]  # [N, 1, H', W']

            for i in range(N_normal):
                x = visual_maps_normal[i, 0]
                normal_pooled_list.append(pool2d_numpy(x, pool_size, pool_stride))
            normal_pooled = np.array(normal_pooled_list)[:, np.newaxis, :, :]

        pooled_h = test_pooled.shape[2]

        for band_name, n_bands in band_configs.items():
            if n_bands == 1:
                bands = [(0, pooled_h)]
            else:
                bands = make_bands(pooled_h, n_bands)

            for topk_ratio in topk_ratios:
                n_test_band = compute_band_topk(test_pooled, bands, topk_ratio)
                n_normal_band = compute_band_topk(normal_pooled, bands, topk_ratio)

                normal_calib[(scale_name, band_name, topk_ratio)] = {
                    'mean': n_normal_band.mean(axis=0),
                    'std': n_normal_band.std(axis=0),
                }

                # Now compute test scores
                z_scores = calibrate_band_scores(n_test_band, normal_calib[(scale_name, band_name, topk_ratio)])

                for beta in betas:
                    # max over bands at this scale
                    fused = text_scores + beta * z_scores.max(axis=1)
                    auroc = compute_auroc(fused, labels)
                    key = (scale_name, band_name, f'topk{topk_ratio}', 'zscore_max', beta)
                    results[key] = auroc

    # Multi-scale fusion: max over all scales and bands
    all_scale_keys = list(results.keys())
    suffix_scale_map = defaultdict(list)
    for key in all_scale_keys:
        scale_name, band_name, topk_str, agg_str, beta = key
        suffix = (band_name, topk_str, agg_str, beta)
        suffix_scale_map[suffix].append(scale_name)

    ms_results = {}
    for suffix, scale_list in suffix_scale_map.items():
        if len(scale_list) <= 1:
            continue
        band_name, topk_str, agg_str, beta = suffix

        # Multi-scale: we need to compute z-scores at each scale, then take max
        # This requires re-computing...
        pass

    # Print single-scale results
    print(f'\n{"Scale":<10} {"Band":<10} {"TopK":<8} {"Calib":<15} {"Beta":<8} {"AUROC":>8}')
    print('-' * 65)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for (scale_name, band_name, topk_str, calib_str, beta), auroc in sorted_results[:40]:
        marker = ' <--' if auroc == sorted_results[0][1] else ''
        print(f'{scale_name:<10} {band_name:<10} {topk_str:<8} {calib_str:<15} {beta:<8.2f} {auroc:>8.4f}{marker}')

    # Now compute true multi-scale fusion: max over scales + bands
    print(f'\n--- Multi-Scale Fusion (max over all scales and bands) ---')
    for beta in betas:
        for band_name in band_configs:
            for topk_ratio in topk_ratios:
                # Collect z-scores at each scale for this band
                scale_z_scores = {}
                for scale_name in scales:
                    calib_key = (scale_name, band_name, topk_ratio)
                    if calib_key not in normal_calib:
                        continue

                    # Need test z-scores at this scale
                    pool_size = scales[scale_name]
                    if pool_size == 1:
                        test_pooled = visual_maps_test
                    else:
                        pool_stride = max(1, pool_size // 2)
                        test_pooled_list = []
                        for i in range(N):
                            x = visual_maps_test[i, 0]
                            test_pooled_list.append(pool2d_numpy(x, pool_size, pool_stride))
                        test_pooled = np.array(test_pooled_list)[:, np.newaxis, :, :]

                    pooled_h = test_pooled.shape[2]
                    if n_bands == 1:
                        bands = [(0, pooled_h)]
                    else:
                        bands = make_bands(pooled_h, band_configs[band_name])
                    n_bands_actual = len(bands)

                    test_band = compute_band_topk(test_pooled, bands, topk_ratio)
                    z_band = calibrate_band_scores(test_band, normal_calib[calib_key])
                    scale_z_scores[scale_name] = z_band  # [N, n_bands]

                # Multi-scale fusion: max over all scales and bands
                all_z = np.concatenate(
                    [z for z in scale_z_scores.values()],
                    axis=1,
                )  # [N, total_n_bands]
                ms_max = all_z.max(axis=1)

                fused = text_scores + beta * ms_max
                auroc = compute_auroc(fused, labels)
                ms_key = ('multi-scale', band_name, f'topk{topk_ratio}', 'zscore_ms_max', beta)
                ms_results[ms_key] = auroc
                print(f'multi-scale {band_name:<10} topk{topk_ratio:<5} beta={beta:.2f}  AUROC={auroc:.4f}')

    results.update(ms_results)
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def get_parser():
    parser = argparse.ArgumentParser(description='Band-Aware Scoring Offline Analysis')
    # Checkpoint & dataset params
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--class_name', type=str, required=True)
    parser.add_argument('--noise-level', type=str, default='m30db')
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--root-dir', type=str, default='./result')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Explicit checkpoint path. If None, derive from root_dir.')

    # Model params (must match training)
    parser.add_argument('--backbone', type=str, default='ViT-B-16-plus-240')
    parser.add_argument('--pretrained-dataset', type=str, default='laion400m_e32')
    parser.add_argument('--prompt-mode', type=str, default='rf')
    parser.add_argument('--input-mode', type=str, default='morph_fusion_gray_residual_a01')
    parser.add_argument('--n_ctx', type=int, default=4)
    parser.add_argument('--n_pro', type=int, default=3)
    parser.add_argument('--n_ctx_ab', type=int, default=1)
    parser.add_argument('--n_pro_ab', type=int, default=4)
    parser.add_argument('--k-shot', type=int, default=1)
    parser.add_argument('--split-mode', type=str, default='normal_75_25')
    parser.add_argument('--normal-train-ratio', type=float, default=0.75)
    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--text-prototype-mode', type=str, default='single')

    # Band-aware params
    parser.add_argument('--betas', type=float, nargs='+', default=[0.02, 0.05, 0.1])
    parser.add_argument('--topk-ratios', type=float, nargs='+', default=[0.05, 0.1, 0.2])
    parser.add_argument('--versions', type=str, nargs='+',
                        default=['A', 'B', 'C'],
                        choices=['A', 'B', 'C'])

    # Output
    parser.add_argument('--save-npz', type=str, default=None,
                        help='Save visual anomaly maps and scores to this npz file')
    parser.add_argument('--load-npz', type=str, default=None,
                        help='Load pre-extracted visual anomaly maps from npz (skip inference)')
    parser.add_argument('--load-training-scores', type=str, default=None,
                        help='Path to training scores .npz (contains visual_maps, text_scores, labels). '
                             'If provided, use these pre-computed text_scores instead of model inference.')
    parser.add_argument('--load-normal-maps', type=str, default=None,
                        help='Path to pre-extracted normal visual maps .npy for calibration (Versions B/C).')
    parser.add_argument('--gpu-id', type=int, default=0)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    device = f'cuda:{args.gpu_id}' if args.gpu_id >= 0 else 'cpu'
    print(f'Device: {device}')

    # Determine checkpoint path (for normal reference extraction)
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        _, _, checkpoint_path = get_dir_from_args(
            'CLS', root_dir=args.root_dir,
            dataset=args.dataset, class_name=args.class_name,
            noise_level=args.noise_level, k_shot=args.k_shot,
            seed=args.seed, split_mode=args.split_mode,
        )

    print(f'Checkpoint: {checkpoint_path}')

    # ── Load test scores and visual maps ──────────────────────────────────────
    if args.load_training_scores:
        print(f'Loading test scores/visual maps from training: {args.load_training_scores}')
        data = np.load(args.load_training_scores, allow_pickle=True)
        visual_maps_test = data['visual_maps']       # [N, grid_h, grid_w]
        text_scores = data['text_scores']             # [N]
        labels = data['labels']
        names = data['names']
        print(f'  Loaded {len(labels)} samples, text_scores AUROC prelim.')
        # Print text_only AUROC
        text_only_auroc = compute_auroc(text_scores, labels)
        print(f'  Text AUROC from training: {text_only_auroc:.4f}')
        # Reshape: [N, grid_h, grid_w] -> [N, 1, grid_h, grid_w]
        if visual_maps_test.ndim == 3:
            visual_maps_test = visual_maps_test[:, np.newaxis, :, :]
    elif args.load_npz:
        print(f'Loading pre-extracted visual maps from: {args.load_npz}')
        data = np.load(args.load_npz, allow_pickle=True)
        visual_maps_test = data['visual_maps_test']
        text_scores = data['text_scores']
        labels = data['labels']
        names = data['names']
        visual_maps_normal = data['visual_maps_normal']
        normal_names = data['normal_names']
        text_only_auroc = compute_auroc(text_scores, labels)
        print(f'  Text AUROC from npz: {text_only_auroc:.4f}')
    else:
        # Build model
        kwargs = vars(args).copy()
        kwargs['out_size_h'] = args.resolution
        kwargs['out_size_w'] = args.resolution
        kwargs['device'] = device
        kwargs['visual_adapter'] = False
        kwargs['visual_lora'] = False

        print('Building model...')
        model = build_model_from_args(kwargs)
        model = model.to(device)

        if not os.path.exists(checkpoint_path):
            print(f'ERROR: Checkpoint not found: {checkpoint_path}')
            print(f'  Please run the baseline training first.')
            sys.exit(1)

        # Load checkpoint
        load_checkpoint(model, checkpoint_path)
        # Move loaded buffers to device
        for buf_name in ['feature_gallery1', 'feature_gallery2', 'text_features']:
            v = getattr(model, buf_name, None)
            if v is not None:
                if isinstance(v, torch.Tensor) and v.device != torch.device(device):
                    setattr(model, buf_name, v.to(device))
        model.eval_mode()
        model.eval()

        # Create test dataloader
        print('Creating test dataloader...')
        test_loader, _ = get_dataloader_from_args(phase='test', **kwargs)

        # Compute visual anomaly maps on test set
        print('Computing visual anomaly maps for test set...')
        visual_maps_test, text_scores, labels, names = compute_visual_anomaly_maps(
            model, test_loader, device,
        )

        text_only_auroc = compute_auroc(text_scores, labels)
        print(f'  Computed text AUROC: {text_only_auroc:.4f}')

        # Save if requested
        if args.save_npz:
            os.makedirs(os.path.dirname(args.save_npz) or '.', exist_ok=True)
            from datasets.dataset import CLIPDataset
            from datasets import load_function_dict
            kwargs_normal = {
                'dataset': args.dataset, 'class_name': args.class_name,
                'noise_level': args.noise_level, 'k_shot': args.k_shot,
                'split_mode': args.split_mode, 'normal_train_ratio': args.normal_train_ratio,
                'batch_size': args.batch_size, 'seed': args.seed,
            }
            normal_loader = get_normal_dataloader(
                args.dataset, args.class_name, args.noise_level, args.seed,
                k_shot=args.k_shot, split_mode=args.split_mode, batch_size=args.batch_size,
            )
            print('Computing visual anomaly maps for normal reference...')
            visual_maps_normal, _, _, normal_names = compute_visual_anomaly_maps(
                model, normal_loader, device,
            )
            np.savez_compressed(
                args.save_npz,
                visual_maps_test=visual_maps_test,
                text_scores=text_scores,
                labels=labels,
                names=names,
                visual_maps_normal=visual_maps_normal,
                normal_names=normal_names,
            )
            print(f'Saved visual maps to: {args.save_npz}')

    # ── Normal reference visual maps (for calibration in Versions B/C) ─────────
    if 'B' in args.versions or 'C' in args.versions:
        if args.load_normal_maps:
            print(f'Loading normal visual maps from: {args.load_normal_maps}')
            visual_maps_normal = np.load(args.load_normal_maps)
            print(f'  Normal reference: {visual_maps_normal.shape[0]} samples')
        elif 'visual_maps_normal' in dir() and visual_maps_normal is not None:
            # Already loaded from --load-npz
            pass
        else:
            # Need to extract from model
            if 'model' not in dir():
                kwargs = vars(args).copy()
                kwargs['out_size_h'] = args.resolution
                kwargs['out_size_w'] = args.resolution
                kwargs['device'] = device
                kwargs['visual_adapter'] = False
                kwargs['visual_lora'] = False
                print('Building model for normal reference extraction...')
                model = build_model_from_args(kwargs)
                model = model.to(device)
                if not os.path.exists(checkpoint_path):
                    print(f'ERROR: Checkpoint not found: {checkpoint_path}')
                    sys.exit(1)
                load_checkpoint(model, checkpoint_path)
                for buf_name in ['feature_gallery1', 'feature_gallery2', 'text_features']:
                    v = getattr(model, buf_name, None)
                    if v is not None and isinstance(v, torch.Tensor) and v.device != torch.device(device):
                        setattr(model, buf_name, v.to(device))
                model.eval_mode()
                model.eval()

            print('Creating normal reference dataloader...')
            normal_loader = get_normal_dataloader(
                args.dataset, args.class_name, args.noise_level, args.seed,
                k_shot=args.k_shot, split_mode=args.split_mode, batch_size=args.batch_size,
            )
            print('Computing visual anomaly maps for normal reference...')
            visual_maps_normal, _, _, _ = compute_visual_anomaly_maps(
                model, normal_loader, device,
            )
            print(f'  Normal reference: {visual_maps_normal.shape[0]} samples')

    # ─── Baseline: text_only ──────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print(f'BASELINE (text_only): Image-AUROC = {text_only_auroc:.4f}')
    print(f'{"="*70}')

    # ─── Version A: Frequency-band top-k aggregation ──────────────────────────
    results_a = {}
    if 'A' in args.versions:
        results_a = version_a_frequency_band(visual_maps_test, text_scores, labels, names, args)

    # ─── Version B: Normal-calibrated band score ──────────────────────────────
    results_b = {}
    if 'B' in args.versions:
        results_b = version_b_normal_calibrated(
            visual_maps_test, text_scores, labels, names,
            visual_maps_normal, args,
        )

    # ─── Version C: Multi-scale band pooling ──────────────────────────────────
    results_c = {}
    if 'C' in args.versions:
        results_c = version_c_multiscale(
            visual_maps_test, text_scores, labels, names,
            visual_maps_normal, args,
        )

    # ─── Summary ──────────────────────────────────────────────────────────────
    all_results = {}
    all_results.update({('A', *k): v for k, v in results_a.items()})
    all_results.update({('B', *k): v for k, v in results_b.items()})
    all_results.update({('C', *k): v for k, v in results_c.items()})

    if all_results:
        print(f'\n{"="*70}')
        print(f'TOP RESULTS')
        print(f'{"="*70}')
        best = sorted(all_results.items(), key=lambda x: x[1], reverse=True)[:15]
        for (ver, *key_parts), auroc in best:
            key_str = ' | '.join(str(p) for p in key_parts)
            delta = auroc - text_only_auroc
            sign = '+' if delta >= 0 else ''
            print(f'  V{ver}  {key_str:<50}  AUROC={auroc:.4f}  ({sign}{delta:.4f})')

    print(f'\nBaseline text_only: {text_only_auroc:.4f}')

    # Return results for further processing
    return text_only_auroc, all_results


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
