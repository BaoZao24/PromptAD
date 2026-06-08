#!/usr/bin/env python3
"""Phase 1: Generate normal_bg_deviation feature visualization sanity check.

Shows for burst/chirp/dsss on Playground_spectrum/m30db:
  original_gray | weak_residual | normal_bg_deviation | fused_rgb

The fused_rgb is the 3-channel tensor as it would be fed to CLIP.
"""
import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from PromptAD.model import NormalBgDeviationChannels


def preprocess_to_gray(img_bgr, img_resize=240, img_cropsize=240):
    """Same preprocessing as model.transform pipeline."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    pil = pil.resize((img_resize, img_resize), Image.BICUBIC)
    from torchvision.transforms import functional as TF
    pil = TF.center_crop(pil, img_cropsize)
    gray = np.array(pil.convert('L'), dtype=np.float32)
    if gray.max() > 1.5:
        gray /= 255.0
    return gray


def load_train_normal_images(data_dir, img_resize=240, img_cropsize=240):
    """Load training normal images for a specific scene.

    Mirrors the dataset loader filtering: only t00000-04000 time range.
    """
    normal_dir = Path(data_dir)
    if not normal_dir.exists():
        raise FileNotFoundError(f'Normal dir not found: {normal_dir}')

    all_gray = []
    png_files = sorted(normal_dir.glob('*.png'))
    # Apply same time filter as the dataset loader (k_shot=1 uses t00000-04000)
    png_files = [p for p in png_files if 't00000-04000' in p.name]
    print(f'Loading {len(png_files)} PNGs from {normal_dir} (filter: t00000-04000)')
    for png_path in tqdm(png_files, desc='Loading normals'):
        img_bgr = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        gray = preprocess_to_gray(img_bgr, img_resize, img_cropsize)
        all_gray.append(gray)

    if not all_gray:
        raise RuntimeError(f'No valid PNGs loaded from {normal_dir}')
    return np.stack(all_gray, axis=0)


def compute_weak_residual(gray, alpha=0.1):
    """Compute weak residual (same as in model.py)."""
    freq_bg = np.median(gray, axis=1, keepdims=True)
    residual = np.abs(gray - freq_bg)
    wr = gray + alpha * residual
    wr = (wr - wr.min()) / (wr.max() - wr.min() + 1e-6)
    return wr


def compute_deviation(gray, median, mad, eps=1e-6):
    """Compute normalized deviation from normal background."""
    abs_dev = np.abs(gray - median)
    deviation = abs_dev / (1.4826 * mad + eps)
    p2, p98 = np.percentile(deviation, 2), np.percentile(deviation, 98)
    deviation = np.clip(deviation, p2, p98)
    deviation = (deviation - p2) / max(p98 - p2, eps)
    return deviation


def main():
    scene = 'Playground_spectrum'
    noise_level = 'm30db'
    img_resize = 240
    img_cropsize = 240

    # Data paths
    normal_dir = f'/mnt/data/wangbei/data/datasets/normal/{scene}/'

    datasets_config = {
        'burst_signal': f'/mnt/data/wangbei/data/datasets/burst/{scene}/',
        'chirp_signal': f'/mnt/data/wangbei/data/datasets/chirp/{scene}/',
        'dsss_signal': f'/mnt/data/wangbei/data/datasets/dsss/{scene}/',
    }

    # Compute normal bg stats
    print(f'\n{"="*60}')
    print('Computing normal background statistics...')
    print(f'Scene: {scene}, Target size: {img_resize}x{img_cropsize}')
    print(f'{"="*60}')
    stack = load_train_normal_images(normal_dir, img_resize, img_cropsize)
    median = np.median(stack, axis=0).astype(np.float32)
    mad = np.median(np.abs(stack - median), axis=0).astype(np.float32)
    print(f'Computed stats from {stack.shape[0]} training normal images')
    print(f'Median(MAD) = {mad.mean():.6f}, Max(MAD) = {mad.max():.6f}')

    # Collect images for visualization
    # For each dataset: 1 normal + 1 abnormal
    rows = []
    row_labels = []

    for ds_name, ds_path in datasets_config.items():
        normal_test_dir = Path(ds_path) / 'normal' / noise_level
        abnormal_test_dir = Path(ds_path) / 'abnormal' / noise_level

        if not normal_test_dir.exists() or not abnormal_test_dir.exists():
            print(f'Skipping {ds_name}: dirs not found at {ds_path}')
            continue

        normal_pngs = sorted(normal_test_dir.glob('*.png'))
        abnormal_pngs = sorted(abnormal_test_dir.glob('*.png'))

        # Pick middle sample for stable representation
        def load_and_prep(path):
            img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                return None
            return preprocess_to_gray(img_bgr, img_resize, img_cropsize)

        normal_gray = load_and_prep(normal_pngs[len(normal_pngs)//2])
        abnormal_gray = load_and_prep(abnormal_pngs[len(abnormal_pngs)//2])

        if normal_gray is None or abnormal_gray is None:
            print(f'Skipping {ds_name}: failed to load images')
            continue

        # For abnormal: original_gray, weak_residual, deviation, fused_rgb
        wr = compute_weak_residual(abnormal_gray)
        dev = compute_deviation(abnormal_gray, median, mad)

        # Fused RGB visualization: stack as (3, H, W) -> (H, W, 3) for display
        gray_vis = abnormal_gray
        wr_vis = wr
        dev_vis = dev
        fused = np.stack([gray_vis, wr_vis, dev_vis], axis=0).transpose(1, 2, 0)
        # Scale fused to [0,1] for display
        fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-6)

        rows.append([gray_vis, wr_vis, dev_vis, fused])
        row_labels.append(ds_name)

        # Also show normal for comparison
        normal_wr = compute_weak_residual(normal_gray)
        normal_dev = compute_deviation(normal_gray, median, mad)
        normal_fused = np.stack([normal_gray, normal_wr, normal_dev], axis=0).transpose(1, 2, 0)
        normal_fused = (normal_fused - normal_fused.min()) / (normal_fused.max() - normal_fused.min() + 1e-6)
        rows.append([normal_gray, normal_wr, normal_dev, normal_fused])
        row_labels.append(f'{ds_name} (normal)')

    # Create figure
    n_rows = len(rows)
    n_cols = 4
    col_titles = ['Original Gray', 'Weak Residual\n(α=0.1)', 'Normal BG Deviation', 'Fused 3-Channel\n(as CLIP input)']

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.0))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            img = rows[r][c]
            if c == 3:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11, fontweight='bold')
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=10, fontweight='bold')

    fig.suptitle(
        f'normal_bg_deviation Feature Visualization\n'
        f'Scene: {scene}, Noise: {noise_level}\n'
        f'Stats from {stack.shape[0]} training normal images',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    # Save
    out_dir = Path(CURRENT_DIR) / 'analysis_outputs' / 'normal_bg_deviation_preview'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'normal_bg_deviation_feature_examples.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nSaved visualization to: {out_path}')
    print('Done.')


if __name__ == '__main__':
    main()
