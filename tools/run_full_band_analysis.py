#!/usr/bin/env python3
"""Full band-aware scoring analysis: compare against harmonic fusion baseline."""

import sys
import numpy as np
sys.path.insert(0, '/mnt/data/wangbei/PromptAD')
from sklearn.metrics import roc_auc_score
from tools.evaluate_band_aware_scoring import (
    compute_band_topk, calibrate_band_scores,
    make_bands, pool2d_numpy, compute_auroc,
)


def harmonic_fusion(text_score, map_score):
    """Same as metric_cal_img: 1/(1/text + 1/max_map)"""
    eps = 1e-8
    return 1.0 / (1.0 / np.maximum(text_score, eps) + 1.0 / np.maximum(map_score, eps))


def evaluate_all(dataset_name, npz_path, normal_maps_path):
    """Run full evaluation for one dataset."""
    d = np.load(npz_path, allow_pickle=True)
    text_scores = d['text_scores']
    visual_maps = d['visual_maps']
    labels = d['labels']

    N, grid_h, grid_w = visual_maps.shape
    visual_maps_4d = visual_maps[:, np.newaxis, :, :]

    normal_maps_4d = np.load(normal_maps_path)  # already [N, 1, grid_h, grid_w]

    print(f'\n{"="*70}')
    print(f'  {dataset_name}: N_test={N}, N_normal={normal_maps_4d.shape[0]}, grid={grid_h}x{grid_w}')
    print(f'{"="*70}')

    # ── BASELINES ──────────────────────────────────────────────────────
    text_auroc = compute_auroc(text_scores, labels)

    full_map_flat = visual_maps.reshape(N, -1)
    max_map = full_map_flat.max(axis=1)
    harmonic_scores = harmonic_fusion(text_scores, max_map)
    harmonic_auroc = compute_auroc(harmonic_scores, labels)

    print(f'  text_only (raw):           {text_auroc:.4f}')
    print(f'  text_only (harmonic):      {harmonic_auroc:.4f}  <-- PRODUCTION BASELINE')

    # ── BAND-AWARE SCORING ─────────────────────────────────────────────
    betas = [0.02, 0.05, 0.1]
    topk_ratios = [0.05, 0.1, 0.2]
    band_configs = {'2bands': 2, '3bands': 3, '4bands': 4, 'full': 1}

    best_overall = {'auroc': 0, 'delta': 0, 'info': '', 'scores': None}
    results = []

    # ── Version A: Frequency-band top-k ────────────────────────────────
    for band_name, n_bands in band_configs.items():
        bands = [(0, grid_h)] if n_bands == 1 else make_bands(grid_h, n_bands)
        for topk_ratio in topk_ratios:
            band_topk = compute_band_topk(visual_maps_4d, bands, topk_ratio)
            band_agg_max = band_topk.max(axis=1)
            for beta in betas:
                fused = text_scores + beta * band_agg_max
                auroc = compute_auroc(fused, labels)
                delta = auroc - harmonic_auroc
                info = f"VA | {band_name:6s} | topk{topk_ratio:.2f} | max_band | b={beta:.2f}"
                results.append((auroc, delta, info, fused))
                if auroc > best_overall['auroc']:
                    best_overall = {'auroc': auroc, 'delta': delta, 'info': info, 'scores': fused}

    # ── Version B: Normal-calibrated ───────────────────────────────────
    for band_name, n_bands in band_configs.items():
        bands = [(0, grid_h)] if n_bands == 1 else make_bands(grid_h, n_bands)
        for topk_ratio in topk_ratios:
            test_band = compute_band_topk(visual_maps_4d, bands, topk_ratio)
            normal_band = compute_band_topk(normal_maps_4d, bands, topk_ratio)
            normal_mean = normal_band.mean(axis=0)
            normal_std = np.maximum(normal_band.std(axis=0), 1e-6)
            z_scores = (test_band - normal_mean[np.newaxis, :]) / normal_std[np.newaxis, :]
            for beta in betas:
                fused = text_scores + beta * z_scores.max(axis=1)
                auroc = compute_auroc(fused, labels)
                delta = auroc - harmonic_auroc
                info = f"VB | {band_name:6s} | topk{topk_ratio:.2f} | zscore_max | b={beta:.2f}"
                results.append((auroc, delta, info, fused))
                if auroc > best_overall['auroc']:
                    best_overall = {'auroc': auroc, 'delta': delta, 'info': info, 'scores': fused}

    # ── Version C: Multi-scale ─────────────────────────────────────────
    scales = {'scale1x': 1, 'scale2x': 2, 'scale4x': 4}

    normal_calib = {}
    for scale_name, pool_size in scales.items():
        if pool_size == 1:
            n_pooled = normal_maps_4d
        else:
            n_pooled_list = []
            for i in range(normal_maps_4d.shape[0]):
                n_pooled_list.append(pool2d_numpy(normal_maps_4d[i, 0], pool_size, max(1, pool_size//2)))
            n_pooled = np.array(n_pooled_list)[:, np.newaxis, :, :]

        for band_name, n_bands in band_configs.items():
            bands_c = [(0, n_pooled.shape[2])] if n_bands == 1 else make_bands(n_pooled.shape[2], n_bands)
            for topk_ratio in topk_ratios:
                n_band = compute_band_topk(n_pooled, bands_c, topk_ratio)
                normal_calib[(scale_name, band_name, topk_ratio)] = {
                    'mean': n_band.mean(axis=0),
                    'std': np.maximum(n_band.std(axis=0), 1e-6),
                    'n_bands_actual': len(bands_c),
                }

    for beta in betas:
        all_z_per_scale = []
        for scale_name, pool_size in scales.items():
            if pool_size == 1:
                t_pooled = visual_maps_4d
            else:
                t_pooled_list = []
                for i in range(N):
                    t_pooled_list.append(pool2d_numpy(visual_maps[i], pool_size, max(1, pool_size//2)))
                t_pooled = np.array(t_pooled_list)[:, np.newaxis, :, :]

            scale_z_list = []
            for band_name, n_bands in band_configs.items():
                for topk_ratio in topk_ratios:
                    calib_key = (scale_name, band_name, topk_ratio)
                    if calib_key not in normal_calib:
                        continue
                    calib = normal_calib[calib_key]
                    bands_c = [(0, t_pooled.shape[2])] if n_bands == 1 else make_bands(t_pooled.shape[2], n_bands)
                    t_band = compute_band_topk(t_pooled, bands_c, topk_ratio)
                    z_band = (t_band - calib['mean'][np.newaxis, :]) / calib['std'][np.newaxis, :]
                    scale_z_list.append(z_band)
            if scale_z_list:
                all_z_per_scale.append(np.concatenate(scale_z_list, axis=1))

        if all_z_per_scale:
            all_z = np.concatenate(all_z_per_scale, axis=1)
            ms_max = all_z.max(axis=1)
            fused = text_scores + beta * ms_max
            auroc = compute_auroc(fused, labels)
            delta = auroc - harmonic_auroc
            info = f"VC | multi-scale | all_bands | zscore_ms_max | b={beta:.2f}"
            results.append((auroc, delta, info, fused))
            if auroc > best_overall['auroc']:
                best_overall = {'auroc': auroc, 'delta': delta, 'info': info, 'scores': fused}

    results.sort(key=lambda x: x[0], reverse=True)
    print(f'  Top 5 band-aware results (delta vs harmonic fusion):')
    for auroc, delta, info, _ in results[:5]:
        sign = '+' if delta >= 0 else ''
        print(f'    {info:55s} AUROC={auroc:.4f} ({sign}{delta:.4f})')

    print(f'\n  >>> BEST: {best_overall["info"]}')
    print(f'      AUROC={best_overall["auroc"]:.4f}, delta_vs_harmonic={best_overall["delta"]:+.4f}')
    print(f'      harmonic_baseline={harmonic_auroc:.4f}')

    return {
        'dataset': dataset_name,
        'text_raw': text_auroc,
        'harmonic_baseline': harmonic_auroc,
        'best_band_auroc': best_overall['auroc'],
        'best_delta': best_overall['delta'],
        'best_info': best_overall['info'],
        'top5': results[:5],
    }


if __name__ == '__main__':
    all_summaries = []
    normal_map_names = {
        'burst_signal': 'normal_maps_burst_Playground_m30db.npy',
        'chirp_signal': 'normal_maps_chirp_signal_Playground_m30db.npy',
        'dsss_signal': 'normal_maps_dsss_signal_Playground_m30db.npy',
    }
    for ds in ['burst_signal', 'chirp_signal', 'dsss_signal']:
        npz_path = f'result/{ds}/Playground_spectrum/m30db/normal_75_25/k_1/scores/Seed_111-image_scores.npz'
        normal_path = f'experiments/band_aware_scoring/{normal_map_names[ds]}'
        summary = evaluate_all(ds, npz_path, normal_path)
        all_summaries.append(summary)

    print(f'\n{"="*80}')
    print(f'FINAL SUMMARY: Band-Aware vs Harmonic Fusion (Production Baseline)')
    print(f'{"="*80}')
    print(f'{"Dataset":<20} {"text_raw":>10} {"harmonic":>10} {"band_best":>10} {"delta":>10} {"vs_harmonic":>14}')
    print(f'{"-"*80}')
    total_delta = 0
    for s in all_summaries:
        verdict = "WIN" if s['best_delta'] > 0 else "LOSE"
        print(f'{s["dataset"]:<20} {s["text_raw"]:>10.4f} {s["harmonic_baseline"]:>10.4f} '
              f'{s["best_band_auroc"]:>10.4f} {s["best_delta"]:>+10.4f} {verdict:>14}')
        total_delta += s["best_delta"]
    avg_delta = total_delta / len(all_summaries)
    print(f'{"-"*80}')
    print(f'{"Average delta":<20} {avg_delta:>+10.4f}')
    print(f'{"="*80}')

    for s in all_summaries:
        print(f'\n{s["dataset"]}: {s["best_info"]}')
        print(f'  harmonic={s["harmonic_baseline"]:.4f}, band_best={s["best_band_auroc"]:.4f}, delta={s["best_delta"]:+.4f}')
