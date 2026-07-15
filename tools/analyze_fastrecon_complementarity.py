#!/usr/bin/env python
"""Complementarity analysis: can FastRecon scores improve the dual-gallery baseline?

For each of the 48 eval cells, align FastRecon (lambda=0) per-sample scores with the
baseline (dual gallery fusion) per-sample scores by name, then sweep fusion weights
and report macro AUROC. If fusion > baseline-alone (91.8955), FastRecon is complementary.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "analysis_outputs/20260629_cls_dual_gallery_fusion/scores"
FASTRECON_DIR = REPO_ROOT / "analysis_outputs/20260703_fastrecon_cls/scores"

# Baseline columns to test (component + best fusion).
BASELINE_BEST = "resnet18_layer3_dual_raw_clip1_resnet0.5"  # == 91.8955 macro
BASELINE_COLS = {
    "promptad": "promptad_scores",            # bare PromptAD text score (~78.41)
    "clip_gallery": "clip_gallery_scores",     # CLIP normal gallery only (~88.65)
    "resnet18_layer3": "resnet18_layer3_scores",  # ResNet18 gallery only (~90.52)
    "dual_best": BASELINE_BEST,                # dual fusion SOTA (~91.8955)
}
FUSION_ALPHAS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]


def safe_auc(labels, scores):
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores) * 100.0)


def minmax(x):
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def fr_key(path):
    return os.path.basename(str(path))[:-4]  # strip .png


def baseline_key(name):
    # baseline names look like: <signal>-<scene>-<jsr>-<...>-<orig_filename_no_ext>
    # the orig filename (no ext) is what FastRecon stores; take the part after the last '-'
    # that begins the patch filename. Robust: the FastRecon key is always a suffix.
    return str(name).split("-")[-1] if False else str(name)  # handled by suffix match below


def align(baseline_npz, fr_npz):
    """Return indices (b_idx, f_idx) of matched samples, plus labels."""
    b_names = [str(n) for n in baseline_npz["names"]]
    f_keys = [fr_key(p) for p in fr_npz["image_paths"]]
    # baseline name ends with the fastrecon key (orig filename without .png)
    f_to_b = {}
    for fi, k in enumerate(f_keys):
        for bi, bn in enumerate(b_names):
            if bn.endswith(k):
                f_to_b[fi] = bi
                break
    if len(f_to_b) != len(f_keys):
        raise RuntimeError(f"Only matched {len(f_to_b)}/{len(f_keys)} samples")
    fi = np.arange(len(f_keys))
    bi = np.array([f_to_b[i] for i in fi])
    # sanity: labels agree
    b_lab = baseline_npz["labels"][bi]
    f_lab = fr_npz["labels"][fi]
    assert np.array_equal(b_lab, f_lab), "label mismatch after alignment"
    return bi, fi, b_lab


def rank_corr(a, b):
    """Spearman-like rank correlation (1-rank normalized)."""
    from scipy.stats import spearmanr
    r, _ = spearmanr(a, b)
    return float(r)


def main():
    cells = []
    for fp in sorted(FASTRECON_DIR.glob("rf_target-*-scores.npz")):
        # rf_target-<signal>-<scene>-<jsr>-scores.npz
        stem = fp.name[len("rf_target-"):-len("-scores.npz")]
        signal, scene, jsr = stem.split("-", 2)
        bp = BASELINE_DIR / f"{signal}-{scene}-{jsr}-scores.npz"
        if not bp.exists():
            print(f"  [skip] no baseline for {signal}/{scene}/{jsr}")
            continue
        cells.append((signal, scene, jsr, bp, fp))

    print(f"matched {len(cells)} cells\n")

    per_cell = []
    all_corr = []
    for signal, scene, jsr, bp, fp in cells:
        b = np.load(bp, allow_pickle=True)
        fr = np.load(fp, allow_pickle=True)
        bi, fi, labels = align(b, fr)
        fr_lam0 = fr["scores_lam0"][fi].astype(np.float64)

        row = {"signal": signal, "scene": scene, "jsr": jsr, "n": len(labels)}
        # alone
        for label, col in BASELINE_COLS.items():
            row[f"{label}_auc"] = safe_auc(labels, b[col][bi].astype(np.float64))
        row["fastrecon_lam0_auc"] = safe_auc(labels, fr_lam0)
        # correlation between baseline-best and fastrecon
        row["corr_dual_best_vs_fr"] = rank_corr(b[BASELINE_BEST][bi].astype(np.float64), fr_lam0)
        all_corr.append(row["corr_dual_best_vs_fr"])
        # fusion: minmax(baseline_col) + alpha * minmax(fastrecon)
        for label, col in BASELINE_COLS.items():
            base_n = minmax(b[col][bi].astype(np.float64))
            fr_n = minmax(fr_lam0)
            for a in FUSION_ALPHAS:
                fused = base_n + a * fr_n
                row[f"fuse_{label}_a{a:g}_auc"] = safe_auc(labels, fused)
        per_cell.append(row)

    # macro
    keys = list(per_cell[0].keys())
    macro = {k: float(np.nanmean([r[k] for r in per_cell if k in r and np.isfinite(r[k])]))
             for k in keys if k not in ("signal", "scene", "jsr", "n")}

    print("=== Alone (macro AUROC) ===")
    for label in BASELINE_COLS:
        print(f"  baseline {label:18s}: {macro[label+'_auc']:.4f}")
    print(f"  fastrecon_lam0          : {macro['fastrecon_lam0_auc']:.4f}")
    print(f"  mean rank-corr(dual_best, fr): {np.mean(all_corr):.4f}")
    print()
    print("=== Fusion: baseline_col + alpha * fastrecon (macro AUROC) ===")
    best_overall = ("", 0.0)
    for label in BASELINE_COLS:
        best_a, best_v = None, -1
        for a in FUSION_ALPHAS:
            v = macro[f"fuse_{label}_a{a:g}_auc"]
            if v > best_v:
                best_v, best_a = v, a
        alone = macro[label + "_auc"]
        delta = best_v - alone
        print(f"  {label:18s}: alone={alone:.4f}  best_fuse(a={best_a:g})={best_v:.4f}  Δ={delta:+.4f}")
        if best_v > best_overall[1]:
            best_overall = (f"{label}@a={best_a:g}", best_v)
    print()
    print(f"=== Best fusion overall: {best_overall[0]} = {best_overall[1]:.4f} ===")
    print(f"    baseline SOTA (dual_best alone): {macro['dual_best_auc']:.4f}")
    print(f"    Δ vs SOTA: {best_overall[1] - macro['dual_best_auc']:+.4f}")

    # write report
    out = REPO_ROOT / "analysis_outputs/20260703_fastrecon_cls/complementarity_report.txt"
    lines = []
    lines.append("FastRecon complementarity analysis (per-sample fusion with dual-gallery baseline)")
    lines.append(f"cells: {len(per_cell)}  |  baseline SOTA col: {BASELINE_BEST}")
    lines.append("")
    lines.append("== Alone (macro AUROC) ==")
    for label in BASELINE_COLS:
        lines.append(f"  baseline {label:18s}: {macro[label+'_auc']:.4f}")
    lines.append(f"  fastrecon_lam0          : {macro['fastrecon_lam0_auc']:.4f}")
    lines.append(f"  mean rank-corr(dual_best, fr): {np.mean(all_corr):.4f}")
    lines.append("")
    lines.append("== Fusion: minmax(baseline_col) + alpha*minmax(fastrecon_lam0) ==")
    for label in BASELINE_COLS:
        best_a, best_v = None, -1
        for a in FUSION_ALPHAS:
            v = macro[f"fuse_{label}_a{a:g}_auc"]
            if v > best_v:
                best_v, best_a = v, a
        alone = macro[label + "_auc"]
        lines.append(f"  {label:18s}: alone={alone:.4f}  best_fuse(a={best_a:g})={best_v:.4f}  Δ={best_v-alone:+.4f}")
    lines.append("")
    lines.append(f"Best fusion overall: {best_overall[0]} = {best_overall[1]:.4f}")
    lines.append(f"baseline SOTA (dual_best alone): {macro['dual_best_auc']:.4f}")
    lines.append(f"Δ vs SOTA: {best_overall[1] - macro['dual_best_auc']:+.4f}")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
