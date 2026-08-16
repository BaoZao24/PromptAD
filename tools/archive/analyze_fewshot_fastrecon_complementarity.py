#!/usr/bin/env python
"""Analyze whether FastRecon is complementary to few-shot PromptAD scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


BASELINE_SCORE_KEYS = {
    "text": "text_scores",
    "vit_max": "vit_map_max_scores",
    "promptad_text_vit": "harmonic_text_vit_max",
    "vit_top0p01": "vit_map_top0p01_scores",
    "promptad_text_vit_top0p01": "harmonic_text_vit_top0p01",
    "vit_patchcore_max": "vit_patchcore_max_scores",
    "text_vit_patchcore_max": "text_vit_patchcore_max",
    "promptad_text_vit_patchcore_max": "harmonic_text_vit_patchcore_max",
    "vit_patchcore_top0p01": "vit_patchcore_top0p01_scores",
    "text_vit_patchcore_top0p01": "text_vit_patchcore_top0p01",
    "promptad_text_vit_patchcore_top0p01": "harmonic_text_vit_patchcore_top0p01",
}


def safe_auc(labels, scores) -> float:
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores) * 100.0)


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def harmonic(a, b, eps=1e-12):
    a = np.clip(np.asarray(a, dtype=np.float64), eps, None)
    b = np.clip(np.asarray(b, dtype=np.float64), eps, None)
    return 1.0 / (1.0 / a + 1.0 / b)


def filename_key(path_or_name):
    stem = Path(str(path_or_name)).stem
    return stem


def align_by_filename(baseline_npz, fastrecon_npz):
    baseline_names = [str(v) for v in baseline_npz["names"]]
    baseline_key_to_idx = {}
    for idx, name in enumerate(baseline_names):
        # Baseline names append the original filename stem at the end.
        for part in name.split("-"):
            if part.endswith("_normal") or part.endswith("_abnormal"):
                baseline_key_to_idx[part] = idx
        baseline_key_to_idx[filename_key(name)] = idx

    b_idx = []
    f_idx = []
    for idx, path in enumerate(fastrecon_npz["image_paths"]):
        key = filename_key(path)
        match = None
        for bi, name in enumerate(baseline_names):
            if name.endswith(key):
                match = bi
                break
        if match is None:
            raise RuntimeError(f"Could not align FastRecon sample {path}")
        b_idx.append(match)
        f_idx.append(idx)

    b_idx = np.asarray(b_idx, dtype=np.int64)
    f_idx = np.asarray(f_idx, dtype=np.int64)
    labels_b = baseline_npz["labels"][b_idx].astype(np.int32)
    labels_f = fastrecon_npz["labels"][f_idx].astype(np.int32)
    if not np.array_equal(labels_b, labels_f):
        raise RuntimeError("Label mismatch after alignment")
    return b_idx, f_idx, labels_b


def parse_cell_from_baseline_name(path: Path):
    stem = path.name.removesuffix("-scores.npz")
    parts = stem.split("-")
    if len(parts) == 2:
        signal, jsr = parts
        scene = "RF_SPE_PNG_public"
        return signal, scene, jsr
    signal, scene, jsr = parts[0], parts[1], "-".join(parts[2:])
    return signal, scene, jsr


def find_fastrecon_path(fastrecon_dir: Path, signal: str, scene: str, jsr: str) -> Path:
    candidates = [
        fastrecon_dir / f"rf_target-{signal}-{scene}-{jsr}-scores.npz",
        fastrecon_dir / f"public_rf-{signal}-{scene}-{jsr}-scores.npz",
        fastrecon_dir / f"rf_public-{signal}-{jsr}-scores.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Missing FastRecon score file. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def summarize_group(rows, group_key, metric_keys):
    groups = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)
    out = []
    for group, items in sorted(groups.items()):
        summary = {group_key: group, "n_cells": len(items)}
        for key in metric_keys:
            values = [float(r[key]) for r in items if np.isfinite(float(r[key]))]
            summary[key] = float(np.mean(values)) if values else float("nan")
        out.append(summary)
    return out


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-score-dir", default="analysis_outputs/20260703_fewshot_vit_patch_gallery_rerun/scores")
    parser.add_argument("--fastrecon-score-dir", default="analysis_outputs/20260703_fewshot_fastrecon_cls/scores")
    parser.add_argument("--output-root", default="analysis_outputs/20260703_fewshot_fastrecon_complementarity")
    parser.add_argument("--fastrecon-key", default="scores_lam0")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_score_dir)
    fastrecon_dir = Path(args.fastrecon_score_dir)
    rows = []

    for baseline_path in sorted(baseline_dir.glob("*-scores.npz")):
        signal, scene, jsr = parse_cell_from_baseline_name(baseline_path)
        fastrecon_path = find_fastrecon_path(fastrecon_dir, signal, scene, jsr)

        b = np.load(baseline_path, allow_pickle=True)
        f = np.load(fastrecon_path, allow_pickle=True)
        b_idx, f_idx, labels = align_by_filename(b, f)
        fr = f[args.fastrecon_key][f_idx].astype(np.float64)
        fr_n = minmax(fr)

        row = {
            "signal": signal,
            "scene": scene,
            "jsr": jsr,
            "n": int(len(labels)),
            "num_normal": int((labels == 0).sum()),
            "num_abnormal": int((labels == 1).sum()),
            "fastrecon_auc": safe_auc(labels, fr),
        }
        for name, key in BASELINE_SCORE_KEYS.items():
            if key not in b:
                continue
            base = b[key][b_idx].astype(np.float64)
            base_n = minmax(base)
            row[f"{name}_auc"] = safe_auc(labels, base)
            corr, _ = spearmanr(base, fr)
            row[f"{name}_fastrecon_spearman"] = float(corr)
            for alpha in args.alphas:
                alpha_key = f"{alpha:g}".replace(".", "p")
                row[f"{name}_add_fr_a{alpha_key}_auc"] = safe_auc(labels, base_n + alpha * fr_n)
                row[f"{name}_harmonic_fr_a{alpha_key}_auc"] = safe_auc(labels, harmonic(base_n, fr_n * alpha))
        rows.append(row)

    metric_keys = [k for k in rows[0] if k.endswith("_auc")]
    macro = {key: float(np.nanmean([row[key] for row in rows])) for key in metric_keys}
    best_key = max(metric_keys, key=lambda k: macro[k])
    by_signal = summarize_group(rows, "signal", metric_keys)
    by_jsr = summarize_group(rows, "jsr", metric_keys)

    out_root = Path(args.output_root)
    write_csv(out_root / "per_cell_complementarity.csv", rows)
    write_csv(out_root / "by_signal_complementarity.csv", by_signal)
    write_csv(out_root / "by_jsr_complementarity.csv", by_jsr)
    summary = {
        "baseline_score_dir": str(baseline_dir),
        "fastrecon_score_dir": str(fastrecon_dir),
        "fastrecon_key": args.fastrecon_key,
        "alphas": args.alphas,
        "num_cells": len(rows),
        "macro": macro,
        "best_macro_key": best_key,
        "best_macro_auc": macro[best_key],
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# FastRecon Few-Shot Complementarity",
        "",
        f"Baseline scores: `{baseline_dir}`",
        f"FastRecon scores: `{fastrecon_dir}`",
        f"FastRecon key: `{args.fastrecon_key}`",
        "",
        "## Macro AUROC",
        "",
    ]
    for key, value in sorted(macro.items(), key=lambda kv: kv[1], reverse=True)[:20]:
        lines.append(f"- `{key}`: {value:.4f}")
    lines.extend([
        "",
        f"Best: `{best_key}` = {macro[best_key]:.4f}",
    ])
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
