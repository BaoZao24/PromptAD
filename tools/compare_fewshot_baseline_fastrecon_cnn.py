#!/usr/bin/env python
"""Compare few-shot RF CLS scores: text+ViT, FastRecon, CNN, PatchCore, and fusions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def safe_auc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores) * 100.0)


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def filename_key(path_or_name):
    return Path(str(path_or_name)).stem


def parse_cell_from_baseline_name(path: Path):
    stem = path.name.removesuffix("-scores.npz")
    signal, scene, jsr = stem.split("-", 2)
    return signal, scene, jsr


def baseline_name_to_file_stem(name):
    name = str(name)
    for part in name.split("-"):
        if part.endswith("_normal") or part.endswith("_abnormal"):
            return part
    return filename_key(name)


def align_by_filename(baseline_npz, other_npz):
    baseline_names = [str(v) for v in baseline_npz["names"]]
    other_names = [str(v) for v in other_npz["image_paths"]] if "image_paths" in other_npz else [str(v) for v in other_npz["names"]]

    b_idx = []
    o_idx = []
    for oi, name in enumerate(other_names):
        if name in baseline_names:
            b_idx.append(baseline_names.index(name))
            o_idx.append(oi)
            continue
        key = filename_key(name)
        match = None
        for bi, b_name in enumerate(baseline_names):
            if b_name.endswith(key) or baseline_name_to_file_stem(b_name) == key:
                match = bi
                break
        if match is None:
            raise RuntimeError(f"Could not align sample {name}")
        b_idx.append(match)
        o_idx.append(oi)

    b_idx = np.asarray(b_idx, dtype=np.int64)
    o_idx = np.asarray(o_idx, dtype=np.int64)
    labels_b = baseline_npz["labels"][b_idx].astype(np.int32)
    labels_o = other_npz["labels"][o_idx].astype(np.int32)
    if not np.array_equal(labels_b, labels_o):
        raise RuntimeError("Label mismatch after score alignment")
    return b_idx, o_idx, labels_b


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(rows, group_key, metric_keys):
    grouped = {}
    for row in rows:
        grouped.setdefault(row[group_key], []).append(row)
    out = []
    for group, items in sorted(grouped.items()):
        summary = {group_key: group, "n_cells": len(items)}
        for key in metric_keys:
            vals = [float(row[key]) for row in items if np.isfinite(float(row[key]))]
            summary[key] = float(np.mean(vals)) if vals else float("nan")
        out.append(summary)
    return out


def evaluate(args):
    baseline_dir = Path(args.baseline_score_dir)
    fastrecon_dir = Path(args.fastrecon_score_dir)
    cnn_dir = Path(args.cnn_score_dir)
    patchcore_dir = Path(args.patchcore_score_dir) if args.patchcore_score_dir else None
    use_patchcore = patchcore_dir is not None
    rows = []
    score_root = Path(args.output_root) / "scores"
    score_root.mkdir(parents=True, exist_ok=True)

    for baseline_path in sorted(baseline_dir.glob("*-scores.npz")):
        signal, scene, jsr = parse_cell_from_baseline_name(baseline_path)
        fastrecon_path = fastrecon_dir / f"rf_target-{signal}-{scene}-{jsr}-scores.npz"
        cnn_path = cnn_dir / baseline_path.name
        patchcore_path = patchcore_dir / f"rf_target-{signal}-{scene}-{jsr}-scores.npz" if patchcore_dir else None
        if not fastrecon_path.exists():
            raise FileNotFoundError(f"Missing FastRecon score file: {fastrecon_path}")
        if not cnn_path.exists():
            raise FileNotFoundError(f"Missing CNN score file: {cnn_path}")
        if patchcore_path is not None and not patchcore_path.exists():
            raise FileNotFoundError(f"Missing PatchCore score file: {patchcore_path}")

        baseline_npz = np.load(baseline_path, allow_pickle=True)
        fastrecon_npz = np.load(fastrecon_path, allow_pickle=True)
        cnn_npz = np.load(cnn_path, allow_pickle=True)
        patchcore_npz = np.load(patchcore_path, allow_pickle=True) if patchcore_path else None

        b_idx_fr, fr_idx, _ = align_by_filename(baseline_npz, fastrecon_npz)
        b_idx_cnn, cnn_idx, _ = align_by_filename(baseline_npz, cnn_npz)
        fr_by_baseline_idx = {int(bi): int(fi) for bi, fi in zip(b_idx_fr.tolist(), fr_idx.tolist())}
        cnn_by_baseline_idx = {int(bi): int(ci) for bi, ci in zip(b_idx_cnn.tolist(), cnn_idx.tolist())}
        common_sets = [set(fr_by_baseline_idx), set(cnn_by_baseline_idx)]
        if patchcore_npz is not None:
            b_idx_patchcore, patchcore_idx, _ = align_by_filename(baseline_npz, patchcore_npz)
            patchcore_by_baseline_idx = {
                int(bi): int(pi) for bi, pi in zip(b_idx_patchcore.tolist(), patchcore_idx.tolist())
            }
            common_sets.append(set(patchcore_by_baseline_idx))
        else:
            patchcore_by_baseline_idx = {}
        common_baseline_idx = sorted(set.intersection(*common_sets))
        if not common_baseline_idx:
            raise RuntimeError(f"No common samples across text+ViT/FastRecon/CNN/PatchCore for {baseline_path.name}")
        baseline_idx = np.asarray(common_baseline_idx, dtype=np.int64)
        fr_idx_reordered = np.asarray([fr_by_baseline_idx[int(bi)] for bi in baseline_idx], dtype=np.int64)
        cnn_idx_reordered = np.asarray([cnn_by_baseline_idx[int(bi)] for bi in baseline_idx], dtype=np.int64)
        patchcore_idx_reordered = (
            np.asarray([patchcore_by_baseline_idx[int(bi)] for bi in baseline_idx], dtype=np.int64)
            if patchcore_npz is not None
            else None
        )
        labels = baseline_npz["labels"][baseline_idx].astype(np.int32)
        labels_fr = fastrecon_npz["labels"][fr_idx_reordered].astype(np.int32)
        labels_cnn = cnn_npz["labels"][cnn_idx_reordered].astype(np.int32)
        if not np.array_equal(labels, labels_fr):
            raise RuntimeError(f"Label mismatch between baseline and FastRecon for {baseline_path.name}")
        if not np.array_equal(labels, labels_cnn):
            raise RuntimeError(f"Label mismatch between FastRecon and CNN for {baseline_path.name}")
        if patchcore_npz is not None:
            labels_patchcore = patchcore_npz["labels"][patchcore_idx_reordered].astype(np.int32)
            if not np.array_equal(labels, labels_patchcore):
                raise RuntimeError(f"Label mismatch between text+ViT and PatchCore for {baseline_path.name}")

        text_score = baseline_npz["text_scores"][baseline_idx].astype(np.float64)
        text_vit_score = baseline_npz[args.baseline_key][baseline_idx].astype(np.float64)
        fastrecon_score = fastrecon_npz[args.fastrecon_key][fr_idx_reordered].astype(np.float64)
        cnn_score = cnn_npz[args.cnn_key][cnn_idx_reordered].astype(np.float64)
        patchcore_score = (
            patchcore_npz[args.patchcore_key][patchcore_idx_reordered].astype(np.float64)
            if patchcore_npz is not None
            else None
        )

        text_n = minmax(text_score)
        text_vit_n = minmax(text_vit_score)
        fastrecon_n = minmax(fastrecon_score)
        cnn_n = minmax(cnn_score)

        row = {
            "method": "fewshot_text_vit_fastrecon_cnn_compare",
            "dataset": signal,
            "scene": scene,
            "jsr": jsr,
            "n": int(len(labels)),
            "num_common_samples": int(len(labels)),
            "num_baseline_samples": int(len(baseline_npz["labels"])),
            "num_fastrecon_samples": int(len(fastrecon_npz["labels"])),
            "num_cnn_samples": int(len(cnn_npz["labels"])),
            "num_patchcore_samples": int(len(patchcore_npz["labels"])) if patchcore_npz is not None else "",
            "num_normal": int((labels == 0).sum()),
            "num_abnormal": int((labels == 1).sum()),
            "text_only_auc": safe_auc(labels, text_score),
            "text_vit_auc": safe_auc(labels, text_vit_score),
            "fastrecon_only_auc": safe_auc(labels, fastrecon_score),
            "cnn_only_auc": safe_auc(labels, cnn_score),
        }
        if patchcore_score is not None:
            row["patchcore_only_auc"] = safe_auc(labels, patchcore_score)
        corr_fr, _ = spearmanr(text_vit_score, fastrecon_score)
        corr_cnn, _ = spearmanr(text_vit_score, cnn_score)
        row["text_vit_fastrecon_spearman"] = float(corr_fr)
        row["text_vit_cnn_spearman"] = float(corr_cnn)
        if patchcore_score is not None:
            corr_patchcore, _ = spearmanr(text_vit_score, patchcore_score)
            row["text_vit_patchcore_spearman"] = float(corr_patchcore)

        payload = {
            "labels": labels,
            "names": baseline_npz["names"][baseline_idx],
            "text": text_score.astype(np.float32),
            "text_vit": text_vit_score.astype(np.float32),
            "fastrecon": fastrecon_score.astype(np.float32),
            "cnn": cnn_score.astype(np.float32),
        }
        if patchcore_score is not None:
            payload["patchcore"] = patchcore_score.astype(np.float32)

        for gamma in args.gammas:
            gamma_key = f"{gamma:g}".replace(".", "p")
            text_fr = text_n + float(gamma) * fastrecon_n
            key = f"text_fastrecon_g{gamma_key}"
            row[f"{key}_auc"] = safe_auc(labels, text_fr)
            payload[key] = text_fr.astype(np.float32)

        for beta in args.betas:
            beta_key = f"{beta:g}".replace(".", "p")
            text_cnn = text_n + float(beta) * cnn_n
            key = f"text_cnn_b{beta_key}"
            row[f"{key}_auc"] = safe_auc(labels, text_cnn)
            payload[key] = text_cnn.astype(np.float32)

        for alpha in args.alphas:
            alpha_key = f"{alpha:g}".replace(".", "p")
            text_vit_fr = text_vit_n + float(alpha) * fastrecon_n
            row[f"text_vit_fastrecon_a{alpha_key}_auc"] = safe_auc(labels, text_vit_fr)
            payload[f"text_vit_fastrecon_a{alpha_key}"] = text_vit_fr.astype(np.float32)
            for beta in args.betas:
                beta_key = f"{beta:g}".replace(".", "p")
                all_score = text_vit_n + float(beta) * cnn_n + float(alpha) * fastrecon_n
                key = f"text_vit_cnn_b{beta_key}_fastrecon_a{alpha_key}"
                row[f"{key}_auc"] = safe_auc(labels, all_score)
                payload[key] = all_score.astype(np.float32)

        rows.append(row)
        np.savez_compressed(score_root / baseline_path.name, **payload)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-score-dir", required=True)
    parser.add_argument("--fastrecon-score-dir", required=True)
    parser.add_argument("--cnn-score-dir", required=True)
    parser.add_argument("--patchcore-score-dir", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-key", default="harmonic_text_vit_max")
    parser.add_argument("--fastrecon-key", default="scores_lam0")
    parser.add_argument("--cnn-key", default="cnn_layer2_guided_by_vit_top0p01")
    parser.add_argument("--patchcore-key", default="scores")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    args = parser.parse_args()

    rows = evaluate(args)
    metric_keys = [key for key in rows[0].keys() if key.endswith("_auc")]
    macro = {key: float(np.nanmean([row[key] for row in rows])) for key in metric_keys}
    best_text_cnn_key = max(
        [key for key in metric_keys if key.startswith("text_cnn_")],
        key=lambda key: macro[key],
    )
    best_text_fr_key = max(
        [key for key in metric_keys if key.startswith("text_fastrecon_g")],
        key=lambda key: macro[key],
    )
    best_text_vit_fr_key = max(
        [key for key in metric_keys if key.startswith("text_vit_fastrecon_a")],
        key=lambda key: macro[key],
    )
    best_all_key = max(
        [key for key in metric_keys if key.startswith("text_vit_cnn_")],
        key=lambda key: macro[key],
    )

    out_root = Path(args.output_root)
    write_csv(out_root / "per_cell_comparison.csv", rows)
    by_signal = summarize_group(rows, "dataset", metric_keys)
    by_jsr = summarize_group(rows, "jsr", metric_keys)
    write_csv(out_root / "by_signal_comparison.csv", by_signal)
    write_csv(out_root / "by_jsr_comparison.csv", by_jsr)

    summary = {
        "text_vit_score_dir": args.baseline_score_dir,
        "fastrecon_score_dir": args.fastrecon_score_dir,
        "cnn_score_dir": args.cnn_score_dir,
        "patchcore_score_dir": args.patchcore_score_dir,
        "text_vit_key": args.baseline_key,
        "fastrecon_key": args.fastrecon_key,
        "cnn_key": args.cnn_key,
        "patchcore_key": args.patchcore_key,
        "alphas": args.alphas,
        "betas": args.betas,
        "gammas": args.gammas,
        "num_cells": len(rows),
        "macro": macro,
        "text_vit_macro_auc": macro["text_vit_auc"],
        "fastrecon_only_macro_auc": macro["fastrecon_only_auc"],
        "patchcore_only_macro_auc": macro.get("patchcore_only_auc"),
        "best_text_cnn_key": best_text_cnn_key,
        "best_text_cnn_auc": macro[best_text_cnn_key],
        "best_text_fastrecon_key": best_text_fr_key,
        "best_text_fastrecon_auc": macro[best_text_fr_key],
        "best_text_vit_fastrecon_key": best_text_vit_fr_key,
        "best_text_vit_fastrecon_auc": macro[best_text_vit_fr_key],
        "best_text_vit_cnn_fastrecon_key": best_all_key,
        "best_text_vit_cnn_fastrecon_auc": macro[best_all_key],
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Few-Shot RF CLS Comparison",
        "",
        f"text+ViT score dir: `{args.baseline_score_dir}`",
        f"FastRecon score dir: `{args.fastrecon_score_dir}`",
        f"CNN score dir: `{args.cnn_score_dir}`",
        f"PatchCore score dir: `{args.patchcore_score_dir}`",
        "",
        "## Headline Macro AUROC",
        "",
        f"- text only: `{macro['text_only_auc']:.4f}`",
        f"- text+ViT: `{macro['text_vit_auc']:.4f}`",
        f"- FastRecon only: `{macro['fastrecon_only_auc']:.4f}`",
        f"- CNN only: `{macro['cnn_only_auc']:.4f}`",
        *( [f"- PatchCore only: `{macro['patchcore_only_auc']:.4f}`"] if "patchcore_only_auc" in macro else [] ),
        f"- text+FastRecon: `{macro[best_text_fr_key]:.4f}` (`{best_text_fr_key}`)",
        f"- text+CNN: `{macro[best_text_cnn_key]:.4f}` (`{best_text_cnn_key}`)",
        f"- text+ViT+FastRecon / FastRecon+ViT+text: `{macro[best_text_vit_fr_key]:.4f}` (`{best_text_vit_fr_key}`)",
        f"- text+ViT+CNN+FastRecon: `{macro[best_all_key]:.4f}` (`{best_all_key}`)",
        "",
        "## By Signal",
        "",
        "| signal | text+ViT | FastRecon only | PatchCore only | text+FastRecon | text+CNN | text+ViT+FastRecon / FastRecon+ViT+text | text+ViT+CNN+FastRecon |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_signal:
        patchcore_cell = f"{row['patchcore_only_auc']:.4f}" if "patchcore_only_auc" in row else ""
        lines.append(
            f"| {row['dataset']} | {row['text_vit_auc']:.4f} | "
            f"{row['fastrecon_only_auc']:.4f} | {patchcore_cell} | {row[best_text_fr_key]:.4f} | "
            f"{row[best_text_cnn_key]:.4f} | "
            f"{row[best_text_vit_fr_key]:.4f} | {row[best_all_key]:.4f} |"
        )
    lines.extend([
        "",
        "## Top Metrics",
        "",
    ])
    for key, value in sorted(macro.items(), key=lambda kv: kv[1], reverse=True)[:20]:
        lines.append(f"- `{key}`: {value:.4f}")
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
