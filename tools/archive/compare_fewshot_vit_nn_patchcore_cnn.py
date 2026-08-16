#!/usr/bin/env python
"""Compare few-shot RF CLS scores with clear branch names.

The branches are:
- text+ViT: PromptAD text score harmonically fused with its ViT anomaly map.
- Official PatchCore: CNN/ResNet PatchCore from references/patchcore-inspection.
- CLIP-ViT NN gallery: nearest-neighbour scoring over PromptAD/CLIP ViT patches.
- CNN gallery: lightweight local CNN nearest-neighbour score, when provided.

Older score files may still use ``vit_patchcore`` keys. This script reports them
as CLIP-ViT NN gallery to avoid confusing them with official PatchCore.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
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


def harmonic(*scores, eps=1e-12):
    clipped = [np.clip(minmax(score), eps, None) for score in scores]
    denom = np.zeros_like(clipped[0], dtype=np.float64)
    for score in clipped:
        denom += 1.0 / score
    return len(clipped) / denom


def additive(*scores):
    out = np.zeros_like(np.asarray(scores[0], dtype=np.float64))
    for score in scores:
        out += minmax(score)
    return out


def filename_key(path_or_name):
    return Path(str(path_or_name)).stem


def baseline_name_to_file_stem(name):
    name = str(name)
    for part in name.split("-"):
        if part.endswith("_normal") or part.endswith("_abnormal"):
            return part
    return filename_key(name)


def parse_cell_from_baseline_name(path: Path, protocol: str):
    stem = path.name.removesuffix("-scores.npz")
    if protocol == "spectrum":
        return stem, "datasets/spectrum", "none"
    if protocol == "public_rf":
        signal, jsr = stem.split("-", 1)
        scene = "RF_SPE_PNG_public"
        return signal, scene, jsr
    signal, scene, jsr = stem.split("-", 2)
    return signal, scene, jsr


def patchcore_path_for(score_dir: Path, protocol: str, signal: str, scene: str, jsr: str) -> Path:
    if protocol == "spectrum":
        return score_dir / f"spectrum-{signal}-datasets_spectrum-none-scores.npz"
    if protocol == "public_rf":
        return score_dir / f"public_rf-{signal}-{scene}-{jsr}-scores.npz"
    return score_dir / f"rf_target-{signal}-{scene}-{jsr}-scores.npz"


def align_to_baseline(baseline_npz, other_npz):
    baseline_names = [str(v) for v in baseline_npz["names"]]
    if "image_paths" in other_npz:
        other_names = [str(v) for v in other_npz["image_paths"]]
    else:
        other_names = [str(v) for v in other_npz["names"]]

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
    return b_idx, o_idx


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(rows, key, metric_keys):
    out = []
    for group, items in sorted(pd.DataFrame(rows).groupby(key), key=lambda x: x[0]):
        row = {key: group, "n_cells": len(items)}
        for metric in metric_keys:
            row[metric] = float(items[metric].mean())
        out.append(row)
    return out


def evaluate(args):
    baseline_dir = Path(args.baseline_score_dir)
    vit_nn_dir = Path(args.vit_nn_score_dir)
    patchcore_dir = Path(args.patchcore_score_dir)
    cnn_dir = Path(args.cnn_score_dir) if args.cnn_score_dir else None

    rows = []
    score_root = Path(args.output_root) / "scores"
    score_root.mkdir(parents=True, exist_ok=True)

    for baseline_path in sorted(baseline_dir.glob("*-scores.npz")):
        signal, scene, jsr = parse_cell_from_baseline_name(baseline_path, args.protocol)
        vit_nn_path = vit_nn_dir / baseline_path.name
        patchcore_path = patchcore_path_for(patchcore_dir, args.protocol, signal, scene, jsr)
        cnn_path = cnn_dir / baseline_path.name if cnn_dir else None
        if not vit_nn_path.exists():
            raise FileNotFoundError(f"Missing CLIP-ViT NN score file: {vit_nn_path}")
        if not patchcore_path.exists():
            raise FileNotFoundError(f"Missing official PatchCore score file: {patchcore_path}")
        if cnn_path is not None and not cnn_path.exists():
            raise FileNotFoundError(f"Missing CNN score file: {cnn_path}")

        baseline_npz = np.load(baseline_path, allow_pickle=True)
        vit_nn_npz = np.load(vit_nn_path, allow_pickle=True)
        patchcore_npz = np.load(patchcore_path, allow_pickle=True)
        cnn_npz = np.load(cnn_path, allow_pickle=True) if cnn_path else None

        b_idx_vit, vit_idx = align_to_baseline(baseline_npz, vit_nn_npz)
        b_idx_pc, pc_idx = align_to_baseline(baseline_npz, patchcore_npz)
        index_sets = [set(b_idx_vit.tolist()), set(b_idx_pc.tolist())]
        if cnn_npz is not None:
            b_idx_cnn, cnn_idx = align_to_baseline(baseline_npz, cnn_npz)
            cnn_by_b = {int(bi): int(ci) for bi, ci in zip(b_idx_cnn.tolist(), cnn_idx.tolist())}
            index_sets.append(set(b_idx_cnn.tolist()))
        else:
            cnn_by_b = {}
        vit_by_b = {int(bi): int(vi) for bi, vi in zip(b_idx_vit.tolist(), vit_idx.tolist())}
        pc_by_b = {int(bi): int(pi) for bi, pi in zip(b_idx_pc.tolist(), pc_idx.tolist())}
        common = sorted(set.intersection(*index_sets))
        if not common:
            raise RuntimeError(f"No common samples for {baseline_path.name}")

        b_idx = np.asarray(common, dtype=np.int64)
        vit_idx = np.asarray([vit_by_b[int(i)] for i in b_idx], dtype=np.int64)
        pc_idx = np.asarray([pc_by_b[int(i)] for i in b_idx], dtype=np.int64)
        labels = baseline_npz["labels"][b_idx].astype(np.int32)
        if not np.array_equal(labels, vit_nn_npz["labels"][vit_idx].astype(np.int32)):
            raise RuntimeError(f"Label mismatch for CLIP-ViT NN {baseline_path.name}")
        if not np.array_equal(labels, patchcore_npz["labels"][pc_idx].astype(np.int32)):
            raise RuntimeError(f"Label mismatch for official PatchCore {baseline_path.name}")

        text = baseline_npz[args.text_key][b_idx].astype(np.float64)
        text_vit = baseline_npz[args.text_vit_key][b_idx].astype(np.float64)
        vit_nn = vit_nn_npz[args.vit_nn_key][vit_idx].astype(np.float64)
        patchcore = patchcore_npz[args.patchcore_key][pc_idx].astype(np.float64)
        payload = {
            "labels": labels,
            "names": baseline_npz["names"][b_idx],
            "text": text.astype(np.float32),
            "text_vit": text_vit.astype(np.float32),
            "clip_vit_nn": vit_nn.astype(np.float32),
            "official_patchcore": patchcore.astype(np.float32),
        }
        row = {
            "method": "fewshot_text_vit_official_patchcore_clip_vit_nn",
            "dataset": signal,
            "scene": scene,
            "jsr": jsr,
            "n": int(len(labels)),
            "num_normal": int((labels == 0).sum()),
            "num_abnormal": int((labels == 1).sum()),
            "text_only_auc": safe_auc(labels, text),
            "text_vit_auc": safe_auc(labels, text_vit),
            "clip_vit_nn_auc": safe_auc(labels, vit_nn),
            "text_clip_vit_nn_auc": safe_auc(labels, harmonic(text, vit_nn)),
            "official_patchcore_auc": safe_auc(labels, patchcore),
            "text_vit_plus_clip_vit_nn_auc": safe_auc(labels, additive(text_vit, vit_nn)),
            "text_vit_plus_patchcore_auc": safe_auc(labels, additive(text_vit, patchcore)),
            "clip_vit_nn_plus_patchcore_auc": safe_auc(labels, additive(vit_nn, patchcore)),
            "text_vit_plus_clip_vit_nn_plus_patchcore_auc": safe_auc(labels, additive(text_vit, vit_nn, patchcore)),
            "harmonic_text_vit_clip_vit_nn_patchcore_auc": safe_auc(labels, harmonic(text_vit, vit_nn, patchcore)),
        }
        payload["text_clip_vit_nn"] = harmonic(text, vit_nn).astype(np.float32)
        payload["text_vit_plus_clip_vit_nn"] = additive(text_vit, vit_nn).astype(np.float32)
        payload["text_vit_plus_patchcore"] = additive(text_vit, patchcore).astype(np.float32)
        payload["clip_vit_nn_plus_patchcore"] = additive(vit_nn, patchcore).astype(np.float32)
        payload["text_vit_plus_clip_vit_nn_plus_patchcore"] = additive(text_vit, vit_nn, patchcore).astype(np.float32)
        payload["harmonic_text_vit_clip_vit_nn_patchcore"] = harmonic(text_vit, vit_nn, patchcore).astype(np.float32)

        if cnn_npz is not None:
            cnn_idx = np.asarray([cnn_by_b[int(i)] for i in b_idx], dtype=np.int64)
            if not np.array_equal(labels, cnn_npz["labels"][cnn_idx].astype(np.int32)):
                raise RuntimeError(f"Label mismatch for CNN {baseline_path.name}")
            cnn = cnn_npz[args.cnn_key][cnn_idx].astype(np.float64)
            payload["cnn_gallery"] = cnn.astype(np.float32)
            row.update({
                "cnn_gallery_auc": safe_auc(labels, cnn),
                "cnn_plus_clip_vit_nn_auc": safe_auc(labels, additive(cnn, vit_nn)),
                "text_vit_plus_cnn_auc": safe_auc(labels, additive(text_vit, cnn)),
                "text_vit_plus_cnn_plus_clip_vit_nn_auc": safe_auc(labels, additive(text_vit, cnn, vit_nn)),
                "cnn_plus_clip_vit_nn_plus_patchcore_auc": safe_auc(labels, additive(cnn, vit_nn, patchcore)),
                "text_vit_plus_cnn_plus_clip_vit_nn_plus_patchcore_auc": safe_auc(labels, additive(text_vit, cnn, vit_nn, patchcore)),
            })
            payload["cnn_plus_clip_vit_nn"] = additive(cnn, vit_nn).astype(np.float32)
            payload["text_vit_plus_cnn"] = additive(text_vit, cnn).astype(np.float32)
            payload["text_vit_plus_cnn_plus_clip_vit_nn"] = additive(text_vit, cnn, vit_nn).astype(np.float32)
            payload["cnn_plus_clip_vit_nn_plus_patchcore"] = additive(cnn, vit_nn, patchcore).astype(np.float32)
            payload["text_vit_plus_cnn_plus_clip_vit_nn_plus_patchcore"] = additive(text_vit, cnn, vit_nn, patchcore).astype(np.float32)

        rows.append(row)
        np.savez_compressed(score_root / baseline_path.name, **payload)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-score-dir", required=True)
    parser.add_argument("--vit-nn-score-dir", required=True)
    parser.add_argument("--patchcore-score-dir", required=True)
    parser.add_argument("--cnn-score-dir", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--protocol", choices=["rf_target", "public_rf", "spectrum"], default="rf_target")
    parser.add_argument("--text-key", default="text_scores")
    parser.add_argument("--text-vit-key", default="harmonic_text_vit_max")
    parser.add_argument("--vit-nn-key", default="vit_patchcore_max_scores")
    parser.add_argument("--patchcore-key", default="scores")
    parser.add_argument("--cnn-key", default="cnn_layer2_guided_by_vit_top0p01")
    args = parser.parse_args()

    rows = evaluate(args)
    out_root = Path(args.output_root)
    metric_keys = [key for key in rows[0] if key.endswith("_auc")]
    macro = {key: float(np.nanmean([row[key] for row in rows])) for key in metric_keys}
    best_key = max(metric_keys, key=lambda key: macro[key])
    by_signal = summarize_group(rows, "dataset", metric_keys)
    by_jsr = summarize_group(rows, "jsr", metric_keys)

    write_csv(out_root / "per_cell_comparison.csv", rows)
    write_csv(out_root / "by_signal_comparison.csv", by_signal)
    write_csv(out_root / "by_jsr_comparison.csv", by_jsr)

    summary = {
        "baseline_score_dir": args.baseline_score_dir,
        "vit_nn_score_dir": args.vit_nn_score_dir,
        "patchcore_score_dir": args.patchcore_score_dir,
        "cnn_score_dir": args.cnn_score_dir,
        "score_keys": {
            "text": args.text_key,
            "text_vit": args.text_vit_key,
            "clip_vit_nn": args.vit_nn_key,
            "official_patchcore": args.patchcore_key,
            "cnn_gallery": args.cnn_key,
        },
        "num_cells": len(rows),
        "macro": macro,
        "best_macro_key": best_key,
        "best_macro_auc": macro[best_key],
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Few-Shot RF CLS: text+ViT vs Official PatchCore vs CLIP-ViT NN Gallery",
        "",
        "This comparison separates official CNN/ResNet PatchCore from the CLIP-ViT patch nearest-neighbour gallery.",
        "",
        "## Macro AUROC",
        "",
    ]
    display_keys = [
        "text_only_auc",
        "text_vit_auc",
        "official_patchcore_auc",
        "clip_vit_nn_auc",
        "text_clip_vit_nn_auc",
        "clip_vit_nn_plus_patchcore_auc",
        "text_vit_plus_clip_vit_nn_auc",
        "text_vit_plus_patchcore_auc",
        "text_vit_plus_clip_vit_nn_plus_patchcore_auc",
        "harmonic_text_vit_clip_vit_nn_patchcore_auc",
        "cnn_gallery_auc",
        "cnn_plus_clip_vit_nn_auc",
        "text_vit_plus_cnn_auc",
        "text_vit_plus_cnn_plus_clip_vit_nn_auc",
        "text_vit_plus_cnn_plus_clip_vit_nn_plus_patchcore_auc",
    ]
    for key in display_keys:
        if key in macro:
            lines.append(f"- `{key}`: {macro[key]:.4f}")
    lines.extend(["", f"Best macro: `{best_key}` = `{macro[best_key]:.4f}`", ""])
    if "cnn_gallery_auc" in macro:
        lines.extend([
            "## By Signal",
            "",
            "| signal | text+ViT | Official PatchCore | CLIP-ViT NN | CNN gallery | text+ViT+CNN+CLIP-ViT NN |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in by_signal:
            lines.append(
                f"| {row['dataset']} | {row['text_vit_auc']:.4f} | {row['official_patchcore_auc']:.4f} | "
                f"{row['clip_vit_nn_auc']:.4f} | {row['cnn_gallery_auc']:.4f} | "
                f"{row['text_vit_plus_cnn_plus_clip_vit_nn_auc']:.4f} |"
            )
    else:
        lines.extend([
            "## By Signal",
            "",
            "| signal | text+ViT | Official PatchCore | CLIP-ViT NN | text+ViT+CLIP-ViT NN+Official PatchCore |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in by_signal:
            lines.append(
                f"| {row['dataset']} | {row['text_vit_auc']:.4f} | {row['official_patchcore_auc']:.4f} | "
                f"{row['clip_vit_nn_auc']:.4f} | {row['text_vit_plus_clip_vit_nn_plus_patchcore_auc']:.4f} |"
            )
    lines.extend([
        "",
        "## Files",
        "",
        "- `per_cell_comparison.csv`: every signal/scene/JSR cell.",
        "- `by_signal_comparison.csv`: signal-level macro.",
        "- `by_jsr_comparison.csv`: JSR-level macro.",
        "- `scores/`: aligned per-sample scores for secondary analysis.",
    ])
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
