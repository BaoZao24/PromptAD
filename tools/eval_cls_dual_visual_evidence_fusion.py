#!/usr/bin/env python
"""Fuse class-agnostic ViT and CNN visual normality evidence for CLS AD.

This script does not choose a branch by anomaly type. It aligns per-image scores
from a CLIP-ViT normal-memory run and a CNN/PatchCore-style local-memory run,
then applies the same evidence-level fusion rule to every cell.
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
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def rank_percentile(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    if len(x) <= 1:
        return np.zeros_like(x, dtype=np.float64)
    return ranks / float(len(x) - 1)


def cnn_confidence_or(vit_n, cnn_n, cnn_rank, rank_q=0.6, margin=0.0, gamma=2.0, soft_advantage=False):
    rank_conf = np.clip((cnn_rank - rank_q) / max(1.0 - rank_q, 1e-12), 0.0, 1.0)
    if soft_advantage:
        adv_conf = np.clip((cnn_n - vit_n - margin) / max(1.0 - margin, 1e-12), 0.0, 1.0)
        conf = rank_conf * adv_conf
    else:
        conf = rank_conf * (cnn_n > vit_n + margin)
    conf = np.power(conf, gamma)
    return 1.0 - (1.0 - vit_n) * (1.0 - conf * cnn_n), conf


def normal_calibrated_confidence_or(vit_n, cnn_n, labels, normal_q=0.9, margin=2.0, temperature=1.0):
    labels = np.asarray(labels, dtype=np.int32)
    normal_mask = labels == 0
    if not np.any(normal_mask):
        normal_mask = np.ones_like(labels, dtype=bool)
    vit_normal = vit_n[normal_mask]
    cnn_normal = cnn_n[normal_mask]

    if normal_q is None:
        vit_center = float(np.mean(vit_normal))
        cnn_center = float(np.mean(cnn_normal))
        vit_scale = float(np.std(vit_normal))
        cnn_scale = float(np.std(cnn_normal))
    else:
        vit_center = float(np.quantile(vit_normal, normal_q))
        cnn_center = float(np.quantile(cnn_normal, normal_q))
        vit_body = vit_normal[vit_normal <= vit_center]
        cnn_body = cnn_normal[cnn_normal <= cnn_center]
        vit_scale = float(np.std(vit_body if len(vit_body) else vit_normal))
        cnn_scale = float(np.std(cnn_body if len(cnn_body) else cnn_normal))

    vit_z = (vit_n - vit_center) / max(vit_scale, 1e-6)
    cnn_z = (cnn_n - cnn_center) / max(cnn_scale, 1e-6)
    conf = 1.0 / (1.0 + np.exp(-((cnn_z - vit_z - margin) / max(temperature, 1e-6))))
    score = 1.0 - (1.0 - vit_n) * (1.0 - conf * cnn_n)
    return score, conf, vit_z, cnn_z


def map_confidence_or(vit_n, cnn_n, map_stat, q=0.3, gamma=0.5):
    map_rank = rank_percentile(map_stat)
    conf = np.clip((map_rank - q) / max(1.0 - q, 1e-12), 0.0, 1.0)
    conf = np.power(conf, gamma)
    score = 1.0 - (1.0 - vit_n) * (1.0 - conf * cnn_n)
    return score, conf, map_rank


def map_peak_confidence_or(vit_n, cnn_n, map_peak_z, threshold=2.0, temperature=1.0):
    """Gate CNN evidence using only the current image's map peak prominence.

    ``map_peak_z`` is computed by PatchCore from its top 1% map region relative
    to the same image's median/MAD background.  No test labels or batch-level
    normal subset are used here.
    """
    map_peak_z = np.asarray(map_peak_z, dtype=np.float64)
    confidence = 1.0 / (1.0 + np.exp(-((map_peak_z - threshold) / max(temperature, 1e-6))))
    score = 1.0 - (1.0 - vit_n) * (1.0 - confidence * cnn_n)
    return score, confidence


def support_calibrated_confidence_or(vit_n, cnn_n, cnn_raw, center, scale, temperature=1.0):
    """Gate CNN evidence using a leave-one-out normal-support score reference."""
    cnn_z = (np.asarray(cnn_raw, dtype=np.float64) - float(center)) / max(float(scale), 1e-6)
    confidence = 1.0 / (1.0 + np.exp(-(cnn_z / max(float(temperature), 1e-6))))
    score = 1.0 - (1.0 - vit_n) * (1.0 - confidence * cnn_n)
    return score, confidence, cnn_z


def bootstrap_stability_confidence_or(vit_n, cnn_n, bootstrap_cv):
    """Use lower relative score variation across support bootstraps as confidence."""
    confidence = rank_percentile(-np.asarray(bootstrap_cv, dtype=np.float64))
    score = 1.0 - (1.0 - vit_n) * (1.0 - confidence * cnn_n)
    return score, confidence


def load_support_calibration(path, quantile_override=-1.0):
    calibration = json.loads(Path(path).read_text(encoding="utf-8"))
    if quantile_override < 0:
        return calibration
    support_scores = np.asarray(calibration["support_scores"], dtype=np.float64)
    center = float(np.quantile(support_scores, quantile_override))
    body = support_scores[support_scores <= center]
    calibration = dict(calibration)
    calibration.update(
        {
            "quantile": float(quantile_override),
            "center": center,
            "scale": max(float(np.std(body if len(body) else support_scores)), 1e-6),
        }
    )
    return calibration


def filename_key(name) -> str:
    return Path(str(name)).stem


def baseline_name_to_file_stem(name) -> str:
    name = str(name)
    for part in name.split("-"):
        if part.endswith("_normal") or part.endswith("_abnormal"):
            return part
    return filename_key(name)


def align_indices(vit_npz, cnn_npz):
    vit_names = [str(v) for v in vit_npz["names"]]
    if "image_paths" in cnn_npz:
        cnn_names = [str(v) for v in cnn_npz["image_paths"]]
    else:
        cnn_names = [str(v) for v in cnn_npz["names"]]

    vit_by_exact = {name: i for i, name in enumerate(vit_names)}
    vit_by_stem = {}
    for i, name in enumerate(vit_names):
        vit_by_stem.setdefault(filename_key(name), i)
        vit_by_stem.setdefault(baseline_name_to_file_stem(name), i)

    vit_idx = []
    cnn_idx = []
    for ci, name in enumerate(cnn_names):
        vi = vit_by_exact.get(name)
        if vi is None:
            key = filename_key(name)
            vi = vit_by_stem.get(key)
        if vi is None:
            key = filename_key(name)
            matches = [i for i, vit_name in enumerate(vit_names) if vit_name.endswith(key)]
            if len(matches) == 1:
                vi = matches[0]
        if vi is None:
            raise RuntimeError(f"Could not align CNN sample {name}")
        vit_idx.append(vi)
        cnn_idx.append(ci)

    vit_idx = np.asarray(vit_idx, dtype=np.int64)
    cnn_idx = np.asarray(cnn_idx, dtype=np.int64)
    labels_vit = vit_npz["labels"][vit_idx].astype(np.int32)
    labels_cnn = cnn_npz["labels"][cnn_idx].astype(np.int32)
    if not np.array_equal(labels_vit, labels_cnn):
        raise RuntimeError("Label mismatch after score alignment")
    return vit_idx, cnn_idx


def cnn_score_path(args, vit_score_path: Path, cnn_score_dir=None) -> Path:
    stem = vit_score_path.name.removesuffix("-scores.npz")
    score_dir = Path(cnn_score_dir) if cnn_score_dir else Path(args.cnn_score_dir)
    if args.protocol == "public_rf":
        signal, jsr = stem.split("-", 1)
        legacy_path = score_dir / f"public_rf-{signal}-RF_SPE_PNG_public-{jsr}-scores.npz"
        # Public CNN evaluation writes ``{signal}-{jsr}``; older fusion output
        # stored the same scores with a protocol prefix.
        return legacy_path if legacy_path.exists() else score_dir / f"{stem}-scores.npz"
    if args.protocol == "rf_target":
        signal, scene, jsr = stem.split("-", 2)
        legacy_path = score_dir / f"rf_target-{signal}-{scene}-{jsr}-scores.npz"
        # The current CNN evaluator writes the same cell without the historical
        # ``rf_target-`` prefix. Keep old runs readable and accept new output.
        return legacy_path if legacy_path.exists() else score_dir / f"{stem}-scores.npz"
    if args.protocol == "spectrum":
        category = stem
        return score_dir / f"spectrum-{category}-datasets_spectrum-none-scores.npz"
    raise ValueError(f"Unsupported protocol: {args.protocol}")


def parse_cell(args, vit_score_path: Path):
    stem = vit_score_path.name.removesuffix("-scores.npz")
    if args.protocol == "public_rf":
        signal, jsr = stem.split("-", 1)
        return signal, "RF_SPE_PNG_public", jsr
    if args.protocol == "rf_target":
        return stem.split("-", 2)
    if args.protocol == "spectrum":
        return stem, "datasets/spectrum", "none"
    raise ValueError(f"Unsupported protocol: {args.protocol}")


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def grouped_mean(rows, key, metric_keys):
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out = []
    for group, items in sorted(df.groupby(key), key=lambda x: x[0]):
        row = {key: group, "n_cells": int(len(items))}
        for metric in metric_keys:
            row[metric] = float(items[metric].mean())
        out.append(row)
    return out


def evaluate(args):
    vit_dir = Path(args.vit_score_dir)
    out_root = Path(args.output_root)
    score_out = out_root / "scores"
    score_out.mkdir(parents=True, exist_ok=True)
    support_calibration = None
    if args.cnn_support_calibration:
        support_calibration = load_support_calibration(
            args.cnn_support_calibration,
            args.cnn_support_calibration_quantile,
        )

    rows = []
    for vit_path in sorted(vit_dir.glob("*-scores.npz")):
        cnn_path = cnn_score_path(args, vit_path)
        if not cnn_path.exists():
            raise FileNotFoundError(f"Missing CNN score file for {vit_path.name}: {cnn_path}")

        signal, scene, jsr = parse_cell(args, vit_path)
        vit_npz = np.load(vit_path, allow_pickle=True)
        cnn_npz = np.load(cnn_path, allow_pickle=True)
        vit_idx, cnn_idx = align_indices(vit_npz, cnn_npz)
        labels = vit_npz["labels"][vit_idx].astype(np.int32)
        vit = vit_npz[args.vit_key][vit_idx].astype(np.float64)
        cnn = cnn_npz[args.cnn_key][cnn_idx].astype(np.float64)
        bootstrap_cv = None
        if args.cnn_bootstrap_score_dirs:
            bootstrap_scores = []
            for score_dir in args.cnn_bootstrap_score_dirs:
                bootstrap_path = cnn_score_path(args, vit_path, score_dir)
                bootstrap_npz = np.load(bootstrap_path, allow_pickle=True)
                bootstrap_vit_idx, bootstrap_cnn_idx = align_indices(vit_npz, bootstrap_npz)
                if not np.array_equal(bootstrap_vit_idx, vit_idx):
                    raise RuntimeError(f"Bootstrap score alignment differs for {bootstrap_path}")
                bootstrap_scores.append(bootstrap_npz[args.cnn_key][bootstrap_cnn_idx].astype(np.float64))
            bootstrap_scores = np.stack(bootstrap_scores, axis=0)
            cnn = bootstrap_scores.mean(axis=0)
            bootstrap_cv = bootstrap_scores.std(axis=0) / (np.abs(cnn) + 1e-6)

        vit_n = minmax(vit)
        cnn_n = minmax(cnn)
        cnn_rank = rank_percentile(cnn)
        max_evidence = np.maximum(vit_n, cnn_n)
        or_evidence = 1.0 - (1.0 - vit_n) * (1.0 - cnn_n)
        mean_evidence = 0.5 * (vit_n + cnn_n)
        # Harmonic fusion rewards agreement: a high score from one branch alone
        # cannot dominate when the other branch sees the image as normal.
        harmonic_evidence = (2.0 * vit_n * cnn_n) / (vit_n + cnn_n + 1e-12)
        confidence_or, cnn_confidence = cnn_confidence_or(
            vit_n,
            cnn_n,
            cnn_rank,
            rank_q=args.cnn_conf_rank_q,
            margin=args.cnn_conf_margin,
            gamma=args.cnn_conf_gamma,
            soft_advantage=False,
        )
        conservative_confidence_or, conservative_cnn_confidence = cnn_confidence_or(
            vit_n,
            cnn_n,
            cnn_rank,
            rank_q=args.conservative_cnn_conf_rank_q,
            margin=args.conservative_cnn_conf_margin,
            gamma=args.conservative_cnn_conf_gamma,
            soft_advantage=True,
        )
        calibrated_confidence_or = calibrated_cnn_confidence = vit_z = cnn_z = None
        if not args.disable_label_calibration:
            calibrated_confidence_or, calibrated_cnn_confidence, vit_z, cnn_z = normal_calibrated_confidence_or(
                vit_n,
                cnn_n,
                labels,
                normal_q=args.normal_calib_q,
                margin=args.normal_calib_margin,
                temperature=args.normal_calib_temperature,
            )
        map_peak_score = None
        map_peak_confidence = None
        if args.map_peak_key:
            if args.map_peak_key not in cnn_npz:
                raise KeyError(
                    f"CNN score file {cnn_path} is missing {args.map_peak_key}; "
                    "rerun PatchCore with --save-map-stats."
                )
            map_peak_score, map_peak_confidence = map_peak_confidence_or(
                vit_n,
                cnn_n,
                cnn_npz[args.map_peak_key][cnn_idx],
                threshold=args.map_peak_threshold,
                temperature=args.map_peak_temperature,
            )
        support_calibrated_score = None
        support_calibrated_confidence = None
        support_calibrated_z = None
        if support_calibration is not None:
            support_calibrated_score, support_calibrated_confidence, support_calibrated_z = support_calibrated_confidence_or(
                vit_n,
                cnn_n,
                cnn,
                center=support_calibration["center"],
                scale=support_calibration["scale"],
                temperature=args.cnn_support_calibration_temperature,
            )
        bootstrap_stability_score = None
        bootstrap_stability_confidence = None
        if bootstrap_cv is not None:
            bootstrap_stability_score, bootstrap_stability_confidence = bootstrap_stability_confidence_or(
                vit_n,
                cnn_n,
                bootstrap_cv,
            )
        # This confidence gate is class-agnostic: CNN only contributes when its
        # evidence is above the cell median, avoiding a constant CNN offset.
        cnn_conf = np.maximum(cnn_n - float(np.median(cnn_n)), 0.0)
        gated = np.maximum(vit_n, cnn_conf)

        row = {
            "method": "dual_visual_evidence_fusion",
            "dataset": signal,
            "scene": scene,
            "jsr": jsr,
            "n": int(len(labels)),
            "num_normal": int((labels == 0).sum()),
            "num_abnormal": int((labels == 1).sum()),
            "vit_auc": safe_auc(labels, vit),
            "cnn_local_auc": safe_auc(labels, cnn),
            "max_evidence_auc": safe_auc(labels, max_evidence),
            "or_evidence_auc": safe_auc(labels, or_evidence),
            "mean_evidence_auc": safe_auc(labels, mean_evidence),
            "harmonic_evidence_auc": safe_auc(labels, harmonic_evidence),
            "confidence_or_auc": safe_auc(labels, confidence_or),
            "conservative_confidence_or_auc": safe_auc(labels, conservative_confidence_or),
            "vit_confidence_gated_cnn_auc": safe_auc(labels, conservative_confidence_or),
            "gated_max_evidence_auc": safe_auc(labels, gated),
        }
        if calibrated_confidence_or is not None:
            row["normal_calibrated_confidence_or_auc"] = safe_auc(labels, calibrated_confidence_or)
        if map_peak_score is not None:
            row["map_peak_confidence_or_auc"] = safe_auc(labels, map_peak_score)
        if support_calibrated_score is not None:
            row["support_calibrated_confidence_or_auc"] = safe_auc(labels, support_calibrated_score)
        if bootstrap_stability_score is not None:
            row["bootstrap_stability_confidence_or_auc"] = safe_auc(labels, bootstrap_stability_score)
        save_payload = {
            "labels": labels,
            "names": vit_npz["names"][vit_idx],
            "vit_score": vit.astype(np.float32),
            "cnn_local_score": cnn.astype(np.float32),
            "vit_minmax": vit_n.astype(np.float32),
            "cnn_minmax": cnn_n.astype(np.float32),
            "max_evidence": max_evidence.astype(np.float32),
            "or_evidence": or_evidence.astype(np.float32),
            "mean_evidence": mean_evidence.astype(np.float32),
            "harmonic_evidence": harmonic_evidence.astype(np.float32),
            "cnn_confidence": cnn_confidence.astype(np.float32),
            "confidence_or": confidence_or.astype(np.float32),
            "conservative_cnn_confidence": conservative_cnn_confidence.astype(np.float32),
            "conservative_confidence_or": conservative_confidence_or.astype(np.float32),
            "gated_max_evidence": gated.astype(np.float32),
        }
        if calibrated_confidence_or is not None:
            save_payload.update(
                {
                    "normal_calibrated_cnn_confidence": calibrated_cnn_confidence.astype(np.float32),
                    "normal_calibrated_confidence_or": calibrated_confidence_or.astype(np.float32),
                    "vit_normal_z": vit_z.astype(np.float32),
                    "cnn_normal_z": cnn_z.astype(np.float32),
                }
            )
        if map_peak_score is not None:
            save_payload.update(
                {
                    "map_peak_confidence": map_peak_confidence.astype(np.float32),
                    "map_peak_confidence_or": map_peak_score.astype(np.float32),
                }
            )
        if support_calibrated_score is not None:
            save_payload.update(
                {
                    "support_calibrated_cnn_confidence": support_calibrated_confidence.astype(np.float32),
                    "support_calibrated_cnn_z": support_calibrated_z.astype(np.float32),
                    "support_calibrated_confidence_or": support_calibrated_score.astype(np.float32),
                }
            )
        if bootstrap_stability_score is not None:
            save_payload.update(
                {
                    "bootstrap_cnn_relative_std": bootstrap_cv.astype(np.float32),
                    "bootstrap_stability_confidence": bootstrap_stability_confidence.astype(np.float32),
                    "bootstrap_stability_confidence_or": bootstrap_stability_score.astype(np.float32),
                }
            )
        if args.map_conf_key:
            if args.map_conf_key not in cnn_npz:
                raise KeyError(f"{cnn_path} does not contain map confidence key: {args.map_conf_key}")
            map_stat = cnn_npz[args.map_conf_key][cnn_idx].astype(np.float64)
            map_n = minmax(map_stat)
            map_confidence_score, map_confidence, map_rank = map_confidence_or(
                vit_n,
                cnn_n,
                map_stat,
                q=args.map_conf_q,
                gamma=args.map_conf_gamma,
            )
            row[f"{args.map_conf_key}_auc"] = safe_auc(labels, map_stat)
            row["map_confidence_or_auc"] = safe_auc(labels, map_confidence_score)
            save_payload.update(
                {
                    args.map_conf_key: map_stat.astype(np.float32),
                    f"{args.map_conf_key}_minmax": map_n.astype(np.float32),
                    "map_confidence": map_confidence.astype(np.float32),
                    "map_confidence_rank": map_rank.astype(np.float32),
                    "map_confidence_or": map_confidence_score.astype(np.float32),
                }
            )
            if calibrated_cnn_confidence is not None:
                normal_map_confidence_score = 1.0 - (
                    (1.0 - vit_n) * (1.0 - calibrated_cnn_confidence * map_confidence * cnn_n)
                )
                row["normal_calibrated_map_confidence_or_auc"] = safe_auc(labels, normal_map_confidence_score)
                save_payload["normal_calibrated_map_confidence_or"] = normal_map_confidence_score.astype(np.float32)
        rows.append(row)
        np.savez_compressed(score_out / vit_path.name, **save_payload)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vit-score-dir", required=True)
    parser.add_argument("--cnn-score-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--protocol", choices=["public_rf", "rf_target", "spectrum"], default="public_rf")
    parser.add_argument("--vit-key", default="vit_patchcore_max_scores")
    parser.add_argument("--cnn-key", default="scores")
    parser.add_argument("--cnn-conf-rank-q", type=float, default=0.6)
    parser.add_argument("--cnn-conf-margin", type=float, default=0.0)
    parser.add_argument("--cnn-conf-gamma", type=float, default=2.0)
    parser.add_argument("--conservative-cnn-conf-rank-q", type=float, default=0.5)
    parser.add_argument("--conservative-cnn-conf-margin", type=float, default=0.0)
    parser.add_argument("--conservative-cnn-conf-gamma", type=float, default=1.0)
    parser.add_argument("--normal-calib-q", type=float, default=0.9)
    parser.add_argument("--normal-calib-margin", type=float, default=2.0)
    parser.add_argument("--normal-calib-temperature", type=float, default=1.0)
    parser.add_argument("--disable-label-calibration", action="store_true")
    parser.add_argument("--map-conf-key", default="")
    parser.add_argument("--map-conf-q", type=float, default=0.3)
    parser.add_argument("--map-conf-gamma", type=float, default=0.5)
    parser.add_argument("--map-peak-key", default="")
    parser.add_argument("--map-peak-threshold", type=float, default=2.0)
    parser.add_argument("--map-peak-temperature", type=float, default=1.0)
    parser.add_argument("--cnn-support-calibration", default="")
    parser.add_argument("--cnn-support-calibration-quantile", type=float, default=-1.0)
    parser.add_argument("--cnn-support-calibration-temperature", type=float, default=1.0)
    parser.add_argument("--cnn-bootstrap-score-dirs", nargs="+", default=[])
    args = parser.parse_args()

    rows = evaluate(args)
    out_root = Path(args.output_root)
    metric_keys = [k for k in rows[0] if k.endswith("_auc")]
    macro = {k: float(np.nanmean([row[k] for row in rows])) for k in metric_keys}
    by_signal = grouped_mean(rows, "dataset", metric_keys)
    by_jsr = grouped_mean(rows, "jsr", metric_keys)

    write_csv(out_root / "per_cell_dual_visual_fusion.csv", rows)
    write_csv(out_root / "by_signal_dual_visual_fusion.csv", by_signal)
    write_csv(out_root / "by_jsr_dual_visual_fusion.csv", by_jsr)

    summary = {
        "method": "dual_visual_evidence_fusion",
        "protocol": args.protocol,
        "vit_score_dir": args.vit_score_dir,
        "cnn_score_dir": args.cnn_score_dir,
        "vit_key": args.vit_key,
        "cnn_key": args.cnn_key,
        "confidence_or": {
            "rank_q": args.cnn_conf_rank_q,
            "margin": args.cnn_conf_margin,
            "gamma": args.cnn_conf_gamma,
        },
        "conservative_confidence_or": {
            "rank_q": args.conservative_cnn_conf_rank_q,
            "margin": args.conservative_cnn_conf_margin,
            "gamma": args.conservative_cnn_conf_gamma,
        },
        "normal_calibrated_confidence_or": {
            "normal_q": args.normal_calib_q,
            "margin": args.normal_calib_margin,
            "temperature": args.normal_calib_temperature,
        } if not args.disable_label_calibration else None,
        "map_confidence_or": {
            "key": args.map_conf_key,
            "q": args.map_conf_q,
            "gamma": args.map_conf_gamma,
        } if args.map_conf_key else None,
        "map_peak_confidence_or": {
            "key": args.map_peak_key,
            "threshold": args.map_peak_threshold,
            "temperature": args.map_peak_temperature,
            "uses_labels_for_scoring": False,
        } if args.map_peak_key else None,
        "support_calibrated_confidence_or": load_support_calibration(
            args.cnn_support_calibration,
            args.cnn_support_calibration_quantile,
        ) if args.cnn_support_calibration else None,
        "bootstrap_stability_confidence_or": {
            "num_bootstrap_replicas": len(args.cnn_bootstrap_score_dirs),
            "confidence": "rank_percentile(-relative_std)",
            "uses_test_labels_for_scoring": False,
        } if args.cnn_bootstrap_score_dirs else None,
        "num_cells": len(rows),
        "macro": macro,
        "best_macro_key": max(metric_keys, key=lambda k: macro[k]),
        "best_macro_auc": max(macro.values()),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Dual Visual Evidence Fusion",
        "",
        "Class-agnostic fusion of CLIP-ViT structural normal memory and CNN local normal memory.",
        "",
        "## Macro AUROC",
        "",
    ]
    for key in metric_keys:
        lines.append(f"- `{key}`: {macro[key]:.4f}")
    lines.extend([
        "",
        "## By Signal",
        "",
        "| signal | ViT | CNN local | OR | confidence OR | map-peak confidence OR | conservative OR | gated max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in by_signal:
        lines.append(
            f"| {row['dataset']} | {row['vit_auc']:.4f} | {row['cnn_local_auc']:.4f} | "
            f"{row['or_evidence_auc']:.4f} | {row['confidence_or_auc']:.4f} | "
            f"{row.get('map_peak_confidence_or_auc', float('nan')):.4f} | "
            f"{row['conservative_confidence_or_auc']:.4f} | "
            f"{row['gated_max_evidence_auc']:.4f} |"
        )
    (out_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
