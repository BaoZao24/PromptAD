"""Screen support-only frequency-stratified false-alarm calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_candidate_methods import build_model, encode_path
from tools.eval_cls_vit_patchcore_gallery import _farthest_indices
import torch


DEFAULT_MANIFEST = ROOT / "analysis_outputs/20260728_rf_five_type_formal_seed111/support_manifest.json"
DEFAULT_CACHE = ROOT / "analysis_outputs/exploratory/20260815_candidate_feature_cache"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--signal", default="burst_signal")
    parser.add_argument("--scene", default="WeaponMuseum_spectrum")
    parser.add_argument("--jsr", default="m10db")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-memory-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--smoothing-prior-count", type=float, default=32.0)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--all-cells", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def patch_columns(count):
    grid = int(round(float(count) ** 0.5))
    if grid * grid != count:
        raise ValueError(f"Non-square patch grid: {count}")
    return np.tile(np.arange(grid, dtype=np.int64), grid)


def find_cell(manifest, signal, scene, jsr):
    scenes = [item for item in manifest["scene_support"] if item["scene"] == scene]
    cells = [
        item
        for item in manifest["cells"]
        if item["signal"] == signal and item["scene"] == scene and item["jsr"] == jsr
    ]
    if len(scenes) != 1 or len(cells) != 1:
        raise KeyError(f"Cannot find unique cell {signal}/{scene}/{jsr}")
    return scenes[0]["support"], cells[0]


def build_base_memory(support_features, ratio, seed):
    raw = np.concatenate(support_features, axis=0).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
    count = max(1, int(round(raw.shape[0] * float(ratio))))
    indices = _farthest_indices(torch.from_numpy(raw), count, seed).numpy()
    return raw, raw[indices], indices


def support_reference_scores(raw, memory, memory_indices):
    similarities = raw @ memory.T
    base_positions = {int(raw_index): position for position, raw_index in enumerate(memory_indices)}
    for raw_index, memory_position in base_positions.items():
        similarities[raw_index, memory_position] = -np.inf
    best = similarities.max(axis=1)
    return (1.0 - best) / 2.0


def frequency_thresholds(reference_scores, groups, target_fpr, mode, prior_count):
    global_threshold = float(np.quantile(reference_scores, 1.0 - target_fpr))
    if mode == "global":
        return np.full(int(groups.max()) + 1, global_threshold, dtype=np.float32)

    thresholds = np.full(int(groups.max()) + 1, global_threshold, dtype=np.float32)
    for group in np.unique(groups):
        values = reference_scores[groups == group]
        local = float(np.quantile(values, 1.0 - target_fpr))
        if mode == "local":
            thresholds[int(group)] = local
        elif mode == "smooth":
            weight = len(values) / (len(values) + float(prior_count))
            thresholds[int(group)] = weight * local + (1.0 - weight) * global_threshold
        else:
            raise ValueError(f"Unknown calibration mode: {mode}")
    return thresholds


def calibrated_image_score(patches, memory, columns, thresholds, excluded_memory_positions=None):
    similarities = patches @ memory.T
    if excluded_memory_positions is not None and len(excluded_memory_positions):
        similarities[:, np.asarray(excluded_memory_positions, dtype=np.int64)] = -np.inf
    patch_scores = (1.0 - similarities.max(axis=1)) / 2.0
    normalized = patch_scores / np.maximum(thresholds[columns], 1e-8)
    return float(normalized.max())


def metric_summary(labels, scores):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels, scores) * 100.0),
        "auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if len(reached) else 100.0,
    }


def evaluate_cell(model, manifest, signal, scene, jsr, args, cache_dir):
    support_items, cell = find_cell(manifest, signal, scene, jsr)
    support_features = [encode_path(model, item["path"], args.device, cache_dir)[0] for item in support_items]
    raw, memory, memory_indices = build_base_memory(support_features, args.base_memory_ratio, args.seed)
    columns = np.tile(patch_columns(support_features[0].shape[0]), len(support_features))
    reference = support_reference_scores(raw, memory, memory_indices)
    # Map every selected memory patch back to the support image it came from.  A
    # support image must not score against *any* of its own selected patches:
    # otherwise its calibration score is optimistically low while test images
    # have no such self-match available.
    support_lengths = [features.shape[0] for features in support_features]
    support_starts = np.cumsum([0, *support_lengths[:-1]])
    memory_owner = np.searchsorted(support_starts[1:], memory_indices, side="right")

    query_items = [(item, 0) for item in cell["test_normals"]]
    query_items.extend((item, 1) for item in cell["test_abnormals"])
    query_features = [(encode_path(model, item["path"], args.device, cache_dir)[0], label) for item, label in query_items]

    modes = {"C1_global": "global", "C2_local": "local", "C3_smooth": "smooth"}
    results = {"C0_raw": {}}
    raw_scores = []
    labels = []
    for patches, label in query_features:
        raw_scores.append(calibrated_image_score(patches, memory, patch_columns(patches.shape[0]), np.ones(int(patch_columns(patches.shape[0]).max()) + 1)))
        labels.append(label)
    results["C0_raw"]["metrics"] = metric_summary(labels, raw_scores)

    for name, mode in modes.items():
        target_results = {}
        for target_fpr in (0.01, 0.05, 0.10):
            thresholds = frequency_thresholds(reference, columns, target_fpr, mode, args.smoothing_prior_count)
            support_scores = []
            test_scores = []
            test_labels = []
            for support_index, features in enumerate(support_features):
                excluded = np.flatnonzero(memory_owner == support_index)
                support_scores.append(
                    calibrated_image_score(
                        features,
                        memory,
                        patch_columns(features.shape[0]),
                        thresholds,
                        excluded_memory_positions=excluded,
                    )
                )
            for features, label in query_features:
                test_scores.append(calibrated_image_score(features, memory, patch_columns(features.shape[0]), thresholds))
                test_labels.append(label)
            alarm_threshold = float(np.quantile(support_scores, 1.0 - target_fpr))
            normal_scores = np.asarray([score for score, label in zip(test_scores, test_labels) if label == 0])
            abnormal_scores = np.asarray([score for score, label in zip(test_scores, test_labels) if label == 1])
            target_results[f"target_{int(target_fpr * 100)}"] = {
                "threshold": alarm_threshold,
                "actual_fpr": float(np.mean(normal_scores > alarm_threshold) * 100.0),
                "tpr": float(np.mean(abnormal_scores > alarm_threshold) * 100.0),
                "fpr_error": float(abs(np.mean(normal_scores > alarm_threshold) - target_fpr) * 100.0),
            }
        # Use the 5% calibration for ranking metrics.
        thresholds = frequency_thresholds(reference, columns, 0.05, mode, args.smoothing_prior_count)
        calibrated_scores = [
            calibrated_image_score(features, memory, patch_columns(features.shape[0]), thresholds)
            for features, _label in query_features
        ]
        results[name] = {
            "metrics": metric_summary(labels, calibrated_scores),
            "target_fpr": target_results,
        }
    result = {
        "signal": signal,
        "scene": scene,
        "jsr": jsr,
        "base_memory_count": int(memory.shape[0]),
        "raw_support_count": int(raw.shape[0]),
        "reference_count": int(reference.shape[0]),
        "results": results,
    }
    print(f"{signal}/{scene}/{jsr} C3 target5={results['C3_smooth']['target_fpr']['target_5']}")
    return result


def main():
    args = parse_args()
    manifest = json.loads(Path(args.support_manifest).read_text(encoding="utf-8"))
    model = build_model(args.device)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if args.all_cells:
        cells = [(item["signal"], item["scene"], item["jsr"]) for item in manifest["cells"]]
    else:
        cells = [(args.signal, args.scene, args.jsr)]
    results = [evaluate_cell(model, manifest, *cell, args, cache_dir) for cell in cells]
    if args.all_cells:
        for key in ("C0_raw", "C1_global", "C2_local", "C3_smooth"):
            metrics = [item["results"][key]["metrics"] for item in results]
            macro = {metric: float(np.mean([row[metric] for row in metrics])) for metric in metrics[0]}
            target5 = [item["results"][key].get("target_fpr", {}).get("target_5") for item in results]
            actual = [row["actual_fpr"] for row in target5 if row is not None]
            tpr = [row["tpr"] for row in target5 if row is not None]
            print(f"macro {key}: metrics={macro} target5_actual_fpr={np.mean(actual) if actual else None} target5_tpr={np.mean(tpr) if tpr else None}")
    if args.output_json:
        Path(args.output_json).write_text(json.dumps({"cells": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
