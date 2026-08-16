"""Screen feature-space local normal-memory interpolation.

The formal evaluator remains unchanged.  This experiment keeps the current
farthest coreset as the base memory, then appends virtual midpoint features
from safe normal-feature pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_candidate_methods import build_model, encode_path
from tools.eval_cls_vit_patchcore_gallery import _farthest_indices


DEFAULT_MANIFEST = ROOT / "analysis_outputs/20260728_rf_five_type_formal_seed111/support_manifest.json"
DEFAULT_CACHE = ROOT / "analysis_outputs/exploratory/20260815_candidate_feature_cache"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--signal", default="burst_signal")
    parser.add_argument("--scene", default="WeaponMuseum_spectrum")
    parser.add_argument("--jsr", default="m10db")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--variant",
        choices=("m0", "global_mutual", "frequency_nearest", "frequency_mutual"),
        default="frequency_mutual",
    )
    parser.add_argument("--augment-ratio", type=float, default=0.5)
    parser.add_argument("--base-memory-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--all-cells", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


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


def find_cell(manifest, signal, scene, jsr):
    scenes = [entry for entry in manifest["scene_support"] if entry["scene"] == scene]
    cells = [
        cell
        for cell in manifest["cells"]
        if cell["signal"] == signal and cell["scene"] == scene and cell["jsr"] == jsr
    ]
    if len(scenes) != 1 or len(cells) != 1:
        raise KeyError(f"Cannot find unique cell {signal}/{scene}/{jsr}")
    return scenes[0]["support"], cells[0]


def patch_columns(patch_count):
    grid = int(round(float(patch_count) ** 0.5))
    if grid * grid != patch_count:
        raise ValueError(f"Patch grid is not square: {patch_count}")
    return np.tile(np.arange(grid, dtype=np.int64), grid)


def nearest_pairs(features, groups, sources, variant, source_count):
    count = features.shape[0]
    similarities = features @ features.T
    np.fill_diagonal(similarities, -np.inf)
    if variant.startswith("frequency"):
        allowed = groups[:, None] == groups[None, :]
    else:
        allowed = np.ones((count, count), dtype=bool)
    if source_count > 1:
        allowed &= sources[:, None] != sources[None, :]
    similarities[~allowed] = -np.inf

    nearest = np.argmax(similarities, axis=1)
    valid = np.isfinite(similarities[np.arange(count), nearest])
    pairs = []
    if variant.endswith("mutual"):
        for left in np.flatnonzero(valid):
            right = int(nearest[left])
            if right > left and valid[right] and int(nearest[right]) == int(left):
                distance = float((1.0 - similarities[left, right]) / 2.0)
                pairs.append((distance, int(left), right))
    else:
        for left in np.flatnonzero(valid):
            right = int(nearest[left])
            if right == int(left):
                continue
            distance = float((1.0 - similarities[left, right]) / 2.0)
            pairs.append((distance, int(left), right))

    pairs.sort(key=lambda item: item[0])
    selected = []
    used = np.zeros(count, dtype=bool)
    for distance, left, right in pairs:
        if used[left] or used[right]:
            continue
        used[left] = True
        used[right] = True
        selected.append((distance, left, right))
    return selected


def build_memory(support_features, variant, augment_ratio, base_memory_ratio, seed):
    raw = np.concatenate(support_features, axis=0).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
    source_ids = np.concatenate(
        [np.full(features.shape[0], index, dtype=np.int64) for index, features in enumerate(support_features)]
    )
    columns = np.tile(patch_columns(support_features[0].shape[0]), len(support_features))

    base_count = max(1, int(round(raw.shape[0] * float(base_memory_ratio))))
    base_indices = _farthest_indices(
        torch.from_numpy(raw),
        base_count,
        seed,
    ).cpu().numpy()
    base = raw[base_indices]
    base_sources = source_ids[base_indices]
    base_columns = columns[base_indices]

    if variant == "m0" or augment_ratio <= 0:
        return base, {"raw_count": int(raw.shape[0]), "base_count": int(base.shape[0]), "pair_count": 0, "virtual_count": 0}

    pairs = nearest_pairs(
        base,
        base_columns if variant.startswith("frequency") else np.zeros(base.shape[0], dtype=np.int64),
        base_sources,
        variant,
        len(support_features),
    )
    target = int(round(base.shape[0] * float(augment_ratio)))
    chosen = pairs[:target]
    virtual = [
        (base[left] + base[right]) / 2.0
        for _distance, left, right in chosen
    ]
    if virtual:
        virtual_array = np.asarray(virtual, dtype=np.float32)
        virtual_array /= np.linalg.norm(virtual_array, axis=1, keepdims=True) + 1e-8
        memory = np.concatenate([base, virtual_array], axis=0)
    else:
        memory = base
    distances = [item[0] for item in chosen]
    return memory, {
        "raw_count": int(raw.shape[0]),
        "base_count": int(base.shape[0]),
        "pair_candidates": int(len(pairs)),
        "pair_count": int(len(chosen)),
        "virtual_count": int(len(virtual)),
        "pair_mean_distance": float(np.mean(distances)) if distances else None,
        "pair_max_distance": float(np.max(distances)) if distances else None,
    }


def evaluate_cell(model, manifest, signal, scene, jsr, args, cache_dir):
    support_items, cell = find_cell(manifest, signal, scene, jsr)
    support_features = [
        encode_path(model, item["path"], args.device, cache_dir)[0]
        for item in support_items
    ]
    memory, diagnostics = build_memory(
        support_features,
        args.variant,
        args.augment_ratio,
        args.base_memory_ratio,
        args.seed,
    )
    labels = []
    scores = []
    test_items = [(item, 0) for item in cell["test_normals"]]
    test_items.extend((item, 1) for item in cell["test_abnormals"])
    for item, label in test_items:
        patches = encode_path(model, item["path"], args.device, cache_dir)[0]
        similarity = patches @ memory.T
        patch_distance = (1.0 - similarity.max(axis=1)) / 2.0
        scores.append(float(patch_distance.max()))
        labels.append(label)
    result = {
        "signal": signal,
        "scene": scene,
        "jsr": jsr,
        "variant": args.variant,
        "augment_ratio": args.augment_ratio,
        "base_memory_ratio": args.base_memory_ratio,
        **diagnostics,
        **metric_summary(labels, scores),
    }
    print(
        f"{signal}/{scene}/{jsr} {args.variant} ratio={args.augment_ratio:.2f} "
        f"virtual={diagnostics['virtual_count']} auc={result['auroc']:.4f}"
    )
    return result


def main():
    args = parse_args()
    if not 0.0 <= args.augment_ratio <= 1.0:
        raise ValueError("--augment-ratio must be in [0, 1]")
    if not 0.0 < args.base_memory_ratio <= 1.0:
        raise ValueError("--base-memory-ratio must be in (0, 1]")
    manifest = json.loads(Path(args.support_manifest).read_text(encoding="utf-8"))
    model = build_model(args.device)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if args.all_cells:
        selected = [(cell["signal"], cell["scene"], cell["jsr"]) for cell in manifest["cells"]]
    else:
        selected = [(args.signal, args.scene, args.jsr)]
    results = [evaluate_cell(model, manifest, *cell, args, cache_dir) for cell in selected]
    macro = {
        key: float(np.mean([row[key] for row in results]))
        for key in ("auroc", "auprc", "fpr95")
    }
    print(f"macro {args.variant} ratio={args.augment_ratio:.2f}: {macro}")
    if args.output_json:
        payload = {"macro": macro, "cells": results}
        Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
