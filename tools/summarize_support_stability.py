#!/usr/bin/env python3
"""Aggregate repeated-support experiments with uncertainty and paired deltas.

The script never pools test images across replicates.  Each replicate is first
reduced to a cell-macro (RF) or scene-macro (OFDMA), then the replicate-level
statistics are reported.  This is the unit of analysis required for a support
stability claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


METRICS = ("auroc", "auprc", "fpr95")
METHOD_KEYS = {
    "ViT-only": "vit_score",
    "CNN-only": "cnn_score",
    "Ours": "confidence_gated_score",
}
PROMPTAD_KEY_CANDIDATES = {
    "self_rf": ("harmonic_text_vit_patchcore_max", "text_vit_patchcore_max"),
    "public_rf": ("text_vit_patchcore_max", "harmonic_text_vit_patchcore_max"),
}
OFDMA_METHOD_KEYS = {
    "ViT-only": "ours_vit",
    "CNN-only": "ours_cnn",
    "Ours": "confidence_gated_dual_visual",
}
SEED_RE = re.compile(r"seed_(\d+)$")
OFDMA_RE = re.compile(r"(test_\d+)_(\d+)shot_observation_scores\.npz$")


def metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return {key: float("nan") for key in METRICS}
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels, scores) * 100.0),
        "auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if len(reached) else 100.0,
    }


def read_npz(path: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        if "labels" not in data.files or key not in data.files:
            raise KeyError(f"{path} must contain labels and {key}; keys={data.files}")
        return np.asarray(data["labels"], dtype=np.int32), np.asarray(data[key], dtype=np.float64)


def read_first_available(path: Path, keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Read the PromptAD-compatible score key used by the corresponding RF evaluator."""

    with np.load(path, allow_pickle=True) as data:
        key = next((candidate for candidate in keys if candidate in data.files), None)
        if key is None:
            raise KeyError(f"{path} has none of the expected score keys: {keys}")
        return np.asarray(data["labels"], dtype=np.int32), np.asarray(data[key], dtype=np.float64)


def score_signature(path: Path) -> str:
    """Hash the ordered test names and labels used by a score file.

    Names alone catch most accidental split changes, but including labels and
    the array length also verifies that two methods are evaluated on the same
    ordered unit partition rather than merely on files with matching names.
    """

    with np.load(path, allow_pickle=True) as data:
        if "names" in data.files:
            names = [str(value) for value in np.asarray(data["names"]).reshape(-1)]
        elif {"target_scene_ids", "observation_ids"}.issubset(data.files):
            # RF score files store a flat ``names`` array.  OFDMA observation
            # files store the same ordered test units as separate scene and
            # observation arrays; canonicalize that pair before hashing.
            scene_ids = [
                str(value) for value in np.asarray(data["target_scene_ids"]).reshape(-1)
            ]
            observation_ids = [
                str(value) for value in np.asarray(data["observation_ids"]).reshape(-1)
            ]
            if len(scene_ids) != len(observation_ids):
                raise ValueError(
                    f"{path} has mismatched target_scene_ids/observation_ids lengths: "
                    f"{len(scene_ids)} != {len(observation_ids)}"
                )
            names = [f"{scene}::{observation}" for scene, observation in zip(scene_ids, observation_ids)]
        else:
            raise KeyError(
                f"{path} has neither names nor target_scene_ids/observation_ids "
                "for test-split validation"
            )
        if "labels" not in data.files:
            raise KeyError(f"{path} has no labels array for test-split validation")
        labels = np.asarray(data["labels"], dtype=np.int8).reshape(-1)
    if len(names) != len(labels):
        raise ValueError(
            f"{path} has mismatched names/labels lengths: {len(names)} != {len(labels)}"
        )
    payload = json.dumps(
        {"names": names, "labels": labels.tolist()},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_test_name_alignment(replicate_dirs: list[Path]) -> None:
    """Fail closed if a repeated-support run changes test units or labels.

    RF fusion and PromptAD-compatible ViT scores are compared under one
    canonical dataset key, so this also checks that every RF method uses the
    exact same ordered test cells within each replicate.  The same signature
    is then required across all support repeats.
    """

    expected: dict[tuple[str, str], str] = {}
    for replicate in replicate_dirs:
        for dataset, score_dir in (
            ("self_rf", replicate / "rf" / "self_fusion" / "scores"),
            ("public_rf", replicate / "rf" / "public_fusion" / "scores"),
            ("self_rf", replicate / "rf" / "self_vit" / "scores"),
            ("public_rf", replicate / "rf" / "public_vit" / "scores"),
            ("ofdma", replicate / "ofdma" / "test" / "scores"),
        ):
            if not score_dir.is_dir():
                continue
            for path in sorted(score_dir.glob("*.npz")):
                key = (dataset, path.name)
                digest = score_signature(path)
                previous = expected.get(key)
                if previous is not None and previous != digest:
                    raise ValueError(
                        "Test unit/label partition mismatch for "
                        f"{dataset}/{path.name}: {previous} != {digest}"
                    )
                expected[key] = digest


def seed_from_root(path: Path) -> int:
    match = SEED_RE.search(path.name)
    if match is None:
        raise ValueError(f"Replicate directory must be named seed_<int>: {path}")
    return int(match.group(1))


def rf_replicate_rows(seed_root: Path, seed: int) -> list[dict]:
    rows: list[dict] = []
    for dataset, fusion_dir in (
        ("self_rf", seed_root / "rf" / "self_fusion" / "scores"),
        ("public_rf", seed_root / "rf" / "public_fusion" / "scores"),
    ):
        if not fusion_dir.is_dir():
            continue
        cell_rows = defaultdict(list)
        for path in sorted(fusion_dir.glob("*.npz")):
            with np.load(path, allow_pickle=True) as data:
                labels = np.asarray(data["labels"], dtype=np.int32)
                for method, key in METHOD_KEYS.items():
                    cell_rows[method].append(metric(labels, np.asarray(data[key], dtype=np.float64)))
                if {"vit_normalized", "cnn_normalized"}.issubset(data.files):
                    ungated = 1.0 - (
                        1.0 - np.asarray(data["vit_normalized"], dtype=np.float64)
                    ) * (1.0 - np.asarray(data["cnn_normalized"], dtype=np.float64))
                    cell_rows["Ungated OR"].append(metric(labels, ungated))
        promptad_dir = (
            seed_root / "rf" / ("self_vit" if dataset == "self_rf" else "public_vit") / "scores"
        )
        promptad_rows = cell_rows["PromptAD"]
        for path in sorted(promptad_dir.glob("*.npz")):
            labels, scores = read_first_available(path, PROMPTAD_KEY_CANDIDATES[dataset])
            promptad_rows.append(metric(labels, scores))
        for method, values in cell_rows.items():
            if not values:
                continue
            rows.append(
                {
                    "seed": seed,
                    "dataset": dataset,
                    "shot": "all",
                    "method": method,
                    "unit": "cell",
                    "n_units": len(values),
                    **{
                        key: float(np.nanmean([item[key] for item in values]))
                        for key in METRICS
                    },
                }
            )
    return rows


def rf_cell_rows(seed_root: Path, seed: int) -> list[dict]:
    """Return per-cell RF metrics for signal-level uncertainty summaries."""

    rows: list[dict] = []
    for dataset, fusion_dir in (
        ("self_rf", seed_root / "rf" / "self_fusion" / "scores"),
        ("public_rf", seed_root / "rf" / "public_fusion" / "scores"),
    ):
        if not fusion_dir.is_dir():
            continue
        for path in sorted(fusion_dir.glob("*.npz")):
            stem = path.stem
            for prefix in ("in_house_rf-", "public_rf-"):
                if stem.startswith(prefix):
                    stem = stem[len(prefix):]
                    break
            signal = stem.split("-", 1)[0].removesuffix("_signal")
            with np.load(path, allow_pickle=True) as data:
                for method, key in METHOD_KEYS.items():
                    values = metric(data["labels"], data[key])
                    rows.append(
                        {
                            "seed": seed,
                            "dataset": dataset,
                            "signal": signal,
                            "cell": path.name,
                            "method": method,
                            "unit": "cell",
                            **values,
                        }
                    )
                if {"vit_normalized", "cnn_normalized"}.issubset(data.files):
                    ungated = 1.0 - (
                        1.0 - np.asarray(data["vit_normalized"], dtype=np.float64)
                    ) * (1.0 - np.asarray(data["cnn_normalized"], dtype=np.float64))
                    rows.append(
                        {
                            "seed": seed,
                            "dataset": dataset,
                            "signal": signal,
                            "cell": path.name,
                            "method": "Ungated OR",
                            "unit": "cell",
                            **metric(data["labels"], ungated),
                        }
                    )
        promptad_dir = (
            seed_root / "rf" / ("self_vit" if dataset == "self_rf" else "public_vit") / "scores"
        )
        for path in sorted(promptad_dir.glob("*.npz")):
            stem = path.stem
            for prefix in ("in_house_rf-", "public_rf-"):
                if stem.startswith(prefix):
                    stem = stem[len(prefix):]
                    break
            signal = stem.split("-", 1)[0].removesuffix("_signal")
            labels, scores = read_first_available(path, PROMPTAD_KEY_CANDIDATES[dataset])
            rows.append(
                {
                    "seed": seed,
                    "dataset": dataset,
                    "signal": signal,
                    "cell": path.name,
                    "method": "PromptAD",
                    "unit": "cell",
                    **metric(labels, scores),
                }
            )
    return rows


def ofdma_replicate_rows(seed_root: Path, seed: int) -> list[dict]:
    score_dir = seed_root / "ofdma" / "test" / "scores"
    if not score_dir.is_dir():
        return []
    per_method_shot: dict[tuple[int, str], list[dict[str, float]]] = defaultdict(list)
    for path in sorted(score_dir.glob("test_*shot_observation_scores.npz")):
        match = OFDMA_RE.fullmatch(path.name)
        if match is None:
            continue
        shot = int(match.group(2))
        for method, key in OFDMA_METHOD_KEYS.items():
            labels, scores = read_npz(path, key)
            per_method_shot[(shot, method)].append(metric(labels, scores))
    rows: list[dict] = []
    for (shot, method), values in sorted(per_method_shot.items()):
        rows.append(
            {
                "seed": seed,
                "dataset": "ofdma",
                "shot": str(shot),
                "method": method,
                "unit": "scene",
                "n_units": len(values),
                **{
                    key: float(np.nanmean([item[key] for item in values]))
                    for key in METRICS
                },
            }
        )
    return rows


def ofdma_scene_rows(seed_root: Path, seed: int) -> list[dict]:
    """Return one metric row per (support seed, scene, shot, method).

    These rows are kept separate from the replicate macro rows so a scene-level
    bootstrap can resample target scenes without ever concatenating observation
    scores from different scenes or support replicates.
    """

    score_dir = seed_root / "ofdma" / "test" / "scores"
    if not score_dir.is_dir():
        return []
    rows: list[dict] = []
    for path in sorted(score_dir.glob("test_*shot_observation_scores.npz")):
        match = OFDMA_RE.fullmatch(path.name)
        if match is None:
            continue
        scene = match.group(1)
        shot = int(match.group(2))
        for method, key in OFDMA_METHOD_KEYS.items():
            labels, scores = read_npz(path, key)
            rows.append(
                {
                    "seed": seed,
                    "scene": scene,
                    "dataset": "ofdma",
                    "shot": str(shot),
                    "method": method,
                    "unit": "scene",
                    **metric(labels, scores),
                }
            )
    return rows


def ofdma_jammer_scene_rows(seed_root: Path, seed: int) -> list[dict]:
    """Return scene-level metrics for each jammer against the same normal set."""

    score_dir = seed_root / "ofdma" / "test" / "scores"
    if not score_dir.is_dir():
        return []
    rows: list[dict] = []
    for path in sorted(score_dir.glob("test_*shot_observation_scores.npz")):
        match = OFDMA_RE.fullmatch(path.name)
        if match is None:
            continue
        scene = match.group(1)
        shot = int(match.group(2))
        with np.load(path, allow_pickle=True) as data:
            labels = np.asarray(data["labels"], dtype=np.int32)
            jammer_types = np.asarray(data["jammer_types"]).astype(str)
            normal_mask = np.char.lower(jammer_types) == "no jammer"
            jammer_names = sorted(
                name for name in np.unique(jammer_types)
                if str(name).lower() != "no jammer"
            )
            for jammer in jammer_names:
                mask = normal_mask | (jammer_types == jammer)
                if np.count_nonzero(mask) == 0:
                    continue
                for method, key in OFDMA_METHOD_KEYS.items():
                    values = metric(labels[mask], np.asarray(data[key], dtype=np.float64)[mask])
                    rows.append(
                        {
                            "seed": seed,
                            "scene": scene,
                            "dataset": "ofdma",
                            "shot": str(shot),
                            "jammer": str(jammer),
                            "method": method,
                            "unit": "scene",
                            **values,
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normal_ci(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size <= 1:
        mean = float(values.mean()) if values.size else float("nan")
        return mean, mean
    half = 1.96 * float(values.std(ddof=1)) / math.sqrt(values.size)
    mean = float(values.mean())
    return mean - half, mean + half


def paired_bootstrap(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size <= 1:
        value = float(values.mean()) if values.size else float("nan")
        return value, value
    indices = rng.integers(0, values.size, size=(int(n_boot), values.size))
    estimates = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))


def sign_permutation_pvalue(values: np.ndarray, rng: np.random.Generator, n_perm: int) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    observed = abs(float(values.mean()))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(int(n_perm), values.size))
    null = np.abs((signs * values[None, :]).mean(axis=1))
    return float((1.0 + np.count_nonzero(null >= observed)) / (len(null) + 1.0))


def scene_bootstrap_ci(
    rows: list[dict],
    *,
    metric_name: str,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, int, int]:
    """Bootstrap target scenes, keeping support seeds paired within each scene."""

    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = float(row[metric_name])
        if np.isfinite(value):
            by_seed[int(row["seed"])][str(row["scene"])] = value
    seeds = sorted(by_seed)
    scenes = sorted(set.intersection(*(set(by_seed[seed]) for seed in seeds))) if seeds else []
    if not scenes:
        return float("nan"), float("nan"), 0, 0
    matrix = np.asarray(
        [[by_seed[seed][scene] for scene in scenes] for seed in seeds],
        dtype=np.float64,
    )
    estimates = np.empty(int(n_boot), dtype=np.float64)
    for index in range(int(n_boot)):
        sampled = rng.integers(0, len(scenes), size=len(scenes))
        estimates[index] = float(matrix[:, sampled].mean())
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high), len(seeds), len(scenes)


def paired_scene_bootstrap(
    ours_rows: list[dict],
    base_rows: list[dict],
    *,
    metric_name: str,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, float, int, int]:
    ours = {(int(row["seed"]), str(row["scene"])): float(row[metric_name]) for row in ours_rows}
    base = {(int(row["seed"]), str(row["scene"])): float(row[metric_name]) for row in base_rows}
    pairs = {
        key: ours[key] - base[key]
        for key in sorted(set(ours) & set(base))
        if np.isfinite(ours[key]) and np.isfinite(base[key])
    }
    seeds = sorted({key[0] for key in pairs})
    scenes = sorted({key[1] for key in pairs})
    if not seeds or not scenes:
        return float("nan"), float("nan"), float("nan"), 0, 0
    matrix = np.asarray(
        [[pairs.get((seed, scene), np.nan) for scene in scenes] for seed in seeds],
        dtype=np.float64,
    )
    valid_scenes = np.all(np.isfinite(matrix), axis=0)
    matrix = matrix[:, valid_scenes]
    scenes = [scene for scene, valid in zip(scenes, valid_scenes) if valid]
    if not scenes:
        return float("nan"), float("nan"), float("nan"), 0, 0
    estimates = np.empty(int(n_boot), dtype=np.float64)
    for index in range(int(n_boot)):
        sampled = rng.integers(0, len(scenes), size=len(scenes))
        estimates[index] = float(matrix[:, sampled].mean())
    low, high = np.quantile(estimates, [0.025, 0.975])
    scene_means = matrix.mean(axis=0)
    pvalue = sign_permutation_pvalue(scene_means, rng, n_boot)
    return float(scene_means.mean()), float(low), float(high), pvalue, len(scenes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()

    replicate_dirs = sorted(
        path
        for path in (args.run_root / "replicates").glob("seed_*")
        if path.is_dir() and (path / "complete.json").is_file()
    )
    if not replicate_dirs:
        raise FileNotFoundError(f"No replicate directories under {args.run_root / 'replicates'}")
    rows: list[dict] = []
    cell_rows: list[dict] = []
    scene_rows: list[dict] = []
    jammer_scene_rows: list[dict] = []
    for replicate in replicate_dirs:
        seed = seed_from_root(replicate)
        rows.extend(rf_replicate_rows(replicate, seed))
        cell_rows.extend(rf_cell_rows(replicate, seed))
        rows.extend(ofdma_replicate_rows(replicate, seed))
        scene_rows.extend(ofdma_scene_rows(replicate, seed))
        jammer_scene_rows.extend(ofdma_jammer_scene_rows(replicate, seed))
    if not rows:
        raise RuntimeError("No completed RF/OFDMA replicate scores found")

    validate_test_name_alignment(replicate_dirs)

    write_csv(args.output_root / "per_replicate_metrics.csv", rows)
    write_csv(args.output_root / "rf_cell_metrics.csv", cell_rows)
    write_csv(args.output_root / "ofdma_scene_metrics.csv", scene_rows)
    write_csv(args.output_root / "ofdma_jammer_scene_metrics.csv", jammer_scene_rows)
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["shot"], row["method"], row["unit"])].append(row)

    summary: list[dict] = []
    for (dataset, shot, method, unit), values in sorted(grouped.items()):
        for metric_name in METRICS:
            array = np.asarray([row[metric_name] for row in values], dtype=np.float64)
            low, high = normal_ci(array)
            summary.append(
                {
                    "dataset": dataset,
                    "shot": shot,
                    "method": method,
                    "unit": unit,
                    "metric": metric_name,
                    "n_replicates": int(np.isfinite(array).sum()),
                    "mean": float(np.nanmean(array)),
                    "std": float(np.nanstd(array, ddof=1)) if np.isfinite(array).sum() > 1 else 0.0,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    write_csv(args.output_root / "mean_std_ci95.csv", summary)

    signal_replicate: dict[tuple[int, str, str, str], list[dict]] = defaultdict(list)
    for row in cell_rows:
        signal_replicate[(int(row["seed"]), row["dataset"], row["signal"], row["method"])].append(row)
    signal_rows: list[dict] = []
    for (seed, dataset, signal, method), values in sorted(signal_replicate.items()):
        signal_rows.append(
            {
                "seed": seed,
                "dataset": dataset,
                "signal": signal,
                "method": method,
                "unit": "cell",
                "n_units": len(values),
                **{
                    metric_name: float(np.nanmean([value[metric_name] for value in values]))
                    for metric_name in METRICS
                },
            }
        )
    write_csv(args.output_root / "rf_signal_replicate_metrics.csv", signal_rows)
    signal_summary: list[dict] = []
    signal_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in signal_rows:
        signal_groups[(row["dataset"], row["signal"], row["method"])].append(row)
    for (dataset, signal, method), values in sorted(signal_groups.items()):
        for metric_name in METRICS:
            array = np.asarray([row[metric_name] for row in values], dtype=np.float64)
            low, high = normal_ci(array)
            signal_summary.append(
                {
                    "dataset": dataset,
                    "signal": signal,
                    "method": method,
                    "metric": metric_name,
                    "n_replicates": int(np.isfinite(array).sum()),
                    "mean": float(np.nanmean(array)),
                    "std": float(np.nanstd(array, ddof=1)) if np.isfinite(array).sum() > 1 else 0.0,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    write_csv(args.output_root / "rf_signal_mean_std_ci95.csv", signal_summary)

    rng = np.random.default_rng(20260801)
    delta_rows: list[dict] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        shots = sorted({row["shot"] for row in rows if row["dataset"] == dataset})
        for shot in shots:
            by_method = {
                method: {
                    row["seed"]: row
                    for row in rows
                    if row["dataset"] == dataset
                    and row["shot"] == shot
                    and row["method"] == method
                }
                for method in ("ViT-only", "CNN-only", "PromptAD", "Ungated OR", "Ours")
            }
            for baseline in ("ViT-only", "CNN-only", "PromptAD", "Ungated OR"):
                common = sorted(set(by_method["Ours"]) & set(by_method[baseline]))
                if not common:
                    continue
                for metric_name in METRICS:
                    deltas = np.asarray(
                        [
                            by_method["Ours"][seed][metric_name]
                            - by_method[baseline][seed][metric_name]
                            for seed in common
                        ],
                        dtype=np.float64,
                    )
                    low, high = paired_bootstrap(
                        deltas, rng, args.bootstrap_replicates
                    )
                    delta_rows.append(
                        {
                            "dataset": dataset,
                            "shot": shot,
                            "comparison": f"Ours - {baseline}",
                            "metric": metric_name,
                            "n_pairs": len(deltas),
                            "delta_mean": float(np.mean(deltas)),
                            "delta_std": float(np.std(deltas, ddof=1))
                            if len(deltas) > 1
                            else 0.0,
                            "bootstrap_ci95_low": low,
                            "bootstrap_ci95_high": high,
                            "sign_permutation_p": sign_permutation_pvalue(
                                deltas, rng, args.bootstrap_replicates
                            ),
                        }
                    )
    write_csv(args.output_root / "paired_deltas.csv", delta_rows)

    scene_summary: list[dict] = []
    if scene_rows:
        scene_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in scene_rows:
            scene_groups[(row["shot"], row["method"])].append(row)
        for (shot, method), values in sorted(scene_groups.items()):
            for metric_name in METRICS:
                low, high, n_seeds, n_scenes = scene_bootstrap_ci(
                    values,
                    metric_name=metric_name,
                    rng=rng,
                    n_boot=args.bootstrap_replicates,
                )
                finite = np.asarray([row[metric_name] for row in values], dtype=np.float64)
                scene_summary.append(
                    {
                        "dataset": "ofdma",
                        "shot": shot,
                        "method": method,
                        "metric": metric_name,
                        "n_replicates": n_seeds,
                        "n_scenes": n_scenes,
                        "scene_macro_mean": float(np.nanmean(finite)),
                        "scene_bootstrap_ci95_low": low,
                        "scene_bootstrap_ci95_high": high,
                    }
                )
    write_csv(args.output_root / "ofdma_scene_bootstrap_ci95.csv", scene_summary)

    scene_delta_rows: list[dict] = []
    for shot in sorted({row["shot"] for row in scene_rows}):
        for baseline in ("ViT-only", "CNN-only"):
            for metric_name in METRICS:
                ours_values = [
                    row for row in scene_rows
                    if row["shot"] == shot and row["method"] == "Ours"
                ]
                baseline_values = [
                    row for row in scene_rows
                    if row["shot"] == shot and row["method"] == baseline
                ]
                mean, low, high, pvalue, n_scenes = paired_scene_bootstrap(
                    ours_values,
                    baseline_values,
                    metric_name=metric_name,
                    rng=rng,
                    n_boot=args.bootstrap_replicates,
                )
                if n_scenes:
                    scene_delta_rows.append(
                        {
                            "dataset": "ofdma",
                            "shot": shot,
                            "comparison": f"Ours - {baseline}",
                            "metric": metric_name,
                            "n_scenes": n_scenes,
                            "delta_mean": mean,
                            "scene_bootstrap_ci95_low": low,
                            "scene_bootstrap_ci95_high": high,
                            "sign_permutation_p": pvalue,
                        }
                    )
    write_csv(args.output_root / "ofdma_paired_scene_bootstrap.csv", scene_delta_rows)

    jammer_replicate_rows: list[dict] = []
    jammer_groups: dict[tuple[int, str, str, str], list[dict]] = defaultdict(list)
    for row in jammer_scene_rows:
        jammer_groups[
            (int(row["seed"]), row["shot"], row["jammer"], row["method"])
        ].append(row)
    for (seed, shot, jammer, method), values in sorted(jammer_groups.items()):
        jammer_replicate_rows.append(
            {
                "seed": seed,
                "dataset": "ofdma",
                "shot": shot,
                "jammer": jammer,
                "method": method,
                "unit": "scene",
                "n_units": len(values),
                **{
                    metric_name: float(np.nanmean([value[metric_name] for value in values]))
                    for metric_name in METRICS
                },
            }
        )
    write_csv(args.output_root / "ofdma_jammer_replicate_metrics.csv", jammer_replicate_rows)
    jammer_summary: list[dict] = []
    jammer_summary_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in jammer_replicate_rows:
        jammer_summary_groups[(row["shot"], row["jammer"], row["method"])].append(row)
    for (shot, jammer, method), values in sorted(jammer_summary_groups.items()):
        for metric_name in METRICS:
            array = np.asarray([value[metric_name] for value in values], dtype=np.float64)
            low, high = normal_ci(array)
            jammer_summary.append(
                {
                    "dataset": "ofdma",
                    "shot": shot,
                    "jammer": jammer,
                    "method": method,
                    "metric": metric_name,
                    "n_replicates": int(np.isfinite(array).sum()),
                    "mean": float(np.nanmean(array)),
                    "std": float(np.nanstd(array, ddof=1))
                    if np.isfinite(array).sum() > 1 else 0.0,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    write_csv(args.output_root / "ofdma_jammer_mean_std_ci95.csv", jammer_summary)

    payload = {
        "run_root": str(args.run_root.resolve()),
        "replicates": [seed_from_root(path) for path in replicate_dirs],
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "unit_of_analysis": {
            "rf": "equal-weighted cell macro per replicate",
            "ofdma": "equal-weighted target-scene macro per replicate and shot",
        },
        "files": [
            "per_replicate_metrics.csv",
            "rf_cell_metrics.csv",
            "rf_signal_replicate_metrics.csv",
            "rf_signal_mean_std_ci95.csv",
            "ofdma_scene_metrics.csv",
            "ofdma_jammer_scene_metrics.csv",
            "ofdma_jammer_replicate_metrics.csv",
            "ofdma_jammer_mean_std_ci95.csv",
            "mean_std_ci95.csv",
            "paired_deltas.csv",
            "ofdma_scene_bootstrap_ci95.csv",
            "ofdma_paired_scene_bootstrap.csv",
        ],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
