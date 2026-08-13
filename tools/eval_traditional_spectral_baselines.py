#!/usr/bin/env python
"""Evaluate classical communication-style statistics on spectrogram PNGs.

This runner deliberately keeps the protocol support-only: all centering and
scaling is fitted on confirmed normal support images, and test labels are read
only for final metrics.  The original ED/CFAR/cyclostationary literature often
assumes IQ samples, multiple sensors, or temporal snapshots.  Since the current
benchmarks provide spectrogram images, the quantitative rows here are explicitly
image-domain adaptations, not claims of a full IQ detector reproduction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.ofdma_spectrum import OFDMASpectrogramPreprocessor  # noqa: E402
from datasets.ofdma_target_scene import (  # noqa: E402
    DEFAULT_TARGET_SCENE_ROOT,
    available_target_scenes,
    build_target_scene_records,
    load_target_scene_manifest,
)
from tools.eval_cls_aux_cnn_gallery import (  # noqa: E402
    PUBLIC_RF_JSRS,
    public_rf_jobs,
    rf_target_jobs,
)
from utils.traditional_spectral import (  # noqa: E402
    METHOD_DISPLAY_NAMES,
    METHOD_REFERENCES,
    TRADITIONAL_METHODS,
    extract_raw_statistics,
    fit_support_calibrator,
)


DEFAULT_PUBLIC_MANIFEST = (
    REPO_ROOT
    / "analysis_outputs"
    / "20260805_public_rf_fixed_test_pool"
    / "support_manifest.json"
)


def safe_metrics(labels, scores) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "fpr95": float("nan")}
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels, scores) * 100.0),
        "auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if len(reached) else float("nan"),
    }


def image_from_sample(sample: dict, protocol: str, preprocessor=None) -> np.ndarray:
    image = cv2.imread(str(sample["path"]), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(sample["path"])
    if protocol == "ofdma":
        return preprocessor(image)
    if protocol in {"rf_target", "public_rf"}:
        height, width = image.shape[:2]
        size = min(max(height, width), 1024)
        if (height, width) != (size, size):
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return image


def public_or_target_jobs(args) -> list[dict]:
    common = SimpleNamespace(
        output_root=str(args.output_root),
        normal_sampling=args.normal_sampling,
        seed=args.seed,
        support_seed=args.support_seed,
        support_manifest=str(args.support_manifest) if args.support_manifest else "",
        public_rf_signals=tuple(args.public_rf_signals),
    )
    jobs = rf_target_jobs(common) if args.protocol == "rf_target" else public_rf_jobs(common)
    args.support_manifest = getattr(common, "support_manifest", args.support_manifest)
    args.support_manifest_sha256 = getattr(common, "support_manifest_sha256", None)
    args.test_normal_paths_sha256 = getattr(common, "test_normal_paths_sha256", None)
    args.support_policy = getattr(common, "support_policy", None)
    args.per_frequency_k = getattr(common, "per_frequency_k", None)
    return jobs


def ofdma_jobs(args) -> tuple[list[dict], dict]:
    dataset_root = Path(args.dataset_root)
    source = json.loads((dataset_root / "protocol.json").read_text())
    normalization = source["source_protocol"]["normalization"]
    manifest = load_target_scene_manifest(dataset_root)
    scene_ids = list(args.scene_ids) if args.scene_ids else available_target_scenes(manifest, args.split)
    unknown = sorted(set(scene_ids) - set(available_target_scenes(manifest, args.split)))
    if unknown:
        raise ValueError(f"Unknown {args.split} target scenes: {unknown}")

    jobs = []
    for shot in sorted(set(args.shots)):
        for scene_id in scene_ids:
            support = build_target_scene_records(
                dataset_root,
                manifest,
                scene_id,
                split=args.split,
                role="support",
                shot=shot,
            )
            test = build_target_scene_records(
                dataset_root,
                manifest,
                scene_id,
                split=args.split,
                role="test",
                max_normal_observations=args.max_normal_observations,
                max_anomaly_observations_per_type=args.max_anomaly_observations_per_type,
            )
            jobs.append(
                {
                    "dataset": "ofdma",
                    "category": "ofdma",
                    "scene": scene_id,
                    "jsr": f"{shot}shot",
                    "shot": int(shot),
                    "train_samples": [
                        {
                            "path": str(record.image_path),
                            "label": 0,
                            "name": f"{scene_id}::{record.observation_id}::su{record.su_id:02d}",
                            "observation_id": record.observation_id,
                            "jammer_type": record.jammer_type,
                        }
                        for record in support
                    ],
                    "test_samples": [
                        {
                            "path": str(record.image_path),
                            "label": int(record.label),
                            "name": f"{scene_id}::{record.observation_id}::su{record.su_id:02d}",
                            "observation_id": record.observation_id,
                            "jammer_type": record.jammer_type,
                        }
                        for record in test
                    ],
                }
            )
    return jobs, {
        "dataset_root": str(dataset_root.resolve()),
        "protocol_name": source["source_protocol"]["protocol_name"],
        "split": args.split,
        "target_scene_ids": scene_ids,
        "normalization": normalization,
    }


def sample_digest(samples: list[dict]) -> str:
    payload = "\n".join(
        f"{sample['path']}::{sample['label']}::{sample.get('name', '')}"
        for sample in samples
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def aggregate_ofdma(
    job: dict,
    image_scores: dict[str, list[float]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(job["test_samples"]):
        groups[str(sample["observation_id"])].append(index)
    labels = []
    jammer_types = []
    for observation_id in sorted(groups):
        indices = groups[observation_id]
        if len(indices) != 21:
            raise ValueError(
                f"{job['scene']}/{observation_id} has {len(indices)} SUs, expected 21"
            )
        sample_group = [job["test_samples"][index] for index in indices]
        labels_unique = {int(sample["label"]) for sample in sample_group}
        jammer_unique = {str(sample["jammer_type"]) for sample in sample_group}
        if len(labels_unique) != 1 or len(jammer_unique) != 1:
            raise ValueError(f"Inconsistent OFDMA observation {observation_id}")
        labels.append(next(iter(labels_unique)))
        jammer_types.append(next(iter(jammer_unique)))
    output = {}
    for method, scores in image_scores.items():
        output[method] = np.asarray(
            [max(scores[index] for index in groups[observation_id]) for observation_id in sorted(groups)],
            dtype=np.float32,
        )
    return np.asarray(labels, dtype=np.int32), {
        "jammer_types": np.asarray(jammer_types),
        **output,
    }


def evaluate_job(job: dict, args, preprocessor=None) -> tuple[list[dict], dict]:
    protocol = args.protocol
    frequency_axis = 0 if protocol == "ofdma" else 1
    support_images = [
        image_from_sample(sample, protocol, preprocessor)
        for sample in job["train_samples"]
    ]
    calibrator = fit_support_calibrator(
        support_images,
        frequency_axis,
        TRADITIONAL_METHODS,
    )
    labels = []
    raw_scores = {method: [] for method in TRADITIONAL_METHODS}
    for index, sample in enumerate(job["test_samples"], 1):
        image = image_from_sample(sample, protocol, preprocessor)
        labels.append(int(sample["label"]))
        individual = calibrator.score_raw(
            extract_raw_statistics(image, frequency_axis)
        )
        for method in TRADITIONAL_METHODS:
            raw_scores[method].append(float(individual[method]))
        if args.progress_every and index % args.progress_every == 0:
            print(
                f"  {job['dataset']}/{job['scene']}/{job['jsr']}: "
                f"{index}/{len(job['test_samples'])} images",
                flush=True,
            )

    if protocol == "ofdma":
        labels_array, aggregated = aggregate_ofdma(job, raw_scores)
        metric_labels = labels_array
        metric_scores = aggregated
        count_normal = int((labels_array == 0).sum())
        count_abnormal = int((labels_array == 1).sum())
    else:
        metric_labels = np.asarray(labels, dtype=np.int32)
        metric_scores = {method: np.asarray(scores, dtype=np.float32) for method, scores in raw_scores.items()}
        count_normal = int((metric_labels == 0).sum())
        count_abnormal = int((metric_labels == 1).sum())

    rows = []
    for method in TRADITIONAL_METHODS:
        metric = safe_metrics(metric_labels, metric_scores[method])
        rows.append(
            {
                "row_type": "cell",
                "dataset": job["dataset"],
                "category": job["category"],
                "scene": job["scene"],
                "jsr": job["jsr"],
                "shot": job.get("shot", "per_frequency"),
                "method": method,
                "method_display": METHOD_DISPLAY_NAMES[method],
                "num_support": len(job["train_samples"]),
                "num_test_normal": count_normal,
                "num_test_abnormal": count_abnormal,
                **metric,
            }
        )
    score_payload = {
        "labels": metric_labels,
        "methods": np.asarray(TRADITIONAL_METHODS),
        "names": np.asarray(
            sorted(
                {
                    str(sample.get("observation_id", sample["name"]))
                    for sample in job["test_samples"]
                }
            )
            if protocol == "ofdma"
            else [sample["name"] for sample in job["test_samples"]]
        ),
    }
    for method in TRADITIONAL_METHODS:
        score_payload[method] = metric_scores[method]
    return rows, score_payload


def append_macro(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["shot"], row["method"])].append(row)
    macro = []
    for (dataset, shot, method), selected in sorted(groups.items()):
        macro.append(
            {
                "row_type": "macro",
                "dataset": dataset,
                "category": "ALL",
                "scene": "ALL",
                "jsr": "ALL",
                "shot": shot,
                "method": method,
                "method_display": METHOD_DISPLAY_NAMES[method],
                "num_support": "",
                "num_test_normal": sum(int(row["num_test_normal"]) for row in selected),
                "num_test_abnormal": sum(int(row["num_test_abnormal"]) for row in selected),
                "auroc": float(np.nanmean([row["auroc"] for row in selected])),
                "auprc": float(np.nanmean([row["auprc"] for row in selected])),
                "fpr95": float(np.nanmean([row["fpr95"] for row in selected])),
            }
        )
    return macro


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["rf_target", "public_rf", "ofdma"], required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--normal-sampling", choices=["per_frequency", "1shot", "2shot", "4shot"], default="per_frequency")
    parser.add_argument("--support-manifest", default="")
    parser.add_argument("--support-seed", type=int, default=111)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--public-rf-signals", nargs="+", choices=list(PUBLIC_RF_JSRS), default=list(PUBLIC_RF_JSRS))
    parser.add_argument("--dataset-root", default=str(DEFAULT_TARGET_SCENE_ROOT))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--scene-ids", nargs="*", default=[])
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--max-normal-observations", type=int, default=0)
    parser.add_argument("--max-anomaly-observations-per-type", type=int, default=0)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.protocol == "public_rf"
        and not args.support_manifest
        and DEFAULT_PUBLIC_MANIFEST.is_file()
    ):
        args.support_manifest = str(DEFAULT_PUBLIC_MANIFEST)
    if args.protocol == "ofdma":
        jobs, dataset_protocol = ofdma_jobs(args)
    else:
        jobs = public_or_target_jobs(args)
        dataset_protocol = {
            "normal_sampling": args.normal_sampling,
            "support_manifest": str(args.support_manifest) if args.support_manifest else None,
            "support_manifest_sha256": getattr(args, "support_manifest_sha256", None),
            "support_policy": getattr(args, "support_policy", None),
            "per_frequency_k": getattr(args, "per_frequency_k", None),
        }
    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]
    if not jobs:
        raise RuntimeError("No jobs selected")
    if args.protocol == "ofdma":
        normalization = dataset_protocol["normalization"]
        preprocessor = OFDMASpectrogramPreprocessor(
            args.dataset_root,
            output_size=240,
            geometry="letterbox",
            min_db=float(normalization["min_db"]),
            max_db=float(normalization["max_db"]),
        )
    else:
        preprocessor = None

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_payload = {
        "entry": "tools/eval_traditional_spectral_baselines.py",
        "protocol": args.protocol,
        "dataset_protocol": dataset_protocol,
        "support_only_calibration": True,
        "test_labels_usage": "metrics_only",
        "frequency_axis": 0 if args.protocol == "ofdma" else 1,
        "methods": {
            method: {
                "display": METHOD_DISPLAY_NAMES[method],
                **METHOD_REFERENCES[method],
            }
            for method in TRADITIONAL_METHODS
        },
        "methods_not_directly_comparable_without_iq": {
            "cyclostationary_detection": {
                "citation": "W. A. Gardner, IEEE Signal Processing Magazine 8(2), 14--36, 1991.",
                "doi": "https://doi.org/10.1109/79.81007",
                "reason": "requires temporal/cyclic spectral correlation from IQ or time-series samples",
            },
            "eigenvalue_sensing": {
                "citation": "S. Dikmese and M. Renfors, CROWNCOM 2012.",
                "doi": "https://doi.org/10.4108/icst.crowncom.2012.248467",
                "reason": "requires covariance snapshots, usually multiple sensors or time samples",
            },
        },
        "num_jobs": len(jobs),
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rows = []
    for index, job in enumerate(jobs, 1):
        print(
            f"[{index}/{len(jobs)}] {job['dataset']} {job['category']} "
            f"{job['scene']} {job['jsr']} support={len(job['train_samples'])} "
            f"test={len(job['test_samples'])}",
            flush=True,
        )
        job_rows, score_payload = evaluate_job(job, args, preprocessor)
        rows.extend(job_rows)
        score_root = output_root / "scores"
        score_root.mkdir(parents=True, exist_ok=True)
        stem = f"{job['dataset']}-{job['category']}-{job['scene']}-{job['jsr']}".replace("/", "_")
        np.savez_compressed(
            score_root / f"{stem}.npz",
            **score_payload,
            support_paths=np.asarray([sample["path"] for sample in job["train_samples"]]),
            support_paths_sha256=np.asarray(sample_digest(job["train_samples"])),
        )
        write_rows(output_root / "metrics_per_cell.csv", rows)

    macros = append_macro(rows)
    write_rows(output_root / "metrics_macro.csv", macros)
    summary = {
        "status": "complete",
        "protocol": args.protocol,
        "num_jobs": len(jobs),
        "methods": list(TRADITIONAL_METHODS),
        "macro": macros,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Traditional spectrogram-domain communication baselines",
        "",
        f"- protocol: `{args.protocol}`",
        f"- jobs: `{len(jobs)}`",
        "- calibration: normal support only; abnormal/test labels are metrics-only",
        "- input caveat: current data are spectrogram PNGs, so ED/entropy/SFM/SK/CFAR are image-domain adaptations",
        "",
        "## Macro results",
        "",
        "| Method | Dataset | Shot | AUROC | AUPRC | FPR@95TPR | Source |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in macros:
        lines.append(
            f"| {row['method_display']} | {row['dataset']} | {row['shot']} | "
            f"{row['auroc']:.2f} | {row['auprc']:.2f} | {row['fpr95']:.2f} | "
            f"{METHOD_REFERENCES[row['method']]['short']} |"
        )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {output_root / 'metrics_macro.csv'}", flush=True)


if __name__ == "__main__":
    main()
