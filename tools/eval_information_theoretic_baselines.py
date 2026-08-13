#!/usr/bin/env python
"""CPU-only formal evaluation for frozen KLD and ICA spectrum detectors.

The evaluator shares the project's established data splits and support pools:

* In-house/Public RF: ``rf_target_jobs`` and ``public_rf_jobs``;
* OFDMA: target-scene ``build_jobs`` with max aggregation over 21 SUs;
* FedJam: benign-only ``select_benign_support`` and ``iter_test_records``.

Only confirmed-normal support images fit histogram ranges and probabilities.
Query labels are retained separately and are read only when metrics are
computed.  Both methods can be evaluated in one image-loading pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

# This runner is intentionally CPU-only, including when imported by a job
# launcher that has made GPUs visible.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.information_theoretic_spectral import (  # noqa: E402
    DEFAULT_ICA_BINS,
    DEFAULT_ICA_CLUSTER_LENGTH,
    DEFAULT_ICA_THRESHOLD_FACTOR,
    DEFAULT_KLD_BINS,
    DEFAULT_PSEUDOCOUNT,
    ICAFrozen,
    KLDRef,
)


METHODS = ("kld_reference", "ica_frozen")
METHOD_DISPLAY = {
    "kld_reference": "KLD-Ref",
    "ica_frozen": "ICA-Frozen",
}
REFERENCE = {
    "authors": "M. Afgani, S. Sinanovic, and H. Haas",
    "title": "The Information Theoretic Approach to Signal Anomaly Detection for Cognitive Radio",
    "venue": "International Journal of Digital Multimedia Broadcasting",
    "year": 2010,
    "doi": "https://doi.org/10.1155/2010/740594",
    "local_pdf": "references/papers/spectrum_domain/17_afgani_2010_information_theoretic.pdf",
}
FEDJAM_LABEL_NAMES = {
    0: "benign",
    1: "pulse",
    2: "single_tone",
    3: "wideband",
}


def safe_metrics(labels: Iterable[int], scores: Iterable[float]) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels_array.size != scores_array.size:
        raise ValueError("labels and scores must have equal length")
    if labels_array.size == 0 or np.unique(labels_array).size < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "fpr95": float("nan")}
    if not np.all(np.isfinite(scores_array)):
        raise ValueError("scores must all be finite")
    fpr, tpr, _ = roc_curve(labels_array, scores_array)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels_array, scores_array) * 100.0),
        "auprc": float(average_precision_score(labels_array, scores_array) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if reached.size else float("nan"),
    }


def validate_methods(methods: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(methods))
    unknown = sorted(set(selected) - set(METHODS))
    if not selected or unknown:
        raise ValueError(f"methods must be selected from {METHODS}; unknown={unknown}")
    return selected


def fit_detectors(
    support_images: Iterable[np.ndarray], methods: Iterable[str]
) -> dict[str, KLDRef | ICAFrozen]:
    """Fit selected detectors from normal support only, using paper defaults."""

    selected = validate_methods(methods)
    images = [np.asarray(image) for image in support_images]
    if not images:
        raise ValueError("At least one normal support image is required")
    detectors: dict[str, KLDRef | ICAFrozen] = {}
    if "kld_reference" in selected:
        detectors["kld_reference"] = KLDRef.fit(images)
    if "ica_frozen" in selected:
        detectors["ica_frozen"] = ICAFrozen.fit(images)
    return detectors


def score_image(
    detectors: dict[str, KLDRef | ICAFrozen], image: np.ndarray, *, time_axis: int
) -> dict[str, float]:
    scores: dict[str, float] = {}
    if "kld_reference" in detectors:
        scores["kld_reference"] = float(detectors["kld_reference"].score(image))
    if "ica_frozen" in detectors:
        scores["ica_frozen"] = float(
            detectors["ica_frozen"].score(image, time_axis=time_axis)
        )
    if not all(np.isfinite(value) for value in scores.values()):
        raise RuntimeError("Information-theoretic detector produced a non-finite score")
    return scores


def limit_binary_samples(samples: list[dict], max_per_class: int) -> list[dict]:
    if max_per_class < 0:
        raise ValueError("max_per_class cannot be negative")
    if max_per_class == 0:
        return samples
    counts = {0: 0, 1: 0}
    selected = []
    for sample in samples:
        label = int(sample["label"])
        if label not in counts:
            raise ValueError(f"Expected binary RF label, got {label}")
        if counts[label] >= max_per_class:
            continue
        selected.append(sample)
        counts[label] += 1
    return selected


def load_path_image(sample: dict, protocol: str, preprocessor=None) -> np.ndarray:
    image = cv2.imread(str(sample["path"]), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(sample["path"])
    if protocol == "ofdma":
        if preprocessor is None:
            raise ValueError("OFDMA requires its formal preprocessor")
        return np.asarray(preprocessor(image))
    height, width = image.shape
    size = min(max(height, width), 1024)
    if (height, width) != (size, size):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(image)


def evaluate_path_job(job: dict, args, preprocessor=None) -> tuple[dict, dict]:
    support_images = [
        load_path_image(sample, args.protocol, preprocessor)
        for sample in job["train_samples"]
    ]
    detectors = fit_detectors(support_images, args.methods)
    names: list[str] = []
    labels: list[int] = []
    observation_ids: list[str] = []
    jammer_types: list[str] = []
    score_bank = {method: [] for method in args.methods}
    for index, sample in enumerate(job["test_samples"], 1):
        image = load_path_image(sample, args.protocol, preprocessor)
        values = score_image(detectors, image, time_axis=args.time_axis)
        for method in args.methods:
            score_bank[method].append(values[method])
        # Labels never enter fitting or scoring; they are copied only after the
        # query score has been finalized for later metric calculation.
        names.append(str(sample["name"]))
        labels.append(int(sample["label"]))
        observation_ids.append(str(sample.get("observation_id", "")))
        jammer_types.append(str(sample.get("jammer_type", "")))
        if args.progress_every and index % args.progress_every == 0:
            print(f"  scored {index}/{len(job['test_samples'])} images", flush=True)
    payload = {
        "names": np.asarray(names),
        "labels": np.asarray(labels, dtype=np.int32),
        "observation_ids": np.asarray(observation_ids),
        "jammer_types": np.asarray(jammer_types),
        **{
            method: np.asarray(score_bank[method], dtype=np.float32)
            for method in args.methods
        },
    }
    support_payload = {
        "support_names": np.asarray([str(item["name"]) for item in job["train_samples"]]),
        "support_paths": np.asarray([str(item["path"]) for item in job["train_samples"]]),
    }
    return payload, support_payload


def aggregate_ofdma(payload: dict, methods: Iterable[str], expected_sus: int = 21) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, observation_id in enumerate(payload["observation_ids"]):
        groups[str(observation_id)].append(index)
    output: dict[str, list] = {
        "observation_ids": [],
        "labels": [],
        "jammer_types": [],
    }
    for method in methods:
        output[method] = []
    for observation_id, indices_list in sorted(groups.items()):
        indices = np.asarray(indices_list, dtype=np.int64)
        if indices.size != expected_sus:
            raise ValueError(
                f"OFDMA observation {observation_id} has {indices.size} SUs, "
                f"expected {expected_sus}"
            )
        labels = np.unique(payload["labels"][indices])
        jammer_types = np.unique(payload["jammer_types"][indices])
        if labels.size != 1 or jammer_types.size != 1:
            raise ValueError(f"Inconsistent OFDMA observation {observation_id}")
        output["observation_ids"].append(observation_id)
        output["labels"].append(int(labels[0]))
        output["jammer_types"].append(str(jammer_types[0]))
        for method in methods:
            output[method].append(float(np.max(payload[method][indices])))
    return {
        "observation_ids": np.asarray(output["observation_ids"]),
        "labels": np.asarray(output["labels"], dtype=np.int32),
        "jammer_types": np.asarray(output["jammer_types"]),
        **{
            method: np.asarray(output[method], dtype=np.float32)
            for method in methods
        },
    }


def metric_row(
    *,
    row_type: str,
    dataset: str,
    category: str,
    scene: str,
    jsr: str,
    shot,
    scope: str,
    method: str,
    num_support,
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict:
    binary = np.asarray(labels, dtype=np.int32)
    return {
        "row_type": row_type,
        "dataset": dataset,
        "category": category,
        "scene": scene,
        "jsr": jsr,
        "shot": shot,
        "scope": scope,
        "method": method,
        "method_display": METHOD_DISPLAY[method],
        "num_support": num_support,
        "num_test_normal": int((binary == 0).sum()),
        "num_test_abnormal": int((binary == 1).sum()),
        **safe_metrics(binary, scores),
    }


def path_job_metric_rows(job: dict, payload: dict, args) -> list[dict]:
    rows = []
    dataset = str(job["dataset"])
    shot = job.get("shot", args.normal_sampling)
    if args.protocol != "ofdma":
        for method in args.methods:
            rows.append(
                metric_row(
                    row_type="cell",
                    dataset=dataset,
                    category=str(job["category"]),
                    scene=str(job["scene"]),
                    jsr=str(job["jsr"]),
                    shot=shot,
                    scope="overall",
                    method=method,
                    num_support=len(job["train_samples"]),
                    labels=payload["labels"],
                    scores=payload[method],
                )
            )
        return rows

    from datasets.ofdma_spectrum import JAMMER_TYPES, NO_JAMMER

    jammer_types = payload["jammer_types"]
    normal_name = NO_JAMMER
    anomaly_types = [name for name in JAMMER_TYPES if name in set(map(str, jammer_types))]
    for scope in ("overall", *anomaly_types):
        mask = (
            np.ones(payload["labels"].shape, dtype=bool)
            if scope == "overall"
            else (jammer_types == normal_name) | (jammer_types == scope)
        )
        for method in args.methods:
            rows.append(
                metric_row(
                    row_type="scene",
                    dataset=dataset,
                    category="ofdma",
                    scene=str(job["scene"]),
                    jsr=str(job["jsr"]),
                    shot=int(job["shot"]),
                    scope=scope,
                    method=method,
                    num_support=len(job["train_samples"]),
                    labels=payload["labels"][mask],
                    scores=payload[method][mask],
                )
            )
    return rows


def append_macro_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["shot"], row["scope"], row["method"])].append(row)
    macro = []
    for (dataset, shot, scope, method), selected in sorted(
        groups.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        def finite_mean(key: str) -> float:
            values = np.asarray([row[key] for row in selected], dtype=np.float64)
            finite = values[np.isfinite(values)]
            return float(np.mean(finite)) if finite.size else float("nan")

        macro.append(
            {
                "row_type": "macro",
                "dataset": dataset,
                "category": "ALL",
                "scene": "ALL",
                "jsr": "ALL",
                "shot": shot,
                "scope": scope,
                "method": method,
                "method_display": METHOD_DISPLAY[method],
                "num_support": "",
                "num_test_normal": sum(int(row["num_test_normal"]) for row in selected),
                "num_test_abnormal": sum(int(row["num_test_abnormal"]) for row in selected),
                "auroc": finite_mean("auroc"),
                "auprc": finite_mean("auprc"),
                "fpr95": finite_mean("fpr95"),
            }
        )
    return macro


def fedjam_attack_macro_rows(rows: list[dict]) -> list[dict]:
    """Average the three benign-vs-attack cells for each shot and method."""

    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(int(row["shot"]), str(row["method"]))].append(row)
    output = []
    for (shot, method), selected in sorted(groups.items()):
        output.append(
            {
                "row_type": "macro",
                "dataset": "fedjam",
                "category": "ALL",
                "scene": "official_test",
                "jsr": f"{shot}shot",
                "shot": shot,
                "scope": "macro_attack",
                "method": method,
                "method_display": METHOD_DISPLAY[method],
                "num_support": shot,
                "num_test_normal": sum(int(row["num_test_normal"]) for row in selected),
                "num_test_abnormal": sum(int(row["num_test_abnormal"]) for row in selected),
                **{
                    key: float(np.mean([row[key] for row in selected]))
                    for key in ("auroc", "auprc", "fpr95")
                },
            }
        )
    return output


def _safe_stem(parts: Iterable[object]) -> str:
    raw = "--".join(map(str, parts))
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:150]}--{digest}"


def _array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_rf_jobs(args) -> tuple[list[dict], dict]:
    from tools.eval_cls_aux_cnn_gallery import public_rf_jobs, rf_target_jobs

    builder_args = SimpleNamespace(
        output_root=str(args.output_root),
        normal_sampling=args.normal_sampling,
        seed=args.seed,
        support_seed=args.support_seed,
        support_manifest=args.support_manifest,
        public_rf_signals=tuple(args.public_rf_signals),
    )
    jobs = rf_target_jobs(builder_args) if args.protocol == "rf_target" else public_rf_jobs(builder_args)
    for job in jobs:
        job["test_samples"] = limit_binary_samples(
            job["test_samples"], args.max_test_per_class
        )
    metadata = {
        "normal_sampling": args.normal_sampling,
        "support_manifest": getattr(builder_args, "support_manifest", None),
        "support_manifest_sha256": getattr(builder_args, "support_manifest_sha256", None),
        "test_normal_paths_sha256": getattr(builder_args, "test_normal_paths_sha256", None),
        "support_policy": getattr(builder_args, "support_policy", None),
        "per_frequency_k": getattr(builder_args, "per_frequency_k", None),
    }
    return jobs, metadata


def build_ofdma_protocol(args) -> tuple[list[dict], dict, object]:
    from datasets.ofdma_spectrum import OFDMASpectrogramPreprocessor
    from datasets.ofdma_target_scene import load_target_scene_manifest
    from tools.eval_ofdma_target_scene_baselines import build_jobs, source_protocol

    dataset_root = Path(args.dataset_root)
    source = source_protocol(dataset_root)
    normalization = source["source_protocol"]["normalization"]
    manifest = load_target_scene_manifest(dataset_root)
    jobs, scene_ids = build_jobs(args, manifest, normalization)
    preprocessor = OFDMASpectrogramPreprocessor(
        dataset_root,
        output_size=240,
        geometry="letterbox",
        min_db=float(normalization["min_db"]),
        max_db=float(normalization["max_db"]),
    )
    metadata = {
        "dataset_root": str(dataset_root.resolve()),
        "dataset_protocol": source["source_protocol"]["protocol_name"],
        "split": args.split,
        "target_scene_ids": scene_ids,
        "shots": sorted(set(args.shots)),
        "support_rule": "first k ranked normal observations from the same target scene",
        "test_rule": "same target scene normal_test plus anomaly_test",
        "su_reduction": "maximum score over 21 sensing units",
        "normalization": normalization,
    }
    return jobs, metadata, preprocessor


def _fedjam_image(record) -> np.ndarray:
    if hasattr(record, "image_bgr"):
        return np.asarray(record.image_bgr)
    if isinstance(record, dict) and "image" in record:
        return np.asarray(record["image"])
    raise TypeError("Unsupported FedJam record")


def _fedjam_label(record) -> int:
    return int(record.label if hasattr(record, "label") else record["label"])


def _fedjam_name(record) -> str:
    return str(record.name if hasattr(record, "name") else record["name"])


def evaluate_fedjam(args, output_root: Path) -> tuple[list[dict], list[dict], dict]:
    from tools.eval_fedjam_fewshot_dual import (
        iter_test_records,
        select_benign_support,
    )

    shots = sorted(set(args.shots))
    support, train_counts, benign_seen = select_benign_support(
        Path(args.data_root), max(shots), args.seed
    )
    detector_bank = {
        shot: fit_detectors([_fedjam_image(item) for item in support[:shot]], args.methods)
        for shot in shots
    }
    score_bank = {
        shot: {method: [] for method in args.methods} for shot in shots
    }
    labels: list[int] = []
    names: list[str] = []
    test_counts = {label: 0 for label in FEDJAM_LABEL_NAMES}
    for index, record in enumerate(
        iter_test_records(Path(args.data_root), args.max_test_per_label), 1
    ):
        image = _fedjam_image(record)
        for shot in shots:
            values = score_image(detector_bank[shot], image, time_axis=args.time_axis)
            for method in args.methods:
                score_bank[shot][method].append(values[method])
        labels.append(_fedjam_label(record))
        names.append(_fedjam_name(record))
        test_counts[labels[-1]] = test_counts.get(labels[-1], 0) + 1
        if args.progress_every and index % args.progress_every == 0:
            print(f"  scored {index} FedJam test images", flush=True)

    labels_array = np.asarray(labels, dtype=np.int32)
    if labels_array.size == 0:
        raise RuntimeError("FedJam test selection is empty")
    score_root = output_root / "scores"
    score_root.mkdir(parents=True, exist_ok=True)
    cell_rows = []
    pooled_rows = []
    support_hashes = []
    for item in support:
        support_hashes.append(
            {
                "name": _fedjam_name(item),
                "label": _fedjam_label(item),
                "image_sha256": _array_digest(_fedjam_image(item)),
            }
        )
    for shot in shots:
        arrays = {
            method: np.asarray(score_bank[shot][method], dtype=np.float32)
            for method in args.methods
        }
        np.savez_compressed(
            score_root / f"fedjam_{shot}shot_image_scores.npz",
            names=np.asarray(names),
            labels=labels_array,
            **arrays,
        )
        for attack_id, attack_name in FEDJAM_LABEL_NAMES.items():
            if attack_id == 0:
                continue
            mask = (labels_array == 0) | (labels_array == attack_id)
            binary = (labels_array[mask] == attack_id).astype(np.int32)
            for method in args.methods:
                cell_rows.append(
                    metric_row(
                        row_type="cell",
                        dataset="fedjam",
                        category=attack_name,
                        scene="official_test",
                        jsr=f"{shot}shot",
                        shot=shot,
                        scope=attack_name,
                        method=method,
                        num_support=shot,
                        labels=binary,
                        scores=arrays[method][mask],
                    )
                )
        overall = (labels_array != 0).astype(np.int32)
        for method in args.methods:
            pooled_rows.append(
                metric_row(
                    row_type="pooled",
                    dataset="fedjam",
                    category="ALL",
                    scene="official_test",
                    jsr=f"{shot}shot",
                    shot=shot,
                    scope="overall",
                    method=method,
                    num_support=shot,
                    labels=overall,
                    scores=arrays[method],
                )
            )
    macro_rows = fedjam_attack_macro_rows(cell_rows)
    metadata = {
        "data_root": str(Path(args.data_root).resolve()),
        "shots": shots,
        "support_selection": "seeded reservoir over train label=benign; nested shot prefixes",
        "train_counts": train_counts,
        "benign_seen": benign_seen,
        "support": support_hashes,
        "test_counts": test_counts,
        "test_selection": "official test split, optionally limited per label for smoke tests",
        "pooled_metrics": pooled_rows,
    }
    return cell_rows, macro_rows, metadata


def method_protocol(time_axis: int) -> dict:
    return {
        "reference": REFERENCE,
        "kld_reference": {
            "bins": DEFAULT_KLD_BINS,
            "pseudocount": DEFAULT_PSEUDOCOUNT,
            "divergence": "D(P_query || Q_normal), base 2",
            "value_range": "frozen from normal support only",
        },
        "ica_frozen": {
            "bins": DEFAULT_ICA_BINS,
            "pseudocount": DEFAULT_PSEUDOCOUNT,
            "threshold": f"{DEFAULT_ICA_THRESHOLD_FACTOR} * std(normal information)",
            "cluster_length": DEFAULT_ICA_CLUSTER_LENGTH,
            "time_axis": time_axis,
            "image_reduction": "max continuous excess-information cluster score",
            "reference_update_at_test": False,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from datasets.ofdma_target_scene import DEFAULT_TARGET_SCENE_ROOT
    from tools.eval_cls_aux_cnn_gallery import PUBLIC_RF_JSRS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["rf_target", "public_rf", "ofdma", "fedjam"], required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--max-jobs", type=int, default=0)

    parser.add_argument("--normal-sampling", choices=["per_frequency", "1shot", "2shot", "4shot"], default="per_frequency")
    parser.add_argument("--support-manifest", default="")
    parser.add_argument("--support-seed", type=int, default=111)
    parser.add_argument("--public-rf-signals", nargs="+", choices=list(PUBLIC_RF_JSRS), default=list(PUBLIC_RF_JSRS))
    parser.add_argument("--max-test-per-class", type=int, default=0)

    parser.add_argument("--dataset-root", default=str(DEFAULT_TARGET_SCENE_ROOT))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--scene-ids", nargs="*", default=[])
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--max-normal-observations", type=int, default=0)
    parser.add_argument("--max-anomaly-observations-per-type", type=int, default=0)

    parser.add_argument("--data-root", default="/mnt/data/wangbei/data/FedJam")
    parser.add_argument("--max-test-per-label", type=int, default=0)
    args = parser.parse_args(argv)
    args.methods = validate_methods(args.methods)
    if not args.shots or min(args.shots) <= 0:
        raise ValueError("--shots must contain positive integers")
    for name in (
        "max_jobs",
        "max_test_per_class",
        "max_normal_observations",
        "max_anomaly_observations_per_type",
        "max_test_per_label",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    args.time_axis = 1 if args.protocol == "ofdma" else 0
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    smoke_limits = {
        "max_jobs": args.max_jobs,
        "max_test_per_class": args.max_test_per_class,
        "max_normal_observations": args.max_normal_observations,
        "max_anomaly_observations_per_type": args.max_anomaly_observations_per_type,
        "max_test_per_label": args.max_test_per_label,
    }
    protocol = {
        "entry": "tools/eval_information_theoretic_baselines.py",
        "protocol": args.protocol,
        "methods": list(args.methods),
        "method_config": method_protocol(args.time_axis),
        "cpu_only": True,
        "support_only_fit": True,
        "test_labels_usage": "metrics_only",
        "test_batch_statistics_used": False,
        "seed": args.seed,
        "smoke_limits": smoke_limits,
        "formal_full_test": not any(smoke_limits.values()),
    }

    if args.protocol == "fedjam":
        cell_rows, macro_rows, metadata = evaluate_fedjam(args, output_root)
        protocol["dataset_protocol"] = metadata
    else:
        if args.protocol == "ofdma":
            jobs, metadata, preprocessor = build_ofdma_protocol(args)
        else:
            jobs, metadata = build_rf_jobs(args)
            preprocessor = None
        if args.max_jobs:
            jobs = jobs[: args.max_jobs]
        if not jobs:
            raise RuntimeError("No evaluation jobs selected")
        protocol["dataset_protocol"] = metadata
        protocol["num_jobs"] = len(jobs)
        (output_root / "protocol.json").write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        score_root = output_root / "scores"
        score_root.mkdir(parents=True, exist_ok=True)
        cell_rows = []
        for index, job in enumerate(jobs, 1):
            print(
                f"[{index}/{len(jobs)}] {job['dataset']} {job['category']} "
                f"{job['scene']} {job['jsr']} support={len(job['train_samples'])} "
                f"test={len(job['test_samples'])}",
                flush=True,
            )
            sample_payload, support_payload = evaluate_path_job(job, args, preprocessor)
            stem = _safe_stem(
                (job["dataset"], job["category"], job["scene"], job["jsr"])
            )
            np.savez_compressed(
                score_root / f"{stem}_image_scores.npz",
                **sample_payload,
                **support_payload,
            )
            metric_payload = sample_payload
            if args.protocol == "ofdma":
                metric_payload = aggregate_ofdma(sample_payload, args.methods)
                np.savez_compressed(
                    score_root / f"{stem}_observation_scores.npz",
                    **metric_payload,
                )
            cell_rows.extend(path_job_metric_rows(job, metric_payload, args))
            write_csv(output_root / "metrics_per_cell.csv", cell_rows)
        macro_rows = append_macro_rows(cell_rows)

    protocol["dataset_protocol"] = protocol.get("dataset_protocol", {})
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_root / "metrics_per_cell.csv", cell_rows)
    write_csv(output_root / "metrics_macro.csv", macro_rows)
    summary = {
        "status": "complete",
        "protocol": args.protocol,
        "formal_full_test": protocol["formal_full_test"],
        "methods": list(args.methods),
        "num_cell_rows": len(cell_rows),
        "macro": macro_rows,
    }
    if args.protocol == "fedjam":
        summary["pooled"] = protocol["dataset_protocol"]["pooled_metrics"]
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"[done] {output_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
