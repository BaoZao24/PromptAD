#!/usr/bin/env python
"""Few-shot, normal-only GRETEL spectrogram adaptation evaluator.

The original paper trains from IQ-derived PSD maps. This entry keeps the
current strict protocol instead: every teacher, student and memory matrix is
fitted only on the normal support selected by the existing formal manifests.
Test labels are read only after test images have been scored.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.ofdma_spectrum import JAMMER_TYPES, NO_JAMMER, NUM_SUS, OFDMASpectrogramPreprocessor
from datasets.ofdma_target_scene import DEFAULT_TARGET_SCENE_ROOT, available_target_scenes, build_target_scene_records, load_target_scene_manifest
from datasets.rf_target import RF_JSR_BY_SIGNAL, RF_SCENES
from tools.eval_fedjam_fewshot_dual import LABEL_NAMES, RawRecord, iter_test_records, select_benign_support
from tools.eval_patchcore_cls import public_rf_jobs, rf_target_jobs
from tools.gretel_spectral import GretelConfig, fit_gretel, image_to_power_map, score_gretel_maps
from utils.rf_frequency_sampling import NORMAL_SAMPLING_CHOICES
from utils.training_utils import setup_seed


METHOD_NAME = "gretel_spectrogram_adaptation"
FORMAL_CONFIG = GretelConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("rf_target", "public_rf", "ofdma", "fedjam"), required=True)
    parser.add_argument("--output-root", default="analysis_outputs/20260826_gretel_spectral")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=111, help="Shared support-manifest seed.")
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size only.")
    parser.add_argument("--validate-only", action="store_true")
    # Existing RF/Public RF protocol fields.
    parser.add_argument("--normal-sampling", choices=NORMAL_SAMPLING_CHOICES, default="per_frequency")
    parser.add_argument("--support-manifest", default="")
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--rf-signals", nargs="+", default=list(RF_JSR_BY_SIGNAL), choices=list(RF_JSR_BY_SIGNAL))
    parser.add_argument("--rf-scenes", nargs="+", default=list(RF_SCENES), choices=list(RF_SCENES))
    parser.add_argument("--public-rf-signals", nargs="+", default=["burst", "chirp", "dsss", "pulse", "deceptive"])
    parser.add_argument("--support-bootstrap-seed", type=int, default=-1)
    # Target-scene OFDMA protocol fields.
    parser.add_argument("--dataset-root", default=str(DEFAULT_TARGET_SCENE_ROOT))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--scene-ids", nargs="*", default=[])
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-normal-observations", type=int, default=0)
    parser.add_argument("--max-anomaly-observations-per-type", type=int, default=0)
    # FedJam protocol fields.
    parser.add_argument("--fedjam-root", default="/mnt/data/wangbei/data/FedJam")
    parser.add_argument("--fedjam-shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-test-per-label", type=int, default=0)
    return parser.parse_args()


def binary_metrics(labels, scores) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "fpr95": float("nan")}
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels, scores) * 100.0),
        "auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if reached.size else float("nan"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def path_image(sample: dict) -> np.ndarray:
    image = cv2.imread(str(sample["path"]), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(sample["path"])
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def maps_from_images(images: Iterable[np.ndarray], config: GretelConfig) -> np.ndarray:
    maps = [image_to_power_map(image, config) for image in images]
    if not maps:
        raise ValueError("No images available")
    return np.stack(maps, axis=0).astype(np.float32, copy=False)


def generic_jobs(args: argparse.Namespace) -> list[dict]:
    if args.protocol == "public_rf":
        return public_rf_jobs(args)
    if args.protocol == "rf_target":
        return rf_target_jobs(args)
    raise ValueError(args.protocol)


def record_effective_generic_protocol(root: Path, args: argparse.Namespace) -> None:
    """Persist the manifest identifiers populated by the shared job builder."""

    path = root / "protocol.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["support_manifest"] = getattr(args, "support_manifest", "") or None
    payload["support_manifest_sha256"] = getattr(args, "support_manifest_sha256", None)
    payload["test_normal_paths_sha256"] = getattr(args, "test_normal_paths_sha256", None)
    payload["support_policy"] = getattr(args, "support_policy", None)
    payload["per_frequency_k"] = getattr(args, "per_frequency_k", None)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generic_support_maps(job: dict, config: GretelConfig) -> np.ndarray:
    return maps_from_images((path_image(sample) for sample in job["train_samples"]), config)


def generic_score_job(model, job: dict, config: GretelConfig, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    labels, names = [], []
    samples = job["test_samples"]
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        values = score_gretel_maps(model, maps_from_images((path_image(sample) for sample in batch), config), config, device)
        for key, value in values.items():
            parts[key].append(value)
        labels.extend(int(sample["label"]) for sample in batch)
        names.extend(str(sample["name"]) for sample in batch)
    return {**{key: np.concatenate(value) for key, value in parts.items()}, "labels": np.asarray(labels, dtype=np.int32), "names": np.asarray(names)}


def target_scene_jobs(args: argparse.Namespace) -> tuple[list[dict], dict]:
    if not set(args.shots).issubset({1, 2, 4}):
        raise ValueError("Target-scene OFDMA --shots must be selected from 1, 2, 4")
    root = Path(args.dataset_root)
    manifest = load_target_scene_manifest(root)
    source = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    normalization = source["source_protocol"]["normalization"]
    available = available_target_scenes(manifest, args.split)
    scene_ids = list(args.scene_ids) if args.scene_ids else available
    unknown = sorted(set(scene_ids) - set(available))
    if unknown:
        raise ValueError(f"Unknown {args.split} target scenes: {unknown}")
    jobs = []
    for shot in sorted(set(args.shots)):
        for scene_id in scene_ids:
            support = build_target_scene_records(root, manifest, scene_id, split=args.split, role="support", shot=shot)
            test = build_target_scene_records(root, manifest, scene_id, split=args.split, role="test", max_normal_observations=args.max_normal_observations, max_anomaly_observations_per_type=args.max_anomaly_observations_per_type)
            jobs.append({"dataset": "ofdma", "target_scene_id": scene_id, "shot": int(shot), "support_records": support, "test_records": test})
    return jobs, {"root": root, "normalization": normalization, "scene_ids": scene_ids}


def target_preprocessor(context: dict) -> OFDMASpectrogramPreprocessor:
    normalization = context["normalization"]
    return OFDMASpectrogramPreprocessor(context["root"], output_size=240, geometry="letterbox", min_db=float(normalization["min_db"]), max_db=float(normalization["max_db"]))


def target_record_image(record, preprocessor: OFDMASpectrogramPreprocessor) -> np.ndarray:
    image = cv2.imread(str(record.image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(record.image_path)
    return preprocessor(image)


def target_support_maps(job: dict, preprocessor: OFDMASpectrogramPreprocessor, config: GretelConfig) -> np.ndarray:
    return maps_from_images((target_record_image(record, preprocessor) for record in job["support_records"]), config)


def target_score_job(model, job: dict, preprocessor, config, device, batch_size) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    labels, names, jammer_types = [], [], []
    records = job["test_records"]
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        values = score_gretel_maps(model, maps_from_images((target_record_image(record, preprocessor) for record in batch), config), config, device)
        for key, value in values.items():
            parts[key].append(value)
        labels.extend(int(record.label) for record in batch)
        names.extend(f"{record.target_scene_id}::{record.observation_id}::su{record.su_id:02d}" for record in batch)
        jammer_types.extend(str(record.jammer_type) for record in batch)
    return {**{key: np.concatenate(value) for key, value in parts.items()}, "labels": np.asarray(labels, dtype=np.int32), "names": np.asarray(names), "jammer_types": np.asarray(jammer_types)}


def cache_target_test_maps(records, preprocessor, config: GretelConfig, batch_size: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Decode one target scene's fixed test frames exactly once.

    OFDMA uses the same test observations for 1/2/4-shot and all model seeds.
    Keeping this scene-local cache avoids repeated disk decoding without
    exposing a test image to fitting or changing any score calculation.
    """

    maps, labels, names, jammer_types = [], [], [], []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        maps.append(maps_from_images((target_record_image(record, preprocessor) for record in batch), config))
        labels.extend(int(record.label) for record in batch)
        names.extend(f"{record.target_scene_id}::{record.observation_id}::su{record.su_id:02d}" for record in batch)
        jammer_types.extend(str(record.jammer_type) for record in batch)
    return np.concatenate(maps, axis=0), {"labels": np.asarray(labels, dtype=np.int32), "names": np.asarray(names), "jammer_types": np.asarray(jammer_types)}


def score_cached_target_maps(model, maps: np.ndarray, metadata: dict[str, np.ndarray], config, device, batch_size) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    for start in range(0, len(maps), batch_size):
        values = score_gretel_maps(model, maps[start : start + batch_size], config, device)
        for key, value in values.items():
            parts[key].append(value)
    return {**{key: np.concatenate(value) for key, value in parts.items()}, **metadata}


def aggregate_ofdma(payload: dict) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(payload["names"]):
        target_scene, observation, _su = str(name).split("::")
        groups[f"{target_scene}::{observation}"].append(index)
    output: dict[str, list] = {"names": [], "labels": [], "jammer_types": []}
    for key in ("score", "reconstruction", "embedding", "attention"):
        output[key] = []
    for observation, indices in sorted(groups.items()):
        if len(indices) != NUM_SUS:
            raise ValueError(f"{observation} has {len(indices)} SUs, expected {NUM_SUS}")
        index_array = np.asarray(indices, dtype=np.int64)
        labels = np.unique(payload["labels"][index_array])
        jammer_types = np.unique(payload["jammer_types"][index_array])
        if len(labels) != 1 or len(jammer_types) != 1:
            raise ValueError(f"Inconsistent OFDMA observation {observation}")
        output["names"].append(observation)
        output["labels"].append(int(labels[0]))
        output["jammer_types"].append(str(jammer_types[0]))
        for key in ("score", "reconstruction", "embedding", "attention"):
            output[key].append(float(np.max(payload[key][index_array])))
    return {
        key: np.asarray(value, dtype=np.int32 if key == "labels" else np.float32 if key not in {"names", "jammer_types"} else str)
        for key, value in output.items()
    }


def fedjam_support_maps(records: list[RawRecord], config: GretelConfig) -> np.ndarray:
    return maps_from_images((record.image_bgr for record in records), config)


def fedjam_score_job(model, args, config, device) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    labels, names, pending = [], [], []
    def score_pending() -> None:
        if not pending:
            return
        values = score_gretel_maps(model, maps_from_images((record.image_bgr for record in pending), config), config, device)
        for key, value in values.items():
            parts[key].append(value)
        labels.extend(int(record.label) for record in pending)
        names.extend(str(record.name) for record in pending)
        pending.clear()
    for record in iter_test_records(Path(args.fedjam_root), args.max_test_per_label):
        pending.append(record)
        if len(pending) >= args.batch_size:
            score_pending()
    score_pending()
    return {**{key: np.concatenate(value) for key, value in parts.items()}, "labels": np.asarray(labels, dtype=np.int32), "names": np.asarray(names)}


def support_digest(names: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(str(name) for name in names).encode("utf-8")).hexdigest()


def save_scores(root: Path, stem: str, payload: dict, model_seed: int, digest: str) -> None:
    score_root = root / "scores"
    score_root.mkdir(parents=True, exist_ok=True)
    saved = {key: np.asarray(value) for key, value in payload.items()}
    saved["model_seed"] = np.asarray(int(model_seed), dtype=np.int32)
    saved["support_digest"] = np.asarray(digest)
    np.savez_compressed(score_root / f"{stem}-seed{model_seed}.npz", **saved)


def base_row(dataset: str, category: str, scene: str, jsr: str, shot, model_seed: int, ntrain: int, labels, losses: dict[str, float], score) -> dict:
    labels = np.asarray(labels, dtype=np.int32)
    return {
        "method": METHOD_NAME, "dataset": dataset, "category": category, "scene": scene,
        "jsr": jsr, "shot": shot, "model_seed": int(model_seed), "num_train_normal": int(ntrain),
        "num_test_normal": int((labels == 0).sum()), "num_test_abnormal": int((labels == 1).sum()),
        **binary_metrics(labels, score), **losses,
    }


def run_generic(args: argparse.Namespace, root: Path, device: torch.device, config: GretelConfig) -> list[dict]:
    jobs = generic_jobs(args)
    record_effective_generic_protocol(root, args)
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for job in jobs:
        groups[tuple(str(sample["path"]) for sample in job["train_samples"])].append(job)
    rows = []
    for group_index, grouped in enumerate(groups.values(), 1):
        support = generic_support_maps(grouped[0], config)
        digest = support_digest(str(sample["path"]) for sample in grouped[0]["train_samples"])
        print(f"[fit {group_index}/{len(groups)}] support={len(support)} jobs={len(grouped)}", flush=True)
        for model_seed in args.model_seeds:
            model, losses = fit_gretel(support, config, device, model_seed)
            for job in grouped:
                payload = generic_score_job(model, job, config, device, args.batch_size)
                stem = f"{job['dataset']}-{job['category']}-{job['scene']}-{job['jsr']}".replace("/", "_")
                save_scores(root, stem, payload, model_seed, digest)
                shot = job.get("per_frequency_k") or (args.normal_sampling if job["dataset"] == "public_rf" else "")
                row = {"row_type": "cell", "scope": "overall", **base_row(job["dataset"], job["category"], job["scene"], job["jsr"], shot, model_seed, len(job["train_samples"]), payload["labels"], losses, payload["score"])}
                rows.append(row)
                print(f"  {job['category']}/{job['scene']}/{job['jsr']} seed={model_seed} AUROC={row['auroc']:.2f}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def run_ofdma(args: argparse.Namespace, root: Path, device: torch.device, config: GretelConfig) -> list[dict]:
    jobs, context = target_scene_jobs(args)
    preprocessor = target_preprocessor(context)
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.update({
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "target_scene_ids": context["scene_ids"],
        "shots": sorted({int(job["shot"]) for job in jobs}),
        "support_rule": "first k ranked normal observations from the same target scene",
        "test_rule": "same target scene normal_test plus anomaly_test",
        "su_reduction": "maximum score over 21 sensing units",
        "source_normalization": context["normalization"],
        "test_cache": "scene-local preprocessed maps; used only for repeated inference",
    })
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for job in jobs:
        by_scene[job["target_scene_id"]].append(job)
    ordered_scenes = sorted(by_scene)
    fit_index = 0
    for scene_index, scene_id in enumerate(ordered_scenes, 1):
        scene_jobs = sorted(by_scene[scene_id], key=lambda item: item["shot"])
        test_maps, test_metadata = cache_target_test_maps(scene_jobs[0]["test_records"], preprocessor, config, args.batch_size)
        print(f"[cache {scene_index}/{len(ordered_scenes)}] OFDMA {scene_id} test_frames={len(test_maps)}", flush=True)
        for job in scene_jobs:
            fit_index += 1
            support = target_support_maps(job, preprocessor, config)
            digest = support_digest(str(record.image_path) for record in job["support_records"])
            print(f"[fit {fit_index}/{len(jobs)}] OFDMA {job['target_scene_id']} {job['shot']}-shot frames={len(support)}", flush=True)
            for model_seed in args.model_seeds:
                model, losses = fit_gretel(support, config, device, model_seed)
                payload = aggregate_ofdma(score_cached_target_maps(model, test_maps, test_metadata, config, device, args.batch_size))
                save_scores(root, f"ofdma-{job['target_scene_id']}-{job['shot']}shot", payload, model_seed, digest)
                for scope in ("overall", *JAMMER_TYPES):
                    mask = np.ones(len(payload["labels"]), dtype=bool) if scope == "overall" else ((payload["jammer_types"] == NO_JAMMER) | (payload["jammer_types"] == scope))
                    rows.append({"row_type": "target_scene", "scope": scope, **base_row("ofdma", "ofdma", job["target_scene_id"], f"{job['shot']}shot", job["shot"], model_seed, len(job["support_records"]) // NUM_SUS, payload["labels"][mask], losses, payload["score"][mask])})
                print(f"  seed={model_seed} AUROC={rows[-len(JAMMER_TYPES)-1]['auroc']:.2f}", flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del test_maps
        gc.collect()
    return rows


def run_fedjam(args: argparse.Namespace, root: Path, device: torch.device, config: GretelConfig) -> list[dict]:
    shots = sorted(set(int(value) for value in args.fedjam_shots))
    if not shots or not set(shots).issubset({1, 2, 4}):
        raise ValueError("FedJam --fedjam-shots must be selected from 1, 2, 4")
    pool, _counts, benign_seen = select_benign_support(Path(args.fedjam_root), max(shots), args.seed)
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["fedjam_support_rule"] = "deterministic benign-only reservoir sample from train"
    protocol["fedjam_benign_seen"] = int(benign_seen)
    protocol["fedjam_support_names"] = [str(record.name) for record in pool]
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for shot in shots:
        support = fedjam_support_maps(pool[:shot], config)
        digest = support_digest(record.name for record in pool[:shot])
        print(f"[fit] FedJam {shot}-shot support={len(support)} benign_pool={benign_seen}", flush=True)
        for model_seed in args.model_seeds:
            model, losses = fit_gretel(support, config, device, model_seed)
            payload = fedjam_score_job(model, args, config, device)
            save_scores(root, f"fedjam-{shot}shot", payload, model_seed, digest)
            labels = payload["labels"]
            scopes = [("overall", np.ones(len(labels), dtype=bool), (labels != 0).astype(np.int32))]
            attack_values = []
            for label_id, label_name in LABEL_NAMES.items():
                if label_id:
                    mask = (labels == 0) | (labels == label_id)
                    binary = (labels[mask] == label_id).astype(np.int32)
                    scopes.append((label_name, mask, binary))
                    attack_values.append(binary_metrics(binary, payload["score"][mask]))
            for scope, mask, binary in scopes:
                rows.append({"row_type": "fedjam", "scope": scope, **base_row("fedjam", "fedjam", "full_test", f"{shot}shot", shot, model_seed, shot, binary, losses, payload["score"][mask])})
            rows.append({"row_type": "fedjam", "scope": "macro_attack", "method": METHOD_NAME, "dataset": "fedjam", "category": "fedjam", "scene": "full_test", "jsr": f"{shot}shot", "shot": shot, "model_seed": int(model_seed), "num_train_normal": shot, "num_test_normal": int((labels == 0).sum()), "num_test_abnormal": int((labels != 0).sum()), **{key: float(np.nanmean([value[key] for value in attack_values])) for key in ("auroc", "auprc", "fpr95")}, **losses})
            print(f"  seed={model_seed} AUROC={rows[-5]['auroc']:.2f}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def append_macro_rows(rows: list[dict]) -> list[dict]:
    if not rows or rows[0]["dataset"] == "fedjam":
        return rows
    # Public/In-house RF need one macro value over every formal cell, rather
    # than one value per JSR. OFDMA is separately grouped by shot and scope.
    keys = sorted({(row["dataset"], str(row["shot"]), row.get("scope", "overall"), row["model_seed"]) for row in rows if row["row_type"] in {"cell", "target_scene"}})
    macro = []
    for dataset, shot, scope, model_seed in keys:
        selected = [row for row in rows if row["dataset"] == dataset and str(row["shot"]) == shot and row.get("scope", "overall") == scope and row["model_seed"] == model_seed and row["row_type"] in {"cell", "target_scene"}]
        macro.append({"row_type": "macro", "scope": scope, "method": METHOD_NAME, "dataset": dataset, "category": "average", "scene": "macro", "jsr": "macro", "shot": selected[0]["shot"], "model_seed": model_seed, "num_train_normal": "", "num_test_normal": int(sum(row["num_test_normal"] for row in selected)), "num_test_abnormal": int(sum(row["num_test_abnormal"] for row in selected)), **{key: float(np.nanmean([row[key] for row in selected])) for key in ("auroc", "auprc", "fpr95")}, **{key: float(np.nanmean([row[key] for row in selected])) for key in selected[0] if key.startswith("teacher_") or key.startswith("student_")}})
    return rows + macro


def summarize(rows: list[dict]) -> list[dict]:
    candidates = [row for row in rows if row["row_type"] == "macro" or (row["dataset"] == "fedjam" and row.get("scope") in {"overall", "macro_attack"})]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in candidates:
        grouped[(row["dataset"], str(row["shot"]), row.get("scope", "overall"), row["row_type"])].append(row)
    return [{"dataset": dataset, "shot_or_protocol": jsr, "scope": scope, "source_row_type": row_type, "num_model_seeds": len(values), **{f"{metric}_mean": float(np.nanmean([row[metric] for row in values])) for metric in ("auroc", "auprc", "fpr95")}, **{f"{metric}_std": float(np.nanstd([row[metric] for row in values], ddof=0)) for metric in ("auroc", "auprc", "fpr95")}} for (dataset, jsr, scope, row_type), values in sorted(grouped.items())]


def protocol_payload(args: argparse.Namespace, config: GretelConfig) -> dict:
    return {
        "entry": "tools/eval_gretel_spectral.py", "method": METHOD_NAME,
        "method_display": "GRETEL (spectrogram adaptation; few-shot normal-only)",
        "paper": "Hussain et al., GRETEL, IEEE IoT Journal, 2026, doi:10.1109/JIOT.2026.3702837",
        "reproduction_status": "paper-based spectrogram adaptation; not raw-IQ GRETEL reproduction",
        "support_only": True, "test_label_usage": "metrics_only", "test_batch_statistics": "not_used",
        "input_adaptation": {"input": "existing PNG/dB-normalized spectrogram, not raw IQ", "time_frequency_size": [config.time_bins, config.frequency_bins], "frequency_axis": "horizontal; current formal evaluator convention", "nodes": config.frequency_nodes, "edges": "self plus immediate frequency neighbours", "node_features": ["band_mean_power", "band_power_std", "relative_band_contrast", "sinusoidal_frequency_position"], "snr_note": "physical SNR is unavailable; relative_band_contrast is an image-domain proxy"},
        "model_config": config.serializable(), "score": "unweighted reconstruction + teacher/student embedding disagreement + attention disagreement", "model_seeds": [int(seed) for seed in args.model_seeds], "support_seed": int(args.seed),
    }


def validate(args: argparse.Namespace, config: GretelConfig) -> dict:
    if args.protocol in {"rf_target", "public_rf"}:
        jobs = generic_jobs(args)
        record_effective_generic_protocol(Path(args.output_root), args)
        if not jobs:
            raise RuntimeError("No RF jobs selected")
        return {"num_jobs": len(jobs), "probe_support_maps": list(generic_support_maps(jobs[0], config).shape)}
    if args.protocol == "ofdma":
        jobs, context = target_scene_jobs(args)
        if not jobs:
            raise RuntimeError("No OFDMA jobs selected")
        return {"num_jobs": len(jobs), "probe_support_maps": list(target_support_maps(jobs[0], target_preprocessor(context), config).shape), "scenes": context["scene_ids"]}
    pool, _counts, benign_seen = select_benign_support(Path(args.fedjam_root), max(args.fedjam_shots), args.seed)
    return {"num_jobs": len(set(args.fedjam_shots)), "probe_support_maps": list(fedjam_support_maps(pool[:1], config).shape), "benign_support_pool": benign_seen}


def main() -> None:
    args = parse_args()
    if not args.model_seeds:
        raise ValueError("At least one --model-seeds value is required")
    if not args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    setup_seed(args.seed)
    config = FORMAL_CONFIG
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload(args, config)
    protocol["requested_protocol"] = args.protocol
    protocol["smoke_limits"] = {"max_test_normals": args.max_test_normals, "max_abnormals": args.max_abnormals, "max_normal_observations": args.max_normal_observations, "max_anomaly_observations_per_type": args.max_anomaly_observations_per_type, "max_test_per_label": args.max_test_per_label}
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.validate_only:
        print(json.dumps(validate(args, config), indent=2), flush=True)
        print("[validated] GRETEL support-only protocol", flush=True)
        return
    if not args.use_cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; rerun with --use-cpu for a smoke test")
    device = torch.device("cpu" if args.use_cpu else "cuda:0")
    if args.protocol in {"rf_target", "public_rf"}:
        rows = run_generic(args, root, device, config)
    elif args.protocol == "ofdma":
        rows = run_ofdma(args, root, device, config)
    else:
        rows = run_fedjam(args, root, device, config)
    rows = append_macro_rows(rows)
    write_csv(root / "results_gretel_spectral.csv", rows)
    summary_rows = summarize(rows)
    write_csv(root / "summary_gretel_spectral.csv", summary_rows)
    (root / "README.md").write_text("# GRETEL Spectrogram Adaptation\n\nThis is a support-only, few-shot spectrogram adaptation of GRETEL; it is not an IQ-domain reproduction.\n\n- `protocol.json`: frozen model and protocol settings\n- `results_gretel_spectral.csv`: per-cell/per-scene metrics for every model seed\n- `summary_gretel_spectral.csv`: mean and standard deviation over model seeds\n- `scores/`: per-sample scores and score components\n", encoding="utf-8")
    print(f"wrote {root / 'results_gretel_spectral.csv'}", flush=True)
    print(json.dumps(summary_rows, indent=2), flush=True)
    gc.collect()


if __name__ == "__main__":
    main()
