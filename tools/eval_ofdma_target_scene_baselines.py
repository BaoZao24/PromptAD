#!/usr/bin/env python
"""Evaluate comparison methods on the formal target-scene OFDMA protocol.

The entry intentionally fixes method hyperparameters to the configurations
already used by the project's baseline evaluators.  Its public options only
select protocol cells and execution resources.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.ofdma_spectrum import (  # noqa: E402
    JAMMER_TYPES,
    NO_JAMMER,
    NUM_SUS,
    OFDMASpectrogramPreprocessor,
)
from datasets.ofdma_target_scene import (  # noqa: E402
    DEFAULT_TARGET_SCENE_ROOT,
    TargetSceneOFDMAPathDataset,
    available_target_scenes,
    build_target_scene_records,
    load_target_scene_manifest,
    parse_target_scene_name,
)
from tools.eval_cls_vit_patch_gallery import harmonic  # noqa: E402
from tools.eval_seg_resnet_gallery_fusion import load_checkpoint  # noqa: E402
from train_rf_target_pooled_universal import (  # noqa: E402
    build_gallery,
    to_model_input,
)
from utils.training_utils import setup_seed  # noqa: E402


METHOD_NAMES = {
    "promptad": "promptad_text_vit_harmonic",
    "vae": "vae_reconstruction",
    "iad_per": "iad_per",
    "udma": "udma_reimplementation",
    "saife": "saife_reconstruction",
    "deep_svdd": "deep_svdd",
    "padim": "padim_diag_resnet18",
    "stfpm": "stfpm_resnet18",
    "winclip": "winclip_fewshot",
    "patchcore": "patchcore_official",
}
ALLOWED_SHOTS = {1, 2, 4}

PROMPTAD_CONFIG = {
    "resolution": 240,
    "img_resize": 240,
    "img_cropsize": 240,
    "batch_size": 128,
    "backbone": "ViT-B-16-plus-240",
    "pretrained_dataset": "laion400m_e32",
    "prompt_mode": "legacy",
    "input_mode": "rgb",
    "text_prototype_mode": "single",
    "cls_score_mode": "text_only",
    "n_ctx": 4,
    "n_ctx_ab": 1,
    "n_pro": 3,
    "n_pro_ab": 4,
}

METHOD_CONFIGS = {
    "vae": {
        "image_size": 64,
        "base_channels": 16,
        "latent_dim": 64,
        "fc_dim": 1024,
        "decoder_fc_dim": 512,
        "epochs": 100,
        "batch_size": 128,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "score_mode": "mse_mean",
        "log_every": 25,
    },
    "iad_per": {
        "image_size": 64,
        "base_channels": 16,
        "latent_dim": 64,
        "fc_dim": 1024,
        "decoder_fc_dim": 512,
        "epochs": 100,
        "batch_size": 128,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "score_mode": "iad_per",
        "per_alpha": 0.05,
        "per_gamma": 3,
        "per_background_weight": 2.0,
        "per_background_percentile": 90.0,
        "per_signal_percentile": 99.0,
        "log_every": 25,
    },
    "udma": {
        "image_height": 64,
        "image_width": 64,
        "batch_size": 32,
        "reference_extractor": "resnet18_imagenet",
        "teacher_epochs": 5,
        "student_epochs": 10,
        "teacher_lr": 1e-3,
        "student_lr": 1e-3,
        "weight_decay": 0.0,
        "log_every": 5,
    },
    "saife": {
        "epochs": 100,
        "batch_size": 32,
        "test_batch_size": 256,
        "latent_dim": 50,
        "hidden_dim": 256,
        "dropout": 0.1,
        "lr": 5e-5,
        "discriminator_lr": 2.5e-5,
        "adversarial_weight": 1.0,
        "weight_decay": 0.0,
        "log_every": 25,
    },
    "deep_svdd": {
        "image_size": 64,
        "base_channels": 32,
        "rep_dim": 128,
        "epochs": 100,
        "batch_size": 128,
        "lr": 1e-3,
        "weight_decay": 1e-6,
        "center_eps": 0.1,
        "log_every": 25,
    },
    "padim": {
        "resize": 256,
        "imagesize": 224,
        "batch_size": 64,
        "weights": "default",
        "embedding_dim": 128,
        "var_eps": 0.01,
    },
    "stfpm": {
        "resize": 256,
        "imagesize": 224,
        "batch_size": 64,
        "epochs": 50,
        "lr": 0.05,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "log_every": 25,
    },
    "winclip": {
        "batch_size": 32,
        "gallery_batch_size": 16,
        "img_resize": 240,
        "img_cropsize": 240,
        "resolution": 240,
        "scales": [2, 3],
        "backbone": "ViT-B-16-plus-240",
        "pretrained_dataset": "laion400m_e32",
        "prompt_class": "radio frequency spectrogram",
    },
    "patchcore": {
        "resize": 256,
        "imagesize": 224,
        "batch_size": 32,
        "backbone": "wideresnet50",
        "layers": ["layer2", "layer3"],
        "pretrain_embed_dimension": 1024,
        "target_embed_dimension": 1024,
        "patchsize": 3,
        "anomaly_scorer_num_nn": 1,
        "sampler": "random",
        "coreset_percentage": 0.1,
        "faiss_num_workers": 8,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formal comparison methods on target-scene OFDMA cold start"
    )
    parser.add_argument("--method", choices=sorted(METHOD_NAMES), required=True)
    parser.add_argument("--dataset-root", default=str(DEFAULT_TARGET_SCENE_ROOT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--scene-ids", nargs="*", default=[])
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--baseline-batch-size",
        type=int,
        default=0,
        help="Optional batch size override for the selected baseline; 0 keeps the frozen config.",
    )
    parser.add_argument(
        "--baseline-faiss-workers",
        type=int,
        default=0,
        help="Optional PatchCore FAISS worker override; 0 keeps the frozen config.",
    )
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument(
        "--checkpoint",
        default=(
            "analysis_outputs/02_current_baselines/promptad_formal_baseline/"
            "pooled_rf_rgb_cls/checkpoint/overall-best.pt"
        ),
    )
    parser.add_argument("--max-normal-observations", type=int, default=0)
    parser.add_argument(
        "--max-anomaly-observations-per-type", type=int, default=0
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def source_protocol(dataset_root: Path) -> dict:
    path = dataset_root / "protocol.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def record_name(record) -> str:
    return (
        f"{record.target_scene_id}::{record.observation_id}::su{record.su_id:02d}"
    )


def record_sample(record, dataset_root: Path, normalization: dict) -> dict:
    return {
        "path": record.image_path,
        "label": int(record.label),
        "category": "ofdma",
        "name": record_name(record),
        "dataset": "ofdma",
        "scene_id": record.target_scene_id,
        "observation_id": record.observation_id,
        "su_id": int(record.su_id),
        "jammer_type": str(record.jammer_type),
        "ofdma_root": str(dataset_root.resolve()),
        "ofdma_preprocess_size": 240,
        "ofdma_geometry": "letterbox",
        "ofdma_min_db": float(normalization["min_db"]),
        "ofdma_max_db": float(normalization["max_db"]),
    }


def build_jobs(args, manifest, normalization) -> tuple[list[dict], list[str]]:
    dataset_root = Path(args.dataset_root)
    available = available_target_scenes(manifest, args.split)
    scene_ids = list(args.scene_ids) if args.scene_ids else available
    unknown = sorted(set(scene_ids) - set(available))
    if unknown:
        raise ValueError(f"Unknown {args.split} target scenes: {unknown}")

    jobs = []
    for shot in sorted(set(args.shots)):
        for scene_id in scene_ids:
            support_records = build_target_scene_records(
                dataset_root,
                manifest,
                scene_id,
                split=args.split,
                role="support",
                shot=shot,
            )
            test_records = build_target_scene_records(
                dataset_root,
                manifest,
                scene_id,
                split=args.split,
                role="test",
                max_normal_observations=args.max_normal_observations,
                max_anomaly_observations_per_type=(
                    args.max_anomaly_observations_per_type
                ),
            )
            jobs.append(
                {
                    "dataset": "ofdma",
                    "category": "ofdma",
                    "scene": scene_id,
                    "jsr": f"{shot}shot",
                    "shot": shot,
                    "target_scene_id": scene_id,
                    "train_samples": [
                        record_sample(record, dataset_root, normalization)
                        for record in support_records
                    ],
                    "test_samples": [
                        record_sample(record, dataset_root, normalization)
                        for record in test_records
                    ],
                    "support_records": support_records,
                    "test_records": test_records,
                }
            )
    return jobs, scene_ids


def method_args(args, normalization) -> SimpleNamespace:
    values = {
        "protocol": "ofdma",
        "output_root": args.output_root,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "ofdma_root": args.dataset_root,
        "ofdma_min_db": float(normalization["min_db"]),
        "ofdma_max_db": float(normalization["max_db"]),
    }
    values.update(METHOD_CONFIGS.get(args.method, {}))
    values.update(PROMPTAD_CONFIG if args.method == "promptad" else {})
    if args.baseline_batch_size > 0 and "batch_size" in values:
        values["batch_size"] = int(args.baseline_batch_size)
    if args.baseline_faiss_workers > 0 and "faiss_num_workers" in values:
        values["faiss_num_workers"] = int(args.baseline_faiss_workers)
    return SimpleNamespace(**values)


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


def aggregate_observations(payload: dict) -> dict:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, name in enumerate(payload["names"]):
        scene_id, observation_id, _ = parse_target_scene_name(str(name))
        groups[(scene_id, observation_id)].append(index)

    out = {
        "target_scene_ids": [],
        "observation_ids": [],
        "labels": [],
        "jammer_types": [],
        "scores": [],
    }
    for (scene_id, observation_id), indices_list in sorted(groups.items()):
        indices = np.asarray(indices_list, dtype=np.int64)
        if len(indices) != NUM_SUS:
            raise ValueError(
                f"{scene_id}/{observation_id} has {len(indices)} SUs, expected {NUM_SUS}"
            )
        labels = np.unique(payload["labels"][indices])
        jammer_types = np.unique(payload["jammer_types"][indices])
        if len(labels) != 1 or len(jammer_types) != 1:
            raise ValueError(f"Inconsistent observation {scene_id}/{observation_id}")
        out["target_scene_ids"].append(scene_id)
        out["observation_ids"].append(observation_id)
        out["labels"].append(int(labels[0]))
        out["jammer_types"].append(str(jammer_types[0]))
        out["scores"].append(float(np.max(payload["scores"][indices])))
    return {
        "target_scene_ids": np.asarray(out["target_scene_ids"]),
        "observation_ids": np.asarray(out["observation_ids"]),
        "labels": np.asarray(out["labels"], dtype=np.int32),
        "jammer_types": np.asarray(out["jammer_types"]),
        "scores": np.asarray(out["scores"], dtype=np.float32),
    }


def metric_rows(job, observation_scores, method_name) -> list[dict]:
    labels = observation_scores["labels"]
    jammer_types = observation_scores["jammer_types"]
    rows = []
    for scope in ("overall", *JAMMER_TYPES):
        mask = (
            np.ones(len(labels), dtype=bool)
            if scope == "overall"
            else (jammer_types == NO_JAMMER) | (jammer_types == scope)
        )
        rows.append(
            {
                "row_type": "target_scene",
                "target_scene_id": job["target_scene_id"],
                "shot": int(job["shot"]),
                "scope": scope,
                "method": method_name,
                "num_observations": int(mask.sum()),
                **safe_metrics(labels[mask], observation_scores["scores"][mask]),
            }
        )
    return rows


def append_macro_rows(rows: list[dict]) -> list[dict]:
    macro_rows = []
    keys = sorted({(row["shot"], row["scope"], row["method"]) for row in rows})
    for shot, scope, method in keys:
        selected = [
            row
            for row in rows
            if row["shot"] == shot
            and row["scope"] == scope
            and row["method"] == method
        ]
        macro_rows.append(
            {
                "row_type": "scene_macro",
                "target_scene_id": "ALL",
                "shot": shot,
                "scope": scope,
                "method": method,
                "num_observations": int(
                    sum(row["num_observations"] for row in selected)
                ),
                **{
                    metric: float(np.nanmean([row[metric] for row in selected]))
                    for metric in ("auroc", "auprc", "fpr95")
                },
            }
        )
    return rows + macro_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prompt_loader(records, preprocessor, args, support: bool) -> DataLoader:
    return DataLoader(
        TargetSceneOFDMAPathDataset(records, preprocessor),
        batch_size=min(int(getattr(args, "batch_size", 128)), max(1, len(records))),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not args.use_cpu,
        persistent_workers=args.num_workers > 0 and not support,
    )


def make_promptad_context(args, method_cfg, device):
    from PromptAD import PromptAD

    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    kwargs = vars(method_cfg).copy()
    kwargs.update(
        {
            "dataset": "ofdma_spectrum",
            "class_name": "radio frequency spectrogram",
            "device": str(device),
            "out_size_h": method_cfg.resolution,
            "out_size_w": method_cfg.resolution,
            "k_shot": max(args.shots) * NUM_SUS,
        }
    )
    model = PromptAD(**kwargs).to(device)
    load_checkpoint(model, args.checkpoint)
    return model


@torch.no_grad()
def score_promptad(model, job, args, preprocessor, device) -> dict:
    support_loader = prompt_loader(
        job["support_records"], preprocessor, args, support=True
    )
    test_loader = prompt_loader(
        job["test_records"], preprocessor, args, support=False
    )
    build_gallery(model, support_loader, str(device))
    model.build_text_feature_gallery()
    model.eval_mode()

    names, labels, jammer_types, scores = [], [], [], []
    for data, _mask, label, name, jammer_type in test_loader:
        data_t = to_model_input(model, data, str(device), rgb_from_bgr=True)
        visual_features = model.encode_image(data_t)
        text_score = np.asarray(
            model.calculate_textual_anomaly_score(visual_features, "cls"),
            dtype=np.float32,
        )
        visual_map = model.calculate_visual_anomaly_score(visual_features)
        visual_score = (
            visual_map.flatten(1).max(dim=1).values.detach().cpu().numpy()
        )
        scores.extend(harmonic(text_score, visual_score).tolist())
        labels.extend(int(value) for value in label.numpy().tolist())
        names.extend(str(value) for value in name)
        jammer_types.extend(str(value) for value in jammer_type)
    return {
        "names": np.asarray(names),
        "labels": np.asarray(labels, dtype=np.int32),
        "jammer_types": np.asarray(jammer_types),
        "scores": np.asarray(scores, dtype=np.float32),
    }


def saved_payload_path(output_root: Path, job) -> Path:
    stem = (
        f"{job['dataset']}-{job['category']}-{job['scene']}-{job['jsr']}"
        .replace("/", "_")
    )
    return output_root / "scores" / f"{stem}-scores.npz"


def attach_jammer_types(payload, job) -> dict:
    names = np.asarray(payload["names"])
    expected_names = np.asarray([sample["name"] for sample in job["test_samples"]])
    if not np.array_equal(names.astype(str), expected_names.astype(str)):
        raise RuntimeError(f"Prediction order mismatch for {job['scene']} {job['jsr']}")
    labels = np.asarray(payload["labels"], dtype=np.int32)
    expected_labels = np.asarray(
        [sample["label"] for sample in job["test_samples"]], dtype=np.int32
    )
    if not np.array_equal(labels, expected_labels):
        raise RuntimeError(f"Prediction labels mismatch for {job['scene']} {job['jsr']}")
    return {
        "names": names,
        "labels": labels,
        "jammer_types": np.asarray(
            [sample["jammer_type"] for sample in job["test_samples"]]
        ),
        "scores": np.asarray(payload["scores"], dtype=np.float32),
    }


def predict_patchcore_scene(module, model, job, cfg, device) -> dict:
    """Score one target scene with one FAISS query instead of one per batch.

    PatchCore's reference dataloader path performs an independent FAISS search
    for every image batch.  The backbone is still evaluated in bounded GPU
    batches here, but the resulting CPU feature matrix is searched once.  The
    reference patch-to-image max reduction is kept exactly, and this path is
    used only by the target-scene runner.
    """

    dataset = module.PatchCorePathDataset(
        job["test_samples"], cfg.resize, cfg.imagesize
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    feature_parts = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(torch.float).to(device, non_blocking=True)
            feature_parts.append(np.asarray(model._embed(images)))
            labels.extend(int(value) for value in batch["is_anomaly"].numpy().tolist())
    if not feature_parts:
        raise RuntimeError("No target-scene images for PatchCore prediction")
    features = np.concatenate(feature_parts, axis=0)
    num_images = len(job["test_samples"])
    if features.shape[0] % num_images != 0:
        raise RuntimeError(
            f"PatchCore feature/image count mismatch: {features.shape[0]} / {num_images}"
        )
    patch_scores = model.anomaly_scorer.predict([features])[0]
    patch_scores = model.patch_maker.unpatch_scores(
        patch_scores, batchsize=num_images
    )
    patch_scores = patch_scores.reshape(num_images, patch_scores.shape[1], -1)
    image_scores = np.asarray(model.patch_maker.score(patch_scores)).reshape(-1)
    if image_scores.shape[0] != num_images:
        raise RuntimeError(
            f"PatchCore score/image count mismatch: {image_scores.shape[0]} / {num_images}"
        )
    print(
        f"    [patchcore_fast] query_features={features.shape} "
        f"images={num_images}",
        flush=True,
    )
    return {
        "names": np.asarray([sample["name"] for sample in job["test_samples"]]),
        "labels": np.asarray(labels, dtype=np.int32),
        "jammer_types": np.asarray(
            [sample["jammer_type"] for sample in job["test_samples"]]
        ),
        "scores": image_scores.astype(np.float32),
    }


def make_context(method: str, cfg, device):
    if method == "padim":
        module = importlib.import_module("tools.eval_padim_cls")
        return module.ResNetFeatureExtractor(cfg.weights).to(device).eval()
    if method == "winclip":
        module = importlib.import_module("tools.eval_winclip_main")
        return module.make_model(cfg, str(device))
    return None


def evaluate_non_prompt_job(method, context, job, cfg, device) -> dict:
    output_root = Path(cfg.output_root)
    setup_seed(cfg.seed)
    if method in {"vae", "iad_per"}:
        module = importlib.import_module("tools.eval_vae_cls")
        model, _history = module.fit_vae(job["train_samples"], cfg, device)
        module.predict_job(model, job, cfg, device)
        del model
    elif method == "saife":
        module = importlib.import_module("tools.eval_saife_ofdma_fewshot")
        model, _history = module.fit_saife(
            job["train_samples"], cfg, device
        )
        payload = module.predict(model, job["test_samples"], cfg, device)
        del model
        return {
            "names": payload["names"],
            "labels": payload["labels"],
            "jammer_types": payload["jammer_types"],
            "scores": payload["scores"],
        }
    elif method == "deep_svdd":
        module = importlib.import_module("tools.eval_deepsvdd_cls")
        model, center, _history = module.fit_deepsvdd(
            job["train_samples"], cfg, device
        )
        module.predict_job(model, center, job, cfg, device)
        del model, center
    elif method == "udma":
        module = importlib.import_module("tools.eval_udma_cls")
        model = module.fit_udma(job["train_samples"], cfg, device)
        module.predict_job(model, job, cfg, device)
        del model
    elif method == "padim":
        module = importlib.import_module("tools.eval_padim_cls")
        stats = module.fit_padim(context, job["train_samples"], cfg, device)
        module.predict_job(context, stats, job, cfg, device)
        del stats
    elif method == "stfpm":
        module = importlib.import_module("tools.eval_stfpm_cls")
        teacher, student, _history = module.fit_stfpm(
            job["train_samples"], cfg, device
        )
        module.predict_job(teacher, student, job, cfg, device)
        del teacher, student
    elif method == "winclip":
        module = importlib.import_module("tools.eval_winclip_main")
        module.build_visual_gallery(
            context, job["train_samples"], cfg, str(device)
        )
        module.predict_job(context, job, cfg, str(device))
    elif method == "patchcore":
        module = importlib.import_module("tools.eval_patchcore_cls")
        model = module.fit_patchcore(job["train_samples"], cfg, device)
        return predict_patchcore_scene(module, model, job, cfg, device)
    else:
        raise ValueError(method)

    with np.load(saved_payload_path(output_root, job), allow_pickle=True) as loaded:
        payload = {key: loaded[key] for key in loaded.files}
    return attach_jammer_types(payload, job)


def write_readme(output_root: Path, args, macro_rows, num_jobs: int) -> None:
    overall = [
        row
        for row in macro_rows
        if row["row_type"] == "scene_macro" and row["scope"] == "overall"
    ]
    lines = [
        f"# OFDMA target-scene baseline — {METHOD_NAMES[args.method]}",
        "",
        "Date: 2026-07-27",
        "",
        "Formal protocol: same-scene normal-only 1/2/4-shot support, 21-SU",
        "maximum observation score, and macro average over held-out target scenes.",
        "",
        "| Shot | AUROC | AUPRC | FPR@95TPR |",
        "|---:|---:|---:|---:|",
    ]
    for row in sorted(overall, key=lambda item: item["shot"]):
        lines.append(
            f"| {row['shot']} | {row['auroc']:.2f} | "
            f"{row['auprc']:.2f} | {row['fpr95']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Completed scene-shot jobs: `{num_jobs}`.",
            "",
            "Test labels were used only to calculate final metrics.",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if not args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if not args.shots or not set(args.shots).issubset(ALLOWED_SHOTS):
        raise ValueError("--shots must be selected from 1, 2, and 4")
    if args.max_normal_observations < 0 or args.max_anomaly_observations_per_type < 0:
        raise ValueError("Smoke limits cannot be negative")

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_target_scene_manifest(dataset_root)
    source = source_protocol(dataset_root)
    normalization = source["source_protocol"]["normalization"]
    jobs, scene_ids = build_jobs(args, manifest, normalization)
    if not jobs:
        raise RuntimeError("No target-scene jobs selected")

    protocol = {
        "entry": "tools/eval_ofdma_target_scene_baselines.py",
        "method": args.method,
        "method_name": METHOD_NAMES[args.method],
        "method_config": (
            PROMPTAD_CONFIG
            if args.method == "promptad"
            else METHOD_CONFIGS[args.method]
        ),
        "dataset_root": str(dataset_root.resolve()),
        "dataset_protocol": source["source_protocol"]["protocol_name"],
        "split": args.split,
        "target_scene_ids": scene_ids,
        "shots": sorted(set(args.shots)),
        "support_rule": "first k ranked normal observations from the same target scene",
        "test_rule": "same target scene normal_test plus anomaly_test",
        "su_reduction": "maximum score over 21 sensing units",
        "test_label_usage": "metrics_only",
        "seed": args.seed,
        "smoke_limits": {
            "max_normal_observations": args.max_normal_observations,
            "max_anomaly_observations_per_type": (
                args.max_anomaly_observations_per_type
            ),
        },
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    sample_raw = cv2.imread(
        str(jobs[0]["support_records"][0].image_path), cv2.IMREAD_GRAYSCALE
    )
    preprocessor = OFDMASpectrogramPreprocessor(
        dataset_root,
        output_size=240,
        geometry="letterbox",
        min_db=float(normalization["min_db"]),
        max_db=float(normalization["max_db"]),
    )
    if preprocessor(sample_raw).shape != (240, 240):
        raise RuntimeError("Target-scene preprocessing probe failed")
    print(json.dumps(protocol, indent=2), flush=True)
    if args.validate_only:
        print("[validated] target-scene baseline protocol", flush=True)
        return

    if not args.use_cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cpu" if args.use_cpu else "cuda:0")
    cfg = method_args(args, normalization)
    # Record the effective runtime configuration, including optional resource
    # overrides, rather than only the frozen defaults shown in METHOD_CONFIGS.
    protocol["method_config"] = vars(cfg).copy()
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    setup_seed(args.seed)
    context = (
        make_promptad_context(args, cfg, device)
        if args.method == "promptad"
        else make_context(args.method, cfg, device)
    )

    score_root = output_root / "formal_scores"
    score_root.mkdir(exist_ok=True)
    rows = []
    for index, job in enumerate(jobs, 1):
        print(
            f"[{index}/{len(jobs)}] {args.method} "
            f"{job['target_scene_id']} {job['shot']}-shot",
            flush=True,
        )
        if args.method == "promptad":
            payload = score_promptad(
                context, job, args, preprocessor, device
            )
        else:
            payload = evaluate_non_prompt_job(
                args.method, context, job, cfg, device
            )
        if not np.all(np.isfinite(payload["scores"])):
            raise RuntimeError(
                f"Non-finite scores for {job['target_scene_id']} {job['shot']}-shot"
            )
        observation_scores = aggregate_observations(payload)
        np.savez_compressed(
            score_root
            / f"{job['target_scene_id']}_{job['shot']}shot_su_scores.npz",
            **payload,
        )
        np.savez_compressed(
            score_root
            / f"{job['target_scene_id']}_{job['shot']}shot_observation_scores.npz",
            **observation_scores,
        )
        rows.extend(
            metric_rows(job, observation_scores, METHOD_NAMES[args.method])
        )
        merged = append_macro_rows(rows)
        write_csv(output_root / "results.csv", merged)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    merged = append_macro_rows(rows)
    macro = [row for row in merged if row["row_type"] == "scene_macro"]
    summary = {
        "status": "complete",
        "method": METHOD_NAMES[args.method],
        "num_scene_shot_jobs": len(jobs),
        "target_scenes": len(scene_ids),
        "shots": sorted(set(args.shots)),
        "scene_macro": macro,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_readme(output_root, args, merged, len(jobs))
    print(f"[done] {output_root / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
