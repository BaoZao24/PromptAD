#!/usr/bin/env python
"""Evaluate the TFAM-AAE spectrogram reconstruction branch.

This is a faithful adaptation of the paper's first metric (Eq. 9) to the
repository's PNG spectrogram protocols.  The paper's ``U_k`` Mahalanobis
branch is not enabled because the available RF data contains dBm spectrograms
and no paired raw IQ samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.tfam_aae import TFAMAAE, reconstruction_mse  # noqa: E402
from tools.eval_patchcore_cls import (  # noqa: E402
    JSR_BY_SIGNAL,
    PUBLIC_RF_JSRS,
    SCENES,
    SPECTRUM_CATEGORIES,
    SPECTRUM_ROOT,
    public_rf_jobs,
    rf_target_jobs,
    spectrum_jobs,
)
from tools.ofdma_fewshot_baseline_common import (  # noqa: E402
    add_ofdma_args,
    load_sample_image,
    ofdma_jobs,
)
from utils.rf_frequency_sampling import NORMAL_SAMPLING_CHOICES  # noqa: E402
from utils.training_utils import setup_seed  # noqa: E402


METHOD = "tfam_aae_spectrogram"
TARGET_SAMPLING_CHOICES = ("per_frequency", "1shot", "2shot", "4shot")


class TFAMPathDataset(Dataset):
    """Load rendered spectrograms into the single-channel TFAM input format."""

    def __init__(self, samples: list[dict], image_height: int, image_width: int):
        self.samples = list(samples)
        self.image_height = int(image_height)
        self.image_width = int(image_width)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        if "image_bgr" in sample:
            # In-memory adapter (FedJam Arrow records); the BGR buffer is the
            # authoritative image and there is no file to read.
            image = cv2.cvtColor(
                np.asarray(sample["image_bgr"], dtype=np.uint8), cv2.COLOR_BGR2GRAY
            )
        elif str(sample.get("dataset", "")) == "ofdma":
            # OFDMA samples must go through the official physics preprocessor;
            # every other protocol reads the rendered PNG directly.
            image = np.asarray(load_sample_image(sample).convert("L"))
        else:
            image = cv2.imread(str(sample["path"]), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(sample["path"])
        image = cv2.resize(
            image,
            (self.image_width, self.image_height),
            interpolation=cv2.INTER_AREA,
        )
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor, int(sample["label"]), str(sample["path"]), str(sample["name"])


def aggregate_ofdma_scores(
    samples: list[dict], labels: list[int], scores: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse per-SU image scores to per-observation scores.

    Each OFDMA observation groups the SU frames of one scene.  The formal
    protocol takes the maximum image score inside an observation, mirroring
    `tools/eval_traditional_spectral_baselines.py::aggregate_ofdma`.
    """

    groups: dict[int, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(int(sample["scene_id"]), []).append(index)
    agg_labels, agg_scores, observation_ids = [], [], []
    for scene_id in sorted(groups):
        indices = groups[scene_id]
        group_labels = {int(labels[index]) for index in indices}
        if len(group_labels) != 1:
            raise ValueError(
                f"Inconsistent OFDMA observation labels for scene {scene_id}"
            )
        agg_labels.append(int(labels[indices[0]]))
        agg_scores.append(max(float(scores[index]) for index in indices))
        observation_ids.append(scene_id)
    return (
        np.asarray(agg_labels, dtype=np.int32),
        np.asarray(agg_scores, dtype=np.float64),
        np.asarray(observation_ids, dtype=np.int64),
    )


def safe_metrics(labels, scores) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    # FedJam keeps four semantic labels in saved payloads; the shared headline
    # task is benign (0) versus any jammer (>0), matching the other baselines.
    if np.unique(labels).size > 2:
        labels = (labels != 0).astype(np.int32)
    if labels.size == 0 or np.unique(labels).size < 2:
        return {
            "image_auroc": float("nan"),
            "image_auprc": float("nan"),
            "fpr95": float("nan"),
        }
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "image_auroc": float(roc_auc_score(labels, scores) * 100.0),
        "image_auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if reached.size else float("nan"),
    }


def make_loader(samples: list[dict], args, shuffle: bool) -> DataLoader:
    return DataLoader(
        TFAMPathDataset(samples, args.image_height, args.image_width),
        batch_size=args.batch_size if shuffle else args.test_batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def train_signature(job: dict) -> tuple[str, ...]:
    return tuple(str(sample["path"]) for sample in job["train_samples"])


def fit_tfam(train_samples: list[dict], args, device: torch.device):
    if not train_samples or any(int(sample["label"]) != 0 for sample in train_samples):
        raise RuntimeError("TFAM support data must be non-empty and normal-only")

    model = TFAMAAE(
        input_shape=(args.image_height, args.image_width),
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        discriminator_hidden_dim=args.discriminator_hidden_dim,
        discriminator_dropout=args.discriminator_dropout,
    ).to(device)
    loader = make_loader(train_samples, args, shuffle=True)
    reconstruction_optimizer = Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    discriminator_optimizer = Adam(
        model.discriminator.parameters(),
        lr=args.discriminator_lr,
        weight_decay=args.weight_decay,
    )
    generator_optimizer = Adam(model.encoder.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        reconstruction_losses = []
        discriminator_losses = []
        generator_losses = []
        for data, _label, _path, _name in loader:
            data = data.to(device, non_blocking=True)

            reconstruction, _latent = model.reconstruct(data)
            reconstruction_loss = F.mse_loss(reconstruction, data)
            reconstruction_optimizer.zero_grad(set_to_none=True)
            reconstruction_loss.backward()
            reconstruction_optimizer.step()

            set_requires_grad(model.discriminator, True)
            with torch.no_grad():
                encoded = model.encode(data)
            prior = torch.randn_like(encoded)
            real_logits = model.discriminator(prior)
            fake_logits = model.discriminator(encoded)
            discriminator_loss = bce(
                real_logits, torch.ones_like(real_logits)
            ) + bce(fake_logits, torch.zeros_like(fake_logits))
            discriminator_optimizer.zero_grad(set_to_none=True)
            discriminator_loss.backward()
            discriminator_optimizer.step()

            set_requires_grad(model.discriminator, False)
            encoded = model.encode(data)
            generator_loss = bce(
                model.discriminator(encoded), torch.ones_like(fake_logits)
            )
            generator_optimizer.zero_grad(set_to_none=True)
            (args.adversarial_weight * generator_loss).backward()
            generator_optimizer.step()
            set_requires_grad(model.discriminator, True)

            reconstruction_losses.append(float(reconstruction_loss.detach().cpu()))
            discriminator_losses.append(float(discriminator_loss.detach().cpu()))
            generator_losses.append(float(generator_loss.detach().cpu()))

        row = {
            "epoch": epoch,
            "reconstruction_loss": float(np.mean(reconstruction_losses)),
            "discriminator_loss": float(np.mean(discriminator_losses)),
            "generator_loss": float(np.mean(generator_losses)),
        }
        history.append(row)
        if args.log_every > 0 and (
            epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0
        ):
            print(
                f"    epoch {epoch}/{args.epochs} "
                f"reconstruction={row['reconstruction_loss']:.6f} "
                f"discriminator={row['discriminator_loss']:.6f} "
                f"generator={row['generator_loss']:.6f}",
                flush=True,
            )
    return model.eval(), history


@torch.no_grad()
def predict_job(model: TFAMAAE, job: dict, args, device: torch.device) -> dict:
    loader = make_loader(job["test_samples"], args, shuffle=False)
    labels, scores, paths, names = [], [], [], []
    model.eval()
    for data, batch_labels, batch_paths, batch_names in loader:
        data = data.to(device, non_blocking=True)
        reconstruction, _latent = model.reconstruct(data)
        if args.score_mode == "mse_mean":
            batch_scores = reconstruction_mse(data, reconstruction)
        elif args.score_mode == "mae_mean":
            batch_scores = (data - reconstruction).abs().flatten(1).mean(dim=1)
        else:
            raise ValueError(f"Unsupported score mode: {args.score_mode}")
        labels.extend(int(value) for value in batch_labels.numpy().tolist())
        scores.extend(float(value) for value in batch_scores.cpu().numpy().tolist())
        paths.extend(str(value) for value in batch_paths)
        names.extend(str(value) for value in batch_names)

    labels_np = np.asarray(labels, dtype=np.int32)
    scores_np = np.asarray(scores, dtype=np.float32)
    if not np.all(np.isfinite(scores_np)):
        raise RuntimeError(
            f"Non-finite TFAM scores for {job['category']} {job['scene']} {job['jsr']}"
        )
    if job["dataset"] == "ofdma":
        metric_labels, metric_scores, observation_ids = aggregate_ofdma_scores(
            job["test_samples"], labels, scores
        )
        count_normal = int((metric_labels == 0).sum())
        count_abnormal = int((metric_labels != 0).sum())
    else:
        metric_labels, metric_scores = labels_np, scores_np.astype(np.float64)
        observation_ids = None
        count_normal = int((metric_labels == 0).sum())
        count_abnormal = int((metric_labels != 0).sum())
    metrics = safe_metrics(metric_labels, metric_scores)
    score_dir = Path(args.output_root) / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{job['dataset']}-{job['category']}-{job['scene']}-{job['jsr']}".replace(
        "/", "_"
    )
    payload = {
        "scores": scores_np,
        "labels": labels_np,
        "image_paths": np.asarray(paths),
        "names": np.asarray(names),
    }
    if observation_ids is not None:
        payload["observation_ids"] = observation_ids
        payload["observation_scores"] = np.asarray(metric_scores, dtype=np.float32)
        payload["observation_labels"] = np.asarray(metric_labels, dtype=np.int32)
    if getattr(args, "support_manifest_sha256", ""):
        payload["support_manifest_sha256"] = np.asarray(args.support_manifest_sha256)
    np.savez_compressed(score_dir / f"{stem}-scores.npz", **payload)
    return {
        "method": METHOD,
        "dataset": job["dataset"],
        "category": job["category"],
        "scene": job["scene"],
        "jsr": job["jsr"],
        "num_train_normal": len(job["train_samples"]),
        "num_test_normal": count_normal,
        "num_test_abnormal": count_abnormal,
        **metrics,
    }


def run_jobs(jobs: list[dict], args, device: torch.device) -> list[dict]:
    grouped: dict[tuple[str, ...], list[tuple[int, dict]]] = {}
    for index, job in enumerate(jobs):
        grouped.setdefault(train_signature(job), []).append((index, job))

    rows = []
    checkpoint_root = Path(args.output_root) / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for group_index, (_signature, grouped_jobs) in enumerate(grouped.items(), 1):
        first_job = grouped_jobs[0][1]
        if not first_job["train_samples"]:
            raise RuntimeError("No TFAM normal support images")
        print(
            f"[fit {group_index}/{len(grouped)}] "
            f"train_normals={len(first_job['train_samples'])} "
            f"eval_jobs={len(grouped_jobs)}",
            flush=True,
        )
        setup_seed(args.seed)
        model, history = fit_tfam(first_job["train_samples"], args, device)
        (Path(args.output_root) / f"training_history_group{group_index:03d}.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save(
            {
                "model": model.state_dict(),
                "support_paths": train_signature(first_job),
                "model_config": {
                    "input_shape": [args.image_height, args.image_width],
                    "base_channels": args.base_channels,
                    "latent_dim": args.latent_dim,
                    "discriminator_hidden_dim": args.discriminator_hidden_dim,
                },
            },
            checkpoint_root / f"tfam_group{group_index:03d}.pt",
        )
        for original_index, job in grouped_jobs:
            if not any(int(sample["label"]) == 1 for sample in job["test_samples"]):
                raise RuntimeError(
                    f"No anomalies for {job['category']} {job['scene']} {job['jsr']}"
                )
            print(
                f"[{original_index + 1}/{len(jobs)}] {job['category']} "
                f"{job['scene']} {job['jsr']}",
                flush=True,
            )
            row = predict_job(model, job, args, device)
            row["train_reconstruction_loss_final"] = history[-1][
                "reconstruction_loss"
            ]
            rows.append(row)
            print(f"    image AUROC={row['image_auroc']:.4f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return sorted(rows, key=lambda row: (row["dataset"], row["category"], row["scene"], row["jsr"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def macro_row(rows: list[dict]) -> dict:
    return {
        "method": METHOD,
        "dataset": rows[0]["dataset"],
        "category": "average",
        "scene": "macro",
        "jsr": "macro",
        "num_train_normal": "",
        "num_test_normal": "",
        "num_test_abnormal": "",
        "image_auroc": float(np.nanmean([row["image_auroc"] for row in rows])),
        "image_auprc": float(np.nanmean([row["image_auprc"] for row in rows])),
        "fpr95": float(np.nanmean([row["fpr95"] for row in rows])),
        "train_reconstruction_loss_final": float(
            np.nanmean([row["train_reconstruction_loss_final"] for row in rows])
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normal-only TFAM-AAE spectrogram reconstruction baseline"
    )
    parser.add_argument(
        "--protocol",
        choices=["rf_target", "public_rf", "spectrum", "ofdma"],
        required=True,
    )
    parser.add_argument("--output-root", default="analysis_outputs/tfam_aae_spectral")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument(
        "--normal-sampling", choices=NORMAL_SAMPLING_CHOICES, default="per_frequency"
    )
    parser.add_argument("--support-manifest", default="")
    parser.add_argument(
        "--rf-signals",
        nargs="+",
        default=list(JSR_BY_SIGNAL),
        choices=list(JSR_BY_SIGNAL),
    )
    parser.add_argument(
        "--rf-scenes", nargs="+", default=list(SCENES), choices=list(SCENES)
    )
    parser.add_argument(
        "--public-rf-signals",
        nargs="+",
        default=list(PUBLIC_RF_JSRS),
        choices=list(PUBLIC_RF_JSRS),
    )
    parser.add_argument("--spectrum-root", default=str(SPECTRUM_ROOT))
    parser.add_argument(
        "--spectrum-categories",
        nargs="+",
        default=list(SPECTRUM_CATEGORIES),
        choices=list(SPECTRUM_CATEGORIES),
    )
    parser.add_argument("--max-train-normals", type=int, default=0)
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--discriminator-hidden-dim", type=int, default=128)
    parser.add_argument("--discriminator-dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--discriminator-lr", type=float, default=2.5e-5)
    parser.add_argument("--adversarial-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--score-mode", choices=["mse_mean", "mae_mean"], default="mse_mean")
    parser.add_argument("--log-every", type=int, default=10)
    add_ofdma_args(parser)
    return parser.parse_args()


def validate_args(args) -> None:
    if args.protocol == "rf_target" and args.normal_sampling not in TARGET_SAMPLING_CHOICES:
        raise ValueError(
            f"rf_target support sampling must be one of {TARGET_SAMPLING_CHOICES}"
        )
    if args.image_height < 8 or args.image_width < 8:
        raise ValueError("image dimensions must be at least 8")
    if args.epochs < 1 or args.batch_size < 1 or args.test_batch_size < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if args.base_channels < 1 or args.latent_dim < 1:
        raise ValueError("base_channels and latent_dim must be positive")
    if args.lr <= 0 or args.discriminator_lr <= 0 or args.adversarial_weight < 0:
        raise ValueError("learning rates must be positive and adversarial weight non-negative")


def main() -> None:
    args = parse_args()
    validate_args(args)
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device(
        "cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda:0"
    )

    if args.protocol == "rf_target":
        jobs = rf_target_jobs(args)
    elif args.protocol == "public_rf":
        jobs = public_rf_jobs(args)
    elif args.protocol == "ofdma":
        jobs = ofdma_jobs(args)
    else:
        jobs = spectrum_jobs(args)
    if not jobs:
        raise RuntimeError(f"No jobs built for {args.protocol}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "method": METHOD,
        "implementation": "TFAM-AAE spectrogram reconstruction branch",
        "paper_metric": "Eq. (9) per-sample MSE reconstruction error",
        "user_discriminator": "not implemented: current data has dBm spectrograms, no raw IQ",
        "protocol": args.protocol,
        "support_protocol": (
            "target_scene" if args.protocol == "rf_target"
            else "official_scene_split" if args.protocol == "ofdma"
            else None
        ),
        "ofdma_shots": list(args.ofdma_shots) if args.protocol == "ofdma" else None,
        "support_manifest": getattr(args, "support_manifest", "") or None,
        "support_manifest_sha256": getattr(args, "support_manifest_sha256", None),
        "normal_sampling": args.normal_sampling,
        "input_shape": [1, args.image_height, args.image_width],
        "input_conversion": "grayscale PNG resized to [0, 1]",
        "training_labels": "normal support only",
        "test_label_usage": "metrics only after scoring",
        "score_mode": args.score_mode,
        "base_channels": args.base_channels,
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "discriminator_lr": args.discriminator_lr,
        "adversarial_weight": args.adversarial_weight,
        "seed": args.seed,
        "device": str(device),
        "num_evaluation_cells": len(jobs),
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2), flush=True)

    rows = run_jobs(jobs, args, device)
    rows_with_macro = rows + [macro_row(rows)]
    result_path = output_root / "results_tfam_cls.csv"
    write_csv(result_path, rows_with_macro)
    summary = {
        "method": METHOD,
        "protocol": args.protocol,
        "num_jobs": len(rows),
        "image_auroc_macro": rows_with_macro[-1]["image_auroc"],
        "image_auprc_macro": rows_with_macro[-1]["image_auprc"],
        "fpr95_macro": rows_with_macro[-1]["fpr95"],
        "user_discriminator_available": False,
        "result_csv": str(result_path),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
