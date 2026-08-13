#!/usr/bin/env python
"""Evaluate a normal-only SAIFE-style AAE on the OFDMA few-shot protocol.

This is a small PyTorch port of the CNN adversarial-autoencoder core in
``references/saife/spec_aae.py``.  OFDMA has only one normal support class, so
SAIFE's categorical latent variable is mathematically constant and its
categorical discriminator/supervised loss contain no usable information.  The
continuous Gaussian latent prior, CNN autoencoder, and latent discriminator are
retained.  Per-image reconstruction MSE is used as the anomaly score.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.ofdma_spectrum import (  # noqa: E402
    JAMMER_TYPES,
    NO_JAMMER,
    NUM_SUS,
    OFDMASpectrogramPreprocessor,
    build_official_scene_split,
    load_ofdma_labels,
    parse_ofdma_name,
)
from tools.ofdma_fewshot_baseline_common import add_ofdma_args, ofdma_jobs  # noqa: E402
from utils.training_utils import setup_seed  # noqa: E402


METHOD = "saife_reconstruction"
NATIVE_SHAPE = (110, 70)
ALLOWED_SHOTS = (1, 2, 4)


class SAIFENativeDataset(Dataset):
    """Load official OFDMA aggregation without square image adaptation."""

    def __init__(
        self,
        samples: list[dict],
        dataset_root: str,
        min_db: float | None = None,
        max_db: float | None = None,
    ):
        self.samples = list(samples)
        self.preprocessor = OFDMASpectrogramPreprocessor(
            dataset_root=dataset_root,
            min_db=min_db,
            max_db=max_db,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = cv2.imread(str(sample["path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(sample["path"])
        image = self.preprocessor.aggregate_subcarriers(image)
        if image.shape != NATIVE_SHAPE:
            raise RuntimeError(f"Unexpected aggregated shape {image.shape} for {sample['path']}")
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        return (
            tensor,
            int(sample["label"]),
            str(sample["path"]),
            str(sample["name"]),
            str(sample["jammer_type"]),
        )


class SAIFEEncoder(nn.Module):
    """Three tanh convolution/max-pool blocks as in the SAIFE source."""

    def __init__(self, input_shape: tuple[int, int], latent_dim: int):
        super().__init__()
        height, width = (int(value) for value in input_shape)
        pool_sizes = []
        for _ in range(3):
            height, width = height // 2, width // 2
            pool_sizes.append((height, width))
        if min(pool_sizes[-1]) < 1:
            raise ValueError(f"SAIFE input is too small for three pooling stages: {input_shape}")
        self.decoder_sizes = (pool_sizes[1], pool_sizes[0], tuple(input_shape))
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            feature = self.features(torch.zeros(1, 1, *input_shape))
        self.feature_shape = tuple(int(value) for value in feature.shape[1:])
        self.to_latent = nn.Linear(int(feature.numel()), latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.features(x).flatten(1))


class SAIFEDecoder(nn.Module):
    """Dynamic counterpart of SAIFE's tanh convolution/upsampling decoder."""

    def __init__(
        self,
        latent_dim: int,
        feature_shape: tuple[int, int, int],
        target_sizes: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    ):
        super().__init__()
        channels, height, width = feature_shape
        # The extra value is the one-class categorical latent used by SAIFE.
        self.feature_shape = feature_shape
        self.target_sizes = target_sizes
        self.from_latent = nn.Linear(latent_dim + 1, channels * height * width)
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.output = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        category = torch.ones((z.size(0), 1), dtype=z.dtype, device=z.device)
        x = torch.tanh(self.from_latent(torch.cat((z, category), dim=1)))
        x = x.view(z.size(0), *self.feature_shape)
        x = F.interpolate(torch.tanh(self.conv1(x)), size=self.target_sizes[0], mode="nearest")
        x = F.interpolate(torch.tanh(self.conv2(x)), size=self.target_sizes[1], mode="nearest")
        x = F.interpolate(torch.tanh(self.conv3(x)), size=self.target_sizes[2], mode="nearest")
        # SAIFE uses a linear reconstruction output rather than a sigmoid.
        return self.output(x)


class SAIFELatentDiscriminator(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(z)


class SAIFEAAE(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        dropout: float,
        input_shape: tuple[int, int] = NATIVE_SHAPE,
    ):
        super().__init__()
        self.encoder = SAIFEEncoder(input_shape, latent_dim)
        self.decoder = SAIFEDecoder(
            latent_dim,
            self.encoder.feature_shape,
            self.encoder.decoder_sizes,
        )
        self.discriminator = SAIFELatentDiscriminator(latent_dim, hidden_dim, dropout)

    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(x)[0]


def make_loader(samples: list[dict], args, shuffle: bool, device: torch.device) -> DataLoader:
    dataset = SAIFENativeDataset(
        samples,
        args.ofdma_root,
        min_db=getattr(args, "ofdma_min_db", None),
        max_db=getattr(args, "ofdma_max_db", None),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size if shuffle else args.test_batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def fit_saife(train_samples: list[dict], args, device: torch.device):
    if not train_samples or any(int(sample["label"]) != 0 for sample in train_samples):
        raise RuntimeError("SAIFE support data must be non-empty and normal-only")

    model = SAIFEAAE(args.latent_dim, args.hidden_dim, args.dropout).to(device)
    loader = make_loader(train_samples, args, shuffle=True, device=device)
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
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_reconstruction = []
        epoch_discriminator = []
        epoch_generator = []
        for data, _label, _path, _name, _jammer_type in loader:
            data = data.to(device, non_blocking=True)

            reconstruction, _ = model.reconstruct(data)
            pixel_error = F.mse_loss(reconstruction, data, reduction="none")
            reconstruction_loss = pixel_error.flatten(1).sum(dim=1).mean()
            reconstruction_optimizer.zero_grad(set_to_none=True)
            reconstruction_loss.backward()
            reconstruction_optimizer.step()

            _set_requires_grad(model.discriminator, True)
            with torch.no_grad():
                encoded = model.encoder(data)
            prior = torch.randn_like(encoded)
            real_logits = model.discriminator(prior)
            fake_logits = model.discriminator(encoded)
            discriminator_loss = bce(real_logits, torch.ones_like(real_logits)) + bce(
                fake_logits, torch.zeros_like(fake_logits)
            )
            discriminator_optimizer.zero_grad(set_to_none=True)
            discriminator_loss.backward()
            discriminator_optimizer.step()

            _set_requires_grad(model.discriminator, False)
            encoded = model.encoder(data)
            fake_logits = model.discriminator(encoded)
            generator_loss = bce(fake_logits, torch.ones_like(fake_logits))
            generator_optimizer.zero_grad(set_to_none=True)
            (args.adversarial_weight * generator_loss).backward()
            generator_optimizer.step()
            _set_requires_grad(model.discriminator, True)

            epoch_reconstruction.append(float(reconstruction_loss.detach().cpu()))
            epoch_discriminator.append(float(discriminator_loss.detach().cpu()))
            epoch_generator.append(float(generator_loss.detach().cpu()))

        row = {
            "epoch": epoch,
            "reconstruction_loss": float(np.mean(epoch_reconstruction)),
            "discriminator_loss": float(np.mean(epoch_discriminator)),
            "generator_loss": float(np.mean(epoch_generator)),
        }
        history.append(row)
        if args.log_every > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0):
            print(
                f"    epoch {epoch}/{args.epochs} reconstruction={row['reconstruction_loss']:.6f} "
                f"discriminator={row['discriminator_loss']:.6f} generator={row['generator_loss']:.6f}",
                flush=True,
            )
    return model.eval(), history


@torch.no_grad()
def predict(model: SAIFEAAE, samples: list[dict], args, device: torch.device) -> dict:
    loader = make_loader(samples, args, shuffle=False, device=device)
    labels: list[int] = []
    scores: list[float] = []
    paths: list[str] = []
    names: list[str] = []
    jammer_types: list[str] = []
    model.eval()
    for data, label, path, name, jammer_type in loader:
        data = data.to(device, non_blocking=True)
        reconstruction = model(data)
        batch_scores = F.mse_loss(reconstruction, data, reduction="none").flatten(1).mean(dim=1)
        labels.extend(int(value) for value in label.numpy().tolist())
        scores.extend(float(value) for value in batch_scores.cpu().numpy().tolist())
        paths.extend(str(value) for value in path)
        names.extend(str(value) for value in name)
        jammer_types.extend(str(value) for value in jammer_type)
    return {
        "scores": np.asarray(scores, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int32),
        "image_paths": np.asarray(paths),
        "names": np.asarray(names),
        "jammer_types": np.asarray(jammer_types),
    }


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


def aggregate_scenes(names, labels, jammer_types, scores, reduction: str):
    groups = defaultdict(list)
    for index, name in enumerate(names):
        scene_id, _ = parse_ofdma_name(str(name))
        groups[scene_id].append(index)
    scene_labels, scene_types, scene_scores = [], [], []
    for scene_id in sorted(groups):
        indices = np.asarray(groups[scene_id], dtype=np.int64)
        if len(np.unique(labels[indices])) != 1 or len(np.unique(jammer_types[indices])) != 1:
            raise RuntimeError(f"Inconsistent labels inside OFDMA scene {scene_id}")
        selected = scores[indices]
        scene_labels.append(int(labels[indices[0]]))
        scene_types.append(str(jammer_types[indices[0]]))
        scene_scores.append(float(selected.mean() if reduction == "mean" else selected.max()))
    return np.asarray(scene_labels), np.asarray(scene_types), np.asarray(scene_scores)


def metric_rows(shot: int, level: str, labels, jammer_types, scores) -> list[dict]:
    rows = []
    per_jammer = []
    for scope in ("overall", *JAMMER_TYPES):
        mask = np.ones(len(labels), dtype=bool) if scope == "overall" else (
            (jammer_types == NO_JAMMER) | (jammer_types == scope)
        )
        metrics = safe_metrics(labels[mask], scores[mask])
        rows.append(
            {
                "shot": shot,
                "level": level,
                "scope": scope,
                "method": METHOD,
                "num_samples": int(mask.sum()),
                **metrics,
            }
        )
        if scope != "overall":
            per_jammer.append(metrics)
    rows.append(
        {
            "shot": shot,
            "level": level,
            "scope": "macro_jammer",
            "method": METHOD,
            "num_samples": int(len(labels)),
            **{
                key: float(np.nanmean([item[key] for item in per_jammer]))
                for key in ("auroc", "auprc", "fpr95")
            },
        }
    )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normal-only SAIFE AAE on the official OFDMA 1/2/4-shot protocol"
    )
    parser.add_argument(
        "--output-root",
        default="analysis_outputs/20260721_ofdma_saife_fewshot",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--discriminator-lr", type=float, default=2.5e-5)
    parser.add_argument("--adversarial-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--validate-only", action="store_true")
    add_ofdma_args(parser)
    return parser.parse_args()


def validate_args(args) -> None:
    shots = tuple(sorted(set(int(value) for value in args.ofdma_shots)))
    if not shots or any(value not in ALLOWED_SHOTS for value in shots):
        raise ValueError(
            f"This evaluator is intentionally few-shot only; --ofdma-shots must be drawn from {ALLOWED_SHOTS}"
        )
    if args.epochs < 1 or args.batch_size < 1 or args.test_batch_size < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if args.latent_dim < 1 or args.hidden_dim < 1:
        raise ValueError("latent and hidden dimensions must be positive")
    if args.lr <= 0 or args.discriminator_lr <= 0 or args.adversarial_weight < 0:
        raise ValueError("learning rates must be positive and adversarial weight non-negative")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")


def main() -> None:
    args = parse_args()
    validate_args(args)
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda:0")

    jobs = sorted(ofdma_jobs(args), key=lambda job: int(job["shot"]))
    if not jobs:
        raise RuntimeError("No OFDMA jobs were built")
    labels_df = load_ofdma_labels(args.ofdma_root)
    split = build_official_scene_split(labels_df)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "scores").mkdir(exist_ok=True)
    (output_root / "checkpoints").mkdir(exist_ok=True)

    probe = SAIFENativeDataset([jobs[0]["train_samples"][0]], args.ofdma_root)[0][0]
    if tuple(probe.shape) != (1, *NATIVE_SHAPE):
        raise RuntimeError(f"Native preprocessing probe failed: {tuple(probe.shape)}")

    protocol = {
        "method": METHOD,
        "implementation": "PyTorch port of references/saife CNN adversarial-autoencoder core",
        "reference_source": str((REPO_ROOT / "references" / "saife").resolve()),
        "dataset_root": str(Path(args.ofdma_root).resolve()),
        "raw_shape": [1320, 70],
        "aggregated_model_shape": [1, *NATIVE_SHAPE],
        "geometry": "native_non_square",
        "normalization": "official 12-subcarrier linear-power aggregation mapped to [0, 1]",
        "shots": [int(job["shot"]) for job in jobs],
        "images_per_scene": int(args.max_sus_per_scene),
        "official_split": {
            "train_normal_pool_scenes": len(split.train_normal),
            "valid_normal_scenes_unused": len(split.valid_normal),
            "test_normal_scenes": len(split.test_normal),
            "test_anomaly_scenes_by_type": {
                key: len(value) for key, value in split.test_abnormal_by_type.items()
            },
        },
        "actual_test_images": len(jobs[0]["test_samples"]),
        "training_labels": "normal support only",
        "test_label_usage": "metrics_only_after_scoring",
        "anomaly_score": "per_image_reconstruction_mse",
        "categorical_branch": "collapsed_to_constant_one_because_support_has_one_class",
        "latent_prior": "standard_normal",
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "discriminator_lr": args.discriminator_lr,
        "adversarial_weight": args.adversarial_weight,
        "seed": args.seed,
        "device": str(device),
        "full_shot_supported": False,
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2), flush=True)

    for job in jobs:
        support = {
            "shot": int(job["shot"]),
            "scene_ids": list(job["support_scene_ids"]),
            "num_images": len(job["train_samples"]),
            "images_per_scene": int(args.max_sus_per_scene),
        }
        (output_root / f"support_{job['shot']}shot.json").write_text(
            json.dumps(support, indent=2) + "\n", encoding="utf-8"
        )
    if args.validate_only:
        print("[validated] protocol, support selection, and native preprocessing", flush=True)
        return

    all_rows: list[dict] = []
    completed = []
    for index, job in enumerate(jobs, 1):
        shot = int(job["shot"])
        print(
            f"[{index}/{len(jobs)}] fit SAIFE {shot}-shot: "
            f"train_images={len(job['train_samples'])} test_images={len(job['test_samples'])}",
            flush=True,
        )
        # Reinitialize each shot reproducibly instead of continuing from the smaller-shot model.
        setup_seed(args.seed)
        model, history = fit_saife(job["train_samples"], args, device)
        payload = predict(model, job["test_samples"], args, device)
        if not np.all(np.isfinite(payload["scores"])):
            raise RuntimeError(f"Non-finite anomaly scores in {shot}-shot evaluation")
        if not np.any(payload["labels"] == 0) or not np.any(payload["labels"] == 1):
            raise RuntimeError(f"Both normal and anomaly samples are required for {shot}-shot metrics")

        legacy_path = output_root / "scores" / f"ofdma-ofdma-official_split-{shot}shot-scores.npz"
        np.savez_compressed(legacy_path, **payload)
        np.savez_compressed(
            output_root / "scores" / f"ofdma_{shot}shot_image_scores.npz",
            names=payload["names"],
            labels=payload["labels"],
            jammer_types=payload["jammer_types"],
            saife_reconstruction=payload["scores"],
        )
        torch.save(
            {
                "model": model.state_dict(),
                "shot": shot,
                "support_scene_ids": job["support_scene_ids"],
                "model_config": {
                    "input_shape": NATIVE_SHAPE,
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                },
            },
            output_root / "checkpoints" / f"saife_{shot}shot.pt",
        )
        (output_root / f"training_history_{shot}shot.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )

        labels = payload["labels"]
        jammer_types = payload["jammer_types"]
        scores = payload["scores"]
        all_rows.extend(metric_rows(shot, "image", labels, jammer_types, scores))
        for reduction in ("mean", "max"):
            scene_labels, scene_types, scene_scores = aggregate_scenes(
                payload["names"], labels, jammer_types, scores, reduction
            )
            all_rows.extend(
                metric_rows(shot, f"scene_{reduction}", scene_labels, scene_types, scene_scores)
            )
        write_csv(output_root / "results.csv", all_rows)

        overall = next(
            row
            for row in all_rows
            if row["shot"] == shot and row["level"] == "image" and row["scope"] == "overall"
        )
        completed.append(
            {
                "shot": shot,
                "final_training_losses": history[-1],
                "image_overall": overall,
            }
        )
        print(f"    image overall AUROC={overall['auroc']:.4f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {"method": METHOD, "completed": completed}
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] results={output_root / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
