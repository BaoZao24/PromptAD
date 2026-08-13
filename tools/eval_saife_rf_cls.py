#!/usr/bin/env python
"""Evaluate SAIFE reconstruction on the self-RF and public-RF protocols.

The evaluator reuses the project-standard normal support/test job builders. It
trains only on normal support images and writes one reconstruction-MSE score per
test image; test labels are read only after scoring to compute metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_patchcore_cls import (  # noqa: E402
    JSR_BY_SIGNAL,
    PUBLIC_RF_JSRS,
    SCENES,
    public_rf_jobs,
    rf_target_jobs,
)
from tools.eval_saife_ofdma_fewshot import SAIFEAAE, safe_metrics  # noqa: E402
from tools.ofdma_fewshot_baseline_common import load_sample_image  # noqa: E402
from utils.training_utils import setup_seed  # noqa: E402


METHOD = "saife_reconstruction"
FEWSHOT_SAMPLING_CHOICES = ("per_frequency", "1shot", "2shot", "4shot")


class SAIFERFPathDataset(Dataset):
    def __init__(self, samples: list[dict], input_shape: tuple[int, int]):
        self.samples = list(samples)
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(input_shape),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self.transform(load_sample_image(sample))
        return image, int(sample["label"]), str(sample["path"]), str(sample["name"])


def input_shape(args) -> tuple[int, int]:
    return int(args.image_height), int(args.image_width)


def make_loader(samples: list[dict], args, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        SAIFERFPathDataset(samples, input_shape(args)),
        batch_size=args.batch_size if shuffle else args.test_batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def fit_saife(train_samples: list[dict], args, device: torch.device):
    if not train_samples or any(int(sample["label"]) != 0 for sample in train_samples):
        raise RuntimeError("SAIFE support data must be non-empty and normal-only")

    model = SAIFEAAE(
        args.latent_dim,
        args.hidden_dim,
        args.dropout,
        input_shape=input_shape(args),
    ).to(device)
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
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        reconstruction_losses = []
        discriminator_losses = []
        generator_losses = []
        for data, _label, _path, _name in loader:
            data = data.to(device, non_blocking=True)

            reconstruction, _ = model.reconstruct(data)
            reconstruction_loss = F.mse_loss(
                reconstruction, data, reduction="none"
            ).flatten(1).sum(dim=1).mean()
            reconstruction_optimizer.zero_grad(set_to_none=True)
            reconstruction_loss.backward()
            reconstruction_optimizer.step()

            set_requires_grad(model.discriminator, True)
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

            set_requires_grad(model.discriminator, False)
            encoded = model.encoder(data)
            fake_logits = model.discriminator(encoded)
            generator_loss = bce(fake_logits, torch.ones_like(fake_logits))
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
        if args.log_every > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0):
            print(
                f"    epoch {epoch}/{args.epochs} reconstruction={row['reconstruction_loss']:.6f} "
                f"discriminator={row['discriminator_loss']:.6f} generator={row['generator_loss']:.6f}",
                flush=True,
            )
    return model.eval(), history


@torch.no_grad()
def predict_job(model: SAIFEAAE, job: dict, args, device: torch.device) -> dict:
    loader = make_loader(job["test_samples"], args, shuffle=False, device=device)
    labels, scores, paths, names = [], [], [], []
    model.eval()
    for data, batch_labels, batch_paths, batch_names in loader:
        data = data.to(device, non_blocking=True)
        reconstruction = model(data)
        batch_scores = F.mse_loss(
            reconstruction, data, reduction="none"
        ).flatten(1).mean(dim=1)
        labels.extend(int(value) for value in batch_labels.numpy().tolist())
        scores.extend(float(value) for value in batch_scores.cpu().numpy().tolist())
        paths.extend(str(value) for value in batch_paths)
        names.extend(str(value) for value in batch_names)

    labels_np = np.asarray(labels, dtype=np.int32)
    scores_np = np.asarray(scores, dtype=np.float32)
    if not np.all(np.isfinite(scores_np)):
        raise RuntimeError(
            f"Non-finite scores for {job['dataset']} {job['category']} {job['scene']} {job['jsr']}"
        )
    metrics = safe_metrics(labels_np, scores_np)
    score_dir = Path(args.output_root) / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{job['dataset']}-{job['category']}-{job['scene']}-{job['jsr']}".replace("/", "_")
    np.savez_compressed(
        score_dir / f"{stem}-scores.npz",
        scores=scores_np,
        labels=labels_np,
        image_paths=np.asarray(paths),
        names=np.asarray(names),
        **(
            {"support_manifest_sha256": np.asarray(args.support_manifest_sha256)}
            if getattr(args, "support_manifest_sha256", "")
            else {}
        ),
    )
    return {
        "method": METHOD,
        "dataset": job["dataset"],
        "category": job["category"],
        "scene": job["scene"],
        "jsr": job["jsr"],
        "num_train_normal": len(job["train_samples"]),
        "num_test_normal": int((labels_np == 0).sum()),
        "num_test_abnormal": int((labels_np == 1).sum()),
        "image_auroc": metrics["auroc"],
        "image_auprc": metrics["auprc"],
        "fpr95": metrics["fpr95"],
    }


def train_signature(job: dict) -> tuple[str, ...]:
    return tuple(str(sample["path"]) for sample in job["train_samples"])


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
    }


def run_jobs(jobs: list[dict], args, device: torch.device) -> list[dict]:
    grouped_jobs: dict[tuple[str, ...], list[tuple[int, dict]]] = {}
    for index, job in enumerate(jobs):
        grouped_jobs.setdefault(train_signature(job), []).append((index, job))

    rows = []
    checkpoint_root = Path(args.output_root) / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for group_index, (signature, grouped) in enumerate(grouped_jobs.items(), 1):
        first_job = grouped[0][1]
        if not signature:
            raise RuntimeError("No normal support images for grouped SAIFE run")
        print(
            f"[fit {group_index}/{len(grouped_jobs)}] train_normals={len(signature)} "
            f"eval_jobs={len(grouped)}",
            flush=True,
        )
        setup_seed(args.seed)
        model, history = fit_saife(first_job["train_samples"], args, device)
        (Path(args.output_root) / f"training_history_group{group_index:03d}.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        torch.save(
            {
                "model": model.state_dict(),
                "support_paths": signature,
                "model_config": {
                    "input_shape": input_shape(args),
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                },
            },
            checkpoint_root / f"saife_group{group_index:03d}.pt",
        )
        for original_index, job in grouped:
            if not any(int(sample["label"]) == 1 for sample in job["test_samples"]):
                raise RuntimeError(
                    f"No anomalies for {job['dataset']} {job['category']} {job['scene']} {job['jsr']}"
                )
            print(
                f"[{original_index + 1}/{len(jobs)}] {job['dataset']} "
                f"{job['category']} {job['scene']} {job['jsr']}",
                flush=True,
            )
            row = predict_job(model, job, args, device)
            row["train_reconstruction_loss_final"] = history[-1]["reconstruction_loss"]
            rows.append(row)
            print(f"    image AUROC={row['image_auroc']:.4f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return sorted(rows, key=lambda row: (row["category"], row["scene"], row["jsr"]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normal-only SAIFE AAE on self-RF or public-RF"
    )
    parser.add_argument("--protocol", choices=["rf_target", "public_rf"], required=True)
    parser.add_argument("--output-root", default="analysis_outputs/saife_rf_cls")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument(
        "--normal-sampling",
        choices=FEWSHOT_SAMPLING_CHOICES,
        default="per_frequency",
    )
    parser.add_argument(
        "--support-manifest",
        default="",
        help="Shared target-scene support manifest for the rf_target protocol.",
    )
    parser.add_argument("--rf-train-mode", choices=["pooled", "per_cell"], default="pooled")
    parser.add_argument(
        "--rf-signals",
        nargs="+",
        default=list(JSR_BY_SIGNAL),
        choices=list(JSR_BY_SIGNAL),
    )
    parser.add_argument(
        "--public-rf-signals",
        nargs="+",
        default=list(PUBLIC_RF_JSRS),
        choices=list(PUBLIC_RF_JSRS),
    )
    parser.add_argument("--rf-scenes", nargs="+", default=SCENES, choices=SCENES)
    parser.add_argument("--max-pooled-train-normals", type=int, default=0)
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--support-bootstrap-seed", type=int, default=-1)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--discriminator-lr", type=float, default=2.5e-5)
    parser.add_argument("--adversarial-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate_args(args) -> None:
    if args.image_height < 8 or args.image_width < 8:
        raise ValueError("image dimensions must be at least 8 for three pooling stages")
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

    jobs = public_rf_jobs(args) if args.protocol == "public_rf" else rf_target_jobs(args)
    if not jobs:
        raise RuntimeError(f"No jobs built for {args.protocol}")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    groups = {}
    for job in jobs:
        groups.setdefault(train_signature(job), []).append(
            f"{job['category']}:{job['scene']}:{job['jsr']}"
        )
    support_manifest = []
    for group_index, (signature, cells) in enumerate(groups.items(), 1):
        support_manifest.append(
            {
                "group": group_index,
                "num_normal_images": len(signature),
                "support_paths": list(signature),
                "evaluation_cells": cells,
            }
        )
    (output_root / "support_manifest.json").write_text(
        json.dumps(support_manifest, indent=2) + "\n", encoding="utf-8"
    )

    probe = SAIFERFPathDataset([jobs[0]["train_samples"][0]], input_shape(args))[0][0]
    if tuple(probe.shape) != (1, *input_shape(args)):
        raise RuntimeError(f"RF preprocessing probe failed: {tuple(probe.shape)}")
    protocol = {
        "method": METHOD,
        "implementation": "shared SAIFE PyTorch CNN adversarial-autoencoder core",
        "reference_source": str((REPO_ROOT / "references" / "saife").resolve()),
        "protocol": args.protocol,
        "support_protocol": "target_scene" if args.protocol == "rf_target" else None,
        "support_manifest": getattr(args, "support_manifest", "") or None,
        "support_manifest_sha256": getattr(args, "support_manifest_sha256", None),
        "normal_sampling": args.normal_sampling,
        "rf_train_mode": args.rf_train_mode if args.protocol == "rf_target" else None,
        "input_shape": [1, *input_shape(args)],
        "input_conversion": "grayscale_resize_to_unit_interval",
        "num_train_groups": len(groups),
        "num_evaluation_cells": len(jobs),
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
        "full_normal_sampling_supported": False,
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2), flush=True)
    if args.validate_only:
        print("[validated] jobs, normal support grouping, and RF preprocessing", flush=True)
        return

    rows = run_jobs(jobs, args, device)
    macro = macro_row(rows)
    write_csv(output_root / "results_saife_cls.csv", rows + [macro])
    summary = {
        "method": METHOD,
        "protocol": args.protocol,
        "support_protocol": "target_scene" if args.protocol == "rf_target" else None,
        "support_manifest": getattr(args, "support_manifest", "") or None,
        "support_manifest_sha256": getattr(args, "support_manifest_sha256", None),
        "normal_sampling": args.normal_sampling,
        "rf_train_mode": args.rf_train_mode if args.protocol == "rf_target" else None,
        "num_jobs": len(rows),
        "num_train_groups": len(groups),
        "image_auroc_macro": macro["image_auroc"],
        "image_auprc_macro": macro["image_auprc"],
        "fpr95_macro": macro["fpr95"],
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] results={output_root / 'results_saife_cls.csv'}", flush=True)


if __name__ == "__main__":
    main()
