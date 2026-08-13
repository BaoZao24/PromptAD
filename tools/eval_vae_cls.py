#!/usr/bin/env python
"""Evaluate a VAE reconstruction baseline on RF/spectrum CLS protocols."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
VAE_SRC = REPO_ROOT / "references" / "vae_ism_ano"
for path in (REPO_ROOT, VAE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model import VAE, vae_loss  # type: ignore  # noqa: E402
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
from utils.iad_per import iad_per_score  # noqa: E402
from utils.training_utils import setup_seed  # noqa: E402


class VAEPathDataset(Dataset):
    def __init__(self, samples, image_size: int):
        self.samples = list(samples)
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = load_sample_image(sample)
        return self.transform(image), int(sample["label"]), str(sample["path"]), str(sample["name"])


def safe_auc(labels, scores) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores) * 100.0)


def safe_metrics(labels, scores) -> dict[str, float]:
    labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores_np = np.asarray(scores, dtype=np.float64).reshape(-1)
    # FedJam retains its four semantic labels in saved payloads; the shared
    # headline task is benign (0) versus any jammer (>0).
    if np.unique(labels_np).size > 2:
        labels_np = (labels_np != 0).astype(np.int32)
    if labels_np.size == 0 or np.unique(labels_np).size < 2:
        return {
            "image_auroc": float("nan"),
            "image_auprc": float("nan"),
            "fpr95": float("nan"),
        }
    fpr, tpr, _ = roc_curve(labels_np, scores_np)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "image_auroc": float(roc_auc_score(labels_np, scores_np) * 100.0),
        "image_auprc": float(average_precision_score(labels_np, scores_np) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if reached.size else float("nan"),
    }


def method_name(args) -> str:
    return "iad_per" if args.score_mode == "iad_per" else "vae_reconstruction"


def train_signature(job):
    return tuple(str(sample["path"]) for sample in job["train_samples"])


def make_model(args, device):
    return VAE(
        input_size=args.image_size,
        in_channels=1,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        fc_dim=args.fc_dim,
        decoder_fc_dim=args.decoder_fc_dim,
    ).to(device)


def make_loader(samples, args, shuffle: bool):
    dataset = VAEPathDataset(samples, args.image_size)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def fit_vae(train_samples, args, device):
    model = make_model(args, device)
    loader = make_loader(train_samples, args, shuffle=True)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    history = []
    for epoch in range(1, args.epochs + 1):
        losses = []
        for data, _label, _path, _name in loader:
            data = data.to(device).float()
            output, mean, logvar = model(data)
            loss = vae_loss(output, mean, logvar, data) / max(data.size(0), 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(epoch_loss)
        if args.log_every > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0):
            print(f"    epoch {epoch}/{args.epochs} loss={epoch_loss:.6f}", flush=True)
    return model.eval(), history


@torch.no_grad()
def predict_job(model, job, args, device):
    loader = make_loader(job["test_samples"], args, shuffle=False)
    labels, scores, paths, names = [], [], [], []
    for data, label, path, name in loader:
        data = data.to(device).float()
        output, _mean, _logvar = model(data)
        diff = (output - data) ** 2
        if args.score_mode == "iad_per":
            batch_scores = iad_per_score(
                data,
                output,
                alpha=args.per_alpha,
                gamma=args.per_gamma,
                background_weight=args.per_background_weight,
                xi=args.per_background_percentile,
                eta=args.per_signal_percentile,
            )
        elif args.score_mode == "mse_mean":
            batch_scores = diff.reshape(data.size(0), -1).mean(dim=1)
        elif args.score_mode == "mse_sum":
            batch_scores = diff.reshape(data.size(0), -1).sum(dim=1)
        elif args.score_mode == "mae_mean":
            batch_scores = (output - data).abs().reshape(data.size(0), -1).mean(dim=1)
        else:
            raise ValueError(f"Unsupported score mode: {args.score_mode}")
        labels.extend(int(x) for x in label.numpy().tolist())
        scores.extend(float(x) for x in batch_scores.cpu().numpy().tolist())
        paths.extend(str(x) for x in path)
        names.extend(str(x) for x in name)

    labels_np = np.asarray(labels, dtype=np.int32)
    scores_np = np.asarray(scores, dtype=np.float32)
    score_dir = Path(args.output_root) / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{job['dataset']}-{job['category']}-{job['scene']}-{job['jsr']}".replace("/", "_")
    np.savez_compressed(
        score_dir / f"{stem}-scores.npz",
        scores=scores_np,
        labels=labels_np,
        image_paths=np.asarray(paths),
        names=np.asarray(names),
    )
    return {
        "method": method_name(args),
        "dataset": job["dataset"],
        "category": job["category"],
        "scene": job["scene"],
        "jsr": job["jsr"],
        "num_train_normal": len(job["train_samples"]),
        "num_test_normal": int((labels_np == 0).sum()),
        "num_test_abnormal": int((labels_np == 1).sum()),
        **safe_metrics(labels, scores),
    }


def run_jobs(jobs, args, device):
    rows = []
    groups = {}
    for idx, job in enumerate(jobs):
        groups.setdefault(train_signature(job), []).append((idx, job))

    for group_idx, grouped in enumerate(groups.values(), 1):
        first_job = grouped[0][1]
        if not first_job["train_samples"]:
            raise RuntimeError("No train samples for grouped VAE run")
        print(
            f"[fit {group_idx}/{len(groups)}] train_normals={len(first_job['train_samples'])} "
            f"eval_jobs={len(grouped)}",
            flush=True,
        )
        model, history = fit_vae(first_job["train_samples"], args, device)
        for idx, job in grouped:
            if not any(sample["label"] == 1 for sample in job["test_samples"]):
                raise RuntimeError(f"No abnormal samples for {job['dataset']} {job['category']} {job['scene']} {job['jsr']}")
            print(f"[{idx + 1}/{len(jobs)}] {job['dataset']} {job['category']} {job['scene']} {job['jsr']}", flush=True)
            row = predict_job(model, job, args, device)
            row["train_loss_final"] = history[-1] if history else ""
            print(f"  image_auroc={row['image_auroc']:.4f}", flush=True)
            rows.append(row)
    rows.sort(key=lambda r: (r["dataset"], r["category"], r["scene"], r["jsr"]))
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_average(rows):
    out = list(rows)
    avg = {key: "" for key in rows[0].keys()}
    avg.update(
        {
            "method": rows[0]["method"],
            "dataset": rows[0]["dataset"],
            "category": "average",
            "scene": "macro",
            "jsr": "macro",
            "num_train_normal": "",
            "num_test_normal": "",
            "num_test_abnormal": "",
            "image_auroc": float(np.mean([row["image_auroc"] for row in rows])),
            "image_auprc": float(np.mean([row["image_auprc"] for row in rows])),
            "fpr95": float(np.mean([row["fpr95"] for row in rows])),
        }
    )
    out.append(avg)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        choices=["spectrum", "public_rf", "rf_target", "ofdma"],
        required=True,
    )
    parser.add_argument("--output-root", default="analysis_outputs/vae_cls")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--normal-sampling", choices=NORMAL_SAMPLING_CHOICES, default="per_frequency")
    parser.add_argument(
        "--support-manifest",
        default="",
        help="Shared target-scene support manifest for the rf_target protocol.",
    )
    parser.add_argument("--rf-train-mode", choices=["pooled", "per_cell"], default="pooled")
    parser.add_argument("--rf-signals", nargs="+", default=list(JSR_BY_SIGNAL.keys()), choices=list(JSR_BY_SIGNAL.keys()))
    parser.add_argument("--public-rf-signals", nargs="+", default=list(PUBLIC_RF_JSRS), choices=list(PUBLIC_RF_JSRS))
    parser.add_argument("--rf-scenes", nargs="+", default=SCENES, choices=SCENES)
    parser.add_argument("--spectrum-root", default=str(SPECTRUM_ROOT))
    parser.add_argument("--spectrum-categories", nargs="+", default=list(SPECTRUM_CATEGORIES), choices=list(SPECTRUM_CATEGORIES))
    parser.add_argument("--max-pooled-train-normals", type=int, default=0)
    parser.add_argument("--max-train-normals", type=int, default=0)
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--fc-dim", type=int, default=1024)
    parser.add_argument("--decoder-fc-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--score-mode",
        choices=["mse_mean", "mse_sum", "mae_mean", "iad_per"],
        default="mse_mean",
    )
    parser.add_argument("--per-alpha", type=float, default=0.05)
    parser.add_argument("--per-gamma", type=int, default=3)
    parser.add_argument("--per-background-weight", type=float, default=2.0)
    parser.add_argument("--per-background-percentile", type=float, default=90.0)
    parser.add_argument("--per-signal-percentile", type=float, default=99.0)
    parser.add_argument("--log-every", type=int, default=25)
    add_ofdma_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda:0")

    if args.protocol == "spectrum":
        jobs = spectrum_jobs(args)
    elif args.protocol == "public_rf":
        jobs = public_rf_jobs(args)
    elif args.protocol == "ofdma":
        jobs = ofdma_jobs(args)
    else:
        jobs = rf_target_jobs(args)

    rows = run_jobs(jobs, args, device)
    rows_with_avg = append_average(rows)
    out_root = Path(args.output_root)
    result_path = out_root / "results_vae_cls.csv"
    write_csv(result_path, rows_with_avg)
    summary = {
        "method": method_name(args),
        "protocol": args.protocol,
        "support_protocol": "target_scene" if args.protocol == "rf_target" else None,
        "support_manifest": getattr(args, "support_manifest", "") or None,
        "support_manifest_sha256": getattr(args, "support_manifest_sha256", None),
        "test_normal_paths_sha256": getattr(args, "test_normal_paths_sha256", None),
        "support_policy": getattr(args, "support_policy", None),
        "per_frequency_k": getattr(args, "per_frequency_k", None),
        "normal_sampling": args.normal_sampling,
        "rf_train_mode": getattr(args, "rf_train_mode", None),
        "image_size": args.image_size,
        "base_channels": args.base_channels,
        "latent_dim": args.latent_dim,
        "fc_dim": args.fc_dim,
        "decoder_fc_dim": args.decoder_fc_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "score_mode": args.score_mode,
        "per": (
            {
                "alpha": args.per_alpha,
                "gamma": args.per_gamma,
                "background_weight": args.per_background_weight,
                "background_percentile": args.per_background_percentile,
                "signal_percentile": args.per_signal_percentile,
            }
            if args.score_mode == "iad_per"
            else None
        ),
        "seed": args.seed,
        "num_jobs": len(rows),
        "image_auroc_macro": float(np.mean([row["image_auroc"] for row in rows])),
        "image_auprc_macro": float(np.mean([row["image_auprc"] for row in rows])),
        "fpr95_macro": float(np.mean([row["fpr95"] for row in rows])),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_root / "README.md").write_text(
        f"# {method_name(args)} CLS Baseline\n\n"
        f"Protocol: `{args.protocol}`\n\n"
        f"Normal sampling: `{args.normal_sampling}`\n\n"
        f"Result CSV: `{result_path.name}`\n\n"
        f"Macro Image-AUROC: `{summary['image_auroc_macro']:.4f}`\n",
        encoding="utf-8",
    )
    print(f"wrote {result_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
