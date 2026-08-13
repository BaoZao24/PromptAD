#!/usr/bin/env python
"""Evaluate a Deep SVDD baseline on RF/spectrum CLS protocols."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
DEEPSVDD_SRC = REPO_ROOT / "references" / "Deep-SVDD-PyTorch"
for path in (REPO_ROOT, DEEPSVDD_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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


class ImagePathDataset(Dataset):
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


class SmallSVDDNet(nn.Module):
    """Bias-free compact encoder for one-class Deep SVDD on spectrograms."""

    def __init__(self, image_size: int, rep_dim: int = 128, base_channels: int = 32):
        super().__init__()
        self.rep_dim = rep_dim
        self.features = nn.Sequential(
            nn.Conv2d(1, base_channels, 5, padding=2, bias=False),
            nn.GroupNorm(4, base_channels, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, 5, padding=2, bias=False),
            nn.GroupNorm(8, base_channels * 2, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 5, padding=2, bias=False),
            nn.GroupNorm(8, base_channels * 4, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, image_size, image_size)
            flat_dim = int(np.prod(self.features(dummy).shape[1:]))
        self.fc = nn.Linear(flat_dim, rep_dim, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)


def safe_auc(labels, scores) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores) * 100.0)


def train_signature(job):
    return tuple(str(sample["path"]) for sample in job["train_samples"])


def make_loader(samples, args, shuffle: bool):
    dataset = ImagePathDataset(samples, args.image_size)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def init_center(loader, model, device, eps: float = 0.1):
    model.eval()
    outputs = []
    for data, _label, _path, _name in loader:
        data = data.to(device).float()
        outputs.append(model(data))
    if not outputs:
        raise RuntimeError("No train samples for Deep SVDD center initialization")
    center = torch.cat(outputs, dim=0).mean(dim=0)
    center[(center.abs() < eps) & (center < 0)] = -eps
    center[(center.abs() < eps) & (center > 0)] = eps
    return center.detach()


def fit_deepsvdd(train_samples, args, device):
    model = SmallSVDDNet(args.image_size, args.rep_dim, args.base_channels).to(device)
    loader = make_loader(train_samples, args, shuffle=True)
    center = init_center(loader, model, device, eps=args.center_eps)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for data, _label, _path, _name in loader:
            data = data.to(device).float()
            output = model(data)
            dist = torch.sum((output - center) ** 2, dim=1)
            loss = torch.mean(dist)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(epoch_loss)
        if args.log_every > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0):
            print(f"    epoch {epoch}/{args.epochs} loss={epoch_loss:.6f}", flush=True)
    return model.eval(), center, history


@torch.no_grad()
def predict_job(model, center, job, args, device):
    loader = make_loader(job["test_samples"], args, shuffle=False)
    labels, scores, paths, names = [], [], [], []
    for data, label, path, name in loader:
        data = data.to(device).float()
        output = model(data)
        batch_scores = torch.sum((output - center) ** 2, dim=1)
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
        "method": "deep_svdd",
        "dataset": job["dataset"],
        "category": job["category"],
        "scene": job["scene"],
        "jsr": job["jsr"],
        "num_train_normal": len(job["train_samples"]),
        "num_test_normal": int((labels_np == 0).sum()),
        "num_test_abnormal": int((labels_np == 1).sum()),
        "image_auroc": safe_auc(labels, scores),
    }


def run_jobs(jobs, args, device):
    rows = []
    groups = {}
    for idx, job in enumerate(jobs):
        groups.setdefault(train_signature(job), []).append((idx, job))

    for group_idx, grouped in enumerate(groups.values(), 1):
        first_job = grouped[0][1]
        if not first_job["train_samples"]:
            raise RuntimeError("No train samples for grouped Deep SVDD run")
        print(
            f"[fit {group_idx}/{len(groups)}] train_normals={len(first_job['train_samples'])} "
            f"eval_jobs={len(grouped)}",
            flush=True,
        )
        model, center, history = fit_deepsvdd(first_job["train_samples"], args, device)
        for idx, job in grouped:
            if not any(sample["label"] == 1 for sample in job["test_samples"]):
                raise RuntimeError(f"No abnormal samples for {job['dataset']} {job['category']} {job['scene']} {job['jsr']}")
            print(f"[{idx + 1}/{len(jobs)}] {job['dataset']} {job['category']} {job['scene']} {job['jsr']}", flush=True)
            row = predict_job(model, center, job, args, device)
            row["train_loss_final"] = history[-1] if history else ""
            print(f"  image_auroc={row['image_auroc']:.4f}", flush=True)
            rows.append(row)
    rows.sort(key=lambda r: (r["dataset"], r["category"], r["scene"], r["jsr"]))
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_average(rows):
    out = list(rows)
    avg = {key: "" for key in rows[0].keys()}
    avg.update(
        {
            "method": "deep_svdd",
            "dataset": rows[0]["dataset"],
            "category": "average",
            "scene": "macro",
            "jsr": "macro",
            "num_train_normal": "",
            "num_test_normal": "",
            "num_test_abnormal": "",
            "image_auroc": float(np.mean([row["image_auroc"] for row in rows])),
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
    parser.add_argument("--output-root", default="analysis_outputs/deepsvdd_cls")
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
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--rep-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--center-eps", type=float, default=0.1)
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
    result_path = out_root / "results_deepsvdd_cls.csv"
    write_csv(result_path, rows_with_avg)
    summary = {
        "method": "deep_svdd",
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
        "rep_dim": args.rep_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "num_jobs": len(rows),
        "image_auroc_macro": float(np.mean([row["image_auroc"] for row in rows])),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_root / "README.md").write_text(
        "# Deep SVDD CLS Baseline\n\n"
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
