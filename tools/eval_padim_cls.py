#!/usr/bin/env python
"""Evaluate a PaDiM-style baseline on RF/spectrum CLS protocols."""

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
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    def __init__(self, samples, resize: int, imagesize: int):
        self.samples = list(samples)
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = load_sample_image(sample)
        return self.transform(image), int(sample["label"]), str(sample["path"]), str(sample["name"])


class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self, weights: str):
        super().__init__()
        if weights == "default":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif weights == "none":
            backbone = models.resnet18(weights=None)
        else:
            raise ValueError(f"Unsupported weights: {weights}")
        self.stem = torch.nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)


def safe_auc(labels, scores) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores) * 100.0)


def train_signature(job):
    return tuple(str(sample["path"]) for sample in job["train_samples"])


def make_loader(samples, args, shuffle: bool):
    dataset = ImagePathDataset(samples, args.resize, args.imagesize)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def make_feature_ids(num_channels: int, target_dim: int, seed: int):
    if target_dim <= 0 or target_dim >= num_channels:
        return torch.arange(num_channels)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_channels, generator=generator)[:target_dim].sort().values


@torch.no_grad()
def extract_embeddings(model, samples, args, device):
    loader = make_loader(samples, args, shuffle=False)
    embeddings, labels, paths, names = [], [], [], []
    model.eval()
    feature_ids = None
    for images, batch_labels, batch_paths, batch_names in loader:
        images = images.to(device)
        feats = model(images).cpu()
        if feature_ids is None:
            feature_ids = make_feature_ids(feats.shape[1], args.embedding_dim, args.seed)
        feats = feats[:, feature_ids]
        embeddings.append(feats)
        labels.extend(int(x) for x in batch_labels.numpy().tolist())
        paths.extend(str(x) for x in batch_paths)
        names.extend(str(x) for x in batch_names)
    if not embeddings:
        raise RuntimeError("No samples for PaDiM feature extraction")
    return torch.cat(embeddings, dim=0).numpy(), np.asarray(labels, dtype=np.int32), paths, names


def fit_padim(model, train_samples, args, device):
    embeddings, _labels, _paths, _names = extract_embeddings(model, train_samples, args, device)
    mean = embeddings.mean(axis=0)
    var = embeddings.var(axis=0) + float(args.var_eps)
    return {"mean": mean.astype(np.float32), "var": var.astype(np.float32)}


def score_embeddings(embeddings, stats):
    diff = embeddings - stats["mean"][None, ...]
    dist = np.sqrt(np.sum((diff * diff) / stats["var"][None, ...], axis=1))
    maps = dist.astype(np.float32)
    flat = maps.reshape(maps.shape[0], -1)
    return flat.max(axis=1).astype(np.float32), maps


def predict_job(model, stats, job, args, device):
    maps = None
    if args.protocol == "ofdma":
        loader = make_loader(job["test_samples"], args, shuffle=False)
        score_parts = []
        labels, paths, names = [], [], []
        feature_ids = None
        model.eval()
        with torch.no_grad():
            for images, batch_labels, batch_paths, batch_names in loader:
                feats = model(images.to(device)).cpu()
                if feature_ids is None:
                    feature_ids = make_feature_ids(
                        feats.shape[1],
                        args.embedding_dim,
                        args.seed,
                    )
                batch_scores, _ = score_embeddings(feats[:, feature_ids].numpy(), stats)
                score_parts.append(batch_scores)
                labels.extend(int(value) for value in batch_labels.numpy().tolist())
                paths.extend(str(value) for value in batch_paths)
                names.extend(str(value) for value in batch_names)
        if not score_parts:
            raise RuntimeError("No OFDMA samples for PaDiM prediction")
        scores = np.concatenate(score_parts).astype(np.float32)
        labels = np.asarray(labels, dtype=np.int32)
    else:
        embeddings, labels, paths, names = extract_embeddings(
            model,
            job["test_samples"],
            args,
            device,
        )
        scores, maps = score_embeddings(embeddings, stats)

    score_dir = Path(args.output_root) / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{job['dataset']}-{job['category']}-{job['scene']}-{job['jsr']}".replace("/", "_")
    payload = {
        "scores": scores,
        "labels": labels,
        "image_paths": np.asarray(paths),
        "names": np.asarray(names),
    }
    if maps is not None:
        payload["anomaly_maps"] = maps
    np.savez_compressed(score_dir / f"{stem}-scores.npz", **payload)
    return {
        "method": "padim_diag_resnet18",
        "dataset": job["dataset"],
        "category": job["category"],
        "scene": job["scene"],
        "jsr": job["jsr"],
        "num_train_normal": len(job["train_samples"]),
        "num_test_normal": int((labels == 0).sum()),
        "num_test_abnormal": int((labels == 1).sum()),
        "image_auroc": safe_auc(labels.tolist(), scores.tolist()),
    }


def run_jobs(jobs, args, device):
    model = ResNetFeatureExtractor(args.weights).to(device).eval()
    rows = []
    groups = {}
    for idx, job in enumerate(jobs):
        groups.setdefault(train_signature(job), []).append((idx, job))

    for group_idx, grouped in enumerate(groups.values(), 1):
        first_job = grouped[0][1]
        if not first_job["train_samples"]:
            raise RuntimeError("No train samples for grouped PaDiM run")
        print(
            f"[fit {group_idx}/{len(groups)}] train_normals={len(first_job['train_samples'])} "
            f"eval_jobs={len(grouped)}",
            flush=True,
        )
        stats = fit_padim(model, first_job["train_samples"], args, device)
        for idx, job in grouped:
            if not any(sample["label"] == 1 for sample in job["test_samples"]):
                raise RuntimeError(f"No abnormal samples for {job['dataset']} {job['category']} {job['scene']} {job['jsr']}")
            print(f"[{idx + 1}/{len(jobs)}] {job['dataset']} {job['category']} {job['scene']} {job['jsr']}", flush=True)
            row = predict_job(model, stats, job, args, device)
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
            "method": "padim_diag_resnet18",
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
    parser.add_argument("--output-root", default="analysis_outputs/padim_cls")
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
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--weights", choices=["default", "none"], default="default")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--var-eps", type=float, default=0.01)
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
    result_path = out_root / "results_padim_cls.csv"
    write_csv(result_path, rows_with_avg)
    summary = {
        "method": "padim_diag_resnet18",
        "protocol": args.protocol,
        "support_protocol": "target_scene" if args.protocol == "rf_target" else None,
        "support_manifest": getattr(args, "support_manifest", "") or None,
        "support_manifest_sha256": getattr(args, "support_manifest_sha256", None),
        "test_normal_paths_sha256": getattr(args, "test_normal_paths_sha256", None),
        "support_policy": getattr(args, "support_policy", None),
        "per_frequency_k": getattr(args, "per_frequency_k", None),
        "normal_sampling": args.normal_sampling,
        "rf_train_mode": getattr(args, "rf_train_mode", None),
        "weights": args.weights,
        "embedding_dim": args.embedding_dim,
        "var_eps": args.var_eps,
        "resize": args.resize,
        "imagesize": args.imagesize,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "num_jobs": len(rows),
        "image_auroc_macro": float(np.mean([row["image_auroc"] for row in rows])),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_root / "README.md").write_text(
        "# PaDiM-style CLS Baseline\n\n"
        "Implementation: diagonal-covariance PaDiM-style ResNet18 feature baseline.\n\n"
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
