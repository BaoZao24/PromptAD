#!/usr/bin/env python
"""Evaluate an STFPM baseline on RF/spectrum CLS protocols."""

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
    SCENES,
    SPECTRUM_CATEGORIES,
    SPECTRUM_ROOT,
    public_rf_jobs,
    rf_target_jobs,
    spectrum_jobs,
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
        image = Image.open(sample["path"]).convert("RGB")
        return self.transform(image), int(sample["label"]), str(sample["path"]), str(sample["name"])


class ResNet18MS3(torch.nn.Module):
    def __init__(self, weights: str):
        super().__init__()
        if weights == "default":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif weights == "none":
            backbone = models.resnet18(weights=None)
        else:
            raise ValueError(f"Unsupported weights: {weights}")
        self.model = torch.nn.Sequential(*list(backbone.children())[:-2])

    def forward(self, x):
        features = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in {"4", "5", "6"}:
                features.append(x)
        return features


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


def feature_loss(teacher_features, student_features):
    loss = 0.0
    for teacher_feat, student_feat in zip(teacher_features, student_features):
        teacher_norm = F.normalize(teacher_feat, dim=1)
        student_norm = F.normalize(student_feat, dim=1)
        loss = loss + torch.sum((teacher_norm - student_norm) ** 2, dim=1).mean()
    return loss


def make_models(args, device):
    teacher = ResNet18MS3("default").to(device).eval()
    student = ResNet18MS3("none").to(device).train()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher, student


def fit_stfpm(train_samples, args, device):
    teacher, student = make_models(args, device)
    loader = make_loader(train_samples, args, shuffle=True)
    optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    history = []
    for epoch in range(1, args.epochs + 1):
        losses = []
        student.train()
        for images, _labels, _paths, _names in loader:
            images = images.to(device)
            with torch.no_grad():
                teacher_features = teacher(images)
            student_features = student(images)
            loss = feature_loss(teacher_features, student_features)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(epoch_loss)
        if args.log_every > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0):
            print(f"    epoch {epoch}/{args.epochs} loss={epoch_loss:.6f}", flush=True)
    return teacher.eval(), student.eval(), history


@torch.no_grad()
def predict_job(teacher, student, job, args, device):
    loader = make_loader(job["test_samples"], args, shuffle=False)
    labels, scores, paths, names, maps = [], [], [], [], []
    for images, batch_labels, batch_paths, batch_names in loader:
        images = images.to(device)
        teacher_features = teacher(images)
        student_features = student(images)
        score_map = None
        for teacher_feat, student_feat in zip(teacher_features, student_features):
            teacher_norm = F.normalize(teacher_feat, dim=1)
            student_norm = F.normalize(student_feat, dim=1)
            layer_map = torch.sum((teacher_norm - student_norm) ** 2, dim=1, keepdim=True)
            layer_map = F.interpolate(layer_map, size=(64, 64), mode="bilinear", align_corners=False)
            score_map = layer_map if score_map is None else score_map * layer_map
        batch_maps = score_map[:, 0].detach().cpu().numpy().astype(np.float32)
        batch_scores = batch_maps.reshape(batch_maps.shape[0], -1).max(axis=1)
        labels.extend(int(x) for x in batch_labels.numpy().tolist())
        scores.extend(float(x) for x in batch_scores.tolist())
        paths.extend(str(x) for x in batch_paths)
        names.extend(str(x) for x in batch_names)
        maps.append(batch_maps)

    labels_np = np.asarray(labels, dtype=np.int32)
    scores_np = np.asarray(scores, dtype=np.float32)
    maps_np = np.concatenate(maps, axis=0) if maps else np.zeros((0, 64, 64), dtype=np.float32)
    score_dir = Path(args.output_root) / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{job['dataset']}-{job['category']}-{job['scene']}-{job['jsr']}".replace("/", "_")
    np.savez_compressed(
        score_dir / f"{stem}-scores.npz",
        scores=scores_np,
        labels=labels_np,
        anomaly_maps=maps_np,
        image_paths=np.asarray(paths),
        names=np.asarray(names),
    )
    return {
        "method": "stfpm_resnet18",
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
            raise RuntimeError("No train samples for grouped STFPM run")
        print(
            f"[fit {group_idx}/{len(groups)}] train_normals={len(first_job['train_samples'])} "
            f"eval_jobs={len(grouped)}",
            flush=True,
        )
        teacher, student, history = fit_stfpm(first_job["train_samples"], args, device)
        for idx, job in grouped:
            if not any(sample["label"] == 1 for sample in job["test_samples"]):
                raise RuntimeError(f"No abnormal samples for {job['dataset']} {job['category']} {job['scene']} {job['jsr']}")
            print(f"[{idx + 1}/{len(jobs)}] {job['dataset']} {job['category']} {job['scene']} {job['jsr']}", flush=True)
            row = predict_job(teacher, student, job, args, device)
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
            "method": "stfpm_resnet18",
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
    parser.add_argument("--protocol", choices=["spectrum", "public_rf", "rf_target"], required=True)
    parser.add_argument("--output-root", default="analysis_outputs/stfpm_cls")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--normal-sampling", choices=NORMAL_SAMPLING_CHOICES, default="per_frequency")
    parser.add_argument("--rf-train-mode", choices=["pooled", "per_cell"], default="pooled")
    parser.add_argument("--rf-signals", nargs="+", default=list(JSR_BY_SIGNAL.keys()), choices=list(JSR_BY_SIGNAL.keys()))
    parser.add_argument("--rf-scenes", nargs="+", default=SCENES, choices=SCENES)
    parser.add_argument("--spectrum-root", default=str(SPECTRUM_ROOT))
    parser.add_argument("--spectrum-categories", nargs="+", default=list(SPECTRUM_CATEGORIES), choices=list(SPECTRUM_CATEGORIES))
    parser.add_argument("--max-pooled-train-normals", type=int, default=0)
    parser.add_argument("--max-train-normals", type=int, default=0)
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=25)
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
    else:
        jobs = rf_target_jobs(args)

    rows = run_jobs(jobs, args, device)
    rows_with_avg = append_average(rows)
    out_root = Path(args.output_root)
    result_path = out_root / "results_stfpm_cls.csv"
    write_csv(result_path, rows_with_avg)
    summary = {
        "method": "stfpm_resnet18",
        "protocol": args.protocol,
        "normal_sampling": args.normal_sampling,
        "rf_train_mode": getattr(args, "rf_train_mode", None),
        "epochs": args.epochs,
        "lr": args.lr,
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
        "# STFPM CLS Baseline\n\n"
        "Source idea: teacher-student feature pyramid matching with ResNet18.\n\n"
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
