#!/usr/bin/env python
"""Evaluate SPADE (kNN feature matching) on RF/spectrum CLS protocols.

Adapted from byungjae89/SPADE-pytorch (src/main.py): image-level score is the
mean distance to the k nearest neighbors in the pooled ResNet feature space.
No training is required; the support gallery is built from the (few-shot)
normal support images exactly as the other baselines do.
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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.rf_target import (
    RF_JSR_BY_SIGNAL as JSR_BY_SIGNAL,
    RF_SCENES as SCENES,
    collect_rf_target_samples as collect_samples,
)
from tools.eval_patchcore_cls import (
    PatchCorePathDataset,
    append_average,
    public_rf_jobs,
    rf_target_jobs,
    safe_auc,
    spectrum_jobs,
)
from tools.ofdma_fewshot_baseline_common import add_ofdma_args, ofdma_jobs
from utils.training_utils import setup_seed


class SpadeGallery:
    """Feature gallery + kNN scorer (official SPADE protocol).

    Image-level scoring only needs the pooled (avgpool) features; the
    layer1/2/3 hook outputs are required by the original SPADE pixel-level
    scoring and are only kept when ``with_pixel_level=True``.
    """

    def __init__(self, model, device, top_k: int = 5, with_pixel_level: bool = False):
        self.model = model
        self.device = device
        self.top_k = top_k
        self.outputs = []
        hooks = [
            model.layer1[-1],
            model.layer2[-1],
            model.layer3[-1],
            model.avgpool,
        ]
        self.with_pixel_level = with_pixel_level
        for module in hooks:
            module.register_forward_hook(self._hook)
        self.gallery = None

    def _hook(self, module, input, output):
        self.outputs.append(output)

    @torch.no_grad()
    def extract(self, dataloader):
        if self.with_pixel_level:
            per_layer = [list() for _ in range(4)]
            for batch in dataloader:
                if isinstance(batch, dict):
                    x = batch["image"]
                else:
                    x, _y, _mask = batch[:3]
                self.outputs = []
                self.model(x.to(self.device))
                for index, feature in enumerate(self.outputs):
                    per_layer[index].append(feature.detach().cpu())
            return [torch.cat(values, 0) for values in per_layer]
        pooled = []
        for batch in dataloader:
            if isinstance(batch, dict):
                x = batch["image"]
            else:
                x, _y, _mask = batch[:3]
            self.outputs = []
            self.model(x.to(self.device))
            pooled.append(self.outputs[-1].detach().cpu())
        return [None, None, None, torch.cat(pooled, 0)]

    def fit(self, support_features):
        self.gallery = support_features

    @torch.no_grad()
    def image_scores(self, test_features):
        query = torch.flatten(test_features[3], 1)
        gallery = torch.flatten(self.gallery[3], 1)
        k = min(self.top_k, gallery.size(0))
        if k < 1:
            raise RuntimeError("SPADE gallery is empty")
        block = 512
        values = []
        for start in range(0, query.size(0), block):
            dist = self._calc_dist_matrix(query[start:start + block], gallery)
            topk_values, _ = torch.topk(dist, k=k, dim=1, largest=False)
            values.append(torch.mean(topk_values, 1).cpu().numpy())
        return np.concatenate(values)

    def _calc_dist_matrix(self, x, y):
        n, m = x.size(0), y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return torch.sqrt(torch.pow(x - y, 2).sum(2))


def predict_job(gallery_model, job, args):
    dataset = PatchCorePathDataset(job["test_samples"], args.resize, args.imagesize)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    features = gallery_model.extract(dataloader)
    scores = gallery_model.image_scores(features)
    labels = np.asarray([int(sample["label"]) for sample in job["test_samples"]], dtype=np.int32)

    score_dir = Path(args.output_root) / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{job['dataset']}-{job['category']}-{job['scene']}-{job['jsr']}".replace("/", "_")
    score_payload = {
        "scores": scores,
        "labels": labels,
        "image_paths": np.asarray([str(s["path"]) for s in job["test_samples"]]),
    }
    if getattr(args, "support_manifest_sha256", ""):
        score_payload["support_manifest_sha256"] = np.asarray(args.support_manifest_sha256)
    np.savez_compressed(score_dir / f"{stem}-scores.npz", **score_payload)
    return {
        "method": "spade",
        "dataset": job["dataset"],
        "category": job["category"],
        "scene": job["scene"],
        "jsr": job["jsr"],
        "num_train_normal": len(job["train_samples"]),
        "num_test_normal": int((labels == 0).sum()),
        "num_test_abnormal": int((labels == 1).sum()),
        "image_auroc": safe_auc(labels.tolist(), scores.tolist()),
    }


def fit_spade(train_samples, args, device):
    """Build the SPADE support gallery (mirrors eval_patchcore_cls.fit_patchcore)."""
    train_dataset = PatchCorePathDataset(train_samples, args.resize, args.imagesize)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model = models.wide_resnet50_2(pretrained=True).to(device).eval()
    gallery_model = SpadeGallery(model, device, top_k=args.top_k, with_pixel_level=args.pixel_level)
    gallery_model.fit(gallery_model.extract(train_loader))
    return gallery_model


def run_jobs(jobs, args, device):
    rows = []
    groups = {}
    for idx, job in enumerate(jobs):
        groups.setdefault(tuple(str(s["path"]) for s in job["train_samples"]), []).append((idx, job))

    for group_idx, grouped in enumerate(groups.values(), 1):
        first_job = grouped[0][1]
        if not first_job["train_samples"]:
            raise RuntimeError("No train samples for grouped SPADE run")
        print(
            f"[gallery {group_idx}/{len(groups)}] "
            f"support_normals={len(first_job['train_samples'])} eval_jobs={len(grouped)}"
        )
        gallery_model = fit_spade(first_job["train_samples"], args, device)

        for idx, job in grouped:
            if not any(s["label"] == 1 for s in job["test_samples"]):
                raise RuntimeError(f"No abnormal samples for {job['dataset']} {job['category']} {job['scene']} {job['jsr']}")
            print(f"[{idx + 1}/{len(jobs)}] {job['dataset']} {job['category']} {job['scene']} {job['jsr']}")
            row = predict_job(gallery_model, job, args)
            print(f"  image_auroc={row['image_auroc']:.4f}")
            rows.append(row)
    rows.sort(key=lambda r: (r["dataset"], r["category"], r["scene"], r["jsr"]))
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["spectrum", "public_rf", "rf_target", "ofdma"], required=True)
    parser.add_argument("--output-root", default="analysis_outputs/20260817_spade_cls")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pixel-level", action="store_true", help="Also keep layer1/2/3 features for pixel-level scoring (much slower).")
    parser.add_argument("--normal-sampling", default="per_frequency")
    parser.add_argument("--support-manifest", default="")
    parser.add_argument("--max-train-normals", type=int, default=0)
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--spectrum-root", default="datasets/spectrum")
    parser.add_argument("--spectrum-categories", nargs="+", default=["16QAM", "CHIRP", "GMSK", "QPSK"])
    parser.add_argument("--rf-signals", nargs="+", default=list(JSR_BY_SIGNAL.keys()), choices=list(JSR_BY_SIGNAL.keys()))
    parser.add_argument("--public-rf-signals", nargs="+", default=["burst", "chirp", "dsss", "pulse", "deceptive"])
    parser.add_argument("--rf-scenes", nargs="+", default=SCENES, choices=SCENES)
    parser.add_argument("--map-only-cell", nargs="*", default=[])
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

    if args.map_only_cell:
        wanted = set(args.map_only_cell)
        jobs = [job for job in jobs if f"{job['category']}:{job['scene']}:{job['jsr']}" in wanted]
        if not jobs:
            raise RuntimeError(f"No requested map cells found: {sorted(wanted)}")

    rows = run_jobs(jobs, args, device)
    rows_with_avg = append_average(rows)
    out_root = Path(args.output_root)
    result_path = out_root / "results_spade_cls.csv"
    write_csv(result_path, rows_with_avg)
    summary = {
        "method": "spade",
        "protocol": args.protocol,
        "top_k": args.top_k,
        "backbone": "wide_resnet50_2",
        "support_manifest": getattr(args, "support_manifest", "") or None,
        "support_manifest_sha256": getattr(args, "support_manifest_sha256", None),
        "normal_sampling": args.normal_sampling,
        "resize": args.resize,
        "imagesize": args.imagesize,
        "seed": args.seed,
        "num_jobs": len(rows),
        "image_auroc_macro": float(np.mean([r["image_auroc"] for r in rows])),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {result_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
