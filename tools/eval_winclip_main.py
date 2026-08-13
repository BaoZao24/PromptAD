#!/usr/bin/env python
"""Evaluate WinCLIP on current RF/Spectrum main CLS protocols."""

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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
WINCLIP_ROOT = REPO_ROOT / "references" / "WinCLIP"
if str(WINCLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(WINCLIP_ROOT))
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
from WinCLIP import WinClipAD  # noqa: E402


class WinCLIPPathDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = list(samples)
        self.transform = transform

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


def train_signature(job):
    return tuple(str(sample["path"]) for sample in job["train_samples"])


def make_loader(samples, transform, args, batch_size: int, shuffle: bool):
    return DataLoader(
        WinCLIPPathDataset(samples, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def make_model(args, device):
    model = WinClipAD(
        out_size_h=args.resolution,
        out_size_w=args.resolution,
        device=device,
        backbone=args.backbone,
        pretrained_dataset=args.pretrained_dataset,
        scales=args.scales,
        img_resize=args.img_resize,
        img_cropsize=args.img_cropsize,
        resolution=args.resolution,
    )
    model = model.to(device)
    model.eval_mode()
    model.build_text_feature_gallery(args.prompt_class)
    return model


@torch.no_grad()
def build_visual_gallery(model, train_samples, args, device):
    loader = make_loader(train_samples, model.transform, args, args.gallery_batch_size, shuffle=False)
    per_scale = None
    for images, _labels, _paths, _names in loader:
        images = images.to(device)
        feats = model.encode_image(images)
        groups = []
        for scale_idx in range(len(model.scale_begin_indx)):
            if scale_idx == len(model.scale_begin_indx) - 1:
                scale_feats = feats[model.scale_begin_indx[scale_idx]:]
            else:
                scale_feats = feats[model.scale_begin_indx[scale_idx]:model.scale_begin_indx[scale_idx + 1]]
            groups.append(torch.cat(scale_feats, dim=0))
        if per_scale is None:
            per_scale = [[group] for group in groups]
        else:
            for idx, group in enumerate(groups):
                per_scale[idx].append(group)
    if per_scale is None:
        raise RuntimeError("No train samples for WinCLIP visual gallery")
    model.visual_gallery = [torch.cat(items, dim=0) for items in per_scale]


@torch.no_grad()
def predict_job(model, job, args, device):
    loader = make_loader(job["test_samples"], model.transform, args, args.batch_size, shuffle=False)
    labels, scores, paths, names = [], [], [], []
    for images, batch_labels, batch_paths, batch_names in loader:
        images = images.to(device)
        batch_maps = model(images)
        for anomaly_map, label, path, name in zip(batch_maps, batch_labels.numpy(), batch_paths, batch_names):
            labels.append(int(label))
            scores.append(float(np.max(anomaly_map)))
            paths.append(str(path))
            names.append(str(name))

    labels_np = np.asarray(labels, dtype=np.int32)
    scores_np = np.asarray(scores, dtype=np.float32)
    score_dir = Path(args.output_root) / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{job['dataset']}-{job['category']}-{job['scene']}-{job['jsr']}".replace("/", "_")
    payload = {
        "scores": scores_np,
        "labels": labels_np,
        "image_paths": np.asarray(paths),
        "names": np.asarray(names),
    }
    np.savez_compressed(score_dir / f"{stem}-scores.npz", **payload)
    return {
        "method": "winclip_fewshot",
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
    model = make_model(args, device)
    rows = []
    groups = {}
    for idx, job in enumerate(jobs):
        groups.setdefault(train_signature(job), []).append((idx, job))

    for group_idx, grouped in enumerate(groups.values(), 1):
        first_job = grouped[0][1]
        if not first_job["train_samples"]:
            raise RuntimeError("No train samples for grouped WinCLIP run")
        print(
            f"[gallery {group_idx}/{len(groups)}] train_normals={len(first_job['train_samples'])} "
            f"eval_jobs={len(grouped)}",
            flush=True,
        )
        build_visual_gallery(model, first_job["train_samples"], args, device)
        print(f"  visual_gallery={[tuple(g.shape) for g in model.visual_gallery]}", flush=True)
        for idx, job in grouped:
            if not any(sample["label"] == 1 for sample in job["test_samples"]):
                raise RuntimeError(f"No abnormal samples for {job['dataset']} {job['category']} {job['scene']} {job['jsr']}")
            print(f"[{idx + 1}/{len(jobs)}] {job['dataset']} {job['category']} {job['scene']} {job['jsr']}", flush=True)
            row = predict_job(model, job, args, device)
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
            "method": "winclip_fewshot",
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
    parser.add_argument("--output-root", default="analysis_outputs/winclip_main")
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gallery-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-resize", type=int, default=240)
    parser.add_argument("--img-cropsize", type=int, default=240)
    parser.add_argument("--resolution", type=int, default=240)
    parser.add_argument("--scales", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--backbone", default="ViT-B-16-plus-240")
    parser.add_argument("--pretrained-dataset", default="laion400m_e32")
    parser.add_argument("--prompt-class", default="radio frequency spectrogram")
    add_ofdma_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda:0"

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
    result_path = out_root / "results_winclip_cls.csv"
    write_csv(result_path, rows_with_avg)
    summary = {
        "method": "winclip_fewshot",
        "protocol": args.protocol,
        "support_protocol": "target_scene" if args.protocol == "rf_target" else None,
        "support_manifest": getattr(args, "support_manifest", "") or None,
        "support_manifest_sha256": getattr(args, "support_manifest_sha256", None),
        "test_normal_paths_sha256": getattr(args, "test_normal_paths_sha256", None),
        "support_policy": getattr(args, "support_policy", None),
        "per_frequency_k": getattr(args, "per_frequency_k", None),
        "normal_sampling": args.normal_sampling,
        "rf_train_mode": getattr(args, "rf_train_mode", None),
        "prompt_class": args.prompt_class,
        "backbone": args.backbone,
        "pretrained_dataset": args.pretrained_dataset,
        "scales": args.scales,
        "img_resize": args.img_resize,
        "img_cropsize": args.img_cropsize,
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "num_jobs": len(rows),
        "image_auroc_macro": float(np.mean([row["image_auroc"] for row in rows])),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_root / "README.md").write_text(
        "# WinCLIP Few-shot CLS Reference\n\n"
        "Source: `references/WinCLIP`. This is a CLIP/WinCLIP reference baseline, not a normal-only visual-memory baseline.\n\n"
        f"Protocol: `{args.protocol}`\n\n"
        f"Normal sampling: `{args.normal_sampling}`\n\n"
        f"Prompt class: `{args.prompt_class}`\n\n"
        f"Result CSV: `{result_path.name}`\n\n"
        f"Macro Image-AUROC: `{summary['image_auroc_macro']:.4f}`\n",
        encoding="utf-8",
    )
    print(f"wrote {result_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
