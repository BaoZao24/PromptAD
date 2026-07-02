#!/usr/bin/env python
"""Evaluate CLS dual-gallery calibration on the built-in spectrum dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PromptAD import PromptAD
from tools.eval_cls_dual_gallery_fusion import fuse_dual_scores, safe_auc, topk_distance
from tools.eval_cls_resnet_gallery_fusion import resnet_patch_image_scores
from tools.eval_seg_resnet_gallery_fusion import (
    ResNet18LocalEncoder,
    build_resnet_gallery,
    load_checkpoint,
)
from train_rf_target_pooled_universal import build_gallery, to_model_input
from utils.training_utils import setup_seed


SPECTRUM_ROOT = Path("datasets/spectrum")
SPECTRUM_CLASSES = ("16QAM", "CHIRP", "GMSK", "QPSK")


class SpectrumPathDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, sample_type, name_prefix = self.samples[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        h, w = img.shape[:2]
        gt = np.zeros((h, w), dtype=np.uint8)
        name = f"{name_prefix}-{sample_type}-{Path(img_path).stem}"
        return img, gt, int(label), name, sample_type


def collect_train_samples(category: str, args):
    paths = sorted((Path(args.spectrum_root) / category / "train" / "good").glob("*.png"))
    if args.max_train_normals > 0:
        paths = paths[: args.max_train_normals]
    return [(p, 0, f"{category}_train_good", f"spectrum-{category}") for p in paths]


def collect_eval_samples(category: str, args):
    root = Path(args.spectrum_root) / category / "test"
    good_paths = sorted((root / "good").glob("*.png"))
    bad_paths = sorted((root / "bad").glob("*.png"))
    if args.max_test_normals > 0:
        good_paths = good_paths[: args.max_test_normals]
    if args.max_abnormals > 0:
        bad_paths = bad_paths[: args.max_abnormals]
    samples = [(p, 0, f"{category}_test_good", f"spectrum-{category}") for p in good_paths]
    samples.extend((p, 1, f"{category}_test_bad", f"spectrum-{category}") for p in bad_paths)
    return samples


def make_loader(samples, args, shuffle=False):
    return DataLoader(
        SpectrumPathDataset(samples),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def encode_clip_gallery(model, loader, device):
    features = []
    model.eval_mode()
    for data, mask, label, name, img_type in tqdm(loader, desc="Build spectrum CLIP normal gallery", leave=False):
        batch = to_model_input(model, data, device, rgb_from_bgr=True)
        features.append(F.normalize(model.encode_image(batch)[0].float(), dim=-1).cpu())
    return torch.cat(features, dim=0)


@torch.no_grad()
def evaluate_category(model, encoder, category, train_loader, eval_loader, args, device):
    build_gallery(model, train_loader, device)
    model.build_text_feature_gallery()
    clip_gallery = encode_clip_gallery(model, train_loader, device)
    print(f"[spectrum_clip_gallery] category={category} images={clip_gallery.shape[0]} topk={args.clip_topk}")
    resnet_galleries = build_resnet_gallery(encoder, train_loader, args, device)

    labels, names = [], []
    promptad_scores, clip_scores = [], []
    resnet_scores = {layer: [] for layer in args.resnet_layers}
    model.eval_mode()
    encoder.eval()

    for data, mask, label, name, img_type in tqdm(eval_loader, desc=f"Eval spectrum {category}", leave=False):
        data_t = to_model_input(model, data, device, rgb_from_bgr=True)
        visual_features = model.encode_image(data_t)
        score_img, _ = model.score_cached(visual_features, "cls")
        cls_features = F.normalize(visual_features[0].float(), dim=-1).cpu()

        promptad_scores.extend(float(x) for x in score_img)
        clip_scores.extend(float(x) for x in topk_distance(cls_features, clip_gallery, args.clip_topk))

        raw = data.permute(0, 3, 1, 2).to(device, non_blocking=True)
        cnn_features = encoder(raw)
        for layer in args.resnet_layers:
            values = resnet_patch_image_scores(
                cnn_features[layer],
                resnet_galleries[layer],
                args.distance_chunk_size,
                args.image_top_ratio,
            )
            resnet_scores[layer].extend(float(x) for x in values)

        labels.extend(int(x) for x in label.numpy().tolist())
        names.extend(list(name))

    labels_np = np.asarray(labels, dtype=np.int32)
    promptad_np = np.asarray(promptad_scores, dtype=np.float32)
    clip_np = np.asarray(clip_scores, dtype=np.float32)
    row = {
        "method": "spectrum_cls_dual_gallery",
        "task": "cls",
        "dataset": "spectrum",
        "category": category,
        "num_normal": int((labels_np == 0).sum()),
        "num_abnormal": int((labels_np == 1).sum()),
        "promptad_rescore_auc": safe_auc(labels, promptad_np),
        f"clip_gallery_top{args.clip_topk}_auc": safe_auc(labels, clip_np),
    }
    npz_payload = {
        "names": np.asarray(names),
        "labels": labels_np,
        "promptad_scores": promptad_np,
        "clip_gallery_scores": clip_np,
    }
    for layer in args.resnet_layers:
        resnet_np = np.asarray(resnet_scores[layer], dtype=np.float32)
        row[f"resnet18_{layer}_gallery_top{args.image_top_ratio:g}_auc"] = safe_auc(labels, resnet_np)
        npz_payload[f"resnet18_{layer}_scores"] = resnet_np
        fused = fuse_dual_scores(
            promptad_np,
            clip_np,
            resnet_np,
            args.clip_lambdas,
            args.resnet_lambdas,
        )
        for key, values in fused.items():
            row[f"resnet18_{layer}_{key}_auc"] = safe_auc(labels, values)
            npz_payload[f"resnet18_{layer}_{key}"] = np.asarray(values, dtype=np.float32)

    score_root = Path(args.output_root) / "scores"
    score_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(score_root / f"{category}-scores.npz", **npz_payload)
    return row


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="analysis_outputs/20260702_spectrum_cls_dual_gallery")
    parser.add_argument("--spectrum-root", default=str(SPECTRUM_ROOT))
    parser.add_argument("--checkpoint", default="analysis_outputs/20260627_method_funnel/runs/pooled_rf_rgb/cls/checkpoint/overall-best.pt")
    parser.add_argument("--categories", nargs="+", default=list(SPECTRUM_CLASSES), choices=list(SPECTRUM_CLASSES))
    parser.add_argument("--resnet-layers", nargs="+", default=["layer3"], choices=["layer2", "layer3"])
    parser.add_argument("--clip-topk", type=int, default=50)
    parser.add_argument("--image-top-ratio", type=float, default=0.1)
    parser.add_argument("--clip-lambdas", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--resnet-lambdas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--max-gallery-patches", type=int, default=50000)
    parser.add_argument("--distance-chunk-size", type=int, default=1024)
    parser.add_argument("--max-train-normals", type=int, default=0)
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument("--img-resize", type=int, default=240)
    parser.add_argument("--img-cropsize", type=int, default=240)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--backbone", default="ViT-B-16-plus-240")
    parser.add_argument("--pretrained_dataset", default="laion400m_e32")
    parser.add_argument("--prompt-mode", default="rf")
    parser.add_argument("--input-mode", default="rgb")
    parser.add_argument("--text-prototype-mode", default="single")
    parser.add_argument("--cls-score-mode", default="text_only")
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=3)
    parser.add_argument("--n_pro_ab", type=int, default=4)
    parser.add_argument("--use-cpu", type=int, default=0)
    args = parser.parse_args()

    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cpu" if args.use_cpu else "cuda:0"

    kwargs = vars(args).copy()
    kwargs.update({
        "dataset": "spectrum",
        "class_name": "radio frequency spectrogram",
        "device": device,
        "out_size_h": args.resolution,
        "out_size_w": args.resolution,
    })
    model = PromptAD(**kwargs).to(device)
    load_checkpoint(model, args.checkpoint)
    encoder = ResNet18LocalEncoder().to(device)

    rows = []
    for category in args.categories:
        train_samples = collect_train_samples(category, args)
        eval_samples = collect_eval_samples(category, args)
        if not train_samples:
            print(f"[skip] no train/good samples for {category}")
            continue
        if not any(int(sample[1]) == 1 for sample in eval_samples):
            print(f"[skip] no test/bad samples for {category}")
            continue
        train_loader = make_loader(train_samples, args)
        eval_loader = make_loader(eval_samples, args)
        rows.append(evaluate_category(model, encoder, category, train_loader, eval_loader, args, device))

    out_root = Path(args.output_root)
    result_path = out_root / "results_cls_spectrum_dual_gallery.csv"
    write_csv(result_path, rows)
    df = pd.DataFrame(rows)
    summary = {
        "method": "spectrum_cls_dual_gallery",
        "checkpoint": args.checkpoint,
        "spectrum_root": args.spectrum_root,
        "categories": args.categories,
        "gallery_protocol": "per-category train/good normal gallery; test/good + test/bad evaluation",
        "clip_topk": args.clip_topk,
        "resnet": "resnet18_imagenet1k_v1",
        "resnet_layers": args.resnet_layers,
        "image_top_ratio": args.image_top_ratio,
        "clip_lambdas": args.clip_lambdas,
        "resnet_lambdas": args.resnet_lambdas,
        "max_gallery_patches": args.max_gallery_patches,
    }
    for col in [c for c in df.columns if c.endswith("_auc")]:
        summary[f"{col}_macro"] = float(df[col].mean())
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_root / "README.md").write_text(
        "# Spectrum CLS Dual Gallery Evaluation\n\n"
        "This experiment evaluates the current dual-gallery CLS method on `datasets/spectrum`.\n\n"
        "Protocol: for each category, `train/good` builds the normal galleries; "
        "`test/good` and `test/bad` are evaluated for Image-AUROC. No SEG metrics are computed.\n\n"
        f"Result CSV: `{result_path.name}`\n\n"
        f"Summary: `{summary_path.name}`\n",
        encoding="utf-8",
    )
    print(f"wrote {result_path}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
