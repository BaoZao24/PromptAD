#!/usr/bin/env python
"""Evaluate CLS dual-gallery calibration on the public RF_SPE_PNG dataset."""

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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PromptAD import PromptAD
from tools.eval_cls_dual_gallery_fusion import (
    fuse_dual_scores,
    safe_auc,
    topk_distance,
)
from tools.eval_cls_resnet_gallery_fusion import resnet_patch_image_scores
from tools.eval_seg_resnet_gallery_fusion import (
    ResNet18LocalEncoder,
    build_resnet_gallery,
    cnn_tta_batch,
    cnn_tta_modes,
    load_checkpoint,
)
from train_rf_target_pooled_universal import build_gallery, to_model_input
from utils.rf_frequency_sampling import NORMAL_SAMPLING_CHOICES, maybe_select_one_per_frequency_band
from utils.training_utils import setup_seed


RF_PUBLIC_ROOT = Path("/mnt/data/wangbei/data/RF_SPE_PNG")
PUBLIC_NORMAL_DIR = RF_PUBLIC_ROOT / "RF_Spectrum_Public_Dataset"
PUBLIC_SIGNALS = ("burst", "chirp", "dsss", "pulse")
JSR_BY_SIGNAL = {
    "burst": ("m10db", "m20db", "m30db", "m40db", "m50db"),
    "chirp": ("m30db", "m40db", "m50db", "m55db", "m60db", "m70db"),
    "dsss": ("m10db", "m20db", "m30db", "m40db", "m50db"),
    "pulse": ("m10db", "m20db", "m30db", "m40db", "m50db"),
}
DEFAULT_THREE_JSR_BY_SIGNAL = {
    "burst": ("m30db", "m40db", "m50db"),
    "chirp": ("m40db", "m50db", "m55db"),
    "dsss": ("m30db", "m40db", "m50db"),
    "pulse": ("m30db", "m40db", "m50db"),
}


class PublicRFPathDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path, label, sample_type, name_prefix = self.samples[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        h, w = img.shape[:2]
        if gt_path == 0:
            gt = np.zeros((h, w), dtype=np.uint8)
        else:
            gt_color = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
            if gt_color is None:
                gt = np.zeros((h, w), dtype=np.uint8)
            else:
                gt = cv2.cvtColor(gt_color, cv2.COLOR_BGR2GRAY)
                gt = ((gt > 0).astype(np.uint8) * 255)

        target_size = min(max(h, w), 1024)
        img = cv2.resize(img, (target_size, target_size))
        gt = cv2.resize(gt, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        name = f"{name_prefix}-{sample_type}-{Path(img_path).stem}"
        return img, gt, int(label), name, sample_type


def public_normal_records():
    records = sorted(
        p for p in PUBLIC_NORMAL_DIR.iterdir()
        if p.is_dir() and p.name.startswith("MeasRes_")
    )
    if not records:
        raise FileNotFoundError(f"No public normal records found under {PUBLIC_NORMAL_DIR}")
    return records


def public_normal_paths(records):
    paths = []
    for record in records:
        paths.extend(sorted(record.glob("*.png")))
    return paths


def all_public_normal_paths():
    records = public_normal_records()
    return public_normal_paths(records)


def select_public_support_and_test_normals(args):
    normal_paths = all_public_normal_paths()
    train_normal_paths = maybe_select_one_per_frequency_band(normal_paths, args.normal_sampling)
    support_set = {str(p) for p in train_normal_paths}
    test_normal_paths = [p for p in normal_paths if str(p) not in support_set]
    return train_normal_paths, test_normal_paths


def build_train_loader(args):
    train_normal_paths, _ = select_public_support_and_test_normals(args)
    samples = [
        (p, 0, 0, "public_train_normal", "public-rf")
        for p in train_normal_paths
    ]
    return DataLoader(
        PublicRFPathDataset(samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def collect_eval_samples(signal: str, jsr: str, args):
    _, test_normal_paths = select_public_support_and_test_normals(args)
    if args.max_test_normals > 0:
        test_normal_paths = test_normal_paths[: args.max_test_normals]
    samples = [
        (p, 0, 0, f"{signal}_{jsr}_public_normal", f"public-rf-{signal}-{jsr}")
        for p in test_normal_paths
    ]

    abnormal_root = RF_PUBLIC_ROOT / signal / "abnormal" / jsr
    gt_root = RF_PUBLIC_ROOT / signal / "groundtruth" / jsr
    abnormal_paths = sorted(abnormal_root.glob("*.png")) if abnormal_root.is_dir() else []
    if args.max_abnormals > 0:
        abnormal_paths = abnormal_paths[: args.max_abnormals]

    for img_path in abnormal_paths:
        gt_name = img_path.name.replace("_abnormal.png", "_groundtruth.png")
        gt_path = gt_root / gt_name
        samples.append((
            img_path,
            gt_path if gt_path.exists() else 0,
            1,
            f"{signal}_{jsr}_public_abnormal",
            f"public-rf-{signal}-{jsr}",
        ))
    return samples


def build_eval_loaders(args):
    loaders = []
    jsr_by_signal = parse_jsrs_by_signal(args.jsrs_by_signal)
    for signal in args.signals:
        for jsr in jsr_by_signal[signal]:
            samples = collect_eval_samples(signal, jsr, args)
            if not any(int(s[2]) == 1 for s in samples):
                print(f"[skip] no abnormal samples for {signal}/{jsr}")
                continue
            loader = DataLoader(
                PublicRFPathDataset(samples),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            loaders.append((signal, "RF_SPE_PNG_public", jsr, loader))
    return loaders


def parse_jsrs_by_signal(value: str | None):
    result = {k: tuple(v) for k, v in DEFAULT_THREE_JSR_BY_SIGNAL.items()}
    if not value:
        return result
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        signal, jsrs = item.split(":", 1)
        signal = signal.strip()
        if signal not in JSR_BY_SIGNAL:
            raise ValueError(f"Unknown public RF signal {signal!r}; expected {sorted(JSR_BY_SIGNAL)}")
        parsed = tuple(x.strip() for x in jsrs.split(",") if x.strip())
        invalid = [x for x in parsed if x not in JSR_BY_SIGNAL[signal]]
        if invalid:
            raise ValueError(f"Invalid JSR for {signal}: {invalid}; available={JSR_BY_SIGNAL[signal]}")
        if not parsed:
            raise ValueError(f"No JSR values provided for {signal}")
        result[signal] = parsed
    return result


@torch.no_grad()
def encode_clip_gallery(model, loader, device):
    features = []
    model.eval_mode()
    for data, mask, label, name, img_type in tqdm(loader, desc="Build public CLIP normal gallery", leave=False):
        batch = to_model_input(model, data, device, rgb_from_bgr=True)
        features.append(F.normalize(model.encode_image(batch)[0].float(), dim=-1).cpu())
    return torch.cat(features, dim=0)


@torch.no_grad()
def evaluate(model, encoder, clip_gallery, resnet_galleries_by_mode, eval_loaders, args, device):
    model.eval_mode()
    encoder.eval()
    rows = []
    score_root = Path(args.output_root) / "scores"
    score_root.mkdir(parents=True, exist_ok=True)

    for signal, scene, jsr, loader in eval_loaders:
        labels, names = [], []
        promptad_scores, clip_scores = [], []
        resnet_scores = {layer: [] for layer in args.resnet_layers}

        for data, mask, label, name, img_type in tqdm(loader, desc=f"Eval public RF {signal}/{jsr}", leave=False):
            data_t = to_model_input(model, data, device, rgb_from_bgr=True)
            visual_features = model.encode_image(data_t)
            score_img, _ = model.score_cached(visual_features, "cls")
            cls_features = F.normalize(visual_features[0].float(), dim=-1).cpu()

            promptad_scores.extend([float(x) for x in score_img])
            clip_scores.extend([float(x) for x in topk_distance(cls_features, clip_gallery, args.clip_topk)])

            mode_scores = {layer: [] for layer in args.resnet_layers}
            for mode, resnet_galleries in resnet_galleries_by_mode.items():
                data_variant = cnn_tta_batch(data, mode, args)
                raw = data_variant.permute(0, 3, 1, 2).to(device, non_blocking=True)
                cnn_features = encoder(raw)
                for layer in args.resnet_layers:
                    values = resnet_patch_image_scores(
                        cnn_features[layer],
                        resnet_galleries[layer],
                        args.distance_chunk_size,
                        args.image_top_ratio,
                        args.cnn_nn_topk,
                    )
                    mode_scores[layer].append(np.asarray(values, dtype=np.float32))
            for layer in args.resnet_layers:
                values = np.max(np.stack(mode_scores[layer], axis=0), axis=0)
                resnet_scores[layer].extend([float(x) for x in values])

            labels.extend([int(x) for x in label.numpy().tolist()])
            names.extend(list(name))

        labels_np = np.asarray(labels, dtype=np.int32)
        promptad_np = np.asarray(promptad_scores, dtype=np.float32)
        clip_np = np.asarray(clip_scores, dtype=np.float32)
        row = {
            "method": "public_rf_cls_dual_gallery",
            "task": "cls",
            "dataset": signal,
            "scene": scene,
            "jsr": jsr,
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

        rows.append(row)
        np.savez_compressed(score_root / f"{signal}-{jsr}-scores.npz", **npz_payload)
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="analysis_outputs/20260702_public_rf_cls_dual_gallery")
    parser.add_argument(
        "--checkpoint",
        default="analysis_outputs/90_rejected_or_aborted/20260706_cleanup_old_results/20260627_method_funnel/runs/pooled_rf_rgb/cls/checkpoint/overall-best.pt",
    )
    parser.add_argument("--signals", nargs="+", default=list(PUBLIC_SIGNALS), choices=list(PUBLIC_SIGNALS))
    parser.add_argument(
        "--jsrs-by-signal",
        default=None,
        help=(
            "Semicolon-separated JSR protocol, e.g. "
            "'burst:m30db,m40db,m50db;chirp:m40db,m50db,m55db;dsss:m30db,m40db,m50db;pulse:m30db,m40db,m50db'. "
            "Defaults to this three-JSR public RF protocol."
        ),
    )
    parser.add_argument("--resnet-layers", nargs="+", default=["layer3"], choices=["layer2", "layer3"])
    parser.add_argument("--clip-topk", type=int, default=50)
    parser.add_argument("--image-top-ratio", type=float, default=0.1)
    parser.add_argument("--clip-lambdas", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--resnet-lambdas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--max-gallery-patches", type=int, default=50000)
    parser.add_argument(
        "--cnn-coreset-size",
        type=int,
        default=0,
        help="Farthest-first CNN memory size; 0 keeps the complete normal memory.",
    )
    parser.add_argument(
        "--cnn-nn-topk",
        type=int,
        default=1,
        help="Average distance to this many nearest CNN normal patches; 1 is legacy nearest neighbour.",
    )
    parser.add_argument(
        "--cnn-paired-tta",
        choices=["none", "stft_shift_blur"],
        default="none",
        help="Optional paired TTA for the CNN branch; default keeps the formal CNN path unchanged.",
    )
    parser.add_argument("--cnn-paired-tta-shift-px", type=int, default=4)
    parser.add_argument("--cnn-paired-tta-blur-ksize", type=int, default=3)
    parser.add_argument("--distance-chunk-size", type=int, default=1024)
    parser.add_argument("--normal-sampling", choices=NORMAL_SAMPLING_CHOICES, default="all")
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
        "dataset": "rf_spe_png",
        "class_name": "radio frequency spectrogram",
        "device": device,
        "out_size_h": args.resolution,
        "out_size_w": args.resolution,
    })
    model = PromptAD(**kwargs).to(device)
    load_checkpoint(model, args.checkpoint)

    train_loader = build_train_loader(args)
    build_gallery(model, train_loader, device)
    model.build_text_feature_gallery()
    clip_gallery = encode_clip_gallery(model, train_loader, device)
    print(f"[public_clip_gallery] images={clip_gallery.shape[0]} topk={args.clip_topk}")

    encoder = ResNet18LocalEncoder().to(device)
    resnet_galleries_by_mode = {
        mode: build_resnet_gallery(encoder, train_loader, args, device, tta_mode=mode)
        for mode in cnn_tta_modes(args)
    }
    rows = evaluate(model, encoder, clip_gallery, resnet_galleries_by_mode, build_eval_loaders(args), args, device)

    out_root = Path(args.output_root)
    result_path = out_root / "results_cls_public_rf_dual_gallery.csv"
    write_csv(result_path, rows)
    df = pd.DataFrame(rows)
    summary = {
        "method": "public_rf_cls_dual_gallery",
        "checkpoint": args.checkpoint,
        "dataset_root": str(RF_PUBLIC_ROOT),
        "normal_sampling": args.normal_sampling,
        "clip_topk": args.clip_topk,
        "resnet": "resnet18_imagenet1k_v1",
        "resnet_layers": args.resnet_layers,
        "jsrs_by_signal": parse_jsrs_by_signal(args.jsrs_by_signal),
        "image_top_ratio": args.image_top_ratio,
        "clip_lambdas": args.clip_lambdas,
        "resnet_lambdas": args.resnet_lambdas,
        "max_gallery_patches": args.max_gallery_patches,
        "cnn_paired_tta": args.cnn_paired_tta,
        "cnn_paired_tta_modes": cnn_tta_modes(args),
        "cnn_paired_tta_shift_px": args.cnn_paired_tta_shift_px,
        "cnn_paired_tta_blur_ksize": args.cnn_paired_tta_blur_ksize,
        "cnn_coreset_size": args.cnn_coreset_size,
        "cnn_nn_topk": args.cnn_nn_topk,
    }
    for col in [c for c in df.columns if c.endswith("_auc")]:
        summary[f"{col}_macro"] = float(df[col].mean())
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_root / "README.md").write_text(
        "# Public RF CLS Dual Gallery Evaluation\n\n"
        "This experiment evaluates the current dual-gallery CLS method on RF_SPE_PNG.\n\n"
        "Normal support is selected by --normal-sampling; selected support normals are excluded from test normals.\n"
        "Abnormal samples are read from RF_SPE_PNG/{burst,chirp,dsss,pulse}/abnormal/{jsr}.\n\n"
        f"Result CSV: `{result_path.name}`\n\n"
        f"Summary: `{summary_path.name}`\n",
        encoding="utf-8",
    )
    print(f"wrote {result_path}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
