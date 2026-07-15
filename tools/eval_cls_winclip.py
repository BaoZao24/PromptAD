#!/usr/bin/env python
"""Evaluate WinCLIP (few-shot) on the pooled 4-signal cls protocol.

Protocol:
- signals: burst / chirp / dsss / pulse (pooled normal support, per-(signal,scene,jsr) eval)
- few-shot normal gallery: one normal sample per RF frequency band

WinCLIP forward returns a 2D anomaly map per image; image-level score = map.max()
(same as WinCLIP's own `metric_cal`).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
WINCLIP_ROOT = REPO_ROOT / "references" / "WinCLIP"
# Order matters: REPO_ROOT must be searched BEFORE WinCLIP_ROOT so that
# `import datasets` / `import utils` resolve to PromptAD's packages, not WinCLIP's.
# WinCLIP's model package uses relative imports (.CLIPAD, .ad_prompts) and does not
# depend on its own top-level `datasets`/`utils`, so this is safe.
if str(WINCLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(WINCLIP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_rf_target_pooled_universal import (  # noqa: E402
    JSR_BY_SIGNAL,
    RFPathDataset,
    SCENES,
    SIGNALS,
    collect_samples,
    to_model_input,
)
from utils.rf_frequency_sampling import select_one_per_frequency_band  # noqa: E402
from utils.training_utils import setup_seed  # noqa: E402

from WinCLIP import WinClipAD  # noqa: E402

CURRENT_RESCORE = 78.4088


def safe_auc(labels, scores) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores) * 100.0)


@torch.no_grad()
def build_winclip_visual_gallery(model, normal_loader, device):
    """Build multi-scale visual gallery batch-wise.

    WinCLIP's `build_image_feature_gallery` resets `self.visual_gallery = []` on each
    call, so calling it per-batch would keep only the last batch. We replicate its
    scale-grouping logic here but accumulate across batches.
    """
    per_scale = None  # list[list[tensor]] -> one list of tensors per scale
    for data, *_ in tqdm(normal_loader, desc="Build WinCLIP gallery", leave=False):
        data = to_model_input(model, data, device, rgb_from_bgr=True)
        feats = model.encode_image(data)  # list of [B, D] per window, L2-normalized
        groups = []
        for si in range(len(model.scale_begin_indx)):
            if si == len(model.scale_begin_indx) - 1:
                scale_feats = feats[model.scale_begin_indx[si]:]
            else:
                scale_feats = feats[model.scale_begin_indx[si]:model.scale_begin_indx[si + 1]]
            groups.append(torch.cat(scale_feats, dim=0))  # [B*windows_in_scale, D]
        if per_scale is None:
            per_scale = [[g] for g in groups]
        else:
            for i, g in enumerate(groups):
                per_scale[i].append(g)
    model.visual_gallery = [torch.cat(pf, dim=0) for pf in per_scale]


@torch.no_grad()
def eval_loader(model, loader, device):
    scores, labels = [], []
    for data, mask, label, name, img_type in loader:
        data = to_model_input(model, data, device, rgb_from_bgr=True)
        maps = model(data)  # list of HxW numpy arrays
        for m, l in zip(maps, label.numpy()):
            scores.append(float(np.max(m)))
            labels.append(int(l))
    return labels, scores


def get_args():
    parser = argparse.ArgumentParser(description="WinCLIP few-shot cls on pooled signals")
    parser.add_argument("--signals", nargs="+", default=SIGNALS, choices=SIGNALS)
    parser.add_argument("--scenes", nargs="+", default=SCENES, choices=SCENES)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gallery-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--img-resize", type=int, default=240)
    parser.add_argument("--img-cropsize", type=int, default=240)
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument("--scales", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--backbone", default="ViT-B-16-plus-240")
    parser.add_argument("--pretrained-dataset", default="laion400m_e32")
    parser.add_argument("--prompt-class", default="signal",
                        help="class name injected into text prompts (pooled -> 'signal')")
    parser.add_argument("--output-root", default="analysis_outputs/winclip_cls")
    parser.add_argument("--smoke-test", action="store_true",
                        help="only eval the first combo with a tiny gallery (chain validation)")
    return parser.parse_args()


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
    setup_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ---- pooled few-shot normal gallery: one sample per frequency band ----
    normal_samples = []
    for signal in args.signals:
        for scene in args.scenes:
            for jsr in JSR_BY_SIGNAL[signal]:
                normal_samples.extend(collect_samples(signal, scene, jsr, "train", k_shot=0))
    normal_samples = select_one_per_frequency_band(
        normal_samples, path_getter=lambda s: s[0]
    )
    print(f"[WinCLIP cls] few-shot normal gallery samples (one per band): {len(normal_samples)}")
    support_paths = {sample[0] for sample in normal_samples}

    # ---- eval combos ----
    eval_combos = [(s, sc, j) for s in args.signals for sc in args.scenes for j in JSR_BY_SIGNAL[s]]
    if args.smoke_test:
        eval_combos = eval_combos[:1]
        normal_samples = normal_samples[:4]
        print(f"[smoke-test] only eval {eval_combos[0]}, gallery={len(normal_samples)}")

    normal_loader = DataLoader(
        RFPathDataset(normal_samples),
        batch_size=args.gallery_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ---- model ----
    model = WinClipAD(
        out_size_h=args.resolution, out_size_w=args.resolution, device=device,
        backbone=args.backbone, pretrained_dataset=args.pretrained_dataset,
        scales=args.scales, img_resize=args.img_resize, img_cropsize=args.img_cropsize,
        resolution=args.resolution,
    )
    model = model.to(device)
    model.eval_mode()
    model.build_text_feature_gallery(args.prompt_class)
    build_winclip_visual_gallery(model, normal_loader, device)
    print(f"[WinCLIP cls] visual gallery shapes: {[tuple(g.shape) for g in model.visual_gallery]}")

    # ---- eval ----
    rows = []
    for signal, scene, jsr in tqdm(eval_combos, desc="Eval combos"):
        samples = collect_samples(signal, scene, jsr, "test", exclude_paths=support_paths)
        loader = DataLoader(RFPathDataset(samples), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
        labels, scores = eval_loader(model, loader, device)
        i_roc = safe_auc(labels, scores)
        n_norm = sum(1 for s in samples if int(s[2]) == 0)
        n_abn = sum(1 for s in samples if int(s[2]) == 1)
        rows.append({
            "signal": signal, "scene": scene, "jsr": jsr,
            "i_roc": i_roc, "num_normal": n_norm, "num_abnormal": n_abn,
        })
        print(f"  {signal}/{scene}/{jsr}: i_roc={i_roc:.2f}  (n_norm={n_norm}, n_abn={n_abn})")

    # ---- summary ----
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "winclip_cls_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== Summary (image-AUROC) ===")
    for signal in args.signals:
        valid = [r["i_roc"] for r in rows if r["signal"] == signal and not np.isnan(r["i_roc"])]
        mean = float(np.mean(valid)) if valid else float("nan")
        print(f"  {signal}: {mean:.4f}  ({len(valid)} combos)")
    valid_all = [r["i_roc"] for r in rows if not np.isnan(r["i_roc"])]
    overall = float(np.mean(valid_all)) if valid_all else float("nan")
    print(f"  OVERALL: {overall:.4f}  ({len(valid_all)} combos)")
    print(f"  vs PromptAD current rescore {CURRENT_RESCORE}: delta={overall - CURRENT_RESCORE:+.4f}")
    print(f"  CSV -> {csv_path}")


if __name__ == "__main__":
    main()
