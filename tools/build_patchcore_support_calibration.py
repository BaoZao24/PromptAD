#!/usr/bin/env python
"""Build a label-free CNN score reference from normal support images.

Each support image is scored with a PatchCore memory fitted on all *other*
support images.  This leave-one-out procedure avoids the zero-distance shortcut
of matching a support image against its own normal patches.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_patchcore_cls import (
    PUBLIC_RF_JSRS,
    PatchCorePathDataset,
    fit_patchcore,
    public_rf_jobs,
    rf_target_jobs,
)
from train_rf_target_pooled_universal import JSR_BY_SIGNAL, SCENES
from utils.rf_frequency_sampling import NORMAL_SAMPLING_CHOICES
from utils.training_utils import setup_seed


@torch.no_grad()
def score_one_normal(model, sample, args):
    dataset = PatchCorePathDataset([sample], args.resize, args.imagesize)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    scores, _segmentations, _labels, _masks = model.predict(loader)
    return float(np.asarray(scores, dtype=np.float64)[0])


def collect_support(args):
    if args.protocol == "public_rf":
        jobs = public_rf_jobs(args)
    else:
        jobs = rf_target_jobs(args)
    if not jobs or not jobs[0]["train_samples"]:
        raise RuntimeError("No normal support images found")
    return jobs[0]["train_samples"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["public_rf", "rf_target"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--backbone", default="wideresnet50")
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--pretrain-embed-dimension", type=int, default=1024)
    parser.add_argument("--target-embed-dimension", type=int, default=1024)
    parser.add_argument("--patchsize", type=int, default=3)
    parser.add_argument("--anomaly-scorer-num-nn", type=int, default=1)
    parser.add_argument("--sampler", choices=["identity", "random", "greedy_coreset", "approx_greedy_coreset"], default="random")
    parser.add_argument("--coreset-percentage", type=float, default=0.1)
    parser.add_argument("--faiss-num-workers", type=int, default=8)
    parser.add_argument("--normal-sampling", choices=NORMAL_SAMPLING_CHOICES, default="per_frequency")
    parser.add_argument("--max-pooled-train-normals", type=int, default=0)
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--rf-signals", nargs="+", default=list(JSR_BY_SIGNAL), choices=list(JSR_BY_SIGNAL))
    parser.add_argument("--public-rf-signals", nargs="+", default=list(PUBLIC_RF_JSRS), choices=list(PUBLIC_RF_JSRS))
    parser.add_argument("--rf-scenes", nargs="+", default=SCENES, choices=SCENES)
    parser.add_argument("--rf-train-mode", choices=["pooled", "per_cell"], default="pooled")
    parser.add_argument("--quantile", type=float, default=0.9)
    args = parser.parse_args()

    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    support = collect_support(args)
    if len(support) < 2:
        raise RuntimeError("Leave-one-out calibration needs at least two normal support images")

    normal_scores = []
    for index, held_out in enumerate(support):
        # Resetting the sampler makes each leave-one-out score reproducible.
        setup_seed(args.seed)
        gallery = support[:index] + support[index + 1 :]
        print(f"[{index + 1}/{len(support)}] score held-out normal: {held_out['path']}", flush=True)
        model = fit_patchcore(gallery, args, device)
        normal_scores.append(score_one_normal(model, held_out, args))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scores = np.asarray(normal_scores, dtype=np.float64)
    center = float(np.quantile(scores, args.quantile))
    body = scores[scores <= center]
    scale = float(np.std(body if len(body) else scores))
    payload = {
        "method": "patchcore_support_leave_one_out",
        "protocol": args.protocol,
        "normal_sampling": args.normal_sampling,
        "num_support": int(len(support)),
        "quantile": float(args.quantile),
        "center": center,
        "scale": max(scale, 1e-6),
        "support_scores": scores.tolist(),
        "support_paths": [str(sample["path"]) for sample in support],
        "uses_test_labels": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
