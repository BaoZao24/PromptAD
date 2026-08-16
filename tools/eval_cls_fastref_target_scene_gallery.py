#!/usr/bin/env python
"""Evaluate a FastRef-style prototype refinement gallery on target scenes.

The implementation follows FastRef's inference-only composition refinement:
each query image independently transfers its patch characteristics to the
support gallery through an alternating closed-form ``W`` update and Sinkhorn
transport ``T`` update. No query batch statistics or learned parameters are
shared across images.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PromptAD import PromptAD
from tools.eval_cls_vit_patchcore_gallery import (
    attach_formal_baseline,
    build_target_scene_eval_loaders,
    build_target_scene_support_loader,
    build_vit_nn_gallery_with_rows,
    prepare_patch_features,
    prepare_target_scene_manifest,
    robust_cosine_distance_chunked,
    top_ratio_score,
)
from tools.eval_seg_resnet_gallery_fusion import load_checkpoint
from train_rf_target_pooled_universal import to_model_input
from utils.training_utils import setup_seed


def safe_auc(labels, scores) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores) * 100.0)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def sinkhorn_transport(cost, epsilon, iterations):
    """Return a uniform-marginal entropy-regularized transport plan."""
    batch_size, num_query, num_proto = cost.shape
    epsilon = max(float(epsilon), 1e-5)
    scaled = cost - cost.amin(dim=(-2, -1), keepdim=True)
    kernel = torch.exp(-scaled / epsilon).clamp_min(1e-12)
    a = torch.full(
        (batch_size, num_query),
        1.0 / float(num_query),
        dtype=cost.dtype,
        device=cost.device,
    )
    b = torch.full(
        (batch_size, num_proto),
        1.0 / float(num_proto),
        dtype=cost.dtype,
        device=cost.device,
    )
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(max(1, int(iterations))):
        u = a / torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
        v = b / torch.bmm(kernel.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
    return u.unsqueeze(-1) * kernel * v.unsqueeze(-2)


class FastRefiner:
    """Per-query FastRef composition refinement for cosine features."""

    def __init__(
        self,
        gallery,
        ridge,
        refine_steps,
        sinkhorn_epsilon,
        sinkhorn_iterations,
        refinement_weight,
    ):
        self.gallery = F.normalize(gallery.float(), dim=-1).contiguous()
        self.refine_steps = max(0, int(refine_steps))
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iterations = max(1, int(sinkhorn_iterations))
        self.refinement_weight = max(0.0, float(refinement_weight))
        num_proto = int(self.gallery.shape[0])
        gram = self.gallery @ self.gallery.t()
        regularized = gram + max(float(ridge), 1e-8) * torch.eye(
            num_proto,
            dtype=gram.dtype,
            device=gram.device,
        )
        self.gram_inverse = torch.linalg.pinv(regularized)

    @torch.no_grad()
    def score(self, query_features):
        query_features = F.normalize(query_features.float(), dim=-1)
        gallery_t = self.gallery.t()
        base_coefficients = torch.matmul(query_features, gallery_t)
        base_coefficients = torch.matmul(base_coefficients, self.gram_inverse)
        coefficients = base_coefficients

        for _ in range(self.refine_steps):
            refined = F.normalize(torch.matmul(coefficients, self.gallery), dim=-1)
            cost = (1.0 - torch.matmul(refined, gallery_t)).mul(0.5).clamp_min(0.0)
            transport = sinkhorn_transport(
                cost,
                self.sinkhorn_epsilon,
                self.sinkhorn_iterations,
            )
            # Sinkhorn uses probability marginals (row mass 1/m).  Normalize
            # rows before applying lambda so the closed-form update has a
            # useful scale for patch features with different grid sizes.
            transport = transport / transport.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            weight = self.refinement_weight
            coefficients = (base_coefficients + weight * transport) / (1.0 + weight)

        refined = F.normalize(torch.matmul(coefficients, self.gallery), dim=-1)
        similarity = torch.matmul(query_features, refined.transpose(1, 2))
        patch_scores = (1.0 - similarity.amax(dim=-1)).mul(0.5)
        return patch_scores


def encode_visual_features(model, data, device):
    data_t = to_model_input(model, data, device, rgb_from_bgr=False)
    return prepare_patch_features(model.encode_image(data_t))


@torch.no_grad()
def evaluate_scene(
    model,
    gallery,
    eval_loaders,
    args,
    device,
):
    refiner = FastRefiner(
        gallery,
        ridge=args.fastref_ridge,
        refine_steps=args.fastref_steps,
        sinkhorn_epsilon=args.fastref_epsilon,
        sinkhorn_iterations=args.fastref_sinkhorn_iterations,
        refinement_weight=args.fastref_lambda,
    )
    gallery = refiner.gallery
    rows = []
    score_dir = Path(args.output_root) / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    for signal, scene, jsr, loader in eval_loaders:
        labels, names = [], []
        memory_scores, fastref_scores = [], []
        memory_ratio_scores = {ratio: [] for ratio in args.map_top_ratios}
        fastref_ratio_scores = {ratio: [] for ratio in args.map_top_ratios}

        for data, _mask, label, name, _img_type in tqdm(
            loader,
            desc=f"Eval FastRef {signal}/{scene}/{jsr}",
            leave=False,
        ):
            query_features = encode_visual_features(model, data, device)
            batch_size, num_patches, _ = query_features.shape
            grid_side = int(round(np.sqrt(num_patches)))
            if grid_side * grid_side != num_patches:
                raise RuntimeError(f"CLIP patch count is not square: {num_patches}")

            memory_map = robust_cosine_distance_chunked(
                query_features.reshape(-1, query_features.shape[-1]),
                gallery,
                args.gallery_chunk_size,
                args,
            ).reshape(batch_size, grid_side, grid_side)
            fastref_map = refiner.score(query_features).reshape(
                batch_size,
                grid_side,
                grid_side,
            )
            memory_np = memory_map.detach().cpu().numpy().astype(np.float32)
            fastref_np = fastref_map.detach().cpu().numpy().astype(np.float32)
            memory_scores.extend(memory_np.reshape(batch_size, -1).max(axis=1).tolist())
            fastref_scores.extend(fastref_np.reshape(batch_size, -1).max(axis=1).tolist())
            for ratio in args.map_top_ratios:
                memory_ratio_scores[ratio].extend(top_ratio_score(memory_np, ratio).tolist())
                fastref_ratio_scores[ratio].extend(top_ratio_score(fastref_np, ratio).tolist())
            labels.extend(int(value) for value in label.numpy().tolist())
            names.extend(list(name))

        labels_np = np.asarray(labels, dtype=np.int32)
        memory_np = np.asarray(memory_scores, dtype=np.float32)
        fastref_np = np.asarray(fastref_scores, dtype=np.float32)
        row = {
            "method": "fastref_target_scene_gallery",
            "task": "cls",
            "dataset": signal,
            "scene": scene,
            "jsr": jsr,
            "patch_layer": "concat",
            "coreset_ratio": args.coreset_ratio,
            "nn_topk": args.nn_topk,
            "nn_agg": args.nn_agg,
            "coreset_method": args.coreset_method,
            "fastref_steps": args.fastref_steps,
            "fastref_lambda": args.fastref_lambda,
            "fastref_epsilon": args.fastref_epsilon,
            "fastref_sinkhorn_iterations": args.fastref_sinkhorn_iterations,
            "fastref_ridge": args.fastref_ridge,
            "memory_max_auc": safe_auc(labels, memory_np),
            "fastref_max_auc": safe_auc(labels, fastref_np),
        }
        payload = {
            "names": np.asarray(names),
            "labels": labels_np,
            "memory_max_scores": memory_np,
            "fastref_max_scores": fastref_np,
        }
        for ratio in args.map_top_ratios:
            key = f"{ratio:g}".replace(".", "p")
            memory_values = np.asarray(memory_ratio_scores[ratio], dtype=np.float32)
            fastref_values = np.asarray(fastref_ratio_scores[ratio], dtype=np.float32)
            row[f"memory_top{key}_auc"] = safe_auc(labels, memory_values)
            row[f"fastref_top{key}_auc"] = safe_auc(labels, fastref_values)
            payload[f"memory_top{key}_scores"] = memory_values
            payload[f"fastref_top{key}_scores"] = fastref_values
        if getattr(args, "support_manifest_sha256", ""):
            payload["support_manifest_sha256"] = np.asarray(args.support_manifest_sha256)
        rows.append(row)
        np.savez_compressed(score_dir / f"{signal}-{scene}-{jsr}-scores.npz", **payload)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="FastRef target-scene RF classification")
    parser.add_argument("--output-root", default="analysis_outputs/exploratory/20260814_fastref_target_scene_gallery")
    parser.add_argument("--formal-baseline-csv", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--signals", nargs="+", default=["burst_signal"], choices=["burst_signal", "chirp_signal", "dsss_signal", "pulse_signal", "deceptive_signal"])
    parser.add_argument("--scenes", nargs="+", default=["WeaponMuseum_spectrum"], choices=["WeaponMuseum_spectrum", "Playground_spectrum", "TimeSquare_spectrum", "Gymnasium_spectrum"])
    parser.add_argument("--normal-sampling", choices=("per_frequency", "1shot", "2shot", "4shot"), default="per_frequency")
    parser.add_argument("--support-manifest", default="")
    parser.add_argument("--coreset-ratio", type=float, default=0.1)
    parser.add_argument("--coreset-method", choices=["random", "farthest"], default="farthest")
    parser.add_argument("--rowwise-coreset", action="store_true")
    parser.add_argument("--support-augment", choices=["none", "rf_weak", "time_shift"], default="none")
    parser.add_argument("--support-shift-px", type=int, default=3)
    parser.add_argument("--support-crop-ratio", type=float, default=0.03)
    parser.add_argument("--paired-tta", choices=["none"], default="none")
    parser.add_argument("--gallery-chunk-size", type=int, default=4096)
    parser.add_argument("--nn-topk", type=int, default=1)
    parser.add_argument("--nn-agg", choices=["mean", "weighted", "adaptive"], default="mean")
    parser.add_argument("--nn-weight-temp", type=float, default=0.05)
    parser.add_argument("--adaptive-sim-margin", type=float, default=0.02)
    parser.add_argument("--fastref-steps", type=int, default=2)
    parser.add_argument("--fastref-lambda", type=float, default=1.0)
    parser.add_argument("--fastref-epsilon", type=float, default=0.05)
    parser.add_argument("--fastref-sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--fastref-ridge", type=float, default=1e-3)
    parser.add_argument("--map-top-ratios", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--support-seed", type=int, default=None)
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
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0.0 < args.coreset_ratio <= 1.0):
        raise ValueError("--coreset-ratio must be in (0, 1]")
    if args.fastref_steps < 0 or args.fastref_sinkhorn_iterations <= 0:
        raise ValueError("FastRef iteration counts must be non-negative/positive")
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cpu" if args.use_cpu else "cuda:0"

    kwargs = vars(args).copy()
    kwargs.update({
        "dataset": "rf_target_test_pool",
        "class_name": "signal",
        "device": device,
        "out_size_h": args.resolution,
        "out_size_w": args.resolution,
    })
    model = PromptAD(**kwargs).to(device)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
    model.eval_mode()

    manifest = prepare_target_scene_manifest(args)
    rows = []
    selected_normals = []
    gallery_feature_dim = None
    gallery_patch_counts = {}
    for scene in args.scenes:
        train_loader, scene_samples = build_target_scene_support_loader(manifest, scene, args)
        gallery, _gallery_rows, _gallery_cols = build_vit_nn_gallery_with_rows(
            model,
            train_loader,
            args,
            device,
        )
        rows.extend(
            evaluate_scene(
                model,
                gallery,
                build_target_scene_eval_loaders(manifest, scene, args),
                args,
                device,
            )
        )
        selected_normals.extend(sample[0] for sample in scene_samples)
        gallery_feature_dim = int(gallery.shape[1])
        gallery_patch_counts[scene] = int(gallery.shape[0])

    rows = attach_formal_baseline(rows, args.formal_baseline_csv)
    out_root = Path(args.output_root)
    result_path = out_root / "results_cls_fastref_target_scene_gallery.csv"
    write_csv(result_path, rows)
    df = pd.DataFrame(rows)
    summary = {
        "method": "fastref_target_scene_gallery",
        "support_protocol": "target_scene",
        "support_manifest": args.support_manifest or None,
        "support_manifest_sha256": getattr(args, "support_manifest_sha256", None),
        "normal_sampling": args.normal_sampling,
        "coreset_ratio": args.coreset_ratio,
        "coreset_method": args.coreset_method,
        "nn_topk": args.nn_topk,
        "nn_agg": args.nn_agg,
        "fastref_steps": args.fastref_steps,
        "fastref_lambda": args.fastref_lambda,
        "fastref_epsilon": args.fastref_epsilon,
        "fastref_sinkhorn_iterations": args.fastref_sinkhorn_iterations,
        "fastref_ridge": args.fastref_ridge,
        "gallery_feature_dim": gallery_feature_dim,
        "gallery_patch_counts": gallery_patch_counts,
        "selected_normal_count": len(selected_normals),
        "selected_normals": selected_normals,
        "num_cells": len(rows),
    }
    for column in [name for name in df.columns if name.endswith("_auc")]:
        summary[f"{column}_macro"] = float(df[column].mean())
    for column in [name for name in df.columns if name.startswith("delta_")]:
        summary[f"{column}_macro"] = float(df[column].mean())
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (out_root / "README.md").write_text(
        "# FastRef Target-Scene Gallery\n\n"
        "Per-query FastRef-style composition refinement over the support-only CLIP-ViT gallery.\n",
        encoding="utf-8",
    )
    print(f"wrote {result_path}")
    print(f"wrote {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
