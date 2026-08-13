#!/usr/bin/env python3
"""Bounded RF feature-level fusion search.

This evaluator keeps the formal ViT/CNN branches frozen and builds a normal-only
patch memory from ViT block 8 tokens and ResNet18 layer2 local features.  It is
intentionally separate from the formal score-gate evaluator: exploratory
fusion choices cannot silently change the paper baseline.
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
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PromptAD import PromptAD
from tools.eval_cls_public_rf_dual_gallery import (
    PUBLIC_SIGNALS,
    build_eval_loaders as build_public_eval_loaders,
    build_train_loader as build_public_train_loader,
)
from tools.eval_cls_vit_patchcore_gallery import (
    SCENES as SELF_SCENES,
    SIGNALS as SELF_SIGNALS,
    build_target_scene_eval_loaders,
    build_target_scene_support_loader,
    prepare_target_scene_manifest,
)
from tools.eval_seg_resnet_gallery_fusion import ResNet18LocalEncoder, load_checkpoint
from train_rf_target_pooled_universal import to_model_input
from tools.eval_cls_public_rf_dual_gallery import parse_jsrs_by_signal
from utils.training_utils import setup_seed


DEFAULT_METHODS = [
    "vit_block8",
    "cnn_layer2",
    "norm_concat_a025",
    "norm_concat_a050",
    "norm_concat_a100",
    "whiten_concat_a050",
    "pca_product",
    "ridge_residual_a050",
    "ridge_product",
    "agreement_gate_a050",
    "joint_whiten",
    "cca_product",
    "cca_diff",
    "centered_ridge_product",
    "centered_ridge_diff",
    "adaptive_residual_a050",
    "residual_only",
    "film_a025",
    "film_a050",
    "film_a100",
    "centered_film_a050",
    "symmetric_interaction_a050",
]


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1]).float()


def fit_pca(x: torch.Tensor, dim: int) -> dict:
    x = x.float()
    mean = x.mean(dim=0)
    centered = x - mean
    rank = max(1, min(int(dim), centered.shape[0] - 1, centered.shape[1]))
    if rank == centered.shape[1]:
        components = torch.eye(centered.shape[1], dtype=torch.float32)
    else:
        _u, _s, v = torch.pca_lowrank(centered, q=rank, center=False, niter=2)
        components = v[:, :rank].contiguous()
    return {"mean": mean, "components": components}


def inverse_sqrt(matrix: torch.Tensor, ridge: float) -> torch.Tensor:
    """Stable inverse square root for a small covariance matrix."""

    matrix = (matrix + matrix.t()) * 0.5
    values, vectors = torch.linalg.eigh(matrix)
    values = values.clamp_min(0.0) + float(ridge)
    return (vectors * values.rsqrt().unsqueeze(0)) @ vectors.t()


def apply_pca(x: torch.Tensor, state: dict) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1]).float()
    projected = (flat - state["mean"].to(flat.device)) @ state["components"].to(flat.device)
    return projected.reshape(*x.shape[:-1], projected.shape[-1])


def fit_fusion_state(vit: torch.Tensor, cnn: torch.Tensor, args) -> dict:
    """Fit only normal-support statistics and cross-modal alignment."""

    v = flatten_tokens(vit)
    c = flatten_tokens(cnn)
    v = normalize_rows(v)
    c = normalize_rows(c)
    state = {
        "vit_mean": v.mean(dim=0),
        "vit_std": v.std(dim=0, unbiased=False).clamp_min(1e-4),
        "cnn_mean": c.mean(dim=0),
        "cnn_std": c.std(dim=0, unbiased=False).clamp_min(1e-4),
        "pca_vit": fit_pca(v, args.pca_dim),
        "pca_cnn": fit_pca(c, args.pca_dim),
    }

    # Normal-only ridge alignment maps local CNN tokens into the high-level
    # ViT block-8 space.  The mapping is fitted on spatially corresponding
    # support patches and is used without abnormal labels.
    ridge = float(args.ridge_lambda)
    gram = (c.t() @ c) / max(1, c.shape[0])
    gram = gram + ridge * torch.eye(c.shape[1], dtype=c.dtype)
    cross = (c.t() @ v) / max(1, c.shape[0])
    state["ridge"] = torch.linalg.solve(gram, cross)

    # Centered ridge alignment includes an intercept implicitly.  This avoids
    # forcing the CNN branch through the origin, which is especially useful
    # when the two frozen encoders have different normal-feature means.
    c_mean = c.mean(dim=0)
    v_mean = v.mean(dim=0)
    c_centered = c - c_mean
    v_centered = v - v_mean
    centered_gram = (c_centered.t() @ c_centered) / max(1, c.shape[0])
    centered_gram = centered_gram + ridge * torch.eye(c.shape[1], dtype=c.dtype)
    centered_cross = (c_centered.t() @ v_centered) / max(1, c.shape[0])
    state["centered_ridge"] = torch.linalg.solve(centered_gram, centered_cross)
    state["centered_c_mean"] = c_mean
    state["centered_v_mean"] = v_mean

    # Joint diagonal whitening preserves a single metric over both modalities
    # instead of normalizing the two branches independently.
    joint = torch.cat([v, c], dim=-1)
    state["joint_mean"] = joint.mean(dim=0)
    state["joint_std"] = joint.std(dim=0, unbiased=False).clamp_min(1e-4)

    # CCA is fitted only on paired normal patches.  It retains directions that
    # are correlated across the global ViT and local CNN representations, then
    # adds an explicit cross-modal interaction term at inference time.
    v_p = apply_pca(v, state["pca_vit"]).reshape(-1, state["pca_vit"]["components"].shape[1])
    c_p = apply_pca(c, state["pca_cnn"]).reshape(-1, state["pca_cnn"]["components"].shape[1])
    v_p_mean = v_p.mean(dim=0)
    c_p_mean = c_p.mean(dim=0)
    v_pc = v_p - v_p_mean
    c_pc = c_p - c_p_mean
    denom = max(1, v_pc.shape[0])
    v_cov = (v_pc.t() @ v_pc) / denom
    c_cov = (c_pc.t() @ c_pc) / denom
    vc_cov = (v_pc.t() @ c_pc) / denom
    v_inv_sqrt = inverse_sqrt(v_cov, ridge)
    c_inv_sqrt = inverse_sqrt(c_cov, ridge)
    whitened_cross = v_inv_sqrt @ vc_cov @ c_inv_sqrt
    u, _s, vh = torch.linalg.svd(whitened_cross, full_matrices=False)
    cca_dim = max(1, min(int(args.cca_dim), u.shape[1], vh.shape[0]))
    state["cca_v_mean"] = v_p_mean
    state["cca_c_mean"] = c_p_mean
    state["cca_v_map"] = v_inv_sqrt @ u[:, :cca_dim]
    state["cca_c_map"] = c_inv_sqrt @ vh.t()[:, :cca_dim]
    return state


def transform_features(method: str, vit: torch.Tensor, cnn: torch.Tensor, state: dict) -> torch.Tensor:
    """Create a patch feature for one candidate method."""

    v = normalize_rows(vit)
    c = normalize_rows(cnn)
    if method == "vit_block8":
        return v
    if method == "cnn_layer2":
        return c

    if method.startswith("norm_concat_a"):
        alpha = float(method.removeprefix("norm_concat_a")) / 100.0
        return normalize_rows(torch.cat([v, alpha * c], dim=-1))

    if method == "whiten_concat_a050":
        v_w = normalize_rows((v - state["vit_mean"].to(v.device)) / state["vit_std"].to(v.device))
        c_w = normalize_rows((c - state["cnn_mean"].to(c.device)) / state["cnn_std"].to(c.device))
        return normalize_rows(torch.cat([v_w, 0.5 * c_w], dim=-1))

    if method == "pca_product":
        v_p = normalize_rows(apply_pca(v, state["pca_vit"]))
        c_p = normalize_rows(apply_pca(c, state["pca_cnn"]))
        return normalize_rows(torch.cat([v_p, c_p, v_p * c_p], dim=-1))

    c_projected = normalize_rows(c @ state["ridge"].to(c.device))
    if method.startswith("ridge_residual_a"):
        alpha = float(method.removeprefix("ridge_residual_a")) / 100.0
        return normalize_rows(v + alpha * c_projected)
    if method == "ridge_product":
        return normalize_rows(torch.cat([v, c_projected, v * c_projected], dim=-1))
    if method == "agreement_gate_a050":
        agreement = ((v * c_projected).sum(dim=-1, keepdim=True) + 1.0) * 0.5
        return normalize_rows(torch.cat([v, 0.5 * agreement * c_projected], dim=-1))
    if method == "joint_whiten":
        joint = torch.cat([v, c], dim=-1)
        joint = (joint - state["joint_mean"].to(joint.device)) / state["joint_std"].to(joint.device)
        return normalize_rows(joint)
    if method in {
        "centered_ridge_product",
        "centered_ridge_diff",
        "adaptive_residual_a050",
        "residual_only",
        "centered_film_a050",
    }:
        c_mean = state["centered_c_mean"].to(c.device)
        v_mean = state["centered_v_mean"].to(c.device)
        centered_projected = (c - c_mean) @ state["centered_ridge"].to(c.device) + v_mean
        c_centered_projected = normalize_rows(centered_projected)
        residual = v - c_centered_projected
        if method == "centered_ridge_product":
            return normalize_rows(torch.cat([v, c_centered_projected, v * c_centered_projected], dim=-1))
        if method == "centered_ridge_diff":
            return normalize_rows(torch.cat([v, c_centered_projected, residual.abs()], dim=-1))
        if method == "residual_only":
            return normalize_rows(residual)
        if method == "centered_film_a050":
            return normalize_rows(v * (1.0 + 0.5 * c_centered_projected))
        agreement = ((v * c_centered_projected).sum(dim=-1, keepdim=True) + 1.0) * 0.5
        anomaly_weight = 1.0 - agreement
        return normalize_rows(v + 0.5 * anomaly_weight * residual.abs())
    if method in {"film_a025", "film_a050", "film_a100", "symmetric_interaction_a050"}:
        if method == "symmetric_interaction_a050":
            return normalize_rows(v + 0.5 * c_projected + 0.5 * (v * c_projected))
        alpha = float(method.removeprefix("film_a")) / 100.0
        return normalize_rows(v * (1.0 + alpha * c_projected))
    if method == "cca_product" or method == "cca_diff":
        v_p = apply_pca(v, state["pca_vit"])
        c_p = apply_pca(c, state["pca_cnn"])
        v_c = (v_p - state["cca_v_mean"].to(v.device)) @ state["cca_v_map"].to(v.device)
        c_c = (c_p - state["cca_c_mean"].to(c.device)) @ state["cca_c_map"].to(c.device)
        v_c = normalize_rows(v_c)
        c_c = normalize_rows(c_c)
        if method == "cca_product":
            return normalize_rows(torch.cat([v_c, c_c, v_c * c_c], dim=-1))
        return normalize_rows(torch.cat([v_c, c_c, (v_c - c_c).abs()], dim=-1))
    raise ValueError(f"Unsupported fusion method: {method}")


def deterministic_subset(count: int, limit: int, seed: int) -> torch.Tensor:
    if limit <= 0 or count <= limit:
        return torch.arange(count, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(count, generator=generator)[: int(limit)]


@torch.no_grad()
def extract_features(model, encoder, data, device, grid_size):
    model_input = to_model_input(model, data, device, rgb_from_bgr=True)
    visual = model.encode_image(model_input)
    if len(visual) < 4:
        raise RuntimeError(f"PromptAD returned {len(visual)} visual feature groups; block8 needs index 3")
    vit = normalize_rows(visual[3].float())
    raw = data.permute(0, 3, 1, 2).to(device, non_blocking=True)
    cnn_map = encoder(raw)["layer2"]
    cnn_map = F.adaptive_avg_pool2d(cnn_map.float(), tuple(grid_size))
    cnn = cnn_map.permute(0, 2, 3, 1).reshape(vit.shape[0], vit.shape[1], -1)
    cnn = normalize_rows(cnn)
    if cnn.shape[:2] != vit.shape[:2]:
        raise RuntimeError(f"Spatial alignment failed: vit={tuple(vit.shape)}, cnn={tuple(cnn.shape)}")
    return vit.detach(), cnn.detach()


@torch.no_grad()
def collect_support_features(model, encoder, loader, args, device):
    vit_chunks, cnn_chunks = [], []
    for data, _mask, _label, _name, _img_type in tqdm(loader, desc="Collect normal support features", leave=False):
        vit, cnn = extract_features(model, encoder, data, device, model.grid_size)
        vit_chunks.append(vit.cpu())
        cnn_chunks.append(cnn.cpu())
    vit = torch.cat(vit_chunks, dim=0)
    cnn = torch.cat(cnn_chunks, dim=0)
    keep = deterministic_subset(vit.shape[0] * vit.shape[1], args.max_fit_patches, args.seed)
    vit = vit.reshape(-1, vit.shape[-1])[keep].reshape(1, -1, vit.shape[-1])
    cnn = cnn.reshape(-1, cnn.shape[-1])[keep].reshape(1, -1, cnn.shape[-1])
    return vit, cnn


@torch.no_grad()
def collect_query_features(model, encoder, loader, args, device):
    batches = []
    for data, _mask, label, name, _img_type in tqdm(loader, desc="Collect query features", leave=False):
        vit, cnn = extract_features(model, encoder, data, device, model.grid_size)
        batches.append((vit.cpu(), cnn.cpu(), np.asarray(label.numpy(), dtype=np.int32), list(name)))
    return batches


def build_galleries(vit_support, cnn_support, methods, state, args):
    galleries = {}
    count = vit_support.shape[1]
    keep = deterministic_subset(count, args.max_gallery_patches, args.seed + 17)
    v = vit_support[:, keep].reshape(1, -1, vit_support.shape[-1])
    c = cnn_support[:, keep].reshape(1, -1, cnn_support.shape[-1])
    for method in methods:
        fused = transform_features(method, v, c, state)
        galleries[method] = fused.reshape(-1, fused.shape[-1]).contiguous()
    return galleries


@torch.no_grad()
def nearest_distance_map(probe, gallery, chunk_size, nn_topk):
    flat = probe.reshape(-1, probe.shape[-1])
    k = max(1, min(int(nn_topk), int(gallery.shape[0])))
    distances = []
    for start in range(0, flat.shape[0], max(1, int(chunk_size))):
        chunk = flat[start : start + max(1, int(chunk_size))]
        similarities = chunk @ gallery.t()
        top = similarities.topk(k, dim=1).values.mean(dim=1)
        distances.append(((1.0 - top) / 2.0).cpu())
    return torch.cat(distances, dim=0).reshape(probe.shape[0], probe.shape[1])


def image_scores(distance_map, ratios):
    out = {"max": distance_map.max(dim=1).values.cpu().numpy()}
    for ratio in ratios:
        k = max(1, int(round(distance_map.shape[1] * float(ratio))))
        key = f"top{str(ratio).replace('.', 'p')}"
        out[key] = distance_map.topk(k, dim=1).values.mean(dim=1).cpu().numpy()
    return out


def metric(labels, scores):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if np.unique(labels).size < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "fpr95": 100.0}
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels, scores) * 100.0),
        "auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if len(reached) else 100.0,
    }


@torch.no_grad()
def evaluate_cell(model, encoder, support_loader, eval_loaders, methods, args, device, cell_prefix):
    support_vit, support_cnn = collect_support_features(model, encoder, support_loader, args, device)
    state = fit_fusion_state(support_vit, support_cnn, args)
    galleries_cpu = build_galleries(support_vit, support_cnn, methods, state, args)
    rows = []
    for signal, scene, jsr, loader in eval_loaders:
        query_batches = collect_query_features(model, encoder, loader, args, device)
        labels = np.concatenate([item[2] for item in query_batches], axis=0)
        for method in methods:
            gallery = galleries_cpu[method].to(device)
            score_bank = {}
            for vit_cpu, cnn_cpu, _labels, _names in query_batches:
                vit = vit_cpu.to(device, non_blocking=True)
                cnn = cnn_cpu.to(device, non_blocking=True)
                fused = transform_features(method, vit, cnn, state)
                distances = nearest_distance_map(fused, gallery, args.gallery_chunk_size, args.nn_topk)
                for key, values in image_scores(distances, args.top_ratios).items():
                    score_bank.setdefault(key, []).append(values)
            del gallery
            for key, chunks in score_bank.items():
                scores = np.concatenate(chunks, axis=0)
                values = metric(labels, scores)
                rows.append({
                    "cell_prefix": cell_prefix,
                    "dataset": args.dataset,
                    "signal": signal,
                    "scene": scene,
                    "jsr": jsr,
                    "method": method,
                    "score": key,
                    **values,
                    "num_samples": int(labels.shape[0]),
                })
    del galleries_cpu, state, support_vit, support_cnn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_model(args, device):
    kwargs = vars(args).copy()
    kwargs.update({
        "dataset": "rf_target_test_pool" if args.dataset == "self_rf" else "rf_spe_png",
        "class_name": "signal" if args.dataset == "self_rf" else "radio frequency spectrogram",
        "device": device,
        "out_size_h": args.resolution,
        "out_size_w": args.resolution,
    })
    model = PromptAD(**kwargs).to(device)
    load_checkpoint(model, args.checkpoint)
    model.eval_mode()
    encoder = ResNet18LocalEncoder().to(device)
    encoder.eval()
    return model, encoder


def run_dataset(args, model, encoder, device):
    rows = []
    if args.dataset == "self_rf":
        manifest = prepare_target_scene_manifest(args)
        for scene in args.scenes:
            support_loader, _samples = build_target_scene_support_loader(manifest, scene, args)
            eval_loaders = build_target_scene_eval_loaders(manifest, scene, args)
            rows.extend(evaluate_cell(model, encoder, support_loader, eval_loaders, args.methods, args, device, scene))
    else:
        support_loader = build_public_train_loader(args)
        eval_loaders = build_public_eval_loaders(args)
        rows.extend(evaluate_cell(model, encoder, support_loader, eval_loaders, args.methods, args, device, "public_rf"))
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("self_rf", "public_rf"), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", default="analysis_outputs/02_current_baselines/promptad_formal_baseline/pooled_rf_rgb_cls/checkpoint/overall-best.pt")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--signals", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--jsrs-by-signal", default=None)
    parser.add_argument("--support-manifest", default="")
    parser.add_argument("--support-seed", type=int, default=111)
    parser.add_argument("--normal-sampling", default="per_frequency")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--max-fit-patches", type=int, default=20000)
    parser.add_argument("--max-gallery-patches", type=int, default=20000)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--cca-dim", type=int, default=32)
    parser.add_argument("--ridge-lambda", type=float, default=0.05)
    parser.add_argument("--nn-topk", type=int, default=5)
    parser.add_argument("--gallery-chunk-size", type=int, default=2048)
    parser.add_argument("--top-ratios", type=float, nargs="+", default=[0.01, 0.05, 0.1])
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
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--use-cpu", type=int, default=0)
    args = parser.parse_args()
    if args.signals is None:
        args.signals = list(SELF_SIGNALS if args.dataset == "self_rf" else PUBLIC_SIGNALS)
    if args.scenes is None:
        args.scenes = list(SELF_SCENES)
    if not args.support_manifest:
        args.support_manifest = ""
    return args


def main():
    args = parse_args()
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cpu" if args.use_cpu else "cuda:0"
    if args.dataset == "public_rf":
        args.jsrs_by_signal = args.jsrs_by_signal
        parse_jsrs_by_signal(args.jsrs_by_signal)

    model, encoder = build_model(args, device)
    rows = run_dataset(args, model, encoder, device)
    if not rows:
        raise RuntimeError("No feature-fusion rows were produced")
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    write_rows(out_root / "feature-fusion-results.csv", rows)
    frame = pd.DataFrame(rows)
    summary = {
        "dataset": args.dataset,
        "methods": args.methods,
        "vit_feature": "block8 (visual_features[3])",
        "cnn_feature": "resnet18.layer2",
        "cnn_alignment": "adaptive_avg_pool2d to ViT patch grid",
        "support_seed": args.support_seed,
        "support_manifest": args.support_manifest,
        "max_fit_patches": args.max_fit_patches,
        "max_gallery_patches": args.max_gallery_patches,
        "pca_dim": args.pca_dim,
        "ridge_lambda": args.ridge_lambda,
        "nn_topk": args.nn_topk,
        "top_ratios": args.top_ratios,
        "metrics": {},
    }
    for (method, score), group in frame.groupby(["method", "score"]):
        summary["metrics"].setdefault(method, {})[score] = {
            "auroc": float(group["auroc"].mean()),
            "auprc": float(group["auprc"].mean()),
            "fpr95": float(group["fpr95"].mean()),
            "num_cells": int(len(group)),
        }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_root / "protocol.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
