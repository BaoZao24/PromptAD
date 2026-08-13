#!/usr/bin/env python
"""Evaluate the formal safe support-only method on target-scene OFDMA cold start.

The public CLI intentionally exposes only protocol choices.  Method settings
are fixed below so the entry cannot silently drift from the formal method.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PromptAD import PromptAD
from datasets.ofdma_spectrum import (
    JAMMER_TYPES,
    NO_JAMMER,
    NUM_SUS,
    OFDMASpectrogramPreprocessor,
)
from datasets.ofdma_target_scene import (
    DEFAULT_TARGET_SCENE_ROOT,
    TargetSceneOFDMAPathDataset,
    available_target_scenes,
    build_target_scene_records,
    load_target_scene_manifest,
    parse_target_scene_name,
)
from tools.eval_cls_resnet_gallery_fusion import resnet_patch_score_bank
from tools.eval_cls_vit_patchcore_gallery import (
    _paired_tta_modes,
    build_vit_nn_gallery_with_rows,
    compute_patch_map,
    prepare_patch_features,
    spectrogram_nn_scores,
)
from tools.eval_seg_resnet_gallery_fusion import (
    ResNet18LocalEncoder,
    build_resnet_gallery,
    cnn_tta_batch,
    load_checkpoint,
)
from train_rf_target_pooled_universal import to_model_input
from utils.confidence_gate import (
    SAFE_SUPPORT_ALPHA,
    SAFE_SUPPORT_RANK_QUANTILE,
    SAFE_SUPPORT_TEMPERATURE,
    confidence_gated_or,
    safe_support_only_gate,
)
from utils.training_utils import setup_seed


METHOD = "confidence_gated_dual_visual"
VIT_BRANCH = "ours_vit"
CNN_BRANCH = "ours_cnn"

# Frozen formal configuration.  These are recorded in protocol.json and are
# deliberately not command-line options.
FORMAL = {
    "seed": 111,
    "resolution": 240,
    "img_resize": 240,
    "img_cropsize": 240,
    "geometry": "letterbox",
    "subcarriers_per_rb": 12,
    "backbone": "ViT-B-16-plus-240",
    "pretrained_dataset": "laion400m_e32",
    "prompt_mode": "legacy",
    "input_mode": "rgb",
    "text_prototype_mode": "single",
    "cls_score_mode": "text_only",
    "n_ctx": 4,
    "n_ctx_ab": 1,
    "n_pro": 3,
    "n_pro_ab": 4,
    "coreset_ratio": 0.5,
    "coreset_method": "farthest",
    "rowwise_coreset": False,
    "memory_mode": "global_nn",
    "support_augment": "none",
    "paired_tta": "ofdma_time_shift_blur",
    "paired_tta_fusion": "max",
    "paired_tta_shift_px": 4,
    "paired_tta_blur_ksize": 3,
    "gallery_chunk_size": 4096,
    "patch_score_rank": 3,
    "freq_window": -1,
    "nn_topk": 1,
    "nn_agg": "mean",
    "nn_weight_temp": 0.05,
    "adaptive_sim_margin": 0.02,
    "position_soft_axis": "frequency",
    "position_soft_weight": 0.0,
    "resnet_layers": ["layer3"],
    "max_gallery_patches": 50000,
    "cnn_coreset_size": 0,
    "cnn_nn_topk": 1,
    "cnn_image_top_ratio": 0.1,
    "distance_chunk_size": 1024,
    "cnn_paired_tta": "none",
    "cnn_paired_tta_shift_px": 4,
    "cnn_paired_tta_blur_ksize": 3,
    "cnn_input_normalization": "raw",
}

# These views are part of the existing OFDMA support-only protocol.  The
# exploratory bundle changes only the query/gallery TTA views; it does not
# rewrite this calibration protocol.
SUPPORT_REFERENCE_VIEWS = (
    "time_shift_up",
    "time_shift_down",
    "time_shift_up_large",
    "time_shift_down_large",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ours-only evaluation for target-scene OFDMA cold start."
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_TARGET_SCENE_ROOT))
    parser.add_argument(
        "--output-root",
        default="analysis_outputs/20260726_ofdma_target_scene_ours",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "analysis_outputs/02_current_baselines/promptad_formal_baseline/"
            "pooled_rf_rgb_cls/checkpoint/overall-best.pt"
        ),
    )
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--scene-ids",
        nargs="*",
        default=[],
        help="Target scene IDs; empty means all scenes in the selected split.",
    )
    parser.add_argument(
        "--max-normal-observations",
        type=int,
        default=0,
        help="Smoke-test limit only; zero uses all normal test observations.",
    )
    parser.add_argument(
        "--max-anomaly-observations-per-type",
        type=int,
        default=0,
        help="Smoke-test limit per jammer; zero uses all anomaly observations.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--gate-protocol",
        choices=["support_only", "transductive"],
        default="support_only",
        help=(
            "Formal default is support_only; transductive is retained only "
            "for protocol comparison."
        ),
    )
    parser.add_argument(
        "--support-reference-root",
        default="",
        help=(
            "Optional directory containing <scene>_<shot>shot.npz support "
            "references. If omitted, references are scored in the same run."
        ),
    )
    parser.add_argument(
        "--support-reference-only",
        action="store_true",
        help=(
            "Score held-out views of normal support only and write reference "
            "branch scores; do not evaluate test observations."
        ),
    )
    parser.add_argument(
        "--support-seed",
        type=int,
        default=None,
        help=(
            "Optional seed for selecting complete normal support observations "
            "within each target scene. Omitted keeps the formal nested first-k rule."
        ),
    )
    return parser.parse_args()


def formal_args(cli: argparse.Namespace) -> SimpleNamespace:
    values = vars(cli).copy()
    values.update(FORMAL)
    return SimpleNamespace(**values)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_loader(records, preprocessor, args, *, support: bool) -> DataLoader:
    dataset = TargetSceneOFDMAPathDataset(records, preprocessor)
    return DataLoader(
        dataset,
        batch_size=min(args.batch_size, max(1, len(dataset))),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not args.use_cpu,
        persistent_workers=args.num_workers > 0 and not support,
    )


def patch_map_image_score(patch_map: torch.Tensor) -> np.ndarray:
    # Ignore the two most extreme patch responses. Validation showed that
    # they are often isolated normal noise amplified by the later SU maximum.
    rank = int(FORMAL["patch_score_rank"])
    ranked = torch.topk(patch_map.flatten(1), k=rank, dim=1).values
    return ranked[:, rank - 1].cpu().numpy().astype(np.float32)


def support_reference_view_batch(raw_batch: torch.Tensor, mode: str, args) -> torch.Tensor:
    """Apply a deterministic held-out view to an OFDMA support batch."""

    shift = int(args.paired_tta_shift_px)
    variants = []
    for arr_tensor in raw_batch:
        arr = arr_tensor.numpy()
        if mode == "time_shift_up":
            dx, dy = 0, -shift
        elif mode == "time_shift_down":
            dx, dy = 0, shift
        elif mode == "time_shift_up_large":
            dx, dy = 0, -2 * shift
        elif mode == "time_shift_down_large":
            dx, dy = 0, 2 * shift
        else:
            raise ValueError(f"Unsupported OFDMA support reference view: {mode}")
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        height, width = arr.shape[:2]
        variants.append(
            cv2.warpAffine(
                arr,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        )
    return torch.as_tensor(np.stack(variants, axis=0))


@torch.no_grad()
def build_memories(model, encoder, support_loader, args, device):
    paired_galleries = []
    for mode in _paired_tta_modes(args):
        gallery, rows, cols = build_vit_nn_gallery_with_rows(
            model,
            support_loader,
            args,
            device,
            paired_tta_mode=mode,
        )
        paired_galleries.append((mode, gallery, rows, cols))
    cnn_galleries = build_resnet_gallery(
        encoder,
        support_loader,
        args,
        device,
        tta_mode="identity",
    )
    return paired_galleries, cnn_galleries[args.resnet_layers[0]]


@torch.no_grad()
def evaluate_su_scores(
    model,
    encoder,
    paired_galleries,
    cnn_gallery,
    loader,
    args,
    device,
):
    model.eval_mode()
    encoder.eval()
    names: list[str] = []
    labels: list[int] = []
    jammer_types: list[str] = []
    vit_scores: list[float] = []
    cnn_scores: list[float] = []

    for data, _, label, name, jammer_type in tqdm(
        loader, desc="Score target-scene SUs", leave=False
    ):
        data_t = to_model_input(model, data, device, rgb_from_bgr=True)
        visual_features = model.encode_image(data_t)

        vit_maps = []
        for mode, gallery, gallery_rows, gallery_cols in paired_galleries:
            if mode == "identity":
                patches = prepare_patch_features(visual_features)
                map_t = spectrogram_nn_scores(
                    patches,
                    gallery,
                    gallery_rows,
                    args,
                    model.grid_size[0],
                    model.grid_size[1],
                    gallery_cols=gallery_cols,
                )
            else:
                map_t = compute_patch_map(
                    model,
                    data,
                    gallery,
                    gallery_rows,
                    gallery_cols,
                    args,
                    device,
                    paired_tta_mode=mode,
                )
            vit_maps.append(map_t)
        stacked = torch.stack(vit_maps, dim=0)
        if args.paired_tta_fusion == "max":
            vit_map = stacked.max(dim=0).values
        else:
            vit_map = stacked.mean(dim=0)
        vit_scores.extend(patch_map_image_score(vit_map).tolist())

        cnn_data = cnn_tta_batch(data, "identity", args)
        raw = cnn_data.permute(0, 3, 1, 2).to(device, non_blocking=True)
        cnn_features = encoder(raw)[args.resnet_layers[0]]
        cnn_bank = resnet_patch_score_bank(
            cnn_features,
            cnn_gallery,
            args.distance_chunk_size,
            [args.cnn_image_top_ratio],
            nn_topk=args.cnn_nn_topk,
        )
        cnn_scores.extend(
            cnn_bank[f"top{args.cnn_image_top_ratio:g}"].astype(np.float32).tolist()
        )

        batch_names = [str(value) for value in name]
        labels.extend(int(value) for value in label.numpy().tolist())
        names.extend(batch_names)
        jammer_types.extend(str(value) for value in jammer_type)

    return {
        "names": np.asarray(names),
        "labels": np.asarray(labels, dtype=np.int32),
        "jammer_types": np.asarray(jammer_types),
        VIT_BRANCH: np.asarray(vit_scores, dtype=np.float32),
        CNN_BRANCH: np.asarray(cnn_scores, dtype=np.float32),
    }


def reduce_observations(su_scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, name in enumerate(su_scores["names"]):
        scene_id, observation_id, _ = parse_target_scene_name(str(name))
        groups[(scene_id, observation_id)].append(index)

    scene_ids = []
    observation_ids = []
    labels = []
    jammer_types = []
    vit_scores = []
    cnn_scores = []
    for (scene_id, observation_id), indices_list in sorted(groups.items()):
        indices = np.asarray(indices_list, dtype=np.int64)
        if len(indices) != NUM_SUS:
            raise ValueError(
                f"{scene_id}/{observation_id} has {len(indices)} SUs, expected {NUM_SUS}"
            )
        selected_labels = np.unique(su_scores["labels"][indices])
        selected_types = np.unique(su_scores["jammer_types"][indices])
        if len(selected_labels) != 1 or len(selected_types) != 1:
            raise ValueError(f"Inconsistent labels for {scene_id}/{observation_id}")
        scene_ids.append(scene_id)
        observation_ids.append(observation_id)
        labels.append(int(selected_labels[0]))
        jammer_types.append(str(selected_types[0]))
        # A directional or weak jammer may be visible at only one sensing unit.
        # Preserve that evidence instead of diluting it across all 21 SUs.
        vit_scores.append(float(su_scores[VIT_BRANCH][indices].max()))
        cnn_scores.append(float(su_scores[CNN_BRANCH][indices].max()))

    vit = np.asarray(vit_scores, dtype=np.float32)
    cnn = np.asarray(cnn_scores, dtype=np.float32)
    return {
        "target_scene_ids": np.asarray(scene_ids),
        "observation_ids": np.asarray(observation_ids),
        "labels": np.asarray(labels, dtype=np.int32),
        "jammer_types": np.asarray(jammer_types),
        VIT_BRANCH: vit,
        CNN_BRANCH: cnn,
    }


def aggregate_observations(
    su_scores: dict[str, np.ndarray],
    *,
    gate_protocol: str,
    support_reference: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    reduced = reduce_observations(su_scores)
    if gate_protocol == "support_only":
        if support_reference is None:
            raise ValueError("support_only aggregation requires support references")
        dual_visual_gate = safe_support_only_gate(
            reduced[VIT_BRANCH],
            reduced[CNN_BRANCH],
            support_reference[VIT_BRANCH],
            support_reference[CNN_BRANCH],
        )
    else:
        dual_visual_gate = confidence_gated_or(
            reduced[VIT_BRANCH], reduced[CNN_BRANCH]
        )
    return {
        **reduced,
        METHOD: dual_visual_gate["score"].astype(np.float32),
        "cnn_gate": dual_visual_gate["gate"].astype(np.float32),
        "vit_normalized": dual_visual_gate["vit_normalized"].astype(np.float32),
        "cnn_normalized": dual_visual_gate["cnn_normalized"].astype(np.float32),
        "cnn_rank": dual_visual_gate["cnn_rank"].astype(np.float32),
    }


@torch.no_grad()
def evaluate_support_reference_scores(
    model,
    encoder,
    paired_galleries,
    cnn_gallery,
    loader,
    args,
    device,
) -> dict[str, np.ndarray]:
    """Score held-out views of normal support and reduce them by observation."""

    model.eval_mode()
    encoder.eval()
    identity_gallery = next(
        (item for item in paired_galleries if item[0] == "identity"),
        paired_galleries[0],
    )
    _, gallery, gallery_rows, gallery_cols = identity_gallery
    reference_parts = []
    for mode in SUPPORT_REFERENCE_VIEWS:
        names: list[str] = []
        labels: list[int] = []
        jammer_types: list[str] = []
        vit_scores: list[float] = []
        cnn_scores: list[float] = []
        for data, _, label, name, jammer_type in tqdm(
            loader,
            desc=f"Score OFDMA support reference/{mode}",
            leave=False,
        ):
            viewed = support_reference_view_batch(data, mode, args)
            data_t = to_model_input(model, viewed, device, rgb_from_bgr=True)
            visual_features = model.encode_image(data_t)
            patches = prepare_patch_features(visual_features)
            vit_map = spectrogram_nn_scores(
                patches,
                gallery,
                gallery_rows,
                args,
                model.grid_size[0],
                model.grid_size[1],
                gallery_cols=gallery_cols,
            )
            vit_scores.extend(patch_map_image_score(vit_map).tolist())

            raw = viewed.permute(0, 3, 1, 2).to(device, non_blocking=True)
            cnn_features = encoder(raw)[args.resnet_layers[0]]
            cnn_bank = resnet_patch_score_bank(
                cnn_features,
                cnn_gallery,
                args.distance_chunk_size,
                [args.cnn_image_top_ratio],
                nn_topk=args.cnn_nn_topk,
            )
            cnn_scores.extend(
                cnn_bank[f"top{args.cnn_image_top_ratio:g}"].astype(np.float32).tolist()
            )
            names.extend(str(value) for value in name)
            labels.extend(int(value) for value in label.numpy().tolist())
            jammer_types.extend(str(value) for value in jammer_type)
        reduced = reduce_observations(
            {
                "names": np.asarray(names),
                "labels": np.asarray(labels, dtype=np.int32),
                "jammer_types": np.asarray(jammer_types),
                VIT_BRANCH: np.asarray(vit_scores, dtype=np.float32),
                CNN_BRANCH: np.asarray(cnn_scores, dtype=np.float32),
            }
        )
        reduced["reference_view"] = np.asarray(mode)
        reference_parts.append(reduced)

    return {
        VIT_BRANCH: np.concatenate([part[VIT_BRANCH] for part in reference_parts]),
        CNN_BRANCH: np.concatenate([part[CNN_BRANCH] for part in reference_parts]),
        "reference_views": np.asarray(
            [part["reference_view"] for part in reference_parts]
        ),
        "support_observations": np.asarray(
            len(reference_parts[0][VIT_BRANCH]), dtype=np.int32
        ),
    }


def load_support_reference(root: Path, scene_id: str, shot: int) -> dict[str, np.ndarray]:
    path = root / f"{scene_id}_{shot}shot.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        for key in (VIT_BRANCH, CNN_BRANCH):
            if key not in data.files:
                raise KeyError(f"{path} does not contain {key!r}")
        return {
            VIT_BRANCH: np.asarray(data[VIT_BRANCH], dtype=np.float64),
            CNN_BRANCH: np.asarray(data[CNN_BRANCH], dtype=np.float64),
        }


def safe_metrics(labels, scores) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "fpr95": float("nan")}
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels, scores) * 100.0),
        "auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if len(reached) else float("nan"),
    }


def scene_metric_rows(scene_id, shot, observation_scores, methods) -> list[dict]:
    labels = observation_scores["labels"]
    jammer_types = observation_scores["jammer_types"]
    rows = []
    for method in methods:
        for scope in ["overall", *JAMMER_TYPES]:
            mask = (
                np.ones(len(labels), dtype=bool)
                if scope == "overall"
                else (jammer_types == NO_JAMMER) | (jammer_types == scope)
            )
            rows.append(
                {
                    "row_type": "target_scene",
                    "target_scene_id": scene_id,
                    "shot": int(shot),
                    "scope": scope,
                    "method": method,
                    "num_observations": int(mask.sum()),
                    **safe_metrics(labels[mask], observation_scores[method][mask]),
                }
            )
    return rows


def append_macro_rows(rows: list[dict]) -> list[dict]:
    macro_rows = []
    keys = sorted({(row["shot"], row["scope"], row["method"]) for row in rows})
    for shot, scope, method in keys:
        selected = [
            row
            for row in rows
            if row["shot"] == shot and row["scope"] == scope and row["method"] == method
        ]
        macro_rows.append(
            {
                "row_type": "scene_macro",
                "target_scene_id": "ALL",
                "shot": shot,
                "scope": scope,
                "method": method,
                "num_observations": int(
                    sum(row["num_observations"] for row in selected)
                ),
                **{
                    metric: float(np.nanmean([row[metric] for row in selected]))
                    for metric in ("auroc", "auprc", "fpr95")
                },
            }
        )
    return rows + macro_rows


def dataset_protocol(dataset_root: Path) -> dict:
    path = dataset_root / "protocol.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing target-scene protocol: {path}")
    return json.loads(path.read_text())


def build_run_protocol(args, source_protocol, scene_ids, manifest) -> dict:
    return {
        "entry": "tools/eval_cls_ofdma_target_scene_ours.py",
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "dataset_protocol_name": source_protocol["source_protocol"]["protocol_name"],
        "split": args.split,
        "target_scene_ids": scene_ids,
        "shots": sorted(set(args.shots)),
        "support_rule": (
            "seeded complete-observation selection from normal_support"
            if args.support_seed is not None
            else "first k ranked normal_support observations from the same target scene"
        ),
        "support_seed": args.support_seed,
        "test_rule": "normal_test plus anomaly_test observations from the same target scene",
        "sensing_units_per_observation": NUM_SUS,
        "su_reduction": "maximum branch score over 21 SUs",
        "gate_cell": (
            "none; calibration uses normal support references"
            if args.gate_protocol == "support_only"
            else "all selected test observations within one target scene and shot"
        ),
        "test_label_usage": "metrics_only",
        "gate_protocol": args.gate_protocol,
        "uses_test_batch_statistics": args.gate_protocol == "transductive",
        "support_reference_only": bool(args.support_reference_only),
        "support_reference_root": args.support_reference_root or None,
        "support_reference_views": list(SUPPORT_REFERENCE_VIEWS),
        "method": METHOD,
        "branches": [VIT_BRANCH, CNN_BRANCH],
        "method_config": FORMAL,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "smoke_limits": {
            "max_normal_observations": args.max_normal_observations,
            "max_anomaly_observations_per_type": (
                args.max_anomaly_observations_per_type
            ),
        },
        "manifest_rows": int(len(manifest)),
    }


def model_kwargs(args, device, max_shot: int) -> dict:
    values = vars(args).copy()
    values.update(
        {
            "dataset": "ofdma_spectrum",
            "class_name": "radio frequency spectrogram",
            "device": device,
            "out_size_h": args.resolution,
            "out_size_w": args.resolution,
            "k_shot": max_shot * NUM_SUS,
        }
    )
    return values


def main() -> None:
    cli = parse_args()
    args = formal_args(cli)
    if not args.shots or not set(args.shots).issubset({1, 2, 4}):
        raise ValueError("--shots must be selected from the formal protocol: 1, 2, 4")
    if args.max_normal_observations < 0 or args.max_anomaly_observations_per_type < 0:
        raise ValueError("Smoke-test observation limits cannot be negative")

    # Select the physical GPU before setup_seed() or any CUDA availability
    # check initializes the CUDA runtime.
    if not args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    setup_seed(args.seed)
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_target_scene_manifest(dataset_root)
    source_protocol = dataset_protocol(dataset_root)
    available = available_target_scenes(manifest, args.split)
    scene_ids = list(args.scene_ids) if args.scene_ids else available
    unknown = sorted(set(scene_ids) - set(available))
    if unknown:
        raise ValueError(f"Unknown {args.split} target scenes: {unknown}")
    if not scene_ids:
        raise ValueError(f"No target scenes selected for split={args.split}")

    normalization = source_protocol["source_protocol"]["normalization"]
    preprocessor = OFDMASpectrogramPreprocessor(
        dataset_root,
        output_size=args.img_cropsize,
        subcarriers_per_rb=args.subcarriers_per_rb,
        geometry=args.geometry,
        min_db=float(normalization["min_db"]),
        max_db=float(normalization["max_db"]),
    )

    counts = []
    for scene_id in scene_ids:
        test_records = build_target_scene_records(
            dataset_root,
            manifest,
            scene_id,
            split=args.split,
            role="test",
            max_normal_observations=args.max_normal_observations,
            max_anomaly_observations_per_type=args.max_anomaly_observations_per_type,
        )
        observation_count = len({record.observation_id for record in test_records})
        counts.append(
            {
                "target_scene_id": scene_id,
                "test_observations": observation_count,
                "test_images": len(test_records),
            }
        )
        for shot in sorted(set(args.shots)):
            build_target_scene_records(
                dataset_root,
                manifest,
                scene_id,
                split=args.split,
                role="support",
                shot=shot,
                support_seed=args.support_seed,
            )

    sample_record = build_target_scene_records(
        dataset_root,
        manifest,
        scene_ids[0],
        split=args.split,
        role="support",
        shot=min(args.shots),
        support_seed=args.support_seed,
    )[0]
    sample_raw = cv2.imread(str(sample_record.image_path), cv2.IMREAD_GRAYSCALE)
    sample = preprocessor(sample_raw)
    if sample.shape != (args.img_cropsize, args.img_cropsize):
        raise RuntimeError(f"Unexpected preprocessed shape: {sample.shape}")

    protocol = build_run_protocol(args, source_protocol, scene_ids, manifest)
    protocol["selected_counts"] = counts
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2))
    if args.validate_only:
        print("[validated] target-scene protocol and paths are consistent")
        return

    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Missing PromptAD checkpoint: {args.checkpoint}")
    if not args.use_cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --use-cpu only for a tiny debug run")
    device = "cpu" if args.use_cpu else "cuda:0"
    model = PromptAD(**model_kwargs(args, device, max(args.shots))).to(device)
    load_checkpoint(model, args.checkpoint)
    encoder = ResNet18LocalEncoder().to(device)

    if args.support_reference_only:
        reference_root = output_root / "support_reference"
        reference_root.mkdir(parents=True, exist_ok=True)
        for shot in sorted(set(args.shots)):
            for scene_index, scene_id in enumerate(scene_ids, start=1):
                print(
                    f"[support reference {scene_index}/{len(scene_ids)}] "
                    f"{scene_id}, {shot}-shot"
                )
                support_records = build_target_scene_records(
                    dataset_root,
                    manifest,
                    scene_id,
                    split=args.split,
                    role="support",
                    shot=shot,
                    support_seed=args.support_seed,
                )
                support_loader = make_loader(
                    support_records, preprocessor, args, support=True
                )
                paired_galleries, cnn_gallery = build_memories(
                    model, encoder, support_loader, args, device
                )
                references = evaluate_support_reference_scores(
                    model,
                    encoder,
                    paired_galleries,
                    cnn_gallery,
                    support_loader,
                    args,
                    device,
                )
                np.savez_compressed(
                    reference_root / f"{scene_id}_{shot}shot.npz",
                    **references,
                )
                del paired_galleries, cnn_gallery, references
                if not args.use_cpu:
                    torch.cuda.empty_cache()
        (output_root / "support_reference_protocol.json").write_text(
            json.dumps(
                {
                    **protocol,
                    "support_only": True,
                    "uses_test_batch_statistics": False,
                    "test_samples_scored": 0,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"[done] support references={reference_root}")
        return

    score_root = output_root / "scores"
    score_root.mkdir(parents=True, exist_ok=True)
    primary_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    for shot in sorted(set(args.shots)):
        for scene_index, scene_id in enumerate(scene_ids, start=1):
            print(
                f"[scene {scene_index}/{len(scene_ids)}] {scene_id}, {shot}-shot"
            )
            support_records = build_target_scene_records(
                dataset_root,
                manifest,
                scene_id,
                split=args.split,
                role="support",
                shot=shot,
                support_seed=args.support_seed,
            )
            test_records = build_target_scene_records(
                dataset_root,
                manifest,
                scene_id,
                split=args.split,
                role="test",
                max_normal_observations=args.max_normal_observations,
                max_anomaly_observations_per_type=args.max_anomaly_observations_per_type,
            )
            support_loader = make_loader(
                support_records, preprocessor, args, support=True
            )
            test_loader = make_loader(test_records, preprocessor, args, support=False)
            paired_galleries, cnn_gallery = build_memories(
                model, encoder, support_loader, args, device
            )
            if args.gate_protocol == "support_only":
                if args.support_reference_root:
                    support_reference = load_support_reference(
                        Path(args.support_reference_root), scene_id, shot
                    )
                else:
                    support_reference = evaluate_support_reference_scores(
                        model,
                        encoder,
                        paired_galleries,
                        cnn_gallery,
                        support_loader,
                        args,
                        device,
                    )
            else:
                support_reference = None
            su_scores = evaluate_su_scores(
                model,
                encoder,
                paired_galleries,
                cnn_gallery,
                test_loader,
                args,
                device,
            )
            observation_scores = aggregate_observations(
                su_scores,
                gate_protocol=args.gate_protocol,
                support_reference=support_reference,
            )
            np.savez_compressed(
                score_root / f"{scene_id}_{shot}shot_observation_scores.npz",
                **observation_scores,
            )
            np.savez_compressed(
                score_root / f"{scene_id}_{shot}shot_su_branch_scores.npz",
                **su_scores,
            )
            primary_rows.extend(
                scene_metric_rows(
                    scene_id, shot, observation_scores, methods=[METHOD]
                )
            )
            diagnostic_rows.extend(
                scene_metric_rows(
                    scene_id,
                    shot,
                    observation_scores,
                    methods=[VIT_BRANCH, CNN_BRANCH],
                )
            )
            write_csv(output_root / "results.csv", append_macro_rows(primary_rows))
            write_csv(
                output_root / "branch_diagnostics.csv",
                append_macro_rows(diagnostic_rows),
            )
            del (
                paired_galleries,
                cnn_gallery,
                su_scores,
                observation_scores,
                support_reference,
            )
            if not args.use_cpu:
                torch.cuda.empty_cache()

    final_primary = append_macro_rows(primary_rows)
    primary_macro = [
        row
        for row in final_primary
        if row["row_type"] == "scene_macro" and row["scope"] == "overall"
    ]
    summary = {
        "primary_method": METHOD,
        "primary_level": (
            "observation (third-highest ViT patch per SU; maximum over 21 sensing units; "
            "ViT+CNN safe support-only confidence gate)"
            if args.gate_protocol == "support_only"
            else "observation (third-highest ViT patch per SU; maximum over 21 sensing units; ViT+CNN transductive gate)"
        ),
        "gate_protocol": args.gate_protocol,
        "uses_test_labels_for_scoring": False,
        "uses_test_batch_statistics": args.gate_protocol == "transductive",
        "support_reference_root": args.support_reference_root or "computed_in_run",
        "safe_gate": (
            {
                "rank_quantile": SAFE_SUPPORT_RANK_QUANTILE,
                "temperature": SAFE_SUPPORT_TEMPERATURE,
                "alpha": SAFE_SUPPORT_ALPHA,
            }
            if args.gate_protocol == "support_only"
            else None
        ),
        "is_smoke_test": bool(
            args.max_normal_observations > 0
            or args.max_anomaly_observations_per_type > 0
        ),
        "overall_scene_macro": primary_macro,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[done] results={output_root / 'results.csv'}")


if __name__ == "__main__":
    main()
