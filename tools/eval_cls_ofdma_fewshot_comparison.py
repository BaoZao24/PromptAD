#!/usr/bin/env python
"""Legacy transductive comparison on the older OFDMA jammer dataset.

This entry is retained only for historical protocol comparisons.  The paper's
formal OFDMA result uses ``eval_cls_ofdma_target_scene_ours.py`` with the safe
support-only gate.  This legacy entry uses the old test-batch gate and must not
be used as the formal cold-start result.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

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
    DEFAULT_OFDMA_ROOT,
    JAMMER_TYPES,
    NO_JAMMER,
    NUM_SUS,
    OFDMAPathDataset,
    OFDMASpectrogramPreprocessor,
    build_frame_records,
    build_official_scene_split,
    load_ofdma_labels,
    parse_ofdma_name,
    select_support_scene_ids,
)
from tools.eval_cls_resnet_gallery_fusion import (
    resnet_patch_image_scores,
    resnet_patch_score_bank,
)
from tools.eval_cls_vit_patch_gallery import harmonic
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
from train_rf_target_pooled_universal import build_gallery, to_model_input
from utils.training_utils import setup_seed
from utils.confidence_gate import confidence_gated_or


PRIMARY_METHOD = "confidence_gated_dual_visual"
PROMPTAD_BASELINE = "promptad_text_vit_harmonic"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_metrics(labels, scores) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "fpr95": float("nan")}
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[reached[0]] * 100.0) if len(reached) else float("nan")
    return {
        "auroc": float(roc_auc_score(labels, scores) * 100.0),
        "auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": fpr95,
    }


def make_loader(records, preprocessor, args, support=False):
    dataset = OFDMAPathDataset(records, preprocessor)
    return DataLoader(
        dataset,
        batch_size=min(args.batch_size, max(1, len(dataset))),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not args.use_cpu,
        persistent_workers=args.num_workers > 0 and not support,
    )


def capped(values, limit: int):
    return tuple(values if limit <= 0 else values[:limit])


def build_test_records(dataset_root, labels, split, args):
    su_ids = tuple(range(min(NUM_SUS, args.max_sus_per_scene)))
    normal_ids = capped(split.test_normal, args.max_test_normal_scenes)
    records = build_frame_records(dataset_root, labels, normal_ids, su_ids)
    for jammer_type in JAMMER_TYPES:
        scene_ids = capped(
            split.test_abnormal_by_type[jammer_type],
            args.max_test_scenes_per_jammer,
        )
        records.extend(build_frame_records(dataset_root, labels, scene_ids, su_ids))
    return records


def validate_records(records) -> None:
    missing = [str(record.image_path) for record in records if not record.image_path.is_file()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} OFDMA images; first paths:\n{preview}")


def model_kwargs(args, device, shot: int):
    kwargs = vars(args).copy()
    kwargs.update(
        {
            "dataset": "ofdma_spectrum",
            "class_name": "radio frequency spectrogram",
            "device": device,
            "out_size_h": args.resolution,
            "out_size_w": args.resolution,
            "k_shot": shot * NUM_SUS,
        }
    )
    return kwargs


def patch_map_image_score(patch_map: torch.Tensor, top_ratio: float) -> np.ndarray:
    """Summarize a patch anomaly map without changing the default max score."""
    flat = patch_map.flatten(1)
    if float(top_ratio) <= 0.0:
        values = flat.max(dim=1).values
    else:
        keep = max(1, int(round(flat.shape[1] * float(top_ratio))))
        values = flat.topk(keep, dim=1).values.mean(dim=1)
    return values.cpu().numpy().astype(np.float32)


@torch.no_grad()
def build_memories(model, encoder, support_loader, args, device):
    # PromptAD baseline: the original two-layer full ViT patch memory.
    build_gallery(model, support_loader, device)
    model.build_text_feature_gallery()

    # Proposed ViT branch: one coreset per paired view.
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

    # Proposed CNN branch: selected frozen ResNet18 layer, local memory, 1-NN.
    cnn_galleries = build_resnet_gallery(
        encoder,
        support_loader,
        args,
        device,
        tta_mode="identity",
    )
    return paired_galleries, cnn_galleries[args.resnet_layers[0]]


@torch.no_grad()
def evaluate_cnn_images(encoder, cnn_gallery, test_loader, args, device):
    """Evaluate only the CNN branch for inexpensive layer/shot ablations."""
    encoder.eval()
    names: list[str] = []
    jammer_types: list[str] = []
    labels: list[int] = []
    cnn_scores: list[float] = []

    for data, _, label, name, jammer_type in tqdm(
        test_loader, desc="Evaluate OFDMA CNN", leave=False
    ):
        cnn_data = cnn_tta_batch(data, "identity", args)
        raw = cnn_data.permute(0, 3, 1, 2).to(device, non_blocking=True)
        cnn_features = encoder(raw)[args.resnet_layers[0]]
        cnn_score = resnet_patch_image_scores(
            cnn_features,
            cnn_gallery,
            args.distance_chunk_size,
            args.cnn_image_top_ratio,
            nn_topk=args.cnn_nn_topk,
        ).astype(np.float32)
        cnn_scores.extend(cnn_score.tolist())
        labels.extend(int(x) for x in label.numpy().tolist())
        names.extend(str(x) for x in name)
        jammer_types.extend(str(x) for x in jammer_type)

    return (
        np.asarray(names),
        np.asarray(labels, dtype=np.int32),
        np.asarray(jammer_types),
        {"ours_cnn": np.asarray(cnn_scores, dtype=np.float32)},
    )


@torch.no_grad()
def evaluate_images(model, encoder, paired_galleries, cnn_gallery, test_loader, args, device):
    model.eval_mode()
    encoder.eval()
    names: list[str] = []
    jammer_types: list[str] = []
    labels: list[int] = []
    score_lists = defaultdict(list)

    for data, _, label, name, jammer_type in tqdm(test_loader, desc="Evaluate OFDMA", leave=False):
        data_t = to_model_input(model, data, device, rgb_from_bgr=True)
        visual_features = model.encode_image(data_t)

        text_score = np.asarray(
            model.calculate_textual_anomaly_score(visual_features, "cls"),
            dtype=np.float32,
        )
        promptad_map = model.calculate_visual_anomaly_score(visual_features)
        promptad_vit = promptad_map.flatten(1).max(dim=1).values.numpy().astype(np.float32)

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
        vit_score = patch_map_image_score(vit_map, 0.0)

        cnn_data = cnn_tta_batch(data, "identity", args)
        raw = cnn_data.permute(0, 3, 1, 2).to(device, non_blocking=True)
        cnn_features = encoder(raw)[args.resnet_layers[0]]
        cnn_score_bank = resnet_patch_score_bank(
            cnn_features,
            cnn_gallery,
            args.distance_chunk_size,
            [args.cnn_image_top_ratio],
            nn_topk=args.cnn_nn_topk,
        )
        cnn_score = cnn_score_bank[
            f"top{args.cnn_image_top_ratio:g}"
        ].astype(np.float32)
        score_lists["promptad_text"].extend(text_score.tolist())
        score_lists["promptad_vit_memory"].extend(promptad_vit.tolist())
        score_lists[PROMPTAD_BASELINE].extend(harmonic(text_score, promptad_vit).tolist())
        score_lists["ours_vit"].extend(vit_score.tolist())
        for top_ratio in args.ablation_vit_top_ratios:
            key = f"ours_vit_top{float(top_ratio):g}"
            score_lists[key].extend(patch_map_image_score(vit_map, top_ratio).tolist())
        score_lists["ours_cnn"].extend(cnn_score.tolist())
        labels.extend(int(x) for x in label.numpy().tolist())
        names.extend(str(x) for x in name)
        jammer_types.extend(str(x) for x in jammer_type)

    scores = {key: np.asarray(value, dtype=np.float32) for key, value in score_lists.items()}
    return (
        np.asarray(names),
        np.asarray(labels, dtype=np.int32),
        np.asarray(jammer_types),
        scores,
    )


def aggregate_scenes(names, labels, jammer_types, scores, reduction: str):
    groups = defaultdict(list)
    for index, name in enumerate(names):
        scene_id, _ = parse_ofdma_name(str(name))
        groups[scene_id].append(index)

    scene_labels = []
    scene_types = []
    scene_scores = {key: [] for key in scores}
    for scene_id in sorted(groups):
        indices = np.asarray(groups[scene_id], dtype=np.int64)
        scene_labels.append(int(labels[indices[0]]))
        scene_types.append(str(jammer_types[indices[0]]))
        for key, values in scores.items():
            selected = values[indices]
            scene_scores[key].append(float(selected.mean() if reduction == "mean" else selected.max()))
    return (
        np.asarray(scene_labels, dtype=np.int32),
        np.asarray(scene_types),
        {key: np.asarray(value, dtype=np.float32) for key, value in scene_scores.items()},
    )


def metric_rows(shot, level, labels, jammer_types, scores):
    rows = []
    metric_methods = list(scores)
    for method in metric_methods:
        type_metrics = []
        for scope in ["overall", *JAMMER_TYPES]:
            if scope == "overall":
                mask = np.ones(len(labels), dtype=bool)
            else:
                mask = (jammer_types == NO_JAMMER) | (jammer_types == scope)
            metrics = safe_metrics(labels[mask], scores[method][mask])
            row = {
                "shot": shot,
                "level": level,
                "scope": scope,
                "method": method,
                "num_samples": int(mask.sum()),
                **metrics,
            }
            rows.append(row)
            if scope != "overall":
                type_metrics.append(metrics)
        rows.append(
            {
                "shot": shot,
                "level": level,
                "scope": "macro_jammer",
                "method": method,
                "num_samples": int(len(labels)),
                **{
                    key: float(np.nanmean([item[key] for item in type_metrics]))
                    for key in ("auroc", "auprc", "fpr95")
                },
            }
        )
    return rows


def protocol_summary(args, labels, split, preprocessor, test_records):
    return {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "axis_semantics": {"horizontal": "time", "vertical": "frequency"},
        "raw_shape": [1320, 70],
        "aggregated_shape": [110, 70],
        "model_shape": [args.img_cropsize, args.img_cropsize],
        "geometry": args.geometry,
        "subcarriers_per_resource_block": args.subcarriers_per_rb,
        "official_split": {
            "train_normal_scenes": len(split.train_normal),
            "valid_normal_scenes": len(split.valid_normal),
            "test_normal_scenes": len(split.test_normal),
            "test_anomaly_scenes_by_type": {
                key: len(value) for key, value in split.test_abnormal_by_type.items()
            },
        },
        "actual_test_images": len(test_records),
        "shots": args.shots,
        "cnn_only": args.cnn_only,
        "resnet_layers": args.resnet_layers,
        "fusion_method": (
            None if args.cnn_only else PRIMARY_METHOD
        ),
        "support_seed": args.seed,
        "label_usage": "metrics_only",
        "checkpoint": args.checkpoint,
        "label_rows": len(labels),
        "normal_label_rows": int((labels["jammer_type"] == NO_JAMMER).sum()),
        "preprocessing_range_db": [preprocessor.min_db, preprocessor.max_db],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(DEFAULT_OFDMA_ROOT))
    parser.add_argument("--output-root", default="analysis_outputs/20260715_ofdma_fewshot_comparison")
    parser.add_argument(
        "--checkpoint",
        default="analysis_outputs/02_current_baselines/promptad_formal_baseline/pooled_rf_rgb_cls/checkpoint/overall-best.pt",
    )
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--geometry", choices=["letterbox", "square_warp"], default="letterbox")
    parser.add_argument("--subcarriers-per-rb", type=int, default=12)
    parser.add_argument("--max-test-normal-scenes", type=int, default=0)
    parser.add_argument("--max-test-scenes-per-jammer", type=int, default=0)
    parser.add_argument("--max-sus-per-scene", type=int, default=NUM_SUS)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--cnn-only",
        action="store_true",
        help="Skip PromptAD/ViT and report only CNN scores for fast layer/shot ablations.",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--resolution", type=int, default=240)
    parser.add_argument("--img-resize", type=int, default=240)
    parser.add_argument("--img-cropsize", type=int, default=240)
    parser.add_argument("--backbone", default="ViT-B-16-plus-240")
    parser.add_argument("--pretrained_dataset", default="laion400m_e32")
    parser.add_argument("--prompt-mode", default="legacy")
    parser.add_argument("--input-mode", default="rgb")
    parser.add_argument("--text-prototype-mode", default="single")
    parser.add_argument("--cls-score-mode", default="text_only")
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=3)
    parser.add_argument("--n_pro_ab", type=int, default=4)

    parser.add_argument("--coreset-ratio", type=float, default=0.5)
    parser.add_argument("--coreset-method", choices=["random", "farthest"], default="farthest")
    parser.add_argument("--rowwise-coreset", action="store_true")
    parser.add_argument("--memory-mode", choices=["global_nn", "row_nn", "row_proto"], default="global_nn")
    parser.add_argument("--support-augment", default="none")
    parser.add_argument("--paired-tta", default="ofdma_time_shift_blur")
    parser.add_argument("--paired-tta-fusion", choices=["mean", "max"], default="max")
    parser.add_argument("--paired-tta-shift-px", type=int, default=4)
    parser.add_argument("--paired-tta-blur-ksize", type=int, default=3)
    parser.add_argument("--paired-tta-background-noise-strength", type=float, default=3.0)
    parser.add_argument("--paired-tta-frequency-response-strength", type=float, default=3.0)
    parser.add_argument("--gallery-chunk-size", type=int, default=4096)
    parser.add_argument("--freq-window", type=int, default=-1)
    parser.add_argument("--nn-topk", type=int, default=5)
    parser.add_argument("--nn-agg", choices=["mean", "weighted", "adaptive"], default="mean")
    parser.add_argument("--nn-weight-temp", type=float, default=0.05)
    parser.add_argument("--adaptive-sim-margin", type=float, default=0.02)
    parser.add_argument("--position-soft-axis", choices=["frequency", "time"], default="frequency")
    parser.add_argument("--position-soft-weight", type=float, default=0.0)

    parser.add_argument(
        "--resnet-layers",
        nargs="+",
        default=["layer3"],
        choices=["layer1", "layer2", "layer3"],
    )
    parser.add_argument("--max-gallery-patches", type=int, default=50000)
    parser.add_argument("--cnn-coreset-size", type=int, default=0)
    parser.add_argument("--cnn-nn-topk", type=int, default=1)
    parser.add_argument("--cnn-image-top-ratio", type=float, default=0.1)
    parser.add_argument("--distance-chunk-size", type=int, default=1024)
    parser.add_argument("--cnn-paired-tta", default="none")
    parser.add_argument("--cnn-paired-tta-shift-px", type=int, default=4)
    parser.add_argument("--cnn-paired-tta-blur-ksize", type=int, default=3)
    parser.add_argument(
        "--cnn-input-normalization",
        choices=["raw", "robust_global"],
        default="raw",
        help="Optional CNN-only per-image robust intensity normalization.",
    )
    parser.add_argument(
        "--ablation-vit-top-ratios",
        type=float,
        nargs="*",
        default=[],
        help="Optional ViT patch-map top-ratio means to report alongside the unchanged max baseline.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.shots or min(args.shots) < 1:
        raise ValueError("--shots must contain positive integers")
    if not 1 <= args.max_sus_per_scene <= NUM_SUS:
        raise ValueError(f"--max-sus-per-scene must be in [1, {NUM_SUS}]")
    if not 0.0 < args.coreset_ratio <= 1.0:
        raise ValueError("--coreset-ratio must be in (0, 1]")
    if any(not 0.0 < ratio <= 1.0 for ratio in args.ablation_vit_top_ratios):
        raise ValueError("--ablation-vit-top-ratios values must be in (0, 1]")
    if args.paired_tta not in {
        "ofdma_time_shift_blur",
        "ofdma_spectral_structure_v1",
        "ofdma_spectral_background_v1",
        "ofdma_spectral_time_background_v1",
        "ofdma_time_alignment_v1",
        "ofdma_frequency_response_only_v1",
        "ofdma_spectral_response_v1",
        "ofdma_spectral_physics_v1",
    }:
        raise ValueError(
            "OFDMA evaluation requires --paired-tta ofdma_time_shift_blur, "
            "ofdma_spectral_structure_v1, ofdma_spectral_background_v1, "
            "ofdma_spectral_time_background_v1, ofdma_time_alignment_v1, "
            "ofdma_frequency_response_only_v1, ofdma_spectral_response_v1, "
            "or ofdma_spectral_physics_v1"
        )

    setup_seed(args.seed)
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    labels = load_ofdma_labels(dataset_root)
    split = build_official_scene_split(labels)
    preprocessor = OFDMASpectrogramPreprocessor(
        dataset_root,
        output_size=args.img_cropsize,
        subcarriers_per_rb=args.subcarriers_per_rb,
        geometry=args.geometry,
    )
    test_records = build_test_records(dataset_root, labels, split, args)
    validate_records(test_records)
    sample = preprocessor(cv2.imread(str(test_records[0].image_path), cv2.IMREAD_GRAYSCALE))
    if sample.shape != (args.img_cropsize, args.img_cropsize):
        raise RuntimeError(f"Unexpected preprocessed shape: {sample.shape}")

    protocol = protocol_summary(args, labels, split, preprocessor, test_records)
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2))
    if args.validate_only:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cpu" if args.use_cpu else "cuda:0"
    encoder = ResNet18LocalEncoder().to(device)
    model = None
    if not args.cnn_only:
        if not Path(args.checkpoint).is_file():
            raise FileNotFoundError(f"Missing PromptAD checkpoint: {args.checkpoint}")
        max_shot = max(args.shots)
        model = PromptAD(**model_kwargs(args, device, max_shot)).to(device)
        load_checkpoint(model, args.checkpoint)
    test_loader = make_loader(test_records, preprocessor, args)

    all_rows = []
    scores_root = output_root / "scores"
    scores_root.mkdir(parents=True, exist_ok=True)
    for shot in sorted(set(args.shots)):
        support_scene_ids = select_support_scene_ids(split.train_normal, shot, args.seed)
        support_records = build_frame_records(dataset_root, labels, support_scene_ids)
        validate_records(support_records)
        support_loader = make_loader(support_records, preprocessor, args, support=True)
        (output_root / f"support_{shot}shot.json").write_text(
            json.dumps(
                {
                    "shot": shot,
                    "scene_ids": support_scene_ids,
                    "num_images": len(support_records),
                    "images_per_scene": NUM_SUS,
                },
                indent=2,
            )
            + "\n"
        )
        if args.cnn_only:
            cnn_galleries = build_resnet_gallery(
                encoder,
                support_loader,
                args,
                device,
                tta_mode="identity",
            )
            cnn_gallery = cnn_galleries[args.resnet_layers[0]]
            names, y_true, jammer_types, scores = evaluate_cnn_images(
                encoder,
                cnn_gallery,
                test_loader,
                args,
                device,
            )
            paired_galleries = []
        else:
            paired_galleries, cnn_gallery = build_memories(
                model,
                encoder,
                support_loader,
                args,
                device,
            )
            names, y_true, jammer_types, scores = evaluate_images(
                model,
                encoder,
                paired_galleries,
                cnn_gallery,
                test_loader,
                args,
                device,
            )
            gate_result = confidence_gated_or(
                scores["ours_vit"],
                scores["ours_cnn"],
            )
            scores[PRIMARY_METHOD] = gate_result["score"].astype(np.float32)
        np.savez_compressed(
            scores_root / f"ofdma_{shot}shot_image_scores.npz",
            names=names,
            labels=y_true,
            jammer_types=jammer_types,
            **(
                {}
                if args.cnn_only
                else {
                    "cnn_gate": gate_result["gate"].astype(np.float32),
                }
            ),
            **scores,
        )
        all_rows.extend(metric_rows(shot, "image", y_true, jammer_types, scores))
        for reduction in ("mean", "max"):
            scene_labels, scene_types, scene_scores = aggregate_scenes(
                names,
                y_true,
                jammer_types,
                scores,
                reduction,
            )
            all_rows.extend(
                metric_rows(shot, f"scene_{reduction}", scene_labels, scene_types, scene_scores)
            )
        # Keep completed-shot metrics durable if a later full evaluation is interrupted.
        write_csv(output_root / "results.csv", all_rows)
        del paired_galleries, cnn_gallery
        if not args.use_cpu:
            torch.cuda.empty_cache()

    summary_methods = (
        {"ours_cnn"}
        if args.cnn_only
        else {PROMPTAD_BASELINE, "ours_vit", "ours_cnn", PRIMARY_METHOD}
    )
    primary = [
        row
        for row in all_rows
        if row["level"] == "image"
        and row["scope"] in {"overall", "macro_jammer"}
        and row["method"] in summary_methods
    ]
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "primary_method": "ours_cnn" if args.cnn_only else PRIMARY_METHOD,
                "promptad_baseline": None if args.cnn_only else PROMPTAD_BASELINE,
                "uses_test_batch_statistics": True,
                "uses_test_labels_for_scoring": False,
                "primary_image_results": primary,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[done] results={output_root / 'results.csv'}")


if __name__ == "__main__":
    main()
