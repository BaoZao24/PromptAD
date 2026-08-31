#!/usr/bin/env python3
"""Evaluate the original ViT+CNN mainline with full normal memory and 1-NN.

The file was initially created for an exploratory ViT/DINO replacement
comparison.  ``--control-only`` is now the supported path for the mainline:
it skips DINO model construction, feature extraction, scoring, and output.
The original method is PromptAD ViT layer1+layer2 + ResNet18 layer3, with the
normal-support spectral TTA bundle, full normal patch memory, 1-NN, and the
support-only confidence gate.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_cls_dinov2_patchcore_gallery import (  # noqa: E402
    build_transform as build_dino_transform,
    extract_patch_features as extract_dino_patch_features,
    to_dino_input,
)
from tools.eval_cls_resnet_gallery_fusion import resnet_patch_image_scores  # noqa: E402
from tools.eval_cls_vit_patchcore_gallery import (  # noqa: E402
    prepare_patch_features,
    select_gallery_subset,
    topk_cosine_distance_chunked,
)
from tools.eval_seg_resnet_gallery_fusion import (  # noqa: E402
    ResNet18LocalEncoder,
    WideResNet50LocalEncoder,
)
from tools.eval_fedjam_fewshot_dual import metric  # noqa: E402
from tools.eval_fedjam_four_branches import build_dino, build_promptad  # noqa: E402
from tools.eval_vit_dino_weighted_rf import (  # noqa: E402
    ImageRecord,
    RF_JSR_BY_SIGNAL,
    RF_SCENES,
    RF_TARGET_SIGNALS,
    aggregate_ofdma_observations,
    cell_entry,
    deduplicate_records,
    ensure_rf_target_scene_manifest,
    inhouse_manifest,
    load_bgr,
    manifest_support_records,
    ofdma_preprocessed_records,
    public_jsrs,
    public_manifest,
    records_to_raw,
    scene_support_entry,
)
from datasets.ofdma_spectrum import OFDMASpectrogramPreprocessor  # noqa: E402
from datasets.ofdma_target_scene import (  # noqa: E402
    available_target_scenes,
    build_target_scene_records,
    load_target_scene_manifest,
)
from datasets.rf_target import collect_rf_target_samples  # noqa: E402
from train_rf_target_pooled_universal import to_model_input  # noqa: E402
from utils.confidence_gate import safe_support_only_gate  # noqa: E402
from utils.spectral_tta import SPECTRAL_TTA_BUNDLES, augment_spectrogram  # noqa: E402
from utils.training_utils import setup_seed  # noqa: E402


METHOD_KEYS = (
    "ours_vit_cnn_gate",
    "ours_dinov2_cnn_gate",
    "vit_only",
    "dinov2_only",
    "cnn_only",
)
METHOD_LABELS = {
    "ours_vit_cnn_gate": "Ours (ViT+CNN, current)",
    "ours_dinov2_cnn_gate": "Ours (DINOv2+CNN, ViT replaced)",
    "vit_only": "ViT-only",
    "dinov2_only": "DINOv2-only",
    "cnn_only": "CNN-only",
}


def branch_keys(args) -> tuple[str, ...]:
    """Return the feature branches that are actually evaluated."""

    return ("vit", "cnn") if args.control_only else ("vit", "dino", "cnn")


def method_labels(args) -> dict[str, str]:
    """Return labels for methods present in this run."""

    keys = ("ours_vit_cnn_gate", "vit_only", "cnn_only")
    if not args.control_only:
        keys = keys + ("ours_dinov2_cnn_gate", "dinov2_only")
    return {key: METHOD_LABELS[key] for key in keys}


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tta_modes(args) -> tuple[str, ...]:
    if args.tta_bundle == "none":
        return ("identity",)
    return tuple(SPECTRAL_TTA_BUNDLES[args.tta_bundle])


def _warp_time(image: np.ndarray, dy: int) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, 0], [0, 1, int(dy)]])
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def augment_image(image: np.ndarray, mode: str, args) -> np.ndarray:
    """Apply a protocol view to one BGR spectrogram."""

    if mode == "identity":
        return image
    if mode == "blur":
        return cv2.GaussianBlur(image, (3, 3), 0)
    if mode in {"time_shift_up", "time_shift_down", "time_shift_up_large", "time_shift_down_large"}:
        shift = int(args.tta_shift_px) * (2 if mode.endswith("_large") else 1)
        return _warp_time(image, -shift if "up" in mode else shift)
    if mode in SPECTRAL_TTA_BUNDLES.get(args.tta_bundle, ()):
        return augment_spectrogram(
            image,
            mode,
            shift_px=int(args.tta_shift_px),
            frequency_response_strength=float(args.tta_frequency_response_strength),
        )
    # The support-reference protocol uses generic up/down names even for the
    # OFDMA bundle.  The physical axis is still the same time-axis shift used
    # by the existing formal evaluator.
    if mode in {"rf_time_shift_up", "rf_time_shift_down", "ofdma_time_shift_left", "ofdma_time_shift_right"}:
        shift = int(args.tta_shift_px)
        sign = -1 if mode.endswith(("up", "left")) else 1
        return _warp_time(image, sign * shift)
    raise ValueError(f"Unsupported TTA mode {mode!r} for bundle {args.tta_bundle!r}")


def augment_raw(raw: torch.Tensor, mode: str, args) -> torch.Tensor:
    images = [augment_image(image.numpy(), mode, args) for image in raw]
    return torch.from_numpy(np.stack(images, axis=0))


@torch.inference_mode()
def encode_vit(model, raw: torch.Tensor, device: torch.device) -> torch.Tensor:
    inputs = to_model_input(model, raw, device, rgb_from_bgr=True)
    return prepare_patch_features(model.encode_image(inputs)).float().contiguous()


@torch.inference_mode()
def encode_dino(model, transform, raw: torch.Tensor, device: torch.device) -> torch.Tensor:
    inputs = to_dino_input(raw, transform, device, rgb_from_bgr=True)
    return extract_dino_patch_features(model, inputs).float().contiguous()


@torch.inference_mode()
def encode_cnn(model, raw: torch.Tensor, device: torch.device) -> torch.Tensor:
    inputs = raw.permute(0, 3, 1, 2).to(device, non_blocking=True)
    return model(inputs)["layer3"].float().contiguous()


def normalize_gallery(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=-1).contiguous()


def maybe_coreset(features: torch.Tensor, ratio: float, args, seed_offset: int) -> torch.Tensor:
    features = normalize_gallery(features)
    if ratio >= 1.0:
        return features
    subset_args = SimpleNamespace(
        coreset_ratio=ratio,
        coreset_method="farthest",
        seed=int(args.seed) + int(seed_offset),
    )
    indices, _ = select_gallery_subset(features, subset_args)
    return features[indices].contiguous()


@torch.inference_mode()
def build_state(
    support: list[ImageRecord],
    dataset: str,
    clip_model,
    dino_model,
    dino_transform,
    cnn_model,
    args,
    device: torch.device,
) -> dict:
    """Build both visual memories from exactly the same support views."""

    raw = records_to_raw(support, dataset)
    vit_parts = []
    dino_parts = [] if not args.control_only else None
    modes = tta_modes(args)
    for mode in modes:
        viewed = raw if mode == "identity" else augment_raw(raw, mode, args)
        vit_features = encode_vit(clip_model, viewed, device)
        vit_parts.append(vit_features.reshape(-1, vit_features.shape[-1]))
        if not args.control_only:
            dino_features = encode_dino(dino_model, dino_transform, viewed, device)
            dino_parts.append(dino_features.reshape(-1, dino_features.shape[-1]))
            del dino_features
        del viewed, vit_features

    vit_gallery = maybe_coreset(
        torch.cat(vit_parts, dim=0), args.vit_coreset_ratio, args, 1001
    )
    dino_gallery = None
    if not args.control_only:
        dino_gallery = maybe_coreset(
            torch.cat(dino_parts, dim=0), args.dino_coreset_ratio, args, 2001
        )
    cnn_features = encode_cnn(cnn_model, raw, device)
    cnn_gallery = normalize_gallery(
        cnn_features.permute(0, 2, 3, 1).reshape(-1, cnn_features.shape[1])
    )
    state = {
        "vit_gallery": vit_gallery,
        "cnn_gallery": cnn_gallery,
        "vit_patch_count": int(vit_gallery.shape[0] // (len(support) * len(modes))),
    }
    if not args.control_only:
        state["dino_gallery"] = dino_gallery
    branch_sizes = [f"vit={vit_gallery.shape[0]}"]
    if not args.control_only:
        branch_sizes.append(f"dino={dino_gallery.shape[0]}")
    branch_sizes.append(f"cnn={cnn_gallery.shape[0]}")
    print(
        f"[gallery] dataset={dataset} support={len(support)} modes={list(modes)} "
        + " ".join(branch_sizes),
        flush=True,
    )
    del raw, vit_parts, dino_parts, cnn_features
    gc.collect()
    return state


@torch.inference_mode()
def score_vit_features(features: torch.Tensor, gallery: torch.Tensor, args) -> np.ndarray:
    distances = topk_cosine_distance_chunked(
        features.reshape(-1, features.shape[-1]),
        gallery,
        args.distance_chunk_size,
        args.vit_nn_topk,
    ).reshape(features.shape[0], -1)
    return distances.amax(dim=1).cpu().numpy().astype(np.float32)


@torch.inference_mode()
def score_dino_features(features: torch.Tensor, gallery: torch.Tensor, args) -> np.ndarray:
    distances = topk_cosine_distance_chunked(
        features.reshape(-1, features.shape[-1]),
        gallery,
        args.distance_chunk_size,
        args.dino_nn_topk,
    ).reshape(features.shape[0], -1)
    keep = max(1, int(round(distances.shape[1] * float(args.dino_top_ratio))))
    return distances.topk(keep, dim=1).values.mean(dim=1).cpu().numpy().astype(np.float32)


@torch.inference_mode()
def score_cnn_features(features: torch.Tensor, gallery: torch.Tensor, args) -> np.ndarray:
    return resnet_patch_image_scores(
        features,
        gallery,
        args.distance_chunk_size,
        args.cnn_top_ratio,
        args.cnn_nn_topk,
    ).astype(np.float32)


def reduce_observation_values(records: list[ImageRecord], values: np.ndarray) -> np.ndarray:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        parts = str(record.name).split("::")
        if len(parts) != 3:
            raise ValueError(f"Expected target-scene SU name, got {record.name!r}")
        groups[parts[1]].append(index)
    output = []
    for _observation, indices in sorted(groups.items()):
        if len(indices) != 21:
            raise ValueError(f"OFDMA observation has {len(indices)} SUs, expected 21")
        output.append(float(np.max(values[indices])))
    return np.asarray(output, dtype=np.float32)


def build_support_references(
    support: list[ImageRecord],
    dataset: str,
    state: dict,
    clip_model,
    dino_model,
    dino_transform,
    cnn_model,
    args,
    device: torch.device,
    *,
    observation_level: bool,
) -> dict[str, np.ndarray]:
    """Calibrate the gate from normal support views only."""

    if args.tta_bundle == "none":
        ref_modes = ("identity",)
        cnn_modes = ("identity",)
    else:
        ref_modes = ("time_shift_up_large", "time_shift_down_large")
        cnn_modes = ("time_shift_up", "time_shift_down", "blur")

    vit_refs: list[np.ndarray] = []
    dino_refs: list[np.ndarray] | None = [] if not args.control_only else None
    cnn_refs: list[np.ndarray] = []
    raw = records_to_raw(support, dataset)
    for mode in ref_modes:
        viewed = raw if mode == "identity" else augment_raw(raw, mode, args)
        vit = score_vit_features(
            encode_vit(clip_model, viewed, device), state["vit_gallery"], args
        )
        if not args.control_only:
            dino = score_dino_features(
                encode_dino(dino_model, dino_transform, viewed, device),
                state["dino_gallery"],
                args,
            )
        if observation_level:
            vit = reduce_observation_values(support, vit)
            if not args.control_only:
                dino = reduce_observation_values(support, dino)
        vit_refs.append(vit)
        if not args.control_only:
            dino_refs.append(dino)

    for mode in cnn_modes:
        viewed = raw if mode == "identity" else augment_raw(raw, mode, args)
        cnn = score_cnn_features(
            encode_cnn(cnn_model, viewed, device), state["cnn_gallery"], args
        )
        if observation_level:
            cnn = reduce_observation_values(support, cnn)
        cnn_refs.append(cnn)

    references = {
        "vit": np.concatenate(vit_refs).astype(np.float32),
        "cnn": np.concatenate(cnn_refs).astype(np.float32),
    }
    if not args.control_only:
        references["dino"] = np.concatenate(dino_refs).astype(np.float32)
    reference_sizes = [f"vit={len(references['vit'])}"]
    if not args.control_only:
        reference_sizes.append(f"dino={len(references['dino'])}")
    reference_sizes.append(f"cnn={len(references['cnn'])}")
    print(
        f"[reference] dataset={dataset} observation_level={observation_level} "
        + " ".join(reference_sizes),
        flush=True,
    )
    del raw
    return references


@torch.inference_mode()
def score_records_multi(
    records: list[ImageRecord],
    dataset: str,
    states: dict[str, dict],
    clip_model,
    dino_model,
    dino_transform,
    cnn_model,
    args,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    unique = deduplicate_records(records)
    indexed = {record.path: index for index, record in enumerate(unique)}
    branches = ("vit", "cnn") if args.control_only else ("vit", "dino", "cnn")
    outputs = {
        key: {name: np.zeros(len(unique), dtype=np.float32) for name in branches}
        for key in states
    }
    for start in tqdm(
        range(0, len(unique), args.batch_size),
        desc=f"Score {dataset} query images",
        leave=False,
    ):
        batch = unique[start : start + args.batch_size]
        raw = records_to_raw(batch, dataset)
        vit_features = encode_vit(clip_model, raw, device)
        dino_features = None if args.control_only else encode_dino(dino_model, dino_transform, raw, device)
        cnn_features = encode_cnn(cnn_model, raw, device)
        for state_key, state in states.items():
            outputs[state_key]["vit"][start : start + len(batch)] = score_vit_features(
                vit_features, state["vit_gallery"], args
            )
            if not args.control_only:
                outputs[state_key]["dino"][start : start + len(batch)] = score_dino_features(
                    dino_features, state["dino_gallery"], args
                )
            outputs[state_key]["cnn"][start : start + len(batch)] = score_cnn_features(
                cnn_features, state["cnn_gallery"], args
            )
        del raw, vit_features, dino_features, cnn_features
    ordered = {}
    for state_key, values in outputs.items():
        ordered[state_key] = {
            branch: np.asarray([values[branch][indexed[record.path]] for record in records], dtype=np.float32)
            for branch in branches
        }
    return ordered


def gated_methods(
    branches: dict[str, np.ndarray], references: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    vit_gate = safe_support_only_gate(
        branches["vit"], branches["cnn"], references["vit"], references["cnn"]
    )
    methods = {
        "ours_vit_cnn_gate": vit_gate["score"].astype(np.float32),
        "vit_only": branches["vit"].astype(np.float32),
        "cnn_only": branches["cnn"].astype(np.float32),
    }
    if "dino" in branches:
        dino_gate = safe_support_only_gate(
            branches["dino"], branches["cnn"], references["dino"], references["cnn"]
        )
        methods.update(
            {
                "ours_dinov2_cnn_gate": dino_gate["score"].astype(np.float32),
                "dinov2_only": branches["dino"].astype(np.float32),
            }
        )
    return methods


def append_metric_rows(
    rows: list[dict],
    output_root: Path,
    cell_id: str,
    metadata: dict,
    records: list[ImageRecord],
    methods: dict[str, np.ndarray],
) -> None:
    labels = np.asarray([record.label for record in records], dtype=np.int32)
    payload = {"labels": labels, "names": np.asarray([record.name for record in records])}
    for key in methods:
        values = methods[key]
        payload[key] = values
        values_metric = metric(labels, values)
        rows.append(
            {
                **metadata,
                "cell": cell_id,
                "method": METHOD_LABELS[key],
                "method_key": key,
                "num_normal": int(np.sum(labels == 0)),
                "num_abnormal": int(np.sum(labels == 1)),
                "auroc": values_metric["auroc"],
                "auprc": values_metric["auprc"],
                "fpr95": values_metric["fpr95"],
            }
        )
    output_root.joinpath("scores").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_root / "scores" / f"{cell_id}.npz", **payload)


def build_models(args, device: torch.device):
    clip_model = build_promptad(
        SimpleNamespace(checkpoint=args.checkpoint, image_size=args.image_size), device
    )
    dino_model = None
    dino_transform = None
    if not args.control_only:
        dino_model = build_dino(
            SimpleNamespace(dino_model=args.dino_model, dino_image_size=args.dino_image_size),
            device,
        )
        dino_transform = build_dino_transform(args.dino_image_size)
    if args.cnn_encoder == "wideresnet50":
        cnn_model = WideResNet50LocalEncoder().to(device).eval()
    else:
        cnn_model = ResNet18LocalEncoder().to(device).eval()
    return clip_model, dino_model, dino_transform, cnn_model


def run_inhouse(args, output_root: Path, device: torch.device) -> list[dict]:
    manifest_path = Path(args.support_manifest) if args.support_manifest else output_root / "support_manifest.json"
    manifest = ensure_rf_target_scene_manifest(
        manifest_path,
        collect_samples=collect_rf_target_samples,
        signals=args.signals,
        scenes=args.scenes,
        jsr_by_signal=RF_JSR_BY_SIGNAL,
        normal_sampling=args.normal_sampling,
        seed=args.support_seed if args.support_seed is not None else args.seed,
    )
    args.support_manifest = str(manifest_path)
    clip_model, dino_model, dino_transform, cnn_model = build_models(args, device)
    rows: list[dict] = []
    for scene in args.scenes:
        support = manifest_support_records(manifest, scene)
        state = build_state(
            support, "inhouse", clip_model, dino_model, dino_transform, cnn_model, args, device
        )
        state["references"] = build_support_references(
            support,
            "inhouse",
            state,
            clip_model,
            dino_model,
            dino_transform,
            cnn_model,
            args,
            device,
            observation_level=False,
        )
        cells = {}
        for signal in args.signals:
            for jsr in RF_JSR_BY_SIGNAL[signal]:
                entry = cell_entry(manifest, signal, scene, jsr)
                normals = entry["test_normals"]
                abnormals = entry["test_abnormals"]
                if args.max_test_normals > 0:
                    normals = normals[: args.max_test_normals]
                if args.max_abnormals > 0:
                    abnormals = abnormals[: args.max_abnormals]
                cells[(signal, jsr)] = [
                    *[ImageRecord(str(item["path"]), 0, f"{signal}-{scene}-{jsr}-normal-{item['path']}") for item in normals],
                    *[ImageRecord(str(item["path"]), 1, f"{signal}-{scene}-{jsr}-abnormal-{item['path']}") for item in abnormals],
                ]
        all_records = deduplicate_records([record for cell in cells.values() for record in cell])
        lookup = score_records_multi(
            all_records,
            "inhouse",
            {scene: state},
            clip_model,
            dino_model,
            dino_transform,
            cnn_model,
            args,
            device,
        )[scene]
        index = {record.path: i for i, record in enumerate(all_records)}
        for (signal, jsr), records in cells.items():
            branches = {
                key: np.asarray(
                    [lookup[key][index[record.path]] for record in records],
                    dtype=np.float32,
                )
                for key in branch_keys(args)
            }
            methods = gated_methods(branches, state["references"])
            append_metric_rows(
                rows,
                output_root,
                f"{signal}-{scene}-{jsr}",
                {"dataset": "In-house RF", "signal": signal, "scene": scene, "jsr": jsr, "support_count": len(support), "normal_sampling": args.normal_sampling},
                records,
                methods,
            )
        del state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def run_public(args, output_root: Path, device: torch.device) -> list[dict]:
    manifest = public_manifest(args, output_root)
    support = [ImageRecord(str(path), 0, f"public-support-{index:04d}") for index, path in enumerate(manifest["support_paths"])]
    clip_model, dino_model, dino_transform, cnn_model = build_models(args, device)
    state = build_state(
        support, "public", clip_model, dino_model, dino_transform, cnn_model, args, device
    )
    state["references"] = build_support_references(
        support,
        "public",
        state,
        clip_model,
        dino_model,
        dino_transform,
        cnn_model,
        args,
        device,
        observation_level=False,
    )
    jsrs = public_jsrs(args)
    cells = {}
    normal_paths = list(manifest["test_paths"])
    for signal in args.signals:
        for jsr in jsrs[signal]:
            normal_subset = normal_paths[: args.max_test_normals] if args.max_test_normals > 0 else normal_paths
            abnormal_root = Path(args.data_root) / signal / "abnormal" / jsr
            abnormal_paths = sorted(abnormal_root.glob("*.png")) if abnormal_root.is_dir() else []
            if args.max_abnormals > 0:
                abnormal_paths = abnormal_paths[: args.max_abnormals]
            cells[(signal, jsr)] = [
                *[ImageRecord(str(path), 0, f"{signal}-{jsr}-normal-{Path(path).name}") for path in normal_subset],
                *[ImageRecord(str(path), 1, f"{signal}-{jsr}-abnormal-{path.name}") for path in abnormal_paths],
            ]
    all_records = deduplicate_records([record for cell in cells.values() for record in cell])
    lookup = score_records_multi(
        all_records,
        "public",
        {"public": state},
        clip_model,
        dino_model,
        dino_transform,
        cnn_model,
        args,
        device,
    )["public"]
    index = {record.path: i for i, record in enumerate(all_records)}
    rows: list[dict] = []
    for (signal, jsr), records in cells.items():
        branches = {
            key: np.asarray(
                [lookup[key][index[record.path]] for record in records],
                dtype=np.float32,
            )
            for key in branch_keys(args)
        }
        methods = gated_methods(branches, state["references"])
        append_metric_rows(
            rows,
            output_root,
            f"{signal}-{jsr}",
            {"dataset": "Public RF", "signal": signal, "scene": "RF_SPE_PNG_public", "jsr": jsr, "support_count": len(support), "normal_sampling": args.normal_sampling},
            records,
            methods,
        )
    return rows


def run_ofdma(args, output_root: Path, device: torch.device) -> list[dict]:
    dataset_root = Path(args.data_root)
    manifest = load_target_scene_manifest(dataset_root)
    available = available_target_scenes(manifest, args.split)
    scene_ids = list(args.scene_ids or available)
    unknown = sorted(set(scene_ids) - set(available))
    if unknown:
        raise ValueError(f"Unknown OFDMA scenes: {unknown}")
    protocol = json.loads((dataset_root / "protocol.json").read_text(encoding="utf-8"))
    normalization = protocol["source_protocol"]["normalization"]
    preprocessor = OFDMASpectrogramPreprocessor(
        dataset_root,
        output_size=args.image_size,
        subcarriers_per_rb=args.ofdma_subcarriers_per_rb,
        geometry=args.ofdma_geometry,
        min_db=float(normalization["min_db"]),
        max_db=float(normalization["max_db"]),
    )
    clip_model, dino_model, dino_transform, cnn_model = build_models(args, device)
    rows: list[dict] = []
    for scene in scene_ids:
        test_frame = build_target_scene_records(
            dataset_root,
            manifest,
            scene,
            split=args.split,
            role="test",
            max_normal_observations=args.max_normal_observations,
            max_anomaly_observations_per_type=args.max_abnormal_observations_per_type,
        )
        test_records = ofdma_preprocessed_records(test_frame, preprocessor)
        states = {}
        for shot in sorted(set(args.shots)):
            support_frame = build_target_scene_records(
                dataset_root,
                manifest,
                scene,
                split=args.split,
                role="support",
                shot=shot,
                support_seed=args.support_seed,
            )
            support = ofdma_preprocessed_records(support_frame, preprocessor)
            state = build_state(
                support, "ofdma", clip_model, dino_model, dino_transform, cnn_model, args, device
            )
            state["references"] = build_support_references(
                support,
                "ofdma",
                state,
                clip_model,
                dino_model,
                dino_transform,
                cnn_model,
                args,
                device,
                observation_level=True,
            )
            states[str(shot)] = state

        branches_by_state = score_records_multi(
            test_records,
            "ofdma",
            states,
            clip_model,
            dino_model,
            dino_transform,
            cnn_model,
            args,
            device,
        )
        # The OFDMA protocol reduces 21 sensing units to one observation by
        # maximum evidence before applying the support-only gate.
        groups: dict[str, list[int]] = defaultdict(list)
        for i, record in enumerate(test_records):
            groups[str(record.name).split("::")[1]].append(i)
        observations = []
        for observation, indices in sorted(groups.items()):
            if len(indices) != 21:
                raise ValueError(f"OFDMA observation {observation} has {len(indices)} SUs")
            labels = {test_records[i].label for i in indices}
            if len(labels) != 1:
                raise ValueError(f"Inconsistent OFDMA labels for {observation}")
            observations.append(ImageRecord(f"observation::{observation}", next(iter(labels)), observation))
        obs_by_state = {}
        for state_key, branches in branches_by_state.items():
            reduced = {
                key: np.asarray(
                    [
                        np.max(branches[key][indices])
                        for _obs, indices in sorted(groups.items())
                    ],
                    dtype=np.float32,
                )
                for key in branch_keys(args)
            }
            obs_by_state[state_key] = gated_methods(reduced, states[state_key]["references"])
            append_metric_rows(
                rows,
                output_root,
                f"{scene}-{state_key}shot",
                {"dataset": "OFDMA", "signal": "all jammer types", "scene": scene, "jsr": "mixed", "support_count": int(state_key) * 21, "normal_sampling": f"{state_key}-shot"},
                observations,
                obs_by_state[state_key],
            )
        del states, branches_by_state, test_records
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["inhouse", "public", "ofdma"], required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--control-only",
        action="store_true",
        help="只评估原主线 ViT+CNN；不构建、不提取、不计算 DINOv2。",
    )
    parser.add_argument(
        "--checkpoint",
        default="analysis_outputs/02_current_baselines/promptad_formal_baseline/pooled_rf_rgb_cls/checkpoint/overall-best.pt",
    )
    parser.add_argument("--data-root", default="")
    parser.add_argument("--signals", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--jsrs-by-signal", default=None)
    parser.add_argument("--normal-sampling", default="per_frequency")
    parser.add_argument("--support-manifest", default="")
    parser.add_argument("--support-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--distance-chunk-size", type=int, default=4096)
    parser.add_argument("--image-size", type=int, default=240)
    parser.add_argument("--dino-model", default="vit_base_patch14_dinov2")
    parser.add_argument("--dino-image-size", type=int, default=224)
    parser.add_argument("--tta-bundle", default=None, choices=["none", "rf_spectral_response_v1", "ofdma_spectral_response_v1"])
    parser.add_argument("--tta-shift-px", type=int, default=4)
    parser.add_argument("--tta-frequency-response-strength", type=float, default=3.0)
    parser.add_argument("--vit-coreset-ratio", type=float, default=1.0)
    parser.add_argument("--dino-coreset-ratio", type=float, default=1.0)
    parser.add_argument("--vit-nn-topk", type=int, default=1)
    parser.add_argument("--dino-nn-topk", type=int, default=1)
    parser.add_argument("--dino-top-ratio", type=float, default=0.1)
    parser.add_argument("--cnn-top-ratio", type=float, default=0.1)
    parser.add_argument("--cnn-nn-topk", type=int, default=1)
    parser.add_argument(
        "--cnn-encoder",
        choices=["resnet18", "wideresnet50"],
        default="resnet18",
        help="Frozen ImageNet backbone for the auxiliary CNN branch (layer3).",
    )
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--scene-ids", nargs="*", default=[])
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-normal-observations", type=int, default=0)
    parser.add_argument("--max-abnormal-observations-per-type", type=int, default=0)
    parser.add_argument("--ofdma-geometry", choices=["letterbox", "square_warp"], default="letterbox")
    parser.add_argument("--ofdma-subcarriers-per-rb", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "inhouse":
        args.signals = list(args.signals or RF_TARGET_SIGNALS)
        args.scenes = list(args.scenes or RF_SCENES)
        args.data_root = args.data_root or "/mnt/data/wangbei/data/datasets"
        args.tta_bundle = args.tta_bundle or "rf_spectral_response_v1"
    elif args.dataset == "public":
        args.data_root = args.data_root or "/mnt/data/wangbei/data/RF_SPE_PNG"
        args.signals = list(args.signals or ("burst", "chirp", "dsss", "pulse", "deceptive"))
        args.scenes = []
        args.tta_bundle = args.tta_bundle or "rf_spectral_response_v1"
    else:
        args.data_root = args.data_root or "/mnt/data/wangbei/data/ofdma-target-scene-coldstart-v2-realistic"
        args.signals = ["all jammer types"]
        args.scenes = list(args.scene_ids)
        args.tta_bundle = args.tta_bundle or "ofdma_spectral_response_v1"
    for name in ("vit_coreset_ratio", "dino_coreset_ratio"):
        value = float(getattr(args, name))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1]")
    if args.vit_nn_topk < 1 or args.dino_nn_topk < 1 or args.cnn_nn_topk < 1:
        raise ValueError("NN top-k values must be positive")
    if not 0.0 < args.dino_top_ratio <= 1.0 or not 0.0 < args.cnn_top_ratio <= 1.0:
        raise ValueError("top ratios must be in (0, 1]")
    if not args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    setup_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda:0")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device} dataset={args.dataset} data_root={args.data_root}", flush=True)

    if args.dataset == "inhouse":
        rows = run_inhouse(args, output_root, device)
    elif args.dataset == "public":
        rows = run_public(args, output_root, device)
    else:
        rows = run_ofdma(args, output_root, device)

    write_csv(output_root / "metrics.csv", rows)
    frame = pd.DataFrame(rows)
    macro = {}
    labels = method_labels(args)
    if not frame.empty:
        for key, group in frame.groupby("method_key"):
            macro[key] = {
                "method": labels[key],
                "auroc": float(group["auroc"].mean()),
                "auprc": float(group["auprc"].mean()),
                "fpr95": float(group["fpr95"].mean()),
                "cells": int(len(group)),
            }
    backbones = {
        "vit": "PromptAD ViT-B-16-plus-240 normalized layer1+layer2",
        "cnn": "ImageNet ResNet18 layer3",
    }
    memory = {
        "vit": "full normal patch memory, 1-NN, maximum patch distance",
        "cnn": f"full normal layer3 patch memory, 1-NN, top {args.cnn_top_ratio:g} patch distance mean",
    }
    if not args.control_only:
        backbones["candidate_dino"] = args.dino_model
        memory["dino"] = (
            f"full normal patch memory, 1-NN, top {args.dino_top_ratio:g} patch distance mean"
        )

    summary = {
        "status": "complete",
        "method": (
            "ours_vit_cnn_full_memory_1nn"
            if args.control_only
            else "ours_vit_vs_dinov2_replacement_cross_dataset"
        ),
        "dataset": args.dataset,
        "data_root": str(Path(args.data_root).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "support_only": True,
        "uses_test_batch_statistics": False,
        "uses_test_labels_for_scoring": False,
        "changed_component": (
            "none: original PromptAD ViT layer1+layer2 + CNN layer3 mainline"
            if args.control_only
            else "PromptAD ViT layer1+layer2 -> DINOv2 patch tokens"
        ),
        "same_as_control": [
            "normal support split",
            "spectral TTA support-memory bundle",
            "full normal patch memory",
            "1-NN distance",
            "ImageNet ResNet18 layer3 branch",
            "support-only confidence gate",
            "independent test set",
        ],
        "backbones": backbones,
        "tta_bundle": args.tta_bundle,
        "tta_modes": list(tta_modes(args)),
        "tta_shift_px": args.tta_shift_px,
        "tta_frequency_response_strength": args.tta_frequency_response_strength,
        "memory": memory,
        "gate": "same safe_support_only_gate applied after branch score (after 21-SU max for OFDMA)",
        "method_labels": labels,
        "macro": macro,
        "num_rows": int(len(rows)),
    }
    save_json(output_root / "summary.json", summary)
    save_json(output_root / "protocol.json", summary)
    readme_title = (
        "# Ours 原主线：全量 memory + 1-NN\n\n"
        if args.control_only
        else "# Ours：ViT 替换为 DINOv2（跨数据集）\n\n"
    )
    readme_body = (
        "本实验只评估原主线 ViT layer1+layer2 + CNN layer3；使用频谱 TTA、全量正常记忆库、1-NN 和 support-only 门控。\n\n"
        if args.control_only
        else "本实验只替换 Ours 的 ViT 视觉记忆分支；CNN layer3、频谱 TTA、全量正常记忆库、1-NN 和 support-only 门控保持一致。\n\n"
    )
    (output_root / "README.md").write_text(
        readme_title
        + readme_body
        + f"- 数据集：`{args.dataset}`\n"
        + f"- TTA：`{args.tta_bundle}`，模式 `{list(tta_modes(args))}`\n"
        + "- 结果：`metrics.csv`；汇总：`summary.json`\n",
        encoding="utf-8",
    )
    if device.type == "cuda":
        print(f"[gpu] max_allocated_mb={torch.cuda.max_memory_allocated(device) / 1024**2:.1f}", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
