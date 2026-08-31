#!/usr/bin/env python3
"""Evaluate ViT+DINOv2 weighted local-memory fusion on RF and OFDMA datasets.

The evaluator keeps the method definition used by the FedJam experiment:

* PromptAD ViT ``layer1 + layer2`` patch features, 5-NN and max patch score;
* DINOv2 patch features, 1-NN and mean of the highest 10% patch distances;
* spectral TTA is applied to *normal DINO support only* and merged into one
  memory; every test image is queried in its original view once;
* the final comparison uses a fixed weighted mean on the bounded cosine
  distance scale.  No test-batch statistics or labels are used.

In-house RF, Public RF, and OFDMA have different support manifests and cell
layouts, so this entry point deliberately preserves each dataset's official
split.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from dataclasses import dataclass
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

from datasets.rf_target import (  # noqa: E402
    RF_JSR_BY_SIGNAL as RF_JSR_BY_SIGNAL,
    RF_SCENES,
    RF_TARGET_SIGNALS,
    collect_rf_target_samples,
)
from datasets.ofdma_spectrum import OFDMASpectrogramPreprocessor  # noqa: E402
from datasets.ofdma_target_scene import (  # noqa: E402
    available_target_scenes,
    build_target_scene_records,
    load_target_scene_manifest,
)
from tools.eval_cls_dinov2_patchcore_gallery import (  # noqa: E402
    build_transform as build_dino_transform,
    extract_patch_features,
    to_dino_input,
)
from tools.eval_cls_public_rf_dual_gallery import (  # noqa: E402
    DEFAULT_THREE_JSR_BY_SIGNAL,
    PUBLIC_SIGNALS,
)
from tools.eval_cls_vit_patchcore_gallery import (  # noqa: E402
    prepare_patch_features,
    select_gallery_subset,
    topk_cosine_distance_chunked,
)
from tools.eval_fedjam_fewshot_dual import metric  # noqa: E402
from tools.eval_fedjam_four_branches import build_dino, build_promptad  # noqa: E402
from train_rf_target_pooled_universal import to_model_input  # noqa: E402
from utils.rf_scene_support import (  # noqa: E402
    cell_entry,
    ensure_rf_target_scene_manifest,
    scene_support_entry,
)
from utils.spectral_tta import SPECTRAL_TTA_BUNDLES, augment_spectrogram  # noqa: E402
from utils.training_utils import setup_seed  # noqa: E402


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: int
    name: str
    image_bgr: np.ndarray | None = None


METHOD_KEYS = ("vit_only", "dino_only", "vit25_dino75", "vit50_dino50", "vit75_dino25")
METHOD_LABELS = {
    "vit_only": "ViT-only",
    "dino_only": "DINO-only",
    "vit25_dino75": "ViT+DINO weighted (25/75)",
    "vit50_dino50": "ViT+DINO weighted (50/50)",
    "vit75_dino25": "ViT+DINO weighted (75/25)",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_paths(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()


def load_bgr(path: str, dataset: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if dataset == "public":
        # Match PublicRFPathDataset: keep the source aspect square and cap
        # very large inputs before PromptAD/DINO preprocessing.
        target = min(max(image.shape[:2]), 1024)
        image = cv2.resize(image, (target, target), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image)


def records_to_raw(records: list[ImageRecord], dataset: str) -> torch.Tensor:
    return torch.from_numpy(
        np.stack(
            [
                record.image_bgr
                if record.image_bgr is not None
                else load_bgr(record.path, dataset)
                for record in records
            ],
            axis=0,
        )
    )


def deduplicate_records(records: list[ImageRecord]) -> list[ImageRecord]:
    by_path: dict[str, ImageRecord] = {}
    for record in records:
        previous = by_path.get(record.path)
        if previous is not None and previous.label != record.label:
            raise ValueError(f"Path has inconsistent labels: {record.path}")
        by_path.setdefault(record.path, record)
    return list(by_path.values())


def dino_tta_modes(args) -> tuple[str, ...]:
    if args.dino_tta == "none":
        return ("identity",)
    return tuple(SPECTRAL_TTA_BUNDLES[args.dino_tta])


def augment_raw_batch(raw: torch.Tensor, mode: str, args) -> torch.Tensor:
    if mode == "identity":
        return raw
    arrays = [
        augment_spectrogram(
            image.numpy(),
            mode,
            shift_px=args.dino_tta_shift_px,
            frequency_response_strength=args.dino_tta_frequency_response_strength,
        )
        for image in raw
    ]
    return torch.from_numpy(np.stack(arrays, axis=0))


@torch.inference_mode()
def encode_batch(
    clip_model,
    dino_model,
    dino_transform,
    raw: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    clip_input = to_model_input(clip_model, raw, device, rgb_from_bgr=True)
    vit = prepare_patch_features(clip_model.encode_image(clip_input)).float().contiguous()
    dino_input = to_dino_input(raw, dino_transform, device, rgb_from_bgr=True)
    dino = extract_patch_features(dino_model, dino_input).float().contiguous()
    return {"vit": vit, "dino": dino}


@torch.inference_mode()
def encode_dino_only(
    dino_model,
    dino_transform,
    raw: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    dino_input = to_dino_input(raw, dino_transform, device, rgb_from_bgr=True)
    return extract_patch_features(dino_model, dino_input).float().contiguous()


def normalize_gallery(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=-1).contiguous()


def build_state(
    support_records: list[ImageRecord],
    dataset: str,
    clip_model,
    dino_model,
    dino_transform,
    args,
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    support_raw = records_to_raw(support_records, dataset)
    support_features = encode_batch(
        clip_model, dino_model, dino_transform, support_raw, device
    )

    vit_gallery = normalize_gallery(
        support_features["vit"].reshape(-1, support_features["vit"].shape[-1])
    )
    if args.vit_coreset_ratio < 1.0:
        subset_args = SimpleNamespace(
            coreset_ratio=args.vit_coreset_ratio,
            coreset_method=args.coreset_method,
            seed=args.seed,
        )
        indices, _ = select_gallery_subset(vit_gallery, subset_args)
        vit_gallery = vit_gallery[indices]

    dino_parts = []
    for mode in dino_tta_modes(args):
        viewed = augment_raw_batch(support_raw, mode, args)
        if mode == "identity":
            features = support_features["dino"]
        else:
            features = encode_dino_only(dino_model, dino_transform, viewed, device)
        dino_parts.append(features.reshape(-1, features.shape[-1]))
        del viewed, features
    dino_gallery = normalize_gallery(torch.cat(dino_parts, dim=0))
    if args.dino_max_gallery_patches > 0 and dino_gallery.shape[0] > args.dino_max_gallery_patches:
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 7919)
        indices = torch.randperm(dino_gallery.shape[0], generator=generator)[: args.dino_max_gallery_patches]
        dino_gallery = dino_gallery[indices.to(dino_gallery.device)]

    state = {
        "vit_gallery": vit_gallery,
        "dino_gallery": dino_gallery,
        "vit_patch_count": int(support_features["vit"].shape[1]),
        "dino_patch_count": int(support_features["dino"].shape[1]),
    }
    print(
        f"[gallery] support={len(support_records)} "
        f"vit_patches={vit_gallery.shape[0]} dino_patches={dino_gallery.shape[0]} "
        f"dino_modes={list(dino_tta_modes(args))}",
        flush=True,
    )
    del support_raw, support_features, dino_parts
    gc.collect()
    return state


@torch.inference_mode()
def score_feature_batch(
    features: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor | int],
    args,
) -> dict[str, np.ndarray]:
    vit_distances = topk_cosine_distance_chunked(
        features["vit"].reshape(-1, features["vit"].shape[-1]),
        state["vit_gallery"],
        args.distance_chunk_size,
        args.vit_nn_topk,
    )
    vit_map_scores = vit_distances.reshape(features["vit"].shape[0], -1)
    rank = int(getattr(args, "vit_patch_score_rank", 0))
    if rank > 0:
        rank = min(rank, int(vit_map_scores.shape[1]))
        vit_scores = vit_map_scores.topk(rank, dim=1).values[:, rank - 1]
    else:
        vit_scores = vit_map_scores.amax(dim=1)
    vit_scores = (
        vit_scores
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    dino_distances = topk_cosine_distance_chunked(
        features["dino"].reshape(-1, features["dino"].shape[-1]),
        state["dino_gallery"],
        args.distance_chunk_size,
        args.dino_nn_topk,
    ).reshape(features["dino"].shape[0], -1)
    keep = max(1, int(round(dino_distances.shape[1] * args.dino_top_ratio)))
    dino_scores = (
        dino_distances.topk(keep, dim=1).values.mean(dim=1)
        .mul(2.0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    vit_evidence = np.clip(vit_scores, 0.0, 1.0)
    dino_evidence = np.clip(dino_scores / 2.0, 0.0, 1.0)
    return {
        "vit_only": vit_scores,
        "dino_only": dino_scores,
        "vit25_dino75": (0.25 * vit_evidence + 0.75 * dino_evidence).astype(np.float32),
        "vit50_dino50": (0.50 * vit_evidence + 0.50 * dino_evidence).astype(np.float32),
        "vit75_dino25": (0.75 * vit_evidence + 0.25 * dino_evidence).astype(np.float32),
    }


@torch.inference_mode()
def score_batch(
    raw: torch.Tensor,
    state: dict[str, torch.Tensor | int],
    clip_model,
    dino_model,
    dino_transform,
    args,
    device: torch.device,
) -> dict[str, np.ndarray]:
    features = encode_batch(clip_model, dino_model, dino_transform, raw, device)
    return score_feature_batch(features, state, args)


def score_records(
    records: list[ImageRecord],
    dataset: str,
    state,
    clip_model,
    dino_model,
    dino_transform,
    args,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray | int]]:
    unique = deduplicate_records(records)
    scores_by_path = {key: {} for key in METHOD_KEYS}
    labels_by_path = {}
    names_by_path = {}
    for start in tqdm(
        range(0, len(unique), args.batch_size),
        desc=f"Score {dataset} unique test images",
        leave=False,
    ):
        batch_records = unique[start : start + args.batch_size]
        raw = records_to_raw(batch_records, dataset)
        batch_scores = score_batch(
            raw,
            state,
            clip_model,
            dino_model,
            dino_transform,
            args,
            device,
        )
        for index, record in enumerate(batch_records):
            labels_by_path[record.path] = record.label
            names_by_path[record.path] = record.name
            for key in METHOD_KEYS:
                scores_by_path[key][record.path] = float(batch_scores[key][index])
        del raw, batch_scores
    return {
        key: {
            "scores": np.asarray([scores_by_path[key][record.path] for record in records], dtype=np.float32),
            "labels": np.asarray([labels_by_path[record.path] for record in records], dtype=np.int32),
            "names": np.asarray([names_by_path[record.path] for record in records]),
        }
        for key in METHOD_KEYS
    }


def score_records_multi(
    records: list[ImageRecord],
    dataset: str,
    states: dict[str, dict[str, torch.Tensor | int]],
    clip_model,
    dino_model,
    dino_transform,
    args,
    device: torch.device,
) -> dict[str, dict[str, dict[str, np.ndarray | int]]]:
    """Score one query feature pass against several support memories.

    This is used by OFDMA, where the same scene/query set is evaluated for
    1/2/4-shot support.  The image encoders run once per query batch; only
    the memory-bank distance calculation is repeated for each shot.
    """
    unique = deduplicate_records(records)
    scores_by_state = {
        state_key: {key: {} for key in METHOD_KEYS}
        for state_key in states
    }
    labels_by_path = {}
    names_by_path = {}
    for start in tqdm(
        range(0, len(unique), args.batch_size),
        desc=f"Score {dataset} unique test images (multi-memory)",
        leave=False,
    ):
        batch_records = unique[start : start + args.batch_size]
        raw = records_to_raw(batch_records, dataset)
        features = encode_batch(clip_model, dino_model, dino_transform, raw, device)
        for state_key, state in states.items():
            batch_scores = score_feature_batch(features, state, args)
            for index, record in enumerate(batch_records):
                labels_by_path[record.path] = record.label
                names_by_path[record.path] = record.name
                for key in METHOD_KEYS:
                    scores_by_state[state_key][key][record.path] = float(
                        batch_scores[key][index]
                    )
            del batch_scores
        del raw, features
    return {
        state_key: {
            key: {
                "scores": np.asarray(
                    [scores_by_state[state_key][key][record.path] for record in records],
                    dtype=np.float32,
                ),
                "labels": np.asarray(
                    [labels_by_path[record.path] for record in records],
                    dtype=np.int32,
                ),
                "names": np.asarray(
                    [names_by_path[record.path] for record in records]
                ),
            }
            for key in METHOD_KEYS
        }
        for state_key in states
    }


def append_cell_results(
    rows: list[dict],
    output_root: Path,
    cell_id: str,
    metadata: dict,
    records: list[ImageRecord],
    score_lookup: dict[str, dict[str, np.ndarray | int]],
) -> None:
    labels = np.asarray([record.label for record in records], dtype=np.int32)
    payload = {"labels": labels, "names": np.asarray([record.name for record in records])}
    for key in METHOD_KEYS:
        values = np.asarray(score_lookup[key]["scores"], dtype=np.float32)
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


def build_model(args, device: torch.device):
    clip_model = build_promptad(
        SimpleNamespace(checkpoint=args.checkpoint, image_size=args.image_size), device
    )
    dino_model = build_dino(
        SimpleNamespace(
            dino_model=args.dino_model,
            dino_image_size=args.dino_image_size,
        ),
        device,
    )
    return clip_model, dino_model, build_dino_transform(args.dino_image_size)


def inhouse_manifest(args, output_root: Path):
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
    return manifest


def manifest_support_records(manifest: dict, scene: str) -> list[ImageRecord]:
    return [
        ImageRecord(str(item["path"]), 0, f"{scene}_support_{index:04d}")
        for index, item in enumerate(scene_support_entry(manifest, scene)["support"])
    ]


def run_inhouse(args, output_root: Path, device: torch.device) -> list[dict]:
    manifest = inhouse_manifest(args, output_root)
    clip_model, dino_model, dino_transform = build_model(args, device)
    rows: list[dict] = []
    for scene in args.scenes:
        support = manifest_support_records(manifest, scene)
        state = build_state(
            support,
            "inhouse",
            clip_model,
            dino_model,
            dino_transform,
            args,
            device,
        )
        cells: dict[tuple[str, str], list[ImageRecord]] = {}
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
                    *[
                        ImageRecord(str(item["path"]), 0, f"{signal}-{scene}-{jsr}-normal-{item['path']}")
                        for item in normals
                    ],
                    *[
                        ImageRecord(str(item["path"]), 1, f"{signal}-{scene}-{jsr}-abnormal-{item['path']}")
                        for item in abnormals
                    ],
                ]
        all_records = deduplicate_records([record for values in cells.values() for record in values])
        index_by_path = {record.path: index for index, record in enumerate(all_records)}
        lookup = score_records(
            all_records,
            "inhouse",
            state,
            clip_model,
            dino_model,
            dino_transform,
            args,
            device,
        )
        for (signal, jsr), records in cells.items():
            ordered_lookup = {
                key: {
                    "scores": np.asarray(
                        [lookup[key]["scores"][index_by_path[record.path]] for record in records],
                        dtype=np.float32,
                    )
                }
                for key in METHOD_KEYS
            }
            append_cell_results(
                rows,
                output_root,
                f"{signal}-{scene}-{jsr}",
                {
                    "dataset": "In-house RF",
                    "signal": signal,
                    "scene": scene,
                    "jsr": jsr,
                    "normal_sampling": args.normal_sampling,
                    "support_count": len(support),
                },
                records,
                ordered_lookup,
            )
        del state
        gc.collect()
    return rows


def public_manifest(args, output_root: Path) -> dict:
    if args.support_manifest:
        path = Path(args.support_manifest)
    else:
        path = output_root / "support_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            "Public RF requires the fixed k-per-frequency support manifest; "
            f"not found: {path}"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    args.support_manifest = str(path)
    return manifest


def public_jsrs(args) -> dict[str, tuple[str, ...]]:
    result = {key: tuple(value) for key, value in DEFAULT_THREE_JSR_BY_SIGNAL.items()}
    if not args.jsrs_by_signal:
        return result
    for item in args.jsrs_by_signal.split(";"):
        signal, values = item.split(":", 1)
        result[signal] = tuple(value.strip() for value in values.split(",") if value.strip())
    return result


def run_public(args, output_root: Path, device: torch.device) -> list[dict]:
    manifest = public_manifest(args, output_root)
    support = [
        ImageRecord(str(path), 0, f"public-support-{index:04d}")
        for index, path in enumerate(manifest["support_paths"])
    ]
    clip_model, dino_model, dino_transform = build_model(args, device)
    state = build_state(
        support,
        "public",
        clip_model,
        dino_model,
        dino_transform,
        args,
        device,
    )
    jsrs = public_jsrs(args)
    cells: dict[tuple[str, str], list[ImageRecord]] = {}
    normal_paths = list(manifest["test_paths"])
    for signal in args.signals:
        for jsr in jsrs[signal]:
            normal_subset = normal_paths[: args.max_test_normals] if args.max_test_normals > 0 else normal_paths
            abnormal_root = Path(args.data_root) / signal / "abnormal" / jsr
            abnormal_paths = sorted(abnormal_root.glob("*.png")) if abnormal_root.is_dir() else []
            if args.max_abnormals > 0:
                abnormal_paths = abnormal_paths[: args.max_abnormals]
            cells[(signal, jsr)] = [
                *[
                    ImageRecord(str(path), 0, f"{signal}-{jsr}-normal-{Path(path).name}")
                    for path in normal_subset
                ],
                *[
                    ImageRecord(str(path), 1, f"{signal}-{jsr}-abnormal-{path.name}")
                    for path in abnormal_paths
                ],
            ]
    all_records = deduplicate_records([record for values in cells.values() for record in values])
    lookup = score_records(
        all_records,
        "public",
        state,
        clip_model,
        dino_model,
        dino_transform,
        args,
        device,
    )
    index_by_path = {record.path: index for index, record in enumerate(all_records)}
    rows: list[dict] = []
    for (signal, jsr), records in cells.items():
        ordered_lookup = {
            key: {
                "scores": np.asarray(
                    [lookup[key]["scores"][index_by_path[record.path]] for record in records],
                    dtype=np.float32,
                )
            }
            for key in METHOD_KEYS
        }
        append_cell_results(
            rows,
            output_root,
            f"{signal}-{jsr}",
            {
                "dataset": "Public RF",
                "signal": signal,
                "scene": "RF_SPE_PNG_public",
                "jsr": jsr,
                "normal_sampling": args.normal_sampling,
                "support_count": len(support),
            },
            records,
            ordered_lookup,
        )
    return rows


def ofdma_preprocessed_records(records, preprocessor) -> list[ImageRecord]:
    converted = []
    for record in records:
        gray = cv2.imread(str(record.image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(record.image_path)
        image = preprocessor(gray)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        converted.append(
            ImageRecord(
                path=str(record.image_path),
                label=int(record.label),
                name=(
                    f"{record.target_scene_id}::{record.observation_id}::"
                    f"su{record.su_id:02d}"
                ),
                image_bgr=np.ascontiguousarray(image_bgr),
            )
        )
    return converted


def aggregate_ofdma_observations(
    records: list[ImageRecord],
    su_lookup: dict[str, dict[str, np.ndarray | int]],
) -> tuple[list[ImageRecord], dict[str, dict[str, np.ndarray | int]]]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        parts = record.name.split("::")
        if len(parts) != 3:
            raise ValueError(f"Invalid OFDMA record name: {record.name}")
        groups.setdefault(parts[1], []).append(index)
    observations = []
    for observation_id, indices in sorted(groups.items()):
        if len(indices) != 21:
            raise ValueError(
                f"OFDMA observation {observation_id} has {len(indices)} SUs, expected 21"
            )
        labels = {records[index].label for index in indices}
        if len(labels) != 1:
            raise ValueError(f"Inconsistent OFDMA labels in {observation_id}")
        observations.append(
            ImageRecord(
                path=f"observation::{observation_id}",
                label=next(iter(labels)),
                name=observation_id,
            )
        )
    output = {}
    for key in METHOD_KEYS:
        output[key] = {
            "scores": np.asarray(
                [
                    np.max(
                        [float(su_lookup[key]["scores"][index]) for index in indices]
                    )
                    for _observation_id, indices in sorted(groups.items())
                ],
                dtype=np.float32,
            ),
            "labels": np.asarray([record.label for record in observations], dtype=np.int32),
            "names": np.asarray([record.name for record in observations]),
        }
    return observations, output


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
    clip_model, dino_model, dino_transform = build_model(args, device)
    args.scenes = scene_ids
    rows: list[dict] = []
    for scene in scene_ids:
        test_frame_records = build_target_scene_records(
            dataset_root,
            manifest,
            scene,
            split=args.split,
            role="test",
            max_normal_observations=args.max_normal_observations,
            max_anomaly_observations_per_type=args.max_abnormal_observations_per_type,
        )
        test_records = ofdma_preprocessed_records(test_frame_records, preprocessor)
        states = {}
        support_counts = {}
        for shot in sorted(set(args.shots)):
            support_frame_records = build_target_scene_records(
                dataset_root,
                manifest,
                scene,
                split=args.split,
                role="support",
                shot=shot,
                support_seed=args.support_seed,
            )
            support_records = ofdma_preprocessed_records(
                support_frame_records, preprocessor
            )
            state = build_state(
                support_records,
                "ofdma",
                clip_model,
                dino_model,
                dino_transform,
                args,
                device,
            )
            states[str(shot)] = state
            support_counts[str(shot)] = len(support_records)

        su_lookups = score_records_multi(
            test_records,
            "ofdma",
            states,
            clip_model,
            dino_model,
            dino_transform,
            args,
            device,
        )
        for shot in sorted(set(args.shots)):
            state_key = str(shot)
            su_lookup = su_lookups[state_key]
            observation_records, observation_lookup = aggregate_ofdma_observations(
                test_records, su_lookup
            )
            append_cell_results(
                rows,
                output_root,
                f"{scene}-{shot}shot",
                {
                    "dataset": "OFDMA",
                    "signal": "all jammer types",
                    "scene": scene,
                    "jsr": "mixed",
                    "normal_sampling": f"{shot}-shot",
                    "support_count": support_counts[state_key],
                },
                observation_records,
                observation_lookup,
            )
        del states, su_lookups
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["inhouse", "public", "ofdma"], required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--checkpoint",
        default=(
            "analysis_outputs/02_current_baselines/promptad_formal_baseline/"
            "pooled_rf_rgb_cls/checkpoint/overall-best.pt"
        ),
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
    parser.add_argument(
        "--dino-tta",
        choices=["none", "rf_spectral_response_v1", "ofdma_spectral_response_v1"],
        default=None,
    )
    parser.add_argument("--dino-tta-shift-px", type=int, default=4)
    parser.add_argument("--dino-tta-frequency-response-strength", type=float, default=3.0)
    parser.add_argument("--vit-coreset-ratio", type=float, default=0.5)
    parser.add_argument("--coreset-method", choices=["random", "farthest"], default="farthest")
    parser.add_argument("--dino-max-gallery-patches", type=int, default=0)
    parser.add_argument("--vit-nn-topk", type=int, default=5)
    parser.add_argument("--dino-nn-topk", type=int, default=1)
    parser.add_argument("--dino-top-ratio", type=float, default=0.1)
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--scene-ids", nargs="*", default=[])
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-normal-observations", type=int, default=0)
    parser.add_argument("--max-abnormal-observations-per-type", type=int, default=0)
    parser.add_argument("--ofdma-geometry", choices=["letterbox", "square_warp"], default="letterbox")
    parser.add_argument("--ofdma-subcarriers-per-rb", type=int, default=12)
    parser.add_argument("--vit-patch-score-rank", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dino_tta is None:
        args.dino_tta = (
            "ofdma_spectral_response_v1"
            if args.dataset == "ofdma"
            else "rf_spectral_response_v1"
        )
    if args.dataset == "inhouse":
        args.signals = list(args.signals or RF_TARGET_SIGNALS)
        args.scenes = list(args.scenes or RF_SCENES)
    elif args.dataset == "public":
        args.data_root = args.data_root or "/mnt/data/wangbei/data/RF_SPE_PNG"
        args.signals = list(args.signals or PUBLIC_SIGNALS)
        args.scenes = []
    else:
        args.data_root = args.data_root or "/mnt/data/wangbei/data/ofdma-target-scene-coldstart-v2-realistic"
        args.signals = ["all jammer types"]
        args.scenes = list(args.scene_ids)
    if not 0.0 < args.vit_coreset_ratio <= 1.0:
        raise ValueError("--vit-coreset-ratio must be in (0, 1]")
    if args.batch_size < 1 or args.distance_chunk_size < 1:
        raise ValueError("batch size and distance chunk size must be positive")
    if args.dino_max_gallery_patches < 0:
        raise ValueError("--dino-max-gallery-patches cannot be negative")
    setup_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda:0")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[vit+dino rf] dataset={args.dataset} device={device}", flush=True)

    if args.dataset == "inhouse":
        rows = run_inhouse(args, output_root, device)
    elif args.dataset == "public":
        rows = run_public(args, output_root, device)
    else:
        rows = run_ofdma(args, output_root, device)

    write_csv(output_root / "metrics.csv", rows)
    frame = pd.DataFrame(rows)
    macro = {}
    if not frame.empty:
        for method_key, group in frame.groupby("method_key"):
            macro[method_key] = {
                "method": METHOD_LABELS[method_key],
                "auroc": float(group["auroc"].mean()),
                "auprc": float(group["auprc"].mean()),
                "fpr95": float(group["fpr95"].mean()),
                "cells": int(len(group)),
            }
    summary = {
        "status": "complete",
        "method": "vit_dino_weighted_rf",
        "dataset": args.dataset,
        "protocol": "dataset-specific official support/test manifest; support-only DINO spectral TTA; original-view query",
        "checkpoint": args.checkpoint,
        "signals": args.signals,
        "scenes": args.scenes,
        "normal_sampling": args.normal_sampling,
        "support_manifest": args.support_manifest,
        "support_manifest_sha256": (
            json.loads(Path(args.support_manifest).read_text(encoding="utf-8")).get("manifest_sha256")
            if args.support_manifest and Path(args.support_manifest).exists() and args.dataset == "public"
            else None
        ),
        "backbones": {
            "vit": "PromptAD ViT-B-16-plus-240 layer1+layer2",
            "dino": args.dino_model,
        },
        "dino_tta": args.dino_tta,
        "dino_tta_modes": list(dino_tta_modes(args)),
        "dino_tta_shift_px": args.dino_tta_shift_px,
        "dino_tta_frequency_response_strength": args.dino_tta_frequency_response_strength,
        "memory": {
            "vit_coreset_ratio": args.vit_coreset_ratio,
            "vit_coreset_method": args.coreset_method,
            "dino_max_gallery_patches": args.dino_max_gallery_patches,
            "vit_nn_topk": args.vit_nn_topk,
            "dino_nn_topk": args.dino_nn_topk,
            "dino_top_ratio": args.dino_top_ratio,
        },
        "weighted_fusion": {
            "normalization": "bounded_distance",
            "formula": "w*clip((1-cos_vit)/2,0,1)+(1-w)*clip((1-cos_dino)/2,0,1)",
            "weights": {"vit25_dino75": [0.25, 0.75], "vit50_dino50": [0.5, 0.5], "vit75_dino25": [0.75, 0.25]},
        },
        "macro": macro,
    }
    # Keep the summary JSON simple and deterministic; the exact cell counts
    # and paths are already present in metrics.csv and per-cell NPZ files.
    summary["num_rows"] = int(len(rows))
    save_json(output_root / "summary.json", summary)
    save_json(
        output_root / "protocol.json",
        {
            **summary,
            "support_memory": "normal support only; no abnormal images, labels, or test-batch statistics enter the memory",
            "query": "original test image once",
            "distance_scale": "both branches mapped to bounded (1-cosine)/2 before weighted mean",
        },
    )
    (output_root / "README.md").write_text(
        "# ViT+DINOv2 weighted cross-dataset experiment\n\n"
        "This directory records the cross-dataset weighted fusion experiment.\n\n"
        f"- dataset: `{args.dataset}`\n"
        f"- DINO TTA: `{args.dino_tta}`; modes: `{list(dino_tta_modes(args))}`\n"
        f"- metrics: `{(output_root / 'metrics.csv').name}`\n"
        f"- summary: `{(output_root / 'summary.json').name}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
