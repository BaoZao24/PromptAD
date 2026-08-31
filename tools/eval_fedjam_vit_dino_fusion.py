#!/usr/bin/env python3
"""Evaluate ViT + DINOv2 local-memory fusion on the formal FedJam protocol.

This is a controlled replacement of the current ViT+CNN branch:

* the PromptAD ViT layer1+layer2 branch is unchanged;
* the ResNet18 layer3 branch is replaced by DINOv2 patch tokens;
* DINO uses a benign-only patch memory and the CNN branch's top-10% image
  aggregation, rather than UniVAD's score fusion;
* the frozen support-only confidence gate is reused with DINO as the local
  branch;
* DINO can use the fixed spectral-response TTA bundle on normal support only,
  while the test image remains an original-view query.

The default run remains identity-only so the previous control is reproducible;
pass ``--dino-tta rf_spectral_response_v1`` for the TTA candidate.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_cls_dinov2_patchcore_gallery import (  # noqa: E402
    build_transform as build_dino_transform,
    extract_patch_features as extract_dino_patch_features,
    to_dino_input,
)
from tools.eval_cls_vit_patchcore_gallery import (  # noqa: E402
    prepare_patch_features,
    select_gallery_subset,
    topk_cosine_distance_chunked,
)
from tools.eval_fedjam_fewshot_dual import (  # noqa: E402
    LABEL_NAMES,
    add_metric_rows,
    iter_test_records,
    metric,
    raw_tensor,
    select_benign_support,
    write_csv,
)
from tools.eval_fedjam_four_branches import build_dino, build_promptad  # noqa: E402
from train_rf_target_pooled_universal import to_model_input  # noqa: E402
from utils.confidence_gate import safe_support_only_gate  # noqa: E402
from utils.normal_subspace import leave_one_out_topk_cosine_distance  # noqa: E402
from utils.spectral_tta import SPECTRAL_TTA_BUNDLES, augment_spectrogram  # noqa: E402
from utils.training_utils import setup_seed  # noqa: E402


BRANCH_LABELS = {
    "vit_only": "ViT-only",
    "dino_only": "DINO-only",
    "vit_dino_gate": "ViT+DINO Confidence Fusion",
}


def weighted_method_key(vit_weight: float) -> str:
    return f"vit_dino_weighted_vit{int(round(float(vit_weight) * 100)):02d}"


def weighted_method_label(vit_weight: float) -> str:
    return (
        "ViT+DINO weighted mean "
        f"(ViT={float(vit_weight):.2f}, DINO={1.0 - float(vit_weight):.2f})"
    )


def weighted_fusion_scores(
    vit_scores: np.ndarray,
    dino_scores: np.ndarray,
    gated: dict[str, np.ndarray],
    args: argparse.Namespace,
    vit_weight: float,
) -> np.ndarray:
    """Return a fixed weighted mean without reading query-batch statistics."""

    if args.weighted_normalization == "bounded_distance":
        # ViT uses (1-cosine)/2; DINO is converted to 1-cosine in score_dino.
        # Both are therefore put on the known [0, 1] distance scale.
        vit_evidence = np.clip(np.asarray(vit_scores, dtype=np.float64), 0.0, 1.0)
        dino_evidence = np.clip(np.asarray(dino_scores, dtype=np.float64) / 2.0, 0.0, 1.0)
    else:
        vit_evidence = np.asarray(gated["vit_probability"], dtype=np.float64)
        dino_evidence = np.asarray(gated["cnn_probability"], dtype=np.float64)
    return (
        float(vit_weight) * vit_evidence
        + (1.0 - float(vit_weight)) * dino_evidence
    ).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/mnt/data/wangbei/data/FedJam")
    parser.add_argument(
        "--checkpoint",
        default=(
            "analysis_outputs/02_current_baselines/promptad_formal_baseline/"
            "pooled_rf_rgb_cls/checkpoint/overall-best.pt"
        ),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--distance-chunk-size", type=int, default=1024)
    parser.add_argument("--coreset-ratio", type=float, default=0.5)
    parser.add_argument("--vit-nn-topk", type=int, default=5)
    parser.add_argument("--dino-nn-topk", type=int, default=1)
    parser.add_argument("--dino-top-ratio", type=float, default=0.1)
    parser.add_argument("--dino-model", default="vit_base_patch14_dinov2")
    parser.add_argument("--dino-image-size", type=int, default=224)
    parser.add_argument(
        "--dino-tta",
        choices=["none", "rf_spectral_response_v1"],
        default="none",
        help=(
            "DINO normal-memory views. rf_spectral_response_v1 merges identity, "
            "time-axis +/-4 px, and frequency-response jitter; query remains original."
        ),
    )
    parser.add_argument("--dino-tta-shift-px", type=int, default=4)
    parser.add_argument("--dino-tta-frequency-response-strength", type=float, default=3.0)
    parser.add_argument(
        "--vit-weight-grid",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75],
        help="ViT weights for the weighted means; DINO gets 1-weight.",
    )
    parser.add_argument(
        "--weighted-normalization",
        choices=["bounded_distance", "support_sigmoid"],
        default="bounded_distance",
        help=(
            "bounded_distance uses ViT [0,1] and DINO (1-cosine)/2; "
            "support_sigmoid is retained for comparison but is degenerate at 1-shot."
        ),
    )
    parser.add_argument("--image-size", type=int, default=240)
    parser.add_argument("--max-test-per-label", type=int, default=0)
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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

    dino = encode_dino_batch(dino_model, dino_transform, raw, device)
    return {"vit": vit, "dino": dino}


def normalize_gallery(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=-1).contiguous()


def dino_tta_modes(args: argparse.Namespace) -> tuple[str, ...]:
    if args.dino_tta == "none":
        return ("identity",)
    return tuple(SPECTRAL_TTA_BUNDLES[args.dino_tta])


def augment_dino_raw(raw: torch.Tensor, mode: str, args: argparse.Namespace) -> torch.Tensor:
    """Apply one axis-aware spectral view to a CPU BGR batch."""

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
def encode_dino_batch(
    dino_model,
    dino_transform,
    raw: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    dino_input = to_dino_input(raw, dino_transform, device, rgb_from_bgr=True)
    return extract_dino_patch_features(dino_model, dino_input).float().contiguous()


def build_galleries(
    support_features: dict[str, torch.Tensor],
    dino_view_features: dict[str, torch.Tensor],
    shots: list[int],
    args: argparse.Namespace,
) -> dict[int, dict[str, torch.Tensor]]:
    galleries: dict[int, dict[str, torch.Tensor]] = {}
    for shot in shots:
        vit = normalize_gallery(
            support_features["vit"][:shot].reshape(-1, support_features["vit"].shape[-1])
        )
        if args.coreset_ratio < 1.0:
            subset_args = SimpleNamespace(
                coreset_ratio=args.coreset_ratio,
                coreset_method="farthest",
                seed=args.seed + shot,
            )
            indices, _ = select_gallery_subset(vit, subset_args)
            vit = vit[indices]

        dino_parts = [
            dino_view_features[mode][:shot].reshape(-1, dino_view_features[mode].shape[-1])
            for mode in dino_tta_modes(args)
        ]
        dino = normalize_gallery(torch.cat(dino_parts, dim=0))
        galleries[shot] = {"vit": vit, "dino": dino}
        print(
            f"[gallery] shot={shot} vit_patches={vit.shape[0]} "
            f"dino_patches={dino.shape[0]} dino_dim={dino.shape[1]}",
            flush=True,
        )
    return galleries


@torch.inference_mode()
def score_vit(
    features: torch.Tensor,
    gallery: torch.Tensor,
    args: argparse.Namespace,
) -> np.ndarray:
    distances = topk_cosine_distance_chunked(
        features.reshape(-1, features.shape[-1]),
        gallery,
        args.distance_chunk_size,
        args.vit_nn_topk,
    )
    return (
        distances.reshape(features.shape[0], -1)
        .amax(dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


@torch.inference_mode()
def score_dino(
    features: torch.Tensor,
    gallery: torch.Tensor,
    args: argparse.Namespace,
) -> np.ndarray:
    distances = topk_cosine_distance_chunked(
        features.reshape(-1, features.shape[-1]),
        gallery,
        args.distance_chunk_size,
        args.dino_nn_topk,
    ).reshape(features.shape[0], -1)
    keep = max(1, int(round(distances.shape[1] * args.dino_top_ratio)))
    # topk_cosine_distance_chunked uses (1-cosine)/2.  Convert to the same
    # raw 1-cosine scale used by the old ResNet local branch.
    return (
        distances.topk(keep, dim=1).values.mean(dim=1)
        .mul(2.0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


@torch.inference_mode()
def build_support_references(
    support_features: dict[str, torch.Tensor],
    dino_view_features: dict[str, torch.Tensor],
    shots: list[int],
    args: argparse.Namespace,
) -> dict[int, dict[str, np.ndarray]]:
    references: dict[int, dict[str, np.ndarray]] = {}
    for shot in shots:
        vit_loo = leave_one_out_topk_cosine_distance(
            support_features["vit"][:shot],
            topk=args.vit_nn_topk,
            chunk_size=args.distance_chunk_size,
            distance_divisor=2.0,
        ).reshape(shot, -1)
        vit_reference = vit_loo.max(dim=1).values.cpu().numpy().astype(np.float32)

        if args.dino_tta == "none":
            dino_loo = leave_one_out_topk_cosine_distance(
                support_features["dino"][:shot],
                topk=args.dino_nn_topk,
                chunk_size=args.distance_chunk_size,
                distance_divisor=1.0,
            ).reshape(shot, -1)
            keep = max(1, int(round(dino_loo.shape[1] * args.dino_top_ratio)))
            dino_reference = (
                dino_loo.topk(keep, dim=1).values.mean(dim=1).cpu().numpy().astype(np.float32)
            )
        else:
            # Remove only the exact identity patches of the support image
            # being calibrated. Its transformed normal views remain in the
            # gallery, so the reference is still support-only and compatible
            # with the merged TTA memory used at test time.
            patch_count = int(support_features["dino"].shape[1])
            reference_values = []
            modes = dino_tta_modes(args)
            for index in range(shot):
                gallery_parts = []
                for mode in modes:
                    view = dino_view_features[mode][:shot].reshape(
                        -1, dino_view_features[mode].shape[-1]
                    )
                    if mode == "identity":
                        keep_mask = torch.ones(
                            view.shape[0], dtype=torch.bool, device=view.device
                        )
                        start = index * patch_count
                        keep_mask[start : start + patch_count] = False
                        view = view[keep_mask]
                    gallery_parts.append(view)
                reference_gallery = normalize_gallery(torch.cat(gallery_parts, dim=0))
                reference_values.append(
                    float(
                        score_dino(
                            support_features["dino"][index : index + 1],
                            reference_gallery,
                            args,
                        )[0]
                    )
                )
            dino_reference = np.asarray(reference_values, dtype=np.float32)
        references[shot] = {"vit": vit_reference, "dino": dino_reference}
        print(
            f"[reference] shot={shot} vit={len(vit_reference)} dino={len(dino_reference)}",
            flush=True,
        )
    return references


def main() -> None:
    args = parse_args()
    shots = sorted(set(int(value) for value in args.shots))
    if shots != [1, 2, 4]:
        raise ValueError("This comparison requires shots 1, 2, and 4")
    if args.batch_size < 1 or args.distance_chunk_size < 1:
        raise ValueError("batch and distance chunk sizes must be positive")
    if not 0.0 < args.coreset_ratio <= 1.0:
        raise ValueError("--coreset-ratio must be in (0, 1]")
    if not 0.0 < args.dino_top_ratio <= 1.0:
        raise ValueError("--dino-top-ratio must be in (0, 1]")
    if args.dino_tta_shift_px < 0:
        raise ValueError("--dino-tta-shift-px cannot be negative")
    if args.dino_tta_frequency_response_strength < 0.0:
        raise ValueError("--dino-tta-frequency-response-strength cannot be negative")
    if not args.vit_weight_grid or any(
        not 0.0 <= float(weight) <= 1.0 for weight in args.vit_weight_grid
    ):
        raise ValueError("--vit-weight-grid values must be in [0, 1]")
    if args.max_test_per_label < 0:
        raise ValueError("--max-test-per-label cannot be negative")

    setup_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device(
        "cuda:0" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[vit+dino] device={device} data_root={data_root}", flush=True)

    support, train_counts, benign_seen = select_benign_support(
        data_root, max_shot=4, seed=args.seed
    )
    save_json(
        output_root / "support_manifest.json",
        {
            "seed": args.seed,
            "shots": shots,
            "train_counts": train_counts,
            "benign_seen": benign_seen,
            "support": [
                {"name": row.name, "label": row.label, "shape": list(row.image_bgr.shape)}
                for row in support
            ],
        },
    )

    model = build_promptad(
        SimpleNamespace(checkpoint=args.checkpoint, image_size=args.image_size), device
    )
    dino_model = build_dino(
        SimpleNamespace(
            dino_model=args.dino_model,
            dino_image_size=args.dino_image_size,
        ),
        device,
    )
    dino_transform = build_dino_transform(args.dino_image_size)

    support_raw = raw_tensor(support)
    support_features = encode_batch(model, dino_model, dino_transform, support_raw, device)
    dino_view_features = {}
    for mode in dino_tta_modes(args):
        if mode == "identity":
            dino_view_features[mode] = support_features["dino"]
            continue
        view_raw = augment_dino_raw(support_raw, mode, args)
        dino_view_features[mode] = encode_dino_batch(
            dino_model,
            dino_transform,
            view_raw,
            device,
        )
        del view_raw
    galleries = build_galleries(support_features, dino_view_features, shots, args)
    references = build_support_references(support_features, dino_view_features, shots, args)
    print(
        f"[dino-tta] modes={list(dino_tta_modes(args))} "
        "memory_layout=merged_support_views_original_query",
        flush=True,
    )
    del support_raw, support_features, support
    gc.collect()

    score_bank = {}
    for shot in shots:
        score_bank[shot] = {
            "vit_only": [],
            "dino_only": [],
            "vit_dino_gate": [],
            "dino_gate": [],
            "dino_rank": [],
            **{weighted_method_key(weight): [] for weight in args.vit_weight_grid},
        }
    labels: list[int] = []
    names: list[str] = []
    pending = []
    processed = 0

    def flush(records: list) -> None:
        nonlocal processed
        if not records:
            return
        raw = raw_tensor(records)
        features = encode_batch(model, dino_model, dino_transform, raw, device)
        for shot in shots:
            vit_scores = score_vit(features["vit"], galleries[shot]["vit"], args)
            dino_scores = score_dino(features["dino"], galleries[shot]["dino"], args)
            gated = safe_support_only_gate(
                vit_scores,
                dino_scores,
                references[shot]["vit"],
                references[shot]["dino"],
            )
            score_bank[shot]["vit_only"].extend(vit_scores.tolist())
            score_bank[shot]["dino_only"].extend(dino_scores.tolist())
            score_bank[shot]["vit_dino_gate"].extend(gated["score"].astype(np.float32).tolist())
            score_bank[shot]["dino_gate"].extend(gated["gate"].astype(np.float32).tolist())
            score_bank[shot]["dino_rank"].extend(gated["cnn_rank"].astype(np.float32).tolist())
            for weight in args.vit_weight_grid:
                key = weighted_method_key(weight)
                weighted = weighted_fusion_scores(
                    vit_scores,
                    dino_scores,
                    gated,
                    args,
                    weight,
                )
                score_bank[shot][key].extend(weighted.tolist())
        labels.extend(int(record.label) for record in records)
        names.extend(record.name for record in records)
        processed += len(records)
        if processed % max(100, args.batch_size * 100) < len(records):
            print(f"[test] processed={processed}", flush=True)
        del raw, features

    for record in iter_test_records(data_root, args.max_test_per_label):
        pending.append(record)
        if len(pending) >= args.batch_size:
            flush(pending)
            pending = []
    flush(pending)

    labels_np = np.asarray(labels, dtype=np.int32)
    if labels_np.size == 0 or np.unique(labels_np).size < 2:
        raise RuntimeError("FedJam test must contain normal and abnormal labels")
    print(f"[test] completed={labels_np.size}", flush=True)

    rows: list[dict] = []
    summary = {
        "status": "complete",
        "method": "fedjam_vit_dino_support_only_fusion",
        "protocol": (
            "benign-only nested 1/2/4-shot; full independent test; spectrogram only; "
            f"DINO TTA={args.dino_tta}"
        ),
        "data_root": str(data_root),
        "test_count": int(labels_np.size),
        "test_label_counts": {
            LABEL_NAMES[int(label)]: int(np.sum(labels_np == int(label)))
            for label in sorted(np.unique(labels_np).tolist())
        },
        "branch_definitions": {
            "vit_only": "PromptAD ViT layer1+layer2; 50% farthest coreset; 5-NN; max patch distance",
            "dino_only": "DINOv2 patch memory; 1-NN; top 10% patch distance mean",
            "vit_dino_gate": "frozen support-only gate; DINO replaces the ResNet18 local branch",
        },
        "backbone": {
            "vit": "PromptAD ViT-B-16-plus-240",
            "dino": args.dino_model,
        },
        "seed": args.seed,
        "batch_size": args.batch_size,
        "coreset_ratio": args.coreset_ratio,
        "vit_nn_topk": args.vit_nn_topk,
        "dino_nn_topk": args.dino_nn_topk,
        "dino_top_ratio": args.dino_top_ratio,
        "dino_tta": args.dino_tta,
        "dino_tta_modes": list(dino_tta_modes(args)),
        "dino_tta_shift_px": args.dino_tta_shift_px,
        "dino_tta_frequency_response_strength": args.dino_tta_frequency_response_strength,
        "dino_memory_layout": "merged_support_views_original_query",
        "weighted_fusion": {
            "normalization": args.weighted_normalization,
            "formula": "w_vit * normalized_vit + (1 - w_vit) * normalized_dino",
            "vit_weights": [float(weight) for weight in args.vit_weight_grid],
        },
        "scores": {},
    }
    score_dir = output_root / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    for shot in shots:
        arrays = {
            key: np.asarray(values, dtype=np.float32)
            for key, values in score_bank[shot].items()
        }
        np.savez_compressed(
            score_dir / f"fedjam_{shot}shot_vit_dino_scores.npz",
            labels=labels_np,
            names=np.asarray(names),
            **arrays,
        )
        summary["scores"][str(shot)] = {}
        for method_key in ("vit_only", "dino_only", "vit_dino_gate"):
            add_metric_rows(rows, shot, BRANCH_LABELS[method_key], labels_np, arrays[method_key])
            summary["scores"][str(shot)][method_key] = metric(
                (labels_np != 0).astype(np.int32), arrays[method_key]
            )
        for weight in args.vit_weight_grid:
            method_key = weighted_method_key(weight)
            add_metric_rows(
                rows,
                shot,
                weighted_method_label(weight),
                labels_np,
                arrays[method_key],
            )
            summary["scores"][str(shot)][method_key] = metric(
                (labels_np != 0).astype(np.int32), arrays[method_key]
            )
        summary["scores"][str(shot)]["gate_active_rate"] = float(
            np.mean(arrays["dino_gate"] > 0.0)
        )

    write_csv(output_root / "metrics.csv", rows)
    save_json(
        output_root / "protocol.json",
        {
            **summary,
            "support_selection": "seeded reservoir over train label=benign; nested prefix for shots",
            "test_selection": "all official test rows unless max-test-per-label is set",
            "memory": "normal support only; no abnormal memory; no training",
            "distance_scale": "ViT uses (1-cosine)/2; DINO is converted to 1-cosine before gating",
        },
    )
    save_json(output_root / "summary.json", summary)
    print(f"wrote {output_root / 'metrics.csv'}", flush=True)
    print(f"wrote {output_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
