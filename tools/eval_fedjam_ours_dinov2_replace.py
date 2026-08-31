#!/usr/bin/env python3
"""Compare current Ours with a strict ViT -> DINOv2 replacement on FedJam.

The control and candidate share the same benign support, TTA views, CNN
layer3 gallery, support-only confidence gate, and independent test split.  The
only changed branch is the visual memory branch:

* control: PromptAD ViT layer1+layer2 patch features;
* candidate: DINOv2 patch-token features.

The default ``matched_vit`` mode uses the same 50% farthest coreset, 5-NN
patch distance and maximum patch score for a strict feature-only swap.  The
``patch_top10`` mode keeps the previously validated DINO patch protocol
(full patch memory, 1-NN, top-10% mean) while leaving the CNN branch, TTA and
support-only gate unchanged.  DINOv2 is not trained and no abnormal samples
are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
from tools.eval_cls_vit_patchcore_gallery import (  # noqa: E402
    select_gallery_subset,
    topk_cosine_distance_chunked,
)
from tools.eval_fedjam_fewshot_dual import (  # noqa: E402
    CNN_REFERENCE_MODES,
    LABEL_NAMES,
    VIT_REFERENCE_MODES,
    _augment_bgr,
    add_metric_rows,
    augment_records,
    build_cnn_galleries,
    build_vit_galleries,
    build_vit_subspaces,
    build_vit_calibrators,
    iter_test_records,
    metric,
    raw_tensor,
    score_batch,
    select_benign_support,
    support_reference_scores,
    vit_features,
    vit_tta_modes,
    write_csv,
)
from tools.eval_fedjam_four_branches import build_dino, build_promptad  # noqa: E402
from tools.eval_seg_resnet_gallery_fusion import ResNet18LocalEncoder  # noqa: E402
from utils.confidence_gate import safe_support_only_gate  # noqa: E402
from utils.normal_subspace import leave_one_out_topk_cosine_distance  # noqa: E402
from utils.training_utils import setup_seed  # noqa: E402


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--distance-chunk-size", type=int, default=1024)
    parser.add_argument("--nn-topk", type=int, default=5)
    parser.add_argument("--coreset-ratio", type=float, default=0.5)
    parser.add_argument("--cnn-top-ratio", type=float, default=0.1)
    parser.add_argument("--cnn-nn-topk", type=int, default=1)
    parser.add_argument(
        "--vit-tta",
        choices=["none", "stft_shift_blur", "rf_spectral_response_v1"],
        default="rf_spectral_response_v1",
    )
    parser.add_argument("--shift-px", type=int, default=4)
    parser.add_argument("--vit-tta-frequency-response-strength", type=float, default=3.0)
    parser.add_argument(
        "--vit-tta-memory-layout",
        choices=["separate", "merged"],
        default="merged",
    )
    parser.add_argument("--vit-normal-model", choices=["memory", "pca", "hybrid"], default="memory")
    parser.add_argument("--pca-variance", type=float, default=0.99)
    parser.add_argument("--pca-max-components", type=int, default=256)
    parser.add_argument("--pca-max-fit-samples", type=int, default=8192)
    parser.add_argument("--dino-model", default="vit_base_patch14_dinov2")
    parser.add_argument("--dino-image-size", type=int, default=224)
    parser.add_argument(
        "--dino-score-protocol",
        choices=["matched_vit", "patch_top10"],
        default="matched_vit",
        help=(
            "matched_vit copies the ViT 5-NN/max-patch rule; patch_top10 uses "
            "the previously validated DINO 1-NN/top-10%% patch mean rule."
        ),
    )
    parser.add_argument("--dino-nn-topk", type=int, default=1)
    parser.add_argument("--dino-top-ratio", type=float, default=0.1)
    parser.add_argument(
        "--dino-coreset-ratio",
        type=float,
        default=None,
        help="DINO gallery coreset ratio; omitted means the ViT ratio, 1.0 keeps all DINO patches.",
    )
    parser.add_argument("--max-test-per-label", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument("--img-resize", type=int, default=240)
    parser.add_argument("--img-cropsize", type=int, default=240)
    return parser.parse_args()


def normalize_gallery(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=-1).contiguous()


def dino_coreset_ratio(args: argparse.Namespace) -> float:
    return args.coreset_ratio if args.dino_coreset_ratio is None else args.dino_coreset_ratio


def dino_score_parameters(args: argparse.Namespace) -> tuple[int, float | None]:
    if args.dino_score_protocol == "patch_top10":
        return args.dino_nn_topk, args.dino_top_ratio
    return args.nn_topk, None


def memory_description(ratio: float, nn_topk: int, aggregation: str) -> str:
    gallery = "full patch memory (no coreset)" if abs(ratio - 1.0) < 1e-12 else f"{ratio:g} farthest coreset"
    return f"{gallery}, {nn_topk}-NN, {aggregation}"


def augment_raw(raw: torch.Tensor, mode: str, args: argparse.Namespace) -> torch.Tensor:
    if mode == "identity":
        return raw
    images = [
        _augment_bgr(image.numpy(), mode, args)
        for image in raw
    ]
    return torch.from_numpy(np.stack(images, axis=0))


@torch.inference_mode()
def encode_dino(raw: torch.Tensor, model, transform, device: torch.device) -> torch.Tensor:
    inputs = to_dino_input(raw, transform, device, rgb_from_bgr=True)
    return extract_dino_patch_features(model, inputs).float().contiguous()


@torch.inference_mode()
def encode_dino_views(
    raw: torch.Tensor,
    model,
    transform,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if args.vit_tta_memory_layout == "merged":
        return {"identity": encode_dino(raw, model, transform, device)}
    return {
        mode: encode_dino(augment_raw(raw, mode, args), model, transform, device)
        for mode in vit_tta_modes(args)
    }


def build_dino_galleries(
    support_views: dict[str, torch.Tensor],
    support_count: int,
    shots: list[int],
    args: argparse.Namespace,
) -> dict[int, torch.Tensor | dict[str, torch.Tensor]]:
    galleries = {}
    for shot in shots:
        subset_args = SimpleNamespace(
            coreset_ratio=args.coreset_ratio,
            coreset_method="farthest",
            seed=args.seed + 1000 + shot,
        )
        if args.vit_tta_memory_layout == "merged":
            modes = vit_tta_modes(args)
            features = torch.cat(
                [
                    support_views[mode][:shot].reshape(-1, support_views[mode].shape[-1])
                    for mode in modes
                ],
                dim=0,
            )
            gallery = normalize_gallery(features)
            if dino_coreset_ratio(args) < 1.0:
                subset_args.coreset_ratio = dino_coreset_ratio(args)
                indices, _ = select_gallery_subset(gallery, subset_args)
                gallery = gallery[indices]
            galleries[shot] = gallery
            print(
                f"[dino_gallery] shot={shot} layout=merged "
                f"patches={gallery.shape[0]} dim={gallery.shape[1]}",
                flush=True,
            )
            continue

        galleries[shot] = {}
        for mode in vit_tta_modes(args):
            features = normalize_gallery(
                support_views[mode][:shot].reshape(-1, support_views[mode].shape[-1])
            )
            if dino_coreset_ratio(args) < 1.0:
                subset_args.coreset_ratio = dino_coreset_ratio(args)
                indices, _ = select_gallery_subset(features, subset_args)
                features = features[indices]
            galleries[shot][mode] = features
            print(
                f"[dino_gallery] shot={shot} mode={mode} "
                f"patches={features.shape[0]} dim={features.shape[1]}",
                flush=True,
            )
    return galleries


@torch.inference_mode()
def score_dino_features(
    features: dict[str, torch.Tensor],
    gallery: torch.Tensor | dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> np.ndarray:
    nn_topk, top_ratio = dino_score_parameters(args)
    if args.vit_tta_memory_layout == "merged":
        probes = features["identity"].reshape(-1, features["identity"].shape[-1])
        distances = topk_cosine_distance_chunked(
            probes, gallery, args.distance_chunk_size, nn_topk
        )
        distances = distances.reshape(features["identity"].shape[0], -1)
        if top_ratio is not None:
            keep = max(1, int(round(distances.shape[1] * top_ratio)))
            values = distances.topk(keep, dim=1).values.mean(dim=1)
        else:
            values = distances.amax(dim=1)
        return (
            values.cpu()
            .numpy()
            .astype(np.float32)
        )

    values = []
    for mode in vit_tta_modes(args):
        probes = features[mode].reshape(-1, features[mode].shape[-1])
        distances = topk_cosine_distance_chunked(
            probes, gallery[mode], args.distance_chunk_size, nn_topk
        )
        distances = distances.reshape(features[mode].shape[0], -1)
        if top_ratio is not None:
            keep = max(1, int(round(distances.shape[1] * top_ratio)))
            values.append(distances.topk(keep, dim=1).values.mean(dim=1))
        else:
            values.append(distances.amax(dim=1))
    return torch.stack(values, dim=0).amax(dim=0).cpu().numpy().astype(np.float32)


@torch.inference_mode()
def score_dino_batch(
    raw: torch.Tensor,
    model,
    transform,
    gallery,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    return score_dino_features(
        encode_dino_views(raw, model, transform, args, device), gallery, args
    )


def build_dino_references(
    support: list,
    shot: int,
    gallery,
    model,
    transform,
    args: argparse.Namespace,
    device: torch.device,
    support_identity: torch.Tensor,
) -> np.ndarray:
    selected = support[:shot]
    if args.vit_tta != "none":
        values = []
        for mode in VIT_REFERENCE_MODES:
            raw = augment_records(selected, mode, args)
            values.extend(score_dino_batch(raw, model, transform, gallery, args, device).tolist())
        return np.asarray(values, dtype=np.float32)

    dino_nn_topk, top_ratio = dino_score_parameters(args)
    loo = leave_one_out_topk_cosine_distance(
        support_identity[:shot],
        topk=dino_nn_topk,
        chunk_size=args.distance_chunk_size,
        distance_divisor=2.0,
    ).reshape(shot, -1)
    if top_ratio is None:
        values = loo.amax(dim=1)
    else:
        keep = max(1, int(round(loo.shape[1] * top_ratio)))
        values = loo.topk(keep, dim=1).values.mean(dim=1)
    return values.cpu().numpy().astype(np.float32)


def write_human_summary(output_root: Path, summary: dict) -> None:
    dino_rule = summary["branch_definitions"]["dinov2_memory"]
    lines = [
        "# Ours：ViT 替换为 DINOv2（FedJam）",
        "",
        "本实验只替换 Ours 的视觉记忆分支：PromptAD ViT layer1+layer2 改为 DINOv2 patch token。",
        "CNN layer3、正常样本记忆库、频谱 TTA、支持集门控、1/2/4-shot 划分和完整测试集均保持不变。",
        "",
        "| shot | 方法 | AUROC | AUPRC | FPR@95%TPR |",
        "|---:|---|---:|---:|---:|",
    ]
    for shot in summary["shots"]:
        for method in ("ours_vit_cnn_gate", "ours_dinov2_cnn_gate"):
            values = summary["scores"][str(shot)][method]
            lines.append(
                f"| {shot} | {summary['method_labels'][method]} | "
                f"{values['auroc']:.2f} | {values['auprc']:.2f} | {values['fpr95']:.2f} |"
            )
    lines += [
        "",
        f"说明：DINOv2 使用 {dino_rule}；",
        "候选方法再与未改变的 ResNet18 layer3 分支通过同一条 support-only confidence gate 融合。",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    shots = sorted(set(int(value) for value in args.shots))
    if shots != [1, 2, 4]:
        raise ValueError("This comparison requires shots 1, 2, and 4")
    if not 0.0 < args.coreset_ratio <= 1.0:
        raise ValueError("--coreset-ratio must be in (0, 1]")
    if args.dino_coreset_ratio is not None and not 0.0 < args.dino_coreset_ratio <= 1.0:
        raise ValueError("--dino-coreset-ratio must be in (0, 1]")
    if args.dino_nn_topk <= 0 or not 0.0 < args.dino_top_ratio <= 1.0:
        raise ValueError("DINO scoring parameters must be positive and top ratio must be in (0, 1]")
    if args.max_test_per_label < 0:
        raise ValueError("--max-test-per-label cannot be negative")

    setup_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if args.use_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
            raise ValueError(f"gpu-id {args.gpu_id} is unavailable; count={torch.cuda.device_count()}")
        device = torch.device(f"cuda:{args.gpu_id}")
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device} data_root={data_root}", flush=True)

    support, train_counts, benign_seen = select_benign_support(data_root, max_shot=4, seed=args.seed)
    save_json(
        output_root / "support_manifest.json",
        {
            "seed": args.seed,
            "shots": shots,
            "train_counts": train_counts,
            "benign_seen": benign_seen,
            "support": [
                {
                    "name": row.name,
                    "label": row.label,
                    "sha256": hashlib.sha256(row.image_bgr.tobytes()).hexdigest(),
                    "shape": list(row.image_bgr.shape),
                }
                for row in support
            ],
        },
    )

    # Build the exact current control model and the only replacement model.
    promptad_args = SimpleNamespace(checkpoint=args.checkpoint, image_size=args.img_resize)
    vit_model = build_promptad(promptad_args, device)
    dino_model = build_dino(
        SimpleNamespace(dino_model=args.dino_model, dino_image_size=args.dino_image_size),
        device,
    )
    dino_transform = build_dino_transform(args.dino_image_size)
    cnn_model = ResNet18LocalEncoder().to(device).eval()

    if hasattr(vit_model, "eval_mode"):
        vit_model.eval_mode()
    else:
        vit_model.eval()

    vit_galleries = build_vit_galleries(vit_model, support, shots, args, device)
    vit_subspaces = build_vit_subspaces(vit_model, support, shots, args, device)
    vit_calibrators = build_vit_calibrators(
        vit_model, support, shots, args, device, vit_galleries, vit_subspaces
    )
    cnn_galleries = build_cnn_galleries(cnn_model, support, shots, device)

    support_raw = raw_tensor(support)
    dino_support_views = {}
    with torch.inference_mode():
        for mode in vit_tta_modes(args):
            dino_support_views[mode] = encode_dino(
                augment_raw(support_raw, mode, args), dino_model, dino_transform, device
            )
    dino_galleries = build_dino_galleries(dino_support_views, len(support), shots, args)

    references = {}
    for shot in shots:
        vit_reference, cnn_reference = support_reference_scores(
            vit_model,
            cnn_model,
            support,
            shot,
            vit_galleries[shot],
            cnn_galleries[shot],
            args,
            device,
            vit_subspaces[shot],
            vit_calibrators[shot],
        )
        dino_reference = build_dino_references(
            support,
            shot,
            dino_galleries[shot],
            dino_model,
            dino_transform,
            args,
            device,
            dino_support_views["identity"],
        )
        references[shot] = {
            "vit": vit_reference,
            "dino": dino_reference,
            "cnn": cnn_reference,
        }
        print(
            f"[reference] shot={shot} vit={len(vit_reference)} "
            f"dino={len(dino_reference)} cnn={len(cnn_reference)}",
            flush=True,
        )

    score_bank = {
        shot: {
            "ours_vit_cnn_gate": [],
            "ours_dinov2_cnn_gate": [],
            "vit_only": [],
            "dinov2_only": [],
            "cnn_only": [],
            "vit_gate": [],
            "dino_gate": [],
        }
        for shot in shots
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
        # The existing Ours path computes ViT and CNN together.  DINO is
        # encoded once per test batch and reused for all three shot galleries.
        dino_features = encode_dino_views(raw, dino_model, dino_transform, args, device)
        for shot in shots:
            current = score_batch(
                vit_model,
                cnn_model,
                raw,
                vit_galleries[shot],
                cnn_galleries[shot],
                args,
                device,
                None,
                vit_subspaces[shot],
                vit_calibrators[shot],
            )
            dino_scores = score_dino_features(dino_features, dino_galleries[shot], args)
            vit_gate = safe_support_only_gate(
                current["vit_scores"],
                current["cnn_scores"],
                references[shot]["vit"],
                references[shot]["cnn"],
            )
            dino_gate = safe_support_only_gate(
                dino_scores,
                current["cnn_scores"],
                references[shot]["dino"],
                references[shot]["cnn"],
            )
            score_bank[shot]["ours_vit_cnn_gate"].extend(vit_gate["score"].astype(np.float32).tolist())
            score_bank[shot]["ours_dinov2_cnn_gate"].extend(dino_gate["score"].astype(np.float32).tolist())
            score_bank[shot]["vit_only"].extend(current["vit_scores"].tolist())
            score_bank[shot]["dinov2_only"].extend(dino_scores.tolist())
            score_bank[shot]["cnn_only"].extend(current["cnn_scores"].tolist())
            score_bank[shot]["vit_gate"].extend(vit_gate["gate"].astype(np.float32).tolist())
            score_bank[shot]["dino_gate"].extend(dino_gate["gate"].astype(np.float32).tolist())
        labels.extend(int(record.label) for record in records)
        names.extend(record.name for record in records)
        processed += len(records)
        if processed % max(100, args.batch_size * 100) < len(records):
            print(f"[test] processed={processed}", flush=True)
        del raw, dino_features

    for record in tqdm(
        iter_test_records(data_root, args.max_test_per_label),
        desc="Evaluate ViT replacement on FedJam",
        unit="row",
    ):
        pending.append(record)
        if len(pending) >= args.batch_size:
            flush(pending)
            pending = []
    flush(pending)

    labels_np = np.asarray(labels, dtype=np.int32)
    if labels_np.size == 0 or np.unique(labels_np).size < 2:
        raise RuntimeError("FedJam test must contain normal and abnormal labels")
    print(f"[test] completed={labels_np.size}", flush=True)

    method_labels = {
        "ours_vit_cnn_gate": "Ours (ViT+CNN, current)",
        "ours_dinov2_cnn_gate": "Ours (DINOv2+CNN, ViT replaced)",
    }
    rows = []
    summary = {
        "status": "complete",
        "method": "fedjam_ours_vit_vs_dinov2_replacement",
        "data_root": str(data_root),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "shots": shots,
        "test_count": int(labels_np.size),
        "test_label_counts": {
            str(label): int(np.sum(labels_np == label)) for label in sorted(np.unique(labels_np))
        },
        "test_batch_statistics_used": False,
        "support_only": True,
        "same_as_control": [
            "benign support selection",
            "normal-memory TTA bundle",
            "CNN ImageNet ResNet18 layer3 gallery",
            "support-only confidence gate",
            "nested 1/2/4-shot split",
            "complete independent test split",
        ],
        "changed_component": "PromptAD ViT layer1+layer2 patch features -> DINOv2 patch tokens",
        "branch_definitions": {
            "ours_vit_cnn_gate": "current ViT patch memory + unchanged CNN layer3 + support-only gate",
            "ours_dinov2_cnn_gate": "DINOv2 patch memory + unchanged CNN layer3 + same support-only gate",
            "vit_memory": memory_description(
                args.coreset_ratio, args.nn_topk, "maximum patch distance"
            ),
            "dinov2_memory": (
                memory_description(
                    dino_coreset_ratio(args), args.nn_topk, "maximum patch distance"
                )
                if args.dino_score_protocol == "matched_vit"
                else (
                    memory_description(
                        dino_coreset_ratio(args),
                        args.dino_nn_topk,
                        f"top {args.dino_top_ratio:g} patch distance mean",
                    )
                )
            ),
            "cnn_memory": "ResNet18 layer3, full support patch memory, top 10% patch distance",
        },
        "backbones": {
            "control_vit": "PromptAD ViT-B-16-plus-240, normalized layer1+layer2 concatenation",
            "candidate_dino": args.dino_model,
            "cnn": "ImageNet ResNet18 layer3",
        },
        "vit_tta": args.vit_tta,
        "vit_tta_modes": list(vit_tta_modes(args)),
        "vit_tta_memory_layout": args.vit_tta_memory_layout,
        "shift_px": args.shift_px,
        "frequency_response_strength": args.vit_tta_frequency_response_strength,
        "coreset_ratio": args.coreset_ratio,
        "nn_topk": args.nn_topk,
        "dino_score_protocol": args.dino_score_protocol,
        "dino_nn_topk": args.dino_nn_topk,
        "dino_top_ratio": args.dino_top_ratio,
        "dino_coreset_ratio": dino_coreset_ratio(args),
        "cnn_top_ratio": args.cnn_top_ratio,
        "cnn_nn_topk": args.cnn_nn_topk,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "method_labels": method_labels,
        "scores": {},
    }
    score_dir = output_root / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    for shot in shots:
        arrays = {key: np.asarray(value, dtype=np.float32) for key, value in score_bank[shot].items()}
        np.savez_compressed(
            score_dir / f"fedjam_{shot}shot_ours_vit_vs_dinov2.npz",
            labels=labels_np,
            names=np.asarray(names),
            **arrays,
        )
        summary["scores"][str(shot)] = {}
        for method in method_labels:
            add_metric_rows(rows, shot, method_labels[method], labels_np, arrays[method])
            summary["scores"][str(shot)][method] = metric(
                (labels_np != 0).astype(np.int32), arrays[method]
            )
        summary["scores"][str(shot)]["vit_only"] = metric(
            (labels_np != 0).astype(np.int32), arrays["vit_only"]
        )
        summary["scores"][str(shot)]["dinov2_only"] = metric(
            (labels_np != 0).astype(np.int32), arrays["dinov2_only"]
        )
        summary["scores"][str(shot)]["cnn_only"] = metric(
            (labels_np != 0).astype(np.int32), arrays["cnn_only"]
        )
        summary["scores"][str(shot)]["gate_active_rate_vit"] = float(np.mean(arrays["vit_gate"] > 0.0))
        summary["scores"][str(shot)]["gate_active_rate_dinov2"] = float(np.mean(arrays["dino_gate"] > 0.0))

    write_csv(output_root / "metrics.csv", rows)
    save_json(output_root / "summary.json", summary)
    save_json(
        output_root / "protocol.json",
        {
            **summary,
            "support_selection": "seeded reservoir over train label=benign; nested prefix for shots",
            "test_selection": "all official test rows unless max-test-per-label is set",
            "memory": "normal support only; no abnormal memory; no training",
            "fusion_formula": "same frozen support-only gate for current and replacement",
        },
    )
    write_human_summary(output_root, summary)
    if device.type == "cuda":
        print(f"[gpu] max_allocated_mb={torch.cuda.max_memory_allocated(device) / 1024**2:.1f}", flush=True)
    print(f"wrote {output_root / 'metrics.csv'}", flush=True)
    print(f"wrote {output_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
