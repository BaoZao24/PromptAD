#!/usr/bin/env python
"""Evaluate four independent visual branches on the formal FedJam protocol.

Branches
--------
* ``vit_local``: current PromptAD CLIP ViT local patch features;
* ``vit_global``: the same CLIP ViT global/CLS image feature;
* ``cnn_local``: frozen ImageNet ResNet18 layer3 patch features;
* ``dino_local``: frozen DINOv2-B patch features.

The CLIP and DINO local branches use the planned full-support-memory and
1-nearest-neighbour rule.  Every branch is scored independently; no fusion
or test-batch calibration is performed in this experiment.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PromptAD import PromptAD  # noqa: E402
from tools.eval_cls_dinov2_patchcore_gallery import (  # noqa: E402
    build_transform as build_dino_transform,
    extract_patch_features as extract_dino_patch_features,
    to_dino_input,
)
from tools.eval_cls_resnet_gallery_fusion import resnet_patch_image_scores  # noqa: E402
from tools.eval_cls_vit_patchcore_gallery import (  # noqa: E402
    min_cosine_distance_chunked,
    prepare_patch_features,
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
from tools.eval_seg_resnet_gallery_fusion import (  # noqa: E402
    ResNet18LocalEncoder,
    load_checkpoint,
)
from train_rf_target_pooled_universal import to_model_input  # noqa: E402
from utils.training_utils import setup_seed  # noqa: E402


BRANCHES = ("vit_local", "vit_global", "cnn_local", "dino_local")
BRANCH_LABELS = {
    "vit_local": "ViT-local",
    "vit_global": "ViT-global",
    "cnn_local": "CNN-local",
    "dino_local": "DINO-local",
}


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
    parser.add_argument("--dino-model", default="vit_base_patch14_dinov2")
    parser.add_argument("--dino-image-size", type=int, default=224)
    parser.add_argument("--max-test-per-label", type=int, default=0)
    parser.add_argument("--cnn-top-ratio", type=float, default=0.1)
    parser.add_argument("--cnn-nn-topk", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=240)
    return parser.parse_args()


def build_promptad(args: argparse.Namespace, device: torch.device) -> PromptAD:
    model = PromptAD(
        dataset="rf_target_test_pool",
        class_name="signal",
        device=str(device),
        out_size_h=400,
        out_size_w=400,
        backbone="ViT-B-16-plus-240",
        pretrained_dataset="laion400m_e32",
        n_ctx=4,
        n_pro=3,
        n_ctx_ab=1,
        n_pro_ab=4,
        precision="fp16",
        k_shot=1,
        img_resize=args.image_size,
        img_cropsize=args.image_size,
        prompt_mode="rf",
        input_mode="rgb",
        text_prototype_mode="single",
        cls_score_mode="text_only",
    ).to(device)
    load_checkpoint(model, str(Path(args.checkpoint).resolve()))
    model.eval_mode()
    return model


def build_dino(args: argparse.Namespace, device: torch.device):
    model = timm.create_model(
        args.dino_model,
        pretrained=True,
        num_classes=0,
        img_size=args.dino_image_size,
        dynamic_img_size=True,
    )
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def encode_batch(
    clip_model: PromptAD,
    cnn_model: ResNet18LocalEncoder,
    dino_model,
    dino_transform,
    raw: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    clip_input = to_model_input(clip_model, raw, device, rgb_from_bgr=True)
    visual = clip_model.encode_image(clip_input)
    vit_local = prepare_patch_features(visual).float().contiguous()
    vit_global = F.normalize(visual[0].float(), dim=-1).contiguous()

    cnn_input = raw.permute(0, 3, 1, 2).to(device, non_blocking=True)
    cnn_layer3 = cnn_model(cnn_input)["layer3"].float().contiguous()

    dino_input = to_dino_input(raw, dino_transform, device, rgb_from_bgr=True)
    dino_local = extract_dino_patch_features(dino_model, dino_input).float().contiguous()
    return {
        "vit_local": vit_local,
        "vit_global": vit_global,
        "cnn_local": cnn_layer3,
        "dino_local": dino_local,
    }


def normalize_gallery(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=-1).contiguous()


def build_galleries(
    support_features: dict[str, torch.Tensor],
    shots: list[int],
) -> dict[int, dict[str, torch.Tensor]]:
    galleries: dict[int, dict[str, torch.Tensor]] = {}
    for shot in shots:
        galleries[shot] = {
            "vit_local": normalize_gallery(
                support_features["vit_local"][:shot].reshape(-1, support_features["vit_local"].shape[-1])
            ),
            "vit_global": normalize_gallery(support_features["vit_global"][:shot]),
            "cnn_local": normalize_gallery(
                support_features["cnn_local"][:shot]
                .permute(0, 2, 3, 1)
                .reshape(-1, support_features["cnn_local"].shape[1])
            ),
            "dino_local": normalize_gallery(
                support_features["dino_local"][:shot].reshape(-1, support_features["dino_local"].shape[-1])
            ),
        }
    return galleries


@torch.inference_mode()
def score_branch_batch(
    branch: str,
    features: dict[str, torch.Tensor],
    gallery: torch.Tensor,
    args: argparse.Namespace,
) -> np.ndarray:
    if branch == "vit_global":
        values = min_cosine_distance_chunked(
            normalize_gallery(features[branch]),
            gallery,
            args.distance_chunk_size,
        )
        return values.detach().cpu().numpy().astype(np.float32)

    if branch == "cnn_local":
        return resnet_patch_image_scores(
            features[branch],
            gallery,
            args.distance_chunk_size,
            args.cnn_top_ratio,
            args.cnn_nn_topk,
        ).astype(np.float32)

    patch_features = features[branch]
    distances = topk_cosine_distance_chunked(
        patch_features.reshape(-1, patch_features.shape[-1]),
        gallery,
        args.distance_chunk_size,
        1,
    )
    return distances.reshape(patch_features.shape[0], -1).amax(dim=1).detach().cpu().numpy().astype(np.float32)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    shots = sorted(set(int(value) for value in args.shots))
    if shots != [1, 2, 4]:
        raise ValueError("This formal branch comparison requires shots 1, 2, and 4")
    if args.max_test_per_label < 0:
        raise ValueError("--max-test-per-label cannot be negative")

    setup_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda:0" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[four-branches] device={device} data_root={data_root}", flush=True)

    support, train_counts, benign_seen = select_benign_support(data_root, max_shot=4, seed=args.seed)
    save_json(
        output_root / "support_manifest.json",
        {
            "seed": args.seed,
            "shots": shots,
            "train_counts": train_counts,
            "benign_seen": benign_seen,
            "support": [
                {"name": record.name, "label": record.label, "shape": list(record.image_bgr.shape)}
                for record in support
            ],
        },
    )

    clip_model = build_promptad(args, device)
    cnn_model = ResNet18LocalEncoder().to(device).eval()
    dino_model = build_dino(args, device)
    dino_transform = build_dino_transform(args.dino_image_size)

    support_raw = raw_tensor(support)
    with torch.inference_mode():
        support_features = encode_batch(
            clip_model, cnn_model, dino_model, dino_transform, support_raw, device
        )
    galleries = build_galleries(support_features, shots)
    del support_raw, support_features
    gc.collect()

    score_bank = {
        shot: {branch: [] for branch in BRANCHES}
        for shot in shots
    }
    labels_multiclass: list[int] = []
    names: list[str] = []
    pending = []
    processed = 0

    def flush(records: list) -> None:
        nonlocal processed
        if not records:
            return
        raw = raw_tensor(records)
        with torch.inference_mode():
            features = encode_batch(
                clip_model, cnn_model, dino_model, dino_transform, raw, device
            )
            for shot in shots:
                for branch in BRANCHES:
                    values = score_branch_batch(branch, features, galleries[shot][branch], args)
                    score_bank[shot][branch].extend(float(value) for value in values)
        labels_multiclass.extend(int(record.label) for record in records)
        names.extend(record.name for record in records)
        processed += len(records)
        if processed % max(args.batch_size * 100, 100) < len(records):
            print(f"[test] processed={processed}", flush=True)
        del raw, features

    for record in iter_test_records(data_root, args.max_test_per_label):
        pending.append(record)
        if len(pending) >= args.batch_size:
            flush(pending)
            pending = []
    flush(pending)

    labels_np = np.asarray(labels_multiclass, dtype=np.int32)
    if labels_np.size == 0 or np.unique(labels_np).size < 2:
        raise RuntimeError("FedJam test must contain normal and abnormal labels")
    print(f"[test] completed={labels_np.size}", flush=True)

    rows: list[dict] = []
    summary: dict = {
        "status": "complete",
        "method": "fedjam_four_independent_branches",
        "protocol": "benign-only nested 1/2/4-shot; full independent test; spectrogram only",
        "data_root": str(data_root),
        "test_count": int(labels_np.size),
        "test_label_counts": {
            LABEL_NAMES[int(label)]: int(np.sum(labels_np == int(label)))
            for label in sorted(np.unique(labels_np).tolist())
        },
        "branch_definitions": {
            "vit_local": "PromptAD CLIP ViT-B local layer1+layer2; all support patches; 1-NN; max patch distance",
            "vit_global": "PromptAD CLIP ViT-B global/CLS; nearest normal support image distance",
            "cnn_local": "ImageNet ResNet18 layer3; full normal patch gallery; top 10% patch distance",
            "dino_local": f"DINOv2 {args.dino_model}; all support patches; 1-NN; max patch distance",
        },
        "seed": args.seed,
        "batch_size": args.batch_size,
        "dino_model": args.dino_model,
        "dino_image_size": args.dino_image_size,
        "scores": {},
    }
    score_dir = output_root / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    for shot in shots:
        arrays = {branch: np.asarray(score_bank[shot][branch], dtype=np.float32) for branch in BRANCHES}
        np.savez_compressed(
            score_dir / f"fedjam_{shot}shot_four_branch_scores.npz",
            labels=labels_np,
            names=np.asarray(names),
            **arrays,
        )
        summary["scores"][str(shot)] = {}
        for branch in BRANCHES:
            method = BRANCH_LABELS[branch]
            add_metric_rows(rows, shot, method, labels_np, arrays[branch])
            summary["scores"][str(shot)][branch] = metric(
                (labels_np != 0).astype(np.int32), arrays[branch]
            )

    write_csv(output_root / "metrics.csv", rows)
    save_json(output_root / "summary.json", summary)
    save_json(
        output_root / "protocol.json",
        {
            **summary,
            "support_selection": "seeded reservoir over train label=benign; nested prefix for shots",
            "test_selection": "all official test rows unless max-test-per-label is set",
            "fusion": "none; each branch is reported independently",
        },
    )
    print(f"wrote {output_root / 'metrics.csv'}", flush=True)
    print(f"wrote {output_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
