#!/usr/bin/env python3
"""Evaluate a frozen ConvNeXt-B local patch branch on FedJam.

The evaluator follows the formal FedJam few-shot protocol used by the other
independent branches: benign-only nested 1/2/4-shot support, an all-test-row
evaluation, a normal patch memory, 1-NN cosine distance, and image-level max
patch aggregation.  ConvNeXt stage2 is used because its 14x14 feature map is
the closest spatial analogue of the existing ResNet18 layer3 branch.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_fedjam_fewshot_dual import (
    LABEL_NAMES,
    add_metric_rows,
    iter_test_records,
    metric,
    raw_tensor,
    select_benign_support,
    write_csv,
)


BRANCH_NAME = "ConvNeXt-B-stage2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/mnt/data/wangbei/data/FedJam")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="convnext_base.fb_in22k_ft_in1k")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--match-chunk-size", type=int, default=2048)
    parser.add_argument("--max-test-per-label", type=int, default=0)
    return parser.parse_args()


def raw_to_normalized_rgb(records: list) -> torch.Tensor:
    """Convert FedJam decoded BGR uint8 records to ImageNet-normalized RGB."""

    raw = raw_tensor(records)
    rgb = raw[..., [2, 1, 0]].permute(0, 3, 1, 2).float() / 255.0
    mean = rgb.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = rgb.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (rgb - mean) / std


def build_model(args: argparse.Namespace, device: torch.device):
    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=0,
    )
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def encode_stage2(model, images: torch.Tensor) -> torch.Tensor:
    """Return normalized ConvNeXt stage2 tokens with shape [B, 196, 512]."""

    _, intermediates = model.forward_intermediates(images, indices=[2])
    if len(intermediates) != 1:
        raise RuntimeError(f"Expected one ConvNeXt intermediate, got {len(intermediates)}")
    feature_map = intermediates[0]
    if feature_map.ndim != 4 or feature_map.shape[-2:] != (14, 14):
        raise RuntimeError(f"Unexpected ConvNeXt stage2 shape: {tuple(feature_map.shape)}")
    tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
    return F.normalize(tokens.float(), dim=-1)


def build_galleries(
    support_features: torch.Tensor, shots: list[int]
) -> dict[int, torch.Tensor]:
    galleries = {}
    for shot in shots:
        gallery = support_features[:shot].reshape(-1, support_features.shape[-1])
        galleries[shot] = F.normalize(gallery.float(), dim=-1).contiguous()
    return galleries


@torch.inference_mode()
def score_batch(
    features: torch.Tensor,
    gallery: torch.Tensor,
    match_chunk_size: int,
) -> np.ndarray:
    """Score each image by the maximum 1-NN patch cosine distance."""

    probe = features.reshape(-1, features.shape[-1])
    best_similarity = []
    gallery_t = gallery.transpose(0, 1).contiguous()
    for start in range(0, probe.shape[0], match_chunk_size):
        similarity = probe[start : start + match_chunk_size] @ gallery_t
        best_similarity.append(similarity.amax(dim=1))
    patch_distance = 1.0 - torch.cat(best_similarity, dim=0)
    image_scores = patch_distance.reshape(features.shape[0], -1).amax(dim=1)
    return image_scores.detach().cpu().numpy().astype(np.float32)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    shots = sorted(set(int(value) for value in args.shots))
    if shots != [1, 2, 4]:
        raise ValueError("This formal comparison requires shots 1, 2, and 4")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.match_chunk_size < 1:
        raise ValueError("--match-chunk-size must be positive")
    if args.max_test_per_label < 0:
        raise ValueError("--max-test-per-label cannot be negative")

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device(
        "cuda:0" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[convnext] device={device} data_root={data_root}", flush=True)

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
                {
                    "name": record.name,
                    "label": record.label,
                    "shape": list(record.image_bgr.shape),
                }
                for record in support
            ],
        },
    )

    model = build_model(args, device)
    support_images = raw_to_normalized_rgb(support).to(device)
    with torch.inference_mode():
        support_features = encode_stage2(model, support_images).contiguous()
    galleries = build_galleries(support_features, shots)
    del support_images, support_features, support
    gc.collect()

    score_bank = {shot: [] for shot in shots}
    labels: list[int] = []
    names: list[str] = []
    pending = []
    processed = 0

    def flush(records: list) -> None:
        nonlocal processed
        if not records:
            return
        images = raw_to_normalized_rgb(records).to(device)
        with torch.inference_mode():
            features = encode_stage2(model, images)
            for shot in shots:
                score_bank[shot].extend(
                    score_batch(features, galleries[shot], args.match_chunk_size).tolist()
                )
        labels.extend(int(record.label) for record in records)
        names.extend(record.name for record in records)
        processed += len(records)
        if processed % max(args.batch_size * 100, 100) < len(records):
            print(f"[test] processed={processed}", flush=True)
        del images, features

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

    score_dir = output_root / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    summary = {
        "status": "complete",
        "method": "fedjam_convnext_b_stage2_patch_memory",
        "protocol": "benign-only nested 1/2/4-shot; full independent test; spectrogram only",
        "data_root": str(data_root),
        "test_count": int(labels_np.size),
        "test_label_counts": {
            LABEL_NAMES[int(label)]: int(np.sum(labels_np == int(label)))
            for label in sorted(np.unique(labels_np).tolist())
        },
        "branch": BRANCH_NAME,
        "backbone": args.model,
        "pretrained": True,
        "input_size": args.image_size,
        "feature_definition": "ConvNeXt-B stage2; 14x14 spatial map; normalized patch tokens; full normal patch memory; 1-NN cosine distance; max patch aggregation",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "scores": {},
    }
    for shot in shots:
        arrays = np.asarray(score_bank[shot], dtype=np.float32)
        np.savez_compressed(
            score_dir / f"fedjam_{shot}shot_convnext_b_stage2_scores.npz",
            labels=labels_np,
            names=np.asarray(names),
            convnext_b_stage2=arrays,
        )
        add_metric_rows(rows, shot, BRANCH_NAME, labels_np, arrays)
        summary["scores"][str(shot)] = metric(
            (labels_np != 0).astype(np.int32), arrays
        )

    write_csv(output_root / "metrics.csv", rows)
    save_json(output_root / "summary.json", summary)
    save_json(
        output_root / "protocol.json",
        {
            **summary,
            "support_selection": "seeded reservoir over train label=benign; nested prefix for shots",
            "test_selection": "all official test rows unless max-test-per-label is set",
            "comparison_note": "stage2 is used as the spatially matched ConvNeXt analogue of ResNet18 layer3; no training or test-batch calibration",
        },
    )
    print(f"wrote {output_root / 'metrics.csv'}", flush=True)
    print(f"wrote {output_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
