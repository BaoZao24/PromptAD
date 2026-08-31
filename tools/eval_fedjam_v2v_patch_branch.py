#!/usr/bin/env python3
"""Evaluate PromptAD's final V2V-attention patch features on FedJam.

This is an exploratory branch experiment.  It uses ``visual_features[1]``
returned by PromptAD's V2VTransformer, whose patch tokens come from the V-V
attention path.  The normal support memory, nearest-neighbour rule, and
image-level max patch aggregation match the formal FedJam visual branch
protocol.  No prompt or visual backbone parameters are trained here.
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
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_cls_vit_patchcore_gallery import topk_cosine_distance_chunked
from tools.eval_fedjam_fewshot_dual import (
    LABEL_NAMES,
    add_metric_rows,
    iter_test_records,
    metric,
    raw_tensor,
    select_benign_support,
    write_csv,
)
from tools.eval_fedjam_four_branches import build_promptad
from train_rf_target_pooled_universal import to_model_input
from utils.training_utils import setup_seed


BRANCH_NAME = "ViT-V2V-final"


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
    parser.add_argument("--image-size", type=int, default=240)
    parser.add_argument("--max-test-per-label", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def encode_v2v(model, raw: torch.Tensor, device: torch.device) -> torch.Tensor:
    clip_input = to_model_input(model, raw, device, rgb_from_bgr=True)
    visual_features = model.encode_image(clip_input)
    # PromptAD's V2VTransformer returns:
    # [global, final_tokens, original_mid_layer1, original_mid_layer2].
    # The patch part of final_tokens is produced by the V-V attention path.
    return F.normalize(visual_features[1].float(), dim=-1).contiguous()


def build_galleries(features: torch.Tensor, shots: list[int]) -> dict[int, torch.Tensor]:
    return {
        shot: F.normalize(features[:shot].reshape(-1, features.shape[-1]).float(), dim=-1).contiguous()
        for shot in shots
    }


@torch.inference_mode()
def score_batch(
    features: torch.Tensor,
    gallery: torch.Tensor,
    distance_chunk_size: int,
) -> np.ndarray:
    distances = topk_cosine_distance_chunked(
        features.reshape(-1, features.shape[-1]),
        gallery,
        distance_chunk_size,
        1,
    )
    return (
        distances.reshape(features.shape[0], -1)
        .amax(dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    shots = sorted(set(int(value) for value in args.shots))
    if shots != [1, 2, 4]:
        raise ValueError("This comparison requires shots 1, 2, and 4")
    if args.batch_size < 1 or args.distance_chunk_size < 1:
        raise ValueError("batch and distance chunk sizes must be positive")
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
    print(f"[v2v] device={device} data_root={data_root}", flush=True)

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
        argparse.Namespace(
            checkpoint=args.checkpoint,
            image_size=args.image_size,
        ),
        device,
    )
    support_raw = raw_tensor(support)
    support_features = encode_v2v(model, support_raw, device).cpu()
    galleries = build_galleries(support_features, shots)
    del support_raw, support_features, support
    gc.collect()

    scores_by_shot = {shot: [] for shot in shots}
    labels: list[int] = []
    names: list[str] = []
    pending = []
    processed = 0

    def flush(records: list) -> None:
        nonlocal processed
        if not records:
            return
        raw = raw_tensor(records)
        features = encode_v2v(model, raw, device)
        for shot in shots:
            scores_by_shot[shot].extend(
                score_batch(features, galleries[shot].to(device), args.distance_chunk_size).tolist()
            )
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
        "method": "fedjam_promptad_v2v_final_patch",
        "protocol": "benign-only nested 1/2/4-shot; full independent test; spectrogram only",
        "data_root": str(data_root),
        "test_count": int(labels_np.size),
        "test_label_counts": {
            LABEL_NAMES[int(label)]: int(np.sum(labels_np == int(label)))
            for label in sorted(np.unique(labels_np).tolist())
        },
        "branch": BRANCH_NAME,
        "feature_definition": "PromptAD V2VTransformer final patch tokens; V-V path; normalized patch memory; 1-NN cosine distance; max patch aggregation",
        "backbone": "PromptAD CLIP ViT-B-16-plus-240",
        "v2v_source": "PromptAD/CLIPAD/transformer.py V2V Attention; q=k=v visual patch attention",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "scores": {},
    }
    score_dir = output_root / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    for shot in shots:
        arrays = np.asarray(scores_by_shot[shot], dtype=np.float32)
        np.savez_compressed(
            score_dir / f"fedjam_{shot}shot_v2v_patch_scores.npz",
            labels=labels_np,
            names=np.asarray(names),
            v2v_patch=arrays,
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
            "comparison_note": "This isolates final V2V patch features; current ViT-local layer1+layer2 uses PromptAD original-path intermediate hooks and is a separate control.",
        },
    )
    print(f"wrote {output_root / 'metrics.csv'}", flush=True)
    print(f"wrote {output_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
