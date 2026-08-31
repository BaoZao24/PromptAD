#!/usr/bin/env python3
"""Evaluate UniVAD texture evidence branches on FedJam.

The official UniVAD implementation has component paths (C3/CAPM/GECM) that
require object/component masks.  FedJam contains spectrograms rather than
objects, so those paths are not claimed here.  This evaluator separates the
four independently measurable evidence sources used by the official whole-
image texture path:

* ``clip_global``: CLIP-L/14-336 global image-feature distance;
* ``clip_patch``: CLIP multi-layer patch matching distance;
* ``dino_patch``: DINOv2-G patch matching distance;
* ``clip_text_map``: CLIP text-prompt patch anomaly map.

``texture_fusion`` is saved as a reference using the same combination as the
existing FedJam UniVAD-Texture-adapted evaluator.  It is not counted as one
of the four independent branches.
"""

from __future__ import annotations

import argparse
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

from tools.eval_fedjam_fewshot_dual import (
    LABEL_NAMES,
    add_metric_rows,
    iter_test_records,
    metric,
    raw_tensor,
    select_benign_support,
    write_csv,
)

# The FedJam helper imports the project's ``utils``/``models`` packages,
# while the reference UniVAD tree has packages with the same top-level names.
# Load the FedJam functions first, then give the UniVAD import its own package
# namespace.  The already-imported FedJam functions do not need those modules
# at runtime for this evaluator.
for _module_name in list(sys.modules):
    if (
        _module_name == "utils"
        or _module_name.startswith("utils.")
        or _module_name == "models"
        or _module_name.startswith("models.")
    ):
        del sys.modules[_module_name]

from tools.eval_univad_rf_fewshot import (
    build_model,
    chunked_max_similarity,
)


BRANCHES = ("clip_global", "clip_patch", "dino_patch", "clip_text_map")
ALL_SCORES = (*BRANCHES, "texture_fusion")
BRANCH_LABELS = {
    "clip_global": "UniVAD-CLIP-global",
    "clip_patch": "UniVAD-CLIP-patch",
    "dino_patch": "UniVAD-DINO-patch",
    "clip_text_map": "UniVAD-CLIP-text-map",
    "texture_fusion": "UniVAD-texture-fusion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/mnt/data/wangbei/data/FedJam")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--encode-batch-size", type=int, default=1)
    parser.add_argument("--match-chunk-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--max-test-per-label", type=int, default=0)
    return parser.parse_args()


def raw_to_rgb_tensor(records: list) -> torch.Tensor:
    """Convert FedJam decoded BGR uint8 records to RGB float tensors."""

    raw = raw_tensor(records)
    return raw[..., [2, 1, 0]].permute(0, 3, 1, 2).float() / 255.0


@torch.inference_mode()
def build_texture_memories(
    model,
    support_images: torch.Tensor,
    shots: list[int],
    encode_batch_size: int,
) -> dict[int, dict[str, object]]:
    """Encode the common support pool once and expose nested shot memories.

    The support pool is selected as a deterministic prefix for 1/2/4-shot.
    All support images can therefore be encoded together, but each shot must
    use only its own prefix when computing nearest-neighbour scores.
    """

    image_features_all = []
    patch_tokens_all = None
    dino_tokens_all = []
    for start in range(0, len(support_images), encode_batch_size):
        support_batch = support_images[start : start + encode_batch_size]
        clip_images = model.transform_clip(support_batch).to(model.device)
        dino_images = model.transform_dino(support_batch).to(model.device)
        image_features, patch_tokens = model.clip_model.encode_image(
            clip_images, model.out_layers
        )
        image_features_all.append(
            F.normalize(image_features[:, 0, :], dim=-1).detach().contiguous()
        )
        patch_tokens = model.decoder(patch_tokens)
        if patch_tokens_all is None:
            patch_tokens_all = [[] for _ in patch_tokens]
        for index, tokens in enumerate(patch_tokens):
            patch_tokens_all[index].append(tokens.detach().contiguous())
        dino_tokens_all.append(
            model.dinov2_net.forward_features(dino_images)["x_norm_patchtokens"]
            .detach()
            .contiguous()
        )
        del clip_images, dino_images, image_features, patch_tokens

    normal_image_features = torch.cat(image_features_all, dim=0)
    normal_patch_tokens = [torch.cat(items, dim=0) for items in patch_tokens_all]
    normal_dino_patches = torch.cat(dino_tokens_all, dim=0)
    memories = {}
    for shot in shots:
        memories[shot] = {
            "normal_image_features": normal_image_features[:shot],
            "normal_patch_tokens": [tokens[:shot] for tokens in normal_patch_tokens],
            "normal_dino_patches": normal_dino_patches[:shot],
        }
    return memories


@torch.inference_mode()
def score_components(
    model,
    image: torch.Tensor,
    memories: dict[int, dict[str, object]],
    chunk_size: int,
) -> dict[int, dict[str, float]]:
    """Return branch scores for each shot using its own nested support memory."""

    clip_input = model.transform_clip(image).to(model.device)
    dino_input = model.transform_dino(image).to(model.device)
    image_features, patch_tokens = model.clip_model.encode_image(
        clip_input, model.out_layers
    )
    image_features = F.normalize(image_features[:, 0, :], dim=-1)
    patch_tokens = model.decoder(patch_tokens)

    global_scores = {}
    clip_maps = {}
    for shot, memory in memories.items():
        global_scores[shot] = float(
            (1.0 - (image_features @ memory["normal_image_features"].T).amax()).item()
        )
        shot_clip_maps = []
        for layer_index, tokens in enumerate(patch_tokens):
            if layer_index % 2 == 0:
                continue
            query = tokens[0]
            reference = memory["normal_patch_tokens"][layer_index].reshape(
                -1, query.shape[-1]
            )
            patch_distance = 1.0 - chunked_max_similarity(
                query, reference, chunk_size
            )
            grid = int(round(patch_distance.numel() ** 0.5))
            shot_clip_maps.append(
                F.interpolate(
                    patch_distance.reshape(1, 1, grid, grid),
                    size=model.image_size,
                    mode="bilinear",
                    align_corners=True,
                )
            )
        clip_maps[shot] = torch.stack(shot_clip_maps).mean(dim=0)

    dino_tokens = model.dinov2_net.forward_features(dino_input)["x_norm_patchtokens"][0]
    dino_maps = {}
    for shot, memory in memories.items():
        dino_reference = memory["normal_dino_patches"].reshape(
            -1, dino_tokens.shape[-1]
        )
        dino_distance = 1.0 - chunked_max_similarity(
            dino_tokens, dino_reference, chunk_size
        )
        dino_grid = int(round(dino_distance.numel() ** 0.5))
        dino_maps[shot] = F.interpolate(
            dino_distance.reshape(1, 1, dino_grid, dino_grid),
            size=model.image_size,
            mode="bilinear",
            align_corners=True,
        )

    vl_tokens = patch_tokens[6] @ model.clip_model.visual.proj
    vl_tokens = vl_tokens / vl_tokens.norm(dim=-1, keepdim=True)
    vl_scores = 100.0 * vl_tokens @ model.text_prompts["object"]
    batch, length, classes = vl_scores.shape
    vl_map = F.interpolate(
        vl_scores.permute(0, 2, 1).reshape(
            batch, classes, int(length**0.5), int(length**0.5)
        ),
        size=model.image_size,
        mode="bilinear",
        align_corners=True,
    )
    probs = torch.softmax(vl_map, dim=1)
    vl_map = (probs[:, 1:2] - probs[:, 0:1] + 1.0) / 2.0

    clip_text_score = float(vl_map.amax().item())
    results = {}
    for shot in memories:
        clip_patch_score = float(clip_maps[shot].amax().item())
        dino_patch_score = float(dino_maps[shot].amax().item())
        fusion_score = float(
            ((clip_maps[shot] + dino_maps[shot] + vl_map) / 3.0).amax().item()
            + global_scores[shot]
        )
        results[shot] = {
            "clip_global": global_scores[shot],
            "clip_patch": clip_patch_score,
            "dino_patch": dino_patch_score,
            "clip_text_map": clip_text_score,
            "texture_fusion": fusion_score,
        }
    return results


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    shots = sorted(set(int(value) for value in args.shots))
    if shots != [1, 2, 4]:
        raise ValueError("This comparison requires shots 1, 2, and 4")
    if args.batch_size != 1:
        raise ValueError("UniVAD official texture scoring currently requires --batch-size 1")
    if args.max_test_per_label < 0:
        raise ValueError("--max-test-per-label cannot be negative")

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device(
        "cuda:0" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[univad-branches] device={device} data_root={data_root}", flush=True)

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

    model = build_model(
        argparse.Namespace(
            image_size=args.image_size,
            device=str(device),
        )
    )
    model.eval()

    support_images = raw_to_rgb_tensor(support)
    memories = build_texture_memories(
        model, support_images, shots, args.encode_batch_size
    )
    del support_images, support
    gc.collect()

    score_bank = {
        shot: {branch: [] for branch in ALL_SCORES}
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
        images = raw_to_rgb_tensor(records)
        with torch.inference_mode():
            for record, image in zip(records, images):
                components_by_shot = score_components(
                    model, image.unsqueeze(0), memories, args.match_chunk_size
                )
                for shot in shots:
                    components = components_by_shot[shot]
                    for branch in ALL_SCORES:
                        score_bank[shot][branch].append(components[branch])
                labels.append(int(record.label))
                names.append(record.name)
        processed += len(records)
        if processed % 100 == 0 or processed == 1:
            print(f"[test] processed={processed}", flush=True)
        del images

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
        "method": "univad_texture_branch_ablation",
        "official_method": "UniVAD",
        "protocol": "benign-only nested 1/2/4-shot; full independent test; spectrogram only",
        "protocol_boundary": "C3/CAPM/GECM component paths omitted because FedJam spectrograms have no object masks",
        "data_root": str(data_root),
        "test_count": int(labels_np.size),
        "test_label_counts": {
            LABEL_NAMES[int(label)]: int(np.sum(labels_np == int(label)))
            for label in sorted(np.unique(labels_np).tolist())
        },
        "branch_definitions": {
            "clip_global": "official UniVAD CLIP-L/14-336 global CLS distance to normal support",
            "clip_patch": "official UniVAD CLIP multi-layer patch matching distance",
            "dino_patch": "official UniVAD DINOv2-G patch matching distance",
            "clip_text_map": "official UniVAD CLIP object text-prompt patch map",
            "texture_fusion": "reference fusion: max((CLIP-patch + DINO-patch + CLIP-text-map)/3) + CLIP-global",
        },
        "backbones": {
            "clip": "ViT-L/14-336",
            "dino": "DINOv2-G/14",
        },
        "seed": args.seed,
        "batch_size": args.batch_size,
        "scores": {},
    }

    for shot in shots:
        arrays = {
            branch: np.asarray(score_bank[shot][branch], dtype=np.float32)
            for branch in ALL_SCORES
        }
        np.savez_compressed(
            score_dir / f"fedjam_{shot}shot_univad_branch_scores.npz",
            labels=labels_np,
            names=np.asarray(names),
            **arrays,
        )
        summary["scores"][str(shot)] = {}
        for branch in ALL_SCORES:
            add_metric_rows(rows, shot, BRANCH_LABELS[branch], labels_np, arrays[branch])
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
            "fusion": "texture_fusion saved only as a reference; four evidence branches are reported independently",
        },
    )
    print(f"wrote {output_root / 'metrics.csv'}", flush=True)
    print(f"wrote {output_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
