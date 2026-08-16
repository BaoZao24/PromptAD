"""Evaluate the UniVAD texture adapter on the formal Public-RF protocol.

This is deliberately a texture-only adaptation.  Public RF spectrograms do
not provide the object/component masks required by UniVAD's C3/CAPM/GECM
branches, so the evaluator keeps the whole-image CLIP/DINO patch path and
labels the result ``UniVAD-Texture-adapted (DINOv2-G)``.

The three nested k-per-frequency support memories are built once.  Each query
batch is encoded once and scored against k=1/2/4 memories, which avoids three
repeated backbone passes while preserving the fixed full test set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from tools.eval_univad_rf_fewshot import (
    build_model,
    chunked_max_similarity,
    load_rgb_tensor,
    metric_summary,
    setup_texture_memory,
)


PUBLIC_ROOT = Path("/mnt/data/wangbei/data/RF_SPE_PNG")
DEFAULT_MANIFEST_ROOT = ROOT / "analysis_outputs/20260810_public_rf_k_per_frequency"
DEFAULT_OURS_ROOT = ROOT / "analysis_outputs/exploratory/20260815_tta_position_no_tta"
SIGNAL_JSRS = {
    "burst": ("m30db", "m40db", "m50db"),
    "chirp": ("m40db", "m50db", "m55db"),
    "dsss": ("m30db", "m40db", "m50db"),
    "pulse": ("m30db", "m40db", "m50db"),
    "deceptive": ("m10db", "m20db", "m30db"),
}


def load_manifest(path: Path) -> dict:
    """Validate the Public-RF manifest without importing project ``utils``.

    The official UniVAD source has its own top-level ``utils`` package, so
    importing ``utils.public_rf_support`` after that package is on sys.path
    would resolve the wrong module.  This small validation keeps the two
    package namespaces independent.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "public_rf_fixed_test_support_pool":
        raise ValueError(f"Unexpected Public-RF protocol in {path}")
    support = [str(value) for value in payload.get("support_paths", [])]
    test = [str(value) for value in payload.get("test_paths", [])]
    if not support or not test or len(set(support)) != len(support) or len(set(test)) != len(test):
        raise ValueError(f"Invalid Public-RF support/test paths in {path}")
    if set(support) & set(test):
        raise ValueError(f"Public-RF support/test overlap in {path}")
    digest = lambda values: hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()
    if payload.get("support_paths_sha256") != digest(support):
        raise ValueError(f"Public-RF support digest mismatch in {path}")
    if payload.get("test_paths_sha256") != digest(test):
        raise ValueError(f"Public-RF test digest mismatch in {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--encode-batch-size", type=int, default=1)
    parser.add_argument("--match-chunk-size", type=int, default=256)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--signals", nargs="+", default=list(SIGNAL_JSRS))
    parser.add_argument("--jsrs", default="")
    parser.add_argument("--max-test-normal", type=int, default=0)
    parser.add_argument("--max-test-abnormal", type=int, default=0)
    parser.add_argument("--manifest-root", default=str(DEFAULT_MANIFEST_ROOT))
    parser.add_argument("--ours-root", default=str(DEFAULT_OURS_ROOT))
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def parse_jsr_override(value: str) -> dict[str, tuple[str, ...]]:
    result = {signal: tuple(jsrs) for signal, jsrs in SIGNAL_JSRS.items()}
    if not value:
        return result
    for item in value.split(";"):
        signal, jsrs = item.split(":", 1)
        if signal not in result:
            raise ValueError(f"Unknown signal: {signal}")
        values = tuple(x.strip() for x in jsrs.split(",") if x.strip())
        if not values:
            raise ValueError(f"No JSR values for {signal}")
        result[signal] = values
    return result


def read_manifest_paths(manifest_root: Path, k: int) -> tuple[list[str], list[str], dict]:
    path = manifest_root / f"k{k}" / "support_manifest.json"
    manifest = load_manifest(path)
    if int(manifest.get("per_frequency_k", -1)) != k:
        raise ValueError(f"Manifest {path} is not k={k}")
    return list(manifest["support_paths"]), list(manifest["test_paths"]), manifest


def build_cells(test_paths: list[str], signal: str, jsr: str, args: argparse.Namespace) -> tuple[list[str], list[int]]:
    normal_paths = list(test_paths)
    if args.max_test_normal > 0:
        normal_paths = normal_paths[: args.max_test_normal]
    abnormal_root = PUBLIC_ROOT / signal / "abnormal" / jsr
    abnormal_paths = sorted(str(path) for path in abnormal_root.glob("*.png"))
    if args.max_test_abnormal > 0:
        abnormal_paths = abnormal_paths[: args.max_test_abnormal]
    if not abnormal_paths:
        raise FileNotFoundError(f"No abnormal images for {signal}/{jsr}: {abnormal_root}")
    return normal_paths + abnormal_paths, [0] * len(normal_paths) + [1] * len(abnormal_paths)


def encode_query(model, images: torch.Tensor) -> dict[str, object]:
    clip_input = model.transform_clip(images).to(model.device)
    dino_input = model.transform_dino(images).to(model.device)
    image_features, patch_tokens = model.clip_model.encode_image(clip_input, model.out_layers)
    image_features = F.normalize(image_features[:, 0, :], dim=-1).contiguous()
    patch_tokens = model.decoder(patch_tokens)
    dino_tokens = model.dinov2_net.forward_features(dino_input)["x_norm_patchtokens"].contiguous()
    vl_tokens = patch_tokens[6] @ model.clip_model.visual.proj
    vl_tokens = F.normalize(vl_tokens, dim=-1)
    vl_scores = 100.0 * vl_tokens @ model.text_prompts["object"]
    return {
        "image": image_features,
        "patch": patch_tokens,
        "dino": dino_tokens,
        "vl": vl_scores,
    }


def subset_max_similarity(
    query: torch.Tensor,
    reference: torch.Tensor,
    support_indices_by_k: dict[int, torch.Tensor],
    patches_per_support: int,
    chunk_size: int,
) -> dict[int, torch.Tensor]:
    """Compute k-subset maxima from one similarity matrix pass.

    The k-per-frequency manifests are nested.  The largest memory therefore
    contains every support patch needed by k=1 and k=2; selecting support rows
    after one matrix multiplication is substantially faster than repeating
    the CLIP/DINO matching three times.
    """

    query = F.normalize(query, dim=-1)
    reference = F.normalize(reference, dim=-1)
    reference_t = reference.transpose(0, 1).contiguous()
    values = {k: [] for k in support_indices_by_k}
    for start in range(0, query.shape[0], chunk_size):
        similarity = torch.mm(query[start : start + chunk_size], reference_t)
        similarity = similarity.reshape(
            similarity.shape[0], reference.shape[0] // patches_per_support, patches_per_support
        )
        for k, indices in support_indices_by_k.items():
            values[k].append(similarity.index_select(1, indices).amax(dim=(1, 2)))
    return {k: torch.cat(items, dim=0) for k, items in values.items()}


def score_encoded_multi(
    model,
    encoded: dict[str, object],
    memory: tuple[torch.Tensor, list[torch.Tensor], torch.Tensor],
    support_indices_by_k: dict[int, torch.Tensor],
    chunk_size: int,
) -> dict[int, torch.Tensor]:
    """Score all nested k memories while encoding each query only once."""

    normal_image, normal_patch, normal_dino = memory
    image_features = encoded["image"]
    patch_tokens = encoded["patch"]
    dino_tokens = encoded["dino"]
    vl_scores = encoded["vl"]
    global_similarity = subset_max_similarity(
        image_features,
        normal_image,
        support_indices_by_k,
        patches_per_support=1,
        chunk_size=chunk_size,
    )
    global_scores = {k: 1.0 - value for k, value in global_similarity.items()}

    clip_maps = {k: [] for k in support_indices_by_k}
    for layer_index, tokens in enumerate(patch_tokens):
        if layer_index % 2 == 0:
            continue
        batch, length, dim = tokens.shape
        reference = normal_patch[layer_index].reshape(-1, dim)
        similarities = subset_max_similarity(
            tokens.reshape(-1, dim),
            reference,
            support_indices_by_k,
            patches_per_support=length,
            chunk_size=chunk_size,
        )
        for k, similarity in similarities.items():
            distance = 1.0 - similarity.reshape(
                batch, int(length**0.5), int(length**0.5)
            )
            clip_maps[k].append(
                F.interpolate(
                    distance.unsqueeze(1),
                    size=model.image_size,
                    mode="bilinear",
                    align_corners=True,
                )
            )
    clip_maps = {
        k: torch.stack(items, dim=0).mean(dim=0) for k, items in clip_maps.items()
    }

    batch, length, dim = dino_tokens.shape
    dino_reference = normal_dino.reshape(-1, dim)
    dino_similarities = subset_max_similarity(
        dino_tokens.reshape(-1, dim),
        dino_reference,
        support_indices_by_k,
        patches_per_support=length,
        chunk_size=chunk_size,
    )
    dino_maps = {}
    for k, similarity in dino_similarities.items():
        dino_distance = 1.0 - similarity.reshape(
            batch, int(length**0.5), int(length**0.5)
        )
        dino_maps[k] = F.interpolate(
            dino_distance.unsqueeze(1),
            size=model.image_size,
            mode="bilinear",
            align_corners=True,
        )

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
    return {
        k: ((clip_maps[k] + dino_maps[k] + vl_map) / 3.0).flatten(1).amax(dim=1)
        + global_scores[k]
        for k in support_indices_by_k
    }


def load_ours_rows(root: Path, k: int) -> dict[tuple[str, str], dict[str, float]]:
    path = root / f"public_fusion_k{k}" / "per_cell_confidence_gate.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["dataset"], row["jsr"]): {
                "auroc": float(row["confidence_gated_auc"]),
                "auprc": float(row["confidence_gated_auprc"]),
                "fpr95": float(row["confidence_gated_fpr95"]),
            }
            for row in csv.DictReader(handle)
        }


def result_payload(args: argparse.Namespace, rows: list[dict], status: str) -> dict:
    by_k = {}
    for k in sorted(set(row["k"] for row in rows)):
        subset = [row for row in rows if row["k"] == k]
        by_k[str(k)] = {
            metric: float(np.mean([row["metrics"][metric] for row in subset]))
            for metric in ("auroc", "auprc", "fpr95")
        }
    return {
        "status": status,
        "method": "UniVAD-Texture-adapted",
        "official_method": "UniVAD",
        "protocol_note": "Public RF k-per-frequency full test; whole-image texture branch; C3/CAPM/GECM omitted",
        "support_protocol": "nested k=1/2/4-per-frequency",
        "ks": sorted(set(args.ks)),
        "cell_count_completed": len(rows),
        "metrics_macro_by_k": by_k,
        "cells": rows,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "match_chunk_size": args.match_chunk_size,
    }


def save_checkpoint(args: argparse.Namespace, rows: list[dict], status: str) -> None:
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result_payload(args, rows, status), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def score_paths(
    model,
    paths: list[str],
    memory: tuple[torch.Tensor, list[torch.Tensor], torch.Tensor],
    support_indices_by_k: dict[int, torch.Tensor],
    args: argparse.Namespace,
    description: str,
) -> dict[int, list[float]]:
    """Encode a path list once and return scores for every nested k."""

    scores_by_k = {k: [] for k in support_indices_by_k}
    with torch.inference_mode():
        for start in range(0, len(paths), args.batch_size):
            image_batch = torch.stack(
                [
                    load_rgb_tensor(path, args.image_size)
                    for path in paths[start : start + args.batch_size]
                ]
            )
            encoded = encode_query(model, image_batch)
            scores = score_encoded_multi(
                model,
                encoded,
                memory,
                support_indices_by_k,
                args.match_chunk_size,
            )
            for k, values in scores.items():
                scores_by_k[k].extend(float(value) for value in values.detach().cpu())
            if (start // args.batch_size + 1) % 100 == 0:
                print(
                    f"  {description} processed={min(start + args.batch_size, len(paths))}/{len(paths)}",
                    flush=True,
                )
    return scores_by_k


def main() -> None:
    args = parse_args()
    args.ks = sorted(set(args.ks))
    if any(k not in (1, 2, 4) for k in args.ks):
        raise ValueError("Public RF only supports k=1,2,4")
    jsrs_by_signal = parse_jsr_override(args.jsrs)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    print(f"[univad-public] ks={args.ks} signals={args.signals} batch={args.batch_size}", flush=True)

    model = build_model(args)
    support_paths_by_k = {}
    manifest_meta = {}
    for k in args.ks:
        support_paths, test_paths, manifest = read_manifest_paths(Path(args.manifest_root), k)
        support_paths_by_k[k] = support_paths
        manifest_meta[str(k)] = {
            "manifest_sha256": manifest["manifest_sha256"],
            "support_count": len(support_paths),
            "test_normal_count": len(test_paths),
        }
    max_k = max(args.ks)
    max_support_paths = support_paths_by_k[max_k]
    support_images = torch.stack(
        [load_rgb_tensor(path, args.image_size) for path in max_support_paths]
    )
    setup_texture_memory(model, support_images, args.encode_batch_size)
    memory = (
        model.normal_image_features,
        model.normal_patch_tokens,
        model.normal_dino_patches,
    )
    max_support_index = {path: index for index, path in enumerate(max_support_paths)}
    support_indices_by_k = {
        k: torch.tensor(
            [max_support_index[path] for path in support_paths_by_k[k]],
            device=model.device,
            dtype=torch.long,
        )
        for k in args.ks
    }
    for k in args.ks:
        print(
            f"[memory k={k}] support={len(support_paths_by_k[k])} "
            f"test_normal={manifest_meta[str(k)]['test_normal_count']}",
            flush=True,
        )

    # All k manifests share the same fixed test pool.  Read it once.
    _, test_paths, _ = read_manifest_paths(Path(args.manifest_root), args.ks[0])
    rows: list[dict] = []
    ours_root = Path(args.ours_root)
    ours_by_k = {k: load_ours_rows(ours_root, k) for k in args.ks}
    normal_paths = list(test_paths)
    if args.max_test_normal > 0:
        normal_paths = normal_paths[: args.max_test_normal]
    print(f"[normal test pool] count={len(normal_paths)}; score once and reuse", flush=True)
    normal_scores_by_k = score_paths(
        model,
        normal_paths,
        memory,
        support_indices_by_k,
        args,
        "normal-pool",
    )
    for signal in args.signals:
        if signal not in SIGNAL_JSRS:
            raise ValueError(f"Unknown Public RF signal {signal}")
        for jsr in jsrs_by_signal[signal]:
            paths, labels = build_cells(test_paths, signal, jsr, args)
            normal_count = labels.count(0)
            abnormal_paths = paths[normal_count:]
            full_labels = [0] * normal_count + [1] * len(abnormal_paths)
            print(
                f"[cell {signal}/{jsr}] normal={normal_count} abnormal={len(abnormal_paths)}",
                flush=True,
            )
            scores_by_k = {k: list(normal_scores_by_k[k]) for k in args.ks}
            abnormal_scores_by_k = score_paths(
                model,
                abnormal_paths,
                memory,
                support_indices_by_k,
                args,
                f"{signal}/{jsr} abnormal",
            )
            for k in args.ks:
                scores_by_k[k].extend(abnormal_scores_by_k[k])

            for k in args.ks:
                metrics = metric_summary(full_labels, scores_by_k[k])
                ours = ours_by_k[k].get((signal, jsr))
                rows.append(
                    {
                        "method": "UniVAD-Texture-adapted",
                        "official_method": "UniVAD",
                        "k": k,
                        "signal": signal,
                        "scene": "RF_SPE_PNG_public",
                        "jsr": jsr,
                        "support_count": manifest_meta[str(k)]["support_count"],
                        "test_normal_count": labels.count(0),
                        "test_abnormal_count": labels.count(1),
                        "metrics": metrics,
                        "ours_existing_metrics_same_cell_full_test": ours,
                    }
                )
            save_checkpoint(args, rows, "running")

    final = result_payload(args, rows, "complete")
    final["manifest"] = manifest_meta
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)
    Path(args.output_json).write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
