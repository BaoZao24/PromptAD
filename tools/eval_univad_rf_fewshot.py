"""Evaluate a transparent UniVAD texture-branch adaptation on RF spectrograms.

The official UniVAD implementation assumes object/component masks and uses a
large DINOv2-G backbone.  RF spectrograms do not contain the object masks used
by UniVAD's C3/GECM branches, so this evaluator deliberately uses only the
paper's whole-image texture path: frozen CLIP-L patch matching plus frozen
DINOv2 patch matching against normal support.  Results must be labelled
``UniVAD-Texture-adapted`` rather than claimed as an exact full UniVAD run.

The reference implementation and its DINOv2 submodule are loaded from
``references/UniVAD``.  This script is an exploratory adapter and is not part
of the SpectraMemAD formal evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
UNIVAD_ROOT = ROOT / "references" / "UniVAD"
if str(UNIVAD_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIVAD_ROOT))

# ``pydensecrf`` is imported by UniVAD's segmentation-only code path.  RF
# texture mode never calls it, so keep the optional dependency out of this
# evaluator and make that boundary explicit.
_crf_stub = types.ModuleType("utils.crf")
_crf_stub.dense_crf = lambda *_args, **_kwargs: None
sys.modules.setdefault("utils.crf", _crf_stub)

# The official import tree eagerly imports GroundingDINO/SAM even though the
# texture branch never calls component segmentation.  Stub only those unused
# boundaries; no segmentation result is used by this adapter.
_component_features_stub = types.ModuleType("models.component_feature_extractor")
_component_features_stub.ComponentFeatureExtractor = type(
    "UnusedComponentFeatureExtractor", (), {"__init__": lambda self, *_a, **_k: None}
)
sys.modules.setdefault("models.component_feature_extractor", _component_features_stub)
_component_segmentation_stub = types.ModuleType("models.component_segmentaion")
_component_segmentation_stub.split_masks_from_one_mask = lambda *_a, **_k: []
_component_segmentation_stub.split_masks_from_one_mask_torch = lambda *_a, **_k: []
_component_segmentation_stub.split_masks_from_one_mask_with_bg = lambda *_a, **_k: ([], [])
sys.modules.setdefault("models.component_segmentaion", _component_segmentation_stub)

import UniVAD as univad_impl  # noqa: E402
from UniVAD import UniVAD, object_type  # noqa: E402


DEFAULT_MANIFEST = ROOT / "analysis_outputs/20260728_rf_five_type_formal_seed111/support_manifest.json"
# Match the current 60-cell In-house protocol: five signal types including
# deceptive_signal and the support-only (TTA=none) mainline result.  The older
# 20260725 directory contains the retired wideband_pulse protocol and must not
# be used for this comparison.
DEFAULT_OURS_ROOT = ROOT / "analysis_outputs/exploratory/20260815_tta_position_no_tta/inhouse_fusion"


class _UnusedDinoFeaturizer(nn.Module):
    """Avoid the official segmentation-only DINO-Small download.

    The texture branch never calls ``dino_net``; it uses UniVAD's DINOv2
    backbone instead.  Keeping this unused module explicit prevents a hidden
    extra checkpoint download during the RF adapter run.
    """

    def forward(self, *_args, **_kwargs):  # pragma: no cover - defensive only
        raise RuntimeError("The segmentation-only DINO featurizer is disabled for RF texture mode")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--signal", default="burst_signal")
    parser.add_argument("--scene", default="WeaponMuseum_spectrum")
    parser.add_argument("--jsr", default="m30db")
    parser.add_argument("--all-cells", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--max-test-normal", type=int, default=8)
    parser.add_argument("--max-test-abnormal", type=int, default=8)
    parser.add_argument("--encode-batch-size", type=int, default=1)
    parser.add_argument("--match-chunk-size", type=int, default=256)
    parser.add_argument(
        "--dino-model",
        choices=("dinov2_vitg14", "dinov2_vitb14"),
        default="dinov2_vitg14",
        help="DINOv2 backbone; G preserves the old protocol and B is the size-matched rerun.",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--ours-csv", default="")
    return parser.parse_args()


def load_rgb_tensor(path: str, image_size: int) -> torch.Tensor:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image).resize((image_size, image_size), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0


def setup_texture_memory(model: UniVAD, support_images: torch.Tensor, encode_batch_size: int) -> None:
    """Build normal memories in small batches to bound temporary GPU memory."""

    image_features_all = []
    patch_tokens_all = None
    dino_tokens_all = []
    with torch.inference_mode():
        for start in range(0, len(support_images), encode_batch_size):
            support_batch = support_images[start : start + encode_batch_size]
            clip_images = model.transform_clip(support_batch).to(model.device)
            dino_images = model.transform_dino(support_batch).to(model.device)
            image_features, patch_tokens = model.clip_model.encode_image(
                clip_images, model.out_layers
            )
            image_features_all.append(F.normalize(image_features[:, 0, :], dim=-1).contiguous())
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
    model.normal_image_features = torch.cat(image_features_all, dim=0)
    model.normal_patch_tokens = [torch.cat(items, dim=0) for items in patch_tokens_all]
    model.normal_dino_patches = torch.cat(dino_tokens_all, dim=0)
    model.gate = object_type.TEXTURE


def chunked_max_similarity(query: torch.Tensor, reference: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Return per-query maximum cosine similarity without a huge 3-D tensor."""

    query = F.normalize(query, dim=-1)
    reference = F.normalize(reference, dim=-1)
    reference_t = reference.transpose(0, 1).contiguous()
    values = []
    for start in range(0, query.shape[0], chunk_size):
        values.append(torch.mm(query[start : start + chunk_size], reference_t).amax(dim=1))
    return torch.cat(values, dim=0)


@torch.inference_mode()
def score_texture(model: UniVAD, image: torch.Tensor, chunk_size: int) -> float:
    """Official texture-branch score with chunked patch matching."""

    clip_input = model.transform_clip(image).to(model.device)
    dino_input = model.transform_dino(image).to(model.device)
    image_features, patch_tokens = model.clip_model.encode_image(clip_input, model.out_layers)
    image_features = F.normalize(image_features[:, 0, :], dim=-1)
    global_score = (1.0 - (image_features @ model.normal_image_features.T).amax()).item()
    patch_tokens = model.decoder(patch_tokens)

    clip_maps = []
    for layer_index, tokens in enumerate(patch_tokens):
        if layer_index % 2 == 0:
            continue
        query = tokens[0]
        reference = model.normal_patch_tokens[layer_index].reshape(-1, query.shape[-1])
        patch_distance = 1.0 - chunked_max_similarity(query, reference, chunk_size)
        grid = int(round(patch_distance.numel() ** 0.5))
        clip_maps.append(patch_distance.reshape(1, 1, grid, grid))
    clip_map = torch.stack(
        [F.interpolate(item, size=model.image_size, mode="bilinear", align_corners=True) for item in clip_maps]
    ).mean(dim=0)

    dino_tokens = model.dinov2_net.forward_features(dino_input)["x_norm_patchtokens"][0]
    dino_reference = model.normal_dino_patches.reshape(-1, dino_tokens.shape[-1])
    dino_distance = 1.0 - chunked_max_similarity(dino_tokens, dino_reference, chunk_size)
    dino_grid = int(round(dino_distance.numel() ** 0.5))
    dino_map = F.interpolate(
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
        vl_scores.permute(0, 2, 1).reshape(batch, classes, int(length**0.5), int(length**0.5)),
        size=model.image_size,
        mode="bilinear",
        align_corners=True,
    )
    probs = torch.softmax(vl_map, dim=1)
    vl_map = (probs[:, 1:2] - probs[:, 0:1] + 1.0) / 2.0
    return float(((clip_map + dino_map + vl_map) / 3.0).amax().item() + global_score)


def build_model(args: argparse.Namespace) -> UniVAD:
    # UniVAD's reference code fixes its device to cuda.  CUDA_VISIBLE_DEVICES
    # can be used by the caller to select a physical GPU safely.
    univad_impl.DinoFeaturizer = _UnusedDinoFeaturizer
    # The official constructor hard-codes ``dinov2_vitg14``. Intercept only
    # that local hub call so the official texture-path setup is unchanged when
    # running the size-matched B/14 experiment. The default remains G/14.
    requested_dino = getattr(args, "dino_model", "dinov2_vitg14")
    if requested_dino not in {"dinov2_vitg14", "dinov2_vitb14"}:
        raise ValueError(f"Unsupported UniVAD DINOv2 model: {requested_dino}")
    original_hub_load = torch.hub.load

    def patched_hub_load(repo_or_dir, model_name, *hub_args, **hub_kwargs):
        if model_name in {"dinov2_vitg14", "dinov2_vitb14"}:
            model_name = requested_dino
        return original_hub_load(repo_or_dir, model_name, *hub_args, **hub_kwargs)

    original_cwd = os.getcwd()
    torch.hub.load = patched_hub_load
    try:
        # The official constructor resolves its DINOv2 submodule relative to
        # the UniVAD repository root.
        os.chdir(UNIVAD_ROOT)
        model = UniVAD(image_size=args.image_size)
    finally:
        torch.hub.load = original_hub_load
        os.chdir(original_cwd)
    model.eval()
    model.dino_model_name = requested_dino
    return model


def find_cells(manifest: dict, args: argparse.Namespace) -> list[tuple[list[dict], dict]]:
    if args.all_cells:
        scene_support = {row["scene"]: row["support"] for row in manifest["scene_support"]}
        return [(scene_support[row["scene"]], row) for row in manifest["cells"]]
    scenes = [row for row in manifest["scene_support"] if row["scene"] == args.scene]
    cells = [
        row
        for row in manifest["cells"]
        if row["signal"] == args.signal and row["scene"] == args.scene and row["jsr"] == args.jsr
    ]
    if len(scenes) != 1 or len(cells) != 1:
        raise KeyError(f"Cannot find unique cell {args.signal}/{args.scene}/{args.jsr}")
    return [(scenes[0]["support"], cells[0])]


def metric_summary(labels: list[int], scores: list[float]) -> dict[str, float]:
    labels_np = np.asarray(labels, dtype=np.int32)
    scores_np = np.asarray(scores, dtype=np.float64)
    fpr, tpr, _ = roc_curve(labels_np, scores_np)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels_np, scores_np) * 100.0),
        "auprc": float(average_precision_score(labels_np, scores_np) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if reached.size else 100.0,
    }


def load_ours_auc(csv_path: Path, args: argparse.Namespace) -> float | None:
    if not csv_path.exists():
        return None
    import csv

    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["dataset"] == args.signal
                and row["scene"] == args.scene
                and row["jsr"] == args.jsr
            ):
                return float(row["confidence_gated_auc"])
    return None


def build_result(args: argparse.Namespace, results: list[dict], status: str) -> dict:
    metrics_macro = {
        metric: float(np.mean([row["metrics"][metric] for row in results]))
        for metric in ("auroc", "auprc", "fpr95")
    } if results else {metric: None for metric in ("auroc", "auprc", "fpr95")}
    ours_values = [
        row["ours_existing_auc_same_cell_full_test"]
        for row in results
        if row["ours_existing_auc_same_cell_full_test"] is not None
    ]
    return {
        "status": status,
        "method": "UniVAD-Texture-adapted",
        "official_method": "UniVAD",
        "dino_model": getattr(args, "dino_model", "dinov2_vitg14"),
        "dino_backbone": {
            "dinov2_vitg14": "DINOv2-G/14",
            "dinov2_vitb14": "DINOv2-B/14",
        }.get(getattr(args, "dino_model", "dinov2_vitg14"), "unknown"),
        "protocol_note": "whole-image texture branch; C3/GECM object components omitted for RF spectrograms",
        "selection": "all_cells" if args.all_cells else "single_cell",
        "signal": None if args.all_cells else args.signal,
        "scene": None if args.all_cells else args.scene,
        "jsr": None if args.all_cells else args.jsr,
        "cell_count": len(results),
        "metrics_macro": metrics_macro,
        "ours_existing_auc_macro": float(np.mean(ours_values)) if ours_values else None,
        "cells": results,
        "image_size": args.image_size,
        "encode_batch_size": args.encode_batch_size,
        "match_chunk_size": args.match_chunk_size,
    }


def save_result(args: argparse.Namespace, results: list[dict], status: str) -> None:
    if not args.output_json:
        return
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_result(args, results, status), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.support_manifest).read_text(encoding="utf-8"))
    selected_cells = find_cells(manifest, args)
    print(f"[univad-texture] cells={len(selected_cells)} device={args.device} image_size={args.image_size}", flush=True)
    model = build_model(args)
    results = []
    current_scene = None
    for index, (support_items, cell) in enumerate(selected_cells, start=1):
        normal_items = cell["test_normals"][: args.max_test_normal]
        abnormal_items = cell["test_abnormals"][: args.max_test_abnormal]
        if not normal_items or not abnormal_items:
            raise ValueError("Each cell needs both normal and abnormal query samples")
        if current_scene != cell["scene"]:
            support_images = torch.stack(
                [load_rgb_tensor(item["path"], args.image_size) for item in support_items]
            )
            setup_texture_memory(model, support_images, args.encode_batch_size)
            current_scene = cell["scene"]
        print(
            f"[cell {index}/{len(selected_cells)}] {cell['signal']}/{cell['scene']}/{cell['jsr']} "
            f"support={len(support_items)} normal={len(normal_items)} abnormal={len(abnormal_items)}",
            flush=True,
        )
        labels: list[int] = []
        scores: list[float] = []
        with torch.inference_mode():
            for label, item in [(0, row) for row in normal_items] + [(1, row) for row in abnormal_items]:
                image = load_rgb_tensor(item["path"], args.image_size).unsqueeze(0).to(model.device)
                labels.append(label)
                scores.append(score_texture(model, image, args.match_chunk_size))
        metrics = metric_summary(labels, scores)
        ours_csv = Path(args.ours_csv) if args.ours_csv else DEFAULT_OURS_ROOT / "per_cell_confidence_gate.csv"
        results.append(
            {
                "method": "UniVAD-Texture-adapted",
                "official_method": "UniVAD",
                "protocol_note": "whole-image texture branch; C3/GECM object components omitted for RF spectrograms",
                "signal": cell["signal"],
                "scene": cell["scene"],
                "jsr": cell["jsr"],
                "support_count": len(support_items),
                "test_normal_count": len(normal_items),
                "test_abnormal_count": len(abnormal_items),
                "metrics": metrics,
                "ours_existing_auc_same_cell_full_test": load_ours_auc(ours_csv, argparse.Namespace(
                    signal=cell["signal"], scene=cell["scene"], jsr=cell["jsr"]
                )),
            }
        )
        save_result(args, results, "running")
    result = build_result(args, results, "complete")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    save_result(args, results, "complete")


if __name__ == "__main__":
    main()
