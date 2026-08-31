#!/usr/bin/env python3
"""Evaluate the official UniVAD DINOv2-G patch evidence on RF datasets.

This evaluator isolates only the DINO patch evidence from UniVAD's whole-image
texture path.  It does not use CLIP, text prompts, component masks, training,
or test-batch calibration.  The in-house protocol uses the existing 24-image
scene support pool; the public protocol uses the existing nested k=1/2/4 per-
frequency support manifests.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
UNIVAD_ROOT = ROOT / "references" / "UniVAD"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


INHOUSE_MANIFEST = ROOT / "analysis_outputs/20260728_rf_five_type_formal_seed111/support_manifest.json"
PUBLIC_ROOT = Path("/mnt/data/wangbei/data/RF_SPE_PNG")
PUBLIC_MANIFEST_ROOT = ROOT / "analysis_outputs/20260810_public_rf_k_per_frequency"
PUBLIC_SIGNALS = {
    "burst": ("m30db", "m40db", "m50db"),
    "chirp": ("m40db", "m50db", "m55db"),
    "dsss": ("m30db", "m40db", "m50db"),
    "pulse": ("m30db", "m40db", "m50db"),
    "deceptive": ("m10db", "m20db", "m30db"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("inhouse", "public", "both"), default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--match-chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--signals", nargs="+", default=list(PUBLIC_SIGNALS))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-inhouse-cells", type=int, default=0)
    parser.add_argument("--max-test-normal", type=int, default=0)
    parser.add_argument("--max-test-abnormal", type=int, default=0)
    return parser.parse_args()


def metric_summary(labels: list[int], scores: list[float]) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    auroc = float(roc_auc_score(y, s) * 100.0)
    auprc = float(average_precision_score(y, s) * 100.0)
    fpr, tpr, _ = roc_curve(y, s)
    reached = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[reached[0]] * 100.0) if reached.size else 100.0
    return {"auroc": auroc, "auprc": auprc, "fpr95": fpr95}


def load_image(path: str, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = tensor.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    return (tensor - mean) / std


def load_dinov2_g(device: torch.device):
    previous = Path.cwd()
    try:
        os.chdir(UNIVAD_ROOT)
        model = torch.hub.load(
            "./models/dinov2",
            "dinov2_vitg14",
            pretrained=True,
            source="local",
        )
    finally:
        os.chdir(previous)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def encode_paths(model, paths: list[str], args: argparse.Namespace, device: torch.device):
    features = []
    for start in range(0, len(paths), args.encode_batch_size):
        batch_paths = paths[start : start + args.encode_batch_size]
        images = torch.stack([load_image(path, args.image_size) for path in batch_paths]).to(device)
        tokens = model.forward_features(images)["x_norm_patchtokens"]
        features.append(F.normalize(tokens.float(), dim=-1).cpu())
    if not features:
        raise ValueError("Cannot encode an empty path list")
    return torch.cat(features, dim=0)


@torch.inference_mode()
def score_features(
    query_features: torch.Tensor,
    reference_features: torch.Tensor,
    args: argparse.Namespace,
) -> np.ndarray:
    """Official DINO branch score: upsampled patch distance-map maximum."""

    query = query_features
    query = F.normalize(query, dim=-1)
    reference = F.normalize(reference_features, dim=-1)
    reference = reference.reshape(-1, reference.shape[-1])
    reference_t = reference.transpose(0, 1).contiguous()
    batch, patches, _ = query.shape
    best = []
    flat_query = query.reshape(-1, query.shape[-1])
    for start in range(0, flat_query.shape[0], args.match_chunk_size):
        similarity = flat_query[start : start + args.match_chunk_size] @ reference_t
        best.append(similarity.amax(dim=1))
    distance = 1.0 - torch.cat(best, dim=0).reshape(batch, patches)
    grid = int(round(patches**0.5))
    if grid * grid != patches:
        raise ValueError(f"DINO patch count is not square: {patches}")
    distance_map = F.interpolate(
        distance.reshape(batch, 1, grid, grid),
        size=args.image_size,
        mode="bilinear",
        align_corners=True,
    )
    return distance_map.flatten(1).amax(dim=1).detach().cpu().numpy().astype(np.float32)


def encode_and_score(
    model,
    paths: list[str],
    reference_features: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    description: str,
) -> np.ndarray:
    scores = []
    reference_features = reference_features.to(device)
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        images = torch.stack([load_image(path, args.image_size) for path in batch_paths]).to(device)
        with torch.inference_mode():
            query = F.normalize(
                model.forward_features(images)["x_norm_patchtokens"].float(), dim=-1
            )
            scores.extend(score_features(query, reference_features.to(device), args).tolist())
        if (start // args.batch_size + 1) % 100 == 0:
            print(f"[{description}] processed={min(start + args.batch_size, len(paths))}/{len(paths)}", flush=True)
    return np.asarray(scores, dtype=np.float32)


def encode_and_score_many(
    model,
    paths: list[str],
    references: dict[int, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    description: str,
) -> dict[int, np.ndarray]:
    """Encode each query once and score it against multiple shot memories."""

    references = {key: value.to(device) for key, value in references.items()}
    scores = {key: [] for key in references}
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        images = torch.stack([load_image(path, args.image_size) for path in batch_paths]).to(device)
        with torch.inference_mode():
            query = F.normalize(
                model.forward_features(images)["x_norm_patchtokens"].float(), dim=-1
            )
            for key, reference in references.items():
                scores[key].extend(score_features(query, reference, args).tolist())
        if (start // args.batch_size + 1) % 100 == 0:
            print(f"[{description}] processed={min(start + args.batch_size, len(paths))}/{len(paths)}", flush=True)
    return {key: np.asarray(value, dtype=np.float32) for key, value in scores.items()}


def read_inhouse_manifest() -> dict:
    return json.loads(INHOUSE_MANIFEST.read_text(encoding="utf-8"))


def read_public_manifest(k: int) -> dict:
    path = PUBLIC_MANIFEST_ROOT / f"k{k}" / "support_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("protocol") != "public_rf_fixed_test_support_pool":
        raise ValueError(f"Unexpected Public RF protocol in {path}")
    support = data["support_paths"]
    test = data["test_paths"]
    digest = lambda values: hashlib.sha256("\n".join(values).encode()).hexdigest()
    if data.get("support_paths_sha256") != digest(support) or data.get("test_paths_sha256") != digest(test):
        raise ValueError(f"Manifest digest mismatch in {path}")
    return data


def public_abnormal_paths(signal: str, jsr: str) -> list[str]:
    root = PUBLIC_ROOT / signal / "abnormal" / jsr
    paths = sorted(str(path) for path in root.glob("*.png"))
    if not paths:
        raise FileNotFoundError(root)
    return paths


def run_inhouse(model, args: argparse.Namespace, device: torch.device) -> list[dict]:
    manifest = read_inhouse_manifest()
    scene_support = {row["scene"]: [item["path"] for item in row["support"]] for row in manifest["scene_support"]}
    cells = manifest["cells"]
    if args.max_inhouse_cells > 0:
        cells = cells[: args.max_inhouse_cells]
    rows = []
    for scene, support_paths in scene_support.items():
        selected = [row for row in cells if row["scene"] == scene]
        if not selected:
            continue
        print(f"[inhouse] scene={scene} support={len(support_paths)} cells={len(selected)}", flush=True)
        support_features = encode_paths(model, support_paths, args, device)
        for index, cell in enumerate(selected, start=1):
            normal = [item["path"] for item in cell["test_normals"]]
            abnormal = [item["path"] for item in cell["test_abnormals"]]
            if args.max_test_normal > 0:
                normal = normal[: args.max_test_normal]
            if args.max_test_abnormal > 0:
                abnormal = abnormal[: args.max_test_abnormal]
            paths = normal + abnormal
            labels = [0] * len(normal) + [1] * len(abnormal)
            scores = encode_and_score(model, paths, support_features, args, device, f"inhouse {index}/{len(selected)}")
            metrics = metric_summary(labels, scores)
            rows.append({
                "method": "UniVAD-DINO-patch",
                "dataset": "inhouse_rf",
                "signal": cell["signal"],
                "scene": cell["scene"],
                "jsr": cell["jsr"],
                "support_count": len(support_paths),
                "test_normal_count": len(normal),
                "test_abnormal_count": len(abnormal),
                "metrics": metrics,
            })
        del support_features
        gc.collect()
    return rows


def run_public(model, args: argparse.Namespace, device: torch.device) -> list[dict]:
    manifests = {k: read_public_manifest(k) for k in (1, 2, 4)}
    max_paths = manifests[4]["support_paths"]
    path_to_index = {path: index for index, path in enumerate(max_paths)}
    support_features = encode_paths(model, max_paths, args, device)
    support_indices = {
        k: [path_to_index[path] for path in manifests[k]["support_paths"]]
        for k in (1, 2, 4)
    }
    rows = []
    normal_paths = list(manifests[1]["test_paths"])
    normal_scores = encode_and_score_many(
        model,
        normal_paths,
        {k: support_features[support_indices[k]] for k in (1, 2, 4)},
        args,
        device,
        "public normal",
    )
    for signal in args.signals:
        if signal not in PUBLIC_SIGNALS:
            raise ValueError(f"Unknown Public RF signal: {signal}")
        for jsr in PUBLIC_SIGNALS[signal]:
            abnormal = public_abnormal_paths(signal, jsr)
            abnormal_scores = encode_and_score_many(
                model,
                abnormal,
                {k: support_features[support_indices[k]] for k in (1, 2, 4)},
                args,
                device,
                f"public {signal}/{jsr} abnormal",
            )
            for k in (1, 2, 4):
                scores = np.concatenate([normal_scores[k], abnormal_scores[k]])
                labels = [0] * len(normal_paths) + [1] * len(abnormal)
                rows.append({
                    "method": "UniVAD-DINO-patch",
                    "dataset": "public_rf",
                    "signal": signal,
                    "scene": "RF_SPE_PNG_public",
                    "jsr": jsr,
                    "k": k,
                    "support_count": len(manifests[k]["support_paths"]),
                    "test_normal_count": len(normal_paths),
                    "test_abnormal_count": len(abnormal),
                    "metrics": metric_summary(labels, scores),
                })
    del support_features
    gc.collect()
    return rows


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.encode_batch_size < 1 or args.match_chunk_size < 1:
        raise ValueError("batch and chunk sizes must be positive")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device(
        "cuda:0" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[dino-g] device={device} dataset={args.dataset} image_size={args.image_size}", flush=True)
    model = load_dinov2_g(device)
    rows = []
    if args.dataset in ("inhouse", "both"):
        rows.extend(run_inhouse(model, args, device))
    if args.dataset in ("public", "both"):
        rows.extend(run_public(model, args, device))

    csv_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key != "metrics"}
        flat.update(row["metrics"])
        csv_rows.append(flat)
    if csv_rows:
        with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    macro = {}
    groups = {}
    for row in rows:
        group = (row["dataset"], row.get("k", "fixed24"))
        groups.setdefault(group, []).append(row["metrics"])
    for group, values in groups.items():
        macro[f"{group[0]}_{group[1]}"] = {
            key: float(np.mean([item[key] for item in values]))
            for key in ("auroc", "auprc", "fpr95")
        }
    summary = {
        "status": "complete",
        "method": "UniVAD-DINO-patch",
        "official_backbone": "DINOv2-G/14 (dinov2_vitg14)",
        "protocol_boundary": "whole-image DINO texture evidence only; no CLIP, text map, component masks, training, or test calibration",
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "encode_batch_size": args.encode_batch_size,
        "match_chunk_size": args.match_chunk_size,
        "row_count": len(rows),
        "macro_metrics": macro,
        "rows": rows,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_root / "protocol.json").write_text(
        json.dumps({**summary, "support_selection": "existing formal RF manifests"}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"row_count": len(rows), "macro_metrics": macro}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
