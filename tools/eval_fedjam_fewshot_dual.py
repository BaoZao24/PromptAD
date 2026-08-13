#!/usr/bin/env python
"""Few-shot evaluation of the formal dual-visual method on FedJam.

FedJam is stored locally as Hugging Face streaming Arrow files.  This
evaluator deliberately does not depend on the external ``datasets`` package:
it reads one Arrow record batch at a time, keeps only a tiny benign support
pool in memory, and evaluates the official test split without loading it into
RAM.

Protocol
--------
* support: deterministic reservoir sample of benign train rows only;
* shots: nested 1/2/4-shot subsets of that support pool;
* ViT: layer1+layer2 concatenated patch features, 50% farthest coreset,
  5-NN mean distance, identity/blur/time-shift paired-TTA with max fusion;
* CNN: frozen ImageNet ResNet18 layer3, complete normal patch gallery,
  top-10% patch distance;
* gate: the frozen support-only rule in ``utils.confidence_gate``;
* test: every row of the independent FedJam test split, unless a smoke-test
  per-label limit is supplied.

The RF checkpoint is used only for the frozen PromptAD/ViT encoder and text
prompt parameters.  FedJam benign images rebuild the normal galleries, so the
large RF gallery contained in the checkpoint is never used for scoring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pyarrow.ipc as pa_ipc
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PromptAD import PromptAD
from tools.eval_cls_resnet_gallery_fusion import resnet_patch_image_scores
from tools.eval_cls_vit_patch_gallery import harmonic
from tools.eval_cls_vit_patchcore_gallery import (
    prepare_patch_features,
    select_gallery_subset,
    topk_cosine_distance_chunked,
)
from tools.eval_seg_resnet_gallery_fusion import ResNet18LocalEncoder, load_checkpoint
from train_rf_target_pooled_universal import to_model_input
from utils.confidence_gate import safe_support_only_gate
from utils.training_utils import setup_seed


LABEL_NAMES = {
    0: "benign",
    1: "pulse",
    2: "single_tone",
    3: "wideband",
}
VIT_TTA_MODES = ("identity", "blur", "time_shift_up", "time_shift_down")
VIT_REFERENCE_MODES = ("time_shift_up_large", "time_shift_down_large")
CNN_REFERENCE_MODES = ("time_shift_up", "time_shift_down", "blur")


@dataclass
class RawRecord:
    """A decoded FedJam image and its integer label."""

    image_bgr: np.ndarray
    label: int
    name: str


def _image_bytes(image_value: dict) -> bytes:
    if not isinstance(image_value, dict) or not image_value.get("bytes"):
        raise ValueError("FedJam row has no embedded image bytes")
    return bytes(image_value["bytes"])


def decode_bgr(image_value: dict | bytes) -> np.ndarray:
    encoded = image_value if isinstance(image_value, (bytes, bytearray)) else _image_bytes(image_value)
    with Image.open(BytesIO(encoded)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    # The card specifies 224x224, but a few shards contain legacy rows with a
    # different encoded geometry.  Normalize the decoded view before batching;
    # the model performs its own 240x240 CLIP preprocessing afterwards.
    if rgb.shape[:2] != (224, 224):
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def iter_arrow_rows(split_dir: Path):
    """Yield image bytes and labels without materializing a split."""

    files = sorted(split_dir.glob("*.arrow"))
    if not files:
        raise FileNotFoundError(f"No Arrow shards under {split_dir}")
    for shard in files:
        with shard.open("rb") as handle:
            reader = pa_ipc.open_stream(handle)
            for batch_index, batch in enumerate(reader):
                images = batch.column("image").to_pylist()
                labels = batch.column("label").to_pylist()
                for row_index, (image, label) in enumerate(zip(images, labels)):
                    yield (
                        _image_bytes(image),
                        int(label),
                        f"{shard.name}:batch{batch_index}:row{row_index}",
                    )


def select_benign_support(data_root: Path, max_shot: int, seed: int):
    """Use one deterministic reservoir pass over train benign rows.

    This avoids the undesirable dependence on Arrow row ordering while keeping
    only ``max_shot`` encoded image blobs in memory.  The returned list is the
    common nested support pool for 1/2/4-shot evaluation.
    """

    rng = np.random.default_rng(seed)
    reservoir: list[tuple[bytes, int, str]] = []
    benign_seen = 0
    counts = {label: 0 for label in LABEL_NAMES}
    train_dir = data_root / "fedjam_dataset" / "train"
    for image_bytes, label, name in tqdm(
        iter_arrow_rows(train_dir),
        desc="Scan FedJam train benign support",
        unit="row",
    ):
        counts[label] = counts.get(label, 0) + 1
        if label != 0:
            continue
        benign_seen += 1
        item = (image_bytes, label, name)
        if len(reservoir) < max_shot:
            reservoir.append(item)
            continue
        replacement = int(rng.integers(0, benign_seen))
        if replacement < max_shot:
            reservoir[replacement] = item

    if len(reservoir) < max_shot:
        raise RuntimeError(
            f"FedJam benign support has only {len(reservoir)} rows; "
            f"requested {max_shot}"
        )
    support = [
        RawRecord(decode_bgr(image_bytes), label, name)
        for image_bytes, label, name in reservoir
    ]
    return support, counts, benign_seen


def iter_test_records(data_root: Path, max_per_label: int = 0):
    """Yield test records, optionally keeping a balanced smoke subset."""

    counts = {label: 0 for label in LABEL_NAMES}
    test_dir = data_root / "fedjam_dataset" / "test"
    for image_bytes, label, name in iter_arrow_rows(test_dir):
        if max_per_label > 0 and counts.get(label, 0) >= max_per_label:
            continue
        counts[label] = counts.get(label, 0) + 1
        yield RawRecord(decode_bgr(image_bytes), label, name)


def batch_records(records: list[RawRecord], batch_size: int):
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def raw_tensor(records: list[RawRecord]) -> torch.Tensor:
    return torch.from_numpy(np.stack([record.image_bgr for record in records], axis=0))


def _augment_bgr(image: np.ndarray, mode: str, shift_px: int = 4) -> np.ndarray:
    if mode == "identity":
        return image
    if mode == "blur":
        return cv2.GaussianBlur(image, (3, 3), 0)
    if mode in {"time_shift_up", "time_shift_down", "time_shift_up_large", "time_shift_down_large"}:
        magnitude = shift_px
        if mode.endswith("_large"):
            magnitude *= 2
        dy = -magnitude if "up" in mode else magnitude
        height, width = image.shape[:2]
        matrix = np.float32([[1, 0, 0], [0, 1, dy]])
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    raise ValueError(f"Unsupported FedJam augmentation: {mode}")


def augment_records(records: list[RawRecord], mode: str, shift_px: int) -> torch.Tensor:
    images = [_augment_bgr(record.image_bgr, mode, shift_px) for record in records]
    return torch.from_numpy(np.stack(images, axis=0))


def vit_features(model, raw: torch.Tensor, device: torch.device):
    data_t = to_model_input(model, raw, device, rgb_from_bgr=True)
    return prepare_patch_features(model.encode_image(data_t))


@torch.inference_mode()
def encode_vit_views(model, raw: torch.Tensor, args, device: torch.device):
    """Encode the four paired views, returning identity text evidence once."""

    features = {}
    text_scores = None
    for mode in VIT_TTA_MODES:
        viewed = augment_records(
            [RawRecord(image.numpy(), 0, "") for image in raw],
            mode,
            args.shift_px,
        )
        data_t = to_model_input(model, viewed, device, rgb_from_bgr=True)
        visual = model.encode_image(data_t)
        features[mode] = prepare_patch_features(visual)
        if mode == "identity":
            text_scores = np.asarray(
                model.calculate_textual_anomaly_score(visual, "cls"),
                dtype=np.float32,
            )
    return features, text_scores


def build_vit_galleries(
    model,
    support: list[RawRecord],
    shots: list[int],
    args,
    device: torch.device,
):
    """Build either paired galleries or one merged augmented normal gallery."""

    support_features = {}
    with torch.inference_mode():
        for mode in VIT_TTA_MODES:
            raw = augment_records(support, mode, args.shift_px)
            support_features[mode] = vit_features(model, raw, device).float()

    galleries = {}
    for shot in shots:
        subset_args = SimpleNamespace(
            coreset_ratio=args.coreset_ratio,
            coreset_method="farthest",
            seed=args.seed + shot,
        )
        if args.vit_tta_memory_layout == "merged":
            per_view_count = support_features[VIT_TTA_MODES[0]][:shot].numel() // support_features[VIT_TTA_MODES[0]].shape[-1]
            # Match the total number retained by four separately rounded
            # coresets.  This matters for 1-shot, where 4 * round(225 * .5)
            # is 448 rather than round(4 * 225 * .5) = 450.
            merged_keep = sum(
                max(1, int(round(per_view_count * args.coreset_ratio)))
                for _mode in VIT_TTA_MODES
            )
            features = torch.cat(
                [
                    support_features[mode][:shot].reshape(-1, support_features[mode].shape[-1])
                    for mode in VIT_TTA_MODES
                ],
                dim=0,
            )
            features = F.normalize(features.float(), dim=-1).contiguous()
            if args.coreset_ratio < 1.0:
                subset_args.coreset_ratio = merged_keep / float(features.shape[0])
                indices, _ = select_gallery_subset(features, subset_args)
                features = features[indices]
            galleries[shot] = features
            print(
                f"[vit_gallery] shot={shot} layout=merged "
                f"patches={features.shape[0]} dim={features.shape[1]}"
            )
            continue

        galleries[shot] = {}
        for mode in VIT_TTA_MODES:
            features = support_features[mode][:shot].reshape(-1, support_features[mode].shape[-1])
            features = F.normalize(features.float(), dim=-1).contiguous()
            if args.coreset_ratio < 1.0:
                indices, _ = select_gallery_subset(features, subset_args)
                features = features[indices]
            galleries[shot][mode] = features
            print(
                f"[vit_gallery] shot={shot} mode={mode} "
                f"patches={features.shape[0]} dim={features.shape[1]}"
            )
    return galleries


def build_cnn_galleries(
    encoder,
    support: list[RawRecord],
    shots: list[int],
    device: torch.device,
):
    with torch.inference_mode():
        raw = raw_tensor(support).permute(0, 3, 1, 2).to(device, non_blocking=True)
        features = encoder(raw)["layer3"].float()

    galleries = {}
    for shot in shots:
        gallery = features[:shot].permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        galleries[shot] = F.normalize(gallery.float(), dim=1).contiguous()
        print(
            f"[cnn_gallery] shot={shot} layer=layer3 "
            f"patches={galleries[shot].shape[0]} dim={galleries[shot].shape[1]}"
        )
    return galleries


@torch.inference_mode()
def vit_query_scores(patch_features: torch.Tensor, galleries: dict, args):
    """Return paired-TTA ViT max-patch scores for one query batch."""

    if args.vit_tta_memory_layout == "merged":
        probes = patch_features["identity"].reshape(-1, patch_features["identity"].shape[-1])
        distances = topk_cosine_distance_chunked(
            probes,
            galleries,
            args.distance_chunk_size,
            args.nn_topk,
        ).reshape(patch_features["identity"].shape[0], -1)
        return distances.max(dim=1).values.cpu().numpy().astype(np.float32)

    scores = []
    for mode in VIT_TTA_MODES:
        gallery = galleries[mode]
        probes = patch_features[mode].reshape(-1, patch_features[mode].shape[-1])
        distances = topk_cosine_distance_chunked(
            probes,
            gallery,
            args.distance_chunk_size,
            args.nn_topk,
        ).reshape(patch_features[mode].shape[0], -1)
        scores.append(distances.max(dim=1).values)
    return torch.stack(scores, dim=0).max(dim=0).values.cpu().numpy().astype(np.float32)


@torch.inference_mode()
def score_vit_batch(model, raw: torch.Tensor, gallery: dict, args, device: torch.device):
    if args.vit_tta_memory_layout == "merged":
        # A merged gallery is queried once.  ``raw`` may already be a held-out
        # shifted support view when constructing the fixed gate reference.
        features = {"identity": vit_features(model, raw, device)}
    else:
        features, _text_scores = encode_vit_views(model, raw, args, device)
    scores = vit_query_scores(features, gallery, args)
    return scores


@torch.inference_mode()
def score_cnn_batch(encoder, raw: torch.Tensor, gallery: torch.Tensor, args, device):
    input_tensor = raw.permute(0, 3, 1, 2).to(device, non_blocking=True)
    layer3 = encoder(input_tensor)["layer3"]
    return resnet_patch_image_scores(
        layer3,
        gallery,
        args.distance_chunk_size,
        args.cnn_top_ratio,
        args.cnn_nn_topk,
    ).astype(np.float32)


def support_reference_scores(
    model,
    encoder,
    support: list[RawRecord],
    shot: int,
    vit_gallery: dict,
    cnn_gallery: torch.Tensor,
    args,
    device: torch.device,
):
    """Score held-out support views for the fixed inductive gate."""

    vit_reference = []
    cnn_reference = []
    selected = support[:shot]
    with torch.inference_mode():
        for mode in VIT_REFERENCE_MODES:
            raw = augment_records(selected, mode, args.shift_px)
            # The formal paired-TTA evaluator applies the same paired query
            # views against the corresponding support galleries, then maxes.
            vit_values = score_vit_batch(model, raw, vit_gallery, args, device)
            vit_reference.extend(vit_values.tolist())

        for mode in CNN_REFERENCE_MODES:
            raw = augment_records(selected, mode, args.shift_px)
            cnn_reference.extend(
                score_cnn_batch(encoder, raw, cnn_gallery, args, device).tolist()
            )
    return np.asarray(vit_reference, dtype=np.float32), np.asarray(cnn_reference, dtype=np.float32)


def metric(labels, scores):
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "fpr95": float("nan")}
    auroc = float(roc_auc_score(labels, scores) * 100.0)
    auprc = float(average_precision_score(labels, scores) * 100.0)
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[reached[0]] * 100.0) if reached.size else 100.0
    return {"auroc": auroc, "auprc": auprc, "fpr95": fpr95}


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_metric_rows(rows, shot, method, labels, scores):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)
    scopes = [("overall", labels != 0)]
    for label_id, name in LABEL_NAMES.items():
        if label_id == 0:
            continue
        scopes.append((name, labels == label_id))
    per_attack = []
    for scope, attack_mask in scopes:
        if scope == "overall":
            mask = np.ones_like(labels, dtype=bool)
            binary = (labels != 0).astype(np.int32)
        else:
            mask = (labels == 0) | attack_mask
            binary = attack_mask[mask].astype(np.int32)
        values = metric(binary, scores[mask])
        row = {
            "shot": int(shot),
            "method": method,
            "scope": scope,
            "n": int(mask.sum()),
            "num_normal": int((labels[mask] == 0).sum()),
            "num_abnormal": int(binary.sum()),
            **values,
        }
        rows.append(row)
        if scope != "overall":
            per_attack.append(values)
    rows.append(
        {
            "shot": int(shot),
            "method": method,
            "scope": "macro_attack",
            "n": int(labels.size),
            "num_normal": int((labels == 0).sum()),
            "num_abnormal": int((labels != 0).sum()),
            "auroc": float(np.nanmean([item["auroc"] for item in per_attack])),
            "auprc": float(np.nanmean([item["auprc"] for item in per_attack])),
            "fpr95": float(np.nanmean([item["fpr95"] for item in per_attack])),
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/mnt/data/wangbei/data/FedJam",
        help="Directory containing fedjam_dataset/train and fedjam_dataset/test",
    )
    parser.add_argument(
        "--output-root",
        default="analysis_outputs/20260803_fedjam_fewshot_dual",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "analysis_outputs/02_current_baselines/promptad_formal_baseline/"
            "pooled_rf_rgb_cls/checkpoint/overall-best.pt"
        ),
    )
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-test-per-label", type=int, default=0)
    parser.add_argument("--distance-chunk-size", type=int, default=1024)
    parser.add_argument("--nn-topk", type=int, default=5)
    parser.add_argument("--cnn-top-ratio", type=float, default=0.1)
    parser.add_argument("--cnn-nn-topk", type=int, default=1)
    parser.add_argument("--coreset-ratio", type=float, default=0.5)
    parser.add_argument("--shift-px", type=int, default=4)
    parser.add_argument(
        "--vit-tta-memory-layout",
        choices=["separate", "merged"],
        default="separate",
        help=(
            "separate: query each augmented view against its paired normal gallery; "
            "merged: combine augmented normal features and query each test view once."
        ),
    )
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--include-promptad-text",
        action="store_true",
        help="Include the legacy PromptAD text score as a diagnostic row; off by default.",
    )
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument("--img-resize", type=int, default=240)
    parser.add_argument("--img-cropsize", type=int, default=240)
    args = parser.parse_args()

    if not args.shots or min(args.shots) <= 0:
        raise ValueError("--shots must contain positive integers")
    if not (0.0 < args.coreset_ratio <= 1.0):
        raise ValueError("--coreset-ratio must be in (0, 1]")
    shots = sorted(set(int(value) for value in args.shots))
    max_shot = max(shots)
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    setup_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if not args.use_cpu and torch.cuda.is_available():
        # The launcher can additionally set CUDA_VISIBLE_DEVICES.  Keeping
        # one explicit device here prevents accidental multi-GPU replication.
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"[device] {device} cuda_count={torch.cuda.device_count()}")

    support, train_counts, benign_seen = select_benign_support(data_root, max_shot, args.seed)
    support_manifest = {
        "seed": args.seed,
        "shots": shots,
        "train_counts": train_counts,
        "benign_seen": benign_seen,
        "support": [
            {
                "name": record.name,
                "label": record.label,
                "sha256": hashlib.sha256(record.image_bgr.tobytes()).hexdigest(),
                "shape": list(record.image_bgr.shape),
            }
            for record in support
        ],
    }
    (output_root / "support_manifest.json").write_text(
        json.dumps(support_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[support] selected={len(support)} benign_seen={benign_seen} counts={train_counts}")

    # The checkpoint was trained with a very large RF gallery.  Instantiate a
    # one-shot buffer and then replace every normal gallery with FedJam
    # support features.  Checkpoint text/prompt weights are retained.
    model_kwargs = {
        "dataset": "rf_target_test_pool",
        "class_name": "signal",
        "device": str(device),
        "out_size_h": args.resolution,
        "out_size_w": args.resolution,
        "backbone": "ViT-B-16-plus-240",
        "pretrained_dataset": "laion400m_e32",
        "n_ctx": 4,
        "n_pro": 3,
        "n_ctx_ab": 1,
        "n_pro_ab": 4,
        "precision": "fp16",
        "k_shot": 1,
        "img_resize": args.img_resize,
        "img_cropsize": args.img_cropsize,
        "prompt_mode": "rf",
        "input_mode": "rgb",
        "text_prototype_mode": "single",
        "cls_score_mode": "text_only",
    }
    model = PromptAD(**model_kwargs).to(device)
    load_checkpoint(model, str(Path(args.checkpoint).resolve()))
    model.eval_mode()
    encoder = ResNet18LocalEncoder().to(device).eval()

    vit_galleries = build_vit_galleries(model, support, shots, args, device)
    cnn_galleries = build_cnn_galleries(encoder, support, shots, device)

    references = {}
    for shot in shots:
        vit_reference, cnn_reference = support_reference_scores(
            model,
            encoder,
            support,
            shot,
            vit_galleries[shot],
            cnn_galleries[shot],
            args,
            device,
        )
        references[shot] = {"vit": vit_reference, "cnn": cnn_reference}
        print(
            f"[reference] shot={shot} vit_n={len(vit_reference)} "
            f"cnn_n={len(cnn_reference)}"
        )

    # Score all shots in one pass over test.  Image encoder features are
    # computed once per TTA view and reused for each shot's small gallery.
    score_bank = {
        shot: {
            "vit_patchcore_max_scores": [],
            "resnet18_layer3_top0.1_scores": [],
            "confidence_gated_score": [],
            "cnn_gate": [],
            "cnn_rank": [],
        }
        for shot in shots
    }
    if args.include_promptad_text:
        for shot in shots:
            score_bank[shot]["promptad_text_scores"] = []
    labels = []
    names = []
    test_counts = {label: 0 for label in LABEL_NAMES}
    test_records = iter_test_records(data_root, args.max_test_per_label)
    with torch.inference_mode():
        pending = []
        for record in tqdm(test_records, desc="Evaluate FedJam test", unit="row"):
            pending.append(record)
            if len(pending) < args.batch_size:
                continue
            raw = raw_tensor(pending)
            batch_labels = [record.label for record in pending]
            batch_names = [record.name for record in pending]

            if args.vit_tta_memory_layout == "merged":
                visual = model.encode_image(to_model_input(model, raw, device, rgb_from_bgr=True))
                vit_features_by_mode = {"identity": prepare_patch_features(visual)}
                text_scores = np.asarray(
                    model.calculate_textual_anomaly_score(visual, "cls"),
                    dtype=np.float32,
                )
            else:
                vit_features_by_mode, text_scores = encode_vit_views(
                    model, raw, args, device
                )
            cnn_raw = raw.permute(0, 3, 1, 2).to(device, non_blocking=True)
            cnn_layer3 = encoder(cnn_raw)["layer3"]

            for shot in shots:
                vit_scores = vit_query_scores(vit_features_by_mode, vit_galleries[shot], args)
                cnn_scores = resnet_patch_image_scores(
                    cnn_layer3,
                    cnn_galleries[shot],
                    args.distance_chunk_size,
                    args.cnn_top_ratio,
                    args.cnn_nn_topk,
                ).astype(np.float32)
                gated = safe_support_only_gate(
                    vit_scores,
                    cnn_scores,
                    references[shot]["vit"],
                    references[shot]["cnn"],
                )
                score_bank[shot]["vit_patchcore_max_scores"].extend(vit_scores.tolist())
                score_bank[shot]["resnet18_layer3_top0.1_scores"].extend(cnn_scores.tolist())
                if args.include_promptad_text:
                    score_bank[shot]["promptad_text_scores"].extend(text_scores.tolist())
                score_bank[shot]["confidence_gated_score"].extend(gated["score"].astype(np.float32).tolist())
                score_bank[shot]["cnn_gate"].extend(gated["gate"].astype(np.float32).tolist())
                score_bank[shot]["cnn_rank"].extend(gated["cnn_rank"].astype(np.float32).tolist())

            labels.extend(batch_labels)
            names.extend(batch_names)
            for label in batch_labels:
                test_counts[label] = test_counts.get(label, 0) + 1
            pending = []

        if pending:
            raw = raw_tensor(pending)
            batch_labels = [record.label for record in pending]
            batch_names = [record.name for record in pending]
            if args.vit_tta_memory_layout == "merged":
                visual = model.encode_image(to_model_input(model, raw, device, rgb_from_bgr=True))
                vit_features_by_mode = {"identity": prepare_patch_features(visual)}
                text_scores = np.asarray(
                    model.calculate_textual_anomaly_score(visual, "cls"),
                    dtype=np.float32,
                )
            else:
                vit_features_by_mode, text_scores = encode_vit_views(
                    model, raw, args, device
                )
            cnn_layer3 = encoder(raw.permute(0, 3, 1, 2).to(device, non_blocking=True))["layer3"]
            for shot in shots:
                vit_scores = vit_query_scores(vit_features_by_mode, vit_galleries[shot], args)
                cnn_scores = resnet_patch_image_scores(
                    cnn_layer3,
                    cnn_galleries[shot],
                    args.distance_chunk_size,
                    args.cnn_top_ratio,
                    args.cnn_nn_topk,
                ).astype(np.float32)
                gated = safe_support_only_gate(
                    vit_scores,
                    cnn_scores,
                    references[shot]["vit"],
                    references[shot]["cnn"],
                )
                score_bank[shot]["vit_patchcore_max_scores"].extend(vit_scores.tolist())
                score_bank[shot]["resnet18_layer3_top0.1_scores"].extend(cnn_scores.tolist())
                if args.include_promptad_text:
                    score_bank[shot]["promptad_text_scores"].extend(text_scores.tolist())
                score_bank[shot]["confidence_gated_score"].extend(gated["score"].astype(np.float32).tolist())
                score_bank[shot]["cnn_gate"].extend(gated["gate"].astype(np.float32).tolist())
                score_bank[shot]["cnn_rank"].extend(gated["cnn_rank"].astype(np.float32).tolist())
            labels.extend(batch_labels)
            names.extend(batch_names)
            for label in batch_labels:
                test_counts[label] = test_counts.get(label, 0) + 1

    labels_np = np.asarray(labels, dtype=np.int32)
    names_np = np.asarray(names)
    if labels_np.size == 0:
        raise RuntimeError("FedJam test split produced no rows")
    print(f"[test] counts={test_counts} total={labels_np.size}")

    result_rows = []
    summary = {
        "method": "fedjam_fewshot_dual_visual",
        "data_root": str(data_root),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "shots": shots,
        "test_counts": test_counts,
        "train_counts": train_counts,
        "support_only": True,
        "test_batch_statistics_used": False,
        "vit_tta_modes": list(VIT_TTA_MODES),
        "vit_tta_memory_layout": args.vit_tta_memory_layout,
        "vit_reference_modes": list(VIT_REFERENCE_MODES),
        "cnn_reference_modes": list(CNN_REFERENCE_MODES),
        "vit_feature": "normalized ViT layer1+layer2 concat",
        "cnn_feature": "ImageNet ResNet18 layer3",
        "vit_coreset_ratio": args.coreset_ratio,
        "vit_nn_topk": args.nn_topk,
        "cnn_top_ratio": args.cnn_top_ratio,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "include_promptad_text": bool(args.include_promptad_text),
    }
    score_dir = output_root / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    for shot in shots:
        arrays = {key: np.asarray(values, dtype=np.float32) for key, values in score_bank[shot].items()}
        np.savez_compressed(
            score_dir / f"fedjam_{shot}shot_scores.npz",
            labels=labels_np,
            names=names_np,
            **arrays,
        )
        method_map = {
            "vit_patchcore": "vit_patchcore_max_scores",
            "cnn_layer3_top0.1": "resnet18_layer3_top0.1_scores",
            "confidence_gated_dual_visual": "confidence_gated_score",
        }
        if args.include_promptad_text:
            method_map = {
                "promptad_text": "promptad_text_scores",
                **method_map,
            }
        for method_name, key in method_map.items():
            add_metric_rows(result_rows, shot, method_name, labels_np, arrays[key])
        overall = {}
        for method_name, key in method_map.items():
            overall[method_name] = metric((labels_np != 0).astype(np.int32), arrays[key])
            overall[method_name]["gate_active_rate"] = float(np.mean(arrays["cnn_gate"] > 0.0)) if method_name == "confidence_gated_dual_visual" else None
        summary[f"shot_{shot}"] = overall

    write_csv(output_root / "metrics.csv", result_rows)
    (output_root / "protocol.json").write_text(
        json.dumps(
            {
                **summary,
                "label_names": LABEL_NAMES,
                "support_selection": "seeded reservoir over train label=benign; nested prefix for shots",
                "test_selection": "all official test rows unless max-test-per-label is set",
                "score_files": [str(path.relative_to(output_root)) for path in sorted(score_dir.glob("*.npz"))],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if device.type == "cuda":
        print(f"[gpu] max_allocated_mb={torch.cuda.max_memory_allocated(device) / 1024**2:.1f}")
    print(f"wrote {output_root / 'metrics.csv'}")
    print(f"wrote {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
