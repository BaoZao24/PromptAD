#!/usr/bin/env python
"""Evaluate the FADE few-shot anomaly detector on the project protocols.

This adapter uses the method proposed by FADE (BMVC 2024): a frozen CLIP/GEM
image encoder, a normal-only patch memory bank, one-nearest-neighbour cosine
distance, and the optional zero-shot language score.  It deliberately does
not train or fine-tune on any target-domain image.

The original FADE repository was written against an older ``open_clip`` API.
The current environment has a newer API, so the model is constructed with
keyword arguments and then wrapped by the official GEMWrapper.  This keeps
the feature and attention implementation from ``gem-torch`` unchanged while
avoiding positional-argument drift.

Supported project protocols:
  * rf_target: in-house target-scene RF;
  * public_rf: public RF nested k-per-frequency support manifests;
  * ofdma: formal target-scene v2 realistic cold-start protocol;
  * fedjam: independent full-test FedJam protocol.

The default configuration is a resource-controlled, fair image-level setting:
ViT-B/16 + OpenAI weights, 224x224 CLIP input, one visual scale, CLIP patch
features, and official FADE ``both`` fusion (language score averaged with the
vision-guided score).  ``--mode vision`` can be used to isolate the
few-shot memory-bank branch.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.gem_wrapper import GEMWrapper  # noqa: E402

from tools.eval_fedjam_fewshot_dual import (  # noqa: E402
    LABEL_NAMES,
    RawRecord,
    iter_test_records,
    select_benign_support,
)
from tools.eval_ofdma_target_scene_baselines import (  # noqa: E402
    aggregate_observations,
    build_jobs as build_ofdma_jobs,
    source_protocol,
)
from tools.eval_patchcore_cls import (  # noqa: E402
    JSR_BY_SIGNAL,
    PUBLIC_RF_JSRS,
    PUBLIC_RF_ROOT,
    SCENES,
    load_sample_image,
    make_sample,
    public_rf_jobs,
    rf_target_jobs,
)
from datasets.ofdma_target_scene import (  # noqa: E402
    available_target_scenes,
    load_target_scene_manifest,
)
from utils.training_utils import setup_seed  # noqa: E402


OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)
DEFAULT_PROMPT_PATH = REPO_ROOT / "references" / "FADE" / "prompts" / "winclip_prompt.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FADE few-shot baseline on the SpectraMemAD protocols"
    )
    parser.add_argument(
        "--protocol", choices=("rf_target", "public_rf", "ofdma", "fedjam"), required=True
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--data-root", default="/mnt/data/wangbei/data/FedJam")
    parser.add_argument("--support-manifest", default="")
    parser.add_argument(
        "--normal-sampling",
        choices=("all", "per_frequency", "per_frequency_k"),
        default="per_frequency",
    )
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=2)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--rf-signals", nargs="+", default=list(JSR_BY_SIGNAL))
    parser.add_argument("--rf-scenes", nargs="+", default=list(SCENES))
    parser.add_argument("--public-rf-signals", nargs="+", default=list(PUBLIC_RF_JSRS))

    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--scene-ids", nargs="*", default=[])
    parser.add_argument("--max-normal-observations", type=int, default=0)
    parser.add_argument("--max-anomaly-observations-per-type", type=int, default=0)

    parser.add_argument("--max-test-per-label", type=int, default=0)
    parser.add_argument("--model-name", default="ViT-B-16")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--model-cache-dir", default="models")
    parser.add_argument("--gem-depth", type=int, default=7)
    parser.add_argument("--feature-sizes", default="224")
    parser.add_argument("--classification-size", type=int, default=224)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--vision-feature", choices=("clip", "gem"), default="clip")
    parser.add_argument("--language-feature", choices=("clip", "gem"), default="clip")
    parser.add_argument("--mode", choices=("vision", "language", "both"), default="both")
    parser.add_argument("--prompt-class", default="radio frequency spectrogram")
    parser.add_argument("--prompt-path", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--vision-segmentation-multiplier", type=float, default=3.5)
    parser.add_argument("--reference-chunk-size", type=int, default=4096)
    parser.add_argument("--max-query-patches", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "fpr95": float("nan")}
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels, scores) * 100.0),
        "auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if reached.size else float("nan"),
    }


def load_fedjam_image(sample: dict) -> Image.Image:
    image_bgr = np.asarray(sample["image_bgr"], dtype=np.uint8)
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Unexpected FedJam image shape: {image_bgr.shape}")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")


def sample_image(sample: dict) -> Image.Image:
    if sample.get("dataset") == "fedjam":
        return load_fedjam_image(sample)
    return load_sample_image(sample)


class FadeDataset(Dataset):
    """Return the same sample in each requested square CLIP resolution."""

    def __init__(self, samples: list[dict], sizes: tuple[int, ...], resize: int):
        self.samples = list(samples)
        self.sizes = tuple(int(size) for size in sizes)
        self.transforms = {}
        for size in self.sizes:
            first_resize = max(int(resize), int(size))
            self.transforms[size] = transforms.Compose(
                [
                    transforms.Resize(
                        first_resize,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.CenterCrop(int(size)),
                    transforms.ToTensor(),
                    transforms.Normalize(OPENAI_MEAN, OPENAI_STD),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = sample_image(sample)
        return {
            "image": {size: self.transforms[size](image) for size in self.sizes},
            "label": int(sample["label"]),
            "name": str(sample["name"]),
            "path": str(sample["path"]),
        }


def parse_sizes(value: str, classification_size: int) -> tuple[int, ...]:
    sizes = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not sizes or any(size < 32 for size in sizes):
        raise ValueError(f"Invalid --feature-sizes={value!r}")
    if classification_size not in sizes:
        sizes.append(int(classification_size))
        sizes.sort()
    return tuple(sizes)


def create_fade_model(args, device: torch.device):
    """Create the official GEM wrapper with the current open_clip API."""

    model_name = str(args.model_name).replace("/", "-")
    clip_model = open_clip.create_model(
        model_name,
        pretrained=args.pretrained,
        precision="fp32",
        device=device,
        # OpenAI's original ViT-B/16 checkpoint uses QuickGELU.  The current
        # open_clip registry otherwise constructs the non-QuickGELU variant
        # and only emits a warning, which would silently make this baseline
        # use a mismatched architecture.
        force_quick_gelu=(str(args.pretrained).lower() == "openai"),
        cache_dir=args.model_cache_dir,
    )
    tokenizer = open_clip.get_tokenizer(model_name=model_name)
    model = GEMWrapper(
        model=clip_model,
        tokenizer=tokenizer,
        depth=int(args.gem_depth),
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def extract_visual(model, batch_images: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    gem_features, clip_features = model.model.visual(
        batch_images.to(device, non_blocking=device.type == "cuda")
    )
    return {
        "gem": {
            "cls": F.normalize(gem_features[:, 0, :], dim=-1),
            "patch": F.normalize(gem_features[:, 1:, :], dim=-1),
        },
        "clip": {
            "cls": F.normalize(clip_features[:, 0, :], dim=-1),
            "patch": F.normalize(clip_features[:, 1:, :], dim=-1),
        },
    }


def make_loader(samples: list[dict], args, sizes: tuple[int, ...]) -> DataLoader:
    dataset = FadeDataset(samples, sizes, args.resize)
    return DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=False,
        persistent_workers=False,
    )


def build_gallery(
    model,
    samples: list[dict],
    args,
    sizes: tuple[int, ...],
    device: torch.device,
) -> dict[int, torch.Tensor]:
    if not samples:
        raise ValueError("FADE requires at least one normal support image")
    loader = make_loader(samples, args, sizes)
    parts: dict[int, list[torch.Tensor]] = {size: [] for size in sizes}
    for batch in tqdm(loader, desc=f"FADE gallery ({len(samples)} support)", leave=False):
        for size in sizes:
            features = extract_visual(model, batch["image"][size], device)
            parts[size].append(features[args.vision_feature]["patch"].detach().cpu())
    gallery = {}
    for size in sizes:
        tensor = torch.cat(parts[size], dim=0)
        gallery[size] = tensor.reshape(-1, tensor.shape[-1]).contiguous()
    return gallery


def stable_binary_softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits - logits.max(dim=1, keepdim=True).values, dim=1)


def build_language_prototypes(model, args, device: torch.device) -> torch.Tensor:
    prompt_path = Path(args.prompt_path)
    if not prompt_path.is_file():
        raise FileNotFoundError(prompt_path)
    prompts = json.loads(prompt_path.read_text())
    normal = [item.format(classname=args.prompt_class) for item in prompts["normal"]["prompts"]]
    abnormal = [item.format(classname=args.prompt_class) for item in prompts["abnormal"]["prompts"]]
    with torch.inference_mode():
        normal_embeddings = model.encode_text(normal).squeeze(0).to(device)
        abnormal_embeddings = model.encode_text(abnormal).squeeze(0).to(device)
        prototypes = torch.stack([normal_embeddings.mean(0), abnormal_embeddings.mean(0)])
    return prototypes


@torch.inference_mode()
def language_scores(
    image_features: dict[str, torch.Tensor],
    prototypes: torch.Tensor,
    feature: str,
) -> torch.Tensor:
    image_cls = image_features[feature]["cls"]
    logits = 100.0 * image_cls @ prototypes.t()
    return stable_binary_softmax(logits)[:, 1]


def patch_distance_to_gallery(
    query: torch.Tensor,
    gallery: torch.Tensor,
    reference_chunk_size: int,
) -> torch.Tensor:
    """Return top-1 ``(1-cosine)/2`` distance for every query patch.

    The query is ``[B, P, D]`` and the gallery is ``[N, D]``.  Reference
    chunks keep the ``B*P*N`` similarity tensor bounded, which matters for
    OFDMA's large test set.
    """

    if query.ndim != 3 or gallery.ndim != 2:
        raise ValueError(f"Unexpected patch shapes: query={query.shape}, gallery={gallery.shape}")
    best = torch.full(
        (query.shape[0], query.shape[1]),
        -1.0,
        device=query.device,
        dtype=query.dtype,
    )
    chunk = max(1, int(reference_chunk_size))
    for start in range(0, gallery.shape[0], chunk):
        ref = gallery[start : start + chunk].to(query.device, non_blocking=query.device.type == "cuda")
        similarity = torch.matmul(query, ref.t()).amax(dim=-1)
        best = torch.maximum(best, similarity)
    return 0.5 * (1.0 - best)


@torch.inference_mode()
def vision_scores(
    image_features_by_size: dict[int, dict[str, dict[str, torch.Tensor]]],
    galleries: dict[int, torch.Tensor],
    args,
) -> torch.Tensor:
    maps = []
    target_grid = max(size // 16 for size in galleries)
    for size in sorted(galleries):
        patch_features = image_features_by_size[size][args.vision_feature]["patch"]
        patch_scores = patch_distance_to_gallery(
            patch_features,
            galleries[size],
            args.reference_chunk_size,
        )
        grid = int(round(math.sqrt(patch_scores.shape[1])))
        if grid * grid != patch_scores.shape[1]:
            raise RuntimeError(
                f"FADE expects a square patch grid, got size={size}, patches={patch_scores.shape[1]}"
            )
        score_map = patch_scores.reshape(-1, 1, grid, grid)
        if grid != target_grid:
            score_map = F.interpolate(
                score_map,
                size=(target_grid, target_grid),
                mode="bilinear",
                align_corners=False,
            )
        maps.append(score_map[:, 0])
    merged = torch.stack(maps, dim=0).mean(dim=0)
    return torch.clamp(
        merged.amax(dim=(-1, -2)) * float(args.vision_segmentation_multiplier),
        min=0.0,
        max=1.0,
    )


@torch.inference_mode()
def score_job(
    model,
    job: dict,
    galleries: dict[int, torch.Tensor] | None,
    language_prototypes: torch.Tensor | None,
    args,
    sizes: tuple[int, ...],
    device: torch.device,
) -> dict[str, np.ndarray]:
    loader = make_loader(job["test_samples"], args, sizes)
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_names: list[str] = []
    all_paths: list[str] = []
    offset = 0
    for batch in tqdm(
        loader,
        desc=f"FADE test {job['category']} {job['scene']} {job['jsr']}",
        leave=False,
    ):
        feature_by_size = {}
        for size in sizes:
            feature_by_size[size] = extract_visual(model, batch["image"][size], device)
        parts = []
        if args.mode in {"vision", "both"}:
            if galleries is None:
                raise RuntimeError("Vision mode requires a support gallery")
            parts.append(vision_scores(feature_by_size, galleries, args))
        if args.mode in {"language", "both"}:
            if language_prototypes is None:
                raise RuntimeError("Language mode requires text prototypes")
            class_size = int(args.classification_size)
            parts.append(
                language_scores(feature_by_size[class_size], language_prototypes, args.language_feature)
            )
        if args.mode == "both":
            scores = torch.stack(parts, dim=0).mean(dim=0)
        else:
            scores = parts[0]
        all_scores.append(scores.detach().cpu().numpy().astype(np.float32))
        all_labels.append(np.asarray(batch["label"], dtype=np.int32))
        names = [str(value) for value in batch["name"]]
        paths = [str(value) for value in batch["path"]]
        all_names.extend(names)
        all_paths.extend(paths)
        offset += len(names)
    if offset != len(job["test_samples"]):
        raise RuntimeError(f"FADE test count mismatch: scored={offset} expected={len(job['test_samples'])}")
    return {
        "scores": np.concatenate(all_scores).astype(np.float32),
        "labels": np.concatenate(all_labels).astype(np.int32),
        "names": np.asarray(all_names),
        "paths": np.asarray(all_paths),
    }


def support_signature(job: dict) -> tuple[str, ...]:
    return tuple(str(sample["path"]) for sample in job["train_samples"])


def build_rf_jobs(args) -> list[dict]:
    common = SimpleNamespace(
        output_root=args.output_root,
        support_manifest=args.support_manifest,
        normal_sampling=args.normal_sampling,
        seed=args.seed,
        max_test_normals=args.max_test_normals,
        max_abnormals=args.max_abnormals,
        max_train_normals=0,
        spectrum_root=str(REPO_ROOT / "datasets" / "spectrum"),
        spectrum_categories=[],
        rf_signals=list(args.rf_signals),
        public_rf_signals=list(args.public_rf_signals),
        rf_scenes=list(args.rf_scenes),
        support_bootstrap_seed=-1,
    )
    if args.protocol == "public_rf":
        return public_rf_jobs(common)
    return rf_target_jobs(common)


def build_ofdma_protocol_jobs(args) -> tuple[list[dict], list[str], dict]:
    from tools.eval_ofdma_target_scene_baselines import build_jobs as official_build_jobs

    dataset_root = Path(args.dataset_root)
    manifest = load_target_scene_manifest(dataset_root)
    source = source_protocol(dataset_root)
    normalization = source["source_protocol"]["normalization"]
    ofdma_args = SimpleNamespace(
        dataset_root=str(dataset_root),
        split=args.split,
        scene_ids=list(args.scene_ids),
        shots=list(args.shots),
        max_normal_observations=args.max_normal_observations,
        max_anomaly_observations_per_type=args.max_anomaly_observations_per_type,
    )
    jobs, scene_ids = official_build_jobs(ofdma_args, manifest, normalization)
    return jobs, scene_ids, source


def build_fedjam_jobs(args) -> tuple[list[dict], dict]:
    data_root = Path(args.data_root)
    support_pool, support_counts, benign_seen = select_benign_support(
        data_root,
        max_shot=max(args.shots),
        seed=args.seed,
    )
    test_records = list(iter_test_records(data_root, max_per_label=args.max_test_per_label))
    test_samples = [
        {
            "path": f"fedjam/test/{record.name}",
            "label": int(record.label != 0),
            "category": "fedjam",
            "name": str(record.name),
            "dataset": "fedjam",
            "image_bgr": np.ascontiguousarray(record.image_bgr),
            "fedjam_label": int(record.label),
        }
        for record in test_records
    ]
    jobs = []
    for shot in sorted(set(args.shots)):
        train_samples = [
            {
                "path": f"fedjam/train/{record.name}",
                "label": 0,
                "category": "fedjam",
                "name": str(record.name),
                "dataset": "fedjam",
                "image_bgr": np.ascontiguousarray(record.image_bgr),
            }
            for record in support_pool[:shot]
        ]
        jobs.append(
            {
                "dataset": "fedjam",
                "category": "fedjam",
                "scene": "independent_test",
                "jsr": f"{shot}shot",
                "shot": shot,
                "train_samples": train_samples,
                "test_samples": test_samples,
            }
        )
    metadata = {
        "support_counts": support_counts,
        "benign_seen": benign_seen,
        "test_counts": {
            str(label): int(sum(record.label == label for record in test_records))
            for label in sorted(LABEL_NAMES)
        },
    }
    return jobs, metadata


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_payload(output_root: Path, job: dict, payload: dict[str, np.ndarray]) -> None:
    score_dir = output_root / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{job['dataset']}-{job['category']}-{job['scene']}-{job['jsr']}".replace("/", "_")
    np.savez_compressed(
        score_dir / f"{stem}-scores.npz",
        scores=payload["scores"],
        labels=payload["labels"],
        names=payload["names"],
        paths=payload["paths"],
    )


def rf_rows(job: dict, payload: dict[str, np.ndarray], method_name: str) -> list[dict]:
    values = metric(payload["labels"], payload["scores"])
    return [
        {
            "row_type": "cell",
            "protocol": job["dataset"],
            "category": job["category"],
            "scene": job["scene"],
            "jsr": job["jsr"],
            "shot": job.get("shot", ""),
            "method": method_name,
            "num_train_normal": len(job["train_samples"]),
            "num_test": len(payload["labels"]),
            **values,
        }
    ]


def append_rf_macro(rows: list[dict], method_name: str) -> list[dict]:
    cells = [row for row in rows if row["row_type"] == "cell"]
    if not cells:
        return list(rows)
    result = list(rows)
    metrics = {
        key: float(np.nanmean([row[key] for row in cells]))
        for key in ("auroc", "auprc", "fpr95")
    }
    result.append(
        {
            "row_type": "macro",
            "protocol": cells[0]["protocol"],
            "category": "ALL",
            "scene": "macro",
            "jsr": "macro",
            "shot": "",
            "method": method_name,
            "num_train_normal": "",
            "num_test": int(sum(row["num_test"] for row in cells)),
            **metrics,
        }
    )
    return result


def fedjam_rows(job: dict, payload: dict[str, np.ndarray], method_name: str) -> list[dict]:
    labels = np.asarray(payload["labels"], dtype=np.int32)
    scores = np.asarray(payload["scores"], dtype=np.float32)
    rows = []
    scopes = [("overall", np.ones(labels.shape[0], dtype=bool))]
    raw_labels = np.asarray([sample.get("fedjam_label", int(sample["label"])) for sample in job["test_samples"]])
    for label_id, label_name in LABEL_NAMES.items():
        if label_id != 0:
            scopes.append((label_name, (raw_labels == 0) | (raw_labels == label_id)))
    for scope, mask in scopes:
        binary = (raw_labels[mask] != 0).astype(np.int32)
        rows.append(
            {
                "row_type": "cell",
                "protocol": "fedjam",
                "category": "fedjam",
                "scene": job["scene"],
                "jsr": job["jsr"],
                "shot": job["shot"],
                "scope": scope,
                "method": method_name,
                "num_train_normal": len(job["train_samples"]),
                "num_test": int(mask.sum()),
                **metric(binary, scores[mask]),
            }
        )
    return rows


def write_protocol(args, output_root: Path, sizes: tuple[int, ...], model_info: dict, extra: dict):
    protocol = {
        "entry": "tools/eval_fade_cls.py",
        "method": "fade",
        "method_name": f"fade_{args.mode}_{args.vision_feature}",
        "date": "2026-08-23",
        "project_protocol": args.protocol,
        "support_rule": "normal-only support; no target-domain training or fine-tuning",
        "test_label_usage": "metrics_only",
        "preprocessing": {
            "resize": args.resize,
            "center_crop": True,
            "feature_sizes": list(sizes),
            "normalization": "OpenAI CLIP mean/std",
        },
        "model": model_info,
        "fade_scoring": {
            "vision_feature": args.vision_feature,
            "language_feature": args.language_feature,
            "mode": args.mode,
            "vision_distance": "top-1 (1-cosine)/2 against support patch gallery",
            "vision_image_reduction": "mean resized patch maps then spatial maximum",
            "vision_segmentation_multiplier": args.vision_segmentation_multiplier,
            "language_prompts": str(Path(args.prompt_path).resolve()),
            "both_fusion": "0.5 * language score + 0.5 * vision score",
            "reference_chunk_size": args.reference_chunk_size,
        },
        **extra,
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def validate_args(args, sizes: tuple[int, ...]) -> None:
    if not args.shots or any(int(shot) < 1 for shot in args.shots):
        raise ValueError("--shots must contain positive integers")
    if args.protocol == "ofdma" and not args.dataset_root:
        raise ValueError("--dataset-root is required for --protocol ofdma")
    if args.protocol in {"rf_target", "public_rf"} and not args.dataset_root:
        # RF builders use the repository/data defaults; this option is kept
        # for OFDMA while preserving the existing RF job constructors.
        args.dataset_root = ""
    if args.mode in {"vision", "both"} and args.reference_chunk_size < 1:
        raise ValueError("--reference-chunk-size must be positive")
    if args.classification_size not in sizes:
        raise ValueError("classification size must be present in feature sizes")


def main() -> None:
    args = parse_args()
    sizes = parse_sizes(args.feature_sizes, args.classification_size)
    validate_args(args, sizes)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    setup_seed(args.seed)
    if args.use_cpu:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; pass --use-cpu explicitly if intended")
        device = torch.device(f"cuda:{int(args.gpu_id)}")
    torch.set_float32_matmul_precision("high")

    if args.protocol in {"rf_target", "public_rf"}:
        jobs = build_rf_jobs(args)
        extra = {
            "support_manifest": args.support_manifest,
            "normal_sampling": args.normal_sampling,
            "shots": "not applicable; RF support is fixed by the shared manifest",
        }
    elif args.protocol == "ofdma":
        jobs, scene_ids, source = build_ofdma_protocol_jobs(args)
        extra = {
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "dataset_protocol": source["source_protocol"].get("protocol_name", "unknown"),
            "split": args.split,
            "target_scene_ids": scene_ids,
            "shots": sorted(set(args.shots)),
            "support_rule": "same target-scene normal support observations, complete 21-SU groups",
            "su_reduction": "maximum score over 21 SUs, then macro over target scenes",
            "smoke_limits": {
                "max_normal_observations": args.max_normal_observations,
                "max_anomaly_observations_per_type": args.max_anomaly_observations_per_type,
            },
        }
    else:
        jobs, fedjam_meta = build_fedjam_jobs(args)
        extra = {
            "dataset_root": str(Path(args.data_root).resolve()),
            "support_rule": "nested benign train support; independent test split",
            "fedjam_sampling": fedjam_meta,
            "max_test_per_label": args.max_test_per_label,
        }
    if not jobs:
        raise RuntimeError("No jobs were built")

    model_info = {
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "force_quick_gelu": str(args.pretrained).lower() == "openai",
        "gem_depth": args.gem_depth,
        "source": "references/FADE + gem-torch official GEMWrapper",
        "trainable_parameters": 0,
        "open_clip": getattr(open_clip, "__version__", "unknown"),
    }
    protocol = write_protocol(args, output_root, sizes, model_info, extra)
    if args.validate_only:
        print(json.dumps({"status": "validated", "num_jobs": len(jobs), "protocol": protocol}, indent=2))
        return

    print(
        json.dumps(
            {
                "status": "starting",
                "protocol": args.protocol,
                "mode": args.mode,
                "jobs": len(jobs),
                "device": str(device),
                "feature_sizes": sizes,
            },
            indent=2,
        ),
        flush=True,
    )
    model = create_fade_model(args, device)
    language_prototypes = (
        build_language_prototypes(model, args, device)
        if args.mode in {"language", "both"}
        else None
    )

    method_name = f"fade_{args.mode}_{args.vision_feature}"
    rows: list[dict] = []
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for job in jobs:
        grouped[support_signature(job)].append(job)
    print(f"support_groups={len(grouped)} jobs={len(jobs)}", flush=True)

    for group_index, grouped_jobs in enumerate(grouped.values(), 1):
        first_job = grouped_jobs[0]
        print(
            f"[support {group_index}/{len(grouped)}] "
            f"support={len(first_job['train_samples'])} eval_jobs={len(grouped_jobs)}",
            flush=True,
        )
        galleries = (
            build_gallery(model, first_job["train_samples"], args, sizes, device)
            if args.mode in {"vision", "both"}
            else None
        )
        for job in grouped_jobs:
            payload = score_job(
                model,
                job,
                galleries,
                language_prototypes,
                args,
                sizes,
                device,
            )
            if not np.all(np.isfinite(payload["scores"])):
                raise RuntimeError(f"Non-finite scores for {job['scene']} {job['jsr']}")
            save_payload(output_root, job, payload)
            if args.protocol == "ofdma":
                observation = aggregate_observations(
                    {
                        "names": payload["names"],
                        "labels": payload["labels"],
                        "jammer_types": np.asarray(
                            [sample["jammer_type"] for sample in job["test_samples"]]
                        ),
                        "scores": payload["scores"],
                    }
                )
                stem = f"{job['scene']}_{job['jsr']}"
                np.savez_compressed(
                    output_root / "scores" / f"{stem}-observation-scores.npz",
                    **observation,
                )
                # Reuse the formal OFDMA metric implementation so the SU and
                # scene aggregation is exactly aligned with other baselines.
                from tools.eval_ofdma_target_scene_baselines import metric_rows

                rows.extend(metric_rows(job, observation, method_name))
            elif args.protocol == "fedjam":
                rows.extend(fedjam_rows(job, payload, method_name))
            else:
                rows.extend(rf_rows(job, payload, method_name))
            current = append_rf_macro(rows, method_name) if args.protocol != "ofdma" and args.protocol != "fedjam" else rows
            write_csv(output_root / "results.csv", current)
            values = metric(payload["labels"], payload["scores"])
            print(
                f"  {job['scene']} {job['jsr']} "
                f"n={len(payload['labels'])} AUROC={values['auroc']:.2f}",
                flush=True,
            )
        del galleries
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.protocol == "ofdma":
        from tools.eval_ofdma_target_scene_baselines import append_macro_rows

        final_rows = append_macro_rows(rows)
    elif args.protocol in {"rf_target", "public_rf"}:
        final_rows = append_rf_macro(rows, method_name)
    else:
        final_rows = rows
    write_csv(output_root / "results.csv", final_rows)
    summary = {
        "status": "complete",
        "method": method_name,
        "protocol": args.protocol,
        "num_jobs": len(jobs),
        "rows": len(final_rows),
        "device": str(device),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[done] {output_root / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
