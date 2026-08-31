#!/usr/bin/env python
"""Target-support-only ResAD adaptation for the SpectraMemAD experiments.

This evaluator keeps the useful ResAD mechanism while replacing its industrial
dataset protocol with this project's frozen spectrum protocols:

* a Wide-ResNet-50-2 extracts three spatial feature levels;
* each query patch is matched to a normal target-support reference patch;
* squared residual features are processed by ResAD's VQ, constraintor, and
  conditional normalizing-flow heads;
* every dataset is evaluated independently;
* the trainable ResAD heads are fitted only from that evaluation unit's normal
  target support; no external RF source, anomaly image, or pixel mask is used;
* the same target support builds the inference reference bank, while a
  deterministic two-fold split prevents self-matching during head fitting.

The implementation is intentionally called ``ResAD-Protocol-Adapted`` rather
than an exact reproduction.  In particular, its log-likelihood branch is
calibrated with support-normal statistics so every target query can be scored
independently; the official implementation normalizes over the complete test
batch, which conflicts with this project's support-only protocol.

No experiment is launched by importing this module.  Use ``--validate-only``
to audit manifests, counts, and support/test isolation without creating a
model or touching CUDA.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import cv2
import numpy as np
import pandas as pd
import pyarrow.ipc as pa_ipc
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Importing this compatibility module exposes the pinned ResAD-2024 components
# and aliases the old timm module paths used by its downloaded source snapshot.
from tools.eval_resad_spectral import (  # noqa: E402
    BoundaryAverager,
    MultiScaleConv,
    MultiScaleVQ,
    applying_EFDM,
    calculate_log_barrier_bi_occ_loss,
    get_logp,
    get_position_encoding,
    load_flow_model,
    resad_flow_train,
)

# The downloaded ResAD snapshot uses generic top-level module names such as
# ``datasets``, ``utils``, and ``models``.  Its imported class/function objects
# keep their own module globals, so after extracting them above we restore the
# normal project import namespace before loading this repository's protocol
# helpers.  This avoids silently importing ResAD's industrial datasets in
# place of the local spectrum datasets.
_resad_root = REPO_ROOT / "references" / "ResAD"
_resad_paths = {str(_resad_root), str(_resad_root / "_deps")}
sys.path[:] = [entry for entry in sys.path if entry not in _resad_paths]
for _prefix in ("datasets", "utils", "models", "losses", "train"):
    for _module_name in list(sys.modules):
        if _module_name == _prefix or _module_name.startswith(f"{_prefix}."):
            del sys.modules[_module_name]

from datasets.ofdma_spectrum import NUM_SUS, OFDMASpectrogramPreprocessor  # noqa: E402
from datasets.ofdma_target_scene import (  # noqa: E402
    DEFAULT_TARGET_SCENE_ROOT,
    available_target_scenes,
    build_target_scene_records,
    load_target_scene_manifest,
)
from datasets.rf_target import RF_JSR_BY_SIGNAL, RF_SCENES  # noqa: E402
from tools.eval_patchcore_cls import (  # noqa: E402
    PUBLIC_RF_JSRS,
    public_rf_jobs,
    rf_target_jobs,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_RF_MANIFEST = (
    REPO_ROOT
    / "analysis_outputs"
    / "20260728_rf_five_type_formal_seed111"
    / "support_manifest.json"
)
DEFAULT_PUBLIC_MANIFEST_ROOT = (
    REPO_ROOT / "analysis_outputs" / "20260810_public_rf_k_per_frequency"
)
METHOD_NAME = "ResAD-Normal-Only"
PROTOCOL_VERSION = "target-support-only-v2"
# Without anomaly supervision, ResAD's shifted-Gaussian abnormal branch is not
# trained.  The fair normal-only baseline therefore reports only the normal
# likelihood branch; retaining an untrained bscore/merged branch would create
# an uncontrolled extra score rather than a valid ablation.
BRANCHES = ("logp",)


@dataclass(frozen=True)
class Sample:
    path: Path | None
    label: int
    group: str
    name: str
    image_bgr: np.ndarray | None = field(default=None, compare=False, repr=False)
    observation_id: str = ""
    jammer_type: str = ""
    su_id: int = -1


@dataclass
class ReferenceBank:
    """CPU-backed raw and normalized patch features for three levels."""

    raw: list[torch.Tensor]
    normalized: list[torch.Tensor]


@dataclass
class ModelBundle:
    encoder: torch.nn.Module
    vq: torch.nn.Module
    constraintor: torch.nn.Module
    estimators: list[torch.nn.Module]
    feature_dims: list[int]
    calibration: list[dict[str, float]]
    fit_metadata: dict


class FormalImageDataset(Dataset):
    """Common image transform used by the formal visual baselines."""

    def __init__(self, samples: Sequence[Sample], resize: int, image_size: int):
        self.samples = list(samples)
        self.image_transform = T.Compose(
            [
                T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_bgr(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(path)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    def __getitem__(self, index: int):
        sample = self.samples[index]
        if sample.image_bgr is not None:
            bgr = sample.image_bgr
        elif sample.path is not None:
            bgr = self._read_bgr(sample.path)
        else:
            raise ValueError(f"Sample {sample.name!r} has neither a path nor image data")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self.image_transform(Image.fromarray(rgb))

        return {
            "image": image,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "group": sample.group,
            "name": sample.name,
            "observation_id": sample.observation_id,
            "jammer_type": sample.jammer_type,
            "su_id": sample.su_id,
        }


def make_loader(samples: Sequence[Sample], args, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        FormalImageDataset(samples, args.resize, args.image_size),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_text(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def sample_identity(sample: Sample) -> str:
    if sample.path is not None:
        return f"path:{sample.path.resolve()}"
    return f"name:{sample.name}"


def support_signature(samples: Sequence[Sample]) -> str:
    return sha256_text(sorted(sample_identity(sample) for sample in samples))


def sample_paths(samples: Iterable[Sample]) -> set[str]:
    return {str(sample.path.resolve()) for sample in samples if sample.path is not None}


def assert_sample_isolation(
    support: Iterable[Sample], test: Iterable[Sample], context: str
) -> None:
    support_ids = {sample_identity(sample) for sample in support}
    test_ids = {sample_identity(sample) for sample in test}
    overlap = support_ids & test_ids
    if overlap:
        preview = "\n".join(sorted(overlap)[:10])
        raise RuntimeError(
            f"{context}: support/test leakage ({len(overlap)} samples); first:\n{preview}"
        )


def metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "fpr95": float("nan")}
    fpr, tpr, _ = roc_curve(y, s)
    valid = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(y, s) * 100.0),
        "auprc": float(average_precision_score(y, s) * 100.0),
        "fpr95": float(fpr[valid[0]] * 100.0) if len(valid) else 100.0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def regroup_sample(sample: Sample, group: str, *, name: str | None = None) -> Sample:
    return Sample(
        sample.path,
        sample.label,
        group,
        sample.name if name is None else name,
        image_bgr=sample.image_bgr,
        observation_id=sample.observation_id,
        jammer_type=sample.jammer_type,
        su_id=sample.su_id,
    )


def read_sample_bgr(sample: Sample) -> np.ndarray:
    if sample.image_bgr is not None:
        return np.ascontiguousarray(sample.image_bgr.copy())
    if sample.path is None:
        raise ValueError(f"Sample {sample.name!r} has no image source")
    return FormalImageDataset._read_bgr(sample.path)


def singleton_training_view(sample: Sample, args, context: str, group: str) -> Sample:
    """Create one deterministic weak view when only one support image exists.

    ResAD learns from residuals between a normal image and a separate normal
    reference.  With exactly one support image, matching the image to itself
    makes every residual identically zero.  A very small photometric jitter is
    therefore used only as the fitting query; the inference memory still uses
    the untouched support image.
    """

    seed_bytes = hashlib.sha256(
        f"{args.seed}:{context}:{sample_identity(sample)}".encode("utf-8")
    ).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
    image = read_sample_bgr(sample).astype(np.float32)
    gain = float(rng.uniform(0.99, 1.01))
    bias = float(rng.uniform(-1.0, 1.0))
    noise = rng.normal(0.0, args.singleton_noise_std, size=image.shape).astype(np.float32)
    image = np.clip(image * gain + bias + noise, 0.0, 255.0).astype(np.uint8)
    return Sample(
        None,
        0,
        group,
        f"fit-view/{sample.name}",
        image_bgr=np.ascontiguousarray(image),
        observation_id=sample.observation_id,
        jammer_type=sample.jammer_type,
        su_id=sample.su_id,
    )


def support_fit_metadata(support: Sequence[Sample], context: str) -> dict:
    if not support:
        raise RuntimeError(f"{context}: empty normal support")
    if any(sample.label != 0 for sample in support):
        raise RuntimeError(f"{context}: support contains a non-normal sample")
    count = len(support)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_scope": context.split("/", 1)[0],
        "fit_key": context,
        "training_supervision": "target support normals only; no anomalies or masks",
        "support_signature": support_signature(support),
        "support_count": count,
        "normal_count": count,
        "anomaly_count": 0,
        "fit_strategy": (
            "deterministic two-fold support residual fitting"
            if count > 1
            else "singleton weak-view residual fitting"
        ),
    }


def build_support_fit_records(
    support: Sequence[Sample], args, context: str
) -> tuple[list[Sample], dict[str, list[Sample]], dict]:
    """Use only one evaluation unit's support to fit the ResAD heads.

    For two or more support images, all samples participate in two folds: one
    fold is queried against the other, then their roles are reversed.  This
    avoids the zero-residual self-match without importing another scene or
    dataset.  A singleton uses one deterministic weak query view as the only
    unavoidable fallback.
    """

    metadata = support_fit_metadata(support, context)
    ordered = sorted(
        support,
        key=lambda sample: hashlib.sha256(
            f"{args.seed}:{context}:{sample_identity(sample)}".encode("utf-8")
        ).hexdigest(),
    )
    if len(ordered) == 1:
        group = f"fit/{context}/singleton"
        normal_records = [singleton_training_view(ordered[0], args, context, group)]
        references = {group: [regroup_sample(ordered[0], group)]}
        metadata.update(
            {
                "fit_normal_count": 1,
                "reference_count": 1,
                "singleton_noise_std": float(args.singleton_noise_std),
            }
        )
        return normal_records, references, metadata

    split = (len(ordered) + 1) // 2
    fold_a = ordered[:split]
    fold_b = ordered[split:]
    group_a = f"fit/{context}/fold_a"
    group_b = f"fit/{context}/fold_b"
    normal_records = [regroup_sample(sample, group_a) for sample in fold_a]
    normal_records.extend(regroup_sample(sample, group_b) for sample in fold_b)
    references = {
        group_a: [regroup_sample(sample, group_a) for sample in fold_b],
        group_b: [regroup_sample(sample, group_b) for sample in fold_a],
    }
    metadata.update(
        {
            "fit_normal_count": len(normal_records),
            "reference_count": sum(len(values) for values in references.values()),
            "fold_sizes": [len(fold_a), len(fold_b)],
        }
    )
    return normal_records, references, metadata


def make_models(args, device: torch.device):
    """Create the ResAD feature extractor and trainable heads."""

    # Reuse the locally cached torchvision checkpoint.  Loading into timm's
    # features-only model intentionally ignores layer4/classifier parameters;
    # all layer1-layer3 keys must be present.
    import timm

    checkpoint = Path(args.backbone_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing Wide-ResNet checkpoint: {checkpoint}")
    encoder = timm.create_model(
        "wide_resnet50_2",
        features_only=True,
        out_indices=(1, 2, 3),
        pretrained=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = encoder.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"Backbone checkpoint is missing used keys: {incompatible.missing_keys[:10]}")
    unexpected_non_layer4 = [
        key for key in incompatible.unexpected_keys if not key.startswith(("layer4.", "fc."))
    ]
    if unexpected_non_layer4:
        raise RuntimeError(f"Unexpected backbone keys outside unused layer4/fc: {unexpected_non_layer4[:10]}")
    encoder = encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    feature_dims = list(encoder.feature_info.channels())
    vq = MultiScaleVQ(args.num_embeddings, feature_dims).to(device)
    constraintor = MultiScaleConv(feature_dims).to(device)
    flow_args = argparse.Namespace(
        flow_arch="conditional_flow_model",
        coupling_layers=args.coupling_layers,
        clamp_alpha=args.clamp_alpha,
        pos_embed_dim=args.pos_embed_dim,
    )
    estimators = [load_flow_model(flow_args, dim).to(device) for dim in feature_dims]
    return encoder, vq, constraintor, estimators, feature_dims


@torch.inference_mode()
def encode_reference_bank(
    encoder: torch.nn.Module,
    samples: Sequence[Sample],
    args,
    device: torch.device,
) -> ReferenceBank:
    chunks: list[list[torch.Tensor]] = [[], [], []]
    for batch in make_loader(samples, args, shuffle=False):
        features = encoder(batch["image"].to(device, non_blocking=True))
        for level, feature in enumerate(features):
            flattened = feature.permute(0, 2, 3, 1).reshape(-1, feature.shape[1])
            chunks[level].append(flattened.float().cpu())
    raw = [torch.cat(values, dim=0).contiguous() for values in chunks]
    normalized = [F.normalize(value, p=2, dim=1).contiguous() for value in raw]
    return ReferenceBank(raw=raw, normalized=normalized)


def encode_group_reference_banks(encoder, references, args, device) -> dict[str, ReferenceBank]:
    return {
        group: encode_reference_bank(encoder, samples, args, device)
        for group, samples in references.items()
    }


def chunked_match_level(
    feature: torch.Tensor,
    bank: ReferenceBank,
    level: int,
    args,
) -> torch.Tensor:
    """Cosine nearest-reference matching without an all-pairs GPU matrix."""

    batch, channels, height, width = feature.shape
    query = feature.permute(0, 2, 3, 1).reshape(-1, channels).contiguous()
    query_n = F.normalize(query.float(), p=2, dim=1)
    raw_cpu = bank.raw[level]
    norm_cpu = bank.normalized[level]
    selected = torch.empty_like(query)
    for q_start in range(0, len(query_n), args.query_chunk_size):
        q_end = min(q_start + args.query_chunk_size, len(query_n))
        q = query_n[q_start:q_end]
        best_value = torch.full((len(q),), -float("inf"), device=q.device)
        best_index = torch.zeros((len(q),), dtype=torch.long, device=q.device)
        for r_start in range(0, len(norm_cpu), args.reference_chunk_size):
            r_end = min(r_start + args.reference_chunk_size, len(norm_cpu))
            refs = norm_cpu[r_start:r_end].to(q.device, non_blocking=True)
            similarity = q @ refs.T
            value, index = similarity.max(dim=1)
            update = value > best_value
            best_value[update] = value[update]
            best_index[update] = index[update] + r_start
            del refs, similarity, value, index
        selected[q_start:q_end] = raw_cpu[best_index.cpu()].to(
            query.device, dtype=query.dtype, non_blocking=True
        )
    return selected.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


def matched_features(
    features: Sequence[torch.Tensor],
    groups: Sequence[str],
    banks: dict[str, ReferenceBank],
    args,
) -> list[torch.Tensor]:
    output = [[] for _ in features]
    for index, group in enumerate(groups):
        if group not in banks:
            raise KeyError(f"No reference bank for support-fit group {group!r}")
        for level, feature in enumerate(features):
            output[level].append(
                chunked_match_level(feature[index : index + 1], banks[group], level, args)
            )
    return [torch.cat(values, dim=0) for values in output]


def target_matched_features(features, bank: ReferenceBank, args) -> list[torch.Tensor]:
    return [chunked_match_level(feature, bank, level, args) for level, feature in enumerate(features)]


def residual_features(features, matched) -> list[torch.Tensor]:
    return [F.mse_loss(feature, reference, reduction="none") for feature, reference in zip(features, matched)]


def train_support_model(
    args,
    encoder,
    vq,
    constraintor,
    estimators,
    normal_records: Sequence[Sample],
    reference_banks: dict[str, ReferenceBank],
    device: torch.device,
) -> list[dict]:
    """Fit ResAD heads from normal target-support samples only.

    The original ResAD has a later anomaly-supervised stage that consumes
    abnormal masks.  This adapter deliberately omits that stage.  A zero
    normality indicator is passed to the compatibility flow routine only
    because its API uses that tensor to distinguish normal from abnormal
    pixels; no annotation mask is read or constructed from an anomaly image.
    """

    normal_loader = make_loader(normal_records, args, shuffle=True)
    optimizer_vq = torch.optim.Adam(vq.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer_constraintor = torch.optim.Adam(
        constraintor.parameters(), lr=args.lr, weight_decay=0.0005
    )
    flow_parameters = [parameter for model in estimators for parameter in model.parameters()]
    optimizer_flow = torch.optim.Adam(flow_parameters, lr=args.lr, weight_decay=0.0005)
    scheduler_vq = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_vq, milestones=args.lr_milestones, gamma=0.1
    )
    scheduler_constraintor = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_constraintor, milestones=args.lr_milestones, gamma=0.1
    )
    scheduler_flow = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_flow, milestones=args.lr_milestones, gamma=0.1
    )
    boundaries = BoundaryAverager(num_levels=3)
    history: list[dict] = []

    for epoch in range(args.epochs):
        stage = "normal-only"
        vq.train()
        constraintor.train()
        for estimator in estimators:
            estimator.train()
        total_loss = 0.0
        total_steps = 0

        for batch in normal_loader:
            images = batch["image"].to(device, non_blocking=True)
            groups = list(batch["group"])
            # This is an internal all-normal indicator required by the
            # downloaded flow-training function, not a ground-truth mask.
            normal_indicator = torch.zeros(
                images.shape[0],
                1,
                images.shape[-2],
                images.shape[-1],
                device=device,
                dtype=images.dtype,
            )
            with torch.no_grad():
                features = [value.detach() for value in encoder(images)]
                matched = matched_features(features, groups, reference_banks, args)
                residuals = residual_features(features, matched)

            level_normality = [
                F.interpolate(
                    normal_indicator, size=value.shape[-2:], mode="nearest"
                ).squeeze(1)
                for value in residuals
            ]

            loss_vq = vq(residuals, level_normality, train=True)
            optimizer_vq.zero_grad(set_to_none=True)
            loss_vq.backward()
            optimizer_vq.step()

            before_neck = [value.detach().clone() for value in residuals]
            neck = constraintor(*residuals)
            loss_neck = 0.0
            for level, (mapped, original) in enumerate(zip(neck, before_neck)):
                mapped_flat = mapped.permute(0, 2, 3, 1).reshape(-1, mapped.shape[1])
                original_flat = original.permute(0, 2, 3, 1).reshape(-1, original.shape[1])
                mask_flat = level_normality[level].reshape(-1)
                loss_level, _normal, _abnormal = calculate_log_barrier_bi_occ_loss(
                    mapped_flat, mask_flat, original_flat
                )
                loss_neck = loss_neck + loss_level
            optimizer_constraintor.zero_grad(set_to_none=True)
            loss_neck.backward()
            optimizer_constraintor.step()

            detached_neck = [value.detach().clone() for value in neck]
            flow_loss, flow_steps = resad_flow_train(
                argparse.Namespace(
                    feature_levels=3,
                    pos_embed_dim=args.pos_embed_dim,
                    device=device,
                    flow_arch="conditional_flow_model",
                    pos_beta=args.pos_beta,
                    margin_tau=args.margin_tau,
                    bgspp_lambda=args.bgspp_lambda,
                ),
                detached_neck,
                estimators,
                optimizer_flow,
                normal_indicator,
                boundaries,
                epoch,
                N_batch=args.flow_batch_points,
                FIRST_STAGE_EPOCH=args.epochs + 1,
            )
            total_loss += float(loss_vq.item()) + float(loss_neck.item()) + float(flow_loss)
            total_steps += 2 + int(flow_steps)

        scheduler_vq.step()
        scheduler_constraintor.step()
        scheduler_flow.step()
        row = {
            "epoch": epoch,
            "stage": stage,
            "loss": total_loss / max(1, total_steps),
            "steps": total_steps,
        }
        history.append(row)
        print(
            f"[support-fit] epoch={epoch + 1}/{args.epochs} stage={stage} "
            f"loss={row['loss']:.6f} steps={total_steps}",
            flush=True,
        )
    return history


def raw_resad_outputs(
    args,
    vq,
    constraintor,
    estimators,
    residuals: Sequence[torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    fdm_features = vq(residuals, train=False)
    adapted = applying_EFDM(residuals, fdm_features, alpha=args.fdm_alpha)
    adapted = constraintor(*adapted)
    logps: list[torch.Tensor] = []
    for value, estimator in zip(adapted, estimators):
        batch, channels, height, width = value.shape
        flattened = value.permute(0, 2, 3, 1).reshape(-1, channels)
        position = get_position_encoding(args.pos_embed_dim, height, width).to(device)
        position = position.unsqueeze(0).repeat(batch, 1, 1, 1)
        position = position.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
        z, logdet = estimator(flattened, [position])
        logp = get_logp(channels, z, logdet) / channels
        logps.append(logp.reshape(batch, height, width))
    return logps


@torch.inference_mode()
def fit_support_calibration(
    args,
    encoder,
    vq,
    constraintor,
    estimators,
    normal_records: Sequence[Sample],
    reference_banks: dict[str, ReferenceBank],
    device: torch.device,
) -> list[dict[str, float]]:
    """Fit per-level robust log-likelihood calibration from support normals only."""

    vq.eval()
    constraintor.eval()
    for estimator in estimators:
        estimator.eval()
    collected: list[list[torch.Tensor]] = [[], [], []]
    counts = [0, 0, 0]
    for batch in make_loader(normal_records, args, shuffle=False):
        images = batch["image"].to(device, non_blocking=True)
        features = [value.detach() for value in encoder(images)]
        matched = matched_features(features, list(batch["group"]), reference_banks, args)
        residuals = residual_features(features, matched)
        logps = raw_resad_outputs(
            args, vq, constraintor, estimators, residuals, device
        )
        for level, value in enumerate(logps):
            remaining = args.calibration_points - counts[level]
            if remaining <= 0:
                continue
            flattened = value.detach().float().cpu().reshape(-1)
            if len(flattened) > remaining:
                step = max(1, len(flattened) // remaining)
                flattened = flattened[::step][:remaining]
            collected[level].append(flattened)
            counts[level] += len(flattened)
        if all(count >= args.calibration_points for count in counts):
            break

    calibration = []
    for level, values in enumerate(collected):
        if not values:
            raise RuntimeError(f"No support-normal calibration values for feature level {level}")
        vector = torch.cat(values)
        q1, median, q3 = torch.quantile(
            vector, torch.tensor([0.25, 0.5, 0.75], dtype=vector.dtype)
        ).tolist()
        calibration.append(
            {
                "level": level,
                "median_logp": float(median),
                "iqr_logp": float(max(q3 - q1, args.calibration_epsilon)),
                "points": int(len(vector)),
            }
        )
    return calibration


def model_config(args) -> dict:
    """Configuration fields that materially change a fitted support model."""

    return {
        "protocol_version": PROTOCOL_VERSION,
        "backbone_checkpoint": str(Path(args.backbone_checkpoint).resolve()),
        "num_embeddings": args.num_embeddings,
        "coupling_layers": args.coupling_layers,
        "clamp_alpha": args.clamp_alpha,
        "pos_embed_dim": args.pos_embed_dim,
        "pos_beta": args.pos_beta,
        "margin_tau": args.margin_tau,
        "bgspp_lambda": args.bgspp_lambda,
        "fdm_alpha": args.fdm_alpha,
        "resize": args.resize,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_milestones": list(args.lr_milestones),
        "flow_batch_points": args.flow_batch_points,
        "singleton_noise_std": args.singleton_noise_std,
        "calibration_points": args.calibration_points,
    }


def checkpoint_payload(bundle: ModelBundle, args, history: list[dict]) -> dict:
    return {
        "method": METHOD_NAME,
        "resad_commit": "08c9b74934c5d25524f6e3bf42761bdb2e6efc34",
        "vq": bundle.vq.state_dict(),
        "constraintor": bundle.constraintor.state_dict(),
        "estimators": [model.state_dict() for model in bundle.estimators],
        "feature_dims": bundle.feature_dims,
        "calibration": bundle.calibration,
        "fit_metadata": bundle.fit_metadata,
        "train_history": history,
        "model_config": model_config(args),
    }


def support_checkpoint_path(args, fit_key: str) -> Path:
    safe_key = fit_key.replace("/", "_").replace(" ", "_")
    return Path(args.output_root) / "support_models" / f"{safe_key}.pt"


def build_or_load_bundle(
    args,
    fit_key: str,
    support_samples: Sequence[Sample],
    test_samples: Sequence[Sample],
    device: torch.device,
) -> ModelBundle:
    checkpoint_path = support_checkpoint_path(args, fit_key)
    assert_sample_isolation(support_samples, test_samples, fit_key)
    normal, references, fit_metadata = build_support_fit_records(
        support_samples, args, fit_key
    )
    encoder, vq, constraintor, estimators, feature_dims = make_models(args, device)

    if checkpoint_path.is_file() and not args.force_refit:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload.get("method") != METHOD_NAME:
            raise ValueError(f"Unexpected support checkpoint method: {checkpoint_path}")
        if payload.get("model_config") != model_config(args):
            raise ValueError(
                f"Support checkpoint configuration mismatch: {checkpoint_path}. "
                "Use matching arguments or pass --force-refit."
            )
        checkpoint_fit = payload.get("fit_metadata", {})
        for key in (
            "protocol_version",
            "dataset_scope",
            "fit_key",
            "support_signature",
            "support_count",
            "normal_count",
            "anomaly_count",
            "fit_normal_count",
            "reference_count",
            "fit_strategy",
        ):
            if checkpoint_fit.get(key) != fit_metadata.get(key):
                raise ValueError(
                    f"Support checkpoint data mismatch for {key}: {checkpoint_path}. "
                    "Pass --force-refit to rebuild it."
                )
        vq.load_state_dict(payload["vq"])
        constraintor.load_state_dict(payload["constraintor"])
        if len(payload.get("estimators", [])) != len(estimators):
            raise ValueError(f"Support checkpoint flow-level mismatch: {checkpoint_path}")
        for estimator, state in zip(estimators, payload["estimators"]):
            estimator.load_state_dict(state)
        calibration = payload["calibration"]
        fit_metadata = payload["fit_metadata"]
        print(f"[support-model] loaded {checkpoint_path}", flush=True)
    else:
        reference_banks = encode_group_reference_banks(
            encoder, references, args, device
        )
        history = train_support_model(
            args,
            encoder,
            vq,
            constraintor,
            estimators,
            normal,
            reference_banks,
            device,
        )
        calibration = fit_support_calibration(
            args,
            encoder,
            vq,
            constraintor,
            estimators,
            normal,
            reference_banks,
            device,
        )
        bundle = ModelBundle(
            encoder,
            vq,
            constraintor,
            estimators,
            feature_dims,
            calibration,
            fit_metadata,
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload(bundle, args, history), checkpoint_path)
        print(f"[support-model] wrote {checkpoint_path}", flush=True)
        del reference_banks

    vq.eval()
    constraintor.eval()
    for estimator in estimators:
        estimator.eval()
    return ModelBundle(
        encoder,
        vq,
        constraintor,
        estimators,
        feature_dims,
        calibration,
        fit_metadata,
    )


@torch.inference_mode()
def score_encoded_features(
    args,
    bundle: ModelBundle,
    features: Sequence[torch.Tensor],
    target_bank: ReferenceBank,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Score one encoded batch without using statistics from other test images."""

    matched = target_matched_features(features, target_bank, args)
    residuals = residual_features(features, matched)
    logps = raw_resad_outputs(
        args,
        bundle.vq,
        bundle.constraintor,
        bundle.estimators,
        residuals,
        device,
    )
    normal_maps: list[torch.Tensor] = []
    for level, logp in enumerate(logps):
        calibration = bundle.calibration[level]
        standardized = (
            float(calibration["median_logp"]) - logp
        ) / float(calibration["iqr_logp"])
        normal_score = torch.sigmoid(standardized)
        normal_maps.append(
            F.interpolate(
                normal_score.unsqueeze(1),
                size=(args.image_size, args.image_size),
                mode="bilinear",
                align_corners=True,
            ).squeeze(1)
        )
    logp_map = torch.stack(normal_maps, dim=0).mean(dim=0).cpu().numpy()

    def image_max(values: np.ndarray) -> np.ndarray:
        if args.gaussian_sigma > 0:
            values = np.stack(
                [gaussian_filter(value, sigma=args.gaussian_sigma) for value in values],
                axis=0,
            )
        return values.reshape(len(values), -1).max(axis=1).astype(np.float32)

    return {
        "logp": image_max(logp_map),
    }


@torch.inference_mode()
def score_samples(
    args,
    bundle: ModelBundle,
    samples: Sequence[Sample],
    target_bank: ReferenceBank,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    labels: list[int] = []
    names: list[str] = []
    scores = {branch: [] for branch in BRANCHES}
    bundle.vq.eval()
    bundle.constraintor.eval()
    for estimator in bundle.estimators:
        estimator.eval()
    for batch in make_loader(samples, args, shuffle=False):
        features = [value.detach() for value in bundle.encoder(batch["image"].to(device))]
        batch_scores = score_encoded_features(args, bundle, features, target_bank, device)
        labels.extend(int(value) for value in batch["label"].tolist())
        names.extend(str(value) for value in batch["name"])
        for branch in BRANCHES:
            scores[branch].extend(float(value) for value in batch_scores[branch])
    return (
        np.asarray(labels, dtype=np.int64),
        {branch: np.asarray(values, dtype=np.float32) for branch, values in scores.items()},
        names,
    )


def job_sample(item: dict, group: str) -> Sample:
    return Sample(
        path=Path(item["path"]),
        label=int(item["label"]),
        group=group,
        name=str(item["name"]),
    )


def common_job_args(args) -> argparse.Namespace:
    """Namespace expected by the project's shared formal job builders."""

    return argparse.Namespace(
        output_root=args.output_root,
        support_manifest=args.support_manifest,
        normal_sampling=args.normal_sampling,
        seed=args.seed,
        max_train_normals=0,
        max_test_normals=args.max_test_normals,
        max_abnormals=args.max_abnormals,
        rf_signals=list(args.rf_signals),
        public_rf_signals=list(args.public_rf_signals),
        rf_scenes=list(args.rf_scenes),
        support_bootstrap_seed=-1,
    )


def build_inhouse_jobs(args) -> tuple[list[dict], dict]:
    shared = common_job_args(args)
    shared.support_manifest = str(args.rf_manifest)
    jobs = rf_target_jobs(shared)
    for job in jobs:
        job["shot"] = "per_frequency"
        job["support_manifest"] = shared.support_manifest
        job["support_manifest_sha256"] = shared.support_manifest_sha256
        job["fit_key"] = f"inhouse_rf/{job['scene']}/per_frequency"
    metadata = {
        "protocol": "In-house RF target-scene per-frequency",
        "support_manifest": shared.support_manifest,
        "support_manifest_sha256": shared.support_manifest_sha256,
        "job_count": len(jobs),
    }
    return jobs, metadata


def build_public_jobs(args) -> tuple[list[dict], dict]:
    jobs: list[dict] = []
    test_digests: set[str] = set()
    manifests = {}
    for shot in args.shots:
        manifest_path = Path(args.public_manifest_root) / f"k{shot}" / "support_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing Public RF k={shot} manifest: {manifest_path}")
        shared = common_job_args(args)
        shared.support_manifest = str(manifest_path)
        shot_jobs = public_rf_jobs(shared)
        test_digests.add(shared.test_normal_paths_sha256)
        manifests[str(shot)] = {
            "path": str(manifest_path),
            "manifest_sha256": json.loads(manifest_path.read_text(encoding="utf-8"))[
                "manifest_sha256"
            ],
            "support_paths_sha256": shared.support_manifest_sha256,
            "test_paths_sha256": shared.test_normal_paths_sha256,
            "support_count": len(shot_jobs[0]["train_samples"]),
        }
        for job in shot_jobs:
            job["shot"] = int(shot)
            job["support_manifest"] = str(manifest_path)
            job["support_manifest_sha256"] = shared.support_manifest_sha256
            job["test_normal_paths_sha256"] = shared.test_normal_paths_sha256
            job["fit_key"] = f"public_rf/k{shot}_per_frequency"
            jobs.append(job)
    if len(test_digests) != 1:
        raise RuntimeError(f"Public RF k=1/2/4 test sets are not fixed: {sorted(test_digests)}")
    metadata = {
        "protocol": "Public RF nested k=1/2/4-per-frequency full-test",
        "manifests": manifests,
        "fixed_test_normal_paths_sha256": next(iter(test_digests)),
        "job_count": len(jobs),
    }
    return jobs, metadata


def target_samples_for_job(job: dict) -> tuple[list[Sample], list[Sample]]:
    group = f"target/{job['dataset']}/{job['scene']}"
    support = [job_sample(item, group) for item in job["train_samples"]]
    test = [job_sample(item, group) for item in job["test_samples"]]
    if any(sample.label != 0 for sample in support):
        raise RuntimeError(f"Target support contains an anomaly: {job['dataset']} {job['scene']}")
    if len({sample.label for sample in test}) != 2:
        raise RuntimeError(f"Target test does not contain both labels: {job['dataset']} {job['scene']}")
    direct_overlap = sample_paths(support) & sample_paths(test)
    if direct_overlap:
        raise RuntimeError(f"Target support/test overlap: {sorted(direct_overlap)[:10]}")
    return support, test


def score_file_stem(job: dict) -> str:
    value = (
        f"{job['dataset']}-{job['category']}-{job['scene']}-{job['jsr']}-"
        f"{job.get('shot', 'na')}"
    )
    return value.replace("/", "_").replace(" ", "_")


def save_job_scores(
    args,
    job: dict,
    labels: np.ndarray,
    score_values: dict[str, np.ndarray],
    names: Sequence[str],
) -> None:
    score_root = Path(args.output_root) / "scores"
    score_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        score_root / f"{score_file_stem(job)}.npz",
        labels=labels,
        names=np.asarray(names),
        image_paths=np.asarray([str(item["path"]) for item in job["test_samples"]]),
        **score_values,
        support_manifest_sha256=np.asarray(job.get("support_manifest_sha256", "")),
        test_normal_paths_sha256=np.asarray(job.get("test_normal_paths_sha256", "")),
    )


def job_metric_rows(job: dict, labels, score_values) -> list[dict]:
    rows = []
    for branch in BRANCHES:
        result = metrics(labels, score_values[branch])
        rows.append(
            {
                "row_type": "cell",
                "method": METHOD_NAME,
                "dataset": job["dataset"],
                "category": job["category"],
                "scene": job["scene"],
                "jsr": job["jsr"],
                "shot": job.get("shot", ""),
                "branch": branch,
                "num_support": len(job["train_samples"]),
                "num_test_normal": int((np.asarray(labels) == 0).sum()),
                "num_test_abnormal": int((np.asarray(labels) == 1).sum()),
                **result,
            }
        )
    return rows


def append_macro_rows(rows: list[dict]) -> list[dict]:
    output = list(rows)
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["dataset"], str(row["shot"]), row["branch"])
        groups.setdefault(key, []).append(row)
    for (dataset, shot, branch), values in groups.items():
        output.append(
            {
                "row_type": "macro",
                "method": METHOD_NAME,
                "dataset": dataset,
                "category": "ALL",
                "scene": "ALL",
                "jsr": "ALL",
                "shot": shot,
                "branch": branch,
                "num_support": "",
                "num_test_normal": sum(int(row["num_test_normal"]) for row in values),
                "num_test_abnormal": sum(int(row["num_test_abnormal"]) for row in values),
                "auroc": float(np.mean([float(row["auroc"]) for row in values])),
                "auprc": float(np.mean([float(row["auprc"]) for row in values])),
                "fpr95": float(np.mean([float(row["fpr95"]) for row in values])),
            }
        )
    return output


def evaluate_path_jobs(
    args, jobs: list[dict], device: torch.device
) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    fit_models: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = {}
    for job in jobs:
        grouped.setdefault(job["fit_key"], []).append(job)

    for fit_key, fit_jobs in grouped.items():
        first_support, _first_test = target_samples_for_job(fit_jobs[0])
        expected_signature = support_signature(first_support)
        all_test: list[Sample] = []
        for job in fit_jobs:
            support, test = target_samples_for_job(job)
            if support_signature(support) != expected_signature:
                raise RuntimeError(f"{fit_key}: grouped jobs do not share the same support")
            all_test.extend(test)
        bundle = build_or_load_bundle(args, fit_key, first_support, all_test, device)
        fit_models[fit_key] = bundle.fit_metadata
        support_cache: dict[tuple[str, ...], ReferenceBank] = {}
        # Public RF reuses the same 5,120 normal test images in every signal/
        # JSR cell.  Cache per-image scores under the support signature so the
        # frozen full-test protocol does not trigger 15 identical forwards.
        score_cache: dict[tuple[tuple[str, ...], str], dict[str, float]] = {}
        for index, job in enumerate(fit_jobs, start=1):
            print(
                f"[target {index}/{len(fit_jobs)}] {job['dataset']} "
                f"{job['category']} {job['scene']} {job['jsr']} shot={job.get('shot')}",
                flush=True,
            )
            support, test = target_samples_for_job(job)
            signature = tuple(str(sample.path) for sample in support)
            if signature not in support_cache:
                support_cache[signature] = encode_reference_bank(
                    bundle.encoder, support, args, device
                )
            missing: list[Sample] = []
            seen_missing: set[str] = set()
            for sample in test:
                if sample.path is None:
                    raise ValueError("Path-job samples must carry an image path")
                path_key = str(sample.path.resolve())
                cache_key = (signature, path_key)
                if cache_key not in score_cache and path_key not in seen_missing:
                    missing.append(sample)
                    seen_missing.add(path_key)
            if missing:
                _labels, missing_scores, _names = score_samples(
                    args, bundle, missing, support_cache[signature], device
                )
                for missing_index, sample in enumerate(missing):
                    path_key = str(sample.path.resolve())
                    score_cache[(signature, path_key)] = {
                        branch: float(missing_scores[branch][missing_index])
                        for branch in BRANCHES
                    }
            labels = np.asarray([sample.label for sample in test], dtype=np.int64)
            names = [sample.name for sample in test]
            score_values = {
                branch: np.asarray(
                    [
                        score_cache[(signature, str(sample.path.resolve()))][branch]
                        for sample in test
                    ],
                    dtype=np.float32,
                )
                for branch in BRANCHES
            }
            save_job_scores(args, job, labels, score_values, names)
            rows.extend(job_metric_rows(job, labels, score_values))
        del bundle, support_cache, score_cache
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows, fit_models


FEDJAM_LABEL_NAMES = {0: "benign", 1: "pulse", 2: "single_tone", 3: "wideband"}


def iter_fedjam_rows(split_dir: Path) -> Iterator[tuple[bytes, int, str]]:
    shards = sorted(split_dir.glob("*.arrow"))
    if not shards:
        raise FileNotFoundError(f"No FedJam Arrow shards under {split_dir}")
    for shard in shards:
        with shard.open("rb") as handle:
            reader = pa_ipc.open_stream(handle)
            for batch_index, batch in enumerate(reader):
                images = batch.column("image").to_pylist()
                labels = batch.column("label").to_pylist()
                for row_index, (image, label) in enumerate(zip(images, labels)):
                    yield (
                        bytes(image["bytes"]),
                        int(label),
                        f"{shard.name}:batch{batch_index}:row{row_index}",
                    )


def decode_fedjam_bgr(image_bytes: bytes) -> np.ndarray:
    with Image.open(BytesIO(image_bytes)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape[:2] != (224, 224):
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def select_fedjam_support(args) -> tuple[list[Sample], dict]:
    root = Path(args.fedjam_root) / "fedjam_dataset"
    rng = np.random.default_rng(args.seed)
    reservoir: list[tuple[bytes, str]] = []
    benign_seen = 0
    train_counts = {label: 0 for label in FEDJAM_LABEL_NAMES}
    for image_bytes, label, name in iter_fedjam_rows(root / "train"):
        train_counts[label] = train_counts.get(label, 0) + 1
        if label != 0:
            continue
        benign_seen += 1
        if len(reservoir) < max(args.shots):
            reservoir.append((image_bytes, name))
        else:
            replacement = int(rng.integers(0, benign_seen))
            if replacement < max(args.shots):
                reservoir[replacement] = (image_bytes, name)
    if len(reservoir) < max(args.shots):
        raise RuntimeError("FedJam does not contain enough benign train support rows")
    support = [
        Sample(
            None,
            0,
            "target/fedjam",
            f"train/{name}",
            image_bgr=decode_fedjam_bgr(image_bytes),
        )
        for image_bytes, name in reservoir
    ]
    manifest = {
        "protocol": "FedJam benign-only nested support",
        "seed": args.seed,
        "shots": list(args.shots),
        "support_names": [sample.name for sample in support],
        "support_names_sha256": sha256_text(sample.name for sample in support),
        "train_label_counts": {FEDJAM_LABEL_NAMES[key]: value for key, value in train_counts.items()},
        "benign_seen": benign_seen,
    }
    return support, manifest


def batched(values: Sequence, size: int) -> Iterator[Sequence]:
    if size < 1:
        raise ValueError(f"Batch size must be positive, got {size}")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def fedjam_test_samples(args) -> Iterator[Sample]:
    root = Path(args.fedjam_root) / "fedjam_dataset" / "test"
    counts = {label: 0 for label in FEDJAM_LABEL_NAMES}
    for image_bytes, raw_label, name in iter_fedjam_rows(root):
        if (
            args.max_fedjam_test_per_label > 0
            and counts.get(raw_label, 0) >= args.max_fedjam_test_per_label
        ):
            continue
        counts[raw_label] = counts.get(raw_label, 0) + 1
        yield Sample(
            None,
            int(raw_label != 0),
            "target/fedjam",
            f"test/{name}",
            image_bgr=decode_fedjam_bgr(image_bytes),
            jammer_type=FEDJAM_LABEL_NAMES[raw_label],
        )


def evaluate_fedjam(args, device: torch.device) -> tuple[list[dict], dict]:
    support, support_manifest = select_fedjam_support(args)
    score_root = Path(args.output_root) / "scores"
    score_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    expected_test_signature: str | None = None
    fit_models: dict[str, dict] = {}
    reverse_labels = {name: key for key, name in FEDJAM_LABEL_NAMES.items()}

    for shot in args.shots:
        shot_support = support[:shot]
        fit_key = f"fedjam/{shot}shot"
        bundle = build_or_load_bundle(args, fit_key, shot_support, [], device)
        fit_models[str(shot)] = bundle.fit_metadata
        bank = encode_reference_bank(bundle.encoder, shot_support, args, device)
        score_values = {branch: [] for branch in BRANCHES}
        binary_labels: list[int] = []
        raw_labels: list[int] = []
        names: list[str] = []
        pending: list[Sample] = []
        processed = 0

        def score_pending(samples: list[Sample]) -> None:
            nonlocal processed
            for batch in make_loader(samples, args, shuffle=False):
                features = [
                    value.detach()
                    for value in bundle.encoder(batch["image"].to(device, non_blocking=True))
                ]
                labels_batch = [int(value) for value in batch["label"].tolist()]
                jammer_batch = [str(value) for value in batch["jammer_type"]]
                binary_labels.extend(labels_batch)
                raw_labels.extend(reverse_labels[jammer] for jammer in jammer_batch)
                names.extend(str(value) for value in batch["name"])
                values = score_encoded_features(args, bundle, features, bank, device)
                for branch in BRANCHES:
                    score_values[branch].extend(float(value) for value in values[branch])
                processed += len(labels_batch)
                if args.progress_every and processed % args.progress_every < len(labels_batch):
                    print(
                        f"[FedJam {shot}-shot] scored {processed} test images",
                        flush=True,
                    )

        for sample in fedjam_test_samples(args):
            if sample.name.startswith("train/"):
                raise RuntimeError("FedJam test iterator returned a training sample")
            pending.append(sample)
            if len(pending) >= args.stream_batch_size:
                score_pending(pending)
                pending = []
        if pending:
            score_pending(pending)
        if not binary_labels:
            raise RuntimeError("FedJam test split is empty")

        y_binary = np.asarray(binary_labels, dtype=np.int64)
        y_raw = np.asarray(raw_labels, dtype=np.int64)
        test_signature = sha256_text(names)
        if expected_test_signature is None:
            expected_test_signature = test_signature
        elif test_signature != expected_test_signature:
            raise RuntimeError("FedJam test set changed between shot values")
        arrays = {
            branch: np.asarray(score_values[branch], dtype=np.float32)
            for branch in BRANCHES
        }
        np.savez_compressed(
            score_root / f"fedjam-{shot}shot.npz",
            labels=y_binary,
            raw_labels=y_raw,
            names=np.asarray(names),
            **arrays,
        )
        for branch in BRANCHES:
            rows.append(
                {
                    "row_type": "pooled",
                    "method": METHOD_NAME,
                    "dataset": "fedjam",
                    "category": "ALL",
                    "scene": "official_test",
                    "jsr": "ALL",
                    "shot": shot,
                    "branch": branch,
                    "num_support": shot,
                    "num_test_normal": int((y_binary == 0).sum()),
                    "num_test_abnormal": int((y_binary == 1).sum()),
                    **metrics(y_binary, arrays[branch]),
                }
            )
            for attack_id, attack_name in FEDJAM_LABEL_NAMES.items():
                if attack_id == 0:
                    continue
                mask = (y_raw == 0) | (y_raw == attack_id)
                attack_labels = (y_raw[mask] == attack_id).astype(np.int64)
                rows.append(
                    {
                        "row_type": "cell",
                        "method": METHOD_NAME,
                        "dataset": "fedjam",
                        "category": attack_name,
                        "scene": "official_test",
                        "jsr": "ALL",
                        "shot": shot,
                        "branch": branch,
                        "num_support": shot,
                        "num_test_normal": int((attack_labels == 0).sum()),
                        "num_test_abnormal": int((attack_labels == 1).sum()),
                        **metrics(attack_labels, arrays[branch][mask]),
                    }
                )
        support_manifest["test_label_counts"] = {
            FEDJAM_LABEL_NAMES[label]: int((y_raw == label).sum())
            for label in FEDJAM_LABEL_NAMES
        }
        support_manifest["test_size"] = len(y_raw)
        del bundle, bank, score_values, arrays
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    support_manifest["fit_protocol"] = PROTOCOL_VERSION
    support_manifest["fit_models"] = fit_models
    support_manifest["fixed_test_names_sha256"] = expected_test_signature
    (Path(args.output_root) / "fedjam_support_manifest.json").write_text(
        json.dumps(support_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return rows, support_manifest


def load_ofdma_protocol(args) -> tuple[pd.DataFrame, dict, list[str], OFDMASpectrogramPreprocessor]:
    root = Path(args.ofdma_root)
    manifest = load_target_scene_manifest(root)
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing OFDMA protocol: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    normalization = protocol["source_protocol"]["normalization"]
    available = available_target_scenes(manifest, args.ofdma_split)
    scenes = list(args.ofdma_scene_ids or available)
    unknown = sorted(set(scenes) - set(available))
    if unknown:
        raise ValueError(f"Unknown OFDMA {args.ofdma_split} scenes: {unknown}")
    if not scenes:
        raise ValueError(f"No OFDMA scenes selected for split={args.ofdma_split}")
    preprocessor = OFDMASpectrogramPreprocessor(
        root,
        output_size=args.resize,
        subcarriers_per_rb=args.ofdma_subcarriers_per_rb,
        geometry=args.ofdma_geometry,
        min_db=float(normalization["min_db"]),
        max_db=float(normalization["max_db"]),
    )
    return manifest, protocol, scenes, preprocessor


def ofdma_sample(record, preprocessor: OFDMASpectrogramPreprocessor) -> Sample:
    gray = cv2.imread(str(record.image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(record.image_path)
    processed = preprocessor(gray)
    image_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    return Sample(
        record.image_path,
        int(record.label),
        f"target/ofdma/{record.target_scene_id}",
        f"{record.target_scene_id}::{record.observation_id}::su{record.su_id:02d}",
        image_bgr=np.ascontiguousarray(image_bgr),
        observation_id=record.observation_id,
        jammer_type=record.jammer_type,
        su_id=record.su_id,
    )


def aggregate_ofdma_scores(
    samples: Sequence[Sample],
    score_values: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(sample.observation_id, []).append(index)
    labels: list[int] = []
    observation_ids: list[str] = []
    jammer_types: list[str] = []
    aggregated = {branch: [] for branch in BRANCHES}
    for observation_id, indices in sorted(groups.items()):
        if len(indices) != NUM_SUS:
            raise RuntimeError(
                f"OFDMA observation {observation_id} has {len(indices)} SUs, expected {NUM_SUS}"
            )
        if sorted(samples[index].su_id for index in indices) != list(range(NUM_SUS)):
            raise RuntimeError(f"OFDMA observation {observation_id} has invalid SU ids")
        label_set = {samples[index].label for index in indices}
        jammer_set = {samples[index].jammer_type for index in indices}
        if len(label_set) != 1 or len(jammer_set) != 1:
            raise RuntimeError(f"Inconsistent OFDMA observation: {observation_id}")
        labels.append(next(iter(label_set)))
        jammer_types.append(next(iter(jammer_set)))
        observation_ids.append(observation_id)
        for branch in BRANCHES:
            aggregated[branch].append(float(np.max(score_values[branch][indices])))
    return (
        np.asarray(labels, dtype=np.int64),
        {branch: np.asarray(values, dtype=np.float32) for branch, values in aggregated.items()},
        np.asarray(observation_ids),
        np.asarray(jammer_types),
    )


def evaluate_ofdma(args, device: torch.device) -> tuple[list[dict], dict]:
    manifest, protocol, scenes, preprocessor = load_ofdma_protocol(args)
    root = Path(args.ofdma_root)
    rows: list[dict] = []
    fit_models: dict[str, dict] = {}
    score_root = Path(args.output_root) / "scores"
    score_root.mkdir(parents=True, exist_ok=True)
    for scene_index, scene in enumerate(scenes, start=1):
        print(f"[OFDMA {scene_index}/{len(scenes)}] scene={scene}", flush=True)
        test_records = build_target_scene_records(
            root,
            manifest,
            scene,
            split=args.ofdma_split,
            role="test",
            max_normal_observations=args.max_ofdma_normal_observations,
            max_anomaly_observations_per_type=args.max_ofdma_anomaly_observations_per_type,
        )
        test_identity_samples = [
            Sample(
                record.image_path,
                record.label,
                f"target/ofdma/{scene}",
                f"{scene}::{record.observation_id}::su{record.su_id:02d}",
                observation_id=record.observation_id,
                jammer_type=record.jammer_type,
                su_id=record.su_id,
            )
            for record in test_records
        ]
        for shot in args.shots:
            print(f"[OFDMA {scene}] fitting {shot}-shot support model", flush=True)
            support_records = build_target_scene_records(
                root,
                manifest,
                scene,
                split=args.ofdma_split,
                role="support",
                shot=shot,
                support_seed=args.ofdma_support_seed,
            )
            support_samples = [ofdma_sample(record, preprocessor) for record in support_records]
            fit_key = f"ofdma/{scene}/{shot}shot"
            bundle = build_or_load_bundle(
                args, fit_key, support_samples, test_identity_samples, device
            )
            fit_models[f"{scene}/{shot}"] = bundle.fit_metadata
            bank = encode_reference_bank(bundle.encoder, support_samples, args, device)
            score_bank = {branch: [] for branch in BRANCHES}
            for record_chunk in batched(test_records, args.stream_batch_size):
                samples = [ofdma_sample(record, preprocessor) for record in record_chunk]
                for batch in make_loader(samples, args, shuffle=False):
                    features = [
                        value.detach()
                        for value in bundle.encoder(batch["image"].to(device, non_blocking=True))
                    ]
                    values = score_encoded_features(args, bundle, features, bank, device)
                    for branch in BRANCHES:
                        score_bank[branch].extend(float(value) for value in values[branch])
            su_scores = {
                branch: np.asarray(values, dtype=np.float32)
                for branch, values in score_bank.items()
            }
            labels, observation_scores, observation_ids, jammer_types = aggregate_ofdma_scores(
                test_identity_samples, su_scores
            )
            np.savez_compressed(
                score_root / f"ofdma-{scene}-{shot}shot.npz",
                labels=labels,
                observation_ids=observation_ids,
                jammer_types=jammer_types,
                **observation_scores,
                su_reduction=np.asarray("max_over_21_sus"),
            )
            for branch in BRANCHES:
                rows.append(
                    {
                        "row_type": "cell",
                        "method": METHOD_NAME,
                        "dataset": "ofdma",
                        "category": "ALL",
                        "scene": scene,
                        "jsr": "ALL",
                        "shot": shot,
                        "branch": branch,
                        "num_support": len(support_samples),
                        "num_test_normal": int((labels == 0).sum()),
                        "num_test_abnormal": int((labels == 1).sum()),
                        **metrics(labels, observation_scores[branch]),
                    }
                )
            del bundle, bank, support_samples, score_bank, su_scores, observation_scores
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    metadata = {
        "protocol": protocol["source_protocol"]["protocol_name"],
        "dataset_root": str(root.resolve()),
        "split": args.ofdma_split,
        "scenes": scenes,
        "shots": list(args.shots),
        "fit_protocol": PROTOCOL_VERSION,
        "fit_models": fit_models,
        "support_rule": "first k ranked normal observations from the target scene",
        "su_reduction": "maximum score over 21 sensing units",
        "normalization": protocol["source_protocol"]["normalization"],
    }
    return append_macro_rows(rows), metadata


def validate_protocol(args) -> dict:
    """Audit the selected formal protocol without constructing a neural model."""

    if args.protocol in {"rf_target", "public_rf"}:
        jobs, metadata = (
            build_inhouse_jobs(args)
            if args.protocol == "rf_target"
            else build_public_jobs(args)
        )
        fit_audits: dict[str, dict] = {}
        counts = []
        for job in jobs:
            support, test = target_samples_for_job(job)
            fit_key = job["fit_key"]
            assert_sample_isolation(support, test, fit_key)
            audit = support_fit_metadata(support, fit_key)
            previous = fit_audits.get(fit_key)
            if previous is not None and previous["support_signature"] != audit["support_signature"]:
                raise RuntimeError(f"{fit_key}: grouped jobs use different supports")
            fit_audits[fit_key] = audit
            counts.append(
                {
                    "dataset": job["dataset"],
                    "category": job["category"],
                    "scene": job["scene"],
                    "jsr": job["jsr"],
                    "shot": job.get("shot"),
                    "support": len(support),
                    "test_normal": sum(sample.label == 0 for sample in test),
                    "test_abnormal": sum(sample.label == 1 for sample in test),
                }
            )
        return {**metadata, "fit_protocol": PROTOCOL_VERSION, "fit_audits": fit_audits, "selected_counts": counts}

    if args.protocol == "fedjam":
        support, metadata = select_fedjam_support(args)
        test_counts = {label: 0 for label in FEDJAM_LABEL_NAMES}
        test_samples: list[Sample] = []
        test_root = Path(args.fedjam_root) / "fedjam_dataset" / "test"
        for _image_bytes, label, name in iter_fedjam_rows(test_root):
            if (
                args.max_fedjam_test_per_label > 0
                and test_counts.get(label, 0) >= args.max_fedjam_test_per_label
            ):
                continue
            test_counts[label] = test_counts.get(label, 0) + 1
            test_samples.append(
                Sample(None, int(label != 0), "target/fedjam", f"test/{name}")
            )
        if not test_counts[0] or not sum(test_counts[label] for label in test_counts if label != 0):
            raise RuntimeError("FedJam test selection lacks benign or abnormal samples")
        metadata["test_label_counts"] = {
            FEDJAM_LABEL_NAMES[label]: count for label, count in test_counts.items()
        }
        metadata["fit_protocol"] = PROTOCOL_VERSION
        metadata["fit_audits"] = {}
        for shot in args.shots:
            shot_support = support[:shot]
            fit_key = f"fedjam/{shot}shot"
            assert_sample_isolation(shot_support, test_samples, fit_key)
            metadata["fit_audits"][str(shot)] = support_fit_metadata(
                shot_support, fit_key
            )
        return metadata

    if args.protocol == "ofdma":
        manifest, protocol, scenes, _preprocessor = load_ofdma_protocol(args)
        root = Path(args.ofdma_root)
        counts = []
        fit_audits: dict[str, dict] = {}
        for scene in scenes:
            test = build_target_scene_records(
                root,
                manifest,
                scene,
                split=args.ofdma_split,
                role="test",
                max_normal_observations=args.max_ofdma_normal_observations,
                max_anomaly_observations_per_type=args.max_ofdma_anomaly_observations_per_type,
            )
            test_samples = [
                Sample(
                    record.image_path,
                    record.label,
                    f"target/ofdma/{scene}",
                    f"{scene}::{record.observation_id}::su{record.su_id:02d}",
                )
                for record in test
            ]
            test_observations = len({record.observation_id for record in test})
            for shot in args.shots:
                support = build_target_scene_records(
                    root,
                    manifest,
                    scene,
                    split=args.ofdma_split,
                    role="support",
                    shot=shot,
                    support_seed=args.ofdma_support_seed,
                )
                support_samples = [
                    Sample(
                        record.image_path,
                        0,
                        f"target/ofdma/{scene}",
                        f"{scene}::{record.observation_id}::su{record.su_id:02d}",
                    )
                    for record in support
                ]
                fit_key = f"ofdma/{scene}/{shot}shot"
                assert_sample_isolation(support_samples, test_samples, fit_key)
                fit_audits[f"{scene}/{shot}"] = support_fit_metadata(
                    support_samples, fit_key
                )
                counts.append(
                    {
                        "scene": scene,
                        "shot": shot,
                        "support_observations": len(support) // NUM_SUS,
                        "support_images": len(support),
                        "test_observations": test_observations,
                        "test_images": len(test),
                    }
                )
        return {
            "protocol": protocol["source_protocol"]["protocol_name"],
            "split": args.ofdma_split,
            "scenes": scenes,
            "shots": list(args.shots),
            "fit_protocol": PROTOCOL_VERSION,
            "selected_counts": counts,
            "fit_audits": fit_audits,
        }
    raise ValueError(f"Unsupported protocol: {args.protocol}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt ResAD to the frozen SpectraMemAD few-shot protocols"
    )
    parser.add_argument(
        "--protocol",
        required=True,
        choices=("rf_target", "public_rf", "fedjam", "ofdma"),
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "analysis_outputs" / "resad_protocol_adapted"),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--max-memory-fraction", type=float, default=0.80)

    parser.add_argument("--rf-manifest", default=str(DEFAULT_RF_MANIFEST))
    parser.add_argument("--support-manifest", default="")
    parser.add_argument("--public-manifest-root", default=str(DEFAULT_PUBLIC_MANIFEST_ROOT))
    parser.add_argument("--normal-sampling", default="per_frequency")
    parser.add_argument("--rf-signals", nargs="+", default=list(RF_JSR_BY_SIGNAL))
    parser.add_argument("--public-rf-signals", nargs="+", default=list(PUBLIC_RF_JSRS))
    parser.add_argument("--rf-scenes", nargs="+", default=list(RF_SCENES))
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)

    parser.add_argument("--fedjam-root", default="/mnt/data/wangbei/data/FedJam")
    parser.add_argument("--max-fedjam-test-per-label", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)

    parser.add_argument("--ofdma-root", default=str(DEFAULT_TARGET_SCENE_ROOT))
    parser.add_argument("--ofdma-split", default="test")
    parser.add_argument("--ofdma-scene-ids", nargs="+", default=[])
    parser.add_argument("--ofdma-support-seed", type=int, default=None)
    parser.add_argument("--ofdma-subcarriers-per-rb", type=int, default=12)
    parser.add_argument(
        "--ofdma-geometry", choices=("letterbox", "square_warp"), default="letterbox"
    )
    parser.add_argument("--max-ofdma-normal-observations", type=int, default=0)
    parser.add_argument("--max-ofdma-anomaly-observations-per-type", type=int, default=0)

    parser.add_argument(
        "--backbone-checkpoint",
        default="/home/wangbei/.cache/torch/hub/checkpoints/wide_resnet50_2-95faca4d.pth",
    )
    parser.add_argument(
        "--force-refit",
        dest="force_refit",
        action="store_true",
        help="refit the target-support model even when a matching checkpoint exists",
    )
    parser.add_argument("--singleton-noise-std", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr-milestones", nargs="+", type=int, default=[70, 90])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stream-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--coupling-layers", type=int, default=10)
    parser.add_argument("--clamp-alpha", type=float, default=1.9)
    parser.add_argument("--pos-embed-dim", type=int, default=256)
    parser.add_argument("--pos-beta", type=float, default=0.05)
    parser.add_argument("--margin-tau", type=float, default=0.1)
    parser.add_argument("--bgspp-lambda", type=float, default=1.0)
    parser.add_argument("--flow-batch-points", type=int, default=512)
    parser.add_argument("--num-embeddings", type=int, default=1536)
    parser.add_argument("--fdm-alpha", type=float, default=0.4)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--gaussian-sigma", type=float, default=4.0)
    parser.add_argument("--calibration-points", type=int, default=200_000)
    parser.add_argument("--calibration-epsilon", type=float, default=1e-6)
    parser.add_argument("--query-chunk-size", type=int, default=512)
    parser.add_argument("--reference-chunk-size", type=int, default=4096)
    return parser.parse_args(argv)


def validate_args(args) -> None:
    args.shots = sorted(set(args.shots))
    if not args.shots or not set(args.shots).issubset({1, 2, 4}):
        raise ValueError("--shots must be selected from the formal values: 1, 2, 4")
    if args.protocol == "rf_target" and args.shots != [1, 2, 4]:
        # In-house RF has a per-frequency support protocol; --shots is unused.
        print("[note] --shots is ignored by rf_target", flush=True)
    nonnegative = (
        "max_test_normals",
        "max_abnormals",
        "max_fedjam_test_per_label",
        "max_ofdma_normal_observations",
        "max_ofdma_anomaly_observations_per_type",
    )
    for name in nonnegative:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    positive = (
        "epochs",
        "batch_size",
        "stream_batch_size",
        "num_embeddings",
        "calibration_points",
        "query_chunk_size",
        "reference_chunk_size",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.singleton_noise_std <= 0:
        raise ValueError("--singleton-noise-std must be positive")
    if not 0.0 <= args.max_memory_fraction <= 1.0:
        raise ValueError("--max-memory-fraction must be in [0, 1]")
    if not args.use_cpu and args.gpu_id == 0:
        raise ValueError("GPU0 is disabled for this evaluator; choose --gpu-id 1 or another GPU")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    if args.validate_only:
        audit = validate_protocol(args)
        print(json.dumps(audit, indent=2, ensure_ascii=False, default=str))
        print("[validated] protocol, sample counts, and support/test isolation are consistent")
        return

    if not args.use_cpu:
        # Set visibility before the first CUDA availability check.  The process
        # sees the selected physical GPU as cuda:0, but physical GPU0 is never
        # accepted by validate_args().
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    seed_everything(args.seed)
    if args.use_cpu:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --use-cpu only for a small debug run")
        device = torch.device("cuda:0")
        if args.max_memory_fraction > 0:
            torch.cuda.set_per_process_memory_fraction(args.max_memory_fraction, device=0)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.protocol == "rf_target":
        jobs, protocol_metadata = build_inhouse_jobs(args)
        raw_rows, fit_models = evaluate_path_jobs(args, jobs, device)
        rows = append_macro_rows(raw_rows)
        protocol_metadata["fit_protocol"] = PROTOCOL_VERSION
        protocol_metadata["fit_models"] = fit_models
    elif args.protocol == "public_rf":
        jobs, protocol_metadata = build_public_jobs(args)
        raw_rows, fit_models = evaluate_path_jobs(args, jobs, device)
        rows = append_macro_rows(raw_rows)
        protocol_metadata["fit_protocol"] = PROTOCOL_VERSION
        protocol_metadata["fit_models"] = fit_models
    elif args.protocol == "fedjam":
        rows, protocol_metadata = evaluate_fedjam(args, device)
    else:
        rows, protocol_metadata = evaluate_ofdma(args, device)

    metrics_path = output_root / "metrics.csv"
    write_csv(metrics_path, rows)
    result = {
        "method": METHOD_NAME,
        "method_scope": "ResAD mechanism adapted to the target-support-only project protocol",
        "target_adaptation": "heads, calibration, and reference bank use only the selected target normal support",
        "training_data": "current dataset and evaluation unit support normals only; no cross-dataset source, anomalies, or masks",
        "test_batch_statistics": "not used",
        "protocol": protocol_metadata,
        "args": vars(args),
        "metrics_file": str(metrics_path),
        "metrics": rows,
    }
    (output_root / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"[done] wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
