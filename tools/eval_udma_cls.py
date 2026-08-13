#!/usr/bin/env python
"""Unified support-only evaluation entry point for the UDMA reimplementation.

The UDMA paper does not identify its pretrained reference network.  This runner
therefore exposes two frozen choices: an ImageNet ResNet18 and a deterministic,
offline spectral-statistics extractor.  Every fitted component sees normal
support images only; the reference extractor is always frozen, and the teacher,
students, memory, and support statistics are frozen before query scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.udma import UDMA  # noqa: E402
from tools.eval_cls_aux_cnn_gallery import (  # noqa: E402
    FEWSHOT_SAMPLING_CHOICES,
    PUBLIC_RF_JSRS,
    public_rf_jobs,
    rf_target_jobs,
)
from tools.ofdma_fewshot_baseline_common import load_sample_image  # noqa: E402
from utils.training_utils import setup_seed  # noqa: E402


METHOD_NAME = "udma_reimplementation"
REFERENCE_CHOICES = ("spectral_stats", "resnet18_imagenet")


def _arg(args, name: str, default):
    return getattr(args, name, default)


class UDMAPathDataset(Dataset):
    """Read a canonical job's image paths as fixed-size grayscale tensors."""

    def __init__(self, samples, image_height: int, image_width: int):
        self.samples = list(samples)
        self.image_height = int(image_height)
        self.image_width = int(image_width)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        loaded = load_sample_image(sample)
        if not isinstance(loaded, Image.Image):
            loaded = Image.fromarray(np.asarray(loaded))
        image = np.asarray(loaded.convert("L"), dtype=np.uint8)
        interpolation = (
            cv2.INTER_AREA
            if image.shape[0] > self.image_height or image.shape[1] > self.image_width
            else cv2.INTER_LINEAR
        )
        image = cv2.resize(
            image,
            (self.image_width, self.image_height),
            interpolation=interpolation,
        )
        tensor = torch.from_numpy(image).unsqueeze(0).float().div_(255.0)
        return (
            tensor,
            int(sample["label"]),
            str(sample["path"]),
            str(sample["name"]),
        )


class FrozenSpectralStatistics(nn.Module):
    """Auditable offline reference: pooled intensity/energy/gradient maps."""

    output_dim = 16

    def forward(self, x: Tensor) -> Tensor:
        horizontal = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
        vertical = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
        maps = torch.cat((x, x.square(), horizontal.abs(), vertical.abs()), dim=1)
        return F.adaptive_avg_pool2d(maps, output_size=(2, 2)).flatten(1)


class FrozenResNet18(nn.Module):
    """Fixed ImageNet-1K ResNet18 global features for teacher distillation."""

    output_dim = 512

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(network.children())[:-1])
        self.register_buffer(
            "mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        rgb = x.repeat(1, 3, 1, 1)
        rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
        return self.features((rgb - self.mean) / self.std).flatten(1)


def make_reference_extractor(args, device) -> tuple[nn.Module, int]:
    choice = _arg(args, "reference_extractor", "resnet18_imagenet")
    if choice == "spectral_stats":
        extractor: nn.Module = FrozenSpectralStatistics()
    elif choice == "resnet18_imagenet":
        extractor = FrozenResNet18()
    else:
        raise ValueError(f"Unsupported reference extractor: {choice}")
    extractor.to(device).eval()
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    return extractor, int(extractor.output_dim)


def make_loader(samples, args, device, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(int(_arg(args, "seed", 111)))
    return DataLoader(
        UDMAPathDataset(
            samples,
            _arg(args, "image_height", 64),
            _arg(args, "image_width", 64),
        ),
        batch_size=int(_arg(args, "batch_size", 8)),
        shuffle=shuffle,
        num_workers=int(_arg(args, "num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
    )


def make_model(args, reference_dim: int, device) -> UDMA:
    model = UDMA(
        in_channels=1,
        feature_channels=int(_arg(args, "feature_channels", 16)),
        teacher_hidden=(
            int(_arg(args, "teacher_hidden_1", 16)),
            int(_arg(args, "teacher_hidden_2", 32)),
        ),
        encoder_channels=(
            int(_arg(args, "encoder_channels_1", 8)),
            int(_arg(args, "encoder_channels_2", 16)),
            int(_arg(args, "encoder_channels_3", 16)),
        ),
        latent_channels=int(_arg(args, "latent_channels", 16)),
        memory_size=int(_arg(args, "memory_size", 10)),
        reference_dim=reference_dim,
        shrink_threshold=_arg(args, "shrink_threshold", None),
        update_threshold=_arg(args, "update_threshold", None),
        memory_update_rate=float(_arg(args, "memory_update_rate", 0.1)),
        separateness_margin=float(_arg(args, "separateness_margin", 1.0)),
        discrepancy_weights=(
            float(_arg(args, "weight_teacher_ae", 0.5)),
            float(_arg(args, "weight_teacher_memae", 0.5)),
            float(_arg(args, "weight_ae_memae", 0.5)),
        ),
        compactness_weight=float(_arg(args, "compactness_weight", 0.1)),
        separateness_weight=float(_arg(args, "separateness_weight", 0.1)),
    )
    return model.to(device)


def _log_epoch(stage: str, epoch: int, total: int, loss: float, args) -> None:
    interval = int(_arg(args, "log_every", 5))
    if interval > 0 and (epoch == 1 or epoch == total or epoch % interval == 0):
        print(f"    {stage} epoch {epoch}/{total} loss={loss:.6f}", flush=True)


@torch.no_grad()
def _fit_teacher_statistics(model: UDMA, loader, device) -> None:
    total = None
    total_square = None
    count = 0
    model.teacher.eval()
    for data, _label, _path, _name in loader:
        features = model.teacher(data.to(device, non_blocking=True))
        batch_sum = features.double().sum(dim=(0, 2, 3), keepdim=True)
        batch_square = features.double().square().sum(dim=(0, 2, 3), keepdim=True)
        total = batch_sum if total is None else total + batch_sum
        total_square = (
            batch_square if total_square is None else total_square + batch_square
        )
        count += features.shape[0] * features.shape[2] * features.shape[3]
    if not count:
        raise RuntimeError("Cannot fit teacher statistics from an empty support set")
    mean = total / count
    variance = (total_square / count - mean.square()).clamp_min(model.eps**2)
    model.set_teacher_statistics(mean.float(), variance.sqrt().float())


def fit_udma(train_samples, args, device):
    """Fit UDMA from confirmed-normal support samples and return a frozen model."""

    train_samples = list(train_samples)
    if not train_samples:
        raise ValueError("UDMA requires at least one normal support sample")
    non_normal = [sample for sample in train_samples if int(sample["label"]) != 0]
    if non_normal:
        raise ValueError("UDMA training is support-only; non-normal samples were provided")

    setup_seed(int(_arg(args, "seed", 111)))
    reference, reference_dim = make_reference_extractor(args, device)
    model = make_model(args, reference_dim, device)
    loader = make_loader(train_samples, args, device, shuffle=True)

    teacher_epochs = int(_arg(args, "teacher_epochs", 5))
    student_epochs = int(_arg(args, "student_epochs", 10))
    if teacher_epochs < 1 or student_epochs < 1:
        raise ValueError("teacher_epochs and student_epochs must both be positive")

    model.configure_phase("teacher")
    teacher_parameters = list(model.teacher.parameters()) + list(
        model.teacher_projection.parameters()
    )
    teacher_optimizer = Adam(
        teacher_parameters,
        lr=float(_arg(args, "teacher_lr", 1e-3)),
        weight_decay=float(_arg(args, "weight_decay", 0.0)),
    )
    teacher_history = []
    for epoch in range(1, teacher_epochs + 1):
        losses = []
        for data, _label, _path, _name in loader:
            data = data.to(device, non_blocking=True)
            with torch.no_grad():
                reference_features = reference(data)
            output = model.teacher_distillation_loss(data, reference_features)
            teacher_optimizer.zero_grad(set_to_none=True)
            output["loss"].backward()
            teacher_optimizer.step()
            losses.append(float(output["loss"].detach().cpu()))
        epoch_loss = float(np.mean(losses))
        teacher_history.append(epoch_loss)
        _log_epoch("teacher", epoch, teacher_epochs, epoch_loss, args)

    statistics_loader = make_loader(train_samples, args, device, shuffle=False)
    _fit_teacher_statistics(model, statistics_loader, device)

    model.configure_phase("students")
    student_parameters = list(model.ae_student.parameters()) + list(
        model.memae_student.parameters()
    )
    student_optimizer = Adam(
        student_parameters,
        lr=float(_arg(args, "student_lr", 1e-3)),
        weight_decay=float(_arg(args, "weight_decay", 0.0)),
    )
    student_history = []
    for epoch in range(1, student_epochs + 1):
        losses = []
        for data, _label, _path, _name in loader:
            data = data.to(device, non_blocking=True)
            output = model.student_training_loss(data)
            student_optimizer.zero_grad(set_to_none=True)
            output["loss"].backward()
            student_optimizer.step()
            model.update_memory(output["memory_queries"].detach())
            losses.append(float(output["loss"].detach().cpu()))
        epoch_loss = float(np.mean(losses))
        student_history.append(epoch_loss)
        _log_epoch("students", epoch, student_epochs, epoch_loss, args)

    model.configure_phase("inference")
    model.eval()
    model.training_history = {
        "teacher_loss": teacher_history,
        "student_loss": student_history,
    }
    model.reference_extractor_name = str(
        _arg(args, "reference_extractor", "resnet18_imagenet")
    )
    return model


def safe_metrics(labels, scores) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if np.unique(labels).size > 2:
        labels = (labels != 0).astype(np.int32)
    if labels.size == 0 or np.unique(labels).size < 2:
        return {
            "image_auroc": float("nan"),
            "image_auprc": float("nan"),
            "fpr95": float("nan"),
        }
    fpr, tpr, _thresholds = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "image_auroc": float(roc_auc_score(labels, scores) * 100.0),
        "image_auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if reached.size else float("nan"),
    }


def _score_stem(job) -> str:
    fields = (
        job.get("dataset", "dataset"),
        job.get("category", "category"),
        job.get("scene", "scene"),
        job.get("jsr", "none"),
    )
    return "-".join(str(value) for value in fields).replace("/", "_")


@torch.no_grad()
def predict_job(model, job, args, device):
    """Score one canonical job with all fitted UDMA state frozen."""

    model.configure_phase("inference")
    model.eval()
    loader = make_loader(job["test_samples"], args, device, shuffle=False)
    labels, scores, image_paths, names = [], [], [], []
    for data, label, path, name in loader:
        output = model.anomaly_outputs(data.to(device, non_blocking=True))
        labels.extend(int(value) for value in label.numpy().tolist())
        scores.extend(float(value) for value in output["score"].cpu().numpy().tolist())
        image_paths.extend(str(value) for value in path)
        names.extend(str(value) for value in name)

    labels_array = np.asarray(labels, dtype=np.int32)
    scores_array = np.asarray(scores, dtype=np.float32)
    score_root = Path(args.output_root) / "scores"
    score_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        score_root / f"{_score_stem(job)}-scores.npz",
        labels=labels_array,
        scores=scores_array,
        names=np.asarray(names),
        image_paths=np.asarray(image_paths),
        method=np.asarray(METHOD_NAME),
    )
    return {
        "method": METHOD_NAME,
        "dataset": str(job["dataset"]),
        "category": str(job["category"]),
        "scene": str(job["scene"]),
        "jsr": str(job["jsr"]),
        "shot": str(job.get("shot", _arg(args, "normal_sampling", "support"))),
        "num_train_normal": len(job["train_samples"]),
        "num_test_normal": int((labels_array == 0).sum()),
        "num_test_abnormal": int((labels_array != 0).sum()),
        **safe_metrics(labels_array, scores_array),
    }


def train_signature(job) -> tuple[str, ...]:
    return tuple(str(sample["path"]) for sample in job["train_samples"])


def _limit_samples(samples, normal_limit: int, abnormal_limit: int):
    normal_seen = 0
    abnormal_seen = 0
    selected = []
    for sample in samples:
        if int(sample["label"]) == 0:
            if normal_limit and normal_seen >= normal_limit:
                continue
            normal_seen += 1
        else:
            if abnormal_limit and abnormal_seen >= abnormal_limit:
                continue
            abnormal_seen += 1
        selected.append(sample)
    return selected


def apply_smoke_limits(jobs, args) -> list[dict]:
    max_jobs = int(_arg(args, "max_jobs", 0))
    if max_jobs > 0:
        jobs = list(jobs)[:max_jobs]
    limited = []
    for source in jobs:
        job = dict(source)
        max_train = int(_arg(args, "max_train_samples", 0))
        job["train_samples"] = list(source["train_samples"])[
            : max_train or None
        ]
        job["test_samples"] = _limit_samples(
            source["test_samples"],
            int(_arg(args, "max_test_normal", 0)),
            int(_arg(args, "max_test_abnormal", 0)),
        )
        limited.append(job)
    return limited


def run_jobs(jobs, args, device) -> list[dict]:
    groups = {}
    for job in apply_smoke_limits(jobs, args):
        groups.setdefault(train_signature(job), []).append(job)
    rows = []
    for group_index, group in enumerate(groups.values(), 1):
        first = group[0]
        if not first["train_samples"]:
            raise RuntimeError("No normal support samples remain after smoke limits")
        print(
            f"[fit {group_index}/{len(groups)}] normals={len(first['train_samples'])} "
            f"jobs={len(group)}",
            flush=True,
        )
        model = fit_udma(first["train_samples"], args, device)
        history = model.training_history
        for job in group:
            if not any(int(sample["label"]) != 0 for sample in job["test_samples"]):
                raise RuntimeError(
                    "Each evaluated job must retain at least one abnormal sample"
                )
            row = predict_job(model, job, args, device)
            row["teacher_loss_final"] = history["teacher_loss"][-1]
            row["student_loss_final"] = history["student_loss"][-1]
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (row["dataset"], row["category"], row["scene"], row["jsr"]),
    )


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _nanmean(rows, key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def build_jobs(args) -> list[dict]:
    if args.protocol == "public_rf" and not args.support_manifest:
        raise ValueError("public_rf requires an explicit --support-manifest")
    if args.protocol == "public_rf" and not Path(args.support_manifest).is_file():
        raise FileNotFoundError(args.support_manifest)
    jobs = rf_target_jobs(args) if args.protocol == "rf_target" else public_rf_jobs(args)
    return jobs


def protocol_payload(args) -> dict:
    return {
        "method": METHOD_NAME,
        "protocol": args.protocol,
        "normal_sampling": args.normal_sampling,
        "support_only_training": True,
        "test_time_state": "fully_frozen",
        "support_manifest": getattr(args, "support_manifest", "") or None,
        "support_manifest_sha256": getattr(args, "support_manifest_sha256", None),
        "test_normal_paths_sha256": getattr(args, "test_normal_paths_sha256", None),
        "support_policy": getattr(args, "support_policy", None),
        "per_frequency_k": getattr(args, "per_frequency_k", None),
        "reference_extractor": args.reference_extractor,
        "reference_extractor_frozen": True,
        "reference_extractor_note": (
            "deterministic 2x2 pooled intensity, squared-intensity, and absolute "
            "horizontal/vertical gradient maps"
            if args.reference_extractor == "spectral_stats"
            else "torchvision ResNet18 IMAGENET1K_V1 global features"
        ),
        "teacher_epochs": args.teacher_epochs,
        "student_epochs": args.student_epochs,
        "teacher_lr": args.teacher_lr,
        "student_lr": args.student_lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "feature_channels": args.feature_channels,
        "teacher_hidden": [args.teacher_hidden_1, args.teacher_hidden_2],
        "encoder_channels": [
            args.encoder_channels_1,
            args.encoder_channels_2,
            args.encoder_channels_3,
        ],
        "latent_channels": args.latent_channels,
        "memory_size": args.memory_size,
        "shrink_threshold": args.shrink_threshold,
        "update_threshold": args.update_threshold,
        "memory_update_rate": args.memory_update_rate,
        "separateness_margin": args.separateness_margin,
        "discrepancy_weights": [
            args.weight_teacher_ae,
            args.weight_teacher_memae,
            args.weight_ae_memae,
        ],
        "compactness_weight": args.compactness_weight,
        "separateness_weight": args.separateness_weight,
        "public_rf_signals": list(args.public_rf_signals),
        "score": "mean of fused teacher-AE, teacher-MemAE, and AE-MemAE maps",
        "seed": args.seed,
        "smoke_limits": {
            "max_jobs": args.max_jobs,
            "max_train_samples": args.max_train_samples,
            "max_test_normal": args.max_test_normal,
            "max_test_abnormal": args.max_test_abnormal,
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("rf_target", "public_rf"), required=True)
    parser.add_argument("--output-root", default="analysis_outputs/udma_cls")
    parser.add_argument(
        "--normal-sampling",
        choices=FEWSHOT_SAMPLING_CHOICES,
        default="per_frequency",
    )
    parser.add_argument("--support-manifest", default="")
    parser.add_argument("--support-seed", type=int, default=None)
    parser.add_argument(
        "--public-rf-signals",
        nargs="+",
        default=list(PUBLIC_RF_JSRS),
        choices=list(PUBLIC_RF_JSRS),
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--reference-extractor",
        choices=REFERENCE_CHOICES,
        default="resnet18_imagenet",
    )
    parser.add_argument("--teacher-epochs", type=int, default=5)
    parser.add_argument("--student-epochs", type=int, default=10)
    parser.add_argument("--teacher-lr", type=float, default=1e-3)
    parser.add_argument("--student-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--feature-channels", type=int, default=16)
    parser.add_argument("--teacher-hidden-1", type=int, default=16)
    parser.add_argument("--teacher-hidden-2", type=int, default=32)
    parser.add_argument("--encoder-channels-1", type=int, default=8)
    parser.add_argument("--encoder-channels-2", type=int, default=16)
    parser.add_argument("--encoder-channels-3", type=int, default=16)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--memory-size", type=int, default=10)
    parser.add_argument("--shrink-threshold", type=float, default=None)
    parser.add_argument("--update-threshold", type=float, default=None)
    parser.add_argument("--memory-update-rate", type=float, default=0.1)
    parser.add_argument("--separateness-margin", type=float, default=1.0)
    parser.add_argument("--weight-teacher-ae", type=float, default=0.5)
    parser.add_argument("--weight-teacher-memae", type=float, default=0.5)
    parser.add_argument("--weight-ae-memae", type=float, default=0.5)
    parser.add_argument("--compactness-weight", type=float, default=0.1)
    parser.add_argument("--separateness-weight", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-normal", type=int, default=0)
    parser.add_argument("--max-test-abnormal", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_seed(args.seed)
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(
        "cpu"
        if args.use_cpu or not torch.cuda.is_available()
        else f"cuda:{args.gpu_id}"
    )
    jobs = build_jobs(args)
    rows = run_jobs(jobs, args, device)
    if not rows:
        raise RuntimeError(f"No UDMA evaluation jobs for {args.protocol}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "results_udma_cls.csv"
    write_csv(result_path, rows)
    protocol = protocol_payload(args)
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        **protocol,
        "num_jobs": len(rows),
        "image_auroc_macro": _nanmean(rows, "image_auroc"),
        "image_auprc_macro": _nanmean(rows, "image_auprc"),
        "fpr95_macro": _nanmean(rows, "fpr95"),
        "results_csv": result_path.name,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {result_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return rows


if __name__ == "__main__":
    main()
