#!/usr/bin/env python
"""Evaluate the fixed ResNet18 auxiliary CNN branch used by the formal method."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.rf_target import (
    RF_JSR_BY_SIGNAL,
    RF_SCENES,
    RF_TARGET_SIGNALS,
    collect_rf_target_samples,
)
from tools.eval_seg_resnet_gallery_fusion import (
    ResNet18LocalEncoder,
    cnn_tta_batch,
    downsample_gallery,
    farthest_first_coreset,
    flatten_feature_map,
)
from utils.rf_frequency_sampling import maybe_select_one_per_frequency_band
from utils.public_rf_support import select_support_and_test_paths
from utils.rf_scene_support import (
    cell_entry,
    ensure_rf_target_scene_manifest,
    scene_support_entry,
)
from utils.training_utils import setup_seed


PUBLIC_RF_ROOT = Path("/mnt/data/wangbei/data/RF_SPE_PNG")
PUBLIC_RF_NORMAL_DIR = PUBLIC_RF_ROOT / "RF_Spectrum_Public_Dataset"
PUBLIC_RF_JSRS = {
    "burst": ("m30db", "m40db", "m50db"),
    "chirp": ("m40db", "m50db", "m55db"),
    "dsss": ("m30db", "m40db", "m50db"),
    "pulse": ("m30db", "m40db", "m50db"),
    "deceptive": ("m10db", "m20db", "m30db"),
}
SPECTRUM_ROOT = REPO_ROOT / "datasets" / "spectrum"
SPECTRUM_CATEGORIES = ("16QAM", "CHIRP", "GMSK", "QPSK")
FEWSHOT_SAMPLING_CHOICES = ("per_frequency", "1shot", "2shot", "4shot")
SCORE_KEYS = {
    "rf_target": "resnet18_layer3_top0.1_scores",
    "public_rf": "resnet18_layer3_scores",
    "spectrum": "resnet18_layer3_scores",
}
CNN_FEATURE_MODES = (
    "layer3",
    "layer1_layer4_concat",
    "layer1_layer3_concat",
    "layer2_layer3_concat",
    "layer2_layer4_concat",
)
CNN_FEATURE_PAIRS = {
    "layer1_layer4_concat": ("layer1", "layer4"),
    "layer1_layer3_concat": ("layer1", "layer3"),
    "layer2_layer3_concat": ("layer2", "layer3"),
    "layer2_layer4_concat": ("layer2", "layer4"),
}


class RawImageDataset(Dataset):
    def __init__(self, samples, square_resize: bool):
        self.samples = list(samples)
        self.square_resize = bool(square_resize)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = cv2.imread(str(sample["path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(sample["path"])
        if self.square_resize:
            height, width = image.shape[:2]
            size = min(max(height, width), 1024)
            image = cv2.resize(image, (size, size))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        return (
            image,
            mask,
            int(sample["label"]),
            sample["name"],
            "abnormal" if int(sample["label"]) else "normal",
        )


def make_sample(path, label: int, name: str) -> dict:
    return {
        "path": str(path),
        "label": int(label),
        "name": str(name),
    }


def safe_auc(labels, scores) -> float:
    if len(set(int(value) for value in labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores) * 100.0)


def public_normal_paths() -> list[Path]:
    records = sorted(
        path
        for path in PUBLIC_RF_NORMAL_DIR.iterdir()
        if path.is_dir() and path.name.startswith("MeasRes_")
    )
    paths = [
        image
        for record in records
        for image in sorted(record.glob("*.png"))
    ]
    if not paths:
        raise FileNotFoundError(
            f"No public normal images found under {PUBLIC_RF_NORMAL_DIR}"
        )
    return paths


def public_rf_jobs(args) -> list[dict]:
    normal_paths = public_normal_paths()
    support_paths, test_normal_paths, manifest = select_support_and_test_paths(
        normal_paths,
        normal_sampling=args.normal_sampling,
        seed=getattr(args, "support_seed", None),
        manifest_path=getattr(args, "support_manifest", None),
    )
    args.support_manifest_sha256 = (
        manifest.get("support_paths_sha256")
        if manifest is not None
        else hashlib.sha256(
            "\n".join(str(path) for path in support_paths).encode("utf-8")
        ).hexdigest()
    )
    args.test_normal_paths_sha256 = (
        manifest.get("test_paths_sha256")
        if manifest is not None
        else hashlib.sha256(
            "\n".join(str(path) for path in test_normal_paths).encode("utf-8")
        ).hexdigest()
    )
    args.support_policy = (
        manifest.get("support_policy")
        if manifest is not None
        else str(args.normal_sampling)
    )
    args.per_frequency_k = (
        int(manifest["per_frequency_k"])
        if manifest is not None and manifest.get("per_frequency_k") is not None
        else None
    )
    shot_label = (
        f"k{args.per_frequency_k}_per_frequency"
        if args.per_frequency_k is not None
        else str(args.normal_sampling)
    )
    train_samples = [
        make_sample(path, 0, f"public-train-{path.stem}")
        for path in support_paths
    ]

    jobs = []
    signals = getattr(args, "public_rf_signals", tuple(PUBLIC_RF_JSRS))
    for signal in signals:
        jsrs = PUBLIC_RF_JSRS[signal]
        for jsr in jsrs:
            test_samples = [
                make_sample(path, 0, f"{signal}-{jsr}-normal-{path.stem}")
                for path in test_normal_paths
            ]
            abnormal_paths = sorted(
                (PUBLIC_RF_ROOT / signal / "abnormal" / jsr).glob("*.png")
            )
            test_samples.extend(
                make_sample(path, 1, f"{signal}-{jsr}-abnormal-{path.stem}")
                for path in abnormal_paths
            )
            jobs.append(
                {
                    "dataset": "public_rf",
                    "category": signal,
                    "scene": "RF_SPE_PNG_public",
                    "jsr": jsr,
                    "train_samples": train_samples,
                    "test_samples": test_samples,
                    "square_resize": True,
                    "shot": shot_label,
                    "support_policy": args.support_policy,
                    "per_frequency_k": args.per_frequency_k,
                }
            )
    return jobs


def spectrum_jobs(args) -> list[dict]:
    if args.normal_sampling == "per_frequency":
        raise ValueError(
            "Spectrum has no frequency-band metadata; use 1shot, 2shot, or 4shot."
        )

    jobs = []
    for category in SPECTRUM_CATEGORIES:
        category_root = SPECTRUM_ROOT / category
        train_paths = maybe_select_one_per_frequency_band(
            sorted((category_root / "train" / "good").glob("*.png")),
            args.normal_sampling,
        )
        good_paths = sorted((category_root / "test" / "good").glob("*.png"))
        bad_paths = sorted((category_root / "test" / "bad").glob("*.png"))
        jobs.append(
            {
                "dataset": "spectrum",
                "category": category,
                "scene": "datasets/spectrum",
                "jsr": "none",
                "train_samples": [
                    make_sample(path, 0, f"{category}-train-{path.stem}")
                    for path in train_paths
                ],
                "test_samples": [
                    make_sample(path, 0, f"{category}-good-{path.stem}")
                    for path in good_paths
                ]
                + [
                    make_sample(path, 1, f"{category}-bad-{path.stem}")
                    for path in bad_paths
                ],
                "square_resize": False,
            }
        )
    return jobs


def rf_target_jobs(args) -> list[dict]:
    manifest_path = (
        Path(args.support_manifest)
        if args.support_manifest
        else Path(args.output_root) / "support_manifest.json"
    )
    manifest = ensure_rf_target_scene_manifest(
        manifest_path,
        collect_samples=collect_rf_target_samples,
        signals=RF_TARGET_SIGNALS,
        scenes=RF_SCENES,
        jsr_by_signal=RF_JSR_BY_SIGNAL,
        normal_sampling=args.normal_sampling,
        seed=(args.support_seed if args.support_seed is not None else args.seed),
    )
    args.support_manifest = str(manifest_path)
    args.support_manifest_sha256 = manifest["manifest_sha256"]

    jobs = []
    for scene in RF_SCENES:
        support = scene_support_entry(manifest, scene)["support"]
        train_samples = [
            make_sample(
                item["path"],
                0,
                f"scene-support-{scene}-{Path(item['path']).stem}",
            )
            for item in support
        ]
        for signal in RF_TARGET_SIGNALS:
            for jsr in RF_JSR_BY_SIGNAL[signal]:
                manifest_cell = cell_entry(manifest, signal, scene, jsr)
                allowed_normals = {
                    item["path"] for item in manifest_cell["test_normals"]
                }
                allowed_abnormals = {
                    item["path"] for item in manifest_cell["test_abnormals"]
                }
                raw_samples = collect_rf_target_samples(
                    signal,
                    scene,
                    jsr,
                    "test",
                    k_shot=1,
                )
                test_samples = [
                    make_sample(
                        sample[0],
                        sample[2],
                        (
                            f"{signal}-{scene}-{jsr}-{sample[3]}-"
                            f"{Path(sample[0]).stem}"
                        ),
                    )
                    for sample in raw_samples
                    if (
                        int(sample[2]) == 0
                        and str(sample[0]) in allowed_normals
                    )
                    or (
                        int(sample[2]) == 1
                        and str(sample[0]) in allowed_abnormals
                    )
                ]
                jobs.append(
                    {
                        "dataset": "rf_target",
                        "category": signal,
                        "scene": scene,
                        "jsr": jsr,
                        "train_samples": train_samples,
                        "test_samples": test_samples,
                        "square_resize": True,
                    }
                )
    return jobs


def make_loader(samples, args, square_resize: bool) -> DataLoader:
    return DataLoader(
        RawImageDataset(samples, square_resize),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def cnn_feature_map(
    features: dict[str, torch.Tensor],
    feature_mode: str,
) -> torch.Tensor:
    """Select a CNN descriptor while keeping the baseline patch grid fixed.

    The experimental descriptor aligns layer2 and layer4 to the current
    layer3 grid (14x14 for a 224px ResNet input), normalizes each branch before
    concatenation, and normalizes the concatenated descriptor once more.
    """

    if feature_mode == "layer3":
        return features["layer3"]
    if feature_mode not in CNN_FEATURE_PAIRS:
        raise ValueError(f"Unsupported CNN feature mode: {feature_mode}")

    target_size = features["layer3"].shape[-2:]
    left_name, right_name = CNN_FEATURE_PAIRS[feature_mode]
    left = F.interpolate(
        features[left_name],
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    right = F.interpolate(
        features[right_name],
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    left = F.normalize(left.float(), dim=1)
    right = F.normalize(right.float(), dim=1)
    return F.normalize(torch.cat([left, right], dim=1), dim=1)


@torch.no_grad()
def build_cnn_gallery(encoder, train_loader, gallery_args, device) -> torch.Tensor:
    """Build one normal patch gallery for either CNN feature mode."""

    encoder.eval()
    chunks = []
    for data, _mask, _label, _name, _img_type in tqdm(
        train_loader,
        desc="Build CNN normal gallery",
        leave=False,
    ):
        data_variant = cnn_tta_batch(data, "identity", gallery_args)
        raw = data_variant.permute(0, 3, 1, 2).to(device, non_blocking=True)
        features = cnn_feature_map(
            encoder(raw),
            gallery_args.cnn_feature_mode,
        )
        chunks.append(flatten_feature_map(features).cpu())

    gallery = torch.cat(chunks, dim=0).to(device)
    gallery_seed = gallery_args.seed + len(gallery_args.cnn_feature_mode)
    gallery = downsample_gallery(
        gallery,
        gallery_args.max_gallery_patches,
        gallery_seed,
    )
    gallery = farthest_first_coreset(
        gallery,
        int(gallery_args.cnn_coreset_size),
        gallery_seed,
    )
    gallery = F.normalize(gallery, dim=1)
    print(
        f"[cnn_gallery] mode={gallery_args.cnn_feature_mode} "
        f"patches={gallery.shape[0]} dim={gallery.shape[1]} "
        f"coreset_size={gallery_args.cnn_coreset_size}"
    )
    return gallery


@torch.no_grad()
def image_scores(
    feature_map: torch.Tensor,
    gallery: torch.Tensor,
    chunk_size: int,
) -> np.ndarray:
    batch_size, _channels, height, width = feature_map.shape
    probes = flatten_feature_map(F.normalize(feature_map.float(), dim=1))
    patch_scores = []
    for start in range(0, probes.shape[0], chunk_size):
        similarities = probes[start : start + chunk_size] @ gallery.t()
        patch_scores.append(
            (1.0 - similarities.max(dim=1).values).cpu()
        )
    patch_scores = torch.cat(patch_scores).reshape(
        batch_size,
        height * width,
    )
    count = max(1, int(round(patch_scores.shape[1] * 0.1)))
    return patch_scores.topk(count, dim=1).values.mean(dim=1).numpy()


@torch.no_grad()
def evaluate_job(encoder, gallery, job, args, device) -> dict:
    labels = []
    names = []
    scores = []
    loader = make_loader(
        job["test_samples"],
        args,
        job["square_resize"],
    )
    for data, _mask, label, name, _sample_type in tqdm(
        loader,
        desc=(
            f"Aux CNN {job['category']}/{job['scene']}/{job['jsr']}"
        ),
        leave=False,
    ):
        raw = data.permute(0, 3, 1, 2).to(device, non_blocking=True)
        features = cnn_feature_map(
            encoder(raw),
            args.cnn_feature_mode,
        )
        values = image_scores(features, gallery, args.distance_chunk_size)
        labels.extend(int(value) for value in label.numpy().tolist())
        names.extend(str(value) for value in name)
        scores.extend(float(value) for value in values)

    labels_array = np.asarray(labels, dtype=np.int32)
    scores_array = np.asarray(scores, dtype=np.float32)
    paths = [str(sample["path"]) for sample in job["test_samples"]]
    score_key = SCORE_KEYS[args.protocol]
    payload = {
        "names": np.asarray(names),
        "image_paths": np.asarray(paths),
        "labels": labels_array,
        score_key: scores_array,
        "cnn_feature_mode": np.asarray(args.cnn_feature_mode),
    }
    if getattr(args, "support_manifest_sha256", ""):
        payload["support_manifest_sha256"] = np.asarray(
            args.support_manifest_sha256
        )
    if getattr(args, "support_manifest_sha256", ""):
        payload["support_manifest_sha256"] = np.asarray(
            args.support_manifest_sha256
        )

    score_dir = Path(args.output_root) / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{job['dataset']}-{job['category']}-{job['scene']}-{job['jsr']}"
        .replace("/", "_")
    )
    np.savez_compressed(score_dir / f"{stem}-scores.npz", **payload)
    return {
        "method": "resnet18_auxiliary_cnn",
        "feature_mode": args.cnn_feature_mode,
        "dataset": job["dataset"],
        "category": job["category"],
        "scene": job["scene"],
        "jsr": job["jsr"],
        "num_train_normal": len(job["train_samples"]),
        "num_test_normal": int((labels_array == 0).sum()),
        "num_test_abnormal": int((labels_array == 1).sum()),
        "image_auroc": safe_auc(labels, scores),
    }


@torch.no_grad()
def support_reference_scores(encoder, gallery, job, args, device) -> tuple[np.ndarray, np.ndarray]:
    """Score held-out normal-support shifts without reading test samples."""

    scores, names = [], []
    loader = make_loader(job["train_samples"], args, job["square_resize"])
    for data, _mask, _label, name, _sample_type in tqdm(
        loader,
        desc=f"Score support-only CNN reference/{job['scene']}",
        leave=False,
    ):
        for mode in ("time_shift_up", "time_shift_down", "blur"):
            data_variant = cnn_tta_batch(data, mode, args)
            raw = data_variant.permute(0, 3, 1, 2).to(device, non_blocking=True)
            features = cnn_feature_map(
                encoder(raw),
                args.cnn_feature_mode,
            )
            values = image_scores(features, gallery, args.distance_chunk_size)
            scores.extend(float(value) for value in values)
            names.extend(str(value) for value in name)
    return np.asarray(scores, dtype=np.float32), np.asarray(names)


def train_signature(job) -> tuple[str, ...]:
    return tuple(str(sample["path"]) for sample in job["train_samples"])


def run_jobs(jobs, args, device) -> list[dict]:
    encoder = ResNet18LocalEncoder().to(device).eval()
    gallery_args = SimpleNamespace(
        cnn_feature_mode=args.cnn_feature_mode,
        max_gallery_patches=50000,
        cnn_coreset_size=0,
        cnn_paired_tta="none",
        cnn_input_normalization="raw",
        seed=args.seed,
    )
    grouped = {}
    for job in jobs:
        grouped.setdefault(train_signature(job), []).append(job)

    rows = []
    for group_index, group in enumerate(grouped.values(), 1):
        first_job = group[0]
        if not first_job["train_samples"]:
            raise RuntimeError(
                f"No normal support for {first_job['dataset']} "
                f"{first_job['category']}"
            )
        print(
            f"[gallery {group_index}/{len(grouped)}] "
            f"normal_images={len(first_job['train_samples'])} "
            f"test_cells={len(group)}"
        )
        loader = make_loader(
            first_job["train_samples"],
            args,
            first_job["square_resize"],
        )
        gallery = build_cnn_gallery(
            encoder,
            loader,
            gallery_args,
            device,
        )
        if args.support_reference_only:
            reference, names = support_reference_scores(
                encoder,
                gallery,
                first_job,
                args,
                device,
            )
            reference_root = Path(args.output_root) / "support_reference"
            reference_root.mkdir(parents=True, exist_ok=True)
            reference_name = (
                first_job["scene"]
                if args.protocol == "rf_target"
                else first_job["dataset"]
            )
            np.savez_compressed(
                reference_root / f"{reference_name}.npz",
                cnn_scores=reference,
                names=names,
                support_count=np.asarray(len(first_job["train_samples"])),
            )
            continue

        for job in group:
            if not any(
                int(sample["label"]) == 1
                for sample in job["test_samples"]
            ):
                raise RuntimeError(
                    f"No abnormal test sample for {job['category']} "
                    f"{job['scene']} {job['jsr']}"
                )
            rows.append(evaluate_job(encoder, gallery, job, args, device))
    return sorted(
        rows,
        key=lambda row: (
            row["dataset"],
            row["category"],
            row["scene"],
            row["jsr"],
        ),
    )


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        choices=("rf_target", "public_rf", "spectrum"),
        required=True,
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--normal-sampling",
        choices=FEWSHOT_SAMPLING_CHOICES,
        required=True,
    )
    parser.add_argument(
        "--public-rf-signals",
        nargs="+",
        default=list(PUBLIC_RF_JSRS),
        choices=list(PUBLIC_RF_JSRS),
    )
    parser.add_argument(
        "--support-manifest",
        default="",
        help=(
            "Shared support manifest for rf_target or public_rf. For public_rf it "
            "also fixes the test normal paths across support replicates."
        ),
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--distance-chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument(
        "--cnn-feature-mode",
        choices=CNN_FEATURE_MODES,
        default="layer3",
        help=(
            "CNN descriptor. Pair modes align both layers to the baseline "
            "layer3 grid before concatenation."
        ),
    )
    parser.add_argument(
        "--support-seed",
        type=int,
        default=None,
        help=(
            "Optional independent seed for public-RF normal support selection; "
            "self-RF uses the shared target-scene manifest instead."
        ),
    )
    parser.add_argument(
        "--support-reference-only",
        action="store_true",
        help="Build only the support-only calibration reference and skip test scoring.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device(
        "cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda:0"
    )

    if args.protocol == "rf_target":
        jobs = rf_target_jobs(args)
    elif args.protocol == "public_rf":
        jobs = public_rf_jobs(args)
    else:
        jobs = spectrum_jobs(args)
    rows = run_jobs(jobs, args, device)
    if args.support_reference_only:
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "support_reference_protocol.json").write_text(
            json.dumps(
                {
                    "method": "resnet18_auxiliary_cnn",
                    "support_only": True,
                    "reference_views": ["time_shift_up", "time_shift_down", "blur"],
                    "protocol": args.protocol,
                    "support_manifest": getattr(args, "support_manifest", None),
                    "support_manifest_sha256": getattr(args, "support_manifest_sha256", None),
                    "support_seed": getattr(args, "support_seed", None),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote support references under {output_root / 'support_reference'}")
        return
    if not rows:
        raise RuntimeError(f"No evaluation jobs for {args.protocol}")

    output_root = Path(args.output_root)
    result_path = output_root / "results_aux_cnn.csv"
    write_csv(result_path, rows)
    summary = {
        "method": "resnet18_auxiliary_cnn",
        "protocol": args.protocol,
        "normal_sampling": args.normal_sampling,
        "support_seed": getattr(args, "support_seed", None),
        "support_protocol": (
            "target_scene" if args.protocol == "rf_target" else None
        ),
        "support_manifest": getattr(args, "support_manifest", "") or None,
        "support_manifest_sha256": getattr(
            args,
            "support_manifest_sha256",
            None,
        ),
        "support_policy": getattr(args, "support_policy", None),
        "per_frequency_k": getattr(args, "per_frequency_k", None),
        "test_normal_paths_sha256": getattr(
            args,
            "test_normal_paths_sha256",
            None,
        ),
        "encoder": "resnet18_imagenet1k_v1",
        "feature_layer": args.cnn_feature_mode,
        "image_score": "mean_top_10_percent_patch_distance",
        "nearest_neighbours": 1,
        "memory": "complete",
        "num_jobs": len(rows),
        "image_auroc_macro": float(
            np.mean([row["image_auroc"] for row in rows])
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {result_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
