#!/usr/bin/env python
"""Evaluate official PatchCore on RF/spectrum CLS protocols."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCHCORE_SRC = REPO_ROOT / "references" / "patchcore-inspection" / "src"
if str(PATCHCORE_SRC) not in sys.path:
    sys.path.insert(0, str(PATCHCORE_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.sampler

from train_rf_target_pooled_universal import JSR_BY_SIGNAL, SCENES, collect_samples
from utils.training_utils import setup_seed
from utils.rf_frequency_sampling import (
    NORMAL_SAMPLING_CHOICES,
    maybe_select_one_per_frequency_band,
    split_train_test_normals,
)


SPECTRUM_ROOT = REPO_ROOT / "datasets" / "spectrum"
SPECTRUM_CATEGORIES = ("16QAM", "CHIRP", "GMSK", "QPSK")
PUBLIC_RF_ROOT = Path("/mnt/data/wangbei/data/RF_SPE_PNG")
PUBLIC_RF_NORMAL_DIR = PUBLIC_RF_ROOT / "RF_Spectrum_Public_Dataset"
PUBLIC_RF_JSRS = {
    "burst": ("m30db", "m40db", "m50db"),
    "chirp": ("m40db", "m50db", "m55db"),
    "dsss": ("m30db", "m40db", "m50db"),
    "pulse": ("m30db", "m40db", "m50db"),
}


class PatchCorePathDataset(Dataset):
    def __init__(self, samples, resize: int, imagesize: int):
        self.samples = samples
        self.imagesize = (3, imagesize, imagesize)
        self.transform_img = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
            ]
        )
        self.data_to_iterate = [
            (sample["category"], "bad" if int(sample["label"]) else "good", str(sample["path"]), None)
            for sample in samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        image = self.transform_img(image)
        mask = torch.zeros([1, image.shape[1], image.shape[2]])
        return {
            "image": image,
            "mask": mask,
            "classname": sample["category"],
            "anomaly": "bad" if int(sample["label"]) else "good",
            "is_anomaly": int(sample["label"]),
            "image_name": sample["name"],
            "image_path": str(sample["path"]),
        }


def safe_auc(labels, scores):
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores) * 100.0)


def bootstrap_support(items, args):
    """Optionally resample normal support images while leaving the test split fixed."""
    seed = int(getattr(args, "support_bootstrap_seed", -1))
    if seed < 0:
        return list(items)
    if not items:
        return []
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(items), size=len(items))
    return [items[int(index)] for index in indices]


def top_ratio_score(maps, ratio):
    flat = maps.reshape(maps.shape[0], -1)
    k = max(1, int(round(flat.shape[1] * float(ratio))))
    part = np.partition(flat, flat.shape[1] - k, axis=1)[:, -k:]
    return part.mean(axis=1)


def map_stats(segmentations):
    maps = np.asarray(segmentations, dtype=np.float32)
    if maps.ndim == 3:
        pass
    elif maps.ndim == 4 and maps.shape[1] == 1:
        maps = maps[:, 0]
    elif maps.ndim == 4 and maps.shape[-1] == 1:
        maps = maps[..., 0]
    else:
        maps = maps.reshape(maps.shape[0], -1)
    flat = maps.reshape(maps.shape[0], -1)
    max_scores = flat.max(axis=1)
    mean_scores = flat.mean(axis=1)
    std_scores = flat.std(axis=1)
    median_scores = np.median(flat, axis=1)
    mad_scores = np.median(np.abs(flat - median_scores[:, None]), axis=1)
    top001 = top_ratio_score(maps, 0.01)
    top005 = top_ratio_score(maps, 0.05)
    top010 = top_ratio_score(maps, 0.10)
    concentration = (max_scores - mean_scores) / (std_scores + 1e-6)
    top_contrast = (top001 - mean_scores) / (std_scores + 1e-6)
    # The top 1% mean represents a small connected high-score region more
    # robustly than a single interpolated map pixel.  MAD normalizes it using
    # only the current image's background distribution.
    # A small standard-deviation floor keeps a perfectly flat map with one
    # interpolated spike from producing an unbounded confidence value.
    robust_scale = np.maximum(1.4826 * mad_scores, 0.1 * std_scores)
    top1pct_robust_z = (top001 - median_scores) / (robust_scale + 1e-6)
    return {
        "map_max": max_scores.astype(np.float32),
        "map_mean": mean_scores.astype(np.float32),
        "map_std": std_scores.astype(np.float32),
        "map_median": median_scores.astype(np.float32),
        "map_mad": mad_scores.astype(np.float32),
        "map_top0p01": top001.astype(np.float32),
        "map_top0p05": top005.astype(np.float32),
        "map_top0p10": top010.astype(np.float32),
        "map_concentration": concentration.astype(np.float32),
        "map_top_contrast": top_contrast.astype(np.float32),
        "map_top1pct_robust_z": top1pct_robust_z.astype(np.float32),
    }


def make_sample(path, label, category, name=None):
    return {
        "path": Path(path),
        "label": int(label),
        "category": category,
        "name": name or Path(path).name,
    }


def public_normal_paths():
    records = sorted(
        p for p in PUBLIC_RF_NORMAL_DIR.iterdir()
        if p.is_dir() and p.name.startswith("MeasRes_")
    )
    if not records:
        raise FileNotFoundError(f"No public normal records found under {PUBLIC_RF_NORMAL_DIR}")
    paths = []
    for record in records:
        paths.extend(sorted(record.glob("*.png")))
    return paths


def select_public_support_and_test_normals(args):
    normal_paths = public_normal_paths()
    base_support_paths = maybe_select_one_per_frequency_band(normal_paths, args.normal_sampling)
    # The test set must stay fixed across bootstrap replicas.
    support_set = {str(p) for p in base_support_paths}
    test_normal_paths = [p for p in normal_paths if str(p) not in support_set]
    train_normal_paths = bootstrap_support(base_support_paths, args)
    return train_normal_paths, test_normal_paths


def public_rf_jobs(args):
    train_normal_paths, test_normal_paths = select_public_support_and_test_normals(args)
    train_samples = [
        make_sample(p, 0, "public_rf", f"public_train_normal-{p.stem}")
        for p in train_normal_paths
    ]

    jobs = []
    for signal in args.public_rf_signals:
        jsrs = PUBLIC_RF_JSRS[signal]
        for jsr in jsrs:
            normals = test_normal_paths[: args.max_test_normals] if args.max_test_normals > 0 else test_normal_paths
            test_samples = [
                make_sample(p, 0, f"{signal}_{jsr}", f"{signal}-{jsr}-normal-{p.stem}")
                for p in normals
            ]
            abnormal_paths = sorted((PUBLIC_RF_ROOT / signal / "abnormal" / jsr).glob("*.png"))
            if args.max_abnormals > 0:
                abnormal_paths = abnormal_paths[: args.max_abnormals]
            test_samples.extend(
                make_sample(p, 1, f"{signal}_{jsr}", f"{signal}-{jsr}-abnormal-{p.stem}")
                for p in abnormal_paths
            )
            jobs.append(
                {
                    "dataset": "public_rf",
                    "category": signal,
                    "scene": "RF_SPE_PNG_public",
                    "jsr": jsr,
                    "train_samples": train_samples,
                    "test_samples": test_samples,
                }
            )
    return jobs


def spectrum_jobs(args):
    jobs = []
    for category in args.spectrum_categories:
        train_paths = sorted((Path(args.spectrum_root) / category / "train" / "good").glob("*.png"))
        train_paths = maybe_select_one_per_frequency_band(train_paths, args.normal_sampling)
        if args.max_train_normals > 0:
            train_paths = train_paths[: args.max_train_normals]
        good_paths = sorted((Path(args.spectrum_root) / category / "test" / "good").glob("*.png"))
        bad_paths = sorted((Path(args.spectrum_root) / category / "test" / "bad").glob("*.png"))
        if args.max_test_normals > 0:
            good_paths = good_paths[: args.max_test_normals]
        if args.max_abnormals > 0:
            bad_paths = bad_paths[: args.max_abnormals]
        jobs.append(
            {
                "dataset": "spectrum",
                "category": category,
                "scene": "datasets/spectrum",
                "jsr": "none",
                "train_samples": [make_sample(p, 0, category, f"{category}-train-{p.stem}") for p in train_paths],
                "test_samples": (
                    [make_sample(p, 0, category, f"{category}-good-{p.stem}") for p in good_paths]
                    + [make_sample(p, 1, category, f"{category}-bad-{p.stem}") for p in bad_paths]
                ),
            }
        )
    return jobs


def collect_rf_target_cell(signal, scene, jsr, args):
    train_raw = collect_samples(signal, scene, jsr, "train", k_shot=0)
    train_samples = [
        make_sample(sample[0], 0, f"{signal}_{scene}_{jsr}", f"{signal}-{scene}-{jsr}-{sample[3]}-{Path(sample[0]).stem}")
        for sample in train_raw
    ]
    base_train_samples = maybe_select_one_per_frequency_band(
        train_samples,
        args.normal_sampling,
        path_getter=lambda sample: sample["path"],
    )
    train_samples = bootstrap_support(base_train_samples, args)
    exclude_support = getattr(args, "exclude_support_from_test", True)
    support_paths = {str(sample["path"]) for sample in base_train_samples} if exclude_support else set()
    test_raw = collect_samples(signal, scene, jsr, "test", k_shot=1, exclude_paths=support_paths)
    test_samples = [
        make_sample(sample[0], sample[2], f"{signal}_{scene}_{jsr}", f"{signal}-{scene}-{jsr}-{sample[3]}-{Path(sample[0]).stem}")
        for sample in test_raw
    ]
    if args.max_test_normals > 0:
        normals = [s for s in test_samples if s["label"] == 0][: args.max_test_normals]
        abnormals = [s for s in test_samples if s["label"] == 1]
        test_samples = normals + abnormals
    if args.max_abnormals > 0:
        normals = [s for s in test_samples if s["label"] == 0]
        abnormals = [s for s in test_samples if s["label"] == 1][: args.max_abnormals]
        test_samples = normals + abnormals
    return train_samples, test_samples


def collect_rf_target_test_samples(signal, scene, jsr, support_paths, args):
    test_raw = collect_samples(signal, scene, jsr, "test", k_shot=1, exclude_paths=support_paths)
    test_samples = [
        make_sample(sample[0], sample[2], f"{signal}_{scene}_{jsr}", f"{signal}-{scene}-{jsr}-{sample[3]}-{Path(sample[0]).stem}")
        for sample in test_raw
    ]
    if args.max_test_normals > 0:
        normals = [s for s in test_samples if s["label"] == 0][: args.max_test_normals]
        abnormals = [s for s in test_samples if s["label"] == 1]
        test_samples = normals + abnormals
    if args.max_abnormals > 0:
        normals = [s for s in test_samples if s["label"] == 0]
        abnormals = [s for s in test_samples if s["label"] == 1][: args.max_abnormals]
        test_samples = normals + abnormals
    return test_samples


def rf_target_jobs(args):
    jobs = []
    if args.rf_train_mode == "pooled":
        pooled_train_raw = []
        for signal in args.rf_signals:
            for scene in args.rf_scenes:
                for jsr in JSR_BY_SIGNAL[signal]:
                    cell_train_raw = collect_samples(signal, scene, jsr, "train", k_shot=0)
                    if args.normal_sampling == "split_75_25":
                        cell_train_raw, _ = split_train_test_normals(
                            cell_train_raw,
                            train_ratio=0.75,
                            seed=args.seed,
                            path_getter=lambda sample: sample[0],
                        )
                    pooled_train_raw.extend(cell_train_raw)
        if args.normal_sampling == "split_75_25":
            base_selected_train_raw = pooled_train_raw
        else:
            base_selected_train_raw = maybe_select_one_per_frequency_band(
                pooled_train_raw,
                args.normal_sampling,
                path_getter=lambda sample: sample[0],
            )
        if args.max_pooled_train_normals > 0:
            base_selected_train_raw = base_selected_train_raw[: args.max_pooled_train_normals]
        selected_train_raw = bootstrap_support(base_selected_train_raw, args)
        train_samples = [
            make_sample(sample[0], 0, "rf_target_pooled", f"pooled-train-{sample[4]}-{Path(sample[0]).stem}")
            for sample in selected_train_raw
        ]
        support_paths = {str(sample[0]) for sample in base_selected_train_raw}
        for signal in args.rf_signals:
            for scene in args.rf_scenes:
                for jsr in JSR_BY_SIGNAL[signal]:
                    test_samples = collect_rf_target_test_samples(signal, scene, jsr, support_paths, args)
                    jobs.append(
                        {
                            "dataset": "rf_target",
                            "category": signal,
                            "scene": scene,
                            "jsr": jsr,
                            "train_samples": train_samples,
                            "test_samples": test_samples,
                        }
                    )
        return jobs

    for signal in args.rf_signals:
        for scene in args.rf_scenes:
            for jsr in JSR_BY_SIGNAL[signal]:
                train_samples, test_samples = collect_rf_target_cell(signal, scene, jsr, args)
                jobs.append(
                    {
                        "dataset": "rf_target",
                        "category": signal,
                        "scene": scene,
                        "jsr": jsr,
                        "train_samples": train_samples,
                        "test_samples": test_samples,
                    }
                )
    return jobs


def make_patchcore(input_shape, args, device):
    backbone = patchcore.backbones.load(args.backbone)
    backbone.name = args.backbone
    backbone.seed = None
    if args.sampler == "identity":
        sampler = patchcore.sampler.IdentitySampler()
    elif args.sampler == "random":
        sampler = patchcore.sampler.RandomSampler(args.coreset_percentage)
    elif args.sampler == "approx_greedy_coreset":
        sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(args.coreset_percentage, device)
    else:
        sampler = patchcore.sampler.GreedyCoresetSampler(args.coreset_percentage, device)

    nn_method = patchcore.common.FaissNN(False, args.faiss_num_workers)
    model = patchcore.patchcore.PatchCore(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=args.layers,
        device=device,
        input_shape=input_shape,
        pretrain_embed_dimension=args.pretrain_embed_dimension,
        target_embed_dimension=args.target_embed_dimension,
        patchsize=args.patchsize,
        anomaly_scorer_num_nn=args.anomaly_scorer_num_nn,
        featuresampler=sampler,
        nn_method=nn_method,
    )
    return model


def fit_patchcore(train_samples, args, device):
    train_dataset = PatchCorePathDataset(train_samples, args.resize, args.imagesize)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model = make_patchcore(train_dataset.imagesize, args, device)
    model.fit(train_loader)
    return model


def predict_job(model, job, args):
    test_dataset = PatchCorePathDataset(job["test_samples"], args.resize, args.imagesize)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    scores, segmentations, labels, _ = model.predict(test_loader)
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    stats = map_stats(segmentations) if args.save_map_stats else {}

    score_dir = Path(args.output_root) / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{job['dataset']}-{job['category']}-{job['scene']}-{job['jsr']}".replace("/", "_")
    np.savez_compressed(
        score_dir / f"{stem}-scores.npz",
        scores=scores,
        labels=labels,
        image_paths=np.asarray([str(s["path"]) for s in job["test_samples"]]),
        **stats,
    )
    if args.save_map_match:
        map_dir = Path(args.output_root) / "maps"
        map_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(job["test_samples"]):
            path_text = str(sample["path"])
            if any(token in path_text for token in args.save_map_match):
                np.savez_compressed(
                    map_dir / f"{stem}-index{index}.npz",
                    image_path=np.asarray(path_text),
                    label=np.asarray(int(sample["label"])),
                    anomaly_map=np.asarray(segmentations[index], dtype=np.float32),
                )
    return {
        "method": "patchcore_official",
        "dataset": job["dataset"],
        "category": job["category"],
        "scene": job["scene"],
        "jsr": job["jsr"],
        "num_train_normal": len(job["train_samples"]),
        "num_test_normal": int((labels == 0).sum()),
        "num_test_abnormal": int((labels == 1).sum()),
        "image_auroc": safe_auc(labels.tolist(), scores.tolist()),
    }


def run_job(job, args, device):
    if not job["train_samples"]:
        raise RuntimeError(f"No train samples for {job['dataset']} {job['category']} {job['scene']} {job['jsr']}")
    if not any(s["label"] == 1 for s in job["test_samples"]):
        raise RuntimeError(f"No abnormal samples for {job['dataset']} {job['category']} {job['scene']} {job['jsr']}")

    return predict_job(fit_patchcore(job["train_samples"], args, device), job, args)


def train_signature(job):
    return tuple(str(sample["path"]) for sample in job["train_samples"])


def run_jobs(jobs, args, device):
    rows = []
    groups = {}
    for idx, job in enumerate(jobs):
        groups.setdefault(train_signature(job), []).append((idx, job))

    for group_idx, grouped in enumerate(groups.values(), 1):
        first_job = grouped[0][1]
        print(
            f"[fit {group_idx}/{len(groups)}] "
            f"train_normals={len(first_job['train_samples'])} "
            f"eval_jobs={len(grouped)}"
        )
        if not first_job["train_samples"]:
            raise RuntimeError("No train samples for grouped PatchCore run")
        model = fit_patchcore(first_job["train_samples"], args, device)
        for idx, job in grouped:
            if not any(s["label"] == 1 for s in job["test_samples"]):
                raise RuntimeError(f"No abnormal samples for {job['dataset']} {job['category']} {job['scene']} {job['jsr']}")
            print(f"[{idx + 1}/{len(jobs)}] {job['dataset']} {job['category']} {job['scene']} {job['jsr']}")
            row = predict_job(model, job, args)
            print(f"  image_auroc={row['image_auroc']:.4f}")
            rows.append(row)
    rows.sort(key=lambda r: (r["dataset"], r["category"], r["scene"], r["jsr"]))
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_average(rows):
    out = list(rows)
    avg = {k: "" for k in rows[0].keys()}
    avg.update(
        {
            "method": "patchcore_official",
            "dataset": rows[0]["dataset"],
            "category": "average",
            "scene": "macro",
            "jsr": "macro",
            "num_train_normal": "",
            "num_test_normal": "",
            "num_test_abnormal": "",
            "image_auroc": float(np.mean([r["image_auroc"] for r in rows])),
        }
    )
    out.append(avg)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["spectrum", "public_rf", "rf_target"], required=True)
    parser.add_argument("--output-root", default="analysis_outputs/20260702_patchcore_cls")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--backbone", default="wideresnet50")
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--pretrain-embed-dimension", type=int, default=1024)
    parser.add_argument("--target-embed-dimension", type=int, default=1024)
    parser.add_argument("--patchsize", type=int, default=3)
    parser.add_argument("--anomaly-scorer-num-nn", type=int, default=1)
    parser.add_argument("--sampler", choices=["identity", "random", "greedy_coreset", "approx_greedy_coreset"], default="random")
    parser.add_argument("--coreset-percentage", type=float, default=0.1)
    parser.add_argument("--faiss-num-workers", type=int, default=8)
    parser.add_argument("--normal-sampling", choices=NORMAL_SAMPLING_CHOICES, default="per_frequency")
    parser.add_argument("--max-pooled-train-normals", type=int, default=0)
    parser.add_argument("--max-train-normals", type=int, default=0)
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--spectrum-root", default=str(SPECTRUM_ROOT))
    parser.add_argument("--spectrum-categories", nargs="+", default=list(SPECTRUM_CATEGORIES), choices=list(SPECTRUM_CATEGORIES))
    parser.add_argument("--rf-signals", nargs="+", default=list(JSR_BY_SIGNAL.keys()), choices=list(JSR_BY_SIGNAL.keys()))
    parser.add_argument("--public-rf-signals", nargs="+", default=list(PUBLIC_RF_JSRS.keys()), choices=list(PUBLIC_RF_JSRS.keys()))
    parser.add_argument("--rf-scenes", nargs="+", default=SCENES, choices=SCENES)
    parser.add_argument("--rf-train-mode", choices=["pooled", "per_cell"], default="pooled")
    parser.add_argument("--support-bootstrap-seed", type=int, default=-1)
    parser.add_argument("--save-map-stats", action="store_true")
    parser.add_argument("--save-map-match", nargs="*", default=[])
    parser.add_argument(
        "--map-only-cell",
        nargs="*",
        default=[],
        help="Keep only cells encoded as signal:scene:jsr after building the shared gallery.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda:0")

    if args.protocol == "spectrum":
        jobs = spectrum_jobs(args)
    elif args.protocol == "public_rf":
        jobs = public_rf_jobs(args)
    else:
        jobs = rf_target_jobs(args)

    if args.map_only_cell:
        wanted = set(args.map_only_cell)
        jobs = [
            job for job in jobs
            if f"{job['category']}:{job['scene']}:{job['jsr']}" in wanted
        ]
        if not jobs:
            raise RuntimeError(f"No requested map cells found: {sorted(wanted)}")

    rows = run_jobs(jobs, args, device)

    rows_with_avg = append_average(rows)
    out_root = Path(args.output_root)
    result_path = out_root / "results_patchcore_cls.csv"
    write_csv(result_path, rows_with_avg)
    summary = {
        "method": "patchcore_official",
        "protocol": args.protocol,
        "rf_train_mode": args.rf_train_mode if args.protocol == "rf_target" else None,
        "normal_sampling": args.normal_sampling,
        "num_train_normal": int(rows[0]["num_train_normal"]) if rows else 0,
        "backbone": args.backbone,
        "layers": args.layers,
        "sampler": args.sampler,
        "coreset_percentage": args.coreset_percentage,
        "resize": args.resize,
        "imagesize": args.imagesize,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "num_jobs": len(rows),
        "image_auroc_macro": float(np.mean([r["image_auroc"] for r in rows])),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_root / "README.md").write_text(
        "# Official PatchCore CLS Evaluation\n\n"
        "Source: `references/patchcore-inspection` cloned from amazon-research/patchcore-inspection.\n\n"
        f"Protocol: `{args.protocol}`\n\n"
        f"Result CSV: `{result_path.name}`\n\n"
        f"Macro Image-AUROC: `{summary['image_auroc_macro']:.4f}`\n",
        encoding="utf-8",
    )
    print(f"wrote {result_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
