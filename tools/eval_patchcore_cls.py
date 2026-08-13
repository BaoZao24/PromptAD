#!/usr/bin/env python
"""Evaluate official PatchCore on RF/spectrum CLS protocols."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from datasets.rf_target import (
    RF_JSR_BY_SIGNAL as JSR_BY_SIGNAL,
    RF_SCENES as SCENES,
    collect_rf_target_samples as collect_samples,
)
from tools.ofdma_fewshot_baseline_common import add_ofdma_args, load_sample_image, ofdma_jobs
from utils.training_utils import setup_seed
from utils.rf_frequency_sampling import (
    NORMAL_SAMPLING_CHOICES,
    maybe_select_one_per_frequency_band,
)
from utils.rf_scene_support import (
    cell_entry,
    ensure_rf_target_scene_manifest,
    scene_support_entry,
)
from utils.public_rf_support import select_support_and_test_paths


SPECTRUM_ROOT = REPO_ROOT / "datasets" / "spectrum"
SPECTRUM_CATEGORIES = ("16QAM", "CHIRP", "GMSK", "QPSK")
PUBLIC_RF_ROOT = Path("/mnt/data/wangbei/data/RF_SPE_PNG")
PUBLIC_RF_NORMAL_DIR = PUBLIC_RF_ROOT / "RF_Spectrum_Public_Dataset"
PUBLIC_RF_JSRS = {
    "burst": ("m30db", "m40db", "m50db"),
    "chirp": ("m40db", "m50db", "m55db"),
    "dsss": ("m30db", "m40db", "m50db"),
    "pulse": ("m30db", "m40db", "m50db"),
    "deceptive": ("m10db", "m20db", "m30db"),
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
        image = load_sample_image(sample)
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
    base_support_paths, test_normal_paths, manifest = select_support_and_test_paths(
        normal_paths,
        normal_sampling=args.normal_sampling,
        seed=getattr(args, "seed", None),
        manifest_path=getattr(args, "support_manifest", None) or None,
    )
    # The test set must stay fixed across bootstrap replicas.  If a manifest is
    # supplied, its support/test split is authoritative for every visual
    # baseline that imports this shared job builder.
    train_normal_paths = bootstrap_support(base_support_paths, args)
    args.support_manifest_sha256 = (
        manifest.get("support_paths_sha256")
        if manifest is not None
        else hashlib.sha256(
            "\n".join(str(path) for path in base_support_paths).encode("utf-8")
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
    return train_normal_paths, test_normal_paths


def public_rf_jobs(args):
    train_normal_paths, test_normal_paths = select_public_support_and_test_normals(args)
    train_samples = [
        make_sample(p, 0, "public_rf", f"public_train_normal-{p.stem}")
        for p in train_normal_paths
    ]

    jobs = []
    signals = getattr(args, "public_rf_signals", None)
    if signals is None:
        signals = getattr(args, "rf_signals", tuple(PUBLIC_RF_JSRS))
    for signal in signals:
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
                    "support_policy": getattr(args, "support_policy", None),
                    "per_frequency_k": getattr(args, "per_frequency_k", None),
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


def collect_rf_target_scene_test_samples(signal, scene, jsr, manifest, args):
    manifest_cell = cell_entry(manifest, signal, scene, jsr)
    allowed_normal_paths = {
        item["path"] for item in manifest_cell["test_normals"]
    }
    allowed_abnormal_paths = {
        item["path"] for item in manifest_cell["test_abnormals"]
    }
    test_raw = collect_samples(signal, scene, jsr, "test", k_shot=1)
    test_samples = [
        make_sample(
            sample[0],
            sample[2],
            f"{signal}_{scene}_{jsr}",
            f"{signal}-{scene}-{jsr}-{sample[3]}-{Path(sample[0]).stem}",
        )
        for sample in test_raw
        if (
            (int(sample[2]) == 0 and str(sample[0]) in allowed_normal_paths)
            or (
                int(sample[2]) == 1
                and str(sample[0]) in allowed_abnormal_paths
            )
        )
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


def prepare_rf_target_scene_manifest(args):
    manifest_path = (
        Path(args.support_manifest)
        if args.support_manifest
        else Path(args.output_root) / "support_manifest.json"
    )
    manifest = ensure_rf_target_scene_manifest(
        manifest_path,
        collect_samples=collect_samples,
        signals=args.rf_signals,
        scenes=args.rf_scenes,
        jsr_by_signal=JSR_BY_SIGNAL,
        normal_sampling=args.normal_sampling,
        seed=args.seed,
    )
    args.support_manifest = str(manifest_path)
    args.support_manifest_sha256 = manifest["manifest_sha256"]
    return manifest


def rf_target_jobs(args):
    jobs = []
    manifest = prepare_rf_target_scene_manifest(args)
    for scene in args.rf_scenes:
        support = scene_support_entry(manifest, scene)["support"]
        train_samples = [
            make_sample(
                item["path"],
                0,
                f"rf_target_scene_{scene}",
                f"scene-support-{scene}-{Path(item['path']).stem}",
            )
            for item in support
        ]
        for signal in args.rf_signals:
            for jsr in JSR_BY_SIGNAL[signal]:
                test_samples = collect_rf_target_scene_test_samples(
                    signal,
                    scene,
                    jsr,
                    manifest,
                    args,
                )
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
    scores, _segmentations, labels, _ = model.predict(test_loader)
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    score_dir = Path(args.output_root) / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{job['dataset']}-{job['category']}-{job['scene']}-{job['jsr']}".replace("/", "_")
    score_payload = {
        "scores": scores,
        "labels": labels,
        "image_paths": np.asarray([str(s["path"]) for s in job["test_samples"]]),
    }
    if getattr(args, "support_manifest_sha256", ""):
        score_payload["support_manifest_sha256"] = np.asarray(
            args.support_manifest_sha256
        )
    np.savez_compressed(score_dir / f"{stem}-scores.npz", **score_payload)
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

    model = fit_patchcore(job["train_samples"], args, device)
    return predict_job(model, job, args)


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
    parser.add_argument(
        "--protocol",
        choices=["spectrum", "public_rf", "rf_target", "ofdma"],
        required=True,
    )
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
    parser.add_argument(
        "--support-manifest",
        default="",
        help="Shared target-scene support manifest for the rf_target protocol.",
    )
    parser.add_argument("--max-train-normals", type=int, default=0)
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    parser.add_argument("--spectrum-root", default=str(SPECTRUM_ROOT))
    parser.add_argument("--spectrum-categories", nargs="+", default=list(SPECTRUM_CATEGORIES), choices=list(SPECTRUM_CATEGORIES))
    parser.add_argument("--rf-signals", nargs="+", default=list(JSR_BY_SIGNAL.keys()), choices=list(JSR_BY_SIGNAL.keys()))
    parser.add_argument("--public-rf-signals", nargs="+", default=list(PUBLIC_RF_JSRS.keys()), choices=list(PUBLIC_RF_JSRS.keys()))
    parser.add_argument("--rf-scenes", nargs="+", default=SCENES, choices=SCENES)
    parser.add_argument("--support-bootstrap-seed", type=int, default=-1)
    parser.add_argument("--save-map-match", nargs="*", default=[])
    parser.add_argument(
        "--map-only-cell",
        nargs="*",
        default=[],
        help="Keep only cells encoded as signal:scene:jsr after building the shared gallery.",
    )
    add_ofdma_args(parser)
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
    elif args.protocol == "ofdma":
        jobs = ofdma_jobs(args)
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
        "support_protocol": "target_scene" if args.protocol == "rf_target" else None,
        "support_manifest": getattr(args, "support_manifest", "") or None,
        "support_manifest_sha256": getattr(args, "support_manifest_sha256", None),
        "test_normal_paths_sha256": getattr(args, "test_normal_paths_sha256", None),
        "support_policy": getattr(args, "support_policy", None),
        "per_frequency_k": getattr(args, "per_frequency_k", None),
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
