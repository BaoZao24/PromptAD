#!/usr/bin/env python
"""Evaluate external visual baselines on the FedJam few-shot image protocol.

This adapter keeps the existing baseline implementations unchanged for their
original protocols.  It only supplies FedJam's embedded Arrow images through
the same ``sample`` interface used by the RF/OFDMA evaluators.

Protocol
--------
* support: deterministic, nested 1/2/4-shot benign train images;
* training: normal support only;
* test: the independent FedJam test split, with labels used only after
  scoring;
* score: image-level AUROC/AUPRC/FPR@95TPR, plus per-attack diagnostics.

The runner is intentionally one method per process by default.  This keeps
large frozen backbones and decoded Arrow images from accumulating across
methods and makes failures easy to isolate.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_fedjam_fewshot_dual import (  # noqa: E402
    LABEL_NAMES,
    iter_test_records,
    select_benign_support,
)
from utils.training_utils import setup_seed  # noqa: E402


METHOD_NAMES = {
    "vae": "vae_reconstruction",
    "iad_per": "iad_per",
    "udma": "udma_reimplementation",
    "saife": "saife_reconstruction",
    "deep_svdd": "deep_svdd",
    "padim": "padim_diag_resnet18",
    "stfpm": "stfpm_resnet18",
    "winclip": "winclip_fewshot",
    "patchcore": "patchcore_official",
}
ALL_METHODS = tuple(METHOD_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedJam few-shot external visual baselines"
    )
    parser.add_argument("--method", choices=("all", *ALL_METHODS), required=True)
    parser.add_argument(
        "--data-root", default="/mnt/data/wangbei/data/FedJam"
    )
    parser.add_argument(
        "--output-root",
        default="analysis_outputs/20260810_fedjam_visual_baselines_formal",
    )
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gallery-batch-size", type=int, default=8)
    parser.add_argument("--max-test-per-label", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stfpm-epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--rep-dim", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=240)
    parser.add_argument("--img-resize", type=int, default=240)
    parser.add_argument("--img-cropsize", type=int, default=240)
    parser.add_argument("--faiss-num-workers", type=int, default=2)
    return parser.parse_args()


def fedjam_image(sample: dict) -> Image.Image:
    image_bgr = np.asarray(sample["image_bgr"], dtype=np.uint8)
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Unexpected FedJam image shape: {image_bgr.shape}")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")


def make_sample(record, split: str) -> dict:
    return {
        "path": f"fedjam/{split}/{record.name}",
        "label": int(record.label),
        "category": "fedjam",
        "name": str(record.name),
        "dataset": "fedjam",
        "image_bgr": np.ascontiguousarray(record.image_bgr),
    }


def install_fedjam_loaders() -> dict[str, object]:
    """Patch only imported loader symbols in the selected baseline modules."""

    module_names = {
        "vae": "tools.eval_vae_cls",
        "iad_per": "tools.eval_vae_cls",
        "udma": "tools.eval_udma_cls",
        "saife": "tools.eval_saife_ofdma_fewshot",
        "deep_svdd": "tools.eval_deepsvdd_cls",
        "padim": "tools.eval_padim_cls",
        "stfpm": "tools.eval_stfpm_cls",
        "winclip": "tools.eval_winclip_main",
        "patchcore": "tools.eval_patchcore_cls",
    }
    modules = {
        method: importlib.import_module(module_name)
        for method, module_name in module_names.items()
    }
    for module in modules.values():
        if hasattr(module, "load_sample_image"):
            module.load_sample_image = fedjam_image
        # The original RF/OFDMA helpers receive binary labels.  FedJam has
        # benign=0 plus three attack labels; the adapter's final metrics keep
        # those labels, while this legacy per-job diagnostic uses the required
        # benign-vs-any-attack reduction.
        if hasattr(module, "safe_auc"):
            module.safe_auc = lambda labels, scores: metric(
                (np.asarray(labels) != 0).astype(np.int32), scores
            )["auroc"]
    return modules


class FedJamSAIFEDataset(Dataset):
    """Small native-shape adapter for the existing SAIFE implementation."""

    def __init__(self, samples, _dataset_root, min_db=None, max_db=None):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        gray = cv2.cvtColor(sample["image_bgr"], cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (70, 110), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0)
        return (
            tensor,
            int(sample["label"]),
            str(sample["path"]),
            str(sample["name"]),
            "fedjam",
        )


def install_saife_loader(module) -> None:
    module.SAIFENativeDataset = FedJamSAIFEDataset


def metric(labels, scores) -> dict[str, float]:
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


def add_metric_rows(rows: list[dict], shot: int, method: str, labels, scores) -> None:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)
    scopes = [("overall", labels != 0)]
    for label_id, label_name in LABEL_NAMES.items():
        if label_id != 0:
            scopes.append((label_name, labels == label_id))

    per_attack = []
    for scope, attack_mask in scopes:
        if scope == "overall":
            mask = np.ones(labels.shape[0], dtype=bool)
            binary = (labels != 0).astype(np.int32)
        else:
            mask = (labels == 0) | attack_mask
            binary = attack_mask[mask].astype(np.int32)
        values = metric(binary, scores[mask])
        rows.append(
            {
                "shot": int(shot),
                "method": method,
                "scope": scope,
                "n": int(mask.sum()),
                "num_normal": int((labels[mask] == 0).sum()),
                "num_abnormal": int(binary.sum()),
                **values,
            }
        )
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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def common_cfg(args, output_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        protocol="ofdma",
        output_root=str(output_root),
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        test_batch_size=max(args.batch_size, 64),
        gallery_batch_size=args.gallery_batch_size,
        resolution=args.resolution,
        img_resize=args.img_resize,
        img_cropsize=args.img_cropsize,
        epochs=args.epochs,
        stfpm_epochs=args.stfpm_epochs,
        log_every=max(1, args.epochs // 2),
        image_size=args.image_size,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        hidden_dim=256,
        dropout=0.1,
        fc_dim=1024,
        decoder_fc_dim=512,
        rep_dim=args.rep_dim,
        lr=1e-3,
        discriminator_lr=2.5e-5,
        adversarial_weight=1.0,
        weight_decay=1e-6,
        score_mode="mse_mean",
        per_alpha=0.05,
        per_gamma=3,
        per_background_weight=2.0,
        per_background_percentile=90.0,
        per_signal_percentile=99.0,
        center_eps=0.1,
        resize=256,
        imagesize=224,
        weights="default",
        embedding_dim=128,
        var_eps=0.01,
        momentum=0.9,
        stfpm_lr=0.05,
        faiss_num_workers=args.faiss_num_workers,
        backbone="wideresnet50",
        layers=["layer2", "layer3"],
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        anomaly_scorer_num_nn=1,
        sampler="random",
        coreset_percentage=0.1,
        save_map_match=[],
        scales=[2, 3],
        pretrained_dataset="laion400m_e32",
        prompt_class="radio frequency spectrogram",
        ofdma_root="/mnt/data/wangbei/data/FedJam",
        ofdma_min_db=None,
        ofdma_max_db=None,
        image_height=args.image_size,
        image_width=args.image_size,
        reference_extractor="resnet18_imagenet",
        teacher_epochs=5,
        student_epochs=10,
        teacher_lr=1e-3,
        student_lr=1e-3,
        weight_teacher_ae=0.5,
        weight_teacher_memae=0.5,
        weight_ae_memae=0.5,
        compactness_weight=0.1,
        separateness_weight=0.1,
        memory_update_rate=0.1,
        separateness_margin=1.0,
    )


def load_payload(output_root: Path, job: dict) -> dict[str, np.ndarray]:
    stem = f"{job['dataset']}-{job['category']}-{job['scene']}-{job['jsr']}".replace(
        "/", "_"
    )
    path = output_root / "scores" / f"{stem}-scores.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as loaded:
        payload = {key: loaded[key] for key in loaded.files}
    if not np.array_equal(
        np.asarray(payload["labels"], dtype=np.int32),
        np.asarray([sample["label"] for sample in job["test_samples"]], dtype=np.int32),
    ):
        raise RuntimeError(f"Prediction label order mismatch: {path}")
    return {
        "labels": np.asarray(payload["labels"], dtype=np.int32),
        "scores": np.asarray(payload["scores"], dtype=np.float32),
        "names": np.asarray(
            payload.get(
                "names",
                [sample["name"] for sample in job["test_samples"]],
            )
        ),
    }


def evaluate_one(method: str, jobs: list[dict], args, modules, device) -> list[dict]:
    output_root = Path(args.output_root) / method
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "scores").mkdir(parents=True, exist_ok=True)
    cfg = common_cfg(args, output_root)
    rows: list[dict] = []
    for job in jobs:
        shot = int(job["shot"])
        print(
            f"[{method}] fit shot={shot} support={len(job['train_samples'])} "
            f"test={len(job['test_samples'])}",
            flush=True,
        )
        setup_seed(args.seed)
        module = modules[method]
        context = None
        if method in {"vae", "iad_per"}:
            if method == "iad_per":
                cfg.score_mode = "iad_per"
            model, history = module.fit_vae(job["train_samples"], cfg, device)
            module.predict_job(model, job, cfg, device)
            payload = load_payload(output_root, job)
            cfg.train_loss_final = history[-1] if history else None
            del model
        elif method == "deep_svdd":
            model, center, history = module.fit_deepsvdd(
                job["train_samples"], cfg, device
            )
            module.predict_job(model, center, job, cfg, device)
            payload = load_payload(output_root, job)
            cfg.train_loss_final = history[-1] if history else None
            del model, center
        elif method == "udma":
            model = module.fit_udma(job["train_samples"], cfg, device)
            module.predict_job(model, job, cfg, device)
            payload = load_payload(output_root, job)
            cfg.teacher_loss_final = model.training_history["teacher_loss"][-1]
            cfg.student_loss_final = model.training_history["student_loss"][-1]
            del model
        elif method == "saife":
            model, history = module.fit_saife(job["train_samples"], cfg, device)
            raw_payload = module.predict(model, job["test_samples"], cfg, device)
            payload = {
                "labels": np.asarray(raw_payload["labels"], dtype=np.int32),
                "scores": np.asarray(raw_payload["scores"], dtype=np.float32),
                "names": np.asarray(raw_payload["names"]),
            }
            cfg.train_loss_final = (
                history[-1]["reconstruction_loss"] if history else None
            )
            del model
        elif method == "padim":
            context = module.ResNetFeatureExtractor(cfg.weights).to(device).eval()
            stats = module.fit_padim(context, job["train_samples"], cfg, device)
            module.predict_job(context, stats, job, cfg, device)
            payload = load_payload(output_root, job)
            del context, stats
        elif method == "stfpm":
            cfg.epochs = args.stfpm_epochs
            cfg.log_every = max(1, args.stfpm_epochs // 2)
            cfg.lr = cfg.stfpm_lr
            teacher, student, history = module.fit_stfpm(
                job["train_samples"], cfg, device
            )
            module.predict_job(teacher, student, job, cfg, device)
            payload = load_payload(output_root, job)
            cfg.train_loss_final = history[-1] if history else None
            del teacher, student
        elif method == "winclip":
            cfg.batch_size = max(args.batch_size, 16)
            cfg.gallery_batch_size = args.gallery_batch_size
            cfg.backbone = "ViT-B-16-plus-240"
            context = module.make_model(cfg, str(device))
            module.build_visual_gallery(
                context, job["train_samples"], cfg, str(device)
            )
            module.predict_job(context, job, cfg, str(device))
            payload = load_payload(output_root, job)
            del context
        elif method == "patchcore":
            cfg.batch_size = args.batch_size
            model = module.fit_patchcore(job["train_samples"], cfg, device)
            module.predict_job(model, job, cfg)
            payload = load_payload(output_root, job)
            del model
        else:
            raise ValueError(method)

        if not np.all(np.isfinite(payload["scores"])):
            raise RuntimeError(f"Non-finite scores from {method} shot={shot}")
        np.savez_compressed(
            output_root / "scores" / f"fedjam_{shot}shot_scores.npz",
            labels=payload["labels"],
            scores=payload["scores"],
            names=np.asarray([sample["name"] for sample in job["test_samples"]]),
        )
        add_metric_rows(rows, shot, METHOD_NAMES[method], payload["labels"], payload["scores"])
        overall = metric(
            (payload["labels"] != 0).astype(np.int32), payload["scores"]
        )
        print(
            f"[{method}] shot={shot} AUROC={overall['auroc']:.4f} "
            f"AUPRC={overall['auprc']:.4f} FPR95={overall['fpr95']:.4f}",
            flush=True,
        )
        write_csv(output_root / "metrics.csv", rows)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return rows


def write_metadata(output_root: Path, method: str, args, support, train_counts, test_counts):
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "entry": "tools/eval_fedjam_visual_baselines.py",
        "method": method,
        "method_name": METHOD_NAMES[method],
        "data_root": str(Path(args.data_root).resolve()),
        "support": "benign label=0 only, deterministic nested seed-111 reservoir",
        "shots": sorted(set(int(value) for value in args.shots)),
        "test": "full independent FedJam test split unless --max-test-per-label is set",
        "test_label_usage": "metrics_only",
        "image_mode": "embedded spectrogram image only; KPI not used",
        "train_counts": train_counts,
        "test_counts": test_counts,
        "support_names": [record.name for record in support],
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_test_per_label": args.max_test_per_label,
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "README.md").write_text(
        "# FedJam external visual baseline\n\n"
        f"Method: `{METHOD_NAMES[method]}`\n\n"
        "Protocol: benign-only 1/2/4-shot support, independent test split, "
        "embedded spectrogram images only.\n\n"
        "The adapter reuses the existing baseline implementation and replaces "
        "only its image loader for FedJam Arrow records. Test labels are used "
        "only for final metrics.\n\n"
        "See `protocol.json`, `metrics.csv`, and `scores/`.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    shots = sorted(set(int(value) for value in args.shots))
    if shots != [value for value in shots if value in {1, 2, 4}]:
        raise ValueError("--shots must be selected from 1, 2, and 4")
    if args.max_test_per_label < 0:
        raise ValueError("--max-test-per-label cannot be negative")
    if not args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    setup_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device(
        "cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda:0"
    )
    print(f"[device] {device}", flush=True)

    data_root = Path(args.data_root).resolve()
    support, train_counts, _benign_seen = select_benign_support(
        data_root, max(shots), args.seed
    )
    test_records = list(iter_test_records(data_root, args.max_test_per_label))
    test_counts = {label: 0 for label in LABEL_NAMES}
    for record in test_records:
        test_counts[record.label] = test_counts.get(record.label, 0) + 1
    if not any(record.label == 0 for record in test_records) or not any(
        record.label != 0 for record in test_records
    ):
        raise RuntimeError("FedJam smoke/formal test needs benign and abnormal rows")
    jobs = [
        {
            "dataset": "fedjam",
            "category": "fedjam",
            "scene": "fewshot",
            "jsr": f"{shot}shot",
            "shot": shot,
            "train_samples": [make_sample(record, "train") for record in support[:shot]],
            "test_samples": [make_sample(record, "test") for record in test_records],
        }
        for shot in shots
    ]
    print(
        f"[data] support={len(support)} test={len(test_records)} "
        f"test_counts={test_counts}",
        flush=True,
    )

    methods = ALL_METHODS if args.method == "all" else (args.method,)
    modules = install_fedjam_loaders()
    install_saife_loader(modules["saife"])
    for method in methods:
        method_root = Path(args.output_root) / method
        write_metadata(method_root, method, args, support, train_counts, test_counts)
        evaluate_one(method, jobs, args, modules, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    print(f"[done] root={Path(args.output_root).resolve()}", flush=True)


if __name__ == "__main__":
    main()
