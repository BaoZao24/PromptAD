#!/usr/bin/env python
"""Unified module ablation for the current visual normality method.

Runs only missing ViT-branch variants and reuses current CNN score files from
the confidence-gate sampling ablation.
"""

from __future__ import annotations

import csv
import json
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Queue


GPUS = [0, 2, 3]
ROOT = Path("analysis_outputs/20260725_confidence_gate_module_ablation")
BASE = Path("analysis_outputs/20260725_confidence_gate_sampling_ablation")
PUB_CKPT = (
    "analysis_outputs/90_rejected_or_aborted/20260706_cleanup_old_results/"
    "20260627_method_funnel/runs/pooled_rf_rgb/cls/checkpoint/overall-best.pt"
)

DATASETS = {
    "self_rf": {"sampling": "per_frequency"},
    "public_rf": {"sampling": "per_frequency"},
    "spectrum": {"sampling": "4shot"},
}

VIT_VARIANTS = {
    "no_all": {
        # Plain ViT normal gallery: no paired views, no coreset, and top-1 NN.
        "paired_tta": "none",
        "paired_tta_fusion": "max",
        "coreset_method": "random",
        "coreset_ratio": "1.0",
        "nn_topk": "1",
    },
    "no_tta": {
        "paired_tta": "none",
        "paired_tta_fusion": "max",
        "coreset_method": "farthest",
        "coreset_ratio": "0.5",
        "nn_topk": "5",
    },
    "no_topk": {
        "paired_tta": "stft_shift_blur",
        "paired_tta_fusion": "max",
        "coreset_method": "farthest",
        "coreset_ratio": "0.5",
        "nn_topk": "1",
    },
    "no_coreset": {
        "paired_tta": "stft_shift_blur",
        "paired_tta_fusion": "max",
        "coreset_method": "random",
        "coreset_ratio": "1.0",
        "nn_topk": "5",
    },
}


def out_base(variant: str, dataset: str) -> Path:
    sampling = DATASETS[dataset]["sampling"]
    return ROOT / variant / sampling / dataset


def existing_vit_dir(dataset: str) -> Path:
    if dataset == "self_rf":
        return BASE / "per_frequency" / "self_vit"
    if dataset == "public_rf":
        return BASE / "per_frequency" / "public_vit"
    if dataset == "spectrum":
        return BASE / "4shot" / "spectrum_vit"
    raise KeyError(dataset)


def existing_cnn_dir(dataset: str) -> Path:
    if dataset == "self_rf":
        return BASE / "per_frequency" / "self_aux_cnn"
    if dataset == "public_rf":
        return BASE / "per_frequency" / "public_aux_cnn"
    if dataset == "spectrum":
        return BASE / "4shot" / "spectrum_aux_cnn"
    raise KeyError(dataset)


def existing_fusion_dir(dataset: str) -> Path:
    if dataset == "self_rf":
        return BASE / "per_frequency" / "self_fusion"
    if dataset == "public_rf":
        return BASE / "per_frequency" / "public_fusion"
    if dataset == "spectrum":
        return BASE / "4shot" / "spectrum_fusion"
    raise KeyError(dataset)


def vit_cmd(variant: str, dataset: str) -> list[str]:
    cfg = VIT_VARIANTS[variant]
    base = out_base(variant, dataset) / "vit"
    common = [
        "--output-root", str(base),
        "--normal-sampling", DATASETS[dataset]["sampling"],
        "--coreset-method", cfg["coreset_method"],
        "--coreset-ratio", cfg["coreset_ratio"],
        "--nn-topk", cfg["nn_topk"],
        "--nn-agg", "mean",
        "--paired-tta", cfg["paired_tta"],
        "--paired-tta-fusion", cfg["paired_tta_fusion"],
        "--gpu-id", "%GPU%",
    ]
    if dataset == "self_rf":
        return [
            "python", "tools/eval_cls_vit_patchcore_gallery.py",
            *common,
            "--support-manifest",
            str(BASE / "per_frequency" / "support_manifest.json"),
        ]
    if dataset == "public_rf":
        return [
            "python", "tools/eval_cls_public_rf_vit_patchcore_gallery.py",
            *common,
            "--checkpoint", PUB_CKPT,
        ]
    if dataset == "spectrum":
        return ["python", "tools/eval_cls_spectrum_vit_nn_gallery.py", *common]
    raise KeyError(dataset)


def fusion_cmd(variant: str, dataset: str) -> list[str]:
    base = out_base(variant, dataset)
    protocol = {"self_rf": "rf_target", "public_rf": "public_rf", "spectrum": "spectrum"}[dataset]
    cmd = [
        "python", "tools/eval_cls_dual_visual_evidence_fusion.py",
        "--protocol", protocol,
        "--vit-score-dir", str(base / "vit" / "scores"),
        "--cnn-score-dir", str(existing_cnn_dir(dataset) / "scores"),
        "--output-root", str(base / "fusion"),
    ]
    return cmd


def marker(kind: str, variant: str, dataset: str) -> Path:
    if kind == "vit":
        return out_base(variant, dataset) / "vit" / "summary.json"
    if kind == "fusion":
        return out_base(variant, dataset) / "fusion" / "summary.json"
    raise KeyError(kind)


def log_line(message: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (ROOT / "scheduler.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(name: str, cmd: list[str], gpu: int | None, log_path: Path, marker_path: Path) -> None:
    if marker_path.exists():
        log_line(f"SKIP {name}: marker exists")
        return
    actual = [str(gpu) if part == "%GPU%" else part for part in cmd]
    log_line(f"START {name} gpu={gpu}: {' '.join(actual)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(actual, stdout=log, stderr=subprocess.STDOUT, text=True)
    log_line(f"DONE {name}: returncode={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed; see {log_path}")


def schedule() -> None:
    gpu_jobs = []
    cpu_jobs = []
    for variant in VIT_VARIANTS:
        for dataset in DATASETS:
            gpu_jobs.append((variant, dataset, vit_cmd(variant, dataset)))
            if variant != "no_all":
                cpu_jobs.append((variant, dataset, fusion_cmd(variant, dataset)))

    gpu_queue: Queue[int] = Queue()
    for gpu in GPUS:
        gpu_queue.put(gpu)

    done_vit = {
        (variant, dataset)
        for variant in VIT_VARIANTS
        for dataset in DATASETS
        if marker("vit", variant, dataset).exists()
    }
    done_fusion = {
        (variant, dataset)
        for variant in VIT_VARIANTS
        if variant != "no_all"
        for dataset in DATASETS
        if marker("fusion", variant, dataset).exists()
    }

    futures: dict[Future, tuple[str, str, str]] = {}
    submitted_fusion: set[tuple[str, str]] = set(done_fusion)

    def submit_gpu(executor: ThreadPoolExecutor, variant: str, dataset: str, cmd: list[str]) -> None:
        gpu = gpu_queue.get()
        name = f"{variant}/{dataset}/vit"

        def wrapped() -> None:
            try:
                run_cmd(
                    name,
                    cmd,
                    gpu,
                    out_base(variant, dataset) / "vit.log",
                    marker("vit", variant, dataset),
                )
            finally:
                gpu_queue.put(gpu)

        futures[executor.submit(wrapped)] = ("vit", variant, dataset)

    with ThreadPoolExecutor(max_workers=len(GPUS) + 3) as executor:
        for variant, dataset, cmd in gpu_jobs:
            if (variant, dataset) in done_vit:
                log_line(f"SKIP {variant}/{dataset}/vit: marker exists")
            else:
                submit_gpu(executor, variant, dataset, cmd)

        while futures:
            for fut in list(futures):
                if not fut.done():
                    continue
                kind, variant, dataset = futures.pop(fut)
                fut.result()
                if kind == "vit":
                    done_vit.add((variant, dataset))
                else:
                    done_fusion.add((variant, dataset))

            for variant, dataset, cmd in cpu_jobs:
                key = (variant, dataset)
                if key in submitted_fusion or key not in done_vit:
                    continue
                name = f"{variant}/{dataset}/fusion"
                futures[executor.submit(
                    run_cmd,
                    name,
                    cmd,
                    None,
                    out_base(variant, dataset) / "fusion.log",
                    marker("fusion", variant, dataset),
                )] = ("fusion", variant, dataset)
                submitted_fusion.add(key)
            time.sleep(1.0)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def vit_auc(dataset: str, path: Path) -> float:
    data = read_json(path)
    key = "clip_vit_nn_max_auc_macro" if dataset == "spectrum" else "vit_patchcore_max_auc_macro"
    return float(data[key])


def fusion_auc(path: Path, key: str) -> float:
    return float(read_json(path)["macro"][key])


def summarize() -> None:
    rows = []
    for dataset in DATASETS:
        current_vit = existing_vit_dir(dataset) / "summary.json"
        current_fusion = existing_fusion_dir(dataset) / "summary.json"
        cnn_summary = existing_cnn_dir(dataset) / "summary.json"
        rows.append({
            "dataset": dataset,
            "variant": "full_current",
            "vit_auc": vit_auc(dataset, current_vit),
            "cnn_auc": read_cnn_auc(dataset, cnn_summary),
            "final_auc": fusion_auc(current_fusion, "confidence_gated_auc"),
        })
        for variant in VIT_VARIANTS:
            v_path = marker("vit", variant, dataset)
            if variant == "no_all":
                if not v_path.exists():
                    continue
                plain_vit_auc = vit_auc(dataset, v_path)
                rows.append({
                    "dataset": dataset,
                    "variant": variant,
                    "vit_auc": plain_vit_auc,
                    "cnn_auc": None,
                    "final_auc": plain_vit_auc,
                })
                continue
            f_path = marker("fusion", variant, dataset)
            if not v_path.exists() or not f_path.exists():
                continue
            rows.append({
                "dataset": dataset,
                "variant": variant,
                "vit_auc": vit_auc(dataset, v_path),
                "cnn_auc": read_cnn_auc(dataset, cnn_summary),
                "final_auc": fusion_auc(f_path, "confidence_gated_auc"),
            })

    out_csv = ROOT / "module_ablation_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Module Ablation",
        "",
        "Date: 2026-07-09",
        "",
        "Rows use the same support protocol as the current method:",
        "",
        "- self RF: `per_frequency`",
        "- public RF: `per_frequency`",
        "- Spectrum: `4shot`",
        "",
        "Variants:",
        "",
        "- `full_current`: ViT farthest50 + top-k5 + `stft_shift_blur`, CNN top-1, confidence gate.",
        "- `no_all`: plain ViT only: single view + full random gallery + top-1 NN; CNN and gate are removed too.",
        "- `no_tta`: disables paired TTA only.",
        "- `no_topk`: changes ViT top-k5 to top-1 only.",
        "- `no_coreset`: changes ViT farthest50 to full random gallery only.",
        "",
        "## Macro Image-AUROC",
        "",
        "| dataset | variant | ViT | CNN | final score |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        cnn = "-" if row["cnn_auc"] is None else f"{row['cnn_auc']:.4f}"
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['vit_auc']:.4f} | "
            f"{cnn} | {row['final_auc']:.4f} |"
        )
    lines.extend([
        "",
        "## CNN Branch Ablation",
        "",
        "Removing the CNN branch also removes the confidence gate; the final "
        "score is therefore the ViT Normal Gallery score.",
        "",
        "| dataset | full final score | no CNN (ViT only) | delta |",
        "|---|---:|---:|---:|",
    ])
    for dataset in DATASETS:
        full = next(row for row in rows if row["dataset"] == dataset and row["variant"] == "full_current")
        delta = full["final_auc"] - full["vit_auc"]
        lines.append(
            f"| {dataset} | {full['final_auc']:.4f} | {full['vit_auc']:.4f} | {delta:+.4f} |"
        )
    (ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log_line(f"WROTE {out_csv}")


def read_cnn_auc(dataset: str, path: Path) -> float:
    data = read_json(path)
    return float(data["image_auroc_macro"])


def main() -> None:
    schedule()
    summarize()


if __name__ == "__main__":
    main()
