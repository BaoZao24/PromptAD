#!/usr/bin/env python
"""Run STFT time-shift TTA v2 ablation for the ViT branch.

This reuses existing CNN local-gallery scores from the sampling ablation and
only recomputes the ViT score files affected by paired TTA.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Queue


GPUS = [0, 2, 3]
ROOT = Path("analysis_outputs/20260709_tta_time_shift_v2")
BASE = Path("analysis_outputs/20260706_sampling_shot_ablation")
PUB_CKPT = (
    "analysis_outputs/90_rejected_or_aborted/20260706_cleanup_old_results/"
    "20260627_method_funnel/runs/pooled_rf_rgb/cls/checkpoint/overall-best.pt"
)


def self_vit_cmd() -> list[str]:
    return [
        "python", "tools/eval_cls_vit_patchcore_gallery.py",
        "--output-root", str(ROOT / "per_frequency" / "self_vit"),
        "--normal-sampling", "per_frequency",
        "--patch-layer", "concat",
        "--coreset-method", "farthest",
        "--coreset-ratio", "0.5",
        "--nn-topk", "5",
        "--nn-agg", "mean",
        "--paired-tta", "stft_time_shift_v2",
        "--paired-tta-fusion", "max",
        "--paired-tta-shift-px", "8",
        "--gpu-id", "%GPU%",
    ]


def public_vit_cmd() -> list[str]:
    return [
        "python", "tools/eval_cls_public_rf_vit_patchcore_gallery.py",
        "--output-root", str(ROOT / "per_frequency" / "public_vit"),
        "--normal-sampling", "per_frequency",
        "--checkpoint", PUB_CKPT,
        "--patch-layer", "concat",
        "--coreset-method", "farthest",
        "--coreset-ratio", "0.5",
        "--nn-topk", "5",
        "--nn-agg", "mean",
        "--paired-tta", "stft_time_shift_v2",
        "--paired-tta-fusion", "max",
        "--paired-tta-shift-px", "8",
        "--gpu-id", "%GPU%",
    ]


def spectrum_vit_cmd() -> list[str]:
    return [
        "python", "tools/eval_cls_spectrum_vit_nn_gallery.py",
        "--output-root", str(ROOT / "4shot" / "spectrum_vit"),
        "--normal-sampling", "4shot",
        "--patch-layer", "concat",
        "--coreset-method", "farthest",
        "--coreset-ratio", "0.5",
        "--nn-topk", "5",
        "--nn-agg", "mean",
        "--paired-tta", "stft_time_shift_v2",
        "--paired-tta-fusion", "max",
        "--paired-tta-shift-px", "8",
        "--gpu-id", "%GPU%",
    ]


def self_fusion_cmd() -> list[str]:
    return [
        "python", "tools/eval_cls_dual_visual_evidence_fusion.py",
        "--protocol", "rf_target",
        "--vit-score-dir", str(ROOT / "per_frequency" / "self_vit" / "scores"),
        "--cnn-score-dir", str(BASE / "per_frequency" / "self_cnn" / "scores"),
        "--output-root", str(ROOT / "per_frequency" / "self_fusion"),
    ]


def public_fusion_cmd() -> list[str]:
    return [
        "python", "tools/eval_cls_dual_visual_evidence_fusion.py",
        "--protocol", "public_rf",
        "--vit-score-dir", str(ROOT / "per_frequency" / "public_vit" / "scores"),
        "--cnn-score-dir", str(BASE / "per_frequency" / "public_cnn" / "scores"),
        "--output-root", str(ROOT / "per_frequency" / "public_fusion"),
    ]


def spectrum_fusion_cmd() -> list[str]:
    return [
        "python", "tools/eval_cls_dual_visual_evidence_fusion.py",
        "--protocol", "spectrum",
        "--vit-score-dir", str(ROOT / "4shot" / "spectrum_vit" / "scores"),
        "--vit-key", "clip_vit_nn_max_scores",
        "--cnn-score-dir", str(BASE / "4shot" / "spectrum_cnn" / "scores"),
        "--output-root", str(ROOT / "4shot" / "spectrum_fusion"),
    ]


GPU_JOBS = [
    ("self_vit", self_vit_cmd()),
    ("public_vit", public_vit_cmd()),
    ("spectrum_vit", spectrum_vit_cmd()),
]
CPU_JOBS = [
    ("self_fusion", self_fusion_cmd(), ["self_vit"]),
    ("public_fusion", public_fusion_cmd(), ["public_vit"]),
    ("spectrum_fusion", spectrum_fusion_cmd(), ["spectrum_vit"]),
]


def marker_for(name: str) -> Path:
    if name == "self_vit":
        return ROOT / "per_frequency" / "self_vit" / "summary.json"
    if name == "public_vit":
        return ROOT / "per_frequency" / "public_vit" / "summary.json"
    if name == "spectrum_vit":
        return ROOT / "4shot" / "spectrum_vit" / "summary.json"
    if name == "self_fusion":
        return ROOT / "per_frequency" / "self_fusion" / "summary.json"
    if name == "public_fusion":
        return ROOT / "per_frequency" / "public_fusion" / "summary.json"
    if name == "spectrum_fusion":
        return ROOT / "4shot" / "spectrum_fusion" / "summary.json"
    raise KeyError(name)


def log_line(message: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (ROOT / "scheduler.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(name: str, cmd: list[str], gpu: int | None) -> None:
    if marker_for(name).exists():
        log_line(f"SKIP {name}: marker exists")
        return
    actual = [str(gpu) if part == "%GPU%" else part for part in cmd]
    log_line(f"START {name} gpu={gpu}: {' '.join(actual)}")
    log_path = marker_for(name).parent.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(actual, stdout=log, stderr=subprocess.STDOUT, text=True)
    log_line(f"DONE {name}: returncode={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed; see {log_path}")


def schedule() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    done = {name for name, _ in GPU_JOBS if marker_for(name).exists()}
    done.update({name for name, _, _ in CPU_JOBS if marker_for(name).exists()})
    lock = threading.Lock()
    gpu_queue: Queue[int] = Queue()
    for gpu in GPUS:
        gpu_queue.put(gpu)

    futures: dict[Future, str] = {}

    def submit_gpu(executor: ThreadPoolExecutor, name: str, cmd: list[str]) -> None:
        gpu = gpu_queue.get()

        def wrapped() -> None:
            try:
                run_cmd(name, cmd, gpu)
            finally:
                gpu_queue.put(gpu)

        futures[executor.submit(wrapped)] = name

    with ThreadPoolExecutor(max_workers=len(GPUS) + 3) as executor:
        for name, cmd in GPU_JOBS:
            if name in done:
                log_line(f"SKIP {name}: marker exists")
            else:
                submit_gpu(executor, name, cmd)

        submitted_cpu: set[str] = set()
        while futures:
            for fut in list(futures):
                if not fut.done():
                    continue
                name = futures.pop(fut)
                fut.result()
                with lock:
                    done.add(name)

            for name, cmd, deps in CPU_JOBS:
                if name in done or name in submitted_cpu:
                    continue
                if all(dep in done for dep in deps):
                    futures[executor.submit(run_cmd, name, cmd, None)] = name
                    submitted_cpu.add(name)
            time.sleep(1.0)


def read_metric(path: Path, key: str) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data[key])


def read_macro_metric(path: Path, key: str) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["macro"][key])


def summarize() -> None:
    rows = [
        {
            "dataset": "self_rf",
            "sampling": "per_frequency",
            "tta": "stft_time_shift_v2",
            "shift_px": 8,
            "vit_auc": read_metric(ROOT / "per_frequency" / "self_vit" / "summary.json", "vit_patchcore_max_auc_macro"),
            "calibrated_auc": read_macro_metric(ROOT / "per_frequency" / "self_fusion" / "summary.json", "normal_calibrated_confidence_or_auc"),
        },
        {
            "dataset": "public_rf",
            "sampling": "per_frequency",
            "tta": "stft_time_shift_v2",
            "shift_px": 8,
            "vit_auc": read_metric(ROOT / "per_frequency" / "public_vit" / "summary.json", "vit_patchcore_max_auc_macro"),
            "calibrated_auc": read_macro_metric(ROOT / "per_frequency" / "public_fusion" / "summary.json", "normal_calibrated_confidence_or_auc"),
        },
        {
            "dataset": "spectrum",
            "sampling": "4shot",
            "tta": "stft_time_shift_v2",
            "shift_px": 8,
            "vit_auc": read_metric(ROOT / "4shot" / "spectrum_vit" / "summary.json", "clip_vit_nn_max_auc_macro"),
            "calibrated_auc": read_macro_metric(ROOT / "4shot" / "spectrum_fusion" / "summary.json", "normal_calibrated_confidence_or_auc"),
        },
    ]
    out = ROOT / "tta_time_shift_v2_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log_line(f"WROTE {out}")


def main() -> None:
    schedule()
    summarize()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_line(f"ERROR {exc}")
        raise
