#!/usr/bin/env python
"""Run CNN local-memory coreset/top-k ablations and fuse with existing ViT scores."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

GPUS = [0, 1, 3]
ROOT = Path("analysis_outputs/20260709_cnn_coreset_topk_ablation")
BASE = Path("analysis_outputs/20260706_sampling_shot_ablation")

VARIANTS = {
    "farthest50_topk5": {
        "sampler": "approx_greedy_coreset",
        "coreset_percentage": "0.5",
        "topk": "5",
    },
    "farthest25_topk9": {
        "sampler": "approx_greedy_coreset",
        "coreset_percentage": "0.25",
        "topk": "9",
    },
}


def cnn_cmd(protocol: str, sampling: str, out_dir: Path, variant: dict[str, str]) -> list[str]:
    cmd = [
        "python", "tools/eval_patchcore_cls.py",
        "--protocol", protocol,
        "--output-root", str(out_dir),
        "--normal-sampling", sampling,
        "--batch-size", "32",
        "--sampler", variant["sampler"],
        "--coreset-percentage", variant["coreset_percentage"],
        "--anomaly-scorer-num-nn", variant["topk"],
        "--gpu-id", "%GPU%",
    ]
    if protocol == "rf_target":
        cmd.extend(["--rf-train-mode", "pooled"])
    return cmd


def fusion_cmd(protocol: str, variant_name: str, role: str, sampling: str) -> list[str]:
    variant_root = ROOT / variant_name
    if protocol == "rf_target":
        return [
            "python", "tools/eval_cls_dual_visual_evidence_fusion.py",
            "--protocol", "rf_target",
            "--vit-score-dir", str(BASE / "per_frequency" / "self_vit" / "scores"),
            "--cnn-score-dir", str(variant_root / "per_frequency" / "self_cnn" / "scores"),
            "--output-root", str(variant_root / "per_frequency" / "self_fusion"),
        ]
    if protocol == "public_rf":
        return [
            "python", "tools/eval_cls_dual_visual_evidence_fusion.py",
            "--protocol", "public_rf",
            "--vit-score-dir", str(BASE / "per_frequency" / "public_vit" / "scores"),
            "--cnn-score-dir", str(variant_root / "per_frequency" / "public_cnn" / "scores"),
            "--output-root", str(variant_root / "per_frequency" / "public_fusion"),
        ]
    if protocol == "spectrum":
        return [
            "python", "tools/eval_cls_dual_visual_evidence_fusion.py",
            "--protocol", "spectrum",
            "--vit-score-dir", str(BASE / "4shot" / "spectrum_vit" / "scores"),
            "--vit-key", "clip_vit_nn_max_scores",
            "--cnn-score-dir", str(variant_root / "4shot" / "spectrum_cnn" / "scores"),
            "--output-root", str(variant_root / "4shot" / "spectrum_fusion"),
        ]
    raise ValueError(protocol)


def build_jobs():
    gpu_jobs = []
    cpu_jobs = []
    for variant_name, variant in VARIANTS.items():
        variant_root = ROOT / variant_name
        gpu_jobs.extend(
            [
                (
                    f"{variant_name}/per_frequency/self_cnn",
                    cnn_cmd("rf_target", "per_frequency", variant_root / "per_frequency" / "self_cnn", variant),
                ),
                (
                    f"{variant_name}/per_frequency/public_cnn",
                    cnn_cmd("public_rf", "per_frequency", variant_root / "per_frequency" / "public_cnn", variant),
                ),
                (
                    f"{variant_name}/4shot/spectrum_cnn",
                    cnn_cmd("spectrum", "4shot", variant_root / "4shot" / "spectrum_cnn", variant),
                ),
            ]
        )
        cpu_jobs.extend(
            [
                (
                    f"{variant_name}/per_frequency/self_fusion",
                    fusion_cmd("rf_target", variant_name, "self_fusion", "per_frequency"),
                    [f"{variant_name}/per_frequency/self_cnn"],
                ),
                (
                    f"{variant_name}/per_frequency/public_fusion",
                    fusion_cmd("public_rf", variant_name, "public_fusion", "per_frequency"),
                    [f"{variant_name}/per_frequency/public_cnn"],
                ),
                (
                    f"{variant_name}/4shot/spectrum_fusion",
                    fusion_cmd("spectrum", variant_name, "spectrum_fusion", "4shot"),
                    [f"{variant_name}/4shot/spectrum_cnn"],
                ),
            ]
        )
    return gpu_jobs, cpu_jobs


def marker_path(name: str) -> Path:
    variant, sampling, role = name.split("/")
    return ROOT / variant / sampling / role / "summary.json"


def log_path(name: str) -> Path:
    variant, sampling, role = name.split("/")
    return ROOT / variant / sampling / f"{role}.log"


def run_subprocess(name: str, cmd_template: list[str], scheduler_log: Path, gpu: int | None = None) -> tuple[str, int]:
    cmd = [str(gpu) if part == "%GPU%" else part for part in cmd_template]
    log = log_path(name)
    log.parent.mkdir(parents=True, exist_ok=True)
    with scheduler_log.open("a", encoding="utf-8") as sched:
        sched.write(f"[{time.strftime('%F %T')}] START {name} gpu={gpu}: {' '.join(cmd)}\n")
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    with scheduler_log.open("a", encoding="utf-8") as sched:
        sched.write(f"[{time.strftime('%F %T')}] DONE {name}: returncode={proc.returncode}\n")
    return name, proc.returncode


def metric(summary: dict, key: str):
    macro = summary.get("macro")
    if isinstance(macro, dict) and key in macro:
        return macro[key]
    return summary.get(f"{key}_macro", "")


def summarize() -> None:
    rows = []
    for variant_name, variant in VARIANTS.items():
        root = ROOT / variant_name
        specs = [
            ("self_rf", "per_frequency", "self_cnn", "self_fusion"),
            ("public_rf", "per_frequency", "public_cnn", "public_fusion"),
            ("spectrum", "4shot", "spectrum_cnn", "spectrum_fusion"),
        ]
        for dataset, sampling, cnn_role, fusion_role in specs:
            cnn_summary = root / sampling / cnn_role / "summary.json"
            fusion_summary = root / sampling / fusion_role / "summary.json"
            if not cnn_summary.exists() or not fusion_summary.exists():
                continue
            cnn = json.loads(cnn_summary.read_text(encoding="utf-8"))
            fusion = json.loads(fusion_summary.read_text(encoding="utf-8"))
            rows.append(
                {
                    "variant": variant_name,
                    "dataset": dataset,
                    "sampling": sampling,
                    "sampler": variant["sampler"],
                    "coreset_percentage": variant["coreset_percentage"],
                    "cnn_topk": variant["topk"],
                    "cnn_auc": cnn.get("image_auroc_macro"),
                    "calibrated_auc": metric(fusion, "normal_calibrated_confidence_or_auc"),
                    "or_auc": metric(fusion, "or_evidence_auc"),
                    "vit_auc": metric(fusion, "vit_auc"),
                    "output_root": str(root / sampling),
                }
            )
    if not rows:
        return
    out_csv = ROOT / "cnn_coreset_topk_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# CNN Coreset Top-k Ablation",
        "",
        "CNN local-memory variants fused with existing ViT scores.",
        "",
        "| variant | dataset | sampling | CNN AUROC | calibrated AUROC | OR AUROC | ViT AUROC |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['dataset']} | {row['sampling']} | "
            f"{float(row['cnn_auc']):.4f} | {float(row['calibrated_auc']):.4f} | "
            f"{float(row['or_auc']):.4f} | {float(row['vit_auc']):.4f} |"
        )
    lines.extend(["", f"CSV: `{out_csv.name}`"])
    (ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    scheduler_log = ROOT / "scheduler.log"
    gpu_jobs, cpu_jobs = build_jobs()
    gpu_q: Queue[int] = Queue()
    for gpu in GPUS:
        gpu_q.put(gpu)

    completed: set[str] = set()
    failed: list[tuple[str, int]] = []
    lock = threading.Lock()

    def gpu_worker(name: str, cmd: list[str]):
        if marker_path(name).exists():
            with lock:
                completed.add(name)
            return name, 0
        gpu = gpu_q.get()
        try:
            return run_subprocess(name, cmd, scheduler_log, gpu=gpu)
        finally:
            gpu_q.put(gpu)

    def cpu_worker(name: str, cmd: list[str]):
        if marker_path(name).exists():
            with lock:
                completed.add(name)
            return name, 0
        return run_subprocess(name, cmd, scheduler_log, gpu=None)

    with ThreadPoolExecutor(max_workers=len(GPUS)) as gpu_pool, ThreadPoolExecutor(max_workers=3) as cpu_pool:
        futures = {gpu_pool.submit(gpu_worker, name, cmd): name for name, cmd in gpu_jobs}
        pending_cpu = list(cpu_jobs)
        submitted_cpu = {}
        while futures or pending_cpu or submitted_cpu:
            done = [future for future in list(futures) if future.done()]
            for future in done:
                name, code = future.result()
                futures.pop(future)
                with lock:
                    if code == 0:
                        completed.add(name)
                    else:
                        failed.append((name, code))
            ready = []
            with lock:
                snapshot = set(completed)
            for item in pending_cpu:
                name, _cmd, deps = item
                if all(dep in snapshot for dep in deps):
                    ready.append(item)
            for item in ready:
                pending_cpu.remove(item)
                name, cmd, _deps = item
                submitted_cpu[cpu_pool.submit(cpu_worker, name, cmd)] = name
            done_cpu = [future for future in list(submitted_cpu) if future.done()]
            for future in done_cpu:
                name, code = future.result()
                submitted_cpu.pop(future)
                with lock:
                    if code == 0:
                        completed.add(name)
                    else:
                        failed.append((name, code))
            time.sleep(2)

    summarize()
    if failed:
        print(json.dumps({"failed": failed}, indent=2), file=sys.stderr)
        return 1
    print(f"wrote {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
