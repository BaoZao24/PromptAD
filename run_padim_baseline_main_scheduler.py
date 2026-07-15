#!/usr/bin/env python
"""Scheduler for main PaDiM-style baseline experiments."""

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

GPUS = [0, 2, 3]
ROOT = Path("analysis_outputs/20260709_padim_baseline_main")


def base_cmd(protocol: str, sampling: str, out_dir: Path) -> list[str]:
    cmd = [
        "python", "tools/eval_padim_cls.py",
        "--protocol", protocol,
        "--normal-sampling", sampling,
        "--output-root", str(out_dir),
        "--resize", "256",
        "--imagesize", "224",
        "--batch-size", "32",
        "--num-workers", "2",
        "--weights", "default",
        "--embedding-dim", "128",
        "--var-eps", "0.01",
        "--gpu-id", "%GPU%",
    ]
    if protocol == "rf_target":
        cmd.extend(["--rf-train-mode", "pooled"])
    return cmd


def build_jobs() -> list[tuple[str, list[str]]]:
    return [
        ("per_frequency/self_padim", base_cmd("rf_target", "per_frequency", ROOT / "per_frequency" / "self_padim")),
        ("per_frequency/public_padim", base_cmd("public_rf", "per_frequency", ROOT / "per_frequency" / "public_padim")),
        ("4shot/spectrum_padim", base_cmd("spectrum", "4shot", ROOT / "4shot" / "spectrum_padim")),
    ]


def marker_path(name: str) -> Path:
    sampling, role = name.split("/")
    return ROOT / sampling / role / "summary.json"


def log_path(name: str) -> Path:
    sampling, role = name.split("/")
    return ROOT / sampling / f"{role}.log"


def run_job(name: str, cmd_template: list[str], gpu: int, scheduler_log: Path) -> tuple[str, int]:
    cmd = [str(gpu) if part == "%GPU%" else part for part in cmd_template]
    out_log = log_path(name)
    out_log.parent.mkdir(parents=True, exist_ok=True)
    with scheduler_log.open("a", encoding="utf-8") as sched:
        sched.write(f"[{time.strftime('%F %T')}] START gpu={gpu} {name}: {' '.join(cmd)}\n")
    with out_log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    with scheduler_log.open("a", encoding="utf-8") as sched:
        sched.write(f"[{time.strftime('%F %T')}] DONE gpu={gpu} {name}: returncode={proc.returncode}\n")
    return name, proc.returncode


def summarize() -> None:
    specs = [
        ("self_rf", "per_frequency", "self_padim"),
        ("public_rf", "per_frequency", "public_padim"),
        ("spectrum", "4shot", "spectrum_padim"),
    ]
    rows = []
    for dataset, sampling, role in specs:
        summary_path = ROOT / sampling / role / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({
            "dataset": dataset,
            "sampling": sampling,
            "padim_auc": summary.get("image_auroc_macro"),
            "weights": summary.get("weights"),
            "embedding_dim": summary.get("embedding_dim"),
            "output_root": str(summary_path.parent),
        })
    if not rows:
        return
    out_csv = ROOT / "padim_baseline_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# PaDiM-style Main Baseline",
        "",
        "Diagonal-covariance PaDiM-style ResNet18 baseline, normal-only, Image-AUROC.",
        "",
        "| dataset | sampling | PaDiM AUROC |",
        "|---|---|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['sampling']} | {float(row['padim_auc']):.4f} |")
    lines.extend(["", f"CSV: `{out_csv.name}`"])
    (ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    scheduler_log = ROOT / "scheduler.log"
    queue: Queue[tuple[str, list[str]]] = Queue()
    for name, cmd in build_jobs():
        if marker_path(name).exists():
            with scheduler_log.open("a", encoding="utf-8") as sched:
                sched.write(f"[{time.strftime('%F %T')}] SKIP {name}: marker exists\n")
            continue
        queue.put((name, cmd))

    failures: list[tuple[str, int]] = []
    lock = threading.Lock()

    def worker(gpu: int):
        while True:
            try:
                name, cmd = queue.get_nowait()
            except Exception:
                return
            result_name, code = run_job(name, cmd, gpu, scheduler_log)
            if code != 0:
                with lock:
                    failures.append((result_name, code))
            queue.task_done()

    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        for gpu in GPUS:
            pool.submit(worker, gpu)

    summarize()
    if failures:
        print(json.dumps({"failures": failures}, indent=2), file=sys.stderr)
        return 1
    print(f"wrote {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
