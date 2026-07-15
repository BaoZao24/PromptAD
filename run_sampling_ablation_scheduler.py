#!/usr/bin/env python
"""Greedy GPU scheduler for the normal-support sampling ablation.

Runs 4 samplings x 4 independent GPU jobs + 2 CPU fusion jobs each.
Pipelines the long public_vit jobs across GPUs 0/2/3 to minimise wall time.
Idempotent: skips any job whose output marker (summary.json) already exists.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from queue import Queue

GPUS = [0, 2, 3]
ROOT = Path("analysis_outputs/20260706_sampling_shot_ablation")
SAMPLINGS = ["per_frequency", "1shot", "2shot", "4shot"]
PUB_CKPT = (
    "analysis_outputs/90_rejected_or_aborted/20260706_cleanup_old_results/"
    "20260627_method_funnel/runs/pooled_rf_rgb/cls/checkpoint/overall-best.pt"
)

# ---- job definitions ---------------------------------------------------------

GPU_JOBS = []   # (name, sampling, role, cmd_template)
CPU_JOBS = []   # (name, sampling, role, cmd, deps)


def self_vit_cmd(s: str, base: Path) -> list[str]:
    return [
        "python", "tools/eval_cls_vit_patchcore_gallery.py",
        "--output-root", str(base / "self_vit"),
        "--normal-sampling", s,
        "--patch-layer", "concat",
        "--coreset-method", "farthest",
        "--coreset-ratio", "0.5",
        "--nn-topk", "5",
        "--nn-agg", "mean",
        "--paired-tta", "stft_shift_blur",
        "--paired-tta-fusion", "max",
        "--gpu-id", "%GPU%",
    ]


def self_cnn_cmd(s: str, base: Path) -> list[str]:
    return [
        "python", "tools/eval_patchcore_cls.py",
        "--protocol", "rf_target",
        "--rf-train-mode", "pooled",
        "--output-root", str(base / "self_cnn"),
        "--normal-sampling", s,
        "--batch-size", "32",
        "--gpu-id", "%GPU%",
    ]


def public_vit_cmd(s: str, base: Path) -> list[str]:
    return [
        "python", "tools/eval_cls_public_rf_vit_patchcore_gallery.py",
        "--output-root", str(base / "public_vit"),
        "--normal-sampling", s,
        "--checkpoint", PUB_CKPT,
        "--patch-layer", "concat",
        "--coreset-method", "farthest",
        "--coreset-ratio", "0.5",
        "--nn-topk", "5",
        "--nn-agg", "mean",
        "--paired-tta", "stft_shift_blur",
        "--paired-tta-fusion", "max",
        "--gpu-id", "%GPU%",
    ]


def public_cnn_cmd(s: str, base: Path) -> list[str]:
    return [
        "python", "tools/eval_patchcore_cls.py",
        "--protocol", "public_rf",
        "--output-root", str(base / "public_cnn"),
        "--normal-sampling", s,
        "--batch-size", "32",
        "--gpu-id", "%GPU%",
    ]


def self_fusion_cmd(s: str, base: Path) -> list[str]:
    return [
        "python", "tools/eval_cls_dual_visual_evidence_fusion.py",
        "--protocol", "rf_target",
        "--vit-score-dir", str(base / "self_vit" / "scores"),
        "--cnn-score-dir", str(base / "self_cnn" / "scores"),
        "--output-root", str(base / "self_fusion"),
    ]


def public_fusion_cmd(s: str, base: Path) -> list[str]:
    return [
        "python", "tools/eval_cls_dual_visual_evidence_fusion.py",
        "--protocol", "public_rf",
        "--vit-score-dir", str(base / "public_vit" / "scores"),
        "--cnn-score-dir", str(base / "public_cnn" / "scores"),
        "--output-root", str(base / "public_fusion"),
    ]


def build_jobs():
    gpu_jobs = []
    cpu_jobs = []
    for s in SAMPLINGS:
        base = ROOT / s
        base.mkdir(parents=True, exist_ok=True)
        gpu_jobs.append(("self_vit", s, self_vit_cmd(s, base)))
        gpu_jobs.append(("self_cnn", s, self_cnn_cmd(s, base)))
        gpu_jobs.append(("public_vit", s, public_vit_cmd(s, base)))
        gpu_jobs.append(("public_cnn", s, public_cnn_cmd(s, base)))
        cpu_jobs.append((
            "self_fusion", s, self_fusion_cmd(s, base),
            [f"{s}/self_vit", f"{s}/self_cnn"],
        ))
        cpu_jobs.append((
            "public_fusion", s, public_fusion_cmd(s, base),
            [f"{s}/public_vit", f"{s}/public_cnn"],
        ))
    return gpu_jobs, cpu_jobs


# ---- scheduler ---------------------------------------------------------------

def marker_path(name: str) -> Path:
    s, role = name.split("/")
    return ROOT / s / role / "summary.json"


def log_path(name: str) -> Path:
    s, role = name.split("/")
    return ROOT / s / f"{role}.log"


def main():
    gpu_jobs, cpu_jobs = build_jobs()

    gpu_q: Queue = Queue()
    for g in GPUS:
        gpu_q.put(g)

    completed = set()
    completed_lock = threading.Lock()
    failed = set()

    def ts():
        return time.strftime("%H:%M:%S")

    def run_subprocess(name, cmd, gpu=None):
        full_cmd = [c.replace("%GPU%", str(gpu)) if gpu is not None else c for c in cmd]
        log = log_path(name)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as fh:
            fh.write(f"cmd: {' '.join(full_cmd)}\n\n")
            fh.flush()
            proc = subprocess.run(full_cmd, stdout=fh, stderr=subprocess.STDOUT)
        return proc.returncode

    def gpu_worker(name, cmd):
        gpu = gpu_q.get()
        print(f"[{ts()}] START {name} on GPU {gpu}", flush=True)
        try:
            rc = run_subprocess(name, cmd, gpu=gpu)
        except Exception as e:
            print(f"[{ts()}] ERROR {name}: {e}", flush=True)
            rc = -1
        finally:
            gpu_q.put(gpu)
        return rc

    def cpu_worker(name, cmd):
        print(f"[{ts()}] START {name} (cpu)", flush=True)
        try:
            rc = run_subprocess(name, cmd)
        except Exception as e:
            print(f"[{ts()}] ERROR {name}: {e}", flush=True)
            rc = -1
        return rc

    gpu_executor = ThreadPoolExecutor(max_workers=len(GPUS))
    cpu_executor = ThreadPoolExecutor(max_workers=4)

    futures: dict[str, Future] = {}

    def on_done(name):
        def cb(fut: Future):
            rc = fut.result()
            with completed_lock:
                if rc == 0:
                    completed.add(name)
                    print(f"[{ts()}] DONE  {name} (rc=0)", flush=True)
                else:
                    failed.add(name)
                    print(f"[{ts()}] FAIL  {name} (rc={rc})", flush=True)
        return cb

    # Submit GPU jobs in LPT-ish order: public_vit (longest) first, then the rest.
    priority = {"public_vit": 0, "self_vit": 1, "self_cnn": 2, "public_cnn": 3}
    gpu_jobs_sorted = sorted(gpu_jobs, key=lambda j: (priority[j[0]], j[1]))

    for role, s, cmd in gpu_jobs_sorted:
        name = f"{s}/{role}"
        if marker_path(name).exists():
            print(f"[{ts()}] SKIP  {name} (marker exists)", flush=True)
            completed.add(name)
            continue
        fut = gpu_executor.submit(gpu_worker, name, cmd)
        fut.add_done_callback(on_done(name))
        futures[name] = fut

    # CPU jobs: submit once deps are all completed. Poll.
    pending_cpu = [(f"{s}/{role}", cmd, deps) for role, s, cmd, deps in cpu_jobs]
    cpu_done = False
    while pending_cpu:
        time.sleep(2)
        with completed_lock:
            snapshot = set(completed)
        ready = [j for j in pending_cpu if all(d in snapshot for d in j[2])]
        for name, cmd, deps in ready:
            pending_cpu.remove((name, cmd, deps))
            if marker_path(name).exists():
                print(f"[{ts()}] SKIP  {name} (marker exists)", flush=True)
                completed.add(name)
                continue
            fut = cpu_executor.submit(cpu_worker, name, cmd)
            fut.add_done_callback(on_done(name))
            futures[name] = fut
        # exit when all cpu jobs submitted and all done
        if not pending_cpu:
            break

    # Wait for everything
    for name, fut in futures.items():
        fut.result()

    gpu_executor.shutdown(wait=True)
    cpu_executor.shutdown(wait=True)

    print(f"\n[{ts()}] === SUMMARY ===", flush=True)
    with completed_lock:
        print(f"completed: {sorted(completed)}", flush=True)
        print(f"failed: {sorted(failed)}", flush=True)

    # final table of markers
    print(f"\n[{ts()}] === MARKERS ===", flush=True)
    for s in SAMPLINGS:
        for role in ["self_vit", "self_cnn", "self_fusion", "public_vit", "public_cnn", "public_fusion"]:
            name = f"{s}/{role}"
            mk = marker_path(name)
            print(f"  {name}: {'OK' if mk.exists() else 'MISSING'}", flush=True)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
