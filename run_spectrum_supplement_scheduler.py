#!/usr/bin/env python
"""Scheduler for the current-method Spectrum supplement.

Extends the confidence-gate sampling ablation with spectrum results
for 1shot/2shot/4shot (no per_frequency — spectrum images carry no RF band labels).

Per sampling: spectrum_vit (GPU) + spectrum_aux_cnn (GPU) independent, then
spectrum_fusion (CPU) once both score dirs exist. Idempotent: skips any job
whose summary.json already exists.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from queue import Queue

GPUS = [0, 2, 3]
ROOT = Path("analysis_outputs/20260725_confidence_gate_sampling_ablation")
SAMPLINGS = ["1shot", "2shot", "4shot"]  # no per_frequency for spectrum


def vit_cmd(s: str, base: Path) -> list[str]:
    # Current-method gallery config (mirrors self_vit / public_vit):
    # farthest coreset 0.5, topk 5, stft_shift_blur TTA. prompt-mode defaults to
    # rf (current method); checkpoint default already points at the moved path.
    return [
        "python", "tools/eval_cls_spectrum_vit_nn_gallery.py",
        "--output-root", str(base / "spectrum_vit"),
        "--normal-sampling", s,
        "--coreset-method", "farthest",
        "--coreset-ratio", "0.5",
        "--nn-topk", "5",
        "--nn-agg", "mean",
        "--paired-tta", "stft_shift_blur",
        "--paired-tta-fusion", "max",
        "--gpu-id", "%GPU%",
    ]


def aux_cnn_cmd(s: str, base: Path) -> list[str]:
    return [
        "python", "tools/eval_cls_aux_cnn_gallery.py",
        "--protocol", "spectrum",
        "--output-root", str(base / "spectrum_aux_cnn"),
        "--normal-sampling", s,
        "--batch-size", "32",
        "--gpu-id", "%GPU%",
    ]


def fusion_cmd(s: str, base: Path) -> list[str]:
    return [
        "python", "tools/eval_cls_dual_visual_evidence_fusion.py",
        "--protocol", "spectrum",
        "--vit-score-dir", str(base / "spectrum_vit" / "scores"),
        "--cnn-score-dir", str(base / "spectrum_aux_cnn" / "scores"),
        "--output-root", str(base / "spectrum_fusion"),
    ]


def marker_path(name: str) -> Path:
    s, role = name.split("/")
    return ROOT / s / role / "summary.json"


def log_path(name: str) -> Path:
    s, role = name.split("/")
    return ROOT / s / f"{role}.log"


def main():
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
    cpu_executor = ThreadPoolExecutor(max_workers=3)
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

    # Submit GPU jobs: vit (longer) first, then cnn.
    gpu_jobs = []
    for s in SAMPLINGS:
        base = ROOT / s
        base.mkdir(parents=True, exist_ok=True)
        gpu_jobs.append(("spectrum_vit", s, vit_cmd(s, base)))
        gpu_jobs.append(("spectrum_aux_cnn", s, aux_cnn_cmd(s, base)))

    for role, s, cmd in gpu_jobs:
        name = f"{s}/{role}"
        if marker_path(name).exists():
            print(f"[{ts()}] SKIP  {name} (marker exists)", flush=True)
            completed.add(name)
            continue
        fut = gpu_executor.submit(gpu_worker, name, cmd)
        fut.add_done_callback(on_done(name))
        futures[name] = fut

    # CPU fusion jobs: submit when deps (vit + cnn) are completed.
    pending_cpu = []
    for s in SAMPLINGS:
        base = ROOT / s
        name = f"{s}/spectrum_fusion"
        deps = [f"{s}/spectrum_vit", f"{s}/spectrum_aux_cnn"]
        pending_cpu.append((name, fusion_cmd(s, base), deps))

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
        if not pending_cpu:
            break

    for name, fut in futures.items():
        fut.result()

    gpu_executor.shutdown(wait=True)
    cpu_executor.shutdown(wait=True)

    print(f"\n[{ts()}] === SUMMARY ===", flush=True)
    with completed_lock:
        print(f"completed: {sorted(completed)}", flush=True)
        print(f"failed: {sorted(failed)}", flush=True)
    for s in SAMPLINGS:
        for role in [
            "spectrum_vit",
            "spectrum_aux_cnn",
            "spectrum_fusion",
        ]:
            name = f"{s}/{role}"
            mk = marker_path(name)
            print(f"  {name}: {'OK' if mk.exists() else 'MISSING'}", flush=True)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
