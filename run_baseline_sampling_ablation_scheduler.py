#!/usr/bin/env python
"""Greedy GPU scheduler for generic PromptAD baseline support sampling ablation."""

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
ROOT = Path("analysis_outputs/20260706_baseline_sampling_shot_ablation")
SAMPLINGS = ["per_frequency", "1shot", "2shot", "4shot"]
SPECTRUM_SAMPLINGS = ["1shot", "2shot", "4shot"]
PUB_CKPT = (
    "analysis_outputs/90_rejected_or_aborted/20260706_cleanup_old_results/"
    "20260627_method_funnel/runs/pooled_rf_rgb/cls/checkpoint/overall-best.pt"
)


def self_baseline_cmd(sampling: str, base: Path) -> list[str]:
    return [
        "python", "tools/eval_cls_vit_patch_gallery.py",
        "--output-root", str(base / "self_baseline"),
        "--normal-sampling", sampling,
        "--prompt-mode", "generic",
        "--input-mode", "rgb",
        "--text-prototype-mode", "single",
        "--cls-score-mode", "text_only",
        "--gpu-id", "%GPU%",
    ]


def public_baseline_cmd(sampling: str, base: Path) -> list[str]:
    return [
        "python", "tools/eval_cls_public_rf_vit_patchcore_gallery.py",
        "--output-root", str(base / "public_baseline"),
        "--normal-sampling", sampling,
        "--checkpoint", PUB_CKPT,
        "--prompt-mode", "generic",
        "--input-mode", "rgb",
        "--text-prototype-mode", "single",
        "--cls-score-mode", "text_only",
        "--coreset-method", "random",
        "--coreset-ratio", "1.0",
        "--nn-topk", "1",
        "--nn-agg", "mean",
        "--paired-tta", "none",
        "--gpu-id", "%GPU%",
    ]


def spectrum_baseline_cmd(sampling: str, base: Path) -> list[str]:
    return [
        "python", "tools/eval_cls_spectrum_vit_nn_gallery.py",
        "--output-root", str(base / "spectrum_baseline"),
        "--normal-sampling", sampling,
        "--checkpoint", PUB_CKPT,
        "--prompt-mode", "generic",
        "--input-mode", "rgb",
        "--text-prototype-mode", "single",
        "--cls-score-mode", "text_only",
        "--coreset-method", "random",
        "--coreset-ratio", "1.0",
        "--nn-topk", "1",
        "--nn-agg", "mean",
        "--paired-tta", "none",
        "--gpu-id", "%GPU%",
    ]


def build_jobs() -> list[tuple[str, str, list[str]]]:
    jobs = []
    for sampling in SAMPLINGS:
        base = ROOT / sampling
        base.mkdir(parents=True, exist_ok=True)
        jobs.append(("self_baseline", sampling, self_baseline_cmd(sampling, base)))
        jobs.append(("public_baseline", sampling, public_baseline_cmd(sampling, base)))
        if sampling in SPECTRUM_SAMPLINGS:
            jobs.append(("spectrum_baseline", sampling, spectrum_baseline_cmd(sampling, base)))
    return jobs


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
    rows = []
    for sampling in SAMPLINGS:
        self_summary_path = ROOT / sampling / "self_baseline" / "summary.json"
        public_summary_path = ROOT / sampling / "public_baseline" / "summary.json"
        spectrum_summary_path = ROOT / sampling / "spectrum_baseline" / "summary.json"
        if not self_summary_path.exists() or not public_summary_path.exists():
            continue
        self_summary = json.loads(self_summary_path.read_text(encoding="utf-8"))
        public_summary = json.loads(public_summary_path.read_text(encoding="utf-8"))
        spectrum_summary = json.loads(spectrum_summary_path.read_text(encoding="utf-8")) if spectrum_summary_path.exists() else {}
        rows.append({
            "sampling": sampling,
            "self_selected_normals": self_summary.get("selected_normal_count"),
            "public_selected_normals": public_summary.get("selected_normal_count"),
            "spectrum_gallery_patches": spectrum_summary.get("gallery_patch_counts"),
            "public_gallery_patches": public_summary.get("gallery_patch_count"),
            "self_text_auc": self_summary.get("text_auc_macro"),
            "self_vit_auc": self_summary.get("vit_map_max_auc_macro"),
            "self_text_vit_baseline_auc": self_summary.get("harmonic_text_vit_max_auc_macro"),
            "public_text_auc": public_summary.get("text_auc_macro"),
            "public_vit_auc": public_summary.get("vit_max_auc_macro"),
            "public_text_vit_baseline_auc": public_summary.get("text_vit_max_auc_macro"),
            "spectrum_text_auc": spectrum_summary.get("text_auc_macro", ""),
            "spectrum_vit_auc": spectrum_summary.get("vit_max_auc_macro", ""),
            "spectrum_text_vit_baseline_auc": spectrum_summary.get("text_vit_max_auc_macro", ""),
        })
    if not rows:
        return
    out_csv = ROOT / "baseline_sampling_shot_ablation_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Generic PromptAD Baseline Sampling Ablation",
        "",
        "This run evaluates the baseline only, not the current dual-visual method.",
        "",
        "Baseline score:",
        "",
        "```text",
        "generic text_score + ViT patch anomaly map",
        "= harmonic(generic text_score, max(ViT patch anomaly map))",
        "```",
        "",
        "All jobs explicitly use `--prompt-mode generic`.",
        "",
        "## Macro Image-AUROC",
        "",
        "| sampling | self baseline | public baseline | spectrum baseline |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        spectrum_auc = row["spectrum_text_vit_baseline_auc"]
        spectrum_text = f"{float(spectrum_auc):.4f}" if spectrum_auc != "" else "N/A"
        lines.append(
            f"| {row['sampling']} | "
            f"{float(row['self_text_vit_baseline_auc']):.4f} | "
            f"{float(row['public_text_vit_baseline_auc']):.4f} | "
            f"{spectrum_text} |"
        )
    lines.extend([
        "",
        "## Files",
        "",
        "- `baseline_sampling_shot_ablation_summary.csv` - compact summary.",
        "- `{sampling}/self_baseline/summary.json` - self RF baseline metrics.",
        "- `{sampling}/public_baseline/summary.json` - public RF baseline metrics.",
        "- `{sampling}/spectrum_baseline/summary.json` - Spectrum baseline metrics.",
        "- `{sampling}/*.log` - job logs.",
    ])
    (ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    scheduler_log = ROOT / "scheduler.log"
    jobs = build_jobs()
    q: Queue[tuple[str, list[str]]] = Queue()
    for role, sampling, cmd in jobs:
        name = f"{sampling}/{role}"
        if marker_path(name).exists():
            with scheduler_log.open("a", encoding="utf-8") as sched:
                sched.write(f"[{time.strftime('%F %T')}] SKIP {name}: marker exists\n")
            continue
        q.put((name, cmd))

    failures: list[tuple[str, int]] = []
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                name, cmd = q.get_nowait()
            except Exception:
                return
            result_name, code = run_job(name, cmd, gpu, scheduler_log)
            if code != 0:
                with lock:
                    failures.append((result_name, code))
            q.task_done()

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
