#!/usr/bin/env python
"""Run coreset/top-k ablations for the independent PatchCore baseline."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue


GPUS = (0, 1, 3)
ROOT = Path("analysis_outputs/20260725_patchcore_baseline_ablation")
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
DATASETS = {
    "self_rf": ("rf_target", "per_frequency"),
    "public_rf": ("public_rf", "per_frequency"),
    "spectrum": ("spectrum", "4shot"),
}


def command(
    protocol: str,
    sampling: str,
    output_root: Path,
    variant: dict[str, str],
) -> list[str]:
    result = [
        "python",
        "tools/eval_patchcore_cls.py",
        "--protocol",
        protocol,
        "--output-root",
        str(output_root),
        "--normal-sampling",
        sampling,
        "--batch-size",
        "32",
        "--sampler",
        variant["sampler"],
        "--coreset-percentage",
        variant["coreset_percentage"],
        "--anomaly-scorer-num-nn",
        variant["topk"],
        "--gpu-id",
        "%GPU%",
    ]
    if protocol == "rf_target":
        result.extend(
            [
                "--support-manifest",
                str(ROOT / "support_manifest.json"),
            ]
        )
    return result


def build_jobs():
    jobs = []
    for variant_name, variant in VARIANTS.items():
        for dataset, (protocol, sampling) in DATASETS.items():
            output_root = ROOT / variant_name / sampling / dataset
            jobs.append(
                (
                    f"{variant_name}/{dataset}",
                    command(protocol, sampling, output_root, variant),
                    output_root,
                )
            )
    return jobs


def run_job(
    name: str,
    command_template: list[str],
    output_root: Path,
    gpu_queue: Queue,
    log_lock: threading.Lock,
) -> tuple[str, int]:
    marker = output_root / "summary.json"
    if marker.exists():
        return name, 0

    gpu = gpu_queue.get()
    try:
        command_line = [
            str(gpu) if part == "%GPU%" else part
            for part in command_template
        ]
        output_root.mkdir(parents=True, exist_ok=True)
        with log_lock:
            print(
                f"[{time.strftime('%F %T')}] START gpu={gpu} {name}",
                flush=True,
            )
        with (output_root / "run.log").open(
            "w",
            encoding="utf-8",
        ) as handle:
            process = subprocess.run(
                command_line,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        with log_lock:
            print(
                f"[{time.strftime('%F %T')}] DONE {name} "
                f"returncode={process.returncode}",
                flush=True,
            )
        return name, process.returncode
    finally:
        gpu_queue.put(gpu)


def summarize() -> None:
    rows = []
    for variant_name, variant in VARIANTS.items():
        for dataset, (_protocol, sampling) in DATASETS.items():
            output_root = ROOT / variant_name / sampling / dataset
            summary_path = output_root / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
            rows.append(
                {
                    "variant": variant_name,
                    "dataset": dataset,
                    "sampling": sampling,
                    "sampler": variant["sampler"],
                    "coreset_percentage": variant[
                        "coreset_percentage"
                    ],
                    "nearest_neighbours": variant["topk"],
                    "patchcore_image_auroc": summary[
                        "image_auroc_macro"
                    ],
                    "output_root": str(output_root),
                }
            )
    if not rows:
        return

    summary_path = ROOT / "patchcore_coreset_topk_summary.csv"
    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# PatchCore Baseline Coreset/Top-k Ablation",
        "",
        "This evaluates the independent PatchCore baseline only. "
        "Its scores are not inputs to the confidence gate.",
        "",
        "| variant | dataset | sampling | PatchCore Image-AUROC |",
        "|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['dataset']} | "
            f"{row['sampling']} | "
            f"{float(row['patchcore_image_auroc']):.4f} |"
        )
    lines.extend(["", f"CSV: `{summary_path.name}`"])
    (ROOT / "README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    gpu_queue: Queue = Queue()
    for gpu in GPUS:
        gpu_queue.put(gpu)
    log_lock = threading.Lock()
    failed = []

    with ThreadPoolExecutor(max_workers=len(GPUS)) as executor:
        futures = [
            executor.submit(
                run_job,
                name,
                command_line,
                output_root,
                gpu_queue,
                log_lock,
            )
            for name, command_line, output_root in build_jobs()
        ]
        for future in as_completed(futures):
            name, returncode = future.result()
            if returncode != 0:
                failed.append((name, returncode))

    summarize()
    if failed:
        print(json.dumps({"failed": failed}, indent=2), file=sys.stderr)
        return 1
    print(f"wrote {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
