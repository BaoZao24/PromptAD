#!/usr/bin/env python
"""Run the pure ViT-only module ablation with a ViT-B-32 backbone.

This is an independent rerun of the historical ``no_all`` branch.  It keeps
the branch protocol fixed and writes to a new timestamped output root so that
the previous ViT-B-16-plus-240 results remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path("analysis_outputs/20260824_vit_b32_module_ablation")
PUB_CKPT = Path(
    "analysis_outputs/02_current_baselines/promptad_formal_baseline/"
    "pooled_rf_rgb_cls/checkpoint/overall-best.pt"
)

DATASETS = {
    "self_rf": {
        "sampling": "per_frequency",
        "script": "tools/eval_cls_vit_patchcore_gallery.py",
    },
    "public_rf": {
        "sampling": "per_frequency",
        "script": "tools/eval_cls_public_rf_vit_patchcore_gallery.py",
    },
    "spectrum": {
        "sampling": "4shot",
        "script": "tools/eval_cls_spectrum_vit_nn_gallery.py",
    },
}


def output_dir(dataset: str) -> Path:
    return ROOT / "no_all" / DATASETS[dataset]["sampling"] / dataset / "vit"


def command(dataset: str, gpu: int) -> list[str]:
    cfg = DATASETS[dataset]
    cmd = [
        "python",
        cfg["script"],
        "--output-root",
        str(output_dir(dataset)),
        "--normal-sampling",
        cfg["sampling"],
        "--coreset-method",
        "random",
        "--coreset-ratio",
        "1.0",
        "--nn-topk",
        "1",
        "--nn-agg",
        "mean",
        "--paired-tta",
        "none",
        "--paired-tta-fusion",
        "max",
        "--backbone",
        "ViT-B-32",
        "--pretrained_dataset",
        "laion400m_e32",
        "--img-resize",
        "240",
        "--img-cropsize",
        "240",
        "--gpu-id",
        str(gpu),
        "--seed",
        "111",
    ]
    if dataset == "public_rf":
        cmd.extend(["--checkpoint", str(PUB_CKPT)])
    return cmd


def run_one(dataset: str, gpu: int) -> None:
    out = output_dir(dataset)
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "summary.json"
    log_path = out / "run.log"
    if summary.exists():
        print(f"[skip] {dataset}: {summary}", flush=True)
        return
    cmd = command(dataset, gpu)
    print(f"[start] {dataset} gpu={gpu}: {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    print(f"[done] {dataset} returncode={proc.returncode}", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{dataset} failed; see {log_path}")


def auc_from_summary(dataset: str, summary: dict) -> float:
    key = "clip_vit_nn_max_auc_macro" if dataset == "spectrum" else "vit_patchcore_max_auc_macro"
    return float(summary[key])


def write_summary() -> None:
    rows = []
    for dataset in DATASETS:
        summary_path = output_dir(dataset) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": dataset,
                "variant": "no_all_vit_b32",
                "backbone": summary.get("backbone", "ViT-B-32"),
                "pretrained_dataset": summary.get("pretrained_dataset", "laion400m_e32"),
                "normal_sampling": summary["normal_sampling"],
                "paired_tta": summary["paired_tta"],
                "coreset_method": summary["coreset_method"],
                "coreset_ratio": summary["coreset_ratio"],
                "nn_topk": summary["nn_topk"],
                "vit_only_auc": auc_from_summary(dataset, summary),
            }
        )
    out_csv = ROOT / "vit_only_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "README.md").write_text(
        "\n".join(
            [
                "# ViT-only module ablation: ViT-B-32",
                "",
                "This reruns the historical `no_all` branch with `ViT-B-32` and `laion400m_e32`.",
                "",
                "- no paired TTA (`paired_tta=none`)",
                "- full random normal gallery (`coreset_ratio=1.0`)",
                "- top-1 nearest-neighbour scoring (`nn_topk=1`)",
                "- original 240x240 image preprocessing",
                "",
                "See `vit_only_summary.csv` for the macro image-AUROC values and each dataset's `summary.json` for the full protocol.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[summary] wrote {out_csv}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()
    if len(args.gpus) < len(DATASETS):
        raise ValueError("provide at least three GPU ids for the three dataset jobs")
    if not PUB_CKPT.is_file():
        raise FileNotFoundError(f"Public RF checkpoint not found: {PUB_CKPT}")
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(DATASETS)) as executor:
        futures = [
            executor.submit(run_one, dataset, args.gpus[index])
            for index, dataset in enumerate(DATASETS)
        ]
        for future in futures:
            future.result()
    write_summary()


if __name__ == "__main__":
    main()
