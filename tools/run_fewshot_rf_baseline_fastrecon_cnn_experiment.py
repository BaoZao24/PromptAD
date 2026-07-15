#!/usr/bin/env python
"""Run the full five-class few-shot RF CLS comparison pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, cmd: list[str], log_path: Path, dry_run: bool = False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(cmd)
    print(f"\n[{name}] {rendered}")
    if dry_run:
        return
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {name}\n")
        log.write(f"# {datetime.now().isoformat(timespec='seconds')}\n")
        log.write(rendered + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}. See {log_path}")


def maybe_run(name: str, output_file: Path, cmd: list[str], log_path: Path, args):
    if output_file.exists() and not args.force:
        print(f"[skip {name}] exists: {output_file}")
        return
    run_step(name, cmd, log_path, args.dry_run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="analysis_outputs/20260704_fewshot_rf_wideband_baseline_fastrecon_cnn")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--normal-sampling", choices=["all", "frequency_one_per_band"], default="frequency_one_per_band")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fastrecon-batch-size", type=int, default=32)
    parser.add_argument("--fastrecon-max-support-patches", type=int, default=10000)
    parser.add_argument("--fastrecon-max-coreset-size", type=int, default=256)
    parser.add_argument("--fastrecon-coreset-ratio", type=float, default=0.05)
    parser.add_argument("--patchcore-batch-size", type=int, default=32)
    parser.add_argument("--patchcore-sampler", choices=["identity", "random", "greedy_coreset", "approx_greedy_coreset"], default="random")
    parser.add_argument("--patchcore-coreset-percentage", type=float, default=0.1)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--cnn-key", default="cnn_layer2_guided_by_vit_top0p01")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output_root)
    baseline_root = out_root / "baseline_vit_patch_gallery"
    cnn_root = out_root / "cnn_guided_gallery"
    fastrecon_root = out_root / "fastrecon"
    patchcore_root = out_root / "patchcore"
    compare_root = out_root / "comparison"
    log_root = out_root / "logs"

    common = [
        "--gpu-id", str(args.gpu_id),
        "--seed", str(args.seed),
        "--normal-sampling", args.normal_sampling,
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
    ]

    baseline_cmd = [
        sys.executable, "tools/eval_cls_vit_patch_gallery.py",
        "--output-root", str(baseline_root),
        *common,
    ]
    if args.checkpoint:
        baseline_cmd.extend(["--checkpoint", args.checkpoint])

    cnn_cmd = [
        sys.executable, "tools/eval_cls_vit_guided_cnn_gallery.py",
        "--output-root", str(cnn_root),
        "--resnet-layers", "layer2",
        "--guided-ratios", "0.01",
        "--betas", *(str(v) for v in args.betas),
        *common,
    ]
    if args.checkpoint:
        cnn_cmd.extend(["--checkpoint", args.checkpoint])

    fastrecon_cmd = [
        sys.executable, "tools/eval_fastrecon_cls.py",
        "--protocol", "rf_target",
        "--rf-train-mode", "per_scene",
        "--output-root", str(fastrecon_root),
        "--gpu-id", str(args.gpu_id),
        "--seed", str(args.seed),
        "--normal-sampling", args.normal_sampling,
        "--batch-size", str(args.fastrecon_batch_size),
        "--num-workers", str(args.num_workers),
        "--lambdas", "0.0",
        "--primary-lam", "0.0",
        "--coreset-ratio", str(args.fastrecon_coreset_ratio),
        "--max-support-patches", str(args.fastrecon_max_support_patches),
        "--max-coreset-size", str(args.fastrecon_max_coreset_size),
    ]

    patchcore_cmd = [
        sys.executable, "tools/eval_patchcore_cls.py",
        "--protocol", "rf_target",
        "--rf-train-mode", "pooled",
        "--output-root", str(patchcore_root),
        "--gpu-id", str(args.gpu_id),
        "--seed", str(args.seed),
        "--normal-sampling", args.normal_sampling,
        "--batch-size", str(args.patchcore_batch_size),
        "--num-workers", str(args.num_workers),
        "--sampler", args.patchcore_sampler,
        "--coreset-percentage", str(args.patchcore_coreset_percentage),
    ]

    compare_cmd = [
        sys.executable, "tools/compare_fewshot_baseline_fastrecon_cnn.py",
        "--baseline-score-dir", str(baseline_root / "scores"),
        "--fastrecon-score-dir", str(fastrecon_root / "scores"),
        "--cnn-score-dir", str(cnn_root / "scores"),
        "--patchcore-score-dir", str(patchcore_root / "scores"),
        "--output-root", str(compare_root),
        "--cnn-key", args.cnn_key,
        "--alphas", *(str(v) for v in args.alphas),
        "--betas", *(str(v) for v in args.betas),
    ]

    maybe_run("baseline", baseline_root / "results_cls_vit_patch_gallery.csv", baseline_cmd, log_root / "baseline.log", args)
    maybe_run("cnn", cnn_root / "results_cls_vit_guided_cnn_gallery.csv", cnn_cmd, log_root / "cnn.log", args)
    maybe_run("fastrecon", fastrecon_root / "results_fastrecon_cls.csv", fastrecon_cmd, log_root / "fastrecon.log", args)
    maybe_run("patchcore", patchcore_root / "results_patchcore_cls.csv", patchcore_cmd, log_root / "patchcore.log", args)
    maybe_run("compare", compare_root / "summary.json", compare_cmd, log_root / "compare.log", args)

    manifest = {
        "output_root": str(out_root),
        "baseline_root": str(baseline_root),
        "cnn_root": str(cnn_root),
        "fastrecon_root": str(fastrecon_root),
        "patchcore_root": str(patchcore_root),
        "comparison_root": str(compare_root),
        "normal_sampling": args.normal_sampling,
        "seed": args.seed,
        "gpu_id": args.gpu_id,
        "checkpoint": args.checkpoint,
        "commands": {
            "baseline": baseline_cmd,
            "cnn": cnn_cmd,
            "fastrecon": fastrecon_cmd,
            "patchcore": patchcore_cmd,
            "compare": compare_cmd,
        },
    }
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
