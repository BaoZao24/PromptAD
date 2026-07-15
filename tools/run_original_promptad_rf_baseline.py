#!/usr/bin/env python
"""Run the unmodified upstream PromptAD code on RF data via an MVTec-style view."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "references" / "PromptAD-original"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_cls_vit_patch_gallery import collect_pooled_train_samples
from train_rf_target_pooled_universal import JSR_BY_SIGNAL, SCENES, SIGNALS, collect_samples


def link_or_copy(src: Path, dst: Path, copy_files: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def reset_dir(path: Path):
    if path.exists() or path.is_symlink():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare_seed_file(reference_root: Path, class_name: str, k_shot: int):
    seed_dir = reference_root / "datasets" / "seeds_mvtec" / class_name
    seed_dir.mkdir(parents=True, exist_ok=True)
    indices = " ".join(str(i) for i in range(k_shot))
    (seed_dir / "selected_samples_per_run.txt").write_text(f"#{k_shot}: {indices}\n", encoding="utf-8")


def prepare_mvtec_view(reference_root: Path, class_name: str, train_samples, test_samples, copy_files: bool):
    class_root = reference_root / "anomaly_detection" / "mvtec_anomaly_detection" / class_name
    reset_dir(class_root)
    train_good = class_root / "train" / "good"
    test_good = class_root / "test" / "good"
    test_bad = class_root / "test" / "bad"
    gt_bad = class_root / "ground_truth" / "bad"
    for folder in [train_good, test_good, test_bad, gt_bad]:
        folder.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(train_samples):
        img_path = Path(sample[0])
        link_or_copy(img_path, train_good / f"train_good_{idx:04d}.png", copy_files)

    normal_i = 0
    abnormal_i = 0
    for sample in test_samples:
        img_path = Path(sample[0])
        gt_path = sample[1]
        label = int(sample[2])
        if label == 0:
            link_or_copy(img_path, test_good / f"test_good_{normal_i:04d}.png", copy_files)
            normal_i += 1
        else:
            stem = f"test_bad_{abnormal_i:04d}"
            link_or_copy(img_path, test_bad / f"{stem}.png", copy_files)
            if gt_path and Path(gt_path).exists():
                link_or_copy(Path(gt_path), gt_bad / f"{stem}_mask.png", copy_files)
            abnormal_i += 1

    return {
        "num_train": len(train_samples),
        "num_test_normal": normal_i,
        "num_test_abnormal": abnormal_i,
    }


def build_jobs(args, support_paths):
    jobs = []
    for signal in args.signals:
        for scene in SCENES:
            for jsr in JSR_BY_SIGNAL[signal]:
                samples = collect_samples(signal, scene, jsr, "test", k_shot=args.rf_k_shot, exclude_paths=support_paths)
                jobs.append({"signal": signal, "scene": scene, "jsr": jsr, "test_samples": samples})
    return jobs


def read_original_metric(csv_root: Path, class_name: str):
    csv_files = sorted(csv_root.rglob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in csv_files:
        try:
            df = pd.read_csv(path, index_col=0)
        except Exception:
            continue
        row_name = f"mvtec-{class_name}"
        if row_name in df.index and "i_roc" in df.columns:
            return float(df.loc[row_name, "i_roc"]), str(path)
        if class_name in df.index and "i_roc" in df.columns:
            return float(df.loc[class_name, "i_roc"]), str(path)
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            return float(numeric.iloc[-1, -1]), str(path)
    raise RuntimeError(f"Could not find metric CSV under {csv_root}")


def run_one_job(args, train_samples, support_paths, job, row_index):
    class_name = args.original_class_name
    prepare_seed_file(REFERENCE_ROOT, class_name, len(train_samples))
    view_info = prepare_mvtec_view(
        REFERENCE_ROOT,
        class_name,
        train_samples,
        job["test_samples"],
        copy_files=args.copy_files,
    )

    run_root = Path(args.output_root) / "original_runs" / f"{row_index:03d}_{job['signal']}_{job['scene']}_{job['jsr']}"
    if run_root.exists() and not args.force:
        summary_path = run_root / "summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
    run_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "train_cls.py",
        "--dataset", "mvtec",
        "--class_name", class_name,
        "--k-shot", str(len(train_samples)),
        "--Epoch", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--gpu-id", str(args.gpu_id),
        "--seed", str(args.seed),
        "--root-dir", str(run_root.resolve()),
        "--vis", "False",
        "--cal-pro", "False",
    ]
    log_path = run_root / "run.log"
    with log_path.open("w", encoding="utf-8") as log_f:
        env = os.environ.copy()
        env.setdefault("MKL_THREADING_LAYER", "GNU")
        proc = subprocess.run(
            cmd,
            cwd=REFERENCE_ROOT,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Original PromptAD failed for {job}; see {log_path}")

    score, source_csv = read_original_metric(run_root, class_name)
    summary = {
        "method": "original_promptad_unmodified_source",
        "dataset": job["signal"],
        "scene": job["scene"],
        "jsr": job["jsr"],
        "i_roc": score,
        "epochs": args.epochs,
        "support_count": len(train_samples),
        "original_class_name": class_name,
        "reference_source": str(REFERENCE_ROOT),
        "source_csv": source_csv,
        "log": str(log_path),
        **view_info,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="analysis_outputs/20260704_original_promptad_rf_baseline")
    parser.add_argument("--signals", nargs="+", default=SIGNALS, choices=SIGNALS)
    parser.add_argument("--normal-sampling", choices=["all", "first", "frequency_one_per_band"], default="frequency_one_per_band")
    parser.add_argument("--rf-k-shot", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--original-class-name", default="carpet")
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--copy-files", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not REFERENCE_ROOT.exists():
        raise FileNotFoundError(f"Missing upstream source: {REFERENCE_ROOT}")

    train_args = argparse.Namespace(signals=SIGNALS, normal_sampling=args.normal_sampling)
    train_samples = collect_pooled_train_samples(train_args)
    support_paths = {sample[0] for sample in train_samples}
    jobs = build_jobs(args, support_paths)
    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]

    rows = []
    for idx, job in enumerate(jobs):
        print(f"[{idx + 1}/{len(jobs)}] {job['signal']} {job['scene']} {job['jsr']}", flush=True)
        rows.append(run_one_job(args, train_samples, support_paths, job, idx))

    out_root = Path(args.output_root)
    result_path = out_root / "results_original_promptad_rf_cls.csv"
    write_csv(result_path, rows)
    df = pd.DataFrame(rows)
    by_signal = df.groupby("dataset", as_index=False)["i_roc"].mean()
    by_signal.to_csv(out_root / "by_signal_original_promptad_rf_cls.csv", index=False)
    summary = {
        "method": "original_promptad_unmodified_source",
        "reference_source": str(REFERENCE_ROOT),
        "epochs": args.epochs,
        "normal_sampling": args.normal_sampling,
        "support_count": len(train_samples),
        "num_cells": len(rows),
        "macro_i_roc": float(df["i_roc"].mean()),
        "by_signal": dict(zip(by_signal["dataset"], by_signal["i_roc"])),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    readme = [
        "# Original PromptAD RF Baseline",
        "",
        "Upstream source is kept unmodified under `references/PromptAD-original`.",
        "RF data is exposed to it through a temporary MVTec-style `carpet` data view.",
        "",
        f"- epochs: `{args.epochs}`",
        f"- normal sampling: `{args.normal_sampling}`",
        f"- support count: `{len(train_samples)}`",
        f"- cells: `{len(rows)}`",
        f"- macro Image-AUROC: `{summary['macro_i_roc']:.4f}`",
        "",
        "## By Signal",
        "",
        "| signal | Image-AUROC |",
        "|---|---:|",
    ]
    for _, row in by_signal.iterrows():
        readme.append(f"| {row['dataset']} | {float(row['i_roc']):.4f} |")
    (out_root / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
