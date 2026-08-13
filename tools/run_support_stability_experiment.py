#!/usr/bin/env python3
"""Run bounded, independently seeded support-repetition experiments.

Every replicate has its own support manifest/reference directory.  The RF
fusion stage and the OFDMA test stage consume only that replicate's support
reference, so no calibration statistics cross support seeds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / (
    "analysis_outputs/02_current_baselines/promptad_formal_baseline/"
    "pooled_rf_rgb_cls/checkpoint/overall-best.pt"
)
OFDMA_ROOT = Path("/mnt/data/wangbei/data/ofdma-target-scene-coldstart-v2-realistic")


def run(name: str, command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed; inspect {log_path}")


def python_command(*parts: str) -> list[str]:
    return ["python", *parts]


def run_rf(seed: int, replicate: Path, gpu_id: int, logs: Path) -> None:
    rf_root = replicate / "rf"
    manifest = rf_root / "self_support_manifest.json"
    public_manifest = rf_root / "public_support_manifest.json"
    self_vit = rf_root / "self_vit"
    self_vit_ref = rf_root / "self_vit_reference"
    self_cnn = rf_root / "self_cnn"
    self_cnn_ref = rf_root / "self_cnn_reference"
    public_vit = rf_root / "public_vit"
    public_vit_ref = rf_root / "public_vit_reference"
    public_cnn = rf_root / "public_cnn"
    public_cnn_ref = rf_root / "public_cnn_reference"

    run(
        "build self RF support manifest",
        python_command(
            "tools/build_rf_target_scene_support_manifest.py",
            "--output",
            str(manifest),
            "--normal-sampling",
            "per_frequency",
            "--seed",
            str(seed),
        ),
        logs / "rf-self-manifest.log",
    )
    run(
        "build public RF support manifest",
        python_command(
            "tools/build_public_rf_support_manifest.py",
            "--output",
            str(public_manifest),
            "--normal-sampling",
            "per_frequency",
            "--seed",
            str(seed),
        ),
        logs / "rf-public-manifest.log",
    )

    vit_common = [
        "--batch-size",
        "256",
        "--normal-sampling",
        "per_frequency",
        "--coreset-method",
        "farthest",
        "--coreset-ratio",
        "0.5",
        "--nn-topk",
        "5",
        "--nn-agg",
        "mean",
        "--paired-tta",
        "none",
        "--paired-tta-fusion",
        "max",
        "--paired-tta-shift-px",
        "4",
        "--gpu-id",
        str(gpu_id),
        "--seed",
        "111",
    ]
    run(
        "self RF ViT test",
        python_command(
            "tools/eval_cls_vit_patchcore_gallery.py",
            "--output-root",
            str(self_vit),
        "--support-manifest",
        str(manifest),
        "--support-seed",
        str(seed),
        *vit_common,
        ),
        logs / "rf-self-vit-test.log",
    )
    run(
        "self RF ViT support reference",
        python_command(
            "tools/eval_cls_vit_patchcore_gallery.py",
            "--output-root",
            str(self_vit_ref),
            "--support-manifest",
            str(manifest),
            "--support-seed",
            str(seed),
            "--support-reference-only",
            *vit_common,
        ),
        logs / "rf-self-vit-reference.log",
    )
    cnn_common = [
        "--protocol",
        "rf_target",
        "--normal-sampling",
        "per_frequency",
        "--support-manifest",
        str(manifest),
        "--support-seed",
        str(seed),
        "--gpu-id",
        str(gpu_id),
        "--seed",
        "111",
        "--batch-size",
        "512",
    ]
    run(
        "self RF CNN test",
        python_command(
            "tools/eval_cls_aux_cnn_gallery.py",
            "--output-root",
            str(self_cnn),
            *cnn_common,
        ),
        logs / "rf-self-cnn-test.log",
    )
    run(
        "self RF CNN support reference",
        python_command(
            "tools/eval_cls_aux_cnn_gallery.py",
            "--output-root",
            str(self_cnn_ref),
            "--support-reference-only",
            *cnn_common,
        ),
        logs / "rf-self-cnn-reference.log",
    )

    public_vit_common = [
        "--batch-size",
        "256",
        "--normal-sampling",
        "per_frequency",
        "--support-manifest",
        str(public_manifest),
        "--support-seed",
        str(seed),
        "--coreset-method",
        "farthest",
        "--coreset-ratio",
        "0.5",
        "--nn-topk",
        "5",
        "--nn-agg",
        "mean",
        "--paired-tta",
        "none",
        "--paired-tta-fusion",
        "max",
        "--paired-tta-shift-px",
        "4",
        "--checkpoint",
        str(CHECKPOINT),
        "--gpu-id",
        str(gpu_id),
        "--seed",
        "111",
    ]
    run(
        "public RF ViT test",
        python_command(
            "tools/eval_cls_public_rf_vit_patchcore_gallery.py",
            "--output-root",
            str(public_vit),
            *public_vit_common,
        ),
        logs / "rf-public-vit-test.log",
    )
    run(
        "public RF ViT support reference",
        python_command(
            "tools/eval_cls_public_rf_vit_patchcore_gallery.py",
            "--output-root",
            str(public_vit_ref),
            "--support-reference-only",
            *public_vit_common,
        ),
        logs / "rf-public-vit-reference.log",
    )
    public_cnn_common = [
        "--protocol",
        "public_rf",
        "--normal-sampling",
        "per_frequency",
        "--support-manifest",
        str(public_manifest),
        "--support-seed",
        str(seed),
        "--gpu-id",
        str(gpu_id),
        "--seed",
        "111",
        "--batch-size",
        "512",
    ]
    run(
        "public RF CNN test",
        python_command(
            "tools/eval_cls_aux_cnn_gallery.py",
            "--output-root",
            str(public_cnn),
            *public_cnn_common,
        ),
        logs / "rf-public-cnn-test.log",
    )
    run(
        "public RF CNN support reference",
        python_command(
            "tools/eval_cls_aux_cnn_gallery.py",
            "--output-root",
            str(public_cnn_ref),
            "--support-reference-only",
            *public_cnn_common,
        ),
        logs / "rf-public-cnn-reference.log",
    )

    run(
        "self RF safe fusion",
        python_command(
            "tools/eval_cls_dual_visual_evidence_fusion.py",
            "--protocol",
            "rf_target",
            "--vit-score-dir",
            str(self_vit / "scores"),
            "--cnn-score-dir",
            str(self_cnn / "scores"),
            "--vit-reference-dir",
            str(self_vit_ref),
            "--cnn-reference-dir",
            str(self_cnn_ref),
            "--output-root",
            str(rf_root / "self_fusion"),
            "--gate-protocol",
            "support_only",
        ),
        logs / "rf-self-fusion.log",
    )
    run(
        "public RF safe fusion",
        python_command(
            "tools/eval_cls_dual_visual_evidence_fusion.py",
            "--protocol",
            "public_rf",
            "--vit-score-dir",
            str(public_vit / "scores"),
            "--cnn-score-dir",
            str(public_cnn / "scores"),
            "--vit-reference-dir",
            str(public_vit_ref),
            "--cnn-reference-dir",
            str(public_cnn_ref),
            "--support-manifest",
            str(public_manifest),
            "--output-root",
            str(rf_root / "public_fusion"),
            "--gate-protocol",
            "support_only",
        ),
        logs / "rf-public-fusion.log",
    )


def run_ofdma(
    seed: int,
    replicate: Path,
    gpu_id: int,
    logs: Path,
    dataset_root: Path,
) -> None:
    ofdma_root = replicate / "ofdma"
    reference = ofdma_root / "reference"
    test = ofdma_root / "test"
    support_manifest = ofdma_root / "support_manifest.json"
    run(
        "build OFDMA support manifest",
        python_command(
            "tools/build_ofdma_support_manifest.py",
            "--dataset-root",
            str(dataset_root),
            "--output",
            str(support_manifest),
            "--split",
            "test",
            "--seed",
            str(seed),
            "--shots",
            "1",
            "2",
            "4",
        ),
        logs / "ofdma-manifest.log",
    )
    common = [
        "--dataset-root",
        str(dataset_root),
        "--split",
        "test",
        "--shots",
        "1",
        "2",
        "4",
        "--support-seed",
        str(seed),
        "--gate-protocol",
        "support_only",
        "--gpu-id",
        str(gpu_id),
        "--batch-size",
        "128",
        "--num-workers",
        "4",
    ]
    run(
        "OFDMA support reference",
        python_command(
            "tools/eval_cls_ofdma_target_scene_ours.py",
            "--output-root",
            str(reference),
            "--support-reference-only",
            *common,
        ),
        logs / "ofdma-reference.log",
    )
    run(
        "OFDMA test",
        python_command(
            "tools/eval_cls_ofdma_target_scene_ours.py",
            "--output-root",
            str(test),
            "--support-reference-root",
            str(reference / "support_reference"),
            *common,
        ),
        logs / "ofdma-test.log",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "autoresearch" / ("support-stability-" + time.strftime("%y%m%d-%H%M%S")),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[111, 222, 333, 444, 555])
    parser.add_argument("--datasets", nargs="+", choices=["rf", "ofdma"], default=["rf", "ofdma"])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--ofdma-dataset-root", type=Path, default=OFDMA_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add one or more seeds to an existing run root (useful for parallel jobs).",
    )
    args = parser.parse_args()

    if args.output_root.exists():
        if not args.append and not args.resume:
            raise FileExistsError(
                f"Output root already exists: {args.output_root}; use --append or --resume"
            )
    else:
        args.output_root.mkdir(parents=True, exist_ok=False)
    config = {
        "scope": "support stability under independent normal-support resampling",
        "seeds": [int(seed) for seed in args.seeds],
        "datasets": args.datasets,
        "rf_protocol": "self target-scene per-frequency; public seeded per-frequency",
        "ofdma_protocol": "target-scene seeded complete-observation selection, shots 1/2/4",
        "model_seed": 111,
        "public_rf_test_split": "fixed_records_outside_support_pool",
        "test_labels_used_for_scoring": False,
        "resume": bool(args.resume),
    }
    config_path = args.output_root / "config.json"
    if config_path.exists() and args.append:
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        config["seeds"] = sorted(
            {
                *[int(value) for value in existing.get("seeds", [])],
                *[int(value) for value in args.seeds],
            }
        )
        config["datasets"] = sorted(
            {*existing.get("datasets", []), *args.datasets}
        )
    if not config_path.exists() or args.append:
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    for seed in args.seeds:
        replicate = args.output_root / "replicates" / f"seed_{int(seed):04d}"
        replicate.mkdir(parents=True, exist_ok=True)
        logs = args.output_root / "logs" / f"seed_{int(seed):04d}"
        marker = replicate / "complete.json"
        if args.resume and marker.is_file():
            continue
        failed_marker = replicate / "failed.json"
        if failed_marker.exists():
            failed_marker.unlink()
        try:
            if "rf" in args.datasets:
                run_rf(int(seed), replicate, args.gpu_id, logs)
            if "ofdma" in args.datasets:
                run_ofdma(int(seed), replicate, args.gpu_id, logs, args.ofdma_dataset_root)
        except Exception as exc:
            (replicate / "failed.json").write_text(
                json.dumps({"seed": int(seed), "error": str(exc)}, indent=2) + "\n",
                encoding="utf-8",
            )
            raise
        marker.write_text(
            json.dumps({"seed": int(seed), "datasets": args.datasets}, indent=2) + "\n",
            encoding="utf-8",
        )
    completed_seeds = sorted(
        int(path.name.removeprefix("seed_"))
        for path in (args.output_root / "replicates").glob("seed_*")
        if (path / "complete.json").is_file()
    )
    handoff = {
        "version": "1.0.0",
        "source": "autoresearch",
        "status": "COMPLETE_EXPERIMENT",
        "run_root": str(args.output_root.resolve()),
        "seeds": completed_seeds,
        "datasets": args.datasets,
        "test_split_used_for_selection": False,
        "decision": "summarize with tools/summarize_support_stability.py",
    }
    (args.output_root / "handoff.json").write_text(
        json.dumps(handoff, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(handoff, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
