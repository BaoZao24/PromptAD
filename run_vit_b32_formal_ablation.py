#!/usr/bin/env python
"""Run the formal ViT-only ablation with a ViT-B-32 visual backbone.

The commands deliberately mirror the four-dataset protocol described in
``docs/paper/现有方案介绍.md``.  Only the ViT backbone changes relative to
the recorded formal settings; the CNN branch is still evaluated by the
OFDMA/FedJam dual-visual entry points so that their frozen branch diagnostics
remain directly comparable, while the final summary selects ``ours_vit``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
CHECKPOINT = REPO_ROOT / (
    "analysis_outputs/02_current_baselines/promptad_formal_baseline/"
    "pooled_rf_rgb_cls/checkpoint/overall-best.pt"
)
OUTPUT_NAME = "20260824_vit_b32_formal_ablation"


def command_specs(output_root: Path, gpu_id: int) -> dict[str, list[str]]:
    common = [
        "--backbone",
        "ViT-B-32",
        "--seed",
        "111",
    ]
    rf_tta = [
        "--paired-tta",
        "rf_spectral_response_v1",
        "--paired-tta-fusion",
        "max",
        "--paired-tta-memory-layout",
        "merged",
        "--paired-tta-shift-px",
        "4",
        "--paired-tta-frequency-response-strength",
        "3",
    ]
    public_tta = [
        "--paired-tta",
        "rf_spectral_response_v1",
        "--paired-tta-fusion",
        "max",
        "--paired-tta-memory-layout",
        "merged",
        "--paired-tta-shift-px",
        "4",
        "--paired-tta-frequency-response-strength",
        "3",
    ]
    checkpoint = str(CHECKPOINT)
    specs: dict[str, list[str]] = {
        "inhouse_rf": [
            sys.executable,
            "tools/eval_cls_vit_patchcore_gallery.py",
            "--output-root",
            str(output_root / "inhouse_rf"),
            "--support-manifest",
            "analysis_outputs/20260728_rf_five_type_formal_seed111/support_manifest.json",
            "--normal-sampling",
            "per_frequency",
            "--checkpoint",
            checkpoint,
            "--coreset-method",
            "farthest",
            "--coreset-ratio",
            "0.5",
            "--memory-selection",
            "farthest",
            "--memory-mode",
            "global_nn",
            "--nn-topk",
            "5",
            "--nn-agg",
            "mean",
            "--batch-size",
            "64",
            "--num-workers",
            "4",
            "--gpu-id",
            str(gpu_id),
            *common,
            *rf_tta,
        ],
    }
    for shot in (1, 2, 4):
        specs[f"public_rf_k{shot}"] = [
            sys.executable,
            "tools/eval_cls_public_rf_vit_patchcore_gallery.py",
            "--output-root",
            str(output_root / f"public_rf_k{shot}"),
            "--support-manifest",
            f"analysis_outputs/20260810_public_rf_k_per_frequency/k{shot}/support_manifest.json",
            "--support-seed",
            "111",
            "--normal-sampling",
            "per_frequency",
            "--checkpoint",
            checkpoint,
            "--coreset-method",
            "farthest",
            "--coreset-ratio",
            "0.5",
            "--memory-mode",
            "global_nn",
            "--nn-topk",
            "5",
            "--nn-agg",
            "mean",
            "--batch-size",
            "96",
            "--num-workers",
            "4",
            "--gpu-id",
            str(gpu_id),
            *common,
            *public_tta,
        ]
    specs["ofdma"] = [
        sys.executable,
        "tools/eval_cls_ofdma_target_scene_ours.py",
        "--dataset-root",
        "/mnt/data/wangbei/data/ofdma-target-scene-coldstart-v2-realistic",
        "--split",
        "test",
        "--shots",
        "1",
        "2",
        "4",
        "--output-root",
        str(output_root / "ofdma"),
        "--checkpoint",
        checkpoint,
        "--backbone",
        "ViT-B-32",
        "--vit-tta-ablation",
        "spectral_response",
        "--gate-protocol",
        "support_only",
        "--batch-size",
        "32",
        "--num-workers",
        "4",
        "--gpu-id",
        str(gpu_id),
    ]
    specs["fedjam"] = [
        sys.executable,
        "tools/eval_fedjam_fewshot_dual.py",
        "--data-root",
        "/mnt/data/wangbei/data/FedJam",
        "--output-root",
        str(output_root / "fedjam"),
        "--checkpoint",
        checkpoint,
        "--shots",
        "1",
        "2",
        "4",
        "--batch-size",
        "8",
        "--distance-chunk-size",
        "1024",
        "--coreset-ratio",
        "0.5",
        "--nn-topk",
        "5",
        "--backbone",
        "ViT-B-32",
        "--vit-tta",
        "rf_spectral_response_v1",
        "--vit-tta-frequency-response-strength",
        "3",
        "--vit-tta-memory-layout",
        "merged",
        "--seed",
        "111",
        "--gpu-id",
        str(gpu_id),
    ]
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["inhouse_rf", "public_rf_k1", "public_rf_k2", "public_rf_k4", "ofdma", "fedjam"],
        default=["inhouse_rf", "public_rf_k1", "public_rf_k2", "public_rf_k4", "ofdma", "fedjam"],
    )
    parser.add_argument("--output-root", default=f"analysis_outputs/{OUTPUT_NAME}")
    parser.add_argument("--gpu-id", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    specs = command_specs(output_root, args.gpu_id)
    selected = {name: specs[name] for name in args.datasets}
    protocol_path = output_root / "run_protocol.json"
    previous_commands = {}
    if protocol_path.is_file():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        previous_commands = previous.get("commands", {})
    run_protocol = {
        "backbone": "ViT-B-32",
        "pretrained_dataset": "laion400m_e32",
        "seed": 111,
        "output_root": str(output_root),
        "formal_scheme": {
            "datasets": ["In-house RF", "Public RF", "OFDMA", "FedJam"],
            "vit_feature": "normalized layer1+layer2 patch features",
            "coreset": "0.5 farthest-first",
            "rf_fedjam_score": "5-NN mean distance",
            "ofdma_score": "1-NN, third-highest patch, max over 21 SUs",
            "vit_tta": "identity, time-axis +/-4 px, frequency-response drift strength 3",
            "test_query": "original view once",
            "support_policy": "normal support only; support-only evaluation",
        },
        "commands": {
            **previous_commands,
            **{name: command for name, command in selected.items()},
        },
    }
    protocol_path.write_text(
        json.dumps(run_protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        print(json.dumps(run_protocol, indent=2, ensure_ascii=False))
        return

    for name, command in selected.items():
        job_root = output_root / name
        job_root.mkdir(parents=True, exist_ok=True)
        log_path = job_root / "run.log"
        print(f"[run] {name}", flush=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            print(f"[failed] {name} returncode={result.returncode} log={log_path}", flush=True)
            raise SystemExit(result.returncode)
        print(f"[done] {name} log={log_path}", flush=True)


if __name__ == "__main__":
    main()
