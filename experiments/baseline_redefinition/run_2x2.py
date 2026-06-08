#!/usr/bin/env python3
"""Baseline Redefinition: 2x2 Ablation Study
Output directory: experiments/baseline_redefinition/
"""

import os
import subprocess
import csv
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_DIR)

RESULTS_BASE = "result/baseline_redefinition"

# 2x2 matrix
CONFIGS = {
    "A": {"prompt": "legacy", "input": "rgb", "label": "Original PromptAD"},
    "B": {"prompt": "rf", "input": "rgb", "label": "RF PromptAD"},
    "C": {"prompt": "legacy", "input": "morph_fusion_gray_residual_a01", "label": "Input-enhanced PromptAD"},
    "D": {"prompt": "rf", "input": "morph_fusion_gray_residual_a01", "label": "Ours-full"},
}

FIXED_ARGS = [
    "--k-shot", "1",
    "--split-mode", "normal_75_25",
    "--seed", "111",
    "--cls-score-mode", "text_only",
    "--vis", "False",
]

DATASETS = ["burst_signal", "chirp_signal", "dsss_signal"]
SCENES = ["Playground_spectrum"]
NOISE_LEVELS = ["m30db"]


def get_csv_path(group, dataset, scene, noise):
    return (Path(RESULTS_BASE) / group / dataset / scene / noise /
            "normal_75_25" / "k_1" / "csv" / "Seed_111-results.csv")


def run_experiment(group, dataset, scene, noise, prompt, input_mode):
    cmd = [
        sys.executable, "train_cls.py",
        "--dataset", dataset,
        "--class_name", scene,
        "--noise-level", noise,
        "--prompt-mode", prompt,
        "--input-mode", input_mode,
        "--root-dir", f"{RESULTS_BASE}/{group}",
    ] + FIXED_ARGS

    print(f"[{group}] {dataset}/{scene}/{noise} | prompt={prompt} | input={input_mode}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-500:]}")
        return None

    # Extract Image-AUROC from output
    for line in result.stdout.splitlines():
        if "Object:" in line and "Image-AUROC:" in line:
            auroc = line.split("Image-AUROC:")[-1].strip()
            print(f"  Image-AUROC: {auroc}")
            return float(auroc)

    # Fallback: read from CSV
    csv_path = get_csv_path(group, dataset, scene, noise)
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)  # header
            row = next(reader)
            auroc = float(row[1])
            print(f"  CSV Image-AUROC: {auroc}")
            return auroc

    print("  Could not find result")
    return None


def main():
    print("=" * 60)
    print("Baseline Redefinition: 2x2 Sanity Check")
    print("=" * 60)

    is_full = "--full" in sys.argv

    # --groups option: comma-separated, e.g. --groups A,D
    group_filter = None
    for i, arg in enumerate(sys.argv):
        if arg == "--groups" and i + 1 < len(sys.argv):
            group_filter = set(sys.argv[i + 1].split(","))

    datasets = DATASETS
    scenes = SCENES if not is_full else [
        "WeaponMuseum_spectrum", "Playground_spectrum",
        "TimeSquare_spectrum", "Gymnasium_spectrum"
    ]
    noises = NOISE_LEVELS if not is_full else ["m10db", "m20db", "m30db"]

    phase = "full" if is_full else "sanity"
    if group_filter:
        phase = f"{phase}_" + "_".join(sorted(group_filter))
    csv_file = Path("experiments/baseline_redefinition") / f"{phase}_2x2_results.csv"

    results = []
    groups = [g for g in ["A", "B", "C", "D"] if group_filter is None or g in group_filter]
    total = len(datasets) * len(scenes) * len(noises) * len(groups)
    count = 0

    for dataset in datasets:
        for scene in scenes:
            for noise in noises:
                for group in [g for g in ["A", "B", "C", "D"] if group_filter is None or g in group_filter]:
                    cfg = CONFIGS[group]
                    count += 1
                    print(f"\n[{count}/{total}]", end=" ")

                    auroc = run_experiment(
                        group, dataset, scene, noise,
                        cfg["prompt"], cfg["input"]
                    )
                    results.append({
                        "group": group,
                        "label": cfg["label"],
                        "dataset": dataset,
                        "scene": scene,
                        "noise": noise,
                        "prompt": cfg["prompt"],
                        "input": cfg["input"],
                        "Image-AUROC": auroc,
                    })

    # Write CSV
    fieldnames = ["group", "label", "dataset", "scene", "noise", "prompt", "input", "Image-AUROC"]
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {csv_file}")

    # Quick summary
    print("\n" + "=" * 60)
    print("Quick Summary (mean Image-AUROC per group)")
    print("=" * 60)
    for group in ["A", "B", "C", "D"]:
        vals = [r["Image-AUROC"] for r in results if r["group"] == group and r["Image-AUROC"] is not None]
        if vals:
            print(f"  {group} ({CONFIGS[group]['label']}): {sum(vals)/len(vals):.2f}% (n={len(vals)})")


if __name__ == "__main__":
    main()
