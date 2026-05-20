import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SCENES = [
    "WeaponMuseum_spectrum",
    "Playground_spectrum",
    "TimeSquare_spectrum",
    "Gymnasium_spectrum",
]
NOISE_LEVELS = ["m10db", "m20db", "m30db"]


def read_i_roc(root_dir, scene, noise_level, split_mode, k_shot, seed):
    csv_path = Path(root_dir) / "dsss_signal" / scene / noise_level / split_mode / f"k_{k_shot}" / "csv" / f"Seed_{seed}-results.csv"
    df = pd.read_csv(csv_path, index_col=0)
    return float(df.loc[f"dsss_signal-{scene}", "i_roc"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", default="result_split_ablation/rf_rgb")
    parser.add_argument("--new-root", default="result_split_ablation/rf_dsss_statistical")
    parser.add_argument("--split-mode", default="normal_75_25")
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--output-dir", default="analysis_outputs/dsss_statistical_vs_baseline")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scene in SCENES:
        for noise in NOISE_LEVELS:
            baseline = read_i_roc(args.baseline_root, scene, noise, args.split_mode, args.k_shot, args.seed)
            new = read_i_roc(args.new_root, scene, noise, args.split_mode, args.k_shot, args.seed)
            rows.append({
                "scene": scene,
                "noise": noise,
                "baseline": baseline,
                "new": new,
                "delta": new - baseline,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dsss_statistical_vs_baseline.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    max_abs = max(abs(df["delta"].min()), abs(df["delta"].max()), 1.0)

    delta_table = df.pivot(index="scene", columns="noise", values="delta").loc[SCENES, NOISE_LEVELS]
    sns.heatmap(delta_table, ax=axes[0], annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-max_abs, vmax=max_abs, linewidths=0.5, linecolor="white", cbar_kws={"label": "Image-AUROC delta"})
    axes[0].set_title("dsss_statistical - rf_rgb")
    axes[0].set_xlabel("ISR")
    axes[0].set_ylabel("")

    compare = pd.DataFrame({
        "baseline": df["baseline"].values,
        "new": df["new"].values,
    })
    compare["label"] = [f"{s}\n{n}" for s, n in zip(df["scene"], df["noise"])]
    compare = compare.set_index("label")
    sns.heatmap(compare, ax=axes[1], annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=100,
                linewidths=0.5, linecolor="white", cbar_kws={"label": "Image-AUROC"})
    axes[1].set_title("baseline vs dsss_statistical")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    fig.savefig(out_dir / "dsss_statistical_vs_baseline.png", dpi=220)
    plt.close(fig)

    summary = df[["baseline", "new", "delta"]].mean().to_frame("overall").T
    summary.to_csv(out_dir / "summary.csv")
    print(summary.round(4))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
