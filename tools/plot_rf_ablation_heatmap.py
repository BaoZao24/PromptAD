import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATASETS = ["burst_signal", "chirp_signal", "dsss_signal"]
SCENES = [
    "WeaponMuseum_spectrum",
    "Playground_spectrum",
    "TimeSquare_spectrum",
    "Gymnasium_spectrum",
]
NOISE_LEVELS = ["m10db", "m20db", "m30db"]


def read_i_roc(root_dir, dataset, scene, noise_level, split_mode, k_shot, seed):
    csv_path = Path(root_dir) / dataset / scene / noise_level / split_mode / f"k_{k_shot}" / "csv" / f"Seed_{seed}-results.csv"
    df = pd.read_csv(csv_path, index_col=0)
    key = f"{dataset}-{scene}"
    return float(df.loc[key, "i_roc"])


def build_rows(args):
    rows = []
    for dataset in DATASETS:
        for scene in SCENES:
            for noise_level in NOISE_LEVELS:
                baseline = read_i_roc(
                    args.baseline_root,
                    dataset,
                    scene,
                    noise_level,
                    args.split_mode,
                    args.k_shot,
                    args.seed,
                )
                improved = read_i_roc(
                    args.improved_root,
                    dataset,
                    scene,
                    noise_level,
                    args.split_mode,
                    args.k_shot,
                    args.seed,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "scene": scene,
                        "noise": noise_level,
                        "baseline": baseline,
                        "improved": improved,
                        "delta": improved - baseline,
                    }
                )
    return pd.DataFrame(rows)


def plot_delta_heatmap(df, output_path, title):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(16, 4.8), constrained_layout=True)

    max_abs = max(abs(df["delta"].min()), abs(df["delta"].max()), 1.0)
    for ax, dataset in zip(axes, DATASETS):
        sub = df[df["dataset"] == dataset]
        table = sub.pivot(index="scene", columns="noise", values="delta").loc[SCENES, NOISE_LEVELS]
        sns.heatmap(
            table,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-max_abs,
            vmax=max_abs,
            linewidths=0.5,
            linecolor="white",
            cbar=(dataset == DATASETS[-1]),
            cbar_kws={"label": "Image-AUROC delta"},
        )
        ax.set_title(dataset)
        ax.set_xlabel("ISR")
        ax.set_ylabel("")

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_score_heatmaps(df, output_path, title):
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(16, 8.4), constrained_layout=True)

    for row_idx, value_col in enumerate(["baseline", "improved"]):
        for col_idx, dataset in enumerate(DATASETS):
            ax = axes[row_idx][col_idx]
            sub = df[df["dataset"] == dataset]
            table = sub.pivot(index="scene", columns="noise", values=value_col).loc[SCENES, NOISE_LEVELS]
            sns.heatmap(
                table,
                ax=ax,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                vmin=0,
                vmax=100,
                linewidths=0.5,
                linecolor="white",
                cbar=(col_idx == len(DATASETS) - 1),
                cbar_kws={"label": "Image-AUROC"},
            )
            ax.set_title(f"{dataset} - {value_col}")
            ax.set_xlabel("ISR")
            ax.set_ylabel("")

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", default="result_split_ablation/rf_rgb")
    parser.add_argument("--feature-root", default="result_split_ablation/rf_signal_adaptive")
    parser.add_argument("--prompt-root", default="result_split_ablation/rf_signal_structured_rgb")
    parser.add_argument("--both-root", default="result_split_ablation/rf_signal_structured_signal_adaptive")
    parser.add_argument("--baseline-name", default="rf + rgb/none")
    parser.add_argument("--feature-name", default="rf + signal_adaptive")
    parser.add_argument("--prompt-name", default="rf_signal_structured + rgb")
    parser.add_argument("--both-name", default="rf_signal_structured + signal_adaptive")
    parser.add_argument("--output-dir", default="analysis_outputs/rf_four_way_compare")
    parser.add_argument("--split-mode", default="normal_75_25")
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in DATASETS:
        for scene in SCENES:
            for noise_level in NOISE_LEVELS:
                baseline = read_i_roc(
                    args.baseline_root,
                    dataset,
                    scene,
                    noise_level,
                    args.split_mode,
                    args.k_shot,
                    args.seed,
                )
                feature = read_i_roc(
                    args.feature_root,
                    dataset,
                    scene,
                    noise_level,
                    args.split_mode,
                    args.k_shot,
                    args.seed,
                )
                prompt = read_i_roc(
                    args.prompt_root,
                    dataset,
                    scene,
                    noise_level,
                    args.split_mode,
                    args.k_shot,
                    args.seed,
                )
                both = read_i_roc(
                    args.both_root,
                    dataset,
                    scene,
                    noise_level,
                    args.split_mode,
                    args.k_shot,
                    args.seed,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "scene": scene,
                        "noise": noise_level,
                        args.baseline_name: baseline,
                        args.feature_name: feature,
                        args.prompt_name: prompt,
                        args.both_name: both,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "four_way_compare.csv", index=False)

    def add_delta_table(left, right, col_name):
        return pd.DataFrame(
            [
                {
                    "dataset": r["dataset"],
                    "scene": r["scene"],
                    "noise": r["noise"],
                    col_name: r[right] - r[left],
                }
                for _, r in df.iterrows()
            ]
        )

    delta_feature = add_delta_table(args.baseline_name, args.feature_name, "delta")
    delta_prompt = add_delta_table(args.baseline_name, args.prompt_name, "delta")
    delta_both = add_delta_table(args.baseline_name, args.both_name, "delta")

    plot_delta_heatmap(delta_feature.rename(columns={"delta": "delta"}), output_dir / "delta_feature_heatmap.png", f"{args.feature_name} - {args.baseline_name}")
    plot_delta_heatmap(delta_prompt.rename(columns={"delta": "delta"}), output_dir / "delta_prompt_heatmap.png", f"{args.prompt_name} - {args.baseline_name}")
    plot_delta_heatmap(delta_both.rename(columns={"delta": "delta"}), output_dir / "delta_both_heatmap.png", f"{args.both_name} - {args.baseline_name}")

    summary_rows = []
    for name in [args.baseline_name, args.feature_name, args.prompt_name, args.both_name]:
        vals = []
        for dataset in DATASETS:
            sub = df[df["dataset"] == dataset][name].tolist()
            vals.extend(sub)
            summary_rows.append({"method": name, "dataset": dataset, "mean": sum(sub) / len(sub)})
        summary_rows.append({"method": name, "dataset": "overall", "mean": sum(vals) / len(vals)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(summary.round(4))
    print(f"Saved heatmaps to {output_dir}")


if __name__ == "__main__":
    main()
