#!/usr/bin/env python3
"""Plot macro-averaged ROC curves for the RF main comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from sklearn.metrics import auc, roc_curve


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis_outputs/20260830_rf_roc_comparison"
FPR_GRID = np.linspace(0.0, 1.0, 501)


@dataclass(frozen=True)
class ScoreSource:
    label: str
    root: Path
    pattern: str
    key: str
    expected_inhouse: int
    expected_public: int
    color: str
    linestyle: str = "-"
    linewidth: float = 1.35


SOURCES = (
    ScoreSource(
        "ED",
        ROOT / "analysis_outputs/20260830_roc_scores/traditional_{dataset}/scores",
        "*.npz",
        "energy_detector",
        60,
        15,
        "#7A7A7A",
        "--",
    ),
    ScoreSource(
        "SCSE",
        ROOT / "analysis_outputs/20260830_roc_scores/traditional_{dataset}/scores",
        "*.npz",
        "statistical_fusion",
        60,
        15,
        "#0072B2",
    ),
    ScoreSource(
        "GRETEL",
        ROOT / "analysis_outputs/20260826_gretel_{dataset}_formal/scores",
        "*.npz",
        "score",
        180,
        45,
        "#CC79A7",
        "-.",
    ),
    ScoreSource(
        "TFAM-AAE",
        ROOT / "analysis_outputs/tfam_aae_spectral_{dataset}/scores",
        "*.npz",
        "scores",
        60,
        15,
        "#E69F00",
    ),
    ScoreSource(
        "SPADE",
        ROOT / "analysis_outputs/20260817_spade_{dataset}/scores",
        "*.npz",
        "scores",
        60,
        15,
        "#009E73",
    ),
    ScoreSource(
        "SpectraMemAD",
        ROOT / "analysis_outputs/exploratory/20260819_ours_wrn50_replace/{dataset}/scores",
        "*.npz",
        "confidence_gated_score",
        60,
        15,
        "#D55E00",
        "-",
        2.2,
    ),
)


DATASET_PATH_KEYS = {
    "inhouse": {
        "traditional": "inhouse",
        "gretel": "inhouse_rf",
        "tfam": "self_rf",
        "spade": "rf_target_full",
        "ours": "inhouse_fusion_r18",
    },
    "public": {
        "traditional": "public_k1",
        "gretel": "public_rf_k1",
        "tfam": "public_rf",
        "spade": "public_rf_k1",
        "ours": "public_fusion_r18_k1",
    },
}


for font_path in (
    Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"),
):
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.4,
        "legend.frameon": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def resolve_root(source: ScoreSource, dataset: str) -> Path:
    keys = DATASET_PATH_KEYS[dataset]
    template = str(source.root)
    if source.label in {"ED", "SCSE"}:
        value = keys["traditional"]
    elif source.label == "GRETEL":
        value = keys["gretel"]
    elif source.label == "TFAM-AAE":
        value = keys["tfam"]
    elif source.label == "SPADE":
        value = keys["spade"]
    else:
        value = keys["ours"]
    return Path(template.format(dataset=value))


def macro_roc(source: ScoreSource, dataset: str) -> tuple[np.ndarray, float, int]:
    root = resolve_root(source, dataset)
    paths = sorted(root.glob(source.pattern))
    expected = source.expected_inhouse if dataset == "inhouse" else source.expected_public
    if len(paths) != expected:
        raise RuntimeError(f"{source.label} {dataset}: expected {expected} files, found {len(paths)} in {root}")

    interpolated = []
    for path in paths:
        with np.load(path, allow_pickle=True) as payload:
            labels = np.asarray(payload["labels"], dtype=np.int32).reshape(-1)
            scores = np.asarray(payload[source.key], dtype=np.float64).reshape(-1)
        fpr, tpr, _ = roc_curve(labels, scores)
        curve = np.interp(FPR_GRID, fpr, tpr)
        curve[0] = 0.0
        curve[-1] = 1.0
        interpolated.append(curve)

    mean_tpr = np.mean(interpolated, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    return mean_tpr, float(auc(FPR_GRID, mean_tpr) * 100.0), len(paths)


def plot_dataset(dataset: str, output_name: str) -> dict[str, dict[str, float | int]]:
    fig, ax = plt.subplots(figsize=(3.55, 3.05))
    summary: dict[str, dict[str, float | int]] = {}
    for source in SOURCES:
        mean_tpr, macro_auroc, cells = macro_roc(source, dataset)
        summary[source.label] = {"macro_auroc": macro_auroc, "num_curves": cells}
        ax.plot(
            FPR_GRID,
            mean_tpr,
            color=source.color,
            linestyle=source.linestyle,
            linewidth=source.linewidth,
            label=source.label,
            zorder=3 if source.label == "SpectraMemAD" else 2,
        )

    ax.plot([0, 1], [0, 1], color="#A0A0A0", linestyle=":", linewidth=0.8, zorder=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.grid(color="#D9DEE3", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        columnspacing=0.9,
        handlelength=2.2,
    )
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.31, top=0.98)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    png = OUTPUT / f"{output_name}.png"
    fig.savefig(png, dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return summary


def main() -> None:
    summary = {
        "inhouse_rf": plot_dataset("inhouse", "fig_roc_comparison_inhouse_rf"),
        "public_rf_k1": plot_dataset("public", "fig_roc_comparison_public_rf_k1"),
    }
    (OUTPUT / "roc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
