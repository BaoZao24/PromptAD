#!/usr/bin/env python3
"""Plot compact 1/2/4-shot ROC panels for the multi-shot datasets."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from sklearn.metrics import auc, roc_curve


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis_outputs/20260830_kshot_roc_comparison"
FPR_GRID = np.linspace(0.0, 1.0, 501)
SHOTS = (1, 2, 4)

METHOD_STYLE = {
    "GRETEL": {"color": "#CC79A7", "linestyle": "-.", "linewidth": 1.25},
    "TFAM-AAE": {"color": "#E69F00", "linestyle": "-", "linewidth": 1.25},
    "SPADE": {"color": "#009E73", "linestyle": "-", "linewidth": 1.35},
    "SpectraMemAD": {"color": "#D55E00", "linestyle": "-", "linewidth": 2.15},
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
        "font.size": 8.0,
        "axes.labelsize": 8.2,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "legend.fontsize": 7.3,
        "legend.frameon": False,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_curve(path: Path, score_key: str, label_key: str = "labels") -> np.ndarray:
    with np.load(path, allow_pickle=True) as payload:
        labels = np.asarray(payload[label_key]).reshape(-1)
        scores = np.asarray(payload[score_key], dtype=np.float64).reshape(-1)
    labels = (labels != 0).astype(np.int32)
    fpr, tpr, _ = roc_curve(labels, scores)
    curve = np.interp(FPR_GRID, fpr, tpr)
    curve[0] = 0.0
    curve[-1] = 1.0
    return curve


def mean_curve(paths: list[Path], score_key: str, label_key: str = "labels") -> np.ndarray:
    if not paths:
        raise RuntimeError(f"No score files found for key {score_key}")
    return np.mean([load_curve(path, score_key, label_key) for path in paths], axis=0)


def public_curves(shot: int) -> dict[str, np.ndarray]:
    suffix = "" if shot == 1 else f"_k{shot}"
    sources = {
        "GRETEL": (
            sorted(
                (
                    ROOT
                    / f"analysis_outputs/20260826_gretel_public_rf_k{shot}_formal/scores"
                ).glob("*.npz")
            ),
            "score",
        ),
        "TFAM-AAE": (
            sorted(
                (ROOT / f"analysis_outputs/tfam_aae_spectral_public_rf{suffix}/scores").glob(
                    "*.npz"
                )
            ),
            "scores",
        ),
        "SPADE": (
            sorted(
                (ROOT / f"analysis_outputs/20260817_spade_public_rf_k{shot}/scores").glob(
                    "*.npz"
                )
            ),
            "scores",
        ),
        "SpectraMemAD": (
            sorted(
                (
                    ROOT
                    / "analysis_outputs/exploratory/20260819_ours_wrn50_replace"
                    / f"public_fusion_r18_k{shot}/scores"
                ).glob("*.npz")
            ),
            "confidence_gated_score",
        ),
    }
    expected = {"GRETEL": 45, "TFAM-AAE": 15, "SPADE": 15, "SpectraMemAD": 15}
    for method, (paths, _) in sources.items():
        if len(paths) != expected[method]:
            raise RuntimeError(f"Public RF {shot}-shot {method}: found {len(paths)} files")
    return {method: mean_curve(paths, key) for method, (paths, key) in sources.items()}


def ofdma_curves(shot: int) -> dict[str, np.ndarray]:
    gretel = sorted(
        (ROOT / "analysis_outputs/20260826_gretel_ofdma_formal/scores").glob(
            f"*-{shot}shot-*.npz"
        )
    )
    ours = sorted(
        (
            ROOT
            / "analysis_outputs/exploratory/20260819_ours_wrn50_replace/ofdma_r18_repro/scores"
        ).glob(f"*_{shot}shot_observation_scores.npz")
    )
    if len(gretel) != 90 or len(ours) != 30:
        raise RuntimeError(
            f"OFDMA {shot}-shot: expected 90 GRETEL and 30 SpectraMemAD files; "
            f"found {len(gretel)} and {len(ours)}"
        )
    tfam = (
        ROOT
        / "analysis_outputs/tfam_aae_spectral_ofdma/scores"
        / f"ofdma-ofdma-official_split-{shot}shot-scores.npz"
    )
    spade = (
        ROOT
        / "analysis_outputs/20260817_spade_ofdma/scores"
        / f"ofdma-ofdma-official_split-{shot}shot-scores.npz"
    )
    return {
        "GRETEL": mean_curve(gretel, "score"),
        "TFAM-AAE": load_curve(tfam, "observation_scores", "observation_labels"),
        "SPADE": load_curve(spade, "scores"),
        "SpectraMemAD": mean_curve(ours, "confidence_gated_dual_visual"),
    }


def fedjam_curves(shot: int) -> dict[str, np.ndarray]:
    gretel = sorted(
        (ROOT / "analysis_outputs/20260826_gretel_fedjam_formal/scores").glob(
            f"*-{shot}shot-*.npz"
        )
    )
    if len(gretel) != 3:
        raise RuntimeError(f"FedJam {shot}-shot GRETEL: found {len(gretel)} files")
    tfam = (
        ROOT
        / "analysis_outputs/tfam_aae_spectral_fedjam/tfam/scores"
        / f"fedjam_{shot}shot_scores.npz"
    )
    spade = (
        ROOT
        / "analysis_outputs/20260817_fedjam_spade_formal/spade/scores"
        / f"fedjam_{shot}shot_scores.npz"
    )
    ours = (
        ROOT
        / "analysis_outputs/exploratory/20260819_ours_wrn50_replace/fedjam_r18_repro/scores"
        / f"fedjam_{shot}shot_scores.npz"
    )
    return {
        "GRETEL": mean_curve(gretel, "score"),
        "TFAM-AAE": load_curve(tfam, "scores"),
        "SPADE": load_curve(spade, "scores"),
        "SpectraMemAD": load_curve(ours, "confidence_gated_score"),
    }


def plot_dataset(dataset: str, loader, filename: str) -> dict[str, dict[str, float]]:
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55), sharex=True, sharey=True)
    summary: dict[str, dict[str, float]] = {}
    handles = []
    labels = []
    for index, (ax, shot) in enumerate(zip(axes, SHOTS)):
        curves = loader(shot)
        summary[f"{shot}-shot"] = {}
        for method, curve in curves.items():
            style = METHOD_STYLE[method]
            (line,) = ax.plot(
                FPR_GRID,
                curve,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                label=method,
                zorder=3 if method == "SpectraMemAD" else 2,
            )
            summary[f"{shot}-shot"][method] = float(auc(FPR_GRID, curve) * 100.0)
            if index == 0:
                handles.append(line)
                labels.append(method)
        ax.plot([0, 1], [0, 1], color="#A0A0A0", linestyle=":", linewidth=0.75, zorder=1)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"({'abc'[index]}) {shot}-shot", loc="left", fontweight="semibold")
        ax.set_xlabel("False positive rate")
        ax.grid(color="#D9DEE3", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("True positive rate")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        columnspacing=1.15,
        handlelength=2.25,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.27, top=0.91, wspace=0.16)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    png = OUTPUT / filename
    fig.savefig(png, dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return summary


def main() -> None:
    summary = {
        "Public RF": plot_dataset(
            "Public RF", public_curves, "fig_public_rf_kshot_roc.png"
        ),
        "OFDMA": plot_dataset("OFDMA", ofdma_curves, "fig_ofdma_kshot_roc.png"),
        "FedJam": plot_dataset("FedJam", fedjam_curves, "fig_fedjam_kshot_roc.png"),
    }
    (OUTPUT / "roc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
