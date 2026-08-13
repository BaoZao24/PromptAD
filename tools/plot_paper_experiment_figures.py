#!/usr/bin/env python3
"""Create the figures used by the paper-style experiment section.

The result tables and protocol in ``docs/现有方案介绍.md`` are the source of
truth for this script.  It only consumes completed score/metric files; it does
not rerun a model or use test labels for inference.  Labels are used here only
to draw the reported evaluation curves.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Iterable, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These paths mirror the current formal result bundles cited by the paper
# experiment document.  The proposed method is the original two-branch
# ViT+CNN confidence-gated fusion; the power-residual exploratory branch is
# deliberately not part of the reported figures.
RF_FORMAL_ROOT = ROOT / "analysis_outputs/20260802_rf_support_only_summary"
RF_FUSION_ROOT = ROOT / "analysis_outputs/20260802_rf_support_only_formal"
PUBLIC_RF_K_ROOT = ROOT / "analysis_outputs/20260810_public_rf_k_per_frequency"
IAD_RF_ROOT = ROOT / "analysis_outputs/20260811_iad_per_rf_target_formal"
UDMA_RF_ROOT = ROOT / "analysis_outputs/20260811_udma_resnet18_rf_target_formal"
KLD_ICA_RF_ROOT = ROOT / "analysis_outputs/20260811_kld_ica_rf_target_formal"
DEFAULT_OFDMA_BASELINE_ROOT = (
    ROOT / "analysis_outputs/20260811_ofdma_v2_realistic_unified_published"
)
OFDMA_FORMAL_ROOT = ROOT / "analysis_outputs/20260731_ofdma_target_scene_ours_topk1_formal"
OFDMA_PATCHCORE_ROOT = ROOT / "analysis_outputs/20260810_ofdma_patchcore_formal"
TRAD_RF_TARGET_ROOT = ROOT / "analysis_outputs/20260805_traditional_rf_target_formal"
TRAD_PUBLIC_RF_ROOT = ROOT / "analysis_outputs/20260805_traditional_public_rf_formal"
TRAD_OFDMA_ROOT = ROOT / "analysis_outputs/20260805_traditional_ofdma_formal"
FEDJAM_FORMAL_ROOT = ROOT / "analysis_outputs/20260810_fedjam_fewshot_formal"
FEDJAM_VISUAL_ROOT = ROOT / "analysis_outputs/20260810_fedjam_visual_baselines_formal"
FEDJAM_TRAD_ROOT = ROOT / "analysis_outputs/20260810_fedjam_traditional_fewshot_formal"
IAD_FEDJAM_ROOT = ROOT / "analysis_outputs/20260811_iad_per_fedjam_formal/iad_per"
UDMA_FEDJAM_ROOT = ROOT / "analysis_outputs/20260811_udma_resnet18_fedjam_formal/udma"
KLD_ICA_FEDJAM_ROOT = ROOT / "analysis_outputs/20260811_kld_ica_fedjam_formal"
RF_STABILITY_ROOT = ROOT / "autoresearch/support-stability-rf-20-fixedtest/summary"

RF_SIGNAL_ORDER = ("burst", "chirp", "dsss", "pulse", "deceptive")
RF_SIGNAL_LABEL = {
    "burst": "Burst",
    "chirp": "Chirp",
    "dsss": "DSSS",
    "pulse": "Pulse",
    "deceptive": "Deceptive",
}

# Colorblind-safe qualitative colors derived from the Okabe-Ito palette.  The
# three main comparison methods keep the same colors in every figure; less
# important baselines are deliberately muted to preserve visual hierarchy.
INK = "#222222"
GRID = "#D9DEE3"
MUTED = "#7A7A7A"
LIGHT_MUTED = "#BFC5CA"
BLUE = "#0072B2"
SKY_BLUE = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"

VISUAL_METHOD_ORDER = [
    "iad_per",
    "saife_reconstruction",
    "udma_reimplementation",
    "patchcore_official",
    "winclip_fewshot",
]
TRADITIONAL_METHOD_ORDER = [
    "energy_detector",
    "ca_cfar",
    "statistical_fusion",
    "kld_reference",
]
# Internal branch ablations are intentionally kept out of the main comparison
# order.  They are loaded for the dedicated ablation plot and diagnostics.
ABLATION_METHOD_ORDER = ["vit_only", "cnn_only"]
METHOD_ORDER = [
    *TRADITIONAL_METHOD_ORDER,
    *VISUAL_METHOD_ORDER,
    "confidence_gated_fusion",
]
METHOD_LABEL = {
    "vae_reconstruction": "VAE-MSE [6]",
    "iad_per": "IAD-PER [6]",
    "saife_reconstruction": "SAIFE [7]",
    "deep_svdd": "Deep SVDD [8]",
    "padim_diag_resnet18": "PaDiM [9]",
    "stfpm_resnet18": "STFPM [10]",
    "winclip_fewshot": "WinCLIP [11]",
    "patchcore_official": "PatchCore [12]",
    "udma_reimplementation": "UDMA-ResNet18 [17]",
    "energy_detector": "ED [1]",
    "spectral_entropy": "Spectral entropy [2]",
    "spectral_flatness": "Spectral flatness [3]",
    "spectral_kurtosis": "Spectral kurtosis [4]",
    "ca_cfar": "CA-CFAR [5]",
    "kld_reference": "KLD-Ref [16]",
    "ica_frozen": "ICA-Frozen [16]",
    "statistical_fusion": "SCSE [1–5]",
    "vit_only": "ViT-only (ablation)",
    "cnn_only": "CNN-only (ablation)",
    "confidence_gated_fusion": "Ours (this work)",
}
METHOD_COLOR = {
    "vae_reconstruction": "#A9A9A9",
    "iad_per": "#4E79A7",
    "saife_reconstruction": "#6F6F6F",
    "deep_svdd": "#C9C9C9",
    "padim_diag_resnet18": PURPLE,
    "stfpm_resnet18": ORANGE,
    "winclip_fewshot": SKY_BLUE,
    "patchcore_official": "#006D77",
    "udma_reimplementation": "#B07AA1",
    "energy_detector": "#8C6D31",
    "spectral_entropy": "#17BECF",
    "spectral_flatness": "#9467BD",
    "spectral_kurtosis": "#BCBD22",
    "ca_cfar": "#7F3C8D",
    "kld_reference": "#F28E2B",
    "ica_frozen": "#59A14F",
    "statistical_fusion": "#B22222",
    "vit_only": GREEN,
    "cnn_only": PURPLE,
    "confidence_gated_fusion": VERMILLION,
}
METHOD_MARKER = {
    "vae_reconstruction": "v",
    "iad_per": "D",
    "saife_reconstruction": "^",
    "deep_svdd": "<",
    "padim_diag_resnet18": ">",
    "stfpm_resnet18": "P",
    "winclip_fewshot": "X",
    "patchcore_official": "s",
    "udma_reimplementation": "8",
    "energy_detector": "v",
    "spectral_entropy": "<",
    "spectral_flatness": ">",
    "spectral_kurtosis": "^",
    "ca_cfar": "P",
    "kld_reference": "h",
    "ica_frozen": "d",
    "statistical_fusion": "*",
    "vit_only": "^",
    "cnn_only": "s",
    "confidence_gated_fusion": "o",
}

TRADITIONAL_METHOD_KEY = {
    "energy_detector": "energy_detector",
    "spectral_entropy": "spectral_entropy",
    "spectral_flatness": "spectral_flatness",
    "spectral_kurtosis": "spectral_kurtosis",
    "ca_cfar": "ca_cfar",
    "statistical_fusion": "statistical_fusion",
}

# The Matplotlib cache in the experiment environment predates the installed
# Microsoft core fonts. Register the actual files so PDF export cannot silently
# fall back to DejaVu Sans.
for font_path in (
    Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold_Italic.ttf"),
):
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed plotting input: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=320, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def configure_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8)


def score_paths(specs: Iterable[tuple[Path, str]]) -> Iterator[Path]:
    for score_dir, prefix in specs:
        if not score_dir.is_dir():
            raise FileNotFoundError(f"Missing score directory: {score_dir}")
        for path in sorted(score_dir.glob(f"{prefix}*.npz")):
            yield path


def load_final_scores(
    specs: Iterable[tuple[Path, str]],
    score_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    labels, scores = [], []
    for path in score_paths(specs):
        with np.load(path, allow_pickle=True) as data:
            if score_key not in data.files:
                raise KeyError(f"{path} does not contain score key {score_key!r}")
            labels.append(np.asarray(data["labels"], dtype=np.int32))
            scores.append(np.asarray(data[score_key], dtype=np.float64))
    if not labels:
        raise FileNotFoundError(f"No score files found for {list(specs)!r}")
    return np.concatenate(labels), np.concatenate(scores)


def score_metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels, scores) * 100.0),
        "auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if len(reached) else 100.0,
    }


def macro_score_metrics(
    specs: Iterable[tuple[Path, str]],
    score_key: str,
) -> dict[str, float]:
    rows = []
    for path in score_paths(specs):
        with np.load(path, allow_pickle=True) as data:
            rows.append(score_metric(data["labels"], data[score_key]))
    if not rows:
        raise FileNotFoundError(f"No score files found for {specs!r}")
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("auroc", "auprc", "fpr95")
    }


def completed_score_metrics(
    root: Path,
    *,
    method: str,
    expected_files: int,
    prefix: str = "",
    score_key: str = "scores",
) -> dict[str, float]:
    score_dir = root / "scores"
    if not score_dir.is_dir():
        raise FileNotFoundError(
            f"{METHOD_LABEL[method]}: missing completed score directory: {score_dir}"
        )
    files = sorted(score_dir.glob(f"{prefix}*.npz"))
    if len(files) != expected_files:
        raise RuntimeError(
            f"{METHOD_LABEL[method]} is incomplete at {score_dir}: expected "
            f"{expected_files} score files, found {len(files)}"
        )
    return macro_score_metrics([(score_dir, prefix)], score_key)


def read_unique_macro(
    path: Path,
    *,
    method: str,
    dataset: str,
    shot: str | None = None,
) -> dict[str, float]:
    selected = [
        row
        for row in read_csv(path)
        if row.get("row_type") == "macro"
        and row.get("method") == method
        and row.get("dataset") == dataset
        and row.get("scope") == "overall"
        and (shot is None or row.get("shot") == shot)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"{METHOD_LABEL[method]}: expected one completed overall macro row "
            f"in {path}, found {len(selected)}"
        )
    return {
        metric: float(selected[0][metric])
        for metric in ("auroc", "auprc", "fpr95")
    }


def add_published_rf_metrics(
    metrics: dict[str, dict[str, dict[str, float]]],
) -> None:
    """Add completed IAD-PER, KLD/ICA, and UDMA RF results."""

    metrics["self_rf"]["iad_per"] = completed_score_metrics(
        IAD_RF_ROOT, method="iad_per", expected_files=60
    )
    metrics["self_rf"]["udma_reimplementation"] = completed_score_metrics(
        UDMA_RF_ROOT, method="udma_reimplementation", expected_files=60
    )
    for method in ("kld_reference", "ica_frozen"):
        metrics["self_rf"][method] = read_unique_macro(
            KLD_ICA_RF_ROOT / "metrics_macro.csv",
            method=method,
            dataset="rf_target",
        )

    public_summary = PUBLIC_RF_K_ROOT / "full_test_metrics_summary.csv"
    public_rows = {row["method"]: row for row in read_csv(public_summary)}
    for method in (
        "iad_per",
        "saife_reconstruction",
        "kld_reference",
        "ica_frozen",
        "udma_reimplementation",
    ):
        if method not in public_rows:
            raise RuntimeError(
                f"Public RF summary is stale or incomplete: {public_summary} lacks "
                f"{METHOD_LABEL[method]}. Run "
                "tools/summarize_public_rf_k_per_frequency_metrics.py after all "
                "20260811/20260812 k=1/2/4 runs complete."
            )
        row = public_rows[method]
        metrics["public_rf"][method] = {
            metric: float(row[f"k1_{metric}"])
            for metric in ("auroc", "auprc", "fpr95")
        }


def signal_from_score_path(path: Path, dataset: str) -> str:
    stem = path.stem
    prefix = "in_house_rf-" if dataset == "self_rf" else "public_rf-"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    return stem.split("-", 1)[0].removesuffix("_signal")


def signal_score_metrics(
    specs: Iterable[tuple[Path, str]],
    score_key: str,
    dataset: str,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for path in score_paths(specs):
        with np.load(path, allow_pickle=True) as data:
            signal = signal_from_score_path(path, dataset)
            grouped[signal].append(score_metric(data["labels"], data[score_key]))
    return {
        signal: {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("auroc", "auprc", "fpr95")
        }
        for signal, rows in grouped.items()
    }


def current_main_metrics() -> dict[str, dict[str, dict[str, float]]]:
    """Read visual and spectrogram-domain macro RF tables."""

    method_key = {
        "VAE": "vae_reconstruction",
        "VAE-MSE": "vae_reconstruction",
        "SAIFE": "saife_reconstruction",
        "Deep SVDD": "deep_svdd",
        "PaDiM": "padim_diag_resnet18",
        "STFPM": "stfpm_resnet18",
        "WinCLIP": "winclip_fewshot",
        "PatchCore": "patchcore_official",
        "Ours": "confidence_gated_fusion",
    }
    metrics: dict[str, dict[str, dict[str, float]]] = {"self_rf": {}, "public_rf": {}}

    rows = read_csv(RF_FORMAL_ROOT / "summary" / "metrics_macro.csv")
    for row in rows:
        dataset = row["dataset"]
        method = method_key.get(row["method"])
        if method is None:
            # The old formal bundle can still contain historical rows.  They
            # are intentionally excluded from the current paper figures.
            continue
        metrics[dataset][method] = {
            "auroc": float(row["auroc"]),
            "auprc": float(row["ap"]),
            "fpr95": float(row["fpr95"]),
        }

    traditional_paths = {
        "self_rf": TRAD_RF_TARGET_ROOT / "metrics_macro.csv",
        "public_rf": TRAD_PUBLIC_RF_ROOT / "metrics_macro.csv",
    }
    for dataset, path in traditional_paths.items():
        for row in read_csv(path):
            method = TRADITIONAL_METHOD_KEY.get(row["method"])
            if method is None:
                continue
            metrics[dataset][method] = {
                "auroc": float(row["auroc"]),
                "auprc": float(row["auprc"]),
                "fpr95": float(row["fpr95"]),
            }

    add_published_rf_metrics(metrics)

    expected = set(METHOD_ORDER)
    for dataset, rows in metrics.items():
        missing = expected - set(rows)
        if missing:
            raise RuntimeError(f"{dataset} is missing formal methods: {sorted(missing)}")
    return metrics


def load_ofdma_metrics(ofdma_root: Path) -> tuple[
    dict[int, dict[str, dict[str, float]]],
    dict[int, dict[str, dict[str, float]]],
    dict[str, dict[str, dict[str, float]]],
]:
    method_alias = {
        "patchcore_style_cnn": "patchcore_official",
        "confidence_gated_dual_visual": "confidence_gated_fusion",
    }
    overall_path = ofdma_root / "overall_scene_macro.csv"
    overall_rows = read_csv(overall_path)
    by_shot: dict[int, dict[str, dict[str, float]]] = {1: {}, 2: {}, 4: {}}
    for row in overall_rows:
        if row["scope"] != "overall":
            continue
        shot = int(row["shot"])
        method = method_alias.get(row["method"], row["method"])
        if method is None or method not in METHOD_ORDER:
            continue
        by_shot[shot][method] = {
            "auroc": float(row["auroc"]),
            "auprc": float(row["auprc"]),
            "fpr95": float(row["fpr95"]),
        }

    # Add the independently evaluated spectrogram-domain communication
    # baselines.  They have macro values but no repeated scene uncertainty in
    # this formal bundle, so their shot curves are drawn without error bars.
    for row in read_csv(TRAD_OFDMA_ROOT / "metrics_macro.csv"):
        method = TRADITIONAL_METHOD_KEY.get(row["method"])
        if method is None:
            continue
        shot = int(row["shot"])
        by_shot[shot][method] = {
            "auroc": float(row["auroc"]),
            "auprc": float(row["auprc"]),
            "fpr95": float(row["fpr95"]),
        }

    # PatchCore was completed later under the same target-scene protocol.  It
    # is read from its dedicated formal bundle rather than from historical
    # OFDMA comparison directories.
    patchcore_rows = read_csv(OFDMA_PATCHCORE_ROOT / "results.csv")
    for row in patchcore_rows:
        if (
            row["row_type"] != "scene_macro"
            or row["target_scene_id"] != "ALL"
            or row["scope"] != "overall"
        ):
            continue
        shot = int(row["shot"])
        by_shot[shot]["patchcore_official"] = {
            "auroc": float(row["auroc"]),
            "auprc": float(row["auprc"]),
            "fpr95": float(row["fpr95"]),
        }

    # The formal OFDMA table is a macro mean over 30 target scenes.  For the
    # plot, use the same per-scene rows to show the scene-to-scene spread as
    # +/- one sample standard deviation.  This keeps the error bars aligned
    # with the reported macro protocol instead of pooling all observations.
    unified_rows = read_csv(ofdma_root / "unified_results.csv")
    per_scene: dict[tuple[int, str], dict[str, list[float]]] = defaultdict(
        lambda: {"auroc": [], "auprc": [], "fpr95": []}
    )
    for row in unified_rows:
        if (
            row["row_type"] != "target_scene"
            or row["target_scene_id"] == "ALL"
            or row["scope"] != "overall"
        ):
            continue
        shot = int(row["shot"])
        method = method_alias.get(row["method"], row["method"])
        if method not in METHOD_ORDER:
            continue
        bucket = per_scene[(shot, method)]
        for metric in ("auroc", "auprc", "fpr95"):
            bucket[metric].append(float(row[metric]))
    for row in patchcore_rows:
        if (
            row["row_type"] != "target_scene"
            or row["scope"] != "overall"
            or row["target_scene_id"] == "ALL"
        ):
            continue
        shot = int(row["shot"])
        bucket = per_scene[(shot, "patchcore_official")]
        for metric in ("auroc", "auprc", "fpr95"):
            bucket[metric].append(float(row[metric]))
    by_shot_std: dict[int, dict[str, dict[str, float]]] = {1: {}, 2: {}, 4: {}}
    for (shot, method), values in per_scene.items():
        by_shot_std[shot][method] = {
            metric: float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
            for metric, samples in values.items()
        }

    jammer_path = ofdma_root / "per_jammer_scene_macro.csv"
    jammer_rows = read_csv(jammer_path)
    jammer: dict[str, dict[str, dict[str, float]]] = {}
    for row in jammer_rows:
        if int(row["shot"]) != 4:
            continue
        scope = row["scope"]
        method = method_alias.get(row["method"], row["method"])
        if method not in METHOD_ORDER:
            continue
        jammer.setdefault(scope, {})[method] = {
            "auroc": float(row["auroc"]),
            "auprc": float(row["auprc"]),
            "fpr95": float(row["fpr95"]),
        }
    for row in patchcore_rows:
        if (
            row["row_type"] != "scene_macro"
            or row["target_scene_id"] != "ALL"
            or row["shot"] != "4"
            or row["scope"] not in {"barrage", "deceptive", "pilot", "random_hop", "sweep"}
        ):
            continue
        jammer.setdefault(row["scope"], {})["patchcore_official"] = {
            "auroc": float(row["auroc"]),
            "auprc": float(row["auprc"]),
            "fpr95": float(row["fpr95"]),
        }
    expected = set(METHOD_ORDER)
    for shot, values in by_shot.items():
        missing = expected - set(values)
        if missing:
            raise RuntimeError(
                f"OFDMA merged result {ofdma_root} is incomplete for {shot}-shot: "
                f"missing methods {[METHOD_LABEL[method] for method in METHOD_ORDER if method in missing]}"
            )
    return by_shot, by_shot_std, jammer


def load_public_rf_k_scaling() -> dict[str, dict[str, dict[str, list[float]]]]:
    """Load the formal Public RF k=1/2/4 summaries by protocol."""

    method_alias = {
        "energy_detector": "energy_detector",
        "spectral_entropy": "spectral_entropy",
        "spectral_flatness": "spectral_flatness",
        "spectral_kurtosis": "spectral_kurtosis",
        "ca_cfar": "ca_cfar",
        "statistical_fusion": "statistical_fusion",
        "kld_reference": "kld_reference",
        "ica_frozen": "ica_frozen",
        "vae_reconstruction": "vae_reconstruction",
        "iad_per": "iad_per",
        "saife_reconstruction": "saife_reconstruction",
        "udma_reimplementation": "udma_reimplementation",
        "deep_svdd": "deep_svdd",
        "padim_diag_resnet18": "padim_diag_resnet18",
        "stfpm_resnet18": "stfpm_resnet18",
        "confidence_gated_dual_visual": "confidence_gated_fusion",
        "patchcore_resnet50": "patchcore_official",
        "winclip_fewshot": "winclip_fewshot",
    }
    paths = {
        "full-test": PUBLIC_RF_K_ROOT / "full_test_metrics_summary.csv",
        "controlled-subset": PUBLIC_RF_K_ROOT / "controlled_test_metrics_summary.csv",
    }
    result: dict[str, dict[str, dict[str, list[float]]]] = {
        protocol: {} for protocol in paths
    }
    for protocol, path in paths.items():
        for row in read_csv(path):
            method = method_alias.get(row["method"])
            if method is None:
                continue
            result[protocol][method] = {
                metric: [
                    float(row[f"k{shot}_{metric}"])
                    for shot in (1, 2, 4)
                ]
                for metric in ("auroc", "auprc", "fpr95")
            }
    required = {
        "full-test": {
            "energy_detector",
            "ca_cfar",
            "statistical_fusion",
            "kld_reference",
            "iad_per",
            "saife_reconstruction",
            "udma_reimplementation",
            "patchcore_official",
            "winclip_fewshot",
            "confidence_gated_fusion",
        },
        "controlled-subset": {"patchcore_official", "winclip_fewshot"},
    }
    for protocol, methods in required.items():
        missing = methods - set(result[protocol])
        if missing:
            raise RuntimeError(
                f"Public RF {protocol} summary is missing {sorted(missing)}"
            )
    return result


def plot_public_rf_k_scaling(
    data: dict[str, dict[str, dict[str, list[float]]]],
    output: Path,
) -> None:
    """Draw the Public RF full-test k-scaling comparison."""

    shots = [1, 2, 4]
    metric_info = (
        ("auroc", "AUROC (%) ↑", (45, 88)),
        ("auprc", "AUPRC (%) ↑", (0, 50)),
        ("fpr95", "FPR@95%TPR (%) ↓", (35, 100)),
    )
    full_methods = tuple(METHOD_ORDER)
    display = {
        "energy_detector": "ED [1]",
        "ca_cfar": "CA-CFAR [5]",
        "statistical_fusion": "SCSE [1–5]",
        "kld_reference": "KLD-Ref [16]",
        "iad_per": "IAD-PER [6]",
        "saife_reconstruction": "SAIFE [7]",
        "udma_reimplementation": "UDMA [17]",
        "patchcore_official": "PatchCore [12]",
        "winclip_fewshot": "WinCLIP [11]",
        "confidence_gated_fusion": "Ours (this work)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.35), sharex=True)
    for column, (metric, ylabel, ylim) in enumerate(metric_info):
        ax = axes[column]
        for method in full_methods:
            highlight = method == "confidence_gated_fusion"
            ax.plot(
                shots,
                data["full-test"][method][metric],
                marker=METHOD_MARKER[method],
                color=METHOD_COLOR[method],
                linewidth=2.2 if highlight else 1.35,
                markersize=5.2 if highlight else 4.2,
                label=display[method],
                zorder=3 if highlight else 2,
            )
        ax.set_xticks(shots, ["k=1", "k=2", "k=4"])
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"({'abc'[column]}) Full-test",
            loc="left",
            fontweight="semibold",
        )
        ax.set_xlabel("Normal reference images per frequency")
        configure_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=5,
        columnspacing=1.2,
        handlelength=2.0,
    )
    fig.text(
        0.5,
        0.005,
        "All curves use the same 5120 normal test images and all abnormal images in each of 15 Public RF cells.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.34, top=0.90, wspace=0.28)
    save_figure(fig, output)


def plot_main_auroc(
    metrics: dict[str, dict[str, dict[str, float]]],
    dataset: str,
    title: str,
    output: Path,
) -> None:
    """Draw one main-method comparison for one dataset only."""

    rows = metrics[dataset]
    methods = [method for method in reversed(METHOD_ORDER) if method in rows]
    values = [rows[method]["auroc"] for method in methods]
    fig, ax = plt.subplots(figsize=(5.4, 6.35))
    y = np.arange(len(methods))
    for method, value, y_pos in zip(methods, values, y):
        is_key = method in {
            "patchcore_official",
            "statistical_fusion",
            "confidence_gated_fusion",
        }
        ax.scatter(
            value,
            y_pos,
            s=42 if is_key else 27,
            marker=METHOD_MARKER[method],
            color=METHOD_COLOR[method],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax.text(value + 0.55, y_pos, f"{value:.2f}", va="center", fontsize=6.6, color=INK)
    ax.set_yticks(y, [METHOD_LABEL[method] for method in methods])
    ax.invert_yaxis()
    ax.set_xlim(40, 100)
    ax.set_xticks(np.arange(50, 101, 10))
    ax.set_xlabel("Macro-averaged AUROC (%)")
    ax.set_title(title, loc="left", fontweight="semibold")
    configure_axes(ax, grid_axis="x")
    fig.text(
        0.5,
        0.005,
        "Published communication/spectrogram and visual anomaly baselines; Ours = this work",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.33, right=0.98, bottom=0.13, top=0.94)
    save_figure(fig, output)


def plot_curves(
    labels: np.ndarray,
    scores: np.ndarray,
    dataset_title: str,
    output: Path,
) -> None:
    """Draw the proposed method's ROC/PR curves for one RF dataset."""

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores) * 100.0
    axes[0].plot(fpr, tpr, color=VERMILLION, linewidth=1.8)
    axes[0].plot([0, 1], [0, 1], "--", color=MUTED, linewidth=0.8)
    axes[0].set_title(f"(a) {dataset_title}: ROC", loc="left", fontweight="semibold")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].text(
        0.97,
        0.05,
        f"pooled AUROC = {auroc:.2f}%",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)

    precision, recall, _ = precision_recall_curve(labels, scores)
    auprc = average_precision_score(labels, scores) * 100.0
    axes[1].plot(recall, precision, color=VERMILLION, linewidth=1.8)
    axes[1].axhline(float(np.mean(labels)), linestyle="--", color=MUTED, linewidth=0.8)
    axes[1].set_title(
        f"(b) {dataset_title}: precision–recall",
        loc="left",
        fontweight="semibold",
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].text(
        0.03,
        0.05,
        f"pooled AUPRC = {auprc:.2f}%",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    for ax in axes:
        configure_axes(ax, grid_axis="both")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.13, top=0.91, wspace=0.28)
    save_figure(fig, output)


def plot_ofdma_shots(
    by_shot: dict[int, dict[str, dict[str, float]]],
    by_shot_std: dict[int, dict[str, dict[str, float]]],
    output: Path,
) -> None:
    shots = [1, 2, 4]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.8))
    for panel, (ax, metric, ylabel, ylim) in enumerate(zip(
        axes,
        ("auroc", "auprc", "fpr95"),
        ("Scene-averaged AUROC (%)", "Scene-averaged AUPRC (%)", "FPR@95%TPR (%) ↓"),
        ((40, 100), (45, 100), (25, 100)),
    )):
        for method in METHOD_ORDER:
            values = [by_shot[s][method][metric] for s in shots if method in by_shot[s]]
            if len(values) != len(shots):
                continue
            errors = [by_shot_std[s].get(method, {}).get(metric, 0.0) for s in shots]
            highlight = method in {"confidence_gated_fusion", "statistical_fusion", "patchcore_official"}
            ax.errorbar(
                shots,
                values,
                yerr=errors,
                fmt="none",
                ecolor=METHOD_COLOR[method],
                elinewidth=0.65 if highlight else 0.5,
                capsize=2.0 if highlight else 1.4,
                alpha=0.82 if highlight else 0.45,
                zorder=2,
            )
            ax.plot(
                shots,
                values,
                marker=METHOD_MARKER[method],
                markersize=4.6 if highlight else 3.4,
                linewidth=2.0 if method == "confidence_gated_fusion" else (1.35 if highlight else 0.75),
                alpha=1.0 if highlight else 0.58,
                color=METHOD_COLOR[method],
                label=METHOD_LABEL[method],
            )
        ax.set_xticks(shots, ["1-shot", "2-shot", "4-shot"])
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(f"({chr(97 + panel)})", loc="left", fontweight="semibold")
        configure_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=5,
        columnspacing=1.0,
        handlelength=1.8,
    )
    fig.text(
        0.5,
        0.005,
        "SCSE is the non-learning spectral ensemble; IAD-PER is the only VAE row in the main comparison",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.39, top=0.94, wspace=0.33)
    save_figure(fig, output)


def plot_ofdma_jammer(jammer: dict[str, dict[str, dict[str, float]]], output: Path) -> None:
    scopes = ["barrage", "deceptive", "pilot", "random_hop", "sweep"]
    candidate_methods = [
        "saife_reconstruction",
        "winclip_fewshot",
        "patchcore_official",
        "confidence_gated_fusion",
    ]
    methods = [
        method
        for method in candidate_methods
        if all(method in jammer.get(scope, {}) for scope in scopes)
    ]
    labels = ["Barrage", "Deceptive", "Pilot", "Random-hop", "Sweep"]
    x = np.arange(len(scopes))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.15), sharex=True)
    for panel, (ax, metric, ylabel) in enumerate(zip(
        axes,
        ("auroc", "fpr95"),
        ("AUROC (%)", "FPR@95%TPR (%) ↓"),
    )):
        for i, method in enumerate(methods):
            values = [jammer[s][method][metric] for s in scopes]
            bars = ax.bar(
                x + (i - (len(methods) - 1) / 2) * width,
                values,
                width,
                label=METHOD_LABEL[method],
                color=METHOD_COLOR[method],
                edgecolor="none",
            )
            if metric == "auroc":
                for bar, value in zip(bars, values):
                    if method == "confidence_gated_fusion":
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            value + 0.9,
                            f"{value:.1f}",
                            ha="center",
                            fontsize=6.8,
                            color=INK,
                        )
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels, rotation=16, ha="right", rotation_mode="anchor")
        ax.set_ylim(0, 106)
        ax.set_title(
            f"({chr(97 + panel)}) {'AUROC' if metric == 'auroc' else 'FPR@95%TPR'}",
            loc="left",
            fontweight="semibold",
        )
        configure_axes(ax)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=5,
        columnspacing=1.3,
        handlelength=2.2,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.24, top=0.82, wspace=0.25)
    save_figure(fig, output)


def _cell_minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    low, high = float(np.min(values)), float(np.max(values))
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def current_fusion_ablation_metrics() -> dict[str, dict[str, dict[str, float]]]:
    """Compute branch/fusion variants from the frozen RF score bundle."""

    variants = ("vit_only", "cnn_only", "ungated_or", "confidence_gated_fusion")
    result: dict[str, dict[str, dict[str, float]]] = {
        "self_rf": {variant: {} for variant in variants},
        "public_rf": {variant: {} for variant in variants},
    }
    for dataset, prefix in (("self_rf", "in_house_rf-"), ("public_rf", "public_rf-")):
        for path in score_paths([(RF_FUSION_ROOT / "scores", prefix)]):
            with np.load(path, allow_pickle=True) as data:
                vit = np.asarray(data["vit_scores"], dtype=np.float64)
                cnn = np.asarray(data["cnn_scores"], dtype=np.float64)
                vit_norm = _cell_minmax(vit)
                cnn_norm = _cell_minmax(cnn)
                scores = {
                    "vit_only": vit,
                    "cnn_only": cnn,
                    "ungated_or": 1.0 - (1.0 - vit_norm) * (1.0 - cnn_norm),
                    "confidence_gated_fusion": np.asarray(
                        data["confidence_gated_score"], dtype=np.float64
                    ),
                }
                labels = np.asarray(data["labels"], dtype=np.int32)
                for variant, variant_scores in scores.items():
                    row = score_metric(labels, variant_scores)
                    for metric in ("auroc", "auprc", "fpr95"):
                        result[dataset][variant].setdefault(metric, []).append(row[metric])
    return {
        dataset: {
            variant: {
                metric: float(np.mean(values))
                for metric, values in metrics.items()
            }
            for variant, metrics in variants_data.items()
        }
        for dataset, variants_data in result.items()
    }


def _mean_ci(values: list[float]) -> tuple[float, float]:
    values_array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values_array))
    if len(values_array) < 2:
        return mean, 0.0
    ci = 1.96 * float(np.std(values_array, ddof=1)) / np.sqrt(len(values_array))
    return mean, ci


def plot_support_stability(dataset: str, dataset_label: str, output: Path) -> None:
    """Draw repeated-support AUROC stability for one RF dataset only."""

    method_map = {
        "ViT-only": "ViT-only (ablation)",
        "CNN-only": "CNN-only (ablation)",
        "Ours": "Ours (this work)",
    }
    colors = {
        "ViT-only": GREEN,
        "CNN-only": PURPLE,
        "Ours": VERMILLION,
    }
    markers = {"ViT-only": "^", "CNN-only": "s", "Ours": "o"}
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in read_csv(RF_STABILITY_ROOT / "per_replicate_metrics.csv"):
        if row["dataset"] != dataset or row["shot"] != "all":
            continue
        if row["method"] not in method_map:
            continue
        grouped[row["method"]].append(float(row["auroc"]))
    methods = [method for method in ("ViT-only", "CNN-only", "Ours") if method in grouped]
    means, errors = [], []
    for method in methods:
        mean, ci = _mean_ci(grouped[method])
        means.append(mean)
        errors.append(ci)
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(5.7, 3.55))
    for x_pos, method, mean, error in zip(x, methods, means, errors):
        ax.errorbar(
            x_pos,
            mean,
            yerr=error,
            fmt=markers[method],
            color=colors[method],
            markersize=7.0,
            capsize=3.0,
            elinewidth=1.0,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=method_map[method],
            zorder=3,
        )
        ax.text(x_pos, mean + error + 0.12, f"{mean:.2f}", ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(x, [method_map[method] for method in methods], rotation=17, ha="right")
    lower = min(mean - error for mean, error in zip(means, errors)) - 0.8
    upper = max(mean + error for mean, error in zip(means, errors)) + 0.8
    ax.set_ylim(lower, upper)
    ax.set_ylabel("AUROC (%)")
    ax.set_title(dataset_label, loc="left", fontweight="semibold")
    configure_axes(ax)
    fig.text(
        0.5,
        0.005,
        "20 repeated supports; error bars are 95% CIs; internal branches are ablations",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.31, top=0.90)
    save_figure(fig, output)


def plot_signal_stability(dataset: str, dataset_label: str, output: Path) -> None:
    """Draw repeated-support type breakdown for one RF dataset only."""

    method_map = {
        "ViT-only": "ViT-only (ablation)",
        "CNN-only": "CNN-only (ablation)",
        "Ours": "Ours (this work)",
    }
    colors = {
        "ViT-only": GREEN,
        "CNN-only": PURPLE,
        "Ours": VERMILLION,
    }
    markers = {"ViT-only": "^", "CNN-only": "s", "Ours": "o"}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in read_csv(RF_STABILITY_ROOT / "rf_signal_replicate_metrics.csv"):
        if row["dataset"] != dataset or row["method"] not in method_map:
            continue
        grouped[(row["method"], row["signal"])].append(float(row["auroc"]))
    fig, ax = plt.subplots(figsize=(5.8, 3.55))
    x = np.arange(len(RF_SIGNAL_ORDER))
    for method in ("ViT-only", "CNN-only", "Ours"):
        means, errors = [], []
        for signal in RF_SIGNAL_ORDER:
            mean, ci = _mean_ci(grouped[(method, signal)])
            means.append(mean)
            errors.append(ci)
        ax.errorbar(
            x,
            means,
            yerr=errors,
            marker=markers[method],
            color=colors[method],
            linewidth=1.4 if method == "Ours" else 1.0,
            markersize=4.8,
            capsize=2.2,
            elinewidth=0.8,
            label=method_map[method],
        )
    ax.set_xticks(x, [RF_SIGNAL_LABEL[signal] for signal in RF_SIGNAL_ORDER], rotation=22, ha="right")
    ax.set_ylabel("AUROC (%)")
    ax.set_ylim(40, 100)
    ax.set_title(dataset_label, loc="left", fontweight="semibold")
    configure_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4,
               columnspacing=0.8, handlelength=1.6)
    fig.text(
        0.5,
        0.005,
        "20 repeated supports; error bars are 95% CIs; internal branches are ablations",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.29, top=0.82)
    save_figure(fig, output)


def plot_fusion_ablation(
    values: dict[str, dict[str, dict[str, float]]],
    dataset: str,
    dataset_label: str,
    output: Path,
) -> None:
    """Draw RF branch/fusion ablation for one dataset only."""

    variants = ("vit_only", "cnn_only", "confidence_gated_fusion")
    variant_labels = (
        "ViT branch (ablation)",
        "CNN branch (ablation)",
        "Confidence-gated fusion (ours)",
    )
    colors = (GREEN, PURPLE, VERMILLION)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15))
    x = np.arange(1)
    offsets = np.linspace(-0.27, 0.27, len(variants))
    for ax, metric, ylabel, ylim in zip(
        axes,
        ("auroc", "fpr95"),
        ("Macro image-level AUROC (%)", "Macro FPR@95%TPR (%) ↓"),
        ((65, 96), (15, 78)),
    ):
        for variant, label, offset, color in zip(variants, variant_labels, offsets, colors):
            value = values[dataset][variant][metric]
            ax.bar(
                x + offset,
                [value],
                width=0.16,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                label=label,
                zorder=3,
            )
            ax.text(
                x[0] + offset,
                value + (0.65 if metric == "auroc" else 1.0),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.4,
                color=INK,
            )
        ax.set_xticks(x, [dataset_label])
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(
            f"({'a' if metric == 'auroc' else 'b'}) {dataset_label}",
            loc="left",
            fontweight="semibold",
        )
        configure_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        columnspacing=0.9,
        handlelength=1.6,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.25, top=0.94, wspace=0.30)
    save_figure(fig, output)


def current_rf_breakdown() -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    rows = read_csv(RF_FORMAL_ROOT / "summary" / "metrics_by_signal.csv")
    method_key = {"Ours": "confidence_gated_fusion"}
    result: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        "self_rf": {},
        "public_rf": {},
    }
    for row in rows:
        if row["method"] not in method_key:
            continue
        dataset = row["dataset"]
        method = method_key[row["method"]]
        result[dataset].setdefault(method, {})[row["signal"]] = {
            "auroc": float(row["auroc"]),
            "auprc": float(row["ap"]),
            "fpr95": float(row["fpr95"]),
        }
    # Branch-only rows are stored by the same cell-macro evaluator.  Include
    # them in the dedicated breakdown so the main comparison remains clean.
    branch_key = {
        "ViT normal memory": "vit_only",
        "CNN normal memory": "cnn_only",
    }
    branch_rows = read_csv(RF_FORMAL_ROOT / "summary" / "ablation_by_signal.csv")
    for row in branch_rows:
        method = branch_key.get(row["method"])
        if method is None:
            continue
        result[row["dataset"]].setdefault(method, {})[row["signal"]] = {
            "auroc": float(row["auroc"]),
            "auprc": float(row["ap"]),
            "fpr95": float(row["fpr95"]),
        }
    return result


def plot_rf_breakdown(
    by_dataset: dict[str, dict[str, dict[str, dict[str, float]]]],
    dataset: str,
    title: str,
    output: Path,
) -> None:
    """Draw the RF interference-type breakdown for one dataset only."""

    fig, ax = plt.subplots(figsize=(5.5, 3.05))
    methods = ("vit_only", "cnn_only", "confidence_gated_fusion")
    names = list(RF_SIGNAL_ORDER)
    x = np.arange(len(names))
    for method in methods:
        scores = [by_dataset[dataset][method][name]["auroc"] for name in names]
        ax.plot(
            x,
            scores,
            marker=METHOD_MARKER[method],
            markersize=5.0,
            linewidth=1.8 if method == "confidence_gated_fusion" else 1.05,
            color=METHOD_COLOR[method],
            label=METHOD_LABEL[method],
        )
    ax.set_xticks(
        x,
        [RF_SIGNAL_LABEL[name] for name in names],
        rotation=22,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_ylabel("Image-level AUROC (%)")
    ax.set_ylim(40, 101)
    configure_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        columnspacing=0.9,
    )
    fig.text(
        0.5,
        0.005,
        "Internal branch ablation; Ungated OR is omitted from figures",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.29, top=0.82)
    save_figure(fig, output)


def load_fedjam_metrics() -> tuple[
    dict[int, dict[str, dict[str, float]]],
    dict[int, dict[str, dict[str, float]]],
]:
    """Load FedJam main methods and the separate branch ablation rows."""

    visual_dirs = {
        "vae": "vae_reconstruction",
        "saife": "saife_reconstruction",
        "deep_svdd": "deep_svdd",
        "padim": "padim_diag_resnet18",
        "stfpm": "stfpm_resnet18",
        "winclip": "winclip_fewshot",
        "patchcore": "patchcore_official",
    }
    by_shot: dict[int, dict[str, dict[str, float]]] = {1: {}, 2: {}, 4: {}}
    ablation: dict[int, dict[str, dict[str, float]]] = {1: {}, 2: {}, 4: {}}

    for directory, method in visual_dirs.items():
        for row in read_csv(FEDJAM_VISUAL_ROOT / directory / "metrics.csv"):
            if row["scope"] != "overall":
                continue
            shot = int(row["shot"])
            by_shot[shot][method] = {
                "auroc": float(row["auroc"]),
                "auprc": float(row["auprc"]),
                "fpr95": float(row["fpr95"]),
            }

    for row in read_csv(FEDJAM_TRAD_ROOT / "metrics.csv"):
        if row["scope"] != "overall":
            continue
        method = TRADITIONAL_METHOD_KEY.get(row["method"])
        if method is None:
            continue
        shot = int(row["shot"])
        by_shot[shot][method] = {
            "auroc": float(row["auroc"]),
            "auprc": float(row["auprc"]),
            "fpr95": float(row["fpr95"]),
        }

    published_visual = {
        "iad_per": IAD_FEDJAM_ROOT / "metrics.csv",
        "udma_reimplementation": UDMA_FEDJAM_ROOT / "metrics.csv",
    }
    for method, path in published_visual.items():
        selected = [
            row
            for row in read_csv(path)
            if row.get("method") == method and row.get("scope") == "overall"
        ]
        if {int(row["shot"]) for row in selected} != {1, 2, 4} or len(selected) != 3:
            raise RuntimeError(
                f"FedJam {METHOD_LABEL[method]} is incomplete at {path}: expected "
                "one overall row for each of 1/2/4-shot"
            )
        for row in selected:
            by_shot[int(row["shot"])][method] = {
                metric: float(row[metric])
                for metric in ("auroc", "auprc", "fpr95")
            }

    information_summary = KLD_ICA_FEDJAM_ROOT / "summary.json"
    if not information_summary.is_file():
        raise FileNotFoundError(
            f"Missing completed FedJam KLD/ICA summary: {information_summary}"
        )
    information_payload = json.loads(information_summary.read_text(encoding="utf-8"))
    pooled_rows = information_payload.get("pooled", [])
    for method in ("kld_reference", "ica_frozen"):
        selected = [
            row
            for row in pooled_rows
            if row.get("method") == method and row.get("scope") == "overall"
        ]
        if {int(row["shot"]) for row in selected} != {1, 2, 4} or len(selected) != 3:
            raise RuntimeError(
                f"FedJam {METHOD_LABEL[method]} is incomplete at {information_summary}: "
                "expected one pooled overall row for each of 1/2/4-shot"
            )
        for row in selected:
            by_shot[int(row["shot"])][method] = {
                metric: float(row[metric])
                for metric in ("auroc", "auprc", "fpr95")
            }

    branch_key = {
        "vit_patchcore": "vit_only",
        "cnn_layer3_top0.1": "cnn_only",
        "confidence_gated_dual_visual": "confidence_gated_fusion",
    }
    for row in read_csv(FEDJAM_FORMAL_ROOT / "metrics.csv"):
        if row["scope"] != "overall":
            continue
        shot = int(row["shot"])
        method = branch_key.get(row["method"])
        if method is None:
            continue
        values = {
            "auroc": float(row["auroc"]),
            "auprc": float(row["auprc"]),
            "fpr95": float(row["fpr95"]),
        }
        ablation[shot][method] = values
        if method == "confidence_gated_fusion":
            by_shot[shot][method] = values

    expected = set(METHOD_ORDER)
    for shot, rows in by_shot.items():
        missing = expected - set(rows)
        if missing:
            raise RuntimeError(f"FedJam {shot}-shot is missing formal methods: {sorted(missing)}")
        if set(ablation[shot]) != {"vit_only", "cnn_only", "confidence_gated_fusion"}:
            raise RuntimeError(f"FedJam {shot}-shot is missing branch ablation rows")
    return by_shot, ablation


def load_ofdma_ablation_metrics() -> dict[int, dict[str, dict[str, float]]]:
    """Load OFDMA ViT/CNN/Ours rows for the dedicated ablation figure."""

    by_shot: dict[int, dict[str, dict[str, float]]] = {1: {}, 2: {}, 4: {}}
    method_key = {"ours_vit": "vit_only", "ours_cnn": "cnn_only"}
    for row in read_csv(OFDMA_FORMAL_ROOT / "branch_diagnostics.csv"):
        if (
            row["row_type"] != "scene_macro"
            or row["target_scene_id"] != "ALL"
            or row["scope"] != "overall"
        ):
            continue
        method = method_key.get(row["method"])
        if method is None:
            continue
        by_shot[int(row["shot"])][method] = {
            "auroc": float(row["auroc"]),
            "auprc": float(row["auprc"]),
            "fpr95": float(row["fpr95"]),
        }
    for row in read_csv(OFDMA_FORMAL_ROOT / "results.csv"):
        if (
            row["row_type"] != "scene_macro"
            or row["target_scene_id"] != "ALL"
            or row["scope"] != "overall"
            or row["method"] != "confidence_gated_dual_visual"
        ):
            continue
        by_shot[int(row["shot"])]["confidence_gated_fusion"] = {
            "auroc": float(row["auroc"]),
            "auprc": float(row["auprc"]),
            "fpr95": float(row["fpr95"]),
        }
    for shot, rows in by_shot.items():
        if set(rows) != {"vit_only", "cnn_only", "confidence_gated_fusion"}:
            raise RuntimeError(f"OFDMA {shot}-shot is missing branch ablation rows")
    return by_shot


def plot_shot_scaling(
    by_shot: dict[int, dict[str, dict[str, float]]],
    output: Path,
    dataset_title: str,
    y_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    with_error_bars: bool = False,
    by_shot_std: dict[int, dict[str, dict[str, float]]] | None = None,
) -> None:
    """Draw a shot-scaling figure for one dataset."""

    shots = [1, 2, 4]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.8))
    for panel, (ax, metric, ylabel, ylim) in enumerate(
        zip(
            axes,
            ("auroc", "auprc", "fpr95"),
            ("Scene-averaged AUROC (%)", "Scene-averaged AUPRC (%)", "FPR@95%TPR (%) ↓"),
            y_limits,
        )
    ):
        for method in METHOD_ORDER:
            values = [by_shot[s][method][metric] for s in shots if method in by_shot[s]]
            if len(values) != len(shots):
                continue
            if with_error_bars and by_shot_std is not None:
                errors = [by_shot_std[s].get(method, {}).get(metric, 0.0) for s in shots]
                highlight = method in {
                    "confidence_gated_fusion",
                    "statistical_fusion",
                    "patchcore_official",
                }
                ax.errorbar(
                    shots,
                    values,
                    yerr=errors,
                    fmt="none",
                    ecolor=METHOD_COLOR[method],
                    elinewidth=0.65 if highlight else 0.5,
                    capsize=2.0 if highlight else 1.4,
                    alpha=0.82 if highlight else 0.45,
                    zorder=2,
                )
            highlight = method in {
                "confidence_gated_fusion",
                "statistical_fusion",
                "patchcore_official",
            }
            ax.plot(
                shots,
                values,
                marker=METHOD_MARKER[method],
                markersize=4.6 if highlight else 3.4,
                linewidth=2.0 if method == "confidence_gated_fusion" else (1.35 if highlight else 0.75),
                alpha=1.0 if highlight else 0.58,
                color=METHOD_COLOR[method],
                label=METHOD_LABEL[method],
            )
        ax.set_xticks(shots, ["1-shot", "2-shot", "4-shot"])
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(f"({'abc'[panel]}) {dataset_title}", loc="left", fontweight="semibold")
        configure_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=4,
        columnspacing=1.0,
        handlelength=1.8,
    )
    fig.text(
        0.5,
        0.005,
        "Main comparison: ED, CA-CFAR, SCSE, KLD-Ref, IAD-PER, SAIFE, UDMA, PatchCore, WinCLIP, and Ours",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.39, top=0.90, wspace=0.33)
    save_figure(fig, output)


def plot_branch_ablation_shots(
    by_shot: dict[int, dict[str, dict[str, float]]],
    dataset_title: str,
    output: Path,
) -> None:
    """Draw only the internal ViT/CNN/fusion ablation for one dataset."""

    shots = [1, 2, 4]
    methods = ("vit_only", "cnn_only", "confidence_gated_fusion")
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.25))
    for panel, (ax, metric, ylabel, ylim) in enumerate(
        zip(
            axes,
            ("auroc", "auprc", "fpr95"),
            ("AUROC (%)", "AUPRC (%)", "FPR@95%TPR (%) ↓"),
            ((40, 100), (45, 100), (25, 100)),
        )
    ):
        for method in methods:
            values = [by_shot[shot][method][metric] for shot in shots]
            ax.plot(
                shots,
                values,
                marker=METHOD_MARKER[method],
                markersize=4.8,
                linewidth=1.9 if method == "confidence_gated_fusion" else 1.1,
                color=METHOD_COLOR[method],
                label=METHOD_LABEL[method],
            )
        ax.set_xticks(shots, ["1-shot", "2-shot", "4-shot"])
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(f"({'abc'[panel]}) {dataset_title}", loc="left", fontweight="semibold")
        configure_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=3,
               columnspacing=1.0, handlelength=1.8)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.24, top=0.90, wspace=0.30)
    save_figure(fig, output)


def build_manifest(output: Path, figures: Iterable[str], ofdma_root: Path) -> None:
    lines = [
        "# Paper experiment figures",
        "",
        "These figures are generated from completed formal result files.  PNG and PDF versions are emitted for each figure.",
        "",
        "| Figure | Purpose |",
        "|---|---|",
        "| `fig_main_auroc_inhouse_rf` | Main comparison on In-house RF only |",
        "| `fig_main_auroc_public_rf` | Main comparison on Public RF only |",
        "| `fig_public_rf_k_scaling` | Public RF k=1/2/4 full-test and controlled-subset scaling |",
        "| `fig_curves_inhouse_rf` | In-house RF ROC and precision–recall curves |",
        "| `fig_curves_public_rf` | Public RF ROC and precision–recall curves |",
        "| `fig_support_stability_inhouse_rf` | In-house RF repeated-support stability |",
        "| `fig_support_stability_public_rf` | Public RF repeated-support stability |",
        "| `fig_signal_stability_inhouse_rf` | In-house RF repeated-support type breakdown |",
        "| `fig_signal_stability_public_rf` | Public RF repeated-support type breakdown |",
        "| `fig_ofdma_shot_scaling` | OFDMA 1/2/4-shot cold-start trend |",
        "| `fig_fedjam_shot_scaling` | FedJam 1/2/4-shot cold-start trend |",
        "| `fig_ofdma_jammer` | OFDMA four-shot breakdown by jammer type |",
        "| `fig_ablation_inhouse_rf` | In-house RF internal branch/fusion ablation |",
        "| `fig_ablation_public_rf` | Public RF internal branch/fusion ablation |",
        "| `fig_ablation_ofdma` | OFDMA internal ViT/CNN/fusion ablation |",
        "| `fig_ablation_fedjam` | FedJam internal ViT/CNN/fusion ablation |",
        "| `fig_rf_breakdown_inhouse_rf` | In-house RF type breakdown |",
        "| `fig_rf_breakdown_public_rf` | Public RF type breakdown |",
        "",
        "RF values use `20260802_rf_support_only_summary/summary/` and the safe support-only score bundle under `20260802_rf_support_only_formal/`.",
        "The RF protocol is the five-type burst/chirp/DSSS/pulse/deceptive protocol with equal-weighted cell macro metrics.",
        f"OFDMA values use the completed common target-scene merge under `{ofdma_root.relative_to(ROOT)}`; PatchCore remains sourced from `20260810_ofdma_patchcore_formal/`.",
        "The ROC/PR figure is a pooled-score visualization; the numeric tables in the experiment document remain the required scene/cell macro metrics.",
        "OFDMA error bars are the sample standard deviation across the same 30 target scenes used by the macro tables.",
        "Main-method labels are unified as ED, CA-CFAR, SCSE, KLD-Ref, IAD-PER, SAIFE, UDMA, PatchCore, WinCLIP, and Ours. ViT/CNN branch labels appear only in dedicated ablation/type figures; legacy VAE-MSE and other exploratory rows remain supplementary.",
        "Every figure is dataset-specific: no current main or ablation figure combines two datasets. ViT-only and CNN-only are internal ablations and are kept out of main-method rankings.",
        "Ungated OR is retained only for numeric diagnostics/tables and is intentionally omitted from every generated figure.",
        "Reproduce with: `python tools/plot_paper_experiment_figures.py --output-dir analysis_outputs/20260810_paper_experiment_figures_cited`.",
        "",
        "Generated files:",
        "",
    ]
    lines.extend(f"- `{name}.png` / `{name}.pdf`" for name in figures)
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis_outputs/20260731_paper_experiment_figures_dual_visual",
    )
    parser.add_argument(
        "--ofdma-results-root",
        type=Path,
        default=DEFAULT_OFDMA_BASELINE_ROOT,
        help="Completed output of tools/merge_ofdma_v2_results.py.",
    )
    parser.add_argument(
        "--public-rf-k-scaling-only",
        action="store_true",
        help="Only draw the Public RF k=1/2/4 full-test figure.",
    )
    args = parser.parse_args()
    output = args.output_dir
    ofdma_root = args.ofdma_results_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.public_rf_k_scaling_only:
        plot_public_rf_k_scaling(
            load_public_rf_k_scaling(),
            output / "fig_public_rf_k_scaling.png",
        )
        return

    metrics = current_main_metrics()
    ofdma_by_shot, ofdma_by_shot_std, jammer = load_ofdma_metrics(ofdma_root)
    fedjam_by_shot, fedjam_ablation = load_fedjam_metrics()
    ofdma_ablation = load_ofdma_ablation_metrics()
    fusion_ablation = current_fusion_ablation_metrics()
    rf_breakdown = current_rf_breakdown()
    self_labels, self_scores = load_final_scores(
        [(RF_FUSION_ROOT / "scores", "in_house_rf-")],
        "confidence_gated_score",
    )
    public_labels, public_scores = load_final_scores(
        [(RF_FUSION_ROOT / "scores", "public_rf-")],
        "confidence_gated_score",
    )
    figures = [
        "fig_main_auroc_inhouse_rf",
        "fig_main_auroc_public_rf",
        "fig_public_rf_k_scaling",
        "fig_curves_inhouse_rf",
        "fig_curves_public_rf",
        "fig_support_stability_inhouse_rf",
        "fig_support_stability_public_rf",
        "fig_signal_stability_inhouse_rf",
        "fig_signal_stability_public_rf",
        "fig_ofdma_shot_scaling",
        "fig_fedjam_shot_scaling",
        "fig_ofdma_jammer",
        "fig_ablation_inhouse_rf",
        "fig_ablation_public_rf",
        "fig_ablation_ofdma",
        "fig_ablation_fedjam",
        "fig_rf_breakdown_inhouse_rf",
        "fig_rf_breakdown_public_rf",
    ]
    plot_main_auroc(metrics, "self_rf", "(a) In-house RF", output / "fig_main_auroc_inhouse_rf.png")
    plot_main_auroc(metrics, "public_rf", "(b) Public RF", output / "fig_main_auroc_public_rf.png")
    plot_public_rf_k_scaling(
        load_public_rf_k_scaling(),
        output / "fig_public_rf_k_scaling.png",
    )
    plot_curves(self_labels, self_scores, "In-house RF", output / "fig_curves_inhouse_rf.png")
    plot_curves(public_labels, public_scores, "Public RF", output / "fig_curves_public_rf.png")
    plot_support_stability("self_rf", "In-house RF", output / "fig_support_stability_inhouse_rf.png")
    plot_support_stability("public_rf", "Public RF", output / "fig_support_stability_public_rf.png")
    plot_signal_stability("self_rf", "In-house RF", output / "fig_signal_stability_inhouse_rf.png")
    plot_signal_stability("public_rf", "Public RF", output / "fig_signal_stability_public_rf.png")
    plot_shot_scaling(
        ofdma_by_shot,
        output / "fig_ofdma_shot_scaling.png",
        "OFDMA",
        ((40, 100), (45, 100), (25, 100)),
        with_error_bars=True,
        by_shot_std=ofdma_by_shot_std,
    )
    plot_shot_scaling(
        fedjam_by_shot,
        output / "fig_fedjam_shot_scaling.png",
        "FedJam",
        ((25, 90), (60, 100), (25, 100)),
    )
    plot_ofdma_jammer(jammer, output / "fig_ofdma_jammer.png")
    plot_fusion_ablation(fusion_ablation, "self_rf", "In-house RF", output / "fig_ablation_inhouse_rf.png")
    plot_fusion_ablation(fusion_ablation, "public_rf", "Public RF", output / "fig_ablation_public_rf.png")
    plot_branch_ablation_shots(ofdma_ablation, "OFDMA", output / "fig_ablation_ofdma.png")
    plot_branch_ablation_shots(fedjam_ablation, "FedJam", output / "fig_ablation_fedjam.png")
    plot_rf_breakdown(rf_breakdown, "self_rf", "(a) In-house RF", output / "fig_rf_breakdown_inhouse_rf.png")
    plot_rf_breakdown(rf_breakdown, "public_rf", "(b) Public RF", output / "fig_rf_breakdown_public_rf.png")
    write_json(
        output / "figure_data.json",
        {
            "main_metrics": metrics,
            "rf_fusion_ablation": fusion_ablation,
            "rf_breakdown": rf_breakdown,
            "ofdma_by_shot": ofdma_by_shot,
            "ofdma_by_shot_std": ofdma_by_shot_std,
            "ofdma_ablation": ofdma_ablation,
            "fedjam_by_shot": fedjam_by_shot,
            "fedjam_ablation": fedjam_ablation,
            "ofdma_four_shot_jammer": jammer,
            "data_sources": {
                "rf_formal": str(RF_FORMAL_ROOT.relative_to(ROOT)),
                "rf_fusion_scores": str(RF_FUSION_ROOT.relative_to(ROOT)),
                "ofdma_baselines": str(ofdma_root.relative_to(ROOT)),
                "ofdma_formal": str(OFDMA_FORMAL_ROOT.relative_to(ROOT)),
                "ofdma_patchcore": str(OFDMA_PATCHCORE_ROOT.relative_to(ROOT)),
                "fedjam_formal": str(FEDJAM_FORMAL_ROOT.relative_to(ROOT)),
                "fedjam_visual": str(FEDJAM_VISUAL_ROOT.relative_to(ROOT)),
                "fedjam_traditional": str(FEDJAM_TRAD_ROOT.relative_to(ROOT)),
            },
        },
    )
    build_manifest(output, figures, ofdma_root)


if __name__ == "__main__":
    main()
