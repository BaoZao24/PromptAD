#!/usr/bin/env python3
"""Build the formal five-type RF tables from unchanged and new score files.

The burst/chirp/DSSS/pulse cells are read from the previously completed formal
runs.  Only the newly added deceptive cells are read from the 2026-07-28 runs.
All reported values are equal-weighted means of per-cell image metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[1]
SIGNALS = ("burst", "chirp", "dsss", "pulse", "deceptive")
SELF_SCENES = (
    "WeaponMuseum_spectrum",
    "Playground_spectrum",
    "TimeSquare_spectrum",
    "Gymnasium_spectrum",
)
SELF_JSRS = {
    "burst": ("m10db", "m20db", "m30db"),
    "chirp": ("m10db", "m20db", "m30db"),
    "dsss": ("m10db", "m20db", "m30db"),
    "pulse": ("m20db", "m30db", "m40db"),
}
PUBLIC_JSRS = {
    "burst": ("m30db", "m40db", "m50db"),
    "chirp": ("m40db", "m50db", "m55db"),
    "dsss": ("m30db", "m40db", "m50db"),
    "pulse": ("m30db", "m40db", "m50db"),
    "deceptive": ("m10db", "m20db", "m30db"),
}
DECEPTIVE_STORAGE_JSR = {
    "WeaponMuseum_spectrum": ("m10db", "m20db", "m30db"),
    "Playground_spectrum": ("m10db", "m20db", "m30db"),
    "TimeSquare_spectrum": ("m10db", "m20db", "m30db"),
    "Gymnasium_spectrum": ("0db", "10db", "20db"),
}
DECEPTIVE_LEVELS = ("strong", "medium", "weak")

OLD_SELF_ROOT = (
    ROOT / "analysis_outputs/20260725_target_scene_self_rf_baselines_seed111"
)
NEW_ROOT = ROOT / "analysis_outputs/20260728_rf_deceptive_baselines_seed111"
SELF_CLASSIC = {
    "VAE": OLD_SELF_ROOT / "vae/scores",
    "SAIFE": OLD_SELF_ROOT / "saife/scores",
    "Deep SVDD": OLD_SELF_ROOT / "deepsvdd/scores",
    "PaDiM": OLD_SELF_ROOT / "padim/scores",
    "STFPM": OLD_SELF_ROOT / "stfpm/scores",
    "WinCLIP": OLD_SELF_ROOT / "winclip/scores",
    "PatchCore": OLD_SELF_ROOT / "patchcore/scores",
}
PUBLIC_CLASSIC = {
    "VAE": ROOT
    / "analysis_outputs/20260709_vae_baseline_sampling/per_frequency/public_vae/scores",
    "SAIFE": ROOT / "analysis_outputs/20260721_saife_public_rf_formal/scores",
    "Deep SVDD": ROOT
    / "analysis_outputs/20260709_deepsvdd_baseline_sampling/per_frequency/public_deepsvdd/scores",
    "PaDiM": ROOT
    / "analysis_outputs/20260709_padim_baseline_main/per_frequency/public_padim/scores",
    "STFPM": ROOT
    / "analysis_outputs/20260709_stfpm_baseline_main/per_frequency/public_stfpm/scores",
    "WinCLIP": ROOT
    / "analysis_outputs/20260709_winclip_reference_main/per_frequency/public_winclip/scores",
    "PatchCore": ROOT / "analysis_outputs/20260704_public_rf_official_patchcore_cls/scores",
}
PROMPTAD_SELF_OLD = OLD_SELF_ROOT / "promptad/scores"
PROMPTAD_PUBLIC_OLD = (
    ROOT
    / "analysis_outputs/20260706_baseline_sampling_shot_ablation/per_frequency/public_baseline/scores"
)
PROMPTAD_SELF_NEW = (
    ROOT / "analysis_outputs/20260728_rf_deceptive_power_residual"
)
PROMPTAD_PUBLIC_NEW = PROMPTAD_SELF_NEW / "public_vit_reused_normals/scores"
TRIPLE_ROOT = ROOT / "analysis_outputs/20260731_rf_dual_visual_scores/scores"
SAFE_ROOT = ROOT / "analysis_outputs/20260802_rf_support_only_formal/scores"

METHOD_ORDER = (
    "VAE",
    "SAIFE",
    "Deep SVDD",
    "PaDiM",
    "STFPM",
    "WinCLIP",
    "PatchCore",
    "PromptAD",
    "Ours",
)
ABLATION_KEYS = {
    "ViT normal memory": "vit_scores",
    "CNN normal memory": "cnn_scores",
    "Ours": "confidence_gated_score",
}


def metric_row(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError(
            f"Expected aligned 1-D labels/scores, got {labels.shape}/{scores.shape}"
        )
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Every formal evaluation cell must contain both classes")
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auroc": float(roc_auc_score(labels, scores) * 100.0),
        "ap": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if len(reached) else 100.0,
    }


def load_score(path: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        if key not in data.files:
            raise KeyError(f"{path} does not contain {key!r}; keys={data.files}")
        return (
            np.asarray(data["labels"], dtype=np.int32),
            np.asarray(data[key], dtype=np.float64),
        )


def add_row(
    rows: list[dict],
    *,
    dataset: str,
    method: str,
    signal: str,
    scene: str,
    jsr: str,
    source: Path,
    score_key: str,
) -> None:
    labels, scores = load_score(source, score_key)
    rows.append(
        {
            "dataset": dataset,
            "method": method,
            "signal": signal,
            "scene": scene,
            "jsr": jsr,
            "n": int(labels.size),
            **metric_row(labels, scores),
            "source": str(source.relative_to(ROOT)),
            "score_key": score_key,
        }
    )


def self_deceptive_source(root: Path, scene: str, jsr: str, prefix: str) -> Path:
    return root / f"{prefix}deceptive_signal-{scene}-{jsr}-scores.npz"


def collect_classic_rows(rows: list[dict]) -> None:
    for method in SELF_CLASSIC:
        old_root = SELF_CLASSIC[method]
        # Directory names intentionally follow the evaluation entry points.
        new_root = {
            "VAE": NEW_ROOT / "self/vae/scores",
            "SAIFE": NEW_ROOT / "self/saife/scores",
            "Deep SVDD": NEW_ROOT / "self/deepsvdd/scores",
            "PaDiM": NEW_ROOT / "self/padim/scores",
            "STFPM": NEW_ROOT / "self/stfpm/scores",
            "WinCLIP": NEW_ROOT / "self/winclip/scores",
            "PatchCore": NEW_ROOT / "self/patchcore/scores",
        }[method]
        for signal in SIGNALS[:-1]:
            for scene in SELF_SCENES:
                for jsr in SELF_JSRS[signal]:
                    add_row(
                        rows,
                        dataset="self_rf",
                        method=method,
                        signal=signal,
                        scene=scene,
                        jsr=jsr,
                        source=old_root
                        / f"rf_target-{signal}_signal-{scene}-{jsr}-scores.npz",
                        score_key="scores",
                    )
        for scene in SELF_SCENES:
            for level in DECEPTIVE_LEVELS:
                add_row(
                    rows,
                    dataset="self_rf",
                    method=method,
                    signal="deceptive",
                    scene=scene,
                    jsr=level,
                    source=self_deceptive_source(
                        new_root, scene, level, "rf_target-"
                    ),
                    score_key="scores",
                )

    for method in PUBLIC_CLASSIC:
        old_root = PUBLIC_CLASSIC[method]
        new_root = {
            "VAE": NEW_ROOT / "public/vae/scores",
            "SAIFE": NEW_ROOT / "public/saife/scores",
            "Deep SVDD": NEW_ROOT / "public/deepsvdd/scores",
            "PaDiM": NEW_ROOT / "public/padim/scores",
            "STFPM": NEW_ROOT / "public/stfpm/scores",
            "WinCLIP": NEW_ROOT / "public/winclip/scores",
            "PatchCore": NEW_ROOT / "public/patchcore/scores",
        }[method]
        for signal in SIGNALS:
            source_root = new_root if signal == "deceptive" else old_root
            for jsr in PUBLIC_JSRS[signal]:
                add_row(
                    rows,
                    dataset="public_rf",
                    method=method,
                    signal=signal,
                    scene="RF_SPE_PNG_public",
                    jsr=jsr,
                    source=source_root
                    / f"public_rf-{signal}-RF_SPE_PNG_public-{jsr}-scores.npz",
                    score_key="scores",
                )


def collect_promptad_rows(rows: list[dict]) -> None:
    for signal in SIGNALS[:-1]:
        for scene in SELF_SCENES:
            for jsr in SELF_JSRS[signal]:
                add_row(
                    rows,
                    dataset="self_rf",
                    method="PromptAD",
                    signal=signal,
                    scene=scene,
                    jsr=jsr,
                    source=PROMPTAD_SELF_OLD
                    / f"{signal}_signal-{scene}-{jsr}-scores.npz",
                    score_key="harmonic_text_vit_max",
                )
    for scene in SELF_SCENES:
        score_root = (
            PROMPTAD_SELF_NEW / "self_gym_vit/scores"
            if scene == "Gymnasium_spectrum"
            else PROMPTAD_SELF_NEW / "self_main3_vit/scores"
        )
        for level, storage_jsr in zip(
            DECEPTIVE_LEVELS, DECEPTIVE_STORAGE_JSR[scene]
        ):
            add_row(
                rows,
                dataset="self_rf",
                method="PromptAD",
                signal="deceptive",
                scene=scene,
                jsr=level,
                source=score_root
                / f"deceptive_signal-{scene}-{storage_jsr}-scores.npz",
                score_key="harmonic_text_vit_patchcore_max",
            )

    for signal in SIGNALS:
        source_root = (
            PROMPTAD_PUBLIC_NEW if signal == "deceptive" else PROMPTAD_PUBLIC_OLD
        )
        for jsr in PUBLIC_JSRS[signal]:
            add_row(
                rows,
                dataset="public_rf",
                method="PromptAD",
                signal=signal,
                scene="RF_SPE_PNG_public",
                jsr=jsr,
                source=source_root / f"{signal}-{jsr}-scores.npz",
                score_key="text_vit_patchcore_max",
            )


def triple_source(
    dataset: str,
    signal: str,
    scene: str,
    jsr: str,
    source_root: Path = TRIPLE_ROOT,
) -> Path:
    prefix = "in_house_rf" if dataset == "self_rf" else "public_rf"
    return source_root / f"{prefix}-{signal}-{scene}-{jsr}.npz"


def collect_triple_rows(
    rows: list[dict],
    method: str,
    score_key: str,
    source_root: Path = TRIPLE_ROOT,
) -> None:
    for signal in SIGNALS[:-1]:
        for scene in SELF_SCENES:
            for jsr in SELF_JSRS[signal]:
                add_row(
                    rows,
                    dataset="self_rf",
                    method=method,
                    signal=signal,
                    scene=scene,
                    jsr=jsr,
                    source=triple_source("self_rf", signal, scene, jsr, source_root),
                    score_key=score_key,
                )
    for scene in SELF_SCENES:
        for level, storage_jsr in zip(
            DECEPTIVE_LEVELS, DECEPTIVE_STORAGE_JSR[scene]
        ):
            add_row(
                rows,
                dataset="self_rf",
                method=method,
                signal="deceptive",
                scene=scene,
                jsr=level,
                source=triple_source(
                    "self_rf", "deceptive", scene, storage_jsr, source_root
                ),
                score_key=score_key,
            )

    for signal in SIGNALS:
        for jsr in PUBLIC_JSRS[signal]:
            add_row(
                rows,
                dataset="public_rf",
                method=method,
                signal=signal,
                scene="RF_SPE_PNG_public",
                jsr=jsr,
                source=triple_source(
                    "public_rf", signal, "RF_SPE_PNG_public", jsr, source_root
                ),
                score_key=score_key,
            )


def average_rows(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    output = []
    for group, items in groups.items():
        output.append(
            {
                **dict(zip(group_keys, group)),
                "n_cells": len(items),
                "auroc": float(np.mean([item["auroc"] for item in items])),
                "ap": float(np.mean([item["ap"] for item in items])),
                "fpr95": float(np.mean([item["fpr95"] for item in items])),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assert_cell_counts(rows: list[dict], methods: tuple[str, ...]) -> None:
    expected = {"self_rf": 60, "public_rf": 15}
    keys = set()
    for row in rows:
        key = (
            row["dataset"],
            row["method"],
            row["signal"],
            row["scene"],
            row["jsr"],
        )
        if key in keys:
            raise RuntimeError(f"Duplicate formal result cell: {key}")
        keys.add(key)
    for dataset, count in expected.items():
        for method in methods:
            actual = sum(
                row["dataset"] == dataset and row["method"] == method
                for row in rows
            )
            if actual != count:
                raise RuntimeError(
                    f"{dataset}/{method}: expected {count} cells, found {actual}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="analysis_outputs/20260728_rf_five_type_formal_seed111/summary",
    )
    args = parser.parse_args()
    output_root = ROOT / args.output_root

    main_rows: list[dict] = []
    collect_classic_rows(main_rows)
    collect_promptad_rows(main_rows)
    collect_triple_rows(
        main_rows,
        "Ours",
        "confidence_gated_score",
        source_root=SAFE_ROOT,
    )
    assert_cell_counts(main_rows, METHOD_ORDER)
    main_rows.sort(
        key=lambda row: (
            row["dataset"],
            METHOD_ORDER.index(row["method"]),
            SIGNALS.index(row["signal"]),
            row["scene"],
            row["jsr"],
        )
    )
    by_signal = average_rows(main_rows, ("dataset", "method", "signal"))
    macro = average_rows(main_rows, ("dataset", "method"))
    write_csv(output_root / "metrics_per_cell.csv", main_rows)
    write_csv(output_root / "metrics_by_signal.csv", by_signal)
    write_csv(output_root / "metrics_macro.csv", macro)

    ablation_rows: list[dict] = []
    for method, key in ABLATION_KEYS.items():
        collect_triple_rows(
            ablation_rows,
            method,
            key,
            source_root=SAFE_ROOT,
        )
    assert_cell_counts(ablation_rows, tuple(ABLATION_KEYS))
    write_csv(
        output_root / "ablation_by_signal.csv",
        average_rows(ablation_rows, ("dataset", "method", "signal")),
    )
    write_csv(
        output_root / "ablation_macro.csv",
        average_rows(ablation_rows, ("dataset", "method")),
    )

    protocol = {
        "signals": list(SIGNALS),
        "cell_aggregation": "equal-weighted arithmetic mean of per-cell metrics",
        "self_rf_cells_per_method": 60,
        "public_rf_cells_per_method": 15,
        "support_manifest": (
            "analysis_outputs/20260728_rf_five_type_formal_seed111/"
            "support_manifest.json"
        ),
        "support_manifest_sha256": (
            "65f4a06db8b490588020088fb2a522b960dc65d964096a2cede7d62ee3c6d296"
        ),
        "reused_signals": list(SIGNALS[:-1]),
        "newly_evaluated_signal": "deceptive",
        "ours_score_key": "confidence_gated_score",
        "ours_protocol": "safe_support_only",
        "transductive_comparison_source": str(TRIPLE_ROOT.relative_to(ROOT)),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_root / 'metrics_macro.csv'}")


if __name__ == "__main__":
    main()
