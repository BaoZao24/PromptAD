#!/usr/bin/env python
"""Evaluate the formal ViT + safe support-only CNN confidence score."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.confidence_gate import (
    RANK_QUANTILE,
    SAFE_SUPPORT_ALPHA,
    SAFE_SUPPORT_RANK_QUANTILE,
    SAFE_SUPPORT_TEMPERATURE,
    confidence_gated_or,
    safe_support_only_gate,
)


VIT_SCORE_KEY = "vit_patchcore_max_scores"
CNN_SCORE_KEYS = {
    "rf_target": "resnet18_layer3_top0.1_scores",
    "public_rf": "resnet18_layer3_scores",
    "spectrum": "resnet18_layer3_scores",
}
FINAL_SCORE_KEY = "confidence_gated_score"
FINAL_AUC_KEY = "confidence_gated_auc"
FINAL_AUPRC_KEY = "confidence_gated_auprc"
FINAL_FPR95_KEY = "confidence_gated_fpr95"


def safe_auc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores) * 100.0)


def score_metrics(labels, scores) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return {"auc": float("nan"), "auprc": float("nan"), "fpr95": float("nan")}
    fpr, tpr, _ = roc_curve(labels, scores)
    reached = np.flatnonzero(tpr >= 0.95)
    return {
        "auc": float(roc_auc_score(labels, scores) * 100.0),
        "auprc": float(average_precision_score(labels, scores) * 100.0),
        "fpr95": float(fpr[reached[0]] * 100.0) if len(reached) else 100.0,
    }


def filename_key(name) -> str:
    return Path(str(name)).stem


def baseline_name_to_file_stem(name) -> str:
    name = str(name)
    for part in name.split("-"):
        if part.endswith("_normal") or part.endswith("_abnormal"):
            return part
    return filename_key(name)


def align_indices(vit_npz, cnn_npz):
    vit_names = [str(value) for value in vit_npz["names"]]
    cnn_name_key = "image_paths" if "image_paths" in cnn_npz else "names"
    cnn_names = [str(value) for value in cnn_npz[cnn_name_key]]
    vit_by_exact = {name: index for index, name in enumerate(vit_names)}
    vit_by_stem = {}
    for index, name in enumerate(vit_names):
        vit_by_stem.setdefault(filename_key(name), index)
        vit_by_stem.setdefault(baseline_name_to_file_stem(name), index)

    vit_indices = []
    cnn_indices = []
    for cnn_index, name in enumerate(cnn_names):
        vit_index = vit_by_exact.get(name)
        if vit_index is None:
            vit_index = vit_by_stem.get(filename_key(name))
        if vit_index is None:
            key = filename_key(name)
            matches = [
                index
                for index, vit_name in enumerate(vit_names)
                if vit_name.endswith(key)
            ]
            if len(matches) == 1:
                vit_index = matches[0]
        if vit_index is None:
            raise RuntimeError(f"Could not align CNN sample {name}")
        vit_indices.append(vit_index)
        cnn_indices.append(cnn_index)

    vit_indices = np.asarray(vit_indices, dtype=np.int64)
    cnn_indices = np.asarray(cnn_indices, dtype=np.int64)
    if not np.array_equal(
        vit_npz["labels"][vit_indices].astype(np.int32),
        cnn_npz["labels"][cnn_indices].astype(np.int32),
    ):
        raise RuntimeError("Label mismatch after score alignment")
    return vit_indices, cnn_indices


def parse_cell(protocol: str, vit_path: Path):
    stem = vit_path.name.removesuffix("-scores.npz")
    if protocol == "public_rf":
        parts = stem.split("-")
        if parts[0] == "public_rf":
            signal = parts[1]
            jsr = parts[-1]
        else:
            signal, jsr = parts[0], parts[-1]
        return signal, "RF_SPE_PNG_public", jsr
    if protocol == "rf_target":
        parts = stem.split("-")
        if parts[0] == "in_house_rf":
            return parts[1], parts[2], "-".join(parts[3:])
        return parts[0], parts[1], "-".join(parts[2:])
    if protocol == "spectrum":
        return stem, "datasets/spectrum", "none"
    raise ValueError(f"Unsupported protocol: {protocol}")


def cnn_score_path(protocol: str, vit_path: Path, cnn_dir: Path) -> Path:
    stem = vit_path.name.removesuffix("-scores.npz")
    if protocol == "public_rf":
        signal, jsr = stem.split("-", 1)
        candidates = (
            cnn_dir
            / f"public_rf-{signal}-RF_SPE_PNG_public-{jsr}-scores.npz",
            cnn_dir / f"{stem}-scores.npz",
        )
    elif protocol == "rf_target":
        candidates = (
            cnn_dir / f"rf_target-{stem}-scores.npz",
            cnn_dir / f"{stem}-scores.npz",
        )
    elif protocol == "spectrum":
        candidates = (
            cnn_dir / f"spectrum-{stem}-datasets_spectrum-none-scores.npz",
            cnn_dir / f"{stem}-scores.npz",
        )
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def support_manifest_sha256(npz) -> str:
    if "support_manifest_sha256" not in npz:
        return ""
    return str(np.asarray(npz["support_manifest_sha256"]).item())


def support_manifest_file_digest(path: Path | None) -> str:
    """Read a model-independent support manifest digest."""

    if path is None:
        return ""
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    for key in ("support_paths_sha256", "manifest_sha256"):
        if value.get(key):
            return str(value[key])
    raise KeyError(f"{path} has no support_paths_sha256/manifest_sha256")


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def support_reference_paths(
    protocol: str,
    scene: str,
    vit_reference_dir: Path,
    cnn_reference_dir: Path,
) -> tuple[Path, Path]:
    """Resolve normal-only reference files for one RF evaluation cell."""

    filename = "public_rf.npz" if protocol == "public_rf" else f"{scene}.npz"
    return (
        vit_reference_dir / "support_reference" / filename,
        cnn_reference_dir / "support_reference" / filename,
    )


def load_reference(path: Path, key: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        if key not in data.files:
            raise KeyError(f"{path} does not contain {key!r}")
        return np.asarray(data[key], dtype=np.float64)


def grouped_mean(rows, key, metric_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    result = []
    for group in sorted(grouped):
        items = grouped[group]
        output = {key: group, "n_cells": len(items)}
        for metric in metric_keys:
            output[metric] = float(
                np.nanmean([item[metric] for item in items])
            )
        result.append(output)
    return result


def evaluate(args):
    vit_dir = Path(args.vit_score_dir)
    cnn_dir = Path(args.cnn_score_dir)
    score_dir = Path(args.output_root) / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    manifest_hashes = set()
    external_manifest = support_manifest_file_digest(
        Path(args.support_manifest) if args.support_manifest else None
    )
    for vit_path in sorted(vit_dir.glob("*-scores.npz")):
        cnn_path = cnn_score_path(args.protocol, vit_path, cnn_dir)
        if not cnn_path.exists():
            raise FileNotFoundError(
                f"Missing CNN score file for {vit_path.name}: {cnn_path}"
            )

        with np.load(vit_path, allow_pickle=True) as vit_npz, np.load(
            cnn_path,
            allow_pickle=True,
        ) as cnn_npz:
            vit_indices, cnn_indices = align_indices(vit_npz, cnn_npz)
            labels = vit_npz["labels"][vit_indices].astype(np.int32)
            vit_scores = vit_npz[VIT_SCORE_KEY][vit_indices].astype(
                np.float64
            )
            cnn_key = CNN_SCORE_KEYS[args.protocol]
            if cnn_key not in cnn_npz:
                raise KeyError(
                    f"{cnn_path} does not contain the fixed auxiliary CNN "
                    f"score key {cnn_key!r}"
                )
            cnn_scores = cnn_npz[cnn_key][cnn_indices].astype(
                np.float64
            )

            vit_manifest = support_manifest_sha256(vit_npz)
            cnn_manifest = support_manifest_sha256(cnn_npz)
            if args.protocol == "public_rf" and external_manifest:
                # Older public-ViT bundles predate the embedded hash field.
                # The model-independent manifest keeps those frozen scores
                # auditable without recomputing test scores.
                vit_manifest = vit_manifest or external_manifest
                cnn_manifest = cnn_manifest or external_manifest
            if vit_manifest or cnn_manifest:
                if vit_manifest != cnn_manifest:
                    raise RuntimeError(
                        "ViT/CNN support manifest mismatch: "
                        f"vit={vit_manifest or '<none>'}, "
                        f"cnn={cnn_manifest or '<none>'}"
                    )
                manifest_hashes.add(vit_manifest)

            signal, scene, jsr = parse_cell(args.protocol, vit_path)
            if args.gate_protocol == "support_only":
                if args.vit_reference_dir is None or args.cnn_reference_dir is None:
                    raise ValueError(
                        "support_only requires --vit-reference-dir and "
                        "--cnn-reference-dir"
                    )
                vit_ref_path, cnn_ref_path = support_reference_paths(
                    args.protocol,
                    scene,
                    Path(args.vit_reference_dir),
                    Path(args.cnn_reference_dir),
                )
                result = safe_support_only_gate(
                    vit_scores,
                    cnn_scores,
                    load_reference(vit_ref_path, "vit_scores"),
                    load_reference(cnn_ref_path, "cnn_scores"),
                )
            else:
                result = confidence_gated_or(vit_scores, cnn_scores)
            names = vit_npz["names"][vit_indices]

        vit_metrics = score_metrics(labels, vit_scores)
        cnn_metrics = score_metrics(labels, cnn_scores)
        fused_metrics = score_metrics(labels, result["score"])
        rows.append(
            {
                "method": "confidence_gated_dual_visual",
                "dataset": signal,
                "scene": scene,
                "jsr": jsr,
                "n": len(labels),
                "num_normal": int((labels == 0).sum()),
                "num_abnormal": int((labels == 1).sum()),
                "vit_auc": vit_metrics["auc"],
                "vit_auprc": vit_metrics["auprc"],
                "vit_fpr95": vit_metrics["fpr95"],
                "cnn_auc": cnn_metrics["auc"],
                "cnn_auprc": cnn_metrics["auprc"],
                "cnn_fpr95": cnn_metrics["fpr95"],
                FINAL_AUC_KEY: fused_metrics["auc"],
                FINAL_AUPRC_KEY: fused_metrics["auprc"],
                FINAL_FPR95_KEY: fused_metrics["fpr95"],
                "gate_mean": float(np.mean(result["gate"])),
                "gate_active_rate": float(np.mean(result["gate"] > 0.0)),
            }
        )
        payload = {
            "names": names,
            "labels": labels,
            "vit_score": vit_scores.astype(np.float32),
            "cnn_score": cnn_scores.astype(np.float32),
            "vit_normalized": result["vit_normalized"].astype(np.float32),
            "cnn_normalized": result["cnn_normalized"].astype(np.float32),
            "cnn_rank": result["cnn_rank"].astype(np.float32),
            "cnn_gate": result["gate"].astype(np.float32),
            "gate_protocol": np.asarray(args.gate_protocol),
            FINAL_SCORE_KEY: result["score"].astype(np.float32),
        }
        if vit_manifest:
            payload["support_manifest_sha256"] = np.asarray(vit_manifest)
        np.savez_compressed(score_dir / vit_path.name, **payload)

    if not rows:
        raise RuntimeError(f"No ViT score files found under {vit_dir}")
    return rows, sorted(manifest_hashes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vit-score-dir", required=True)
    parser.add_argument("--cnn-score-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--protocol",
        choices=["public_rf", "rf_target", "spectrum"],
        required=True,
    )
    parser.add_argument(
        "--gate-protocol",
        choices=["support_only", "transductive"],
        default="support_only",
        help=(
            "Formal default is support_only; transductive is retained only "
            "for protocol comparison."
        ),
    )
    parser.add_argument(
        "--vit-reference-dir",
        help="Normal-only ViT reference output containing support_reference/*.npz",
    )
    parser.add_argument(
        "--cnn-reference-dir",
        help="Normal-only CNN reference output containing support_reference/*.npz",
    )
    parser.add_argument(
        "--support-manifest",
        help=(
            "Optional model-independent public-RF support manifest used to "
            "audit score bundles that lack an embedded hash."
        ),
    )
    args = parser.parse_args()

    rows, manifest_hashes = evaluate(args)
    metric_keys = (
        "vit_auc",
        "vit_auprc",
        "vit_fpr95",
        "cnn_auc",
        "cnn_auprc",
        "cnn_fpr95",
        FINAL_AUC_KEY,
        FINAL_AUPRC_KEY,
        FINAL_FPR95_KEY,
    )
    macro = {
        key: float(np.nanmean([row[key] for row in rows]))
        for key in metric_keys
    }
    by_signal = grouped_mean(rows, "dataset", metric_keys)
    by_jsr = grouped_mean(rows, "jsr", metric_keys)

    output_root = Path(args.output_root)
    write_csv(output_root / "per_cell_confidence_gate.csv", rows)
    write_csv(output_root / "by_signal_confidence_gate.csv", by_signal)
    write_csv(output_root / "by_jsr_confidence_gate.csv", by_jsr)

    summary = {
        "method": "confidence_gated_dual_visual",
        "protocol": args.protocol,
        "gate_protocol": args.gate_protocol,
        "vit_score_dir": args.vit_score_dir,
        "cnn_score_dir": args.cnn_score_dir,
        "support_manifest": args.support_manifest,
        "cnn_score_key": CNN_SCORE_KEYS[args.protocol],
        "support_manifest_sha256": (
            manifest_hashes[0]
            if len(manifest_hashes) == 1
            else manifest_hashes or None
        ),
        "uses_test_batch_statistics": args.gate_protocol == "transductive",
        "uses_test_labels_for_scoring": False,
        "gate": {
            "rank_quantile": (
                SAFE_SUPPORT_RANK_QUANTILE
                if args.gate_protocol == "support_only"
                else RANK_QUANTILE
            ),
            "temperature": (
                SAFE_SUPPORT_TEMPERATURE
                if args.gate_protocol == "support_only"
                else None
            ),
            "alpha": (
                SAFE_SUPPORT_ALPHA if args.gate_protocol == "support_only" else None
            ),
        },
        "num_cells": len(rows),
        "macro": macro,
        "primary_metric": FINAL_AUC_KEY,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "README.md").write_text(
        "\n".join(
            [
                "# Confidence-Gated Dual Visual Fusion",
                "",
                "Formal default is safe support-only; transductive is retained only as a protocol comparison.",
                "",
                "## Macro image metrics",
                "",
                f"- ViT AUROC/AUPRC/FPR95: `{macro['vit_auc']:.4f}` / `{macro['vit_auprc']:.4f}` / `{macro['vit_fpr95']:.4f}`",
                f"- CNN AUROC/AUPRC/FPR95: `{macro['cnn_auc']:.4f}` / `{macro['cnn_auprc']:.4f}` / `{macro['cnn_fpr95']:.4f}`",
                f"- Confidence gate AUROC/AUPRC/FPR95: `{macro[FINAL_AUC_KEY]:.4f}` / `{macro[FINAL_AUPRC_KEY]:.4f}` / `{macro[FINAL_FPR95_KEY]:.4f}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
