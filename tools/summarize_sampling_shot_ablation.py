#!/usr/bin/env python
"""Summarize normal-support sampling ablation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SAMPLINGS = ("per_frequency", "1shot", "2shot", "4shot")


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary, key):
    if not summary:
        return ""
    macro = summary.get("macro")
    if isinstance(macro, dict) and key in macro:
        return macro[key]
    return summary.get(f"{key}_macro", "")


def selected_count(summary):
    if not summary:
        return ""
    return summary.get("selected_normal_count", summary.get("num_train_normal", ""))


def fmt(value):
    if value == "" or value is None:
        return ""
    return f"{float(value):.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="analysis_outputs/20260725_confidence_gate_sampling_ablation",
    )
    parser.add_argument("--write-readme", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for sampling in SAMPLINGS:
        base = root / sampling
        self_vit = load_json(base / "self_vit" / "summary.json")
        self_cnn = load_json(base / "self_aux_cnn" / "summary.json")
        self_fusion = load_json(base / "self_fusion" / "summary.json")
        public_vit = load_json(base / "public_vit" / "summary.json")
        public_cnn = load_json(base / "public_aux_cnn" / "summary.json")
        public_fusion = load_json(base / "public_fusion" / "summary.json")
        spectrum_vit = load_json(base / "spectrum_vit" / "summary.json")
        spectrum_cnn = load_json(
            base / "spectrum_aux_cnn" / "summary.json"
        )
        spectrum_fusion = load_json(base / "spectrum_fusion" / "summary.json")
        rows.append(
            {
                "sampling": sampling,
                "self_selected_normals": selected_count(self_vit),
                "public_selected_normals": selected_count(public_vit),
                "public_gallery_patches": public_vit.get("gallery_patch_count", "") if public_vit else "",
                "spectrum_available": bool(spectrum_vit and spectrum_cnn and spectrum_fusion),
                "self_vit_auc": metric(self_vit, "vit_patchcore_max_auc"),
                "self_cnn_auc": self_cnn.get("image_auroc_macro", "") if self_cnn else "",
                "self_confidence_gated_auc": metric(self_fusion, "confidence_gated_auc"),
                "public_vit_auc": metric(public_vit, "vit_patchcore_max_auc"),
                "public_cnn_auc": public_cnn.get("image_auroc_macro", "") if public_cnn else "",
                "public_confidence_gated_auc": metric(public_fusion, "confidence_gated_auc"),
                "spectrum_vit_auc": metric(spectrum_vit, "clip_vit_nn_max_auc"),
                "spectrum_cnn_auc": spectrum_cnn.get("image_auroc_macro", "") if spectrum_cnn else "",
                "spectrum_confidence_gated_auc": metric(spectrum_fusion, "confidence_gated_auc"),
            }
        )

    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "sampling_shot_ablation_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Normal Support Sampling Ablation",
        "",
        "| sampling | self normals | self ViT | self CNN | self confidence gate | public normals | public ViT | public CNN | public confidence gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sampling']} | {row['self_selected_normals']} | "
            f"{fmt(row['self_vit_auc'])} | {fmt(row['self_cnn_auc'])} | {fmt(row['self_confidence_gated_auc'])} | "
            f"{row['public_selected_normals']} | {fmt(row['public_vit_auc'])} | "
            f"{fmt(row['public_cnn_auc'])} | {fmt(row['public_confidence_gated_auc'])} |"
        )
    lines.extend(
        [
            "",
            "Generated from per-sampling `summary.json` files.",
            "",
            f"CSV: `{csv_path.name}`",
        ]
    )
    print(f"wrote {csv_path}")
    if args.write_readme:
        readme_path = root / "README.md"
        readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {readme_path}")


if __name__ == "__main__":
    main()
