#!/usr/bin/env python3
"""Merge the v2 realistic OFDMA method bundles into one paper table.

Each method is evaluated independently and writes the same row schema.  This
utility only concatenates completed result files and extracts the scene-macro
rows consumed by the plotting script; it never recomputes scores or uses
labels beyond the metrics already written by each evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SHOTS = (1, 2, 4)
SCOPES = ("overall", "barrage", "deceptive", "pilot", "random_hop", "sweep")
EXPECTED_TARGET_SCENES = 30

LEGACY_METHOD_DIRS = {
    "vae_reconstruction": "vae",
    "saife_reconstruction": "saife",
    "deep_svdd": "deep_svdd",
    "padim_diag_resnet18": "padim",
    "stfpm_resnet18": "stfpm",
    "winclip_fewshot": "winclip",
    "confidence_gated_dual_visual": "ours",
}
NEW_DIRECT_METHOD_DIRS = {
    "iad_per": "20260811_iad_per_ofdma_v2_realistic_formal",
    "udma_reimplementation": "20260811_udma_resnet18_ofdma_v2_realistic_formal",
}
INFORMATION_THEORETIC_DIR = "20260811_kld_ica_ofdma_v2_realistic_formal"
METHOD_ORDER = (
    "kld_reference",
    "ica_frozen",
    "vae_reconstruction",
    "iad_per",
    "saife_reconstruction",
    "deep_svdd",
    "padim_diag_resnet18",
    "stfpm_resnet18",
    "winclip_fewshot",
    "udma_reimplementation",
    "confidence_gated_dual_visual",
)
METHOD_INDEX = {method: index for index, method in enumerate(METHOD_ORDER)}
METHOD_ALIASES = {"patchcore_style_cnn": "patchcore_official"}
METHOD_DISPLAY = {
    "kld_reference": "KLD-Ref",
    "ica_frozen": "ICA-Frozen",
    "vae_reconstruction": "VAE-MSE",
    "iad_per": "IAD-PER",
    "saife_reconstruction": "SAIFE",
    "deep_svdd": "Deep SVDD",
    "padim_diag_resnet18": "PaDiM",
    "stfpm_resnet18": "STFPM",
    "winclip_fewshot": "WinCLIP",
    "udma_reimplementation": "UDMA-ResNet18",
    "confidence_gated_dual_visual": "Ours: support-only confidence fusion",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed method result: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def canonicalize_rows(
    rows: list[dict[str, str]], *, expected_method: str, source: Path
) -> list[dict[str, str]]:
    output = []
    for original in rows:
        row = dict(original)
        row["method"] = METHOD_ALIASES.get(row.get("method", ""), row.get("method", ""))
        if row["method"] == expected_method:
            output.append(row)
    if not output:
        raise RuntimeError(
            f"OFDMA {expected_method}: no matching rows found in completed result {source}"
        )
    return output


def convert_information_theoretic_rows(
    root: Path, *, method: str
) -> tuple[list[dict[str, str]], list[Path]]:
    cell_path = root / "metrics_per_cell.csv"
    macro_path = root / "metrics_macro.csv"
    cell_rows = read_rows(cell_path)
    macro_rows = read_rows(macro_path)
    output: list[dict[str, str]] = []
    for row in cell_rows:
        if row.get("row_type") != "scene" or row.get("method") != method:
            continue
        output.append(
            {
                "row_type": "target_scene",
                "target_scene_id": row["scene"],
                "shot": row["shot"],
                "scope": row["scope"],
                "method": method,
                "num_observations": str(
                    int(row["num_test_normal"]) + int(row["num_test_abnormal"])
                ),
                "auroc": row["auroc"],
                "auprc": row["auprc"],
                "fpr95": row["fpr95"],
            }
        )
    for row in macro_rows:
        if row.get("row_type") != "macro" or row.get("method") != method:
            continue
        output.append(
            {
                "row_type": "scene_macro",
                "target_scene_id": "ALL",
                "shot": row["shot"],
                "scope": row["scope"],
                "method": method,
                "num_observations": str(
                    int(row["num_test_normal"]) + int(row["num_test_abnormal"])
                ),
                "auroc": row["auroc"],
                "auprc": row["auprc"],
                "fpr95": row["fpr95"],
            }
        )
    if not output:
        raise RuntimeError(
            f"OFDMA {method}: no matching scene/macro rows in {cell_path} and {macro_path}"
        )
    return output, [cell_path, macro_path]


def validate_complete_method(
    rows: list[dict[str, str]], *, method: str, source: Path
) -> None:
    required_fields = {
        "row_type",
        "target_scene_id",
        "shot",
        "scope",
        "method",
        "num_observations",
        "auroc",
        "auprc",
        "fpr95",
    }
    for index, row in enumerate(rows, 2):
        missing = required_fields - set(row)
        if missing:
            raise RuntimeError(
                f"OFDMA {method}: {source} row {index} lacks fields {sorted(missing)}"
            )
        for metric in ("auroc", "auprc", "fpr95"):
            try:
                finite = math.isfinite(float(row[metric]))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                raise RuntimeError(
                    f"OFDMA {method}: non-finite {metric} at {source} row {index}"
                )

    expected_macro = {(str(shot), scope) for shot in SHOTS for scope in SCOPES}
    macros = [
        row
        for row in rows
        if row["row_type"] == "scene_macro" and row["target_scene_id"] == "ALL"
    ]
    macro_keys = [(row["shot"], row["scope"]) for row in macros]
    missing_macro = expected_macro - set(macro_keys)
    duplicates = sorted({key for key in macro_keys if macro_keys.count(key) > 1})
    unexpected = set(macro_keys) - expected_macro
    if missing_macro or duplicates or unexpected:
        raise RuntimeError(
            f"OFDMA {method} is incomplete at {source}: expected exactly one scene-macro "
            f"for every 1/2/4-shot × scope combination; missing={sorted(missing_macro)}, "
            f"duplicates={duplicates}, unexpected={sorted(unexpected)}"
        )

    scenes = [row for row in rows if row["row_type"] == "target_scene"]
    for shot, scope in sorted(expected_macro):
        selected = [
            row for row in scenes if row["shot"] == shot and row["scope"] == scope
        ]
        scene_ids = {row["target_scene_id"] for row in selected}
        if len(selected) != EXPECTED_TARGET_SCENES or len(scene_ids) != EXPECTED_TARGET_SCENES:
            raise RuntimeError(
                f"OFDMA {method} is incomplete at {source}: {shot}-shot/{scope} has "
                f"{len(selected)} rows over {len(scene_ids)} unique target scenes; "
                f"expected {EXPECTED_TARGET_SCENES}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--ours-root", type=Path, required=True)
    parser.add_argument(
        "--new-results-root",
        type=Path,
        default=None,
        help="Directory containing the 20260811 published-baseline runs "
        "(default: parent of --input-root).",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    new_results_root = args.new_results_root or args.input_root.parent
    rows_by_method: dict[str, list[dict[str, str]]] = {}
    source_files: dict[str, list[Path]] = {}
    for method, directory in LEGACY_METHOD_DIRS.items():
        root = args.ours_root if directory == "ours" else args.input_root / directory
        path = root / "results.csv"
        method_rows = canonicalize_rows(read_rows(path), expected_method=method, source=path)
        rows_by_method[method] = method_rows
        source_files[method] = [path]

    for method, directory in NEW_DIRECT_METHOD_DIRS.items():
        path = new_results_root / directory / "results.csv"
        method_rows = canonicalize_rows(read_rows(path), expected_method=method, source=path)
        rows_by_method[method] = method_rows
        source_files[method] = [path]

    information_root = new_results_root / INFORMATION_THEORETIC_DIR
    for method in ("kld_reference", "ica_frozen"):
        method_rows, paths = convert_information_theoretic_rows(
            information_root, method=method
        )
        rows_by_method[method] = method_rows
        source_files[method] = paths

    rows: list[dict[str, str]] = []
    for method in METHOD_ORDER:
        method_rows = rows_by_method.get(method)
        if method_rows is None:
            raise RuntimeError(f"OFDMA merge configuration omitted method {method}")
        validate_complete_method(
            method_rows, method=method, source=source_files[method][0]
        )
        for row in method_rows:
            row["method_display"] = METHOD_DISPLAY[method]
        rows.extend(method_rows)

    rows.sort(
        key=lambda row: (
            row["row_type"],
            int(row["shot"]),
            row["scope"],
            METHOD_INDEX[row["method"]],
            row["target_scene_id"],
        )
    )
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "unified_results.csv", rows)
    write_rows(
        output / "overall_scene_macro.csv",
        [row for row in rows if row["row_type"] == "scene_macro" and row["scope"] == "overall"],
    )
    write_rows(
        output / "per_jammer_scene_macro.csv",
        [row for row in rows if row["row_type"] == "scene_macro" and row["scope"] != "overall"],
    )

    protocol = {
        "dataset_protocol": "ofdma_target_scene_coldstart_v2_realistic",
        "input_root": str(args.input_root.resolve()),
        "ours_root": str(args.ours_root.resolve()),
        "new_results_root": str(new_results_root.resolve()),
        "methods": list(METHOD_ORDER),
        "source_files": {
            method: [str(path.resolve()) for path in source_files[method]]
            for method in METHOD_ORDER
        },
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# OFDMA v2 realistic unified results\n\n"
        "All methods use the same target-scene cold-start v2-realistic protocol; "
        "Ours is the ViT+CNN safe support-only confidence-gated method without "
        "the exploratory power-residual branch. Alternative test-time fitting "
        "variants are kept outside this main comparison.\n",
        encoding="utf-8",
    )
    print(f"[done] {output / 'unified_results.csv'}")


if __name__ == "__main__":
    main()
