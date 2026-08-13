#!/usr/bin/env python3
"""Merge scene-sharded OFDMA baseline runs into one formal result bundle.

The OFDMA evaluators intentionally process one target scene at a time.  This
utility combines independent scene shards after they finish, recomputing only
the same unweighted scene macro rows used by the evaluators.  It never reads
test labels except through the already-written per-scene metric rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


SCOPES = ("overall", "barrage", "deceptive", "pilot", "random_hop", "sweep")
SHOTS = (1, 2, 4)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def chunk_dirs(root: Path, filename: str) -> list[Path]:
    paths = sorted(path for path in root.iterdir() if path.is_dir() and (path / filename).is_file())
    if not paths:
        raise RuntimeError(f"No chunk directories containing {filename}: {root}")
    return paths


def finite_mean(rows: list[dict], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def copy_scores(chunks: list[Path], output: Path) -> None:
    destination = output / "scores"
    destination.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        source = chunk / "scores"
        if not source.is_dir():
            continue
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, destination / path.name)


def merge_direct(chunks: list[Path], output: Path, expected_scenes: int) -> None:
    rows: list[dict[str, str]] = []
    protocol = None
    for chunk in chunks:
        rows.extend(row for row in read_csv(chunk / "results.csv") if row["row_type"] == "target_scene")
        if protocol is None and (chunk / "protocol.json").is_file():
            protocol = json.loads((chunk / "protocol.json").read_text(encoding="utf-8"))
    key = lambda row: (row["target_scene_id"], int(row["shot"]), row["scope"], row["method"])
    rows.sort(key=key)
    keys = [key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate direct-method scene rows found across chunks")
    scenes = {row["target_scene_id"] for row in rows}
    expected = {(scene, shot, scope, rows[0]["method"]) for scene in scenes for shot in SHOTS for scope in SCOPES}
    actual = set(keys)
    methods = {row["method"] for row in rows}
    expected = {(scene, shot, scope, method) for scene in scenes for shot in SHOTS for scope in SCOPES for method in methods}
    if len(scenes) != expected_scenes or actual != expected:
        raise RuntimeError(
            f"Incomplete direct merge: scenes={len(scenes)}, methods={sorted(methods)}, "
            f"rows={len(rows)}, expected={len(expected)}"
        )
    macro: list[dict] = []
    for shot in SHOTS:
        for scope in SCOPES:
            for method in sorted(methods):
                selected = [row for row in rows if int(row["shot"]) == shot and row["scope"] == scope and row["method"] == method]
                macro.append(
                    {
                        "row_type": "scene_macro",
                        "target_scene_id": "ALL",
                        "shot": shot,
                        "scope": scope,
                        "method": method,
                        "num_observations": sum(int(row["num_observations"]) for row in selected),
                        "auroc": finite_mean(selected, "auroc"),
                        "auprc": finite_mean(selected, "auprc"),
                        "fpr95": finite_mean(selected, "fpr95"),
                    }
                )
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "results.csv", rows + macro)
    copy_scores(chunks, output)
    merged_protocol = dict(protocol or {})
    raw_dataset_protocol = merged_protocol.get("dataset_protocol", {})
    dataset_protocol = (
        dict(raw_dataset_protocol)
        if isinstance(raw_dataset_protocol, dict)
        else {"dataset_protocol": raw_dataset_protocol}
    )
    dataset_protocol["target_scene_ids"] = sorted(scenes)
    dataset_protocol["num_jobs"] = len(scenes) * len(SHOTS)
    merged_protocol["dataset_protocol"] = dataset_protocol
    merged_protocol["target_scene_ids"] = sorted(scenes)
    merged_protocol.update({"merged_from_chunks": [str(path.resolve()) for path in chunks], "formal_full_test": True})
    (output / "protocol.json").write_text(json.dumps(merged_protocol, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({"status": "complete", "num_scene_rows": len(rows), "num_macro_rows": len(macro)}, indent=2) + "\n", encoding="utf-8")


def merge_information(chunks: list[Path], output: Path, expected_scenes: int) -> None:
    rows: list[dict[str, str]] = []
    protocol = None
    for chunk in chunks:
        rows.extend(row for row in read_csv(chunk / "metrics_per_cell.csv") if row["row_type"] == "scene")
        if protocol is None and (chunk / "protocol.json").is_file():
            protocol = json.loads((chunk / "protocol.json").read_text(encoding="utf-8"))
    key = lambda row: (row["scene"], int(row["shot"]), row["scope"], row["method"])
    rows.sort(key=key)
    keys = [key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate information-theoretic scene rows found across chunks")
    scenes = {row["scene"] for row in rows}
    methods = {row["method"] for row in rows}
    expected = {(scene, shot, scope, method) for scene in scenes for shot in SHOTS for scope in SCOPES for method in methods}
    if len(scenes) != expected_scenes or set(keys) != expected:
        raise RuntimeError(
            f"Incomplete information-theoretic merge: scenes={len(scenes)}, methods={sorted(methods)}, "
            f"rows={len(rows)}, expected={len(expected)}"
        )
    macro: list[dict] = []
    for shot in SHOTS:
        for scope in SCOPES:
            for method in sorted(methods):
                selected = [row for row in rows if int(row["shot"]) == shot and row["scope"] == scope and row["method"] == method]
                first = selected[0]
                macro.append(
                    {
                        "row_type": "macro",
                        "dataset": first["dataset"],
                        "category": "ALL",
                        "scene": "ALL",
                        "jsr": "ALL",
                        "shot": shot,
                        "scope": scope,
                        "method": method,
                        "method_display": first["method_display"],
                        "num_support": "",
                        "num_test_normal": sum(int(row["num_test_normal"]) for row in selected),
                        "num_test_abnormal": sum(int(row["num_test_abnormal"]) for row in selected),
                        "auroc": finite_mean(selected, "auroc"),
                        "auprc": finite_mean(selected, "auprc"),
                        "fpr95": finite_mean(selected, "fpr95"),
                    }
                )
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "metrics_per_cell.csv", rows)
    write_csv(output / "metrics_macro.csv", macro)
    copy_scores(chunks, output)
    merged_protocol = dict(protocol or {})
    dataset_protocol = dict(merged_protocol.get("dataset_protocol", {}))
    dataset_protocol["target_scene_ids"] = sorted(scenes)
    dataset_protocol["num_jobs"] = len(scenes) * len(SHOTS)
    merged_protocol["dataset_protocol"] = dataset_protocol
    merged_protocol.update({"merged_from_chunks": [str(path.resolve()) for path in chunks], "formal_full_test": True})
    (output / "protocol.json").write_text(json.dumps(merged_protocol, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({"status": "complete", "num_scene_rows": len(rows), "num_macro_rows": len(macro)}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("direct", "information"), required=True)
    parser.add_argument("--chunk-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-scenes", type=int, default=30)
    args = parser.parse_args()
    if args.expected_scenes <= 0:
        raise ValueError("--expected-scenes must be positive")
    filename = "results.csv" if args.kind == "direct" else "metrics_per_cell.csv"
    chunks = chunk_dirs(args.chunk_root, filename)
    if args.kind == "direct":
        merge_direct(chunks, args.output_root, args.expected_scenes)
    else:
        merge_information(chunks, args.output_root, args.expected_scenes)
    print(f"[done] merged {len(chunks)} chunks into {args.output_root}")


if __name__ == "__main__":
    main()
