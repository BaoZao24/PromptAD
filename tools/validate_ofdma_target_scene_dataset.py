#!/usr/bin/env python
"""Validate and consolidate a generated target-scene OFDMA dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_scene(
    root: Path,
    scene_dir: Path,
    split: str,
    observation_counts: dict[str, int],
    jammer_types: list[str],
    expected_sus: int,
    check_images: bool,
) -> tuple[list[dict[str, str]], list[str], dict[str, Any]]:
    errors: list[str] = []
    scene_id = scene_dir.name
    context_path = scene_dir / "context.json"
    manifest_path = scene_dir / "manifest.csv"
    completion_path = scene_dir / "_COMPLETE.json"
    require(context_path.is_file(), f"{scene_id}: missing context.json", errors)
    require(manifest_path.is_file(), f"{scene_id}: missing manifest.csv", errors)
    require(completion_path.is_file(), f"{scene_id}: missing _COMPLETE.json", errors)
    if errors:
        return [], errors, {"target_scene_id": scene_id, "split": split}

    context = read_json(context_path)
    rows = read_csv(manifest_path)
    require(context["target_scene_id"] == scene_id, f"{scene_id}: context id mismatch", errors)
    require(context["split"] == split, f"{scene_id}: split mismatch", errors)

    by_observation: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_observation[row["observation_id"]].append(row)

    expected_observations = (
        observation_counts["normal_support"]
        + observation_counts["normal_test"]
        + len(jammer_types) * observation_counts["anomaly_test_per_type"]
    )
    require(
        len(by_observation) == expected_observations,
        f"{scene_id}: expected {expected_observations} observations, got {len(by_observation)}",
        errors,
    )
    require(
        len(rows) == expected_observations * expected_sus,
        f"{scene_id}: expected {expected_observations * expected_sus} image rows, got {len(rows)}",
        errors,
    )

    role_counts: Counter[str] = Counter()
    jammer_counts: Counter[str] = Counter()
    normal_resources: dict[str, str] = {}
    support_ranks: list[int] = []
    image_paths: set[str] = set()
    for observation_id, observation_rows in by_observation.items():
        first = observation_rows[0]
        role_counts[first["role"]] += 1
        if first["role"] == "anomaly_test":
            jammer_counts[first["jammer_type"]] += 1
        if first["role"] == "normal_test":
            normal_resources[observation_id] = first["resource_fingerprint"]
        if first["role"] == "normal_support":
            support_ranks.append(int(first["support_rank"]))

        require(
            len(observation_rows) == expected_sus,
            f"{scene_id}/{observation_id}: expected {expected_sus} SUs, got {len(observation_rows)}",
            errors,
        )
        su_ids = {int(row["su_id"]) for row in observation_rows}
        require(
            su_ids == set(range(expected_sus)),
            f"{scene_id}/{observation_id}: invalid SU ids {sorted(su_ids)}",
            errors,
        )
        require(
            len({row["context_fingerprint"] for row in observation_rows}) == 1
            and first["context_fingerprint"] == context["context_fingerprint"],
            f"{scene_id}/{observation_id}: context fingerprint mismatch",
            errors,
        )
        require(
            len({row["resource_fingerprint"] for row in observation_rows}) == 1,
            f"{scene_id}/{observation_id}: resource fingerprint mismatch within observation",
            errors,
        )
        expected_label = "1" if first["role"] == "anomaly_test" else "0"
        require(
            {row["label"] for row in observation_rows} == {expected_label},
            f"{scene_id}/{observation_id}: invalid label for role {first['role']}",
            errors,
        )
        for row in observation_rows:
            relative_path = row["image_path"]
            require(
                relative_path not in image_paths,
                f"{scene_id}: duplicate image path {relative_path}",
                errors,
            )
            image_paths.add(relative_path)
            image_path = root / relative_path
            require(
                image_path.is_file(),
                f"{scene_id}: missing image {relative_path}",
                errors,
            )
            if check_images and image_path.is_file():
                with Image.open(image_path) as image:
                    require(
                        image.mode == "L",
                        f"{scene_id}: non-grayscale image {relative_path}: {image.mode}",
                        errors,
                    )
                    require(
                        image.size == (70, 1320),
                        f"{scene_id}: unexpected image size {relative_path}: {image.size}",
                        errors,
                    )

    require(
        role_counts["normal_support"] == observation_counts["normal_support"],
        f"{scene_id}: wrong support count {role_counts['normal_support']}",
        errors,
    )
    require(
        role_counts["normal_test"] == observation_counts["normal_test"],
        f"{scene_id}: wrong normal-test count {role_counts['normal_test']}",
        errors,
    )
    require(
        role_counts["anomaly_test"]
        == len(jammer_types) * observation_counts["anomaly_test_per_type"],
        f"{scene_id}: wrong anomaly-test count {role_counts['anomaly_test']}",
        errors,
    )
    require(
        sorted(support_ranks)
        == list(range(1, observation_counts["normal_support"] + 1)),
        f"{scene_id}: support ranks are not nested: {sorted(support_ranks)}",
        errors,
    )
    for jammer_type in jammer_types:
        require(
            jammer_counts[jammer_type]
            == observation_counts["anomaly_test_per_type"],
            f"{scene_id}: wrong {jammer_type} count {jammer_counts[jammer_type]}",
            errors,
        )

    for observation_id, observation_rows in by_observation.items():
        first = observation_rows[0]
        if first["role"] != "anomaly_test":
            continue
        paired_id = first["paired_normal_observation_id"]
        require(
            paired_id in normal_resources,
            f"{scene_id}/{observation_id}: missing paired normal {paired_id}",
            errors,
        )
        if paired_id in normal_resources:
            require(
                first["resource_fingerprint"] == normal_resources[paired_id],
                f"{scene_id}/{observation_id}: paired normal resource mismatch",
                errors,
            )

    summary = {
        "target_scene_id": scene_id,
        "split": split,
        "context_seed": context["context_seed"],
        "context_fingerprint": context["context_fingerprint"],
        "observations": len(by_observation),
        "images": len(rows),
        "role_counts": dict(role_counts),
        "jammer_counts": dict(jammer_counts),
    }
    return rows, errors, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a generated target-scene OFDMA dataset."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Open every image and verify its mode and dimensions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    effective = read_json(protocol_path)
    source = effective["source_protocol"]
    scene_counts = effective["effective_scene_counts"]
    observation_counts = effective["effective_observations_per_scene"]
    jammer_types = source["jammer_types"]
    expected_sus = int(source["sensing_units"])

    all_rows: list[dict[str, str]] = []
    errors: list[str] = []
    scene_summaries: list[dict[str, Any]] = []
    context_seeds: set[int] = set()
    for split, expected_scene_count in scene_counts.items():
        split_dir = root / "scenes" / split
        scene_dirs = sorted(path for path in split_dir.glob("*") if path.is_dir())
        require(
            len(scene_dirs) == expected_scene_count,
            f"{split}: expected {expected_scene_count} scenes, got {len(scene_dirs)}",
            errors,
        )
        for scene_dir in scene_dirs:
            rows, scene_errors, summary = validate_scene(
                root,
                scene_dir,
                split,
                observation_counts,
                jammer_types,
                expected_sus,
                check_images=args.check_images,
            )
            errors.extend(scene_errors)
            all_rows.extend(rows)
            scene_summaries.append(summary)
            if "context_seed" in summary:
                require(
                    summary["context_seed"] not in context_seeds,
                    f"duplicate context seed {summary['context_seed']}",
                    errors,
                )
                context_seeds.add(summary["context_seed"])

    require(
        len({row["image_path"] for row in all_rows}) == len(all_rows),
        "global manifest contains duplicate image paths",
        errors,
    )
    report = {
        "status": "passed" if not errors else "failed",
        "root": str(root),
        "scenes": len(scene_summaries),
        "observations": sum(item.get("observations", 0) for item in scene_summaries),
        "images": len(all_rows),
        "check_images": args.check_images,
        "errors": errors,
        "scene_summaries": scene_summaries,
    }
    write_json_atomic(root / "validation_report.json", report)
    if errors:
        for error in errors[:100]:
            print(f"ERROR: {error}")
        raise SystemExit(f"Dataset validation failed with {len(errors)} errors")
    if not all_rows:
        raise SystemExit("Dataset contains no manifest rows")
    write_csv_atomic(root / "manifest.csv", all_rows)
    print(json.dumps({key: report[key] for key in ("status", "scenes", "observations", "images")}))


if __name__ == "__main__":
    main()
