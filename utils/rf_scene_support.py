"""Target-scene normal-support protocol for the self-collected RF dataset.

Normal backgrounds are scene-specific, while signal type and JSR describe the
injected anomaly.  This module therefore builds one content-deduplicated normal
support set per scene and reuses it for every signal/JSR cell in that scene.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import cv2

from datasets.rf_target import SUPPORT_TIME_RANGE, TEST_MIN_TIME_START
from utils.rf_frequency_sampling import frequency_band_key


MANIFEST_PROTOCOL = "rf_target_scene"
MANIFEST_VERSION = 2
CONTENT_IDENTITY = "sha256_decoded_pixels"
SCENE_SUPPORT_SAMPLING_CHOICES = (
    "per_frequency",
    "frequency_one_per_band",
    "1shot",
    "2shot",
    "4shot",
)
TIME_RANGE_RE = re.compile(r"_t(\d+)-(\d+)_")
TIME_BLOCK_ISOLATION = {
    "strategy": "reserved_support_window",
    "support_time_range": list(SUPPORT_TIME_RANGE),
    "test_min_time_start": TEST_MIN_TIME_START,
    "interval_semantics": "half_open",
}


def image_content_sha256(path: str | Path) -> str:
    """Hash decoded pixels, ignoring PNG metadata and encoding differences."""

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(image.tobytes(order="C"))
    return digest.hexdigest()


def _manifest_digest(manifest: Mapping) -> str:
    payload = copy.deepcopy(dict(manifest))
    payload.pop("manifest_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _time_range(path: str | Path) -> list[int] | None:
    match = TIME_RANGE_RE.search(Path(path).name)
    if not match:
        return None
    return [int(match.group(1)), int(match.group(2))]


def _record(path: str | Path, signal: str, scene: str, jsr: str, digest: str) -> dict:
    band = frequency_band_key(path)
    return {
        "path": str(path),
        "sha256": digest,
        "signal": signal,
        "scene": scene,
        "jsr": jsr,
        "frequency_band_mhz": list(band) if band is not None else None,
        "time_range": _time_range(path),
    }


def _seeded_key(record: Mapping, seed: int) -> tuple[str, str]:
    token = f"{int(seed)}:{record['sha256']}:{record['path']}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest(), str(record["path"])


def _deduplicate_by_content(records: Iterable[dict]) -> tuple[list[dict], int]:
    by_hash: dict[str, dict] = {}
    raw_count = 0
    for record in records:
        raw_count += 1
        digest = str(record["sha256"])
        current = by_hash.get(digest)
        if current is None or str(record["path"]) < str(current["path"]):
            by_hash[digest] = record
    unique = sorted(by_hash.values(), key=lambda item: str(item["path"]))
    return unique, raw_count - len(unique)


def _select_support(records: Sequence[dict], sampling: str, seed: int) -> list[dict]:
    if sampling not in SCENE_SUPPORT_SAMPLING_CHOICES:
        raise ValueError(
            f"Target-scene support does not allow {sampling!r}; "
            f"choose one of {SCENE_SUPPORT_SAMPLING_CHOICES}. Full-shot is intentionally disabled."
        )

    if sampling in {"per_frequency", "frequency_one_per_band"}:
        by_band: dict[tuple[float, float], list[dict]] = defaultdict(list)
        missing = []
        for record in records:
            band = record["frequency_band_mhz"]
            if band is None:
                missing.append(record["path"])
                continue
            by_band[(float(band[0]), float(band[1]))].append(record)
        if missing:
            raise ValueError(
                "Target-scene per-frequency support requires a frequency band in every filename; "
                f"missing examples={missing[:3]}"
            )
        selected = [
            min(by_band[band], key=lambda item: _seeded_key(item, seed))
            for band in sorted(by_band)
        ]
    else:
        count = int(sampling.removesuffix("shot"))
        selected = sorted(records, key=lambda item: _seeded_key(item, seed))[:count]

    if not selected:
        raise RuntimeError("No target-scene normal support images were selected")
    if len({item["sha256"] for item in selected}) != len(selected):
        raise AssertionError("Target-scene support contains duplicate image content")
    return selected


def build_rf_target_scene_manifest(
    *,
    collect_samples: Callable,
    signals: Sequence[str],
    scenes: Sequence[str],
    jsr_by_signal: Mapping[str, Sequence[str]],
    normal_sampling: str,
    seed: int,
) -> dict:
    """Build one deduplicated normal support set per target scene."""

    signals = list(signals)
    scenes = list(scenes)
    hash_cache: dict[str, str] = {}
    normal_by_cell: dict[tuple[str, str, str], list] = {}
    abnormal_by_cell: dict[tuple[str, str, str], list] = {}

    def digest(path: str | Path) -> str:
        key = str(path)
        if key not in hash_cache:
            hash_cache[key] = image_content_sha256(key)
        return hash_cache[key]

    for signal in signals:
        if signal not in jsr_by_signal:
            raise KeyError(f"Missing JSR definition for signal {signal!r}")
        for scene in scenes:
            for jsr in jsr_by_signal[signal]:
                normal_samples = collect_samples(signal, scene, jsr, "train", k_shot=0)
                if not normal_samples:
                    raise RuntimeError(f"No normal samples for {signal}/{scene}/{jsr}")
                test_samples = collect_samples(signal, scene, jsr, "test", k_shot=1)
                abnormal_samples = [
                    sample for sample in test_samples if int(sample[2]) == 1
                ]
                if not abnormal_samples:
                    raise RuntimeError(
                        f"No abnormal samples for {signal}/{scene}/{jsr}"
                    )
                normal_by_cell[(signal, scene, jsr)] = normal_samples
                abnormal_by_cell[(signal, scene, jsr)] = abnormal_samples

    scene_entries = []
    cell_entries = []
    for scene in scenes:
        candidate_records = []
        raw_scene_normal_count = 0
        for (signal, cell_scene, jsr), samples in normal_by_cell.items():
            if cell_scene != scene:
                continue
            raw_scene_normal_count += len(samples)
            for sample in samples:
                path = str(sample[0])
                record = _record(path, signal, scene, jsr, digest(path))
                if record["time_range"] == list(SUPPORT_TIME_RANGE):
                    candidate_records.append(record)

        unique_candidates, duplicate_count = _deduplicate_by_content(candidate_records)
        support = _select_support(unique_candidates, normal_sampling, seed)
        support_hashes = {item["sha256"] for item in support}
        support_paths = {item["path"] for item in support}
        excluded_content_copies = 0

        for signal in signals:
            for jsr in jsr_by_signal[signal]:
                normal_samples = normal_by_cell[(signal, scene, jsr)]
                abnormal_samples = abnormal_by_cell[(signal, scene, jsr)]
                test_normals = []
                test_abnormals = []
                excluded_content = 0
                excluded_temporal_normals = 0
                excluded_temporal_abnormals = 0
                for sample in normal_samples:
                    path = str(sample[0])
                    record = _record(path, signal, scene, jsr, digest(path))
                    time_range = record["time_range"]
                    if time_range is None:
                        raise ValueError(f"RF normal filename has no time range: {path}")
                    if int(time_range[0]) < TEST_MIN_TIME_START:
                        excluded_temporal_normals += 1
                        continue
                    image_hash = digest(path)
                    if image_hash in support_hashes:
                        excluded_content += 1
                        continue
                    test_normals.append(record)
                for sample in abnormal_samples:
                    path = str(sample[0])
                    record = _record(path, signal, scene, jsr, digest(path))
                    time_range = record["time_range"]
                    if time_range is None:
                        raise ValueError(f"RF abnormal filename has no time range: {path}")
                    if int(time_range[0]) < TEST_MIN_TIME_START:
                        excluded_temporal_abnormals += 1
                        continue
                    if record["sha256"] in support_hashes:
                        raise ValueError(
                            f"Abnormal test content matches support content: {path}"
                        )
                    test_abnormals.append(record)
                excluded_content_copies += excluded_content
                cell_entries.append(
                    {
                        "signal": signal,
                        "scene": scene,
                        "jsr": jsr,
                        "raw_normal_count": len(normal_samples),
                        "raw_abnormal_count": len(abnormal_samples),
                        "test_normal_count": len(test_normals),
                        "test_abnormal_count": len(test_abnormals),
                        "excluded_support_content_count": excluded_content,
                        "excluded_temporal_normal_count": excluded_temporal_normals,
                        "excluded_temporal_abnormal_count": excluded_temporal_abnormals,
                        "test_normals": test_normals,
                        "test_abnormals": test_abnormals,
                    }
                )

        scene_entries.append(
            {
                "scene": scene,
                "raw_scene_normal_count": raw_scene_normal_count,
                "raw_candidate_count": len(candidate_records),
                "unique_content_count": len(unique_candidates),
                "duplicate_candidate_count": duplicate_count,
                "support_count": len(support),
                "support_paths": sorted(support_paths),
                "support_hashes": sorted(support_hashes),
                "excluded_test_content_copies": excluded_content_copies,
                "support": support,
            }
        )

    manifest = {
        "protocol": MANIFEST_PROTOCOL,
        "version": MANIFEST_VERSION,
        "normal_sampling": normal_sampling,
        "seed": int(seed),
        "signals": signals,
        "scenes": scenes,
        "jsr_by_signal": {
            signal: list(jsr_by_signal[signal])
            for signal in signals
        },
        "content_identity": CONTENT_IDENTITY,
        "time_block_isolation": TIME_BLOCK_ISOLATION,
        "scene_support": scene_entries,
        "cells": cell_entries,
    }
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    validate_rf_target_scene_manifest(manifest)
    return manifest


def validate_rf_target_scene_manifest(manifest: Mapping) -> None:
    if manifest.get("protocol") != MANIFEST_PROTOCOL:
        raise ValueError(f"Unexpected support protocol: {manifest.get('protocol')!r}")
    if int(manifest.get("version", -1)) != MANIFEST_VERSION:
        raise ValueError(f"Unsupported target-scene manifest version: {manifest.get('version')!r}")
    if manifest.get("content_identity") != CONTENT_IDENTITY:
        raise ValueError(
            f"Unsupported content identity: {manifest.get('content_identity')!r}"
        )
    if manifest.get("time_block_isolation") != TIME_BLOCK_ISOLATION:
        raise ValueError(
            "Target-scene manifest does not use the required disjoint time blocks"
        )
    expected_digest = _manifest_digest(manifest)
    if manifest.get("manifest_sha256") != expected_digest:
        raise ValueError("Target-scene support manifest digest does not match its content")

    scene_hashes = {}
    for entry in manifest["scene_support"]:
        scene = entry["scene"]
        support = entry["support"]
        hashes = {item["sha256"] for item in support}
        if len(hashes) != len(support):
            raise ValueError(f"Duplicate support content in scene {scene}")
        if any(item["scene"] != scene for item in support):
            raise ValueError(f"Support scene mismatch in scene {scene}")
        if any(item["time_range"] != list(SUPPORT_TIME_RANGE) for item in support):
            raise ValueError(
                f"Support outside reserved time range in scene {scene}"
            )
        scene_hashes[scene] = hashes

    expected_cells = {
        (signal, scene, jsr)
        for signal in manifest["signals"]
        for scene in manifest["scenes"]
        for jsr in manifest["jsr_by_signal"][signal]
    }
    actual_cells = {
        (cell["signal"], cell["scene"], cell["jsr"])
        for cell in manifest["cells"]
    }
    if actual_cells != expected_cells:
        missing = sorted(expected_cells - actual_cells)
        extra = sorted(actual_cells - expected_cells)
        raise ValueError(f"Manifest cell mismatch; missing={missing}, extra={extra}")

    for cell in manifest["cells"]:
        support_hashes = scene_hashes[cell["scene"]]
        test_items = [*cell["test_normals"], *cell["test_abnormals"]]
        for item in test_items:
            time_range = item.get("time_range")
            if time_range is None or int(time_range[0]) < TEST_MIN_TIME_START:
                raise ValueError(
                    f"Test sample overlaps the support time block: {item.get('path')}"
                )
        test_hashes = {item["sha256"] for item in test_items}
        overlap = support_hashes & test_hashes
        if overlap:
            raise ValueError(
                f"Support/test content overlap in "
                f"{cell['signal']}/{cell['scene']}/{cell['jsr']}: {len(overlap)} hashes"
            )
        if int(cell["test_normal_count"]) != len(cell["test_normals"]):
            raise ValueError("test_normal_count does not match test_normals")
        if int(cell["test_abnormal_count"]) != len(cell["test_abnormals"]):
            raise ValueError("test_abnormal_count does not match test_abnormals")
        if not cell["test_normals"] or not cell["test_abnormals"]:
            raise ValueError(
                f"Empty isolated test class in "
                f"{cell['signal']}/{cell['scene']}/{cell['jsr']}"
            )


def write_rf_target_scene_manifest(path: str | Path, manifest: Mapping) -> Path:
    validate_rf_target_scene_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)
    return path


def load_rf_target_scene_manifest(path: str | Path) -> dict:
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_rf_target_scene_manifest(manifest)
    return manifest


def ensure_rf_target_scene_manifest(
    path: str | Path,
    *,
    collect_samples: Callable,
    signals: Sequence[str],
    scenes: Sequence[str],
    jsr_by_signal: Mapping[str, Sequence[str]],
    normal_sampling: str,
    seed: int,
) -> dict:
    path = Path(path)
    if path.exists():
        manifest = load_rf_target_scene_manifest(path)
        requested_signals = list(signals)
        requested_scenes = list(scenes)
        manifest_signals = set(manifest["signals"])
        manifest_scenes = set(manifest["scenes"])
        mismatches = {}
        if manifest.get("normal_sampling") != normal_sampling:
            mismatches["normal_sampling"] = (
                manifest.get("normal_sampling"),
                normal_sampling,
            )
        if manifest.get("seed") != int(seed):
            mismatches["seed"] = (manifest.get("seed"), int(seed))
        missing_signals = sorted(set(requested_signals) - manifest_signals)
        if missing_signals:
            mismatches["missing_signals"] = missing_signals
        missing_scenes = sorted(set(requested_scenes) - manifest_scenes)
        if missing_scenes:
            mismatches["missing_scenes"] = missing_scenes
        jsr_mismatches = {
            signal: (
                manifest["jsr_by_signal"].get(signal),
                list(jsr_by_signal[signal]),
            )
            for signal in requested_signals
            if signal in manifest_signals
            and manifest["jsr_by_signal"].get(signal)
            != list(jsr_by_signal[signal])
        }
        if jsr_mismatches:
            mismatches["jsr_by_signal"] = jsr_mismatches
        if mismatches:
            raise ValueError(
                f"Existing target-scene support manifest does not match requested protocol: {mismatches}"
            )
        return manifest

    manifest = build_rf_target_scene_manifest(
        collect_samples=collect_samples,
        signals=signals,
        scenes=scenes,
        jsr_by_signal=jsr_by_signal,
        normal_sampling=normal_sampling,
        seed=seed,
    )
    write_rf_target_scene_manifest(path, manifest)
    return manifest


def scene_support_entry(manifest: Mapping, scene: str) -> dict:
    matches = [entry for entry in manifest["scene_support"] if entry["scene"] == scene]
    if len(matches) != 1:
        raise KeyError(f"Expected one support entry for scene {scene!r}, found {len(matches)}")
    return matches[0]


def cell_entry(manifest: Mapping, signal: str, scene: str, jsr: str) -> dict:
    matches = [
        entry
        for entry in manifest["cells"]
        if entry["signal"] == signal and entry["scene"] == scene and entry["jsr"] == jsr
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one manifest cell for {signal}/{scene}/{jsr}, found {len(matches)}"
        )
    return matches[0]
