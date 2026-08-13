"""Shared support/test split for repeated public-RF experiments.

The public RF directory contains many complete normal measurement records.  A
support replicate must not change the test set, so the protocol reserves a
fixed, model-independent pool of records for support sampling and keeps every
image from the remaining records in the test set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from utils.rf_frequency_sampling import (
    frequency_band_key,
    maybe_select_one_per_frequency_band,
    select_k_per_frequency_band,
)


PUBLIC_RF_ROOT = Path("/mnt/data/wangbei/data/RF_SPE_PNG")
PUBLIC_RF_NORMAL_DIR = PUBLIC_RF_ROOT / "RF_Spectrum_Public_Dataset"
SUPPORT_POOL_RECORD_COUNT = 20
MANIFEST_PROTOCOL = "public_rf_fixed_test_support_pool"
MANIFEST_VERSION = 1


def public_normal_records() -> list[Path]:
    records = sorted(
        path
        for path in PUBLIC_RF_NORMAL_DIR.iterdir()
        if path.is_dir() and path.name.startswith("MeasRes_")
    )
    if not records:
        raise FileNotFoundError(
            f"No public normal records found under {PUBLIC_RF_NORMAL_DIR}"
        )
    return records


def public_normal_paths(records: Sequence[Path] | None = None) -> list[Path]:
    records = list(public_normal_records() if records is None else records)
    paths = [image for record in records for image in sorted(record.glob("*.png"))]
    if not paths:
        raise FileNotFoundError(
            f"No public normal images found under {PUBLIC_RF_NORMAL_DIR}"
        )
    return paths


def _paths_digest(paths: Sequence[str | Path]) -> str:
    payload = "\n".join(str(path) for path in paths).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_digest(payload: Mapping) -> str:
    encoded = json.dumps(
        dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(*, seed: int, normal_sampling: str = "per_frequency") -> dict:
    records = public_normal_records()
    if len(records) <= SUPPORT_POOL_RECORD_COUNT:
        raise ValueError(
            "Public RF must contain more records than the fixed support pool; "
            f"got {len(records)} records and pool size {SUPPORT_POOL_RECORD_COUNT}."
        )
    pool_records = records[:SUPPORT_POOL_RECORD_COUNT]
    test_records = records[SUPPORT_POOL_RECORD_COUNT:]
    pool_paths = public_normal_paths(pool_records)
    test_paths = public_normal_paths(test_records)
    support_paths = maybe_select_one_per_frequency_band(
        pool_paths, normal_sampling, seed=int(seed)
    )
    support_strings = [str(path) for path in support_paths]
    pool_strings = [str(path) for path in pool_paths]
    test_strings = [str(path) for path in test_paths]
    payload = {
        "protocol": MANIFEST_PROTOCOL,
        "version": MANIFEST_VERSION,
        "normal_sampling": str(normal_sampling),
        "seed": int(seed),
        "support_pool_record_count": SUPPORT_POOL_RECORD_COUNT,
        "support_pool_records": [str(path) for path in pool_records],
        "test_records": [str(path) for path in test_records],
        "candidate_count": len(public_normal_paths(records)),
        "support_pool_count": len(pool_strings),
        "support_count": len(support_strings),
        "test_normal_count": len(test_strings),
        "support_pool_paths": pool_strings,
        "support_paths": support_strings,
        "test_paths": test_strings,
        "support_paths_sha256": _paths_digest(support_strings),
        "test_paths_sha256": _paths_digest(test_strings),
    }
    payload["manifest_sha256"] = _manifest_digest(payload)
    return payload


def build_k_per_frequency_manifest(*, seed: int, k: int) -> dict:
    """Build a fixed-test manifest with ``k`` normal images per RF band.

    This is an additive exploratory protocol.  ``normal_sampling`` remains
    ``per_frequency`` for compatibility with existing evaluators, while the
    explicit ``support_policy`` and ``per_frequency_k`` fields record that
    the support set contains k samples in every band.
    """
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    records = public_normal_records()
    if len(records) <= SUPPORT_POOL_RECORD_COUNT:
        raise ValueError(
            "Public RF must contain more records than the fixed support pool; "
            f"got {len(records)} records and pool size {SUPPORT_POOL_RECORD_COUNT}."
        )
    pool_records = records[:SUPPORT_POOL_RECORD_COUNT]
    test_records = records[SUPPORT_POOL_RECORD_COUNT:]
    pool_paths = public_normal_paths(pool_records)
    test_paths = public_normal_paths(test_records)
    support_paths = select_k_per_frequency_band(pool_paths, k=k, seed=int(seed))
    support_strings = [str(path) for path in support_paths]
    pool_strings = [str(path) for path in pool_paths]
    test_strings = [str(path) for path in test_paths]
    bands = sorted({frequency_band_key(path) for path in pool_paths})
    payload = {
        "protocol": MANIFEST_PROTOCOL,
        "version": MANIFEST_VERSION,
        "normal_sampling": "per_frequency",
        "support_policy": "k_per_frequency",
        "per_frequency_k": k,
        "seed": int(seed),
        "support_pool_record_count": SUPPORT_POOL_RECORD_COUNT,
        "support_pool_records": [str(path) for path in pool_records],
        "test_records": [str(path) for path in test_records],
        "frequency_band_count": len(bands),
        "frequency_bands_mhz": [list(band) for band in bands],
        "candidate_count": len(public_normal_paths(records)),
        "support_pool_count": len(pool_strings),
        "support_count": len(support_strings),
        "test_normal_count": len(test_strings),
        "support_pool_paths": pool_strings,
        "support_paths": support_strings,
        "test_paths": test_strings,
        "support_paths_sha256": _paths_digest(support_strings),
        "test_paths_sha256": _paths_digest(test_strings),
    }
    payload["manifest_sha256"] = _manifest_digest(payload)
    return payload


def load_manifest(path: str | Path) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != MANIFEST_PROTOCOL:
        raise ValueError(
            f"Unexpected public-RF support protocol in {path}: {payload.get('protocol')!r}"
        )
    if int(payload.get("version", -1)) != MANIFEST_VERSION:
        raise ValueError(f"Unsupported public-RF support manifest version in {path}")
    digest_payload = dict(payload)
    digest_payload.pop("manifest_sha256", None)
    if payload.get("manifest_sha256") != _manifest_digest(digest_payload):
        raise ValueError(f"Public-RF support manifest digest mismatch: {path}")
    support = [str(value) for value in payload.get("support_paths", [])]
    test = [str(value) for value in payload.get("test_paths", [])]
    if not support or not test:
        raise ValueError(f"Public-RF support manifest has an empty split: {path}")
    if len(set(support)) != len(support) or len(set(test)) != len(test):
        raise ValueError(f"Public-RF support manifest contains duplicate paths: {path}")
    if set(support) & set(test):
        raise ValueError(f"Public-RF support/test overlap in {path}")
    if payload.get("support_paths_sha256") != _paths_digest(support):
        raise ValueError(f"Public-RF support path digest mismatch: {path}")
    if payload.get("test_paths_sha256") != _paths_digest(test):
        raise ValueError(f"Public-RF test path digest mismatch: {path}")
    return payload


def select_support_and_test_paths(
    normal_paths: Sequence[Path],
    *,
    normal_sampling: str,
    seed: int | None,
    manifest_path: str | Path | None = None,
) -> tuple[list[Path], list[Path], dict | None]:
    """Return support and fixed test paths, validating a shared manifest."""

    all_paths = {str(path) for path in normal_paths}
    if manifest_path:
        manifest = load_manifest(manifest_path)
        if manifest.get("normal_sampling") != str(normal_sampling):
            raise ValueError(
                "Public-RF support manifest sampling does not match the evaluator: "
                f"{manifest.get('normal_sampling')!r} vs {normal_sampling!r}"
            )
        if seed is not None and int(manifest["seed"]) != int(seed):
            raise ValueError(
                "Public-RF support manifest seed does not match the evaluator: "
                f"{manifest['seed']} vs {seed}"
            )
        support_strings = [str(value) for value in manifest["support_paths"]]
        test_strings = [str(value) for value in manifest["test_paths"]]
        if not set(support_strings).issubset(all_paths):
            raise ValueError("Public-RF support manifest contains paths outside the dataset")
        if not set(test_strings).issubset(all_paths):
            raise ValueError("Public-RF test manifest contains paths outside the dataset")
        pool_strings = {str(value) for value in manifest.get("support_pool_paths", [])}
        if pool_strings and not pool_strings.issubset(all_paths):
            raise ValueError("Public-RF support pool contains paths outside the dataset")
        return (
            [Path(value) for value in support_strings],
            [Path(value) for value in test_strings],
            manifest,
        )

    support = maybe_select_one_per_frequency_band(
        list(normal_paths), normal_sampling, seed=seed
    )
    support_set = {str(path) for path in support}
    test = [path for path in normal_paths if str(path) not in support_set]
    return list(support), test, None
