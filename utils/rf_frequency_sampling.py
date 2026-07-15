"""Helpers for selecting RF normal samples by frequency band."""

from __future__ import annotations

import re
import hashlib
from pathlib import Path
from typing import Iterable, Sequence, TypeVar


T = TypeVar("T")

FREQ_RE = re.compile(r"_f([0-9.]+)-([0-9.]+)MHz")
NORMAL_SAMPLING_CHOICES = (
    "all",
    "per_frequency",
    "frequency_one_per_band",
    "1shot",
    "2shot",
    "4shot",
    "split_75_25",
)


def frequency_band_key(path) -> tuple[float, float] | None:
    match = FREQ_RE.search(Path(path).name)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def select_one_per_frequency_band(items: Sequence[T], path_getter=lambda item: item) -> list[T]:
    """Keep one item per parsed RF frequency band.

    If no item has a parseable frequency band, return the original items. This
    keeps non-RF datasets such as `datasets/spectrum` unchanged.
    """
    keyed = []
    unkeyed = []
    for item in items:
        path = path_getter(item)
        key = frequency_band_key(path)
        if key is None:
            unkeyed.append(item)
        else:
            keyed.append((key, str(path), item))

    if not keyed:
        return list(items)

    selected = []
    seen = set()
    for key, _path, item in sorted(keyed, key=lambda row: (row[0], row[1])):
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
    return selected


def select_first_n(items: Sequence[T], n: int, path_getter=lambda item: item) -> list[T]:
    """Select the first ``n`` items in deterministic path order."""
    keyed = [(str(path_getter(item)), item) for item in items]
    return [item for _path, item in sorted(keyed, key=lambda row: row[0])[:n]]


def split_train_test_normals(
    items: Sequence[T],
    train_ratio: float = 0.75,
    seed: int = 111,
    path_getter=lambda item: item,
) -> tuple[list[T], list[T]]:
    """Deterministically split normal samples into train/test subsets.

    The split is path-hash based so the same cell receives the same partition
    regardless of caller order. For cells with more than one normal, at least
    one sample is kept for each side.
    """
    items = list(items)
    if not items:
        return [], []
    if len(items) == 1:
        return items, []

    def key(item: T) -> tuple[str, str]:
        path = str(path_getter(item))
        digest = hashlib.sha1(f"{seed}:{path}".encode("utf-8")).hexdigest()
        return digest, path

    ordered = sorted(items, key=key)
    train_count = int(round(len(ordered) * float(train_ratio)))
    train_count = min(max(1, train_count), len(ordered) - 1)
    return ordered[:train_count], ordered[train_count:]


def maybe_select_one_per_frequency_band(items: Sequence[T], mode: str, path_getter=lambda item: item) -> list[T]:
    if mode in {"all", "", None}:
        return list(items)
    if mode == "split_75_25":
        train_items, _test_items = split_train_test_normals(items, path_getter=path_getter)
        return train_items
    if mode in {"frequency_one_per_band", "per_frequency"}:
        return select_one_per_frequency_band(items, path_getter=path_getter)
    if mode in {"1shot", "2shot", "4shot"}:
        return select_first_n(items, int(mode.removesuffix("shot")), path_getter=path_getter)
    raise ValueError(f"Unsupported normal sampling mode: {mode}")
