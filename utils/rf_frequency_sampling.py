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


def select_one_per_frequency_band(
    items: Sequence[T],
    path_getter=lambda item: item,
    seed: int | None = None,
) -> list[T]:
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
    grouped: dict[tuple[float, float], list[tuple[tuple[float, float], str, T]]] = {}
    for row in keyed:
        grouped.setdefault(row[0], []).append(row)
    for key in sorted(grouped):
        candidates = grouped[key]
        if seed is None:
            chosen = min(candidates, key=lambda row: row[1])
        else:
            # Hash-based ordering is independent of caller order and Python's
            # process-level hash randomization.  The same support seed is
            # therefore shared exactly by all branch evaluators.
            chosen = min(
                candidates,
                key=lambda row: hashlib.sha256(
                    f"{int(seed)}:{row[1]}".encode("utf-8")
                ).hexdigest(),
            )
        if key in seen:
            continue
        seen.add(key)
        selected.append(chosen[2])
    return selected


def select_k_per_frequency_band(
    items: Sequence[T],
    k: int,
    path_getter=lambda item: item,
    seed: int | None = None,
) -> list[T]:
    """Keep ``k`` distinct items per parsed RF frequency band.

    Candidate order is deterministic and independent of the caller's input
    order.  Consequently, selections are nested: the first item selected for
    a band at ``k=1`` is also selected at ``k=2`` and ``k=4`` when the same
    seed is used.  The function is deliberately strict about malformed RF
    inputs so a k-per-frequency experiment cannot silently fall back to a
    global shot count.
    """
    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    grouped: dict[tuple[float, float], list[tuple[str, T]]] = {}
    unkeyed = []
    for item in items:
        path = path_getter(item)
        key = frequency_band_key(path)
        if key is None:
            unkeyed.append(str(path))
            continue
        grouped.setdefault(key, []).append((str(path), item))

    if not grouped:
        raise ValueError("k-per-frequency requires parseable RF frequency bands")
    if unkeyed:
        raise ValueError(
            "k-per-frequency received paths without a parseable frequency band: "
            + ", ".join(sorted(unkeyed)[:3])
        )

    selected = []
    for key in sorted(grouped):
        candidates = grouped[key]
        if seed is None:
            ordered = sorted(candidates, key=lambda row: row[0])
        else:
            ordered = sorted(
                candidates,
                key=lambda row: (
                    hashlib.sha256(
                        f"{int(seed)}:{row[0]}".encode("utf-8")
                    ).hexdigest(),
                    row[0],
                ),
            )
        if len(ordered) < int(k):
            raise ValueError(
                f"Frequency band {key[0]:.5f}-{key[1]:.5f}MHz has only "
                f"{len(ordered)} candidates; cannot select k={int(k)}"
            )
        selected.extend(item for _path, item in ordered[: int(k)])
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


def maybe_select_one_per_frequency_band(
    items: Sequence[T],
    mode: str,
    path_getter=lambda item: item,
    seed: int | None = None,
) -> list[T]:
    if mode in {"all", "", None}:
        return list(items)
    if mode == "split_75_25":
        train_items, _test_items = split_train_test_normals(items, path_getter=path_getter)
        return train_items
    if mode in {"frequency_one_per_band", "per_frequency"}:
        return select_one_per_frequency_band(items, path_getter=path_getter, seed=seed)
    if mode in {"1shot", "2shot", "4shot"}:
        return select_first_n(items, int(mode.removesuffix("shot")), path_getter=path_getter)
    raise ValueError(f"Unsupported normal sampling mode: {mode}")
