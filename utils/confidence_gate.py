"""Fixed confidence gates used by the formal anomaly detectors."""

from __future__ import annotations

import numpy as np


RANK_QUANTILE = 0.5

# Formal inductive gate configuration.  These values are frozen and are not
# exposed as evaluator command-line tuning knobs.  They were selected in the
# RF support-only exploration and transferred unchanged to OFDMA.
SAFE_SUPPORT_RANK_QUANTILE = 0.8
SAFE_SUPPORT_TEMPERATURE = 2.5
SAFE_SUPPORT_ALPHA = 2.5


def _minmax(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _percentile_rank(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size <= 1:
        return np.zeros_like(values)
    # Use mid-ranks for ties instead of depending on the input order.  The
    # returned rank is normalized to [0, 1], with the largest value at 1.
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.arange(values.size, dtype=np.float64)
    ranks_sorted = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks_sorted[start:end] = float(sorted_ranks[start:end].mean())
        start = end
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks / float(values.size - 1)


def support_empirical_rank(values, reference) -> np.ndarray:
    """Map values to a support-only empirical CDF with mid-ranks.

    The reference array must contain only normal support views.  Values below
    or above the reference range are clipped to the endpoint ranks, so the
    result is defined for every test sample without reading its batch peers.
    """

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    reference = np.sort(np.asarray(reference, dtype=np.float64).reshape(-1))
    reference = reference[np.isfinite(reference)]
    if reference.size <= 1:
        return np.zeros_like(values)
    left = np.searchsorted(reference, values, side="left").astype(np.float64)
    right = np.searchsorted(reference, values, side="right").astype(np.float64)
    ordinal = (left + right - 1.0) / 2.0
    return np.clip(ordinal / float(reference.size - 1), 0.0, 1.0)


def _support_robust_stats(reference) -> tuple[float, float]:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    reference = reference[np.isfinite(reference)]
    if reference.size == 0:
        raise ValueError("Support reference must contain at least one finite score")
    q25, q50, q75 = np.quantile(reference, [0.25, 0.5, 0.75])
    return float(q50), max(float(q75 - q25), 1e-6)


def safe_support_only_gate(
    vit_scores,
    cnn_scores,
    vit_reference,
    cnn_reference,
) -> dict[str, np.ndarray]:
    """Fuse scores using normal support references only.

    ViT remains the base score.  CNN can add a non-negative correction only
    when its support empirical rank is in the frozen upper tail and its
    robust sigmoid evidence exceeds ViT.  No test-batch min/max or ranks are
    accessed, so each sample can be evaluated independently.
    """

    vit = np.asarray(vit_scores, dtype=np.float64).reshape(-1)
    cnn = np.asarray(cnn_scores, dtype=np.float64).reshape(-1)
    if vit.shape != cnn.shape:
        raise ValueError(
            f"ViT/CNN score shape mismatch: vit={vit.shape}, cnn={cnn.shape}"
        )
    if vit.size == 0:
        raise ValueError("ViT/CNN score arrays must not be empty")

    vit_median, vit_iqr = _support_robust_stats(vit_reference)
    cnn_median, cnn_iqr = _support_robust_stats(cnn_reference)
    vit_z = (vit - vit_median) / vit_iqr
    cnn_z = (cnn - cnn_median) / cnn_iqr
    # Clipping avoids overflow for unusual scores while preserving the
    # sigmoid result at machine precision for all relevant values.
    vit_prob = 1.0 / (1.0 + np.exp(-np.clip(vit_z / SAFE_SUPPORT_TEMPERATURE, -60.0, 60.0)))
    cnn_prob = 1.0 / (1.0 + np.exp(-np.clip(cnn_z / SAFE_SUPPORT_TEMPERATURE, -60.0, 60.0)))
    correction = np.clip(cnn_prob - vit_prob, 0.0, 1.0)
    cnn_rank = support_empirical_rank(cnn, cnn_reference)
    rank_gate = (cnn_rank >= SAFE_SUPPORT_RANK_QUANTILE).astype(np.float64)
    correction *= rank_gate
    score = vit + SAFE_SUPPORT_ALPHA * vit_iqr * correction
    return {
        "score": score,
        "cnn_rank": cnn_rank,
        "vit_probability": vit_prob,
        "cnn_probability": cnn_prob,
        "vit_normalized": vit_prob,
        "cnn_normalized": cnn_prob,
        "correction": correction,
        "gate": correction,
    }


def confidence_gated_or(vit_scores, cnn_scores) -> dict[str, np.ndarray]:
    """Fuse one test cell with the fixed method from commit ``ac8fe3c``.

    ViT is the main score. CNN contributes only in the upper half of its
    within-cell rank and only by the amount that its normalized evidence
    exceeds ViT. Each branch is min-max normalized to ``[0, 1]`` within the
    cell; a constant branch maps to all zeros. Exact score ties use mid-ranks
    (the mean of the occupied ordinal positions). There are intentionally no
    tuning arguments.
    """

    vit = np.asarray(vit_scores, dtype=np.float64).reshape(-1)
    cnn = np.asarray(cnn_scores, dtype=np.float64).reshape(-1)
    if vit.shape != cnn.shape:
        raise ValueError(
            f"ViT/CNN score shape mismatch: vit={vit.shape}, cnn={cnn.shape}"
        )
    if vit.size == 0:
        raise ValueError("ViT/CNN score arrays must not be empty")

    vit_normalized = _minmax(vit)
    cnn_normalized = _minmax(cnn)
    cnn_rank = _percentile_rank(cnn)
    rank_confidence = np.clip(
        (cnn_rank - RANK_QUANTILE) / (1.0 - RANK_QUANTILE),
        0.0,
        1.0,
    )
    cnn_advantage = np.clip(cnn_normalized - vit_normalized, 0.0, 1.0)
    gate = rank_confidence * cnn_advantage
    score = 1.0 - (1.0 - vit_normalized) * (
        1.0 - gate * cnn_normalized
    )
    return {
        "score": score,
        "gate": gate,
        "vit_normalized": vit_normalized,
        "cnn_normalized": cnn_normalized,
        "cnn_rank": cnn_rank,
    }
