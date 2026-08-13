"""Frozen-reference information-theoretic scores for spectrogram images.

The core equations and defaults follow Afgani, Sinanovic, and Haas (2010):

* KLD uses ``D(P || Q) = sum(P * log2(P / Q))`` and the
  Krichevsky--Trofimov histogram preload of 0.5 (32 bins by default).
* ICA uses ``I(x) = -log2(p(x))`` (20 bins by default) and a cluster of
  three contiguous events, corresponding to ``N = 2`` in the paper.

The paper updates ICA's histogram online.  This module intentionally freezes
the normal reference after fitting so that query images cannot alter the
model.  Histogram edges are either supplied explicitly or learned only from
normal support images; query values are clipped to those frozen edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


DEFAULT_PSEUDOCOUNT = 0.5
DEFAULT_KLD_BINS = 32
DEFAULT_ICA_BINS = 20
DEFAULT_ICA_CLUSTER_LENGTH = 3
DEFAULT_ICA_THRESHOLD_FACTOR = 1.2


def _as_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a finite float64 grayscale image without per-image scaling."""

    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[-1] == 1:
            array = array[..., 0]
        elif array.shape[-1] in (3, 4):
            # Spectrogram RGB images encode one scalar through colour.  A
            # fixed channel mean is deterministic and introduces no fitted
            # or query-dependent conversion parameters.
            array = array[..., :3].astype(np.float64).mean(axis=-1)
        else:
            raise ValueError(
                "Expected an RGB/RGBA image with channels last; "
                f"got shape {array.shape}"
            )
    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2-D grayscale or 3-D RGB image, got shape {array.shape}"
        )
    if array.size == 0:
        raise ValueError("Image must not be empty")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("Image values must all be finite")
    return array


def _normal_image_list(normal_images: Iterable[np.ndarray]) -> list[np.ndarray]:
    images = [_as_grayscale(image) for image in normal_images]
    if not images:
        raise ValueError("At least one normal reference image is required")
    return images


def _validate_histogram_parameters(n_bins: int, pseudocount: float) -> None:
    if isinstance(n_bins, bool) or int(n_bins) != n_bins or int(n_bins) < 2:
        raise ValueError(f"n_bins must be an integer >= 2, got {n_bins!r}")
    if not np.isfinite(pseudocount) or float(pseudocount) <= 0.0:
        raise ValueError(
            f"pseudocount must be a positive finite value, got {pseudocount!r}"
        )


def _fit_value_range(
    images: list[np.ndarray], value_range: tuple[float, float] | None
) -> tuple[float, float]:
    if value_range is not None:
        lower, upper = map(float, value_range)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(
                "value_range must contain two finite values with lower < upper"
            )
        return lower, upper

    lower = min(float(image.min()) for image in images)
    upper = max(float(image.max()) for image in images)
    if lower == upper:
        # A nonzero, support-only interval keeps constant normal supports
        # usable while remaining independent of every future query.
        radius = max(abs(lower), 1.0) * 1e-6
        lower -= radius
        upper += radius
    return lower, upper


def _read_only(array: np.ndarray) -> np.ndarray:
    frozen = np.array(array, dtype=np.float64, copy=True)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True)
class FrozenHistogram:
    """A pooled normal histogram whose range and PMF cannot be query-adapted."""

    bin_edges: np.ndarray
    counts: np.ndarray
    probabilities: np.ndarray
    pseudocount: float

    @property
    def n_bins(self) -> int:
        return int(self.probabilities.size)

    @property
    def value_range(self) -> tuple[float, float]:
        return float(self.bin_edges[0]), float(self.bin_edges[-1])

    def counts_for(self, image: np.ndarray) -> np.ndarray:
        """Count query values using frozen bins, clipping only at fixed edges."""

        gray = _as_grayscale(image)
        clipped = np.clip(gray, self.bin_edges[0], self.bin_edges[-1])
        counts, _ = np.histogram(clipped, bins=self.bin_edges)
        return counts.astype(np.float64)

    def pmf_for(self, image: np.ndarray) -> np.ndarray:
        counts = self.counts_for(image) + self.pseudocount
        return counts / float(counts.sum())

    def bin_indices_for(self, image: np.ndarray) -> np.ndarray:
        """Map an image to frozen histogram bin indices."""

        gray = _as_grayscale(image)
        clipped = np.clip(gray, self.bin_edges[0], self.bin_edges[-1])
        indices = np.searchsorted(self.bin_edges, clipped, side="right") - 1
        return np.clip(indices, 0, self.n_bins - 1).astype(np.intp)


def fit_frozen_histogram(
    normal_images: Iterable[np.ndarray],
    *,
    n_bins: int,
    value_range: tuple[float, float] | None = None,
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
) -> FrozenHistogram:
    """Fit one pooled histogram from normal images and freeze all parameters."""

    _validate_histogram_parameters(n_bins, pseudocount)
    images = _normal_image_list(normal_images)
    lower, upper = _fit_value_range(images, value_range)
    edges = np.linspace(lower, upper, int(n_bins) + 1, dtype=np.float64)
    counts = np.zeros(int(n_bins), dtype=np.float64)
    for image in images:
        clipped = np.clip(image, lower, upper)
        image_counts, _ = np.histogram(clipped, bins=edges)
        counts += image_counts
    smoothed = counts + float(pseudocount)
    probabilities = smoothed / float(smoothed.sum())
    return FrozenHistogram(
        bin_edges=_read_only(edges),
        counts=_read_only(counts),
        probabilities=_read_only(probabilities),
        pseudocount=float(pseudocount),
    )


@dataclass(frozen=True)
class KLDRef:
    """Normal-reference adaptation of Afgani et al.'s KLD detector."""

    reference: FrozenHistogram

    @classmethod
    def fit(
        cls,
        normal_images: Iterable[np.ndarray],
        *,
        n_bins: int = DEFAULT_KLD_BINS,
        value_range: tuple[float, float] | None = None,
        pseudocount: float = DEFAULT_PSEUDOCOUNT,
    ) -> "KLDRef":
        return cls(
            fit_frozen_histogram(
                normal_images,
                n_bins=n_bins,
                value_range=value_range,
                pseudocount=pseudocount,
            )
        )

    def score(self, image: np.ndarray) -> float:
        """Return ``D(P_query || Q_normal)`` in bits."""

        query = self.reference.pmf_for(image)
        normal = self.reference.probabilities
        score = np.sum(query * np.log2(query / normal), dtype=np.float64)
        return float(score)

    def score_many(self, images: Iterable[np.ndarray]) -> np.ndarray:
        return np.asarray([self.score(image) for image in images], dtype=np.float64)


def _reduce_image_scores(
    values: np.ndarray,
    *,
    reduction: str,
    topk_fraction: float,
    quantile: float,
) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if reduction == "max":
        return float(np.max(flat))
    if reduction == "mean":
        return float(np.mean(flat))
    if reduction == "topk_mean":
        if not np.isfinite(topk_fraction) or not 0.0 < topk_fraction <= 1.0:
            raise ValueError("topk_fraction must be in (0, 1]")
        count = max(1, int(np.ceil(flat.size * float(topk_fraction))))
        return float(np.mean(np.partition(flat, flat.size - count)[-count:]))
    if reduction == "quantile":
        if not np.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be in [0, 1]")
        return float(np.quantile(flat, float(quantile)))
    raise ValueError(
        "image_reduction must be one of: max, mean, topk_mean, quantile"
    )


@dataclass(frozen=True)
class ICAFrozen:
    """Frozen-reference adaptation of information content analysis (ICA).

    The returned image score is continuous: each contiguous window is scored
    by the minimum excess information over the threshold, so it is positive
    exactly when every event in that window exceeds the threshold.  The
    image-level reduction is explicit and configurable.
    """

    reference: FrozenHistogram
    information_threshold: float
    threshold_factor: float | None

    @classmethod
    def fit(
        cls,
        normal_images: Iterable[np.ndarray],
        *,
        n_bins: int = DEFAULT_ICA_BINS,
        value_range: tuple[float, float] | None = None,
        pseudocount: float = DEFAULT_PSEUDOCOUNT,
        threshold_factor: float = DEFAULT_ICA_THRESHOLD_FACTOR,
        information_threshold: float | None = None,
    ) -> "ICAFrozen":
        reference = fit_frozen_histogram(
            normal_images,
            n_bins=n_bins,
            value_range=value_range,
            pseudocount=pseudocount,
        )
        factor: float | None
        if information_threshold is None:
            if not np.isfinite(threshold_factor) or float(threshold_factor) <= 0.0:
                raise ValueError("threshold_factor must be a positive finite value")
            information = -np.log2(reference.probabilities)
            threshold = float(threshold_factor) * float(np.std(information))
            factor = float(threshold_factor)
        else:
            threshold = float(information_threshold)
            if not np.isfinite(threshold):
                raise ValueError("information_threshold must be finite")
            factor = None
        return cls(reference, threshold, factor)

    def information_map(self, image: np.ndarray) -> np.ndarray:
        """Return ``-log2(p_normal(x))`` for every spectrogram event."""

        indices = self.reference.bin_indices_for(image)
        return -np.log2(self.reference.probabilities[indices])

    def cluster_score_map(
        self,
        image: np.ndarray,
        *,
        time_axis: int,
        cluster_length: int = DEFAULT_ICA_CLUSTER_LENGTH,
    ) -> np.ndarray:
        """Score each contiguous event window along ``time_axis``.

        A window score is the minimum ``I(x) - I_th`` in that window.  Thus,
        scores above zero exactly match the paper's requirement that all
        ``N + 1`` contiguous events exceed the information threshold.
        """

        if isinstance(cluster_length, bool) or int(cluster_length) != cluster_length:
            raise ValueError("cluster_length must be a positive integer")
        cluster_length = int(cluster_length)
        if cluster_length < 1:
            raise ValueError("cluster_length must be a positive integer")
        if time_axis not in (0, 1):
            raise ValueError(f"time_axis must be 0 or 1, got {time_axis!r}")
        information = self.information_map(image)
        if cluster_length > information.shape[time_axis]:
            raise ValueError(
                f"cluster_length={cluster_length} exceeds time-axis size "
                f"{information.shape[time_axis]}"
            )
        excess = information - self.information_threshold
        windows = np.lib.stride_tricks.sliding_window_view(
            excess, window_shape=cluster_length, axis=time_axis
        )
        return np.min(windows, axis=-1)

    def score(
        self,
        image: np.ndarray,
        *,
        time_axis: int,
        cluster_length: int = DEFAULT_ICA_CLUSTER_LENGTH,
        image_reduction: str = "max",
        topk_fraction: float = 0.01,
        quantile: float = 0.99,
    ) -> float:
        cluster_scores = self.cluster_score_map(
            image, time_axis=time_axis, cluster_length=cluster_length
        )
        return _reduce_image_scores(
            cluster_scores,
            reduction=image_reduction,
            topk_fraction=topk_fraction,
            quantile=quantile,
        )

    def score_many(
        self,
        images: Iterable[np.ndarray],
        *,
        time_axis: int,
        cluster_length: int = DEFAULT_ICA_CLUSTER_LENGTH,
        image_reduction: str = "max",
        topk_fraction: float = 0.01,
        quantile: float = 0.99,
    ) -> np.ndarray:
        return np.asarray(
            [
                self.score(
                    image,
                    time_axis=time_axis,
                    cluster_length=cluster_length,
                    image_reduction=image_reduction,
                    topk_fraction=topk_fraction,
                    quantile=quantile,
                )
                for image in images
            ],
            dtype=np.float64,
        )
