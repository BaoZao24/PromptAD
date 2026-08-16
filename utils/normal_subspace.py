"""Small support-only normal subspace models for frozen feature scoring.

The implementation is deliberately independent of any dataset or backbone.
It fits PCA only on normal support features and scores a query by the norm of
its residual after projection onto the retained normal subspace.  No query or
test-batch statistics are used while fitting or scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NormalSubspace:
    """PCA model of normal features fitted from support data only."""

    mean: torch.Tensor
    components: torch.Tensor
    explained_variance_ratio: float
    total_rank: int
    fit_samples: int

    @property
    def rank(self) -> int:
        return int(self.components.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.components.shape[1])

    def to(self, device: torch.device | str) -> "NormalSubspace":
        return NormalSubspace(
            mean=self.mean.to(device),
            components=self.components.to(device),
            explained_variance_ratio=self.explained_variance_ratio,
            total_rank=self.total_rank,
            fit_samples=self.fit_samples,
        )

    def score(self, features: torch.Tensor) -> torch.Tensor:
        """Return per-feature reconstruction residuals.

        ``features`` may have any leading dimensions as long as its last
        dimension equals the fitted feature dimension.
        """

        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                "Feature dimension mismatch: "
                f"query={features.shape[-1]} model={self.feature_dim}"
            )
        flat = features.reshape(-1, features.shape[-1]).float()
        centered = flat - self.mean
        projected = (centered @ self.components.t()) @ self.components
        residual = centered - projected
        return torch.linalg.vector_norm(residual, dim=-1).reshape(features.shape[:-1])


@dataclass(frozen=True)
class SupportScoreCalibrator:
    """Robust scale learned from normal-support scores only.

    The calibrator deliberately stores only a location and a positive scale.
    It is used to put two frozen normality models on a comparable scale before
    an agreement operation; no query or test score participates in fitting.
    """

    center: float
    scale: float

    def normalize(self, scores: torch.Tensor) -> torch.Tensor:
        return (scores.float() - float(self.center)) / float(self.scale)


def fit_support_score_calibrator(scores: torch.Tensor) -> SupportScoreCalibrator:
    """Fit a robust location/scale from a one-dimensional support score set."""

    values = scores.detach().float().reshape(-1).cpu()
    if values.numel() < 2:
        raise ValueError("At least two support scores are required")
    quantiles = torch.quantile(values, torch.tensor([0.25, 0.50, 0.75, 0.95]))
    center = float(quantiles[1].item())
    iqr = float((quantiles[2] - quantiles[0]).item())
    upper_tail = float((quantiles[3] - quantiles[1]).item()) / 1.644854
    std = float(values.std(unbiased=False).item())
    scale = max(iqr, upper_tail, std, 1e-6)
    return SupportScoreCalibrator(center=center, scale=scale)


@torch.no_grad()
def leave_one_out_cosine_distance(
    features: torch.Tensor,
    *,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Return each support vector's nearest *other* cosine distance.

    ``features`` are normalized internally.  The operation is bounded by a
    chunked similarity computation, so it does not materialize a full
    feature-by-feature matrix for the wide ViT representations.
    """

    matrix = torch.nn.functional.normalize(features.detach().float(), dim=-1)
    if matrix.shape[0] < 2:
        raise ValueError("At least two support features are required")
    out = torch.empty(matrix.shape[0], device=matrix.device, dtype=torch.float32)
    step = max(1, int(chunk_size))
    for start in range(0, matrix.shape[0], step):
        end = min(start + step, matrix.shape[0])
        probe = matrix[start:end]
        best = torch.full((end - start,), -float("inf"), device=matrix.device)
        for gallery_start in range(0, matrix.shape[0], step):
            gallery_end = min(gallery_start + step, matrix.shape[0])
            similarity = probe @ matrix[gallery_start:gallery_end].t()
            overlap_start = max(start, gallery_start)
            overlap_end = min(end, gallery_end)
            if overlap_start < overlap_end:
                local_rows = torch.arange(
                    overlap_start - start,
                    overlap_end - start,
                    device=matrix.device,
                )
                local_cols = torch.arange(
                    overlap_start - gallery_start,
                    overlap_end - gallery_start,
                    device=matrix.device,
                )
                similarity[local_rows, local_cols] = -float("inf")
            best = torch.maximum(best, similarity.max(dim=-1).values)
        out[start:end] = (1.0 - best) / 2.0
    return out


@torch.no_grad()
def leave_one_out_topk_cosine_distance(
    features: torch.Tensor,
    *,
    topk: int = 1,
    chunk_size: int = 4096,
    distance_divisor: float = 2.0,
) -> torch.Tensor:
    """Score support features against other original support features.

    The exact self-match is removed by flattened feature index before the
    nearest-neighbour search. This remains defined with one support image
    because its other patches are still available.
    """

    if float(distance_divisor) <= 0:
        raise ValueError("distance_divisor must be positive")
    matrix = torch.nn.functional.normalize(
        features.detach().float().reshape(-1, features.shape[-1]), dim=-1
    )
    total = int(matrix.shape[0])
    if total < 2:
        raise ValueError("At least two support features are required")
    k = max(1, min(int(topk), total - 1))
    out = torch.empty(total, device=matrix.device, dtype=torch.float32)
    step = max(1, int(chunk_size))
    for start in range(0, total, step):
        end = min(start + step, total)
        similarity = matrix[start:end] @ matrix.t()
        local = torch.arange(end - start, device=matrix.device)
        similarity[local, start + local] = -float("inf")
        nearest = torch.topk(similarity, k=k, dim=-1).values
        out[start:end] = (1.0 - nearest).mean(dim=-1) / float(distance_divisor)
    return out.reshape(features.shape[:-1])


def fuse_support_calibrated_scores(
    memory_scores: torch.Tensor,
    subspace_scores: torch.Tensor,
    calibrator: tuple[SupportScoreCalibrator, SupportScoreCalibrator],
) -> torch.Tensor:
    """Keep the conservative agreement of memory and subspace evidence."""

    memory_calibrator, subspace_calibrator = calibrator
    memory = memory_calibrator.normalize(memory_scores)
    subspace = subspace_calibrator.normalize(subspace_scores)
    return torch.minimum(memory, subspace)


def fit_normal_subspace(
    features: torch.Tensor,
    *,
    variance_threshold: float = 0.99,
    max_components: int = 64,
    max_fit_samples: int = 8192,
    seed: int = 111,
) -> NormalSubspace:
    """Fit a compact PCA normal subspace from support features.

    The optional sample cap is deterministic and only limits computation; it
    never introduces test data. ``torch.pca_lowrank`` avoids materializing a
    dense feature covariance matrix, which keeps this suitable for the
    relatively wide ViT patch features used by the project.
    """

    if not (0.0 < float(variance_threshold) <= 1.0):
        raise ValueError("variance_threshold must be in (0, 1]")
    if int(max_components) <= 0:
        raise ValueError("max_components must be positive")
    if int(max_fit_samples) <= 1:
        raise ValueError("max_fit_samples must be greater than one")

    matrix = features.detach().float().reshape(-1, features.shape[-1]).cpu()
    if matrix.shape[0] < 2:
        raise ValueError("At least two normal feature vectors are required")
    if matrix.shape[0] > int(max_fit_samples):
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        indices = torch.randperm(matrix.shape[0], generator=generator)[: int(max_fit_samples)]
        matrix = matrix[indices]

    mean = matrix.mean(dim=0)
    centered = matrix - mean
    n_samples, feature_dim = centered.shape
    q = min(int(max_components), int(n_samples - 1), int(feature_dim))
    if q <= 0:
        raise ValueError("PCA needs at least two samples")

    # A low-rank randomized SVD is enough for the compact normal manifold and
    # avoids the O(D^2) covariance matrix used by a full eigendecomposition.
    _u, singular_values, v = torch.pca_lowrank(
        centered,
        q=q,
        center=False,
        niter=2,
    )
    variance = singular_values.square()
    total_variance = float(centered.square().sum().item())
    if total_variance <= 1e-12:
        rank = 1
        retained_ratio = 1.0
    else:
        cumulative = torch.cumsum(variance, dim=0) / total_variance
        target = torch.tensor(float(variance_threshold), dtype=cumulative.dtype)
        reached = torch.nonzero(cumulative >= target, as_tuple=False).flatten()
        rank = int(reached[0].item() + 1) if reached.numel() else int(q)
        rank = max(1, min(rank, int(q)))
        retained_ratio = float(cumulative[rank - 1].item())

    components = v[:, :rank].t().contiguous()
    return NormalSubspace(
        mean=mean.contiguous(),
        components=components,
        explained_variance_ratio=retained_ratio,
        total_rank=int(q),
        fit_samples=int(matrix.shape[0]),
    )
