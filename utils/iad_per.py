"""Percentile (PER) anomaly score used by IAD-PER.

This module implements Equation (10) from Tian et al., *Unsupervised
Spectrum Anomaly Detection Method for Unauthorized Bands* (2022), while
matching the authors' released ``util.py`` behavior.

Two details of that implementation are intentional:

* pooling at image boundaries uses only in-image neighbours (equivalent to
  ``-inf`` padding for max pooling and ``+inf`` padding for min pooling);
* pixels outside each mask are set to zero *before* taking the percentile, so
  those zeros remain part of the percentile population.

Inputs are batched spectrograms in BCHW layout. NumPy arrays produce NumPy
outputs and PyTorch tensors produce tensors on the input device.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_ALPHA = 0.05
DEFAULT_GAMMA = 3
DEFAULT_BACKGROUND_WEIGHT = 2.0
DEFAULT_BACKGROUND_PERCENTILE = 90.0
DEFAULT_SIGNAL_PERCENTILE = 99.0


def _validate_parameters(
    alpha: float,
    gamma: int,
    background_weight: float,
    xi: float,
    eta: float,
) -> None:
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if not isinstance(gamma, int) or isinstance(gamma, bool) or gamma <= 0 or gamma % 2 == 0:
        raise ValueError("gamma must be a positive odd integer")
    if not np.isfinite(background_weight):
        raise ValueError("background_weight must be finite")
    if not 0.0 <= xi <= 100.0:
        raise ValueError("xi must lie in [0, 100]")
    if not 0.0 <= eta <= 100.0:
        raise ValueError("eta must lie in [0, 100]")


def _validate_shape(x: Any, reconstruction: Any) -> None:
    if x.ndim != 4 or reconstruction.ndim != 4:
        raise ValueError("input and reconstruction must both have BCHW shape")
    if tuple(x.shape) != tuple(reconstruction.shape):
        raise ValueError(
            "input and reconstruction must have identical shapes; "
            f"got {tuple(x.shape)} and {tuple(reconstruction.shape)}"
        )
    if any(size <= 0 for size in x.shape):
        raise ValueError("all BCHW dimensions must be non-empty")


def _numpy_pool2d(array: np.ndarray, kernel_size: int, mode: str) -> np.ndarray:
    """Stride-one pooling with clipped boundary neighbourhoods."""
    radius = kernel_size // 2
    constant = -np.inf if mode == "max" else np.inf
    padded = np.pad(
        array,
        ((0, 0), (0, 0), (radius, radius), (radius, radius)),
        mode="constant",
        constant_values=constant,
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (kernel_size, kernel_size), axis=(-2, -1)
    )
    if mode == "max":
        return windows.max(axis=(-2, -1))
    if mode == "min":
        return windows.min(axis=(-2, -1))
    raise ValueError(f"unsupported pooling mode: {mode}")


def _numpy_iad_per(
    x: np.ndarray,
    reconstruction: np.ndarray,
    *,
    alpha: float,
    gamma: int,
    background_weight: float,
    xi: float,
    eta: float,
    return_components: bool,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    _validate_shape(x, reconstruction)
    dtype = np.result_type(x.dtype, reconstruction.dtype, np.float32)
    x_float = x.astype(dtype, copy=False)
    reconstruction_float = reconstruction.astype(dtype, copy=False)

    pooled_input = _numpy_pool2d(x_float, gamma, "max")
    background_mask = pooled_input < alpha
    signal_mask = ~background_mask
    pooled_error = _numpy_pool2d(np.abs(x_float - reconstruction_float), gamma, "min")

    flattened_background = (pooled_error * background_mask).reshape(x.shape[0], -1)
    flattened_signal = (pooled_error * signal_mask).reshape(x.shape[0], -1)
    background_score = np.percentile(flattened_background, xi, axis=1)
    signal_score = np.percentile(flattened_signal, eta, axis=1)
    score = background_weight * background_score + signal_score

    if not return_components:
        return score
    return score, {
        "background": background_score,
        "signal": signal_score,
        "background_mask": background_mask,
        "signal_mask": signal_mask,
        "pooled_error": pooled_error,
    }


def _torch_iad_per(
    x: torch.Tensor,
    reconstruction: torch.Tensor,
    *,
    alpha: float,
    gamma: int,
    background_weight: float,
    xi: float,
    eta: float,
    return_components: bool,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _validate_shape(x, reconstruction)
    if x.device != reconstruction.device:
        raise ValueError("input and reconstruction must be on the same device")

    dtype = torch.promote_types(x.dtype, reconstruction.dtype)
    if not dtype.is_floating_point or dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    x_float = x.to(dtype=dtype)
    reconstruction_float = reconstruction.to(dtype=dtype)

    padding = gamma // 2
    pooled_input = F.max_pool2d(x_float, kernel_size=gamma, stride=1, padding=padding)
    background_mask = pooled_input < alpha
    signal_mask = ~background_mask
    pooled_error = -F.max_pool2d(
        -torch.abs(x_float - reconstruction_float),
        kernel_size=gamma,
        stride=1,
        padding=padding,
    )

    flattened_background = (pooled_error * background_mask).flatten(start_dim=1)
    flattened_signal = (pooled_error * signal_mask).flatten(start_dim=1)
    background_score = torch.quantile(flattened_background, xi / 100.0, dim=1)
    signal_score = torch.quantile(flattened_signal, eta / 100.0, dim=1)
    score = background_weight * background_score + signal_score

    if not return_components:
        return score
    return score, {
        "background": background_score,
        "signal": signal_score,
        "background_mask": background_mask,
        "signal_mask": signal_mask,
        "pooled_error": pooled_error,
    }


def iad_per_score(
    x: np.ndarray | torch.Tensor,
    reconstruction: np.ndarray | torch.Tensor,
    *,
    alpha: float = DEFAULT_ALPHA,
    gamma: int = DEFAULT_GAMMA,
    background_weight: float = DEFAULT_BACKGROUND_WEIGHT,
    xi: float = DEFAULT_BACKGROUND_PERCENTILE,
    eta: float = DEFAULT_SIGNAL_PERCENTILE,
    return_components: bool = False,
) -> (
    np.ndarray
    | torch.Tensor
    | tuple[np.ndarray, dict[str, np.ndarray]]
    | tuple[torch.Tensor, dict[str, torch.Tensor]]
):
    """Compute one IAD-PER anomaly score per BCHW sample.

    The background mask is ``max_pool2d(x, gamma) < alpha``. Its complement is
    the signal mask. The absolute reconstruction error is min-pooled, multiplied
    by each mask, flattened across C/H/W, and reduced with percentiles ``xi``
    and ``eta``. The final score is ``background_weight * background + signal``.

    When ``return_components=True``, returns ``(score, components)``. Components
    contains the two per-sample score terms plus the masks and pooled error.
    """
    _validate_parameters(alpha, gamma, background_weight, xi, eta)
    if isinstance(x, torch.Tensor) and isinstance(reconstruction, torch.Tensor):
        return _torch_iad_per(
            x,
            reconstruction,
            alpha=alpha,
            gamma=gamma,
            background_weight=background_weight,
            xi=xi,
            eta=eta,
            return_components=return_components,
        )
    if isinstance(x, np.ndarray) and isinstance(reconstruction, np.ndarray):
        return _numpy_iad_per(
            x,
            reconstruction,
            alpha=alpha,
            gamma=gamma,
            background_weight=background_weight,
            xi=xi,
            eta=eta,
            return_components=return_components,
        )
    raise TypeError("input and reconstruction must use the same NumPy or PyTorch backend")


per_score = iad_per_score


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_GAMMA",
    "DEFAULT_BACKGROUND_WEIGHT",
    "DEFAULT_BACKGROUND_PERCENTILE",
    "DEFAULT_SIGNAL_PERCENTILE",
    "iad_per_score",
    "per_score",
]
