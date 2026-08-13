"""Physics-aware paired TTA for spectrogram images.

The transforms preserve the semantic axes instead of applying arbitrary image
flips or rotations.  They are intended for paired use: the same transform is
applied when constructing a normal support memory and when querying it.
"""

from __future__ import annotations

import cv2
import numpy as np


SPECTRAL_TTA_BUNDLES = {
    "rf_spectral_structure_v1": (
        "identity",
        "rf_frequency_shift_left",
        "rf_frequency_shift_right",
        "rf_time_shift_up",
        "rf_time_shift_down",
        "rf_frequency_blur",
    ),
    "rf_spectral_background_v1": (
        "identity",
        "rf_frequency_shift_left",
        "rf_frequency_shift_right",
        "rf_time_shift_up",
        "rf_time_shift_down",
        "rf_background_noise_up",
        "rf_background_noise_down",
    ),
    "rf_spectral_time_background_v1": (
        "identity",
        "rf_time_shift_up",
        "rf_time_shift_down",
        "rf_background_noise_jitter",
    ),
    "rf_spectral_physics_v1": (
        "identity",
        "rf_frequency_shift_left",
        "rf_frequency_shift_right",
        "rf_time_shift_up",
        "rf_time_shift_down",
        "rf_frequency_blur",
        "rf_power_gain",
        "rf_power_loss",
    ),
    "rf_spectral_physics_v2": (
        "identity",
        "rf_frequency_shift_left",
        "rf_frequency_shift_right",
        "rf_time_shift_up",
        "rf_time_shift_down",
        "rf_bandwidth_narrow",
        "rf_bandwidth_wide",
        "rf_frequency_blur",
    ),
    "stft_physical_time_v1": (
        "identity",
        "stft_time_offset_up",
        "stft_time_offset_down",
    ),
    "stft_physical_cfo_v1": (
        "identity",
        "stft_cfo_left",
        "stft_cfo_right",
    ),
    "stft_physical_nuisance_v1": (
        "identity",
        "stft_time_offset_up",
        "stft_time_offset_down",
        "stft_cfo_left",
        "stft_cfo_right",
    ),
    "ofdma_spectral_structure_v1": (
        "identity",
        "ofdma_time_shift_left",
        "ofdma_time_shift_right",
        "ofdma_frequency_shift_up",
        "ofdma_frequency_shift_down",
        "ofdma_frequency_blur",
    ),
    "ofdma_spectral_background_v1": (
        "identity",
        "ofdma_time_shift_left",
        "ofdma_time_shift_right",
        "ofdma_frequency_shift_up",
        "ofdma_frequency_shift_down",
        "ofdma_background_noise_up",
        "ofdma_background_noise_down",
    ),
    "ofdma_spectral_time_background_v1": (
        "identity",
        "ofdma_time_shift_left",
        "ofdma_time_shift_right",
        "ofdma_background_noise_jitter",
    ),
    "ofdma_spectral_physics_v1": (
        "identity",
        "ofdma_time_shift_left",
        "ofdma_time_shift_right",
        "ofdma_frequency_shift_up",
        "ofdma_frequency_shift_down",
        "ofdma_frequency_blur",
        "ofdma_power_gain",
        "ofdma_power_loss",
    ),
}


def _shift_image(array: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    height, width = array.shape[:2]
    matrix = np.float32([[1, 0, int(dx)], [0, 1, int(dy)]])
    return cv2.warpAffine(
        array,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _axis_blur(array: np.ndarray, axis: str, ksize: int) -> np.ndarray:
    ksize = int(ksize)
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError("spectral blur kernel must be an odd integer >= 3")
    kernel = (ksize, 1) if axis == "x" else (1, ksize)
    return cv2.GaussianBlur(array, kernel, 0)


def _shift_stft_grid(array: np.ndarray, *, dx: int = 0, dy: int = 0) -> np.ndarray:
    """Shift an STFT magnitude grid without interpolation or wrap-around.

    Frequency-offset and capture-start perturbations are integer translations
    of the time-frequency grid. Newly exposed bins are filled with the local
    median noise floor instead of reflected signal structure.
    """

    dx, dy = int(dx), int(dy)
    if dx and dy:
        raise ValueError("STFT physical shift supports one axis at a time")
    if not dx and not dy:
        return array.copy()

    shifted = np.empty_like(array)
    height, width = array.shape[:2]
    if dx:
        if abs(dx) >= width:
            raise ValueError("frequency shift must be smaller than image width")
        row_floor = np.median(array, axis=1, keepdims=True).astype(array.dtype)
        shifted[...] = row_floor
        if dx > 0:
            shifted[:, dx:] = array[:, : width - dx]
        else:
            shifted[:, : width + dx] = array[:, -dx:]
        return shifted

    if abs(dy) >= height:
        raise ValueError("time shift must be smaller than image height")
    column_floor = np.median(array, axis=0, keepdims=True).astype(array.dtype)
    shifted[...] = column_floor
    if dy > 0:
        shifted[dy:] = array[: height - dy]
    else:
        shifted[: height + dy] = array[-dy:]
    return shifted


def _bandwidth_scale(array: np.ndarray, scale: float) -> np.ndarray:
    height, width = array.shape[:2]
    center_x = (width - 1) / 2.0
    matrix = np.float32(
        [[float(scale), 0, (1.0 - float(scale)) * center_x], [0, 1, 0]]
    )
    return cv2.warpAffine(
        array,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _power_affine(array: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(array.astype(np.float32) * float(gain), 0, 255).astype(np.uint8)


def _gray_view(array: np.ndarray) -> np.ndarray:
    """Return a float grayscale view for detecting the low-power background."""

    if array.ndim == 2:
        return array.astype(np.float32)
    if array.ndim == 3:
        if array.shape[2] == 1:
            return array[..., 0].astype(np.float32)
        if array.shape[2] == 3:
            return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY).astype(np.float32)
    raise ValueError("spectrogram image must be 2-D grayscale or 3-D BGR")


def _smooth_noise_field(shape: tuple[int, int], *, seed: int) -> np.ndarray:
    """Create a deterministic, spatially correlated background fluctuation."""

    height, width = shape
    small_h = max(4, height // 32)
    small_w = max(4, width // 32)
    rng = np.random.default_rng(int(seed))
    coarse = rng.normal(0.0, 1.0, size=(small_h, small_w)).astype(np.float32)
    field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=2.0, sigmaY=2.0)
    field -= np.median(field)
    scale = float(np.percentile(np.abs(field), 90))
    return field / max(scale, 1e-6)


def _background_noise_floor(
    array: np.ndarray,
    direction: float,
    *,
    strength: float = 6.0,
    background_quantile: float = 65.0,
    seed: int = 20260813,
) -> np.ndarray:
    """Perturb only low-power background pixels with a smooth noise-floor change.

    The mask is estimated independently from each image, while the spatial
    noise field is deterministic.  This keeps the transform reproducible and
    prevents bright signal structures from being altered directly.
    """

    gray = _gray_view(array)
    cutoff = float(np.percentile(gray, float(background_quantile)))
    background = gray <= cutoff
    smooth = np.abs(_smooth_noise_field(gray.shape, seed=int(seed)))
    rng = np.random.default_rng(int(seed) + 1)
    grain = np.abs(rng.normal(0.0, 1.0, size=gray.shape)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (3, 3), sigmaX=0.6, sigmaY=0.6)
    grain /= max(float(np.percentile(grain, 90)), 1e-6)
    field = 0.65 * smooth + 0.35 * grain
    field /= max(float(np.percentile(field, 90)), 1e-6)
    delta = float(direction) * float(strength) * field

    result = array.astype(np.float32).copy()
    if result.ndim == 2:
        result[background] += delta[background]
    else:
        result[background, :] += delta[background, None]

    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        result = np.clip(result, info.min, info.max)
    else:
        # The project images are uint8, but keep normalized float inputs valid.
        upper = 1.0 if float(np.nanmax(array)) <= 1.0 else 255.0
        result = np.clip(result, 0.0, upper)
    return result.astype(array.dtype, copy=False)


def _background_noise_jitter(
    array: np.ndarray,
    *,
    strength: float = 6.0,
    background_quantile: float = 65.0,
    seed: int = 20260813,
) -> np.ndarray:
    """Add a zero-centred, smooth noise-floor jitter only to the background.

    Unlike the separate ``up``/``down`` transforms, this is one balanced view:
    the same deterministic field contains both positive and negative local
    fluctuations.  It is therefore suitable for a four-view bundle without
    introducing a global brightness bias.
    """

    gray = _gray_view(array)
    cutoff = float(np.percentile(gray, float(background_quantile)))
    background = gray <= cutoff

    smooth = _smooth_noise_field(gray.shape, seed=int(seed))
    rng = np.random.default_rng(int(seed) + 2)
    grain = rng.normal(0.0, 1.0, size=gray.shape).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (3, 3), sigmaX=0.6, sigmaY=0.6)
    grain -= float(np.mean(grain))
    field = 0.8 * smooth + 0.2 * grain
    if np.any(background):
        field = field - float(np.mean(field[background]))
    field /= max(float(np.percentile(np.abs(field[background]), 90)), 1e-6)
    delta = float(strength) * field

    result = array.astype(np.float32).copy()
    if result.ndim == 2:
        result[background] += delta[background]
    else:
        result[background, :] += delta[background, None]

    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        result = np.clip(result, info.min, info.max)
    else:
        upper = 1.0 if float(np.nanmax(array)) <= 1.0 else 255.0
        result = np.clip(result, 0.0, upper)
    return result.astype(array.dtype, copy=False)


def _frequency_shift_px(shift_px: int) -> int:
    """Use a smaller displacement on the frequency axis than on time."""

    return max(1, int(round(float(shift_px) / 2.0)))


def augment_spectrogram(
    array: np.ndarray,
    mode: str,
    *,
    shift_px: int = 2,
    blur_ksize: int = 3,
    background_noise_strength: float = 3.0,
) -> np.ndarray:
    """Apply one axis-aware paired TTA transform to a BGR/grayscale image."""

    if mode == "identity":
        return array
    if mode == "rf_frequency_shift_left":
        return _shift_image(array, dx=-_frequency_shift_px(shift_px))
    if mode == "rf_frequency_shift_right":
        return _shift_image(array, dx=_frequency_shift_px(shift_px))
    if mode == "rf_time_shift_up":
        return _shift_image(array, dy=-int(shift_px))
    if mode == "rf_time_shift_down":
        return _shift_image(array, dy=int(shift_px))
    if mode == "rf_frequency_blur":
        return _axis_blur(array, "x", blur_ksize)
    if mode == "rf_background_noise_up":
        return _background_noise_floor(array, +1.0, strength=background_noise_strength)
    if mode == "rf_background_noise_down":
        return _background_noise_floor(array, -1.0, strength=background_noise_strength)
    if mode == "rf_background_noise_jitter":
        return _background_noise_jitter(array, strength=background_noise_strength)
    if mode == "rf_bandwidth_narrow":
        return _bandwidth_scale(array, 0.97)
    if mode == "rf_bandwidth_wide":
        return _bandwidth_scale(array, 1.03)
    if mode == "rf_power_gain":
        return _power_affine(array, 1.06)
    if mode == "rf_power_loss":
        return _power_affine(array, 0.94)
    if mode == "stft_time_offset_up":
        return _shift_stft_grid(array, dy=-int(shift_px))
    if mode == "stft_time_offset_down":
        return _shift_stft_grid(array, dy=int(shift_px))
    if mode == "stft_cfo_left":
        return _shift_stft_grid(array, dx=-_frequency_shift_px(shift_px))
    if mode == "stft_cfo_right":
        return _shift_stft_grid(array, dx=_frequency_shift_px(shift_px))
    # OFDMA PNGs are frequency-by-time (height=freq, width=time), which is
    # the transpose of the RF target image convention used above.
    if mode == "ofdma_time_shift_left":
        return _shift_image(array, dx=-int(shift_px))
    if mode == "ofdma_time_shift_right":
        return _shift_image(array, dx=int(shift_px))
    if mode == "ofdma_frequency_shift_up":
        return _shift_image(array, dy=-_frequency_shift_px(shift_px))
    if mode == "ofdma_frequency_shift_down":
        return _shift_image(array, dy=_frequency_shift_px(shift_px))
    if mode == "ofdma_frequency_blur":
        return _axis_blur(array, "y", blur_ksize)
    if mode == "ofdma_background_noise_up":
        return _background_noise_floor(array, +1.0, strength=background_noise_strength)
    if mode == "ofdma_background_noise_down":
        return _background_noise_floor(array, -1.0, strength=background_noise_strength)
    if mode == "ofdma_background_noise_jitter":
        return _background_noise_jitter(array, strength=background_noise_strength)
    if mode == "ofdma_power_gain":
        return _power_affine(array, 1.06)
    if mode == "ofdma_power_loss":
        return _power_affine(array, 0.94)
    raise ValueError(f"Unsupported spectral TTA transform: {mode}")
