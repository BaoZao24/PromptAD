"""Image-domain adaptations of classical spectrum-sensing statistics.

The RF and OFDMA protocols in this repository expose spectrogram PNGs rather
than complex IQ samples.  The functions below therefore implement deterministic
spectrogram-domain statistics and make that limitation explicit in the
experiment protocol.  They do not fit on abnormal samples or use test-batch
statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TRADITIONAL_METHODS = (
    "energy_detector",
    "spectral_entropy",
    "spectral_flatness",
    "spectral_kurtosis",
    "ca_cfar",
    "statistical_fusion",
)

METHOD_DISPLAY_NAMES = {
    "energy_detector": "ED (spectrogram)",
    "spectral_entropy": "Spectral entropy",
    "spectral_flatness": "Spectral flatness",
    "spectral_kurtosis": "Spectral kurtosis",
    "ca_cfar": "CA-CFAR (spectrogram)",
    "statistical_fusion": "Fixed spectral-statistics fusion",
}

METHOD_REFERENCES = {
    "energy_detector": {
        "short": "Urkowitz (1967)",
        "citation": "H. Urkowitz, Proceedings of the IEEE 55(4), 523--531, 1967.",
        "doi": "https://doi.org/10.1109/PROC.1967.5573",
        "adaptation": "sum/mean of squared spectrogram intensity; no IQ samples",
    },
    "spectral_entropy": {
        "short": "Power-spectrum entropy detector",
        "citation": "Power-spectrum entropy detection for covert communication signals, 2023.",
        "doi": "https://doi.org/10.19363/J.cnki.cn10-1380/tn.2023.08.21",
        "adaptation": "normalized Shannon entropy of the frequency-marginal power profile",
    },
    "spectral_flatness": {
        "short": "Gurugopinath (2017)",
        "citation": "S. Gurugopinath, Electronics Letters 53(13), 890--892, 2017.",
        "doi": "https://doi.org/10.1049/el.2016.4712",
        "adaptation": "geometric/arithmetic mean ratio of the frequency-marginal power profile",
    },
    "spectral_kurtosis": {
        "short": "Antoni (2006)",
        "citation": "J. Antoni, Mechanical Systems and Signal Processing 20(2), 282--307, 2006.",
        "doi": "https://doi.org/10.1016/j.ymssp.2004.09.001",
        "adaptation": "95th percentile of temporal excess kurtosis across frequency bins",
    },
    "ca_cfar": {
        "short": "Rohling (1983)",
        "citation": "H. Rohling, IEEE Transactions on AES AES-19(4), 608--621, 1983.",
        "doi": "https://doi.org/10.1109/TAES.1983.309350",
        "adaptation": "cell-averaging local threshold on the frequency-marginal profile",
    },
    "statistical_fusion": {
        "short": "Fixed support-calibrated fusion",
        "citation": "Derived baseline; component statistics use the references above.",
        "doi": "",
        "adaptation": "maximum positive robust z-score over the five fixed statistics",
    },
}


# The direction is defined after converting the raw statistic to an anomaly
# evidence value.  Lower entropy/flatness indicates a more concentrated
# spectrum, while the remaining statistics are high-is-anomalous.
_HIGH_IS_ANOMALOUS = {
    "energy_detector": 1.0,
    "spectral_entropy": -1.0,
    "spectral_flatness": -1.0,
    "spectral_kurtosis": 1.0,
    "ca_cfar": 1.0,
}


def _gray_float(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        array = array.astype(np.float32).mean(axis=2)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D spectrogram or 3-D image, got {array.shape}")
    array = array.astype(np.float32, copy=False)
    if array.size == 0:
        raise ValueError("Spectrogram is empty")
    if float(np.nanmax(array)) > 1.5:
        array = array / 255.0
    return np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)


def _power_profile(image: np.ndarray, frequency_axis: int) -> tuple[np.ndarray, np.ndarray]:
    array = _gray_float(image)
    axis = int(frequency_axis)
    if axis not in (0, 1):
        raise ValueError(f"frequency_axis must be 0 or 1, got {frequency_axis}")
    time_axis = 1 - axis
    power = np.square(np.maximum(array, 0.0), dtype=np.float32)
    profile = power.mean(axis=time_axis).astype(np.float64)
    return power.astype(np.float64), np.maximum(profile, np.finfo(np.float64).eps)


def _spectral_entropy(profile: np.ndarray) -> float:
    probability = profile / max(float(profile.sum()), np.finfo(np.float64).eps)
    entropy = -float(np.sum(probability * np.log(probability + 1e-12)))
    return entropy / max(np.log(float(profile.size)), 1.0)


def _spectral_flatness(profile: np.ndarray) -> float:
    log_mean = float(np.mean(np.log(profile + 1e-12)))
    arithmetic_mean = max(float(np.mean(profile)), 1e-12)
    return float(np.exp(log_mean) / arithmetic_mean)


def _spectral_kurtosis(power: np.ndarray, frequency_axis: int) -> float:
    time_axis = 1 - int(frequency_axis)
    centered = power - power.mean(axis=time_axis, keepdims=True)
    second = np.mean(np.square(centered), axis=time_axis)
    fourth = np.mean(np.power(centered, 4), axis=time_axis)
    excess = fourth / np.square(second + 1e-8) - 3.0
    excess = np.nan_to_num(excess, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.quantile(np.maximum(excess, 0.0), 0.95))


def _ca_cfar(profile: np.ndarray) -> float:
    length = int(profile.size)
    if length < 8:
        return float((profile - profile.mean()).max() / (profile.std() + 1e-8))
    guard = max(1, int(round(length * 0.02)))
    train = max(2, int(round(length * 0.08)))
    radius = guard + train
    kernel = np.ones(2 * radius + 1, dtype=np.float64)
    kernel[radius - guard : radius + guard + 1] = 0.0
    denominator = float(kernel.sum())
    padded = np.pad(profile, radius, mode="reflect")
    local_sum = np.convolve(padded, kernel, mode="valid") / denominator
    local_sq = np.convolve(np.square(padded), kernel, mode="valid") / denominator
    local_std = np.sqrt(np.maximum(local_sq - np.square(local_sum), 0.0))
    standardized = (profile - local_sum) / (local_std + 1e-6)
    return float(np.max(standardized))


def extract_raw_statistics(image: np.ndarray, frequency_axis: int) -> dict[str, float]:
    """Extract the fixed image-domain statistics for one spectrogram."""

    power, profile = _power_profile(image, frequency_axis)
    return {
        "energy_detector": float(power.mean()),
        "spectral_entropy": _spectral_entropy(profile),
        "spectral_flatness": _spectral_flatness(profile),
        "spectral_kurtosis": _spectral_kurtosis(power, frequency_axis),
        "ca_cfar": _ca_cfar(profile),
    }


@dataclass(frozen=True)
class SupportCalibrator:
    """Normal-only robust calibration for the fixed statistics."""

    methods: tuple[str, ...]
    centers: dict[str, float]
    scales: dict[str, float]

    def score_raw(self, raw: dict[str, float]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for method in self.methods:
            evidence = float(_HIGH_IS_ANOMALOUS[method]) * float(raw[method])
            scores[method] = (
                evidence - self.centers[method]
            ) / self.scales[method]
        if self.methods:
            positive = np.maximum(
                np.asarray([scores[method] for method in self.methods], dtype=np.float64),
                0.0,
            )
            scores["statistical_fusion"] = float(np.max(positive))
        return scores


def fit_support_calibrator(
    support_images: list[np.ndarray],
    frequency_axis: int,
    methods: tuple[str, ...] | list[str] = TRADITIONAL_METHODS,
) -> SupportCalibrator:
    selected = tuple(method for method in methods if method != "statistical_fusion")
    unknown = set(selected) - set(_HIGH_IS_ANOMALOUS)
    if unknown:
        raise ValueError(f"Unsupported traditional statistic(s): {sorted(unknown)}")
    if not support_images:
        raise ValueError("At least one normal support image is required")
    raw_rows = [extract_raw_statistics(image, frequency_axis) for image in support_images]
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for method in selected:
        values = np.asarray(
            [float(_HIGH_IS_ANOMALOUS[method]) * row[method] for row in raw_rows],
            dtype=np.float64,
        )
        center = float(np.median(values))
        iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
        mad = float(np.median(np.abs(values - center)))
        scale = max(iqr / 1.349, 1.4826 * mad, 1e-6)
        centers[method] = center
        scales[method] = scale
    return SupportCalibrator(selected, centers, scales)


def score_image(
    image: np.ndarray,
    calibrator: SupportCalibrator,
    frequency_axis: int,
) -> dict[str, float]:
    return calibrator.score_raw(extract_raw_statistics(image, frequency_axis))
