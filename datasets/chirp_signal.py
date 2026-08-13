"""Compatibility wrapper for the canonical self-RF target loader."""

from .rf_target import (
    RF_SCENES,
    extract_frequency_band,
    extract_time_range,
    load_legacy_rf_signal,
)


chirp_signal_classes = list(RF_SCENES)
_extract_time_range = extract_time_range
_extract_freq = extract_frequency_band


def load_chirp_signal(
    category,
    k_shot,
    noise_level="m10db",
    freq=None,
    train_category=None,
):
    return load_legacy_rf_signal(
        "chirp_signal",
        category,
        k_shot,
        noise_level=noise_level,
        freq=freq,
        train_category=train_category,
    )
