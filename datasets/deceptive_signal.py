"""Compatibility wrapper for the retired deceptive-signal dataset."""

from .rf_target import (
    RF_SCENES,
    extract_frequency_band,
    extract_time_range,
    load_legacy_rf_signal,
)


deceptive_signal_classes = list(RF_SCENES)
_extract_time_range = extract_time_range
_extract_freq = extract_frequency_band


def load_deceptive_signal(category, k_shot, freq=None, train_category=None):
    return load_legacy_rf_signal(
        "deceptive_signal",
        category,
        k_shot,
        noise_level="0db",
        freq=freq,
        train_category=train_category,
    )
