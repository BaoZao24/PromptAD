"""Compatibility wrapper for the canonical self-RF target loader."""

from .rf_target import RF_SCENES, load_legacy_rf_signal


wideband_pulse_classes = list(RF_SCENES)


def load_wideband_pulse(
    category,
    k_shot,
    noise_level="m20db",
    freq=None,
    train_category=None,
):
    return load_legacy_rf_signal(
        "wideband_pulse",
        category,
        k_shot,
        noise_level=noise_level,
        freq=freq,
        train_category=train_category,
    )
