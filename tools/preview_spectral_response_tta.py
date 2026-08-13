#!/usr/bin/env python
"""Render the frequency-response paired-TTA candidate on RF spectrograms."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.spectral_tta import augment_spectrogram


ROOT = Path("analysis_outputs/01_figures/paper_arch_tta/tta_view_samples")
OUT = Path("analysis_outputs/20260813_spectral_response_tta_preview")
MODES = (
    ("identity", "Original"),
    ("rf_time_shift_up", "Time alignment −4"),
    ("rf_time_shift_down", "Time alignment +4"),
    ("rf_frequency_response_jitter", "Frequency-response calibration"),
)


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def render(name: str) -> Path:
    normal = load_gray(ROOT / "normal_originals" / f"{name}.png")
    abnormal = load_gray(ROOT / "abnormal_originals" / f"{name}.png")
    figure, axes = plt.subplots(2, len(MODES), figsize=(11.2, 5.5), squeeze=False)
    for row_index, (row_name, image) in enumerate((("Normal", normal), ("Abnormal", abnormal))):
        for col_index, (mode, title) in enumerate(MODES):
            view = augment_spectrogram(
                image,
                mode,
                shift_px=4,
                frequency_response_strength=3.0,
            )
            axis = axes[row_index][col_index]
            axis.imshow(view, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            if row_index == 0:
                axis.set_title(title, fontsize=10)
            if col_index == 0:
                axis.set_ylabel(row_name, fontsize=11)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle("Spectral paired TTA: time alignment and frequency-response calibration", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.2)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"rf_spectral_response_v1_{name}.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


if __name__ == "__main__":
    for sample_path in sorted((ROOT / "normal_originals").glob("*.png")):
        print(render(sample_path.stem))
