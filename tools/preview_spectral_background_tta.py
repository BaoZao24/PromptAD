#!/usr/bin/env python
"""Render a visual comparison for the background-noise spectral TTA candidate."""

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
OUT = Path("analysis_outputs/20260813_spectral_background_tta_preview")
MODES = (
    ("identity", "Original"),
    ("rf_frequency_shift_left", "Freq -2 px"),
    ("rf_frequency_shift_right", "Freq +2 px"),
    ("rf_time_shift_up", "Time -4 px"),
    ("rf_time_shift_down", "Time +4 px"),
    ("rf_frequency_blur", "Frequency blur"),
    ("rf_background_noise_up", "Noise floor +"),
    ("rf_background_noise_down", "Noise floor -"),
)


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def render(name: str) -> Path:
    normal = load_gray(ROOT / "normal_originals" / f"{name}.png")
    abnormal = load_gray(ROOT / "abnormal_originals" / f"{name}.png")
    rows = [("Normal", normal), ("Abnormal", abnormal)]

    fig, axes = plt.subplots(2, len(MODES), figsize=(19, 5.6), squeeze=False)
    for row_idx, (row_name, image) in enumerate(rows):
        for col_idx, (mode, title) in enumerate(MODES):
            view = augment_spectrogram(
                image,
                mode,
                shift_px=4,
                blur_ksize=3,
            )
            ax = axes[row_idx][col_idx]
            ax.imshow(view, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            if row_idx == 0:
                ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(row_name, fontsize=11)
    fig.suptitle(
        "RF spectral TTA preview: axis shifts, frequency blur, and background noise-floor perturbation",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.2)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"rf_spectral_background_v1_{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    for sample_path in sorted((ROOT / "normal_originals").glob("*.png")):
        print(render(sample_path.stem))
