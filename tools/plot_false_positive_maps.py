#!/usr/bin/env python
"""Plot selected self-RF false-positive images and CNN maps."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path("analysis_outputs/20260712_false_positive_maps")
FUSION = Path("analysis_outputs/20260711_exp_harmonic_fusion/self/scores")
SELECTED = [
    ("burst", "burst_signal-WeaponMuseum_spectrum-m30db", 117),
    ("dsss", "dsss_signal-TimeSquare_spectrum-m30db", 35),
    ("pulse", "pulse_signal-Playground_spectrum-m30db", 1),
]


def main():
    output = ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    for label, stem, index in SELECTED:
        map_file = ROOT / "maps" / f"rf_target-{stem}-index{index}.npz"
        map_data = np.load(map_file, allow_pickle=True)
        image_path = Path(str(map_data["image_path"]))
        anomaly_map = np.asarray(map_data["anomaly_map"], dtype=np.float32)
        fusion = np.load(FUSION / f"{stem}-scores.npz", allow_pickle=True)

        image = np.asarray(Image.open(image_path).convert("RGB"))
        map_2d = anomaly_map.squeeze()
        if map_2d.ndim != 2:
            map_2d = map_2d.reshape(int(np.sqrt(map_2d.size)), -1)
        map_2d = (map_2d - map_2d.min()) / max(float(map_2d.max() - map_2d.min()), 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        axes[0].imshow(image)
        axes[0].set_title("Normal input")
        axes[1].imshow(map_2d, cmap="magma")
        axes[1].set_title("CNN anomaly map")
        axes[2].imshow(image)
        axes[2].imshow(map_2d, cmap="magma", alpha=0.48, vmin=0, vmax=1)
        axes[2].set_title("CNN overlay")
        for ax in axes:
            ax.axis("off")

        text = (
            f"{label} | {image_path.name}\n"
            f"ViT={float(fusion['vit_minmax'][index]):.3f}  "
            f"CNN={float(fusion['cnn_minmax'][index]):.3f}  "
            f"OR={float(fusion['or_evidence'][index]):.3f}"
        )
        fig.suptitle(text, fontsize=9)
        fig.tight_layout()
        fig.savefig(output / f"false_positive_{label}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
