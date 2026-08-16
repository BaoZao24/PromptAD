#!/usr/bin/env python
"""Illustrative CNN vs ViT anomaly maps for the architecture diagram (TRANSPARENT BG).

NOT real model output - synthesized from groundtruth to show the VISUAL CHARACTER
of each branch's anomaly map. Saved as RGBA PNGs with TRANSPARENT background so
they can be overlaid on the original abnormal spectrogram in Visio.

  - CNN (ResNet18 layer3) : smooth, coarse, blob-like
  - ViT (CLIP ViT-B/16+)   : patch-grid aligned, blocky, finer localization

Output: 2 folders under .../tta_view_samples/
  cnn_anomaly_map/  - 5 RGBA PNGs (alpha follows score: 0 -> transparent, high -> opaque)
  vit_anomaly_map/  - 5 RGBA PNGs (same)
"""
import os, cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

PAIRS = [
    ("burst_signal",   "/mnt/data/wangbei/data/datasets/burst/TimeSquare_spectrum/groundtruth/m30db/TimeSquare_spectrum_burst_m30db_t00000-04000_f92.00-93.50MHz_patch006_groundtruth.png"),
    ("chirp_signal",   "/mnt/data/wangbei/data/datasets/chirp/TimeSquare_spectrum/groundtruth/m30db/TimeSquare_spectrum_chirp_m30db_t14000-18000_f105.60-107.10MHz_patch191_groundtruth.png"),
    ("dsss_signal",    "/mnt/data/wangbei/data/datasets/dsss/TimeSquare_spectrum/groundtruth/m30db/TimeSquare_spectrum_dsss_m30db_t10000-14000_f105.60-107.10MHz_patch143_groundtruth.png"),
    ("pulse_signal",   "/mnt/data/wangbei/data/datasets/pulse/TimeSquare_spectrum/groundtruth/m30db/TimeSquare_spectrum_pulse_m30db_t12000-16000_f100.00-101.50MHz_patch160_groundtruth.png"),
    ("wideband_pulse", "/mnt/data/wangbei/data/datasets/wideband_pulse/TimeSquare_spectrum/groundtruth/m30db/TimeSquare_spectrum_wideband_pulse_m30db_t04000-08000_f92.00-93.50MHz_patch054_groundtruth.png"),
]

CNN_GRID = 15        # ResNet18 layer3 @ 240 input: 240/16 = 15
CNN_BLUR = 9
VIT_GRID = 16        # CLIP ViT-B/16+ @ 240 input: ~16 patches per side
OUT_SIZE = 256

# gamma on alpha so subtle anomalies remain visible when overlaid
ALPHA_GAMMA = 0.5

inferno = cm.get_cmap("inferno")


def to_rgba(gray_float, gmin, gmax):
    """Map [gmin, gmax] to RGBA. score=0 -> alpha=0 (transparent)."""
    norm = np.clip((gray_float - gmin) / max(gmax - gmin, 1e-9), 0.0, 1.0)
    rgb = (inferno(norm)[:, :, :3] * 255.0).astype(np.uint8)         # RGB
    alpha = np.clip(norm, 0.0, 1.0) ** ALPHA_GAMMA
    alpha = (alpha * 255.0).astype(np.uint8)                            # 0..255
    rgba = np.dstack([rgb, alpha])                                     # H,W,4 (RGB+A)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)                      # BGRA for cv2.imwrite


def cnn_look(gt, grid=CNN_GRID, blur_sigma=CNN_BLUR, out=OUT_SIZE):
    mask = (gt > 0).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, k, iterations=2)
    coarse = cv2.resize(mask, (grid, grid), interpolation=cv2.INTER_AREA)
    up = cv2.resize(coarse, (out, out), interpolation=cv2.INTER_LINEAR)
    up = cv2.GaussianBlur(up, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    return up


def vit_look(gt, grid=VIT_GRID, out=OUT_SIZE):
    mask = (gt > 0).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, k, iterations=1)
    coarse = cv2.resize(mask, (grid, grid), interpolation=cv2.INTER_AREA)
    up = cv2.resize(coarse, (out, out), interpolation=cv2.INTER_NEAREST)  # blocky patch grid
    up = cv2.GaussianBlur(up, (0, 0), sigmaX=0.8, sigmaY=0.8)
    return up


def main():
    base = "analysis_outputs/01_figures/paper_arch_tta/tta_view_samples"
    cnn_dir = os.path.join(base, "cnn_anomaly_map")
    vit_dir = os.path.join(base, "vit_anomaly_map")
    os.makedirs(cnn_dir, exist_ok=True)
    os.makedirs(vit_dir, exist_ok=True)

    cnn_maps, vit_maps = [], []
    for sig, gp in PAIRS:
        gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(gp)
        if gt.shape != (OUT_SIZE, OUT_SIZE):
            gt = cv2.resize(gt, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_NEAREST)
        cnn_maps.append((sig, cnn_look(gt)))
        vit_maps.append((sig, vit_look(gt)))

    all_vals = np.concatenate([m.ravel() for _, m in cnn_maps + vit_maps])
    gmin, gmax = float(all_vals.min()), float(all_vals.max())
    print(f"common color scale: [{gmin:.4f}, {gmax:.4f}]   alpha_gamma={ALPHA_GAMMA}")

    for sig, m in cnn_maps:
        cv2.imwrite(os.path.join(cnn_dir, f"{sig}.png"), to_rgba(m, gmin, gmax))
    for sig, m in vit_maps:
        cv2.imwrite(os.path.join(vit_dir, f"{sig}.png"), to_rgba(m, gmin, gmax))

    # README update
    readme_path = os.path.join(base, "README.md")
    with open(readme_path, "a", encoding="utf-8") as f:
        f.write("\n\n## Illustrative anomaly maps (CNN vs ViT, TRANSPARENT BG)\n\n"
                "NOT real model output. Synthesized from `groundtruth/` to show the visual character of each branch.\n\n"
                "- `cnn_anomaly_map/`: ResNet18 (layer3) style - smooth, coarse blob (CNN_GRID=15, blur_sigma=9).\n"
                "- `vit_anomaly_map/`: CLIP ViT-B/16+ style - patch-aligned 16x16 blocky grid (nearest upsample).\n\n"
                "Saved as **RGBA PNG** (4 channels). `alpha = (norm ^ 0.5) * 255` so background (score=0) is fully transparent and the heatmap only shows where the anomaly is. Drag these directly on top of the abnormal spectrogram in Visio/PPT for overlay.\n"
                "All 10 maps share one inferno color scale (see `color_scale.txt`).\n")
    with open(os.path.join(base, "color_scale.txt"), "w") as f:
        f.write(f"global_min={gmin:.6f}\nglobal_max={gmax:.6f}\ncolormap=inferno\nalpha_gamma={ALPHA_GAMMA}\nformat=RGBA (transparent BG)\n")

    print(f"wrote 5 RGBA PNGs -> {cnn_dir}/")
    print(f"wrote 5 RGBA PNGs -> {vit_dir}/")
    print(f"updated {readme_path}")


if __name__ == "__main__":
    main()
