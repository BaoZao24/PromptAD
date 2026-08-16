import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F
from torchvision import transforms


class WidebandFeatureInspector:
    def __init__(
        self,
        sigma=1.0,
        percentile=0.995,
        local_time_kernel=9,
        local_freq_kernel=49,
        bg_time_kernel=33,
        bg_freq_kernel=97,
        band_mix_v2=0.28,
        band_mix_v3=0.10,
    ):
        self.sigma = sigma
        self.percentile = percentile
        self.local_time_kernel = local_time_kernel
        self.local_freq_kernel = local_freq_kernel
        self.bg_time_kernel = bg_time_kernel
        self.bg_freq_kernel = bg_freq_kernel
        self.band_mix_v2 = band_mix_v2
        self.band_mix_v3 = band_mix_v3
        self.to_tensor = transforms.ToTensor()

    def _robust_normalize(self, channel):
        flat = channel.flatten()
        if flat.numel() == 0:
            return channel
        scale = torch.quantile(flat, self.percentile).clamp_min(1e-6)
        return (channel / scale).clamp(0.0, 1.0)

    @staticmethod
    def _minmax(channel):
        c_min = channel.amin(dim=(-2, -1), keepdim=True)
        c_max = channel.amax(dim=(-2, -1), keepdim=True)
        return (channel - c_min) / (c_max - c_min).clamp_min(1e-6)

    def _band_contrast(self, gray):
        x = gray.unsqueeze(0)
        local_band = F.avg_pool2d(
            x,
            kernel_size=(self.local_freq_kernel, self.local_time_kernel),
            stride=1,
            padding=(self.local_freq_kernel // 2, self.local_time_kernel // 2),
        ).squeeze(0)
        broad_bg = F.avg_pool2d(
            x,
            kernel_size=(self.bg_freq_kernel, self.bg_time_kernel),
            stride=1,
            padding=(self.bg_freq_kernel // 2, self.bg_time_kernel // 2),
        ).squeeze(0)
        return (local_band - broad_bg).clamp_min(0.0)

    def _wideband_feature_v2(self, gray):
        band_contrast = self._robust_normalize(self._band_contrast(gray))
        return self._minmax(gray + self.band_mix_v2 * band_contrast)

    def _wideband_feature_v3(self, gray):
        band_contrast = self._robust_normalize(self._band_contrast(gray))
        blended = (gray + self.band_mix_v3 * band_contrast).clamp(0.0, 1.0)
        return blended

    def compute(self, image):
        gray = self.to_tensor(image.convert("L"))
        gray_np = gray.squeeze(0).numpy()

        smoothed = gaussian_filter(gray_np, sigma=self.sigma)
        time_grad_np = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        freq_grad_np = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        time_grad = torch.from_numpy(np.abs(time_grad_np)).unsqueeze(0)
        freq_grad = torch.from_numpy(np.abs(freq_grad_np)).unsqueeze(0)
        grad_mag = torch.sqrt(time_grad * time_grad + freq_grad * freq_grad)

        return {
            "original": np.array(image.convert("RGB")),
            "grad_mag": self._robust_normalize(grad_mag).squeeze(0).numpy(),
            "wideband_feature_v2": self._wideband_feature_v2(gray).squeeze(0).numpy(),
            "wideband_feature_v3": self._wideband_feature_v3(gray).squeeze(0).numpy(),
        }


def export_figures(manifest_csv: Path, output_dir: Path):
    inspector = WidebandFeatureInspector()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with manifest_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["dataset"] != "wideband_pulse":
                continue

            image_path = Path(row["image_path"])
            image = Image.open(image_path).convert("RGB")
            feats = inspector.compute(image)

            fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
            axes[0].imshow(feats["original"])
            axes[0].set_title("original")
            axes[1].imshow(feats["grad_mag"], cmap="magma", vmin=0.0, vmax=1.0)
            axes[1].set_title("grad_mag")
            axes[2].imshow(feats["wideband_feature_v2"], cmap="gray", vmin=0.0, vmax=1.0)
            axes[2].set_title("wideband_feature_v2")
            axes[3].imshow(feats["wideband_feature_v3"], cmap="gray", vmin=0.0, vmax=1.0)
            axes[3].set_title("wideband_feature_v3")
            for ax in axes:
                ax.axis("off")

            fig.suptitle(
                f"{row['dataset']} | {row['scene']} | {row['noise']} | {image_path.name}",
                fontsize=11,
            )
            out_name = Path(row["figure_path"]).name
            out_path = output_dir / out_name
            fig.savefig(out_path, dpi=160)
            plt.close(fig)

            rows.append(
                {
                    "scene": row["scene"],
                    "noise": row["noise"],
                    "image_path": row["image_path"],
                    "figure_path": str(out_path),
                    "grad_mag_mean": f"{float(feats['grad_mag'].mean()):.4f}",
                    "wideband_feature_v2_mean": f"{float(feats['wideband_feature_v2'].mean()):.4f}",
                    "wideband_feature_v3_mean": f"{float(feats['wideband_feature_v3'].mean()):.4f}",
                }
            )

    manifest_path = output_dir / "comparison_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene",
                "noise",
                "image_path",
                "figure_path",
                "grad_mag_mean",
                "wideband_feature_v2_mean",
                "wideband_feature_v3_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    readme_path = output_dir / "README.md"
    lines = [
        "# Wideband Feature Compare V3",
        "",
        "Source samples: `analysis_outputs/morph_fusion_featuremaps/selected_samples.csv` filtered to `wideband_pulse`.",
        "Panels: original, grad_mag, wideband_feature_v2, wideband_feature_v3.",
        "V3 is intentionally gentler: lower enhancement alpha, no min-max remap on the final blend, and preserved grayscale dynamic range.",
        "",
        "Per-sample means:",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['scene']}` | `{row['noise']}` | "
            f"`grad_mean={row['grad_mag_mean']}` | "
            f"`v2_mean={row['wideband_feature_v2_mean']}` | "
            f"`v3_mean={row['wideband_feature_v3_mean']}`"
        )
    readme_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("analysis_outputs/morph_fusion_featuremaps/selected_samples.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/wideband_feature_compare_v3"),
    )
    args = parser.parse_args()
    export_figures(args.manifest_csv, args.output_dir)


if __name__ == "__main__":
    main()
