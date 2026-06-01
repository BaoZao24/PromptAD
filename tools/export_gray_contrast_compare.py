import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms


class GrayContrastInspector:
    def __init__(
        self,
        sigma=1.0,
        percentile=0.995,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=8,
    ):
        self.sigma = sigma
        self.percentile = percentile
        self.to_tensor = transforms.ToTensor()
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size),
        )

    def _robust_normalize(self, channel):
        flat = channel.flatten()
        if flat.size == 0:
            return channel
        scale = np.quantile(flat, self.percentile)
        scale = max(scale, 1e-6)
        return np.clip(channel / scale, 0.0, 1.0)

    def compute(self, image):
        gray = self.to_tensor(image.convert("L")).squeeze(0).numpy()
        gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        gray_contrast = self.clahe.apply(gray_u8).astype(np.float32) / 255.0

        smoothed = gaussian_filter(gray, sigma=self.sigma)
        gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx * gx + gy * gy)
        grad_mag = self._robust_normalize(grad_mag)

        freq_background = np.median(gray, axis=1, keepdims=True)
        residual = np.abs(gray - freq_background)
        weak_residual = gray + 0.2 * residual
        wr_min = weak_residual.min()
        wr_max = weak_residual.max()
        weak_residual = (weak_residual - wr_min) / max(wr_max - wr_min, 1e-6)

        return {
            "original": np.array(image.convert("RGB")),
            "gray": gray,
            "gray_contrast": gray_contrast,
            "weak_residual": weak_residual,
            "grad_mag": grad_mag,
        }


def export_figures(manifest_csv: Path, output_dir: Path):
    inspector = GrayContrastInspector()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with manifest_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = Path(row["image_path"])
            gt_path = Path(row["gt_path"])
            image = Image.open(image_path).convert("RGB")
            gt_mask = np.array(Image.open(gt_path).convert("L"))
            feats = inspector.compute(image)

            fig, axes = plt.subplots(1, 6, figsize=(22, 4.2), constrained_layout=True)
            axes[0].imshow(feats["original"])
            axes[0].set_title("original")
            axes[1].imshow(gt_mask, cmap="magma")
            axes[1].set_title("gt mask")
            axes[2].imshow(feats["gray"], cmap="gray", vmin=0.0, vmax=1.0)
            axes[2].set_title("gray")
            axes[3].imshow(feats["gray_contrast"], cmap="gray", vmin=0.0, vmax=1.0)
            axes[3].set_title("gray_contrast")
            axes[4].imshow(feats["weak_residual"], cmap="gray", vmin=0.0, vmax=1.0)
            axes[4].set_title("weak_residual")
            axes[5].imshow(feats["grad_mag"], cmap="magma", vmin=0.0, vmax=1.0)
            axes[5].set_title("grad_mag")
            for ax in axes:
                ax.axis("off")

            fig.suptitle(
                f"{row['dataset']} | {row['scene']} | {row['noise']} | {image_path.name}",
                fontsize=11,
            )
            out_name = Path(row["figure_path"]).name
            out_path = output_dir / row["dataset"] / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=160)
            plt.close(fig)

            rows.append(
                {
                    "dataset": row["dataset"],
                    "scene": row["scene"],
                    "noise": row["noise"],
                    "image_path": row["image_path"],
                    "figure_path": str(out_path),
                    "gray_mean": f"{float(feats['gray'].mean()):.4f}",
                    "gray_contrast_mean": f"{float(feats['gray_contrast'].mean()):.4f}",
                    "weak_residual_mean": f"{float(feats['weak_residual'].mean()):.4f}",
                    "grad_mag_mean": f"{float(feats['grad_mag'].mean()):.4f}",
                }
            )

    manifest_path = output_dir / "comparison_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "scene",
                "noise",
                "image_path",
                "figure_path",
                "gray_mean",
                "gray_contrast_mean",
                "weak_residual_mean",
                "grad_mag_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    readme_path = output_dir / "README.md"
    lines = [
        "# Gray Contrast Compare",
        "",
        "Source samples: `analysis_outputs/morph_fusion_featuremaps/selected_samples.csv`.",
        "Panels: original, gt mask, gray, gray_contrast (CLAHE), weak_residual, grad_mag.",
        "Gray contrast uses conservative local histogram equalization (CLAHE) so the image remains faithful to the source while local intensity contrast is enhanced.",
        "",
        "Per-sample means:",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['dataset']}` | `{row['scene']}` | `{row['noise']}` | "
            f"`gray={row['gray_mean']}` | "
            f"`gray_contrast={row['gray_contrast_mean']}` | "
            f"`weak_residual={row['weak_residual_mean']}` | "
            f"`grad_mag={row['grad_mag_mean']}`"
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
        default=Path("analysis_outputs/gray_contrast_compare"),
    )
    args = parser.parse_args()
    export_figures(args.manifest_csv, args.output_dir)


if __name__ == "__main__":
    main()
