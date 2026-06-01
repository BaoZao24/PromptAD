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
from torchvision import transforms


class MorphFusionDualGradInspector:
    def __init__(self, sigma=1.0, percentile=0.995, alpha=0.2, mad_scale=3.0, floor_percentile=0.75):
        self.sigma = sigma
        self.percentile = percentile
        self.alpha = alpha
        self.mad_scale = mad_scale
        self.floor_percentile = floor_percentile
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

    def _denoise_grad_mag(self, grad_mag):
        flat = grad_mag.flatten()
        median = torch.median(flat)
        mad = torch.median(torch.abs(flat - median)).clamp_min(1e-6)
        floor = torch.quantile(flat, self.floor_percentile)
        threshold = torch.maximum(median + self.mad_scale * mad, floor)
        suppressed = torch.relu(grad_mag - threshold)
        return self._robust_normalize(suppressed)

    def compute(self, image):
        gray = self.to_tensor(image.convert('L'))
        gray_np = gray.squeeze(0).numpy()

        smoothed = gaussian_filter(gray_np, sigma=self.sigma)
        time_grad_np = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        freq_grad_np = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)

        time_grad = torch.from_numpy(np.abs(time_grad_np)).unsqueeze(0)
        freq_grad = torch.from_numpy(np.abs(freq_grad_np)).unsqueeze(0)
        grad_mag = torch.sqrt(time_grad * time_grad + freq_grad * freq_grad)

        freq_background = gray.median(dim=-1, keepdim=True).values
        residual = (gray - freq_background).abs()
        weak_residual = self._minmax(gray + self.alpha * residual)

        return {
            'original': np.array(image.convert('RGB')),
            'weak_residual': weak_residual.squeeze(0).numpy(),
            'time_grad': self._robust_normalize(time_grad).squeeze(0).numpy(),
            'freq_grad': self._robust_normalize(freq_grad).squeeze(0).numpy(),
            'grad_mag': self._robust_normalize(grad_mag).squeeze(0).numpy(),
            'denoised_grad_mag': self._denoise_grad_mag(grad_mag).squeeze(0).numpy(),
        }


def export_figures(manifest_csv: Path, output_dir: Path):
    inspector = MorphFusionDualGradInspector()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with manifest_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = Path(row['image_path'])
            image = Image.open(image_path).convert('RGB')
            feats = inspector.compute(image)

            fig, axes = plt.subplots(1, 6, figsize=(21, 4.2), constrained_layout=True)
            axes[0].imshow(feats['original'])
            axes[0].set_title('original')
            axes[1].imshow(feats['weak_residual'], cmap='gray', vmin=0.0, vmax=1.0)
            axes[1].set_title('weak_residual')
            axes[2].imshow(feats['time_grad'], cmap='viridis', vmin=0.0, vmax=1.0)
            axes[2].set_title('|time_grad|')
            axes[3].imshow(feats['freq_grad'], cmap='viridis', vmin=0.0, vmax=1.0)
            axes[3].set_title('|freq_grad|')
            axes[4].imshow(feats['grad_mag'], cmap='magma', vmin=0.0, vmax=1.0)
            axes[4].set_title('grad_mag')
            axes[5].imshow(feats['denoised_grad_mag'], cmap='magma', vmin=0.0, vmax=1.0)
            axes[5].set_title('denoised_grad_mag')
            for ax in axes:
                ax.axis('off')
            fig.suptitle(f"{row['dataset']} | {row['scene']} | {row['noise']} | {image_path.name}", fontsize=11)

            out_name = Path(row['figure_path']).name
            out_path = output_dir / row['dataset'] / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=160)
            plt.close(fig)

            nonzero_grad = float((feats['grad_mag'] > 0.05).mean())
            nonzero_clean = float((feats['denoised_grad_mag'] > 0.05).mean())
            rows.append({
                'dataset': row['dataset'],
                'scene': row['scene'],
                'noise': row['noise'],
                'image_path': row['image_path'],
                'figure_path': str(out_path),
                'grad_active_frac_gt_0.05': f"{nonzero_grad:.4f}",
                'clean_active_frac_gt_0.05': f"{nonzero_clean:.4f}",
            })

    summary_csv = output_dir / 'comparison_manifest.csv'
    with summary_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            'dataset', 'scene', 'noise', 'image_path', 'figure_path',
            'grad_active_frac_gt_0.05', 'clean_active_frac_gt_0.05'
        ])
        writer.writeheader()
        writer.writerows(rows)

    readme = output_dir / 'README.md'
    lines = [
        '# Morph Fusion Gradient Cleanup Comparison',
        '',
        'Source samples: `analysis_outputs/morph_fusion_featuremaps/selected_samples.csv`.',
        'Panels: original, weak_residual, |time_grad|, |freq_grad|, grad_mag, denoised_grad_mag.',
        'Denoising: Gaussian-smoothed Sobel gradients, gradient magnitude, then soft thresholding with `max(q75, median + 3*MAD)` before robust percentile normalization.',
        '',
        'Per-sample activity shrinkage (`>0.05` fraction):',
        '',
    ]
    for row in rows:
        lines.append(
            f"- `{row['dataset']}` | `{row['scene']}` | `{row['noise']}` | "
            f"active `{row['grad_active_frac_gt_0.05']}` -> `{row['clean_active_frac_gt_0.05']}`"
        )
    readme.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-csv', type=Path, default=Path('analysis_outputs/morph_fusion_featuremaps/selected_samples.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('analysis_outputs/morph_fusion_gradient_cleanup_compare'))
    args = parser.parse_args()
    export_figures(args.manifest_csv, args.output_dir)


if __name__ == '__main__':
    main()
