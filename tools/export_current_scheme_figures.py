#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms


class GrayResidualInspector:
    def __init__(self, alpha=0.1, clahe_clip_limit=2.0, clahe_tile_grid_size=8):
        self.alpha = alpha
        self.to_tensor = transforms.ToTensor()
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size),
        )

    @staticmethod
    def _minmax(channel):
        c_min = channel.amin(dim=(-2, -1), keepdim=True)
        c_max = channel.amax(dim=(-2, -1), keepdim=True)
        return (channel - c_min) / (c_max - c_min).clamp_min(1e-6)

    def compute(self, image):
        original = image.convert('RGB')
        gray = self.to_tensor(original.convert('L'))
        gray_np = gray.squeeze(0).numpy()
        gray_u8 = np.clip(gray_np * 255.0, 0, 255).astype(np.uint8)
        gray_contrast = self.clahe.apply(gray_u8).astype(np.float32) / 255.0
        freq_background = gray.median(dim=-1, keepdim=True).values
        residual = (gray - freq_background).abs()
        weak_residual = self._minmax(gray + self.alpha * residual)
        fused = torch.cat([
            torch.from_numpy(gray_contrast).unsqueeze(0),
            weak_residual,
            gray,
        ], dim=0).permute(1, 2, 0).numpy()
        return {
            'original': np.array(original),
            'gray_contrast': gray_contrast,
            'weak_residual': weak_residual.squeeze(0).numpy(),
            'original_gray': gray.squeeze(0).numpy(),
            'fused': fused,
        }


def default_samples():
    return [
        Path('/mnt/data/wangbei/data/datasets/burst/Playground_spectrum/abnormal/m10db/Playground_spectrum_burst_m10db_t00000-04000_f89.60-91.10MHz_patch003_abnormal.png'),
        Path('/mnt/data/wangbei/data/datasets/chirp/Playground_spectrum/abnormal/m10db/Playground_spectrum_chirp_m10db_t06000-10000_f89.60-91.10MHz_patch075_abnormal.png'),
        Path('/mnt/data/wangbei/data/datasets/dsss/Playground_spectrum/abnormal/m10db/Playground_spectrum_dsss_m10db_t00000-04000_f88.80-90.30MHz_patch002_abnormal.png'),
    ]


def export_feature_figure(samples, output_path, alpha, paper=False):
    inspector = GrayResidualInspector(alpha=alpha)
    rows = []
    for sample in samples:
        if sample.exists():
            rows.append((sample, inspector.compute(Image.open(sample))))
    if not rows:
        raise FileNotFoundError('No sample images found.')

    if paper:
        titles = [
            'Input spectrogram',
            'Local structure view',
            'Background-deviation view',
            'Energy-preserving view',
        ]
        fig, axes = plt.subplots(len(rows), 4, figsize=(12.8, 2.85 * len(rows)), constrained_layout=True)
    else:
        titles = [
            'Original spectrogram',
            'Local structure view',
            'Background-deviation view',
            'Energy-preserving view',
            'Fused 3-channel input',
        ]
        fig, axes = plt.subplots(len(rows), 5, figsize=(18, 3.6 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = axes[None, :]

    dataset_labels = {
        'burst': 'Burst',
        'chirp': 'Chirp',
        'dsss': 'DSSS',
    }
    for r, (sample, feats) in enumerate(rows):
        images = [
            (feats['original'], None),
            (feats['gray_contrast'], 'gray'),
            (feats['weak_residual'], 'gray'),
            (feats['original_gray'], 'gray'),
        ]
        if not paper:
            images.append((feats['fused'], None))
        for c, (image, cmap) in enumerate(images):
            ax = axes[r, c]
            if cmap is None:
                ax.imshow(image, vmin=0.0, vmax=1.0 if image.dtype != np.uint8 else None)
            else:
                ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
            if r == 0:
                ax.set_title(titles[c], fontsize=12 if paper else 11, pad=8)
            ax.axis('off')
        dataset = sample.parts[6] if len(sample.parts) > 6 else sample.parent.name
        axes[r, 0].set_ylabel(dataset_labels.get(dataset, dataset), fontsize=12, rotation=90, labelpad=12)
    title = 'Examples of structure-aware grayscale residual views' if paper else 'Visualization of the proposed structure-aware grayscale residual fusion'
    fig.suptitle(title, fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220 if paper else 180)
    plt.close(fig)


def draw_centered(draw, box, text, font, fill=(30, 30, 30)):
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=6, align='center')
    x = box[0] + (box[2] - box[0] - (bbox[2] - bbox[0])) / 2
    y = box[1] + (box[3] - box[1] - (bbox[3] - bbox[1])) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=6, align='center')


def arrow(draw, start, end, fill=(80, 80, 80)):
    draw.line([start, end], fill=fill, width=4)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head = 14
    p1 = (end[0] - head * ux + 7 * px, end[1] - head * uy + 7 * py)
    p2 = (end[0] - head * ux - 7 * px, end[1] - head * uy - 7 * py)
    draw.polygon([end, p1, p2], fill=fill)


def export_pipeline_figure(output_path):
    width, height = 1500, 620
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    try:
        font_title = ImageFont.truetype('DejaVuSans-Bold.ttf', 36)
        font = ImageFont.truetype('DejaVuSans.ttf', 24)
        font_small = ImageFont.truetype('DejaVuSans.ttf', 20)
    except OSError:
        font_title = font = font_small = ImageFont.load_default()

    draw.text((70, 38), 'Structure-aware grayscale residual fusion', font=font_title, fill=(20, 20, 20))
    boxes = {
        'input': (70, 230, 290, 360),
        'gray': (390, 230, 610, 360),
        'local': (735, 100, 1000, 210),
        'residual': (735, 255, 1000, 365),
        'energy': (735, 410, 1000, 520),
        'fusion': (1140, 230, 1405, 360),
    }
    colors = {
        'input': (232, 242, 255),
        'gray': (242, 242, 242),
        'local': (235, 248, 238),
        'residual': (255, 245, 225),
        'energy': (242, 242, 242),
        'fusion': (238, 232, 255),
    }
    labels = {
        'input': 'Pseudo-color\nspectrogram',
        'gray': 'Grayscale\nconversion',
        'local': 'Local spectral\nstructure view',
        'residual': 'Background-deviation\nview (alpha = 0.1)',
        'energy': 'Energy-preserving\ngrayscale view',
        'fusion': '3-channel input\nto PromptAD',
    }
    for key, box in boxes.items():
        draw.rounded_rectangle(box, radius=18, fill=colors[key], outline=(110, 110, 110), width=3)
        draw_centered(draw, box, labels[key], font)

    arrow(draw, (290, 295), (390, 295))
    for target in ['local', 'residual', 'energy']:
        arrow(draw, (610, 295), (735, (boxes[target][1] + boxes[target][3]) // 2))
    for source in ['local', 'residual', 'energy']:
        arrow(draw, (1000, (boxes[source][1] + boxes[source][3]) // 2), (1140, 295))

    note = 'The method replaces non-physical RGB channels with three complementary grayscale views.'
    draw.text((70, 555), note, font=font_small, fill=(70, 70, 70))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('analysis_outputs/current_scheme_figures'))
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--samples', nargs='*', type=Path, default=default_samples())
    args = parser.parse_args()
    export_feature_figure(args.samples, args.output_dir / 'gray_residual_feature_examples.png', args.alpha)
    export_pipeline_figure(args.output_dir / 'gray_residual_pipeline.png')


if __name__ == '__main__':
    main()
