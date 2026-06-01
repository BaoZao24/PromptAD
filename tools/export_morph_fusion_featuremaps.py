import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from PromptAD.model import MorphFusionDualGradChannels

SCENES = [
    "WeaponMuseum_spectrum",
    "Playground_spectrum",
    "TimeSquare_spectrum",
    "Gymnasium_spectrum",
]
DATASET_CONFIG = {
    "burst_signal": {
        "root": Path("/mnt/data/wangbei/data/datasets/burst"),
        "noise_levels": ["m10db", "m20db", "m30db"],
    },
    "chirp_signal": {
        "root": Path("/mnt/data/wangbei/data/datasets/chirp"),
        "noise_levels": ["m10db", "m20db", "m30db"],
    },
    "dsss_signal": {
        "root": Path("/mnt/data/wangbei/data/datasets/dsss"),
        "noise_levels": ["m10db", "m20db", "m30db"],
    },
    "wideband_pulse": {
        "root": Path("/mnt/data/wangbei/data/datasets/wideband_pulse"),
        "noise_levels": ["m20db", "m30db", "m40db"],
    },
}


@dataclass
class Sample:
    dataset: str
    scene: str
    noise: str
    image_path: Path
    gt_path: Path
    gt_area: int


def abnormal_to_groundtruth(path: Path) -> Path:
    gt_name = path.name.replace('_abnormal.png', '_groundtruth.png')
    return path.parents[2] / 'groundtruth' / path.parent.name / gt_name


def gt_area(gt_path: Path) -> int:
    mask = np.array(Image.open(gt_path).convert('L'))
    return int((mask > 0).sum())


def collect_top_samples(dataset: str, top_k: int) -> list[Sample]:
    cfg = DATASET_CONFIG[dataset]
    candidates: list[Sample] = []
    for scene in SCENES:
        for noise in cfg['noise_levels']:
            abnormal_dir = cfg['root'] / scene / 'abnormal' / noise
            if not abnormal_dir.is_dir():
                continue
            for image_path in sorted(abnormal_dir.glob('*.png')):
                gt_path = abnormal_to_groundtruth(image_path)
                if not gt_path.exists():
                    continue
                candidates.append(Sample(
                    dataset=dataset,
                    scene=scene,
                    noise=noise,
                    image_path=image_path,
                    gt_path=gt_path,
                    gt_area=gt_area(gt_path),
                ))
    candidates.sort(key=lambda x: (x.gt_area, x.scene, x.noise, x.image_path.name), reverse=True)
    return candidates[:top_k]


def export_figure(sample: Sample, output_dir: Path, transform: MorphFusionDualGradChannels) -> dict:
    image = Image.open(sample.image_path).convert('RGB')
    channels = transform(image).numpy()
    original = np.array(image)
    gt_mask = np.array(Image.open(sample.gt_path).convert('L'))

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), constrained_layout=True)
    axes[0].imshow(original)
    axes[0].set_title('original')
    axes[1].imshow(gt_mask, cmap='magma')
    axes[1].set_title(f'gt mask\narea={sample.gt_area}')
    axes[2].imshow(channels[0], cmap='gray', vmin=0.0, vmax=1.0)
    axes[2].set_title('weak_residual')
    axes[3].imshow(channels[1], cmap='viridis', vmin=0.0, vmax=1.0)
    axes[3].set_title('|time_grad|')
    axes[4].imshow(channels[2], cmap='viridis', vmin=0.0, vmax=1.0)
    axes[4].set_title('|freq_grad|')
    for ax in axes:
        ax.axis('off')

    short_name = sample.image_path.stem.replace('_abnormal', '')
    out_name = f"{sample.dataset}__{sample.scene}__{sample.noise}__gt{sample.gt_area:05d}__{short_name}.png"
    fig.suptitle(f"{sample.dataset} | {sample.scene} | {sample.noise} | {sample.image_path.name}", fontsize=11)
    out_path = output_dir / sample.dataset / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    return {
        'dataset': sample.dataset,
        'scene': sample.scene,
        'noise': sample.noise,
        'gt_area': sample.gt_area,
        'image_path': str(sample.image_path),
        'gt_path': str(sample.gt_path),
        'figure_path': str(out_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--output-dir', type=Path, default=Path('analysis_outputs/morph_fusion_featuremaps'))
    args = parser.parse_args()

    transform = MorphFusionDualGradChannels()
    manifest_rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in DATASET_CONFIG:
        samples = collect_top_samples(dataset, args.top_k)
        for sample in samples:
            manifest_rows.append(export_figure(sample, args.output_dir, transform))

    manifest_path = args.output_dir / 'selected_samples.csv'
    with manifest_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'dataset', 'scene', 'noise', 'gt_area', 'image_path', 'gt_path', 'figure_path'
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_path = args.output_dir / 'README.md'
    lines = [
        '# Morph Fusion DualGrad Feature Maps',
        '',
        'Mainline visualized here: `morph_fusion_dualgrad`.',
        'Panels: original input, GT mask, weak_residual, |time_grad|, |freq_grad|.',
        '',
        'Selected samples:',
        '',
    ]
    for row in manifest_rows:
        lines.append(f"- `{row['dataset']}` | `{row['scene']}` | `{row['noise']}` | `gt_area={row['gt_area']}` | `{Path(row['image_path']).name}`")
    summary_path.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
