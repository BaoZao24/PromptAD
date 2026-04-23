# PromptAD — Spectrum Signal Anomaly Detection

This repository extends [PromptAD (CVPR 2024)](http://arxiv.org/abs/2404.05231) for **few-shot cross-site anomaly detection on radio-frequency spectrum signals**.

The model is trained with only normal (clean) spectrogram patches from one site and tested on anomalous spectrogram patches (Burst / Chirp / DSSS jamming) from other sites.

---

## Installation

```bash
conda create -n prompt_ad python=3.10
conda activate prompt_ad
bash install.sh
```

---

## Dataset

### Directory layout

```
/mnt/data/wangbei/data/datasets/
├── normal/                        # Normal (clean) training patches, shared across all signal types
│   ├── WeaponMuseum_spectrum/     # ~192 patches per site
│   ├── Playground_spectrum/
│   ├── TimeSquare_spectrum/
│   └── Gymnasium_spectrum/
├── burst/                         # Burst jamming test data
│   └── {site}/
│       ├── normal/{noise_level}/  # Clean test patches
│       ├── abnormal/{noise_level}/# Anomalous patches
│       └── groundtruth/{noise_level}/
├── chirp/                         # Chirp jamming (same structure)
└── dsss/                          # DSSS jamming (same structure)
```

### Sites and noise levels

| Variable | Values |
|---|---|
| Sites | `WeaponMuseum_spectrum`, `Playground_spectrum`, `TimeSquare_spectrum`, `Gymnasium_spectrum` |
| Signal types | `burst_signal`, `chirp_signal`, `dsss_signal` |
| Noise levels (JSR) | `m10db` (−10 dB), `m20db` (−20 dB), `m30db` (−30 dB) |

---

## Cross-site experiments

**Train on `WeaponMuseum_spectrum`, test on the other 3 sites.**

### Run all signal types in parallel (recommended)

```bash
# Auto-detect all GPUs, run 30 jobs (3 datasets × 3 sites × 3 JSR) in parallel
python run_cross_site_all.py --vis

# Specify GPUs explicitly
python run_cross_site_all.py --gpus 0 1 2 3 --epochs 50 --vis

# Preview commands without running
python run_cross_site_all.py --dry-run
```

After all jobs finish, a summary table is printed and `plot_heatmap.py` is called automatically.

### Run one signal type at a time

```bash
python run_cross_site_burst.py [--epochs 50] [--gpu-id 0] [--dry-run]
python run_cross_site_chirp.py [--epochs 50] [--gpu-id 0] [--dry-run]
python run_cross_site_dsss.py  [--epochs 50] [--gpu-id 0] [--dry-run]
```

### Result directory layout

```
./result/
└── {dataset}/
    └── WeaponMuseum_spectrum_to_{test_site}/
        └── {noise_level}/
            └── k_24/
                ├── csv/Seed_111-results.csv   # Image-AUROC, Pixel-AUROC
                ├── checkpoint/
                └── imgs/                      # Score maps (when --vis True)
```

---

## Visualization

### Heatmap (AUROC across sites × JSR × signal types)

```bash
python plot_heatmap.py --root-dir ./result --seed 111 --out ./result/heatmap_cross_site.png
```

### Score map comparison (input / ground truth / anomaly score)

```bash
# Sample 1 image per condition from all completed experiments
python plot_scoremap.py --root-dir ./result --out ./result/scoremap_compare.png

# Filter to a specific dataset / site / noise level
python plot_scoremap.py --dataset burst_signal --scene Playground_spectrum --noise m10db
```

---

## Training a single run directly

```bash
# Cross-site (train on WeaponMuseum_spectrum, test on Playground_spectrum)
python train_cls.py \
    --dataset burst_signal \
    --class_name Playground_spectrum \
    --train-site WeaponMuseum_spectrum \
    --noise-level m10db \
    --k-shot 24 --Epoch 50 --gpu-id 0 --vis True
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `mvtec` | `burst_signal` / `chirp_signal` / `dsss_signal` / `mvtec` / `visa` |
| `--class_name` | — | Test site name |
| `--train-site` | same as `class_name` | Train site (cross-site only) |
| `--noise-level` | `m10db` | JSR level for signal datasets |
| `--k-shot` | `1` | Number of normal training samples |
| `--Epoch` | `50` | Training epochs |
| `--vis` | `False` | Save score map images |
| `--backbone` | `ViT-B-16-plus-240` | CLIP backbone |
| `--seed` | `111` | Random seed |

---

## Project structure

```
PromptAD/
├── train_cls.py              # Main training / evaluation script
├── run_cross_site_all.py     # Parallel multi-GPU cross-site runner (all datasets)
├── run_cross_site_burst.py   # Sequential cross-site runner — burst
├── run_cross_site_chirp.py   # Sequential cross-site runner — chirp
├── run_cross_site_dsss.py    # Sequential cross-site runner — DSSS
├── plot_heatmap.py           # AUROC heatmap visualization
├── plot_scoremap.py          # Score map comparison visualization
├── datasets/
│   ├── __init__.py           # Dataloader factory
│   ├── dataset.py            # CLIPDataset (image loading + resize)
│   ├── burst_signal.py       # Burst signal loader
│   ├── chirp_signal.py       # Chirp signal loader
│   ├── dsss_signal.py        # DSSS signal loader
│   ├── mvtec.py / visa.py    # Original MVTec / VisA loaders
│   └── spectrum.py           # Generic spectrum loader
├── PromptAD/
│   ├── model.py              # PromptAD model definition
│   └── CLIPAD/               # CLIP backbone wrappers
└── utils/
    ├── training_utils.py     # Output directory helpers
    ├── csv_utils.py          # Result CSV writing
    ├── metrics.py            # AUROC / AUPRO metrics
    ├── eval_utils.py         # Evaluation utilities
    └── visualization.py      # Score map saving
```

---

## Citation

```bibtex
@article{li2024promptad,
  title={PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection},
  author={Li, Xiaofan and Zhang, Zhizhong and Tan, Xin and Chen, Chengwei and Qu, Yanyun and Xie, Yuan and Ma, Lizhuang},
  journal={arXiv preprint arXiv:2404.05231},
  year={2024}
}
```

## Acknowledgements

We thank [WinCLIP](https://github.com/caoyunkang/WinClip.git) and [CoOp](https://github.com/KaiyangZhou/CoOp.git) for their great open-source work.
