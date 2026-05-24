# PromptAD — Spectrum Signal Anomaly Detection

This repository extends [PromptAD (CVPR 2024)](http://arxiv.org/abs/2404.05231) for **few-shot anomaly detection on radio-frequency spectrum signals**.

The model is trained with only normal (clean) spectrogram patches and tested on anomalous spectrogram patches (Burst / Chirp / DSSS jamming).

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

### Result directory layout

```
./result/
└── {dataset}/
    └── {scene}/
        └── {noise_level}/
            └── k_24/
                ├── csv/Seed_111-results.csv   # Image-AUROC, Pixel-AUROC
                ├── checkpoint/
                └── imgs/                      # Score maps (when --vis True)
```

---

## Visualization

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
python train_cls.py \
    --dataset burst_signal \
    --class_name Playground_spectrum \
    --noise-level m10db \
    --k-shot 24 --Epoch 50 --gpu-id 0 --vis True
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `mvtec` | `burst_signal` / `chirp_signal` / `dsss_signal` / `mvtec` / `visa` |
| `--class_name` | — | Test site name |

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
