# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection (CVPR2024). The project implements few-shot anomaly detection using CLIP-based prompt learning with only normal samples for training.

## Development Environment

### Setup
```bash
conda create -n prompt_ad python=3.10
conda activate prompt_ad
bash install.sh
```

### Key Dependencies
- PyTorch with CUDA 11.8
- open_clip_torch (CLIP model backbone)
- timm, transformers
- opencv-python, scikit-learn, pandas

## Running Experiments

### Train and Test Classification (Image-Level)
```bash
python train_cls.py --dataset mvtec --class_name carpet --k-shot 1
```

### Train and Test Segmentation (Pixel-Level)
```bash
python train_seg.py --dataset mvtec --class_name carpet --k-shot 1
```

### Run All Classes
```bash
python run_cls.py  # image-level for all classes
python run_seg.py  # pixel-level for all classes
```

### Supported Datasets
- `mvtec` - MVTec AD dataset
- `visa` - VisA dataset
- `spectrum` - Spectrum dataset
- `sample` - Sample dataset
- `dcase2025` - DCASE2025 audio dataset

### Common Arguments
- `--dataset`: dataset name (mvtec/visa/spectrum/sample/dcase2025)
- `--class_name`: object/category name
- `--k-shot`: number of few-shot samples (1, 2, 4)
- `--backbone`: ViT-B-16-plus-240 or ViT-B-16
- `--n_ctx`, `--n_ctx_ab`: context tokens for normal/abnormal prompts
- `--n_pro`, `--n_pro_ab`: number of prompts for normal/abnormal
- `--Epoch`: training epochs (10 for cls, 100 for seg)
- `--lr`: learning rate (default 0.002)
- `--vis`: enable visualization (default True for seg)

## Codebase Architecture

```
PromptAD/
├── PromptAD/           # Core model implementation
│   ├── CLIPAD/         # CLIP model wrapper (adapted from OpenCLIP)
│   ├── model.py        # PromptAD model: PromptLearner + CLIP backbone
│   └── ad_prompts.py   # Prompt templates for anomaly descriptions
├── datasets/           # Dataset loaders
│   ├── dataset.py      # Base CLIPDataset class
│   ├── mvtec.py        # MVTec dataset loader
│   ├── visa.py         # VisA dataset loader
│   └── ...             # Other datasets
├── utils/              # Utilities
│   ├── training_utils.py   # Seed setup, experiment directory management
│   ├── eval_utils.py       # Evaluation helpers
│   ├── metrics.py          # AUROC, PRO metrics calculation
│   ├── csv_utils.py        # Results logging
│   └── visualization.py    # Anomaly map visualization
├── train_cls.py        # Classification training entry point
├── train_seg.py        # Segmentation training entry point
├── run_cls.py          # Batch run classification for all classes
└── run_seg.py          # Batch run segmentation for all classes
```

## Core Components

### PromptLearner (`PromptAD/model.py:28-156`)
Generates learnable prompt embeddings:
- **Normal prompts**: Learnable context vectors + class name
- **Abnormal prompts (handle)**: Predefined anomaly templates (e.g., "damaged {}", "flawed {}")
- **Abnormal prompts (learned)**: Learnable context + anomaly prefix + class name

### PromptAD Model (`PromptAD/model.py:158-403`)
Main model with two anomaly scoring mechanisms:
1. **Textual anomaly score**: Image features vs. text prompt features similarity
2. **Visual anomaly score**: Image features vs. normal image feature gallery distance

### Training Pipeline
1. **Feature Gallery Building**: Encode all normal training samples
2. **Prompt Tuning**: Optimize prompt parameters using:
   - Vision-to-Text cross-entropy loss
   - Triplet loss (anchor=normal, positive=abnormal text features)
   - Abnormal prompt consistency loss
3. **Evaluation**: Compute Image-AUROC (cls) or Pixel-AUROC (seg)

### Data Flow (`datasets/__init__.py`)
All datasets expose:
- `dataset_classes`: List of category names
- `load_<dataset>()`: Returns train/test splits with image paths, ground truth masks, labels

## Experiment Output Structure
```
result/<dataset>/<class_name>/k_<shot>/
├── csv/            # CSV files with AUROC results
├── checkpoint/     # Saved model checkpoints
└── imgs/           # Visualization of anomaly maps
```

## Git Workflow

Follow the project's Git workflow:
1. Check current branch status before starting work
2. Create feature branches: `git checkout -b task/<description>`
3. Make atomic commits with descriptive messages
