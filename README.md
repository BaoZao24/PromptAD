# PromptAD for RF Spectrogram Anomaly Detection

> **English** | [中文](./README_zh.md)

This repository adapts PromptAD for radio-frequency (RF) spectrogram anomaly detection.

The current research target is **few-shot anomaly detection**.

## Current Method

The active baseline is a PatchCore-style CNN local normal-memory detector:

```text
PatchCore-style baseline
= CNN local normal gallery
+ nearest-neighbour anomaly score
```

The current extension is:

```text
Confidence-Gated Dual Visual Normality Fusion

ViT Normal Gallery
+ CNN Local Gallery
+ label-free CNN confidence-gated OR fusion
```

Branch roles:

| Branch | Role |
|---|---|
| ViT Normal Gallery | Scores structural patch-level deviation from few-shot normal spectrograms. |
| CNN Local Gallery | Scores local texture and energy-detail deviation from few-shot normal spectrograms. |
| Confidence-gated OR fusion | Uses the CNN score rank as a label-free gate; CNN evidence contributes only when it is relatively stronger than the ViT evidence. |

PromptAD `text + ViT patch anomaly map` is retained only as a reference baseline. The current method uses visual ViT and CNN normal-memory evidence.

Detailed method description: [docs/现有方案介绍.md](./docs/现有方案介绍.md).

## Experiment Protocol

Current experiments must use a few-shot normal-only protocol:

```text
support / gallery:
  target normal samples only
  one normal image per frequency band
  no target abnormal samples

test:
  independent test normal samples
  test abnormal samples
  evaluated by anomaly type and JSR
```

Important constraints:

- Do not use target abnormal samples for training or gallery construction.
- Use PatchCore-style CNN local normal memory as the main baseline.
- Keep PromptAD `text + ViT patch anomaly map` only as a reference comparison.
- Report per-anomaly, per-JSR, per-anomaly average, and overall macro average.

## Current Result Status

The few-shot protocol is being redefined.

Results should be filled only after rerunning the few-shot protocol.

## Installation

```bash
conda create -n prompt_ad python=3.10
conda activate prompt_ad
bash install.sh
```

Key dependencies include PyTorch, `open_clip_torch`, `timm`, `transformers`, `opencv-python`, `scikit-learn`, `pandas`, `loguru`, and `tqdm`.

## Dataset Layout

Expected RF data root:

```text
/mnt/data/wangbei/data/datasets/
├── normal/
│   ├── WeaponMuseum_spectrum/
│   ├── Playground_spectrum/
│   ├── TimeSquare_spectrum/
│   └── Gymnasium_spectrum/
├── burst/
├── chirp/
├── dsss/
├── pulse/
└── wideband_pulse/
```

Each signal dataset contains per-scene folders with:

```text
normal/{jsr}/
abnormal/{jsr}/
groundtruth/{jsr}/
```

## Useful Commands

PatchCore-style CNN local normal-memory baseline:

```bash
python tools/eval_patchcore_cls.py \
  --protocol rf_target \
  --rf-train-mode pooled \
  --normal-sampling per_frequency \
  --output-root analysis_outputs/patchcore_fewshot_baseline
```

Calibrated dual visual fusion from existing ViT/CNN scores:

```bash
python tools/eval_cls_dual_visual_evidence_fusion.py \
  --protocol public_rf \
  --vit-score-dir analysis_outputs/20260706_pairtta_stft_shift_blur_public/scores \
  --cnn-score-dir analysis_outputs/20260704_public_rf_official_patchcore_cls/scores \
  --output-root analysis_outputs/public_rf_calibrated_dual_visual
```

The exact output directory should be changed per formal run.

## Useful Documentation

- [Current method](./docs/现有方案介绍.md)
- [Docs index](./docs/README.md)
- [Score mechanism](./docs/分数机制说明.md)
- [Prompt mechanism](./docs/提示词机制说明.md)

## Notes

The upstream MVTec and VisA interfaces are still present for compatibility, but the active RF research line is the few-shot target-normal protocol described above.
