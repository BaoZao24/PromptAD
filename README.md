# PromptAD for RF Spectrogram Anomaly Detection

> **English** | [中文](./README_zh.md)

This repository adapts PromptAD for radio-frequency (RF) spectrogram anomaly detection.

The current method is **Dual Visual Normality Calibration for RF PromptAD**:

```text
PromptAD RF prompt score
+ CLIP target-normal gallery distance
+ ResNet18 local normal-gallery distance
=> final anomaly score
```

The model is trained with target-domain normal samples only. Abnormal samples are used only for evaluation.

## Current Method

The current system has three evidence branches:

| Branch | Role |
|---|---|
| PromptAD RF prompt branch | Provides the semantic normal/abnormal direction for RF spectrograms. |
| CLIP normal gallery | Measures whether a test image is far from target normal examples in the PromptAD/CLIP feature space. |
| ResNet18 local normal gallery | Measures local morphology deviation with frozen ResNet18 features. |

Image-level score:

```text
S_final = S_promptad + lambda_clip * D_clip + lambda_resnet * D_resnet
```

Segmentation map, used as a supporting localization experiment:

```text
M_final = M_text * (1 + beta_clip * M_clip) * (1 + beta_resnet * M_resnet)
```

Detailed method description: [docs/现有方案介绍.md](./docs/现有方案介绍.md).

## Experiment Protocol

Current experiments use a pooled target-domain normal-only protocol:

```text
signals:
  burst_signal, chirp_signal, dsss_signal, pulse_signal

scenes:
  WeaponMuseum_spectrum
  Playground_spectrum
  TimeSquare_spectrum
  Gymnasium_spectrum

JSR:
  burst_signal: m10db, m20db, m30db
  chirp_signal: m10db, m20db, m30db
  dsss_signal : m10db, m20db, m30db
  pulse_signal: m20db, m30db, m40db
```

Training data:

```text
48 cells = 4 signal types x 4 scenes x 3 JSR
each cell contributes 75% normal samples
all normal samples are pooled to train one universal checkpoint
```

Test data:

```text
each cell is evaluated separately
test = remaining 25% normal + all abnormal samples for that cell
```

Important constraints:

- No target abnormal sample is used for training.
- No per-class best checkpoint is used.
- One universal `overall-best.pt` is selected by macro average.

## Latest Results

The following results are from the current in-house RF target dataset only:

```text
/mnt/data/wangbei/data/datasets/
burst / chirp / dsss / pulse
```

CLS Image-AUROC macro on this dataset:

| Method | Image-AUROC |
|---|---:|
| PromptAD RF+RGB baseline | 78.4088 |
| CLIP normal gallery | 88.6536 |
| ResNet18 layer3 gallery | 90.5237 |
| **PromptAD + CLIP gallery + ResNet18 gallery** | **91.8955** |

SEG Pixel-AUROC macro on this dataset:

| Method | Pixel-AUROC |
|---|---:|
| pooled RF+RGB baseline | 89.3342 |
| CLIP gallery gate | 91.5458 |
| ResNet18 gallery gate | 91.5577 |
| **Dual gallery gate** | **92.1947** |

Result files:

```text
analysis_outputs/20260629_cls_dual_gallery_fusion/
analysis_outputs/20260629_seg_dual_gallery_fusion/
```

Public `RF_SPE_PNG` CLS external validation has been run with the following 12 abnormal cells:

```text
burst: m30db / m40db / m50db
chirp: m40db / m50db / m55db
dsss:  m30db / m40db / m50db
pulse: m30db / m40db / m50db
```

RF_SPE_PNG CLS Image-AUROC macro:

| Method | Image-AUROC |
|---|---:|
| PromptAD score with current checkpoint | 48.4355 |
| ResNet18 layer3 normal gallery | 70.3481 |
| Best dual fusion in sweep | 77.0137 |
| **CLIP normal gallery only** | **79.4368** |

Result files:

```text
analysis_outputs/20260702_public_rf_cls_dual_gallery_3jsr/
```

RF_SPE_PNG SEG and spectrum CLS/SEG are still pending. External dataset results must be reported separately from the in-house target dataset.

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
└── pulse/
```

Each signal dataset contains per-scene folders with:

```text
normal/{jsr}/
abnormal/{jsr}/
groundtruth/{jsr}/
```

## Main Training Entry

Train/evaluate the pooled universal PromptAD baseline:

```bash
python train_rf_target_pooled_universal.py \
  --task cls \
  --method pooled_rf_rgb \
  --output-root analysis_outputs/20260627_method_funnel \
  --epochs 20 \
  --eval-every 1 \
  --seed 111 \
  --batch-size 400
```

Segmentation baseline:

```bash
python train_rf_target_pooled_universal.py \
  --task seg \
  --method pooled_rf_rgb \
  --output-root analysis_outputs/20260627_method_funnel \
  --epochs 20 \
  --eval-every 5 \
  --seed 111 \
  --batch-size 400
```

Supported pooled methods are defined in `train_rf_target_pooled_universal.py`.

## Gallery Evaluation

Evaluate the current CLS dual-gallery method from a trained pooled checkpoint:

```bash
python tools/eval_cls_dual_gallery_fusion.py \
  --output-root analysis_outputs/20260629_cls_dual_gallery_fusion \
  --formal-baseline-csv analysis_outputs/20260627_method_funnel/results_cls_pooled_rf_rgb.csv \
  --gpu-id 0 \
  --batch-size 200 \
  --resnet-layers layer3 \
  --clip-topk 50 \
  --image-top-ratio 0.1 \
  --clip-lambdas 0.5 1.0 1.5 \
  --resnet-lambdas 0.5 1.0 1.5 2.0
```

Evaluate the SEG dual-gallery method:

```bash
python tools/eval_seg_dual_gallery_fusion.py \
  --output-root analysis_outputs/20260629_seg_dual_gallery_fusion \
  --formal-baseline-csv analysis_outputs/20260627_method_funnel/results_seg_pooled_rf_rgb.csv \
  --gpu-id 0 \
  --batch-size 100 \
  --resnet-layers layer2 layer3 \
  --clip-betas 0.1 0.25 0.5 \
  --resnet-betas 0.1 0.25 0.5
```

## Useful Documentation

- [Current method](./docs/现有方案介绍.md)
- [Docs index](./docs/README.md)
- [Score mechanism](./docs/分数机制说明.md)
- [Prompt mechanism](./docs/提示词机制说明.md)
- [Current experiment overview](./docs/agent_handoff_method_funnel_overview.md)

## Notes

The upstream MVTec and VisA interfaces are still present for compatibility, but the active RF research line is the pooled target-normal protocol described above.
