# PromptAD 射频频谱图异常检测

> [English](./README.md) | **中文**

本仓库基于 PromptAD 改造，用于射频频谱图（RF spectrogram）异常检测。

当前研究目标是 **少样本 anomaly detection**。

## 当前方案

当前主 baseline 使用 PatchCore-style CNN 局部正常特征库：

```text
PatchCore-style baseline
= CNN local normal gallery
+ nearest-neighbour anomaly score
```

当前改进方案是：

```text
Confidence-Gated Dual Visual Normality Fusion

ViT Normal Gallery
+ CNN Local Gallery
+ 无标签 CNN 置信度门控 OR 融合
```

各分支作用：

| 分支 | 作用 |
|---|---|
| ViT Normal Gallery | 用少量正常频谱图的 ViT patch 特征，计算结构级异常偏离。 |
| CNN Local Gallery | 用少量正常频谱图的 CNN 局部特征，计算局部纹理和能量细节偏离。 |
| Confidence-gated OR fusion | 根据 CNN 分数的相对排名计算无标签置信度，仅在 CNN 证据强于 ViT 时增强最终分数。 |

PromptAD `text + ViT patch anomaly map` 只保留为参考 baseline；当前方法使用 ViT 与 CNN 的视觉 normal memory 证据。

详细说明见：[docs/现有方案介绍.md](./docs/现有方案介绍.md)。

## 当前实验协议

当前实验必须使用少样本 normal-only 协议：

```text
support / gallery:
  只使用 target normal 样本
  每个频段选 1 张正常图
  不使用 target abnormal 样本

test:
  独立 test normal 样本
  test abnormal 样本
  按异常类型和 JSR 分别评估
```

关键约束：

- 不使用 target abnormal 训练或建库。
- PatchCore-style CNN local normal memory 是当前主 baseline。
- PromptAD `text + ViT patch anomaly map` 只作为参考对照。
- 结果需要报告各异常、各 JSR、各异常均值和总 macro average。

## 当前结果状态

少样本协议正在重新整理。

正式结果需要按少样本协议重跑后再填写。

## 安装

```bash
conda create -n prompt_ad python=3.10
conda activate prompt_ad
bash install.sh
```

主要依赖包括 PyTorch、`open_clip_torch`、`timm`、`transformers`、`opencv-python`、`scikit-learn`、`pandas`、`loguru`、`tqdm`。

## 数据组织

RF 数据默认路径：

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

每类信号数据包含：

```text
normal/{jsr}/
abnormal/{jsr}/
groundtruth/{jsr}/
```

## 常用命令

PatchCore-style CNN 局部正常特征库 baseline：

```bash
python tools/eval_patchcore_cls.py \
  --protocol rf_target \
  --rf-train-mode pooled \
  --normal-sampling per_frequency \
  --output-root analysis_outputs/patchcore_fewshot_baseline
```

基于已有 ViT/CNN 分数做校准双视觉融合：

```bash
python tools/eval_cls_dual_visual_evidence_fusion.py \
  --protocol public_rf \
  --vit-score-dir analysis_outputs/20260706_pairtta_stft_shift_blur_public/scores \
  --cnn-score-dir analysis_outputs/20260704_public_rf_official_patchcore_cls/scores \
  --output-root analysis_outputs/public_rf_calibrated_dual_visual
```

正式实验时需要按具体 run 修改输出目录。

## 重要文档

- [现有方案介绍](./docs/现有方案介绍.md)
- [文档索引](./docs/README.md)
- [分数机制说明](./docs/分数机制说明.md)
- [提示词机制说明](./docs/提示词机制说明.md)

## 说明

仓库仍保留上游 MVTec / VisA 接口以兼容原 PromptAD，但当前 RF 主线以本文描述的少样本 target-normal 协议为准。
