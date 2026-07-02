# PromptAD 射频频谱图异常检测

> [English](./README.md) | **中文**

本仓库基于 PromptAD 改造，用于射频频谱图（RF spectrogram）异常检测。

当前最新方案是：

```text
Dual Visual Normality Calibration for RF PromptAD
```

也就是：

```text
PromptAD RF 语义分数
+ CLIP target normal gallery 距离
+ ResNet18 local normal gallery 距离
=> 最终异常分数
```

训练阶段只使用 target normal 样本。异常样本只用于测试。

## 当前方案

当前系统有三条证据分支：

| 分支 | 作用 |
|---|---|
| PromptAD RF prompt 分支 | 提供 RF 频谱图的 normal / abnormal 语义方向。 |
| CLIP normal gallery | 在 PromptAD/CLIP 特征空间里判断测试图像是否远离 target normal 样本。 |
| ResNet18 local normal gallery | 用冻结 ResNet18 局部特征判断频谱局部形态是否偏离正常。 |

图像级分数：

```text
S_final = S_promptad + lambda_clip * D_clip + lambda_resnet * D_resnet
```

分割图作为辅助定位实验：

```text
M_final = M_text * (1 + beta_clip * M_clip) * (1 + beta_resnet * M_resnet)
```

详细说明见：[docs/现有方案介绍.md](./docs/现有方案介绍.md)。

## 当前实验协议

当前实验使用 target pooled normal-only 协议：

```text
信号类型:
  burst_signal, chirp_signal, dsss_signal, pulse_signal

场景:
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

训练集：

```text
48 个 cell = 4 类信号 x 4 个场景 x 3 个 JSR
每个 cell 取 75% normal
所有 normal 合并训练一个 universal checkpoint
```

测试集：

```text
每个 cell 单独测试
test = 剩余 25% normal + 当前 cell 的全部 abnormal
```

关键约束：

- 不使用 target abnormal 训练。
- 不使用 per-class best checkpoint。
- 只选择一个 `overall-best.pt` 作为通用 checkpoint。

## 最新结果

CLS Image-AUROC macro：

| 方法 | Image-AUROC |
|---|---:|
| pooled RF+RGB baseline | 86.3382 |
| CLIP normal gallery | 88.6536 |
| ResNet18 layer3 gallery | 90.5237 |
| **PromptAD + CLIP gallery + ResNet18 gallery** | **91.8955** |

SEG Pixel-AUROC macro：

| 方法 | Pixel-AUROC |
|---|---:|
| pooled RF+RGB baseline | 89.3342 |
| CLIP gallery gate | 91.5458 |
| ResNet18 gallery gate | 91.5577 |
| **Dual gallery gate** | **92.1947** |

结果目录：

```text
analysis_outputs/20260629_cls_dual_gallery_fusion/
analysis_outputs/20260629_seg_dual_gallery_fusion/
```

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
└── pulse/
```

每类信号数据包含：

```text
normal/{jsr}/
abnormal/{jsr}/
groundtruth/{jsr}/
```

## 主训练入口

训练/评估 pooled universal PromptAD baseline：

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

SEG baseline：

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

支持的方法定义在 `train_rf_target_pooled_universal.py` 里。

## Gallery 评估入口

CLS dual-gallery 评估：

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

SEG dual-gallery 评估：

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

## 重要文档

- [现有方案介绍](./docs/现有方案介绍.md)
- [文档索引](./docs/README.md)
- [分数机制说明](./docs/分数机制说明.md)
- [提示词机制说明](./docs/提示词机制说明.md)
- [当前实验总览](./docs/agent_handoff_method_funnel_overview.md)

## 说明

仓库仍保留上游 MVTec / VisA 接口以兼容原 PromptAD，但当前 RF 主线以本文描述的 pooled target-normal 协议为准。
