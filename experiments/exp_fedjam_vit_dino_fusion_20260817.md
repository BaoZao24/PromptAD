# FedJam：ViT + DINOv2 局部分支替换实验（2026-08-17）

## 实验目的

验证 DINOv2 patch 特征能否替换当前 ViT+CNN 架构中的 CNN 局部分支。

## 实验设置

- ViT 分支保持不变：PromptAD `ViT-B-16-plus-240` 的 `layer1+layer2` patch 特征。
- DINO 分支：冻结的 DINOv2-B/14（`vit_base_patch14_dinov2`，224×224）。
- DINO 使用正常 support 建立 patch memory，不训练、不使用异常样本建库。
- DINO 图像级分数沿用 CNN 分支的 top 10% patch 距离均值，不使用 UniVAD 的融合公式。
- ViT 使用原有 50% farthest coreset、5-NN 和最大 patch 距离。
- 融合使用原有 support-only confidence gate，只将 CNN 分数替换为 DINO 分数。
- FedJam：benign-only 嵌套 1/2/4-shot，完整 7200 张独立测试集，TTA=none。

## 结果

| shot | 分支 | AUROC ↑ | AUPRC ↑ | FPR@95%TPR ↓ |
|---:|---|---:|---:|---:|
| 1 | ViT-only | 84.24 | 94.93 | 81.44 |
| 1 | DINO-only | **86.18** | **95.61** | **76.61** |
| 1 | ViT+DINO confidence fusion | 84.24 | 94.93 | 81.44 |
| 2 | ViT-only | 84.49 | 94.90 | 77.39 |
| 2 | DINO-only | **89.52** | **96.66** | **64.28** |
| 2 | ViT+DINO confidence fusion | 84.59 | 94.93 | 76.94 |
| 4 | ViT-only | 85.05 | 95.00 | 72.28 |
| 4 | DINO-only | **91.31** | **97.19** | **53.22** |
| 4 | ViT+DINO confidence fusion | 85.19 | 95.04 | **70.78** |

当前 ViT+CNN 无 TTA 主线的完整结果为：

| shot | ViT+CNN AUROC | ViT+CNN AUPRC | ViT+CNN FPR@95%TPR |
|---:|---:|---:|---:|
| 1 | 84.24 | 94.92 | 81.44 |
| 2 | 84.53 | 94.91 | 76.56 |
| 4 | 85.05 | 95.00 | 72.22 |

ViT+CNN 对照来源：[`20260815_tta_position_no_tta/fedjam/summary.json`](../analysis_outputs/exploratory/20260815_tta_position_no_tta/fedjam/summary.json)。

## 结果分析

1. DINO-only 明显优于 CNN-only，说明在 FedJam 频谱图上，DINOv2 的局部 patch memory
   比 ResNet18 layer3 更能区分局部异常。相对 CNN-only，DINO-only 的 AUROC 提升为
   11.07、10.12、8.04 个百分点（1/2/4-shot）。
2. 直接套用当前 confidence gate 后，ViT+DINO 只比 ViT-only 略高。现有门控是为 CNN
   局部分支设计的保守补充规则，并不会充分利用一个明显强于 ViT 的 DINO 分支。
3. 1-shot 的 support reference 只有一个样本，support rank gate 不会激活，因此融合结果
   与 ViT-only 完全相同。
4. 目前可以确认“DINOv2 局部记忆分支有价值”，但还不能直接确认“把 CNN 替换成 DINO
   后，现有门控架构就能显著提升”。下一步应单独评估更适合 DINO 分支的融合方式。

## 结果文件

- [评估脚本](../tools/eval_fedjam_vit_dino_fusion.py)
- [完整结果](../autoresearch/fedjam-vit-dino-260817/)
- [汇总结果](../autoresearch/fedjam-vit-dino-260817/summary.json)
- [指标表](../autoresearch/fedjam-vit-dino-260817/metrics.csv)
