# Ours 中 ViT 替换为 DINOv2：FedJam 协议核对与校正结果

日期：2026-08-17
数据集：FedJam 公开数据集
完整测试：7200 条（1800 正常、5400 异常）

## 先说结论

之前的 DINO patch 强结果没有做错，且使用的也不是 G/14，而是 DINOv2-B/14：
`vit_base_patch14_dinov2`。这次第一版替换实验下降，主要是因为我把 DINO 的评分方式错误地改成了 ViT 的评分方式。

之前表现较强的 DINO patch 协议是：

- DINOv2-B/14，224×224；
- 正常 patch memory 保留全部 patch，不做 50% coreset；
- 1-NN；
- 取最异常 top 10% patch 距离的均值。

第一版“严格特征替换”则使用了 ViT 的 50% coreset、5-NN 和最大 patch 距离。因此它可以作为“完全照搬 ViT 评分器”的控制实验，但不能用来判断 DINO patch 本身是否有效。

## 公平替换设置

校正实验保留当前 Ours 的：

- CNN 分支：ImageNet ResNet18 layer3；
- 频谱 TTA：原图、时间方向双向偏移、频率响应扰动；
- support-only confidence gate；
- benign-only 嵌套 1/2/4-shot；
- 完整独立测试集；
- 不训练、不使用异常样本建库。

只将 ViT 视觉分支替换为 DINOv2 patch 分支，并沿用已经验证过的 DINO patch 评分协议。

## 结果

| shot | 当前 Ours：ViT+CNN |  |  | DINOv2+CNN（替换 ViT） |  |  |
|---:|---:|---:|---:|---:|---:|---:|
|  | AUROC | AUPRC | FPR@95%TPR | AUROC | AUPRC | FPR@95%TPR |
| 1 | 85.14 | 95.21 | 79.61 | **87.81** | **96.16** | **71.56** |
| 2 | 85.90 | 95.36 | 71.67 | **89.61** | **96.64** | **59.89** |
| 4 | 86.11 | 95.37 | 70.67 | **90.98** | **97.05** | **52.56** |

相对当前 Ours，DINOv2 替换版 AUROC 提升：

- 1-shot：+2.67 个百分点；
- 2-shot：+3.71 个百分点；
- 4-shot：+4.87 个百分点。

## 两组实验的含义

| 版本 | DINO 评分方式 | 1/2/4-shot AUROC | 用途 |
|---|---|---:|---|
| 严格特征替换 | 50% coreset、5-NN、最大 patch 距离 | 83.20/84.33/86.06 | 控制变量检查，不作为 DINO 最佳结果 |
| 校正替换 | 全 patch memory、1-NN、top10% 均值 | 87.81/89.61/90.98 | 与之前强 DINO patch 协议一致，作为候选方案 |

校正结果中的 DINO-only 分数与之前 DINO patch + 频谱 TTA 实验基本一致（87.82/89.61/90.97），说明特征提取、support/test 划分和 TTA 都是对的。

## 最新设置：全量 patch memory + 1-NN

按最新方案，不再使用 50% farthest coreset，ViT 和 DINO 两条视觉记忆分支都使用 support 的全部 patch 和 1-NN。DINO 仍保留已验证的 top10% patch 均值聚合，CNN、TTA 和门控不变。

| shot | 全量 1-NN 当前 Ours：ViT+CNN | DINOv2+CNN（替换 ViT） |
|---:|---:|---:|
| 1 | 85.20 | **87.81** |
| 2 | 85.77 | **89.61** |
| 4 | 86.28 | **90.98** |

全量 1-NN 没有改变 DINO 的结果，但使当前 ViT 基线略有变化；因此后续主实验可以统一采用全量 1-NN，避免 coreset 引入额外变量。

## G/14 说明

本次 FedJam 两组 DINO 实验均使用 DINOv2-B/14，不是 G/14。之前 RF 上的 UniVAD-Texture-adapted 实验才使用过官方 DINOv2-G/14。不能把两者的结果混在一起比较。

## 结果文件

- 严格替换版：[autoresearch/ours-vit-vs-dinov2-fedjam-260817-full](../autoresearch/ours-vit-vs-dinov2-fedjam-260817-full)
- 校正替换版：[autoresearch/ours-vit-vs-dinov2-fedjam-260817-dino-protocol-full](../autoresearch/ours-vit-vs-dinov2-fedjam-260817-dino-protocol-full)
- 全量 1-NN 版：[autoresearch/ours-full-memory-1nn-vs-dinov2-fedjam-260817-full](../autoresearch/ours-full-memory-1nn-vs-dinov2-fedjam-260817-full)
- 评估脚本：[eval_fedjam_ours_dinov2_replace.py](../tools/eval_fedjam_ours_dinov2_replace.py)
