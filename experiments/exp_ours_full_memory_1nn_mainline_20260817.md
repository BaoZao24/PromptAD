# 原主线：全量正常 memory + 1-NN 主实验

日期：2026-08-17

本次只评估原来的 ViT+CNN 主线，不再运行或纳入 DINO 替换版。

## 结论

原主线在统一的“全量正常记忆库 + 1-NN”协议下已经完成三个数据集实验：

| 数据集 | 设置 | AUROC ↑ | AUPRC ↑ | FPR@95%TPR ↓ |
|---|---:|---:|---:|---:|
| In-house RF | 24 个正常 support / scene | **92.25** | **81.61** | **22.15** |
| Public RF | k=1 | **81.15** | **44.23** | **47.14** |
| Public RF | k=2 | **81.63** | **44.79** | **47.07** |
| Public RF | k=4 | **82.22** | **44.62** | **46.54** |
| OFDMA | 1-shot，30 个 test scene 平均 | **84.23** | **87.81** | **70.67** |
| OFDMA | 2-shot，30 个 test scene 平均 | **88.24** | **90.79** | **61.07** |
| OFDMA | 4-shot，30 个 test scene 平均 | **91.10** | **93.04** | **49.53** |

OFDMA 的总体 1/2/4-shot 平均结果分别为 84.23/88.24/91.10 AUROC，说明正常 support 数量增加时，主线整体稳定提升。

## 方法设置

- 不训练模型，只使用正常 support 建立记忆库。
- ViT：PromptAD 的 layer1+layer2 特征，使用全部正常 patch，1-NN，取最大 patch 距离。
- CNN：ImageNet ResNet18 的 layer3 局部特征，使用全部正常 patch，1-NN，取最异常 10% patch 距离的均值。
- 融合：使用 support-only confidence gate。门控阈值只由正常 support 得到，不使用异常样本、测试标签或测试批次统计量。
- RF 的记忆增强视图：原图、时间方向上移、时间方向下移、频率响应扰动。
- OFDMA 的记忆增强视图：原图、时间方向左移、时间方向右移、频率响应扰动。
- 所有增强视图都只由正常 support 产生，并放入全量正常 memory；不做 50% farthest coreset，也不使用 5-NN。
- OFDMA 先对每个观测的 21 个 sensing units 取最大异常证据，再计算观测级指标。

这里的“全量 memory”是指保留 support 的全部正常 patch 及其 TTA 视图；“1-NN”是每个测试 patch 只寻找距离最近的一个正常 patch。

## OFDMA 分场景设置

OFDMA 使用官方 test split 的 30 个场景，每个场景分别进行 1/2/4-shot support。每个 shot 的表格数值是 30 个场景的宏平均，不是把所有场景的图片直接混在一起计算。

## 结果文件

- In-house RF：[ours-full-memory-1nn-inhouse-260817](../autoresearch/ours-full-memory-1nn-inhouse-260817)
- Public RF k=1：[ours-full-memory-1nn-public-k1-pure-260817](../autoresearch/ours-full-memory-1nn-public-k1-pure-260817)
- Public RF k=2：[ours-full-memory-1nn-public-k2-pure-260817](../autoresearch/ours-full-memory-1nn-public-k2-pure-260817)
- Public RF k=4：[ours-full-memory-1nn-public-k4-pure-260817](../autoresearch/ours-full-memory-1nn-public-k4-pure-260817)
- OFDMA：[ours-full-memory-1nn-ofdma-pure-260817](../autoresearch/ours-full-memory-1nn-ofdma-pure-260817)
- 评估脚本：[eval_ours_dinov2_replacement_cross.py](../tools/eval_ours_dinov2_replacement_cross.py)（本次使用 `--control-only`，只执行原主线 ViT+CNN）

旧的 DINO 替换结果保留为历史探索记录，不进入本次主结果表，也不作为后续主线实验。
