# UniVAD 纹理适配版：DINOv2-B/14 公平规模复现实验（2026-08-18）

## 1. 实验目的

此前的 UniVAD 频谱图适配结果使用了 DINOv2-G/14。G/14 是超大规模视觉骨干，直接与我们的 ViT-B/16 + ResNet18 主线比较不够公平。本次只将 DINOv2 骨干替换为 B/14，其他设置保持不变，重新完成已有 UniVAD 适配实验。

这里的 UniVAD 仍然是**频谱图整体纹理路径适配版**，不是完整官方 UniVAD：频谱图没有 UniVAD 所需的物体/组件分割掩码，因此没有使用 C³、CAPM 和 GECM 组件路径。

## 2. 统一设置

| 项目 | 本次设置 |
|---|---|
| 方法名称 | `UniVAD-Texture-adapted (DINOv2-B/14)` |
| CLIP 分支 | CLIP-L/14-336，保持不变 |
| DINO 分支 | 冻结 DINOv2-B/14，模型名 `dinov2_vitb14` |
| 输入尺寸 | 336×336 |
| 适配方式 | 只用正常 support 建立 global/patch memory |
| 目标域训练 | 不训练、不微调、不使用异常样本 |
| 匹配方式 | 正常 memory 中的 1-NN 余弦相似度匹配 |
| 分数融合 | CLIP patch、DINO patch、CLIP 文本图固定等权平均，再加 global 分数 |

本次与旧 G/14 结果使用相同的 support、测试划分、shot 规则、图像预处理、评分代码路径和指标计算方式，唯一实质变化是 DINOv2-G/14 → DINOv2-B/14。

需要说明的是：B/14 解决的是原先 DINOv2-G/14 过大的主要不公平问题，但 UniVAD 仍保留 CLIP-L/14-336，因此这是一组**更公平、资源更可控的 baseline**，不是与 Ours 完全同参数量的严格等算力比较。

## 3. DINOv2-B/14 正式结果

指标说明：AUROC/AUPRC 越高越好，FPR@95%TPR 越低越好。所有数值均为百分比。

### 3.1 In-house RF

60 个 `signal × scene × JSR` 测试单元等权宏平均。

| 方法 | DINO 骨干 | AUROC | AUPRC | FPR@95%TPR |
|---|---|---:|---:|---:|
| UniVAD-Texture-adapted | B/14 | **88.94** | **78.37** | **28.79** |

### 3.2 Public RF

使用固定的 5,120 张正常测试池和完整异常测试集；support 为 nested k-per-frequency，15 个 signal/JSR 单元按 k 分别统计。

| Support | DINO 骨干 | AUROC | AUPRC | FPR@95%TPR |
|---:|---|---:|---:|---:|
| k=1 | B/14 | 80.47 | 39.26 | 46.40 |
| k=2 | B/14 | 81.80 | 40.52 | 44.19 |
| k=4 | B/14 | **83.10** | **44.05** | **44.29** |

### 3.3 FedJam

完整独立测试集 7,200 条，正常 support 使用 nested 1/2/4-shot。

| Support | DINO 骨干 | AUROC | AUPRC | FPR@95%TPR |
|---:|---|---:|---:|---:|
| 1-shot | B/14 | 89.86 | 96.99 | 72.44 |
| 2-shot | B/14 | 90.53 | 97.13 | 65.89 |
| 4-shot | B/14 | **91.46** | **97.36** | **58.22** |

### 3.4 OFDMA

沿用已有的 `test_000` 完整场景：4 个正常 support observation、200 个测试 observation，每个 observation 包含 21 个 SU 频谱图；先在 SU 级评分，再取 observation 内最大分数。

| Support | DINO 骨干 | AUROC | AUPRC | FPR@95%TPR |
|---:|---|---:|---:|---:|
| 1-shot | B/14 | 87.96 | 90.81 | 68.00 |
| 2-shot | B/14 | **90.15** | **92.18** | **59.00** |
| 4-shot | B/14 | 89.73 | 91.77 | 58.00 |

## 4. 与旧 DINOv2-G/14 结果的规模对照

下面只用于说明骨干规模影响，不把 G/14 和 B/14 当作同等计算预算下的方法排名。

| 数据集/Support | G/14 AUROC | B/14 AUROC | B−G |
|---|---:|---:|---:|
| In-house RF 宏平均 | 88.47 | 88.94 | +0.47 |
| Public RF k=1 | 81.04 | 80.47 | −0.58 |
| Public RF k=2 | 81.66 | 81.80 | +0.14 |
| Public RF k=4 | 82.33 | 83.10 | +0.76 |
| FedJam 1-shot | 93.24 | 89.86 | −3.38 |
| FedJam 2-shot | 93.69 | 90.53 | −3.16 |
| FedJam 4-shot | 94.06 | 91.46 | −2.61 |
| OFDMA 1-shot | 80.30 | 87.96 | +7.66 |
| OFDMA 2-shot | 83.73 | 90.15 | +6.42 |
| OFDMA 4-shot | 85.85 | 89.73 | +3.88 |

结果说明：减小模型并不会在所有数据集上同步提升性能。FedJam 对 G/14 更有利，而 OFDMA 对 B/14 更有利；这说明模型容量、频谱域差异和数据集异常类型之间存在交互。B/14 的价值主要是提供更公平、资源更可控的 baseline，而不是保证数值一定超过 G/14。

## 5. 论文中的使用建议

1. 主 baseline 表使用 `UniVAD-Texture-adapted (DINOv2-B/14)`，因为它与我们的主线规模更接近。
2. G/14 结果保留在补充表或模型规模对照表中，明确标注为“大规模骨干参考结果”，不与 B/14 和 Ours 做无条件排名。
3. 不能把本实验称为完整官方 UniVAD 复现，应继续使用“UniVAD 纹理路径频谱适配版”的表述。
4. 论文中同时引用 UniVAD 方法论文和 DINOv2 backbone 论文；本实验的实现是统一 RF/OFDMA 协议下的透明适配，不声称复现官方物体分割流程。

## 6. 结果文件与代码

- In-house RF：[B/14 全量 JSON](../autoresearch/univad-b14-260818/inhouse_full_60cells.json)；[运行日志](../autoresearch/univad-b14-260818/inhouse_full_60cells.log)
- Public RF：[B/14 全量 JSON](../autoresearch/univad-b14-260818/public_full_k1k2k4.json)；[运行日志](../autoresearch/univad-b14-260818/public_full_k1k2k4.log)
- FedJam：[B/14 全量 JSON](../autoresearch/univad-b14-260818/fedjam_full_1shot2shot4shot.json)；[运行日志](../autoresearch/univad-b14-260818/fedjam_full_1shot2shot4shot.log)
- OFDMA：[B/14 `test_000` JSON](../autoresearch/univad-b14-260818/ofdma_test000_full.json)；[运行日志](../autoresearch/univad-b14-260818/ofdma_test000_full.log)
- In-house 入口：[eval_univad_rf_fewshot.py](../tools/eval_univad_rf_fewshot.py)
- Public RF 入口：[eval_univad_public_rf.py](../tools/eval_univad_public_rf.py)
- FedJam 入口：[eval_univad_fedjam.py](../tools/eval_univad_fedjam.py)
- OFDMA 入口：[eval_univad_ofdma.py](../tools/eval_univad_ofdma.py)
