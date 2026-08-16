# UniVAD RF 适配实验记录（2026-08-15）

> 全部数据集的 UniVAD 结果已另行汇总到：[UniVAD 全部实验结果独立记录](exp_univad_all_results_20260816.md)。

## 当前状态

已完成 In-house RF 的正式 60 个测试单元。由于官方 UniVAD 依赖物体/组件分割，而 RF 频谱图没有对应的
工业物体，本实验只使用其整体纹理路径，统一命名为 `UniVAD-Texture-adapted (DINOv2-G)`，不声称是
官方完整 UniVAD 复现。

正式 In-house RF 宏平均结果如下：UniVAD-Texture-adapted 的 AUROC/AUPRC/FPR@95%TPR 为
`88.47/76.75/30.17`，当前 Ours（support-only、TTA=none）为 `91.41/80.18/23.58`。
因此在这套完整协议上，UniVAD 适配版 AUROC 低 `2.94` 个百分点，AUPRC 低 `3.43` 个百分点，
FPR@95%TPR 高 `6.58` 个百分点。早期 3 单元预筛不能代表完整 In-house RF 结论。

使用 DINOv2-B 的早期适配版平均 AUROC 为 84.51%，Ours 为 93.96%；这组结果仅用于检查
适配流程，不能代表官方 backbone。

## 适配边界

官方 UniVAD 的主要组成是 C³ 组件聚类、CAPM 组件内 patch matching 和 GECM 图增强组件建模。
这些模块需要物体分割或组件掩码。当前 RF 频谱图是完整的时间—频率信号图，没有工业物体边界，
因此本实验采用：

- 保留 UniVAD 的 CLIP-L/14 整体/patch matching 思路；
- 预筛版使用 DINOv2-B/14；正式全量版使用官方 DINOv2-G/14 权重作为 DINOv2 特征分支；
- 仅使用正常 support 建立 CLIP 与 DINO patch memory；
- 不使用 C³、GroundingDINO、SAM 和 GECM；
- 不训练目标域模型，不使用异常样本建库或调参。

官方 DINOv2-G 权重约 4.23 GB，现已完整下载并使用分块编码/分块近邻匹配运行成功。官方代码
原本一次性建立 24 张 support 的全 patch 相似度矩阵，会申请约 30 GB 临时显存；修正为
分块后，GPU 1 峰值约 7.7 GB，没有 OOM。

## 预筛结果

所有单元使用 In-house RF 正式 manifest、同一正常 support 和独立测试集。下面两张预筛表中的
Ours 数值也已改为当前 60 单元 support-only（TTA=none）主线对应单元；正式结论以全量表为准。

| 信号 | 场景 | 强度 | UniVAD-Texture-B AUROC | Ours AUROC | 差值 |
|---|---|---:|---:|---:|---:|
| burst | WeaponMuseum | m30 dB | 67.01 | 87.59 | -20.581 |
| chirp | Playground | m20 dB | 93.22 | 99.70 | -6.481 |
| DSSS | TimeSquare | m20 dB | 93.31 | 94.60 | -1.293 |
| **三单元宏平均** | — | — | **84.51** | **93.96** | **-9.452** |

UniVAD 适配版的三单元 AUPRC 分别为 49.77%、88.63% 和 85.26%；FPR@95%TPR 分别为
90.57%、17.50% 和 16.94%。

## 官方 DINOv2-G 纹理适配结果

该组使用官方 DINOv2-G 权重、CLIP-L/14 和分块匹配；仍然只运行整体纹理路径，不包含 C³、
GroundingDINO、SAM 和 GECM。因此名称仍为 `UniVAD-Texture-adapted`，不能称为完整 UniVAD。

| 信号 | 场景 | 强度 | UniVAD-Texture-G AUROC | Ours AUROC | 差值 |
|---|---|---:|---:|---:|---:|
| burst | WeaponMuseum | m30 dB | 91.58 | 87.59 | +3.997 |
| chirp | Playground | m20 dB | 99.73 | 99.70 | +0.035 |
| DSSS | TimeSquare | m20 dB | 95.32 | 94.60 | +0.712 |
| **三单元宏平均** | — | — | **95.54** | **93.96** | **+1.581** |

对应 AUPRC 为 78.38%、99.65%、87.87%；FPR@95%TPR 为 45.28%、0.00%、11.29%。

## 正式 In-house RF 全量结果（60 单元）

本组实验使用全部正常/异常测试样本，不再采用此前的 8+8 冒烟数量；每个单元单独计算指标，最后对
60 个单元做等权宏平均。Ours 使用当前主线的 support-only confidence fusion，`TTA=none`。

| 方法 | AUROC ↑ | AUPRC ↑ | FPR@95%TPR ↓ |
|---|---:|---:|---:|
| UniVAD-Texture-adapted (DINOv2-G) | 88.47 | 76.75 | 30.17 |
| **Ours（当前主线）** | **91.41** | **80.18** | **23.58** |
| UniVAD − Ours | -2.94 | -3.43 | +6.58 |

这里的 UniVAD 结果仍只包含整体 CLIP/DINO patch matching，不包含需要物体组件掩码的 C³、CAPM
组件路径和 GECM；因此它是面向频谱图的透明适配 baseline，而不是完整官方 UniVAD。

结果文件：
[`inhouse_full_60cells.json`](../autoresearch/univad-official-260816/inhouse_full_60cells.json)。

## 正式实验范围

当前已完成 In-house RF、Public RF 和 FedJam；OFDMA 先完成了一个完整 target scene，用于
确认 21-SU 聚合和跨 shot 行为，30 场景宏平均仍未运行。

- [x] In-house RF：60 个 `signal × scene × JSR` 单元；每频段 1 张正常 support；宏平均 AUROC、AUPRC、FPR@95%TPR；
- [x] Public RF：15 个单元，嵌套 `k=1/2/4-per-frequency`；三个 k 均报告 AUROC、AUPRC、FPR@95%TPR；
- [~] OFDMA：已完成 `test_000` 的 1/2/4-shot、200 observation；30 场景宏平均待补；
- [x] FedJam：benign-only 1/2/4-shot；只用 spectrogram；完整 7,200 条 test；报告三项指标。

四套实验必须复用现有 support manifest、测试划分和 shot 规则。UniVAD 结果统一标为
`UniVAD-Texture-adapted (DINOv2-G)`，并注明省略 C³/GECM；Ours 使用对应正式协议的现有结果。

## Public RF 全量结果（15 单元 × 3 个 k）

Public RF 使用固定 5,120 张正常 test 图和全部异常 test 图；三个 support memory 严格嵌套，
每张正常 test 图只编码一次。表中差值为 `UniVAD − Ours`；AUROC/AUPRC 越高越好，FPR@95%TPR
越低越好。

| Support | UniVAD AUROC | Ours AUROC | UniVAD AUPRC | Ours AUPRC | UniVAD FPR95 | Ours FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| k=1 | 81.04 | **81.41** | 37.83 | **45.08** | **43.46** | 46.31 |
| k=2 | 81.66 | **83.55** | 40.38 | **45.30** | 44.00 | **43.98** |
| k=4 | 82.33 | **84.18** | 41.18 | **45.73** | 43.91 | **43.50** |

Public RF 上 UniVAD 随 support 增加而提升，但 AUROC 和 AUPRC 三档均低于当前 Ours；FPR 只在
k=1 略低，k=2/4 基本持平或略高。

## FedJam 全量结果（7,200 条 test）

FedJam 只使用 spectrogram 图像；train 中仅抽取 benign 作为嵌套 1/2/4-shot support，test
包含 1,800 条 benign 和 5,400 条异常记录，KPI 时序没有参与评分。

| Support | UniVAD AUROC | Ours AUROC | UniVAD AUPRC | Ours AUPRC | UniVAD FPR95 | Ours FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| 1-shot | **93.24** | 84.24 | **98.02** | 94.92 | **58.44** | 81.44 |
| 2-shot | **93.69** | 84.53 | **98.12** | 94.91 | **49.56** | 76.56 |
| 4-shot | **94.06** | 85.05 | **98.21** | 95.00 | **46.22** | 72.22 |

FedJam 是目前 UniVAD 适配版优势最明显的数据集；但这仍是整体纹理路径适配，不代表
完整 UniVAD 的 C³/CAPM/GECM 分支结果。

## OFDMA 单场景完整验证（test_000）

本次完成 `test_000` 的全部 200 个 observation（每个 observation 21 张 SU 图），先对 SU
图逐张评分，再取 observation 内 21 个分数的最大值。该表只代表一个 target scene，不能替代
30 场景宏平均。

| Support | UniVAD AUROC | Ours AUROC | UniVAD AUPRC | Ours AUPRC | UniVAD FPR95 | Ours FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| 1-shot | **80.30** | 79.86 | **84.54** | 84.53 | **78.00** | 84.00 |
| 2-shot | 83.73 | **89.24** | 86.80 | **91.43** | 65.00 | **51.00** |
| 4-shot | 85.85 | **89.03** | 88.12 | **91.70** | 62.00 | 62.00 |

单场景结果显示 UniVAD 随 support 增加而提升，但在 2/4-shot 仍低于 Ours；需要完成其余
29 个场景后，才能下 OFDMA 宏平均结论。

## 判断

跨数据集结果说明：在 In-house RF 和 Public RF 上，UniVAD 适配版整体低于 Ours；在 FedJam
上明显高于 Ours；OFDMA 的单场景结果则随 shot 增加而逐渐接近但仍低于 Ours。UniVAD 在
不同数据集上的差异说明，通用视觉 patch matching 并不能自动适应所有频谱分布；但它仍是
一个有代表性的强少样本视觉 baseline。

In-house RF 的正式 60 单元结果和 Public RF 的 45 个 cell-k 结果可以用于跨数据集结论；
OFDMA 当前只有 `test_000`，不能写成 30 场景宏平均。早期 3 单元中出现的轻微领先，也不能
作为跨场景结论。

它仍不是完整 UniVAD，不能把这组结果解读为完整 C³/GECM 方法在 RF 上优于 Ours。

论文中如需列出，应单独写成 `UniVAD-Texture-adapted (DINOv2-G)`，并在表注中说明省略了
物体组件分支；不能简称为官方完整 UniVAD。正式进入主表前，还需要在完整 In-house RF 和
至少一个 Public RF/FedJam 协议上完成相同适配。

## 复现入口

- 适配代码：`tools/eval_univad_rf_texture_b_adapted.py`
- 官方尝试入口：`tools/eval_univad_rf_fewshot.py`
- Public RF 入口：`tools/eval_univad_public_rf.py`
- FedJam 入口：`tools/eval_univad_fedjam.py`
- OFDMA 入口：`tools/eval_univad_ofdma.py`
- 实验结果：`autoresearch/univad-official-260816/`
- UniVAD 源码：`references/UniVAD/`
- UniVAD 论文：`references/papers/univad_cvpr_2025.pdf`
