# GRETEL 频谱图适配正式实验

日期：2026-08-26
方法出处：[GRETEL: A Graph Attention Network for Low-SNR Spectrum Anomaly Detection in IoT Communications](../references/Hussain%20等%20-%202026%20-%20GRE%E2%84%A1%20A%20Graph%20Attention%20Network%20for%20Low-SNR%20Spectrum%20Anomaly%20Detection%20in%20IoT%20Communications.pdf)

## 1. 实验结论

GRETEL 已按本文统一的 normal-only few-shot 协议完成四类数据集的正式实验。结果表明，
当前将 GRETEL 从原始频谱输入适配到单张 PNG 频谱图后，整体检测能力明显弱于本文
SpectraMemAD，也没有表现出稳定的 shot 增益。因此它适合作为真实存在的通信频谱图网络
基线保留在主比较表中，但不适合作为本文方法的竞争性主线。

## 2. 与原方法的关系

本实验是 **GRETEL 的频谱图域适配**，不是原论文输入条件下的逐位复现。当前项目的统一
输入是 PNG/dB 频谱图，没有原始 IQ 快拍和物理 SNR，因此做了以下最小适配：

- 将频率轴划分为 16 个相邻频带节点；每个节点使用频带平均功率、功率标准差、相对频带对比度和位置编码作为节点特征。
- 图边采用节点自身连接与相邻频带连接，保留 GRETEL 的图注意力建模思路。
- 使用教师图自编码器、冻结教师、带正常记忆的学生网络，并以重构误差、教师—学生嵌入差异和注意力差异之和作为异常分数。
- 由于缺少物理 SNR，节点的相对频带对比度只作为图像域代理变量；不能把它解释为原论文中的真实 SNR 输入。

GRETEL 在本协议中需要在当前目标场景的正常 support 上拟合教师和学生网络，因此它属于
**normal-only、support-fitted baseline**，不是本文免训练的记忆式方法。训练过程中不读取
异常样本、测试样本或测试标签；模型种子为 101、202、303，最终报告三次模型种子的均值和
标准差。

## 3. 统一实验协议

| 数据集 | 正常 support | 测试与汇总 | 结果目录 |
|---|---|---|---|
| In-house RF | 每个正式评估单元使用固定 seed=111 的 1 个宽带正常观测；输入裁切块仍属于同一次观测 | 60 个评估单元，图像级指标等权宏平均 | [`20260826_gretel_inhouse_rf_formal`](../analysis_outputs/20260826_gretel_inhouse_rf_formal/) |
| Public RF | `k=1/2/4-per-frequency`，分别为 16/32/64 张正常 support；三个 support 清单嵌套 | 15 个评估单元，完整测试集，单元等权宏平均 | [`k1`](../analysis_outputs/20260826_gretel_public_rf_k1_formal/)、[`k2`](../analysis_outputs/20260826_gretel_public_rf_k2_formal/)、[`k4`](../analysis_outputs/20260826_gretel_public_rf_k4_formal/) |
| OFDMA | 每个目标场景使用同场景 1/2/4 个正常观测；每个观测含 21 个 SU 频谱图 | 30 个目标场景；先取 21 个 SU 分数最大值，再做场景宏平均 | [`20260826_gretel_ofdma_formal`](../analysis_outputs/20260826_gretel_ofdma_formal/) |
| FedJam | 从 benign 训练样本中按固定 seed=111 抽取 1/2/4-shot support | 完整测试集 7,200 张频谱图 | [`20260826_gretel_fedjam_formal`](../analysis_outputs/20260826_gretel_fedjam_formal/) |

所有结果都同时报告 AUROC、AUPRC 和 FPR@95%TPR。AUROC/AUPRC 越高越好，
FPR@95%TPR 越低越好。每个结果目录中的 `protocol.json` 保存了输入适配、support 规则、
模型配置、随机种子和数据划分信息；`scores/` 保存逐样本分数。

## 4. 正式结果

结果为三次模型种子的均值，括号内为标准差。

### In-house RF（1-shot）

| AUROC | AUPRC | FPR@95%TPR |
|---:|---:|---:|
| 63.96 ± 0.26 | 43.80 ± 0.42 | 68.61 ± 0.44 |

### Public RF（full-test）

| support | AUROC | AUPRC | FPR@95%TPR |
|---|---:|---:|---:|
| k=1 | 57.30 ± 0.10 | 12.78 ± 0.60 | 88.90 ± 0.29 |
| k=2 | 57.89 ± 0.15 | 13.06 ± 0.90 | 88.46 ± 0.45 |
| k=4 | 57.27 ± 0.26 | 13.72 ± 0.67 | 87.57 ± 0.44 |

### OFDMA（目标场景冷启动）

| support | AUROC | AUPRC | FPR@95%TPR |
|---|---:|---:|---:|
| 1-shot | 63.58 ± 0.30 | 69.71 ± 0.28 | 90.81 ± 0.23 |
| 2-shot | 63.92 ± 0.31 | 69.91 ± 0.29 | 90.89 ± 0.20 |
| 4-shot | 63.39 ± 0.13 | 69.60 ± 0.19 | 90.78 ± 0.17 |

### FedJam（spectrogram-only）

| support | AUROC | AUPRC | FPR@95%TPR |
|---|---:|---:|---:|
| 1-shot | 45.66 ± 0.45 | 71.78 ± 0.18 | 95.46 ± 0.10 |
| 2-shot | 43.14 ± 0.46 | 70.55 ± 0.15 | 97.04 ± 0.16 |
| 4-shot | 42.28 ± 0.22 | 70.33 ± 0.07 | 96.76 ± 0.13 |

## 5. 结果解释与论文使用方式

GRETEL 在 OFDMA 的 barrage 类异常上能够得到较高的 AUROC，但在 pilot、random-hop 和
sweep 等类型上接近随机，导致整体宏平均只有约 63% AUROC。Public RF 上增加 support
后 AUROC 只在 k=2 略有提升，k=4 又回落；FedJam 上 shot 增加反而使整体 AUROC 下降。
这说明图注意力和重构式训练在当前 PNG 频谱协议中容易把数据集特有的功率、背景或占用模式
当作正常结构，不能稳定覆盖跨场景、跨异常类型变化。

因此论文中建议将其表述为：

> GRETEL 是真实存在的通信频谱图网络方法；本文在统一 PNG、normal-only few-shot 协议下
> 对其进行了频谱图域适配，并如实报告适配结果。该结果用于说明通信专用图网络在当前
> 冷启动协议中的适用边界，不声称为原论文 raw-IQ 实验的完整复现。

## 6. 复现入口

- 模型与训练：[gretel_spectral.py](../tools/gretel_spectral.py)
- 统一评估器：[eval_gretel_spectral.py](../tools/eval_gretel_spectral.py)
- OFDMA 汇总：[results_gretel_spectral.csv](../analysis_outputs/20260826_gretel_ofdma_formal/results_gretel_spectral.csv)
- OFDMA 协议：[protocol.json](../analysis_outputs/20260826_gretel_ofdma_formal/protocol.json)
