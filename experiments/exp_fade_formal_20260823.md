# FADE baseline：统一少样本频谱异常检测实验

日期：2026-08-23

## 1. 方法与源码

本实验适配 FADE（*FADE: Few-shot/zero-shot Anomaly Detection Engine using Large Vision-Language Model*，2024）到本项目的频谱图像级异常检测协议。

- 论文：[arXiv:2409.00556](https://arxiv.org/abs/2409.00556)
- 官方源码：[BMVC-FADE/BMVC-FADE](https://github.com/BMVC-FADE/BMVC-FADE)
- 官方 GEM 组件：[WalBouss/GEM](https://github.com/WalBouss/GEM)
- 本地源码：`references/FADE/`
- 适配代码：[tools/eval_fade_cls.py](../tools/eval_fade_cls.py)

FADE 的核心逻辑保持为：冻结 CLIP/GEM 视觉模型、仅使用正常 support 的 patch 建立记忆库、计算测试 patch 到正常 gallery 的 top-1 cosine distance，并可与语言提示分数做平均融合。本实验不训练、不微调，也不使用测试标签建立记忆库。

由于官方代码依赖旧版 `open_clip`，适配器使用当前环境的关键字参数创建模型，再包裹官方 `GEMWrapper`；GEM 的特征和自注意力实现没有改写。为保证 OpenAI ViT-B/16 权重与结构匹配，显式启用了 QuickGELU。

## 2. 统一配置

| 项目 | 设置 |
|---|---|
| 视觉模型 | `ViT-B-16`，OpenAI 权重 |
| GEM 深度 | 7 |
| 输入 | Resize 256 → CenterCrop 224 |
| 归一化 | OpenAI CLIP mean/std |
| 视觉特征 | CLIP patch feature |
| 语言特征 | CLIP global feature |
| 特征尺度 | 单尺度 224（资源受控的图像级适配） |
| 视觉距离 | support patch gallery 的 top-1 `(1-cosine)/2` |
| 视觉图像分数 | patch 距离图空间最大值 |
| 语言提示 | `references/FADE/prompts/winclip_prompt.json`，类别为 `radio frequency spectrogram` |
| `both` 融合 | `(language score + vision score) / 2` |
| TTA | 不使用；外部 baseline 只用原始正常 support |

这是 FADE 方法在本项目协议上的可复现实验配置，不宣称等同于官方 MVTec/VisA 的原始排行榜设置。官方代码支持多尺度；本次主结果固定为 224 单尺度，使各目标域实验资源和预处理可控。每个输出目录的 `protocol.json` 保存了实际运行配置。

## 3. 项目协议结果

指标均为百分数：AUROC、AUPRC 越高越好，FPR@95%TPR 越低越好。

### 3.1 Public RF：k-per-frequency

三档使用同一测试集，只改变正常 support manifest。

| Support | AUROC | AUPRC | FPR@95%TPR | 输出 |
|---:|---:|---:|---:|---|
| k=1 | 59.78 | 10.66 | 91.92 | [results.csv](../analysis_outputs/fade_public_rf_k1_20260823/results.csv) |
| k=2 | 65.20 | 13.40 | 88.19 | [results.csv](../analysis_outputs/fade_public_rf_k2_20260823/results.csv) |
| k=4 | 67.40 | 15.41 | 86.56 | [results.csv](../analysis_outputs/fade_public_rf_k4_20260823/results.csv) |

每档 15 个 signal×JSR cell，测试集固定，未使用测试标签选 support。

### 3.2 In-house RF

当前 RF 协议共 4 个场景、5 类信号、60 个测试 cell；每个目标场景按每频段 1 张正常图建立 support。

| Support | AUROC | AUPRC | FPR@95%TPR | 输出 |
|---|---:|---:|---:|---|
| per-frequency | 52.12 | 31.92 | 90.11 | [results.csv](../analysis_outputs/fade_inhouse_rf_20260823/results.csv) |

本次使用的是当前数据定义中的 `deceptive_signal`。早期 manifest 中的 `wideband_pulse` 属于旧版协议，因此没有复用。

### 3.3 OFDMA v2 realistic

使用 30 个 target scene；每个场景分别建立 1/2/4 个正常观测 support。每个观测包含 21 个 SU，先取 SU 最大异常分数，再对 30 个场景做 macro average。

| Support | AUROC | AUPRC | FPR@95%TPR |
|---:|---:|---:|---:|
| 1-shot | 55.02 | 55.82 | 94.80 |
| 2-shot | 60.00 | 60.30 | 91.80 |
| 4-shot | 60.59 | 61.27 | 90.57 |

完整结果：[results.csv](../analysis_outputs/fade_ofdma_20260823/results.csv)，共 90 个 scene-shot job、540 个 scene-scope 结果行和 18 个宏平均行。

### 3.4 FedJam

只使用 spectrogram 图像；support 从 train 中的 benign（label=0）抽取，测试使用独立 test 全部 7,200 条记录，异常标签只用于最终指标。

| Support | AUROC | AUPRC | FPR@95%TPR |
|---:|---:|---:|---:|
| 1-shot | 59.00 | 79.37 | 86.28 |
| 2-shot | 58.71 | 79.27 | 87.44 |
| 4-shot | 59.31 | 79.78 | 85.56 |

完整结果：[results.csv](../analysis_outputs/fade_fedjam_20260823/results.csv)。FedJam 的异常比例与其他数据集不同，因此 AUPRC 不应跨数据集直接比较。

## 4. 运行输出

- Public RF k=1：[analysis_outputs/fade_public_rf_k1_20260823](../analysis_outputs/fade_public_rf_k1_20260823)
- Public RF k=2：[analysis_outputs/fade_public_rf_k2_20260823](../analysis_outputs/fade_public_rf_k2_20260823)
- Public RF k=4：[analysis_outputs/fade_public_rf_k4_20260823](../analysis_outputs/fade_public_rf_k4_20260823)
- In-house RF：[analysis_outputs/fade_inhouse_rf_20260823](../analysis_outputs/fade_inhouse_rf_20260823)
- OFDMA：[analysis_outputs/fade_ofdma_20260823](../analysis_outputs/fade_ofdma_20260823)
- FedJam：[analysis_outputs/fade_fedjam_20260823](../analysis_outputs/fade_fedjam_20260823)

本次 FADE 使用 GPU1；GPU0 未使用，GPU2 上原有 ResAD 任务未被抢占。
