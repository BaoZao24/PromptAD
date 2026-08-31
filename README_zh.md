# SpectraMemAD：训练免费射频频谱异常检测

> [English](./README.md) | **中文**

本仓库以 PromptAD 的视觉编码器为实现基础，用于训练免费的射频频谱图（RF spectrogram）异常检测。

当前研究目标是 **少样本 anomaly detection**。

## 当前方案

论文主表统一比较传统频谱统计、已完成的外部异常检测基线和本文方法。ViT-only、
CNN-only 等内部视觉分支单独放在消融表和消融图中，不作为主表独立方法。外部方法的覆盖
范围和文献编号见 [论文实验部分](./docs/paper/论文实验部分.md)。

```text
CNN-only（本文内部消融）
= ResNet18 layer3 local normal gallery
+ nearest-neighbour anomaly score

传统频谱统计
= ED / 谱熵 / 谱平坦度
+ 谱峭度 / CA-CFAR
```

当前方法是：

```text
Confidence Fusion（ViT+CNN）

ViT Normal Gallery
+ CNN Local Gallery
+ 置信度融合
```

各分支作用：

| 分支 | 作用 |
|---|---|
| ViT Normal Gallery | 用少量正常频谱图的 ViT patch 特征，计算结构级异常偏离。 |
| CNN Local Gallery | 用少量正常频谱图的 CNN 局部特征，计算局部纹理和能量细节偏离。 |
| 置信度融合 | 只用正常 support 参考分布中的 CNN 排名与 CNN-over-ViT 优势形成非负修正，不读取其他测试图像，不暴露融合调参接口。 |

外部 PatchCore [12](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf) 和传统频谱统计是正式对比方法；CNN-only 是本文内部消融，不能
与外部 PatchCore 混称。当前方法使用 ViT 与 CNN 的视觉 normal memory 证据。仓库
保留上游 PromptAD 代码仅用于实现兼容，不将其作为论文对比行。

详细说明见：[docs/paper/现有方案介绍.md](./docs/paper/现有方案介绍.md)。

## 当前实验协议

当前实验必须使用少样本 normal-only 协议：

```text
support / gallery:
  只使用 target normal 样本
  按数据集协议从 target normal 样本建立 support；In-house RF 每场景使用 1 个宽带正常
  观测（1-shot，频率裁切块不单独计数），Public RF 使用每频段 1/2/4 个正常观测，
  OFDMA/FedJam 使用 1/2/4-shot
  不使用 target abnormal 样本

test:
  独立 test normal 样本
  test abnormal 样本
  按异常类型和 JSR 分别评估
```

关键约束：

- 不使用 target abnormal 训练或建库。
- 外部异常检测基线和有出处的传统频谱统计是对比基线；其中 SAIFE 是通信频谱深度生成式
  方法，不归入通用视觉方法；ViT/CNN 单分支只作内部消融。
- ViT normal-memory 单分支只作内部消融，不属于 baseline。
- 四个数据集统一使用 ED、CA-CFAR、SCSE、KLD-Ref、IAD-PER、SAIFE、UDMA、PatchCore、
  WinCLIP 和 Ours；其中 IAD-PER 是主表中唯一的 VAE 基线。ViT-only、CNN-only 只作为内部
  消融；谱熵、谱平坦度、谱峭度、ICA-Frozen、VAE-MSE、Deep SVDD、PaDiM、STFPM 的结果
  保留在补充记录中。

## 当前结果状态

当前正式主线启用 Paired TTA 作为辅助正常记忆扩充。它不属于核心创新；论文表格必须如实标明
协议设置，并保留同划分的无 TTA 消融。仅因这一协议调整，无需重跑外部基线，但缺少带 TTA 主线
结果的数据集需要补跑 Ours。

当前正式 RF support-only 实验在 In-house RF 上得到
**91.97/81.80/22.00**（AUROC/AUPRC/FPR@95%TPR）。Public RF 采用主线
`k=1/2/4-per-frequency` 协议，Ours 的三档 AUROC 为 **81.74/82.97/83.74**；完整
三指标表见 [`docs/paper/论文实验部分.md`](./docs/paper/论文实验部分.md) 表 1 和表 1b。

下面的 RF 结果来自当前正式的 ViT+CNN 置信度融合版本；功率残差探索分支不进入主表。

| 数据集 | Ours：Confidence Fusion AUROC |
|---|---:|
| In-house RF（本项目自测数据 + 合成干扰） | **91.97** |
| Public RF [14](https://doi.org/10.1007/s11036-009-0199-9) | **81.74 / 82.97 / 83.74**（k=1/2/4） |

ViT-only、CNN-only、Direct OR 等内部支线不作为独立主方法，统一放在论文实验部分的
内部消融表和图中。

数据集出处直接列在这里：In-house RF 的正常背景来自本项目自测/自采集的 RF 记录，并在其上
注入合成干扰生成异常样本，因此属于本项目自测的派生数据集，不是外部公开数据集。Public RF
的正常底座来自
[14](https://doi.org/10.1007/s11036-009-0199-9)及其公开测量数据入口，异常样本由本项目
注入干扰派生；OFDMA 基于 [15](https://arxiv.org/abs/2606.02102) 的公开模拟论文、源码和
数据改编生成；FedJam 直接使用未修改的公开数据集，出处为
[13](https://arxiv.org/abs/2508.09369)。

In-house RF 正式协议包含 4 个场景（`WeaponMuseum_spectrum`、`Playground_spectrum`、
`TimeSquare_spectrum`、`Gymnasium_spectrum`）、5 类合成注入（burst、chirp、DSSS、pulse、
deceptive）和 60 个 signal/scene/strength 单元；每个场景使用 1 个宽带正常观测
（1-shot），并因网络输入尺寸限制裁切为 24 个频率窗口。代码中的 `per_frequency` 仅为
裁切块的内部组织名称。正式强度为：burst/chirp/DSSS 使用 −10/−20/−30 dB，
pulse 使用 −20/−30/−40 dB，deceptive 使用 strong/medium/weak；`wideband_pulse` 不进入
当前五类主比较。场景、频谱切片和各类注入参数详见
[`docs/paper/论文实验部分.md`](./docs/paper/论文实验部分.md) 第 4.1.1 节。

数据集参考文献：

13. I. Panitsas, I. Ofeidis, and L. Tassiulas, “FedJam: Multimodal Federated Learning Framework for Jamming Detection,” [原论文](https://arxiv.org/abs/2508.09369)、[数据集](https://huggingface.co/datasets/panitsasi/FedJam)、[代码](https://github.com/panitsasi/fedJam)。
14. M. Wellens and P. Mähönen, “Lessons Learned from an Extensive Spectrum Occupancy Measurement Campaign and a Stochastic Duty Cycle Model,” [DOI](https://doi.org/10.1007/s11036-009-0199-9)、[公开测量数据入口](http://download.mobnets.rwth-aachen.de)。
15. A. Schösser, M. Salehi, S. Ma, P. Schulz, and G. Fettweis, “Spectrum Anomaly Detection in OFDMA Systems: Simulation Framework and Benchmark Dataset,” [论文](https://arxiv.org/abs/2606.02102)、[源码](https://github.com/akdd11/ofdma-spectrum-anomalies-simulation)、[Zenodo 数据](https://doi.org/10.5281/zenodo.20341906)。

包括 VAE、SAIFE、Deep SVDD、PaDiM、STFPM、WinCLIP 和官方 PatchCore 在内的 In-house RF
新版 support 全部重跑记录见
[`analysis_outputs/20260725_target_scene_self_rf_baselines_seed111/README.md`](./analysis_outputs/20260725_target_scene_self_rf_baselines_seed111/README.md)。

已核对的结果文件见
[`analysis_outputs/20260713_final_visual_metrics/README.md`](./analysis_outputs/20260713_final_visual_metrics/README.md)。

Public RF 五类合并宏平均和 wideband 逐单元结果见
[`analysis_outputs/20260730_public_rf_wideband_formal_seed111/summary/`](./analysis_outputs/20260730_public_rf_wideband_formal_seed111/summary/)。

OFDMA target-scene 冷启动的正式方案分别计算 21 个 SU 的分数，取最大值后再使用
ViT+CNN support-only 置信度融合。30 个独立 test 场景的 Ours 结果如下：

| Shot | AUROC | AUPRC | FPR@95%TPR |
|---:|---:|---:|---:|
| 1 | 85.30 | 88.80 | 69.77 |
| 2 | 89.53 | 92.01 | 57.63 |
| 4 | 92.75 | 94.55 | 45.90 |

完整协议、逐方法结果和统一表见
[`analysis_outputs/20260731_ofdma_v2_realistic_unified/README.md`](./analysis_outputs/20260731_ofdma_v2_realistic_unified/README.md)。
同一 target-scene 协议下补做的官方 PatchCore 结果见
[`analysis_outputs/20260810_ofdma_patchcore_formal/README.md`](./analysis_outputs/20260810_ofdma_patchcore_formal/README.md)：
1/2/4-shot 的 AUROC/AUPRC/FPR@95%TPR 分别为
70.14/76.78/86.03、75.83/81.29/81.37、81.81/86.14/72.50。

FedJam 的 1/2/4-shot 图像模态补充实验，以及带文献出处的 ED、CA-CFAR、SCSE、KLD-Ref、
IAD-PER、SAIFE、UDMA、WinCLIP、PatchCore 结果分别见
[`analysis_outputs/20260810_fedjam_visual_baselines_formal/`](./analysis_outputs/20260810_fedjam_visual_baselines_formal/)
和 [`analysis_outputs/20260810_fedjam_traditional_fewshot_formal/`](./analysis_outputs/20260810_fedjam_traditional_fewshot_formal/)。
完整论文表格见 [`docs/paper/论文实验部分.md`](./docs/paper/论文实验部分.md) 的表 9–11。

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
  --normal-sampling per_frequency \
  --output-root analysis_outputs/patchcore_fewshot_baseline
```

基于当前 ViT/CNN 分数运行置信度融合：

```bash
python tools/eval_cls_aux_cnn_gallery.py \
  --protocol public_rf \
  --normal-sampling per_frequency \
  --output-root analysis_outputs/current_public_aux_cnn

python tools/eval_cls_dual_visual_evidence_fusion.py \
  --protocol public_rf \
  --vit-score-dir analysis_outputs/current_public_vit/scores \
  --cnn-score-dir analysis_outputs/current_public_aux_cnn/scores \
  --vit-reference-dir analysis_outputs/current_public_vit_reference \
  --cnn-reference-dir analysis_outputs/current_public_cnn_reference \
  --output-root analysis_outputs/current_public_confidence_fusion
```

辅助 CNN 在所有数据集固定使用 ResNet18 layer3；融合入口默认使用 support-only
置信度校准，不读取测试批次统计量，也不提供 CNN 类型或融合调参参数。代码中的旧
`gate` 字段和兼容路径只属于实现标识，不是论文方法名称。

## 重要文档

- [现有方案介绍](./docs/paper/现有方案介绍.md)
- [文档索引](./docs/README.md)
- [分数机制说明](./docs/method/分数机制说明.md)
- [提示词机制说明](./docs/method/提示词机制说明.md)

## 说明

仓库仍保留上游 MVTec / VisA 接口以兼容原 PromptAD，但当前 RF 主线以本文描述的少样本 target-normal 协议为准。

[12]: https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf
