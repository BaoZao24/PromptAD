# Ours 替换 ViT-L-14 实验（20260824）

## 目的

在 Ours 的双分支结构中，将全局 ViT-B-16-plus-240 替换为 ViT-L-14，CNN
分支、support-only confidence gate、正常参考构建、随机种子和数据协议保持不变，
检查更大 ViT 对四个数据集的影响。

## 实现与协议

- 新增 `ViT-L-14` backbone 选项；使用 `laion400m_e32` 预训练权重。
- ViT-L-14 使用 224×224 输入（其位置编码网格为 224 输入）；特征维度为
  layer1+layer2 拼接后的 2048。
- 为兼容更深的 ViT，V2V 特征 hook 保留最后 12 个 V2V block 内的相对位置，
  ViT-L-14 对应 block 14/19；ViT-B 的原有位置不变。
- ViT 使用 farthest coreset 0.5、5-NN mean、当前正式的
  `rf_spectral_response_v1` merged TTA；CNN 使用 ResNet18 layer3；融合使用
  support-only gate，seed=111。
- In-house 为 60 个评估单元，Public 为 k=1/2/4，OFDMA 为 30 场景的场景宏平均，
  FedJam 使用完整 7200 条测试样本。

## Ours 融合结果

指标顺序均为 **AUROC / AUPRC / FPR@95%TPR**（单位：%）。

| 数据集 | ViT-L-14 Ours |
|---|---:|
| In-house RF 1-shot | 90.16 / 72.95 / 26.70 |
| Public RF k=1 | 82.41 / 44.47 / 46.09 |
| Public RF k=2 | 83.23 / 45.70 / 44.47 |
| Public RF k=4 | 83.75 / 46.14 / 43.79 |
| OFDMA 1-shot | 77.52 / 81.68 / 79.10 |
| OFDMA 2-shot | 79.28 / 83.34 / 77.07 |
| OFDMA 4-shot | 82.86 / 86.38 / 71.90 |
| FedJam 1-shot | 85.07 / 95.19 / 77.44 |
| FedJam 2-shot | 86.21 / 95.61 / 77.56 |
| FedJam 4-shot | 88.11 / 96.17 / 70.72 |

### 与《现有方案介绍.md》中 B+ 主结果的参考对照

下面的 Δ 为 **ViT-L-14 − 文档 B+**；AUROC/AUPRC 的正值更好，FPR 的负值更好。
该对照用于判断趋势，文档中的 B+ 数值不是本次重新跑出的严格 paired baseline。

| 设置 | Δ AUROC / Δ AUPRC / Δ FPR |
|---|---:|
| In-house 1-shot | -1.81 / -8.85 / +4.70 |
| Public k=1 | +0.67 / +0.04 / +0.40 |
| Public k=2 | +0.26 / +1.27 / +0.20 |
| Public k=4 | +0.01 / +0.75 / -0.33 |
| OFDMA 1-shot | -7.78 / -7.12 / +9.33 |
| OFDMA 2-shot | -10.25 / -8.67 / +19.44 |
| OFDMA 4-shot | -9.89 / -8.17 / +26.00 |
| FedJam 1-shot | +1.36 / +0.57 / -1.06 |
| FedJam 2-shot | +1.73 / +0.85 / +4.34 |
| FedJam 4-shot | +3.31 / +1.42 / +1.89 |

## 结论

ViT-L-14 在 48 GB 显存环境下运行没有显存瓶颈，当前评测 batch 的峰值约 1.1
GiB。效果上，Public 基本持平略升，FedJam 的 AUROC/AUPRC 有提升；In-house
下降，OFDMA 明显下降，因此暂不建议直接将 ViT-L-14 设为正式默认 backbone。
OFDMA 的下降尤其需要后续做输入分辨率、V2V hook 层位和特征归一化的配对消融。

旧的 B+ `overall-best.pt` checkpoint 可以找到，但其中 gallery/prompt 等参数与
ViT-L-14 形状不匹配，本次日志显示 `loaded_keys=0`；ViT-L-14 的视觉主干按
`laion400m_e32` 加载，正常 support memory 在本次运行中重新构建。这不影响本次
“更换冻结 ViT 主干”的评估，但不应将该 checkpoint 解释为已迁移到 ViT-L 的训练
状态。

## 输出与复现

- 驱动脚本：`../run_ours_vitl14_experiment.sh`
- 结果根目录：`../analysis_outputs/exploratory/20260824_ours_vitl14_replace/`
- 各阶段的 `summary.json`、`metrics.csv`、`results.csv` 和运行日志均保留在结果根目录。
- 14 个阶段均已完成，FedJam 完整处理 7200 条测试样本。
