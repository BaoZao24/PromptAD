# FedJam：DINOv2 局部分支加入频谱 TTA（2026-08-17）

## 实验目的

在已经完成的 ViT+DINOv2 分支替换实验上，只给 DINO 分支加入频谱 TTA，判断 DINO 的局部
patch memory 是否能从频谱专用的正常视图扩充中获益。

## 实验设置

- 数据集：FedJam，完整独立测试集 7200 张，正常 support 为 benign-only 嵌套 1/2/4-shot。
- ViT 分支保持无 TTA，使用原有 `layer1+layer2`、50% farthest coreset、5-NN。
- DINO 分支使用冻结的 DINOv2-B/14（`vit_base_patch14_dinov2`，224×224）。
- DINO 正常 memory 启用四视图：原图、时间方向双向 `±4 px`、频率响应漂移（strength=3）。
- 四个 DINO support 视图合并为一个 memory；测试图只使用原图查询一次。
- DINO 图像分数为 top 10% patch 距离均值；融合仍使用原有 support-only gate。
- 不训练、不使用异常样本建库、不使用测试集统计量。

## DINO-only 结果

与同脚本、同 support/test 划分的无 TTA 结果比较：

| shot | 无 TTA AUROC | DINO TTA AUROC | 变化 | 无 TTA AUPRC | DINO TTA AUPRC | 无 TTA FPR95 | DINO TTA FPR95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 86.18 | **87.82** | **+1.63** | 95.61 | **96.16** | 76.61 | **71.17** |
| 2 | 89.52 | **89.61** | +0.09 | **96.66** | 96.64 | 64.28 | **59.89** |
| 4 | **91.31** | 90.97 | -0.33 | **97.19** | 97.05 | 53.22 | **52.56** |

## ViT+DINO confidence fusion

| shot | 无 TTA AUROC | DINO TTA AUROC | 无 TTA AUPRC | DINO TTA AUPRC | 无 TTA FPR95 | DINO TTA FPR95 | TTA gate active rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 84.24 | 84.24 | 94.93 | 94.93 | 81.44 | 81.44 | 0.00% |
| 2 | 84.59 | **84.66** | 94.93 | **94.95** | 76.94 | **76.72** | 99.94% |
| 4 | **85.19** | 85.19 | **95.04** | 95.03 | **70.78** | 70.83 | 41.88% |

## 结论

1. DINO-only 在 1-shot 上明显受益，AUROC 提升 1.63 个百分点，FPR95 降低 5.44 个百分点；
   2-shot 基本持平，4-shot 略有回落。
2. TTA 对 DINO 的主要作用是扩大正常局部 patch 的覆盖范围，support 越少越容易体现；当
   support 增加到 4-shot 时，原始 DINO memory 已经足够丰富，额外视图会带来冗余竞争。
3. 当前 ViT+DINO gate 几乎没有获得同等幅度的提升，说明瓶颈仍在融合规则，而不是 DINO
   特征本身。1-shot gate 不激活；2/4-shot 的保守门控只对少数分数进行修正。
4. 这组结果支持“DINO 局部 memory + 频谱 TTA”作为探索方向，但暂时不替换论文主线，也不
   把 TTA 单独包装成核心创新。还需要在 RF/Public RF 等数据集上按相同协议复核其稳定性。

## 结果文件

- [实验脚本](../tools/eval_fedjam_vit_dino_fusion.py)
- [TTA 完整结果](../autoresearch/dino-tta-260817-1552-full/)
- [TTA 汇总](../autoresearch/dino-tta-260817-1552-full/summary.json)
- [TTA 指标表](../autoresearch/dino-tta-260817-1552-full/metrics.csv)
- [无 TTA 对照](../autoresearch/fedjam-vit-dino-260817/summary.json)
- [加权平均融合实验](exp_fedjam_vit_dino_weighted_fusion_20260817.md)
