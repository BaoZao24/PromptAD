# Confidence-fusion complementarity experiment

日期：2026-07-27

## 目的

检查“confidence fusion 与 ViT-only 差距很小”是实现错误，还是因为 CNN
证据本身只在少数样本上提供增量信息。实验只读取已经保存的原始 ViT/CNN 分数，
没有修改正式融合代码、测试集划分或论文结果。

## 对照

- 当前 confidence correction：CNN 位于本单元分数上半段，且 CNN 归一化证据高于 ViT 时才加入。
- 只按 CNN 排名、不同排名阈值、不同 correction 强度（幂次）等固定变体。
- 指标：每个测试单元先计算 Image-AUROC、AP、FPR95，再做宏平均。

## 结果

| 协议 | ViT AUROC | 当前 confidence fusion AUROC | AUROC 差值 | 观察 |
|---|---:|---:|---:|---|
| self RF | 94.97 | 95.06 | +0.09 | AP +0.53，FPR95 -1.70 |
| public RF（15 个单元） | 89.26 | 90.11 | +0.84 | AP +0.90，FPR95 -4.88 |
| OFDMA 1-shot | 87.99 | 87.98 | -0.01 | CNN 近似中性 |
| OFDMA 2-shot | 92.00 | 91.97 | -0.03 | CNN 近似中性 |
| OFDMA 4-shot | 94.89 | 94.83 | -0.05 | CNN 近似中性 |

self RF 上 confidence correction 平均激活比例约 19.6%，平均 correction 值约 0.016，说明它本来就是
“少数样本校正”，不是固定权重的双分支平均。更激进的 rank-based correction 没有稳定
提高 AUROC，且会损害 AP 或 FPR95；因此没有把这些临时变体保留为正式接口。

## 结论

差距小不是计算错误。ViT 在这些协议上已经提供了主要排序能力，CNN 与它的独立
互补信息有限。当前方法的可辩护贡献是：在不牺牲 ViT 主体判断的情况下，减少边界
误报；public RF 上这一点表现为 FPR95 降低 4.88 个百分点。后续论文展示应同时
报告 AUROC、AP 和 FPR95，并按异常类型给出 confidence correction 的增量，而不是人为放大 CNN 权重。
