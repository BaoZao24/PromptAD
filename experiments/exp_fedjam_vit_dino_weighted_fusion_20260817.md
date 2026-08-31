# FedJam：ViT+DINOv2 加权平均融合（2026-08-17）

## 实验目的

在 DINOv2 局部分支已经加入频谱 TTA 的基础上，把原来的 support-only confidence gate 换成
固定加权平均，观察两个分支是否能更充分地互补。

## 实验设置

- FedJam 完整独立测试集 7200 张，benign-only 嵌套 1/2/4-shot。
- ViT：PromptAD `layer1+layer2`、50% farthest coreset、5-NN；不加入 TTA。
- DINO：冻结 DINOv2-B/14，正常 support 使用原图、时间方向 `±4 px` 和频率响应漂移构建
  合并 memory；测试图只使用原图查询。
- 加权前的尺度对齐不读取测试集统计量：ViT 分数为 `(1-cosine)/2`，本身在 `[0,1]`；
  DINO 分数为 `1-cosine`，除以 2 后也映射到 `[0,1]`。
- 融合公式：

  $$s_w=w,s_V^{\mathrm{norm}}+(1-w)\,s_D^{\mathrm{norm}}.$$

- 测试 ViT 权重 `w=0.25/0.50/0.75`，对应 DINO 权重 `0.75/0.50/0.25`。

## 结果

| shot | 融合权重（ViT/DINO） | AUROC ↑ | AUPRC ↑ | FPR@95%TPR ↓ |
|---:|---:|---:|---:|---:|
| 1 | 0.25 / 0.75 | **88.28** | **96.35** | 71.44 |
| 1 | 0.50 / 0.50 | 87.73 | 96.19 | **73.28** |
| 1 | 0.75 / 0.25 | 86.39 | 95.74 | 77.83 |
| 2 | 0.25 / 0.75 | **89.58** | **96.67** | 62.50 |
| 2 | 0.50 / 0.50 | 88.50 | 96.35 | 66.22 |
| 2 | 0.75 / 0.25 | 86.82 | 95.79 | **71.61** |
| 4 | 0.25 / 0.75 | **90.63** | **96.96** | 55.94 |
| 4 | 0.50 / 0.50 | 89.28 | 96.54 | 60.33 |
| 4 | 0.75 / 0.25 | 87.44 | 95.92 | **66.39** |

## 与其他融合方式比较

| shot | ViT-only AUROC | 原 confidence gate AUROC | 加权平均（0.25/0.75）AUROC | DINO-only AUROC |
|---:|---:|---:|---:|---:|
| 1 | 84.24 | 84.24 | **88.28** | 87.82 |
| 2 | 84.49 | 84.66 | **89.58** | **89.61** |
| 4 | 85.05 | 85.19 | 90.63 | **90.97** |

## 结论

1. 加权平均明显优于当前 ViT+DINO 的保守门控：AUROC 在 1/2/4-shot 分别提高约 4.03、
   4.92、5.44 个百分点。
2. DINO 权重较高时效果更好，`ViT=0.25、DINO=0.75` 是本次三个固定比例中最稳定的选择。
3. 但加权平均仍没有在所有 shot 上超过 DINO-only；它更像是一个比门控更有效的双分支融合
   基线，而不是新的主线创新。
4. 第一版尝试用 support median/IQR sigmoid 证据做加权，在 1-shot 下因只有一个 support
   参考导致尺度退化，三个权重都接近随机排序；该版本已判为无效，不采用。最终结果使用
   已知余弦距离范围归一化，避免读取测试批次统计量。

## 结果文件

- [评估脚本](../tools/eval_fedjam_vit_dino_fusion.py)
- [最终完整结果](../autoresearch/dino-weighted-bounded-260817-1552-full/)
- [汇总结果](../autoresearch/dino-weighted-bounded-260817-1552-full/summary.json)
- [指标表](../autoresearch/dino-weighted-bounded-260817-1552-full/metrics.csv)
- [被弃用的 support-IQR 版本](../autoresearch/dino-weighted-260817-1552-full/summary.json)
