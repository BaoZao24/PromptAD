# ViT+DINOv2 加权融合：跨数据集实验（2026-08-17）

## 实验目的

把 FedJam 上验证过的 ViT+DINOv2 方案扩展到其他频谱数据集，检查 DINO 局部特征和固定加权融合是否稳定。
本实验属于跨数据集探索性验证，不替换当前论文主线结论。

## 统一设置

- ViT 分支：PromptAD ViT-B/16-plus-240 的 `layer1+layer2`，50% farthest coreset，5-NN；RF 数据集使用最大 patch 分数，OFDMA 按现有协议使用第 3 大 patch 分数。
- DINO 分支：冻结的 DINOv2-B/14（`vit_base_patch14_dinov2`），1-NN，取最异常的 10% patch 距离均值。
- 只用正常 support 建立 memory。DINO support 使用四种视图：原图、时间方向左移、时间方向右移、频率响应漂移；待测图只用原图查询。
- In-house RF 按 60 个 signal/scene/JSR cells 评估；Public RF 使用固定的 15 个测试 cells，分别复核 `k=1/2/4-per-frequency`；OFDMA 使用完整 30 个测试场景，每个 shot 形成 30 个 cells，每个 cell 为 100 个正常观测和 100 个异常观测，并按 21 个 SU 的最大分数聚合。
- 两个分支都先映射到有界距离，再做固定加权平均：

  $$s_V=\operatorname{clip}\left(\frac{1-\cos(V_q,V_m)}{2},0,1\right),\qquad
  s_D=\operatorname{clip}\left(\frac{1-\cos(D_q,D_m)}{2},0,1\right),$$

  $$s_w=w s_V+(1-w)s_D,$$

  其中测试了 `w=0.25/0.50/0.75`，分别表示 ViT/DINO 为 `25/75、50/50、75/25`。
- 不训练、不使用异常样本建库、不使用测试批次统计量。

## AUROC 结果（宏平均）

| 数据集 / 协议 | ViT-only | DINO-only | ViT/DINO 25/75 | ViT/DINO 50/50 | ViT/DINO 75/25 |
|---|---:|---:|---:|---:|---:|
| In-house RF（60 cells） | 91.03 | 86.20 | 90.10 | 91.03 | **91.21** |
| Public RF，k=1（15 cells） | 80.00 | 79.58 | 81.37 | **81.50** | 80.88 |
| Public RF，k=2（15 cells） | 80.78 | 80.55 | **82.27** | 82.24 | 81.60 |
| Public RF，k=4（15 cells） | 81.88 | 81.06 | 83.11 | **83.17** | 82.62 |
| OFDMA，1/2/4-shot（90 cells） | 88.36 | 79.75 | 83.83 | 87.24 | **88.84** |

## 三项指标与最优观察

| 数据集 / 协议 | 代表配置 | AUROC | AUPRC | FPR@95%TPR |
|---|---|---:|---:|---:|
| In-house RF | 75/25 | **91.21** | **81.04** | **24.22** |
| Public RF，k=1 | 50/50 | **81.50** | 38.58 | **44.63** |
| Public RF，k=2 | 25/75 | **82.27** | 32.80 | 43.07 |
| Public RF，k=4 | 50/50 | **83.17** | 39.28 | **42.06** |
| OFDMA，宏平均 | 75/25 | **88.84** | **91.42** | **57.14** |

Public RF 的 AUPRC 在各个 k 下仍以 ViT-only 略高（k=1/2/4 分别为 43.92/44.11/44.38），说明 DINO 分支主要改善 AUROC 和 FPR，而不保证改善所有指标。

## OFDMA 分 shot 结果

| shot | 配置 | AUROC | AUPRC | FPR@95%TPR |
|---:|---|---:|---:|---:|
| 1 | ViT-only | 84.40 | 87.90 | 69.40 |
| 1 | ViT/DINO 75/25 | **85.28** | **88.66** | **67.37** |
| 2 | ViT-only | 88.79 | 91.38 | 60.30 |
| 2 | ViT/DINO 75/25 | **89.00** | **91.57** | **57.67** |
| 4 | ViT-only | 91.89 | 93.84 | 49.90 |
| 4 | ViT/DINO 75/25 | **92.25** | **94.04** | **46.40** |

## 结论

1. DINO-only 在三个 RF 数据集上都弱于或接近 ViT-only；它不适合作为单独主线。
2. DINO 与 ViT 融合后，AUROC 在 In-house RF、Public RF 的 k=1/2/4 以及 OFDMA 上分别比对应 ViT-only 提高 `+0.19/+1.49/+1.50/+1.29/+0.48` 个百分点（按表中最优固定权重）。
3. `75/25` 在 In-house RF 和 OFDMA 上最稳定；Public RF 更偏向 `50/50` 或 `25/75`，说明固定权重存在数据集依赖，暂时应作为探索性增强或代表性对比，不包装成跨数据集统一最优创新点。

## 结果文件

- [评估脚本](../tools/eval_vit_dino_weighted_rf.py)
- [In-house RF 原始结果](../autoresearch/vit-dino-rf-inhouse-260817-1618/)
- [Public RF k=1 原始结果](../autoresearch/vit-dino-rf-public-k1-260817-1623/)
- [Public RF k=2 原始结果](../autoresearch/vit-dino-rf-public-k2-260817-1630/)
- [Public RF k=4 原始结果](../autoresearch/vit-dino-rf-public-k4-260817-1640/)
- [OFDMA 全量原始结果](../autoresearch/vit-dino-ofdma-full-260817-1705/)
