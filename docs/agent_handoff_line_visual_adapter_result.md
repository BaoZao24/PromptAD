# Line: Visual Adapter — 结果记录

跑期: 2026-06-27 → 2026-06-28（agent4 收尾），GPU 3。

## 配置
- method = `pooled_rf_rgb_visual_adapter`
- visual_adapter=True, bottleneck_ratio=0.25, alpha=0.2
- epochs=20, seed=111, batch-size=400
- output_root = `analysis_outputs/20260627_method_funnel`

## 结果（pooled baseline pooled_rf_rgb 同根目录对比）

### cls Image-AUROC (i_roc)
| 数据集 | baseline | visual_adapter | Δ |
|---|---|---|---|
| burst | 92.249 | 92.256 | +0.006 |
| chirp | 92.520 | 92.524 | +0.004 |
| dsss  | 81.754 | 81.781 | +0.027 |
| pulse | 78.830 | 78.836 | +0.006 |
| **OVERALL** | **86.338** | **86.349** | **+0.011** |

pulse m40 单档: 46.140 → 46.131（Δ=−0.009）。

### seg Pixel-AUROC (p_roc)
| 数据集 | baseline | visual_adapter | Δ |
|---|---|---|---|
| burst | 98.041 | 98.026 | −0.015 |
| chirp | 95.077 | 95.065 | −0.012 |
| dsss  | 72.641 | 72.603 | −0.038 |
| pulse | 91.579 | 91.562 | −0.017 |
| **OVERALL** | **89.334** | **89.314** | **−0.020** |

## 训练轨迹（说明 best ≈ baseline 的原因）

cls 训练 i_roc macro 每 epoch:
```
1 -> 86.35  <-- 最佳，被保留
2 -> 85.65
3 -> 79.81
4-8 ->  ~77.2
...
20 -> 77.80
```

Adapter 近 identity 初始化，所以 epoch 1 几乎等价 baseline；
后续 19 个 epoch 训练让 adapter 偏离 identity 反而损害 i_roc（−9 个点），
overall-best.pt 实质就保住了 epoch 1 的"近 baseline"权重。

## 判断（按 handoff 阈值）

- "dsss / pulse m40 有改善" → 没有（dsss +0.03，pulse m40 −0.01，都在噪声内）
- "overall 下降超过 1.0 停止" → 没有
- adapter 本身 = no-op（best 落在 epoch 1）

**结论: REJECT，不进 50 epoch 复验。** 该方向对 RF 频谱跨域无效。

## 工件
- `analysis_outputs/20260627_method_funnel/results_cls_pooled_rf_rgb_visual_adapter.csv`
- `analysis_outputs/20260627_method_funnel/results_seg_pooled_rf_rgb_visual_adapter.csv`
- `analysis_outputs/20260627_method_funnel/logs/visual_adapter_cls.log`
- `analysis_outputs/20260627_method_funnel/runs/pooled_rf_rgb_visual_adapter/`
