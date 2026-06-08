# P1: Frequency-Band-Aware Multi-Scale Scoring

## Status: NEGATIVE RESULT — 不推荐作为第二创新点

## Method Definition

频谱异常检测中的 anomaly map 具有明确的物理轴：
- **水平轴 (时间)**: 信号随时间变化的模式
- **垂直轴 (频率)**: 不同频率上的能量分布

传统 patch-level anomaly map 使用全图 max 聚合（通过 harmonic fusion），本方法尝试：
- 沿频率轴切分为多个 band
- 每个 band 内独立做 top-k / z-score 聚合
- 再跨 band 取 max

## 关键发现：Harmonic Fusion 已经很强

`metric_cal_img` 中的隐式 harmonic fusion 是本次分析的关键发现：

```python
img_scores = 1.0 / (1.0 / max_map_scores + 1.0 / img_scores)
```

即使 `cls_score_mode=text_only`，评估时 `max_map_scores`（pixel-level anomaly map 的 max）也通过 harmonic mean 混入了最终分数。

训练日志中的 Image-AUROC（91-94%）已经包含了 harmonic fusion 的 visual 信息。

## 为什么之前的 delta 被夸大

之前的分析比较了 `text_score + beta * band_score` vs raw `text_score`（无 harmonic fusion），产生了 ~+0.30 的虚假收益。

正确比较：band-aware 方法 vs harmonic fusion baseline（生产环境）。

## 为什么不是 selection

- 不依赖异常类型标签
- 统一使用 max over bands 聚合
- Normal-only calibration（仅用 normal 样本统计）

## Versions

### Version A: Frequency-Band Top-K Aggregation
- 沿频率轴切分 visual_anomaly_map 为 2/3/4 bands
- 每个 band 内计算 top-k mean
- max over bands → final = text_score + beta * band_score

### Version B: Normal-Calibrated Band Score
- 用训练 normal 样本计算每个 band 的 mean/std
- 测试时 z-score = (band_score - normal_mean) / normal_std
- max over calibrated bands → final = text_score + beta * calib_score

### Version C: Multi-Scale Band Pooling
- 对 visual_anomaly_map 做 1x/2x/4x max pooling
- 每个尺度再做 band aggregation + z-score calibration
- max over all scales and bands → final = text_score + beta * score

## Sanity Check Results (正确比较)

**Setup**: Playground_spectrum | m30db | Seed 111 | Epochs 50
**Config**: morph_fusion_gray_residual_a01 + rf + text_only

### Baseline 对比

| Dataset | raw text_only | harmonic fusion (PRODUCTION) |
|---------|---------------|------------------------------|
| burst   | 0.5474        | **0.9450**                   |
| chirp   | 0.7164        | **0.9109**                   |
| dsss    | 0.6292        | **0.9400**                   |
| Avg     | 0.6310        | **0.9320**                   |

> 训练日志中的 91-94% 是 harmonic fusion 的结果，不是 raw text。

### 方法 A：additive fusion (text + beta * band_score)

| Dataset | harmonic_baseline | best_band | delta |
|---------|-------------------|-----------|-------|
| burst   | 0.9450            | 0.9588    | +0.0138 |
| chirp   | 0.9109            | 0.8951    | **-0.0158** |
| dsss    | 0.9400            | 0.9514    | +0.0113 |
| **Avg** | **0.9320**        | **0.9351** | **+0.0031** |

### 方法 B：harmonic-fusion-compatible (harmonic fusion 兼容)

| Dataset | harmonic_baseline | best_band | delta |
|---------|-------------------|-----------|-------|
| burst   | 0.9450            | 0.9602    | +0.0153 |
| chirp   | 0.9109            | 0.9087    | **-0.0023** |
| dsss    | 0.9400            | 0.9516    | +0.0116 |
| **Avg** | **0.9320**        | **0.9402** | **+0.0082** |

> 方法 B 的最佳配置：`harmonic(text, b*zscore_max)` | 4bands | topk0.05 | b=0.02-0.10

### Chirp 详细分析

| Method | Chirp AUC | vs harmonic |
|--------|-----------|-------------|
| harmonic fusion (baseline) | 0.9109 | -- |
| Additive: best band-aware | 0.8951 | **-0.0158** |
| Harmonic-compatible: best band-aware | 0.9087 | **-0.0023** |
| harmonic(text+b*zscore, max_map) 2bands b=0.02 | 0.9087 | -0.0023 |

> **Chirp 在所有 band-aware 变体中都低于 harmonic fusion baseline。**
> 最优方法也只能将差距缩小到 -0.0023，本质上持平。

## Conclusion

### 方法不达标的理由

1. **平均提升接近零**：+0.003 ~ +0.008，远低于 +0.2 的成功标准
2. **Chirp 始终下降**：所有变体中 chirp 都 < harmonic fusion baseline
3. **Burst/dsss 提升微弱**：+1.1% ~ +1.5%，在单个场景/噪声下不具统计显著性
4. **Harmonic fusion 已经足够好**：`1/(1/text + 1/max_map)` 用全图 max pixel 几乎完美捕获了 visual evidence

### 为什么 harmonic fusion 不可超越

Harmonic fusion 的本质特性：
- 当 text_score 和 max_pixel_score 都很小时，harmonic 很小 → 正常样本
- 当任意一个 score 很大时，harmonic 趋向接近较小者 → 异常样本
- 频谱图中的异常通常在某个 patch 上有极高的 anomaly response → max_map 天然适合

Band-aware scoring 试图通过分频带聚合来"更好"地利用 visual map，但：
1. **Max over bands of topk = max patch per band**，而全图 harmonic fusion 已经在用 **全局 max patch**
2. 对于 burst（窄带异常），最大异常 patch 必然在对应频带 → max_map 已经捕获
3. 对于 chirp（扫频异常），最大异常 patch 可能在任意频带 → max_map 同样捕获
4. 对于 dsss（宽频异常），异常散布在所有频带 → max_map 和 band-aware 效果趋同

唯一的潜在优势（频带内 top-k mean > 单点 max）被 harmonic mean 的数学性质抵消：
harmonic mean 天然对 outlier（单个高响应 patch）足够敏感。

### 建议

**不推荐** Frequency-Band-Aware Multi-Scale Scoring 作为第二创新点。
现有的 harmonic fusion baseline 已经有效利用了 visual anomaly map，
band-aware aggregation 无法提供增量收益。

如果后续要继续探索 visual scoring 改进，建议方向：
- 不使用 harmonic fusion（即修改 metric_cal_img），从 text-only 重新起步
- 或者在 feature 层面做 band-aware（而非 anomaly map 后处理）
