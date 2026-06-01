# Improve 开发报告：频谱结构增强方案

更新时间：2026-06-01

## 1. 当前结论

本阶段主推方案确定为：

```text
morph_fusion_gray_residual_a01 = gray_contrast + weak_residual(alpha=0.1) + original_gray
```

研究对象暂定为三类频谱异常：

```text
burst_signal
chirp_signal
dsss_signal
```

`wideband_pulse` 暂不纳入主研究对象。原因不是简单删除失败结果，而是实验和可视化都说明：`wideband_pulse` 尤其 `m40db` 属于极低对比度块状异常，当前基于频谱 PNG 的传统特征增强方法很难稳定分离暗弱区域。它会把主线结论拉向另一个问题：低对比度区域检测。因此本阶段聚焦在更符合当前方法优势的突发、线性调频和扩频纹理异常。

## 2. 为什么放弃 selection 作为主线

早期探索过“根据异常类型选择不同特征提取方法”的方案，但这个逻辑不适合作为最终方法。

真实生产环境中，模型在推理前通常不知道异常类型。如果先验地知道输入是 `chirp`、`dsss` 或 `burst`，再选择对应特征提取器，这相当于把标签信息提前给了模型。因此 selection 方案只保留为探索记录，不作为论文主方案。

最终主线必须满足：

```text
对所有纳入研究的异常类型使用同一个输入变换
不依赖异常类型先验
训练和测试协议一致
```

`morph_fusion_gray_residual_a01` 满足这个条件。

## 3. 方法设计

### 3.1 三个通道

| 通道 | 含义 | 作用 |
|---|---|---|
| `gray_contrast` | 对原灰度频谱图做 CLAHE 对比度增强 | 保留原始能量形态，同时提升弱结构可见度 |
| `weak_residual(alpha=0.1)` | 行背景中位数残差的轻量增强版本 | 抑制部分稳定背景，保留相对背景偏离 |
| `original_gray` | 原始灰度频谱图 | 保留原图能量分布，避免输入完全偏向增强特征 |

直观理解：

```text
gray_contrast 负责增强弱结构
weak_residual(alpha=0.1) 负责补充相对背景偏离
original_gray 负责保留原始频谱分布
```

### 3.2 为什么使用轻量 residual

`burst`、`chirp`、`dsss` 都不是靠颜色判断，而是靠频谱图中的强度变化和结构变化判断：

- `burst_signal`：短时突发，具有明显时间局部变化；
- `chirp_signal`：斜线轨迹明显，方向性强；
- `dsss_signal`：扩频纹理和频带内部结构明显。

实验中发现，单独引入方向纹理增强会提升 `burst` 和 `chirp`，但会使 `dsss` 略有下降。进一步实验表明，更稳的统一输入方式是保留原始灰度图，并加入轻量 residual。这样既能增强异常相对背景的偏离，又不会把输入过度改造成某一种特定结构特征。

这里 residual 权重设为 `alpha=0.1`，不是为了追求最强视觉增强，而是为了避免把正常背景纹理也放大。对于统一方案来说，稳定性比单个异常类型上的极限提升更重要。

## 4. 实验协议

主实验采用自测数据集三类异常：

```text
/mnt/data/wangbei/data/datasets/burst
/mnt/data/wangbei/data/datasets/chirp
/mnt/data/wangbei/data/datasets/dsss
```

场景：

```text
WeaponMuseum_spectrum
Playground_spectrum
TimeSquare_spectrum
Gymnasium_spectrum
```

训练/测试协议：

```text
normal_75_25
75% normal 样本用于训练
25% normal 样本用于测试
异常样本只用于测试
seed = 111
epochs = 50
prompt_mode = rf
```

注意：脚本日志中仍可能出现 `k-shot=1` 字样，这是历史日志字段；当前主实验解释以 `normal_75_25` 为准。

## 5. 主结果表

主结果表位置：

```text
analysis_outputs/universal_gray_residual_a01/method_comparison.csv
analysis_outputs/universal_gray_residual_a01/summary.md
```

| 数据集 | Baseline RGB | gabor_residual | gray_residual_a01 | gray_residual_a01 - baseline |
|---|---:|---:|---:|---:|
| `burst_signal` | 86.3117 | 90.0117 | 90.0708 | +3.7591 |
| `chirp_signal` | 78.7308 | 85.3342 | 84.9758 | +6.2450 |
| `dsss_signal` | 96.4883 | 95.3358 | 96.6483 | +0.1600 |
| **三类平均** | **87.1769** | **90.2272** | **90.5650** | **+3.3881** |

## 6. 结果解读

`gray_residual_a01` 相比 RGB baseline 有明确提升：三类平均提升 `+3.3881`。

与 `gabor_residual` 相比，`gray_residual_a01` 的三类平均从 `90.2272` 提升到 `90.5650`。主要变化是 `dsss_signal` 从 `95.3358` 提升到 `96.6483`，避免了 Gabor 方向纹理通道对 DSSS 的负影响；`burst_signal` 基本持平；`chirp_signal` 略有下降，但仍明显高于 RGB baseline。

因此当前主方案选择 `gray_residual_a01`。它不是单独针对某一种异常类型优化，而是在统一输入变换下兼顾三类异常，更符合实际应用中未知异常类型的设置。

## 7. wideband 为什么暂不纳入

`wideband_pulse` 的 m20/m30 在 Gabor 下表现很好，但 m40db 非常不稳定：

| 场景 | wideband m40db AUC |
|---|---:|
| TimeSquare | 100.00 |
| Playground | 79.42 |
| WeaponMuseum | 75.21 |
| Gymnasium | 63.62 |

可视化位置：

```text
analysis_outputs/wideband_m40_gabor_featuremaps
analysis_outputs/wideband_m40_residual_alpha_compare
analysis_outputs/wideband_m40_local_residual_compare
```

实验现象说明：暗弱块状区域在原图中已经接近背景，`weak_residual`、局部 residual、Gabor 都不能稳定把它提取出来。因此 wideband 更像是另一个低对比度块状异常检测问题，后续可以单独研究，但不放入当前主线。

## 8. Prompt / Adapter / LoRA / VAE 的当前定位

- Prompt 实验：已做过探索，但没有稳定收益。原因可能是 PromptAD/CLIP 已经具备一定文本泛化能力，简单增加描述词无法显著改变视觉特征提取瓶颈。
- Adapter / LoRA：属于参数高效微调路线，理论上可行，但目前不是主创新点。当前主线先收敛在输入特征构造。
- VAE/AE：适合低对比度异常和正常分布建模，尤其可能适合 wideband m40db。但这会变成“重建式异常检测 + PromptAD 融合”的新路线，建议作为后续扩展，不混入当前主实验。

## 9. 参考文献和它们的作用

1. Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization. Graphics Gems IV.  
   用途：支撑 CLAHE 作为局部对比度增强方法的依据。

2. Jain, A. K., & Farrokhnia, F. (1991). Unsupervised texture segmentation using Gabor filters. Pattern Recognition. DOI: https://doi.org/10.1016/0031-3203(91)90143-S  
   用途：作为 Gabor 探索实验的参考依据，不是当前主方案的核心方法。

3. Daugman, J. G. (1985). Uncertainty relation for resolution in space, spatial frequency, and orientation optimized by two-dimensional visual cortical filters. JOSA A. https://opg.optica.org/abstract.cfm?uri=josaa-2-7-1160  
   用途：作为方向纹理增强探索的参考依据。

4. Gao et al. CLIP-Adapter: Better Vision-Language Models with Feature Adapters. https://arxiv.org/abs/2110.04544  
   用途：作为后续 Adapter 路线参考，不是当前主方案。

5. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685  
   用途：作为后续参数高效微调路线参考，不是当前主方案。
