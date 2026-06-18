# Improve 开发报告：RF 提示词与频谱结构增强方案

更新时间：2026-06-01

## 1. 当前结论

本阶段已经收敛出一套可直接写入论文主线的方法，不再是分散的探索记录。主推方案确定为：

```text
prompt_mode = rf
input_mode = morph_fusion_gray_residual_a01
morph_fusion_gray_residual_a01 = gray_contrast + weak_residual(alpha=0.1) + original_gray
```

研究对象暂定为三类频谱异常：

```text
burst_signal
chirp_signal
dsss_signal
```

## 1.1 本阶段贡献

本阶段最终沉淀下来的贡献可以概括为两点：

1. **RF 提示词适配**：把原始 PromptAD 面向通用工业图像的 prompt，改成面向射频频谱异常的 prompt，使文本原型不再描述“defect / damage”，而是描述“abnormal signal energy / unexpected interference / abnormal time-frequency structure”。
2. **统一频谱输入构造**：提出 `morph_fusion_gray_residual_a01`，用 `gray_contrast + weak_residual(alpha=0.1) + original_gray` 替代伪彩色 RGB 输入，在不依赖异常类型先验的前提下兼顾 `burst/chirp/dsss` 三类异常。

这两部分组合起来，构成了当前真正有效的完整方案，而不是单独某一个通道或某一个 prompt 小改动。

## 1.2 取得的主提升

相对真正 baseline `legacy + RGB`，当前方案在自测主实验上的结果为：

- `burst_signal`: `86.8375 -> 90.0708`，提升 `+3.2333`
- `chirp_signal`: `78.9358 -> 84.9758`，提升 `+6.0400`
- `dsss_signal`: `96.4792 -> 96.6483`，提升 `+0.1691`
- **三类平均**: `87.4175 -> 90.5650`，提升 `+3.1475`

在更新后的公开协议上，当前方案也保持了正收益：

- `burst` 均值：`80.52 -> 83.55`，提升 `+3.03`
- `chirp` 均值：`70.14 -> 73.82`，提升 `+3.68`
- `dsss` 均值：`84.14 -> 83.76`，下降 `-0.38`
- **九条总体均值**：`78.27 -> 80.37`，提升 `+2.11`

公开协议说明，这套方法不是只对自测数据有效；同时，`dsss` 的结果也提醒我们，它的跨协议泛化仍然是后续需要继续优化的问题。

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

`rf + morph_fusion_gray_residual_a01` 满足这个条件。

## 3. 方法设计
### 2.1 三个特征的公式表达

`original_gray`：

```text
I_gray = Gray(I_rgb)
```

`gray_contrast`：

```text
I_contrast = CLAHE(I_gray; clipLimit=2.0, tileGridSize=8×8)
```

`weak_residual(alpha=0.1)`：

```text
B(f) = median_t I_gray(f, t)
R(f, t) = |I_gray(f, t) - B(f)|
I_res(f, t) = Normalize(I_gray(f, t) + 0.1 * R(f, t))
```


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
input_mode = morph_fusion_gray_residual_a01
```

真正 baseline 使用原始 PromptAD 设置：

```text
prompt_mode = legacy
input_mode = rgb
```

注意：脚本日志中仍可能出现 `k-shot=1` 字样，这是历史日志字段；当前主实验解释以 `normal_75_25` 为准。

## 5. 主结果表

主结果表位置：

```text
analysis_outputs/universal_gray_residual_a01/method_comparison.csv
analysis_outputs/universal_gray_residual_a01/summary.md
experiments/baseline_redefinition/original_promptad_legacy_rgb_summary.csv
```

| 数据集 | 原始 PromptAD baseline (`legacy` + RGB) | 现有方案 (`rf` + `gray_residual_a01`) | 提升幅度 |
|---|---:|---:|---:|
| `burst_signal` | 86.8375 | 90.0708 | +3.2333 |
| `chirp_signal` | 78.9358 | 84.9758 | +6.0400 |
| `dsss_signal` | 96.4792 | 96.6483 | +0.1691 |
| **三类平均** | **87.4175** | **90.5650** | **+3.1475** |

## 6. 结果解读

`rf + morph_fusion_gray_residual_a01` 相比真正原始 PromptAD baseline 有明确提升：三类平均提升 `+3.1475`。这说明当前方法的主要价值，不是对单一异常做极限优化，而是给 PromptAD 提供了一套更符合频谱图判别方式的统一输入与统一文本语义。

这里的 baseline 指 `legacy prompt + RGB input`，不是之前文档中误称为 baseline 的 `rf prompt + RGB input`。后者已经包含 RF 手工提示词，属于中间消融设置，不再作为主结果表的对比对象。

因此当前主方案选择 `rf + gray_residual_a01`。它不是单独针对某一种异常类型优化，而是在 RF 场景提示词和统一输入变换下兼顾三类异常，更符合实际应用中未知异常类型的设置。

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

- Prompt 实验：RF 场景提示词是当前完整方案的一部分，用于解决原始 PromptAD 文本语义与频谱图不匹配的问题。后续继续增加复杂 prompt、分组异常原型或更多可学习 abnormal token 的实验没有稳定收益，因此不作为额外创新点。
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
