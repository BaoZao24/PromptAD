# 已尝试改进路线与结论汇总

更新时间：2026-06-02

本文档用于整理目前已经尝试过的改进方向，明确哪些方向已经被实验或逻辑约束排除，避免后续重复投入。

## 当前主线结论

目前稳定成立的正向方案是 RF 场景提示词和输入特征层面的统一增强：

```text
prompt_mode = rf
input_mode = morph_fusion_gray_residual_a01
morph_fusion_gray_residual_a01 = gray_contrast + weak_residual(alpha=0.1) + original_gray
```

主研究对象暂定为三类异常：

```text
burst_signal
chirp_signal
dsss_signal
```

该方案的核心优点是：

```text
不依赖异常类型先验
不需要异常训练样本
用 RF 场景提示词适配文本语义
直接改善频谱 PNG 的视觉输入表征
```

在自测三类异常上，相比真正原始 PromptAD baseline：

| 方法 | burst_signal | chirp_signal | dsss_signal | 三类平均 |
|---|---:|---:|---:|---:|
| 原始 PromptAD baseline (`legacy` + RGB) | 86.8375 | 78.9358 | 96.4792 | 87.4175 |
| 现有方案 (`rf` + `morph_fusion_gray_residual_a01`) | 90.0708 | 84.9758 | 96.6483 | 90.5650 |
| 提升 | +3.2333 | +6.0400 | +0.1691 | +3.1475 |

## 失败或不作为主线的方向

### 1. Prompt 手工改写

尝试内容：

- RF 通用 prompt；
- `rf_signal_structured` 结构化 prompt；
- 更具体的 burst/chirp/dsss 描述词；
- wideband 中强调矩形块状异常；
- object-agnostic 风格 prompt。

实验现象：

- 相比当前 `prompt_mode=rf`，收益基本为噪声级；
- 更具体的信号形态描述没有稳定提升；
- 部分 structured prompt 反而略降。

结论：

```text
不作为创新点。
```

原因：

CLIP / PromptAD 的文本编码器已经具备较强的 normal / abnormal 语义泛化能力。单纯增加异常描述词，无法显著改变视觉特征提取瓶颈。当前任务的主要限制不在“文本描述不够精确”，而在“频谱图视觉输入是否能突出异常结构”。

### 2. Learnable abnormal prompt token 数量调整

尝试内容：

- `n_ctx_ab=2`
- `n_ctx_ab=4`

实验现象：

- `n_ctx_ab=2` 三类平均约 `90.1794`；
- `n_ctx_ab=4` 结果不完整，已完成部分也没有稳定超过主方案；
- 没有证据说明更多 abnormal learnable token 能提升主结果。

结论：

```text
不继续作为主线。
```

原因：

`n_ctx_ab=1` 时，模型已经能学习到 normal / abnormal 的主要区分方向。继续增加 learnable abnormal token 容易带来冗余自由度，不一定提供新的有效语义。

### 3. Grouped abnormal prototype

尝试内容：

将 abnormal prototype 拆成多组：

```text
normal prototype
burst abnormal prototype
chirp abnormal prototype
dsss abnormal prototype
```

并尝试 `grouped_max` 等聚合方式。

实验结果：

| 方法 | burst_signal | chirp_signal | dsss_signal | 三类平均 |
|---|---:|---:|---:|---:|
| `single` | 90.0708 | 84.9758 | 96.6483 | 90.5650 |
| `grouped_max` | 89.9717 | 85.0708 | 96.4883 | 90.5103 |
| delta | -0.0991 | +0.0950 | -0.1600 | -0.0547 |

结论：

```text
机制可以作为探索记录，但不作为主线。
```

原因：

多异常原型理论上保留了异常形态差异，但实际推理时不能知道异常类型。`max` 聚合会引入非目标异常 prototype 的噪声：例如 chirp 样本仍会受到 burst / dsss prototype 的干扰。最终整体不如单一 abnormal prototype 稳定。

### 4. Selection 按异常类型选择特征

尝试内容：

早期尝试过根据异常类型选择不同输入特征，例如对 chirp、dsss、burst 使用不同增强方式。

实验现象：

- 某些单类/单场景可以提升；
- 但方法依赖“测试前知道异常类型”。

结论：

```text
逻辑上不成立，只保留为探索记录。
```

原因：

真实生产场景中，模型在检测前不知道异常属于 burst、chirp 还是 dsss。如果先知道异常类型再选择特征提取方法，相当于提前使用了标签信息。因此 selection 不能作为论文主方法。

### 5. Gabor / dualgrad / 方向纹理增强

尝试内容：

- Gabor 方向纹理；
- dualgrad 梯度方向图；
- grad magnitude；
- denoised grad；
- residual + gradient 组合。

实验现象：

- Gabor 对 burst、chirp 有一定提升；
- 但 DSSS 上会下降；
- dualgrad / denoised gradient 容易引入背景纹理噪声；
- denoised gradient 会把部分弱异常也抹掉；
- 梯度图整体偏脏，正常背景也容易被增强。

结论：

```text
不作为最终主方案，但 Gabor 可作为对比实验。
```

原因：

方向纹理增强适合斜线或边缘结构明显的异常，但 DSSS 更像扩频纹理/能量分布变化，过强方向滤波会改变其原始结构。最终 `gray_residual_a01` 更稳，因为它保留原始灰度，同时只做轻量残差补充。

### 6. Wideband pulse

尝试内容：

- RGB；
- spectral gradient；
- weak residual；
- Gabor；
- local residual；
- residual alpha 0.2 / 0.5；
- contrast enhancement；
- wideband-specific response。

实验现象：

- m20/m30 下部分方法有效；
- m40db 极不稳定；
- 暗弱块状异常在原图中接近背景，很多增强方法会整体抬高背景或引入不存在的干扰；
- 局部增强无法稳定突出低对比度矩形区域。

结论：

```text
wideband_pulse 暂不纳入当前主研究对象。
```

原因：

wideband m40db 更像低对比度块状目标检测问题，而不是当前三类频谱结构异常问题。把它混入主线会使问题定义变得不一致。

### 7. Score-level patch fusion / top-k fusion

尝试内容：

- `visual_topk`
- `visual_topk_max`
- `visual_topk_freq`
- `text + beta * patch_score`
- beta 从 `0.02` 到 `0.2`
- 36 jobs、17 种 scoring 变体。

核心结果：

| Variant | Avg iAUROC | vs text_only |
|---|---:|---:|
| `text_only` | 90.5631 | baseline |
| `topk_beta002` | 90.5581 | -0.005 |
| `max_beta010` | 90.5398 | -0.023 |
| `topk_beta020` | 90.4949 | -0.068 |
| `visual_topk_only` | 88.8877 | -1.675 |

结论：

```text
不作为第二创新点。
```

原因：

patch score 对局部异常有一定价值，但频谱异常不总是局部 patch 异常。尤其 chirp 是跨时间和频率的全局扫频模式，单个 patch 看起来可能像正常纹理。直接把 patch top-k 融入 image score 会拖累 chirp。

另外，`metric_cal_img` 中已经存在隐式 harmonic fusion：

```python
img_scores = 1.0 / (1.0 / max_map_scores + 1.0 / img_scores)
```

因此训练日志中的 Image-AUROC 已经混入了 max pixel anomaly map。再叠加 top-k patch score 信息高度重复。

### 8. Frequency-band-aware multi-scale scoring

尝试内容：

- 沿频率轴切分 2/3/4 个 band；
- 每个 band 内做 top-k / z-score；
- multi-scale pooling；
- 和 harmonic fusion baseline 做公平对比。

关键结果：

| Dataset | harmonic baseline | harmonic-compatible best | delta |
|---|---:|---:|---:|
| burst | 0.9450 | 0.9602 | +0.015 |
| chirp | 0.9109 | 0.9087 | -0.002 |
| dsss | 0.9400 | 0.9516 | +0.012 |
| Avg | 0.9320 | 0.9402 | +0.008 |

结论：

```text
不推荐作为第二创新点。
```

注意：

这组 best 是在测试变体中挑出的 oracle best，真实固定配置下收益可能更低。

原因：

band-aware 后处理没有稳定超过现有 harmonic fusion。它对 burst 和 dsss 有轻微帮助，但 chirp 仍然下降，不满足“不能牺牲 chirp”的约束。说明在 anomaly map 后处理层面做频带聚合，不能稳定提供增量。

### 9. Global normal distribution scoring

尝试内容：

- `normal_center`
- `normal_mahalanobis`
- `text_normal_center`
- `text_normal_mahalanobis`

实验现象：

- 单独使用 global normal distance 明显变差；
- 和 text score 融合后也没有超过 baseline；
- 部分场景略降。

结论：

```text
不作为主线。
```

原因：

global CLIP image feature 太粗，容易丢掉局部时频异常结构。频谱异常往往体现在局部或结构化区域，不能只靠全图 feature 到 normal center 的距离判断。

### 10. VAE reconstruction error

尝试内容：

- 使用已有 VAE 模型；
- 对测试图计算全图 reconstruction MSE；
- 与 PromptAD 分数融合。

实验现象：

| 异常类型 | VAE 单独 AUC | 现象 | 融合结果 |
|---|---:|---|---|
| burst | 0.38-0.55 | 方向反向 | 变差 |
| chirp | 0.47-0.50 | 方向反向 | 变差 |
| dsss | 0.31-0.86 | 部分正向 | 微弱且不稳定 |

结论：

```text
不推荐作为第二创新点。
```

原因：

全图 MSE 是像素级全局度量，容易被背景噪声主导。burst/chirp 的异常是较规则的结构化线条，VAE 反而可能更容易重建，导致异常图 MSE 低于正常图，方向反了。DSSS 有时有效，是因为它更像全局能量/纹理扰动，但整体仍不稳定。

### 11. Adapter / LoRA

尝试内容：

- 讨论过 Adapter 和 LoRA；
- 初步定位为参数高效微调路线。

结论：

```text
目前不作为主创新点。
```

原因：

Adapter / LoRA 需要训练视觉或文本侧参数，工程复杂度和过拟合风险更高。目前我们已经证明最有效收益来自输入频谱结构增强，因此不优先展开参数微调路线。

## 当前判断

已经排除的方向可以概括为：

```text
prompt 堆词无效
异常类型 selection 不成立
score 后处理无稳定增益
global normal modeling 太粗
VAE 全图重建方向不稳
方向滤波单独使用不够稳
```

当前最稳的论文主线是：

```text
频谱异常检测的关键瓶颈在视觉输入表征；
相比改 prompt 或后处理分数，结构感知灰度残差输入能更直接改善 CLIP/PromptAD 对频谱异常的感知。
```

## 仍可探索但尚未验证的方向

如果后续还需要第二创新点，优先考虑和当前成功路线一致的输入/特征层方法：

### A. Normal-only 频谱背景偏离输入通道

参考 PaDiM 的 normal patch distribution 思想，但不在 score 层做 Mahalanobis，而是构造输入通道：

```text
gray_contrast + weak_residual + normal_background_deviation
```

其中 `normal_background_deviation` 由训练 normal 样本估计每个频率位置或 patch 位置的正常均值/方差，再计算测试图偏离程度。

优点：

- 有 normal-only 文献依据；
- 不需要异常样本；
- 不依赖异常类型；
- 和当前输入增强路线一致。

风险：

- 如果场景 normal 背景本身变化大，normal deviation map 可能引入场景噪声；
- 需要严格保证只使用训练 normal 样本统计。

### B. Feature-level 多层 CLIP 融合

参考 AnomalyCLIP 的 DPAM 思想，在 CLIP 中间层保留更多局部细节。

优点：

- 仍是视觉表征层改进；
- 比 prompt 和 score 后处理更可能影响模型感知。

风险：

- 需要改 `encode_image`；
- 工程成本高于输入通道。

### C. Feature-level 多尺度窗口

参考 WinCLIP 的 multi-scale window 思想，但不要做后处理 top-k，而是在 feature 层形成多尺度表示。

优点：

- 有文献支撑；
- 理论上适合不同尺度的频谱异常。

风险：

- 容易退化成另一个 patch score fusion；
- 如果只在 anomaly map 后处理层实现，已经被 P1 实验证明收益不足。

## 后续原则

后续任何新方向都应满足：

```text
不使用异常类型先验
不使用测试异常样本调参
不只在单一异常类型上提升
不能明显牺牲 chirp
必须和 morph_fusion_gray_residual_a01 + rf + 当前评估方式比较
```

如果新方向只带来噪声级提升，或者依赖测试集 oracle best，就不应作为论文创新点。
