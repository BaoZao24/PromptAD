# 频谱异常检测特征提取 Selection 方法报告

> 说明：这份文档现在保留为历史探索记录，用来说明按异常类型做特征选择时曾经得到过哪些结论。当前后续主线不再把 selection 作为最终方法，而是优先考虑不依赖异常类型先验的通用融合输入，例如 `morph_fusion_dualgrad`。

## 1. 研究目标

本项目的核心问题不是简单地提高某一个数据集上的分数，而是为频谱异常检测设计一套可以解释、可以复现的特征提取选择方法。

PromptAD 原始流程把频谱图当作普通 RGB 图像输入 CLIP 视觉编码器。这个做法可以作为 baseline，但它没有显式利用频谱图的时频结构。不同异常类型在频谱图中的形态差异很明显：

| 异常类型 | 主要频谱形态 | 对特征提取的要求 |
|---|---|---|
| Burst | 短时突发、局部能量增强，边界较明显 | 强化时间方向和频率方向的局部变化 |
| Chirp | 随时间变化的斜线或扫频轨迹 | 强化斜率、边缘和方向性结构 |
| DSSS | 宽带、弱能量、铺展式扰动 | 在保留原始能量图的同时突出弱残差 |
| Wideband pulse | 宽带块状或片状能量异常 | 保留整体能量块结构，避免过度边缘化 |

因此，本报告提出的 selection 不是按场景硬编码，而是按异常信号的频谱形态选择输入特征表示。

## 2. Selection 规则

当前最终采用的是单层规则：按异常形态选择特征提取方法。

### 2.1 常规规则

| 数据集 / 异常类型 | 选择的特征提取方法 | 选择理由 |
|---|---|---|
| `burst_signal` | `optimized spectral gradient` | burst 的关键是短时突发边缘，在平滑、Sobel 梯度和稳健归一化后更稳定 |
| `chirp_signal` | `optimized spectral gradient` | chirp 在自测常规难度下仍具有可见斜向轨迹，对斜向结构增强更稳定 |
| `dsss_signal` | `dsss_weak_residual` | DSSS 异常弱且铺展，弱残差能突出异常能量，同时不过度破坏原图 |
| `wideband_pulse` | `rgb` | wideband 的块状能量结构已经明显，梯度会在低信噪比下破坏整体区域信息 |

对应代码中的自动选择版本可以理解为常规难度下的优化版 signal-adaptive：

```text
burst_signal   -> optimized spectral gradient
chirp_signal   -> optimized spectral gradient
dsss_signal    -> dsss_weak_residual
wideband_pulse -> rgb
```

## 3. 各特征提取方法的含义

### 3.1 RGB baseline

RGB baseline 是原始 PromptAD 的输入方式：将频谱图转成 3 通道 RGB，然后送入 CLIP 视觉编码器。

它的优点是稳定、简单，不会人为改变图像结构。缺点是没有显式告诉模型哪些方向的变化更重要，因此对 burst、chirp 这类结构性异常不够敏感。

### 3.2 Spectral Gradient

`optimized spectral gradient` 将输入三通道改为：

| 通道 | 含义 |
|---|---|
| Channel 1 | 原始灰度频谱图 |
| Channel 2 | 时间方向梯度 |
| Channel 3 | 频率方向梯度 |

与早期直接做相邻差分的 `spectral_gradient` 相比，优化后的版本加入了三步预处理：

1. **高斯平滑**：先对灰度频谱做轻微平滑；
2. **Sobel 梯度**：再用 Sobel 算子提取时间方向和频率方向梯度；
3. **稳健归一化**：最后不用最大值归一化，而是使用高分位数尺度做裁剪归一化。

可以概括为：

```text
gray -> gaussian smoothing -> Sobel(time/freq) -> robust normalization
```

这样做的原因分别是：

- **先做高斯平滑**：频谱图里常有细碎噪声、插值纹理和孤立亮点。如果直接做差分，这些高频扰动会被误当成异常边缘。先平滑可以压低这类伪结构，让后续梯度更集中在真正连续的时频变化上。
- **再做 Sobel**：Sobel 不是只看两个相邻像素的差，而是在局部邻域内估计梯度，响应更平滑，对 burst 的局部突变边界和 chirp 的连续轨迹都更稳定。
- **最后做稳健归一化**：旧版使用最大值归一化，容易被极少数异常尖点或噪声点带偏，导致整张图的梯度尺度失真。改成高分位数归一化后，梯度通道对极端值不那么敏感，跨场景和跨档位更稳。

因此，优化后的 `spectral_gradient` 本质上仍然是在突出时频结构，但它不再粗糙地放大所有局部变化，而是尽量保留真正与异常形态相关的边缘、斜率和局部突变结构。这也是为什么它比旧版更适合 burst 和 chirp。

### 3.3 DSSS Weak Residual

`dsss_weak_residual` 针对 DSSS 设计。它先估计每个频率位置的背景，再计算残差：

```text
freq_background = median(gray, time_axis)
residual = |gray - freq_background|
weak_residual = normalize(gray + alpha * residual), alpha = 0.2
```

最终三通道为：

| 通道 | 含义 |
|---|---|
| Channel 1 | 原始灰度频谱图 |
| Channel 2 | 原始灰度频谱图 |
| Channel 3 | 原图 + 弱残差增强 |

这样做的原因是 DSSS 的异常不是清晰边缘，而是宽带、弱能量、铺展式扰动。如果像 `spectral_gradient` 那样只强调边界，容易丢掉 DSSS 的整体能量分布；如果残差增强太强，又可能放大噪声。因此采用 `alpha=0.2` 的弱残差，在保留原图主体结构的同时轻微突出异常。

### 3.4 Wideband Pulse 保留 RGB

wideband pulse 的异常通常是宽带块状或片状区域。实验发现，`spectral_gradient` 对它不稳定，尤其在 m40db 低信噪比下下降明显。这说明 wideband 的关键信息不是边缘本身，而是整体能量块的存在和位置。梯度会把块状区域拆成边界，反而破坏了模型需要的区域信息。

因此，wideband pulse 当前不使用额外特征增强，保留 RGB baseline。

## 4. Selection 的依据

本方法的选择依据分为三层。

第一层是频谱形态先验。不同异常在频谱图上的可见结构不同，不能默认所有异常都适合同一种特征提取方式。burst/chirp 更像局部结构变化，DSSS 更像弱残差铺展，wideband pulse 更像整体能量块。

第二层是候选方法与形态的匹配关系。`optimized spectral gradient` 匹配边缘、斜率、局部突变，并通过平滑、Sobel 与稳健归一化减少噪声放大；`dsss_weak_residual` 匹配弱能量残差；RGB 匹配本身已经足够明显的块状能量区域。

第三层才是实验验证。最终 selection 不是只靠直觉，而是用 baseline 对比实验筛掉无效方法：

在自测数据上，`optimized spectral gradient` 已进一步替代旧 `spectral_gradient` 作为 burst/chirp 的默认版本。对比表见 `analysis_outputs/optimized_spectral_gradient_selftest_compare.csv`，其中：burst 从 `86.3117` 提升到 `89.4200`，chirp 从 `78.7308` 提升到 `85.2983`。

| 异常类型 | RGB baseline mean | selected mean | delta |
|---|---:|---:|---:|
| Burst | 86.3117 | 89.4200 | +3.1083 |
| Chirp | 78.7308 | 85.2983 | +6.5675 |
| DSSS | 89.5625 | 94.5833 | +5.0208 |
| Wideband pulse | 96.1333 | 96.1333 | +0.0000 |
| ALL | 87.6846 | 91.3587 | +3.6741 |

其中 wideband pulse 没有强行加入 `spectral_gradient`，因为它的对比实验显示整体下降：

| group | RGB baseline | `spectral_gradient` | delta |
|---|---:|---:|---:|
| m20db mean | 99.1225 | 97.2800 | -1.8425 |
| m30db mean | 98.2400 | 99.3050 | +1.0650 |
| m40db mean | 91.0375 | 73.6700 | -17.3675 |
| overall | 96.1333 | 90.0850 | -6.0483 |

这也是 selection 方法的重要原则：不是所有“更复杂”的输入都更好。如果某种增强破坏了异常的主要结构，就应该保留 baseline。

## 5. 与 Adapter / LoRA 的关系

Adapter 和 LoRA 属于视觉编码器参数适配方法，目标是让 CLIP backbone 通过少量可训练参数适应频谱图。它们和本报告的 selection 方法不是同一个层面的改进：

| 方法 | 改动位置 | 当前结论 |
|---|---|---|
| Feature selection | 输入特征表示 | 当前主创新点，改动小，可解释，已有稳定收益 |
| Adapter | 视觉特征层后接轻量残差模块 | 已尝试，收益不明显，不作为主线 |
| LoRA | 对视觉编码器部分线性层做低秩微调 | 已尝试，整体不稳定，不作为主线 |

因此，当前论文叙述建议把重点放在“基于频谱形态的输入特征 selection”，而不是强调 Adapter 或 LoRA。

## 6. 公开数据 / RF_SPE_PNG 结果说明

当前公开数据统一按 `RF_SPE_PNG + RF_Spectrum_Public_Dataset` 协议整理。正常数据来自：

```text
/mnt/data/wangbei/data/RF_SPE_PNG/RF_Spectrum_Public_Dataset/MeasRes_*
```

异常目录位于：

```text
/mnt/data/wangbei/data/RF_SPE_PNG/{class}/abnormal/{noise}
```

对应总表如下：

| 异常类型 | baseline mean | improved mean |
|---|---:|---:|
| Burst | 92.8500 | 93.5250 |
| Chirp | 99.7225 | 99.7400 |
| DSSS | 86.8850 | 87.3400 |
| Wideband pulse | 96.9467 | 96.9467 |
| ALL_AVAILABLE | 94.1011 | 94.3879 |

需要注意：公开数据上的 burst/chirp/dsss 与 wideband pulse 在难度和协议上并不完全一致，因此这里更适合作为补充验证和汇总结果，不适合作为主创新点的唯一证据。论文主线仍建议以自测四场景数据上的 feature selection 结果为主。

另外，`m50db` 极低功率实验结果仍保留在 `analysis_outputs/rf_spe_png_new_levels/`，但当前将其视为探索性补充结果，不纳入主表均值与主结论。

在补充公开数据实验中，`RF_SPE_PNG` 的新增档位统一使用 `RF_Spectrum_Public_Dataset` 中的正常记录目录：

```text
train normal: first k_shot MeasRes_* folders under RF_Spectrum_Public_Dataset
test normal:  remaining MeasRes_* folders under RF_Spectrum_Public_Dataset
abnormal:     /mnt/data/wangbei/data/RF_SPE_PNG/{class}/abnormal/{noise}
```

新增 `m40db` 结果如下：

| class | noise | RGB baseline | selected feature | selected | delta |
|---|---|---:|---|---:|---:|
| burst | m40db | 81.9700 | `optimized spectral gradient` | 84.3100 | +2.3400 |
| chirp | m40db | 98.9000 | `optimized spectral gradient` | 98.9700 | +0.0700 |
| dsss | m40db | 67.7000 | `dsss_weak_residual` | 69.2100 | +1.5100 |
| wideband pulse | m40db | 94.0400 | `rgb` | 94.0400 | +0.0000 |

这组结果说明两点。第一，降低 ISR 到 `m40db` 已经能把公开数据从“偏简单”变成更有区分度的验证集。第二，自测数据上得到的 selection 规则在公开数据 `m40db` 条件下仍可保留小幅正收益，说明这套按形态选择输入表示的方法具备一定跨协议适应性。

## 7. 生产场景下如何选择

实际生产中通常不知道具体场景特点，也可能不知道异常类型。因此 selection 不能写成“测试时先知道异常是 burst/chirp/DSSS，再选择特征”。那样会使用标签先验，不适合作为开放异常检测流程。

更严谨的使用方式分为两种。

第一种是已知威胁模型。如果部署任务本身已经明确要检测某一类干扰，例如只检测 DSSS 或只检测 chirp，那么可以直接按任务类型选择输入。

| 已知检测目标 | 推荐输入 |
|---|---|
| burst | `optimized spectral gradient` |
| chirp | `optimized spectral gradient` |
| dsss | `dsss_weak_residual` |
| wideband pulse | `rgb` |

第二种是未知异常类型。此时不应该先判断异常类型，而应把 selection 扩展为多视角异常评分：

```text
same test spectrogram
-> RGB view
-> spectral_gradient view
-> dsss_weak_residual view
-> anomaly scores from each view
-> fixed fusion or validation-calibrated selection
```

可用的融合规则包括：

| 规则 | 含义 |
|---|---|
| max score | 任一视角给出高异常分数，就认为可疑 |
| average score | 多视角平均，降低单一视角误报 |
| validation-weighted score | 在验证集上确定 RGB/gradient/residual 的固定权重 |
| normal-calibrated score | 用训练正常样本的分数分布校准不同视角的尺度 |

因此，当前论文主实验可以表述为“按异常类型划分的数据集级 feature selection”；面向真实开放部署时，更合理的扩展方向是“morphology-aware multi-view anomaly scoring”。

## 8. 结论

本项目最终采用的 selection 方法可以概括为：

```text
根据频谱异常的形态选择输入特征表示。
```

它的优势是：

- 改动小，只改变输入特征构造，不需要大规模重训 backbone；
- 可解释，每个选择都能对应到具体频谱形态；
- 有实验支撑，自测四场景总体从 87.6846 提升到 91.3587；
- 不盲目增强，对 wideband pulse 这种梯度会破坏结构的异常，明确保留 RGB。

因此，这个方法可以直接表述为一套基于频谱形态的 feature selection 策略，并作为当前工作的主要创新点来组织论文叙述。
