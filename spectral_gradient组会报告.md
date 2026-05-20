# 组会报告：Spectral Gradient 输入表示改进

> 说明：这里的 `spectral ingredient` 按项目实现应写作 `spectral_gradient`。它不是一个新模型，而是 PromptAD 输入表示层面的改进方法。

## 1. 报告题目

**Spectral Gradient：面向频谱异常检测的三通道输入表示改进**

建议开场说明：

当前 PromptAD 原本面向工业图像异常检测，输入通常是普通 RGB 图片。但我们的任务是射频频谱异常检测，频谱图的横轴是时间，纵轴是频率，亮度表示信号能量。直接把频谱图当普通 RGB 图片输入，会让模型更多关注颜色和纹理，而不是信号在时间、频率方向上的结构变化。因此我们设计了 `spectral_gradient` 输入表示，把一张频谱图转换成更符合时频信号特征的三通道输入。

配图建议：

- 图 1：放一张原始频谱图，标注横轴 time、纵轴 frequency、颜色/亮度表示 power。
- 图 2：放 PromptAD 原始流程图，强调本次改动发生在图像输入预处理阶段，不改变 CLIP 主干和 PromptLearner。

## 2. 问题背景：为什么不能简单当 RGB 图像处理

原始 PromptAD 的视觉编码器来自 CLIP，CLIP 习惯处理自然图像或普通 RGB 图片。但频谱图不是自然图像，它有明确物理含义：

```text
横轴：时间
纵轴：频率
像素强度：信号能量
```

不同干扰信号在时频图上表现为不同结构：

```text
burst：短时突发，边界明显
chirp：随时间扫频，表现为斜线轨迹
dsss：宽带扩频，能量分布更弥散
```

如果只使用 RGB 输入，模型看到的是“频谱图长什么样”；但异常检测真正需要的是“哪里发生了时间突变、哪里出现了频率边界、哪里有异常时频结构”。

配图建议：

- 图 3：放 burst、chirp、dsss 三类异常频谱示例。
- 在图上用箭头标注：
  - burst 的短时突发边缘；
  - chirp 的斜向扫频轨迹；
  - dsss 的宽带弥散区域。

讲解重点：

burst 和 chirp 的异常不只是颜色变化，而是时频结构变化。因此输入端如果能显式提供时间方向和频率方向的变化信息，就更容易让后续 CLIP patch feature 捕捉异常区域。

## 3. 方法核心：什么是 spectral_gradient

`spectral_gradient` 的核心思想是：

> 不再把频谱图简单当成 RGB 图片，而是构造“强度 + 时间梯度 + 频率梯度”的三通道输入。

具体来说，原始频谱图先转成灰度图 `I`，然后构造三个通道：

```text
通道 1：原始灰度强度 I
通道 2：时间方向梯度 |I(t) - I(t-1)|
通道 3：频率方向梯度 |I(f) - I(f-1)|
```

在代码里对应：

```python
gray = image.convert("L")
time_grad[:, :, 1:] = abs(gray[:, :, 1:] - gray[:, :, :-1])
freq_grad[:, 1:, :] = abs(gray[:, 1:, :] - gray[:, :-1, :])
input = concat(gray, time_grad, freq_grad)
```

三个通道分别表达：

| 通道 | 物理含义 | 对异常的帮助 |
|---|---|---|
| 灰度强度 | 当前时频点的能量强弱 | 保留原始信号能量分布 |
| 时间梯度 | 相邻时间帧之间的能量变化 | 强化 burst 的突然出现和消失 |
| 频率梯度 | 相邻频率 bin 之间的能量变化 | 强化频带边界和 chirp 轨迹 |

配图建议：

- 图 4：画一个三分图：
  - 左：原始灰度频谱图；
  - 中：时间方向梯度图；
  - 右：频率方向梯度图。
- 图标题可以写：`Spectral Gradient Input: Intensity + Temporal Gradient + Frequency Gradient`。

讲解重点：

这个方法不是新增一个复杂网络，而是在输入端把频谱图转换成更有物理意义的三通道表示。这样做的好处是改动很小，不破坏 PromptAD 主体结构，也不增加明显训练成本。

## 4. 与已有文献的关系

`spectral_gradient` 不是某篇论文中原封不动提出的专有方法，而是受音频和频谱图深度学习中的两类工作启发。

第一类是 spectrogram-based CNN。Salamon 和 Bello 在环境声音分类中指出，CNN 适合从 spectrogram-like inputs 中学习 discriminative spectro-temporal patterns。这说明把频谱图作为二维结构输入 CNN，并让模型学习时间-频率局部模式，是合理的。

第二类是 delta / frequency-delta 特征。Takahashi 等人在声学场景分类中使用 static mel-spectrogram 和 frequency-delta features，说明在频谱图上显式建模频率方向变化可以增强模型对声学场景的表征能力。

本项目把这两个思想迁移到 RF spectrogram 异常检测中：

```text
音频任务：log-mel spectrogram + delta / frequency-delta
本项目：RF spectrogram + temporal gradient + frequency gradient
```

配图建议：

- 图 5：插入 Salamon & Bello 论文中关于 spectrogram/CNN 输入或网络结构的图，用来说明“频谱图可以作为 CNN 输入学习时频模式”。
- 图 6：插入 Takahashi 等论文中关于 frequency-delta 特征或多宽度 frequency-delta 数据增强的图，用来说明“频谱差分特征能提供额外结构信息”。
- 旁边放我们自己的三通道示意图，说明本文不是照搬，而是做了 RF 任务适配。

可放在 PPT 页脚的引用：

```text
Salamon & Bello, Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification, IEEE SPL, 2017.
Takahashi et al., Acoustic Scene Classification Using CNN and Multiple-Width Frequency-Delta Data Augmentation, DCASE, 2016.
```

## 5. 在 PromptAD 中怎么接入

在项目中，`spectral_gradient` 是 `--input-mode` 的一个选项：

```bash
--input-mode spectral_gradient
```

它只改变输入图像的三通道构造方式，后面的 PromptAD 主体保持不变：

```text
频谱图
  -> spectral_gradient 三通道
  -> CLIP image encoder
  -> patch feature / global feature
  -> textual anomaly score + visual gallery score
```

代码位置：

```text
PromptAD/model.py
SpectrogramGradientChannels
```

当前还支持：

```text
rgb：原始 RGB 输入
spectral_gradient：强度 + 时间梯度 + 频率梯度
signal_adaptive：burst/chirp 用 spectral_gradient，dsss 用 rgb
```

配图建议：

- 图 7：画一张方法流程图：

```text
Original spectrogram
      |
      v
Intensity / Temporal Gradient / Frequency Gradient
      |
      v
CLIP Visual Encoder
      |
      v
PromptAD anomaly score
```

讲解重点：

这说明 `spectral_gradient` 是一个轻量、可插拔的输入增强模块。它不需要修改 PromptLearner，不需要重新设计 loss，也不需要训练额外模型。

## 6. 为什么 burst 和 chirp 更适合 spectral_gradient

burst 的特点是短时间突然出现，异常边界主要体现在时间方向。

```text
burst -> 时间方向突变明显 -> temporal gradient 有帮助
```

chirp 的特点是扫频轨迹，在时频图上表现为斜线。斜线本质上同时包含时间变化和频率变化。

```text
chirp -> 时频方向连续变化 -> temporal/frequency gradient 有帮助
```

因此 `spectral_gradient` 对这两类异常更匹配。它让模型不仅看到哪里能量高，还看到哪里能量变化快。

配图建议：

- 图 8：放 burst 原图和梯度图对比，标出时间边界被增强。
- 图 9：放 chirp 原图和梯度图对比，标出斜线轨迹被增强。

## 7. 为什么 DSSS 不一定适合 spectral_gradient

DSSS 是宽带扩频信号，异常通常表现为较宽频带内的弥散能量变化。它不像 burst 那样有很明显的短时边界，也不像 chirp 那样有清晰斜线轨迹。

因此，如果强行使用梯度输入，模型会更关注边缘和局部变化，反而可能破坏 DSSS 的整体宽带纹理。

项目实验也支持这一点。在 DSSS 单独实验中：

| grouping | rf + rgb/none | rf + spectral_gradient | delta |
|---|---:|---:|---:|
| m10db mean | 96.3875 | 93.3850 | -3.0025 |
| m20db mean | 90.4525 | 87.6050 | -2.8475 |
| m30db mean | 81.8475 | 80.0700 | -1.7775 |
| overall | 89.5625 | 87.0200 | -2.5425 |

结论是：

```text
spectral_gradient 对 DSSS 整体无效，平均下降 2.5425。
```

这也是为什么后续设计了 `signal_adaptive`：

```text
burst  -> spectral_gradient
chirp  -> spectral_gradient
dsss   -> rgb
```

配图建议：

- 图 10：放 DSSS 的原始频谱图和梯度图对比，说明梯度图可能削弱宽带纹理。
- 图 11：放 DSSS 实验 heatmap：
  - `analysis_outputs/rf_grad_vs_rf_rgb/delta_heatmap.png`
  - 或 `analysis_outputs/rf_grad_vs_rf_rgb/score_heatmaps.png`

## 8. 实验结果：spectral_gradient 的有效性

在早期 sanity 实验中，数据集为 `chirp_signal`，训练站点为 `WeaponMuseum_spectrum`，测试站点为 `Playground_spectrum`，JSR 为 `m30db`，k-shot 为 4，epoch 为 1。

| Prompt | Input | Image-AUROC |
|---|---|---:|
| legacy | rgb | 84.88 |
| rf | rgb | 84.96 |
| legacy | spectral_gradient | 85.59 |
| rf | spectral_gradient | 85.49 |

可以看到，在这个小规模 sanity 实验中，主要收益来自输入表示的变化，也就是 `spectral_gradient`。

在 36 组小批量实验中，整体结果如下：

| 方案 | Mean Image-AUROC |
|---|---:|
| legacy prompt + rgb | 85.0733 |
| legacy prompt + spectral_gradient | 85.8528 |
| rf prompt + rgb | 84.8683 |
| rf prompt + spectral_gradient | 85.8861 |
| rf prompt + signal_adaptive | 86.7336 |

按异常类型看：

| 方案 | burst | chirp | dsss |
|---|---:|---:|---:|
| legacy prompt + rgb | 86.8367 | 78.9425 | 89.4408 |
| legacy prompt + spectral_gradient | 87.6533 | 82.8317 | 87.0733 |
| rf prompt + rgb | 86.3117 | 78.7308 | 89.5625 |
| rf prompt + spectral_gradient | 87.9800 | 82.6583 | 87.0200 |
| rf prompt + signal_adaptive | 87.9800 | 82.6583 | 89.5625 |

关键结论：

```text
1. spectral_gradient 对 burst 和 chirp 有明显帮助；
2. spectral_gradient 对 dsss 不稳定，甚至下降；
3. signal_adaptive 综合最好，因为它对不同信号类型采用不同输入。
```

配图建议：

- 图 12：画柱状图，对比五种方案的 Mean Image-AUROC。
- 图 13：画分组柱状图，对比 burst / chirp / dsss 三类异常上的 AUROC。
- 如果组会时间有限，只放图 13，因为它最能解释为什么需要 signal_adaptive。

## 9. 方法优点和局限

优点：

- 改动小，只改输入预处理；
- 不增加模型参数；
- 不改变 PromptAD 主体训练流程；
- 对 burst 和 chirp 这类结构明显的异常有效；
- 和 RF prompt、visual gallery 等模块兼容。

局限：

- 对 DSSS 这类宽带弥散信号不适合；
- 梯度会放大边缘，也可能放大噪声；
- 目前只是简单一阶差分，没有引入更复杂的频谱统计；
- 对不同场景、不同 ISR 的稳定性还需要继续验证。

后续改进方向：

```text
burst / chirp：继续使用 spectral_gradient 或多尺度梯度；
dsss：尝试 log-power、局部方差、谱平坦度、VAE 重建分数；
整体策略：使用 signal_adaptive，让输入表示随信号类型变化。
```

## 10. 总结页

可以用三句话总结：

```text
第一，spectral_gradient 是一种频谱图三通道输入表示，把 RGB 改成强度、时间梯度和频率梯度。

第二，它的动机来自频谱图 CNN 和 delta / frequency-delta 特征，目的是让模型显式看到时频结构变化。

第三，它适合 burst 和 chirp，但不适合 DSSS，因此最终更推荐 signal_adaptive：burst/chirp 用 spectral_gradient，DSSS 用 RGB。
```

建议最后一页配图：

```text
RGB input
   vs
Spectral Gradient input
   vs
Signal Adaptive input
```

并在图下写：

```text
From generic image input to RF-aware signal representation.
```

## 参考文献

1. Justin Salamon and Juan Pablo Bello. **Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification**. IEEE Signal Processing Letters, 2017. https://arxiv.org/abs/1608.04363
2. Naoya Takahashi et al. **Acoustic Scene Classification Using Convolutional Neural Network and Multiple-Width Frequency-Delta Data Augmentation**. DCASE, 2016. https://arxiv.org/abs/1607.02383
3. PromptAD project implementation: `PromptAD/model.py::SpectrogramGradientChannels`.
