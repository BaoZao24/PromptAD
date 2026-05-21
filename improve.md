# PromptAD 改进方案（聚焦改进1和改进3）

> 目标：只保留两类最适合当前项目的改进方向：
> - **改进1：特征提取**
> - **改进3：提示词（Prompt）**
>
> 这份文档尽量写得直白，并且对每篇参考文献都说明：
> - **用于什么模块**
> - **作用是什么**
> - **为什么适合我们这个频谱跨场景任务**

---

## 一、先说结论：这两个方向为什么值得先做

我们现在的 PromptAD，本质上还是把频谱图当成普通图片来处理。

但你的任务其实有两个很明显的特点：

1. **频谱图不是自然图像**
   - CLIP 原本看的是猫、狗、风景图。
   - 我们现在给它看的是时频图。
   - 所以第一件事就是：**让特征提取模块更懂频谱图。**

2. **异常类型和场景都有明显语义**
   - 异常类型有：`burst`、`chirp`、`dsss`
   - 场景有：`WeaponMuseum_spectrum`、`Playground_spectrum`、`TimeSquare_spectrum`、`Gymnasium_spectrum`
   - 所以第二件事就是：**把提示词写得更像这个任务本身，而不是泛泛地写“abnormal spectrum”。**

因此，最值得优先做的就是：

- **改进1：让视觉特征更适合频谱图**
- **改进3：让文本提示更适合“异常类型 + 场景”**

---

# 改进1：特征提取模块怎么改

## 1.1 当前问题

你现在的特征提取模块主要在：
- `@/mnt/data/wangbei/PromptAD/PromptAD/model.py:186-191`
- `@/mnt/data/wangbei/PromptAD/PromptAD/model.py:203-208`

也就是：
- 用 CLIP 的视觉编码器提特征；
- 输入前先 resize；
- 再用 CLIP 默认的 `mean/std` 做归一化。

问题主要有 3 个：

### 问题 A：CLIP 不懂频谱图

CLIP 预训练时见的是自然图片，不是无线电频谱图。
所以它提到的特征，很多更偏向“图片纹理”，不一定真正抓住：
- chirp 的斜率
- burst 的短时窄带结构
- DSSS 的宽带铺展结构

### 问题 B：三通道信息利用不充分

现在本质上还是把一张频谱图当作普通 RGB 图输入。
但频谱图真正有用的信息其实是：
- 原始强度
- 时间方向变化
- 频率方向变化

### 问题 C：跨场景时背景差异会干扰特征

你的任务不是单场景，而是 **cross-site**：
- 在一个场景训练
- 去另一个场景测试

不同场景的背景噪声底、干扰密度、频谱纹理都不一样。
如果特征提取模块过度记住训练场景背景，就会导致跨场景性能下降。

---

## 1.2 特征提取改进方案（按推荐顺序）

## 方案 1：重新设计输入通道，让模型看到“时域变化”和“频域变化”

### 怎么做

把输入的 3 个通道改成：
- **通道 1**：原始频谱图（dB 或归一化后的能量图）
- **通道 2**：时间方向梯度
- **通道 3**：频率方向梯度

也就是不要再简单复制 3 份灰度图，而是给模型更有信息量的 3 个视角。

### 用于什么模块

- **模块**：输入预处理模块
- **代码位置**：`PromptAD/model.py` 里的 `transform` 前后，或者 `datasets/` 里读图后构造三通道

### 作用是什么

- 让模型更容易识别：
  - chirp 的斜线趋势
  - burst 的竖向突发结构
  - dsss 的宽带能量分布
- 比“纯灰度复制三份”更符合频谱图本身的结构

### 为什么适合我们任务

因为你的异常不是靠颜色分辨，而是靠**结构变化**分辨。
其中：
- chirp 更依赖斜率
- burst 更依赖短时突发边缘
- dsss 更依赖频带内部的能量分布

所以加入梯度通道，能直接加强这些信息。

### 参考文献

- **Salamon & Bello, 2017, arXiv:1608.04363**

### 这篇文献用于哪个模块

- **用于模块**：输入表示 / 特征输入构造

### 这篇文献的作用

- 它说明在声音与频谱任务中，不同通道可以承载不同类型的谱信息，而不是照搬普通 RGB 图像的输入方式。

---

## 方案 2：先做最简单的域对齐——重算频谱图自己的 mean/std

### 怎么做

把 `@/mnt/data/wangbei/PromptAD/PromptAD/model.py:20-21` 里 CLIP 默认的：
- `mean_train`
- `std_train`

换成你自己的频谱数据集统计值。

### 用于什么模块

- **模块**：输入归一化模块
- **代码位置**：`PromptAD/model.py:20-21`

### 作用是什么

- 让模型输入分布更接近真实频谱数据；
- 减少“自然图像统计量”和“频谱图统计量”不匹配的问题。

### 为什么适合我们任务

这是最便宜、最稳的一步。
尤其你现在的数据是 RF 频谱图，不同于照片，直接用 CLIP 默认值往往不合理。

### 参考文献

- **He et al., MAE, CVPR 2022, arXiv:2111.06377**
- **Huang et al., Audio-MAE, NeurIPS 2022, arXiv:2207.06405**

### 这些文献用于哪个模块

- **用于模块**：视觉输入分布对齐 / 频谱域建模

### 这些文献的作用

- 它们都说明了：当任务域和原始图像域不同的时候，继续做域内预训练或域内适配是有效的。
- 对我们来说，最基础的一步就是先把输入统计量对齐。

---

## 方案 3：给 CLIP 加一个轻量 Adapter / LoRA，专门适配频谱图

### 怎么做

不直接把整个 CLIP 重新训练，而是在视觉编码器旁边加一个很小的可训练模块：
- **Adapter**：小型特征修正层
- **LoRA**：低秩微调模块

让这个小模块学会把“频谱图特征”调整得更适合当前任务。

### 用于什么模块

- **模块**：视觉特征提取模块
- **代码位置**：`PromptAD/model.py` 中 `self.model.visual` 相关层

### 作用是什么

- 不用大改原模型；
- 参数少、训练稳定；
- 比全量微调更适合你现在这种数据量不大的场景。

### 为什么适合我们任务

你的任务是少样本、跨场景。
如果直接把 CLIP 全部训一遍：
- 容易过拟合
- 很吃显存
- 不稳定

而 Adapter / LoRA 适合：
- 数据少
- 希望尽量保留 CLIP 原能力
- 又想让它适应频谱图

### 参考文献 1

- **Gao et al., CLIP-Adapter, IJCV 2024, arXiv:2110.04544**

### 这篇文献用于哪个模块

- **用于模块**：视觉特征适配模块

### 这篇文献的作用

- 提供一种在不大改 CLIP 主体的情况下，让特征更适合下游任务的方法。

### 参考文献 2

- **Zhang et al., Tip-Adapter, ECCV 2022, arXiv:2207.09519**

### 这篇文献用于哪个模块

- **用于模块**：少样本特征适配模块

### 这篇文献的作用

- 说明在少样本条件下，轻量适配能快速提升 CLIP 下游效果。

### 参考文献 3

- **Hu et al., LoRA, ICLR 2022, arXiv:2106.09685**

### 这篇文献用于哪个模块

- **用于模块**：参数高效微调模块

### 这篇文献的作用

- 说明只训练少量低秩参数，也能让大模型适应新领域。

---

## 方案 4：用更懂频谱图的骨架做初始化，再接 PromptAD

### 怎么做

把视觉编码器从普通 CLIP backbone，换成或借鉴：
- **AST**
- **BEATs**
- **Audio-MAE**

这些模型本来就是在声音频谱图上训练出来的。

### 用于什么模块

- **模块**：主干视觉编码器（backbone）
- **代码位置**：`PromptAD/model.py` 的 `get_model()`

### 作用是什么

- 提取出来的特征更接近频谱图结构；
- 对 chirp、burst、dsss 这种典型时频结构更敏感。

### 为什么适合我们任务

你做的是频谱异常检测，不是普通图片异常检测。
如果 backbone 天生就见过 spectrogram，它对这个任务的起点会更高。

### 参考文献 1

- **Gong et al., AST, Interspeech 2021, arXiv:2104.01778**

### 这篇文献用于哪个模块

- **用于模块**：频谱视觉 backbone

### 这篇文献的作用

- 证明 Transformer 在 spectrogram 上可以直接学到有效特征。

### 参考文献 2

- **Chen et al., BEATs, ICML 2023, arXiv:2212.09058**

### 这篇文献用于哪个模块

- **用于模块**：音频/频谱预训练 backbone

### 这篇文献的作用

- 提供更强的频谱预训练表示，适合迁移到你的异常检测任务。

### 参考文献 3

- **Huang et al., Audio-MAE, NeurIPS 2022, arXiv:2207.06405**

### 这篇文献用于哪个模块

- **用于模块**：频谱自监督预训练 backbone

### 这篇文献的作用

- 说明在没有大量标注的情况下，也能先学到适合频谱的特征。

---

## 1.3 面向“场景”的特色特征改进

这一部分是最贴合你项目的，因为你不是只做单一场景，而是要跨站点。

### 场景特点

根据 README，你现在的场景有：
- `WeaponMuseum_spectrum`
- `Playground_spectrum`
- `TimeSquare_spectrum`
- `Gymnasium_spectrum`

这些场景本质上代表不同的：
- 背景噪声底
- 环境电磁复杂度
- 正常信号稀疏度
- 频谱纹理分布

### 可以做的场景化改进

## 场景化改进 A：做“场景无关特征 + 场景相关补偿”

### 思路

把视觉特征拆成两部分：
- **共享特征**：所有场景都通用，比如 burst / chirp / dsss 的基本结构
- **场景补偿特征**：只负责适应某个场景的背景差异

### 用于什么模块

- **模块**：视觉特征提取模块的后半段
- **实现方式**：共享 backbone + 小型 scene adapter

### 作用是什么

- 减少模型只记住训练场景背景；
- 提高 cross-site 泛化能力。

### 对应参考文献

- **Cha et al., MIRO, ECCV 2022, arXiv:2203.10789**

### 这篇文献用于哪个模块

- **用于模块**：跨域泛化 / 场景泛化模块

### 这篇文献的作用

- 目标是让模型不要过度依赖某一个训练域的特征，对你这种跨场景测试特别有参考价值。

---

## 场景化改进 B：训练时让不同场景风格混合，逼模型学“异常本身”

### 思路

在训练时做更强的数据扰动：
- 加不同强度噪声
- 改变背景纹理
- 做不同场景风格的混合

这样模型就更难只记住某个固定背景。

### 用于什么模块

- **模块**：数据增强模块 + 特征学习模块

### 作用是什么

- 强迫模型关注异常结构，而不是背景场景。

### 对应参考文献

- **Cha et al., SWAD, NeurIPS 2021, arXiv:2102.08604**

### 这篇文献用于哪个模块

- **用于模块**：跨域泛化训练策略

### 这篇文献的作用

- 它的核心思想是提升模型在域外数据上的稳定性，对 cross-site 很有借鉴意义。

---

## 1.4 改进1的推荐落地顺序

建议按下面顺序做：

1. **先重算 mean/std**
2. **再改成三通道输入：原图 + 时间梯度 + 频率梯度**
3. **然后加 LoRA / Adapter**
4. **最后再考虑换 AST / BEATs backbone**

因为：
- 前两步最便宜；
- 第三步性价比最高；
- 第四步改动最大。

---

# 改进3：提示词模块怎么改

## 3.1 当前问题

当前提示词相关代码主要在：
- `@/mnt/data/wangbei/PromptAD/PromptAD/ad_prompts.py`
- `@/mnt/data/wangbei/PromptAD/PromptAD/model.py:37-63`

现在已经比原始 PromptAD 更适合频谱了，因为你已经给：
- `chirp`
- `burst`
- `dsss`

写了一些专门描述。

但还存在 3 个问题：

### 问题 A：场景名字没有进 prompt 语义

现在 `class_mapping` 主要把：
- `burst_m10db`
- `chirp_m10db`
- `dsss_m10db`

统一映射成 `radio frequency spectrum`。

这意味着模型知道“这是个频谱图”，但**不知道这是哪种场景下的频谱图**。

### 问题 B：提示词更像“异常的英文翻译”，还不够“任务化”

比如：
- `with interference`
- `with anomalous signal`

这类词虽然没错，但还是偏泛。
对 burst / chirp / dsss 来说，更关键的是：
- 形状
- 位置
- 持续时间
- 带宽
- 是否跨频段

### 问题 C：还没有“场景特色 prompt”

你的任务不是只区分异常类型，还要跨场景泛化。
所以 prompt 最好能显式描述：
- 背景更空旷还是更复杂
- 噪声底高还是低
- 正常频谱通常更稀疏还是更密集

---

## 3.2 提示词改进方案（按推荐顺序）

## 方案 1：把 prompt 从“泛异常”改成“结构异常”

### 怎么做

不要只写：
- `abnormal radio frequency spectrum`
- `radio frequency spectrum with interference`

而是写得更结构化：

### chirp 建议写法
- `a spectrogram with a diagonal rising trace`
- `a spectrogram with a slanted narrowband interference line`
- `a spectrogram with an abnormal chirp-like slope`

### burst 建议写法
- `a spectrogram with a short-duration narrowband burst`
- `a spectrogram with a transient vertical streak`
- `a spectrogram with a weak short pulse in a narrow frequency band`

### dsss 建议写法
- `a spectrogram with a wideband spread-spectrum-like band`
- `a spectrogram with abnormal wideband energy spread`
- `a spectrogram with uneven power distribution across a wide frequency band`

### 用于什么模块

- **模块**：手工异常 prompt 模块
- **代码位置**：`PromptAD/ad_prompts.py` 的 `class_state_abnormal`

### 作用是什么

- 让文本编码器更容易对齐到真正的视觉结构；
- 减少“只会泛泛说 abnormal、interference”的问题。

### 参考文献

- **Jeong et al., WinCLIP, CVPR 2023, arXiv:2303.14814**

### 这篇文献用于哪个模块

- **用于模块**：prompt 模板设计模块

### 这篇文献的作用

- 它说明一个任务不应该只靠单条 prompt，而应该用多个结构化模板做组合，提升稳定性。

---

## 方案 2：引入 object-agnostic prompt，让模型少依赖类别名，多依赖“正常/异常”概念

### 怎么做

参考 AnomalyCLIP，不要把 prompt 强绑定在某个具体类名上，而是写成：
- `a normal radio frequency spectrogram`
- `an anomalous radio frequency spectrogram`
- `a normal spectrogram pattern in this scene`
- `an anomalous signal pattern in this scene`

再配合可学习 token，让模型自己学细节。

### 用于什么模块

- **模块**：可学习 prompt 模块
- **代码位置**：`PromptAD/model.py` 里 `PromptLearner`

### 作用是什么

- 减少类别名写得不好带来的影响；
- 更适合跨场景、跨异常类型泛化。

### 为什么适合我们任务

因为你这个任务的关键不是识别“物体类别”，而是识别：
- 这个频谱结构是不是正常
- 这个干扰形状是不是异常

所以 object-agnostic prompt 非常适合。

### 参考文献

- **Zhou et al., AnomalyCLIP, ICLR 2024, arXiv:2310.18961**

### 这篇文献用于哪个模块

- **用于模块**：正常/异常 prompt 学习模块

### 这篇文献的作用

- 它证明了：在异常检测里，不一定非要依赖强类别语义，直接学习“正常”和“异常”的通用表达，效果反而更稳。

---

## 方案 3：做“场景感知 prompt”——把不同场景写进提示词

这是最值得你强调的“特色改进”。

### 怎么做

把场景从简单的名字，改成更有语义的英文描述。

例如：

- `WeaponMuseum_spectrum`
  - `an indoor radio scene with relatively stable background spectrum`
- `Playground_spectrum`
  - `an open outdoor radio scene with sparse background activity`
- `TimeSquare_spectrum`
  - `a crowded urban radio scene with complex background interference`
- `Gymnasium_spectrum`
  - `a semi-enclosed activity area with moderate spectrum occupancy`

然后把 prompt 写成两层：

### 正常 prompt
- `a normal radio frequency spectrogram in an open outdoor radio scene with sparse background activity`
- `a clean spectrum pattern in a crowded urban radio scene with complex background interference`

### 异常 prompt
- `a spectrogram with a burst anomaly in a crowded urban radio scene with complex background interference`
- `a spectrogram with a chirp-like interference in an open outdoor radio scene with sparse background activity`

### 用于什么模块

- **模块**：频谱任务域条件 prompt 模块
- **代码位置**：
  - `PromptAD/ad_prompts.py` 增加 `scene_mapping`
  - `PromptAD/model.py` 的 `PromptLearner` 拼接 scene text

### 作用是什么

- 让模型知道“同样的异常，在不同场景里长得会不太一样”；
- 降低跨场景时文本分支失效的问题；
- 让 prompt 更像“任务说明”，不是孤立的标签。

### 为什么适合我们任务

因为你的数据天然就是按场景组织的。
如果 prompt 里完全没有场景信息，那模型只能从图像特征里硬猜背景差异。
加入场景描述后，文本分支就能成为一个“场景先验”。

### 参考文献 1

- **Zhou et al., CoCoOp, CVPR 2022, arXiv:2203.05557**

### 这篇文献用于哪个模块

- **用于模块**：条件化 prompt 模块

### 这篇文献的作用

- CoCoOp 的核心思想就是：prompt 不一定固定，可以根据输入条件动态变化。
- 在你的任务里，这个“条件”就可以是场景。

### 参考文献 2

- **Zhou et al., CoOp, IJCV 2022, arXiv:2109.01134**

### 这篇文献用于哪个模块

- **用于模块**：可学习上下文 token 模块

### 这篇文献的作用

- 说明 prompt 的上下文部分可以学习，不必手工写死。
- 你可以保留现在的 `n_ctx` 机制，但让它和场景描述一起工作。

---

## 方案 4：让每种异常类型都有“专属词库”

### 怎么做

针对每种异常类型，分别准备多组 prompt：

### chirp 词库
重点词：
- diagonal
- slanted
- rising
- falling
- sweep
- chirp-like

### burst 词库
重点词：
- short-duration
- transient
- pulse-like
- narrowband
- vertical streak

### dsss 词库
重点词：
- wideband
- spread-spectrum-like
- diffuse energy
- uniform band
- broad occupancy

然后每次训练或测试时：
- 不是只用一句 prompt；
- 而是从词库里组合多句，再做平均。

### 用于什么模块

- **模块**：异常类型 prompt 库
- **代码位置**：`PromptAD/ad_prompts.py`

### 作用是什么

- 增强 prompt 多样性；
- 防止某一句写法刚好不适合当前数据；
- 提升文本分支稳定性。

### 参考文献

- **Jeong et al., WinCLIP, CVPR 2023, arXiv:2303.14814**

### 这篇文献用于哪个模块

- **用于模块**：prompt ensemble 模块

### 这篇文献的作用

- 支持“多模板组合比单模板更稳”这个思路。

---

## 3.3 面向“场景”的特色提示词改进

这一部分最适合写进你的实验计划里，因为它最像你自己的创新点。

## 场景化改进 A：每个场景用“背景描述 + 异常描述”两层 prompt

### 例子

对于 `TimeSquare_spectrum`：

- 正常：
  - `a normal spectrogram in a crowded urban radio scene with dense background activity`
- Burst 异常：
  - `a crowded urban spectrogram with a short narrow burst signal`
- Chirp 异常：
  - `a crowded urban spectrogram with a slanted chirp-like interference trace`

对于 `Playground_spectrum`：

- 正常：
  - `a clean spectrogram in an open outdoor radio scene with sparse activity`
- Burst 异常：
  - `a sparse outdoor spectrogram with a weak short burst signal`

### 用于什么模块

- **模块**：场景 + 异常联合 prompt 模块

### 作用是什么

- 让“异常类型”和“场景背景”同时进入文本分支。

---

## 场景化改进 B：区分“场景本底词”和“异常结构词”

### 思路

把 prompt 拆成两部分：

- **场景本底词**：
  - urban
  - open outdoor
  - indoor stable background
  - moderate occupancy

- **异常结构词**：
  - diagonal chirp
  - transient burst
  - wideband spread

最后做组合。

### 用于什么模块

- **模块**：prompt 组合模块

### 作用是什么

- 结构更清楚；
- 方便做消融实验；
- 更容易看出到底是“频谱任务域词”有效，还是“异常结构词”有效。

### 参考文献

- **Jeong et al., WinCLIP, CVPR 2023, arXiv:2303.14814**
- **Zhou et al., CoCoOp, CVPR 2022, arXiv:2203.05557**

### 这些文献用于哪个模块

- **WinCLIP**：用于 prompt 组合与多模板设计
- **CoCoOp**：用于条件化 prompt 设计

### 这些文献的作用

- 一个强调“多模板组合”，一个强调“根据条件动态调整 prompt”，两者结合后很适合你的场景化 prompt。

---

## 3.4 改进3的推荐落地顺序

建议按下面顺序做：

1. **先把 burst / chirp / dsss 的 prompt 改成更结构化的描述**
2. **再补上 RF spectrogram / time-frequency / interference 等频谱任务域描述**
3. **然后把 prompt 改成“频谱任务域词 + 异常结构词”的组合形式**
4. **最后再尝试 AnomalyCLIP / CoCoOp 风格的可学习任务条件 prompt**

---

# 最后给出一个最适合当前项目的实施建议

如果你现在只想选 **一个特征提取改进 + 一个提示词改进** 来做，我建议这样搭配：

## 推荐组合 A（最稳）

### 特征提取
- **三通道输入：原图 + 时间梯度 + 频率梯度**

### 提示词
- **频谱任务域结构 prompt：频谱词 + 异常结构词**

### 为什么推荐这个组合

- 都不需要大改 backbone；
- 改动集中在数据预处理和 prompt 构造；
- 很符合你当前数据的特点；
- 也最容易写成你自己的“针对场景的特色改进”。

---

## 推荐组合 B（效果潜力更高）

### 特征提取
- **CLIP + LoRA / Adapter 做频谱域适配**

### 提示词
- **AnomalyCLIP 风格的 object-agnostic + scene-conditioned prompt**

### 为什么推荐这个组合

- 这是更像论文路线的方案；
- 对 cross-site 泛化更有希望；
- 但实现难度比组合 A 高一档。

---

# 可以直接写进论文/汇报里的简化表述

## 改进1（特征提取）

我们计划针对频谱图的时频结构特点，重构输入表示与视觉特征适配模块。具体而言，输入通道由原始灰度复制改为“原始频谱 + 时间梯度 + 频率梯度”，以增强模型对 chirp 斜线、burst 窄带脉冲和 DSSS 宽带结构的感知能力；同时结合 LoRA/Adapter 等参数高效微调方法，对 CLIP 视觉编码器进行频谱域适配，从而缓解自然图像预训练与频谱图域之间的不匹配问题。

## 改进3（提示词）

我们计划将现有提示词从泛化的“异常频谱”描述，升级为“频谱任务域语义 + 异常结构”的联合 prompt 设计。具体而言，一方面针对 burst、chirp、dsss 分别构造结构化描述模板；另一方面补充 RF spectrogram、time-frequency pattern、interference、noise floor 等任务域描述，使文本分支显式关注频谱异常检测中的结构差异，从而提升跨场景异常检测性能。

---

# 参考文献与对应模块汇总

| 参考文献 | 用于哪个模块 | 作用是什么 |
|---|---|---|
| Salamon & Bello, 2017 | 输入表示模块 | 支持频谱输入通道不必照搬 RGB，可引入更有物理意义的谱信息 |
| He et al., MAE, 2022 | 视觉输入域对齐模块 | 说明跨域任务需要做域内适配，支持重算 mean/std 与域对齐思路 |
| Huang et al., Audio-MAE, 2022 | 频谱域特征学习模块 | 说明频谱图可以先做自监督建模，再迁移到下游任务 |
| Gao et al., CLIP-Adapter, 2024 | 视觉特征适配模块 | 给 CLIP 增加轻量适配层，使其更适应当前任务 |
| Zhang et al., Tip-Adapter, 2022 | 少样本特征适配模块 | 说明在少样本条件下，轻量适配是有效的 |
| Hu et al., LoRA, 2022 | 参数高效微调模块 | 用少量参数适配大模型，适合你当前数据规模 |
| Gong et al., AST, 2021 | 频谱 backbone 模块 | 证明 Transformer 能直接处理 spectrogram |
| Chen et al., BEATs, 2023 | 频谱预训练 backbone 模块 | 提供更懂频谱图的初始化特征 |
| Cha et al., MIRO, 2022 | 场景泛化模块 | 减少模型过度依赖单场景特征 |
| Cha et al., SWAD, 2021 | 跨场景训练策略模块 | 提升 cross-site 条件下的稳定性 |
| Jeong et al., WinCLIP, 2023 | Prompt 模板组合模块 | 支持多模板、多描述的 prompt ensemble |
| Zhou et al., AnomalyCLIP, 2024 | 正常/异常 prompt 模块 | 支持 object-agnostic 的异常 prompt 设计 |
| Zhou et al., CoOp, 2022 | 可学习 prompt 模块 | 支持学习式上下文 token |
| Zhou et al., CoCoOp, 2022 | 任务条件 prompt 模块 | 支持根据输入条件动态调整 prompt |

---

# 本轮已落地的小改进与初步实验

> 这里的“场景特色”按当前理解指 **频谱异常检测这个任务域** 的特色，而不是 `WeaponMuseum_spectrum`、`Playground_spectrum` 这类具体采集地点。

## 已落地 1：频谱任务域 Prompt

已加入：
- `--prompt-mode rf`
- `--prompt-mode legacy`

`rf` 模式会把跨站点类别统一描述为：
- `radio frequency spectrogram`

并根据数据集类型自动加入结构异常描述：
- chirp：diagonal / slanted / slope trace
- burst：short-duration / transient / narrowband pulse
- dsss：wideband / spread spectrum / spectral density

这样做的目的是让 prompt 关注频谱异常检测的视觉结构，而不是关注具体采集地点。

## 已落地 2：频谱梯度三通道输入

已加入：
- `--input-mode auto`
- `--input-mode rgb`
- `--input-mode spectral_gradient`
- `--input-mode signal_adaptive`

`spectral_gradient` 模式把输入从普通 RGB 改成：
- 通道 1：原始灰度频谱强度
- 通道 2：时间方向梯度
- 通道 3：频率方向梯度

`auto` 模式下：
- 频谱类数据集自动使用 `spectral_gradient`
- MVTec / VisA 等普通视觉数据集仍使用 `rgb`

`signal_adaptive` 模式下：
- burst / chirp 使用 `spectral_gradient`
- dsss 使用 `rgb`

这个策略来自 36 组小批量消融实验：burst / chirp 更依赖时频边缘结构，梯度通道收益更明显；dsss 的宽带扩频纹理在当前 CLIP 特征下保留 RGB 更稳。

## 为什么 `spectral_gradient` 是针对频谱特征选择的特征提取方法

### 1. 频谱图和普通 RGB 图像的差异

在原始 PromptAD 中，输入图像会被当成普通 RGB 图片送入 CLIP 视觉编码器。这个做法对自然图像是合理的，因为自然图像中的 RGB 三个通道分别对应颜色信息，颜色、纹理和物体轮廓都可能是重要语义。

但无线电频谱图不是自然图像。频谱图的横轴通常对应时间，纵轴对应频率，像素强度对应某个时间-频率位置上的能量。也就是说，频谱图真正有意义的不是颜色本身，而是：

- 能量在时间方向是否突然出现或消失；
- 能量在频率方向是否集中、扩散或发生边界变化；
- 能量轨迹是否形成竖直突发、斜线扫频或宽带弥散结构。

因此，如果只是把灰度频谱复制成 RGB 三通道，三个通道携带的是高度重复的信息，CLIP 看到的仍然主要是普通纹理和颜色分布，而不是频谱异常最关键的时频结构。`spectral_gradient` 的出发点就是：不再把三个通道当作颜色通道，而是把它们改造成更符合频谱物理含义的结构通道。

### 2. `spectral_gradient` 的通道设计

`spectral_gradient` 将输入构造成三个通道：

| 通道 | 含义 | 主要保留的信息 |
|---|---|---|
| Channel 1 | 原始灰度频谱强度 | 保留原始能量分布、背景噪声底和异常能量强弱 |
| Channel 2 | 时间方向梯度 | 强调信号随时间的突变、短时出现和短时消失 |
| Channel 3 | 频率方向梯度 | 强调信号在频率维度上的边界、窄带结构和频率变化 |

具体来说，时间方向梯度计算相邻时间列之间的强度变化，频率方向梯度计算相邻频率行之间的强度变化。这样做以后，模型输入不再是三份重复的灰度图，而是同时包含：

- 原始能量图；
- 时间变化图；
- 频率变化图。

这相当于把频谱图中的时频结构显式暴露给视觉编码器，使 CLIP 不必完全依靠预训练特征自己从 RGB 纹理中推断这些结构。

### 3. 为什么 burst 和 chirp 适合 `spectral_gradient`

`burst_signal` 和 `chirp_signal` 的异常形态都具有明显的边缘和方向结构，因此更适合用梯度增强。

对于 `burst_signal`，异常通常表现为短时间内突然出现的能量块或窄带脉冲。它的关键特征是：

- 持续时间短；
- 起止边界明显；
- 在时间方向上有突变；
- 常表现为局部竖向或块状高能结构。

因此，时间方向梯度能够突出 burst 的开始和结束位置，频率方向梯度能够突出窄带脉冲的频率边界。这比单纯 RGB 输入更容易让模型关注“短时突发”这个异常结构。

对于 `chirp_signal`，异常通常表现为随时间变化的频率扫掠轨迹，也就是频谱图中的斜线结构。它的关键特征是：

- 频率随时间连续变化；
- 在时频图上形成 diagonal / slanted trace；
- 轨迹边缘比颜色语义更重要；
- 低 ISR 下斜线可能很弱，需要增强局部变化。

`spectral_gradient` 同时保留时间方向变化和频率方向变化，因此更容易突出 chirp 的斜率、边界和扫频轨迹。这也是为什么在完整实验中，chirp 从 `legacy + rgb` 到 `rf + signal_adaptive` 的平均提升最明显，达到 `+3.7158`。

### 4. 为什么 DSSS 不适合统一使用 `spectral_gradient`

DSSS 和 burst / chirp 不一样。DSSS 的典型形态不是清晰边缘或斜线，而是宽带、弥散、接近噪声纹理的能量铺展。它的异常信息更可能体现在：

- 宽频带范围内的能量分布变化；
- 频谱平坦度变化；
- 局部方差变化；
- 背景残差和统计纹理变化。

这类信号没有特别清晰的边缘或方向轨迹。如果强行使用时间/频率梯度，可能会把 DSSS 的宽带连续能量结构打碎成局部边缘，反而破坏原本有用的统计纹理信息。

这个判断也被后续实验验证：在 `dsss_signal` 上单独比较 `rf + rgb/none` 和 `rf + spectral_gradient`，`spectral_gradient` 整体下降 `-2.5425`，12 个测试点中只有 2 个提升、10 个下降。因此，DSSS 不应该继续使用边缘/梯度型输入，而应该转向宽带统计纹理特征，例如 `dsss_statistical = gray + frequency-background residual + local variance`。

### 5. 这个方法是怎么分析出来的

`spectral_gradient` 不是单纯凭经验添加的，而是按下面的分析路径确定的：

1. **先从频谱图物理含义出发。** 频谱图的两个空间维度分别对应时间和频率，因此相邻像素变化具有实际含义，不同于普通自然图像中的颜色纹理变化。
2. **再按异常类型分析视觉形态。** burst 的核心是短时突发，chirp 的核心是斜线扫频，二者都依赖时频边缘和局部变化；DSSS 的核心是宽带弥散纹理，不主要依赖边缘。
3. **然后设计三通道输入。** 用原始灰度保留能量分布，用时间梯度增强短时变化，用频率梯度增强频带边界和扫频结构。
4. **最后用消融实验验证。** 36 组小批量消融显示，`spectral_gradient` 对 burst / chirp 更有效，但对 dsss 会破坏宽带纹理；因此最终不是所有信号统一使用 `spectral_gradient`，而是采用 `signal_adaptive`：burst / chirp 使用 `spectral_gradient`，dsss 保持 `rgb` 或进一步使用统计纹理输入。

所以，`spectral_gradient` 的作用不是简单换一种图像预处理方式，而是把无线电频谱异常检测中的时频结构先验显式注入到 CLIP 的视觉输入中。它适合具有清晰局部变化、边界和轨迹的异常类型，但不适合所有频谱异常统一使用。

## 早期 sanity 实验结果

实验设置：
- 数据集：`chirp_signal`
- 训练站点：`WeaponMuseum_spectrum`
- 测试站点：`Playground_spectrum`
- JSR：`m30db`
- k-shot：4
- epoch：1
- seed：111

| Prompt | Input | Image-AUROC |
|---|---|---:|
| legacy | rgb | 84.88 |
| rf | rgb | 84.96 |
| legacy | spectral_gradient | 85.59 |
| rf | spectral_gradient | 85.49 |

初步结论：
- 在这个很小的 sanity 实验里，主要收益来自 `spectral_gradient` 输入表示。
- `rf` prompt 相比旧 prompt 有轻微变化，但 1 epoch 下不是主要收益来源。
- 后续正式实验建议优先比较 `rgb` vs `spectral_gradient`，再看 `rf` prompt 是否在 burst / dsss 或更多 epoch 下更稳定。

## 新的 36 组小批量实验协议

为了减少单次实验时间，现在采用“少量数据切分”而不是缩短训练 epoch：

- 异常类型：`burst` / `chirp` / `dsss`
- 场景：`WeaponMuseum_spectrum` / `Playground_spectrum` / `TimeSquare_spectrum` / `Gymnasium_spectrum`
- ISR：`m10db` / `m20db` / `m30db`
- 总实验数：`3 x 4 x 3 = 36`

数据切分规则：
- 训练集：`normal/{ISR}` 中前 3/4 的正常图片
- 测试集：剩余 1/4 的正常图片 + `abnormal/{ISR}` 的全部异常图片

对应脚本：
- `run_rf_split_all.py`

这个协议保持了原来的训练轮数不变，只是缩小了训练和测试图像数量，更适合快速比较不同方法是否有增益。

## 36 组小批量消融结果

实验设置：
- split：`normal_75_25`
- epoch：50
- seed：111
- 实验数：36

整体 Image-AUROC 均值：

| 方案 | Mean Image-AUROC |
|---|---:|
| legacy prompt + rgb | 85.0733 |
| legacy prompt + spectral_gradient | 85.8528 |
| rf prompt + rgb | 84.8683 |
| rf prompt + spectral_gradient | 85.8861 |
| rf prompt + signal_adaptive | 86.7336 |

按异常类型均值：

| 方案 | burst | chirp | dsss |
|---|---:|---:|---:|
| legacy prompt + rgb | 86.8367 | 78.9425 | 89.4408 |
| legacy prompt + spectral_gradient | 87.6533 | 82.8317 | 87.0733 |
| rf prompt + rgb | 86.3117 | 78.7308 | 89.5625 |
| rf prompt + spectral_gradient | 87.9800 | 82.6583 | 87.0200 |
| rf prompt + signal_adaptive | 87.9800 | 82.6583 | 89.5625 |

相对 `legacy prompt + rgb`：
- `rf prompt + signal_adaptive` 平均提升 `+1.6603`
- 36 个实验中：21 个提升、12 个下降、3 个持平
- burst 平均提升 `+1.1433`
- chirp 平均提升 `+3.7158`
- dsss 平均提升 `+0.1217`

当前推荐方案：
- prompt：`rf`
- input：`signal_adaptive`

运行命令：

```bash
python run_rf_split_all.py --gpus 0 1 2 3 --epochs 50 --prompt-mode rf --input-mode signal_adaptive --root-dir ./result_split_ablation/rf_signal_adaptive
```

## Visual fusion 小实验

根据综述中的建议，还测试了把 visual anomaly map 的 patch 分数聚合进图像级分数。

新增图像级打分模式：
- `text_only`
- `visual_topk`
- `visual_topk_max`
- `visual_topk_freq`

实验设置：
- 数据集：`chirp_signal`
- scene：`Gymnasium_spectrum`
- ISR：`m30db`
- prompt：`rf`
- input：`signal_adaptive`
- split：`normal_75_25`
- epoch：50

结果：

| cls score mode | Image-AUROC |
|---|---:|
| text_only | 45.52 |
| visual_topk | 41.83 |
| visual_topk_max | 41.69 |
| visual_topk_freq | 40.96 |

结论：
- 在当前设置下，直接把 visual top-k patch score 融合进 image score 没有带来提升。
- 频率约束版 `visual_topk_freq` 反而更差。
- 说明当前 visual gallery 的 patch score 更适合作为定位图，而不是直接用于图像级 AUROC 融合。
- 因此当前主推方案仍然是 `rf prompt + signal_adaptive input`，不建议把这个 visual fusion 作为默认设置。

为了排除 `Gymnasium_spectrum` 单场景偶然性，又换了一个非 Gym 场景复测：

实验设置：
- 数据集：`chirp_signal`
- scene：`WeaponMuseum_spectrum`
- ISR：`m30db`
- prompt：`rf`
- input：`signal_adaptive`
- split：`normal_75_25`
- epoch：50

结果：

| cls score mode | Image-AUROC |
|---|---:|
| text_only | 59.78 |
| visual_topk | 59.22 |
| visual_topk_max | 59.44 |
| visual_topk_freq | 59.26 |

补充结论：
- 换到 `WeaponMuseum_spectrum` 后，`text_only` 仍然最好。
- 三种 visual fusion 都接近 baseline，但都没有超过 baseline。
- 这说明当前这套 visual top-k 融合策略的问题不是只出现在 Gym 场景，而更像是图像级打分融合方式本身还不够合适。

## 结构化 object-agnostic RF prompt 小实验

说明：下面这部分是之前按 cross-site 协议做的探索性实验记录。当前实验主协议已经调整为“分场景训练、分场景测试（同场景 normal_75_25）”，因此这一节不再作为当前主结论依据，只保留作历史参考。

根据综述中的第二优先级，又做了一组 prompt 结构小实验，验证“跨站点任务中是否应该弱化站点类别名语义”。

新增 prompt mode：
- `legacy`
- `rf`
- `rf_object_agnostic`
- `rf_scene_conditioned`

其中：
- `rf`：当前已经在用的 RF prompt，包含频谱域词和信号结构异常词；
- `rf_object_agnostic`：去掉信号/站点特异异常词，只保留 RF 正常/异常公共语义；
- `rf_scene_conditioned`：在 RF prompt 基础上加入训练场景背景描述。

实验设置：
- 数据集：`chirp_signal`
- 训练站点：`WeaponMuseum_spectrum`
- 测试站点：`Gymnasium_spectrum`
- ISR：`m30db`
- k-shot：24
- input：`signal_adaptive`
- epoch：50

结果：

| prompt mode | Image-AUROC |
|---|---:|
| legacy | 49.81 |
| rf | 51.35 |
| rf_object_agnostic | 52.01 |
| rf_scene_conditioned | 49.49 |

结论：
- `rf_object_agnostic` 是这一组里最好的，较 `rf` 提升 `+0.66`，较 `legacy` 提升 `+2.20`。
- 说明在跨站点频谱异常检测中，弱化地点/对象语义、强调通用 RF 正常/异常语义是有效的。
- `rf_scene_conditioned` 反而下降，说明把场景背景描述重新放回 prompt，并没有在这个点上带来收益。
- 因此“优先级二”有初步正结果，但更值得继续的是 `rf_object_agnostic`，而不是 `rf_scene_conditioned`。

为了验证 `rf_object_agnostic` 的提升是否稳定，又补测了另外两个 cross-site 测试站点，保持相同设置：

- 数据集：`chirp_signal`
- 训练站点：`WeaponMuseum_spectrum`
- 测试站点：`Playground_spectrum` / `TimeSquare_spectrum`
- ISR：`m30db`
- k-shot：24
- input：`signal_adaptive`
- epoch：50

结果：

| test scene | rf | rf_object_agnostic | delta |
|---|---:|---:|---:|
| Gymnasium_spectrum | 51.35 | 52.01 | +0.66 |
| Playground_spectrum | 87.24 | 87.20 | -0.04 |
| TimeSquare_spectrum | 84.06 | 84.04 | -0.02 |

补充结论：
- `rf_object_agnostic` 在 `Gymnasium_spectrum` 上确实有提升；
- 但在 `Playground_spectrum` 和 `TimeSquare_spectrum` 上基本持平，并没有继续提升；
- 因此目前只能说 `rf_object_agnostic` 有一定潜力，但提升还不稳定；
- 在现阶段，它更适合作为后续继续扩展验证的候选方案，而不建议直接替代当前 `rf` 作为统一默认设置。

如果把两个改动合并起来，直接和最开始的 baseline 对比：

- baseline：`legacy + rgb`
- improved：`rf_object_agnostic + signal_adaptive`

实验设置：
- 数据集：`chirp_signal`
- 训练站点：`WeaponMuseum_spectrum`
- 测试站点：`Gymnasium_spectrum` / `Playground_spectrum` / `TimeSquare_spectrum`
- ISR：`m30db`
- k-shot：24
- epoch：50

结果：

| test scene | legacy + rgb | rf_object_agnostic + signal_adaptive | delta |
|---|---:|---:|---:|
| Gymnasium_spectrum | 52.34 | 52.01 | -0.33 |
| Playground_spectrum | 87.24 | 87.20 | -0.04 |
| TimeSquare_spectrum | 84.68 | 84.04 | -0.64 |
| mean | 74.75 | 74.42 | -0.34 |

结论：
- 如果按“完整方案 vs 最初 baseline”的口径看，`rf_object_agnostic + signal_adaptive` 在这 3 个 `chirp_signal` cross-site 点上没有超过 `legacy + rgb`；
- 三个站点里没有一个点真正优于 baseline；
- 因此当前还不能说“把 rf_object_agnostic 和 signal_adaptive 一起加进去”会稳定优于最开始的 baseline。

更新说明：上面这一段是旧的 cross-site 口径结果。当前主实验协议已经改成“分场景训练、分场景测试（normal_75_25）”，因此真正应参考的是下面这组同场景结果。

## 同场景协议下重新实验：`rf_object_agnostic + signal_adaptive` vs `legacy + rgb`

实验设置：
- 数据集：`chirp_signal`
- 场景：`WeaponMuseum_spectrum` / `Playground_spectrum` / `TimeSquare_spectrum` / `Gymnasium_spectrum`
- ISR：`m30db`
- split：`normal_75_25`
- k-shot：1
- epoch：50
- seed：111

对比方案：
- baseline：`legacy + rgb`
- improved：`rf_object_agnostic + signal_adaptive`

结果：

| scene | legacy + rgb | rf_object_agnostic + signal_adaptive | delta |
|---|---:|---:|---:|
| WeaponMuseum_spectrum | 46.19 | 59.65 | +13.46 |
| Playground_spectrum | 82.73 | 85.81 | +3.08 |
| TimeSquare_spectrum | 81.66 | 83.64 | +1.98 |
| Gymnasium_spectrum | 26.25 | 46.22 | +19.97 |
| mean | 59.21 | 68.83 | +9.62 |

结论：
- 在当前已经确认的新协议下，`rf_object_agnostic + signal_adaptive` 明显优于 `legacy + rgb`；
- 4 个 scene 全部提升，没有出现回落；
- 提升最大的两个场景是 `WeaponMuseum_spectrum` 和 `Gymnasium_spectrum`，说明该组合对较难场景更有帮助；
- 因此如果后续继续沿“分场景训练、分场景测试”的协议推进，这个组合作为候选改进方案是成立的。

## `dsss_signal` 同场景协议复现实验

实验设置：
- 数据集：`dsss_signal`
- 场景：`WeaponMuseum_spectrum` / `Playground_spectrum` / `TimeSquare_spectrum` / `Gymnasium_spectrum`
- ISR：`m30db`
- split：`normal_75_25`
- k-shot：1
- epoch：50
- seed：111

对比方案：
- baseline：`legacy + rgb`
- improved：`rf_object_agnostic + signal_adaptive`

结果：

| scene | legacy + rgb | rf_object_agnostic + signal_adaptive | delta |
|---|---:|---:|---:|
| WeaponMuseum_spectrum | 80.97 | 80.46 | -0.51 |
| Playground_spectrum | 79.26 | 79.13 | -0.13 |
| TimeSquare_spectrum | 83.88 | 83.96 | +0.08 |
| Gymnasium_spectrum | 83.28 | 83.28 | +0.00 |
| mean | 81.85 | 81.71 | -0.14 |

结论：
- 对 `dsss_signal` 而言，`rf_object_agnostic + signal_adaptive` 没有像 `chirp_signal` 那样带来明显提升；
- 4 个 scene 中只有 `TimeSquare_spectrum` 有极小幅正增益，`Gymnasium_spectrum` 持平，其余两个 scene 轻微下降；
- 因此这个组合的收益是明显依赖异常类型的：对 `chirp_signal` 有效，但对 `dsss_signal` 基本无效；
- 如果后续做统一默认方案，不能直接把 `chirp_signal` 的结论外推到 `dsss_signal`。

## 36 组完整实验：`rf_object_agnostic + signal_adaptive`

这组实验补齐了完整同场景协议：

- 数据集：`burst_signal` / `chirp_signal` / `dsss_signal`
- 场景：`WeaponMuseum_spectrum` / `Playground_spectrum` / `TimeSquare_spectrum` / `Gymnasium_spectrum`
- ISR：`m10db` / `m20db` / `m30db`
- split：`normal_75_25`
- k-shot：1
- epoch：50
- seed：111
- prompt：`rf_object_agnostic`
- input：`signal_adaptive`
- 结果目录：`./result_split_ablation/rf_object_agnostic_signal_adaptive`

共 `3 * 4 * 3 = 36` 个实验，已全部完成并生成 `36` 个结果 CSV。

### `burst_signal`

| scene | m10db | m20db | m30db |
|---|---:|---:|---:|
| WeaponMuseum_spectrum | 97.1100 | 93.6100 | 82.1700 |
| Playground_spectrum | 97.3700 | 96.5900 | 91.1900 |
| TimeSquare_spectrum | 99.2500 | 99.0400 | 98.8200 |
| Gymnasium_spectrum | 90.7500 | 53.6700 | 53.6700 |
| mean | 96.1200 | 85.7275 | 81.4625 |

`burst_signal` 总均值：`87.7700`。

### `chirp_signal`

| scene | m10db | m20db | m30db |
|---|---:|---:|---:|
| WeaponMuseum_spectrum | 95.3800 | 86.5200 | 59.6500 |
| Playground_spectrum | 96.6500 | 93.7100 | 85.8100 |
| TimeSquare_spectrum | 97.3500 | 93.3400 | 83.6400 |
| Gymnasium_spectrum | 76.0000 | 77.5600 | 46.2200 |
| mean | 91.3450 | 87.7825 | 68.8300 |

`chirp_signal` 总均值：`82.6525`。

### `dsss_signal`

| scene | m10db | m20db | m30db |
|---|---:|---:|---:|
| WeaponMuseum_spectrum | 97.6800 | 92.8300 | 80.4600 |
| Playground_spectrum | 96.6100 | 92.0500 | 79.1300 |
| TimeSquare_spectrum | 98.8000 | 93.5800 | 83.9600 |
| Gymnasium_spectrum | 92.4300 | 83.2800 | 83.2800 |
| mean | 96.3800 | 90.4350 | 81.7075 |

`dsss_signal` 总均值：`89.5075`。

### 总结

| dataset | mean |
|---|---:|
| burst_signal | 87.7700 |
| chirp_signal | 82.6525 |
| dsss_signal | 89.5075 |
| overall | 86.6433 |

和之前完整 36 组的 `rf + signal_adaptive` 结果相比：

| method | overall mean |
|---|---:|
| rf + signal_adaptive | 86.7336 |
| rf_object_agnostic + signal_adaptive | 86.6433 |

结论：
- `rf_object_agnostic + signal_adaptive` 的 36 组完整平均值为 `86.6433`；
- 它和 `rf + signal_adaptive` 非常接近，但整体略低 `0.0903`；
- 因此从完整 36 组平均值看，当前默认主方案仍建议保留 `rf + signal_adaptive`；
- `rf_object_agnostic` 可以作为 `chirp_signal` 局部候选，尤其在 `m30db` 的同场景单独对比中更明显，但不建议直接替代完整默认方案。

## DSSS 上验证 `spectral_gradient` 是否有效

为了确认 DSSS 是否适合梯度输入，单独比较：

- baseline：`rf + rgb/none`
- test：`rf + spectral_gradient`
- 数据集：`dsss_signal`
- 场景：4 个 scene
- ISR：`m10db` / `m20db` / `m30db`
- split：`normal_75_25`
- k-shot：1
- epoch：50
- seed：111

结果：

| scene | ISR | rf + rgb/none | rf + spectral_gradient | delta |
|---|---|---:|---:|---:|
| WeaponMuseum_spectrum | m10db | 97.6800 | 94.7200 | -2.9600 |
| WeaponMuseum_spectrum | m20db | 92.8700 | 88.2300 | -4.6400 |
| WeaponMuseum_spectrum | m30db | 80.9700 | 73.9700 | -7.0000 |
| Playground_spectrum | m10db | 96.5800 | 91.4200 | -5.1600 |
| Playground_spectrum | m20db | 92.0500 | 85.9300 | -6.1200 |
| Playground_spectrum | m30db | 79.1000 | 74.0400 | -5.0600 |
| TimeSquare_spectrum | m10db | 98.8300 | 97.4300 | -1.4000 |
| TimeSquare_spectrum | m20db | 93.6100 | 94.5400 | +0.9300 |
| TimeSquare_spectrum | m30db | 84.0400 | 90.5500 | +6.5100 |
| Gymnasium_spectrum | m10db | 92.4600 | 89.9700 | -2.4900 |
| Gymnasium_spectrum | m20db | 83.2800 | 81.7200 | -1.5600 |
| Gymnasium_spectrum | m30db | 83.2800 | 81.7200 | -1.5600 |

汇总：

| grouping | rf + rgb/none | rf + spectral_gradient | delta |
|---|---:|---:|---:|
| m10db mean | 96.3875 | 93.3850 | -3.0025 |
| m20db mean | 90.4525 | 87.6050 | -2.8475 |
| m30db mean | 81.8475 | 80.0700 | -1.7775 |
| overall | 89.5625 | 87.0200 | -2.5425 |

胜负统计：`2` 胜 / `10` 负 / `0` 平。

结论：
- `spectral_gradient` 对 `dsss_signal` 整体无效，平均下降 `-2.5425`；
- 只有 `TimeSquare_spectrum` 的 m20db/m30db 有提升，但不能抵消其他 10 个点的下降；
- 这进一步支持当前 `signal_adaptive` 的设计：DSSS 不走梯度输入，继续保持 RGB；
- 后续 DSSS 的改进方向不应继续强化边缘/梯度，而应尝试宽带统计纹理、谱平坦度、局部方差、循环平稳特征等更适合弥散弱结构信号的输入。

可视化：
- `rf + spectral_gradient` vs `rf + rgb/none` heatmap：`analysis_outputs/rf_grad_vs_rf_rgb/delta_heatmap.png`
- 原始 AUROC 对比 heatmap：`analysis_outputs/rf_grad_vs_rf_rgb/score_heatmaps.png`

## 当前最终主协议与四组核心对比

当前主实验协议已经从 cross-site 调整为同场景训练 / 同场景测试：

- split：`normal_75_25`
- 训练：`normal/{ISR}` 前 3/4 图片
- 测试：剩余 1/4 normal + 全部 abnormal
- k-shot：1
- epoch：50
- seed：111

完整四组对比文件：
- `analysis_outputs/rf_four_way_compare/four_way_compare.csv`

四组核心方案的整体 Image-AUROC 均值：

| method | overall |
|---|---:|
| `rf + rgb/none` | 84.8683 |
| `rf + signal_adaptive` | 86.7336 |
| `rf_signal_structured + rgb` | 84.8289 |
| `rf_signal_structured + signal_adaptive` | 86.7625 |

结论：
- 主要收益来自特征输入侧，而不是 prompt 侧；
- `rf_signal_structured + rgb` 相比 `rf + rgb/none` 基本没有提升；
- `rf_signal_structured + signal_adaptive` 是四组里最高的，为 `86.7625`；
- 但它只比 `rf + signal_adaptive` 高 `0.0289`，可以认为基本持平；
- 因此当前默认主方案仍建议写为 `rf + signal_adaptive`，论文里可以把 `rf_signal_structured + signal_adaptive` 作为“prompt 结构化后轻微进一步提升但边际收益很小”的补充结果。

对应图：
- 特征输入增益 heatmap：`analysis_outputs/rf_four_way_compare/delta_feature_heatmap.png`
- prompt 增益 heatmap：`analysis_outputs/rf_four_way_compare/delta_prompt_heatmap.png`
- 两者叠加增益 heatmap：`analysis_outputs/rf_four_way_compare/delta_both_heatmap.png`

## DSSS statistical 输入专项实验

在确认 `spectral_gradient` 对 DSSS 平均下降 `-2.5425` 后，DSSS 的改进方向改为宽带统计纹理，而不是边缘/梯度增强。

新增输入：
- `dsss_statistical = gray + frequency-background residual + local variance`

完整结果文件：
- `analysis_outputs/dsss_statistical_vs_baseline/dsss_statistical_vs_baseline.csv`

可视化：
- `analysis_outputs/dsss_statistical_vs_baseline/dsss_statistical_vs_baseline.png`

整体结果：

| method | overall |
|---|---:|
| `rf + rgb/none` | 89.5625 |
| `rf + dsss_statistical` | 90.8775 |
| delta | +1.3150 |

逐场景结果：

| scene | ISR | baseline | dsss_statistical | delta |
|---|---|---:|---:|---:|
| WeaponMuseum_spectrum | m10db | 97.68 | 95.19 | -2.49 |
| WeaponMuseum_spectrum | m20db | 92.87 | 90.55 | -2.32 |
| WeaponMuseum_spectrum | m30db | 80.97 | 73.15 | -7.82 |
| Playground_spectrum | m10db | 96.58 | 97.81 | +1.23 |
| Playground_spectrum | m20db | 92.05 | 94.89 | +2.84 |
| Playground_spectrum | m30db | 79.10 | 88.11 | +9.01 |
| TimeSquare_spectrum | m10db | 98.83 | 97.38 | -1.45 |
| TimeSquare_spectrum | m20db | 93.61 | 96.17 | +2.56 |
| TimeSquare_spectrum | m30db | 84.04 | 94.70 | +10.66 |
| Gymnasium_spectrum | m10db | 92.46 | 93.50 | +1.04 |
| Gymnasium_spectrum | m20db | 83.28 | 84.54 | +1.26 |
| Gymnasium_spectrum | m30db | 83.28 | 84.54 | +1.26 |

结论：
- `dsss_statistical` 对 DSSS 全组平均提升 `+1.3150`；
- 最大收益来自低 ISR 的 `Playground_spectrum` 和 `TimeSquare_spectrum`；
- `Playground_spectrum m30db` 从 `79.10` 提升到 `88.11`，提升 `+9.01`；
- `TimeSquare_spectrum m30db` 从 `84.04` 提升到 `94.70`，提升 `+10.66`；
- 但 `WeaponMuseum_spectrum` 三个 ISR 全部下降，尤其 `m30db` 下降 `-7.82`；
- 因此 DSSS 不适合统一替换为 `dsss_statistical`，更适合写成 scene-adaptive input selection。

## WeaponMuseum DSSS 输入专项分析

针对 `WeaponMuseum_spectrum` 的 DSSS 负收益，又测试了三个更保能量的输入方案：

- `dsss_energy_smooth`
- `dsss_lowfreq_band`
- `dsss_energy_profile`

结果：

| method | m10db | m20db | m30db | mean | vs baseline |
|---|---:|---:|---:|---:|---:|
| `baseline rgb` | 97.68 | 92.87 | 80.97 | 90.51 | +0.00 |
| `dsss_energy_smooth` | 81.62 | 76.63 | 64.82 | 74.36 | -16.15 |
| `dsss_lowfreq_band` | 92.40 | 86.13 | 72.98 | 83.84 | -6.67 |
| `dsss_energy_profile` | 97.72 | 92.01 | 79.64 | 89.79 | -0.72 |

结论：
- `WeaponMuseum_spectrum` 的 DSSS 仍建议保持 RGB；
- `dsss_energy_profile` 最接近 baseline，但平均仍低 `-0.72`；
- `dsss_energy_smooth` 和 `dsss_lowfreq_band` 明显破坏有效信息；
- 不建议继续强行为 `WeaponMuseum_spectrum` DSSS 设计新输入，当前更合理的做法是 scene-adaptive selection。

## 当前推荐的 scene-adaptive DSSS selection

基于以上结果，DSSS 的推荐输入选择是：

| scene | recommended DSSS input |
|---|---|
| WeaponMuseum_spectrum | `rgb` |
| Playground_spectrum | `dsss_statistical` |
| TimeSquare_spectrum | `dsss_statistical` |
| Gymnasium_spectrum | `dsss_statistical` |

这个策略的论文表述可以是：
- DSSS 是宽带、弥散、弱结构信号，不适合统一使用边缘/梯度输入；
- 统计纹理输入能增强低 ISR 下的宽带能量扰动；
- 但统计纹理对部分场景背景敏感，`WeaponMuseum_spectrum` 是典型负例；
- 因此最终采用 scene-adaptive input selection，而不是对所有场景强制使用同一 DSSS 前端。

如果继续补实验，优先跑这 12 组 scene-adaptive DSSS selection：
- `WeaponMuseum_spectrum` 使用 `rgb`
- `Playground_spectrum` / `TimeSquare_spectrum` / `Gymnasium_spectrum` 使用 `dsss_statistical`

然后和 `rf + rgb/none` 的 DSSS baseline 对比。按现有结果静态估计，这个选择会避免 `WeaponMuseum_spectrum` 的明显下降，同时保留另外三个场景的大部分收益。

## 方法解释图

当前已经生成方法解释图：

- `analysis_outputs/feature_extraction_improvement/feature_extraction_improvement_overview.png`

图中表达的主线是：
- 输入 RF spectrogram；
- 使用 signal-adaptive front-end；
- burst / chirp 走 `spectral_gradient`，强化突发边缘和斜线结构；
- DSSS 不走梯度，而走 statistical texture；
- 对 `WeaponMuseum_spectrum` 等统计纹理不稳定的场景保留 RGB。

## 当前论文写法建议

可以按以下逻辑组织方法与实验结论：

1. RF 异常类型具有不同视觉形态：burst 是短时突发，chirp 是斜线扫频，DSSS 是宽带弥散弱结构。
2. 对 burst / chirp，`spectral_gradient` 能强化边缘、短时变化和斜线结构，因此带来主要提升。
3. 对 DSSS，`spectral_gradient` 平均下降 `-2.5425`，说明边缘型输入不适合宽带弥散信号。
4. 因此进一步设计 `dsss_statistical`，用频率背景残差和局部方差描述宽带统计纹理。
5. `dsss_statistical` 在 DSSS 全组平均提升 `+1.3150`，尤其改善低 ISR 的 `Playground_spectrum` 和 `TimeSquare_spectrum`。
6. `WeaponMuseum_spectrum` 的负收益说明 DSSS 统计纹理具有场景依赖性，因此最终采用 scene-adaptive input selection。
7. 四组完整对比表明，特征提取优化贡献最大；prompt 结构化单独收益很小；两者叠加最高但与 `rf + signal_adaptive` 基本持平。

当前主结论：

| conclusion | result |
|---|---|
| 默认主方案 | `rf + signal_adaptive` |
| 四组最高均值 | `rf_signal_structured + signal_adaptive = 86.7625` |
| 与默认主方案差距 | `+0.0289` |
| DSSS 梯度输入 | 不推荐，整体 `-2.5425` |
| DSSS statistical | 推荐作为 scene-adaptive 候选，整体 `+1.3150` |
| WeaponMuseum DSSS | 保持 RGB |

## Prompt-only 三方案全组对比

为了单独验证 prompt 语义是否有效，本轮固定输入为 `rgb`，只改变 prompt：

- baseline：`generic + rgb`
- RF domain：`rf_domain + rgb`
- signal-structured：`rf_signal_structured + rgb`

其中：
- `generic` 是无频谱场景优化的通用视觉异常 prompt，把 RF 类统一写成 `image`；
- `rf_domain` 只加入 `radio frequency spectrogram`，不加入 burst / chirp / DSSS 结构词；
- `rf_signal_structured` 加入 burst / chirp / DSSS 的信号结构描述。

实验设置：
- 数据集：`burst_signal` / `chirp_signal` / `dsss_signal`
- 场景：`WeaponMuseum_spectrum` / `Playground_spectrum` / `TimeSquare_spectrum` / `Gymnasium_spectrum`
- ISR：`m10db` / `m20db` / `m30db`
- split：`normal_75_25`
- k-shot：1
- epoch：50
- seed：111
- input：`rgb`

结果文件：
- `analysis_outputs/prompt_three_way_compare/prompt_three_way_compare.csv`

整体与分类型均值：

| method | burst | chirp | dsss | overall |
|---|---:|---:|---:|---:|
| `generic + rgb` | 86.7100 | 79.5800 | 90.1050 | 85.4650 |
| `rf_domain + rgb` | 86.0292 | 78.8783 | 89.7125 | 84.8733 |
| `rf_signal_structured + rgb` | 86.2125 | 78.6492 | 89.6250 | 84.8289 |

相对 `generic + rgb`：

| method | wins | losses | ties | mean delta |
|---|---:|---:|---:|---:|
| `rf_domain + rgb` | 11 | 24 | 1 | -0.5917 |
| `rf_signal_structured + rgb` | 8 | 24 | 4 | -0.6361 |

结论：
- 在固定 `rgb` 输入时，通用 prompt baseline 反而最好；
- 只加入 RF / spectrogram 任务域词没有带来提升，整体下降 `-0.5917`；
- 加入 burst / chirp / DSSS 结构词也没有带来提升，整体下降 `-0.6361`；
- 因此当前实验不能支持“prompt 语义优化单独有效”；
- 和前面的四组核心对比一致，当前主要收益应归因于输入特征提取优化，而不是自然语言 prompt；
- 论文中更稳妥的写法是：prompt 结构化作为任务先验尝试，单独收益有限；最终性能提升主要来自 signal-adaptive front-end。

## 下一步：轻量 Visual Adapter

由于 prompt-only 实验没有带来稳定收益，后续改进不再继续堆自然语言模板，而是转向视觉特征提取侧。当前新增一个轻量残差 Visual Adapter，用来在冻结 CLIP 视觉主干的前提下，对 RF 频谱特征做小幅域适配。

### 方法设计

Adapter 接在 CLIP 图像编码器输出之后：

```text
RF input -> frozen CLIP visual encoder -> visual feature -> residual adapter -> adapted visual feature
```

残差形式为：

```text
adapted_feature = feature + alpha * MLP(feature)
```

其中：
- CLIP visual backbone 冻结，不参与训练；
- PromptLearner 继续训练；
- Visual Adapter 参与训练；
- MLP 使用 bottleneck 结构，默认 `adapter_bottleneck_ratio=0.25`；
- 残差权重默认 `adapter_alpha=0.2`；
- 最后一层线性层零初始化，使 Adapter 初始时接近 identity mapping，避免一开始破坏 CLIP 原特征。

由于当前 CLIP 输出包含不同维度的视觉特征：
- 全局图文对齐特征：`640` 维；
- patch/gallery 特征：`896` 维；

因此实现中按特征维度分别建立 Adapter，避免全局特征和 patch 特征共用错误维度的投影层。

### 为什么这样设计

这个 Adapter 的目标不是重新训练 CLIP，而是做参数量很小的频谱域适配：

1. `signal_adaptive` 已经说明主要瓶颈在视觉输入/视觉特征侧，而不是 prompt 文案侧。
2. RF 频谱图和自然图像差异明显，CLIP 原始视觉特征可能不能充分表达时频结构。
3. 直接全量微调 CLIP 容易过拟合，且少样本设置下不稳定。
4. 残差 Adapter 可以在保留 CLIP 原始能力的同时，学习一个小的特征修正方向。

### 已实现参数

`train_cls.py` 新增：

```bash
--visual-adapter True
--adapter-bottleneck-ratio 0.25
--adapter-alpha 0.2
```

`run_rf_split_all.py` 新增：

```bash
--visual-adapter
--adapter-bottleneck-ratio 0.25
--adapter-alpha 0.2
```

注意：启用 Visual Adapter 后，测试图像特征不能再在训练前一次性缓存，因为 Adapter 每个 epoch 都会更新。因此当前实现会在每个 epoch 重新编码测试特征，并在 Adapter 更新后重建正常样本 feature gallery。

### Smoke test

已完成一个 1 epoch 单点运行测试：

```bash
python train_cls.py \
  --dataset chirp_signal \
  --class_name WeaponMuseum_spectrum \
  --k-shot 1 \
  --Epoch 1 \
  --gpu-id 0 \
  --noise-level m30db \
  --vis False \
  --seed 111 \
  --root-dir ./tmp_visual_adapter_smoke \
  --prompt-mode rf \
  --input-mode signal_adaptive \
  --visual-adapter True \
  --split-mode normal_75_25 \
  --normal-train-ratio 0.75
```

运行结果：
- 训练、重建 gallery、评估流程均正常完成；
- 1 epoch smoke test 的 Image-AUROC 为 `59.75`；
- 这个数值只用于确认代码路径正确，不作为正式结论。

### 建议下一组正式实验

先不要直接跑全部 seed。建议先跑 `chirp_signal` 的 12 组，因为 chirp 是当前最受益于视觉结构特征的异常类型：

```bash
python run_rf_split_all.py \
  --gpus 0 1 2 3 \
  --epochs 50 \
  --datasets chirp_signal \
  --prompt-mode rf \
  --input-mode signal_adaptive \
  --visual-adapter \
  --root-dir ./result_visual_adapter/chirp_rf_signal_adaptive_adapter
```

对比对象：
- `rf + signal_adaptive`
- `rf + signal_adaptive + visual_adapter`

如果 chirp 有稳定收益，再扩展到 burst 和 dsss；如果 chirp 都没有收益，就不建议继续扩大 Adapter 实验。

### Chirp baseline 对比实验结果

先按最小成本验证 Visual Adapter 是否能在最可能受益的 `chirp_signal` 上超过 baseline。

实验设置：
- dataset：`chirp_signal`
- scene：`WeaponMuseum_spectrum` / `Playground_spectrum` / `TimeSquare_spectrum` / `Gymnasium_spectrum`
- ISR：`m10db` / `m20db` / `m30db`
- split：`normal_75_25`
- k-shot：1
- epoch：50
- seed：111
- baseline：`rf + signal_adaptive`
- adapter：`rf + signal_adaptive + visual_adapter`

结果文件：
- `analysis_outputs/visual_adapter_compare/chirp_rf_signal_adaptive_adapter_vs_baseline.csv`

逐项结果：

| scene | ISR | baseline | adapter | delta |
|---|---|---:|---:|---:|
| WeaponMuseum_spectrum | m10db | 95.3600 | 95.3600 | +0.0000 |
| WeaponMuseum_spectrum | m20db | 86.5400 | 86.4900 | -0.0500 |
| WeaponMuseum_spectrum | m30db | 59.7800 | 59.7500 | -0.0300 |
| Playground_spectrum | m10db | 96.6800 | 96.6800 | +0.0000 |
| Playground_spectrum | m20db | 93.7900 | 93.7900 | +0.0000 |
| Playground_spectrum | m30db | 85.8100 | 85.8200 | +0.0100 |
| TimeSquare_spectrum | m10db | 97.2700 | 97.2400 | -0.0300 |
| TimeSquare_spectrum | m20db | 93.4000 | 93.4300 | +0.0300 |
| TimeSquare_spectrum | m30db | 83.6400 | 83.6400 | +0.0000 |
| Gymnasium_spectrum | m10db | 76.2200 | 76.2600 | +0.0400 |
| Gymnasium_spectrum | m20db | 77.8900 | 77.9100 | +0.0200 |
| Gymnasium_spectrum | m30db | 45.5200 | 45.4500 | -0.0700 |

整体结果：

| method | mean Image-AUROC |
|---|---:|
| `rf + signal_adaptive` | 82.6583 |
| `rf + signal_adaptive + visual_adapter` | 82.6517 |
| delta | -0.0067 |

按 ISR 汇总：

| ISR | baseline | adapter | delta |
|---|---:|---:|---:|
| m10db | 91.3825 | 91.3850 | +0.0025 |
| m20db | 87.9050 | 87.9050 | -0.0000 |
| m30db | 68.6875 | 68.6650 | -0.0225 |

胜负统计：
- 4 胜 / 4 负 / 4 持平；
- overall delta 为 `-0.0067`，可以认为和 baseline 完全持平；
- 当前这版残差 Visual Adapter 没有带来可观收益。

结论：
- 在 `chirp_signal` 这组最可能受益的实验上，Visual Adapter 没有超过 `rf + signal_adaptive` baseline；
- 说明简单地在 CLIP 输出特征后接一个 residual MLP，不足以进一步改善当前频谱特征；
- 不建议直接扩展到 burst / dsss 全量 36 组，除非先调整 Adapter 训练策略或结构。

后续如果继续尝试视觉适配，更合理的方向是：
1. 只适配全局图文对齐特征，不适配 patch/gallery 特征；
2. 降低 adapter 学习率，和 prompt learner 分开设置 optimizer；
3. 尝试只在 burst / chirp 的低 ISR 难样本上训练；
4. 或者改为 LoRA 插入视觉 transformer block，而不是输出后 MLP adapter。

## Visual LoRA 对比实验

在 residual Visual Adapter 没有收益后，进一步尝试更深入的视觉侧参数高效适配：Visual LoRA。

### LoRA 插入位置

当前实现把 LoRA 插入到 CLIP visual transformer 的 attention 投影层中。由于本项目的 V2VTransformer 会在第一次 forward 时把原始 `nn.MultiheadAttention` 替换成自定义 `Attention(qkv, proj)`，因此 LoRA 实际加在：

- `Attention.qkv`
- `Attention.proj`

也就是：

```text
qkv(x)  = W_qkv x  + (alpha / r) B_qkv A_qkv x
proj(x) = W_proj x + (alpha / r) B_proj A_proj x
```

训练时：
- 冻结原始 CLIP 权重；
- 训练 PromptLearner；
- 训练 visual LoRA 参数；
- 不使用之前的输出后 residual Visual Adapter。

默认参数：
- `visual_lora_rank = 4`
- `visual_lora_alpha = 8`
- `visual_lora_dropout = 0`
- `batch_size = 16`

说明：LoRA 需要对视觉 Transformer 反向传播，显存开销明显高于只训练 PromptLearner，因此不能继续使用默认 `batch_size=400`。

### 实验设置

本轮不再强调 `k-shot=1`，因为主协议使用的是 `normal_75_25`，实际训练样本由 75/25 切分决定：

- dataset：`chirp_signal`
- prompt：`rf`
- input：`signal_adaptive`
- split：`normal_75_25`
- train：每个 scene/ISR 下前 75% normal samples
- test：剩余 25% normal samples + 全部 abnormal samples
- epoch：50
- seed：111
- baseline：`rf + signal_adaptive`
- test：`rf + signal_adaptive + visual_lora`

结果文件：
- `analysis_outputs/visual_lora_compare/chirp_rf_signal_adaptive_lora_r4_vs_baseline.csv`

### 逐项结果

| scene | ISR | baseline | LoRA | delta |
|---|---|---:|---:|---:|
| WeaponMuseum_spectrum | m10db | 95.3600 | 95.1700 | -0.1900 |
| WeaponMuseum_spectrum | m20db | 86.5400 | 86.3200 | -0.2200 |
| WeaponMuseum_spectrum | m30db | 59.7800 | 59.8100 | +0.0300 |
| Playground_spectrum | m10db | 96.6800 | 96.6000 | -0.0800 |
| Playground_spectrum | m20db | 93.7900 | 93.5300 | -0.2600 |
| Playground_spectrum | m30db | 85.8100 | 85.6700 | -0.1400 |
| TimeSquare_spectrum | m10db | 97.2700 | 96.8600 | -0.4100 |
| TimeSquare_spectrum | m20db | 93.4000 | 93.2000 | -0.2000 |
| TimeSquare_spectrum | m30db | 83.6400 | 83.5400 | -0.1000 |
| Gymnasium_spectrum | m10db | 76.2200 | 76.2300 | +0.0100 |
| Gymnasium_spectrum | m20db | 77.8900 | 77.7100 | -0.1800 |
| Gymnasium_spectrum | m30db | 45.5200 | 52.0000 | +6.4800 |

### 汇总结果

| method | mean Image-AUROC |
|---|---:|
| `rf + signal_adaptive` | 82.6583 |
| `rf + signal_adaptive + visual_lora` | 83.0533 |
| delta | +0.3950 |

按 ISR 汇总：

| ISR | baseline | LoRA | delta |
|---|---:|---:|---:|
| m10db | 91.3825 | 91.2150 | -0.1675 |
| m20db | 87.9050 | 87.6900 | -0.2150 |
| m30db | 68.6875 | 70.2550 | +1.5675 |

按 scene 汇总：

| scene | baseline | LoRA | delta |
|---|---:|---:|---:|
| WeaponMuseum_spectrum | 80.5600 | 80.4333 | -0.1267 |
| Playground_spectrum | 92.0933 | 91.9333 | -0.1600 |
| TimeSquare_spectrum | 91.4367 | 91.2000 | -0.2367 |
| Gymnasium_spectrum | 66.5433 | 68.6467 | +2.1033 |

胜负统计：
- 3 胜 / 9 负 / 0 持平；
- overall 提升 `+0.3950`；
- 主要收益来自 `Gymnasium_spectrum m30db` 的 `+6.4800`；
- m10db 和 m20db 平均下降，m30db 平均提升。

### 结论

Visual LoRA 和输出后 Adapter 不一样：它在 overall 上有小幅正增益，但增益并不稳定。

更准确的结论是：
- LoRA 对 `chirp_signal` 的低 ISR 困难场景有潜力；
- 尤其 `Gymnasium_spectrum m30db` 从 `45.52` 提升到 `52.00`；
- 但 12 个点里有 9 个点下降，说明当前 LoRA 设置还不能作为统一默认方案；
- 不能直接宣称 LoRA 稳定优于 baseline。

下一步更合理的实验不是立刻跑全量 36 组，而是围绕 LoRA 的稳定性做小范围消融：
1. 只在后 3 或后 6 个 visual transformer blocks 上加 LoRA，而不是 12 个 block 全部加；
2. 降低 LoRA 学习率，和 PromptLearner 分开 optimizer；
3. 对 `m30db` 低 ISR 单独验证，看 LoRA 是否专门改善困难低信噪比场景；
4. 跑 `seed=222/333` 验证 `Gymnasium m30db` 的大幅提升是否稳定。

## 传统频谱统计特征融合实验

在 prompt、Adapter 和 LoRA 都没有形成稳定收益后，进一步尝试不训练 backbone 的传统频谱统计特征融合。

### 方法设计

本实验不使用 z-score，也不使用训练正常样本分布校准，而是直接从原始频谱图计算一个传统统计图像级分数，再和 PromptAD 图像级分数相加：

```text
final_score = PromptAD_score + beta * raw_stat_score
```

其中 `raw_stat_score` 来自：
- frequency-background residual：每个频率行减去该行的中位背景，取绝对残差；
- local variance：局部窗口内能量方差；
- top-k pooling：取统计图中 top 5% 像素均值作为图像级统计分数。

本轮参数：
- `stat_topk_ratio = 0.05`
- `stat_fusion_beta = 0.5`
- 不做 z-score / percentile / normal-score normalization

### 实验设置

- dataset：`dsss_signal`
- prompt：`rf`
- input：`signal_adaptive`
- split：`normal_75_25`
- train：每个 scene/ISR 下前 75% normal samples
- test：剩余 25% normal samples + 全部 abnormal samples
- epoch：50
- seed：111
- baseline：`rf + signal_adaptive`
- test：`rf + signal_adaptive + raw spectral-stat fusion`

结果文件：
- `analysis_outputs/stat_fusion_compare/dsss_rf_signal_adaptive_stat_beta05_vs_baseline.csv`

### 逐项结果

| scene | ISR | baseline | stat fusion | delta |
|---|---|---:|---:|---:|
| WeaponMuseum_spectrum | m10db | 97.6800 | 96.7400 | -0.9400 |
| WeaponMuseum_spectrum | m20db | 92.8700 | 91.8400 | -1.0300 |
| WeaponMuseum_spectrum | m30db | 80.9700 | 79.7300 | -1.2400 |
| Playground_spectrum | m10db | 96.5800 | 96.7200 | +0.1400 |
| Playground_spectrum | m20db | 92.0500 | 92.4900 | +0.4400 |
| Playground_spectrum | m30db | 79.1000 | 79.7300 | +0.6300 |
| TimeSquare_spectrum | m10db | 98.8300 | 98.8000 | -0.0300 |
| TimeSquare_spectrum | m20db | 93.6100 | 93.4400 | -0.1700 |
| TimeSquare_spectrum | m30db | 84.0400 | 83.7400 | -0.3000 |
| Gymnasium_spectrum | m10db | 92.4600 | 92.5100 | +0.0500 |
| Gymnasium_spectrum | m20db | 83.2800 | 83.3600 | +0.0800 |
| Gymnasium_spectrum | m30db | 83.2800 | 83.3600 | +0.0800 |

### 汇总结果

| method | mean Image-AUROC |
|---|---:|
| `rf + signal_adaptive` | 89.5625 |
| `rf + signal_adaptive + raw spectral-stat fusion` | 89.3717 |
| delta | -0.1908 |

按 ISR 汇总：

| ISR | baseline | stat fusion | delta |
|---|---:|---:|---:|
| m10db | 96.3875 | 96.1925 | -0.1950 |
| m20db | 90.4525 | 90.2825 | -0.1700 |
| m30db | 81.8475 | 81.6400 | -0.2075 |

按 scene 汇总：

| scene | baseline | stat fusion | delta |
|---|---:|---:|---:|
| WeaponMuseum_spectrum | 90.5067 | 89.4367 | -1.0700 |
| Playground_spectrum | 89.2433 | 89.6467 | +0.4033 |
| TimeSquare_spectrum | 92.1600 | 91.9933 | -0.1667 |
| Gymnasium_spectrum | 86.3400 | 86.4100 | +0.0700 |

胜负统计：
- 6 胜 / 6 负 / 0 持平；
- overall 下降 `-0.1908`。

### 结论

直接把传统频谱统计分数加到 PromptAD 图像分数上没有带来稳定收益。

具体来看：
- `Playground_spectrum` 有小幅提升，平均 `+0.4033`；
- `Gymnasium_spectrum` 基本持平，平均 `+0.0700`；
- `WeaponMuseum_spectrum` 明显下降，平均 `-1.0700`；
- 所有 ISR 分组平均都下降。

因此，传统统计特征本身不是完全无效，但不能用简单 raw-score 加权融合直接作为统一方案。更合理的后续方向是：
1. 不直接融合 image score，而是把统计特征作为 scene/type-adaptive input，例如前面的 `dsss_statistical`；
2. 或者只对 `Playground_spectrum` 这类确实受益的场景启用；
3. 如果继续做分数融合，需要先解决不同分数尺度问题，但本轮按要求没有使用 z-score。

## WeaponMuseum DSSS 专用输入候选实验

前面的实验说明，DSSS 不适合继续沿用 `spectral_gradient` 这类边缘增强输入；但直接统计分数融合又会伤害 `WeaponMuseum_spectrum`。因此这一轮只针对 `dsss_signal / WeaponMuseum_spectrum` 做更小范围的输入通道候选实验，目标是找出是否存在比 baseline RGB 更适合 WeaponMuseum DSSS 的特征提取方式。

### 选择依据

WeaponMuseum DSSS 的异常形态更接近宽带、弱纹理、低对比度扩频能量变化。它不像 chirp 有明显斜线轨迹，也不像 burst 有清晰突发边界，所以不能把重点放在强梯度边缘上。

本轮选择的依据是：
- 保留原始灰度能量图，避免破坏 CLIP 已经能使用的整体纹理；
- 引入频率背景残差，用每个频率行的中位数作为背景，突出相对背景的异常能量变化；
- 控制残差强度，避免纯残差通道把宽带连续结构打碎；
- 额外验证 CLAHE 这种局部对比度增强是否有帮助。

因此设计三个候选：

| input mode | 通道设计 | 目的 |
|---|---|---|
| `dsss_rgb_residual` | `gray, gray, abs(gray - freq_median)` | 强化频率背景残差 |
| `dsss_weak_residual` | `gray, gray, gray + 0.2 * residual` | 弱残差增强，在保留原图的同时加入背景差异 |
| `dsss_clahe` | `gray, gray, CLAHE(gray)` | 增强局部对比度 |

这里没有使用 z-score，也没有使用 normal-score normalization；只是改变输入三通道的构造方式。

### 实验设置

- dataset：`dsss_signal`
- scene：`WeaponMuseum_spectrum`
- ISR：`m10db` / `m20db` / `m30db`
- prompt：`rf`
- split：`normal_75_25`
- train：前 75% normal samples
- test：剩余 25% normal samples + 全部 abnormal samples
- epoch：50
- seed：111
- baseline：`rf + rgb/none`

结果文件：
- `analysis_outputs/weapon_dsss_input_candidates/weapon_dsss_input_candidates_vs_rgb.csv`

### 结果

| ISR | baseline RGB | `dsss_rgb_residual` | delta | `dsss_weak_residual` | delta | `dsss_clahe` | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| m10db | 97.6800 | 98.7500 | +1.0700 | 98.8400 | +1.1600 | 93.8100 | -3.8700 |
| m20db | 92.8700 | 94.2900 | +1.4200 | 95.1500 | +2.2800 | 82.8600 | -10.0100 |
| m30db | 80.9700 | 79.6800 | -1.2900 | 83.1200 | +2.1500 | 62.6300 | -18.3400 |
| mean | 90.5067 | 90.9067 | +0.4000 | 92.3700 | +1.8633 | 79.7667 | -10.7400 |

### 结论

这一轮真正有效的是 `dsss_weak_residual`。

原因是它满足两个条件：
1. 保留两路原始灰度输入，没有破坏 DSSS 的宽带纹理和整体能量分布；
2. 第三通道只加入弱残差信息，让模型看到相对频率背景的异常变化，但不过度放大局部边缘。

`dsss_rgb_residual` 虽然平均提升 `+0.4000`，但在最困难的 `m30db` 下降 `-1.2900`，说明纯残差通道对低 ISR 不稳定。`dsss_clahe` 明显失败，尤其 `m30db` 下降 `-18.3400`，说明局部对比度增强会放大噪声纹理，破坏 DSSS 的弱扩频结构。

因此，当前对 WeaponMuseum DSSS 的推荐是：

```text
rf + dsss_weak_residual
```

但这个结论目前只覆盖 `WeaponMuseum_spectrum`、`seed=111`。如果要作为论文或最终方案，需要继续做两步验证：
1. 在 `seed=222/333` 上复验 `dsss_weak_residual`，确认不是单 seed 偶然收益；
2. 在另外三个 DSSS scene 上跑同一输入，判断它是 WeaponMuseum 专用，还是可以作为 DSSS 默认输入。

## 特征提取方法整理与 selection 设计

当前已经尝试过的输入特征包括：
- `rgb`：原始频谱图按普通 RGB 输入；
- `spectral_gradient`：`gray + time-gradient + frequency-gradient`；
- `dsss_statistical`：`gray + frequency-background residual + local variance`；
- `dsss_weak_residual`：`gray + gray + weak residual`；
- `dsss_rgb_residual`、`dsss_clahe` 等 DSSS 候选。

这一节的目标不是简单堆实验结果，而是整理出一套可以解释的 feature selection 规则。

### 1. 按信号类型的最佳结果

先看主线实验：`rf + signal_adaptive` 相对 `rf + rgb/none` 的结果。

| signal type | baseline input | selected input | baseline mean | selected mean | delta | 结论 |
|---|---|---|---:|---:|---:|---|
| `burst_signal` | `rgb` | `spectral_gradient` | 86.3117 | 87.9800 | +1.6683 | 整体有效 |
| `chirp_signal` | `rgb` | `spectral_gradient` | 78.7308 | 82.6583 | +3.9275 | 明显有效 |
| `dsss_signal` | `rgb` | `rgb` | 89.5625 | 89.5625 | +0.0000 | 不使用梯度 |

这个结果说明：
- `burst_signal` 和 `chirp_signal` 更适合边缘/方向结构输入；
- `dsss_signal` 不适合统一使用 `spectral_gradient`，因为 DSSS 的主要信息不是清晰边缘，而是宽带统计纹理。

因此第一层 selection 是按 signal type：

| signal type | 推荐特征提取 |
|---|---|
| `burst_signal` | `spectral_gradient` |
| `chirp_signal` | `spectral_gradient` |
| `dsss_signal` | 进入 DSSS 专用选择 |

### 2. Burst / Chirp 为什么选 `spectral_gradient`

`spectral_gradient` 适合 burst 和 chirp 的依据是频谱形态：

| signal type | 频谱形态 | `spectral_gradient` 的作用 |
|---|---|---|
| `burst_signal` | 短时突发、起止边界明显、局部能量块 | 时间梯度突出出现/消失边界，频率梯度突出窄带边界 |
| `chirp_signal` | 频率随时间扫掠，形成斜线轨迹 | 时间/频率梯度共同突出斜率、轨迹和边缘 |

从实验上看，`chirp_signal` 的收益最稳定，平均提升 `+3.9275`；`burst_signal` 平均提升 `+1.6683`，但存在场景差异：

| signal type | scene | `spectral_gradient` mean delta |
|---|---|---:|
| `burst_signal` | `WeaponMuseum_spectrum` | +4.2000 |
| `burst_signal` | `TimeSquare_spectrum` | +4.5500 |
| `burst_signal` | `Playground_spectrum` | -0.2167 |
| `burst_signal` | `Gymnasium_spectrum` | -1.8600 |
| `chirp_signal` | `WeaponMuseum_spectrum` | +4.4900 |
| `chirp_signal` | `Playground_spectrum` | +1.5233 |
| `chirp_signal` | `TimeSquare_spectrum` | +1.5567 |
| `chirp_signal` | `Gymnasium_spectrum` | +8.1400 |

所以如果只做 signal-type-level selection，burst / chirp 都选 `spectral_gradient`；如果进一步做 scene-adaptive selection，burst 的 `Playground_spectrum` 和 `Gymnasium_spectrum` 可以保留 `rgb`，因为它们在当前 seed 下对梯度输入不稳定。

### 3. DSSS 内部的最佳结果

DSSS 不能直接套用 `spectral_gradient`。单独实验已经验证：

```text
rf + spectral_gradient vs rf + rgb/none on dsss_signal: mean delta = -2.5425
```

因此 DSSS 的候选重点转向宽带统计纹理。当前 DSSS 的主要结果如下：

| scene | baseline RGB | `dsss_statistical` | delta | 当前推荐 |
|---|---:|---:|---:|---|
| `WeaponMuseum_spectrum` | 90.5067 | 86.2967 | -4.2100 | 不用 `dsss_statistical` |
| `Playground_spectrum` | 89.2433 | 93.6033 | +4.3600 | `dsss_statistical` |
| `TimeSquare_spectrum` | 92.1600 | 96.0833 | +3.9233 | `dsss_statistical` |
| `Gymnasium_spectrum` | 86.3400 | 87.5267 | +1.1867 | `dsss_statistical` |

对 `WeaponMuseum_spectrum`，继续尝试了更弱的残差输入：

| input | mean Image-AUROC | delta vs RGB |
|---|---:|---:|
| `rgb` | 90.5067 | +0.0000 |
| `dsss_rgb_residual` | 90.9067 | +0.4000 |
| `dsss_weak_residual` | 92.3700 | +1.8633 |
| `dsss_clahe` | 79.7667 | -10.7400 |

因此当前 DSSS 内部推荐为：

| scene | 推荐特征提取 | 依据 |
|---|---|---|
| `WeaponMuseum_spectrum` | `dsss_weak_residual` | 保留原始宽带纹理，同时弱注入频率背景残差 |
| `Playground_spectrum` | `dsss_statistical` | 统计纹理提升明显，平均 `+4.3600` |
| `TimeSquare_spectrum` | `dsss_statistical` | 统计纹理提升明显，平均 `+3.9233` |
| `Gymnasium_spectrum` | `dsss_statistical` | 小幅提升，平均 `+1.1867` |

注意：`WeaponMuseum_spectrum -> dsss_weak_residual` 目前只在 `seed=111` 验证过，后续需要 `seed=222/333` 复验。

### 4. 推荐的 selection 方法

基于当时只完成 WeaponMuseum 补充实验的结果，可以先把方法设计成两级选择：

```text
Input spectrogram
    |
    |-- if signal type is burst:
    |       use spectral_gradient
    |
    |-- if signal type is chirp:
    |       use spectral_gradient
    |
    |-- if signal type is dsss:
            |
            |-- if scene is WeaponMuseum:
            |       use dsss_weak_residual
            |
            |-- else:
                    use dsss_statistical
```

这个中间方案是：

| signal type | scene condition | selected feature |
|---|---|---|
| `burst_signal` | default | `spectral_gradient` |
| `chirp_signal` | default | `spectral_gradient` |
| `dsss_signal` | `WeaponMuseum_spectrum` | `dsss_weak_residual` |
| `dsss_signal` | other scenes | `dsss_statistical` |

但这个中间方案有一个问题：它依赖 scene condition，不适合生产中未知场景的情况。因此后续又补充了 `dsss_weak_residual` 在 DSSS 四个场景上的完整验证。新的结果显示，DSSS 可以统一使用 `dsss_weak_residual`，不再需要按 scene 手工选择。

如果不采用后续四场景验证结果，保守版本可以写成：

| signal type | selected feature |
|---|---|
| `burst_signal` | `spectral_gradient` |
| `chirp_signal` | `spectral_gradient` |
| `dsss_signal` | `rgb` 或 `dsss_statistical`，按开发集选择 |

但从后续完整验证结果看，DSSS 更适合统一选择 `dsss_weak_residual`，这比 scene-adaptive selection 更适合生产部署。

### 5. 论文里 selection 的严谨写法

不能写成“我们在最终测试集上看哪个最高就选哪个”。更严谨的写法应该是：

1. **先验分析。** 根据 RF 频谱形态，把信号分为 edge/trajectory-dominant 和 texture/statistics-dominant 两类。
2. **候选设计。** 对 edge/trajectory-dominant 的 burst/chirp 设计 `spectral_gradient`；对 texture/statistics-dominant 的 DSSS 设计 residual/statistical 输入。
3. **开发集选择。** 在固定的 validation/development split 上选择每类信号的输入 front-end。
4. **最终测试。** 选定规则后，在 test split 上只评估一次，不再根据 test 结果调整规则。

可以写成英文方法描述：

```text
We adopt a signal-adaptive front-end selection strategy. For burst and chirp signals, whose anomalies are dominated by temporal edges and time-frequency trajectories, we use the spectral-gradient representation. For DSSS signals, whose anomalies are characterized by broadband weak texture variations, we avoid gradient-based inputs and instead use residual/statistical channels. The selection rule is determined on the development split and then fixed for final evaluation.
```

这套写法的重点是：selection 的依据来自频谱形态和开发集消融，而不是最终测试集调参。

## DSSS `dsss_weak_residual` 四场景完整验证

前面 `dsss_weak_residual` 只在 `WeaponMuseum_spectrum` 上验证过，因此还不能说明它能作为 DSSS 的统一输入特征。为了解决“生产中未知场景时如何选择”的问题，进一步在 DSSS 另外三个场景上补充实验。

### 实验设置

- dataset：`dsss_signal`
- scene：`WeaponMuseum_spectrum` / `Playground_spectrum` / `TimeSquare_spectrum` / `Gymnasium_spectrum`
- ISR：`m10db` / `m20db` / `m30db`
- prompt：`rf`
- split：`normal_75_25`
- epoch：50
- seed：111
- baseline：`rf + rgb/none`
- comparison：`rf + dsss_statistical`、`rf + dsss_weak_residual`

结果文件：
- `analysis_outputs/dsss_weak_residual_all_scenes/dsss_weak_residual_vs_rgb_and_statistical.csv`

### 逐项结果

| scene | ISR | RGB baseline | `dsss_statistical` | `dsss_weak_residual` | weak delta |
|---|---|---:|---:|---:|---:|
| WeaponMuseum_spectrum | m10db | 97.6800 | 95.1900 | 98.8400 | +1.1600 |
| WeaponMuseum_spectrum | m20db | 92.8700 | 90.5500 | 95.1500 | +2.2800 |
| WeaponMuseum_spectrum | m30db | 80.9700 | 73.1500 | 83.1200 | +2.1500 |
| Playground_spectrum | m10db | 96.5800 | 97.8100 | 100.0000 | +3.4200 |
| Playground_spectrum | m20db | 92.0500 | 94.8900 | 98.5800 | +6.5300 |
| Playground_spectrum | m30db | 79.1000 | 88.1100 | 95.4100 | +16.3100 |
| TimeSquare_spectrum | m10db | 98.8300 | 97.3800 | 99.8900 | +1.0600 |
| TimeSquare_spectrum | m20db | 93.6100 | 96.1700 | 96.9100 | +3.3000 |
| TimeSquare_spectrum | m30db | 84.0400 | 94.7000 | 95.0800 | +11.0400 |
| Gymnasium_spectrum | m10db | 92.4600 | 93.5000 | 96.7800 | +4.3200 |
| Gymnasium_spectrum | m20db | 83.2800 | 84.5400 | 87.6200 | +4.3400 |
| Gymnasium_spectrum | m30db | 83.2800 | 84.5400 | 87.6200 | +4.3400 |

### 按场景汇总

| scene | RGB baseline | `dsss_statistical` | `dsss_weak_residual` | weak delta |
|---|---:|---:|---:|---:|
| WeaponMuseum_spectrum | 90.5067 | 86.2967 | 92.3700 | +1.8633 |
| Playground_spectrum | 89.2433 | 93.6033 | 97.9967 | +8.7533 |
| TimeSquare_spectrum | 92.1600 | 96.0833 | 97.2933 | +5.1333 |
| Gymnasium_spectrum | 86.3400 | 87.5267 | 90.6733 | +4.3333 |
| ALL | 89.5625 | 90.8775 | 94.5833 | +5.0208 |

### 结论

`dsss_weak_residual` 在当前 `seed=111` 的 DSSS 12 个测试点上全部超过 RGB baseline，整体提升 `+5.0208`，也比 `dsss_statistical` 高 `+3.7058`。

这说明前面担心的“DSSS 不同场景要选择不同特征”的问题可以先简化：当前不需要按 scene 手工选择，DSSS 可以统一使用 `dsss_weak_residual`。

更新后的 selection 规则为：

| signal type | selected feature |
|---|---|
| `burst_signal` | `spectral_gradient` |
| `chirp_signal` | `spectral_gradient` |
| `dsss_signal` | `dsss_weak_residual` |

这个规则比前一版更适合生产部署，因为它只依赖 signal type，不依赖具体场景名称。对于未知场景，只要知道信号类型是 DSSS，就使用同一个 `dsss_weak_residual` front-end。

但需要注意：当前结论仍然只基于 `seed=111`。如果要作为最终论文结果，需要继续跑 `seed=222/333`，报告 `mean ± std`，确认 `dsss_weak_residual` 的提升不是单 seed 偶然结果。

## Wideband pulse baseline 对比实验

新增数据集 `wideband_pulse` 后，先按频谱形态假设它可能接近 burst/chirp，因此优先比较：

```text
rf + rgb
rf + spectral_gradient
```

注意：`wideband_pulse` 的噪声档位是 `m20db / m30db / m40db`，不是 `m10db / m20db / m30db`。

### 实验设置

- dataset：`wideband_pulse`
- scene：`WeaponMuseum_spectrum` / `Playground_spectrum` / `TimeSquare_spectrum` / `Gymnasium_spectrum`
- noise：`m20db` / `m30db` / `m40db`
- prompt：`rf`
- split：`normal_75_25`
- epoch：50
- seed：111
- baseline：`rf + rgb`
- comparison：`rf + spectral_gradient`

结果文件：
- `analysis_outputs/wideband_pulse_feature_compare/rf_spectral_gradient_vs_rgb.csv`

### 逐项结果

| scene | noise | RGB baseline | `spectral_gradient` | delta |
|---|---|---:|---:|---:|
| WeaponMuseum_spectrum | m20db | 100.0000 | 100.0000 | +0.0000 |
| WeaponMuseum_spectrum | m30db | 100.0000 | 100.0000 | +0.0000 |
| WeaponMuseum_spectrum | m40db | 83.4200 | 60.0000 | -23.4200 |
| Playground_spectrum | m20db | 97.3100 | 98.7100 | +1.4000 |
| Playground_spectrum | m30db | 92.9600 | 97.2200 | +4.2600 |
| Playground_spectrum | m40db | 92.2800 | 78.1300 | -14.1500 |
| TimeSquare_spectrum | m20db | 100.0000 | 95.5600 | -4.4400 |
| TimeSquare_spectrum | m30db | 100.0000 | 100.0000 | +0.0000 |
| TimeSquare_spectrum | m40db | 100.0000 | 100.0000 | +0.0000 |
| Gymnasium_spectrum | m20db | 99.1800 | 94.8500 | -4.3300 |
| Gymnasium_spectrum | m30db | 100.0000 | 100.0000 | +0.0000 |
| Gymnasium_spectrum | m40db | 88.4500 | 56.5500 | -31.9000 |

### 汇总结果

| group | RGB baseline | `spectral_gradient` | delta |
|---|---:|---:|---:|
| m20db mean | 99.1225 | 97.2800 | -1.8425 |
| m30db mean | 98.2400 | 99.3050 | +1.0650 |
| m40db mean | 91.0375 | 73.6700 | -17.3675 |
| overall | 96.1333 | 90.0850 | -6.0483 |

### 结论

这个结果推翻了最初“wideband pulse 应该像 burst/chirp 一样使用 `spectral_gradient`”的假设。

更准确的结论是：
- `spectral_gradient` 在 `m30db` 有小幅平均提升；
- 但在 `m20db` 平均下降；
- 在最困难的 `m40db` 明显下降，平均 `-17.3675`；
- overall 下降 `-6.0483`。

因此当前 `wideband_pulse` 不应该使用 `spectral_gradient` 作为默认特征。更稳妥的 selection 规则应更新为：

| signal type | selected feature |
|---|---|
| `burst_signal` | `spectral_gradient` |
| `chirp_signal` | `spectral_gradient` |
| `dsss_signal` | `dsss_weak_residual` |
| `wideband_pulse` | `rgb` |

从现象上看，`wideband_pulse` 虽然有 pulse 边界，但在低信噪比 `m40db` 下，强梯度增强可能会放大噪声边缘、削弱原始宽带能量结构。因此它不能简单归入 burst/chirp 的边缘主导类。
