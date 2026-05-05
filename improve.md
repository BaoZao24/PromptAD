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

- **模块**：场景条件 prompt 模块
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
- 更容易看出到底是“场景词”有效，还是“异常词”有效。

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
2. **再给 4 个场景补上英文语义描述**
3. **然后把 prompt 改成“场景词 + 异常词”的组合形式**
4. **最后再尝试 AnomalyCLIP / CoCoOp 风格的可学习场景条件 prompt**

---

# 最后给出一个最适合当前项目的实施建议

如果你现在只想选 **一个特征提取改进 + 一个提示词改进** 来做，我建议这样搭配：

## 推荐组合 A（最稳）

### 特征提取
- **三通道输入：原图 + 时间梯度 + 频率梯度**

### 提示词
- **场景化结构 prompt：场景词 + 异常结构词**

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

我们计划将现有提示词从泛化的“异常频谱”描述，升级为“场景语义 + 异常结构”的联合 prompt 设计。具体而言，一方面针对 burst、chirp、dsss 分别构造结构化描述模板；另一方面为不同测试场景补充场景语义描述，使文本分支能够显式建模不同场景下的背景频谱差异，从而提升跨场景异常检测性能。

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
| Zhou et al., CoCoOp, 2022 | 场景条件 prompt 模块 | 支持根据场景动态调整 prompt |

---

如果你愿意，我下一步可以继续帮你做两件很具体的事：

1. **把这份文档再压缩成“老师汇报版”一页纸**
2. **直接按照这份方案去改 `ad_prompts.py`，先把“场景化 prompt”写出来**
