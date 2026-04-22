# PromptAD 在频谱异常检测场景下的改进分析

结合仓库中现有实现（`@/mnt/data/wangbei/PromptAD/PromptAD/model.py:28-156`、`@/mnt/data/wangbei/PromptAD/PromptAD/ad_prompts.py:1-146`、`@/mnt/data/wangbei/PromptAD/datasets/dataset.py:1-63`、`@/mnt/data/wangbei/PromptAD/train_cls.py:83-234`），PromptAD 是一个在工业自然图像上训练的 CLIP + Prompt Tuning 框架。频谱图虽然表面上也是二维图像，但本质是 time–frequency 结构化信号，与 CLIP 的预训练分布差异极大。以下从 **域差、提示、视觉特征、训练、融合、架构** 六个角度给出改进建议。

---

## 1. 根本问题：CLIP 预训练域 vs. 频谱图域

- **背景**：`@/mnt/data/wangbei/PromptAD/PromptAD/model.py:198-208` 使用 `ViT-B-16-plus-240` + `laion400m_e32`，训练分布是自然图像。
- **归一化常数** `mean_train/std_train`（`@/mnt/data/wangbei/PromptAD/PromptAD/model.py:20-21`）是 ImageNet/CLIP 的 RGB 分布，频谱图（伪彩/灰度/dB 值映射）与之分布完全错位。

**改进建议**

- **更换或再对齐 backbone**：
  - 用频谱/时频图上自监督预训练的 ViT（MAE/DINOv2 在 spectrogram 上继续预训练）替换 CLIP 视觉塔；
  - 若要保留文本分支，使用 **SigLIP/EVA-CLIP** + 在频谱图 + 文本 caption 上做一段 LoRA/Adapter 对齐（冻结大部分，只训 adapter）。
- **替换归一化统计量**：对所选数据集估计 spectrogram 的 `mean/std`，而不是直接用 CLIP 统计量。
- **保留数值语义**：频谱中"强度 dB 值"本身承载信息，不要做 `BGR → RGB` 的三通道拷贝，推荐三通道分别承载：`[dB 归一化, time-gradient, frequency-gradient]`，让模型显式感知时/频方向。

---

## 2. 输入分辨率与 Patch 破坏频谱结构

- `@/mnt/data/wangbei/PromptAD/PromptAD/model.py:186-191` 会把频谱图 Resize 到 `240×240`，`@/mnt/data/wangbei/PromptAD/datasets/dataset.py:55-58` 又先 resize 到 1024 再交给模型。
- 频谱异常（chirp 斜线、burst 窄条、DSSS 宽带）都是**极细的线状/条状结构**，双三次插值 + ViT 16×16 patch 化会直接抹掉斜率 / 宽度这类关键特征。

**改进建议**

- **抗混叠的各向异性 resize**：时间轴和频率轴分别保持原 sampling，避免等比缩放把窄条压没。
- **更小 patch 或窗口化推理**：改用 ViT-B/8 或使用 WinCLIP 式的滑窗 + 多尺度 patch，能更好保留窄带/短时结构。
- **保留时/频坐标先验**：在 patch embedding 上拼接显式的 `(t, f)` 位置编码（而不是通用 2D sincos），让模型知道"纵轴是频率、横轴是时间"。

---

## 3. 文本 Prompt 的重构

目前 `@/mnt/data/wangbei/PromptAD/PromptAD/ad_prompts.py:64-144` 已经为 chirp/burst/dsss/deceptive 写了视觉化描述，但还存在几个问题：

1. **classname 无语义**：场景名 `BinBo / CaoChang / ShiJianGuangChang / TiYuGuan`（拼音）进入 CLIP 文本编码器等价于随机 token，`abnormal_prompt_prefix + classname` 的语义一半作废（`@/mnt/data/wangbei/PromptAD/PromptAD/model.py:39-63`）。
2. **提示多样性差**：不同场景全用同一套 "interference/injected" 描述，失去类间区分。
3. **缺物理量描述**：没有"斜率 (chirp rate)""带宽""持续时间""信噪比"等 domain 术语。

**改进建议**

- **把 classname 替换为可理解的语义锚点**，例如：
  - `BinBo → "wideband urban RF scene"`
  - `CaoChang → "open field low-noise RF scene"`
  - 在 `class_mapping` 里加英文描述，而不是直接丢拼音。
- **按信号类型分层 prompt**：
  - 结构级：`"spectrogram with a narrow vertical streak"`、`"spectrogram with a diagonal slope from low to high frequency"`；
  - 物理级：`"RF signal with chirp rate"`、`"short-duration burst in narrow band"`；
  - 对比级（triplet 用）：把正常 prompt 也写丰富，例如 `"clean background noise floor"`、`"no man-made signal in band"`，而不是只靠可学习 `N ... N`。
- **引入 learnable scene-conditioned prompt**：当前 `n_ctx=4`, `n_pro=3`（`@/mnt/data/wangbei/PromptAD/train_cls.py:312-315`），可把 `normal_ctx` 拆为"全局共享 ctx + 每场景独立 ctx"，避免 4 个场景强行共享同一组上下文。

---

## 4. Visual Feature Gallery 的调整

`@/mnt/data/wangbei/PromptAD/PromptAD/model.py:290-348` 用两层 patch token 的最近邻距离做 visual score（PatchCore 思想）。在频谱场景下：

- **问题 1**：正常样本里也包含随机的背景噪声斑点，patch 级最近邻对"新出现的斑点"非常敏感 → 大量假阳性。
- **问题 2**：K-shot=1 时 gallery 只有一个时间片段，覆盖率低，正常 burst/间歇信号容易被误判。

**改进建议**

- **Row-wise / Column-wise gallery**：频谱中同一行（固定频率）和同一列（固定时刻）本来就具有时/频对称性，把 gallery 拆成 per-row / per-column 统计（均值+方差），对异常的判定更贴近物理意义。
- **Patch coreset + 频率 bin 条件化**：将 patch 特征按所属频率 bin 分组，比较时只与同频率 bin 的正常 patch 做最近邻，避免把"不同频段正常 patch"当成候选。
- **使用 rank/percentile 距离** 代替 `min`（`@/mnt/data/wangbei/PromptAD/PromptAD/model.py:340-344`），对 outlier patch 更鲁棒。

---

## 5. 训练损失与融合策略

- 当前 loss（`@/mnt/data/wangbei/PromptAD/train_cls.py:162-169`）：`V2T CE + Triplet + handle/learned 一致性`，没有 **利用频谱已知的 synthetic 异常**。
- 最终 score 用 harmonic mean：
  ```@/mnt/data/wangbei/PromptAD/PromptAD/model.py:358
              anomaly_map = 1. / (1. / textual_anomaly_map + 1. / visual_anomaly_map)
  ```
  在频谱域下 textual 分支常常不可靠（CLIP 文本先验 OOD），harmonic mean 会被较小的 textual score "拖死"。

**改进建议**

- **合成异常做自监督辅助**：频谱场景下合成异常几乎免费：
  - 在正常样本上 **贴一条 chirp 斜线 / 窄带 burst / DSSS 宽带条** → 得到 `pseudo-abnormal` 配对；
  - 加一个监督项 `BCE(score, 合成 mask)`，pixel-level 监督直接让 anomaly map 对齐物理位置，比纯 triplet 强得多。
- **MixUp in spectrum domain**：随机把不同频段或不同场景的正常 patch 拼接，作为 hard negative 提升判别力。
- **可学习的融合权重**：把 `textual`/`visual` 融合权重做成 per-class 可学习 scalar（甚至 per-pixel gate），避免 harmonic mean 被某一支主导：
  ```
  anomaly = sigmoid(alpha) * textual + (1-sigmoid(alpha)) * visual
  ```
- **用 ground-truth mask（训练时不可用，但测试时有）重新校准**：测试数据已给了 groundtruth（`@/mnt/data/wangbei/PromptAD/datasets/burst_signal.py:120-124`），可以用极少量异常 sample 做 unsupervised 阈值自适应（PRO/F1 最优点）。

---

## 6. 架构级增强（针对频谱）

- **Axial attention / FFT-enhanced branch**：在视觉塔上并一条 frequency-axis 的 1D attention 分支，让模型显式建模"跨频率 / 跨时间"的长程相关性。
- **Physics-guided head**：在输出前加一个小的分类头，显式预测 `{ chirp, burst, dsss, deceptive, clean }`，用 prompt-level 标签做弱监督，可与 anomaly map 共享 backbone。
- **Cross-noise robustness**：现有设置对 `m10db/m20db/m30db` 是分别训的（`@/mnt/data/wangbei/PromptAD/train_cls.py:319`）。改为训练时随机加噪声退化作为增广，测试一个模型覆盖多 SNR，更贴近实战。
- **Cross-scene generalization**：目前 cross-site 是换 `train_category`（`@/mnt/data/wangbei/PromptAD/datasets/burst_signal.py:59-60`），但 prompt/gallery 还都绑定在训练场景上。可以把 visual gallery 设计成"场景无关的频率统计量"，让模型对未见场景更稳。

---

## 建议的优先级（性价比排序）

- **P0（最容易显著提升）**
  - 合成异常 + pixel-level 辅助监督（第 5 节）
  - 可学习融合权重替换 harmonic mean（第 5 节）
  - classname 语义化 + 场景条件 prompt（第 3 节）
- **P1**
  - Patch gallery 按频率 bin 条件化（第 4 节）
  - 三通道输入包含时/频梯度（第 1 节）
- **P2（改动较大）**
  - 在 spectrogram 上做 CLIP LoRA/MAE 对齐（第 1 节）
  - Axial/FFT 辅助分支（第 6 节）

---

如果需要，我可以先从 **P0 的合成异常辅助监督** 或 **可学习融合权重** 入手，直接在 `@/mnt/data/wangbei/PromptAD/train_cls.py` 和 `@/mnt/data/wangbei/PromptAD/PromptAD/model.py` 中给出具体的代码改动草案。你希望从哪一项开始？