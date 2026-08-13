# CLIP 少样本异常检测改进方法调研（2023–2026）

> 生成日期：2026-06-23
> 目标：找到能在 image-level AUROC 上超过 PromptAD (CVPR 2024) 基线，且能融入现有 PromptAD 框架的方法。
> 当前基线 PromptAD 做法：可学习 prompt context + 类名 → normal/abnormal text embedding + visual feature gallery 打分。
> 调研流程：18 篇文献，77 条论断抽取，25 条对抗式核验（24 通过、1 否决）。

---

## 一、各候选方法快查

| 方法 | 年份/会场 | 协议 | 仓库 | 是否真比 PromptAD 强 | 核心改进 |
|---|---|---|---|---|---|
| **AnoPLe** | MM 2024 | few-shot, normal-only ✅ | [YoojLee/AnoPLe](https://github.com/YoojLee/AnoPLe) | ✅ MVTec 1-shot 94.5 vs PromptAD 91.2（多类协议，**注意协议差异**） | 双向 text↔visual prompt 交互 + scale-aware prefix |
| **APRIL-GAN** | CVPR'23 VAND 冠军 | zero/few-shot | [ByChelsea/VAND-APRIL-GAN](https://github.com/ByChelsea/VAND-APRIL-GAN) | 改进版 WinCLIP，无直接对照 | 线性投影把多层 CLIP visual feature 映射到 text joint space |
| **MVFA-AD** | CVPR'24 Highlight | few/zero-shot | [MediaBrain-SJTU/MVFA-AD](https://github.com/MediaBrain-SJTU/MVFA-AD) | +6.24/+7.33% AUC，但**仅医学域**，无 MVTec 数据 | CLIP visual encoder 各层插入轻量 residual adapter |
| **AdaptCLIP** | arXiv 2505 (2025) | zero/few-shot | [arxiv 2505.09926](https://arxiv.org/abs/2505.09926) | 未独立验证 | 3 个简单 adapter (visual/textual/prompt-query) + 交替训练 |
| **AnomalyCLIP** | ICLR'24 | **zero-shot**（需 auxiliary 标注） | [zqhang/AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) | 协议不可比 | object-agnostic prompt（丢掉类名） |
| **FiLo** | MM'24 | **zero-shot** | [CASIA-IVA-Lab/FiLo](https://github.com/CASIA-IVA-Lab/FiLo) | README 无 AUROC 表，**无法证实** | LLM 生成细粒度异常描述 + adaptive template |
| **AdaCLIP** | ECCV'24 | zero-shot, **需异常标注** | [caoyunkang/AdaCLIP](https://github.com/caoyunkang/AdaCLIP) | 训练协议与 PromptAD 冲突 | static + dynamic 混合 prompt |
| **SOWA** | arXiv 2407 | – | – | 无对照 | Soldier-Officer 窗口自注意力 + 分层 prompt |

> ⚠️ **核验提示**：FiLo / AdaCLIP / AnomalyCLIP 均为 zero-shot 且部分需异常标注，与 PromptAD "normal-only few-shot" 不在同一协议。它们的 SOTA 数字 **不能直接当作"超过 PromptAD"**——只有其**架构组件**可移植，训练流程不可。

---

## 二、Top-5 推荐集成（按可落地优先级排序）

### 🥇 1. AnoPLe — 双向 prompt 交互【中等难度，最值得做】
- **为何排第一**：唯一与 PromptAD 完全同协议（few-shot, normal-only），且有正式对照数字。
- **核心思想**：让 visual patch tokens 反过来 condition text prompt context，不是单向地用文本去查图像。
- **落地方式**：在 `PromptAD/model.py:28-156` 的 `PromptLearner` 里加一条 visual prompt 分支——用 image encoder 中间层 token 经过小 MLP 得到 visual prompt，拼到 text context 前面。
- **重要警告**：AnoPLe 的提升用的是 "multi-class few-shot" 协议（一套 prompt 同时覆盖多类），与 PromptAD repo 的 per-category 协议**不一定一致**。在 burst/chirp/dsss 上 A/B 验证再下结论。
- **代码**：`git clone https://github.com/YoojLee/AnoPLe`

### 🥈 2. APRIL-GAN 跨模态投影【低难度，最容易做】
- **为何排第二**：改动局部、风险低；对**非自然图像**（频谱图）更友好——学的是 visual→text 空间的线性映射，比纯文本 prompt 工程对类名语义依赖少。
- **核心思想**：选 CLIP 若干中间层 visual feature，过 `nn.Linear` 投影到 text embedding 空间，再算余弦相似度。
- **落地方式**：在 `PromptAD/model.py` 评分流程里加 `proj_layers = nn.ModuleList([nn.Linear(d_v, d_t) for _ in selected_layers])`；仅训练这些投影层（与 PromptLearner 一起）。
- **代码**：`git clone https://github.com/ByChelsea/VAND-APRIL-GAN`，参考 `model.py` 中的 `Linear_Projection` 与 `features_list`。

### 🥉 3. MVFA-AD 多层 residual adapter【低-中难度，对 RF 域可能最受益】
- **为何值得做**：你的频谱图属于"非自然图像"，与医学影像同样存在 CLIP 训练分布外问题；MVFA 在医学上 +7.33% AUC，**比工业图上做的方法更可能迁移到 RF**。
- **核心思想**：CLIP visual encoder 若干 transformer block 后插入 `Adapter(d) = Linear(d, d/r) → GELU → Linear(d/r, d) + skip`，CLIP 冻结，只训 adapter。
- **落地方式**：包装 `PromptAD/CLIPAD/` 中的 visual encoder，hook 进 adapter；与现有 PromptLearner 联合训练。
- **注意**：MVFA 仅在医学上验证，**未在 MVTec/VisA 跑过**，"+7.33% AUC" 不能当工业域承诺。
- **代码**：`git clone https://github.com/MediaBrain-SJTU/MVFA-AD`

### 4. AdaCLIP 的 dynamic prompt 设计（仅架构）【中等难度】
- **可借鉴**：static prompt（全局）+ dynamic prompt（per-image 生成）的混合设计，让 prompt 对每张频谱图自适应。
- **不可借鉴**：训练用异常标注，违反 PromptAD 协议；需把 dynamic prompt 的训练 loss 换成 **normal-only 自监督**（对比 / 重建）。
- **代码**：`git clone https://github.com/caoyunkang/AdaCLIP`，只抽其 `PromptMaker`。

### 5. FiLo 的 LLM 细粒度描述【极低难度，但收益未证实】
- **思路**：把 PromptAD 当前的硬编码 anomaly handle（`damaged {}`、`flawed {}`，见 `PromptAD/ad_prompts.py`）换成 LLM 离线生成的细粒度描述。
- **场景特化**：针对 RF 域，用 GPT 离线生成 `"spectrogram with deceptive interference burst at {scene}"` 这类描述加进 abnormal handle。
- **风险**：FiLo 仓库**无公开 AUROC 表**，量化收益未验证，建议作为零成本试验。
- **代码**：`git clone https://github.com/CASIA-IVA-Lab/FiLo`

---

## 三、对 RF/spectrogram 场景的关键判断

1. **没有任何调研到的方法在 RF/spectrogram 上跑过**——所有迁移性论述都是工程外推，不是已验证事实。
2. **架构选型偏好**（基于"CLIP text encoder 对 RF 几乎无先验"这一事实）：
   - ✅ **优先**：visual-side 改动（MVFA adapter / APRIL-GAN 投影）——不依赖类名语义。
   - ⚠️ **谨慎**：依赖类名的 prompt 设计（PromptAD、AnoPLe、FiLo）——类名 `BinBo / CaoChang` 在 CLIP text encoder 几乎是 OOV，提升可能打折。
   - 💡 **值得一试**：AnomalyCLIP 的 object-agnostic prompt 思路（丢掉类名）可能特别适合 RF——但它是 zero-shot，需改写训练循环。

---

## 四、建议的实验路径（按 ROI 排序）

| 优先级 | 实验 | 预计工作量 | 预期收益 |
|---|---|---|---|
| P0 | 移植 APRIL-GAN 的线性投影层到 PromptAD 评分流程 | 1–2 天 | 中（架构成熟、风险低） |
| P0 | 把 PromptAD 的 abnormal handle 替换成 LLM 生成的 RF 专属描述（FiLo 思路） | 半天 | 低-中（零成本） |
| P1 | 在 CLIPAD visual encoder 加 MVFA-style residual adapter | 3–5 天 | 中-高（对 RF 域最对路） |
| P2 | 实现 AnoPLe 的双向 prompt 交互 | 1 周 | 中-高（同协议唯一证据） |
| P3 | 实验 object-agnostic prompt（AnomalyCLIP 思路）下的 PromptAD 变体 | 1 周 | 不确定（探索性） |

---

## 五、调研盲区（未充分覆盖，可发起二轮）

- **InCTRL** (CVPR 2024 in-context residual)
- **AnomalyGPT**
- **GlassNet / SimpleNet** 等纯 normal-only 合成异常方法
- **score fusion / ensemble** 最新做法

特别是 **InCTRL** 和 **GlassNet**——这两个直接对应"用 normal 数据合成 hard negative"的想法。

---

## 六、引用来源（按调研角度归类）

**SOTA 基准对比**
- AnoPLe — https://arxiv.org/abs/2408.13516 · https://github.com/YoojLee/AnoPLe
- SOWA — https://arxiv.org/abs/2407.03634
- AdaptCLIP — https://arxiv.org/abs/2505.09926

**Object-agnostic / 可泛化 prompt**
- AnomalyCLIP — https://arxiv.org/abs/2310.18961 · https://github.com/zqhang/AnomalyCLIP
- FiLo — https://arxiv.org/abs/2404.13671 · https://github.com/CASIA-IVA-Lab/FiLo
- AdaCLIP — https://arxiv.org/abs/2407.15795 · https://github.com/caoyunkang/AdaCLIP

**Adapter / 跨模态对齐**
- MVFA-AD — https://arxiv.org/abs/2403.12570 · https://github.com/MediaBrain-SJTU/MVFA-AD
- APRIL-GAN — https://arxiv.org/abs/2305.17382 · https://github.com/ByChelsea/VAND-APRIL-GAN
- AnomalyGPT — https://github.com/CASIA-IVA-Lab/AnomalyGPT

**In-context / 残差 / 合成异常**
- InCTRL — https://arxiv.org/abs/2403.06495 · https://github.com/mala-lab/InCTRL
- SimpleNet — https://arxiv.org/abs/2303.15140 · https://github.com/DonaldRR/SimpleNet

---

## 七、调研注意事项与开放问题

**Caveats**
1. 只有 AnoPLe 有验证过的与 PromptAD 头对头数字，且用的是 multi-class 协议，**不是** PromptAD repo 的 per-category 协议。
2. FiLo / AdaCLIP / AnomalyCLIP 均为 zero-shot，其 MVTec/VisA 数字不能引用为"超过 PromptAD"。AdaCLIP 额外需要异常标注，违反 normal-only 约束；只能移植架构，不能移植训练。
3. MVFA-AD 的 +7.33% AUC 是医学域，工业域和 RF 域均未验证。
4. SOWA 的论断仅在架构层面被核实，AUROC 未独立验证。
5. **零** 调研到的工作在 RF/spectrogram 数据上做过基准——所有对 `/mnt/data/wangbei/PromptAD` 信号场景的迁移性表述都是工程推断。
6. InCTRL / AnomalyGPT / GlassNet / SimpleNet-port 未在本轮取得验证论断。

**开放问题**
1. 在 PromptAD 的 per-category 协议下，AnoPLe 在 MVTec/VisA k=1/2/4 是否仍优于 PromptAD？
2. MVFA-AD 的 residual adapter 架构移植到 MVTec/VisA few-shot，医学域的收益能否迁移到工业图？
3. InCTRL / AnomalyGPT / GlassNet / SimpleNet-port 在 image-AUROC 上与 PromptAD 的对比如何？
4. RF spectrogram 域下，object-agnostic prompt（AnomalyCLIP 风格，去掉类名）能否优于 PromptAD 的类名 prompt？
5. AdaCLIP 的 dynamic per-image prompt generator 能否改造成 normal-only 自监督训练，从而摆脱对异常标注的依赖？
