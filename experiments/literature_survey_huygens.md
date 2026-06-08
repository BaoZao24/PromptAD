# 文献调研：频谱图/CLIP 异常检测可借鉴方法（Huygens）

> 调研日期：2026-06-02
> 目的：为 PromptAD 寻找可形成第二创新点的方向

---

**重要约束**
1. 当前主线不是 selection，不能假设测试前知道异常类型
2. 暂不把 wideband_pulse 纳入主研究对象
3. 实验优先使用 burst_signal / chirp_signal / dsss_signal
4. 协议优先使用 normal_75_25, seed 111
5. 当前主方案：`prompt_mode=rf`, `input_mode=morph_fusion_gray_residual_a01`, `cls_score_mode=text_only`

---

## 一、候选文献总表（10 篇）

| # | 论文 | 方法核心 | 需要原始信号？ | 仅需PNG？ | 接PromptAD？ | 适合层 |
|---|------|----------|--------------|----------|-------------|--------|
| 1 | **AnomalyCLIP** (ICLR'24) | object-agnostic prompt + DPAM multi-layer fusion | ❌ | ✅ | ⭐⭐⭐⭐⭐ | prompt层+score层 |
| 2 | **WinCLIP** (CVPR'23) | multi-scale window + normal reference memory + CPE | ❌ | ✅ | ⭐⭐⭐⭐ | score层+特征层 |
| 3 | **AF-CLIP** (ACM MM'25) | multi-scale spatial aggregation + attention adapter + patch alignment loss | ❌ | ✅ | ⭐⭐⭐⭐ | 特征层+score层 |
| 4 | **AA-CLIP** (CVPR'25) | anomaly-aware text anchor + patch-level feature alignment + disentanglement | ❌ | ✅ | ⭐⭐⭐⭐ | prompt层+score层 |
| 5 | **EfficientAD** (2024) | teacher-student distillation + hard feature loss + structural/logical dual-branch | ❌ | ✅ | ⭐⭐⭐ | 特征层+训练策略 |
| 6 | **PaDiM** (ICPR'20) | patch distribution modeling + Mahalanobis distance | ❌ | ✅ | ⭐⭐⭐⭐⭐ | score层 |
| 7 | **PatchCore** (CVPR'22) | patch feature memory bank + coreset sampling + nearest-neighbor scoring | ❌ | ✅ | ⭐⭐⭐⭐⭐ | score层+特征层 |
| 8 | **MGVCLIP** (2025) | multi-scale convolution adapter + guided context optimization + variance correlation loss | ❌ | ✅ | ⭐⭐⭐ | 特征层+prompt层 |
| 9 | **RareCLIP** (ICCV'25) | prototype memory bank + rarity-based patch aggregation + online scoring | ❌ | ✅ | ⭐⭐⭐⭐ | score层 |
| 10 | **Training-Free TAD** (arXiv'24) | wavelet transform → image foundation model → training-free detection | ⚠️ need wavelet | ❌ | ⭐⭐ | 特征层（但需重做变换） |

### 筛选排除说明

| 被排除论文 | 原因 |
|-----------|------|
| Deep Domain-Adversarial Contrastive Network (Wu 2024) | 需要原始时域信号做 Wavelet Scattering |
| Incrementally GAN OCL (2024) | 必须有时序数据训练GAN |
| Unified Feature Learning Network (2024) | 需要CWT变换原始振动信号 |
| TorchSig YOLOv8 (2024) | 需要有原始IQ做STFT和bounding box标注 |
| Nuclear-Shaped Anomaly Detection (2024) | 遥感领域，不相关 |

---

## 二、3 篇重点论文详细分析

### 论文1：AnomalyCLIP — Object-agnostic Prompt Learning for Zero-shot Anomaly Detection (ICLR 2024)

**论文链接**：arxiv.org/abs/2310.18961 | **代码**: github.com/zqhang/AnomalyCLIP

#### 方法核心

```
输入图片 → CLIP Image Encoder ──→ multi-layer patch features (L3, L4)
                                        │
                                  DPAM: Detail-Preserving Anomaly Map
                                  (多层特征融合 + cosine similarity with text)
                                        │
                                        └──→ anomaly score map
           CLIP Text Encoder ← learnable object-agnostic prompts
           "normal object"  ← [V₁][V₂]...[V_E][object]
           "damaged object" ← [W₁][W₂]...[W_E][damaged][object]
```

关键设计：
1. **Object-agnostic prompt**：异常 prompt 中类名用 "object" 替代，让模型关注"是否异常"而非"是什么物体"
2. **DPAM (Detail-Preserving Anomaly Map)**：融合 CLIP 中间层 (L3, L4) 的 patch 特征，做 pixel-level 对齐，保留细节
3. **联合优化**：image-level cross-entropy + pixel-level focal loss + Dice loss

#### 和我们任务的关系

| 维度 | 评估 |
|------|------|
| 是否需要原始信号？ | ❌ 不需要，仅需PNG频谱图 |
| 是否需要异常训练样本？ | ❌ 不需要，仅需 normal + text 描述 |
| 是否需要知道异常类型？ | ❌ 不需要，object-agnostic |
| 能否接入 PromptAD？ | ⭐⭐⭐⭐⭐ 高度可接 |

**可以借鉴的点**：
- **Object-agnostic 思想 → RF 频谱版**：当前 prompt_mode=rf 的异常 prompt 会带上类名（如 "burst signal radio frequency spectrogram"），可增加一个 object-agnostic 的变体，让 prompt 说 "radio frequency spectrogram" 而不带具体信号类型 → 更好的跨信号泛化
- **DPAM 多层特征融合**：当前 PromptAD 只用 CLIP 最后一层 feature map 做 patch scoring，可以加入中间层 L3 的特征，用类似 DPAM 的方式做 multi-layer fusion → 增强对小尺度异常（burst 窄脉冲）的感知
- **Pixel-level alignment loss**：可以在训练中额外加入 patch 级别的 alignment loss（正常 patch 对齐 normal text，异常 patch 对齐 abnormal text），替代当前仅 image-level 的 v2t loss

#### 对我们的限制
- DPAM 需要 CLIP 中间层特征，当前代码只取了最后一层 → 需改 `encode_image` 返回多层特征
- Object-agnostic prompt 在 RF 场景下的语义是否合理？"abnormal radio frequency spectrogram" vs "abnormal burst signal radio frequency spectrogram"——需要实验验证
- Pixel-level alignment loss 需要 pseudo-label（哪些 patch 异常），AnomalyCLIP 通过训练中动态估计 → 引入额外复杂度

---

### 论文2：PatchCore — Towards Total Recall in Industrial Anomaly Detection (CVPR 2022)

**论文链接**：arxiv.org/abs/2206.06676 | **代码**: github.com/amazon-science/patchcore-inspection

#### 方法核心

```
训练阶段（仅 normal）:
  所有normal图片 → Wide-ResNet-101 提取 layer2+layer3 patch特征
                              ↓
                   局部邻域聚合 (avg pooling over 3×3 neighborhood)
                              ↓
              Coreset Subsampling (greedy max-min)
              → Nominal Patch Memory Bank M (存储最具代表性的patch特征)

测试阶段:
  测试图片 → 提取 patch特征 → 对每个patch p_test:
    s(p_test) = max_{m∈M} cosine_sim(p_test, m)  # 相似度越高=越正常
    anomaly_score(p_test) = 1 - s(p_test)

  图像级分数: max over all patches 或 top-k mean
  像素级分数: 直接使用 patch-level anomaly map
```

关键设计：
1. **Patch-level memory bank**：存储正常样本最具代表性的局部特征，而非全局
2. **Coreset subsampling**：避免存储海量 patch 特征，降低内存和计算开销
3. **Nearest-neighbor scoring**：简单的余弦相似度最近邻搜索，无训练

#### 和我们任务的关系

| 维度 | 评估 |
|------|------|
| 是否需要原始信号？ | ❌ 不需要 |
| 是否需要异常训练样本？ | ❌ 不需要 |
| 是否能接入 PromptAD？ | ⭐⭐⭐⭐⭐ |

**可以借鉴的点**：
- **Nominal Patch Feature Bank**：PromptAD 已有 `feature_gallery1/2`（全局特征），可以扩展为 patch-level gallery → 对每个 patch 计算与正常特征的最近邻距离，得到 visual patch anomaly score
- **这个 score 可与 text-based score 融合** → 当前 `cls_score_mode=text_only` 只用了文本分数，加入 visual patch score 可互补
- **Coreset 思想**：当前 PromptAD 存储全量训练特征，可改为 coreset 采样 → 降低正常样本数量敏感度

#### 对我们最重要的启示

当前 PromptAD 已经有 `feature_gallery1` 和 `feature_gallery2`（全局图像特征），可以做以下扩展：

```python
# 在 feature gallery building 阶段加：
model.build_patch_feature_gallery(feature_map1, feature_map2)
# 存储正常样本的 patch-level features

# 在 scoring 阶段：
visual_patch_score = 1 - max_cosine_similarity(test_patch, patch_gallery)
# 与 text_score 融合
final_score = alpha * text_score + beta * visual_patch_score
```

**修改量评估**：
- `model.py`: ~50行 新增 patch gallery building + scoring
- `train_cls.py`: ~20行 适配
- 训练速度：几乎不变（gallery building 在训练前一次完成）
- 测试速度：增加 patch 最近邻搜索 (~O(N*D) per image)

---

### 论文3：WinCLIP — Zero-/Few-Shot Anomaly Classification and Segmentation (CVPR 2023)

**论文链接**：arxiv.org/abs/2211.13445 | **代码**: github.com/zqhang/Accurate-WinCLIP-pytorch

#### 方法核心

```
1. 组合提示集成 (CPE):
   normal_prompt = mean_{state, template} ( text_encoder("template state [class]") )
   abnormal_prompt = mean_{state, template} ( text_encoder("template state [class]") )
   → 比单一 prompt 更鲁棒

2. 多尺度窗口特征 (WinCLIP):
   对每个尺度 (image, mid, small):
     窗口滑动 → 提取局部特征 → 与 text feature 计算 cosine similarity
   → 多尺度 anomaly maps 融合

3. WinCLIP+ (few-shot with normal samples):
   存储少量 normal 样本的特征作为 reference
   测试时计算与 reference 的 similarity
   → 与 text-based anomaly score 互补
```

关键设计：
1. **CPE (Compositional Prompt Ensemble)**：多状态 × 多模板的 prompt 集成 → 增强鲁棒性
2. **Multi-scale windows**：不同尺度窗口捕获不同大小的异常
3. **Visual reference correlation**：用少量 normal 样本特征校准

#### 和我们任务的关系

| 维度 | 评估 |
|------|------|
| 是否需要原始信号？ | ❌ 不需要 |
| 是否需要异常训练样本？ | ❌ 不需要 |
| 是否需要知道异常类型？ | ❌ 不需要 |
| 能否接入 PromptAD？ | ⭐⭐⭐⭐ |

**可以借鉴的点**：
- **CPE → 增强版 prompt ensemble**：当前 PromptAD 的 prompt 集是固定的几个模板取平均 → 可以用 CPE 的方式做更系统的 ensemble（多状态 x 多模板），提升异常 prompt 的表达力
- **Multi-scale window → 多尺度 anomaly map**：当前仅用单一分辨率的 patch feature，可以用不同窗口大小的 pooling/attention 来捕获多尺度异常 → 特别对 burst（窄脉冲）和 chirp（斜线）有帮助
- **Visual reference → score calibration**：用 normal 样本的统计量（均值、方差）对 text-based anomaly score 做 z-score normalization 或校准 → 类似 `normal_center` / `normal_mahalanobis` 模式但更轻量

#### 对我们的限制
- WinCLIP 的多尺度窗口在推理时需要多次前向 → 计算开销大
- 但我们可以用不同尺度的 pooling 而非实际裁剪，减少开销
- 论文未开源，但第三方复现可用

---

## 三、两个最适合当前代码的可实现创新点

### 创新点1：Patch-level Nominal Feature Gallery + Hybrid Score Fusion

**▸ 借鉴来源**：PatchCore + AnomalyCLIP

**▸ 为什么适合**：
- PromptAD 已有 `feature_gallery1/2`（全局特征）+ `feature_map1/2`（中间层特征）
- 现在 `cls_score_mode=text_only` 只用文本分数
- 加入 patch 级别的 visual score 可以互补，当前代码框架完全支持

**▸ 具体方案**：
```
阶段1（同现在）：训练 prompt → 文本分数
阶段2（新增）：用 normal 训练样本的中间层 patch 特征建 memory bank
              → 测试时对每个 patch 计算与 bank 中最近邻的 cosine similarity
              → visual_patch_score = 1 - max_cosine_sim (对每张图做 top-k max)
阶段3（新增）：hybrid_score = alpha * text_score + beta * visual_patch_score
              （alpha、beta 可以固定或用 normal 样本校准）
```

**▸ 需要改的代码**：
1. `PromptAD/model.py`:
   - 在 `build_image_feature_gallery()` 中增加 `build_patch_feature_gallery()` (~30行)
   - 新增 `visual_patch_score()` 方法 (~40行)
   - 新增 `hybrid_score()` 方法 (~20行)
2. `train_cls.py`:
   - 在 `fit()` 的 scoring 阶段调用 hybrid_score (~15行)
   - 新增 `--patch-bank-topk`, `--patch-bank-alpha`, `--patch-bank-beta` 参数 (~10行)

**▸ 预计实验成本**：
- 代码实现：~150行改动
- 实验时间：每 run 增加 ~2-5s（patch bank 相似度搜索），几乎不增加训练时间
- 需要跑的实验：3 种信号 × 3 seeds 的 A/B 对比 → 约 9 runs

**▸ 风险**：
- 中等：patch-level 的 cosine similarity 可能在频谱图背景下不够敏感（背景噪声的 patch 差异也很大）
- 缓解：use coreset sampling 只保留最具代表性的正常 patch；用 top-k 而非 single max 聚合

---

### 创新点2：Frequency-Band-Aware Multi-Scale Patch Scoring（频带感知多尺度打分）

**▸ 借鉴来源**：WinCLIP (multi-scale) + AnomalyCLIP (multi-layer DPAM) + 频谱特性

**▸ 为什么适合**：
- 频谱图有天然的纵轴（频率）和横轴（时间）语义
- 不同信号类型对频带的敏感度不同：burst 关注窄带局部、chirp 关注斜线轨迹、DSSS 关注宽带分布
- 当前方法对全图做统一的 patch scoring，不区分频带

**▸ 具体方案**：
```
对每张测试频谱图：
  1. Multi-scale patch feature extraction (3个尺度):
     - fine: 原分辨率 patch (捕获 burst 窄脉冲)
     - medium: 2×下采样 patch (捕获 chirp 断裂)
     - coarse: 4×下采样 patch (捕获 DSSS 大范围能量异常)

  2. Frequency-band division (3个频带):
     - low-freq band (0-33%): 背景能量分布
     - mid-freq band (33-66%): 信号主要活动区域
     - high-freq band (66-100%): 弱信号/噪声区域

  3. 每个(scale, band)对，计算与 normal gallery 的偏离程度
     → 3×3 = 9 个 anomaly score

  4. Adaptive aggregation:
     - 每个 band/scale 的权重由 normal 样本的方差自动决定
     - 方差越大的 band/scale → 可能是背景噪声 → 权重越低
     - 方差越小的 band/scale → 信号稳定 → 异常更可信 → 权重越高

  5. final_score = weighted_sum of all band/scale scores
```

**▸ 需要改的代码**：
1. `PromptAD/model.py`:
   - 新增 `MultiScaleBandScorer` 类 (~80行)
   - 在 `build_image_feature_gallery()` 中建不同频带的 gallery (~30行)
   - 在 `score_cached()` 中加入 multi-scale band scoring (~40行)
2. `train_cls.py`:
   - 新增参数 `--multi-scale-band` 开关 (~10行)

**▸ 预计实验成本**：
- 代码实现：~200行改动
- 推理时间：增加约 1.5-3×（多尺度 + 多频带）
- 需要跑的实验：3 种信号 × 3 seeds → 约 9 runs

**▸ 风险**：
- 较高：多尺度和频带划分的合理性需要验证；
  burst 和 chirp 可能在某个特定频带更敏感，但 DSSS 的宽带特征可能被频带划分"切碎"
- 缓解：先在小规模上做消融（只用多尺度 or 只用频带划分），确认有效再组合

---

## 四、其他可探索的方向（次要）

### 4.1 Prompt Ensemble via CPE (WinCLIP 启发)
- 当前异常 prompt 取所有 template 的 mean → 信息损失
- 改为：每个 template 单独计算异常分数 → 取 ensemble（max/mean/weighted） → 可能更鲁棒
- 改造成本低 (~30行)，风险低，但预期提升有限（当前 rf prompt 模板差异不够大）

### 4.2 Normal Calibration with Z-score (PaDiM 启发)
- 用 normal 训练样本的 anomaly score 分布做 z-score normalization
- 当前 `cls_score_mode` 已有 `normal_center` / `normal_mahalanobis`，可以在其基础上做 calibration
- 改造成本极低 (~15行)，可作为 baseline 对比

### 4.3 Contrastive Patch Alignment Loss (AF-CLIP 启发)
- 在训练中增加 patch 级别的 alignment loss
- 正常区域的 patch 应与 normal text features 更近
- 需要 pseudo-label → 用当前 anomaly map 在线生成
- 改造成本中等 (~60行)，训练时间增加 10-20%

---

## 五、行动建议

**优先级排序**：

| 优先级 | 方向 | 改造成本 | 预期增益 | 风险 | 备注 |
|--------|------|---------|---------|------|------|
| **P0** | 创新点1: Patch Feature Bank + Hybrid Scoring | 低 (~150行) | 中高 | 中 | 最稳妥，代码框架天然支持 |
| **P1** | 创新点2: Frequency-Band Multi-Scale Scoring | 中 (~200行) | 高 | 中高 | 频谱领域特色，能写故事 |
| P2 | Normal Calibration | 极低 (~15行) | 低 | 低 | 先做baseline验证 |
| P3 | Prompt CPE Ensemble | 低 (~30行) | 低 | 低 | 小改 |
| P4 | Patch Alignment Loss | 中 (~60行) | 中 | 中 | 需要pseudo-label质量验证 |

**建议路线**：
1. 先做 P0（Patch Feature Bank），这是最安全且一定能出结果的
2. 如果 P0 有效果，将其作为 baseline，再叠加 P1（Multi-Scale Band）
3. 用 P2（Normal Calibration）作为对照实验，验证 score calibration 的重要性

---

*End of survey*
