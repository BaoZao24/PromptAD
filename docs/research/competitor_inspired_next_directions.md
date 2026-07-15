# 竞争方法启发下的下一步探索方向

更新时间：2026-07-05

## 当前判断

我们现在的有效主线是：

```text
CLIP-ViT patch memory
+ frequency-balanced normal gallery
+ farthest50
+ top-k5
```

已经验证过：

- text 基本没有贡献。
- DINOv2-small 单独不够强。
- DINOv2 / FastRecon 都有一点互补，但直接扫融合权重不够干净。
- row-wise hard matching 已舍弃。

所以后续不要继续围绕 prompt/text 做主创新。

更应该围绕：

```text
few-shot normal reference 如何更好地搜索、对齐、生成、投影
```

## 1. VisionAD / Search is All You Need

关键词：

```text
training-free
nearest-neighbor search
support augmentation
query augmentation
multi-layer feature integration
class-aware visual memory bank
```

对我们的启发：

我们现在已经有 search/memory，但还缺两件东西：

```text
support augmentation:
  normal gallery 只由原图 patch 构成
  没有对 normal reference 做增强扩展

query augmentation:
  每张测试图只评估一次
  没有多视角/多尺度一致性
```

最值得试：

```text
RF support augmentation:
  对少量 normal 图做轻微强度扰动、时间平移、频率平移、轻微模糊
  生成更丰富的 normal patch gallery

RF query augmentation:
  对 test 图做 2-4 个轻微增强
  每个增强都算 anomaly score
  最后取平均或保守最大
```

为什么适合 RF：

RF 频谱图的正常形态会有轻微时间/频率漂移。正常 gallery 太少时，增强可以补正常变化，而不需要异常标注。

优先级：高。

## 2. FoundAD

关键词：

```text
foundation visual encoder
normal manifold projection
nonlinear projection operator
pure visual
few-shot
```

对我们的启发：

我们现在是直接比较：

```text
test patch vs normal patch memory
```

FoundAD 的思想更像：

```text
把 test feature 投影/拉回 normal manifold
看投影前后差异
```

这比最近邻更像一个“视觉正常化算子”。

最值得试的轻量版本：

```text
Normal Feature Projector

1. 用 few-shot normal patch features 训练一个很小的 denoising/projector
2. 输入 test patch feature
3. 输出 projected-normal feature
4. score = ||feature - projected_feature||
```

可以先不用神经网络，先做线性版本：

```text
PCA / low-rank normal subspace projection
score = reconstruction residual
```

这和 FastRecon 有点像，但更轻、更容易固定，不需要复杂权重融合。

优先级：高。

## 3. FOCT

关键词：

```text
foreground-aware
online conditional transport
structural distance
feature relevance
transductive memory
```

对我们的启发：

我们现在每个 test patch 独立找 normal patch，缺少“整体结构匹配”。

FOCT 类思想可以改成：

```text
RF Conditional Transport

query patch set
vs
normal patch set

不是逐 patch 最近邻，
而是计算两组 patch 分布之间的匹配代价。
```

RF 版本可以用：

```text
Sinkhorn / optimal transport distance
```

在局部窗口或频段 component 内计算。

为什么适合 RF：

异常不是单个 patch 变化，而是时间-频率结构整体变化。OT/transport 能更好描述“整段频谱结构对不上”。

优先级：中高。

## 4. One-to-Normal / OneNIP

关键词：

```text
one-to-normal transformation
normal image prompt
reconstruction/restoration
compare query with normal-restored query
```

对我们的启发：

现在我们比较的是：

```text
query vs support normal
```

One-to-Normal/OneNIP 的思想是：

```text
query -> normal-like query
query 与 normal-like query 的差异就是异常
```

RF 轻量版本：

```text
Spectrogram Normalizer

输入:
  query spectrogram
  few-shot normal gallery

输出:
  normal-restored spectrogram 或 feature map

score:
  residual(query, restored_query)
```

不用上 diffusion，先做简单版本：

```text
patch replacement:
  每个异常可疑 patch 用 normal gallery 中最相似的 normal patch 替换
  得到 pseudo-normal feature map
  residual = 原 feature - 替换后 feature
```

这其实是我们现在 NN memory 的升级：

```text
从“只算距离”
变成“显式构造正常化后的 query”
```

优先级：中。

## 5. UniVAD

关键词：

```text
component-aware patch matching
graph-enhanced component modeling
multi-level anomaly score
```

对我们的启发：

完整 UniVAD 太重，不适合直接搬。

但 component-aware 很适合 RF：

```text
RF components:
  frequency band
  time-frequency tile
  high-energy connected region
```

最值得试：

```text
Component-Aware RF Memory

global patch score:
  当前 CLIP-ViT memory score

component score:
  按频段/时间块计算 patch distribution distance

final:
  fixed rule combine global score and component score
```

优先级：中。

## 推荐实验顺序

### 实验 A：RF Support Augmentation Memory

最小改动：

```text
在 build normal gallery 时，对 support normal 做增强
增强只作用于 normal gallery，不改变 test
```

增强候选：

```text
brightness/contrast jitter
small Gaussian blur
time shift
frequency shift
small random crop-resize
```

评价：

```text
自测 RF 五类
公开 RF
对比当前 CLIP-ViT farthest50 + top-k5
```

### 实验 B：Query Test-Time Augmentation

最小改动：

```text
test image 做 4 个弱增强
每个增强单独算 score
最后 mean / max
```

这可以借鉴 VisionAD 的 query augmentation，但要控制增强别破坏 RF 语义。

### 实验 C：Normal Subspace Projection

最小改动：

```text
用 normal patch features 做 PCA
test patch 投影到 normal subspace
score = residual norm
```

这借鉴 FoundAD 的 normal manifold projection，比训练 projector 更简单。

### 实验 D：Pseudo-Normal Patch Replacement

最小改动：

```text
每个 test patch 找 top-k normal patches
用 normal patch 加权平均重构 pseudo-normal patch
score = residual
```

这借鉴 One-to-Normal/OneNIP，但不用生成模型。

### 实验 E：RF Component OT Distance

最小改动：

```text
把图切成频段/时间块
每块算 query patches 与 normal patches 的 Sinkhorn/OT distance
```

这借鉴 FOCT / UniVAD 的结构级匹配。

## 当前最推荐

优先做：

```text
A. support augmentation memory
B. query augmentation
C. normal subspace projection
```

理由：

- 都是纯视觉。
- 不需要异常样本。
- 不需要训练大模型。
- 不依赖 text。
- 和当前代码结构最接近。
- 能从竞争方法里讲出清晰来源。

