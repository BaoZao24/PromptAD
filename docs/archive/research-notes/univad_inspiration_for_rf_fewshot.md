# UniVAD 对当前 RF 少样本 AD 的启发

更新时间：2026-07-05

源码位置：

```text
references/UniVAD/
```

## 1. UniVAD 做了什么

UniVAD 是 training-free few-shot visual anomaly detection。

它的输入是少量 normal reference，测试时不针对新类别训练模型。

核心由三块组成：

```text
C3:
  Contextual Component Clustering
  先把图像分成组件

CAPM:
  Component-Aware Patch Matching
  在组件内部做 normal patch matching

GECM:
  Graph-Enhanced Component Modeling
  建模组件之间的关系异常
```

## 2. 源码里的关键路径

主文件：

```text
references/UniVAD/UniVAD.py
```

few-shot normal reference 在 `setup()` 里建立：

```text
normal_image_features
normal_patch_tokens
normal_dino_patches
normal_clip_part_patch_features
normal_dino_part_patch_features
normal_component_feats
```

测试时 `forward()` 里主要计算：

```text
CLIP patch matching anomaly map
DINOv2 patch matching anomaly map
text/VLM anomaly map
component-level matching score
```

## 3. 和我们当前方案最像的部分

UniVAD 的 texture 分支本质上是：

```text
CLIP patch NN map
+ DINOv2 patch NN map
+ text/VLM map
平均融合
```

这和我们当前主线非常接近：

```text
CLIP-ViT patch normal memory
+ farthest50
+ top-k5
```

区别是：

```text
我们目前只有 CLIP/PromptAD ViT patch feature
UniVAD 同时用了 CLIP patch feature 和 DINOv2 patch feature
```

## 4. 不能直接照搬的部分

UniVAD 的完整组件分割依赖：

```text
GroundingDINO
SAM / HQ-SAM
DINO / DINOv2
CLIP-L
预先生成 masks/
```

这对我们的 RF 频谱图不合适：

- RF 频谱图没有自然图像里的 object/component。
- GroundingDINO/SAM 对频谱图未必有语义。
- 整套系统太重，会破坏我们当前轻量少样本方法的优势。

所以不建议直接移植 UniVAD 全流程。

## 5. 最值得借鉴的点

### 5.1 多视觉编码器互补

UniVAD 同时用：

```text
CLIP patch feature
DINOv2 patch feature
```

这对我们很有启发。

我们已经发现 text 基本没贡献，真正有用的是视觉 normal memory。
因此下一步可以试：

```text
CLIP-ViT memory
+ DINOv2 memory
```

而不是继续优化 prompt/text。

### 5.2 从 object component 改成 spectrogram component

UniVAD 的 component 对自然图像是物体部件。

RF 频谱图里更合理的 component 是：

```text
frequency band component
time-frequency tile component
energy ridge component
```

也就是说，我们不需要 SAM 分割，可以用 RF 先验切分：

```text
按频段切块
按时间窗口切块
按能量连通区域切块
```

然后在每个 component 内做 normal patch matching。

### 5.3 组件级异常分数

UniVAD 不只看 patch，还看 component feature：

```text
area
color
position
CLIP image feature
DINO image feature
```

RF 里可以替换成：

```text
component energy
band position
time duration
frequency span
local texture feature
```

这比单纯 patch max 更有故事：

```text
patch score 负责局部异常
component score 负责结构异常
```

## 6. 建议的新实验路线

不要直接跑完整 UniVAD。

建议先做轻量版：

```text
RF-UniVAD-lite

1. 保留当前 CLIP-ViT farthest50 + top-k5 memory
2. 加一个 DINOv2 patch memory 分支
3. 不用 SAM/GroundingDINO
4. 用 RF 频谱图规则生成 component:
   - frequency band
   - time-frequency grid
   - high-energy connected region
5. 分别计算:
   - global patch memory score
   - component-aware patch memory score
6. 最终先用固定规则融合，不扫 test 权重
```

优先级：

```text
第一优先级:
  DINOv2 patch memory

第二优先级:
  component-aware matching by frequency/time tiles

第三优先级:
  high-energy connected component matching
```

## 7. 当前判断

UniVAD 给我们的最大启发不是 text，也不是 SAM。

真正有用的是：

```text
多视觉 backbone
+ normal reference patch matching
+ component-aware matching
```

这正好支持我们把主线改成纯视觉：

```text
Frequency-Balanced Visual Normal Memory
```

后续如果要继续提升，最值得试的是：

```text
CLIP-ViT memory + DINOv2 memory
```

而不是继续加 prompt learner。

