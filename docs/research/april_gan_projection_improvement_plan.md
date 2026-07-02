# APRIL-GAN 投影分支改进方案

> 依据：`docs/research/clip_fsad_survey_2023_2026.md` 中排序第 2 的 APRIL-GAN 跨模态投影方案；源码参考 `references/VAND-APRIL-GAN/`。
>
> **状态说明（2026-07-02）**：本文是历史研究草案。文中涉及 `public source` 和
> `train_rf_public_to_target_dense.py` 的命令已经不属于当前主线；对应跨库入口已删除。
> 当前实验只保留 target pooled normal-only 协议。

## 1. 结论先说

当前 dense/VCP 方案主要改善的是 **pixel-level anomaly map**，但 image score 仍基本依赖原 PromptAD 文本分数或简单 map 聚合，所以 Image ROC 很难稳定超过 baseline。

APRIL-GAN 给我们的启发是：不要只训练一个像素 head，也不要只调一个融合权重，而是让 **CLIP patch token 经过可训练线性投影后直接和 normal/abnormal text prototype 对齐**。这样训练信号既来自 mask，又保留了 CLIP 的 text/image 语义空间，适合补上当前方案缺少的图级判别能力。

建议新方案叫：**Text-Aligned Dense Projection Head**，简称 `ta_dense`。

## 2. APRIL-GAN 源码中真正值得移植的部分

源码核心非常小：

- `references/VAND-APRIL-GAN/model.py`
  - `LinearLayer(dim_in, dim_out, k, model)`：每个 selected visual layer 一个 `nn.Linear`。
  - ViT token 输入时会去掉 CLS：`tokens[i][:, 1:, :]`。
- `references/VAND-APRIL-GAN/train.py`
  - 冻结 CLIP image/text encoder。
  - `patch_tokens = trainable_layer(patch_tokens)`。
  - 每层 patch token 和 `[normal_text, abnormal_text]` 做相似度。
  - 得到 `(B, 2, H, W)` anomaly map 后，用 focal + dice 监督 mask。
- `references/VAND-APRIL-GAN/test.py`
  - image score 仍用全局 image feature 和 text feature。
  - pixel map 用多层 projected patch-text map 求和。
  - few-shot 额外加 normal memory 最近邻图。

它不是复杂网络，关键是 **visual patch -> text embedding space** 这个约束。

## 3. 当前 PromptAD 的问题定位

我们现在的 `dense_mask_head`：

```text
visual_features[2], visual_features[3]
  -> DenseMaskHead(raw_head/post_head)
  -> sigmoid dense map
  -> top-k 或 map-only eval
```

问题是：

1. `DenseMaskHead` 的输出是一个自由二分类 map，没有被约束到 CLIP text space。
2. mask loss 让它学会“哪里像异常”，但不一定学会“整张图是否异常”。
3. 图级分数目前常见路径是：
   - 原 PromptAD text score；或
   - dense map top-k；或
   - text + alpha * dense。
4. 这会导致 pixel AUROC 可能提升，但 Image ROC 不稳定，尤其 DSSS 这种异常外观接近正常结构时，简单 top-k 容易误判。

所以改进重点不是再调 `alpha`，而是让 dense 分支产生一个更有语义约束的 image score。

## 4. 新方案结构

### 4.1 新增模块

在 `PromptAD/model.py` 中新增：

```text
TextAlignedDenseProjectionHead
  input : visual_features[2], visual_features[3]
  train : 每路一个 Linear(dim_visual -> dim_text)
  output:
    - projected text-space patch tokens
    - per-layer patch-text anomaly logits
    - text-aligned anomaly map
    - text-aligned image score
```

推荐先只接两路现有 patch feature：

```text
feature_map1 = visual_features[2]
feature_map2 = visual_features[3]
```

不先改 CLIP encoder 去取更多中间层，因为那会扩大风险；等两路版本有效后，再考虑多层 token。

### 4.2 前向流程

```text
source / target spectrogram
  -> morph_fusion_gray_residual_a01
  -> frozen CLIP image encoder
  -> feature_map1, feature_map2
  -> Linear projection to text dim
  -> normalize
  -> dot([normal_text, abnormal_text])
  -> softmax / margin
  -> text-aligned anomaly map
  -> top-k / mean+top-k image score
```

这里的 text prototype 直接复用 PromptAD 当前 `self.text_features`：

```text
text_features[0] = normal prototype
text_features[1] = abnormal prototype
```

如果当前使用 grouped abnormal prototypes，则先做两个版本：

- `single`：normal vs averaged abnormal，最稳定。
- `grouped_max`：normal vs max over burst/chirp/dsss abnormal prototypes，后续 ablation。

## 5. 训练目标

训练仍保持不作弊协议：

```text
public source: normal + abnormal + mask
目标 target: 只用 target source normal gallery
评估 target test: burst/chirp/dsss × 多 JSR
```

新增 loss：

```text
L_total = L_dense_mask + lambda_ta * L_text_aligned
```

其中：

```text
L_text_aligned =
  BCEWithLogits(projected_patch_abnormal_logit, source_mask)
  + Dice(sigmoid(projected_patch_abnormal_logit), source_mask)
  + optional CE(image_logit, image_label)
```

图级 CE 可以先加很小权重，例如 `lambda_img = 0.1`，因为 source/target 图级分布有域差异，过强会让模型记 source 亮度风格。

推荐初始权重：

```text
lambda_dense = 1.0
lambda_ta_mask = 1.0
lambda_ta_img = 0.1
lr_projection = 1e-3
Epoch = 8
```

训练轮数先不要设太长。APRIL-GAN reference 的脚本里 MVTec 只训 3 epoch、VisA 训 15 epoch；我们先用 8 epoch 做主验证，确认方向有效后再扩到 15 epoch。

## 6. 推理时怎么用分数

不要再把 image score 交给固定 alpha。

推荐三种 eval mode：

### A. `ta_map_only`

```text
Image ROC: 原 PromptAD image score
Pixel ROC: text-aligned dense map
```

目的：确认投影分支是否比当前 dense/VCP map 更好。

### B. `ta_image_only`

```text
Image ROC: text-aligned image score
Pixel ROC: text-aligned dense map
```

目的：验证 APRIL-GAN 风格的 projected patch-text 分支是否真正能独立做图级异常判断。

### C. `ta_harmonic`

```text
Image ROC: harmonic(original_promptad_score, ta_image_score, max(ta_map))
Pixel ROC: text-aligned dense map
```

目的：如果单独 `ta_image_score` 有偏差，用调和融合要求“文本全局 + patch 局部”都支持异常才报警，减少 DSSS 正常结构误报。

## 7. 和当前 dense/VCP 的关系

不建议直接抛弃当前 dense branch。

更稳的结构是并联：

```text
visual_features[2]/[3]
  ├─ DenseMaskHead                  -> free dense map
  └─ TextAlignedProjectionHead      -> text-space dense map + image score
```

原因：

- `DenseMaskHead` 更自由，可能定位强。
- `TextAlignedProjectionHead` 受 text prototype 约束，可能图级分数更稳。
- 两者可以先分别评估，后续再考虑融合。

第一阶段不要训练复杂 fusion head；先看两条分支各自的真实能力。

## 8. 预期能解决什么

重点解决：

1. Image ROC 难提升：新增一个被 mask + image label 监督的图级分数。
2. DSSS 容易关注正常结构：projection 分支不是只看亮度/局部响应，而是要求 patch feature 在 text abnormal prototype 方向上更像异常。
3. alpha 调不动：不再靠固定比例，把问题改成“学习一个 text-aligned patch classifier”。

不能保证解决：

- 如果 PromptAD 的 abnormal text prototype 本身对 RF 语义很差，projection 也会被弱文本锚点限制。
- 如果 public source 的 DSSS mask 与 target DSSS 形态仍有显著差异，仍需要 target normal memory 或合成异常辅助。

## 9. 实施顺序

### Step 1: 最小可行版本

- 新增 `TextAlignedDenseProjectionHead`。
- 接 `visual_features[2]` 和 `visual_features[3]`。
- 使用当前 `self.text_features[:2]`。
- 增加 `calculate_text_aligned_dense_score()`。
- 新增 eval mode：`ta_map_only`, `ta_image_only`, `ta_harmonic`。

### Step 2: 训练入口接入

在 `train_rf_public_to_target_dense.py` 增加：

```text
--text-aligned-dense true/false
--ta-loss-weight
--ta-image-loss-weight
--ta-lr
--dense-eval-mode ta_map_only|ta_image_only|ta_harmonic
```

optimizer 同时训练：

```text
dense_mask_head + text_aligned_projection_head
```

checkpoint 保存：

```text
dense_mask_head.*
text_aligned_projection_head.*
prompt_learner.*
```

### Step 3: 实验矩阵

保持同一个不作弊协议：

```text
public source 全异常训练 + target source normal gallery + target test 全异常全 JSR
```

最少跑：

| 方法 | image score | pixel map | 目的 |
|---|---|---|---|
| baseline | 原 PromptAD | 原 PromptAD map | 当前基线 |
| dense_map_only | 原 PromptAD | DenseMaskHead | 已有 VCP 定位收益 |
| ta_map_only | 原 PromptAD | TA map | 比较 map 是否更好 |
| ta_image_only | TA image score | TA map | 看 image score 是否能独立提升 |
| ta_harmonic | PromptAD + TA harmonic | TA map | 主候选 |

每个方法输出仍保持：

```text
burst/chirp/dsss × m10/m20/m30 ROC
每类 avg
all avg
best overall checkpoint
score npz
```

## 10. 成败判据

优先看 Image ROC，但不能牺牲 pixel 太多：

```text
主指标：all avg Image ROC > baseline rf + morph_fusion_gray_residual_a01
次指标：DSSS avg Image ROC 不低于 baseline 太多，最好提升
辅助：Pixel AUROC 不低于当前 dense/VCP map
```

如果结果是：

- `ta_map_only` 好、`ta_image_only` 差：说明 projection 只适合定位，图级聚合方式还要改。
- `ta_image_only` 好、`ta_harmonic` 差：说明融合压制了有效信号，应该直接用 TA image。
- DSSS 仍差：优先检查 projected map 是否仍盯正常结构；如果是，下一步做 DSSS-specific text prototype 或 source DSSS hard-negative 增强。

## 11. 一句话方案

把 APRIL-GAN 的 `LinearLayer visual->text projection` 移植成 PromptAD 的并联 text-aligned dense 分支，让 public source mask 监督的不只是“哪里异常”，还监督“哪些 patch 在 CLIP 文本异常方向上异常”，再用它产生可评估的 image score。

## 12. 当前实现入口

当前已实现最小版 `ta_topk`：

```text
image score = 原 PromptAD image score + ta_score_beta * top-k(text-aligned dense map)
pixel map   = text-aligned dense map
```

checkpoint 选择只保留通用版本：

```text
best checkpoint = 3 类 × 3 JSR 的 Image ROC 宏平均最高的 epoch
保存文件        = ...all_test...overall-best.pt
不再保存        = burst/chirp/dsss 各自单独 best checkpoint
```

推荐先跑：

```bash
python train_rf_public_to_target_dense.py \
  --method-tag ta_topk \
  --dense-eval-mode ta_topk \
  --ta-score-beta 1.0 \
  --ta-loss-weight 1.0 \
  --ta-image-loss-weight 0.1 \
  --source-noise-level m10db \
  --eval-jsrs m10db,m20db,m30db \
  --target-classes burst,chirp,dsss \
  --pool-classes burst,chirp,dsss \
  --input-mode morph_fusion_gray_residual_a01 \
  --prompt-mode rf \
  --text-prototype-mode single \
  --image-roc-source raw \
  --Epoch 8 \
  --seed 111
```

如果 `raw` image ROC 不好，再用同一 checkpoint 做 `--eval-only --dense-eval-mode ta_harmonic` 诊断融合是否能压误报；主候选仍然优先看 `ta_topk`。

建议实验节奏：

```text
3 epoch  : smoke，确认代码、loss、结果文件没问题
8 epoch  : 主验证，先看是否超过 baseline
15 epoch : 只有 8 epoch 有希望时再跑正式长一点版本
```
