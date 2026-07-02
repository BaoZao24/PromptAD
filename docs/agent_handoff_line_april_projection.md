# Agent Handoff: Line 5 APRIL-GAN / Text-Aligned Projection

## 当前状态

这条线暂时不要直接跑正式实验。

仓库里已有 `text_aligned_dense` 相关模块，但 normal-only pooled 协议下还缺少稳定训练信号。APRIL-GAN 风格 projection 通常需要 dense mask 或更明确的 patch-level 对齐监督；本轮 pooled normal-only 不能直接把它当成熟方法。

## 后续要做的实现

先设计训练信号，再跑实验：

```text
visual patch token -> linear projection -> text space
patch score = abnormal prototype margin over normal prototype
```

可选训练方式：

1. 只用 normal，把所有 normal patch 拉向 normal text prototype。
2. 加 synthetic anomaly / CutPaste 式伪异常，再训练 abnormal margin。
3. 如果允许 mask supervision，再用 abnormal mask 训练 projection map。

## 暂定实验名

```text
pooled_rf_rgb_text_aligned_projection
```

## 判断

只有实现完训练信号后，才能和 pooled `rf + rgb` 比较。不要用未训练的 projection head 宣称结果。
