# Agent Handoff: Line 6 AnoPLe-Style Prompt Interaction

## 当前状态

survey 第一名是 AnoPLe，不是 VCPA。

这条线尚未在当前代码中实现，不能直接跑。

## 目标

让 visual patch tokens 反过来 condition text prompt context，使 prompt 不再只依赖固定类别词。

## 需要新增的结构

建议实验名：

```text
pooled_rf_rgb_anople_interaction
```

最小实现：

```text
visual tokens -> small MLP -> visual prompt tokens
visual prompt tokens + learnable text context + class token -> text encoder
```

实现位置：

```text
PromptAD/model.py
PromptLearner
```

## 训练协议

实现后必须使用同一 pooled universal 协议：

```text
burst/chirp/dsss/pulse pooled normal train
48 cell eval
20 epoch
cls + seg
```

## 判断

优先看 cls overall 和 dsss。若 AnoPLe 只提升 seg map 但 cls 不动，应当作为 pixel 分支，而不是主 image-level 方法。
