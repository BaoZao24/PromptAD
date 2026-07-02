# Agent Handoff: Line 7 MVFA-Style Visual Encoder Adapter

## 当前状态

当前代码已有后特征 `visual_adapter`，但这不是完整 MVFA。

MVFA-style adapter 指的是在 CLIP visual encoder 多个 transformer block 内插 residual adapter。当前尚未实现。

## 目标

让 CLIP visual encoder 更适配 RF 频谱图分布。

## 需要新增的结构

建议实验名：

```text
pooled_rf_rgb_mvfa_adapter
```

实现位置：

```text
PromptAD/CLIPAD/
```

结构：

```text
x = x + alpha * Adapter(LayerNorm(x))
Adapter = Linear(d, d/r) -> GELU -> Linear(d/r, d)
```

CLIP 主干参数冻结，只训练 adapter。

## 训练协议

实现后必须使用 pooled universal：

```text
20 epoch 初筛
cls + seg
48 cell macro average
```

## 判断

这条线成本较高，只有前面轻量线都不行时再做。
