# FastRecon 互补性发现

更新时间：2026-07-04

## 结论

FastRecon 不适合作为 PromptAD 的替代方法，但适合作为少样本 RF 异常检测的互补分支。

核心发现：

```text
PromptAD baseline: text + ViT patch anomaly map = 91.27
FastRecon lambda=0 = 88.94
ViT max + 1.5 * FastRecon = 93.15
PromptAD baseline + 1.5 * FastRecon = 93.15
```

也就是说，FastRecon 单独不如 PromptAD，但它提供了 PromptAD / ViT 没有覆盖好的 reconstruction residual 证据。

## 实验协议

当前协议是少样本 AD：

```text
support/gallery:
  每个频段 1 张 target normal
  共 24 张 support normal

test:
  48 个 cell
  support 图从 test normal 中排除
```

不使用 target abnormal 训练或建库。

## 分支含义

PromptAD / ViT patch gallery 判断：

```text
这个 patch 离 few-shot normal gallery 有多远？
```

FastRecon 判断：

```text
这个 patch 能不能被 normal patch coreset 重构出来？
```

所以两者不是重复分支：

```text
ViT gallery = similarity-based normality
FastRecon = reconstruction-based normality
```

## 按异常类型结果

| 异常类型 | PromptAD baseline | FastRecon | ViT + FastRecon |
|---|---:|---:|---:|
| burst | 92.9407 | 86.8355 | **95.5892** |
| chirp | **94.7551** | 88.7990 | 94.5394 |
| dsss | 95.3147 | 95.4710 | **96.0951** |
| pulse | 82.0791 | 84.6348 | **86.3927** |

解释：

- `burst` 和 `pulse` 更偏局部突发/瞬态异常，FastRecon 的重构残差补充明显。
- `dsss` 小幅受益。
- `chirp` 的结构很规则，PromptAD / ViT 已经很强，FastRecon 反而可能引入噪声。

## Lambda 结论

FastRecon 论文默认的分布正则在当前 RF 频谱图上不适合：

| lambda | Image-AUROC macro |
|---:|---:|
| 0 | **88.9351** |
| 0.5 | 87.3704 |
| 2 | 76.3915 |
| 10 | 68.3307 |

当前应使用：

```text
lambda = 0
```

也就是只使用 normal coreset reconstruction residual，不加分布正则。

## 推荐写法

当前方法可以表述为：

```text
final_score
= minmax(ViT patch anomaly score)
+ alpha * minmax(FastRecon reconstruction residual)
```

故事重点：

```text
我们用 PromptAD / ViT 捕获结构型正常性距离，
用 FastRecon 捕获局部异常的不可重构性。
两者分别对应 similarity-based normality 和 reconstruction-based normality。
```

## 结果文件

```text
analysis_outputs/20260703_fewshot_vit_patch_gallery_rerun/
analysis_outputs/20260703_fewshot_fastrecon_cls/
analysis_outputs/20260703_fewshot_fastrecon_complementarity_wide_alpha/
```

对应代码：

```text
tools/eval_cls_vit_patch_gallery.py
tools/eval_fastrecon_cls.py
tools/archive/analyze_fewshot_fastrecon_complementarity.py
```
