# Public RF_SPE_PNG Validation for `morph_fusion_gray_residual_a01`

正式公开数据对比表已经更新为新的混合协议：

```text
analysis_outputs/public_current_vs_original/public_current_vs_original.csv
analysis_outputs/public_current_vs_original/summary.md
```

本目录保留为“现有方案公开数据记录”。

当前公开数据协议下，现有方案 `rf + morph_fusion_gray_residual_a01` 的 9 条结果为：

| anomaly | level 1 | level 2 | level 3 | mean |
|---|---:|---:|---:|---:|
| `burst` | 96.88 | 88.47 | 65.29 | 83.55 |
| `chirp` | 94.16 | 71.75 | 55.55 | 73.82 |
| `dsss` | 94.99 | 84.40 | 71.88 | 83.76 |

注：对应档位分别为：
- `burst`: `m30db / m40db / m50db`
- `chirp`: `m50db / m55db / m60db`
- `dsss`: `m20db / m30db / m40db`
