# Public RF_SPE_PNG: Current Scheme vs Original PromptAD

## Protocol

```text
dataset = rf_spe_png
normal source = /mnt/data/wangbei/data/RF_SPE_PNG/RF_Spectrum_Public_Dataset/
train normal = sorted MeasRes_* directories, first k_shot=1 directory
test normal = remaining MeasRes_* directories
burst abnormal = /mnt/data/wangbei/data/RF_SPE_PNG/burst/abnormal/{m30db,m40db,m50db}
chirp abnormal = /mnt/data/wangbei/data/RF_SPE_PNG/chirp/abnormal/{m50db,m55db,m60db}
dsss abnormal = /mnt/data/wangbei/data/RF_SPE_PNG/dsss/abnormal/{m20db,m30db,m40db}
seed = 111
epochs = 50
```

对比对象：

```text
真正 baseline = original PromptAD = legacy prompt + RGB input
现有方案 = rf prompt + morph_fusion_gray_residual_a01
```

其中：

```text
morph_fusion_gray_residual_a01 = gray_contrast + weak_residual(alpha=0.1) + original_gray
```

## Full Results

| anomaly | level 1 baseline | level 1 current | delta | level 2 baseline | level 2 current | delta | level 3 baseline | level 3 current | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `burst` | 94.87 | 96.88 | +2.01 | 84.29 | 88.47 | +4.18 | 62.39 | 65.29 | +2.90 |
| `chirp` | 90.11 | 94.16 | +4.05 | 66.14 | 71.75 | +5.61 | 54.18 | 55.55 | +1.37 |
| `dsss` | 96.83 | 94.99 | -1.84 | 85.80 | 84.40 | -1.40 | 69.78 | 71.88 | +2.10 |

注：对应档位分别为：
- `burst`: `m30db / m40db / m50db`
- `chirp`: `m50db / m55db / m60db`
- `dsss`: `m20db / m30db / m40db`

## Mean Results

| anomaly | baseline mean | current mean | delta |
|---|---:|---:|---:|
| `burst` | 80.52 | 83.55 | +3.03 |
| `chirp` | 70.14 | 73.82 | +3.68 |
| `dsss` | 84.14 | 83.76 | -0.38 |
| **overall** | **78.27** | **80.37** | **+2.11** |

## Interpretation

这版公开协议的关键变化是：`burst` 和 `chirp` 都不再使用过于容易的低档位，而是统一往更难的区间移动，因此更能体现方法差异。

- `burst` 在 `m30/m40/m50` 三档上都是正收益，均值提升 `+3.03`
- `chirp` 在 `m50/m55/m60` 上仍然保持明确增益，尤其 `m50/m55` 提升明显
- `dsss` 仍然不够稳定，`m20/m30` 回落，`m40` 回升

因此，新的公开数据补充验证支持的结论是：

1. 当前方案对 `burst` 和 `chirp` 都有稳定增益；
2. `dsss` 仍然是公开协议下最不稳定的一类；
3. 在这组混合协议下，九条结果总体均值从 `78.27` 升到 `80.37`，总体增益为 `+2.11`。
