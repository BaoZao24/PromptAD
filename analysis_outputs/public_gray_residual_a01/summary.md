# Public RF_SPE_PNG Validation for `morph_fusion_gray_residual_a01`

Protocol:

```text
dataset = rf_spe_png
normal source = RF_Spectrum_Public_Dataset / MeasRes_*
train normal = first k_shot=1 MeasRes_* folder
test normal = remaining MeasRes_* folders
test abnormal = RF_SPE_PNG/{class}/abnormal/m40db
seed = 111
epochs = 50
prompt_mode = rf
input_mode = morph_fusion_gray_residual_a01
```

Current unified scheme:

```text
morph_fusion_gray_residual_a01 = gray_contrast + weak_residual(alpha=0.1) + original_gray
```

Comparison file:

```text
analysis_outputs/public_gray_residual_a01/m40_comparison.csv
```

## Results at m40db

| anomaly | RGB baseline | historical exploratory input | current gray_residual_a01 | delta vs baseline | delta vs historical |
|---|---:|---:|---:|---:|---:|
| `burst` | 81.97 | 84.31 | 88.47 | +6.50 | +4.16 |
| `chirp` | 98.90 | 98.97 | 99.30 | +0.40 | +0.33 |
| `dsss` | 67.70 | 69.21 | 71.88 | +4.18 | +2.67 |
| **mean** | **82.86** | **84.16** | **86.55** | **+3.69** | **+2.39** |

## Interpretation

On the public `RF_SPE_PNG` validation protocol, the current unified input scheme still improves all three anomaly types at `m40db`.

Compared with RGB baseline, the gain is largest on `burst` and `dsss`, which are also the harder low-SNR cases on this protocol. `chirp` was already very strong with RGB, so the gain is small but still positive.

Compared with the earlier exploratory input setting, the current unified scheme is also better on all three classes. The key point is not to keep selection as a method, but to show that the unified input can improve public-data performance without anomaly-type-specific prior knowledge.
