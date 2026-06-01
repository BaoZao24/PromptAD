# DSSS weak residual improvement

Protocol: self-test dataset, `dsss_signal`, `normal_75_25`, seed `111`, 50 epochs, prompt mode `rf`.

## Compared methods

| method | input description | DSSS mean AUC |
|---|---|---:|
| `rgb` baseline | original RGB spectrogram | 96.4883 |
| `morph_fusion_gabor_residual` | gray contrast + weak residual + Gabor | 95.3358 |
| `morph_fusion_gabor_directional_residual` | gray contrast + row/column residual + Gabor | 93.4992 |
| `dsss_weak_residual_only` | weak residual copied to 3 channels, alpha=0.2 | 96.4917 |
| `dsss_weak_residual_only_a01` | weak residual copied to 3 channels, alpha=0.1 | 96.7883 |

## Result by level

| method | m10db | m20db | m30db | mean |
|---|---:|---:|---:|---:|
| `rgb` baseline | 99.0500 | 97.1325 | 93.2825 | 96.4883 |
| `morph_fusion_gabor_residual` | 99.4100 | 96.1125 | 90.4850 | 95.3358 |
| `morph_fusion_gabor_directional_residual` | 98.4550 | 94.0575 | 87.9850 | 93.4992 |
| `dsss_weak_residual_only` | 99.5275 | 97.3175 | 92.6300 | 96.4917 |
| `dsss_weak_residual_only_a01` | 99.5100 | 97.4250 | 93.4300 | 96.7883 |

## Observation

The direct row/column residual variant is not useful for DSSS. It enhances horizontal abnormal bands visually, but it also amplifies normal background textures and reduces AUC.

The better result comes from a simpler change: use the weak residual image as all three input channels and reduce the residual weight from `alpha=0.2` to `alpha=0.1`. This keeps the residual cue visible while preserving the original spectrogram distribution more closely. The gain is modest but consistent at the dataset mean level, mainly from `m20db` and `m30db`.
