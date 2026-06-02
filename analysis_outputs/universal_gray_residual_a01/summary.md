# Universal gray residual a01 experiment

Protocol: self-test dataset, `normal_75_25`, seed `111`, 50 epochs, prompt mode `rf`.

Candidate method: `morph_fusion_gray_residual_a01`

Table files:

```text
analysis_outputs/universal_gray_residual_a01/method_comparison.csv
analysis_outputs/universal_gray_residual_a01/universal_gray_residual_a01_results.xlsx
analysis_outputs/universal_gray_residual_a01/gray_residual_a01_detailed_results.xlsx
analysis_outputs/universal_gray_residual_a01/contrast_ablation_results.xlsx
```

Input channels:

| channel | content |
|---|---|
| 1 | CLAHE gray contrast |
| 2 | weak residual, `alpha=0.1` |
| 3 | original gray |

## Main comparison

| method | burst | chirp | dsss | three-class mean |
|---|---:|---:|---:|---:|
| RGB baseline | 86.3117 | 78.7308 | 96.4883 | 87.1769 |
| `morph_fusion_gabor_residual` | 90.0117 | 85.3342 | 95.3358 | 90.2272 |
| `morph_fusion_gray_residual_a01` | 90.0708 | 84.9758 | 96.6483 | 90.5650 |

## Observation

Replacing the Gabor channel with the original gray channel fixes the DSSS degradation while preserving most of the burst/chirp gain.

Compared with `morph_fusion_gabor_residual`, the new candidate:

- improves DSSS from `95.3358` to `96.6483`;
- keeps burst almost unchanged, `90.0117` to `90.0708`;
- slightly decreases chirp, `85.3342` to `84.9758`;
- improves the three-class mean from `90.2272` to `90.5650`.

This is currently the better candidate for a unified feature fusion scheme because it does not require knowing the anomaly type in advance.

## Contrast ablation

| method | burst | chirp | dsss | three-class mean | delta vs full |
|---|---:|---:|---:|---:|---:|
| full: `gray_contrast + weak_residual_a01 + gray` | 90.0708 | 84.9758 | 96.6483 | 90.5650 | 0.0000 |
| no contrast: `gray + weak_residual_a01 + gray` | 90.0008 | 85.1058 | 96.2350 | 90.4472 | -0.1178 |
| contrast only: `gray_contrast * 3` | 79.6417 | 72.8858 | 94.9592 | 82.4956 | -8.0694 |

The ablation shows that CLAHE contrast is not effective as a standalone input. However, removing it from the full fusion slightly reduces the three-class mean and especially reduces DSSS. Therefore, the current interpretation should be conservative: `gray_contrast` is an auxiliary weak-structure enhancement channel, while the stability of the full method comes from combining contrast enhancement with original-gray preservation and weak residual information.
