# RF prompt + gray residual a01 experiment

Protocol: self-test dataset, `normal_75_25`, seed `111`, 50 epochs.

True baseline: original PromptAD, `prompt_mode=legacy`, `input_mode=rgb`.

Current scheme: `prompt_mode=rf`, `input_mode=morph_fusion_gray_residual_a01`.

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
| Original PromptAD baseline (`legacy` + RGB) | 86.8375 | 78.9358 | 96.4792 | 87.4175 |
| Current scheme (`rf` + `morph_fusion_gray_residual_a01`) | 90.0708 | 84.9758 | 96.6483 | 90.5650 |
| Delta | +3.2333 | +6.0400 | +0.1691 | +3.1475 |

## Observation

The current scheme improves the true original PromptAD baseline by `+3.1475` average Image-AUROC points across the three studied RF anomaly types.

The previous `RF prompt + RGB` result should not be called the original baseline because it already contains RF-domain handcrafted prompt states. This summary therefore keeps only the headline comparison: original PromptAD vs current RF-adapted scheme.

## Contrast ablation

| method | burst | chirp | dsss | three-class mean | delta vs full |
|---|---:|---:|---:|---:|---:|
| full: `gray_contrast + weak_residual_a01 + gray` | 90.0708 | 84.9758 | 96.6483 | 90.5650 | 0.0000 |
| no contrast: `gray + weak_residual_a01 + gray` | 90.0008 | 85.1058 | 96.2350 | 90.4472 | -0.1178 |
| contrast only: `gray_contrast * 3` | 79.6417 | 72.8858 | 94.9592 | 82.4956 | -8.0694 |

The ablation shows that CLAHE contrast is not effective as a standalone input. However, removing it from the full fusion slightly reduces the three-class mean and especially reduces DSSS. Therefore, the current interpretation should be conservative: `gray_contrast` is an auxiliary weak-structure enhancement channel, while the stability of the full method comes from combining contrast enhancement with original-gray preservation and weak residual information.
