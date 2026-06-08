# Baseline Redefinition: Original PromptAD vs Current RF Scheme

Date: 2026-06-03

## Why This Experiment Was Needed

Previous tables used the name `RGB baseline`, but that run was not the original PromptAD baseline. It used:

```text
prompt_mode = rf
input_mode = rgb
```

The `rf` prompt mode already contains RF-domain handcrafted prompt states. Therefore, the correct original baseline should be:

```text
Original PromptAD baseline = legacy prompt + RGB input
```

## Experiment Setup

```text
prompt_mode = legacy
input_mode = rgb
cls_score_mode = text_only
split = normal_75_25
seed = 111
epochs = 50
datasets = burst_signal, chirp_signal, dsss_signal
scenes = WeaponMuseum_spectrum, Playground_spectrum, TimeSquare_spectrum, Gymnasium_spectrum
noise levels = m10db, m20db, m30db
```

Note: the final Image-AUROC follows the existing `metric_cal_img` implementation, which applies harmonic fusion between image score and max anomaly-map score.

## Results

| Method | burst_signal | chirp_signal | dsss_signal | Three-class mean |
|---|---:|---:|---:|---:|
| Original PromptAD (`legacy` + RGB) | 86.8375 | 78.9358 | 96.4792 | 87.4175 |
| Current full scheme (`rf` + `morph_fusion_gray_residual_a01`) | 90.0708 | 84.9758 | 96.6483 | 90.5650 |
| Delta | +3.2333 | +6.0400 | +0.1691 | +3.1475 |

## Interpretation

The current full scheme improves the original PromptAD baseline by `+3.1475` average Image-AUROC points across the three studied RF anomaly types.

This comparison is the correct headline comparison for the thesis:

```text
Original PromptAD -> RF-adapted PromptAD with structure-aware grayscale residual input
```

The previous `RF prompt + RGB` result should not be called the original baseline. It is an intermediate RF-prompt baseline.

## Output Files

```text
experiments/baseline_redefinition/original_promptad_legacy_rgb_results.csv
experiments/baseline_redefinition/original_promptad_legacy_rgb_summary.csv
```
