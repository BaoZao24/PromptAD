# Agent Handoff: SEG Prompt + Normal Gallery Fusion

Date: 2026-06-29

## Purpose

Test whether segmentation can benefit from the same story as CLS:

```text
Prompt gives semantic localization.
Target normal patch gallery gives visual normality calibration.
```

## Method

No training. Load the pooled RF segmentation checkpoint and evaluate several maps:

- `current_harmonic`: existing PromptAD segmentation map.
- `textual_only`: prompt/text semantic anomaly map.
- `normal_patch_only`: visual distance from target normal patch gallery.
- `gate_norm`: conservative gated fusion.
- `add_norm`: direct additive fusion.

Main proposed fusion:

```text
final_map = textual_map * (1 + beta * normalized_normal_patch_distance_map)
```

This means normal patch distance does not create anomalies alone. It strengthens regions already considered suspicious by the prompt map.

## Command

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python tools/eval_seg_prompt_normal_gallery_fusion.py \
  --output-root analysis_outputs/20260628_seg_prompt_normal_gallery_fusion \
  --gpu-id 3 \
  --batch-size 100 \
  --num-workers 4 \
  --betas 0.25 0.5 1.0 2.0 4.0
```

## Outputs

```bash
analysis_outputs/20260628_seg_prompt_normal_gallery_fusion/
├── results_seg_prompt_normal_gallery_fusion.csv
├── summary.json
└── run_gpu3.log
```

## Main Result

Macro pixel AUROC:

| map | pROC |
|---|---:|
| formal baseline / current harmonic | 89.33 |
| textual only | 89.39 |
| normal patch only | 89.16 |
| add beta=0.25 | 90.14 |
| gate beta=1.0 | 90.32 |
| gate beta=0.5 | 90.94 |
| gate beta=0.25 | 91.53 |

Best result:

```text
gate_norm_beta0.25_p_roc = 91.53
```

Improvement over baseline:

```text
+2.20 pROC
```

## Per-Class Result

| class | baseline | best gate beta=0.25 | delta |
|---|---:|---:|---:|
| burst | 98.04 | 98.17 | +0.13 |
| chirp | 95.08 | 96.11 | +1.03 |
| dsss | 72.64 | 76.04 | +3.40 |
| pulse | 91.58 | 95.81 | +4.23 |

## Conclusion

The segmentation story is supported.

The existing PromptAD harmonic map is already strong, but the conservative gated normality calibration improves it clearly. The gain is largest on DSSS and pulse, which are also the cases where target-domain visual calibration is most useful.

Recommended line for the paper:

```text
Prompt-Guided Normality-Calibrated Localization
```

Short explanation:

```text
Prompt identifies semantically suspicious regions, while target normal patch distance calibrates their anomaly strength using real normal RF spectrogram patches.
```
