# Important Update: Prompt-Guided Normality Calibration

Date: 2026-06-29

## Summary

This update establishes a successful direction for both classification and segmentation:

```text
Prompt provides guidance.
Target-domain normal visual distance provides calibration.
```

For classification, the calibration is based on global image distance to target normal examples.

For segmentation, the calibration is based on patch-level distance to target normal patch examples.

## Classification Result

Active baseline for this line:

```text
current-code regenerated PromptAD image score
```

Best score:

```text
final_score = PromptAD image score + 1.5 * target_normal_top50_distance
```

Macro image AUROC:

| method | AUROC |
|---|---:|
| current PromptAD baseline | 78.41 |
| target normal gallery top50 | 88.64 |
| PromptAD + 1.0 * normal distance | 90.14 |
| PromptAD + 1.5 * normal distance | 90.23 |
| PromptAD + 2.0 * normal distance | 90.17 |

Best observed improvement:

```text
78.41 -> 90.23
+11.82 image AUROC
```

Result files:

```bash
analysis_outputs/20260628_promptad_normal_gallery_fusion_cls/results_cls_promptad_normal_gallery_fusion.csv
analysis_outputs/20260628_promptad_normal_gallery_fusion_cls/lambda_sweep_current_baseline.csv
analysis_outputs/20260628_promptad_normal_gallery_fusion_cls/summary.json
```

## Segmentation Result

Best map:

```text
final_map = textual_map * (1 + 0.25 * normalized_normal_patch_distance_map)
```

Macro pixel AUROC:

| map | pROC |
|---|---:|
| formal baseline / current harmonic | 89.33 |
| textual only | 89.39 |
| normal patch only | 89.16 |
| add beta=0.25 | 90.14 |
| gate beta=0.5 | 90.94 |
| gate beta=0.25 | 91.53 |

Best observed improvement:

```text
89.33 -> 91.53
+2.20 pixel AUROC
```

Per-class pROC:

| class | baseline | gate beta=0.25 | delta |
|---|---:|---:|---:|
| burst | 98.04 | 98.17 | +0.13 |
| chirp | 95.08 | 96.11 | +1.03 |
| dsss | 72.64 | 76.04 | +3.40 |
| pulse | 91.58 | 95.81 | +4.23 |

Result files:

```bash
analysis_outputs/20260628_seg_prompt_normal_gallery_fusion/results_seg_prompt_normal_gallery_fusion.csv
analysis_outputs/20260628_seg_prompt_normal_gallery_fusion/summary.json
```

## Interpretation

This supports the main paper story:

```text
Prompt-guided normality calibration.
```

Plain explanation:

```text
Prompt tells the model what kind of region is suspicious.
Target normal gallery tells the model whether the region really deviates from normal RF spectrograms.
```

The segmentation gate result is especially important because direct addition is weaker than gated fusion. That means the normal patch distance should not create anomalies by itself; it should strengthen prompt-suspicious regions.

## Implementation

New evaluation scripts:

```bash
tools/eval_formal_promptad_normal_gallery_fusion_cls.py
tools/eval_seg_prompt_normal_gallery_fusion.py
```

Detailed handoff documents:

```bash
docs/agent_handoff_promptad_normal_gallery_fusion_cls.md
docs/agent_handoff_seg_prompt_normal_gallery_fusion.md
```
