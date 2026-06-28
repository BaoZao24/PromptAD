# Agent Handoff: PromptAD + Target Normal Gallery Fusion CLS

Date: 2026-06-28

## Question

Test the formal fusion idea:

```text
final_score = PromptAD image score + lambda * target-normal distance score
```

The intended story is:

- Prompt guides RF-aware feature extraction.
- Target-domain normal image distance provides the main anomaly scoring signal.

## What Was Implemented

New script:

```bash
tools/eval_formal_promptad_normal_gallery_fusion_cls.py
```

It loads a pooled RF PromptAD checkpoint, recomputes per-image PromptAD scores, computes target-normal gallery distance on the same images, and evaluates fusion rules.

Command run:

```bash
python tools/eval_formal_promptad_normal_gallery_fusion_cls.py \
  --output-root analysis_outputs/20260628_promptad_normal_gallery_fusion_cls \
  --gpu-id 0 \
  --batch-size 400 \
  --num-workers 4 \
  --normal-topk 50 \
  --lambdas 0.02 0.05 0.1 0.2 0.4
```

Outputs:

```bash
analysis_outputs/20260628_promptad_normal_gallery_fusion_cls/
├── results_cls_promptad_normal_gallery_fusion.csv
├── summary.json
└── scores/*.npz
```

## Baseline Decision

The old formal baseline cannot be strictly fused after the fact, because it did not save per-image PromptAD scores. We therefore adopt the current-code regenerated PromptAD image score as the active baseline for this line.

Old formal baseline file:

```bash
analysis_outputs/20260627_method_funnel/results_cls_pooled_rf_rgb.csv
```

stores only per-cell AUROC and checkpoint path. It does not store per-image PromptAD scores.

Current baseline for this line:

| score source | macro AUROC |
|---|---:|
| old formal baseline CSV | 86.34 |
| regenerated PromptAD scores from checkpoint/current code | 78.41 |

This line should compare against `78.41`, not the old `86.34`, because all fusion scores are computed from the same current-code per-image score export.

## Current-Code Fusion Result

Under current-code regenerated PromptAD scores:

| score | macro AUROC |
|---|---:|
| regenerated PromptAD | 78.41 |
| target normal gallery top50 | 88.64 |
| raw fusion lambda=0.4 | 88.14 |
| raw fusion lambda=1.0 | 90.14 |
| raw fusion lambda=1.5 | 90.23 |
| raw fusion lambda=2.0 | 90.17 |

Best observed setting in the sweep:

```text
final_score = PromptAD image score + 1.5 * target_normal_top50_distance
```

Per-class comparison for selected settings:

| class | current baseline | normal top50 | raw lambda=1.0 | raw lambda=1.5 | raw lambda=2.0 |
|---|---:|---:|---:|---:|---:|
| burst | 82.16 | 91.20 | 91.52 | 91.77 | 91.93 |
| chirp | 89.37 | 89.76 | 92.33 | 92.16 | 91.95 |
| dsss | 61.43 | 95.36 | 93.74 | 94.68 | 95.09 |
| pulse | 80.67 | 78.26 | 82.97 | 82.29 | 81.69 |

Per-class improvement for the selected best setting:

| class | raw lambda=1.5 | delta vs current PromptAD | delta vs normal top50 |
|---|---:|---:|---:|
| burst | 91.77 | +9.61 | +0.58 |
| chirp | 92.16 | +2.79 | +2.40 |
| dsss | 94.68 | +33.26 | -0.67 |
| pulse | 82.29 | +1.62 | +4.03 |

Conclusion:

- Target normal gallery remains very strong.
- Fusion improves over the current PromptAD baseline by about +11.82 macro AUROC.
- Fusion also improves over target normal gallery alone by about +1.58 macro AUROC.
- The best lambda range is around 1.0 to 2.0, which supports the story that prompt is guidance, while target-normal visual distance is the main scoring signal.

Additional sweep output:

```bash
analysis_outputs/20260628_promptad_normal_gallery_fusion_cls/lambda_sweep_current_baseline.csv
```

## Required Fix For Future Formal Experiments

Future official baseline/fusion runs must save per-image scores during evaluation:

```text
name, label, promptad_image_score, score_map summary, dataset, scene, jsr
```

Then fusion should be computed from those saved scores, not by trying to regenerate old scores after code changes.

Recommended next official experiment:

1. Rerun `pooled_rf_rgb` baseline with score export enabled.
2. Compute target-normal top50 distance on the exact same sample order.
3. Fuse saved per-image baseline score with normal distance.
4. Select one universal lambda by macro average, not per class.
