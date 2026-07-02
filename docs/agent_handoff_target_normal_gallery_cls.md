# Agent Handoff: Target Normal Gallery CLS

Date: 2026-06-28

## Purpose

Test a lightweight image-only score for RF anomaly classification:

> anomaly score = distance from the test image feature to the target-domain normal image gallery.

This does not train a new network and does not use target abnormal samples. It uses target normal samples as the normal reference, then evaluates on target test cells.

## Why This Line Exists

Previous source-abnormal prototype tests showed that text/source-abnormal prototypes are unstable, while target normal distance is a strong signal. In plain terms: for RF spectrograms, CLIP may not reliably understand abnormal type names, but it can still tell whether an image looks far from target normal examples.

## Code

New script:

```bash
tools/eval_target_normal_gallery_cls.py
```

The script:

- Builds the same pooled target-normal gallery used by `train_rf_target_pooled_universal.py`.
- Encodes target normal samples with PromptAD image encoder.
- Evaluates all target test cells for burst/chirp/dsss/pulse.
- Compares each cell against the formal `rf+rgb` baseline CSV.
- Saves one long-table CSV and a summary JSON.

## Command Run

```bash
python tools/eval_target_normal_gallery_cls.py \
  --output-root analysis_outputs/20260628_target_normal_gallery_cls \
  --gpu-id 0 \
  --batch-size 400 \
  --num-workers 4 \
  --normal-topk 1 5 10 20 50
```

## Outputs

```bash
analysis_outputs/20260628_target_normal_gallery_cls/
├── results_cls_target_normal_gallery.csv
├── summary.json
└── target_normal_gallery_features.npz
```

Formal baseline used for comparison:

```bash
analysis_outputs/20260627_method_funnel/results_cls_pooled_rf_rgb.csv
```

## Main Result

Macro image AUROC over all evaluated cells:

| score | AUROC |
|---|---:|
| formal baseline rf+rgb | 86.34 |
| normal distance top1 | 85.97 |
| normal distance top10 | 86.76 |
| normal distance top20 | 88.13 |
| normal distance top50 | 88.64 |
| normal distance mean | 74.97 |

Best setting: `normal_dist_top50_auc`, +2.31 over the formal baseline macro average.

## Per-Class Result

`normal_dist_top50_auc` versus formal baseline:

| class | baseline | target normal top50 | delta |
|---|---:|---:|---:|
| burst | 92.25 | 91.20 | -1.05 |
| chirp | 92.52 | 89.76 | -2.76 |
| dsss | 81.75 | 95.36 | +13.60 |
| pulse | 78.83 | 78.26 | -0.56 |

## Interpretation

This method is not a universal replacement for the PromptAD image score.

It strongly fixes DSSS, because DSSS abnormal images are better separated by distance from target normal visual features than by text prompts or source abnormal prototypes.

It slightly hurts burst/chirp, where the current baseline is already strong. For pulse it is roughly tied on average, but helps weak m40 cells and hurts some easy m20/m30 cells.

## Recommended Next Step

Use target normal gallery as a complementary image score, not as the only score.

The next experiment should export per-image formal PromptAD scores and test simple fusion:

```text
final_score = baseline_image_score + lambda * target_normal_top50_score
```

Suggested lambdas:

```text
0.05, 0.10, 0.20
```

Selection rule should still be one universal average-best rule, not per-class best.

## Caveats

- This run is classification/image-level only.
- It does not evaluate pixel-level segmentation.
- It does not train a checkpoint.
- It uses only target normal gallery, so it is acceptable for an anomaly detection setting where target normal training samples are available.
