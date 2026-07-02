# Public RF CLS Dual Gallery Evaluation

This experiment evaluates the current dual-gallery CLS method on RF_SPE_PNG.

Normal samples are split by public normal record folders: 75% for gallery, 25% for test normals.
Abnormal samples are read from RF_SPE_PNG/{burst,chirp,dsss,pulse}/abnormal/{jsr}.

Evaluated JSR cells:

```text
burst: m30db / m40db / m50db
chirp: m40db / m50db / m55db
dsss:  m30db / m40db / m50db
pulse: m30db / m40db / m50db
```

Macro Image-AUROC over 12 cells:

| Method | Image-AUROC |
|---|---:|
| PromptAD score with current checkpoint | 48.4355 |
| ResNet18 layer3 normal gallery | 70.3481 |
| Best dual fusion in sweep | 77.0137 |
| **CLIP normal gallery only** | **79.4368** |

Per-signal mean:

| Signal | PromptAD score | CLIP gallery | ResNet18 gallery |
|---|---:|---:|---:|
| burst | 46.4851 | 75.7335 | 55.0101 |
| chirp | 48.6451 | 76.0282 | 66.4049 |
| dsss | 59.1777 | 82.5171 | 87.4986 |
| pulse | 39.4343 | 83.4684 | 72.4786 |

Result CSV: `results_cls_public_rf_dual_gallery.csv`

Summary: `summary.json`
