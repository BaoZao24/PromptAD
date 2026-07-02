# Spectrum CLS Dual Gallery Evaluation

This experiment evaluates the current dual-gallery CLS method on `datasets/spectrum`.

Protocol: for each category, `train/good` builds the normal galleries; `test/good` and `test/bad` are evaluated for Image-AUROC. No SEG metrics are computed.

Categories:

```text
16QAM / CHIRP / GMSK / QPSK
```

Macro Image-AUROC over 4 categories:

| Method | Image-AUROC |
|---|---:|
| PromptAD score with current checkpoint | 70.9360 |
| CLIP normal gallery | 95.0072 |
| ResNet18 layer3 normal gallery | 99.7432 |
| **Best dual fusion in sweep** | **99.7609** |

Per-category Image-AUROC:

| Category | PromptAD score | CLIP gallery | ResNet18 gallery | Best dual fusion |
|---|---:|---:|---:|---:|
| 16QAM | 71.9231 | 95.0044 | 99.6285 | 99.6350 |
| CHIRP | 68.5028 | 96.2324 | 99.9306 | 99.9465 |
| GMSK | 81.0445 | 94.2265 | 99.7911 | 99.8208 |
| QPSK | 62.2737 | 94.5654 | 99.6227 | 99.6411 |

Result CSV: `results_cls_spectrum_dual_gallery.csv`

Summary: `summary.json`
