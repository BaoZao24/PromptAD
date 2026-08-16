# Frequency-Balanced Reliable Memory Screening

## Protocol

This experiment implements `docs/method/频率均衡可靠记忆库方案.md` without changing
the frozen ViT features or the detector.  RF frequency groups use the patch-grid
**columns** (`gallery_cols`); the total gallery capacity remains `coreset_ratio=0.1`.
All runs use the formal In-house manifest, `paired_tta=none`, and `nn_topk=1` to
match the existing 60-cell baseline.

- B0: global farthest coreset;
- B1: equal frequency-group allocation, no reliability filtering;
- B2: within-frequency reliability filtering, then farthest coreset;
- B3: reliability filtering plus equal frequency-group allocation;
- reliability neighbors: 5;
- filtering ratios: 5% and 10%.

## In-house RF Results

Macro average over 60 cells. AUROC/AUPRC are higher-is-better; FPR95 is
lower-is-better.

| Variant | Drop | AUROC | AUPRC | FPR95 |
|---|---:|---:|---:|---:|
| B0 global farthest | 0% | 92.3211 | 82.1395 | 22.5535 |
| B1 frequency balanced | 0% | 92.1606 | 81.0130 | 22.1630 |
| B2 reliable only | 5% | 89.6374 | 76.9081 | 27.1755 |
| B2 reliable only | 10% | 88.8709 | 75.8642 | 29.0012 |
| B3 frequency + reliable | 5% | 89.2799 | 76.7747 | 26.5376 |
| B3 frequency + reliable | 10% | 89.1265 | 75.9704 | 29.0404 |

## Decision

The proposed memory cleanup does not pass the first-stage criterion:

- B1 reduces FPR95 by only 0.3905 pp, but reduces AUROC by 0.1606 pp and AUPRC by
  1.1265 pp;
- reliability filtering removes useful normal variation rather than isolated noise;
- combining filtering and balancing is worse than either baseline behavior;
- no variant has a positive overall improvement, so Public RF/FedJam follow-up and
  multi-seed expansion are not justified.

The implementation remains available as an exploratory option in
`tools/eval_cls_vit_patchcore_gallery.py`:
`--memory-selection {farthest,frequency_balanced,reliable,frequency_reliable}`.
The default `farthest` path reproduces the formal baseline.
