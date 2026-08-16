# Frequency-Local Feature Interpolation Screening

## Protocol

This experiment implements `docs/method/频率局部扩充记忆库方案.md` as a
support-only feature-space adapter.  It keeps the frozen CLIP features and
does not transform query images.  RF frequency groups use patch-grid columns.
Pairs are selected greedily so each base feature participates at most once.

Two base-memory interpretations were checked:

- **Fair current-capacity check**: first use the formal `0.1` farthest coreset,
  then append virtual features; this isolates the interpolation effect at the
  current memory scale.
- **Literal full-support check**: keep all raw support patches, then append
  virtual features; this follows the wording that no original feature is
  removed, but has a larger memory than the formal baseline.

## Single-Cell Screening

Cell: `burst_signal/WeaponMuseum_spectrum/m10db`.

| Variant | Base ratio | Virtual ratio | AUROC | AUPRC | FPR95 |
|---|---:|---:|---:|---:|---:|
| M0 original | 0.1 | 0% | 97.3188 | 96.6978 | 31.1321 |
| M1 global mutual | 0.1 | 50% | 97.6912 | 96.8487 | 28.3019 |
| M2 same-frequency ordinary | 0.1 | 50% | 97.1698 | 96.5923 | 26.4151 |
| M3 same-frequency mutual | 0.1 | 25% | 97.7408 | 96.7901 | 27.3585 |
| M0 original raw | 1.0 | 0% | 97.6415 | 96.8015 | 35.8491 |
| M3 raw | 1.0 | 25% | 96.8471 | 96.4305 | 45.2830 |

The single-cell M3 gain was not treated as evidence because previous
single-cell TTA gains did not generalize.

## Formal In-house Result

The fair-capacity M3 configuration was evaluated over all 60 formal cells:

| Variant | AUROC | AUPRC | FPR95 |
|---|---:|---:|---:|
| M0 formal baseline | 92.3211 | 82.1395 | 22.5535 |
| M3, 25% virtual features | 91.6199 | 81.6469 | 22.7606 |

Changes are `-0.7012` AUROC, `-0.4926` AUPRC, and `+0.2071` FPR95.
Virtual pair counts were roughly 114–127 per scene after the current-capacity
coreset, so the 50% and 100% targets saturated at the available safe pairs.

## Decision

The same-frequency mutual interpolation is promising in one cell but fails the
formal macro test.  The full-support check also declines, so the result is not
explained only by the current coreset capacity.  The likely failure mode is
that midpoint features enlarge the normal region enough to make some abnormal
patches easier to match.  Do not extend this candidate to Public RF/FedJam or
additional seeds without a new theoretical constraint.

Implementation: `tools/eval_feature_local_interpolation.py`.
