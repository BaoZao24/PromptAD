# Candidate Method Screening: SubspaceAD, RAD, RareCLIP

## Purpose

Screen three recent training-free/few-shot industrial anomaly detection ideas
as possible replacements for the spectral TTA contribution.  The screening
uses the formal In-house target-scene manifest and the existing frozen CLIP
ViT-B-16-plus-240 layer1+layer2 patch features.  No target anomalies are used
to build the support memory.

The adapters are intentionally minimal rather than claims of reproducing the
original DINOv2/DINOv3 or trained RareCLIP checkpoints:

- **SubspaceAD**: PCA subspace of normal patch features; image score is the
  maximum patch reconstruction residual.
- **RAD**: global image-feature retrieval of `k_image` support images followed
  by patch nearest-neighbor matching within those images.
- **RareCLIP-style rarity**: normal patch prototype density is used to weight
  patch nearest-neighbor distances.  This is an offline normal-only proxy for
  the original online RareCLIP memory mechanism.

## Results

Formal In-house macro average over 60 cells:

| Method | Setting | AUROC | Delta vs. formal baseline |
|---|---|---:|---:|
| Existing ViT memory | formal no TTA | 92.3211 | -- |
| SubspaceAD adapter | PCA variance 0.70 | 91.0820 | -1.2391 pp |
| RAD adapter | `k_image=4` | 90.4389 | -1.8822 pp |
| RAD adapter | `k_image=8` | 90.8505 | -1.4706 pp |
| RareCLIP-style adapter | density top-k 10, alpha 0.25 | 89.6207 | -2.7004 pp |

The single `burst_signal/WeaponMuseum_spectrum/m10db` cell was positive for
SubspaceAD (97.3188) and RAD with four references (97.2691), but this did not
survive the 60-cell macro average.  The full protocol result is therefore the
decision criterion.

## Decision

None of the three candidates is strong enough to replace the current main
method.  The apparent single-cell SubspaceAD gain is not stable, RAD mostly
benefits from using more support patches rather than its retrieval step, and
the rarity proxy is consistently worse.  Do not deepen these candidates or
present them as innovations without a separate, properly trained reproduction.

## Source Code

Downloaded under `references/` for inspection:

- `references/SubspaceAD` — CVPR 2026
- `references/RAD` — ICML 2026
- `references/RareCLIP` — ICCV 2025

Screening entry point: `tools/eval_candidate_methods.py`.
