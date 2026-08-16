# TTA Positioning: No-TTA Controls

## Protocol

This is the no-TTA control required by `docs/research/TTA定位与复核计划.md`.
TTA remains available in the evaluators and is not removed; the default formal
path is identity-only. Explicit TTA replay keeps the legacy reference-view
protocol, while the no-TTA support-only gate uses original support patch
leave-one-out references.

Frozen configuration:

- ViT layer1+layer2 normalized patch features;
- 50% farthest-first coreset and dataset-specific fixed NN settings;
- ResNet18 ImageNet layer3 complete local memory;
- support-only confidence gate: `q=0.8`, `T=2.5`, `alpha=2.5`;
- existing support manifests, test paths, and seed 111;
- no external baseline rerun.

## Results

All values are percentages: AUROC / AUPRC / FPR@95%TPR.

### In-house RF

| Branch | Result |
|---|---:|
| ViT-only | 91.0268 / 80.0926 / 24.4496 |
| CNN-only | 90.0500 / 78.1616 / 34.9398 |
| Confidence fusion | **91.4071 / 80.1784 / 23.5834** |

The run uses the formal 60-cell manifest with 50% farthest coreset and 5-NN.

### Public RF

| Support | ViT-only | CNN-only | Confidence fusion |
|---|---:|---:|---:|
| k=1 | 79.9349 / 43.9207 / 48.3229 | 75.0801 / 30.3332 / 59.3607 | **81.4121 / 45.0805 / 46.3073** |
| k=2 | 80.8761 / 43.9045 / 46.9271 | 75.6651 / 31.3277 / 59.6081 | **83.5498 / 45.3001 / 43.9805** |
| k=4 | 81.8158 / 44.3778 / 44.9805 | 75.6368 / 32.4644 / 62.3958 | **84.1797 / 45.7330 / 43.4961** |

The ViT score files reuse the existing exact no-TTA k=1/2/4 runs. New LOO
reference files and CNN scores were generated for this control.

### OFDMA target-scene

| Shot | ViT-only | CNN-only | Confidence fusion |
|---|---:|---:|---:|
| 1 | 84.6217 / 88.0635 / 69.2667 | 69.4443 / 73.9300 / 84.5667 | 84.6217 / 88.0635 / 69.2667 |
| 2 | 89.0097 / 91.5743 / 57.6667 | 75.4143 / 79.4207 / 80.1333 | 89.0220 / 91.5826 / 57.7333 |
| 4 | 92.2470 / 94.1327 / 49.4000 | 79.6563 / 83.1076 / 75.4333 | 92.2327 / 94.1219 / 49.8000 |

The no-TTA gate uses original support patch leave-one-out references. Fusion
does not produce a consistent improvement over ViT-only and is retained as an
internal ablation conclusion.

### FedJam

| Shot | ViT-only | CNN-only | Confidence fusion |
|---|---:|---:|---:|
| 1 | 84.2379 / 94.9238 / 81.4444 | 75.1162 / 91.0049 / 84.5556 | 84.2379 / 94.9238 / 81.4444 |
| 2 | 84.4821 / 94.9008 / 77.1111 | 79.3989 / 92.7038 / 79.0556 | 84.5280 / 94.9115 / 76.5556 |
| 4 | 85.0426 / 95.0000 / 72.5000 | 83.2673 / 94.0922 / 69.2778 | 85.0514 / 95.0017 / 72.2222 |

The run covers all 7,200 official test rows. The 1-shot fusion is exactly the
ViT score because the support-only CNN rank gate has only one reference value.

## Outputs

Main output root:
`analysis_outputs/exploratory/20260815_tta_position_no_tta/`

- `inhouse_vit`, `inhouse_cnn`, `inhouse_fusion`;
- `public_cnn_k1/k2/k4`, `public_fusion_k1/k2/k4`;
- `ofdma`;
- `fedjam`;
- corresponding `*_reference_*` directories contain support-only LOO files.

No paper table or README was replaced in this screening step. TTA remains an
auxiliary memory-expansion/ablation option rather than a core contribution.
