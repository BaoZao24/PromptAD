# Published spectrum-baseline experiment log

Date: 2026-08-11

## Objective

Implement and evaluate four published spectrum-anomaly baselines without
changing the frozen test pools or the existing method:

- IAD-PER (Tian et al., 2022);
- KLD-Ref and ICA-Frozen, adapted from Afgani et al. (2010);
- UDMA, reimplemented from Qi et al. (2024).

The hypotheses are that the paper-specific PER score is a more faithful VAE
baseline than generic MSE, that information-theoretic scores provide useful
CPU-only communication baselines, and that UDMA can be evaluated as a
paper-driven reimplementation despite the missing official code.

## Frozen evaluation protocols

| Dataset | Support/test protocol | Aggregation |
|---|---|---|
| In-house RF | Existing target-scene support manifest; 60 fixed cells | Cell-level macro |
| Public RF | Fixed disjoint k-per-frequency manifests for k=1/2/4 | 15-cell macro |
| OFDMA | Target-scene cold start, 1/2/4-shot, 30 test scenes | Max over 21 SUs, then scene macro |
| FedJam | Nested benign-only 1/2/4-shot, full independent test | Image-level overall and per-attack |

All normal-reference fitting and checkpoint selection must use support data
only. Test labels are metrics-only. Report AUROC, AUPRC, and FPR@95%TPR.

## Implementation boundary

- IAD-PER uses the official VAE/PER definition. Existing VAE-MSE behavior is
  preserved as a separate score mode.
- KLD-Ref freezes a support-fitted histogram and uses Krichevsky--Trofimov
  pseudocount 0.5. It is a normal-reference spectrogram adaptation, not the
  original periodic twin-window experiment.
- ICA-Frozen freezes the support histogram to prevent test-stream adaptation.
  It is a support-only spectrogram adaptation of the paper's online ICA.
- UDMA follows the published teacher/AE/MemAE discrepancy design. Any choices
  required because the paper omits a pretrained-teacher identity or dynamic
  memory dimensions are recorded in the implementation and protocol JSON.

## Resource policy

GPU experiments run one process on one card with conservative workers and
batch size. They do not start while all available GPUs are heavily occupied or
thermally saturated. CPU-only KLD/ICA experiments run first.

## Runs

Commands, output directories, metrics, and conclusions are appended here after
each smoke or formal run. Failed smoke tests remain recorded when informative.

### Core verification

- IAD-PER: `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m
  unittest tests/test_iad_per.py -v` — 6/6 passed.
- KLD-Ref/ICA-Frozen: `PYTHONDONTWRITEBYTECODE=1 python -m unittest
  tests/test_information_theoretic_spectral.py -v` — 8/8 passed.
- UDMA: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m
  unittest tests/test_udma.py -v` — 3/3 passed.

### IAD-PER smoke runs

OFDMA target-scene, one test scene and one normal plus one anomaly observation
per type (all 21 SUs retained), CPU:

```bash
CUDA_VISIBLE_DEVICES='' python tools/eval_ofdma_target_scene_baselines.py \
  --method iad_per \
  --output-root analysis_outputs/20260811_smoke_iad_per_ofdma \
  --use-cpu --num-workers 0 --shots 1 --scene-ids test_000 \
  --max-normal-observations 1 --max-anomaly-observations-per-type 1 \
  --baseline-batch-size 8
```

The full 100-epoch smoke completed and produced finite SU and observation
scores plus all three metrics.

Protocol correction: this first smoke and three subsequently started formal
runs used the evaluator's historical v1 default directory. The paper's OFDMA
main tables use `ofdma-target-scene-coldstart-v2-realistic`; the v1 formal jobs
were therefore stopped, their partial outputs are excluded from every table,
and all OFDMA results below are rerun with the v2-realistic path supplied
explicitly. This correction changes no RF or FedJam result.

FedJam, one test image per class, one training epoch, CPU:

```bash
CUDA_VISIBLE_DEVICES='' python tools/eval_fedjam_visual_baselines.py \
  --method iad_per \
  --output-root analysis_outputs/20260811_smoke_iad_per_fedjam \
  --use-cpu --num-workers 0 --shots 1 --max-test-per-label 1 \
  --epochs 1 --batch-size 4
```

The first attempt exposed that FedJam's saved labels are multiclass. The VAE
adapter was corrected to reduce them to benign versus any jammer only inside
its diagnostic metric function, while preserving the original labels in the
score archive. The repeated smoke then completed successfully.

### IAD-PER formal runs

In-house RF was started CPU-only because every GPU was simultaneously at
89--99% utilization and 81--88 degrees C:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' \
python tools/eval_vae_cls.py \
  --protocol rf_target --score-mode iad_per \
  --output-root analysis_outputs/20260811_iad_per_rf_target_formal \
  --use-cpu --normal-sampling per_frequency \
  --support-manifest \
    analysis_outputs/20260728_rf_five_type_formal_seed111/support_manifest.json \
  --epochs 100 --batch-size 64 --num-workers 2 --log-every 25
```

Result: 60/60 cells completed; AUROC 52.9988, AUPRC 35.7167, FPR95
79.4777. This is worse than the existing VAE-MSE row and is retained as a
negative result rather than tuned against test labels.

Public RF used the three explicit nested manifests and the same fixed test
hash `709f6485206389298d86d2838b15251c70d1a1a0c78b2dcf7724c5832c9a1ff9`.
The k=1/2/4 results were respectively:

| k | AUROC | AUPRC | FPR95 |
|---:|---:|---:|---:|
| 1 | 52.8897 | 9.1074 | 90.7982 |
| 2 | 52.7015 | 8.3726 | 91.2552 |
| 4 | 51.8031 | 8.0767 | 91.4609 |

FedJam full-test results were:

| shot | AUROC | AUPRC | FPR95 |
|---:|---:|---:|---:|
| 1 | 48.7703 | 73.2371 | 94.2222 |
| 2 | 42.7736 | 69.8749 | 96.6111 |
| 4 | 50.5101 | 74.3480 | 94.5000 |

### KLD-Ref and ICA-Frozen formal results completed so far

In-house RF (60-cell macro):

| method | AUROC | AUPRC | FPR95 |
|---|---:|---:|---:|
| KLD-Ref | 55.6623 | 36.1301 | 82.7117 |
| ICA-Frozen | 58.5659 | 33.2421 | 85.8206 |

FedJam full-test pooled normal-versus-any-jammer results:

| method | shot | AUROC | AUPRC | FPR95 |
|---|---:|---:|---:|---:|
| KLD-Ref | 1 | 55.4542 | 80.8815 | 95.6111 |
| KLD-Ref | 2 | 49.5758 | 74.3644 | 95.7222 |
| KLD-Ref | 4 | 50.4303 | 74.7617 | 92.8333 |
| ICA-Frozen | 1 | 49.3058 | 74.9057 | 98.1667 |
| ICA-Frozen | 2 | 49.3058 | 74.9057 | 98.1667 |
| ICA-Frozen | 4 | 49.3058 | 74.9057 | 98.1667 |

Public RF k=1: KLD-Ref achieved 55.0091/11.8959/91.1198 and ICA-Frozen
59.2573/6.8987/85.7174 (AUROC/AUPRC/FPR95).

Public RF k=2 and k=4 subsequently completed on the same test hash:

| method | k | AUROC | AUPRC | FPR95 |
|---|---:|---:|---:|---:|
| KLD-Ref | 2 | 55.0948 | 11.8724 | 91.5391 |
| KLD-Ref | 4 | 54.9720 | 11.8236 | 92.0690 |
| ICA-Frozen | 2 | 59.4711 | 6.9251 | 83.7474 |
| ICA-Frozen | 4 | 59.6423 | 6.9496 | 81.7188 |

### UDMA implementation choice and completed results

The paper does not identify its pretrained reference network. Main UDMA runs
therefore use a frozen torchvision ImageNet-1K ResNet18. A deterministic
spectral-statistics reference was also run as an implementation-sensitivity
check and is not used as the main-table result.

In-house RF with frozen ResNet18 reference completed all 60 cells: AUROC
57.9462, AUPRC 38.7719, FPR95 75.7646. The spectral-reference sensitivity run
was very similar (57.9268/39.6145/75.6932).

FedJam full-test UDMA-ResNet18 results:

| shot | AUROC | AUPRC | FPR95 |
|---:|---:|---:|---:|
| 1 | 50.4528 | 73.1616 | 90.2778 |
| 2 | 46.6807 | 71.2857 | 92.3333 |
| 4 | 47.6847 | 71.7812 | 91.6111 |

Public RF UDMA-ResNet18 results:

| k | AUROC | AUPRC | FPR95 |
|---:|---:|---:|---:|
| 1 | 52.1459 | 9.0591 | 92.5234 |
| 2 | 52.1957 | 9.0311 | 92.7292 |
| 4 | 51.9064 | 9.5620 | 91.9922 |

### Corrected OFDMA v2-realistic formal runs

After the protocol audit, the OFDMA runs were restarted with the explicit
dataset root
`/mnt/data/wangbei/data/ofdma-target-scene-coldstart-v2-realistic`. To keep
CPU usage bounded while avoiding a long serial queue, the 30 target scenes
were split into six five-scene shards per method. Each shard used one CPU
thread, and the completed shards were merged with
`tools/merge_ofdma_method_chunks.py`. The merged files were checked for all
30 scenes × 3 shots × 6 scopes before entering the main tables.

| method | 1-shot AUROC/AUPRC/FPR95 | 2-shot AUROC/AUPRC/FPR95 | 4-shot AUROC/AUPRC/FPR95 |
|---|---:|---:|---:|
| KLD-Ref | 58.72 / 67.06 / 95.40 | 58.90 / 67.14 / 95.40 | 58.78 / 67.13 / 96.10 |
| ICA-Frozen | 48.35 / 49.28 / 99.80 | 47.00 / 48.72 / 99.90 | 46.57 / 48.53 / 99.60 |
| IAD-PER | 53.08 / 56.44 / 93.83 | 53.89 / 57.53 / 93.20 | 55.48 / 59.23 / 92.53 |
| UDMA-ResNet18 | 57.77 / 62.85 / 92.50 | 57.43 / 62.78 / 92.90 | 57.69 / 63.23 / 93.23 |

The unified OFDMA bundle is
`analysis_outputs/20260811_ofdma_v2_realistic_unified_published/`; it
contains no PromptAD row and records the v2-realistic protocol in
`protocol.json`. The earlier v1 partial runs remain diagnostic artifacts only
and are excluded from all reported tables and figures.
