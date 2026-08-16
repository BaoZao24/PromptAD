# SpectraMemAD: Training-Free RF Spectrogram Anomaly Detection

> **English** | [中文](./README_zh.md)

This repository uses the visual encoder of PromptAD as an implementation base for
training-free radio-frequency (RF) spectrogram anomaly detection.

The current research target is **few-shot anomaly detection**.

## Current Method

The paper uses one unified main comparison across all four datasets: classical
spectrogram-domain communication statistics, completed external anomaly-detection baselines,
and the proposed method. The two visual branches are internal ablations and are
reported separately; coverage and references are listed in
[the experiment section](./docs/paper/论文实验部分.md).

```text
CNN-only (internal ablation)
= ResNet18 layer3 local normal gallery
+ nearest-neighbour anomaly score

Classical spectrum statistics
= ED / spectral entropy / spectral flatness
+ spectral kurtosis / CA-CFAR
```

The current method is:

```text
Confidence Fusion (ViT+CNN)

ViT Normal Gallery
+ CNN Local Gallery
+ confidence fusion
```

Branch roles:

| Branch | Role |
|---|---|
| ViT Normal Gallery | Scores structural patch-level deviation from few-shot normal spectrograms. |
| CNN Local Gallery | Scores local texture and energy-detail deviation from few-shot normal spectrograms. |
| Confidence fusion | Uses only the CNN rank and CNN-over-ViT advantage calibrated from normal support to form a non-negative correction; no test-batch statistics or fusion parameters are exposed. |

External PatchCore [12](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf) and the classical spectrum statistics are formal comparison
methods; CNN-only is an internal branch ablation and is not the external PatchCore
baseline. The current method uses visual ViT and CNN normal-memory evidence. The repository
retains the upstream PromptAD code for implementation compatibility, but it is
not a paper comparison row.

Detailed method description: [docs/paper/现有方案介绍.md](./docs/paper/现有方案介绍.md).

## Experiment Protocol

Current experiments must use a few-shot normal-only protocol:

```text
support / gallery:
  target normal samples only
  dataset-specific normal support: one image per RF frequency band and 1/2/4-shot
  normal observations for OFDMA/FedJam
  no target abnormal samples

test:
  independent test normal samples
  test abnormal samples
  evaluated by anomaly type and JSR
```

Important constraints:

- Do not use target abnormal samples for training or gallery construction.
- Use the cited external anomaly-detection baselines and spectrogram-domain communication statistics as comparison baselines. SAIFE is a spectrum-domain generative baseline rather than a generic visual method. The main comparison keeps one representative VAE row (IAD-PER), together with ED, CA-CFAR, SCSE, KLD-Ref, SAIFE, UDMA, PatchCore, WinCLIP, and Ours; other completed baselines remain supplementary.
- Keep ViT/CNN single branches as internal ablations.
- Use the same core method names across In-house RF, Public RF, OFDMA, and FedJam. The current four-dataset comparison tables contain the completed methods; a future unrun method must be marked explicitly rather than assigned a placeholder score.

## Current Result Status

Some frozen Ours protocols retain paired TTA as an auxiliary normal-memory
expansion. It is not a core method innovation; every paper table must state the
protocol setting and retain a matched no-TTA ablation. External baselines do
not need to be rerun solely because of this documentation decision.

The current formal support-only RF run achieves
**91.97/81.80/22.00** (AUROC/AUPRC/FPR@95%TPR) on In-house RF. Public RF uses the main
`k=1/2/4-per-frequency` protocol; Ours obtains **81.74/82.97/83.74** AUROC across the
three k values. The complete three-metric tables are in [`docs/paper/论文实验部分.md`](./docs/paper/论文实验部分.md),
Tables 1 and 1b.

The RF results below are from the current formal ViT+CNN confidence-fusion method;
the exploratory power-residual branch is not part of the main table.

| Dataset | Ours: Confidence Fusion AUROC |
|---|---:|
| In-house RF (self-measured + synthetic interference) | **91.97** |
| Public RF [14](https://doi.org/10.1007/s11036-009-0199-9) | **81.74 / 82.97 / 83.74** (k=1/2/4) |

ViT-only、CNN-only、Direct OR 等内部支线不作为独立主方法，统一放在论文实验部分的
内部消融表和图中。

Dataset provenance is cited directly here: In-house RF is collected through this project’s
own RF measurements, and synthetic interference is injected into those normal recordings to
construct the abnormal samples; it is a self-measured derived benchmark rather than an external
public dataset. Public RF uses normal spectrum-occupancy
measurements from [14](https://doi.org/10.1007/s11036-009-0199-9), whose public measurement
data are available from the cited data portal; the anomaly samples are derived by injecting
interference. OFDMA is derived by adapting the released paper, source code, and data under
[15](https://arxiv.org/abs/2606.02102). FedJam is used as an unchanged public dataset under
[13](https://arxiv.org/abs/2508.09369).

The formal In-house RF protocol contains four scenes (`WeaponMuseum_spectrum`,
`Playground_spectrum`, `TimeSquare_spectrum`, and `Gymnasium_spectrum`), five synthetic
injection types (burst, chirp, DSSS, pulse, deceptive), and 60 signal/scene/strength cells.
It uses 24 candidate frequency bands per scene for `per_frequency` support. The formal strength
sets are −10/−20/−30 dB for burst/chirp/DSSS, −20/−30/−40 dB for pulse, and
strong/medium/weak for deceptive; `wideband_pulse` is not part of the five-type main comparison.
Detailed scene and injection parameters are recorded in
[`docs/paper/论文实验部分.md`](./docs/paper/论文实验部分.md), Section 4.1.1.

Dataset references:

13. I. Panitsas, I. Ofeidis, and L. Tassiulas, “FedJam: Multimodal Federated Learning Framework for Jamming Detection,” [paper](https://arxiv.org/abs/2508.09369), [dataset](https://huggingface.co/datasets/panitsasi/FedJam), [code](https://github.com/panitsasi/fedJam).
14. M. Wellens and P. Mähönen, “Lessons Learned from an Extensive Spectrum Occupancy Measurement Campaign and a Stochastic Duty Cycle Model,” [DOI](https://doi.org/10.1007/s11036-009-0199-9), [measurement data portal](http://download.mobnets.rwth-aachen.de).
15. A. Schösser, M. Salehi, S. Ma, P. Schulz, and G. Fettweis, “Spectrum Anomaly Detection in OFDMA Systems: Simulation Framework and Benchmark Dataset,” [paper](https://arxiv.org/abs/2606.02102), [code](https://github.com/akdd11/ofdma-spectrum-anomalies-simulation), [Zenodo data](https://doi.org/10.5281/zenodo.20341906).

The complete self-RF rerun, including VAE, SAIFE, Deep SVDD, PaDiM, STFPM,
WinCLIP, and official PatchCore, is recorded in
[`analysis_outputs/20260725_target_scene_self_rf_baselines_seed111/README.md`](./analysis_outputs/20260725_target_scene_self_rf_baselines_seed111/README.md).

See
[`analysis_outputs/20260713_final_visual_metrics/README.md`](./analysis_outputs/20260713_final_visual_metrics/README.md)
for the verified result files.

The public-RF five-type macro table and the wideband per-cell results are in
[`analysis_outputs/20260730_public_rf_wideband_formal_seed111/summary/`](./analysis_outputs/20260730_public_rf_wideband_formal_seed111/summary/).

For the formal OFDMA target-scene cold-start protocol, scores are computed for each
of the 21 SUs, followed by maximum pooling and the support-only ViT+CNN confidence fusion.
Ours results on 30 independent test scenes are:

| Shot | AUROC | AUPRC | FPR@95%TPR |
|---:|---:|---:|---:|
| 1 | 85.30 | 88.80 | 69.77 |
| 2 | 89.53 | 92.01 | 57.63 |
| 4 | 92.75 | 94.55 | 45.90 |

The complete protocol, per-method results, and unified table are documented in
[`analysis_outputs/20260731_ofdma_v2_realistic_unified/README.md`](./analysis_outputs/20260731_ofdma_v2_realistic_unified/README.md).
The subsequently completed official PatchCore comparison under the identical
target-scene protocol is in
[`analysis_outputs/20260810_ofdma_patchcore_formal/README.md`](./analysis_outputs/20260810_ofdma_patchcore_formal/README.md):
70.14/76.78/86.03, 75.83/81.29/81.37, and 81.81/86.14/72.50
(AUROC/AUPRC/FPR@95%TPR for 1/2/4-shot).

The FedJam 1/2/4-shot image-only supplement and the unified main comparison
(ED, CA-CFAR, SCSE, KLD-Ref, IAD-PER, SAIFE, UDMA, PatchCore, WinCLIP, and Ours)
are documented in
[`analysis_outputs/20260810_fedjam_visual_baselines_formal/`](./analysis_outputs/20260810_fedjam_visual_baselines_formal/)
and [`analysis_outputs/20260810_fedjam_traditional_fewshot_formal/`](./analysis_outputs/20260810_fedjam_traditional_fewshot_formal/).
The paper tables are in [`docs/paper/论文实验部分.md`](./docs/paper/论文实验部分.md), Tables 9–11.

## Installation

```bash
conda create -n prompt_ad python=3.10
conda activate prompt_ad
bash install.sh
```

Key dependencies include PyTorch, `open_clip_torch`, `timm`, `transformers`, `opencv-python`, `scikit-learn`, `pandas`, `loguru`, and `tqdm`.

## Dataset Layout

Expected RF data root:

```text
/mnt/data/wangbei/data/datasets/
├── normal/
│   ├── WeaponMuseum_spectrum/
│   ├── Playground_spectrum/
│   ├── TimeSquare_spectrum/
│   └── Gymnasium_spectrum/
├── burst/
├── chirp/
├── dsss/
├── pulse/
└── wideband_pulse/
```

Each signal dataset contains per-scene folders with:

```text
normal/{jsr}/
abnormal/{jsr}/
groundtruth/{jsr}/
```

## Useful Commands

PatchCore-style CNN local normal-memory baseline:

```bash
python tools/eval_patchcore_cls.py \
  --protocol rf_target \
  --normal-sampling per_frequency \
  --output-root analysis_outputs/patchcore_fewshot_baseline
```

Confidence fusion from current ViT/CNN scores:

```bash
python tools/eval_cls_aux_cnn_gallery.py \
  --protocol public_rf \
  --normal-sampling per_frequency \
  --output-root analysis_outputs/current_public_aux_cnn

python tools/eval_cls_dual_visual_evidence_fusion.py \
  --protocol public_rf \
  --vit-score-dir analysis_outputs/current_public_vit/scores \
  --cnn-score-dir analysis_outputs/current_public_aux_cnn/scores \
  --vit-reference-dir analysis_outputs/current_public_vit_reference \
  --cnn-reference-dir analysis_outputs/current_public_cnn_reference \
  --output-root analysis_outputs/current_public_confidence_fusion
```

The auxiliary CNN has one fixed ResNet18-layer3 configuration across datasets. The
fusion entry defaults to support-only confidence calibration and has no CNN-selection or
fusion-tuning arguments. Existing output keys and compatibility module names containing
`gate` are implementation identifiers, not the paper method name.

## Useful Documentation

- [Current method](./docs/paper/现有方案介绍.md)
- [Docs index](./docs/README.md)
- [Score mechanism](./docs/method/分数机制说明.md)
- [Prompt mechanism](./docs/method/提示词机制说明.md)

## Notes

The upstream MVTec and VisA interfaces are still present for compatibility, but the active RF research line is the few-shot target-normal protocol described above.

[12]: https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf
