# PromptAD — Few-Shot Anomaly Detection for RF Spectrograms

> **English** | [中文](./README_zh.md)

This repository adapts [PromptAD (CVPR 2024)](http://arxiv.org/abs/2404.05231) for **few-shot anomaly detection on radio-frequency (RF) spectrograms**.
The model is trained with only **normal (clean) spectrogram patches** and evaluated against anomalies caused by `burst / chirp / dsss / wideband_pulse` jamming.

> The MVTec / VisA interfaces are kept from the upstream code. The main contributions of this fork are: RF prompt adaptation, multi-channel spectrogram input construction, score fusion, cross-library transfer, and a set of lightweight adapter modules (VCPA / Visual Adapter / Visual LoRA / RN50 guidance).

---

## 1. Main Pipeline & Default Setup

The current production-line configuration:

```text
prompt_mode      = rf                                # RF-domain prompt templates
input_mode       = morph_fusion_gray_residual_a01    # gray_contrast + weak_residual(α=0.1) + original_gray
cls_score_mode   = text_only                         # textual anomaly score only
split_mode       = normal_75_25                      # 3/4 normal for training, 1/4 normal + all abnormal for test
seed             = 111
k-shot           = 1
Epoch            = 50
backbone         = ViT-B-16-plus-240 (laion400m_e32)
```

Default benchmark targets (excluding `wideband_pulse`):

| Signal | Noise level (JSR) |
|---|---|
| `burst_signal`, `chirp_signal`, `dsss_signal` | `m10db`, `m20db`, `m30db` |
| `wideband_pulse` | `m20db`, `m30db`, `m40db` |

Four fixed test scenes: `WeaponMuseum_spectrum`, `Playground_spectrum`, `TimeSquare_spectrum`, `Gymnasium_spectrum`.

Background and motivation: see [`现有方案介绍.md`](./现有方案介绍.md), [`improve.md`](./improve.md), [`提示词机制说明.md`](./提示词机制说明.md), [`分数机制说明.md`](./分数机制说明.md) (in Chinese).

---

## 2. Installation

```bash
conda create -n prompt_ad python=3.10
conda activate prompt_ad
bash install.sh
```

Key dependencies: PyTorch (CUDA 11.8), `open_clip_torch`, `timm`, `transformers`, `opencv-python`, `scikit-learn`, `pandas`, `loguru`, `tqdm`.

---

## 3. Dataset Layout

```
/mnt/data/wangbei/data/datasets/
├── normal/                                # Normal (clean) training patches; shared across all signal types
│   ├── WeaponMuseum_spectrum/
│   ├── Playground_spectrum/
│   ├── TimeSquare_spectrum/
│   └── Gymnasium_spectrum/
├── burst/                                 # Burst jamming test set
│   └── {scene}/
│       ├── normal/{noise_level}/          # Clean test patches
│       ├── abnormal/{noise_level}/        # Anomalous patches
│       └── groundtruth/{noise_level}/     # Pixel-level GT masks
├── chirp/                                 # Chirp jamming (same structure)
├── dsss/                                  # DSSS jamming (same structure)
├── wideband/                              # Wideband pulse (same structure, different JSR levels)
└── deceptive/                             # Deceptive (stealthy) jamming, 0db only
```

All loaders live in [`datasets/`](./datasets/) and are constructed through `datasets.__init__.get_dataloader_from_args(...)`.
Train/test splitting is controlled by `--split-mode`:

- `legacy` — original PromptAD few-shot split;
- `normal_75_25` — repository default. 3/4 normal patches for training; the rest (plus all abnormal) for test. Avoids leakage.

---

## 4. Project Layout

```
PromptAD/
├── train_cls.py              # Single-run image-level training / evaluation (CLS task)
├── train_seg.py              # Pixel-level (SEG) training entry
├── test_cls.py / test_seg.py # Eval-only (no retraining)
├── run_cls.py / run_seg.py   # Legacy batch scripts (mvtec/visa/spectrum/sample)
├── run_rf_split_all.py       # MAIN batch script: 3 signals × 4 scenes × 3 JSR × N GPU
├── plot_scoremap.py          # Input / GT / score-map triplet visualization
│
├── PromptAD/
│   ├── ad_prompts.py         # Prompt templates (generic / rf / rf_domain / rf_signal_structured ...)
│   ├── model.py              # All core model classes (see §5)
│   └── CLIPAD/               # OpenCLIP-derived vision/text encoders
│
├── datasets/
│   ├── __init__.py           # Dataloader factory + main entry get_dataloader_from_args
│   ├── dataset.py            # CLIPDataset base class (shared loading / cropping)
│   ├── burst_signal.py       # burst loader (with normal_75_25 splitter)
│   ├── chirp_signal.py       # chirp loader
│   ├── dsss_signal.py        # DSSS loader
│   ├── wideband_pulse.py     # wideband loader (PNG / NPY variants)
│   ├── deceptive_signal.py   # deceptive jamming
│   ├── rf_spe_png.py         # Generic RF PNG dataset
│   ├── spectrum.py / sample.py
│   ├── rf_split_utils.py     # normal_75_25 splitting helpers
│   ├── seeds_mvtec/, seeds_visa/  # Upstream few-shot seeds
│   └── mvtec.py / visa.py / prepare_visa_public.py
│
├── utils/
│   ├── training_utils.py     # Output dir helpers, setup_seed, TripletLoss, ...
│   ├── csv_utils.py          # Result CSV writing
│   ├── metrics.py            # AUROC / AUPRO / metric_cal_img (harmonic fusion)
│   ├── eval_utils.py         # specify_resolution and other eval utilities
│   └── visualization.py      # plot_sample_cv2 score-map output
│
├── tools/                    # Offline analysis & plotting scripts (see §8)
├── experiments/              # Recorded experiments: baseline redef., band-aware, normal_bg, VAE fusion ...
├── result/                   # Auto-generated output root
├── archive_results/          # Archived historical results
│
├── 现有方案介绍.md            # Main pipeline & narrative (zh)
├── improve.md                # Stage-wise progress report (zh)
├── 提示词机制说明.md          # PromptLearner walkthrough (zh)
├── 分数机制说明.md            # Score-fusion mechanism (zh)
└── 跨库训练.md                # Cross-library transfer notes (zh)
```

---

## 5. Core Model Components

[`PromptAD/model.py`](./PromptAD/model.py) is large (~2.8k lines) but factors into three layers:

### 5.1 Input channel construction (multi-channel spectrogram fusion)

To exploit the "frequency-domain structure + weak residual energy" nature of RF spectrograms, the codebase ships several three-channel fusion strategies:

| Class | Description |
|---|---|
| `MorphFusionGrayResidualChannels` (`morph_fusion_gray_residual_a01`) | **Main pipeline**: gray_contrast + weak_residual(α) + original_gray |
| `MorphFusionGrayResEnergyChannels` / `MorphFusionGrayResBandChannels` | Replace weak_residual with energy / band statistics |
| `MorphFusionGaborResidualChannels` family | Gabor directional filtering + residual |
| `MorphFusionPlusChannels` / `MorphFusionDualGradChannels` / `MorphFusionBalancedChannels` | Earlier ablation variants |
| `DSSSWeakResidualChannels` / `DSSSEnergyProfileChannels` ... | Narrowband weak-signal enhancers for DSSS |
| `ChirpDirectionalChannels` / `ChirpRidgeChannels` / `ChirpTrackEnhanceChannels` | Chirp time-frequency ridge enhancers |
| `NormalBgDeviationChannels` | Normal/background deviation normalization (triggered by `'normal_bg' in input_mode`) |
| `LogPowerChannels` | Single-channel log-power compression |

`--input-mode auto` picks a default per dataset; `--input-mode signal_adaptive*` switches by anomaly type at runtime.

### 5.2 Prompt / text branch

- `PromptLearner` — CoOp-style learnable prompts:
  - normal prompt: learnable context + class name;
  - abnormal prompt (handle): handcrafted templates such as `rf_state_anomaly`;
  - abnormal prompt (learned): learnable context + abnormal prefix + class name.
- Prompt template pool is selected by `--prompt-mode`: `generic / rf_domain / rf / legacy / rf_object_agnostic / rf_scene_conditioned / rf_signal_structured`.
- Text-prototype aggregation: `--text-prototype-mode {single, grouped_max, grouped_mean, grouped_meanmax, grouped_softmax}`.

### 5.3 Visual branch & anomaly score

The `PromptAD` main class wires together:

1. **CLIP vision encoder** (default ViT-B-16-plus-240) → patch tokens + cls token;
2. **Feature gallery** built from all normal training samples (`feature_gallery1/2`, `global_features`) for visual distance scoring;
3. **Textual anomaly score** — cls feature vs. (normal − abnormal) text prototypes;
4. **Visual anomaly map** — nearest-neighbor distance from each patch token to the normal gallery;
5. **Image-level fusion** — selected by `--cls-score-mode` (`text_only / visual_topk / visual_topk_max / visual_topk_freq / normal_center / normal_mahalanobis / text_normal_center / text_normal_mahalanobis`), weighted by `α / β / γ`.

Optional extension modules (all off by default, enabled per flag):

| Flag | Description |
|---|---|
| `--visual-adapter` | `ResidualVisualAdapter`: residual bottleneck after frozen CLIP, only the adapter is trained |
| `--visual-lora` | LoRA on CLIP visual transformer attention |
| `--visual-class-prompt` (VCPA) | `VisualClassPromptAdapter`: maps the normal visual prototype into soft class tokens injected into the text branch |
| `--learnable-score-fusion` | `LearnableScoreFusionHead`: BCE-supervised residual correction on top of the text score |
| `--rn50-visual-fusion` | Frozen CLIP-RN50 branch — `global` / `local_topk` scoring; `--rn50-fusion-mode guided_vit` uses RN50 local map to guide ViT patch aggregation |
| `--cnn-vit-mamba-fusion` | `CNNMambaLocalBranch`: CNN + ViT + Tiny-Mamba local branch as auxiliary score and normal-alignment loss |
| `--multiview-fusion` | Online fusion (max / mean / conservative) over `(rgb, spectral_gradient_v2, dsss_weak_residual)` views at evaluation |
| `--stat-fusion` | Linearly fuse a classical spectrogram top-k statistic into the model score |

> Implementations live in `model.py`, e.g. `ResidualVisualAdapter:1306`, `VisualClassPromptAdapter:1325`, `LearnableScoreFusionHead:1351`, `CNNMambaLocalBranch:1420`, `PromptLearner:1438`, `PromptAD:1698`.

---

## 6. Single Training / Evaluation Run

```bash
python train_cls.py \
    --dataset burst_signal \
    --class_name Playground_spectrum \
    --noise-level m10db \
    --k-shot 1 --Epoch 50 --gpu-id 0 \
    --prompt-mode rf \
    --input-mode morph_fusion_gray_residual_a01 \
    --cls-score-mode text_only \
    --split-mode normal_75_25 --normal-train-ratio 0.75 \
    --seed 111 --vis True
```

Common arguments (full list in `train_cls.py:get_args`):

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `mvtec` | `burst_signal / chirp_signal / dsss_signal / wideband_pulse / wideband_pulse_png / rf_spe_png / mvtec / visa / spectrum / sample / deceptive_signal` |
| `--class_name` | `carpet` | Test scene name (`*_spectrum` for the RF tasks) |
| `--noise-level` | `m10db` | JSR level |
| `--k-shot` | `1` | Number of normal samples (under `normal_75_25` it controls the gradient-active subset) |
| `--Epoch` | `50` | Training epochs |
| `--backbone` | `ViT-B-16-plus-240` | CLIP backbone (alternative: `ViT-B-16`) |
| `--seed` | `111` | Random seed |
| `--vis` | `False` | Save score-map visualizations |
| `--prompt-mode` | `rf` | Prompt template pool |
| `--input-mode` | `auto` | Multi-channel spectrogram input mode |
| `--cls-score-mode` | `text_only` | Image-level score fusion |
| `--split-mode` | `legacy` | `legacy / normal_75_25` |
| `--root-dir` | `./result` | Output root |

> The reported metric uses `utils.metrics.metric_cal_img`, which **harmonically** fuses the image-level score with the max of the anomaly map before computing AUROC. See [`分数机制说明.md`](./分数机制说明.md).

---

## 7. Batch Runner (recommended entry for the main pipeline)

```bash
# Default: 3 signals × 4 scenes × 3 JSR levels, parallel over auto-detected GPUs
python run_rf_split_all.py \
    --epochs 50 \
    --prompt-mode rf \
    --input-mode morph_fusion_gray_residual_a01 \
    --cls-score-mode text_only

# Choose specific GPUs / subsets
python run_rf_split_all.py --gpus 0 1 2 3 \
    --datasets burst_signal chirp_signal dsss_signal \
    --epochs 50 --seed 111

# Print the commands without launching
python run_rf_split_all.py --dry-run
```

The script:
1. skips jobs that already wrote a `Seed_111-results.csv` with `i_roc > 0` (unless `--force`);
2. logs each job to `/tmp/rf_split_<dataset>_<scene>_<noise>_gpu<gid>.log`;
3. prints a 4×3 `i_roc` summary per dataset once finished.

For the RF cross-library adapter plan, see [`docs/PromptAD_adapter_experiment_plan.md`](./docs/PromptAD_adapter_experiment_plan.md). The official PromptAD RF cross-library entrypoint still needs to be implemented.

---

## 8. Output Layout & Visualization

```
result/<dataset>/<scene>/<noise>/<split>/k_<shot>/
├── csv/Seed_<seed>-results.csv      # i_roc / p_roc / pro and other metrics
├── checkpoint/                      # PromptLearner / Adapter / Gallery state
├── scores/Seed_<seed>-image_scores.npz   # names / scores / labels / visual_maps / text_scores
└── imgs/                            # plot_sample_cv2 output (when --vis True)
```

Score-map comparison tool:

```bash
# Default: walk ./result, sample one image per (dataset/scene/noise)
python plot_scoremap.py --root-dir ./result --out ./result/scoremap_compare.png

# Filter
python plot_scoremap.py --dataset burst_signal --scene Playground_spectrum --noise m10db
```

`tools/` also hosts numerous offline analysis scripts (PR curves, feature-map exporters, ablation plots). The most-used ones:

- `tools/export_morph_fusion_featuremaps.py` — input feature comparison for the morph_fusion family;
- `tools/run_full_band_analysis.py` — full-band PR analysis;
- `tools/plot_rf_ablation_heatmap.py` — RF ablation heatmap;
- `tools/extract_normal_maps.py` — sample anomaly maps on normal patches;
- `tools/vae_fusion_experiment.py` — offline VAE-fusion experiment.

---

## 9. Recorded Experiments

- [`experiments/baseline_redefinition/`](./experiments/baseline_redefinition/) — Original PromptAD (`legacy` prompt + RGB) vs. the current RF pipeline; clarifies that "RGB baseline ≠ original PromptAD".
- [`experiments/band_aware_scoring/`](./experiments/band_aware_scoring/) — Band-aware patch scoring.
- [`experiments/normal_bg_deviation/`](./experiments/normal_bg_deviation/) — Ablation of `NormalBgDeviationChannels`.
- [`experiments/vae_fusion/`](./experiments/vae_fusion/) — VAE-fusion branch.
- [`experiments/failed_directions_summary.md`](./experiments/failed_directions_summary.md) — Summary of directions that did **not** work.
- [`docs/PromptAD_adapter_experiment_plan.md`](./docs/PromptAD_adapter_experiment_plan.md) — RF cross-library adapter experiment plan.

> Internal convention: every new method must be compared against the main pipeline (`rf` + `morph_fusion_gray_residual_a01` + `text_only` + `normal_75_25`, seed=111, k=1) and must **not** assume the anomaly type is known at test time.

---

## 10. Upstream Work & Citation

```bibtex
@article{li2024promptad,
  title={PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection},
  author={Li, Xiaofan and Zhang, Zhizhong and Tan, Xin and Chen, Chengwei and Qu, Yanyun and Xie, Yuan and Ma, Lizhuang},
  journal={arXiv preprint arXiv:2404.05231},
  year={2024}
}
```

Acknowledgements: [WinCLIP](https://github.com/caoyunkang/WinClip.git), [CoOp](https://github.com/KaiyangZhou/CoOp.git), [OpenCLIP](https://github.com/mlfoundations/open_clip).
