# TFAM-AAE Spectrogram Baseline

Status: the reconstruction branch is implemented, smoke-tested, and reviewed
(`models/tfam_aae.py`, `tools/eval_tfam_spectral.py`,
`tests/test_tfam_aae.py`). Unit tests run with
`pytest tests/test_tfam_aae.py` in the `prompt_ad` environment. All four paper
protocols (In-house RF `rf_target`, Public RF k=1/2/4, `ofdma`, FedJam via
`tools/eval_fedjam_visual_baselines.py --method tfam`) are wired and
smoke-tested; the exploratory `spectrum` protocol is also available.

## Paper Protocol

The reference is `references/TFAM-AAE-Uk_A_Dual-Metric_Spectrum_Anomaly_Detection_Algorithm.pdf`:

- The spectrum input is a single-channel time-frequency matrix.
- TFAM-AAE reconstructs the input and uses the per-sample MSE in Eq. (9) as
  the first anomaly metric.
- `U_k` receives two-channel IQ samples of length 1024, learns a 256-dimensional
  fingerprint, and uses the minimum class-conditional Mahalanobis distance in
  Eqs. (10)-(11) as the second metric.
- The simulation samples are 16 x 512 spectrograms, and the LU* test set has
  100 samples per anomaly type (high-power, ultra-bandwidth, frequency-shift).
- The actual experiment uses 16 x 654 spectrum samples and 2 x 1024 IQ samples.
- The exact train/test counts of both experiments live in TABLE II and TABLE IV
  of the paper, which are embedded table images not extractable by `pdftotext`
  in this environment; consult the PDF directly before quoting them.

## Attention Implementation Deviations

`models/tfam_aae.py` adapts the paper's TFAM rather than copying it:

- Eq. (5)'s `f3x3` (three stacked 3x3 convolutions) is implemented as a single
  3x3 convolution per attention branch.
- Eq. (6)'s `(At (x) F') + (Af (x) F')` convolution combination is implemented
  as multiplicative gating `F' * (1 + 0.5 * (At + Af))` followed by a
  refinement convolution in the decoder path.

These choices keep the block compact for 64x64 inputs and must be stated
whenever the row name `tfam_aae_spectrogram` appears in a table.

## Repository Adaptation

`models/tfam_aae.py` and `tools/eval_tfam_spectral.py` implement the first,
spectrogram-only branch. They use the repository's target-scene manifest and
normal-only support training. The current RF arrays are dBm STFT matrices and
the sliced benchmark exposes PNGs; no paired raw IQ data is available for a
faithful `U_k` implementation.

The resulting method name is `tfam_aae_spectrogram`, not the full
`TFAM-AAE-U_k`. Results must not be reported as a full reproduction until an IQ
dataset and its user labels are provided. The rf_target support sampling is
deliberately restricted to `per_frequency`/`1shot`/`2shot`/`4shot`
(`frequency_one_per_band` is rejected) to match the formal few-shot protocol.

## Usage as a Baseline

Positioning: this is a reconstruction-type baseline in the same column as
SAIFE/VAE, not a full reproduction of `TFAM-AAE-U_k`. When it appears in a
comparison table, use the display name `TFAM-AAE (spectrogram)` and state that
only the Eq. (9) reconstruction branch is active because no paired IQ data
exists for `U_k`.

All commands below are the formal full runs (GPU recommended; no smoke
truncation flags). Every run writes `results_tfam_cls.csv` (per-cell rows plus
a macro row), `summary.json`, `protocol.json`, `scores/*.npz`, and
`checkpoints/tfam_group*.pt` under its `--output-root`. Test labels are read
only after scoring; training uses normal support images only.

The paper protocol (`docs/paper/现有方案介绍.md`) covers exactly four datasets:
In-house RF, Public RF, OFDMA, and FedJam. The `spectrum` protocol below is
kept only as an exploratory entry and is not part of the paper comparison.

### 1. In-house RF (rf_target) — formal 60-cell run

Four trainings (one per scene), then every signal x scene x JSR cell is scored.
Support follows the shared target-scene manifest protocol (time-isolated,
content-deduplicated; `docs/research/rf_target_data_protocol.md`):

```bash
python tools/eval_tfam_spectral.py \
  --protocol rf_target \
  --normal-sampling per_frequency \
  --support-manifest analysis_outputs/20260728_rf_five_type_formal_seed111/support_manifest.json \
  --output-root analysis_outputs/tfam_aae_spectral_self_rf
```

Notes:

- `--normal-sampling` accepts `per_frequency` (default, one per frequency band)
  or `1shot`/`2shot`/`4shot` for the shot ablation.
- The paper's In-house RF row is `per_frequency` only (the audit in
  `analysis_outputs/20260810_public_rf_k_per_frequency/README.md` shows most
  frequency bands hold fewer than 4 independent normal contents, so strict
  k=2/4 is not honest on this data).

### 2. Public RF — nested k=1/2/4-per-frequency

The paper protocol is the nested k-per-frequency comparison with a fixed test
pool. Use the three shared manifests from
`analysis_outputs/20260810_public_rf_k_per_frequency/`; the manifests declare
`normal_sampling=per_frequency` for evaluator compatibility and record
`support_policy=k_per_frequency` plus `per_frequency_k`:

```bash
for k in 1 2 4; do
  python tools/eval_tfam_spectral.py \
    --protocol public_rf \
    --normal-sampling per_frequency \
    --support-manifest \
      analysis_outputs/20260810_public_rf_k_per_frequency/k${k}/support_manifest.json \
    --output-root analysis_outputs/tfam_aae_spectral_public_rf_k${k}
done
```

The k=1 manifest is byte-identical (support and test) to the earlier
`20260805_public_rf_fixed_test_pool` manifest, so the existing
`tfam_aae_spectral_public_rf` run is exactly the k=1 row. Five signals x three
levels = 15 cells per k; one training per k (16/32/64 support images).
### 3. OFDMA — official scene split, observation-level metrics

Nested 1/2/4-shot scene-level support on the official split. OFDMA images pass
through the official physics preprocessor (dB normalization + letterbox), and
image scores are aggregated to observation level (max over the SU frames of
each scene) before metrics, matching the other OFDMA baselines:

```bash
python tools/eval_tfam_spectral.py \
  --protocol ofdma \
  --ofdma-shots 1 2 4 \
  --output-root analysis_outputs/tfam_aae_spectral_ofdma
```

Each shot trains one model (three trainings total). The saved `scores/*.npz`
contains both per-image and per-observation scores/labels.

### 4. FedJam — benign-only 1/2/4-shot, spectrogram-only

FedJam runs through the shared visual-baseline adapter, which selects a
deterministic nested seed-111 reservoir of benign train rows as support and
evaluates the full 7,200-row test split. Only the embedded spectrogram image is
used (no KPI). TFAM consumes the in-memory BGR buffers directly:

```bash
python tools/eval_fedjam_visual_baselines.py \
  --method tfam \
  --shots 1 2 4 \
  --gpu-id 0 \
  --output-root analysis_outputs/tfam_aae_spectral_fedjam
```

Output lands in `.../tfam/` with `metrics.csv` (overall + per-attack +
macro_attack rows per shot) and `scores/fedjam_{shot}shot_scores.npz`. The
adapter's shared hyperparameters (`epochs=100`, `lr=1e-3`,
`discriminator_lr=2.5e-5`, `image 64x64`) follow the other generative FedJam
baselines (VAE/SAIFE).

### 5. Spectrum (16QAM/CHIRP/GMSK/QPSK) — exploratory, not in the paper

This dataset is not part of the four-dataset paper protocol; keep it only for
exploratory checks. It has no frequency-band metadata, so `per_frequency`
raises and shot sampling is required:

```bash
python tools/eval_tfam_spectral.py \
  --protocol spectrum \
  --normal-sampling 4shot \
  --output-root analysis_outputs/tfam_aae_spectral_spectrum
```

### 6. Smoke tests (sanity check before a formal run)

```bash
# Self-RF single scene, CPU, ~1 minute
python tools/eval_tfam_spectral.py \
  --protocol rf_target \
  --rf-signals burst_signal \
  --rf-scenes Playground_spectrum \
  --normal-sampling 1shot \
  --use-cpu --epochs 2 --batch-size 1 --test-batch-size 8 --num-workers 0 \
  --max-test-normals 4 --max-abnormals 4 \
  --image-height 32 --image-width 32 \
  --output-root /home/wangbei/tmp/opencode/tfam_spectral_smoke

# OFDMA single shot, CPU, ~1 minute
python tools/eval_tfam_spectral.py \
  --protocol ofdma \
  --ofdma-shots 1 \
  --use-cpu --epochs 1 --batch-size 2 --test-batch-size 8 --num-workers 0 \
  --max-test-normal-scenes 2 --max-test-scenes-per-jammer 1 --max-sus-per-scene 3 \
  --image-height 32 --image-width 32 \
  --output-root /home/wangbei/tmp/opencode/tfam_ofdma_smoke

# FedJam 1shot, CPU, ~1 minute (balanced 20-row test subset)
python tools/eval_fedjam_visual_baselines.py \
  --method tfam --shots 1 --use-cpu --epochs 1 --max-test-per-label 5 \
  --output-root /home/wangbei/tmp/opencode/tfam_fedjam_smoke
```

Smoke results are protocol checks only (tiny truncated test sets); never quote
them as baseline numbers.

### Not implemented

- The `U_k` branch: requires raw IQ recordings plus legal-user labels, which
  the current data does not contain.

### Hyperparameters

For `rf_target`/`public_rf`/`spectrum`/`ofdma`, defaults follow the SAIFE
evaluator conventions (`epochs=100`, `lr=5e-5`, `discriminator_lr=2.5e-5`,
`image 64x64`, `latent_dim=64`, `base_channels=16`, `seed=111`,
`score_mode=mse_mean`). For FedJam, the shared adapter conventions apply
(`epochs=100`, `lr=1e-3`, `discriminator_lr=2.5e-5`, `image 64x64`) so the
TFAM row matches the VAE/SAIFE FedJam rows. Keep them unchanged for the formal
baseline rows; tuning them turns the row into a different experiment.
