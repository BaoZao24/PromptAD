# Normal Background Deviation Input Channel Experiment

> Date: 2026-06-03
> Status: **Completed — Failed, not proceeding to full-scale**

## 1. Method Definition

`normal_bg_deviation` is a per-pixel deviation map computed from **training normal sample statistics only**. It measures how much each pixel deviates from the "expected normal background" at that location.

### 1.1 Formula

Let $\{I^{(n)}\}_{n=1}^N$ be the $N$ training normal spectrogram images (grayscale, 240×240 after preprocessing).

**Step 1 — Normal background estimation** (training phase, offline):

$$\text{median}(f, t) = \underset{n}{\text{median}}\; I^{(n)}(f, t)$$

$$\text{MAD}(f, t) = \underset{n}{\text{median}}\; \left| I^{(n)}(f, t) - \text{median}(f, t) \right|$$

**Step 2 — Deviation computation** (per test image):

$$\text{deviation}(f, t) = \frac{|I(f, t) - \text{median}(f, t)|}{1.4826 \cdot \text{MAD}(f, t) + \epsilon}$$

**Step 3 — Robust normalization**:

$$\text{deviation}_{\text{norm}} = \text{clip}(\text{deviation}, p_2, p_{98}) \rightarrow \text{min-max normalize to } [0, 1]$$

### 1.2 Robustness

- Uses **median** instead of mean for robustness to outliers in the training set
- Uses **MAD** (Median Absolute Deviation) instead of standard deviation
- The 1.4826 factor makes MAD consistent with standard deviation for normally distributed data
- Percentile clipping ($p_2$, $p_{98}$) prevents extreme outliers from compressing the dynamic range

### 1.3 Two Variants

**方案A: `morph_fusion_normal_bg_residual`**
```
original_gray + weak_residual(α=0.1) + normal_bg_deviation
```
Replaces `gray_contrast` with `normal_bg_deviation` as the 3rd channel.

**方案B: `morph_fusion_contrast_residual_normal_bg`**
```
gray_contrast + weak_residual(α=0.1) + normal_bg_deviation
```
Keeps `gray_contrast`, replaces `original_gray` with `normal_bg_deviation`.

## 2. How Normal Background Statistics Are Computed

1. **Only training normal** images are used (label=0, from the `normal_75_25` train split)
2. Images go through the **same preprocessing** as the model transform pipeline:
   - BGR numpy → RGB PIL → Resize(240) → CenterCrop(240) → convert('L') → [0,1] grayscale
3. All preprocessed images are stacked into a $(N, 240, 240)$ array
4. Pixel-wise median and MAD are computed across the $N$ dimension
5. Result is two $(240, 240)$ arrays (`median`, `mad`) that capture the normal background distribution

**Key constraint**: No test data, no abnormal data, no anomaly-type labels are used in stats computation.

## 3. Why This Is NOT Selection

The `normal_bg_deviation` channel is a **unified input transformation** applied identically to all anomaly types. Unlike the deprecated `selection` approach:

- selection: `if burst → use method_A; if chirp → use method_B` — requires anomaly type prior
- normal_bg_deviation: `for all types → compute deviation from training normal background`

The deviation map captures *any* deviation from normal, regardless of the specific anomaly pattern. This is the standard paradigm in anomaly detection — learn the normal distribution, flag deviations.

## 4. Why Better Than CLAHE Contrast for Papers (in theory)

| Aspect | CLAHE (gray_contrast) | Normal BG Deviation |
|---|---|---|
| **Principle** | Generic image contrast enhancement | Normal-only background modeling |
| **Theoretical basis** | Histogram equalization (1994) | Anomaly detection theory (deviation from normal) |
| **Task alignment** | General-purpose enhancement | Specifically designed for anomaly detection |
| **Use of training data** | None (per-image transform) | Leverages training normal distribution |
| **Paper writability** | Weak — "we applied CLAHE" | Strong — "we learned normal background distribution and measured per-pixel deviation" |

## 5. Phase 1: Visualization Sanity Check

**Setup**: Scene=Playground_spectrum, Noise=m30db, 24 training normal images

**Output**: `analysis_outputs/normal_bg_deviation_preview/normal_bg_deviation_feature_examples.png`

**Observations**:
- The normal_bg_deviation map captures energy differences but is noisy around spectrogram edges
- Normal samples show non-zero deviation due to natural patch-to-patch variation in time-frequency windows
- Chirp signal shows weaker deviation than expected because the chirp structure has moderate energy

## 6. Phase 2: Small-Scale Experiment Results

**Setup**: Playground_spectrum/m30db, seed=111, epochs=50, prompt_mode=rf, cls_score_mode=text_only

| Dataset | baseline (gray_residual_a01) | 方案A (normal_bg_residual) | 方案B (contrast_residual_normal_bg) |
|---|---:|---:|---:|
| burst_signal | 94.50 | 94.03 | 92.72 |
| chirp_signal | 91.07 | 88.41 | 88.35 |
| dsss_signal | 94.10 | 94.21 | 93.55 |
| **三类平均** | **93.22** | **92.22** | **91.54** |

**Delta vs baseline**:

| Dataset | 方案A Δ | 方案B Δ |
|---|---:|---:|
| burst_signal | -0.47 | -1.78 |
| chirp_signal | **-2.66** | **-2.72** |
| dsss_signal | +0.11 | -0.55 |
| **三类平均** | **-1.01** | **-1.68** |

## 7. Decision: Proceed to Full-Scale?

**Criteria**:
- [ ] 三类平均 ≥ 90.5650 → **FAIL**: 方案A=92.22 (note: this is single-scene, while 90.5650 is 4-scene mean; for sanity check on single scene, baseline=93.22, both variants lower)
- [ ] chirp_signal not significantly degraded → **FAIL**: chirp drops -2.66 (方案A) and -2.72 (方案B)
- [ ] At least one of burst/dsss stable improvement → Marginal: dsss +0.11 in 方案A, but chirp degradation dominates

**Decision**: **DO NOT proceed to full-scale. This approach is a failure.**

## 8. Failure Analysis

### Why the approach failed

The `normal_bg_deviation` channel underperforms the current main scheme across all anomaly types (except marginal dsss in 方案A). Three root causes:

1. **Pixel-level rigid registration assumption**: Per-pixel median/MAD assumes training and test images are perfectly registered at the pixel level. However, spectrogram PNG patches come from different time-frequency windows (`t00000-04000`, `t02000-06000`, etc.). Even within the `t00000-04000` training filter, different frequency windows produce slightly different background energy distributions at the same pixel coordinate.

2. **Insufficient training samples for robust statistics**: With only 24-132 training normal images per scene, the pixel-wise MAD estimates are noisy. The MAD-based normalization can amplify noise instead of suppressing it.

3. **Translation invariance needed**: Anomalies like chirp signals appear at different frequency locations. A pixel-wise deviation map penalizes any deviation from the exact normal image at that pixel, but the chirp structure is anomalous because of its *pattern* (sweeping line), not because of energy at a specific pixel. The normal_bg_deviation map doesn't capture this structural nature of anomalies.

### Why the current gray_contrast works better

`gray_contrast` (CLAHE) enhances local contrast adaptively for each image, without assuming pixel-level alignment across images. It amplifies edges and structures that are present in the current image, making them more salient to CLIP. This is better matched to the CLIP visual encoder's sensitivity to texture and edge patterns.

### Conclusion

**Normal-only pixel background statistics are unsuitable for the current spectrogram PNG input format because they require pixel-level alignment that does not exist between different time-frequency patches. The approach cannot replace `gray_contrast` as a main scheme channel.**

The `normal_bg_deviation` approach may be better suited for scenarios with:
- Registered images captured from the same sensor position
- Larger training normal sample sizes
- Anomalies that manifest as localized energy changes (rather than structural patterns)

### Classification: **Failed Route**

This experiment is classified as a failed exploration and will NOT be submitted as part of the main method.

## 9. Code Artifacts

- **Channel classes**: `NormalBgDeviationChannels`, `MorphFusionNormalBgResidualChannels`, `MorphFusionContrastResidualNormalBgChannels` in `PromptAD/model.py`
- **Stats computation**: `NormalBgDeviationChannels.compute_normal_bg_stats_from_dataloader()`
- **Training injection**: `train_cls.py` `fit()` function
- **Preview script**: `tools/generate_normal_bg_preview.py`
- **Input modes**: `morph_fusion_normal_bg_residual`, `morph_fusion_contrast_residual_normal_bg`

All code is preserved for documentation of the exploration but will NOT be promoted to the main method.
