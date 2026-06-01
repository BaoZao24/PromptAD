# Gray Contrast ResGrad Full Self-Test Summary

Input mode: `morph_fusion_gray_contrast_resgrad = gray_contrast + weak_residual + grad_mag`.
Prompt mode: `rf`.
Seed: `111`.

Overall mean across 4 datasets: `89.1985`

## burst_signal

- Noise means: `m10db=96.6125`, `m20db=86.3675`, `m30db=75.955`
- Dataset mean: `86.3117`
- Baseline RGB: `86.3117` (delta `+0.0000`)
- DualGrad: `89.8767` (delta `-3.5650`)
- GrayResGrad: `86.3792` (delta `-0.0675`)

## chirp_signal

- Noise means: `m10db=90.755`, `m20db=86.575`, `m30db=58.8625`
- Dataset mean: `78.7308`
- Baseline RGB: `78.7308` (delta `+0.0000`)
- DualGrad: `84.8008` (delta `-6.0700`)
- GrayResGrad: `83.0617` (delta `-4.3309`)

## dsss_signal

- Noise means: `m10db=99.05`, `m20db=97.1325`, `m30db=93.2825`
- Dataset mean: `96.4883`
- Baseline RGB: `96.4883` (delta `+0.0000`)
- DualGrad: `95.8283` (delta `+0.6600`)
- GrayResGrad: `96.1608` (delta `+0.3275`)

## wideband_pulse

- Noise means: `m20db=98.5975`, `m30db=97.935`, `m40db=89.2575`
- Dataset mean: `95.2633`
- Baseline RGB: `96.1333` (delta `-0.8700`)
- DualGrad: `92.6858` (delta `+2.5775`)
- GrayResGrad: `92.83` (delta `+2.4333`)

