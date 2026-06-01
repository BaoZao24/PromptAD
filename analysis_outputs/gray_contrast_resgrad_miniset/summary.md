# Gray Contrast ResGrad Minimal Experiment

Input mode: `morph_fusion_gray_contrast_resgrad = gray_contrast + weak_residual + grad_mag`.
Prompt mode: `rf`.
Seed: `111`.
Datasets: `wideband_pulse`, `dsss_signal`.

## wideband_pulse

- Noise means: `m20db=98.5975`, `m30db=97.935`, `m40db=89.2575`
- Dataset mean: `95.2633`
- Baseline RGB: `96.1333`  (delta `-0.8700`)
- DualGrad: `92.6858`  (delta `+2.5775`)
- GrayResGrad: `92.83`  (delta `+2.4333`)

## dsss_signal

- Noise means: `m10db=99.05`, `m20db=97.1325`, `m30db=93.2825`
- Dataset mean: `96.4883`
- Baseline RGB: `96.4883`  (delta `+0.0000`)
- DualGrad: `95.8283`  (delta `+0.6600`)
- GrayResGrad: `96.1608`  (delta `+0.3275`)

