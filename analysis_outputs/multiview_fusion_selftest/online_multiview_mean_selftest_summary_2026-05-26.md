# Online Multiview Fusion Self-Test Summary

Run:
- root: `result_split_ablation/rf_multiview_mean_selftest`
- seed: `111`
- prompt-mode: `rf`
- input-mode: `rgb`
- multiview-fusion: `True`
- multiview-fusion-rule: `mean`
- views: `rgb`, `spectral_gradient_v2`, `dsss_weak_residual`

Per-dataset mean tables:

- burst_signal: `m10 95.1125`, `m20 87.7425`, `m30 79.0325`, dataset mean `87.2958`
- chirp_signal: `m10 93.9325`, `m20 89.4825`, `m30 73.7750`, dataset mean `85.7300`
- dsss_signal: `m10 98.9950`, `m20 96.1850`, `m30 90.5175`, dataset mean `95.2325`
- wideband_pulse: `m20 98.8300`, `m30 99.0825`, `m40 84.5950`, dataset mean `94.1692`

Overall mean across the 12 dataset/noise means: `90.6069`

Baseline comparison:
- burst_signal baseline dataset mean: `86.3117` -> delta `+0.9842`
- chirp_signal baseline dataset mean: `78.7308` -> delta `+6.9992`
- dsss_signal baseline dataset mean: `96.4883` -> delta `-1.2558`
- wideband_pulse baseline dataset mean: `96.1333` -> delta `-1.9642`
- overall baseline mean: `89.4160` -> delta `+1.1908`
