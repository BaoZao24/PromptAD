# Morph Fusion Gradient Cleanup Comparison

Source samples: `analysis_outputs/morph_fusion_featuremaps/selected_samples.csv`.
Panels: original, weak_residual, |time_grad|, |freq_grad|, grad_mag, denoised_grad_mag.
Denoising: Gaussian-smoothed Sobel gradients, gradient magnitude, then soft thresholding with `max(q75, median + 3*MAD)` before robust percentile normalization.

Per-sample activity shrinkage (`>0.05` fraction):

- `burst_signal` | `WeaponMuseum_spectrum` | `m30db` | active `0.3284` -> `0.2120`
- `burst_signal` | `WeaponMuseum_spectrum` | `m30db` | active `0.2728` -> `0.2183`
- `burst_signal` | `WeaponMuseum_spectrum` | `m30db` | active `0.2312` -> `0.1776`
- `chirp_signal` | `WeaponMuseum_spectrum` | `m30db` | active `0.2242` -> `0.1701`
- `chirp_signal` | `WeaponMuseum_spectrum` | `m30db` | active `0.2344` -> `0.1531`
- `chirp_signal` | `WeaponMuseum_spectrum` | `m30db` | active `0.1258` -> `0.0950`
- `dsss_signal` | `WeaponMuseum_spectrum` | `m30db` | active `0.3119` -> `0.1348`
- `dsss_signal` | `WeaponMuseum_spectrum` | `m30db` | active `0.3843` -> `0.2011`
- `dsss_signal` | `WeaponMuseum_spectrum` | `m30db` | active `0.3480` -> `0.1687`
- `wideband_pulse` | `WeaponMuseum_spectrum` | `m40db` | active `0.3188` -> `0.1891`
- `wideband_pulse` | `WeaponMuseum_spectrum` | `m40db` | active `0.4121` -> `0.1629`
- `wideband_pulse` | `WeaponMuseum_spectrum` | `m40db` | active `0.3061` -> `0.1803`
