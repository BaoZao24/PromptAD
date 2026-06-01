# Gray Contrast ResGrad Score-Fusion Rule Compare (Miniset)

Views: `gray_contrast_only`, `weak_residual_only`, `grad_mag_only`.
Datasets: `wideband_pulse`, `dsss_signal`.

| rule | dataset | mean |
|---|---|---:|
| mean | wideband_pulse | 92.8383 |
| mean | dsss_signal | 89.3783 |
| max | wideband_pulse | 84.4342 |
| max | dsss_signal | 66.9117 |
| conservative_lam0.25 | wideband_pulse | 84.6733 |
| conservative_lam0.25 | dsss_signal | 77.2575 |

Takeaway:

- `mean` is bad for `dsss_signal` and not good enough for `wideband_pulse`.
- `max` collapses both datasets and is not usable.
- `conservative_lam0.25` is still substantially worse than the unified-input baseline and does not rescue either dataset.

