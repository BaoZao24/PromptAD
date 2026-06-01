# Transient Wideband Response Comparison

Source samples: `analysis_outputs/morph_fusion_featuremaps/selected_samples.csv`.
Panels: original, gt mask, weak_residual, grad_mag, transient_wideband_response.
Response design: frequency-tall and time-short local average energy minus a broader local background, then robust percentile normalization.

Per-sample means:

- `burst_signal` | `WeaponMuseum_spectrum` | `m30db` | `grad_mean=0.1416` | `twr_mean=0.2173`
- `burst_signal` | `WeaponMuseum_spectrum` | `m30db` | `grad_mean=0.1268` | `twr_mean=0.2188`
- `burst_signal` | `WeaponMuseum_spectrum` | `m30db` | `grad_mean=0.0994` | `twr_mean=0.1995`
- `chirp_signal` | `WeaponMuseum_spectrum` | `m30db` | `grad_mean=0.0972` | `twr_mean=0.1992`
- `chirp_signal` | `WeaponMuseum_spectrum` | `m30db` | `grad_mean=0.0922` | `twr_mean=0.1784`
- `chirp_signal` | `WeaponMuseum_spectrum` | `m30db` | `grad_mean=0.0672` | `twr_mean=0.1929`
- `dsss_signal` | `WeaponMuseum_spectrum` | `m30db` | `grad_mean=0.0914` | `twr_mean=0.1608`
- `dsss_signal` | `WeaponMuseum_spectrum` | `m30db` | `grad_mean=0.1247` | `twr_mean=0.1661`
- `dsss_signal` | `WeaponMuseum_spectrum` | `m30db` | `grad_mean=0.1038` | `twr_mean=0.1381`
- `wideband_pulse` | `WeaponMuseum_spectrum` | `m40db` | `grad_mean=0.1148` | `twr_mean=0.1036`
- `wideband_pulse` | `WeaponMuseum_spectrum` | `m40db` | `grad_mean=0.1101` | `twr_mean=0.1454`
- `wideband_pulse` | `WeaponMuseum_spectrum` | `m40db` | `grad_mean=0.1100` | `twr_mean=0.1133`
