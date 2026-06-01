# Wideband Feature Compare V2

Source samples: `analysis_outputs/morph_fusion_featuremaps/selected_samples.csv`.
Panels: original, gt mask, weak_residual, grad_mag, wideband_feature_v2.
Feature design: grayscale base plus a lightly mixed frequency-tall, time-short local band contrast against a broader local background.

Per-sample means:

- `burst_signal` | `WeaponMuseum_spectrum` | `m30db` | `grad_mean=0.1416` | `wideband_v2_mean=0.2080`
- `burst_signal` | `WeaponMuseum_spectrum` | `m30db` | `grad_mean=0.1268` | `wideband_v2_mean=0.1688`
- `burst_signal` | `WeaponMuseum_spectrum` | `m30db` | `grad_mean=0.0994` | `wideband_v2_mean=0.1789`
- `chirp_signal` | `WeaponMuseum_spectrum` | `m30db` | `grad_mean=0.0972` | `wideband_v2_mean=0.1786`
- `chirp_signal` | `WeaponMuseum_spectrum` | `m30db` | `grad_mean=0.0922` | `wideband_v2_mean=0.1576`
- `chirp_signal` | `WeaponMuseum_spectrum` | `m30db` | `grad_mean=0.0672` | `wideband_v2_mean=0.1451`
- `dsss_signal` | `WeaponMuseum_spectrum` | `m30db` | `grad_mean=0.0914` | `wideband_v2_mean=0.1254`
- `dsss_signal` | `WeaponMuseum_spectrum` | `m30db` | `grad_mean=0.1247` | `wideband_v2_mean=0.1505`
- `dsss_signal` | `WeaponMuseum_spectrum` | `m30db` | `grad_mean=0.1038` | `wideband_v2_mean=0.1565`
- `wideband_pulse` | `WeaponMuseum_spectrum` | `m40db` | `grad_mean=0.1148` | `wideband_v2_mean=0.1166`
- `wideband_pulse` | `WeaponMuseum_spectrum` | `m40db` | `grad_mean=0.1101` | `wideband_v2_mean=0.1342`
- `wideband_pulse` | `WeaponMuseum_spectrum` | `m40db` | `grad_mean=0.1100` | `wideband_v2_mean=0.1281`
