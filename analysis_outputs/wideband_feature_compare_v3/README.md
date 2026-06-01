# Wideband Feature Compare V3

Source samples: `analysis_outputs/morph_fusion_featuremaps/selected_samples.csv` filtered to `wideband_pulse`.
Panels: original, grad_mag, wideband_feature_v2, wideband_feature_v3.
V3 is intentionally gentler: lower enhancement alpha, no min-max remap on the final blend, and preserved grayscale dynamic range.

Per-sample means:

- `WeaponMuseum_spectrum` | `m40db` | `grad_mean=0.1148` | `v2_mean=0.1166` | `v3_mean=0.3909`
- `WeaponMuseum_spectrum` | `m40db` | `grad_mean=0.1101` | `v2_mean=0.1342` | `v3_mean=0.3888`
- `WeaponMuseum_spectrum` | `m40db` | `grad_mean=0.1100` | `v2_mean=0.1281` | `v3_mean=0.3924`
