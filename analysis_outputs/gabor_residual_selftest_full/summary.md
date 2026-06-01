# Gabor residual selftest summary

Input mode: `morph_fusion_gabor_residual = gray_contrast + weak_residual + gabor_broad`.
Protocol: self-test normal_75_25 where available; wideband result path follows its existing split layout, seed 111, prompt mode rf, 50 epochs.

## Method comparison

| anomaly        |   baseline_rgb |   dualgrad |   gray_contrast_resgrad |   gabor_residual |   gabor_minus_baseline |   gabor_minus_dualgrad |
|:---------------|---------------:|-----------:|------------------------:|-----------------:|-----------------------:|-----------------------:|
| burst_signal   |        86.3117 |    89.8767 |                 86.3117 |          90.0117 |                 3.7000 |                 0.1350 |
| chirp_signal   |        78.7308 |    84.8008 |                 78.7308 |          85.3342 |                 6.6034 |                 0.5334 |
| dsss_signal    |        96.4883 |    95.8283 |                 96.4883 |          95.3358 |                -1.1525 |                -0.4925 |
| wideband_pulse |        96.1333 |    92.6858 |                 95.2633 |          91.8917 |                -4.2416 |                -0.7941 |
| overall        |        89.4160 |    90.7979 |                 89.1985 |          90.6433 |                 1.2273 |                -0.1546 |

## Mean by level

| anomaly        | level   |   image_auroc |
|:---------------|:--------|--------------:|
| burst_signal   | m10db   |       98.1475 |
| burst_signal   | m20db   |       87.8575 |
| burst_signal   | m30db   |       84.0300 |
| chirp_signal   | m10db   |       94.0350 |
| chirp_signal   | m20db   |       88.9125 |
| chirp_signal   | m30db   |       73.0550 |
| dsss_signal    | m10db   |       99.4100 |
| dsss_signal    | m20db   |       96.1125 |
| dsss_signal    | m30db   |       90.4850 |
| wideband_pulse | m20db   |       98.4200 |
| wideband_pulse | m30db   |       97.6925 |
| wideband_pulse | m40db   |       79.5625 |

## Notes

- Gabor is strong on wideband m20/m30 and many non-Gymnasium chirp/DSSS cases.
- It is unstable on weak m40 wideband in WeaponMuseum/Playground/Gymnasium and on Gymnasium burst/chirp.
- As a universal replacement for dualgrad, the full average is lower; as a Gabor/wideband-oriented ablation, it is useful evidence.
