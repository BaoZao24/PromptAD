# Three-class Gabor residual main results

Research scope: `burst_signal`, `chirp_signal`, `dsss_signal`.

`wideband_pulse` is excluded from the main research scope for this stage because m40db low-contrast block anomalies were unstable under traditional feature engineering. It remains an exploratory/limitation case, not a main benchmark.

## Main comparison

| anomaly          |   baseline_rgb |   morph_fusion_dualgrad |   morph_fusion_gabor_residual |   gabor_minus_baseline |   gabor_minus_dualgrad |
|:-----------------|---------------:|------------------------:|------------------------------:|-----------------------:|-----------------------:|
| burst_signal     |        86.3117 |                 89.8767 |                       90.0117 |                 3.7000 |                 0.1350 |
| chirp_signal     |        78.7308 |                 84.8008 |                       85.3342 |                 6.6034 |                 0.5334 |
| dsss_signal      |        96.4883 |                 95.8283 |                       95.3358 |                -1.1525 |                -0.4925 |
| three_class_mean |        87.1769 |                 90.1686 |                       90.2272 |                 3.0503 |                 0.0586 |

## Files

- `threeclass_method_comparison.csv`: main table used by the report.
- `gabor_threeclass_per_case.csv`: per scene/noise-level results for gabor residual.
- `gabor_threeclass_mean_by_level.csv`: gabor residual averaged by anomaly and level.
- `threeclass_gabor_results.xlsx`: Excel workbook with the three sheets above.
