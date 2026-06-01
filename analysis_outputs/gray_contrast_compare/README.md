# Gray Contrast Compare

Source samples: `analysis_outputs/morph_fusion_featuremaps/selected_samples.csv`.
Panels: original, gt mask, gray, gray_contrast (CLAHE), weak_residual, grad_mag.
Gray contrast uses conservative local histogram equalization (CLAHE) so the image remains faithful to the source while local intensity contrast is enhanced.

Per-sample means:

- `burst_signal` | `WeaponMuseum_spectrum` | `m30db` | `gray=0.5119` | `gray_contrast=0.5206` | `weak_residual=0.1119` | `grad_mag=0.1416`
- `burst_signal` | `WeaponMuseum_spectrum` | `m30db` | `gray=0.4769` | `gray_contrast=0.4847` | `weak_residual=0.0821` | `grad_mag=0.1268`
- `burst_signal` | `WeaponMuseum_spectrum` | `m30db` | `gray=0.4883` | `gray_contrast=0.5000` | `weak_residual=0.0990` | `grad_mag=0.0994`
- `chirp_signal` | `WeaponMuseum_spectrum` | `m30db` | `gray=0.4882` | `gray_contrast=0.5010` | `weak_residual=0.0986` | `grad_mag=0.0972`
- `chirp_signal` | `WeaponMuseum_spectrum` | `m30db` | `gray=0.4885` | `gray_contrast=0.5073` | `weak_residual=0.0996` | `grad_mag=0.0922`
- `chirp_signal` | `WeaponMuseum_spectrum` | `m30db` | `gray=0.4874` | `gray_contrast=0.5139` | `weak_residual=0.0655` | `grad_mag=0.0672`
- `dsss_signal` | `WeaponMuseum_spectrum` | `m30db` | `gray=0.3818` | `gray_contrast=0.4054` | `weak_residual=0.0736` | `grad_mag=0.0914`
- `dsss_signal` | `WeaponMuseum_spectrum` | `m30db` | `gray=0.3860` | `gray_contrast=0.3990` | `weak_residual=0.1042` | `grad_mag=0.1247`
- `dsss_signal` | `WeaponMuseum_spectrum` | `m30db` | `gray=0.4074` | `gray_contrast=0.4233` | `weak_residual=0.1473` | `grad_mag=0.1038`
- `wideband_pulse` | `WeaponMuseum_spectrum` | `m40db` | `gray=0.3806` | `gray_contrast=0.3969` | `weak_residual=0.1071` | `grad_mag=0.1148`
- `wideband_pulse` | `WeaponMuseum_spectrum` | `m40db` | `gray=0.3742` | `gray_contrast=0.3913` | `weak_residual=0.1093` | `grad_mag=0.1101`
- `wideband_pulse` | `WeaponMuseum_spectrum` | `m40db` | `gray=0.3811` | `gray_contrast=0.3952` | `weak_residual=0.1047` | `grad_mag=0.1100`
