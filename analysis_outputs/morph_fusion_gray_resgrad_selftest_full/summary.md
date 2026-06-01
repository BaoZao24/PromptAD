# morph_fusion_gray_resgrad Full Self-Test Summary

## morph_fusion_gray_resgrad

- `burst_signal`: `m10db 94.3400`, `m20db 83.9625`, `m30db 80.8350`, dataset mean `86.3792`
- `chirp_signal`: `m10db 92.3525`, `m20db 87.1500`, `m30db 69.6825`, dataset mean `83.0617`
- `dsss_signal`: `m10db 99.6225`, `m20db 96.8725`, `m30db 91.9875`, dataset mean `96.1608`
- `wideband_pulse`: `m20db 98.4200`, `m30db 99.3550`, `m40db 80.7150`, dataset mean `92.8300`
- overall mean: `89.6079`

## baseline_rgb

- `burst_signal`: `m10db 96.6125`, `m20db 86.3675`, `m30db 75.9550`, dataset mean `86.3117`
- `chirp_signal`: `m10db 90.7550`, `m20db 86.5750`, `m30db 58.8625`, dataset mean `78.7308`
- `dsss_signal`: `m10db 99.0500`, `m20db 97.1325`, `m30db 93.2825`, dataset mean `96.4883`
- `wideband_pulse`: `m20db 99.1225`, `m30db 98.2400`, `m40db 91.0375`, dataset mean `96.1333`
- overall mean: `89.4160`

## morph_fusion_dualgrad

- `burst_signal`: `m10db 96.2675`, `m20db 88.7650`, `m30db 84.5975`, dataset mean `89.8767`
- `chirp_signal`: `m10db 91.7550`, `m20db 90.0000`, `m30db 72.6475`, dataset mean `84.8008`
- `dsss_signal`: `m10db 99.3725`, `m20db 97.4375`, `m30db 90.6750`, dataset mean `95.8283`
- `wideband_pulse`: `m20db 98.0700`, `m30db 99.2800`, `m40db 80.7075`, dataset mean `92.6858`
- overall mean: `90.7979`

## Comparison

- `burst_signal`: gray_resgrad `86.3792` | baseline `86.3117` | dualgrad `89.8767` | delta vs baseline `+0.0675` | delta vs dualgrad `-3.4975`
- `chirp_signal`: gray_resgrad `83.0617` | baseline `78.7308` | dualgrad `84.8008` | delta vs baseline `+4.3308` | delta vs dualgrad `-1.7392`
- `dsss_signal`: gray_resgrad `96.1608` | baseline `96.4883` | dualgrad `95.8283` | delta vs baseline `-0.3275` | delta vs dualgrad `+0.3325`
- `wideband_pulse`: gray_resgrad `92.8300` | baseline `96.1333` | dualgrad `92.6858` | delta vs baseline `-3.3033` | delta vs dualgrad `+0.1442`
- overall: gray_resgrad `89.6079` | baseline `89.4160` | dualgrad `90.7979` | delta vs baseline `+0.1919` | delta vs dualgrad `-1.1900`
