# Wideband Prompt and Input Comparison

Fixed setup: `dataset=wideband_pulse`, `seed=111`, `epochs=50`.

## rf+rgb

- `m20db 99.1225`
- `m30db 98.2400`
- `m40db 91.0375`
- dataset mean: `96.1333`
- delta vs `rf+gray_resgrad`: `+3.3033`

## rf_signal_structured+rgb

- `m20db 98.8300`
- `m30db 98.0400`
- `m40db 89.5600`
- dataset mean: `95.4767`
- delta vs `rf+gray_resgrad`: `+2.6467`

## rf+gray_resgrad

- `m20db 98.4200`
- `m30db 99.3550`
- `m40db 80.7150`
- dataset mean: `92.8300`
- delta vs `rf+gray_resgrad`: `+0.0000`

## rf_signal_structured+gray_resgrad

- `m20db 98.4500`
- `m30db 99.3550`
- `m40db 79.7325`
- dataset mean: `92.5125`
- delta vs `rf+gray_resgrad`: `-0.3175`

## Key Delta

- `m20db +0.0300`
- `m30db +0.0000`
- `m40db -0.9825`
- dataset mean: `-0.3175`
