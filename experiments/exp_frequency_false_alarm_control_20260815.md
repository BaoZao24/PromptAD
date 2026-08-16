# Support-Driven Frequency False-Alarm Control Screening

## Implementation

Implemented the document `docs/method/支持集驱动的频率虚警控制方案.md` as a
separate support-only evaluator.  The raw ViT memory score is unchanged.  The
calibrator uses non-self support patch scores against the formal farthest
memory and tests:

- C0: raw score;
- C1: one global normal-support threshold;
- C2: independent threshold for each frequency column;
- C3: local threshold smoothed toward the global threshold using support count.

The reported target-FPR threshold is selected only from support-derived image
scores.  The first implementation had an invalid asymmetry here: a support
image could still match the memory patches that originated from itself, while
a test image could not.  This made the support score unrealistically small
and invalidated the 100% false-alarm conclusion below.  The evaluator now
uses leave-one-support-image-out scoring when estimating the image-level
alarm threshold: all memory patches originating from the scored support image
are masked.  Test labels are used only for final metric calculation.

## Superseded preliminary result

Formal In-house 60-cell macro result, with `base-memory-ratio=0.1`:

| Calibrator | AUROC | AUPRC | FPR95 | Actual FPR at target 5% | TPR at target 5% |
|---|---:|---:|---:|---:|---:|
| C0 raw | 92.2615 | 82.1073 | 21.5723 | -- | -- |
| C1 global | 92.2615 | 82.1073 | 21.5723 | 100.00% | 100.00% |
| C2 frequency-local | 92.5208 | 82.0631 | 21.3400 | 99.54% | 100.00% |
| C3 smoothed-local | 92.5097 | 82.0720 | 21.3443 | 99.54% | 100.00% |

The single `burst_signal/WeaponMuseum_spectrum/m10db` cell already showed the
same failure: C1/C2/C3 all produced 100% actual FPR at target 5%.  This is
**not a valid result** because the support-image self-match described above
was present.  These numbers are retained only as an implementation record and
must not be used in the paper or to reject the candidate method.

## Corrected single-cell re-check

The corrected evaluator was re-run on
`burst_signal/WeaponMuseum_spectrum/m10db` using GPU 1.  The result is stored
in `analysis_outputs/exploratory/20260815_fpr_calibration_single_cell_loo.json`.

| Calibrator | AUROC | AUPRC | FPR95 | Actual FPR at target 5% | TPR at target 5% |
|---|---:|---:|---:|---:|---:|
| C0 raw | 97.32 | 96.70 | 31.13 | -- | -- |
| C1 global | 97.32 | 96.70 | 31.13 | 16.04% | 94.74% |
| C2 frequency-local | 97.19 | 96.67 | 35.85 | 16.04% | 94.74% |
| C3 smoothed-local | 97.22 | 96.68 | 34.91 | 16.04% | 94.74% |

The self-match fix changes the false-alarm result from 100.00% to 16.04%, so
the earlier full result cannot be cited.  However, the corrected 16.04% is
still far from the requested 5% target.  More importantly, C2/C3 do not lower
the false-alarm rate relative to the global C1 baseline, while both slightly
reduce ranking metrics.

## Decision

Reject this candidate in its present form and do not run the 60-cell, Public
RF, or FedJam extensions.  The problem is not an implementation defect after
the correction: frequency-wise scaling changes the score ranking a little,
but does not make the support-derived image-level threshold transfer to this
later normal test window.  It therefore cannot support the claimed
"target-FPR controllable" innovation.

Implementation: `tools/eval_frequency_false_alarm_control.py`.
