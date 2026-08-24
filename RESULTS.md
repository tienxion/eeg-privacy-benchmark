# Benchmark v1 Results

The complete conservative narrative is in `paper/manuscript.md`. Curated
aggregate evidence is indexed in `results/v1/README.md`.

## Supported or qualified findings

- BNCI bottleneck EEGNet reduces expanded-scope subject-identification accuracy
  by 0.2683 while task balanced accuracy improves by 0.0106. Membership effects
  are small and attacker-dependent.
- Lee bottleneck EEGNet reduces subject-identification accuracy by 0.2340 while
  task balanced accuracy improves by 0.0019, but threshold membership AUCs
  regress by 0.0145 to 0.0176.
- PhysioNet has no stable defense default after random-subset, nonlinear-attacker
  seed extension, cross-run holdout, and bounded defense-grid validation.

## Systems diagnostics

- PhysioNet federation changes mean task balanced accuracy from 0.6574 to
  0.5185. Because it fails the 0.60 utility gate, privacy differences are
  diagnostic and do not support a defense claim.
- The pairwise-mask simulator matches FedAvg within `2.384e-07`, estimates
  0.0649% one-time seed-setup communication relative to the measured run, and
  has a 217.77x local runtime ratio. It is not production cryptography and does
  not protect output-level membership.

## Claim boundary

No result establishes formal privacy, attack invariance, resistance to adaptive
white-box adversaries, raw-signal reconstruction protection, or safety under
repeated releases. Lower leakage for one measured objective must not be reported
as lower leakage for another.
