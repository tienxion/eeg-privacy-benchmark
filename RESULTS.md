# Benchmark v1.1 Results

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

## v1.1 Cho2017 confirmatory result

- All 50 confirmatory subjects passed objective cache and signal-structure QA.
- Plain compact EEGNet passed its utility gate at mean subject balanced
  accuracy `0.6287`, with 95% subject-bootstrap CI `[0.6012, 0.6585]`.
- Compact bottleneck EEGNet (`dim=6`) was utility-noninferior: mean balanced
  accuracy `0.6256`, paired loss `0.0032`, and one-sided 95% upper loss bound
  `0.0075` against the preregistered `0.030` margin.
- All 600 frozen cache-only membership jobs validated. Threshold and logistic-
  regression family AUCs were effectively tied, while the MLP reduction was
  `0.0014`. All multiplicity-controlled intervals crossed zero.
- The membership gate recorded **NO PROMOTION**: `0/3` families qualified,
  versus `2/3` required.

See [the v1.1 evidence index](results/v1.1/README.md) and [the Cho2017 claim
audit](docs/V1_1_CHO2017_CLAIM_AUDIT.md). The negative privacy result does not
prove absence of leakage or model equivalence.

## Claim boundary

No result establishes formal privacy, attack invariance, resistance to adaptive
white-box adversaries, raw-signal reconstruction protection, or safety under
repeated releases. Lower leakage for one measured objective must not be reported
as lower leakage for another.
