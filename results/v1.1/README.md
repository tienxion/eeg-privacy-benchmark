# v1.1 Cho2017 Confirmatory Evidence

v1.1 adds a preregistered Cho2017 extension without changing the frozen v1
BNCI, Lee, or PhysioNet evidence.

## Decision summary

- Confirmatory cohort: subjects 3-52 (`50` subjects); pilot subjects 1-2 were
  excluded before confirmation.
- Plain compact EEGNet utility: mean subject BA `0.6287`, 95% bootstrap CI
  `[0.6012, 0.6585]`; gate passed.
- Bottleneck dim-6 utility: mean BA `0.6256`, paired loss `0.0032`, one-sided
  95% upper bound `0.0075` against margin `0.030`; noninferiority passed.
- Membership evaluation: `600/600` cache-only jobs validated across threshold,
  logistic-regression, and MLP families.
- Privacy promotion: **NO PROMOTION**. Mean reductions were `-0.00003`,
  `+0.00002`, and `+0.00140`; `0/3` families qualified versus `2/3` required.

## Records

- [Validation contract](../../docs/V1_1_CHO2017_VALIDATION_CONTRACT.md)
- [Cache QA](../../docs/V1_1_CHO2017_CACHE_QA.md)
- [Baseline utility gate](../../docs/V1_1_CHO2017_BASELINE_GATE.md)
- [Bottleneck utility gate](../../docs/V1_1_CHO2017_BOTTLENECK_GATE.md)
- [Membership-privacy gate](../../docs/V1_1_CHO2017_PRIVACY_GATE.md)
- [Claim audit](../../docs/V1_1_CHO2017_CLAIM_AUDIT.md)

Machine-readable records are under `configs/cho2017_confirmatory_*_v1.*`. They
contain contracts, aggregate metrics, subject identifiers, and artifact hashes,
but no raw EEG or posterior arrays.

## Claim boundary

The fixed bottleneck is supported as task-utility noninferior on this
confirmatory cohort. It is not promoted as a Cho2017 membership-privacy
improvement. Near-chance family means are not proof that leakage is absent, and
the study did not establish privacy equivalence or a new subject-identification
claim.
