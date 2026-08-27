# EEG Privacy Benchmark v1.1.0

Version 1.1.0 adds a preregistered Cho2017 confirmatory extension while keeping
the v1.0.0 BNCI, Lee, and PhysioNet evidence frozen.

## Cho2017 confirmatory result

- Fifty previously untouched confirmatory subjects passed checksum-backed
  cache and signal-structure QA.
- Plain compact EEGNet passed the baseline utility gate with mean subject
  balanced accuracy `0.6287` and 95% subject-bootstrap interval
  `[0.6012, 0.6585]`.
- Compact bottleneck EEGNet (`dim=6`) was utility-noninferior: mean balanced
  accuracy `0.6256`, paired loss `0.0032`, and one-sided 95% upper loss bound
  `0.0075` against the preregistered `0.030` margin.
- All `600/600` frozen cache-only membership jobs completed across threshold,
  logistic-regression, and MLP attacker families.
- The membership-privacy gate recorded **NO PROMOTION**. Mean AUC reductions
  were `-0.00003`, `+0.00002`, and `+0.00140`; all multiplicity-controlled
  intervals crossed zero, and `0/3` families qualified versus `2/3` required.

The supported conclusion is utility noninferiority for the fixed bottleneck on
this Cho2017 cohort. Version 1.1.0 does not claim a Cho2017 membership-privacy
improvement, absence of leakage, or privacy equivalence.

## Reproducibility additions

- Frozen Cho2017 validation, cache, utility, and privacy-gate records
- Subject-disjoint split and attacker-partition plumbing
- Resumable, dry-run-by-default baseline, bottleneck, and cache-only attack
  runners
- Subject-level bootstrap and multiplicity-controlled claim auditing
- Versioned v1.1.0 evidence manifest and SHA-256 checksums
- Expanded no-download/no-training public CI checks

No raw EEG, posterior arrays, model checkpoints, or generated run outputs are
included. Dataset access remains subject to the upstream providers' terms.

## Verification

```bash
python scripts/check_public_smoke.py
PYTHONPATH=src python scripts/check_cho2017_validation_contract.py
PYTHONPATH=src python scripts/check_cho2017_split_plumbing.py
python scripts/check_public_release.py --allow-remote origin
```

See `results/v1.1/README.md` for the evidence index and
`docs/V1_1_CHO2017_CLAIM_AUDIT.md` for the complete claim boundary.
