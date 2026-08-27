# EEG Privacy Benchmark v1.2.0

Version 1.2.0 adds a preregistered Shin2017A temporal-validation study while
preserving the v1.0.0 and v1.1.0 evidence unchanged.

## Shin2017A temporal-validation result

- Twenty-seven confirmatory subjects from NEMAR `nm000267` version `v1.0.3`
  were evaluated across three imagery sessions.
- Plain compact EEGNet passed the baseline utility gate with mean subject
  balanced accuracy `0.6341` and 95% subject-bootstrap interval
  `[0.6008, 0.6705]`.
- A cache-only linear subject-identification probe reached mean cross-session
  accuracy `0.3198`, above 27-subject chance `0.0370`, satisfying the frozen
  detectability prerequisite for that protocol.
- Compact bottleneck EEGNet (`dim=6`) completed `12/12` jobs with mean balanced
  accuracy `0.6137` and paired mean utility loss `0.02037`.
- The bottleneck did not establish noninferiority: its one-sided 95% upper loss
  bound was `0.03071`, which failed the strict `<0.030` requirement.
- Under the preregistered stop rule, no bottleneck subject-identification or
  membership evaluation was run.

The supported conclusion is a negative, bounded result: the fixed dim-6
bottleneck was not eligible for privacy evaluation because utility
noninferiority was not established. Version 1.2.0 does not claim a Shin2017A
privacy improvement, privacy equivalence, or absence of leakage.

## Reproducibility additions

- Historical temporal-validation contract and aggregate gate records
- Deterministic temporal-fold and attacker-partition plumbing
- Version-pinned NEMAR loader and cache inventory checks
- Explicit two-flag dataset acquisition gate requiring acknowledgement of the
  upstream terms
- No-download/no-training public evidence validation
- Versioned v1.2.0 evidence manifest and SHA-256 checksums

No raw EEG, feature arrays, posterior arrays, model checkpoints, internal
authorization records, or generated run outputs are included. Dataset access
remains subject to the upstream provider's current terms.

## Verification

```bash
python scripts/check_public_smoke.py
PYTHONPATH=src python scripts/check_shin2017a_public_evidence.py
PYTHONPATH=src python scripts/check_shin2017a_split_plumbing.py
PYTHONPATH=src python scripts/check_shin2017a_nemar_loader.py
python scripts/check_public_release.py --allow-remote origin
```

See `results/v1.2/README.md` for the evidence index and
`docs/V1_2_SHIN2017A_CLAIM_AUDIT.md` for the complete claim boundary.
