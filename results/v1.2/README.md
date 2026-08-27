# v1.2 Shin2017A Temporal-Validation Evidence

Version 1.2 adds a preregistered temporal validation on 27 confirmatory
Shin2017A subjects. It does not alter the frozen v1.0 or v1.1 evidence.

## Decision summary

- Dataset: NEMAR `nm000267` version `v1.0.3`, imagery sessions only.
- Confirmatory cohort: subjects 3-29; subjects 1-2 were reserved for pilot QA.
- Plain compact EEGNet: mean subject balanced accuracy `0.6341`, 95%
  subject-bootstrap interval `[0.6008, 0.6705]`; utility gate passed.
- Plain-model subject-identification probe: mean cross-session accuracy
  `0.3198`, compared with `0.0370` chance for 27 subjects; the frozen
  detectability prerequisite passed for this linear-probe protocol.
- Compact bottleneck EEGNet (`dim=6`): `12/12` jobs validated, mean subject
  balanced accuracy `0.6137`, and paired mean loss `0.02037`.
- Bottleneck utility decision: **NONINFERIORITY NOT ESTABLISHED**. The
  one-sided 95% upper loss bound was `0.03071`, which did not satisfy the
  strict `<0.030` rule.
- The preregistered stop rule prevented bottleneck subject-identification and
  membership evaluations. No Shin2017A privacy-defense claim is promoted.

## Records

- [Historical validation contract](../../configs/shin2017a_temporal_validation_v1.yaml)
- [Baseline utility gate](../../docs/V1_2_SHIN2017A_BASELINE_UTILITY_GATE.md)
- [Subject-ID detectability gate](../../docs/V1_2_SHIN2017A_SUBJECT_ID_DETECTABILITY_GATE.md)
- [Bottleneck utility gate](../../docs/V1_2_SHIN2017A_BOTTLENECK_UTILITY_GATE.md)
- [Claim audit](../../docs/V1_2_SHIN2017A_CLAIM_AUDIT.md)

The corresponding machine-readable aggregate records are under
`configs/shin2017a_confirmatory_*_v1.json`. The public evidence checker verifies
their cross-record hashes and decision states without downloading data or
training a model:

```bash
PYTHONPATH=src python scripts/check_shin2017a_public_evidence.py
```

## Claim boundary

The supported findings are that plain compact EEGNet passed the named temporal
utility gate and that the named linear probe detected cross-session subject
identity above chance. The fixed dim-6 bottleneck did not establish utility
noninferiority under the preregistered strict rule. No bottleneck privacy,
membership-resistance, absence-of-leakage, or privacy-equivalence conclusion is
supported.
