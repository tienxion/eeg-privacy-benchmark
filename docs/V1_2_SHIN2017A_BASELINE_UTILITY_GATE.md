# Shin2017A v1.2 Baseline Utility Gate

## Status

The preregistered plain-EEGNet temporal-utility gate passed. All 24 result files and all 24 representation/posterior
feature caches validated before aggregation. Producing this record performed
no download, model training, subject-ID probe, membership attack, or defense
execution.

The separate plain-EEGNet subject-identification detectability gate has not
been run and remains pending authorization. Therefore, this utility result
does not authorize the bottleneck stage by itself.

## Subject-level utility

Each subject's 12 observations (three temporal folds by four task seeds) were
averaged first. The 27 subject aggregates were then treated as the inferential
observations. Intervals are deterministic two-sided 95% percentile bootstraps
over subjects (100,000 replicates; seed `7001`).

| Model | Mean balanced accuracy | Subject SD | 95% subject-bootstrap CI | Mean macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| CSP-LDA | 0.6006 | 0.0947 | [0.5660, 0.6364] | 0.5390 |
| Plain compact EEGNet | 0.6341 | 0.0940 | [0.6008, 0.6705] | 0.6138 |

## Gate decision

| Criterion | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Plain-EEGNet mean subject balanced accuracy | >= 0.55 | 0.6341 | PASS |
| 95% lower subject-bootstrap bound | > 0.50 | 0.6008 | PASS |

The CSP-LDA comparison is descriptive. Plain EEGNet minus CSP-LDA has
mean paired subject difference `+0.0335`.

## Fold diagnostics

Fold values are descriptive means over 27 subjects and four seeds; folds
and seeds are not treated as independent inferential observations.

| Fold | CSP-LDA | Plain compact EEGNet |
| ---: | ---: | ---: |
| 1 | 0.6481 | 0.6329 |
| 2 | 0.5870 | 0.6005 |
| 3 | 0.5667 | 0.6690 |

## Reproducibility record

The tracked raw-data-free artifact is
[`configs/shin2017a_confirmatory_baseline_utility_gate_v1.json`](../configs/shin2017a_confirmatory_baseline_utility_gate_v1.json).
It contains subject aggregates, diagnostics, and source hashes, but no raw
EEG, learned representations, or posterior arrays.

Verify the public historical contract and aggregate evidence with:

```bash
PYTHONPATH=src python scripts/check_shin2017a_public_evidence.py
```
