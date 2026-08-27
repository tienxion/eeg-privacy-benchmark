# Shin2017A v1.2 Bottleneck Utility Noninferiority Gate

## Status

The preregistered bottleneck utility noninferiority gate did not pass.

All 12 fixed compact bottleneck EEGNet (`dim=6`) jobs and all 12
six-dimensional feature caches validated before aggregation. Producing this
record performed no download, training, subject-ID probe, membership attack,
model selection, or hyperparameter change.

## Paired subject-level comparison

The three temporal folds and four task seeds were averaged within each of
27 subjects before pairing plain EEGNet with bottleneck EEGNet. Loss is
plain minus bottleneck balanced accuracy; positive values indicate lower
bottleneck utility.

| Measure | Plain EEGNet | Bottleneck dim 6 | Paired loss |
| --- | ---: | ---: | ---: |
| Mean subject balanced accuracy | 0.6341 | 0.6137 | 0.0204 |

The paired two-sided 95% subject-bootstrap interval for mean loss is [0.0080, 0.0327]. The preregistered one-sided 95% upper bound is 0.0307.

## Gate decision

| Criterion | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Mean balanced-accuracy loss | < 0.030 | 0.02037 | PASS |
| One-sided 95% upper subject-bootstrap bound | < 0.030 | 0.03071 | FAIL |

The mean loss is inside the margin, but the one-sided bound exceeds it by `0.00071`. Under the predeclared strict rule, noninferiority is therefore not established.
The stop rule blocks confirmatory privacy promotion and downstream
membership attacks. The margin, model, and cohort will not be changed
after observing this outcome.

## Seed diagnostics

Seed rows are descriptive and are not treated as independent inferential
observations.

| Seed | Plain EEGNet | Bottleneck dim 6 | Loss |
| ---: | ---: | ---: | ---: |
| 13 | 0.6358 | 0.6185 | 0.0173 |
| 17 | 0.6235 | 0.6358 | -0.0123 |
| 19 | 0.6481 | 0.6216 | 0.0265 |
| 23 | 0.6290 | 0.5790 | 0.0500 |

## Reproducibility record

The tracked raw-data-free artifact is
[`configs/shin2017a_confirmatory_bottleneck_utility_gate_v1.json`](../configs/shin2017a_confirmatory_bottleneck_utility_gate_v1.json).
It contains source hashes and aggregate metrics, but no raw EEG, learned
representations, posterior arrays, or trial-level predictions.

Verify the public historical contract and aggregate evidence with:

```bash
PYTHONPATH=src python scripts/check_shin2017a_public_evidence.py
```
