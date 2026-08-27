# Cho2017 Confirmatory Bottleneck Utility Gate

## Status

The preregistered bottleneck utility noninferiority gate passed.

All 20 frozen compact bottleneck EEGNet (`dim=6`) jobs and all 20
posterior caches were validated before comparison. Producing this record
performed no training, download, model selection, or privacy attack.

## Paired subject-level comparison

The four task seeds were averaged within each held-out subject for each model.
The analysis then paired plain EEGNet and bottleneck utility within the same
50 subjects. Loss is defined as plain EEGNet minus bottleneck balanced
accuracy; positive values therefore indicate lower bottleneck utility.

| Measure | Plain compact EEGNet | Bottleneck dim 6 | Paired loss |
| --- | ---: | ---: | ---: |
| Mean subject balanced accuracy | 0.6287 | 0.6256 | 0.0032 |

The paired two-sided 95% subject-bootstrap interval for mean loss is
[-0.0021, 0.0084]. The one-sided 95% upper bound used for the gate is 0.0075.

## Gate decision

| Criterion | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Mean balanced-accuracy loss | < 0.030 | 0.0032 | PASS |
| One-sided 95% upper subject-bootstrap bound | < 0.030 | 0.0075 | PASS |

A passing result authorizes only the frozen attacker families on the saved
plain and bottleneck posterior caches. It is not evidence of a privacy benefit;
privacy promotion still requires the separate multiplicity-controlled attack gate.

## Fold diagnostics

Fold values are descriptive means of ten paired subject aggregates and are not
treated as five independent inferential observations.

| Fold | Plain EEGNet | Bottleneck dim 6 | Loss |
| ---: | ---: | ---: | ---: |
| 1 | 0.5899 | 0.5905 | -0.0006 |
| 2 | 0.6195 | 0.6146 | 0.0049 |
| 3 | 0.6134 | 0.6156 | -0.0022 |
| 4 | 0.6756 | 0.6643 | 0.0113 |
| 5 | 0.6454 | 0.6428 | 0.0026 |

## Reproducibility record

The tracked machine-readable artifact is
[`configs/cho2017_confirmatory_bottleneck_gate_v1.json`](../configs/cho2017_confirmatory_bottleneck_gate_v1.json).
It contains the paired subject aggregates, fold and seed diagnostics,
source result hashes, and cache hashes, but no raw EEG or posterior arrays.

After reproducing the local baseline and bottleneck jobs, verify this record with:

```bash
PYTHONPATH=src .venv/bin/python scripts/freeze_cho2017_bottleneck_gate.py --check
```
