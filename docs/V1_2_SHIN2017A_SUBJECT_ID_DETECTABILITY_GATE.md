# Shin2017A v1.2 Subject-ID Detectability Gate

## Status

The frozen plain-EEGNet cross-session identity-detectability gate passed.
All 12 cache-only probe jobs reproduced with their exact source feature-cache
hashes. No raw EEG was loaded and no task model was retrained.

## Subject-level result

Accuracy was averaged across three temporal folds and four task seeds within
each subject before the 27 subject aggregates were averaged.

| Quantity | Value |
| --- | ---: |
| Chance accuracy (`1/27`) | 0.0370 |
| Required accuracy above chance | 0.1000 |
| Required mean accuracy | 0.1370 |
| Observed mean subject accuracy | 0.3198 |
| Observed accuracy above chance | 0.2827 |
| Descriptive 95% subject-bootstrap interval | [0.2588, 0.3864] |

Gate decision: **PASS**.

The bootstrap interval is descriptive and was not added as a new criterion.
Passing means only that the named linear probe detects identity in the frozen
plain-EEGNet representations. It does not establish a privacy harm beyond this
threat model and does not itself authorize bottleneck execution.

## Diagnostics

Mean job-level macro-F1: `0.2898`.
Mean job-level top-3 accuracy: `0.5637`.

| Fold | Mean accuracy over four seeds |
| ---: | ---: |
| 1 | 0.2597 |
| 2 | 0.3454 |
| 3 | 0.3542 |

## Reproducibility record

The tracked raw-data-free artifact is
[`configs/shin2017a_confirmatory_subject_id_detectability_gate_v1.json`](../configs/shin2017a_confirmatory_subject_id_detectability_gate_v1.json).
It contains source hashes, aggregate metrics, and subject-level accuracies,
but no raw EEG, learned representations, posterior arrays, or trial-level
predictions.

Verify the public historical contract and aggregate evidence with:

```bash
PYTHONPATH=src python scripts/check_shin2017a_public_evidence.py
```
