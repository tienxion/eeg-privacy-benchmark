# Cho2017 Confirmatory Baseline Utility Gate

## Status

The preregistered plain-EEGNet utility gate passed. All 40 frozen baseline jobs and all 20 plain-EEGNet
posterior caches were validated before aggregation. No training, download,
model selection, or privacy attack was performed while producing this record.

## Subject-level utility

Repeated task seeds were averaged within each held-out subject first. The 50
subject aggregates were then treated as the inferential observations. The
interval is a deterministic two-sided 95% percentile bootstrap over subjects
(100,000 replicates; seed `20260826`).

| Model | Mean balanced accuracy | Subject SD | 95% subject-bootstrap CI | Mean macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| CSP-LDA | 0.5916 | 0.1026 | [0.5653, 0.6218] | 0.5415 |
| Plain compact EEGNet | 0.6287 | 0.1037 | [0.6012, 0.6585] | 0.6076 |

## Gate decision

| Criterion | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Plain-EEGNet mean subject balanced accuracy | >= 0.60 | 0.6287 | PASS |
| 95% lower subject-bootstrap bound | > 0.50 | 0.6012 | PASS |

The gate result authorizes only the frozen compact bottleneck EEGNet
(`dim=6`) stage. It is not evidence of a privacy benefit and does not
authorize attacks until the bottleneck utility noninferiority gate passes.

## Fold diagnostics

Fold values are descriptive means of the ten held-out subject aggregates;
they are not treated as five independent inferential observations.

| Fold | CSP-LDA | Plain compact EEGNet |
| ---: | ---: | ---: |
| 1 | 0.5714 | 0.5899 |
| 2 | 0.5640 | 0.6195 |
| 3 | 0.5695 | 0.6134 |
| 4 | 0.6368 | 0.6756 |
| 5 | 0.6164 | 0.6454 |

## Reproducibility record

The tracked machine-readable artifact is
[`configs/cho2017_confirmatory_baseline_gate_v1.json`](../configs/cho2017_confirmatory_baseline_gate_v1.json).
It contains the 50 subject aggregates, fold and seed diagnostics, source
result hashes, and posterior-cache hashes, but no raw EEG or posterior arrays.

After reproducing the 40 local jobs, verify this record with:

```bash
PYTHONPATH=src .venv/bin/python scripts/freeze_cho2017_baseline_gate.py --check
```
