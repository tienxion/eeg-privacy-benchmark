# Cho2017 Confirmatory Cache QA Record

## Status

The pre-model data-quality gate passed on 2026-08-26 for all 50 frozen
confirmatory subjects (`3–52`). No subject met an objective exclusion criterion,
so the study remains within its preregistered stopping rule and may advance to
the staged CSP-LDA and plain EEGNet baseline evaluation.

This record does not contain model results and does not change the frozen
evaluation design.

## Audited cohort

| Check | Result |
| --- | ---: |
| Subjects and checksum-backed MAT files | 50 |
| Total cached bytes | 10,295,233,855 |
| Subjects with 100 trials per class | 47 |
| Subjects with 120 trials per class | 3 |
| Subjects excluded | 0 |
| Files with finite EEG | 50/50 |
| Files with nonzero variance in every EEG channel | 50/50 |
| Files whose event-onset count matches the source trial count | 50/50 |

The two source shapes are expected. Subjects `7`, `9`, and `46` contain 120
trials per class; the other 47 contain 100. This makes the raw source shapes
non-identical while remaining compatible with the fixed post-preprocessing
representation in the confirmatory contract.

## Artifact metadata policy

The audit recorded 775 raw upstream bad-trial index entries across 35 subjects.
These values are provenance metadata, not failed QA checks or exclusions. Their
nested-cell class order and MATLAB/Python index mapping are not authoritatively
documented, and the pinned MOABB loader does not apply them. The primary analysis
therefore retains all MOABB trials, exactly as frozen in the contract. Any later
artifact-filtered analysis requires a separately preregistered sensitivity
protocol.

## Reproducibility record

The tracked machine-readable manifest is
[`configs/cho2017_confirmatory_cache_manifest_v1.json`](../configs/cho2017_confirmatory_cache_manifest_v1.json).
It records every file name, byte count, SHA-256 checksum, source trial count,
event count, and raw artifact-index cell. It contains no raw EEG samples or
machine-specific cache paths.

After acquiring the authorized local cache, reproduce and verify the record with:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_cho2017_cached_subjects.py \
  --mne-data-dir raw_data/mne_data \
  --subjects 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52
PYTHONPATH=src .venv/bin/python scripts/freeze_cho2017_cache_manifest.py --check
```

The QA audit reads local data only and launches neither network access nor model
training. Raw data remains ignored and must not be committed.
