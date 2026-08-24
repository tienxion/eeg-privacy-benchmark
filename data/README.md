# Data Metadata

This directory contains the small, non-recording metadata required to reproduce
the benchmark protocols.

- `splits/` contains versioned train/test assignments used by the documented
  benchmark examples.
- `manifests/` is created locally when dataset metadata are generated and is
  excluded from version control.

EEG recordings are never stored in this directory. Dataset acquisition and
rights information is documented in [`DATASETS.md`](../DATASETS.md).
