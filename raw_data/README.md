# Raw data

Use this directory for downloaded EEG recordings and dataset caches.

Recommended layout:
- `raw_data/mne_data/` for MNE and MOABB downloads

Notes:
- Some MOABB datasets, including `lee2019_mi`, use dataset-specific cache keys in addition to `MNE_DATA`. The benchmark CLI now maps `--mne-data-dir` into those keys so cache resolution stays under `raw_data/mne_data/`.
- If a dataset is only partially present in `raw_data/mne_data/`, MOABB will still try to download the missing files.
- Use `PYTHONPATH=src .venv/bin/python -m eeg_privacy_benchmark.cli check-dataset-cache --mne-data-dir raw_data/mne_data` to inspect cache completeness before running a manifest or training command.
- Use `PYTHONPATH=src .venv/bin/python -m eeg_privacy_benchmark.cli import-dataset-cache ...` to copy expected files from an existing external MNE/MOABB cache into this repo-local cache.

Do not commit downloaded recordings.
