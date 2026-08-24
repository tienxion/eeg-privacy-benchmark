# Datasets and Data Rights

This repository contains no EEG recordings. It stores only small split metadata
and code that can access upstream datasets through MOABB/MNE. Users must review
the current upstream terms before downloading or processing data.

## PhysioNet EEG Motor Movement/Imagery Dataset

- Upstream record: https://physionet.org/content/eegmmidb/1.0.0/
- Benchmark key: `physionet_motor_imagery`
- Scope: six imagined left/right movement runs per subject; primary v1 analysis
  uses subjects 1-46 and a cross-subject protocol.
- Citations: Schalk, G. (2009), EEG Motor Movement/Imagery Dataset; Goldberger et
  al. (2000), PhysioBank, PhysioToolkit, and PhysioNet; Schalk et al. (2004),
  BCI2000.

PhysioNet displays the dataset's current credentialing and license information
on the upstream record. Those terms, not this repository's licenses, govern the
recordings.

## BNCI2014-001 / BCI Competition IV 2a

- MOABB documentation:
  https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html
- Benchmark key: `bnci2014_001`
- Scope: binary left-hand versus right-hand imagery with cross-session splits.
- Citation: Tangermann et al. (2012), Review of the BCI Competition IV,
  Frontiers in Neuroscience, doi:10.3389/fnins.2012.00055.

MOABB provides acquisition code but does not transfer dataset rights. Review the
dataset page reached through MOABB and the original provider's current terms.

## Lee2019 MI / OpenBMI

- Data record: https://doi.org/10.5524/100542
- Data paper: https://doi.org/10.1093/gigascience/giz002
- Benchmark key: `lee2019_mi`
- Scope: binary left-hand versus right-hand imagery with cross-session splits.
- Citation: Lee et al. (2019), EEG dataset and OpenBMI toolbox for three BCI
  paradigms, GigaScience.

Review the GigaDB record's current access and reuse terms. This repository does
not assert additional rights or redistribute the recordings.

## Local acquisition

```bash
eeg-privacy-benchmark build-manifest \
  --dataset bnci2014_001 --subjects 1 \
  --mne-data-dir raw_data/mne_data \
  --output data/manifests/bnci-subject1.json
```

This explicit command may download missing upstream files. Use
`check-dataset-cache` first when downloads are not desired.
