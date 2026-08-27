# Datasets and Data Rights

This repository contains no EEG recordings. It stores only small split metadata
and code that can access upstream datasets through provider-specific helpers or
MOABB/MNE. Users must review the current upstream terms before downloading or
processing data.

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

## Cho2017

- MOABB documentation:
  https://moabb.neurotechx.com/docs/generated/moabb.datasets.Cho2017.html
- Data record: https://doi.org/10.5524/100295
- Benchmark key: `cho2017`
- v1.1 scope: pilot subjects 1-2 are excluded from confirmation; confirmatory
  subjects are 3-52 under five frozen outer subject folds.
- Citation: Cho et al. (2017), The Korea University EEG data set for motor
  imagery brain-computer interface, GigaScience, doi:10.1093/gigascience/gix034.

The v1.1 manifest records checksums and source-shape QA, not EEG samples. Review
the current GigaDB record terms before downloading or processing the data. The
full confirmatory cache is approximately 10.3 GB, remains under ignored
`raw_data/mne_data/`, and is not redistributed by this repository.

## Shin2017A / NEMAR nm000267

- Official record: https://nemar.org/dataset/nm000267
- Dataset DOI: https://doi.org/10.82901/nemar.nm000267
- Benchmark key: `shin2017a`
- Frozen v1.2 provider/version: NEMAR `nm000267`, `v1.0.3`
- v1.2 scope: subjects 1-2 were pilot-only; confirmatory subjects are 3-29;
  only imagery sessions `0imagery`, `2imagery`, and `4imagery` are evaluated.
- Citation: Shin et al. (2017), Open access dataset for EEG+NIRS single-trial
  classification, IEEE Transactions on Neural Systems and Rehabilitation
  Engineering, doi:10.1109/TNSRE.2016.2628057.

The official record currently identifies GPL-3.0 licensing and provides the
authoritative download/citation information. Review the current record before
using the data. This repository includes no EEG recordings and grants no
dataset rights.

The public acquisition helper defaults to a plan and requires two separate
flags before downloading: `--confirm-download` and
`--accept-dataset-terms`. The latter records only the current user's
acknowledgement; the historical project authorization is not transferable.
Downloads remain under ignored `raw_data/mne_data/NEMAR-nm000267-v1.0.3/`.

## Local acquisition

```bash
eeg-privacy-benchmark build-manifest \
  --dataset bnci2014_001 --subjects 1 \
  --mne-data-dir raw_data/mne_data \
  --output data/manifests/bnci-subject1.json
```

This explicit command may download missing upstream files. Use
`check-dataset-cache` first when downloads are not desired.
