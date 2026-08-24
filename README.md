# EEG Privacy Benchmark v1

A reproducible benchmark for asking a narrow question: when an EEG decoder is
trained for left- versus right-hand motor imagery, what subject-identity and
training-membership signals remain in its representations and outputs?

The benchmark evaluates BNCI2014-001, Lee2019 MI/OpenBMI, and PhysioNet EEG
Motor Movement/Imagery with CSP-LDA, compact EEGNet variants, subject-identity
probes, and threshold, logistic, and nonlinear membership attacks. Privacy
objectives are reported separately. A favorable metric is evidence only for the
named attacker, split, model, and release surface; it is not a general privacy
guarantee.

## Main v1 findings

- Bottleneck EEGNet gives a qualified utility/subject-identification tradeoff on
  BNCI and Lee, but not an attack-invariant membership defense.
- PhysioNet has no supported default: rankings change by attacker and protocol.
- The PhysioNet federated comparison is diagnostic only because mean task
  balanced accuracy falls below the preregistered 0.60 utility gate.
- The pairwise-mask secure-aggregation simulator matches FedAvg numerically but
  is 217.77x slower in the measured local aggregation benchmark. It is not
  production cryptography and does not protect model outputs, membership,
  malicious clients, dropout, collusion, authentication, or metadata.

See [RESULTS.md](RESULTS.md), the [venue-neutral manuscript](paper/manuscript.md),
and the [curated v1 evidence](results/v1/README.md).

## Installation

Python 3.9 or newer is supported. The exact top-level versions used for the
validated v1 environment are in `requirements-lock.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

The dependency set includes PyTorch, MNE, and MOABB. Installation can therefore
be large even though the no-training checks are lightweight.

## Lightweight quickstart

These commands do not download EEG data or train a model:

```bash
python scripts/check_public_smoke.py
python scripts/check_public_release.py --allow-remote origin
eeg-privacy-benchmark --help
```

Inspect a local cache without downloading missing files:

```bash
eeg-privacy-benchmark check-dataset-cache \
  --dataset bnci2014_001 --subjects 1 \
  --mne-data-dir raw_data/mne_data
```

## Data acquisition

No EEG recordings are included. Review [DATASETS.md](DATASETS.md) and each
upstream dataset's current access terms before downloading. Dataset access is
performed through MOABB/MNE only when an explicit manifest or training command
is run. Keep all downloads under ignored `raw_data/mne_data/`.

## Reproduction tiers

- Tier 0, seconds: inspect the committed manuscript, tables, summaries, release
  manifest, and checksums. No environment or data is required.
- Tier 1, minutes: install dependencies and run public smoke/release checks. No
  data, downloads, or training are used.
- Tier 2, minutes to hours: acquire one upstream dataset, generate manifests,
  and run a one-subject or dry-run workflow.
- Tier 3, hours to days: reproduce frozen multi-seed neural, federated, and
  attacker suites. Use sequential jobs and thread caps on laptops; see
  [REPRODUCING.md](REPRODUCING.md).

## Repository layout

- `src/eeg_privacy_benchmark/`: benchmark package and CLI
- `configs/`: frozen v1 and explicitly labeled diagnostic configurations
- `scripts/`: runners, plan/result tools, smoke tests, and release validation
- `data/splits/`: small frozen split definitions
- `raw_data/`: ignored local dataset cache placeholder
- `outputs/`: ignored local run-output placeholder
- `paper/`: venue-neutral manuscript and bibliography
- `results/v1/`: curated aggregate evidence, figures, tables, and checksums

## Licensing and citation

Code is Apache-2.0. The manuscript, figures, tables, and curated report assets
are CC BY 4.0. Dataset rights remain with their upstream providers and are not
granted by this repository. See [LICENSE_SCOPE.md](LICENSE_SCOPE.md),
[DATASETS.md](DATASETS.md), and [CITATION.cff](CITATION.cff).

## Release status

The `v1.0.0-rc1` tag is the first public release candidate. The versioned
release manifest records its private source checkpoint and curated-evidence
checksums.
