# EEG Privacy Benchmark v1.2

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

## v1.1 Cho2017 confirmatory extension

v1.1 adds a preregistered transfer evaluation on 50 previously untouched
Cho2017 subjects. Plain compact EEGNet reached mean subject balanced accuracy
`0.6287` (95% subject-bootstrap CI `[0.6012, 0.6585]`). The fixed compact
bottleneck (`dim=6`) reached `0.6256`; its paired utility loss was `0.0032`,
with a one-sided 95% upper bound of `0.0075` below the frozen `0.030` margin.

The subsequent 600-job cache-only membership evaluation did **not** promote a
privacy claim. Mean AUC reductions for threshold, logistic-regression, and MLP
attacker families were `-0.00003`, `+0.00002`, and `+0.00140`; all
Holm-controlled intervals crossed zero, and `0/3` families passed versus `2/3`
required. The supported result is utility noninferiority, not a Cho2017
membership-privacy improvement.

See the [v1.1 evidence index](results/v1.1/README.md), [validation
contract](docs/V1_1_CHO2017_VALIDATION_CONTRACT.md), and [claim
audit](docs/V1_1_CHO2017_CLAIM_AUDIT.md).

## v1.2 Shin2017A temporal validation

v1.2 adds a preregistered 27-subject, three-session temporal validation using
Shin2017A imagery sessions from NEMAR `nm000267` version `v1.0.3`. Plain compact
EEGNet passed its temporal-utility gate at mean subject balanced accuracy
`0.6341` (95% subject-bootstrap CI `[0.6008, 0.6705]`). A cache-only linear
probe reached mean cross-session subject-ID accuracy `0.3198`, compared with
27-subject chance `0.0370`, so the frozen identity-detectability prerequisite
passed for that named threat model.

The fixed compact bottleneck (`dim=6`) completed all 12 jobs at mean subject
balanced accuracy `0.6137`. Its paired mean utility loss was `0.02037`, inside
the frozen `0.030` margin, but the preregistered one-sided 95% upper loss bound
was `0.03071`. Noninferiority was therefore not established under the strict
rule, and the study stopped before bottleneck identity promotion or membership
evaluation. This is a narrow negative temporal-transfer result, not a supported
Shin2017A privacy-defense claim.

See the [historical contract](configs/shin2017a_temporal_validation_v1.yaml),
[aggregate evidence check](scripts/check_shin2017a_public_evidence.py), and
[claim audit](docs/V1_2_SHIN2017A_CLAIM_AUDIT.md). The complete curated record
is indexed under [results/v1.2](results/v1.2/README.md).

## Post-release output-stream development studies

For a passive observer with identity-labeled enrollment outputs, revealing fixed
confidence bands increased achieved twelve-output identity BA from 19.91% to
46.99% on nine reused BNCI participants. The advantage was not reproduced under
a different Shin protocol: ten-output BA was 7.10% with confidence versus 7.41%
with labels on 27 reused participants. These are different enrollment/model/
session designs, not an exact replication or a general privacy guarantee.

The [research log](docs/RESEARCH_LOG.md) reports both outcomes and the earlier
negative results. A [small aggregate bundle](results/development/output-stream-identity.json)
and [standard-library checker](scripts/check_output_stream_evidence.py) let readers
verify the arithmetic and figure without data or dependencies. They do not
reproduce the full attack pipeline. This development update does not change
the frozen v1.0–v1.2 releases, package version, or original Shin stop rule.

A separate [local output-stream runner](scripts/output_stream_identity.py) now
fits and evaluates the same fixed attacker on caller-supplied binary scores.
Try `python scripts/output_stream_identity.py demo` without installing any
dependencies. The demo is synthetic, not research evidence. See the
[two-stage input and evaluation instructions](REPRODUCING.md#local-output-stream-identity-development-runner)
before using your own outputs; enrollment tables and trial-level inputs must
stay private. This tool does not train an EEG model or reproduce a dataset.

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
performed through explicit provider-specific helpers or MOABB/MNE commands.
Keep all downloads under ignored `raw_data/mne_data/`.

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
- `data/`: frozen split definitions and metadata documentation
- `paper/`: venue-neutral manuscript and bibliography
- `results/v1/`: curated aggregate evidence, figures, tables, and checksums
- `results/v1.1/`: Cho2017 confirmatory-extension evidence index
- `results/v1.2/`: Shin2017A temporal-validation evidence index
- `docs/V1_2_SHIN2017A_CLAIM_AUDIT.md`: Shin2017A temporal-validation claim boundary

Dataset caches, generated manifests, and run outputs are written to the ignored
`raw_data/`, `data/manifests/`, and `outputs/` paths when required; generated
artifacts are not part of the source distribution.

## Licensing and citation

Code is Apache-2.0. The manuscript, figures, tables, and curated report assets
are CC BY 4.0. Dataset rights remain with their upstream providers and are not
granted by this repository. See [LICENSE_SCOPE.md](LICENSE_SCOPE.md),
[DATASETS.md](DATASETS.md), and [CITATION.cff](CITATION.cff).

## Release status

The `v1.0.0` evidence remains frozen. v1.1 adds the separate Cho2017
confirmatory extension, and v1.2 adds the stopped Shin2017A temporal validation;
neither revises the original BNCI, Lee, or PhysioNet claims. Versioned release
manifests record reviewed source boundaries and curated evidence checksums.
Secret-scan and manuscript-render checks for v1 are recorded in
[the v1 publication QA note](release/PUBLICATION_QA.md). Public inventory,
links, paths, licenses, commit identity, and checksums are validated by
`scripts/check_public_release.py`.
