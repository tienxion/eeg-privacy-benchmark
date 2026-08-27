# Reproducing Benchmark v1.2

## Provenance

The frozen v1 core is derived from development checkpoint
`5dee43de85a2b4011fc97581efcf65f35fa5a4aa`; the v1.1 and v1.2 scientific
source checkpoints are `c737c1a` and `0ed3c3c`. Curated evidence is separated
under `results/v1/`, `results/v1.1/`, and `results/v1.2/`, with a versioned
manifest and checksum file for each release.

## No-download validation

```bash
python scripts/check_public_smoke.py
python scripts/check_public_release.py --allow-remote origin
```

The smoke suite compiles Python, parses YAML/CFF, imports the package, checks the
CLI, exercises synthetic dataset-cache and privacy plumbing, verifies secure
aggregation against the curated result, and checks manuscript/table structure.
It neither downloads data nor trains models.

## Dataset and split preparation

Read `DATASETS.md`, create `raw_data/mne_data/`, then run
`check-dataset-cache`. `build-manifest` may download missing files and must be an
explicit user action. Frozen split examples are under `data/splits/`.

The Cho2017 v1.1 contract and split plumbing can be checked without EEG data:

```bash
PYTHONPATH=src python scripts/check_cho2017_validation_contract.py
PYTHONPATH=src python scripts/check_cho2017_split_plumbing.py
```

Inspect the confirmatory acquisition plan before downloading:

```bash
PYTHONPATH=src python scripts/acquire_cho2017_confirmatory_cache.py
```

Actual acquisition requires `--confirm-download`, can require approximately
10.3 GB, and writes only to ignored local cache paths. The model and attack
runners default to dry runs and require `--confirm-run`; use `--max-jobs 1` for
bounded first execution. The complete sequence and stop rules are frozen in
`docs/V1_1_CHO2017_VALIDATION_CONTRACT.md`.

## Training and attack reproduction

The root CLI runs individual baselines, defenses, privacy attacks, and sweeps.
Scripts named `run_*` provide the frozen multi-seed and plan-based workflows.
Plan generators write stable `_v1` artifacts under ignored `outputs/` and can be
reviewed before execution.

On a laptop, run jobs sequentially with thread caps:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts/run_federated_multiseed.py --help
```

Neural and federated suites can require hours to days depending on hardware and
dataset cache state. No command in CI launches these suites.

## Evidence interpretation

Always apply the utility gate before interpreting privacy deltas. In particular,
the PhysioNet federated result is diagnostic because mean task balanced accuracy
is 0.5185, below the frozen 0.60 gate. The secure-aggregation result establishes
functional equivalence and measured local overhead only.

For Cho2017, both utility gates passed, but the three-family membership gate did
not. Reproducing a favorable individual attack row does not override the tracked
`NO_PROMOTION` decision in
`configs/cho2017_confirmatory_privacy_gate_v1.json`.

## Shin2017A v1.2 historical evidence

The committed v1.2 contract and aggregate gate artifacts can be checked without
EEG data, downloads, model training, or privacy-probe fitting:

```bash
PYTHONPATH=src python scripts/check_shin2017a_public_evidence.py
PYTHONPATH=src python scripts/check_shin2017a_split_plumbing.py
PYTHONPATH=src python scripts/check_shin2017a_nemar_loader.py
```

The contract is a historical execution record. Its project-owner acceptance
fields do not accept the NEMAR terms for another user. Before downloading,
review https://nemar.org/dataset/nm000267 and inspect the bounded plan:

```bash
PYTHONPATH=src python scripts/acquire_shin2017a_nemar.py
```

Actual acquisition requires both explicit flags:

```bash
PYTHONPATH=src python scripts/acquire_shin2017a_nemar.py \
  --confirmatory --confirm-download --accept-dataset-terms
```

This command writes only to the ignored local cache and performs no model
training. The public v1.2 release preserves aggregate evidence and no-download
plumbing; internal authorization records and local feature/result caches are
not release artifacts.
