# Reproducing Benchmark v1.1

## Provenance

The public release tree is derived from development checkpoint
`5dee43de85a2b4011fc97581efcf65f35fa5a4aa`. Curated evidence is versioned under
`results/v1/` and covered by `release/v1.0.0-manifest.json` plus
`release/SHA256SUMS`.

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
