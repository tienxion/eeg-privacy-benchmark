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

## Local output-stream identity development runner

`scripts/output_stream_identity.py` runs the fixed passive observer used in the
post-release output-stream studies on **your own local binary model scores**.
It needs Python 3.9+ and no third-party packages. It does not load EEG, train or
run a task model, download data, upload anything, or calculate EEG task utility.
It is separate from the frozen v1.0–v1.2 release workflows and stop rules.

Start with a complete, explicitly synthetic example:

```sh
python scripts/output_stream_identity.py demo --output-dir outputs/output-stream-demo
python scripts/check_output_stream_runner.py
```

The demo writes `source.json`, `evaluation.json`, `enrollment.json` and
`summary.json` under that ignored directory. It constructs three artificial
identities with deliberately distinctive outputs; its perfect fine-symbol
accuracy is a plumbing example, **not research evidence**. Existing files are
never overwritten; use another output directory to repeat the demo.

### Input contract

Use the generated demo files as complete schema examples. All inputs are UTF-8
JSON, with no duplicate fields or nonfinite numbers. Inputs are capped at 64 MiB
and 100,000 trials. Only declared fields are accepted; validation data, true
task labels, embeddings, routing metadata and alternate model outputs do not
belong in these inputs.

Enrollment JSON has exactly `schema_version` (1), `model_id` (a stable string),
`score_kind` (`logits` or `probabilities`) and `trials`. Each trial has exactly:

```json
{"trial_id": "source-trial-1", "identity": 2, "scores": [0.8, 0.2]}
```

This is one record, not a complete enrollment file. Supply 2–256 identities as
distinct nonnegative integers, with the **same nonzero number of enrollment
trials per identity**. Numeric identity ordering defines the exact likelihood
tie rule. Trial identifiers must be unique. Scores are two finite numbers in
consistent task-class order. Probabilities must lie in [0,1] and sum to one
within 1e-5; they are normalized before quantization. Exact argmax ties choose
task class zero.

Evaluation JSON has exactly `schema_version` (1), the same `model_id` and
`score_kind`, and `bags`. Each bag has `identity` and `trials`; each trial has
`trial_id` and `scores` but **no identity field**. All bags must contain exactly
the chosen number of observations, belong to one enrolled person, and occur in
equal numbers per person. Source and evaluation trial identifiers must be
disjoint, and evaluation identifiers cannot repeat across bags. Grouping is
explicit; the runner does not infer sessions/runs or establish chronological
adjacency. Never combine outputs from different models in one input.

Identity and trial metadata are used by the evaluator for allocation checks
and scoring, not supplied as features to the prediction kernel. The observer
receives only emitted symbols, identity-labeled enrollment and the knowledge
that each bag comes from one of the enrolled people. If an observer knows the
EEG input or identifying routing metadata, this restricted threat model does
not describe it.

### Fit first, record the table hash, evaluate second

Prepare enrollment and held-out bags using a protocol fixed before looking at
the evaluation outcomes. Then run:

```sh
python scripts/output_stream_identity.py fit \
  --source outputs/your-enrollment.json \
  --stream-budget 10 \
  --output outputs/your-output-stream-table.json
```

`fit` opens only the enrollment input and prints `enrollment_sha256=...`.
Record that exact 64-character digest outside the editable table before
evaluation; for a prospective study, commit its hash and protocol first.
The budget may be 2–128. There is no bin, prior or classifier tuning option:
sixteen fine symbols, pseudocount one, coherent hard-label pseudocount eight,
uniform identity priors and exact product comparisons are fixed.

Replace `COPY_RECORDED_DIGEST_HERE` below with the recorded digest, not a newly
computed digest of a potentially changed table:

```sh
python scripts/output_stream_identity.py evaluate \
  --enrollment outputs/your-output-stream-table.json \
  --expected-sha256 COPY_RECORDED_DIGEST_HERE \
  --evaluation outputs/your-held-out-bags.json \
  --output outputs/your-output-stream-result.json
```

The recorded hash is checked before the held-out input is opened. Evaluation
reports hard-label and confidence-symbol identity BA/F1 at K=1 and the fixed
stream budget, using all the same observations, and the exact primary stream
BA difference. It writes aggregate metrics, not individual predictions.
Generated files use owner-only permissions on systems supporting POSIX modes;
inputs and enrollment tables nevertheless contain sensitive identity data and
must stay private. Review any aggregate output before sharing it.

### Limits of the safeguards and of reproduction

Hash checking binds the table bytes; it does not certify that enrollment was
outcome-blind. Duplicate checks trust supplied identifiers: relabeling a copied
trial defeats them. A matching `model_id` is a caller assertion, not verified
checkpoint provenance. Decide participant roles, model provenance, observation
grouping, utility prerequisites and study stopping rules independently.

This implements the attack calculation, not the upstream model/data pipeline.
The separate aggregate checker validates the published arithmetic; neither
command alone regenerates the historical EEG models or cohorts. Low attack
accuracy does not prove privacy, and confidence symbols contain the hard label
even when this fitted rule performs worse with them. Suppressing confidence
removes confidence-dependent utility. The single-symbol binary alphabet bound
does not apply to a stream. No deployment policy, membership-defense claim or
new significance claim follows from running this tool.
