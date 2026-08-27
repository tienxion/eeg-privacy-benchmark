# Cho2017 Confirmatory Validation Contract

## Study objective

Determine whether the frozen compact bottleneck EEGNet (`dim=6`) transfers to
Cho2017 without materially reducing cross-subject task utility and whether it
reduces black-box trial-membership leakage versus architecture-matched compact
EEGNet. CSP-LDA remains a task-utility anchor.

The public `v1.0.0` evidence remains frozen, and the two-subject Cho pilot is
treated as plumbing-only rather than confirmatory evidence.

## Why the design differs from the pilot

Cho2017 has one MOABB session and one run per subject. It therefore cannot
replicate the existing cross-session subject-identification claim. A naive
membership attack using training subjects as members and unseen test subjects as
nonmembers would also confound membership with subject/domain shift.

The confirmatory design instead reserves nonmember trials within every fitting
subject before task-model fitting. Members and nonmembers therefore come from the
same subject pool. Attacker-training and attacker-evaluation subjects remain
disjoint.

## Cohort and outer folds

Pilot subjects 1 and 2 are permanently excluded from confirmatory estimates.
Subjects 3–52 form the confirmatory cohort. Partition seed `20260826` produces
five frozen 10-subject blocks:

1. `51,30,8,52,29,46,3,13,44,19`
2. `32,24,45,25,38,27,31,42,26,40`
3. `17,20,23,6,34,18,15,21,4,39`
4. `36,50,47,49,9,14,48,28,10,11`
5. `16,35,12,22,43,7,5,41,37,33`

For outer fold `k`, block `k` is the task test set, the next block cyclically is
the task validation set, and the remaining three blocks are the fitting set.
Every fold therefore has 30 fitting, 10 validation, and 10 test subjects. Fold
assignment never changes with model or attacker seeds.

## Data-quality gate

Before any model metric is produced, the project must record cache checksums and
verify labels, session/run identity, channel order, epoch shape, finite signals,
and nonzero channel variance. Each class should contain 100 or 120 source trials.
More than two subjects failing these objective checks stops the confirmatory
study; exclusions must be decided without inspecting model outcomes.

Every MAT file contains upstream `bad_trial_indices`. The source paper identifies
the two artifact fields as voltage-magnitude and EMG-correlation criteria, but it
does not explicitly document the left/right order of each field's nested cells or
their index base. MOABB does not apply these indices. The primary analysis is
therefore frozen to the reproducible MOABB all-trial path, while recording the raw
artifact metadata for audit. It must not silently remove trials using inferred
semantics. An artifact-filtered result is allowed only as a separately labelled,
preregistered sensitivity analysis after authoritative mapping evidence exists.

MOABB 1.2.0 reconstructs a continuous recording from trial segments using zero
buffers and warns about edge effects. That dataset-specific limitation remains
mandatory in every report.

## Frozen evaluation

- Neural preprocessing: 8–32 Hz, 0–3 seconds, 250 Hz, per-trial/channel
  normalization, global-average classifier head.
- Neural training: learning rate `0.001`, batch size `32`, 60 epochs, patience
  `15`.
- Models: CSP-LDA (`8` components, native-rate anchor), compact EEGNet, and
  compact bottleneck EEGNet (`dim=6`).
- Task seeds: `13,17,19,23`.
- Attacker seeds: `101,103,107`.
- Forbidden additions after results: alternate bottleneck dimensions,
  adversarial EEGNet, mixup, confidence penalty, or new hyperparameters.

Within each fitting subject, 20% of class-balanced trials are reserved as strict
nonmembers before task fitting. Outer validation and test subjects are not used
for membership inference. The frozen attacker families are threshold scores,
logistic regression, and MLP posterior attacks. Threshold calibration uses only
attacker-training subjects.

## Promotion and stop rules

Plain EEGNet must reach mean subject-level balanced accuracy of at least `0.60`,
with its 95% lower confidence bound above `0.50`. Otherwise the study stops as a
negative task-transfer result.

Bottleneck utility is noninferior only when its mean balanced-accuracy loss is
less than `0.030` and a one-sided 95% subject-cluster interval supports that
margin. Failure stops privacy promotion.

Privacy promotion requires at least `0.010` lower AUC in at least two of the
three attacker families, multiplicity-adjusted interval support, consistent
direction across all three attacker seeds, and no attacker-family regression
larger than `0.010`.

Subjects are the inferential units. Trials, outer folds, task seeds, and attacker
seeds are repeated measurements, not independent samples. Unfavorable completed
results remain in the record. Changing a fold, seed, model, attack, threshold, or
preprocessing choice after inspecting outcomes starts a separately labelled
exploratory study.

## Staging

1. Complete metadata, checksum, signal-integrity, and raw artifact-metadata QA.
2. Run CSP-LDA and plain compact EEGNet across all frozen folds.
3. Apply the baseline utility gate.
4. Run the bottleneck model only if baseline utility is valid.
5. Apply bottleneck utility noninferiority.
6. Run attacks from frozen posterior caches without retraining task models.
7. Produce subject-cluster inference and a claim audit.

Only one heavy job runs at a time.

The machine-readable source of truth is
[`configs/cho2017_confirmatory_validation_v1.yaml`](../configs/cho2017_confirmatory_validation_v1.yaml).

## Dataset provenance

- [Original Cho et al. dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5493744/)
- [MOABB Cho2017 dataset documentation](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Cho2017.html)
- [Pinned MOABB 1.2.0 Cho2017 loader](https://github.com/NeuroTechX/moabb/blob/v1.2.0/moabb/datasets/gigadb.py)
