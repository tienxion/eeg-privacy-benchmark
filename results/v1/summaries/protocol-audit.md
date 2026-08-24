# Protocol Verification Record

This record documents the benchmark protocol details verified in the frozen release code and the constraints that apply when reporting them.

Companion files:
- [`protocol-audit.csv`](protocol-audit.csv)
- [`protocol-dependency-defaults.json`](protocol-dependency-defaults.json)

## Summary

- Verified protocol items: `18`.
- Verified-with-caveat items: `0`.
- Installed dependency snapshot: MOABB `1.2.0`, MNE `1.8.0`, PyTorch `2.8.0`, scikit-learn `1.5.2`.
- The MOABB preprocessing defaults are verified against the installed MOABB source snapshot; final paper text should cite MOABB and report the dependency version.
- Splits, task-model defaults, privacy-attack definitions, and report-layer membership semantics are pinned in the release code.

## Installed MOABB LeftRightImagery Snapshot

- Module: `moabb.paradigms.motor_imagery`.
- Source module: `moabb.paradigms.motor_imagery` from MOABB 1.2.0.
- Constructor signature: `(**kwargs)`.
- Parent MotorImagery signature: `(n_classes=None, **kwargs)`.
- Events: `['left_hand', 'right_hand']`.
- Filters: `[[8, 32]]`.
- Time window: `tmin=0.0`, `tmax=None`.
- Resampling: `None`.
- Channels: `None`.

## Audit Matrix

| Component | Status | Verified detail | Source | Paper-safe language | Follow-up |
| --- | --- | --- | --- | --- | --- |
| benchmark scope | verified | Version-1 config lists PhysioNet Motor Imagery, BNCI2014-001, and Lee2019_MI with a left_vs_right_motor_imagery task. | `configs/benchmark_v1.yaml` | We evaluate binary left-versus-right motor imagery on three public MOABB-backed datasets. | Dataset citations are recorded in `paper/references.bib` and `DATASETS.md`. |
| dataset registry | verified | Dataset keys map to MOABB source IDs PhysionetMI, BNCI2014_001, and Lee2019_MI; BNCI and Lee are marked as two-session datasets. | `src/eeg_privacy_benchmark/datasets/registry.py` | The benchmark uses fixed dataset identifiers and records session structure for cross-session evaluation where available. | Dataset access and reuse boundaries are documented in `DATASETS.md`. |
| MOABB loading | verified | The loader instantiates MOABB's LeftRightImagery() with no explicit preprocessing arguments, then calls paradigm.get_data(dataset=..., subjects=...). Installed-source introspection of MOABB 1.2.0 reports filters [[8, 32]], tmin 0.0, tmax None, resample None, channels None, baseline None, reject None, and events left_hand/right_hand. | `src/eeg_privacy_benchmark/datasets/moabb_loader.py; installed moabb/paradigms/motor_imagery.py` | Trial arrays are obtained through MOABB's LeftRightImagery paradigm using the installed MOABB 1.2.0 defaults. | For final venue formatting, cite MOABB and place the dependency snapshot in the reproducibility table or appendix. |
| trial metadata | verified | Trial records preserve dataset key, subject_id, session_id, run_id, trial_id, raw_label, and canonical_label from MOABB metadata. | `src/eeg_privacy_benchmark/datasets/moabb_loader.py` | Each extracted trial is aligned with subject, session, run, and label metadata. | None for local metadata construction. |
| cross-subject split | verified | Subjects are sorted, shuffled with Python Random(seed), and split by subject with an 80 percent default train fraction while forcing non-empty train/test subject sets. | `src/eeg_privacy_benchmark/datasets/splits.py` | Cross-subject evaluation holds out subjects, so no subject appears in both task-model train and test splits. | Random subject-subset validation is recorded in `physionet-random-subset-validation.md`. |
| cross-session split | verified | Sessions are sorted, shuffled with Python Random(seed), and one session is assigned to train while the remaining session(s) are assigned to test. | `src/eeg_privacy_benchmark/datasets/splits.py` | Cross-session evaluation trains on one session and tests on held-out session data from the same subjects. | Confirm session naming/order from generated manifests if reporting per-session details. |
| cross-run split | verified | Runs are sorted, shuffled with Python Random(seed), and one run is assigned to train while the remaining run(s) are assigned to test. | `src/eeg_privacy_benchmark/datasets/splits.py` | Cross-run evaluation trains on one recording run and tests on held-out run data from the same subjects. | Use this for PhysioNet protocol-sensitivity checks because MOABB exposes one session but repeated run IDs for the left/right imagery trials. |
| EEGNet normalization | verified | Deep models cast features to float32 and normalize using per-channel mean/std estimated only from the post-validation training subset. | `src/eeg_privacy_benchmark/models/eegnet.py` | Deep-model inputs are standardized using training-subset statistics and then applied to validation and test trials. | None; this is pinned in local code. |
| EEGNet validation | verified | A stratified validation subset is drawn from task-model training indices using validation_fraction=0.2 by default. | `src/eeg_privacy_benchmark/models/eegnet.py` | Deep models use an internal stratified validation split for early stopping. | Report validation_fraction if describing training details. |
| EEGNet training defaults | verified | Default training uses 20 epochs, batch size 32, Adam learning rate 1e-3, and early-stopping patience 5. | `src/eeg_privacy_benchmark/models/eegnet.py; src/eeg_privacy_benchmark/cli.py` | Unless otherwise stated, EEGNet-family task models use 20 requested epochs, batch size 32, learning rate 1e-3, and early stopping on validation loss. | When writing final tables, prefer result JSON epochs_trained over requested epochs. |
| EEGNet architecture | verified | The compact EEGNet uses F1=8, depth_multiplier=2, F2=16, temporal/depthwise/separable-style convolutions, ELU activations, average pooling, dropout p=0.25, and a linear classifier. | `src/eeg_privacy_benchmark/models/eegnet.py` | We use a compact EEGNet-style convolutional decoder with dropout and optional bottleneck/noise/regularization variants. | If space allows, include a concise architecture table rather than prose-only architecture details. |
| CSP-LDA baseline | verified | CSP-LDA defaults to n_components=8, CSP(reg=None, log=True, norm_trace=False), and scikit-learn LinearDiscriminantAnalysis. | `src/eeg_privacy_benchmark/models/csp_lda.py` | The classical baseline is CSP followed by LDA with eight CSP components unless otherwise specified. | None for local defaults. |
| subject-ID probe | verified | Subject-identification probes are restricted to cross_session, fit LogisticRegression(max_iter=2000), and evaluate subject prediction from frozen task-model embeddings or CSP features. | `src/eeg_privacy_benchmark/privacy/subject_id.py` | Subject-identification leakage is measured by a linear probe on frozen features in the cross-session setting. | Do not apply subject-ID conclusions to PhysioNet cross-subject runs without a separate protocol. |
| membership labels | verified | For EEGNet-family membership inference, task-model train plus validation trials are members and held-out test trials are nonmembers. | `src/eeg_privacy_benchmark/privacy/membership_inference.py` | Membership inference distinguishes task-model train/validation trials from held-out test trials. | Make this member definition explicit because validation trials are included as members. |
| threshold membership attacks | verified | Threshold attacks support label_known_log_probability, max_probability, and negative_entropy scalar scores and select the best threshold on the evaluation set. | `src/eeg_privacy_benchmark/privacy/membership_inference.py` | Threshold attacks use scalar confidence scores and report AUC, average precision, and best-threshold balanced accuracy. | Mention that best-threshold balanced accuracy is an attack diagnostic, not a deployment protocol. |
| learned membership attacks | verified | Learned logistic-regression and MLP attacks use posterior probabilities with or without a true-label one-hot feature; member and nonmember sets are split 50/50 into attack-train and attack-eval partitions. | `src/eeg_privacy_benchmark/privacy/membership_inference.py` | Learned attacks train separate classifiers on task-model posterior features and evaluate on held-out attack examples. | For final claims, keep learned(logistic) and learned(mlp) families separate because rankings differ. |
| MLP attacker defaults | verified | The nonlinear learned attacker is StandardScaler plus MLPClassifier(hidden_layer_sizes=(32,), relu, alpha=1e-4, learning_rate_init=1e-3, max_iter=500, early_stopping=True, n_iter_no_change=20). | `src/eeg_privacy_benchmark/privacy/membership_inference.py` | The nonlinear learned membership attacker is a one-hidden-layer MLP trained on posterior-vector features. | If the paper emphasizes attacker strength, add repeated attacker refits as a future or supplemental check. |
| defense parameters | verified | CLI defaults include label_smoothing=0.1, bottleneck_dim=8, feature_noise_std=0.1, mixup_alpha=0.2, confidence_penalty_beta=0.05, adversarial_weight=0.5, adversarial_schedule=constant, and ramp_up_fraction=0.4; many paper artifacts report explicit non-default variants. | `src/eeg_privacy_benchmark/cli.py; results/v1/PROVENANCE.md` | Defense variants are reported with explicit hyperparameters in result tables and captions. | Always cite the hyperparameter printed in the artifact label, not only the CLI default. |

## Methods Text To Prefer

We obtained left-versus-right motor-imagery trial arrays through MOABB's `LeftRightImagery` paradigm using the repository's installed dependency defaults. The benchmark then applied deterministic subject-, session-, or run-level splits, trained CSP-LDA and EEGNet-family task models, and evaluated subject-identification and membership-inference leakage on held-out split partitions.

For deep models, validation trials are drawn from the task-model training partition and are treated as members for membership-inference evaluation. Inputs are standardized using statistics from the post-validation training subset only. Membership attacks are reported by attack family because threshold, learned logistic, and nonlinear learned attackers can induce different defense rankings.

## Interpretation Constraints

- MOABB preprocessing defaults are version-bound and must be tied to the dependency snapshot.
- PhysioNet does not support the same subject-identification interpretation as the BNCI/Lee cross-session probes.
- Validation trials are members under the frozen membership definition.
- Explicit artifact hyperparameters take precedence over CLI defaults.

## Venue-Specific Reporting

- Report the frozen `moabb`, `mne`, `numpy`, `torch`, and `scikit-learn` versions in the environment table or appendix.
- Preserve the dataset, model, and attack citations recorded in `paper/references.bib`.
- Report trial counts from generated manifests when a venue requests per-protocol appendix detail.
