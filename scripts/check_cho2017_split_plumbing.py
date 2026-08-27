from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from eeg_privacy_benchmark.cho2017_validation import (
    build_model_trial_split,
    build_outer_folds,
    partition_attacker_subjects,
    reserve_membership_nonmembers,
)
from eeg_privacy_benchmark.datasets.splits import (
    ExplicitTrialSplit,
    resolve_explicit_trial_split,
)
from eeg_privacy_benchmark.models.csp_lda import fit_csp_lda
from eeg_privacy_benchmark.models.eegnet import (
    _resolve_training_validation_indices,
    fit_bottleneck_eegnet,
    fit_eegnet,
    run_bottleneck_eegnet,
)
from eeg_privacy_benchmark.privacy.membership_inference import (
    PosteriorFeatureCache,
    _centralized_membership_indices,
    _import_attack_dependencies,
    _resolve_attacker_index_partition,
    _resolve_learned_attack_indices,
    _run_membership_attack_from_cache,
    load_posterior_feature_cache,
    run_eegnet_membership_inference_from_artifacts,
    write_posterior_feature_cache,
)
import eeg_privacy_benchmark.privacy.membership_inference as membership_inference


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cho2017_confirmatory_validation_v1.yaml"


@dataclass(frozen=True)
class SyntheticTrial:
    trial_id: str
    subject_id: str
    canonical_label: str


def _synthetic_trials() -> tuple[SyntheticTrial, ...]:
    return tuple(
        SyntheticTrial(
            trial_id=f"cho2017:sub-{subject}:{label}:trial-{trial}",
            subject_id=f"sub-{subject}",
            canonical_label=label,
        )
        for subject in range(3, 53)
        for label in ("left_hand", "right_hand")
        for trial in range(10)
    )


def _raises_value_error(callback) -> bool:
    try:
        callback()
    except ValueError:
        return True
    return False


def _synthetic_partitioned_cache(attacker) -> PosteriorFeatureCache:
    subjects = [*attacker.training_subjects, *attacker.evaluation_subjects]
    member_indices = list(range(30))
    nonmember_indices = list(range(30, 60))
    trial_subject_ids = [f"sub-{subject}" for subject in subjects] * 2
    training_subjects = set(attacker.training_subjects)
    scores = [
        0.90 if subject in training_subjects else 0.70 for subject in subjects
    ] + [
        0.60 if subject in training_subjects else 0.55 for subject in subjects
    ]
    probabilities = np.asarray([[score, 1.0 - score] for score in scores])
    return PosteriorFeatureCache(
        task_model="eegnet",
        dataset_key="cho2017_synthetic",
        protocol="cho2017_confirmatory_fold_1",
        seed=13,
        subjects=subjects,
        task_balanced_accuracy=0.65,
        task_macro_f1=0.64,
        class_probabilities=probabilities,
        encoded_labels=np.zeros(60, dtype=int),
        member_indices=member_indices,
        nonmember_indices=nonmember_indices,
        trial_subject_ids=trial_subject_ids,
        attacker_training_subjects=list(attacker.training_subjects),
        attacker_evaluation_subjects=list(attacker.evaluation_subjects),
    )


class _RecordingAttackModel:
    def __init__(self, calls: list[tuple[str, tuple[float, ...]]]):
        self.calls = calls

    def fit(self, features, labels):
        del labels
        self.calls.append(("fit", tuple(sorted(set(features[:, 0])))))
        return self

    def predict_proba(self, features):
        self.calls.append(("predict", tuple(sorted(set(features[:, 0])))))
        scores = features[:, 0]
        return np.column_stack([1.0 - scores, scores])


def _run_fake_learned_attack(cache, attack_type: str):
    calls: list[tuple[str, tuple[float, ...]]] = []
    dependencies = _import_attack_dependencies()
    dependencies["LogisticRegression"] = lambda **kwargs: _RecordingAttackModel(calls)
    dependencies["MLPClassifier"] = lambda **kwargs: object()
    dependencies["Pipeline"] = lambda steps: _RecordingAttackModel(calls)
    with patch.object(
        membership_inference,
        "_import_attack_dependencies",
        return_value=dependencies,
    ):
        result = _run_membership_attack_from_cache(
            cache,
            attack_type=attack_type,
            score_type="posterior_probabilities",
            attack_seed=101,
        )
    return result, calls


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    blocks = config["cohort"]["shuffled_subject_blocks"]
    folds = build_outer_folds(blocks)
    trials = _synthetic_trials()
    checks: list[tuple[str, bool]] = []

    checks.append(("five_outer_folds", len(folds) == 5))
    role_counts = Counter()
    reservations = []
    for fold in folds:
        checks.append(
            (
                f"fold_{fold.fold_index}_roles",
                len(fold.fitting_subjects) == 30
                and len(fold.validation_subjects) == 10
                and len(fold.test_subjects) == 10
                and not (
                    set(fold.fitting_subjects) & set(fold.validation_subjects)
                    or set(fold.fitting_subjects) & set(fold.test_subjects)
                    or set(fold.validation_subjects) & set(fold.test_subjects)
                ),
            )
        )
        for subject in fold.fitting_subjects:
            role_counts[(subject, "fit")] += 1
        for subject in fold.validation_subjects:
            role_counts[(subject, "validation")] += 1
        for subject in fold.test_subjects:
            role_counts[(subject, "test")] += 1

        reservation = reserve_membership_nonmembers(
            trials,
            fitting_subjects=fold.fitting_subjects,
            seed=config["seeds"]["nonmember_reservation"],
            fraction=config["membership_design"]["nonmember_fraction_per_fitting_subject"],
        )
        reservations.append(reservation)
        nonmember_counts = Counter()
        for trial_id in reservation.nonmember_trial_ids:
            _, subject_id, label, _ = trial_id.split(":")
            nonmember_counts[(subject_id, label)] += 1
        checks.append(
            (
                f"fold_{fold.fold_index}_class_balanced_reservation",
                len(reservation.nonmember_trial_ids) == 120
                and len(reservation.fitting_trial_ids) == 480
                and all(count == 2 for count in nonmember_counts.values()),
            )
        )

        model_split = build_model_trial_split(
            trials,
            fold=fold,
            reservation=reservation,
        )
        resolved = resolve_explicit_trial_split(trials, model_split)
        train_indices = resolved.train_indices
        validation_indices = resolved.validation_indices
        test_indices = resolved.test_indices
        assigned_ids = {
            *(trials[index].trial_id for index in train_indices),
            *(trials[index].trial_id for index in validation_indices),
            *(trials[index].trial_id for index in test_indices),
        }
        checks.append(
            (
                f"fold_{fold.fold_index}_model_facing_indices",
                model_split.protocol
                == f"cho2017_confirmatory_fold_{fold.fold_index}"
                and len(train_indices) == 480
                and len(validation_indices) == 200
                and len(test_indices) == 200
                and len(resolved.nonmember_indices) == 120
                and not assigned_ids & set(reservation.nonmember_trial_ids)
                and {
                    trials[index].trial_id for index in resolved.nonmember_indices
                }
                == set(reservation.nonmember_trial_ids),
            )
        )
        outer_subject_ids = {
            *(f"sub-{subject}" for subject in fold.validation_subjects),
            *(f"sub-{subject}" for subject in fold.test_subjects),
        }
        checks.append(
            (
                f"fold_{fold.fold_index}_outer_subjects_excluded",
                all(
                    trial_id.split(":")[1] not in outer_subject_ids
                    for trial_id in (
                        *reservation.fitting_trial_ids,
                        *reservation.nonmember_trial_ids,
                    )
                ),
            )
        )

        attacker = partition_attacker_subjects(
            fold.fitting_subjects,
            seed=config["seeds"]["attacker_subject_partition"],
            fold_index=fold.fold_index,
        )
        checks.append(
            (
                f"fold_{fold.fold_index}_attacker_partition",
                len(attacker.training_subjects) == 15
                and len(attacker.evaluation_subjects) == 15
                and not set(attacker.training_subjects) & set(attacker.evaluation_subjects)
                and set(attacker.training_subjects) | set(attacker.evaluation_subjects)
                == set(fold.fitting_subjects),
            )
        )

    checks.append(
        (
            "each_subject_role_counts",
            all(
                role_counts[(subject, "fit")] == 3
                and role_counts[(subject, "validation")] == 1
                and role_counts[(subject, "test")] == 1
                for subject in range(3, 53)
            ),
        )
    )
    repeated = reserve_membership_nonmembers(
        trials,
        fitting_subjects=folds[0].fitting_subjects,
        seed=config["seeds"]["nonmember_reservation"],
        fraction=0.20,
    )
    changed = reserve_membership_nonmembers(
        trials,
        fitting_subjects=folds[0].fitting_subjects,
        seed=config["seeds"]["nonmember_reservation"] + 1,
        fraction=0.20,
    )
    checks.extend(
        [
            ("reservation_exactly_deterministic", repeated == reservations[0]),
            ("reservation_seed_is_effective", changed != reservations[0]),
            (
                "model_apis_accept_explicit_split",
                "explicit_trial_split" in inspect.signature(fit_eegnet).parameters
                and "explicit_trial_split"
                in inspect.signature(fit_csp_lda).parameters
                and "explicit_trial_split"
                in inspect.signature(fit_bottleneck_eegnet).parameters
                and "explicit_trial_split"
                in inspect.signature(run_bottleneck_eegnet).parameters
                and "attacker_subject_partition"
                in inspect.signature(
                    run_eegnet_membership_inference_from_artifacts
                ).parameters,
            ),
            (
                "explicit_validation_bypasses_random_builder",
                _resolve_training_validation_indices(
                    None,
                    [3, 1],
                    None,
                    13,
                    0.2,
                    explicit_validation_indices=[7, 5],
                )
                == ([1, 3], [5, 7]),
            ),
            (
                "reserved_nonmembers_drive_membership_cache_roles",
                _centralized_membership_indices(
                    SimpleNamespace(
                        train_indices=(1, 2),
                        validation_indices=(3,),
                        test_indices=(4,),
                        nonmember_indices=(5, 6),
                    )
                )
                == ([1, 2], [5, 6]),
            ),
            (
                "explicit_split_rejects_overlap",
                _raises_value_error(
                    lambda: resolve_explicit_trial_split(
                        trials,
                        ExplicitTrialSplit(
                            protocol="invalid",
                            train_trial_ids=(trials[0].trial_id,),
                            validation_trial_ids=(trials[0].trial_id,),
                            test_trial_ids=(trials[1].trial_id,),
                        ),
                    )
                ),
            ),
            (
                "explicit_split_rejects_unknown_id",
                _raises_value_error(
                    lambda: resolve_explicit_trial_split(
                        trials,
                        ExplicitTrialSplit(
                            protocol="invalid",
                            train_trial_ids=(trials[0].trial_id,),
                            validation_trial_ids=(trials[1].trial_id,),
                            test_trial_ids=("unknown-trial",),
                        ),
                    )
                ),
            ),
        ]
    )

    attacker = partition_attacker_subjects(
        folds[0].fitting_subjects,
        seed=config["seeds"]["attacker_subject_partition"],
        fold_index=folds[0].fold_index,
    )
    partitioned_cache = _synthetic_partitioned_cache(attacker)
    attack_indices = _resolve_attacker_index_partition(
        member_indices=partitioned_cache.member_indices,
        nonmember_indices=partitioned_cache.nonmember_indices,
        trial_subject_ids=partitioned_cache.trial_subject_ids,
        attacker_training_subjects=partitioned_cache.attacker_training_subjects,
        attacker_evaluation_subjects=partitioned_cache.attacker_evaluation_subjects,
    )
    member_train, member_eval, nonmember_train, nonmember_eval = (
        _resolve_learned_attack_indices(
            None,
            member_indices=partitioned_cache.member_indices,
            nonmember_indices=partitioned_cache.nonmember_indices,
            seed=101,
            attack_index_partition=attack_indices,
        )
    )
    training_subject_ids = {f"sub-{subject}" for subject in attacker.training_subjects}
    evaluation_subject_ids = {
        f"sub-{subject}" for subject in attacker.evaluation_subjects
    }
    checks.append(
        (
            "learned_attacker_uses_subject_partition_without_random_split",
            attack_indices is not None
            and len(member_train) == len(nonmember_train) == 15
            and len(member_eval) == len(nonmember_eval) == 15
            and {
                partitioned_cache.trial_subject_ids[index]
                for index in member_train + nonmember_train
            }
            == training_subject_ids
            and {
                partitioned_cache.trial_subject_ids[index]
                for index in member_eval + nonmember_eval
            }
            == evaluation_subject_ids,
        )
    )
    threshold_result = _run_membership_attack_from_cache(
        partitioned_cache,
        attack_type="threshold",
        score_type="max_probability",
        attack_seed=101,
    )
    checks.append(
        (
            "threshold_calibrates_on_training_and_reports_evaluation_subjects",
            abs(threshold_result.best_threshold - 0.90) < 1e-12
            and abs(threshold_result.attack_balanced_accuracy - 0.50) < 1e-12
            and threshold_result.member_trials == 15
            and threshold_result.nonmember_trials == 15
            and threshold_result.attacker_training_subjects
            == list(attacker.training_subjects)
            and threshold_result.attacker_evaluation_subjects
            == list(attacker.evaluation_subjects),
        )
    )
    for attack_type in ("logistic_regression", "mlp"):
        learned_result, learned_calls = _run_fake_learned_attack(
            partitioned_cache,
            attack_type,
        )
        checks.append(
            (
                f"{attack_type}_fits_train_and_reports_evaluation_subjects",
                learned_calls
                == [
                    ("fit", (0.6, 0.9)),
                    ("predict", (0.55, 0.7)),
                    ("predict", (0.6, 0.9)),
                ]
                and learned_result.member_trials == 15
                and learned_result.nonmember_trials == 15
                and learned_result.attacker_evaluation_subjects
                == list(attacker.evaluation_subjects),
            )
        )
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "partitioned_cache.npz"
        write_posterior_feature_cache(partitioned_cache, cache_path)
        loaded_cache = load_posterior_feature_cache(cache_path)
        checks.append(
            (
                "attacker_subject_partition_cache_round_trip",
                loaded_cache.trial_subject_ids
                == partitioned_cache.trial_subject_ids
                and loaded_cache.attacker_training_subjects
                == partitioned_cache.attacker_training_subjects
                and loaded_cache.attacker_evaluation_subjects
                == partitioned_cache.attacker_evaluation_subjects,
            )
        )
    checks.append(
        (
            "explicit_attacker_partition_rejects_uncovered_subjects",
            _raises_value_error(
                lambda: _resolve_attacker_index_partition(
                    member_indices=partitioned_cache.member_indices,
                    nonmember_indices=partitioned_cache.nonmember_indices,
                    trial_subject_ids=[
                        *partitioned_cache.trial_subject_ids[:-1],
                        "sub-999",
                    ],
                    attacker_training_subjects=(
                        partitioned_cache.attacker_training_subjects
                    ),
                    attacker_evaluation_subjects=(
                        partitioned_cache.attacker_evaluation_subjects
                    ),
                )
            ),
        )
    )

    for name, passed in checks:
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    passed_count = sum(passed for _, passed in checks)
    print(f"passed={passed_count}/{len(checks)}")
    if passed_count != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
