from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from eeg_privacy_benchmark.datasets.splits import resolve_explicit_trial_split
from eeg_privacy_benchmark.shin2017a_validation import (
    build_model_trial_split,
    build_temporal_folds,
    partition_attacker_subjects,
    reserve_training_session_nonmembers,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "shin2017a_temporal_validation_v1.yaml"


@dataclass(frozen=True)
class SyntheticTrial:
    trial_id: str
    subject_id: str
    session_id: str
    canonical_label: str


def _synthetic_trials() -> tuple[SyntheticTrial, ...]:
    return tuple(
        SyntheticTrial(
            trial_id=(
                f"shin2017a:sub-{subject}:ses-{session}:{label}:trial-{trial}"
            ),
            subject_id=f"sub-{subject}",
            session_id=session,
            canonical_label=label,
        )
        for subject in range(3, 30)
        for session in ("0imagery", "2imagery", "4imagery")
        for label in ("left_hand", "right_hand")
        for trial in range(10)
    )


def _raises_value_error(callback) -> bool:
    try:
        callback()
    except ValueError:
        return True
    return False


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    subjects = config["cohort"]["confirmatory_subjects"]
    folds = build_temporal_folds(config["outer_evaluation"]["folds"])
    trials = _synthetic_trials()
    checks: list[tuple[str, bool]] = []

    checks.append(("synthetic_manifest_size", len(trials) == 1620))
    checks.append(("three_temporal_folds", len(folds) == 3))
    session_roles = Counter()
    reservations = []
    for fold in folds:
        session_roles[(fold.task_train_session, "train")] += 1
        session_roles[(fold.validation_session, "validation")] += 1
        session_roles[(fold.task_test_session, "test")] += 1
        reservation = reserve_training_session_nonmembers(
            trials,
            subjects=subjects,
            task_train_session=fold.task_train_session,
            seed=config["seeds"]["nonmember_reservation"],
            fraction=config["membership_design"][
                "nonmember_fraction_per_subject_in_task_training_session"
            ],
        )
        reservations.append(reservation)
        reserved_counts = Counter()
        by_id = {trial.trial_id: trial for trial in trials}
        for trial_id in reservation.nonmember_trial_ids:
            trial = by_id[trial_id]
            reserved_counts[(trial.subject_id, trial.canonical_label)] += 1
        checks.append(
            (
                f"fold_{fold.fold_index}_same_session_reservation",
                len(reservation.member_trial_ids) == 432
                and len(reservation.nonmember_trial_ids) == 108
                and all(count == 2 for count in reserved_counts.values())
                and all(
                    by_id[trial_id].session_id == fold.task_train_session
                    for trial_id in (
                        *reservation.member_trial_ids,
                        *reservation.nonmember_trial_ids,
                    )
                ),
            )
        )

        split = build_model_trial_split(
            trials,
            subjects=subjects,
            fold=fold,
            reservation=reservation,
        )
        resolved = resolve_explicit_trial_split(trials, split)
        checks.append(
            (
                f"fold_{fold.fold_index}_model_roles",
                split.protocol == f"shin2017a_temporal_fold_{fold.fold_index}"
                and len(resolved.train_indices) == 432
                and len(resolved.validation_indices) == 540
                and len(resolved.test_indices) == 540
                and len(resolved.nonmember_indices) == 108
                and all(
                    trials[index].session_id == fold.task_train_session
                    for index in resolved.train_indices
                )
                and all(
                    trials[index].session_id == fold.validation_session
                    for index in resolved.validation_indices
                )
                and all(
                    trials[index].session_id == fold.task_test_session
                    for index in resolved.test_indices
                ),
            )
        )

    checks.append(
        (
            "each_session_occupies_each_role_once",
            all(
                session_roles[(session, role)] == 1
                for session in ("0imagery", "2imagery", "4imagery")
                for role in ("train", "validation", "test")
            ),
        )
    )
    repeated = reserve_training_session_nonmembers(
        trials,
        subjects=subjects,
        task_train_session=folds[0].task_train_session,
        seed=config["seeds"]["nonmember_reservation"],
        fraction=0.20,
    )
    changed = reserve_training_session_nonmembers(
        trials,
        subjects=subjects,
        task_train_session=folds[0].task_train_session,
        seed=config["seeds"]["nonmember_reservation"] + 1,
        fraction=0.20,
    )
    checks.extend(
        [
            ("reservation_deterministic", repeated == reservations[0]),
            ("reservation_seed_sensitive", changed != reservations[0]),
        ]
    )

    attacker = partition_attacker_subjects(
        subjects,
        seed=config["seeds"]["attacker_subject_partition"],
    )
    checks.append(
        (
            "attacker_subject_partition",
            len(attacker.training_subjects) == 13
            and len(attacker.evaluation_subjects) == 14
            and not set(attacker.training_subjects)
            & set(attacker.evaluation_subjects)
            and set(attacker.training_subjects)
            | set(attacker.evaluation_subjects)
            == set(subjects),
        )
    )
    checks.extend(
        [
            (
                "reject_wrong_attacker_cohort",
                _raises_value_error(
                    lambda: partition_attacker_subjects(
                        subjects[:-1],
                        seed=config["seeds"]["attacker_subject_partition"],
                    )
                ),
            ),
            (
                "reject_incomplete_session_rotation",
                _raises_value_error(
                    lambda: build_temporal_folds(
                        config["outer_evaluation"]["folds"][:2]
                    )
                ),
            ),
        ]
    )

    for name, passed in checks:
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    passed_count = sum(passed for _, passed in checks)
    print(f"passed={passed_count}/{len(checks)}")
    if passed_count != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
