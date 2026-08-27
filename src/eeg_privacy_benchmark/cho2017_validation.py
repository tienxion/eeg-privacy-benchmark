"""Deterministic split primitives for the preregistered Cho2017 validation."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable, Protocol

from eeg_privacy_benchmark.datasets.splits import ExplicitTrialSplit


class TrialLike(Protocol):
    trial_id: str
    subject_id: str
    canonical_label: str


@dataclass(frozen=True)
class ChoOuterFold:
    fold_index: int
    fitting_subjects: tuple[int, ...]
    validation_subjects: tuple[int, ...]
    test_subjects: tuple[int, ...]


@dataclass(frozen=True)
class MembershipReservation:
    fitting_trial_ids: tuple[str, ...]
    nonmember_trial_ids: tuple[str, ...]


@dataclass(frozen=True)
class AttackerSubjectPartition:
    training_subjects: tuple[int, ...]
    evaluation_subjects: tuple[int, ...]


def build_model_trial_split(
    trial_records: Iterable[TrialLike],
    *,
    fold: ChoOuterFold,
    reservation: MembershipReservation,
) -> ExplicitTrialSplit:
    """Build exact task-model roles while leaving reserved nonmembers unused."""

    records = tuple(trial_records)
    record_ids = [record.trial_id for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Cho2017 trial records contain duplicate trial IDs")

    expected_subjects = {
        *(f"sub-{subject}" for subject in fold.fitting_subjects),
        *(f"sub-{subject}" for subject in fold.validation_subjects),
        *(f"sub-{subject}" for subject in fold.test_subjects),
    }
    observed_subjects = {record.subject_id for record in records}
    if observed_subjects != expected_subjects:
        raise ValueError("Cho2017 trial records do not match the fold cohort")

    fitting_subjects = {f"sub-{subject}" for subject in fold.fitting_subjects}
    fitting_ids = {
        record.trial_id for record in records if record.subject_id in fitting_subjects
    }
    member_ids = set(reservation.fitting_trial_ids)
    nonmember_ids = set(reservation.nonmember_trial_ids)
    if member_ids & nonmember_ids or member_ids | nonmember_ids != fitting_ids:
        raise ValueError("membership reservation does not cover the fitting subjects")

    validation_subjects = {f"sub-{subject}" for subject in fold.validation_subjects}
    test_subjects = {f"sub-{subject}" for subject in fold.test_subjects}
    validation_ids = tuple(
        sorted(
            record.trial_id
            for record in records
            if record.subject_id in validation_subjects
        )
    )
    test_ids = tuple(
        sorted(
            record.trial_id for record in records if record.subject_id in test_subjects
        )
    )
    return ExplicitTrialSplit(
        protocol=f"cho2017_confirmatory_fold_{fold.fold_index}",
        train_trial_ids=tuple(sorted(member_ids)),
        validation_trial_ids=validation_ids,
        test_trial_ids=test_ids,
        nonmember_trial_ids=tuple(sorted(nonmember_ids)),
    )


def build_outer_folds(blocks: Iterable[Iterable[int]]) -> tuple[ChoOuterFold, ...]:
    """Build cyclic 30/10/10 fitting/validation/test roles from five blocks."""

    frozen_blocks = tuple(tuple(int(subject) for subject in block) for block in blocks)
    if len(frozen_blocks) != 5 or any(len(block) != 10 for block in frozen_blocks):
        raise ValueError("Cho2017 validation requires five 10-subject blocks")
    flattened = [subject for block in frozen_blocks for subject in block]
    if len(set(flattened)) != 50:
        raise ValueError("Cho2017 blocks must contain 50 unique subjects")

    all_subjects = set(flattened)
    folds = []
    for index in range(5):
        test = set(frozen_blocks[index])
        validation = set(frozen_blocks[(index + 1) % 5])
        fitting = all_subjects - test - validation
        folds.append(
            ChoOuterFold(
                fold_index=index + 1,
                fitting_subjects=tuple(sorted(fitting)),
                validation_subjects=tuple(sorted(validation)),
                test_subjects=tuple(sorted(test)),
            )
        )
    return tuple(folds)


def _stable_group_seed(seed: int, *parts: str) -> int:
    material = ":".join([str(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def reserve_membership_nonmembers(
    trial_records: Iterable[TrialLike],
    *,
    fitting_subjects: Iterable[int],
    seed: int,
    fraction: float = 0.20,
) -> MembershipReservation:
    """Reserve class-balanced nonmembers within fitting subjects only."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("nonmember fraction must be between zero and one")
    fitting_subject_ids = {f"sub-{int(subject)}" for subject in fitting_subjects}
    grouped: dict[tuple[str, str], list[str]] = {}
    fitting_ids: set[str] = set()
    for record in trial_records:
        if record.subject_id not in fitting_subject_ids:
            continue
        key = (record.subject_id, record.canonical_label)
        grouped.setdefault(key, []).append(record.trial_id)
        fitting_ids.add(record.trial_id)
    if not grouped:
        raise ValueError("no fitting-subject trials were provided")

    nonmember_ids: set[str] = set()
    for (subject_id, label), trial_ids in sorted(grouped.items()):
        ordered = sorted(trial_ids)
        reserve_count = int(len(ordered) * fraction)
        if reserve_count < 1 or reserve_count >= len(ordered):
            raise ValueError(
                f"group {subject_id}/{label} cannot support fraction {fraction}"
            )
        rng = random.Random(_stable_group_seed(seed, subject_id, label))
        rng.shuffle(ordered)
        nonmember_ids.update(ordered[:reserve_count])

    member_ids = fitting_ids - nonmember_ids
    if not member_ids or not nonmember_ids or member_ids & nonmember_ids:
        raise RuntimeError("invalid member/nonmember reservation")
    return MembershipReservation(
        fitting_trial_ids=tuple(sorted(member_ids)),
        nonmember_trial_ids=tuple(sorted(nonmember_ids)),
    )


def partition_attacker_subjects(
    fitting_subjects: Iterable[int],
    *,
    seed: int,
    fold_index: int,
) -> AttackerSubjectPartition:
    """Split 30 fitting subjects into disjoint 15-subject attacker roles."""

    subjects = sorted({int(subject) for subject in fitting_subjects})
    if len(subjects) != 30:
        raise ValueError("attacker partition requires exactly 30 fitting subjects")
    rng = random.Random(_stable_group_seed(seed, f"fold-{fold_index}"))
    rng.shuffle(subjects)
    return AttackerSubjectPartition(
        training_subjects=tuple(sorted(subjects[:15])),
        evaluation_subjects=tuple(sorted(subjects[15:])),
    )
