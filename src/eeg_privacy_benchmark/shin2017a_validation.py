"""Deterministic split primitives for the preregistered Shin2017A validation."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable, Protocol

from eeg_privacy_benchmark.datasets.splits import ExplicitTrialSplit


class TrialLike(Protocol):
    trial_id: str
    subject_id: str
    session_id: str | None
    canonical_label: str


@dataclass(frozen=True)
class ShinTemporalFold:
    fold_index: int
    task_train_session: str
    validation_session: str
    task_test_session: str


@dataclass(frozen=True)
class ShinMembershipReservation:
    member_trial_ids: tuple[str, ...]
    nonmember_trial_ids: tuple[str, ...]


@dataclass(frozen=True)
class ShinAttackerSubjectPartition:
    training_subjects: tuple[int, ...]
    evaluation_subjects: tuple[int, ...]


def _stable_seed(seed: int, *parts: str) -> int:
    material = ":".join([str(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def build_temporal_folds(
    fold_rows: Iterable[dict[str, object]],
) -> tuple[ShinTemporalFold, ...]:
    """Build the exact three-fold train/validation/test session rotation."""

    folds = tuple(
        ShinTemporalFold(
            fold_index=int(row["fold"]),
            task_train_session=str(row["task_train_session"]),
            validation_session=str(row["early_stopping_session"]),
            task_test_session=str(row["task_test_session"]),
        )
        for row in fold_rows
    )
    if len(folds) != 3 or [fold.fold_index for fold in folds] != [1, 2, 3]:
        raise ValueError("Shin2017A validation requires folds 1, 2, and 3")
    sessions = {
        folds[0].task_train_session,
        folds[0].validation_session,
        folds[0].task_test_session,
    }
    if len(sessions) != 3:
        raise ValueError("each Shin2017A fold requires three distinct sessions")
    for fold in folds:
        if {
            fold.task_train_session,
            fold.validation_session,
            fold.task_test_session,
        } != sessions:
            raise ValueError("every Shin2017A fold must cover the same sessions")
    for session in sessions:
        roles = (
            sum(fold.task_train_session == session for fold in folds),
            sum(fold.validation_session == session for fold in folds),
            sum(fold.task_test_session == session for fold in folds),
        )
        if roles != (1, 1, 1):
            raise ValueError("each Shin2017A session must occupy each role once")
    return folds


def reserve_training_session_nonmembers(
    trial_records: Iterable[TrialLike],
    *,
    subjects: Iterable[int],
    task_train_session: str,
    seed: int,
    fraction: float = 0.20,
) -> ShinMembershipReservation:
    """Reserve class-balanced nonmembers within the task-training session."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("nonmember fraction must be between zero and one")
    expected_subjects = {f"sub-{int(subject)}" for subject in subjects}
    grouped: dict[tuple[str, str], list[str]] = {}
    observed_subjects: set[str] = set()
    for record in trial_records:
        if record.session_id != task_train_session:
            continue
        if record.subject_id not in expected_subjects:
            raise ValueError("task-training session contains an unexpected subject")
        grouped.setdefault(
            (record.subject_id, record.canonical_label), []
        ).append(record.trial_id)
        observed_subjects.add(record.subject_id)
    if observed_subjects != expected_subjects:
        raise ValueError("task-training session does not contain the full cohort")

    member_ids: set[str] = set()
    nonmember_ids: set[str] = set()
    for (subject_id, label), trial_ids in sorted(grouped.items()):
        ordered = sorted(trial_ids)
        reserve_count = int(len(ordered) * fraction)
        if reserve_count < 1 or reserve_count >= len(ordered):
            raise ValueError(
                f"group {subject_id}/{label} cannot support fraction {fraction}"
            )
        rng = random.Random(
            _stable_seed(seed, task_train_session, subject_id, label)
        )
        rng.shuffle(ordered)
        nonmember_ids.update(ordered[:reserve_count])
        member_ids.update(ordered[reserve_count:])
    if not member_ids or not nonmember_ids or member_ids & nonmember_ids:
        raise RuntimeError("invalid Shin2017A member/nonmember reservation")
    return ShinMembershipReservation(
        member_trial_ids=tuple(sorted(member_ids)),
        nonmember_trial_ids=tuple(sorted(nonmember_ids)),
    )


def build_model_trial_split(
    trial_records: Iterable[TrialLike],
    *,
    subjects: Iterable[int],
    fold: ShinTemporalFold,
    reservation: ShinMembershipReservation,
) -> ExplicitTrialSplit:
    """Build exact model roles while keeping reserved nonmembers unused."""

    records = tuple(trial_records)
    record_ids = [record.trial_id for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Shin2017A trial records contain duplicate trial IDs")
    expected_subjects = {f"sub-{int(subject)}" for subject in subjects}
    if {record.subject_id for record in records} != expected_subjects:
        raise ValueError("Shin2017A trial records do not match the frozen cohort")
    expected_sessions = {
        fold.task_train_session,
        fold.validation_session,
        fold.task_test_session,
    }
    if {str(record.session_id) for record in records} != expected_sessions:
        raise ValueError("Shin2017A trial records do not match the frozen sessions")

    task_train_ids = {
        record.trial_id
        for record in records
        if record.session_id == fold.task_train_session
    }
    member_ids = set(reservation.member_trial_ids)
    nonmember_ids = set(reservation.nonmember_trial_ids)
    if member_ids & nonmember_ids or member_ids | nonmember_ids != task_train_ids:
        raise ValueError("membership reservation does not cover task-training trials")

    validation_ids = tuple(
        sorted(
            record.trial_id
            for record in records
            if record.session_id == fold.validation_session
        )
    )
    test_ids = tuple(
        sorted(
            record.trial_id
            for record in records
            if record.session_id == fold.task_test_session
        )
    )
    return ExplicitTrialSplit(
        protocol=f"shin2017a_temporal_fold_{fold.fold_index}",
        train_trial_ids=tuple(sorted(member_ids)),
        validation_trial_ids=validation_ids,
        test_trial_ids=test_ids,
        nonmember_trial_ids=tuple(sorted(nonmember_ids)),
    )


def partition_attacker_subjects(
    subjects: Iterable[int],
    *,
    seed: int,
) -> ShinAttackerSubjectPartition:
    """Freeze disjoint 13/14-subject attacker training/evaluation roles."""

    resolved = sorted({int(subject) for subject in subjects})
    if len(resolved) != 27:
        raise ValueError("Shin2017A attacker partition requires 27 subjects")
    rng = random.Random(_stable_seed(seed, "shin2017a-attacker-subjects"))
    rng.shuffle(resolved)
    return ShinAttackerSubjectPartition(
        training_subjects=tuple(sorted(resolved[:13])),
        evaluation_subjects=tuple(sorted(resolved[13:])),
    )
