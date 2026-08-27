"""Deterministic split generation for benchmark evaluation protocols."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable

from eeg_privacy_benchmark.datasets.base import DatasetManifest, TrialRecord


@dataclass(frozen=True)
class SplitAssignment:
    """Assignment of one trial to a split."""

    trial_id: str
    split: str


@dataclass(frozen=True)
class SplitManifest:
    """Frozen split assignments for one evaluation run."""

    dataset_key: str
    protocol: str
    seed: int
    subject_ids: tuple[str, ...]
    assignments: tuple[SplitAssignment, ...]


@dataclass(frozen=True)
class ExplicitTrialSplit:
    """Exact model-facing trial roles for a preregistered evaluation."""

    protocol: str
    train_trial_ids: tuple[str, ...]
    validation_trial_ids: tuple[str, ...]
    test_trial_ids: tuple[str, ...]
    nonmember_trial_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedExplicitTrialSplit:
    """Array indices aligned to an explicit trial split."""

    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    nonmember_indices: tuple[int, ...]


def resolve_explicit_trial_split(
    trial_records: Iterable[TrialRecord],
    split: ExplicitTrialSplit,
) -> ResolvedExplicitTrialSplit:
    """Resolve exact trial IDs to aligned indices with strict leakage guards."""

    records = tuple(trial_records)
    record_ids = [record.trial_id for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("trial records contain duplicate trial IDs")

    role_ids = {
        "train": tuple(split.train_trial_ids),
        "validation": tuple(split.validation_trial_ids),
        "test": tuple(split.test_trial_ids),
        "nonmember": tuple(split.nonmember_trial_ids),
    }
    if any(not role_ids[name] for name in ("train", "validation", "test")):
        raise ValueError("explicit train, validation, and test roles must be non-empty")
    if any(len(set(ids)) != len(ids) for ids in role_ids.values()):
        raise ValueError("an explicit trial role contains duplicate IDs")

    train_ids = set(role_ids["train"])
    validation_ids = set(role_ids["validation"])
    test_ids = set(role_ids["test"])
    nonmember_ids = set(role_ids["nonmember"])
    all_roles = (train_ids, validation_ids, test_ids, nonmember_ids)
    if any(
        left & right
        for index, left in enumerate(all_roles)
        for right in all_roles[index + 1 :]
    ):
        raise ValueError("explicit trial roles must be disjoint")
    missing = (train_ids | validation_ids | test_ids | nonmember_ids) - set(record_ids)
    if missing:
        raise ValueError(f"explicit trial split references {len(missing)} unknown IDs")

    index_by_id = {trial_id: index for index, trial_id in enumerate(record_ids)}
    return ResolvedExplicitTrialSplit(
        train_indices=tuple(sorted(index_by_id[trial_id] for trial_id in train_ids)),
        validation_indices=tuple(
            sorted(index_by_id[trial_id] for trial_id in validation_ids)
        ),
        test_indices=tuple(sorted(index_by_id[trial_id] for trial_id in test_ids)),
        nonmember_indices=tuple(
            sorted(index_by_id[trial_id] for trial_id in nonmember_ids)
        ),
    )


def _group_trials_by_subject(
    trial_records: tuple[TrialRecord, ...],
) -> dict[str, list[TrialRecord]]:
    grouped: dict[str, list[TrialRecord]] = {}
    for record in trial_records:
        grouped.setdefault(record.subject_id, []).append(record)
    return grouped


def _group_trials_by_session(
    trial_records: tuple[TrialRecord, ...],
) -> dict[str, list[TrialRecord]]:
    grouped: dict[str, list[TrialRecord]] = {}
    for record in trial_records:
        session_id = record.session_id or "unknown_session"
        grouped.setdefault(session_id, []).append(record)
    return grouped


def _group_trials_by_run(
    trial_records: tuple[TrialRecord, ...],
) -> dict[str, list[TrialRecord]]:
    grouped: dict[str, list[TrialRecord]] = {}
    for record in trial_records:
        run_id = record.run_id or "unknown_run"
        grouped.setdefault(run_id, []).append(record)
    return grouped


def generate_cross_subject_split(
    manifest: DatasetManifest,
    *,
    seed: int,
    train_fraction: float = 0.8,
) -> SplitManifest:
    """Split by subject so no subject appears in both train and test."""

    grouped = _group_trials_by_subject(manifest.trial_records)
    subject_ids = sorted(grouped)
    if len(subject_ids) < 2:
        raise ValueError(
            f"Need at least two subjects for cross-subject split on {manifest.dataset_key}."
        )

    rng = Random(seed)
    rng.shuffle(subject_ids)
    cutoff = max(1, min(len(subject_ids) - 1, int(len(subject_ids) * train_fraction)))
    train_subjects = set(subject_ids[:cutoff])

    assignments = tuple(
        SplitAssignment(
            trial_id=record.trial_id,
            split="train" if record.subject_id in train_subjects else "test",
        )
        for record in manifest.trial_records
    )
    return SplitManifest(
        dataset_key=manifest.dataset_key,
        protocol="cross_subject",
        seed=seed,
        subject_ids=tuple(sorted(subject_ids)),
        assignments=assignments,
    )


def generate_cross_run_split(
    manifest: DatasetManifest,
    *,
    seed: int,
) -> SplitManifest:
    """Split by recording run for datasets with repeated runs but one session."""

    grouped = _group_trials_by_run(manifest.trial_records)
    run_ids = sorted(grouped)
    if len(run_ids) < 2:
        raise ValueError(
            f"Need at least two runs for cross-run split on {manifest.dataset_key}."
        )

    rng = Random(seed)
    rng.shuffle(run_ids)
    train_run = run_ids[0]

    assignments = tuple(
        SplitAssignment(
            trial_id=record.trial_id,
            split="train" if record.run_id == train_run else "test",
        )
        for record in manifest.trial_records
    )
    return SplitManifest(
        dataset_key=manifest.dataset_key,
        protocol="cross_run",
        seed=seed,
        subject_ids=tuple(sorted({record.subject_id for record in manifest.trial_records})),
        assignments=assignments,
    )


def generate_cross_session_split(
    manifest: DatasetManifest,
    *,
    seed: int,
) -> SplitManifest:
    """Split by session for datasets with repeated sessions."""

    grouped = _group_trials_by_session(manifest.trial_records)
    session_ids = sorted(grouped)
    if len(session_ids) < 2:
        raise ValueError(
            f"Need at least two sessions for cross-session split on {manifest.dataset_key}."
        )

    rng = Random(seed)
    rng.shuffle(session_ids)
    train_session = session_ids[0]

    assignments = tuple(
        SplitAssignment(
            trial_id=record.trial_id,
            split="train" if record.session_id == train_session else "test",
        )
        for record in manifest.trial_records
    )
    return SplitManifest(
        dataset_key=manifest.dataset_key,
        protocol="cross_session",
        seed=seed,
        subject_ids=tuple(sorted({record.subject_id for record in manifest.trial_records})),
        assignments=assignments,
    )
