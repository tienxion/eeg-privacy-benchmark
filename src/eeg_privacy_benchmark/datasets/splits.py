"""Deterministic split generation for benchmark evaluation protocols."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

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
