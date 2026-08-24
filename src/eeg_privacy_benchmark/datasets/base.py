"""Base dataset abstractions for the EEG privacy benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class TrialRecord:
    """Canonical metadata row for one EEG trial."""

    dataset: str
    subject_id: str
    session_id: str | None
    run_id: str | None
    trial_id: str
    raw_label: str
    canonical_label: str | None


@dataclass(frozen=True)
class DatasetManifest:
    """Trial metadata and task coverage for one dataset."""

    dataset_key: str
    trial_records: tuple[TrialRecord, ...]
    available_labels: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class LoadedArrayDataset:
    """Feature array, labels, and aligned trial metadata."""

    dataset_key: str
    features: Any
    labels: Any
    trial_records: tuple[TrialRecord, ...]


class EEGDatasetLoader(Protocol):
    """Protocol implemented by benchmark dataset loaders."""

    dataset_key: str

    def load_manifest(self) -> DatasetManifest:
        """Return canonical trial metadata for split generation."""
