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
    resample_hz: float | None = None
    frequency_band_hz: tuple[float, float] | None = None
    epoch_seconds: tuple[float, float] | None = None


def validate_loaded_array_dataset(
    loaded: LoadedArrayDataset,
    *,
    dataset_key: str,
    subjects: list[int] | None,
    resample_hz: float | None,
    frequency_band_hz: tuple[float, float] | None,
    epoch_seconds: tuple[float, float] | None,
) -> None:
    """Fail closed when preloaded arrays do not match their requested provenance."""

    if loaded.dataset_key != dataset_key:
        raise ValueError("loaded dataset key must match dataset_key")
    if len(loaded.features) != len(loaded.labels) or len(loaded.labels) != len(
        loaded.trial_records
    ):
        raise ValueError("loaded features, labels, and trial records must align")
    if subjects is not None:
        observed_subjects = {
            int(record.subject_id.removeprefix("sub-"))
            for record in loaded.trial_records
        }
        if observed_subjects != set(subjects):
            raise ValueError("loaded dataset subjects must match requested subjects")
    provenance = {
        "resample_hz": (loaded.resample_hz, resample_hz),
        "frequency_band_hz": (loaded.frequency_band_hz, frequency_band_hz),
        "epoch_seconds": (loaded.epoch_seconds, epoch_seconds),
    }
    for name, (observed, expected) in provenance.items():
        if observed != expected:
            raise ValueError(f"loaded dataset {name} must match requested preprocessing")


class EEGDatasetLoader(Protocol):
    """Protocol implemented by benchmark dataset loaders."""

    dataset_key: str

    def load_manifest(self) -> DatasetManifest:
        """Return canonical trial metadata for split generation."""
