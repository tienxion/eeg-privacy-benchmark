"""MOABB-backed dataset loading for benchmark version 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

from eeg_privacy_benchmark.datasets.base import (
    DatasetManifest,
    LoadedArrayDataset,
    TrialRecord,
)
from eeg_privacy_benchmark.datasets.registry import DATASET_REGISTRY, DatasetSpec


def _import_moabb_datasets() -> Any:
    try:
        from moabb import datasets as moabb_datasets
    except ImportError as exc:
        raise ImportError(
            "MOABB is required to load EEG datasets. Install project dependencies "
            "before running benchmark data ingestion."
        ) from exc
    return moabb_datasets


def _import_left_right_paradigm() -> Any:
    try:
        from moabb.paradigms.motor_imagery import LeftRightImagery
    except ImportError as exc:
        raise ImportError(
            "MOABB motor imagery paradigms are required to build benchmark manifests."
        ) from exc
    return LeftRightImagery


def _build_trial_records(dataset_key: str, labels: Any, metadata: Any) -> tuple[TrialRecord, ...]:
    records = []
    for row_index, (label, metadata_row) in enumerate(zip(labels, metadata.itertuples())):
        subject_id = f"sub-{metadata_row.subject}"
        session_id = str(metadata_row.session) if metadata_row.session is not None else None
        run_id = str(metadata_row.run) if metadata_row.run is not None else None
        trial_id = (
            f"{dataset_key}:{subject_id}:"
            f"ses-{session_id or 'na'}:run-{run_id or 'na'}:trial-{row_index}"
        )
        records.append(
            TrialRecord(
                dataset=dataset_key,
                subject_id=subject_id,
                session_id=session_id,
                run_id=run_id,
                trial_id=trial_id,
                raw_label=str(label),
                canonical_label=str(label),
            )
        )
    return tuple(records)


@dataclass(frozen=True)
class MOABBDatasetLoader:
    """Thin wrapper around MOABB dataset classes.

    This class currently focuses on locating the correct MOABB dataset class and
    establishing a stable dataset identity. Trial extraction is intentionally left
    as a separate step because version-1 development should freeze metadata
    assumptions before pulling large raw datasets.
    """

    dataset_key: str
    resample_hz: float | None = None

    @property
    def dataset_spec(self) -> DatasetSpec:
        try:
            return DATASET_REGISTRY[self.dataset_key]
        except KeyError as exc:
            raise KeyError(f"Unknown benchmark dataset: {self.dataset_key}") from exc

    def instantiate_dataset(self) -> Any:
        """Instantiate the underlying MOABB dataset class."""

        moabb_datasets = _import_moabb_datasets()
        dataset_class = getattr(moabb_datasets, self.dataset_spec.source_id, None)
        if dataset_class is None:
            raise AttributeError(
                f"MOABB dataset class {self.dataset_spec.source_id} is not available."
            )
        return dataset_class()

    def load_manifest(self, subjects: list[int] | None = None) -> DatasetManifest:
        """Extract a canonical manifest using MOABB's left-vs-right imagery paradigm."""

        loaded = self.load_array_data(subjects=subjects)
        available_labels = tuple(sorted({str(label) for label in loaded.labels}))
        return DatasetManifest(
            dataset_key=self.dataset_key,
            trial_records=loaded.trial_records,
            available_labels=available_labels,
            notes=self.dataset_spec.notes,
        )

    def load_array_data(self, subjects: list[int] | None = None) -> LoadedArrayDataset:
        """Load filtered left-vs-right trial arrays, labels, and aligned metadata."""

        dataset = self.instantiate_dataset()
        LeftRightImagery = _import_left_right_paradigm()
        paradigm = LeftRightImagery(resample=self.resample_hz)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"warnEpochs .*",
                category=UserWarning,
            )
            features, labels, metadata = paradigm.get_data(
                dataset=dataset,
                subjects=subjects,
            )
        return LoadedArrayDataset(
            dataset_key=self.dataset_key,
            features=features,
            labels=labels,
            trial_records=_build_trial_records(self.dataset_key, labels, metadata),
        )
