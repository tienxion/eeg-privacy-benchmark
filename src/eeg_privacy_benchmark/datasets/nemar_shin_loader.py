"""Version-pinned NEMAR BIDS loader for the Shin2017A v1.2 study."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np

from eeg_privacy_benchmark.datasets.base import (
    DatasetManifest,
    LoadedArrayDataset,
    TrialRecord,
)
from eeg_privacy_benchmark.datasets.registry import DATASET_REGISTRY, DatasetSpec


NEMAR_CACHE_DIRECTORY = "NEMAR-nm000267-v1.0.3"
IMAGERY_SESSIONS = ("0imagery", "2imagery", "4imagery")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


@dataclass(frozen=True)
class NEMARShin2017ALoader:
    """Load the frozen NEMAR nm000267 v1.0.3 imagery derivative."""

    dataset_key: str = "shin2017a"
    resample_hz: float | None = None
    frequency_band_hz: tuple[float, float] | None = None
    epoch_seconds: tuple[float, float] | None = None

    @property
    def dataset_spec(self) -> DatasetSpec:
        return DATASET_REGISTRY[self.dataset_key]

    @property
    def cache_base(self) -> Path:
        mne_data = Path(os.environ.get("MNE_DATA", "raw_data/mne_data")).resolve()
        return mne_data / NEMAR_CACHE_DIRECTORY

    def instantiate_dataset(self) -> Any:
        """Expose pinned MOABB metadata without invoking its legacy downloader."""

        from moabb.datasets import Shin2017A

        return Shin2017A()

    def _resolve_subjects(self, subjects: list[int] | None) -> list[int]:
        from eeg_privacy_benchmark.datasets.cache_status import inspect_dataset_cache

        resolved = list(range(1, 30)) if subjects is None else sorted(subjects)
        if not resolved or any(subject not in range(1, 30) for subject in resolved):
            raise ValueError("Shin2017A subjects must be between 1 and 29")
        status = inspect_dataset_cache(
            dataset_key=self.dataset_key,
            cache_dir=self.cache_base.parent,
            subjects=resolved,
        )
        if not status.complete:
            raise FileNotFoundError(
                f"Shin2017A NEMAR cache is incomplete: "
                f"{status.missing_files}/{status.expected_files} files missing"
            )
        return resolved

    def _session_paths(
        self,
        subject: int,
        session: str,
    ) -> tuple[Path, Path, Path]:
        eeg_dir = self.cache_base / f"sub-{subject}" / f"ses-{session}" / "eeg"
        prefix = f"sub-{subject}_ses-{session}_task-imagery_run-0"
        return (
            eeg_dir / f"{prefix}_eeg.bdf",
            eeg_dir / f"{prefix}_events.tsv",
            eeg_dir / f"{prefix}_channels.tsv",
        )

    def _trial_records(self, subjects: list[int]) -> tuple[TrialRecord, ...]:
        records = []
        for subject in subjects:
            for session in IMAGERY_SESSIONS:
                _, events_path, _ = self._session_paths(subject, session)
                for row in _read_tsv(events_path):
                    label = row["trial_type"]
                    sample = int(row["sample"])
                    records.append(
                        TrialRecord(
                            dataset=self.dataset_key,
                            subject_id=f"sub-{subject}",
                            session_id=session,
                            run_id="0",
                            trial_id=(
                                f"{self.dataset_key}:sub-{subject}:ses-{session}:"
                                f"run-0:sample-{sample}"
                            ),
                            raw_label=label,
                            canonical_label=label,
                        )
                    )
        return tuple(records)

    def load_manifest(self, subjects: list[int] | None = None) -> DatasetManifest:
        resolved = self._resolve_subjects(subjects)
        records = self._trial_records(resolved)
        return DatasetManifest(
            dataset_key=self.dataset_key,
            trial_records=records,
            available_labels=("left_hand", "right_hand"),
            notes=self.dataset_spec.notes,
        )

    def load_array_data(
        self,
        subjects: list[int] | None = None,
    ) -> LoadedArrayDataset:
        """Extract aligned 0-3 second EEG epochs from the BIDS derivative."""

        import mne

        resolved = self._resolve_subjects(subjects)
        frequency_band = self.frequency_band_hz or (8.0, 32.0)
        epoch_seconds = self.epoch_seconds or (0.0, 3.0)
        features = []
        labels = []
        records = []
        expected_channel_names: list[str] | None = None
        for subject in resolved:
            for session in IMAGERY_SESSIONS:
                bdf_path, events_path, channels_path = self._session_paths(
                    subject, session
                )
                event_rows = _read_tsv(events_path)
                channel_rows = _read_tsv(channels_path)
                eeg_names = [
                    row["name"] for row in channel_rows if row["type"] == "EEG"
                ]
                if len(eeg_names) != 30:
                    raise RuntimeError("Shin2017A NEMAR session lacks 30 EEG channels")
                if expected_channel_names is None:
                    expected_channel_names = eeg_names
                elif eeg_names != expected_channel_names:
                    raise RuntimeError("Shin2017A NEMAR EEG channel order changed")

                raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose="ERROR")
                raw.pick(eeg_names)
                raw.filter(
                    l_freq=frequency_band[0],
                    h_freq=frequency_band[1],
                    verbose="ERROR",
                )
                if self.resample_hz is not None:
                    raw.resample(self.resample_hz, verbose="ERROR")
                sampling_hz = float(raw.info["sfreq"])
                start_offset = int(round(epoch_seconds[0] * sampling_hz))
                epoch_samples = int(
                    round((epoch_seconds[1] - epoch_seconds[0]) * sampling_hz)
                )
                source_sampling_hz = float(channel_rows[0]["sampling_frequency"])
                for row in event_rows:
                    source_sample = int(row["sample"])
                    event_sample = int(
                        round(source_sample * sampling_hz / source_sampling_hz)
                    )
                    start = event_sample + start_offset
                    stop = start + epoch_samples
                    epoch = raw.get_data(start=start, stop=stop)
                    if epoch.shape != (30, epoch_samples):
                        raise RuntimeError("Shin2017A epoch extraction shape mismatch")
                    label = row["trial_type"]
                    features.append(epoch)
                    labels.append(label)
                    records.append(
                        TrialRecord(
                            dataset=self.dataset_key,
                            subject_id=f"sub-{subject}",
                            session_id=session,
                            run_id="0",
                            trial_id=(
                                f"{self.dataset_key}:sub-{subject}:ses-{session}:"
                                f"run-0:sample-{source_sample}"
                            ),
                            raw_label=label,
                            canonical_label=label,
                        )
                    )
        return LoadedArrayDataset(
            dataset_key=self.dataset_key,
            features=np.asarray(features),
            labels=np.asarray(labels),
            trial_records=tuple(records),
            resample_hz=self.resample_hz,
            frequency_band_hz=self.frequency_band_hz,
            epoch_seconds=self.epoch_seconds,
        )
