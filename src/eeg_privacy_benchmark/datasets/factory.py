"""Factory helpers for benchmark dataset loaders."""

from __future__ import annotations

from eeg_privacy_benchmark.datasets.base import EEGDatasetLoader
from eeg_privacy_benchmark.datasets.moabb_loader import MOABBDatasetLoader
from eeg_privacy_benchmark.datasets.nemar_shin_loader import NEMARShin2017ALoader
from eeg_privacy_benchmark.datasets.registry import DATASET_REGISTRY


def build_dataset_loader(
    dataset_key: str,
    *,
    resample_hz: float | None = None,
    frequency_band_hz: tuple[float, float] | None = None,
    epoch_seconds: tuple[float, float] | None = None,
) -> EEGDatasetLoader:
    """Construct the benchmark loader for a dataset key."""

    spec = DATASET_REGISTRY.get(dataset_key)
    if spec is None:
        raise KeyError(f"Unknown benchmark dataset: {dataset_key}")
    if spec.source_library != "moabb":
        raise ValueError(
            f"Unsupported source library {spec.source_library} for {dataset_key}."
        )
    if dataset_key == "shin2017a":
        return NEMARShin2017ALoader(
            dataset_key=dataset_key,
            resample_hz=resample_hz,
            frequency_band_hz=frequency_band_hz,
            epoch_seconds=epoch_seconds,
        )
    return MOABBDatasetLoader(
        dataset_key=dataset_key,
        resample_hz=resample_hz,
        frequency_band_hz=frequency_band_hz,
        epoch_seconds=epoch_seconds,
    )
