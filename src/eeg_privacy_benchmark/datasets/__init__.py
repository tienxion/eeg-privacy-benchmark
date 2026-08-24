"""Dataset loaders and split definitions for EEG benchmark experiments."""

from eeg_privacy_benchmark.datasets.factory import build_dataset_loader
from eeg_privacy_benchmark.datasets.registry import DATASET_REGISTRY

__all__ = ["DATASET_REGISTRY", "build_dataset_loader"]
