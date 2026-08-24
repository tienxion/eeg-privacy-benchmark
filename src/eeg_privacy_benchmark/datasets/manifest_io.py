"""Serialization helpers for benchmark dataset manifests and splits."""

from __future__ import annotations

import json
from pathlib import Path

from eeg_privacy_benchmark.datasets.base import DatasetManifest
from eeg_privacy_benchmark.datasets.splits import SplitManifest


def write_dataset_manifest(manifest: DatasetManifest, output_path: str | Path) -> None:
    """Write a dataset manifest to JSON."""

    path = Path(output_path)
    payload = {
        "dataset_key": manifest.dataset_key,
        "available_labels": list(manifest.available_labels),
        "notes": manifest.notes,
        "trial_records": [record.__dict__ for record in manifest.trial_records],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_split_manifest(split_manifest: SplitManifest, output_path: str | Path) -> None:
    """Write split assignments to JSON."""

    path = Path(output_path)
    payload = {
        "dataset_key": split_manifest.dataset_key,
        "protocol": split_manifest.protocol,
        "seed": split_manifest.seed,
        "subject_ids": list(split_manifest.subject_ids),
        "assignments": [assignment.__dict__ for assignment in split_manifest.assignments],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
