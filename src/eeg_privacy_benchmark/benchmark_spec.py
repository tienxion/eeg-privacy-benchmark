"""Frozen version-1 benchmark decisions.

This module intentionally keeps the project scope narrow so that early work
stays aligned with the charter and does not sprawl into federation or
cryptographic engineering before the baseline is stable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSpec:
    datasets: tuple[str, ...]
    task: str
    baselines: tuple[str, ...]
    first_attack: str
    deferred_work: tuple[str, ...]


V1_SPEC = BenchmarkSpec(
    datasets=(
        "physionet_motor_imagery",
        "bnci2014_001",
        "lee2019_mi",
    ),
    task="left_vs_right_motor_imagery",
    baselines=("csp_lda", "eegnet"),
    first_attack="subject_identification_linear_probe",
    deferred_work=(
        "membership_inference",
        "attribute_inference",
        "federated_learning",
        "secure_aggregation",
        "encrypted_inference",
    ),
)
