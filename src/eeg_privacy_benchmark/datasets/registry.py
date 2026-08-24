"""Frozen dataset registry for benchmark version 1."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    source_library: str
    source_id: str
    sessions: int | None
    notes: str


PHYSIONET_MOTOR_IMAGERY = DatasetSpec(
    key="physionet_motor_imagery",
    display_name="PhysioNet EEG Motor Movement/Imagery Dataset",
    source_library="moabb",
    source_id="PhysionetMI",
    sessions=None,
    notes=(
        "Use this dataset for scale. Restrict trials to left-hand and "
        "right-hand imagery conditions in version 1."
    ),
)

BNCI2014_001 = DatasetSpec(
    key="bnci2014_001",
    display_name="BNCI2014-001",
    source_library="moabb",
    source_id="BNCI2014_001",
    sessions=2,
    notes=(
        "Standard motor-imagery benchmark with session structure. Use for "
        "cross-subject and cross-session evaluation."
    ),
)

LEE2019_MI = DatasetSpec(
    key="lee2019_mi",
    display_name="Lee2019_MI",
    source_library="moabb",
    source_id="Lee2019_MI",
    sessions=2,
    notes=(
        "Large two-session motor-imagery dataset that helps measure temporal "
        "stability and privacy leakage across sessions."
    ),
)

DATASET_REGISTRY = {
    spec.key: spec
    for spec in (
        PHYSIONET_MOTOR_IMAGERY,
        BNCI2014_001,
        LEE2019_MI,
    )
}
