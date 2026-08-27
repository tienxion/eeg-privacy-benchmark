"""Dataset registry for released benchmarks and bounded extension pilots."""

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

CHO2017 = DatasetSpec(
    key="cho2017",
    display_name="Cho2017 Motor Imagery",
    source_library="moabb",
    source_id="Cho2017",
    sessions=1,
    notes=(
        "Version 1.1 integration pilot only. Restrict trials to left-hand and "
        "right-hand imagery and do not use pilot metrics to change benchmark claims."
    ),
)

SHIN2017A = DatasetSpec(
    key="shin2017a",
    display_name="Shin2017A EEG Motor Imagery",
    source_library="moabb",
    source_id="Shin2017A",
    sessions=3,
    notes=(
        "Version 1.2 temporal-generalization study using the version-pinned "
        "NEMAR BIDS record. Acquisition requires explicit review of the upstream "
        "license and provenance before download."
    ),
)

DATASET_REGISTRY = {
    spec.key: spec
    for spec in (
        PHYSIONET_MOTOR_IMAGERY,
        BNCI2014_001,
        LEE2019_MI,
        CHO2017,
        SHIN2017A,
    )
}
