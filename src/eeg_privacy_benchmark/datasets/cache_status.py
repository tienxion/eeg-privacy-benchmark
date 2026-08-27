"""Local dataset cache inspection helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil

from eeg_privacy_benchmark.datasets.factory import build_dataset_loader


@dataclass(frozen=True)
class DatasetCacheStatus:
    dataset_key: str
    cache_root: str
    subjects: list[int]
    expected_files: int
    present_files: int
    missing_files: int
    complete: bool
    sample_missing_paths: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetCacheImportResult:
    dataset_key: str
    source_cache_root: str
    destination_cache_root: str
    subjects: list[int]
    expected_files: int
    copied_files: int
    skipped_existing_files: int
    missing_source_files: int
    sample_missing_source_paths: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _resolve_subjects(dataset_key: str, subjects: list[int] | None) -> list[int]:
    if subjects:
        return sorted(subjects)
    loader = build_dataset_loader(dataset_key)
    dataset = loader.instantiate_dataset()
    return sorted(int(subject) for subject in dataset.subject_list)


def _cache_root(cache_dir: str | Path, dataset_key: str) -> Path:
    root = Path(cache_dir).resolve()
    if dataset_key == "bnci2014_001":
        return root / "MNE-bnci-data"
    if dataset_key == "physionet_motor_imagery":
        return root / "MNE-eegbci-data"
    if dataset_key == "lee2019_mi":
        return root / "MNE-lee2019-mi-data"
    if dataset_key == "cho2017":
        return root / "MNE-gigadb-data"
    raise ValueError(f"Unsupported dataset cache inspection target: {dataset_key}")


def _expected_relative_paths(dataset_key: str, subjects: list[int]) -> list[Path]:
    relative_paths: list[Path] = []

    if dataset_key == "bnci2014_001":
        for subject in subjects:
            for split in ("T", "E"):
                relative_paths.append(
                    Path("database/data-sets/001-2014") / f"A{subject:02d}{split}.mat"
                )
        return relative_paths

    if dataset_key == "physionet_motor_imagery":
        # PhysionetMI defaults to imagined, non-executed trials; baseline runs
        # 1 and 2 are not read by the benchmark's LeftRightImagery paradigm.
        runs = (4, 6, 8, 10, 12, 14)
        for subject in subjects:
            subject_dir = Path("files/eegmmidb/1.0.0") / f"S{subject:03d}"
            for run in runs:
                relative_paths.append(subject_dir / f"S{subject:03d}R{run:02d}.edf")
        return relative_paths

    if dataset_key == "lee2019_mi":
        for subject in subjects:
            for session in (1, 2):
                relative_paths.append(
                    Path("gigadb-datasets/live/pub/10.5524/100001_101000/100542")
                    / f"session{session}"
                    / f"s{subject}"
                    / f"sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
                )
        return relative_paths

    if dataset_key == "cho2017":
        for subject in subjects:
            relative_paths.append(
                Path("gigadb-datasets/live/pub/10.5524/100001_101000/100295/mat_data")
                / f"s{subject:02d}.mat"
            )
        return relative_paths

    raise ValueError(f"Unsupported dataset cache inspection target: {dataset_key}")


def _expected_paths_for_subjects(
    *,
    dataset_key: str,
    cache_dir: str | Path,
    subjects: list[int] | None,
) -> tuple[list[int], Path, list[Path]]:
    resolved_subjects = _resolve_subjects(dataset_key, subjects)
    dataset_cache_root = _cache_root(cache_dir, dataset_key)
    expected_relative_paths = _expected_relative_paths(dataset_key, resolved_subjects)
    expected_paths = [
        dataset_cache_root / relative_path for relative_path in expected_relative_paths
    ]
    return resolved_subjects, dataset_cache_root, expected_paths


def inspect_dataset_cache(
    *,
    dataset_key: str,
    cache_dir: str | Path,
    subjects: list[int] | None = None,
    sample_limit: int = 5,
) -> DatasetCacheStatus:
    resolved_subjects, dataset_cache_root, expected_paths = _expected_paths_for_subjects(
        dataset_key=dataset_key,
        cache_dir=cache_dir,
        subjects=subjects,
    )
    missing_paths = [path for path in expected_paths if not path.is_file()]
    present_files = len(expected_paths) - len(missing_paths)

    return DatasetCacheStatus(
        dataset_key=dataset_key,
        cache_root=str(dataset_cache_root),
        subjects=resolved_subjects,
        expected_files=len(expected_paths),
        present_files=present_files,
        missing_files=len(missing_paths),
        complete=not missing_paths,
        sample_missing_paths=[str(path) for path in missing_paths[:sample_limit]],
    )


def import_dataset_cache(
    *,
    dataset_key: str,
    source_cache_dir: str | Path,
    destination_cache_dir: str | Path,
    subjects: list[int] | None = None,
    sample_limit: int = 5,
) -> DatasetCacheImportResult:
    resolved_subjects, source_root, source_paths = _expected_paths_for_subjects(
        dataset_key=dataset_key,
        cache_dir=source_cache_dir,
        subjects=subjects,
    )
    _, destination_root, destination_paths = _expected_paths_for_subjects(
        dataset_key=dataset_key,
        cache_dir=destination_cache_dir,
        subjects=subjects,
    )

    copied_files = 0
    skipped_existing_files = 0
    missing_source_paths: list[Path] = []
    for source_path, destination_path in zip(source_paths, destination_paths):
        if not source_path.is_file():
            missing_source_paths.append(source_path)
            continue
        if destination_path.is_file():
            skipped_existing_files += 1
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        copied_files += 1

    return DatasetCacheImportResult(
        dataset_key=dataset_key,
        source_cache_root=str(source_root),
        destination_cache_root=str(destination_root),
        subjects=resolved_subjects,
        expected_files=len(source_paths),
        copied_files=copied_files,
        skipped_existing_files=skipped_existing_files,
        missing_source_files=len(missing_source_paths),
        sample_missing_source_paths=[
            str(path) for path in missing_source_paths[:sample_limit]
        ],
    )
