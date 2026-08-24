from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from eeg_privacy_benchmark.datasets.cache_status import inspect_dataset_cache


def _touch(root: Path, relative_paths: list[str]) -> None:
    for relative_path in relative_paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def main() -> None:
    checks: list[tuple[str, bool]] = []
    with TemporaryDirectory() as directory:
        root = Path(directory)
        physionet_root = root / "MNE-eegbci-data/files/eegmmidb/1.0.0"
        imagined_runs = (4, 6, 8, 10, 12, 14)
        _touch(
            physionet_root,
            [
                f"S{subject:03d}/S{subject:03d}R{run:02d}.edf"
                for subject in (1, 2)
                for run in imagined_runs
            ],
        )
        status = inspect_dataset_cache(
            dataset_key="physionet_motor_imagery",
            cache_dir=root,
            subjects=[1, 2],
        )
        checks.append(("physionet_imagery_runs_complete", status.complete))
        checks.append(("physionet_expected_file_count", status.expected_files == 12))

        (physionet_root / "S002/S002R14.edf").unlink()
        status = inspect_dataset_cache(
            dataset_key="physionet_motor_imagery",
            cache_dir=root,
            subjects=[1, 2],
        )
        checks.append(
            (
                "physionet_missing_file_detected",
                status.present_files == 11 and status.missing_files == 1,
            )
        )

        _touch(
            root / "MNE-bnci-data/database/data-sets/001-2014",
            ["A01T.mat", "A01E.mat"],
        )
        status = inspect_dataset_cache(
            dataset_key="bnci2014_001",
            cache_dir=root,
            subjects=[1],
        )
        checks.append(("bnci_cache_contract", status.complete))

        lee_root = (
            root
            / "MNE-lee2019-mi-data/gigadb-datasets/live/pub/"
            "10.5524/100001_101000/100542"
        )
        _touch(
            lee_root,
            [
                "session1/s1/sess01_subj01_EEG_MI.mat",
                "session2/s1/sess02_subj01_EEG_MI.mat",
            ],
        )
        status = inspect_dataset_cache(
            dataset_key="lee2019_mi",
            cache_dir=root,
            subjects=[1],
        )
        checks.append(("lee_cache_contract", status.complete))

    for name, passed in checks:
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    passed_count = sum(passed for _, passed in checks)
    print(f"passed={passed_count}/{len(checks)}")
    if passed_count != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
