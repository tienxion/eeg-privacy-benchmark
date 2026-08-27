from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

from eeg_privacy_benchmark.datasets.cache_status import (
    expected_dataset_relative_paths,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "nm000267"
VERSION = "v1.0.3"
MANIFEST_URL = f"https://data.nemar.org/{DATASET_ID}/{VERSION}/manifest.json"
DATASET_RECORD_URL = "https://nemar.org/dataset/nm000267"
DEFAULT_CACHE_ROOT = (
    ROOT / "raw_data" / "mne_data" / "NEMAR-nm000267-v1.0.3"
)
PILOT_SUBJECTS = [1, 2]
CONFIRMATORY_SUBJECTS = list(range(3, 30))


def _selected_file_manifest_sha256(
    rows: list[dict[str, object]],
) -> str:
    stable_rows = [
        {
            "path": str(row["path"]),
            "bytes": int(row["size"]),
            "checksum_algorithm": str(row["checksum_algorithm"]),
            "checksum": str(row["checksum"]),
        }
        for row in rows
    ]
    canonical = json.dumps(
        sorted(stable_rows, key=lambda row: row["path"]),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _fetch_manifest() -> tuple[list[dict[str, object]], bytes]:
    request = urllib.request.Request(
        MANIFEST_URL,
        headers={"User-Agent": "eeg-privacy-benchmark-v1.2-acquisition"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    rows = json.loads(payload)
    if not isinstance(rows, list):
        raise RuntimeError("NEMAR manifest is not a list")
    return rows, payload


def _checksum(path: Path, algorithm: str) -> str:
    if algorithm == "sha256":
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if algorithm == "git":
        content = path.read_bytes()
        header = f"blob {len(content)}\0".encode("ascii")
        return hashlib.sha1(header + content).hexdigest()
    raise ValueError(f"Unsupported NEMAR checksum algorithm: {algorithm}")


def _valid_existing(path: Path, row: dict[str, object]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(row["size"])
        and _checksum(path, str(row["checksum_algorithm"]))
        == str(row["checksum"])
    )


def _download(row: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(
        str(row["bytes_url"]),
        headers={"User-Agent": "eeg-privacy-benchmark-v1.2-acquisition"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    if not _valid_existing(partial, row):
        raise RuntimeError(f"Downloaded file failed checksum: {row['path']}")
    os.replace(partial, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or acquire a bounded NEMAR v1.0.3 Shin2017A imagery-only "
            "cache. Model training is never performed. Review the upstream "
            "dataset record and terms before downloading."
        )
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--confirm-download",
        action="store_true",
        help="Perform the explicitly confirmed bounded download.",
    )
    parser.add_argument(
        "--accept-dataset-terms",
        action="store_true",
        help=(
            "Record that the current user reviewed and accepts the upstream "
            "NEMAR dataset terms for this download."
        ),
    )
    parser.add_argument(
        "--confirmatory",
        action="store_true",
        help="Select the frozen subjects 3-29 confirmatory cohort.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.confirm_download and not args.accept_dataset_terms:
        raise RuntimeError(
            "--confirm-download requires --accept-dataset-terms after reviewing "
            f"{DATASET_RECORD_URL}"
        )
    cache_root = args.cache_root.resolve()
    subjects = CONFIRMATORY_SUBJECTS if args.confirmatory else PILOT_SUBJECTS
    stage = "confirmatory" if args.confirmatory else "pilot"
    rows, manifest_bytes = _fetch_manifest()
    by_path = {str(row["path"]): row for row in rows}
    expected = expected_dataset_relative_paths("shin2017a", subjects)
    expected_strings = [path.as_posix() for path in expected]
    missing_manifest_rows = [path for path in expected_strings if path not in by_path]
    if missing_manifest_rows:
        raise RuntimeError(
            "NEMAR manifest lacks frozen selected files: "
            + ", ".join(missing_manifest_rows[:5])
        )
    selected = [by_path[path] for path in expected_strings]
    selected_file_manifest_sha256 = _selected_file_manifest_sha256(selected)
    required_bytes = sum(int(row["size"]) for row in selected)
    free_bytes = shutil.disk_usage(cache_root.parent).free
    existing = sum(
        _valid_existing(cache_root / str(row["path"]), row) for row in selected
    )
    print(
        f"Shin2017A NEMAR {stage} provider={DATASET_ID} version={VERSION} "
        f"subjects={subjects} files={len(selected)} "
        f"bytes={required_bytes} existing_valid={existing}"
    )
    print(f"cache_root={cache_root}")
    print(f"free_bytes={free_bytes}")
    print(f"selected_file_manifest_sha256={selected_file_manifest_sha256}")
    print(f"dataset_record={DATASET_RECORD_URL}")
    print(f"terms_accepted_by_current_user={args.accept_dataset_terms}")
    if free_bytes < required_bytes * 3:
        raise RuntimeError("Insufficient free space for bounded acquisition")
    if not args.confirm_download:
        print(
            "DRY RUN: no files downloaded; after reviewing the upstream record, "
            "add --confirm-download --accept-dataset-terms to execute"
        )
        return

    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_snapshot = cache_root / ".nemar_manifest_v1.0.3.json"
    manifest_snapshot.write_bytes(manifest_bytes)
    downloaded = 0
    skipped = 0
    for index, row in enumerate(selected, start=1):
        destination = cache_root / str(row["path"])
        if _valid_existing(destination, row):
            skipped += 1
            print(f"[{index}/{len(selected)}] valid existing {row['path']}")
            continue
        if destination.exists():
            raise RuntimeError(
                f"Existing cache file has wrong size or checksum: {destination}"
            )
        print(f"[{index}/{len(selected)}] downloading {row['path']}")
        _download(row, destination)
        downloaded += 1

    invalid = [
        str(row["path"])
        for row in selected
        if not _valid_existing(cache_root / str(row["path"]), row)
    ]
    if invalid:
        raise RuntimeError(f"Acquisition verification failed: {invalid[:5]}")
    source_manifest_snapshot_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    print(
        f"PASS Shin2017A NEMAR {stage} files={len(selected)} "
        f"downloaded={downloaded} skipped={skipped} "
        f"source_manifest_snapshot_sha256={source_manifest_snapshot_sha256} "
        f"selected_file_manifest_sha256={selected_file_manifest_sha256}"
    )
    print("No model training was performed.")


if __name__ == "__main__":
    main()
