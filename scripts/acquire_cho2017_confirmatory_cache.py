from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cho2017_confirmatory_validation_v1.yaml"
AUDIT_SCRIPT = ROOT / "scripts" / "audit_cho2017_cached_subjects.py"
RELATIVE_CACHE_ROOT = Path(
    "MNE-gigadb-data/gigadb-datasets/live/pub/10.5524/"
    "100001_101000/100295/mat_data"
)
GIB = 1024**3


def _parse_subjects(value: str) -> list[int]:
    try:
        subjects = sorted(
            {int(item.strip()) for item in value.split(",") if item.strip()}
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError("subjects must be comma-separated integers") from exc
    if not subjects:
        raise argparse.ArgumentTypeError("at least one subject is required")
    return subjects


def _load_confirmatory_subjects() -> list[int]:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return [int(subject) for subject in payload["cohort"]["confirmatory_subjects"]]


def _estimate_missing_bytes(cache_root: Path, missing_count: int) -> int | None:
    cached_sizes = [
        path.stat().st_size
        for path in sorted(cache_root.glob("s*.mat"))
        if path.is_file() and path.stat().st_size > 0
    ]
    if not cached_sizes:
        return None
    return int(sum(cached_sizes) / len(cached_sizes) * missing_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or sequentially acquire the frozen Cho2017 confirmatory "
            "cache, then run metadata/integrity QA. No model training is performed."
        )
    )
    parser.add_argument("--mne-data-dir", type=Path, required=True)
    parser.add_argument(
        "--subjects",
        type=_parse_subjects,
        help="Optional confirmatory subset. Defaults to frozen subjects 3-52.",
    )
    parser.add_argument(
        "--minimum-free-gib-after-download",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--confirm-download",
        action="store_true",
        help="Actually download missing files. Without this flag, only preflight runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    confirmatory = _load_confirmatory_subjects()
    subjects = args.subjects or confirmatory
    outside_contract = sorted(set(subjects) - set(confirmatory))
    if outside_contract:
        raise ValueError(
            f"subjects outside the frozen confirmatory cohort: {outside_contract}"
        )
    if args.minimum_free_gib_after_download < 0:
        raise ValueError("minimum free space must be non-negative")

    data_dir = args.mne_data_dir.resolve()
    cache_root = data_dir / RELATIVE_CACHE_ROOT
    existing = [
        subject for subject in subjects if (cache_root / f"s{subject:02d}.mat").is_file()
    ]
    missing = [subject for subject in subjects if subject not in existing]
    estimated_bytes = _estimate_missing_bytes(cache_root, len(missing))
    free_bytes = shutil.disk_usage(data_dir).free
    estimate_text = (
        f"{estimated_bytes / GIB:.2f} GiB"
        if estimated_bytes is not None
        else "unknown (no cached size sample)"
    )
    print(
        f"Cho2017 preflight subjects={len(subjects)} existing={len(existing)} "
        f"missing={len(missing)} estimated_download={estimate_text} "
        f"free={free_bytes / GIB:.2f} GiB"
    )

    reserve_bytes = int(args.minimum_free_gib_after_download * GIB)
    if estimated_bytes is not None and free_bytes - estimated_bytes < reserve_bytes:
        raise RuntimeError(
            "insufficient free space for the estimated download plus the required reserve"
        )

    if missing and not args.confirm_download:
        print("Preflight only; pass --confirm-download to acquire the missing files.")
        return

    if missing:
        from moabb.datasets import Cho2017

        dataset = Cho2017()
        for position, subject in enumerate(missing, start=1):
            print(f"[{position}/{len(missing)}] acquiring Cho2017 subject {subject}")
            resolved = Path(
                dataset.data_path(
                    subject,
                    path=str(data_dir),
                    force_update=False,
                    verbose=True,
                )
            )
            if not resolved.is_file() or resolved.stat().st_size == 0:
                raise RuntimeError(f"subject {subject} did not produce a valid cache file")

    subprocess.run(
        [
            sys.executable,
            str(AUDIT_SCRIPT),
            "--mne-data-dir",
            str(data_dir),
            "--subjects",
            ",".join(str(subject) for subject in subjects),
        ],
        check=True,
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
