from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import tempfile

from eeg_privacy_benchmark.datasets.cache_status import (
    expected_dataset_relative_paths,
    inspect_dataset_cache,
)
from eeg_privacy_benchmark.datasets.factory import build_dataset_loader


SESSIONS = ("0imagery", "2imagery", "4imagery")


def _build_synthetic_inventory(cache_parent: Path) -> None:
    cache_root = cache_parent / "NEMAR-nm000267-v1.0.3"
    for relative in expected_dataset_relative_paths("shin2017a", [1, 2]):
        path = cache_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for subject in (1, 2):
        for session in SESSIONS:
            events = (
                cache_root
                / f"sub-{subject}"
                / f"ses-{session}"
                / "eeg"
                / f"sub-{subject}_ses-{session}_task-imagery_run-0_events.tsv"
            )
            rows = ["sample\ttrial_type"]
            rows.extend(f"{1000 + index * 700}\tleft_hand" for index in range(10))
            rows.extend(f"{8000 + index * 700}\tright_hand" for index in range(10))
            events.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="shin2017a-loader-check-") as temporary:
        cache_parent = Path(temporary)
        _build_synthetic_inventory(cache_parent)
        os.environ["MNE_DATA"] = str(cache_parent)

        status = inspect_dataset_cache(
            dataset_key="shin2017a",
            cache_dir=cache_parent,
            subjects=[1, 2],
        )
        loader = build_dataset_loader(
            "shin2017a",
            frequency_band_hz=(8.0, 32.0),
            epoch_seconds=(0.0, 3.0),
        )
        manifest = loader.load_manifest(subjects=[1, 2])
        label_counts = Counter(record.canonical_label for record in manifest.trial_records)
        session_counts = Counter(record.session_id for record in manifest.trial_records)
        subject_counts = Counter(record.subject_id for record in manifest.trial_records)
        checks.extend(
            [
                ("synthetic_cache_complete", status.complete),
                ("manifest_trials", len(manifest.trial_records) == 120),
                ("labels_balanced", label_counts == {"left_hand": 60, "right_hand": 60}),
                (
                    "sessions_balanced",
                    session_counts == {session: 40 for session in SESSIONS},
                ),
                ("subjects_balanced", subject_counts == {"sub-1": 60, "sub-2": 60}),
                (
                    "trial_ids_unique",
                    len({record.trial_id for record in manifest.trial_records}) == 120,
                ),
            ]
        )

    for name, passed in checks:
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    passed_count = sum(passed for _, passed in checks)
    if passed_count != len(checks):
        raise RuntimeError("Shin2017A NEMAR loader check failed")
    print(f"passed={passed_count}/{len(checks)} training=0 downloads=0")


if __name__ == "__main__":
    main()
