from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "v1_1_cho2017_qa"
    / "cho2017_cache_qa.json"
)
DEFAULT_OUTPUT = ROOT / "configs" / "cho2017_confirmatory_cache_manifest_v1.json"
EXPECTED_SUBJECTS = list(range(3, 53))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bad_entry_count(row: dict[str, Any]) -> int:
    return sum(
        len(indices)
        for source in row["bad_trial_indices_raw_matlab"].values()
        for indices in source.values()
    )


def _build_manifest(source_path: Path) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    rows = source["rows"]
    subjects = [int(row["subject"]) for row in rows]

    _require(source["status"] == "PASS", "source QA status is not PASS")
    _require(source["network_or_training"] is False, "source QA launched network or training")
    _require(source["subjects"] == EXPECTED_SUBJECTS, "source QA subject list is not 3-52")
    _require(source["subject_count"] == 50, "source QA subject count is not 50")
    _require(subjects == EXPECTED_SUBJECTS, "source QA rows are not ordered subjects 3-52")

    for row in rows:
        subject = int(row["subject"])
        _require(row["file"] == f"s{subject:02d}.mat", f"subject {subject}: file mismatch")
        _require(len(row["sha256"]) == 64, f"subject {subject}: invalid SHA-256")
        _require(row["sampling_hz"] == 512, f"subject {subject}: sampling-rate mismatch")
        _require(row["source_channels"] == 68, f"subject {subject}: source-channel mismatch")
        _require(row["eeg_channels"] == 64, f"subject {subject}: EEG-channel mismatch")
        _require(row["trials_per_class"] in {100, 120}, f"subject {subject}: trial-count mismatch")
        _require(row["event_onsets"] == row["trials_per_class"], f"subject {subject}: event mismatch")
        _require(row["finite_eeg"] is True, f"subject {subject}: nonfinite EEG")
        _require(row["nonzero_channel_variance"] is True, f"subject {subject}: zero variance")

    trial_counts = Counter(int(row["trials_per_class"]) for row in rows)
    source_shapes = sorted(
        {
            (
                int(row["source_channels"]),
                int(row["samples_per_class_recording"]),
                int(row["trials_per_class"]),
            )
            for row in rows
        }
    )
    subjects_with_raw_bad_indices = [
        int(row["subject"]) for row in rows if _bad_entry_count(row) > 0
    ]

    manifest_rows = []
    for row in rows:
        manifest_rows.append(
            {
                "subject": int(row["subject"]),
                "file": row["file"],
                "bytes": int(row["bytes"]),
                "sha256": row["sha256"],
                "trials_per_class": int(row["trials_per_class"]),
                "event_onsets": int(row["event_onsets"]),
                "samples_per_class_recording": int(row["samples_per_class_recording"]),
                "raw_bad_trial_index_entry_count": _bad_entry_count(row),
                "bad_trial_indices_raw_matlab": row["bad_trial_indices_raw_matlab"],
            }
        )

    return {
        "schema_version": 1,
        "dataset": "Cho2017",
        "cohort": "confirmatory_subjects_3_52",
        "qa_status": "PASS",
        "objective_subject_exclusions": [],
        "primary_analysis_trial_policy": "moabb_all_trials",
        "raw_bad_trial_index_mapping": "unmapped_upstream_metadata_only",
        "network_or_training": False,
        "summary": {
            "subject_count": len(rows),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "trials_per_class_counts": {
                str(trial_count): count
                for trial_count, count in sorted(trial_counts.items())
            },
            "source_shape_variants": [
                {
                    "source_channels": channels,
                    "samples_per_class_recording": samples,
                    "trials_per_class": trials,
                }
                for channels, samples, trials in source_shapes
            ],
            "identical_source_shape": source["identical_source_shape"],
            "identical_source_shape_explanation": (
                "Expected source-level variation: 47 subjects have 100 trials per class "
                "and 3 have 120. The frozen post-preprocessing representation remains fixed."
            ),
            "raw_bad_trial_index_entry_count": sum(_bad_entry_count(row) for row in rows),
            "subjects_with_raw_bad_trial_indices": subjects_with_raw_bad_indices,
            "finite_eeg_all_subjects": all(row["finite_eeg"] for row in rows),
            "nonzero_channel_variance_all_subjects": all(
                row["nonzero_channel_variance"] for row in rows
            ),
        },
        "provenance": {
            "source_qa_sha256": _sha256(source_path),
            "audit_command": (
                "PYTHONPATH=src .venv/bin/python scripts/audit_cho2017_cached_subjects.py "
                "--mne-data-dir raw_data/mne_data --subjects 3,4,...,52"
            ),
            "freeze_command": (
                "PYTHONPATH=src .venv/bin/python scripts/freeze_cho2017_cache_manifest.py"
            ),
        },
        "rows": manifest_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze or verify the tracked Cho2017 confirmatory cache QA manifest."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the tracked manifest with the current audited cache instead of writing it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.input.resolve()
    output_path = args.output.resolve()
    manifest = _build_manifest(source_path)
    rendered = json.dumps(manifest, indent=2) + "\n"

    if args.check:
        _require(output_path.is_file(), f"tracked manifest missing: {output_path}")
        _require(
            output_path.read_text(encoding="utf-8") == rendered,
            "tracked manifest does not match the audited local cache",
        )
        print(
            "PASS Cho2017 tracked cache manifest "
            f"subjects={manifest['summary']['subject_count']} "
            f"files={manifest['summary']['file_count']}"
        )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(
        f"Wrote {output_path} with {manifest['summary']['subject_count']} "
        "checksum-backed subjects"
    )


if __name__ == "__main__":
    main()
