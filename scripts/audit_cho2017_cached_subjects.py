from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "v1_1_cho2017_qa" / "cho2017_cache_qa.json"
RELATIVE_CACHE_ROOT = Path(
    "MNE-gigadb-data/gigadb-datasets/live/pub/10.5524/"
    "100001_101000/100295/mat_data"
)


def _parse_subjects(value: str) -> list[int]:
    subjects = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not subjects or any(subject < 1 or subject > 52 for subject in subjects):
        raise argparse.ArgumentTypeError("subjects must be comma-separated integers from 1 to 52")
    return subjects


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_list(value: Any) -> list[int]:
    import numpy as np

    array = np.atleast_1d(value)
    if array.size == 0:
        return []
    return sorted(int(item) for item in array.reshape(-1).tolist())


def _raw_bad_index_cells(container: Any) -> dict[str, dict[str, list[int]]]:
    import numpy as np

    result: dict[str, dict[str, list[int]]] = {}
    for field in ("bad_trial_idx_voltage", "bad_trial_idx_mi"):
        values = np.atleast_1d(getattr(container, field))
        if len(values) != 2:
            raise RuntimeError(f"Unexpected {field} nested-cell structure: {values!r}")
        result[field] = {
            "cell_1_unmapped": _index_list(values[0]),
            "cell_2_unmapped": _index_list(values[1]),
        }
    return result


def _audit_subject(path: Path, subject: int) -> dict[str, Any]:
    import numpy as np
    from scipy.io import loadmat

    eeg = loadmat(
        path,
        squeeze_me=True,
        struct_as_record=False,
        verify_compressed_data_integrity=False,
    )["eeg"]
    required_fields = {
        "bad_trial_indices",
        "imagery_event",
        "imagery_left",
        "imagery_right",
        "n_imagery_trials",
        "srate",
    }
    observed_fields = set(getattr(eeg, "_fieldnames", []))
    missing_fields = sorted(required_fields - observed_fields)
    if missing_fields:
        raise RuntimeError(f"s{subject:02d}.mat missing fields: {missing_fields}")

    left = np.asarray(eeg.imagery_left)
    right = np.asarray(eeg.imagery_right)
    event = np.asarray(eeg.imagery_event)
    n_trials = int(eeg.n_imagery_trials)
    if n_trials not in {100, 120}:
        raise RuntimeError(f"subject {subject}: unexpected trials per class {n_trials}")
    if int(eeg.srate) != 512:
        raise RuntimeError(f"subject {subject}: unexpected sampling rate {eeg.srate}")
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != 68 or right.shape[0] != 68:
        raise RuntimeError(
            f"subject {subject}: unexpected imagery shapes left={left.shape} right={right.shape}"
        )
    if left.shape != right.shape or event.shape != (left.shape[1],):
        raise RuntimeError(
            f"subject {subject}: incompatible left/right/event shapes "
            f"left={left.shape} right={right.shape} event={event.shape}"
        )
    if not np.isfinite(left[:64]).all() or not np.isfinite(right[:64]).all():
        raise RuntimeError(f"subject {subject}: nonfinite EEG samples")
    if np.any(np.var(left[:64], axis=1) == 0) or np.any(np.var(right[:64], axis=1) == 0):
        raise RuntimeError(f"subject {subject}: zero-variance EEG channel")
    event_onsets = int(np.count_nonzero((event[1:] > 0) & (event[:-1] == 0)))
    if event_onsets + int(event[0] > 0) != n_trials:
        raise RuntimeError(
            f"subject {subject}: event onsets do not match n_imagery_trials"
        )

    return {
        "subject": subject,
        "file": f"s{subject:02d}.mat",
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "sampling_hz": int(eeg.srate),
        "source_channels": int(left.shape[0]),
        "eeg_channels": 64,
        "samples_per_class_recording": int(left.shape[1]),
        "trials_per_class": n_trials,
        "event_onsets": event_onsets + int(event[0] > 0),
        "bad_trial_indices_raw_matlab": _raw_bad_index_cells(eeg.bad_trial_indices),
        "finite_eeg": True,
        "nonzero_channel_variance": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit cached Cho2017 MAT metadata without downloading data or training."
    )
    parser.add_argument("--mne-data-dir", type=Path, required=True)
    parser.add_argument("--subjects", type=_parse_subjects, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_root = args.mne_data_dir.resolve() / RELATIVE_CACHE_ROOT
    missing = [
        cache_root / f"s{subject:02d}.mat"
        for subject in args.subjects
        if not (cache_root / f"s{subject:02d}.mat").is_file()
    ]
    if missing:
        names = ", ".join(path.name for path in missing[:10])
        raise FileNotFoundError(f"Cho2017 cache is incomplete; missing: {names}")

    rows = [
        _audit_subject(cache_root / f"s{subject:02d}.mat", subject)
        for subject in args.subjects
    ]
    first_shape = (
        rows[0]["source_channels"],
        rows[0]["samples_per_class_recording"],
        rows[0]["trials_per_class"],
    )
    identical_shape = all(
        (
            row["source_channels"],
            row["samples_per_class_recording"],
            row["trials_per_class"],
        )
        == first_shape
        for row in rows
    )
    payload = {
        "status": "PASS",
        "subjects": args.subjects,
        "subject_count": len(rows),
        "identical_source_shape": identical_shape,
        "network_or_training": False,
        "bad_trial_index_interpretation": (
            "Raw upstream values only. Nested cells are intentionally unmapped because "
            "their class order and index base are not explicitly documented. The frozen "
            "primary analysis follows MOABB and applies no artifact-index exclusions."
        ),
        "rows": rows,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"PASS Cho2017 cache QA subjects={len(rows)} "
        f"identical_source_shape={identical_shape}"
    )
    for row in rows:
        bad = row["bad_trial_indices_raw_matlab"]
        bad_count = sum(
            len(indices)
            for source in bad.values()
            for indices in source.values()
        )
        print(
            f"subject={row['subject']} trials_per_class={row['trials_per_class']} "
            f"bad_index_entries={bad_count} sha256={row['sha256']}"
        )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
