from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from moabb.paradigms.motor_imagery import LeftRightImagery

from eeg_privacy_benchmark.datasets.base import (
    LoadedArrayDataset,
    TrialRecord,
    validate_loaded_array_dataset,
)
from eeg_privacy_benchmark.datasets.factory import build_dataset_loader
from eeg_privacy_benchmark.models.csp_lda import fit_csp_lda
from eeg_privacy_benchmark.models.eegnet import fit_eegnet


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_cho2017_confirmatory_baselines.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_cho2017_confirmatory_baselines_under_test",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load Cho2017 baseline runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture_main(runner, arguments: list[str]) -> str:
    output = io.StringIO()
    with patch.object(sys, "argv", [str(RUNNER_PATH), *arguments]), redirect_stdout(output):
        runner.main()
    return output.getvalue()


def _fake_payload(runner, model: str, fold, seed: int) -> dict:
    config, manifest = runner._load_sources()
    fold_index = fold.fold_index
    trials_per_class = {
        int(row["subject"]): int(row["trials_per_class"])
        for row in manifest["rows"]
    }
    role_counts = runner._expected_role_counts(fold, trials_per_class)
    attacker = runner.partition_attacker_subjects(
        fold.fitting_subjects,
        seed=config["seeds"]["attacker_subject_partition"],
        fold_index=fold_index,
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "stage": "cho2017_confirmatory_baseline",
        "model": model,
        "fold_index": fold_index,
        "seed": seed,
        "protocol": f"cho2017_confirmatory_fold_{fold_index}",
        "contract_sha256": runner._sha256(runner.CONFIG),
        "cache_manifest_sha256": runner._sha256(runner.CACHE_MANIFEST),
        "roles": {
            "fitting_subjects": list(fold.fitting_subjects),
            "validation_subjects": list(fold.validation_subjects),
            "test_subjects": list(fold.test_subjects),
            "attacker_training_subjects": list(attacker.training_subjects),
            "attacker_evaluation_subjects": list(attacker.evaluation_subjects),
            **role_counts,
        },
        "task_result": {
            "protocol": f"cho2017_confirmatory_fold_{fold_index}",
            "balanced_accuracy": 1.0,
            "macro_f1": 1.0,
        },
        "subject_metrics": [
            {
                "subject": subject,
                "trials": 4,
                "balanced_accuracy": 1.0,
                "macro_f1": 1.0,
            }
            for subject in fold.test_subjects
        ],
        "posterior_cache": None,
    }


def main() -> None:
    runner = _load_runner()
    checks: list[tuple[str, bool]] = []

    csp_signature = inspect.signature(fit_csp_lda)
    eegnet_signature = inspect.signature(fit_eegnet)
    checks.append(
        (
            "model_fitters_accept_preloaded_dataset",
            "loaded_dataset" in csp_signature.parameters
            and "loaded_dataset" in eegnet_signature.parameters,
        )
    )
    paradigm = LeftRightImagery(fmin=8, fmax=32, tmin=0, tmax=3, resample=250)
    checks.append(
        (
            "pinned_moabb_accepts_frozen_preprocessing",
            paradigm.filters == [[8, 32]]
            and paradigm.tmin == 0
            and paradigm.tmax == 3
            and paradigm.resample == 250,
        )
    )

    loader = build_dataset_loader(
        "cho2017",
        resample_hz=250,
        frequency_band_hz=(8, 32),
        epoch_seconds=(0, 3),
    )
    checks.append(
        (
            "loader_freezes_confirmatory_preprocessing",
            loader.resample_hz == 250
            and loader.frequency_band_hz == (8, 32)
            and loader.epoch_seconds == (0, 3),
        )
    )

    record = TrialRecord(
        dataset="cho2017",
        subject_id="sub-3",
        session_id="0",
        run_id="0",
        trial_id="cho2017:sub-3:trial-0",
        raw_label="left_hand",
        canonical_label="left_hand",
    )
    preloaded = LoadedArrayDataset(
        dataset_key="cho2017",
        features=np.zeros((1, 2, 3)),
        labels=np.asarray(["left_hand"]),
        trial_records=(record,),
        resample_hz=250,
        frequency_band_hz=(8, 32),
        epoch_seconds=(0, 3),
    )
    validate_loaded_array_dataset(
        preloaded,
        dataset_key="cho2017",
        subjects=[3],
        resample_hz=250,
        frequency_band_hz=(8, 32),
        epoch_seconds=(0, 3),
    )
    mismatch_rejected = False
    try:
        validate_loaded_array_dataset(
            preloaded,
            dataset_key="cho2017",
            subjects=[3],
            resample_hz=200,
            frequency_band_hz=(8, 32),
            epoch_seconds=(0, 3),
        )
    except ValueError:
        mismatch_rejected = True
    checks.append(("preloaded_provenance_mismatch_is_rejected", mismatch_rejected))

    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        output_dir = temporary_root / "baseline"
        with patch.object(
            runner,
            "_run_job",
            side_effect=AssertionError("dry run attempted training"),
        ):
            dry_output = _capture_main(runner, ["--output-dir", str(output_dir)])
        dry_plan = json.loads(dry_output.split("\nDRY RUN", 1)[0])
        checks.extend(
            [
                (
                    "default_dry_run_is_full_frozen_plan",
                    dry_plan["status"] == "DRY_RUN"
                    and dry_plan["network_downloads"] is False
                    and dry_plan["models"] == ["csp_lda", "compact_eegnet"]
                    and dry_plan["folds"] == [1, 2, 3, 4, 5]
                    and dry_plan["seeds"] == [13, 17, 19, 23]
                    and dry_plan["total_jobs"] == 40,
                ),
                (
                    "dry_run_does_not_write_or_train",
                    not output_dir.exists(),
                ),
                (
                    "dry_run_role_counts_reserve_nonmembers",
                    all(
                        counts["member_training_trials"] > 0
                        and counts["reserved_nonmember_trials"] > 0
                        and counts["validation_trials"] >= 2000
                        and counts["test_trials"] >= 2000
                        for counts in dry_plan["fold_role_counts"].values()
                    ),
                ),
            ]
        )

        calls = []

        def fake_run_job(*, model, fold, seed, **kwargs):
            del kwargs
            calls.append((model, fold.fold_index, seed))
            return _fake_payload(runner, model, fold, seed)

        arguments = [
            "--output-dir",
            str(output_dir),
            "--models",
            "csp_lda",
            "--folds",
            "1",
            "--seeds",
            "13",
            "--confirm-run",
            "--max-jobs",
            "1",
        ]
        with patch.object(runner, "_verify_local_cache"), patch.object(
            runner, "_run_job", side_effect=fake_run_job
        ):
            first_output = _capture_main(runner, arguments)
            second_output = _capture_main(runner, arguments)
        result_path = output_dir / "results" / "csp_lda_fold1_seed13.json"
        checks.append(
            (
                "confirmed_runner_is_resumable",
                calls == [("csp_lda", 1, 13)]
                and result_path.is_file()
                and "COMPLETE csp_lda fold=1 seed=13" in first_output
                and "SKIP csp_lda fold=1 seed=13" in second_output,
            )
        )

        stale_payload = json.loads(result_path.read_text(encoding="utf-8"))
        stale_payload["contract_sha256"] = "0" * 64
        result_path.write_text(json.dumps(stale_payload), encoding="utf-8")
        with patch.object(runner, "_verify_local_cache"), patch.object(
            runner, "_run_job", side_effect=fake_run_job
        ):
            stale_output = _capture_main(runner, arguments)
        checks.append(
            (
                "resume_rejects_stale_contract_provenance",
                calls == [("csp_lda", 1, 13), ("csp_lda", 1, 13)]
                and "COMPLETE csp_lda fold=1 seed=13" in stale_output,
            )
        )

        cache_root = temporary_root / "mne" / runner.RELATIVE_CACHE_ROOT
        cache_root.mkdir(parents=True)
        cache_file = cache_root / "s03.mat"
        cache_file.write_bytes(b"confirmatory-cache-test")
        cache_manifest = {
            "rows": [
                {
                    "file": cache_file.name,
                    "bytes": cache_file.stat().st_size,
                    "sha256": runner._sha256(cache_file),
                }
            ]
        }
        runner._verify_local_cache(temporary_root / "mne", cache_manifest)
        cache_file.write_bytes(b"modified")
        checksum_rejected = False
        try:
            runner._verify_local_cache(temporary_root / "mne", cache_manifest)
        except RuntimeError:
            checksum_rejected = True
        checks.append(("confirmed_run_fails_closed_on_cache_integrity", checksum_rejected))

    records = tuple(
        SimpleNamespace(subject_id=f"sub-{subject}")
        for subject in range(3, 13)
        for _ in range(4)
    )
    labels = np.tile(np.asarray([0, 0, 1, 1]), 10)
    artifacts = SimpleNamespace(
        test_indices=tuple(range(40)),
        trial_records=records,
        encoded_labels=labels,
    )
    metrics = runner._subject_metrics(artifacts, labels.copy())
    checks.append(
        (
            "subject_metrics_keep_subject_as_unit",
            len(metrics) == 10
            and all(row["trials"] == 4 for row in metrics)
            and all(row["balanced_accuracy"] == 1.0 for row in metrics),
        )
    )

    source = RUNNER_PATH.read_text(encoding="utf-8")
    checks.append(
        (
            "runner_is_baseline_only",
            "run_membership" not in source
            and "_run_membership_attack_from_cache" not in source
            and "compact_bottleneck" not in source
            and 'task_model="compact_eegnet"' in source
            and "--confirm-run" in source,
        )
    )

    for name, passed in checks:
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    passed_count = sum(passed for _, passed in checks)
    print(f"passed={passed_count}/{len(checks)}")
    if passed_count != len(checks):
        raise AssertionError("Cho2017 baseline runner check failed")


if __name__ == "__main__":
    main()
