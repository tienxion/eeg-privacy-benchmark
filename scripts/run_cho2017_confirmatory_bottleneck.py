from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
from typing import Any

from eeg_privacy_benchmark.datasets.factory import build_dataset_loader
from eeg_privacy_benchmark.models.eegnet import batched_eegnet_logits, fit_eegnet
from eeg_privacy_benchmark.privacy.membership_inference import (
    load_posterior_feature_cache,
    PosteriorFeatureCache,
    write_posterior_feature_cache,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_RUNNER = ROOT / "scripts" / "run_cho2017_confirmatory_baselines.py"
GATE_FREEZER = ROOT / "scripts" / "freeze_cho2017_baseline_gate.py"
GATE_ARTIFACT = ROOT / "configs" / "cho2017_confirmatory_baseline_gate_v1.json"
BASELINE_OUTPUT = ROOT / "outputs" / "v1_1_cho2017_baseline"
DEFAULT_OUTPUT = ROOT / "outputs" / "v1_1_cho2017_bottleneck"
MODEL = "compact_bottleneck_eegnet_dim6"
BOTTLENECK_DIM = 6


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load required script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _baseline_runner():
    return _load_script(BASELINE_RUNNER, "cho2017_baseline_runner_for_bottleneck")


def _load_gate(runner) -> dict[str, Any]:
    gate = json.loads(GATE_ARTIFACT.read_text(encoding="utf-8"))
    valid = (
        gate["schema_version"] == 1
        and gate["stage"] == "cho2017_confirmatory_baseline_utility_gate"
        and gate["status"] == "PASS"
        and gate["plain_eegnet_utility_gate"]["gate_pass"] is True
        and gate["plain_eegnet_utility_gate"]["next_stage"] == MODEL
        and gate["provenance"]["contract_sha256"] == runner._sha256(runner.CONFIG)
        and gate["provenance"]["cache_manifest_sha256"]
        == runner._sha256(runner.CACHE_MANIFEST)
        and gate["provenance"]["validated_result_jobs"] == 40
        and gate["provenance"]["validated_posterior_caches"] == 20
    )
    if not valid:
        raise RuntimeError("tracked baseline utility gate does not authorize bottleneck execution")
    return gate


def _verify_local_baseline_gate() -> dict[str, Any]:
    runner = _baseline_runner()
    tracked = _load_gate(runner)
    freezer = _load_script(GATE_FREEZER, "cho2017_gate_freezer_for_bottleneck")
    rebuilt = freezer._build_payload(BASELINE_OUTPUT.resolve())
    if rebuilt != tracked:
        raise RuntimeError(
            "local baseline results do not reproduce the tracked utility gate"
        )
    return tracked


def _parse_ints(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("values must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def _job_path(output_dir: Path, fold_index: int, seed: int) -> Path:
    return output_dir / "results" / f"{MODEL}_fold{fold_index}_seed{seed}.json"


def _cache_path(output_dir: Path, fold_index: int, seed: int) -> Path:
    return output_dir / "posterior_caches" / f"{MODEL}_fold{fold_index}_seed{seed}.npz"


def _valid_completed_job(
    path: Path,
    *,
    fold,
    seed: int,
    output_dir: Path,
    expected_role_counts: dict[str, int],
    attacker_partition,
    runner,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fold_index = fold.fold_index
        roles = payload["roles"]
        task_result = payload["task_result"]
        cache_path = _cache_path(output_dir, fold_index, seed)
        valid = (
            payload["schema_version"] == 1
            and payload["status"] == "complete"
            and payload["stage"] == "cho2017_confirmatory_bottleneck"
            and payload["model"] == MODEL
            and payload["fold_index"] == fold_index
            and payload["seed"] == seed
            and payload["protocol"] == f"cho2017_confirmatory_fold_{fold_index}"
            and payload["contract_sha256"] == runner._sha256(runner.CONFIG)
            and payload["cache_manifest_sha256"] == runner._sha256(runner.CACHE_MANIFEST)
            and payload["baseline_gate_sha256"] == runner._sha256(GATE_ARTIFACT)
            and task_result["protocol"] == f"cho2017_confirmatory_fold_{fold_index}"
            and task_result["bottleneck_dim"] == BOTTLENECK_DIM
            and task_result["resample_hz"] == 250
            and task_result["normalization"] == "per_trial_channel"
            and task_result["classifier_head"] == "global_average"
            and len(payload["subject_metrics"]) == 10
            and {row["subject"] for row in payload["subject_metrics"]}
            == set(fold.test_subjects)
            and roles["fitting_subjects"] == list(fold.fitting_subjects)
            and roles["validation_subjects"] == list(fold.validation_subjects)
            and roles["test_subjects"] == list(fold.test_subjects)
            and roles["attacker_training_subjects"]
            == list(attacker_partition.training_subjects)
            and roles["attacker_evaluation_subjects"]
            == list(attacker_partition.evaluation_subjects)
            and all(roles[key] == value for key, value in expected_role_counts.items())
            and payload["posterior_cache"]["relative_path"]
            == str(cache_path.relative_to(output_dir))
            and cache_path.is_file()
        )
        if not valid:
            return False
        if runner._sha256(cache_path) != payload["posterior_cache"]["sha256"]:
            return False
        cache = load_posterior_feature_cache(cache_path)
        return bool(
            cache.task_model == MODEL
            and cache.dataset_key == "cho2017"
            and cache.protocol == f"cho2017_confirmatory_fold_{fold_index}"
            and cache.seed == seed
            and cache.bottleneck_dim == BOTTLENECK_DIM
            and cache.attacker_training_subjects
            == list(attacker_partition.training_subjects)
            and cache.attacker_evaluation_subjects
            == list(attacker_partition.evaluation_subjects)
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _run_job(
    *,
    fold,
    seed: int,
    config: dict[str, Any],
    output_dir: Path,
    runner,
) -> dict[str, Any]:
    subjects = list(config["cohort"]["confirmatory_subjects"])
    loaded = build_dataset_loader(
        "cho2017",
        resample_hz=config["preprocessing"]["neural_resample_hz"],
        frequency_band_hz=tuple(config["preprocessing"]["frequency_band_hz"]),
        epoch_seconds=tuple(config["preprocessing"]["epoch_seconds"]),
    ).load_array_data(subjects=subjects)
    observed_subjects = {
        int(record.subject_id.removeprefix("sub-")) for record in loaded.trial_records
    }
    if observed_subjects != set(subjects):
        raise RuntimeError("loaded Cho2017 cohort does not match the frozen subjects")

    reservation = runner.reserve_membership_nonmembers(
        loaded.trial_records,
        fitting_subjects=fold.fitting_subjects,
        seed=config["seeds"]["nonmember_reservation"],
        fraction=config["membership_design"]["nonmember_fraction_per_fitting_subject"],
    )
    split = runner.build_model_trial_split(
        loaded.trial_records,
        fold=fold,
        reservation=reservation,
    )
    attacker_partition = runner.partition_attacker_subjects(
        fold.fitting_subjects,
        seed=config["seeds"]["attacker_subject_partition"],
        fold_index=fold.fold_index,
    )
    settings = config["models"]["neural_settings"]
    artifacts = fit_eegnet(
        "cho2017",
        protocol=split.protocol,
        seed=seed,
        subjects=subjects,
        epochs=settings["max_epochs"],
        batch_size=settings["batch_size"],
        learning_rate=settings["learning_rate"],
        early_stopping_patience=settings["early_stopping_patience"],
        bottleneck_dim=BOTTLENECK_DIM,
        resample_hz=config["preprocessing"]["neural_resample_hz"],
        normalization=config["preprocessing"]["neural_normalization"],
        classifier_head=config["preprocessing"]["neural_classifier_head"],
        frequency_band_hz=tuple(config["preprocessing"]["frequency_band_hz"]),
        epoch_seconds=tuple(config["preprocessing"]["epoch_seconds"]),
        explicit_trial_split=split,
        loaded_dataset=loaded,
    )
    logits = batched_eegnet_logits(artifacts)
    probabilities = logits.softmax(dim=1).numpy()
    predictions = probabilities[list(artifacts.test_indices)].argmax(axis=1)
    cache = PosteriorFeatureCache(
        task_model=MODEL,
        dataset_key="cho2017",
        protocol=artifacts.protocol,
        seed=artifacts.seed,
        subjects=subjects,
        task_balanced_accuracy=artifacts.result.balanced_accuracy,
        task_macro_f1=artifacts.result.macro_f1,
        class_probabilities=probabilities,
        encoded_labels=artifacts.encoded_labels,
        member_indices=list(artifacts.train_indices),
        nonmember_indices=list(artifacts.nonmember_indices),
        epochs_trained=artifacts.result.epochs_trained,
        best_checkpoint_epoch=artifacts.result.best_checkpoint_epoch,
        best_validation_loss=artifacts.result.best_validation_loss,
        resample_hz=artifacts.result.resample_hz,
        normalization=artifacts.result.normalization,
        classifier_head=artifacts.result.classifier_head,
        label_smoothing=artifacts.result.label_smoothing,
        bottleneck_dim=artifacts.result.bottleneck_dim,
        feature_noise_std=artifacts.result.feature_noise_std,
        mixup_alpha=artifacts.result.mixup_alpha,
        confidence_penalty_beta=artifacts.result.confidence_penalty_beta,
        trial_subject_ids=[record.subject_id for record in artifacts.trial_records],
        attacker_training_subjects=list(attacker_partition.training_subjects),
        attacker_evaluation_subjects=list(attacker_partition.evaluation_subjects),
    )
    cache_path = _cache_path(output_dir, fold.fold_index, seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = cache_path.with_name(cache_path.stem + ".tmp.npz")
    write_posterior_feature_cache(cache, temporary_cache)
    temporary_cache.replace(cache_path)
    load_posterior_feature_cache(cache_path)

    payload = {
        "schema_version": 1,
        "status": "complete",
        "stage": "cho2017_confirmatory_bottleneck",
        "model": MODEL,
        "fold_index": fold.fold_index,
        "seed": seed,
        "protocol": split.protocol,
        "contract_sha256": runner._sha256(runner.CONFIG),
        "cache_manifest_sha256": runner._sha256(runner.CACHE_MANIFEST),
        "baseline_gate_sha256": runner._sha256(GATE_ARTIFACT),
        "roles": {
            "fitting_subjects": list(fold.fitting_subjects),
            "validation_subjects": list(fold.validation_subjects),
            "test_subjects": list(fold.test_subjects),
            "attacker_training_subjects": list(attacker_partition.training_subjects),
            "attacker_evaluation_subjects": list(attacker_partition.evaluation_subjects),
            "member_training_trials": len(split.train_trial_ids),
            "reserved_nonmember_trials": len(split.nonmember_trial_ids),
            "validation_trials": len(split.validation_trial_ids),
            "test_trials": len(split.test_trial_ids),
        },
        "task_result": artifacts.result.to_dict(),
        "subject_metrics": runner._subject_metrics(artifacts, predictions),
        "posterior_cache": {
            "relative_path": str(cache_path.relative_to(output_dir)),
            "sha256": runner._sha256(cache_path),
        },
    }
    del artifacts, loaded
    gc.collect()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or run the frozen Cho2017 compact bottleneck EEGNet dim-6 stage. "
            "Dry-run is the default and never loads data or trains a model."
        )
    )
    parser.add_argument("--mne-data-dir", type=Path, default=ROOT / "raw_data" / "mne_data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=_parse_ints, default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=_parse_ints)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument(
        "--max-jobs",
        type=int,
        help="Stop after this many newly completed jobs; existing results are skipped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = _baseline_runner()
    config, manifest = runner._load_sources()
    gate = _load_gate(runner)
    seeds = args.seeds or list(config["seeds"]["task"])
    if sorted(set(args.folds)) != sorted(args.folds) or not set(args.folds) <= set(range(1, 6)):
        raise ValueError("folds must be unique values from 1 to 5")
    if sorted(set(seeds)) != sorted(seeds) or not set(seeds) <= set(config["seeds"]["task"]):
        raise ValueError("seeds must be unique values from the frozen task seeds")
    if args.max_jobs is not None and args.max_jobs < 1:
        raise ValueError("max-jobs must be positive")

    output_dir = args.output_dir.resolve()
    folds = {
        fold.fold_index: fold
        for fold in runner.build_outer_folds(config["cohort"]["shuffled_subject_blocks"])
    }
    trials_per_class = {
        int(row["subject"]): int(row["trials_per_class"])
        for row in manifest["rows"]
    }
    role_counts = {
        index: runner._expected_role_counts(folds[index], trials_per_class)
        for index in args.folds
    }
    attacker_partitions = {
        index: runner.partition_attacker_subjects(
            folds[index].fitting_subjects,
            seed=config["seeds"]["attacker_subject_partition"],
            fold_index=index,
        )
        for index in args.folds
    }
    jobs = [(folds[fold_index], seed) for fold_index in args.folds for seed in seeds]
    completed = [
        (fold.fold_index, seed)
        for fold, seed in jobs
        if _valid_completed_job(
            _job_path(output_dir, fold.fold_index, seed),
            fold=fold,
            seed=seed,
            output_dir=output_dir,
            expected_role_counts=role_counts[fold.fold_index],
            attacker_partition=attacker_partitions[fold.fold_index],
            runner=runner,
        )
    ]
    plan = {
        "status": "DRY_RUN" if not args.confirm_run else "AUTHORIZED",
        "network_downloads": False,
        "model": MODEL,
        "bottleneck_dim": BOTTLENECK_DIM,
        "folds": args.folds,
        "seeds": seeds,
        "total_jobs": len(jobs),
        "completed_jobs": len(completed),
        "pending_jobs": len(jobs) - len(completed),
        "one_heavy_job_at_a_time": True,
        "baseline_gate": gate["status"],
        "attacks_authorized": False,
    }
    print(json.dumps(plan, indent=2))
    if not args.confirm_run:
        print("DRY RUN only; add --confirm-run to execute pending jobs.")
        return

    _verify_local_baseline_gate()
    runner._verify_local_cache(args.mne_data_dir, manifest)
    runner._configure_cache(args.mne_data_dir)
    newly_completed = 0
    for position, (fold, seed) in enumerate(jobs, start=1):
        path = _job_path(output_dir, fold.fold_index, seed)
        if _valid_completed_job(
            path,
            fold=fold,
            seed=seed,
            output_dir=output_dir,
            expected_role_counts=role_counts[fold.fold_index],
            attacker_partition=attacker_partitions[fold.fold_index],
            runner=runner,
        ):
            print(f"[{position}/{len(jobs)}] SKIP {MODEL} fold={fold.fold_index} seed={seed}")
            continue
        print(f"[{position}/{len(jobs)}] RUN {MODEL} fold={fold.fold_index} seed={seed}")
        payload = _run_job(
            fold=fold,
            seed=seed,
            config=config,
            output_dir=output_dir,
            runner=runner,
        )
        runner._write_json_atomic(path, payload)
        print(
            f"COMPLETE {MODEL} fold={fold.fold_index} seed={seed} "
            f"balanced_accuracy={payload['task_result']['balanced_accuracy']:.4f}"
        )
        newly_completed += 1
        if args.max_jobs is not None and newly_completed >= args.max_jobs:
            print(f"Reached --max-jobs={args.max_jobs}; resume with the same command.")
            break


if __name__ == "__main__":
    main()
