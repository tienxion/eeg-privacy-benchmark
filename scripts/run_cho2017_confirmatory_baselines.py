from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from eeg_privacy_benchmark.cho2017_validation import (
    build_model_trial_split,
    build_outer_folds,
    partition_attacker_subjects,
    reserve_membership_nonmembers,
)
from eeg_privacy_benchmark.datasets.factory import build_dataset_loader
from eeg_privacy_benchmark.models.csp_lda import fit_csp_lda
from eeg_privacy_benchmark.models.eegnet import (
    batched_eegnet_logits,
    fit_eegnet,
)
from eeg_privacy_benchmark.privacy.membership_inference import (
    PosteriorFeatureCache,
    load_posterior_feature_cache,
    write_posterior_feature_cache,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cho2017_confirmatory_validation_v1.yaml"
CACHE_MANIFEST = ROOT / "configs" / "cho2017_confirmatory_cache_manifest_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "v1_1_cho2017_baseline"
ALLOWED_MODELS = ("csp_lda", "compact_eegnet")
RELATIVE_CACHE_ROOT = Path(
    "MNE-gigadb-data/gigadb-datasets/live/pub/10.5524/"
    "100001_101000/100295/mat_data"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_models(value: str) -> list[str]:
    models = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(models) - set(ALLOWED_MODELS))
    if not models or unknown:
        raise argparse.ArgumentTypeError(
            f"models must be a comma-separated subset of {ALLOWED_MODELS}; unknown={unknown}"
        )
    return models


def _configure_cache(mne_data_dir: Path) -> None:
    resolved = str(mne_data_dir.resolve())
    os.environ["MNE_DATA"] = resolved
    os.environ["MNE_DATASETS_GIGADB_PATH"] = resolved
    runtime_cache = ROOT / "outputs" / ".runtime_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(runtime_cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(runtime_cache / "xdg"))


def _load_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    if manifest["qa_status"] != "PASS":
        raise RuntimeError("tracked Cho2017 cache manifest is not PASS")
    if manifest["objective_subject_exclusions"]:
        raise RuntimeError("tracked Cho2017 cache manifest contains exclusions")
    if manifest["summary"]["subject_count"] != 50:
        raise RuntimeError("tracked Cho2017 cache manifest is not the 50-subject cohort")
    return config, manifest


def _verify_local_cache(mne_data_dir: Path, manifest: dict[str, Any]) -> None:
    cache_root = mne_data_dir.resolve() / RELATIVE_CACHE_ROOT
    failures = []
    for row in manifest["rows"]:
        path = cache_root / row["file"]
        if not path.is_file():
            failures.append(f"{row['file']}: missing")
            continue
        if path.stat().st_size != row["bytes"]:
            failures.append(f"{row['file']}: byte count mismatch")
            continue
        if _sha256(path) != row["sha256"]:
            failures.append(f"{row['file']}: SHA-256 mismatch")
    if failures:
        sample = "; ".join(failures[:5])
        raise RuntimeError(
            "Cho2017 local cache failed manifest verification; refusing to call "
            f"MOABB so no download or training can start: {sample}"
        )
    print(f"PASS local cache preflight files={len(manifest['rows'])} checksums=verified")


def _job_path(output_dir: Path, model: str, fold_index: int, seed: int) -> Path:
    return output_dir / "results" / f"{model}_fold{fold_index}_seed{seed}.json"


def _cache_path(output_dir: Path, fold_index: int, seed: int) -> Path:
    return output_dir / "posterior_caches" / f"compact_eegnet_fold{fold_index}_seed{seed}.npz"


def _expected_role_counts(fold, trials_per_class: dict[int, int]) -> dict[str, int]:
    def _trials(subjects) -> int:
        return sum(2 * trials_per_class[int(subject)] for subject in subjects)

    nonmembers = sum(
        2 * int(trials_per_class[int(subject)] * 0.20)
        for subject in fold.fitting_subjects
    )
    fitting_total = _trials(fold.fitting_subjects)
    return {
        "member_training_trials": fitting_total - nonmembers,
        "reserved_nonmember_trials": nonmembers,
        "validation_trials": _trials(fold.validation_subjects),
        "test_trials": _trials(fold.test_subjects),
    }


def _valid_completed_job(
    path: Path,
    *,
    model: str,
    fold,
    seed: int,
    output_dir: Path,
    expected_role_counts: dict[str, int],
    attacker_partition,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fold_index = fold.fold_index
        roles = payload["roles"]
        valid = (
            payload["status"] == "complete"
            and payload["model"] == model
            and payload["fold_index"] == fold_index
            and payload["seed"] == seed
            and len(payload["subject_metrics"]) == 10
            and len({row["subject"] for row in payload["subject_metrics"]}) == 10
            and payload["contract_sha256"] == _sha256(CONFIG)
            and payload["cache_manifest_sha256"] == _sha256(CACHE_MANIFEST)
            and payload["protocol"] == f"cho2017_confirmatory_fold_{fold_index}"
            and payload["task_result"]["protocol"]
            == f"cho2017_confirmatory_fold_{fold_index}"
            and roles["fitting_subjects"] == list(fold.fitting_subjects)
            and roles["validation_subjects"] == list(fold.validation_subjects)
            and roles["test_subjects"] == list(fold.test_subjects)
            and roles["attacker_training_subjects"]
            == list(attacker_partition.training_subjects)
            and roles["attacker_evaluation_subjects"]
            == list(attacker_partition.evaluation_subjects)
            and all(roles[key] == value for key, value in expected_role_counts.items())
            and {row["subject"] for row in payload["subject_metrics"]}
            == set(fold.test_subjects)
        )
        if model == "compact_eegnet":
            cache_path = _cache_path(output_dir, fold_index, seed)
            valid = valid and payload["posterior_cache"]["relative_path"] == str(
                cache_path.relative_to(output_dir)
            )
            valid = valid and cache_path.is_file()
            if valid:
                valid = _sha256(cache_path) == payload["posterior_cache"]["sha256"]
                cache = load_posterior_feature_cache(cache_path)
                valid = (
                    valid
                    and cache.task_model == "compact_eegnet"
                    and cache.protocol == f"cho2017_confirmatory_fold_{fold_index}"
                    and cache.seed == seed
                )
        return bool(valid)
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def _subject_metrics(artifacts, predictions) -> list[dict[str, Any]]:
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, f1_score

    prediction_array = np.asarray(predictions)
    test_indices = list(artifacts.test_indices)
    if prediction_array.shape != (len(test_indices),):
        raise RuntimeError("prediction count does not match test indices")
    metrics = []
    subject_ids = sorted(
        {artifacts.trial_records[index].subject_id for index in test_indices},
        key=lambda value: int(value.removeprefix("sub-")),
    )
    for subject_id in subject_ids:
        positions = [
            position
            for position, index in enumerate(test_indices)
            if artifacts.trial_records[index].subject_id == subject_id
        ]
        labels = np.asarray(
            [artifacts.encoded_labels[test_indices[position]] for position in positions]
        )
        subject_predictions = prediction_array[positions]
        metrics.append(
            {
                "subject": int(subject_id.removeprefix("sub-")),
                "trials": len(positions),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels, subject_predictions)
                ),
                "macro_f1": float(f1_score(labels, subject_predictions, average="macro")),
            }
        )
    if len(metrics) != 10:
        raise RuntimeError("each outer test fold must produce 10 subject metrics")
    return metrics


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_job(
    *,
    model: str,
    fold,
    seed: int,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    subjects = list(config["cohort"]["confirmatory_subjects"])
    resample_hz = 250 if model == "compact_eegnet" else None
    loaded = build_dataset_loader(
        "cho2017",
        resample_hz=resample_hz,
        frequency_band_hz=tuple(config["preprocessing"]["frequency_band_hz"]),
        epoch_seconds=tuple(config["preprocessing"]["epoch_seconds"]),
    ).load_array_data(subjects=subjects)
    observed_subjects = {
        int(record.subject_id.removeprefix("sub-")) for record in loaded.trial_records
    }
    if observed_subjects != set(subjects):
        raise RuntimeError("loaded Cho2017 cohort does not match the frozen subjects")

    reservation = reserve_membership_nonmembers(
        loaded.trial_records,
        fitting_subjects=fold.fitting_subjects,
        seed=config["seeds"]["nonmember_reservation"],
        fraction=config["membership_design"]["nonmember_fraction_per_fitting_subject"],
    )
    split = build_model_trial_split(
        loaded.trial_records,
        fold=fold,
        reservation=reservation,
    )
    attacker_partition = partition_attacker_subjects(
        fold.fitting_subjects,
        seed=config["seeds"]["attacker_subject_partition"],
        fold_index=fold.fold_index,
    )

    if model == "csp_lda":
        artifacts = fit_csp_lda(
            "cho2017",
            protocol=split.protocol,
            seed=seed,
            subjects=subjects,
            n_components=config["preprocessing"]["csp_anchor"]["n_components"],
            frequency_band_hz=tuple(config["preprocessing"]["frequency_band_hz"]),
            epoch_seconds=tuple(config["preprocessing"]["epoch_seconds"]),
            explicit_trial_split=split,
            loaded_dataset=loaded,
        )
        predictions = artifacts.pipeline.named_steps["lda"].predict(
            artifacts.all_features[list(artifacts.test_indices)]
        )
        posterior_record = None
    else:
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
            task_model="compact_eegnet",
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
        posterior_record = {
            "relative_path": str(cache_path.relative_to(output_dir)),
            "sha256": _sha256(cache_path),
        }

    subject_metrics = _subject_metrics(artifacts, predictions)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "stage": "cho2017_confirmatory_baseline",
        "model": model,
        "fold_index": fold.fold_index,
        "seed": seed,
        "protocol": split.protocol,
        "contract_sha256": _sha256(CONFIG),
        "cache_manifest_sha256": _sha256(CACHE_MANIFEST),
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
        "subject_metrics": subject_metrics,
        "posterior_cache": posterior_record,
    }
    del artifacts, loaded
    gc.collect()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or run the frozen Cho2017 CSP-LDA/plain-EEGNet baseline stage. "
            "Dry-run is the default and never loads data or trains a model."
        )
    )
    parser.add_argument("--mne-data-dir", type=Path, default=ROOT / "raw_data" / "mne_data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", type=_parse_models, default=list(ALLOWED_MODELS))
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
    config, manifest = _load_sources()
    seeds = args.seeds or list(config["seeds"]["task"])
    if len(set(args.models)) != len(args.models):
        raise ValueError("models must be unique")
    if sorted(set(args.folds)) != sorted(args.folds) or not set(args.folds) <= set(range(1, 6)):
        raise ValueError("folds must be unique values from 1 to 5")
    if sorted(set(seeds)) != sorted(seeds) or not set(seeds) <= set(config["seeds"]["task"]):
        raise ValueError("seeds must be unique values from the frozen task seeds")
    if args.max_jobs is not None and args.max_jobs < 1:
        raise ValueError("max-jobs must be positive")

    output_dir = args.output_dir.resolve()
    folds_by_index = {
        fold.fold_index: fold
        for fold in build_outer_folds(config["cohort"]["shuffled_subject_blocks"])
    }
    trials_per_class = {
        int(row["subject"]): int(row["trials_per_class"])
        for row in manifest["rows"]
    }
    role_counts_by_fold = {
        index: _expected_role_counts(folds_by_index[index], trials_per_class)
        for index in args.folds
    }
    attacker_partitions_by_fold = {
        index: partition_attacker_subjects(
            folds_by_index[index].fitting_subjects,
            seed=config["seeds"]["attacker_subject_partition"],
            fold_index=index,
        )
        for index in args.folds
    }
    jobs = [
        (model, folds_by_index[fold_index], seed)
        for model in args.models
        for fold_index in args.folds
        for seed in seeds
    ]
    completed = [
        (model, fold.fold_index, seed)
        for model, fold, seed in jobs
        if _valid_completed_job(
            _job_path(output_dir, model, fold.fold_index, seed),
            model=model,
            fold=fold,
            seed=seed,
            output_dir=output_dir,
            expected_role_counts=role_counts_by_fold[fold.fold_index],
            attacker_partition=attacker_partitions_by_fold[fold.fold_index],
        )
    ]
    plan = {
        "status": "DRY_RUN" if not args.confirm_run else "AUTHORIZED",
        "network_downloads": False,
        "models": args.models,
        "folds": args.folds,
        "seeds": seeds,
        "total_jobs": len(jobs),
        "completed_jobs": len(completed),
        "pending_jobs": len(jobs) - len(completed),
        "one_heavy_job_at_a_time": True,
        "fold_role_counts": {
            str(index): role_counts_by_fold[index] for index in args.folds
        },
    }
    print(json.dumps(plan, indent=2))
    if not args.confirm_run:
        print("DRY RUN only; add --confirm-run to execute pending jobs.")
        return

    _verify_local_cache(args.mne_data_dir, manifest)
    _configure_cache(args.mne_data_dir)
    newly_completed = 0
    for position, (model, fold, seed) in enumerate(jobs, start=1):
        path = _job_path(output_dir, model, fold.fold_index, seed)
        if _valid_completed_job(
            path,
            model=model,
            fold=fold,
            seed=seed,
            output_dir=output_dir,
            expected_role_counts=role_counts_by_fold[fold.fold_index],
            attacker_partition=attacker_partitions_by_fold[fold.fold_index],
        ):
            print(f"[{position}/{len(jobs)}] SKIP {model} fold={fold.fold_index} seed={seed}")
            continue
        print(f"[{position}/{len(jobs)}] RUN {model} fold={fold.fold_index} seed={seed}")
        payload = _run_job(
            model=model,
            fold=fold,
            seed=seed,
            config=config,
            output_dir=output_dir,
        )
        _write_json_atomic(path, payload)
        print(
            f"COMPLETE {model} fold={fold.fold_index} seed={seed} "
            f"balanced_accuracy={payload['task_result']['balanced_accuracy']:.4f}"
        )
        newly_completed += 1
        if args.max_jobs is not None and newly_completed >= args.max_jobs:
            print(f"Reached --max-jobs={args.max_jobs}; resume with the same command.")
            break


if __name__ == "__main__":
    main()
