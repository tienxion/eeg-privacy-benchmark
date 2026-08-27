"""Command-line entrypoints for the EEG privacy benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from eeg_privacy_benchmark.datasets.cache_status import (
    import_dataset_cache,
    inspect_dataset_cache,
)
from eeg_privacy_benchmark.datasets.factory import build_dataset_loader
from eeg_privacy_benchmark.datasets.manifest_io import (
    write_dataset_manifest,
    write_split_manifest,
)
from eeg_privacy_benchmark.datasets.registry import DATASET_REGISTRY
from eeg_privacy_benchmark.datasets.splits import (
    generate_cross_run_split,
    generate_cross_session_split,
    generate_cross_subject_split,
)
from eeg_privacy_benchmark.experiments import (
    run_adversarial_weight_sweep,
    run_cached_membership_inference_sweep,
    run_membership_inference_sweep,
    run_subject_id_sweep,
)
from eeg_privacy_benchmark.models import (
    run_csp_lda,
    run_eegnet,
    run_federated_eegnet,
)
from eeg_privacy_benchmark.privacy import (
    run_adversarial_eegnet_membership_inference_attack,
    run_adversarial_eegnet_subject_id_probe,
    run_bottleneck_eegnet_membership_inference_attack,
    run_bottleneck_eegnet_subject_id_probe,
    run_cached_membership_inference_attack,
    run_confidence_penalty_eegnet_membership_inference_attack,
    run_confidence_penalty_eegnet_subject_id_probe,
    run_csp_membership_inference_attack,
    run_csp_subject_id_probe,
    run_eegnet_membership_inference_attack,
    run_eegnet_subject_id_probe,
    run_feature_noise_eegnet_membership_inference_attack,
    run_feature_noise_eegnet_subject_id_probe,
    run_federated_eegnet_membership_inference_attack,
    run_federated_eegnet_subject_id_probe,
    run_label_smoothing_eegnet_membership_inference_attack,
    run_label_smoothing_eegnet_subject_id_probe,
    run_mixup_eegnet_membership_inference_attack,
    run_mixup_eegnet_subject_id_probe,
)
from eeg_privacy_benchmark.reporting import (
    build_adversarial_defense_report,
    build_membership_inference_report,
    build_subject_id_comparison_report,
)
from eeg_privacy_benchmark.results import write_result_json
from eeg_privacy_benchmark.workflows import inspect_workflows


REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_subjects(subjects_arg: str | None) -> list[int] | None:
    if not subjects_arg:
        return None
    return [int(value.strip()) for value in subjects_arg.split(",") if value.strip()]


def _run_repo_script(script_name: str, extra_args: list[str] | None = None) -> None:
    script_path = REPO_ROOT / "scripts" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    command = [sys.executable, str(script_path), *(extra_args or [])]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _parse_posterior_cache_paths(
    *,
    posterior_caches: str | None,
    posterior_cache_dir: str | None,
) -> list[Path]:
    paths = []
    if posterior_caches:
        explicit_paths = [
            Path(value.strip()) for value in posterior_caches.split(",") if value.strip()
        ]
        missing_paths = [path for path in explicit_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(
                "Posterior cache path does not exist: "
                + ", ".join(str(path) for path in missing_paths)
            )
        non_file_paths = [path for path in explicit_paths if not path.is_file()]
        if non_file_paths:
            raise ValueError(
                "Posterior cache path is not a file: "
                + ", ".join(str(path) for path in non_file_paths)
            )
        paths.extend(explicit_paths)
    if posterior_cache_dir:
        cache_dir = Path(posterior_cache_dir)
        if not cache_dir.exists():
            raise FileNotFoundError(
                f"Posterior cache directory does not exist: {cache_dir}"
            )
        if not cache_dir.is_dir():
            raise NotADirectoryError(
                f"Posterior cache directory is not a directory: {cache_dir}"
            )
        directory_paths = sorted(cache_dir.glob("*.npz"))
        if not directory_paths:
            raise ValueError(
                f"Posterior cache directory contains no .npz files: {cache_dir}"
            )
        paths.extend(directory_paths)
    if not paths:
        raise ValueError("Provide --posterior-caches, --posterior-cache-dir, or both.")
    return paths


def _subjects_slug(subjects: list[int] | None) -> str:
    if not subjects:
        return ""
    return "_subject" + "".join(str(subject) for subject in subjects)


def _configure_runtime_cache_dirs() -> None:
    cache_root = Path(
        os.environ.get("EEG_PRIVACY_BENCHMARK_CACHE_DIR", "outputs/.runtime_cache")
    ).resolve()
    matplotlib_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


def _dataset_mne_config_keys(dataset_key: str | None) -> list[str]:
    if not dataset_key:
        return []

    spec = DATASET_REGISTRY.get(dataset_key)
    if spec is None:
        return []

    dataset_specific_tokens = {
        "bnci2014_001": {"BNCI"},
        "physionet_motor_imagery": {"EEGBCI", "PhysionetMotorImagery"},
        "lee2019_mi": {"Lee2019-MI"},
        "cho2017": {"GIGADB"},
        "shin2017a": {"BBCIFNIRS"},
    }
    tokens = {
        dataset_key,
        dataset_key.replace("_", "-"),
        spec.source_id,
        spec.source_id.replace("_", "-"),
        *dataset_specific_tokens.get(dataset_key, set()),
    }
    return sorted({f"MNE_DATASETS_{token.upper()}_PATH" for token in tokens})


def _configure_mne_data_dir(mne_data_dir: str | None, dataset_key: str | None = None) -> None:
    if not mne_data_dir:
        return
    path = Path(mne_data_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    resolved = str(path)
    os.environ["MNE_DATA"] = resolved
    for key in _dataset_mne_config_keys(dataset_key):
        os.environ[key] = resolved


def _emit_result(result, output: str | None) -> None:
    payload = result.to_dict() if hasattr(result, "to_dict") else result.__dict__
    for key, value in payload.items():
        if key == "history":
            print(f"history_entries={len(value)}")
            continue
        if isinstance(value, float):
            print(f"{key}={value:.4f}")
        else:
            print(f"{key}={value}")
    if output:
        write_result_json(payload, output)
        print(f"wrote_result={Path(output).resolve()}")


def _cli_cache_location(args: argparse.Namespace) -> str | None:
    mne_data_dir = getattr(args, "mne_data_dir", None)
    if mne_data_dir:
        return str(Path(mne_data_dir).resolve())
    return os.environ.get("MNE_DATA")


def _handle_cli_exception(exc: Exception, args: argparse.Namespace) -> None:
    dataset = getattr(args, "dataset", "dataset")
    cache_location = _cli_cache_location(args)
    message = str(exc)
    exc_name = exc.__class__.__name__
    exc_module = exc.__class__.__module__

    if message.startswith("Posterior cache "):
        raise SystemExit(message) from exc

    if isinstance(exc, FileNotFoundError) and cache_location:
        raise SystemExit(
            f"{dataset}: dataset cache lookup failed under {cache_location}. {message}"
        ) from exc

    if exc_name == "ConnectionError" and exc_module.startswith("requests"):
        cache_hint = cache_location or "the configured MNE cache"
        raise SystemExit(
            f"{dataset}: required dataset files are not fully present under {cache_hint}, "
            "and the CLI could not download the missing files. Populate the cache first "
            "or rerun with network access."
        ) from exc

    raise exc


def build_manifest_command(args: argparse.Namespace) -> None:
    _configure_mne_data_dir(args.mne_data_dir, args.dataset)
    loader = build_dataset_loader(args.dataset)
    manifest = loader.load_manifest(subjects=_parse_subjects(args.subjects))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_dataset_manifest(manifest, output_path)
    print(f"Wrote manifest to {output_path}")
    print(f"Trials: {len(manifest.trial_records)}")
    print(f"Labels: {', '.join(manifest.available_labels)}")


def build_splits_command(args: argparse.Namespace) -> None:
    _configure_mne_data_dir(args.mne_data_dir, args.dataset)
    requested_subjects = _parse_subjects(args.subjects)
    loader = build_dataset_loader(args.dataset)
    manifest = loader.load_manifest(subjects=requested_subjects)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_slug = _subjects_slug(requested_subjects)

    try:
        subject_split = generate_cross_subject_split(manifest, seed=args.seed)
    except ValueError as exc:
        print(f"Skipping cross-subject split: {exc}")
    else:
        cross_subject_filename = (
            f"{args.dataset}{subset_slug}_cross_subject_seed{args.seed}.json"
        )
        write_split_manifest(
            subject_split,
            output_dir / cross_subject_filename,
        )
        print(
            "Wrote cross-subject split to "
            f"{output_dir / cross_subject_filename}"
        )

    try:
        session_split = generate_cross_session_split(manifest, seed=args.seed)
    except ValueError as exc:
        print(f"Skipping cross-session split: {exc}")
    else:
        cross_session_filename = (
            f"{args.dataset}{subset_slug}_cross_session_seed{args.seed}.json"
        )
        write_split_manifest(
            session_split,
            output_dir / cross_session_filename,
        )
        print(
            "Wrote cross-session split to "
            f"{output_dir / cross_session_filename}"
        )

    try:
        run_split = generate_cross_run_split(manifest, seed=args.seed)
    except ValueError as exc:
        print(f"Skipping cross-run split: {exc}")
    else:
        cross_run_filename = f"{args.dataset}{subset_slug}_cross_run_seed{args.seed}.json"
        write_split_manifest(
            run_split,
            output_dir / cross_run_filename,
        )
        print(
            "Wrote cross-run split to "
            f"{output_dir / cross_run_filename}"
        )


def check_dataset_cache_command(args: argparse.Namespace) -> None:
    dataset_keys = [args.dataset] if args.dataset else sorted(DATASET_REGISTRY)
    requested_subjects = _parse_subjects(args.subjects)
    cache_dir = args.mne_data_dir or os.environ.get("MNE_DATA") or "raw_data/mne_data"

    for index, dataset_key in enumerate(dataset_keys):
        status = inspect_dataset_cache(
            dataset_key=dataset_key,
            cache_dir=cache_dir,
            subjects=requested_subjects,
        )
        print(f"dataset={status.dataset_key}")
        print(f"cache_root={status.cache_root}")
        print(f"subjects={status.subjects}")
        print(f"expected_files={status.expected_files}")
        print(f"present_files={status.present_files}")
        print(f"missing_files={status.missing_files}")
        print(f"complete={status.complete}")
        if status.sample_missing_paths:
            print("sample_missing_paths=")
            for path in status.sample_missing_paths:
                print(path)
        if index != len(dataset_keys) - 1:
            print("")


def import_dataset_cache_command(args: argparse.Namespace) -> None:
    result = import_dataset_cache(
        dataset_key=args.dataset,
        source_cache_dir=args.source_mne_data_dir,
        destination_cache_dir=args.mne_data_dir,
        subjects=_parse_subjects(args.subjects),
    )
    print(f"dataset={result.dataset_key}")
    print(f"source_cache_root={result.source_cache_root}")
    print(f"destination_cache_root={result.destination_cache_root}")
    print(f"subjects={result.subjects}")
    print(f"expected_files={result.expected_files}")
    print(f"copied_files={result.copied_files}")
    print(f"skipped_existing_files={result.skipped_existing_files}")
    print(f"missing_source_files={result.missing_source_files}")
    if result.sample_missing_source_paths:
        print("sample_missing_source_paths=")
        for path in result.sample_missing_source_paths:
            print(path)


def run_csp_lda_command(args: argparse.Namespace) -> None:
    _configure_mne_data_dir(args.mne_data_dir, args.dataset)
    result = run_csp_lda(
        dataset_key=args.dataset,
        protocol=args.protocol,
        seed=args.seed,
        subjects=_parse_subjects(args.subjects),
        n_components=args.n_components,
    )
    _emit_result(result, args.output)


def run_eegnet_command(args: argparse.Namespace) -> None:
    _configure_mne_data_dir(args.mne_data_dir, args.dataset)
    result = run_eegnet(
        dataset_key=args.dataset,
        protocol=args.protocol,
        seed=args.seed,
        subjects=_parse_subjects(args.subjects),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        early_stopping_patience=args.early_stopping_patience,
    )
    _emit_result(result, args.output)


def run_federated_eegnet_command(args: argparse.Namespace) -> None:
    _configure_mne_data_dir(args.mne_data_dir, args.dataset)
    result = run_federated_eegnet(
        dataset_key=args.dataset,
        protocol=args.protocol,
        seed=args.seed,
        subjects=_parse_subjects(args.subjects),
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_path=args.checkpoint,
        checkpoint_interval=args.checkpoint_interval,
        resume_from_checkpoint=not args.no_resume,
        resample_hz=args.resample_hz,
        normalization=args.normalization,
        classifier_head=args.classifier_head,
        bottleneck_dim=args.bottleneck_dim,
        confidence_penalty_beta=args.confidence_penalty_beta,
    )
    _emit_result(result, args.output)


def run_subject_id_command(args: argparse.Namespace) -> None:
    _configure_mne_data_dir(args.mne_data_dir, args.dataset)
    if args.model == "federated_eegnet":
        result = run_federated_eegnet_subject_id_probe(
            dataset_key=args.dataset,
            protocol=args.protocol,
            seed=args.seed,
            subjects=_parse_subjects(args.subjects),
            federated_rounds=args.federated_rounds,
            federated_local_epochs=args.federated_local_epochs,
            task_batch_size=args.task_batch_size,
            task_learning_rate=args.task_learning_rate,
            validation_fraction=args.validation_fraction,
            early_stopping_patience=args.early_stopping_patience,
        )
    elif args.model == "eegnet":
        result = run_eegnet_subject_id_probe(
            dataset_key=args.dataset,
            protocol=args.protocol,
            seed=args.seed,
            subjects=_parse_subjects(args.subjects),
            task_epochs=args.task_epochs,
            task_batch_size=args.task_batch_size,
            task_learning_rate=args.task_learning_rate,
            validation_fraction=args.validation_fraction,
            early_stopping_patience=args.early_stopping_patience,
        )
    elif args.model == "label_smoothing_eegnet":
        result = run_label_smoothing_eegnet_subject_id_probe(
            dataset_key=args.dataset,
            protocol=args.protocol,
            seed=args.seed,
            subjects=_parse_subjects(args.subjects),
            task_epochs=args.task_epochs,
            task_batch_size=args.task_batch_size,
            task_learning_rate=args.task_learning_rate,
            validation_fraction=args.validation_fraction,
            early_stopping_patience=args.early_stopping_patience,
            label_smoothing=args.label_smoothing,
        )
    elif args.model == "bottleneck_eegnet":
        result = run_bottleneck_eegnet_subject_id_probe(
            dataset_key=args.dataset,
            protocol=args.protocol,
            seed=args.seed,
            subjects=_parse_subjects(args.subjects),
            task_epochs=args.task_epochs,
            task_batch_size=args.task_batch_size,
            task_learning_rate=args.task_learning_rate,
            validation_fraction=args.validation_fraction,
            early_stopping_patience=args.early_stopping_patience,
            bottleneck_dim=args.bottleneck_dim,
        )
    elif args.model == "feature_noise_eegnet":
        result = run_feature_noise_eegnet_subject_id_probe(
            dataset_key=args.dataset,
            protocol=args.protocol,
            seed=args.seed,
            subjects=_parse_subjects(args.subjects),
            task_epochs=args.task_epochs,
            task_batch_size=args.task_batch_size,
            task_learning_rate=args.task_learning_rate,
            validation_fraction=args.validation_fraction,
            early_stopping_patience=args.early_stopping_patience,
            feature_noise_std=args.feature_noise_std,
        )
    elif args.model == "mixup_eegnet":
        result = run_mixup_eegnet_subject_id_probe(
            dataset_key=args.dataset,
            protocol=args.protocol,
            seed=args.seed,
            subjects=_parse_subjects(args.subjects),
            task_epochs=args.task_epochs,
            task_batch_size=args.task_batch_size,
            task_learning_rate=args.task_learning_rate,
            validation_fraction=args.validation_fraction,
            early_stopping_patience=args.early_stopping_patience,
            mixup_alpha=args.mixup_alpha,
        )
    elif args.model == "confidence_penalty_eegnet":
        result = run_confidence_penalty_eegnet_subject_id_probe(
            dataset_key=args.dataset,
            protocol=args.protocol,
            seed=args.seed,
            subjects=_parse_subjects(args.subjects),
            task_epochs=args.task_epochs,
            task_batch_size=args.task_batch_size,
            task_learning_rate=args.task_learning_rate,
            validation_fraction=args.validation_fraction,
            early_stopping_patience=args.early_stopping_patience,
            confidence_penalty_beta=args.confidence_penalty_beta,
        )
    elif args.model == "adversarial_eegnet":
        result = run_adversarial_eegnet_subject_id_probe(
            dataset_key=args.dataset,
            protocol=args.protocol,
            seed=args.seed,
            subjects=_parse_subjects(args.subjects),
            task_epochs=args.task_epochs,
            task_batch_size=args.task_batch_size,
            task_learning_rate=args.task_learning_rate,
            validation_fraction=args.validation_fraction,
            early_stopping_patience=args.early_stopping_patience,
            adversarial_weight=args.adversarial_weight,
            adversarial_schedule=args.adversarial_schedule,
            ramp_up_fraction=args.ramp_up_fraction,
        )
    else:
        result = run_csp_subject_id_probe(
            dataset_key=args.dataset,
            protocol=args.protocol,
            seed=args.seed,
            subjects=_parse_subjects(args.subjects),
            n_components=args.n_components,
        )
    _emit_result(result, args.output)


def run_membership_inference_command(args: argparse.Namespace) -> None:
    _configure_mne_data_dir(args.mne_data_dir, args.dataset)
    if args.model == "federated_eegnet":
        result = run_federated_eegnet_membership_inference_attack(
            dataset_key=args.dataset,
            protocol=args.protocol,
            seed=args.seed,
            subjects=_parse_subjects(args.subjects),
            federated_rounds=args.federated_rounds,
            federated_local_epochs=args.federated_local_epochs,
            task_batch_size=args.task_batch_size,
            task_learning_rate=args.task_learning_rate,
            validation_fraction=args.validation_fraction,
            early_stopping_patience=args.early_stopping_patience,
            attack_type=args.attack_type,
            score_type=args.score_type,
            attack_seed=args.attack_seed,
            posterior_cache_output=args.posterior_cache_output,
        )
    elif args.model == "eegnet":
        result = run_eegnet_membership_inference_attack(
            dataset_key=args.dataset,
            protocol=args.protocol,
            seed=args.seed,
            subjects=_parse_subjects(args.subjects),
            task_epochs=args.task_epochs,
            task_batch_size=args.task_batch_size,
            task_learning_rate=args.task_learning_rate,
            validation_fraction=args.validation_fraction,
            early_stopping_patience=args.early_stopping_patience,
            attack_type=args.attack_type,
            score_type=args.score_type,
            attack_seed=args.attack_seed,
            posterior_cache_output=args.posterior_cache_output,
        )
    elif args.model == "label_smoothing_eegnet":
        result = run_label_smoothing_eegnet_membership_inference_attack(
            dataset_key=args.dataset,
            protocol=args.protocol,
            seed=args.seed,
            subjects=_parse_subjects(args.subjects),
            task_epochs=args.task_epochs,
            task_batch_size=args.task_batch_size,
            task_learning_rate=args.task_learning_rate,
            validation_fraction=args.validation_fraction,
            early_stopping_patience=args.early_stopping_patience,
            attack_type=args.attack_type,
            score_type=args.score_type,
            attack_seed=args.attack_seed,
            label_smoothing=args.label_smoothing,
            posterior_cache_output=args.posterior_cache_output,
        )
    elif args.model == "bottleneck_eegnet":
        result = run_bottleneck_eegnet_membership_inference_attack(
            dataset_key=args.dataset,
            protocol=args.protocol,
            seed=args.seed,
            subjects=_parse_subjects(args.subjects),
            task_epochs=args.task_epochs,
            task_batch_size=args.task_batch_size,
            task_learning_rate=args.task_learning_rate,
            validation_fraction=args.validation_fraction,
            early_stopping_patience=args.early_stopping_patience,
            attack_type=args.attack_type,
            score_type=args.score_type,
            attack_seed=args.attack_seed,
            bottleneck_dim=args.bottleneck_dim,
            posterior_cache_output=args.posterior_cache_output,
        )
    elif args.model == "feature_noise_eegnet":
        result = run_feature_noise_eegnet_membership_inference_attack(
            dataset_key=args.dataset,
            protocol=args.protocol,
            seed=args.seed,
            subjects=_parse_subjects(args.subjects),
            task_epochs=args.task_epochs,
            task_batch_size=args.task_batch_size,
            task_learning_rate=args.task_learning_rate,
            validation_fraction=args.validation_fraction,
            early_stopping_patience=args.early_stopping_patience,
            attack_type=args.attack_type,
            score_type=args.score_type,
            attack_seed=args.attack_seed,
            feature_noise_std=args.feature_noise_std,
            posterior_cache_output=args.posterior_cache_output,
        )
    elif args.model == "mixup_eegnet":
        result = run_mixup_eegnet_membership_inference_attack(
            dataset_key=args.dataset,
            protocol=args.protocol,
            seed=args.seed,
            subjects=_parse_subjects(args.subjects),
            task_epochs=args.task_epochs,
            task_batch_size=args.task_batch_size,
            task_learning_rate=args.task_learning_rate,
            validation_fraction=args.validation_fraction,
            early_stopping_patience=args.early_stopping_patience,
            attack_type=args.attack_type,
            score_type=args.score_type,
            attack_seed=args.attack_seed,
            mixup_alpha=args.mixup_alpha,
            posterior_cache_output=args.posterior_cache_output,
        )
    elif args.model == "confidence_penalty_eegnet":
        result = run_confidence_penalty_eegnet_membership_inference_attack(
            dataset_key=args.dataset,
            protocol=args.protocol,
            seed=args.seed,
            subjects=_parse_subjects(args.subjects),
            task_epochs=args.task_epochs,
            task_batch_size=args.task_batch_size,
            task_learning_rate=args.task_learning_rate,
            validation_fraction=args.validation_fraction,
            early_stopping_patience=args.early_stopping_patience,
            attack_type=args.attack_type,
            score_type=args.score_type,
            attack_seed=args.attack_seed,
            confidence_penalty_beta=args.confidence_penalty_beta,
            posterior_cache_output=args.posterior_cache_output,
        )
    elif args.model == "adversarial_eegnet":
        result = run_adversarial_eegnet_membership_inference_attack(
            dataset_key=args.dataset,
            protocol=args.protocol,
            seed=args.seed,
            subjects=_parse_subjects(args.subjects),
            task_epochs=args.task_epochs,
            task_batch_size=args.task_batch_size,
            task_learning_rate=args.task_learning_rate,
            validation_fraction=args.validation_fraction,
            early_stopping_patience=args.early_stopping_patience,
            adversarial_weight=args.adversarial_weight,
            adversarial_schedule=args.adversarial_schedule,
            ramp_up_fraction=args.ramp_up_fraction,
            attack_type=args.attack_type,
            score_type=args.score_type,
            attack_seed=args.attack_seed,
            posterior_cache_output=args.posterior_cache_output,
        )
    else:
        result = run_csp_membership_inference_attack(
            dataset_key=args.dataset,
            protocol=args.protocol,
            seed=args.seed,
            subjects=_parse_subjects(args.subjects),
            n_components=args.n_components,
            attack_type=args.attack_type,
            score_type=args.score_type,
            attack_seed=args.attack_seed,
            posterior_cache_output=args.posterior_cache_output,
        )
    _emit_result(result, args.output)


def run_membership_inference_from_cache_command(args: argparse.Namespace) -> None:
    result = run_cached_membership_inference_attack(
        args.posterior_cache,
        attack_type=args.attack_type,
        score_type=args.score_type,
        attack_seed=args.attack_seed,
    )
    _emit_result(result, args.output)


def run_subject_id_sweep_command(args: argparse.Namespace) -> None:
    _configure_mne_data_dir(args.mne_data_dir, args.dataset)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    summary = run_subject_id_sweep(
        model=args.model,
        dataset_key=args.dataset,
        protocol=args.protocol,
        seeds=seeds,
        subjects=_parse_subjects(args.subjects),
        output_dir=args.output_dir,
        task_epochs=args.task_epochs,
        task_batch_size=args.task_batch_size,
        task_learning_rate=args.task_learning_rate,
        validation_fraction=args.validation_fraction,
        early_stopping_patience=args.early_stopping_patience,
        adversarial_weight=args.adversarial_weight,
        adversarial_schedule=args.adversarial_schedule,
        ramp_up_fraction=args.ramp_up_fraction,
        n_components=args.n_components,
        label_smoothing=args.label_smoothing,
        bottleneck_dim=args.bottleneck_dim,
        feature_noise_std=args.feature_noise_std,
        mixup_alpha=args.mixup_alpha,
        confidence_penalty_beta=args.confidence_penalty_beta,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


def run_membership_inference_sweep_command(args: argparse.Namespace) -> None:
    _configure_mne_data_dir(args.mne_data_dir, args.dataset)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    summary = run_membership_inference_sweep(
        model=args.model,
        dataset_key=args.dataset,
        protocol=args.protocol,
        seeds=seeds,
        subjects=_parse_subjects(args.subjects),
        output_dir=args.output_dir,
        task_epochs=args.task_epochs,
        task_batch_size=args.task_batch_size,
        task_learning_rate=args.task_learning_rate,
        validation_fraction=args.validation_fraction,
        early_stopping_patience=args.early_stopping_patience,
        adversarial_weight=args.adversarial_weight,
        adversarial_schedule=args.adversarial_schedule,
        ramp_up_fraction=args.ramp_up_fraction,
        n_components=args.n_components,
        label_smoothing=args.label_smoothing,
        bottleneck_dim=args.bottleneck_dim,
        feature_noise_std=args.feature_noise_std,
        mixup_alpha=args.mixup_alpha,
        confidence_penalty_beta=args.confidence_penalty_beta,
        attack_type=args.attack_type,
        score_type=args.score_type,
        attack_seed=args.attack_seed,
        posterior_cache_dir=args.posterior_cache_dir,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


def run_membership_inference_cache_sweep_command(args: argparse.Namespace) -> None:
    summary = run_cached_membership_inference_sweep(
        posterior_cache_paths=_parse_posterior_cache_paths(
            posterior_caches=args.posterior_caches,
            posterior_cache_dir=args.posterior_cache_dir,
        ),
        output_dir=args.output_dir,
        attack_type=args.attack_type,
        score_type=args.score_type,
        attack_seed=args.attack_seed,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


def run_adversarial_weight_sweep_command(args: argparse.Namespace) -> None:
    _configure_mne_data_dir(args.mne_data_dir, args.dataset)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    weights = [float(value.strip()) for value in args.weights.split(",") if value.strip()]
    summary = run_adversarial_weight_sweep(
        dataset_key=args.dataset,
        protocol=args.protocol,
        seeds=seeds,
        subjects=_parse_subjects(args.subjects),
        adversarial_weights=weights,
        output_dir=args.output_dir,
        task_epochs=args.task_epochs,
        task_batch_size=args.task_batch_size,
        task_learning_rate=args.task_learning_rate,
        validation_fraction=args.validation_fraction,
        early_stopping_patience=args.early_stopping_patience,
        adversarial_schedule=args.adversarial_schedule,
        ramp_up_fraction=args.ramp_up_fraction,
    )
    for key, value in summary.items():
        if key == "weight_summaries":
            print(f"weight_summaries={len(value)}")
        else:
            print(f"{key}={value}")


def build_subject_id_report_command(args: argparse.Namespace) -> None:
    build_subject_id_comparison_report(
        summary_paths=[path.strip() for path in args.summary_paths.split(",") if path.strip()],
        output_path=args.output,
    )
    print(f"wrote_report={Path(args.output).resolve()}")


def build_membership_report_command(args: argparse.Namespace) -> None:
    build_membership_inference_report(
        summary_paths=[path.strip() for path in args.summary_paths.split(",") if path.strip()],
        output_path=args.output,
    )
    print(f"wrote_report={Path(args.output).resolve()}")


def build_defense_report_command(args: argparse.Namespace) -> None:
    build_adversarial_defense_report(
        baseline_summary_path=args.baseline_summary,
        defense_summary_paths=[
            path.strip() for path in args.defense_summary_paths.split(",") if path.strip()
        ],
        output_path=args.output,
    )
    print(f"wrote_report={Path(args.output).resolve()}")


def build_physionet_cross_protocol_holdout_plan_command(
    args: argparse.Namespace,
) -> None:
    _run_repo_script("generate_physionet_cross_protocol_holdout_plan.py")


def run_physionet_cross_protocol_holdout_plan_command(
    args: argparse.Namespace,
) -> None:
    extra_args = []
    if args.plan_csv:
        extra_args.extend(["--plan-csv", args.plan_csv])
    if args.candidate:
        extra_args.extend(["--candidate", args.candidate])
    if args.attack_label:
        extra_args.extend(["--attack-label", args.attack_label])
    if args.limit is not None:
        extra_args.extend(["--limit", str(args.limit)])
    if args.dry_run:
        extra_args.append("--dry-run")
    if args.force:
        extra_args.append("--force")
    _run_repo_script("run_physionet_cross_protocol_holdout_plan.py", extra_args)


def summarize_physionet_cross_protocol_holdout_command(
    args: argparse.Namespace,
) -> None:
    _run_repo_script("generate_physionet_cross_protocol_holdout_results.py")


def build_physionet_defense_grid_plan_command(args: argparse.Namespace) -> None:
    extra_args = []
    if args.seeds:
        extra_args.extend(["--seeds", args.seeds])
    if args.task_epochs is not None:
        extra_args.extend(["--task-epochs", str(args.task_epochs)])
    if args.task_batch_size is not None:
        extra_args.extend(["--task-batch-size", str(args.task_batch_size)])
    if args.mne_data_dir:
        extra_args.extend(["--mne-data-dir", args.mne_data_dir])
    _run_repo_script("generate_physionet_defense_grid_plan.py", extra_args)


def run_physionet_defense_grid_plan_command(args: argparse.Namespace) -> None:
    extra_args = []
    if args.plan_csv:
        extra_args.extend(["--plan-csv", args.plan_csv])
    if args.candidate:
        extra_args.extend(["--candidate", args.candidate])
    if args.attack_label:
        extra_args.extend(["--attack-label", args.attack_label])
    if args.limit is not None:
        extra_args.extend(["--limit", str(args.limit)])
    if args.dry_run:
        extra_args.append("--dry-run")
    if args.force:
        extra_args.append("--force")
    _run_repo_script("run_physionet_defense_grid_plan.py", extra_args)


def summarize_physionet_defense_grid_command(args: argparse.Namespace) -> None:
    _run_repo_script("generate_physionet_defense_grid_results.py")


def build_physionet_random_subset_validation_plan_command(
    args: argparse.Namespace,
) -> None:
    extra_args = []
    if args.subset_size is not None:
        extra_args.extend(["--subset-size", str(args.subset_size)])
    if args.subset_count is not None:
        extra_args.extend(["--subset-count", str(args.subset_count)])
    if args.sampling_seed is not None:
        extra_args.extend(["--sampling-seed", str(args.sampling_seed)])
    if args.seeds:
        extra_args.extend(["--seeds", args.seeds])
    if args.task_epochs is not None:
        extra_args.extend(["--task-epochs", str(args.task_epochs)])
    if args.task_batch_size is not None:
        extra_args.extend(["--task-batch-size", str(args.task_batch_size)])
    if args.mne_data_dir:
        extra_args.extend(["--mne-data-dir", args.mne_data_dir])
    _run_repo_script("generate_physionet_random_subset_validation_plan.py", extra_args)


def run_physionet_random_subset_validation_plan_command(
    args: argparse.Namespace,
) -> None:
    extra_args = []
    if args.plan_csv:
        extra_args.extend(["--plan-csv", args.plan_csv])
    if args.tier:
        extra_args.extend(["--tier", args.tier])
    if args.limit is not None:
        extra_args.extend(["--limit", str(args.limit)])
    if args.dry_run:
        extra_args.append("--dry-run")
    if args.force:
        extra_args.append("--force")
    _run_repo_script("run_physionet_random_subset_validation_plan.py", extra_args)


def summarize_physionet_random_subset_validation_command(
    args: argparse.Namespace,
) -> None:
    extra_args = []
    if args.plan_csv:
        extra_args.extend(["--plan-csv", args.plan_csv])
    _run_repo_script("generate_physionet_random_subset_validation_results.py", extra_args)


def build_physionet_mlp_seed_extension_plan_command(
    args: argparse.Namespace,
) -> None:
    extra_args = []
    if args.seeds:
        extra_args.extend(["--seeds", args.seeds])
    if args.task_epochs is not None:
        extra_args.extend(["--task-epochs", str(args.task_epochs)])
    if args.task_batch_size is not None:
        extra_args.extend(["--task-batch-size", str(args.task_batch_size)])
    if args.mne_data_dir:
        extra_args.extend(["--mne-data-dir", args.mne_data_dir])
    _run_repo_script("generate_physionet_mlp_seed_extension_plan.py", extra_args)


def run_physionet_mlp_seed_extension_plan_command(
    args: argparse.Namespace,
) -> None:
    extra_args = []
    if args.plan_csv:
        extra_args.extend(["--plan-csv", args.plan_csv])
    if args.limit is not None:
        extra_args.extend(["--limit", str(args.limit)])
    if args.dry_run:
        extra_args.append("--dry-run")
    if args.force:
        extra_args.append("--force")
    _run_repo_script("run_physionet_mlp_seed_extension_plan.py", extra_args)


def summarize_physionet_mlp_seed_extension_command(
    args: argparse.Namespace,
) -> None:
    _run_repo_script("generate_physionet_mlp_seed_extension_results.py")


def build_bnci_lee_validation_plan_command(args: argparse.Namespace) -> None:
    extra_args = []
    if args.seeds:
        extra_args.extend(["--seeds", args.seeds])
    if args.task_epochs is not None:
        extra_args.extend(["--task-epochs", str(args.task_epochs)])
    if args.task_batch_size is not None:
        extra_args.extend(["--task-batch-size", str(args.task_batch_size)])
    if args.mne_data_dir:
        extra_args.extend(["--mne-data-dir", args.mne_data_dir])
    if args.tiers:
        extra_args.extend(["--tiers", args.tiers])
    _run_repo_script("generate_bnci_lee_validation_plan.py", extra_args)


def run_bnci_lee_validation_plan_command(args: argparse.Namespace) -> None:
    extra_args = []
    if args.plan_csv:
        extra_args.extend(["--plan-csv", args.plan_csv])
    for tier in args.tier or []:
        extra_args.extend(["--tier", tier])
    if args.limit is not None:
        extra_args.extend(["--limit", str(args.limit)])
    if args.dry_run:
        extra_args.append("--dry-run")
    if args.force:
        extra_args.append("--force")
    _run_repo_script("run_bnci_lee_validation_plan.py", extra_args)


def summarize_bnci_lee_validation_command(args: argparse.Namespace) -> None:
    _run_repo_script("generate_bnci_lee_validation_results.py")


def build_attacker_refit_plan_command(args: argparse.Namespace) -> None:
    _run_repo_script("generate_attacker_refit_plan.py")


def run_attacker_refit_plan_command(args: argparse.Namespace) -> None:
    extra_args = []
    if args.plan_csv:
        extra_args.extend(["--plan-csv", args.plan_csv])
    for priority in args.priority or []:
        extra_args.extend(["--priority", priority])
    for dataset in args.dataset or []:
        extra_args.extend(["--dataset", dataset])
    for model in args.model or []:
        extra_args.extend(["--model", model])
    for attack_seed in args.attack_seed or []:
        extra_args.extend(["--attack-seed", str(attack_seed)])
    if args.limit is not None:
        extra_args.extend(["--limit", str(args.limit)])
    if args.dry_run:
        extra_args.append("--dry-run")
    if args.force:
        extra_args.append("--force")
    _run_repo_script("run_attacker_refit_plan.py", extra_args)


def summarize_attacker_refit_command(args: argparse.Namespace) -> None:
    _run_repo_script("generate_attacker_refit_results.py")


def list_experiment_workflows_command(args: argparse.Namespace) -> None:
    statuses = inspect_workflows(REPO_ROOT)
    if args.json:
        print(json.dumps([status.to_dict() for status in statuses], indent=2))
        return

    for status in statuses:
        print(
            f"{status.key}: status={status.status} "
            f"completed={status.completed_rows}/{status.planned_rows} "
            f"plan={status.plan_csv}"
        )
        if status.status_note:
            print(f"  note: {status.status_note}")
        if args.verbose:
            print(f"  build: {status.build_command}")
            print(f"  run: {status.run_command}")
            print(f"  summarize: {status.summarize_command}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EEG privacy benchmark utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser(
        "build-manifest",
        help="Download or load one benchmark dataset and write a trial manifest.",
    )
    manifest_parser.add_argument("--dataset", required=True)
    manifest_parser.add_argument("--output", required=True)
    manifest_parser.add_argument(
        "--subjects",
        default=None,
        help="Optional comma-separated subject ids for smoke tests.",
    )
    manifest_parser.add_argument(
        "--mne-data-dir",
        default=None,
        help="Optional MNE/MOABB download cache directory.",
    )
    manifest_parser.set_defaults(func=build_manifest_command)

    split_parser = subparsers.add_parser(
        "build-splits",
        help="Build deterministic split files from a benchmark dataset manifest.",
    )
    split_parser.add_argument("--dataset", required=True)
    split_parser.add_argument("--output-dir", required=True)
    split_parser.add_argument("--seed", type=int, default=13)
    split_parser.add_argument(
        "--subjects",
        default=None,
        help="Optional comma-separated subject ids for smoke tests.",
    )
    split_parser.add_argument(
        "--mne-data-dir",
        default=None,
        help="Optional MNE/MOABB download cache directory.",
    )
    split_parser.set_defaults(func=build_splits_command)

    cache_parser = subparsers.add_parser(
        "check-dataset-cache",
        help="Inspect whether expected local dataset files are already present.",
    )
    cache_parser.add_argument(
        "--dataset",
        default=None,
        choices=sorted(DATASET_REGISTRY),
        help="Optional dataset key. If omitted, inspect all frozen datasets.",
    )
    cache_parser.add_argument(
        "--subjects",
        default=None,
        help="Optional comma-separated subject ids to inspect.",
    )
    cache_parser.add_argument(
        "--mne-data-dir",
        default="raw_data/mne_data",
        help="MNE/MOABB cache directory to inspect.",
    )
    cache_parser.set_defaults(func=check_dataset_cache_command)

    import_cache_parser = subparsers.add_parser(
        "import-dataset-cache",
        help="Copy expected dataset files from another MNE/MOABB cache into this repo.",
    )
    import_cache_parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_REGISTRY),
    )
    import_cache_parser.add_argument(
        "--source-mne-data-dir",
        required=True,
        help="Source MNE/MOABB cache directory to copy from.",
    )
    import_cache_parser.add_argument(
        "--mne-data-dir",
        default="raw_data/mne_data",
        help="Destination MNE/MOABB cache directory inside the repo.",
    )
    import_cache_parser.add_argument(
        "--subjects",
        default=None,
        help="Optional comma-separated subject ids to import.",
    )
    import_cache_parser.set_defaults(func=import_dataset_cache_command)

    csp_parser = subparsers.add_parser(
        "run-csp-lda",
        help="Run the classical CSP plus LDA baseline on one dataset and protocol.",
    )
    csp_parser.add_argument("--dataset", required=True)
    csp_parser.add_argument(
        "--protocol",
        required=True,
        choices=["cross_subject", "cross_session", "cross_run"],
    )
    csp_parser.add_argument("--seed", type=int, default=13)
    csp_parser.add_argument("--n-components", type=int, default=8)
    csp_parser.add_argument(
        "--subjects",
        default=None,
        help="Optional comma-separated subject ids for smoke tests.",
    )
    csp_parser.add_argument(
        "--mne-data-dir",
        default=None,
        help="Optional MNE/MOABB download cache directory.",
    )
    csp_parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file for saving result metrics.",
    )
    csp_parser.set_defaults(func=run_csp_lda_command)

    eegnet_parser = subparsers.add_parser(
        "run-eegnet",
        help="Run the EEGNet baseline on one dataset and protocol.",
    )
    eegnet_parser.add_argument("--dataset", required=True)
    eegnet_parser.add_argument(
        "--protocol",
        required=True,
        choices=["cross_subject", "cross_session", "cross_run"],
    )
    eegnet_parser.add_argument("--seed", type=int, default=13)
    eegnet_parser.add_argument("--epochs", type=int, default=20)
    eegnet_parser.add_argument("--batch-size", type=int, default=32)
    eegnet_parser.add_argument("--learning-rate", type=float, default=1e-3)
    eegnet_parser.add_argument("--validation-fraction", type=float, default=0.2)
    eegnet_parser.add_argument("--early-stopping-patience", type=int, default=5)
    eegnet_parser.add_argument(
        "--subjects",
        default=None,
        help="Optional comma-separated subject ids for smoke tests.",
    )
    eegnet_parser.add_argument(
        "--mne-data-dir",
        default=None,
        help="Optional MNE/MOABB download cache directory.",
    )
    eegnet_parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file for saving result metrics.",
    )
    eegnet_parser.set_defaults(func=run_eegnet_command)

    federated_eegnet_parser = subparsers.add_parser(
        "run-federated-eegnet",
        help="Run subject-partitioned all-client FedAvg with EEGNet.",
    )
    federated_eegnet_parser.add_argument("--dataset", required=True)
    federated_eegnet_parser.add_argument(
        "--protocol",
        required=True,
        choices=["cross_subject", "cross_session", "cross_run"],
    )
    federated_eegnet_parser.add_argument("--seed", type=int, default=13)
    federated_eegnet_parser.add_argument("--rounds", type=int, default=20)
    federated_eegnet_parser.add_argument("--local-epochs", type=int, default=1)
    federated_eegnet_parser.add_argument("--batch-size", type=int, default=32)
    federated_eegnet_parser.add_argument("--learning-rate", type=float, default=1e-3)
    federated_eegnet_parser.add_argument(
        "--validation-fraction", type=float, default=0.2
    )
    federated_eegnet_parser.add_argument(
        "--early-stopping-patience", type=int, default=5
    )
    federated_eegnet_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional round-level checkpoint path for resumable training.",
    )
    federated_eegnet_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Save a checkpoint after this many completed rounds.",
    )
    federated_eegnet_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing checkpoint and start from round zero.",
    )
    federated_eegnet_parser.add_argument("--resample-hz", type=float, default=None)
    federated_eegnet_parser.add_argument(
        "--normalization",
        choices=["federated_sufficient_statistics", "per_trial_channel"],
        default="federated_sufficient_statistics",
    )
    federated_eegnet_parser.add_argument(
        "--classifier-head",
        choices=["flatten", "global_average"],
        default="flatten",
    )
    federated_eegnet_parser.add_argument("--bottleneck-dim", type=int, default=None)
    federated_eegnet_parser.add_argument(
        "--confidence-penalty-beta", type=float, default=0.0
    )
    federated_eegnet_parser.add_argument(
        "--subjects",
        default=None,
        help="Optional comma-separated subject ids for smoke tests.",
    )
    federated_eegnet_parser.add_argument(
        "--mne-data-dir",
        default=None,
        help="Optional MNE/MOABB download cache directory.",
    )
    federated_eegnet_parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file for saving result metrics.",
    )
    federated_eegnet_parser.set_defaults(func=run_federated_eegnet_command)

    subject_id_parser = subparsers.add_parser(
        "run-subject-id",
        help="Run a linear subject-identification probe on frozen task-model features.",
    )
    subject_id_parser.add_argument("--dataset", required=True)
    subject_id_parser.add_argument(
        "--model",
        required=True,
        choices=[
            "eegnet",
            "federated_eegnet",
            "label_smoothing_eegnet",
            "bottleneck_eegnet",
            "feature_noise_eegnet",
            "mixup_eegnet",
            "confidence_penalty_eegnet",
            "adversarial_eegnet",
            "csp_lda",
        ],
    )
    subject_id_parser.add_argument(
        "--protocol",
        required=True,
        choices=["cross_session"],
    )
    subject_id_parser.add_argument("--seed", type=int, default=13)
    subject_id_parser.add_argument("--task-epochs", type=int, default=20)
    subject_id_parser.add_argument("--federated-rounds", type=int, default=20)
    subject_id_parser.add_argument("--federated-local-epochs", type=int, default=1)
    subject_id_parser.add_argument("--task-batch-size", type=int, default=32)
    subject_id_parser.add_argument("--task-learning-rate", type=float, default=1e-3)
    subject_id_parser.add_argument("--validation-fraction", type=float, default=0.2)
    subject_id_parser.add_argument("--early-stopping-patience", type=int, default=5)
    subject_id_parser.add_argument("--adversarial-weight", type=float, default=0.5)
    subject_id_parser.add_argument(
        "--adversarial-schedule",
        choices=["constant", "linear_ramp"],
        default="constant",
    )
    subject_id_parser.add_argument("--ramp-up-fraction", type=float, default=0.4)
    subject_id_parser.add_argument("--label-smoothing", type=float, default=0.1)
    subject_id_parser.add_argument("--bottleneck-dim", type=int, default=8)
    subject_id_parser.add_argument("--feature-noise-std", type=float, default=0.1)
    subject_id_parser.add_argument("--mixup-alpha", type=float, default=0.2)
    subject_id_parser.add_argument("--confidence-penalty-beta", type=float, default=0.05)
    subject_id_parser.add_argument("--n-components", type=int, default=8)
    subject_id_parser.add_argument(
        "--subjects",
        default=None,
        help="Optional comma-separated subject ids for smoke tests.",
    )
    subject_id_parser.add_argument(
        "--mne-data-dir",
        default=None,
        help="Optional MNE/MOABB download cache directory.",
    )
    subject_id_parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file for saving result metrics.",
    )
    subject_id_parser.set_defaults(func=run_subject_id_command)

    membership_parser = subparsers.add_parser(
        "run-membership-inference",
        help="Run a membership-inference attack on task-model outputs.",
    )
    membership_parser.add_argument("--dataset", required=True)
    membership_parser.add_argument(
        "--model",
        required=True,
        choices=[
            "eegnet",
            "federated_eegnet",
            "label_smoothing_eegnet",
            "bottleneck_eegnet",
            "feature_noise_eegnet",
            "mixup_eegnet",
            "confidence_penalty_eegnet",
            "adversarial_eegnet",
            "csp_lda",
        ],
    )
    membership_parser.add_argument(
        "--protocol",
        required=True,
        choices=["cross_subject", "cross_session", "cross_run"],
    )
    membership_parser.add_argument("--seed", type=int, default=13)
    membership_parser.add_argument("--task-epochs", type=int, default=20)
    membership_parser.add_argument("--federated-rounds", type=int, default=20)
    membership_parser.add_argument("--federated-local-epochs", type=int, default=1)
    membership_parser.add_argument("--task-batch-size", type=int, default=32)
    membership_parser.add_argument("--task-learning-rate", type=float, default=1e-3)
    membership_parser.add_argument("--validation-fraction", type=float, default=0.2)
    membership_parser.add_argument("--early-stopping-patience", type=int, default=5)
    membership_parser.add_argument("--adversarial-weight", type=float, default=0.5)
    membership_parser.add_argument(
        "--adversarial-schedule",
        choices=["constant", "linear_ramp"],
        default="constant",
    )
    membership_parser.add_argument("--ramp-up-fraction", type=float, default=0.4)
    membership_parser.add_argument("--label-smoothing", type=float, default=0.1)
    membership_parser.add_argument("--bottleneck-dim", type=int, default=8)
    membership_parser.add_argument("--feature-noise-std", type=float, default=0.1)
    membership_parser.add_argument("--mixup-alpha", type=float, default=0.2)
    membership_parser.add_argument("--confidence-penalty-beta", type=float, default=0.05)
    membership_parser.add_argument("--n-components", type=int, default=8)
    membership_parser.add_argument(
        "--attack-type",
        choices=["threshold", "logistic_regression", "mlp"],
        default="threshold",
    )
    membership_parser.add_argument(
        "--score-type",
        choices=[
            "label_known_log_probability",
            "max_probability",
            "negative_entropy",
            "posterior_probabilities",
            "posterior_probabilities_plus_true_label",
        ],
        default="label_known_log_probability",
    )
    membership_parser.add_argument(
        "--attack-seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for learned membership-attack splitting and "
            "attacker initialization. Defaults to --seed."
        ),
    )
    membership_parser.add_argument(
        "--subjects",
        default=None,
        help="Optional comma-separated subject ids for smoke tests.",
    )
    membership_parser.add_argument(
        "--mne-data-dir",
        default=None,
        help="Optional MNE/MOABB download cache directory.",
    )
    membership_parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file for saving result metrics.",
    )
    membership_parser.add_argument(
        "--posterior-cache-output",
        default=None,
        help=(
            "Optional .npz path for saving task-model posterior features, labels, "
            "and member/nonmember indices for attacker-only reruns."
        ),
    )
    membership_parser.set_defaults(func=run_membership_inference_command)

    cached_membership_parser = subparsers.add_parser(
        "run-membership-inference-from-cache",
        help="Run a membership-inference attack from cached posterior features.",
    )
    cached_membership_parser.add_argument(
        "--posterior-cache",
        required=True,
        help="Path to a .npz cache written by run-membership-inference.",
    )
    cached_membership_parser.add_argument(
        "--attack-type",
        choices=["threshold", "logistic_regression", "mlp"],
        required=True,
    )
    cached_membership_parser.add_argument(
        "--score-type",
        choices=[
            "label_known_log_probability",
            "max_probability",
            "negative_entropy",
            "posterior_probabilities",
            "posterior_probabilities_plus_true_label",
        ],
        required=True,
    )
    cached_membership_parser.add_argument(
        "--attack-seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for learned membership-attack splitting and "
            "attacker initialization. Defaults to the cached task seed."
        ),
    )
    cached_membership_parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file for saving result metrics.",
    )
    cached_membership_parser.set_defaults(
        func=run_membership_inference_from_cache_command
    )

    sweep_parser = subparsers.add_parser(
        "run-subject-id-sweep",
        help="Run repeated subject-identification experiments across multiple seeds.",
    )
    sweep_parser.add_argument("--dataset", required=True)
    sweep_parser.add_argument(
        "--model",
        required=True,
        choices=[
            "eegnet",
            "label_smoothing_eegnet",
            "bottleneck_eegnet",
            "feature_noise_eegnet",
            "mixup_eegnet",
            "confidence_penalty_eegnet",
            "adversarial_eegnet",
            "csp_lda",
        ],
    )
    sweep_parser.add_argument(
        "--protocol",
        required=True,
        choices=["cross_session"],
    )
    sweep_parser.add_argument("--seeds", required=True)
    sweep_parser.add_argument("--task-epochs", type=int, default=20)
    sweep_parser.add_argument("--task-batch-size", type=int, default=32)
    sweep_parser.add_argument("--task-learning-rate", type=float, default=1e-3)
    sweep_parser.add_argument("--validation-fraction", type=float, default=0.2)
    sweep_parser.add_argument("--early-stopping-patience", type=int, default=5)
    sweep_parser.add_argument("--adversarial-weight", type=float, default=0.5)
    sweep_parser.add_argument(
        "--adversarial-schedule",
        choices=["constant", "linear_ramp"],
        default="constant",
    )
    sweep_parser.add_argument("--ramp-up-fraction", type=float, default=0.4)
    sweep_parser.add_argument("--label-smoothing", type=float, default=0.1)
    sweep_parser.add_argument("--bottleneck-dim", type=int, default=8)
    sweep_parser.add_argument("--feature-noise-std", type=float, default=0.1)
    sweep_parser.add_argument("--mixup-alpha", type=float, default=0.2)
    sweep_parser.add_argument("--confidence-penalty-beta", type=float, default=0.05)
    sweep_parser.add_argument("--n-components", type=int, default=8)
    sweep_parser.add_argument(
        "--subjects",
        default=None,
        help="Optional comma-separated subject ids for smoke tests.",
    )
    sweep_parser.add_argument(
        "--mne-data-dir",
        default=None,
        help="Optional MNE/MOABB download cache directory.",
    )
    sweep_parser.add_argument("--output-dir", required=True)
    sweep_parser.set_defaults(func=run_subject_id_sweep_command)

    membership_sweep_parser = subparsers.add_parser(
        "run-membership-inference-sweep",
        help="Run repeated membership-inference experiments across multiple seeds.",
    )
    membership_sweep_parser.add_argument("--dataset", required=True)
    membership_sweep_parser.add_argument(
        "--model",
        required=True,
        choices=[
            "eegnet",
            "label_smoothing_eegnet",
            "bottleneck_eegnet",
            "feature_noise_eegnet",
            "mixup_eegnet",
            "confidence_penalty_eegnet",
            "adversarial_eegnet",
            "csp_lda",
        ],
    )
    membership_sweep_parser.add_argument(
        "--protocol",
        required=True,
        choices=["cross_subject", "cross_session", "cross_run"],
    )
    membership_sweep_parser.add_argument("--seeds", required=True)
    membership_sweep_parser.add_argument("--task-epochs", type=int, default=20)
    membership_sweep_parser.add_argument("--task-batch-size", type=int, default=32)
    membership_sweep_parser.add_argument("--task-learning-rate", type=float, default=1e-3)
    membership_sweep_parser.add_argument("--validation-fraction", type=float, default=0.2)
    membership_sweep_parser.add_argument("--early-stopping-patience", type=int, default=5)
    membership_sweep_parser.add_argument("--adversarial-weight", type=float, default=0.5)
    membership_sweep_parser.add_argument(
        "--adversarial-schedule",
        choices=["constant", "linear_ramp"],
        default="constant",
    )
    membership_sweep_parser.add_argument("--ramp-up-fraction", type=float, default=0.4)
    membership_sweep_parser.add_argument("--label-smoothing", type=float, default=0.1)
    membership_sweep_parser.add_argument("--bottleneck-dim", type=int, default=8)
    membership_sweep_parser.add_argument("--feature-noise-std", type=float, default=0.1)
    membership_sweep_parser.add_argument("--mixup-alpha", type=float, default=0.2)
    membership_sweep_parser.add_argument("--confidence-penalty-beta", type=float, default=0.05)
    membership_sweep_parser.add_argument("--n-components", type=int, default=8)
    membership_sweep_parser.add_argument(
        "--attack-type",
        choices=["threshold", "logistic_regression", "mlp"],
        default="threshold",
    )
    membership_sweep_parser.add_argument(
        "--score-type",
        choices=[
            "label_known_log_probability",
            "max_probability",
            "negative_entropy",
            "posterior_probabilities",
            "posterior_probabilities_plus_true_label",
        ],
        default="label_known_log_probability",
    )
    membership_sweep_parser.add_argument(
        "--attack-seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for learned membership-attack splitting and "
            "attacker initialization. Defaults to each task seed."
        ),
    )
    membership_sweep_parser.add_argument(
        "--subjects",
        default=None,
        help="Optional comma-separated subject ids for smoke tests.",
    )
    membership_sweep_parser.add_argument(
        "--mne-data-dir",
        default=None,
        help="Optional MNE/MOABB download cache directory.",
    )
    membership_sweep_parser.add_argument(
        "--posterior-cache-dir",
        default=None,
        help=(
            "Optional directory for saving one posterior-feature .npz cache per "
            "task seed during the sweep."
        ),
    )
    membership_sweep_parser.add_argument("--output-dir", required=True)
    membership_sweep_parser.set_defaults(func=run_membership_inference_sweep_command)

    membership_cache_sweep_parser = subparsers.add_parser(
        "run-membership-inference-cache-sweep",
        help="Run repeated membership-inference attacks from posterior-feature caches.",
    )
    membership_cache_sweep_parser.add_argument(
        "--posterior-caches",
        default=None,
        help="Optional comma-separated .npz posterior cache paths.",
    )
    membership_cache_sweep_parser.add_argument(
        "--posterior-cache-dir",
        default=None,
        help="Optional directory containing .npz posterior cache files.",
    )
    membership_cache_sweep_parser.add_argument(
        "--attack-type",
        choices=["threshold", "logistic_regression", "mlp"],
        required=True,
    )
    membership_cache_sweep_parser.add_argument(
        "--score-type",
        choices=[
            "label_known_log_probability",
            "max_probability",
            "negative_entropy",
            "posterior_probabilities",
            "posterior_probabilities_plus_true_label",
        ],
        required=True,
    )
    membership_cache_sweep_parser.add_argument(
        "--attack-seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for learned membership-attack splitting and "
            "attacker initialization. Defaults to each cached task seed."
        ),
    )
    membership_cache_sweep_parser.add_argument("--output-dir", required=True)
    membership_cache_sweep_parser.set_defaults(
        func=run_membership_inference_cache_sweep_command
    )

    defense_sweep_parser = subparsers.add_parser(
        "run-adversarial-weight-sweep",
        help="Run repeated subject-adversarial EEGNet experiments across multiple weights.",
    )
    defense_sweep_parser.add_argument("--dataset", required=True)
    defense_sweep_parser.add_argument(
        "--protocol",
        required=True,
        choices=["cross_session"],
    )
    defense_sweep_parser.add_argument("--seeds", required=True)
    defense_sweep_parser.add_argument("--weights", required=True)
    defense_sweep_parser.add_argument("--task-epochs", type=int, default=20)
    defense_sweep_parser.add_argument("--task-batch-size", type=int, default=32)
    defense_sweep_parser.add_argument("--task-learning-rate", type=float, default=1e-3)
    defense_sweep_parser.add_argument("--validation-fraction", type=float, default=0.2)
    defense_sweep_parser.add_argument("--early-stopping-patience", type=int, default=5)
    defense_sweep_parser.add_argument(
        "--adversarial-schedule",
        choices=["constant", "linear_ramp"],
        default="constant",
    )
    defense_sweep_parser.add_argument("--ramp-up-fraction", type=float, default=0.4)
    defense_sweep_parser.add_argument(
        "--subjects",
        default=None,
        help="Optional comma-separated subject ids for smoke tests.",
    )
    defense_sweep_parser.add_argument(
        "--mne-data-dir",
        default=None,
        help="Optional MNE/MOABB download cache directory.",
    )
    defense_sweep_parser.add_argument("--output-dir", required=True)
    defense_sweep_parser.set_defaults(func=run_adversarial_weight_sweep_command)

    report_parser = subparsers.add_parser(
        "build-subject-id-report",
        help="Build a Markdown comparison report from subject-ID sweep summaries.",
    )
    report_parser.add_argument("--summary-paths", required=True)
    report_parser.add_argument("--output", required=True)
    report_parser.set_defaults(func=build_subject_id_report_command)

    membership_report_parser = subparsers.add_parser(
        "build-membership-report",
        help="Build a Markdown comparison report from membership-inference summaries.",
    )
    membership_report_parser.add_argument("--summary-paths", required=True)
    membership_report_parser.add_argument("--output", required=True)
    membership_report_parser.set_defaults(func=build_membership_report_command)

    defense_report_parser = subparsers.add_parser(
        "build-defense-report",
        help="Build a Markdown report from plain and adversarial EEGNet sweep summaries.",
    )
    defense_report_parser.add_argument("--baseline-summary", required=True)
    defense_report_parser.add_argument("--defense-summary-paths", required=True)
    defense_report_parser.add_argument("--output", required=True)
    defense_report_parser.set_defaults(func=build_defense_report_command)

    holdout_plan_parser = subparsers.add_parser(
        "build-physionet-cross-protocol-holdout-plan",
        help="Generate the frozen 11-row PhysioNet cross-run holdout plan.",
    )
    holdout_plan_parser.set_defaults(
        func=build_physionet_cross_protocol_holdout_plan_command
    )

    holdout_runner_parser = subparsers.add_parser(
        "run-physionet-cross-protocol-holdout-plan",
        help="Run or dry-run rows from the frozen PhysioNet cross-run holdout plan.",
    )
    holdout_runner_parser.add_argument(
        "--plan-csv",
        default=None,
        help="Optional holdout plan CSV. Defaults to the generated localcache plan.",
    )
    holdout_runner_parser.add_argument(
        "--candidate",
        default=None,
        help="Optional candidate_id filter, such as eegnet_baseline or mixup_alpha_0p20.",
    )
    holdout_runner_parser.add_argument(
        "--attack-label",
        choices=["mlp_posterior", "threshold_max_probability", "threshold_label_known"],
        default=None,
        help="Optional attack-label filter.",
    )
    holdout_runner_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of filtered rows to run or print.",
    )
    holdout_runner_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected rows without launching training.",
    )
    holdout_runner_parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun rows even when a summary JSON already exists.",
    )
    holdout_runner_parser.set_defaults(
        func=run_physionet_cross_protocol_holdout_plan_command
    )

    holdout_summary_parser = subparsers.add_parser(
        "summarize-physionet-cross-protocol-holdout",
        help="Summarize completed PhysioNet cross-run holdout rows and deltas.",
    )
    holdout_summary_parser.set_defaults(
        func=summarize_physionet_cross_protocol_holdout_command
    )

    defense_grid_plan_parser = subparsers.add_parser(
        "build-physionet-defense-grid-plan",
        help="Generate the pre-registered PhysioNet non-adversarial defense-grid plan.",
    )
    defense_grid_plan_parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated task seeds for the generated plan.",
    )
    defense_grid_plan_parser.add_argument(
        "--task-epochs",
        type=int,
        default=None,
        help="Optional task-training epoch count for planned neural rows.",
    )
    defense_grid_plan_parser.add_argument(
        "--task-batch-size",
        type=int,
        default=None,
        help="Optional task-training batch size for planned neural rows.",
    )
    defense_grid_plan_parser.add_argument(
        "--mne-data-dir",
        default=None,
        help="Optional MNE/MOABB cache directory for planned commands.",
    )
    defense_grid_plan_parser.set_defaults(
        func=build_physionet_defense_grid_plan_command
    )

    defense_grid_runner_parser = subparsers.add_parser(
        "run-physionet-defense-grid-plan",
        help="Run or dry-run rows from the PhysioNet defense-grid plan.",
    )
    defense_grid_runner_parser.add_argument(
        "--plan-csv",
        default=None,
        help="Optional defense-grid plan CSV. Defaults to the generated localcache plan.",
    )
    defense_grid_runner_parser.add_argument(
        "--candidate",
        default=None,
        help="Optional candidate_id filter, such as mixup_alpha_0p10.",
    )
    defense_grid_runner_parser.add_argument(
        "--attack-label",
        choices=["mlp_posterior", "threshold_max_probability", "threshold_label_known"],
        default=None,
        help="Optional attack-label filter.",
    )
    defense_grid_runner_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of filtered rows to run or print.",
    )
    defense_grid_runner_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected rows without launching training.",
    )
    defense_grid_runner_parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun rows even when a summary JSON already exists.",
    )
    defense_grid_runner_parser.set_defaults(
        func=run_physionet_defense_grid_plan_command
    )

    defense_grid_summary_parser = subparsers.add_parser(
        "summarize-physionet-defense-grid",
        help="Summarize completed PhysioNet defense-grid rows and stop-rule decisions.",
    )
    defense_grid_summary_parser.set_defaults(
        func=summarize_physionet_defense_grid_command
    )

    random_subset_plan_parser = subparsers.add_parser(
        "build-physionet-random-subset-validation-plan",
        help="Generate the deterministic PhysioNet random-subset validation plan.",
    )
    random_subset_plan_parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Optional number of subjects per deterministic subset.",
    )
    random_subset_plan_parser.add_argument(
        "--subset-count",
        type=int,
        default=None,
        help="Optional number of deterministic subsets to generate.",
    )
    random_subset_plan_parser.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="Optional deterministic subject-subset sampling seed.",
    )
    random_subset_plan_parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated task seeds for the generated plan.",
    )
    random_subset_plan_parser.add_argument(
        "--task-epochs",
        type=int,
        default=None,
        help="Optional task-training epoch count for planned neural rows.",
    )
    random_subset_plan_parser.add_argument(
        "--task-batch-size",
        type=int,
        default=None,
        help="Optional task-training batch size for planned neural rows.",
    )
    random_subset_plan_parser.add_argument(
        "--mne-data-dir",
        default=None,
        help="Optional MNE/MOABB cache directory for planned commands.",
    )
    random_subset_plan_parser.set_defaults(
        func=build_physionet_random_subset_validation_plan_command
    )

    random_subset_runner_parser = subparsers.add_parser(
        "run-physionet-random-subset-validation-plan",
        help="Run or dry-run a tier from the PhysioNet random-subset validation plan.",
    )
    random_subset_runner_parser.add_argument(
        "--plan-csv",
        default=None,
        help="Optional random-subset plan CSV. Defaults to the generated localcache plan.",
    )
    random_subset_runner_parser.add_argument(
        "--tier",
        choices=["A", "B", "C"],
        default="A",
        help="Validation tier to run or print.",
    )
    random_subset_runner_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of tier rows to run or print.",
    )
    random_subset_runner_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected rows without launching training.",
    )
    random_subset_runner_parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun rows even when a summary JSON already exists.",
    )
    random_subset_runner_parser.set_defaults(
        func=run_physionet_random_subset_validation_plan_command
    )

    random_subset_summary_parser = subparsers.add_parser(
        "summarize-physionet-random-subset-validation",
        help="Summarize completed PhysioNet random-subset validation rows.",
    )
    random_subset_summary_parser.add_argument(
        "--plan-csv",
        default=None,
        help="Optional random-subset plan CSV. Defaults to the generated localcache plan.",
    )
    random_subset_summary_parser.set_defaults(
        func=summarize_physionet_random_subset_validation_command
    )

    mlp_seed_plan_parser = subparsers.add_parser(
        "build-physionet-mlp-seed-extension-plan",
        help="Generate the frozen PhysioNet MLP-posterior seed-extension plan.",
    )
    mlp_seed_plan_parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated extension seeds for the generated plan.",
    )
    mlp_seed_plan_parser.add_argument(
        "--task-epochs",
        type=int,
        default=None,
        help="Optional task-training epoch count for planned neural rows.",
    )
    mlp_seed_plan_parser.add_argument(
        "--task-batch-size",
        type=int,
        default=None,
        help="Optional task-training batch size for planned neural rows.",
    )
    mlp_seed_plan_parser.add_argument(
        "--mne-data-dir",
        default=None,
        help="Optional MNE/MOABB cache directory for planned commands.",
    )
    mlp_seed_plan_parser.set_defaults(
        func=build_physionet_mlp_seed_extension_plan_command
    )

    mlp_seed_runner_parser = subparsers.add_parser(
        "run-physionet-mlp-seed-extension-plan",
        help="Run or dry-run rows from the PhysioNet MLP seed-extension plan.",
    )
    mlp_seed_runner_parser.add_argument(
        "--plan-csv",
        default=None,
        help="Optional seed-extension plan CSV. Defaults to the generated localcache plan.",
    )
    mlp_seed_runner_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of plan rows to run or print.",
    )
    mlp_seed_runner_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected rows without launching training.",
    )
    mlp_seed_runner_parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun rows even when a summary JSON already exists.",
    )
    mlp_seed_runner_parser.set_defaults(
        func=run_physionet_mlp_seed_extension_plan_command
    )

    mlp_seed_summary_parser = subparsers.add_parser(
        "summarize-physionet-mlp-seed-extension",
        help="Combine existing and extension PhysioNet MLP-posterior seed results.",
    )
    mlp_seed_summary_parser.set_defaults(
        func=summarize_physionet_mlp_seed_extension_command
    )

    bnci_lee_plan_parser = subparsers.add_parser(
        "build-bnci-lee-validation-plan",
        help="Generate the staged BNCI/Lee expanded-subject validation plan.",
    )
    bnci_lee_plan_parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated task seeds for the generated plan.",
    )
    bnci_lee_plan_parser.add_argument(
        "--task-epochs",
        type=int,
        default=None,
        help="Optional task-training epoch count for planned neural rows.",
    )
    bnci_lee_plan_parser.add_argument(
        "--task-batch-size",
        type=int,
        default=None,
        help="Optional task-training batch size for planned neural rows.",
    )
    bnci_lee_plan_parser.add_argument(
        "--mne-data-dir",
        default=None,
        help="Optional MNE/MOABB cache directory for planned commands.",
    )
    bnci_lee_plan_parser.add_argument(
        "--tiers",
        default=None,
        help="Optional comma-separated tiers to include. Defaults to A,B,C.",
    )
    bnci_lee_plan_parser.set_defaults(func=build_bnci_lee_validation_plan_command)

    bnci_lee_runner_parser = subparsers.add_parser(
        "run-bnci-lee-validation-plan",
        help="Run or dry-run tiers from the BNCI/Lee validation plan.",
    )
    bnci_lee_runner_parser.add_argument(
        "--plan-csv",
        default=None,
        help="Optional validation plan CSV. Defaults to the generated localcache plan.",
    )
    bnci_lee_runner_parser.add_argument(
        "--tier",
        action="append",
        choices=["A", "B", "C"],
        help="Tier to run. Can be repeated; defaults to all tiers.",
    )
    bnci_lee_runner_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of selected rows to run or print.",
    )
    bnci_lee_runner_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected rows without launching training.",
    )
    bnci_lee_runner_parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun rows even when a summary JSON already exists.",
    )
    bnci_lee_runner_parser.set_defaults(func=run_bnci_lee_validation_plan_command)

    bnci_lee_summary_parser = subparsers.add_parser(
        "summarize-bnci-lee-validation",
        help="Summarize completed BNCI/Lee validation rows and deltas.",
    )
    bnci_lee_summary_parser.set_defaults(func=summarize_bnci_lee_validation_command)

    attacker_refit_plan_parser = subparsers.add_parser(
        "build-attacker-refit-plan",
        help="Generate the frozen learned-attacker refit calibration plan.",
    )
    attacker_refit_plan_parser.set_defaults(func=build_attacker_refit_plan_command)

    attacker_refit_runner_parser = subparsers.add_parser(
        "run-attacker-refit-plan",
        help="Run or dry-run filtered rows from the attacker-refit plan.",
    )
    attacker_refit_runner_parser.add_argument(
        "--plan-csv",
        default=None,
        help="Optional attacker-refit plan CSV. Defaults to the generated localcache plan.",
    )
    attacker_refit_runner_parser.add_argument(
        "--priority",
        action="append",
        choices=["P1", "P2"],
        help="Priority to run. Can be repeated; defaults to all priorities.",
    )
    attacker_refit_runner_parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset key to run. Can be repeated.",
    )
    attacker_refit_runner_parser.add_argument(
        "--model",
        action="append",
        help="Model key to run. Can be repeated.",
    )
    attacker_refit_runner_parser.add_argument(
        "--attack-seed",
        type=int,
        action="append",
        help="Attacker seed to run. Can be repeated.",
    )
    attacker_refit_runner_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of selected rows to run or print.",
    )
    attacker_refit_runner_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected rows without launching training.",
    )
    attacker_refit_runner_parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun rows even when a summary JSON already exists.",
    )
    attacker_refit_runner_parser.set_defaults(func=run_attacker_refit_plan_command)

    attacker_refit_summary_parser = subparsers.add_parser(
        "summarize-attacker-refit",
        help="Summarize completed learned-attacker refit rows and stability deltas.",
    )
    attacker_refit_summary_parser.set_defaults(func=summarize_attacker_refit_command)

    workflow_list_parser = subparsers.add_parser(
        "list-experiment-workflows",
        help="List experiment workflows and derive completion from existing plans.",
    )
    workflow_list_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print each workflow's plan, run, and summarize commands.",
    )
    workflow_list_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable workflow status JSON.",
    )
    workflow_list_parser.set_defaults(func=list_experiment_workflows_command)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    _configure_runtime_cache_dirs()
    try:
        args.func(args)
    except Exception as exc:
        _handle_cli_exception(exc, args)


if __name__ == "__main__":
    main()
