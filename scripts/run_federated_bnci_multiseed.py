from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_privacy_benchmark.cli import _configure_mne_data_dir
from eeg_privacy_benchmark.models import fit_federated_eegnet
from eeg_privacy_benchmark.privacy import (
    load_posterior_feature_cache,
    run_cached_membership_inference_attack,
    run_federated_eegnet_membership_inference_from_artifacts,
    run_federated_eegnet_subject_id_from_artifacts,
    write_posterior_feature_cache,
)
from eeg_privacy_benchmark.results import write_result_json


DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "federated_bnci_multiseed"
ATTACKS = (
    ("threshold_label_known", "threshold", "label_known_log_probability"),
    ("threshold_max_probability", "threshold", "max_probability"),
    ("logistic_posterior", "logistic_regression", "posterior_probabilities"),
    ("mlp_posterior", "mlp", "posterior_probabilities"),
)


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be a non-empty list of unique integers.")
    return seeds


def _parse_subjects(value: str) -> list[int]:
    subjects = []
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending subject range: {token}")
            subjects.extend(range(start, end + 1))
        else:
            subjects.append(int(token))
    if len(subjects) < 2 or len(subjects) != len(set(subjects)):
        raise ValueError("Subjects must contain at least two unique integers.")
    return subjects


def _seed_paths(output_root: Path, seed: int) -> dict[str, Path]:
    seed_root = output_root / f"seed{seed}"
    paths = {
        "task": seed_root / "federated_eegnet_task.json",
        "subject_id": seed_root / "federated_eegnet_subject_id.json",
        "posterior_cache": seed_root / "federated_eegnet_posteriors.npz",
        "checkpoint": seed_root / "federated_eegnet_checkpoint.pt",
        "suite": seed_root / "federated_eegnet_suite.json",
    }
    for label, _, _ in ATTACKS:
        paths[label] = seed_root / f"federated_eegnet_membership_{label}.json"
    return paths


def _suite_complete(paths: dict[str, Path], *, include_subject_id: bool) -> bool:
    outputs_exist = all(
        path.exists()
        for key, path in paths.items()
        if key != "checkpoint" and (include_subject_id or key != "subject_id")
    )
    if not outputs_exist:
        return False
    try:
        suite = json.loads(paths["suite"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return suite.get("metadata_schema_version") == 2


def _component_outputs_complete(
    paths: dict[str, Path], *, include_subject_id: bool
) -> bool:
    return all(
        path.exists()
        for key, path in paths.items()
        if key not in {"suite", "checkpoint"}
        and (include_subject_id or key != "subject_id")
    )


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _best_checkpoint_round(task: dict) -> int:
    if "best_checkpoint_round" in task:
        return int(task["best_checkpoint_round"])
    return int(min(task["history"], key=lambda row: row["validation_loss"])["round"])


def _upgrade_federated_provenance(
    paths: dict[str, Path], *, include_subject_id: bool
) -> int:
    task = json.loads(paths["task"].read_text(encoding="utf-8"))
    best_round = _best_checkpoint_round(task)
    checkpoint_row = next(
        row for row in task["history"] if int(row["round"]) == best_round
    )
    task["best_checkpoint_round"] = best_round
    task["checkpoint_validation_balanced_accuracy"] = checkpoint_row[
        "validation_balanced_accuracy"
    ]
    write_result_json(task, paths["task"])

    result_paths = [paths[label] for label, _, _ in ATTACKS]
    if include_subject_id:
        result_paths.append(paths["subject_id"])
    for result_path in result_paths:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["federated_best_checkpoint_round"] = best_round
        write_result_json(result, result_path)

    cache = load_posterior_feature_cache(paths["posterior_cache"])
    if cache.federated_best_checkpoint_round != best_round:
        write_posterior_feature_cache(
            replace(cache, federated_best_checkpoint_round=best_round),
            paths["posterior_cache"],
        )
    return best_round


def _write_suite_from_outputs(
    *,
    paths: dict[str, Path],
    dataset_key: str,
    protocol: str,
    subjects: list[int],
    seed: int,
    include_subject_id: bool,
    task_model: str = "federated_eegnet",
) -> None:
    best_checkpoint_round = _upgrade_federated_provenance(
        paths,
        include_subject_id=include_subject_id,
    )
    task = json.loads(paths["task"].read_text(encoding="utf-8"))
    attacks = {
        label: json.loads(paths[label].read_text(encoding="utf-8"))
        for label, _, _ in ATTACKS
    }
    suite = {
        "metadata_schema_version": 2,
        "task_model": task_model,
        "dataset_key": dataset_key,
        "protocol": protocol,
        "subjects": subjects,
        "seed": seed,
        "rounds": task["rounds_trained"],
        "best_checkpoint_round": best_checkpoint_round,
        "local_epochs": task["local_epochs"],
        "task_balanced_accuracy": task["balanced_accuracy"],
        "task_macro_f1": task["macro_f1"],
        "subject_id_accuracy": None,
        "estimated_communication_bytes": task["estimated_communication_bytes"],
        "resample_hz": task.get("resample_hz"),
        "normalization": task.get(
            "normalization", "federated_sufficient_statistics"
        ),
        "classifier_head": task.get("classifier_head", "flatten"),
        "bottleneck_dim": task.get("bottleneck_dim"),
        "confidence_penalty_beta": task.get("confidence_penalty_beta"),
        "membership_auc": {
            label: result["attack_auc"] for label, result in attacks.items()
        },
        "outputs": {
            key: _display_path(path)
            for key, path in paths.items()
            if include_subject_id or key != "subject_id"
        },
    }
    if include_subject_id:
        subject = json.loads(paths["subject_id"].read_text(encoding="utf-8"))
        suite["subject_id_accuracy"] = subject["subject_id_accuracy"]
    write_result_json(suite, paths["suite"])


def _run_seed(
    *,
    dataset_key: str,
    protocol: str,
    subjects: list[int],
    seed: int,
    paths: dict[str, Path],
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_fraction: float,
    early_stopping_patience: int,
    include_subject_id: bool,
    task_model: str,
    resample_hz: float | None,
    normalization: str,
    classifier_head: str,
    bottleneck_dim: int | None,
    confidence_penalty_beta: float,
) -> None:
    print(f"seed={seed} status=training")
    artifacts = fit_federated_eegnet(
        dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        checkpoint_path=paths["checkpoint"],
        checkpoint_interval=1,
        resume_from_checkpoint=True,
        resample_hz=resample_hz,
        normalization=normalization,
        classifier_head=classifier_head,
        bottleneck_dim=bottleneck_dim,
        confidence_penalty_beta=confidence_penalty_beta,
    )
    task_result = artifacts.result.to_dict()
    task_result["task_model"] = task_model
    write_result_json(task_result, paths["task"])

    first_label, first_attack_type, first_score_type = ATTACKS[0]
    first_attack = run_federated_eegnet_membership_inference_from_artifacts(
        artifacts,
        subjects=subjects,
        task_model=task_model,
        attack_type=first_attack_type,
        score_type=first_score_type,
        posterior_cache_output=paths["posterior_cache"],
    )
    write_result_json(first_attack.to_dict(), paths[first_label])

    attack_results = {first_label: first_attack}
    for label, attack_type, score_type in ATTACKS[1:]:
        result = run_cached_membership_inference_attack(
            paths["posterior_cache"],
            attack_type=attack_type,
            score_type=score_type,
        )
        attack_results[label] = result
        write_result_json(result.to_dict(), paths[label])

    subject_result = None
    if include_subject_id:
        subject_result = run_federated_eegnet_subject_id_from_artifacts(
            artifacts,
            task_model=task_model,
        )
        write_result_json(subject_result.to_dict(), paths["subject_id"])
    suite = {
        "metadata_schema_version": 2,
        "task_model": task_model,
        "dataset_key": artifacts.dataset_key,
        "protocol": artifacts.protocol,
        "subjects": subjects,
        "seed": seed,
        "rounds": artifacts.result.rounds_trained,
        "best_checkpoint_round": artifacts.result.best_checkpoint_round,
        "local_epochs": artifacts.result.local_epochs,
        "task_balanced_accuracy": artifacts.result.balanced_accuracy,
        "task_macro_f1": artifacts.result.macro_f1,
        "subject_id_accuracy": (
            subject_result.subject_id_accuracy if subject_result is not None else None
        ),
        "estimated_communication_bytes": (
            artifacts.result.estimated_communication_bytes
        ),
        "resample_hz": artifacts.result.resample_hz,
        "normalization": artifacts.result.normalization,
        "classifier_head": artifacts.result.classifier_head,
        "bottleneck_dim": artifacts.result.bottleneck_dim,
        "confidence_penalty_beta": artifacts.result.confidence_penalty_beta,
        "membership_auc": {
            label: result.attack_auc for label, result in attack_results.items()
        },
        "outputs": {
            key: _display_path(path)
            for key, path in paths.items()
            if include_subject_id or key != "subject_id"
        },
    }
    write_result_json(suite, paths["suite"])
    status = (
        f"seed={seed} status=complete "
        f"task_balanced_accuracy={artifacts.result.balanced_accuracy:.6f}"
    )
    if subject_result is not None:
        status += f" subject_id_accuracy={subject_result.subject_id_accuracy:.6f}"
    print(status)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a calibrated federated EEGNet suite once per task seed."
    )
    parser.add_argument("--dataset", default="bnci2014_001")
    parser.add_argument("--protocol", default="cross_session")
    parser.add_argument("--subjects", default="1,2,3,4")
    parser.add_argument("--seeds", default="17,19,23")
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=80)
    parser.add_argument("--mne-data-dir", default="raw_data/mne_data")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--task-model", default="federated_eegnet")
    parser.add_argument("--resample-hz", type=float, default=None)
    parser.add_argument(
        "--normalization",
        choices=["federated_sufficient_statistics", "per_trial_channel"],
        default="federated_sufficient_statistics",
    )
    parser.add_argument(
        "--classifier-head",
        choices=["flatten", "global_average"],
        default="flatten",
    )
    parser.add_argument("--bottleneck-dim", type=int, default=None)
    parser.add_argument("--confidence-penalty-beta", type=float, default=0.0)
    parser.add_argument(
        "--subject-id",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Run the subject-identification probe. By default it runs only for "
            "cross-session protocols."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    subjects = _parse_subjects(args.subjects)
    include_subject_id = (
        args.protocol == "cross_session" if args.subject_id is None else args.subject_id
    )
    _configure_mne_data_dir(args.mne_data_dir, args.dataset)
    for seed in _parse_seeds(args.seeds):
        paths = _seed_paths(args.output_root, seed)
        if _suite_complete(paths, include_subject_id=include_subject_id) and not args.force:
            print(f"seed={seed} status=skipped_complete")
            continue
        if (
            _component_outputs_complete(paths, include_subject_id=include_subject_id)
            and not args.force
        ):
            _write_suite_from_outputs(
                paths=paths,
                dataset_key=args.dataset,
                protocol=args.protocol,
                subjects=subjects,
                seed=seed,
                include_subject_id=include_subject_id,
                task_model=args.task_model,
            )
            print(f"seed={seed} status=recovered_suite")
            continue
        if args.dry_run:
            print(f"seed={seed} status=planned output={paths['suite']}")
            continue
        _run_seed(
            dataset_key=args.dataset,
            protocol=args.protocol,
            subjects=subjects,
            seed=seed,
            paths=paths,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_fraction=args.validation_fraction,
            early_stopping_patience=args.early_stopping_patience,
            include_subject_id=include_subject_id,
            task_model=args.task_model,
            resample_hz=args.resample_hz,
            normalization=args.normalization,
            classifier_head=args.classifier_head,
            bottleneck_dim=args.bottleneck_dim,
            confidence_penalty_beta=args.confidence_penalty_beta,
        )


if __name__ == "__main__":
    main()
