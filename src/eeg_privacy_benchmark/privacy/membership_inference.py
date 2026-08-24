"""Membership-inference attacks on trained task models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from eeg_privacy_benchmark.models import (
    fit_bottleneck_eegnet,
    fit_confidence_penalty_eegnet,
    fit_csp_lda,
    fit_feature_noise_eegnet,
    fit_label_smoothing_eegnet,
    fit_mixup_eegnet,
)
from eeg_privacy_benchmark.models.eegnet import (
    batched_eegnet_logits,
    fit_eegnet,
)
from eeg_privacy_benchmark.models.eegnet_adversarial import fit_adversarial_eegnet
from eeg_privacy_benchmark.models.eegnet_federated import fit_federated_eegnet


def _import_attack_dependencies():
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score,
            balanced_accuracy_score,
            roc_auc_score,
        )
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "Membership-inference attack requires numpy and scikit-learn."
        ) from exc
    return {
        "np": np,
        "LogisticRegression": LogisticRegression,
        "MLPClassifier": MLPClassifier,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "average_precision_score": average_precision_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "roc_auc_score": roc_auc_score,
    }


@dataclass(frozen=True)
class MembershipInferenceResult:
    task_model: str
    dataset_key: str
    protocol: str
    seed: int
    attack_seed: int
    attack_type: str
    score_type: str
    task_balanced_accuracy: float
    task_macro_f1: float | None
    member_trials: int
    nonmember_trials: int
    attack_auc: float
    attack_average_precision: float
    attack_balanced_accuracy: float
    member_score_mean: float
    nonmember_score_mean: float
    best_threshold: float
    attack_seed_policy: str | None = None
    epochs_trained: int | None = None
    best_checkpoint_epoch: int | None = None
    best_validation_loss: float | None = None
    resample_hz: float | None = None
    normalization: str | None = None
    classifier_head: str | None = None
    adversarial_weight: float | None = None
    adversarial_schedule: str | None = None
    ramp_up_fraction: float | None = None
    label_smoothing: float | None = None
    bottleneck_dim: int | None = None
    feature_noise_std: float | None = None
    mixup_alpha: float | None = None
    confidence_penalty_beta: float | None = None
    federated_rounds_trained: int | None = None
    federated_best_checkpoint_round: int | None = None
    federated_local_epochs: int | None = None
    federated_clients: int | None = None
    estimated_communication_bytes: int | None = None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class PosteriorFeatureCache:
    task_model: str
    dataset_key: str
    protocol: str
    seed: int
    subjects: list[int] | None
    task_balanced_accuracy: float
    task_macro_f1: float | None
    class_probabilities: object
    encoded_labels: object
    member_indices: list[int]
    nonmember_indices: list[int]
    epochs_trained: int | None = None
    best_checkpoint_epoch: int | None = None
    best_validation_loss: float | None = None
    resample_hz: float | None = None
    normalization: str | None = None
    classifier_head: str | None = None
    adversarial_weight: float | None = None
    adversarial_schedule: str | None = None
    ramp_up_fraction: float | None = None
    label_smoothing: float | None = None
    bottleneck_dim: int | None = None
    feature_noise_std: float | None = None
    mixup_alpha: float | None = None
    confidence_penalty_beta: float | None = None
    federated_rounds_trained: int | None = None
    federated_best_checkpoint_round: int | None = None
    federated_local_epochs: int | None = None
    federated_clients: int | None = None
    estimated_communication_bytes: int | None = None


def _posterior_cache_metadata(cache: PosteriorFeatureCache) -> dict:
    metadata = asdict(cache)
    metadata.pop("class_probabilities")
    metadata.pop("encoded_labels")
    metadata.pop("member_indices")
    metadata.pop("nonmember_indices")
    metadata["cache_schema_version"] = 1
    return metadata


def _validate_posterior_cache_metadata(metadata: dict) -> None:
    deps = _import_attack_dependencies()
    np = deps["np"]

    def _is_finite_number(value) -> bool:
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False

    required_strings = ["task_model", "dataset_key", "protocol"]
    for field in required_strings:
        value = metadata.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Posterior cache metadata {field} must be a non-empty string.")
    seed = metadata.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Posterior cache metadata seed must be an integer.")
    subjects = metadata.get("subjects")
    if subjects is not None:
        if not isinstance(subjects, list) or not all(
            isinstance(subject, int) and not isinstance(subject, bool)
            for subject in subjects
        ):
            raise ValueError(
                "Posterior cache metadata subjects must be null or a list of integers."
            )
    for field in ["task_balanced_accuracy", "task_macro_f1"]:
        value = metadata.get(field)
        if value is not None and not _is_finite_number(value):
            raise ValueError(f"Posterior cache metadata {field} must be finite.")
    optional_integers = [
        "epochs_trained",
        "best_checkpoint_epoch",
        "bottleneck_dim",
        "federated_rounds_trained",
        "federated_best_checkpoint_round",
        "federated_local_epochs",
        "federated_clients",
        "estimated_communication_bytes",
    ]
    for field in optional_integers:
        value = metadata.get(field)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise ValueError(
                f"Posterior cache metadata {field} must be null or an integer."
            )
    optional_numbers = [
        "best_validation_loss",
        "resample_hz",
        "adversarial_weight",
        "ramp_up_fraction",
        "label_smoothing",
        "feature_noise_std",
        "mixup_alpha",
        "confidence_penalty_beta",
    ]
    for field in optional_numbers:
        value = metadata.get(field)
        if value is not None and not _is_finite_number(value):
            raise ValueError(
                f"Posterior cache metadata {field} must be null or finite."
            )
    schedule = metadata.get("adversarial_schedule")
    if schedule is not None and (not isinstance(schedule, str) or not schedule):
        raise ValueError(
            "Posterior cache metadata adversarial_schedule must be null or a non-empty string."
        )
    for field in ["normalization", "classifier_head"]:
        value = metadata.get(field)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(
                f"Posterior cache metadata {field} must be null or a non-empty string."
            )


def _validate_posterior_feature_cache_arrays(
    class_probabilities,
    encoded_labels,
    member_indices,
    nonmember_indices,
) -> None:
    deps = _import_attack_dependencies()
    np = deps["np"]
    probabilities = np.asarray(class_probabilities)
    labels = np.asarray(encoded_labels)
    member = np.asarray(member_indices, dtype=int)
    nonmember = np.asarray(nonmember_indices, dtype=int)

    if probabilities.ndim != 2:
        raise ValueError("Posterior cache class_probabilities must be a 2D array.")
    if probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError("Posterior cache class_probabilities must be non-empty.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Posterior cache class_probabilities must be finite.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(
            "Posterior cache class_probabilities must be in the [0, 1] range."
        )
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(
            "Posterior cache class_probabilities rows must sum to one."
        )
    if labels.ndim != 1:
        raise ValueError("Posterior cache encoded_labels must be a 1D array.")
    if labels.shape[0] != probabilities.shape[0]:
        raise ValueError(
            "Posterior cache encoded_labels length must match probability rows."
        )
    if member.ndim != 1 or nonmember.ndim != 1:
        raise ValueError("Posterior cache member/nonmember indices must be 1D arrays.")
    if member.size == 0 or nonmember.size == 0:
        raise ValueError("Posterior cache member/nonmember indices must be non-empty.")
    all_indices = np.concatenate([member, nonmember])
    if np.any(all_indices < 0) or np.any(all_indices >= probabilities.shape[0]):
        raise ValueError(
            "Posterior cache member/nonmember indices must be within probability rows."
        )
    if len(set(all_indices.tolist())) != all_indices.size:
        raise ValueError("Posterior cache member/nonmember indices must be unique.")
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError(
            "Posterior cache encoded_labels must be valid class indices."
        )


def write_posterior_feature_cache(
    cache: PosteriorFeatureCache,
    path: str | Path,
) -> Path:
    """Write task-model posteriors and split indices for attacker-only reruns."""

    deps = _import_attack_dependencies()
    np = deps["np"]
    class_probabilities = np.asarray(cache.class_probabilities)
    encoded_labels = np.asarray(cache.encoded_labels)
    member_indices = np.asarray(cache.member_indices, dtype=int)
    nonmember_indices = np.asarray(cache.nonmember_indices, dtype=int)
    _validate_posterior_cache_metadata(_posterior_cache_metadata(cache))
    _validate_posterior_feature_cache_arrays(
        class_probabilities,
        encoded_labels,
        member_indices,
        nonmember_indices,
    )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        metadata=json.dumps(_posterior_cache_metadata(cache), sort_keys=True),
        class_probabilities=class_probabilities,
        encoded_labels=encoded_labels,
        member_indices=member_indices,
        nonmember_indices=nonmember_indices,
    )
    return out_path


def load_posterior_feature_cache(path: str | Path) -> PosteriorFeatureCache:
    """Load a posterior-feature cache written by write_posterior_feature_cache."""

    deps = _import_attack_dependencies()
    np = deps["np"]
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"]))
        version = metadata.pop("cache_schema_version", None)
        if version != 1:
            raise ValueError(f"Unsupported posterior cache schema version: {version}")
        _validate_posterior_cache_metadata(metadata)
        class_probabilities = payload["class_probabilities"].copy()
        encoded_labels = payload["encoded_labels"].copy()
        member_indices = payload["member_indices"].astype(int)
        nonmember_indices = payload["nonmember_indices"].astype(int)
        _validate_posterior_feature_cache_arrays(
            class_probabilities,
            encoded_labels,
            member_indices,
            nonmember_indices,
        )
        return PosteriorFeatureCache(
            **metadata,
            class_probabilities=class_probabilities,
            encoded_labels=encoded_labels,
            member_indices=member_indices.tolist(),
            nonmember_indices=nonmember_indices.tolist(),
        )


def _run_membership_attack_from_cache(
    cache: PosteriorFeatureCache,
    *,
    attack_type: str,
    score_type: str,
    attack_seed: int | None = None,
) -> MembershipInferenceResult:
    attack_kwargs = dict(
        task_model=cache.task_model,
        dataset_key=cache.dataset_key,
        protocol=cache.protocol,
        seed=cache.seed,
        attack_seed=attack_seed,
        task_balanced_accuracy=cache.task_balanced_accuracy,
        task_macro_f1=cache.task_macro_f1,
        class_probabilities=cache.class_probabilities,
        encoded_labels=cache.encoded_labels,
        member_indices=cache.member_indices,
        nonmember_indices=cache.nonmember_indices,
        score_type=score_type,
        epochs_trained=cache.epochs_trained,
        best_validation_loss=cache.best_validation_loss,
        adversarial_weight=cache.adversarial_weight,
        adversarial_schedule=cache.adversarial_schedule,
        ramp_up_fraction=cache.ramp_up_fraction,
        label_smoothing=cache.label_smoothing,
        bottleneck_dim=cache.bottleneck_dim,
        feature_noise_std=cache.feature_noise_std,
        mixup_alpha=cache.mixup_alpha,
        confidence_penalty_beta=cache.confidence_penalty_beta,
    )
    if attack_type == "threshold":
        result = _run_threshold_membership_attack(**attack_kwargs)
    else:
        result = _run_learned_membership_attack(
            attack_type=attack_type,
            **attack_kwargs,
        )
    return replace(
        result,
        best_checkpoint_epoch=cache.best_checkpoint_epoch,
        resample_hz=cache.resample_hz,
        normalization=cache.normalization,
        classifier_head=cache.classifier_head,
        federated_rounds_trained=cache.federated_rounds_trained,
        federated_best_checkpoint_round=cache.federated_best_checkpoint_round,
        federated_local_epochs=cache.federated_local_epochs,
        federated_clients=cache.federated_clients,
        estimated_communication_bytes=cache.estimated_communication_bytes,
    )


def run_cached_membership_inference_attack(
    cache_path: str | Path,
    *,
    attack_type: str,
    score_type: str,
    attack_seed: int | None = None,
) -> MembershipInferenceResult:
    """Run a membership-inference attack from cached task-model posterior features."""

    cache = load_posterior_feature_cache(cache_path)
    return _run_membership_attack_from_cache(
        cache,
        attack_type=attack_type,
        score_type=score_type,
        attack_seed=attack_seed,
    )


def _save_cache_if_requested(
    cache_output: str | Path | None,
    cache: PosteriorFeatureCache,
) -> None:
    if cache_output is not None:
        write_posterior_feature_cache(cache, cache_output)


def _label_known_log_probability_scores(probabilities, encoded_labels):
    deps = _import_attack_dependencies()
    np = deps["np"]
    clipped = np.clip(
        probabilities[np.arange(len(encoded_labels)), encoded_labels],
        1e-8,
        1.0,
    )
    return np.log(clipped)


def _max_probability_scores(probabilities, encoded_labels):
    del encoded_labels
    deps = _import_attack_dependencies()
    np = deps["np"]
    return np.max(probabilities, axis=1)


def _negative_entropy_scores(probabilities, encoded_labels):
    del encoded_labels
    deps = _import_attack_dependencies()
    np = deps["np"]
    clipped = np.clip(probabilities, 1e-8, 1.0)
    return np.sum(clipped * np.log(clipped), axis=1)


def _membership_scores(probabilities, encoded_labels, score_type: str):
    score_builders = {
        "label_known_log_probability": _label_known_log_probability_scores,
        "max_probability": _max_probability_scores,
        "negative_entropy": _negative_entropy_scores,
    }
    try:
        builder = score_builders[score_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported membership score type: {score_type}") from exc
    return builder(probabilities, encoded_labels)


def _validate_attack_configuration(attack_type: str, score_type: str) -> None:
    if attack_type == "threshold":
        if score_type in {
            "posterior_probabilities",
            "posterior_probabilities_plus_true_label",
        }:
            raise ValueError(
                f"score_type={score_type} requires attack_type=logistic_regression."
            )
        return
    if attack_type in {"logistic_regression", "mlp"}:
        if score_type not in {
            "posterior_probabilities",
            "posterior_probabilities_plus_true_label",
        }:
            raise ValueError(
                f"attack_type={attack_type} currently supports only "
                "score_type=posterior_probabilities or "
                "score_type=posterior_probabilities_plus_true_label."
            )
        return
    raise ValueError(f"Unsupported membership attack type: {attack_type}")


def _membership_attack_features(probabilities, encoded_labels, score_type: str):
    deps = _import_attack_dependencies()
    np = deps["np"]
    if score_type == "posterior_probabilities":
        return probabilities
    if score_type == "posterior_probabilities_plus_true_label":
        n_classes = probabilities.shape[1]
        true_label_one_hot = np.eye(n_classes, dtype=probabilities.dtype)[encoded_labels]
        return np.concatenate([probabilities, true_label_one_hot], axis=1)
    raise ValueError(f"Unsupported learned membership score type: {score_type}")


def _split_attack_indices(np, indices: list[int], seed: int) -> tuple[list[int], list[int]]:
    shuffled = list(indices)
    np.random.default_rng(seed).shuffle(shuffled)
    split_point = max(1, int(round(len(shuffled) * 0.5)))
    split_point = min(split_point, len(shuffled) - 1)
    return sorted(shuffled[:split_point]), sorted(shuffled[split_point:])


def _best_balanced_accuracy_threshold(scores, membership_labels) -> tuple[float, float]:
    deps = _import_attack_dependencies()
    np = deps["np"]
    balanced_accuracy_score = deps["balanced_accuracy_score"]
    candidate_thresholds = np.unique(scores)

    best_threshold = float(candidate_thresholds[0])
    best_accuracy = 0.0
    for threshold in candidate_thresholds:
        predictions = (scores >= threshold).astype(int)
        accuracy = float(balanced_accuracy_score(membership_labels, predictions))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_accuracy, best_threshold


def _run_threshold_membership_attack(
    *,
    task_model: str,
    dataset_key: str,
    protocol: str,
    seed: int,
    attack_seed: int | None,
    task_balanced_accuracy: float,
    task_macro_f1: float | None,
    class_probabilities,
    encoded_labels,
    member_indices: list[int],
    nonmember_indices: list[int],
    score_type: str = "label_known_log_probability",
    epochs_trained: int | None = None,
    best_validation_loss: float | None = None,
    adversarial_weight: float | None = None,
    adversarial_schedule: str | None = None,
    ramp_up_fraction: float | None = None,
    label_smoothing: float | None = None,
    bottleneck_dim: int | None = None,
    feature_noise_std: float | None = None,
    mixup_alpha: float | None = None,
    confidence_penalty_beta: float | None = None,
) -> MembershipInferenceResult:
    _validate_attack_configuration("threshold", score_type)
    deps = _import_attack_dependencies()
    np = deps["np"]
    average_precision_score = deps["average_precision_score"]
    roc_auc_score = deps["roc_auc_score"]

    scores = _membership_scores(
        class_probabilities,
        encoded_labels,
        score_type=score_type,
    )
    membership_labels = np.zeros(len(scores), dtype=int)
    membership_labels[member_indices] = 1

    evaluation_indices = member_indices + nonmember_indices
    evaluation_scores = scores[evaluation_indices]
    evaluation_labels = membership_labels[evaluation_indices]

    attack_balanced_accuracy, best_threshold = _best_balanced_accuracy_threshold(
        evaluation_scores,
        evaluation_labels,
    )

    return MembershipInferenceResult(
        task_model=task_model,
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        attack_seed=seed if attack_seed is None else attack_seed,
        attack_type="threshold",
        score_type=score_type,
        task_balanced_accuracy=task_balanced_accuracy,
        task_macro_f1=task_macro_f1,
        member_trials=len(member_indices),
        nonmember_trials=len(nonmember_indices),
        attack_auc=float(roc_auc_score(evaluation_labels, evaluation_scores)),
        attack_average_precision=float(
            average_precision_score(evaluation_labels, evaluation_scores)
        ),
        attack_balanced_accuracy=attack_balanced_accuracy,
        member_score_mean=float(scores[member_indices].mean()),
        nonmember_score_mean=float(scores[nonmember_indices].mean()),
        best_threshold=best_threshold,
        attack_seed_policy=(
            "task_seed_default" if attack_seed is None else "explicit_fixed_seed"
        ),
        epochs_trained=epochs_trained,
        best_validation_loss=best_validation_loss,
        adversarial_weight=adversarial_weight,
        adversarial_schedule=adversarial_schedule,
        ramp_up_fraction=ramp_up_fraction,
        label_smoothing=label_smoothing,
        bottleneck_dim=bottleneck_dim,
        feature_noise_std=feature_noise_std,
        mixup_alpha=mixup_alpha,
        confidence_penalty_beta=confidence_penalty_beta,
    )


def _run_learned_membership_attack(
    *,
    attack_type: str,
    task_model: str,
    dataset_key: str,
    protocol: str,
    seed: int,
    attack_seed: int | None,
    task_balanced_accuracy: float,
    task_macro_f1: float | None,
    class_probabilities,
    encoded_labels,
    member_indices: list[int],
    nonmember_indices: list[int],
    score_type: str = "posterior_probabilities",
    epochs_trained: int | None = None,
    best_validation_loss: float | None = None,
    adversarial_weight: float | None = None,
    adversarial_schedule: str | None = None,
    ramp_up_fraction: float | None = None,
    label_smoothing: float | None = None,
    bottleneck_dim: int | None = None,
    feature_noise_std: float | None = None,
    mixup_alpha: float | None = None,
    confidence_penalty_beta: float | None = None,
) -> MembershipInferenceResult:
    _validate_attack_configuration(attack_type, score_type)
    deps = _import_attack_dependencies()
    np = deps["np"]
    LogisticRegression = deps["LogisticRegression"]
    MLPClassifier = deps["MLPClassifier"]
    Pipeline = deps["Pipeline"]
    StandardScaler = deps["StandardScaler"]
    average_precision_score = deps["average_precision_score"]
    roc_auc_score = deps["roc_auc_score"]
    effective_attack_seed = seed if attack_seed is None else attack_seed

    member_attack_train, member_attack_eval = _split_attack_indices(
        np,
        member_indices,
        effective_attack_seed,
    )
    nonmember_attack_train, nonmember_attack_eval = _split_attack_indices(
        np,
        nonmember_indices,
        effective_attack_seed + 1,
    )

    attack_train_indices = member_attack_train + nonmember_attack_train
    attack_eval_indices = member_attack_eval + nonmember_attack_eval
    attack_train_labels = np.zeros(len(attack_train_indices), dtype=int)
    attack_train_labels[: len(member_attack_train)] = 1
    attack_eval_labels = np.zeros(len(attack_eval_indices), dtype=int)
    attack_eval_labels[: len(member_attack_eval)] = 1

    attack_features = _membership_attack_features(
        class_probabilities,
        encoded_labels,
        score_type,
    )
    attack_train_features = attack_features[attack_train_indices]
    attack_eval_features = attack_features[attack_eval_indices]
    if attack_type == "logistic_regression":
        attack_model = LogisticRegression(
            random_state=effective_attack_seed,
            max_iter=1000,
        )
    elif attack_type == "mlp":
        attack_model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(32,),
                        activation="relu",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=500,
                        early_stopping=True,
                        n_iter_no_change=20,
                        random_state=effective_attack_seed,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported learned membership attack type: {attack_type}")
    attack_model.fit(attack_train_features, attack_train_labels)
    attack_scores = attack_model.predict_proba(attack_eval_features)[:, 1]
    attack_balanced_accuracy, best_threshold = _best_balanced_accuracy_threshold(
        attack_scores,
        attack_eval_labels,
    )

    return MembershipInferenceResult(
        task_model=task_model,
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        attack_seed=effective_attack_seed,
        attack_type=attack_type,
        score_type=score_type,
        task_balanced_accuracy=task_balanced_accuracy,
        task_macro_f1=task_macro_f1,
        member_trials=len(member_attack_eval),
        nonmember_trials=len(nonmember_attack_eval),
        attack_auc=float(roc_auc_score(attack_eval_labels, attack_scores)),
        attack_average_precision=float(
            average_precision_score(attack_eval_labels, attack_scores)
        ),
        attack_balanced_accuracy=attack_balanced_accuracy,
        member_score_mean=float(attack_scores[: len(member_attack_eval)].mean()),
        nonmember_score_mean=float(attack_scores[len(member_attack_eval) :].mean()),
        best_threshold=best_threshold,
        attack_seed_policy=(
            "task_seed_default" if attack_seed is None else "explicit_fixed_seed"
        ),
        epochs_trained=epochs_trained,
        best_validation_loss=best_validation_loss,
        adversarial_weight=adversarial_weight,
        adversarial_schedule=adversarial_schedule,
        ramp_up_fraction=ramp_up_fraction,
        label_smoothing=label_smoothing,
        bottleneck_dim=bottleneck_dim,
        feature_noise_std=feature_noise_std,
        mixup_alpha=mixup_alpha,
        confidence_penalty_beta=confidence_penalty_beta,
    )


def run_eegnet_membership_inference_attack(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    task_epochs: int = 20,
    task_batch_size: int = 32,
    task_learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    attack_type: str = "threshold",
    score_type: str = "label_known_log_probability",
    attack_seed: int | None = None,
    posterior_cache_output: str | Path | None = None,
) -> MembershipInferenceResult:
    """Run a scalar-score membership-inference attack on EEGNet."""

    artifacts = fit_eegnet(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        epochs=task_epochs,
        batch_size=task_batch_size,
        learning_rate=task_learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
    )

    return run_eegnet_membership_inference_from_artifacts(
        artifacts,
        subjects=subjects,
        attack_type=attack_type,
        score_type=score_type,
        attack_seed=attack_seed,
        posterior_cache_output=posterior_cache_output,
    )


def run_eegnet_membership_inference_from_artifacts(
    artifacts,
    *,
    subjects: list[int] | None = None,
    task_model: str = "eegnet",
    attack_type: str = "threshold",
    score_type: str = "label_known_log_probability",
    attack_seed: int | None = None,
    posterior_cache_output: str | Path | None = None,
) -> MembershipInferenceResult:
    """Attack an already-fitted centralized EEGNet without retraining it."""

    logits = batched_eegnet_logits(artifacts)
    probabilities = logits.softmax(dim=1).numpy()
    result = artifacts.result
    cache = PosteriorFeatureCache(
        task_model=task_model,
        dataset_key=artifacts.dataset_key,
        protocol=artifacts.protocol,
        seed=artifacts.seed,
        subjects=subjects,
        task_balanced_accuracy=result.balanced_accuracy,
        task_macro_f1=result.macro_f1,
        class_probabilities=probabilities,
        encoded_labels=artifacts.encoded_labels,
        member_indices=(
            list(artifacts.train_indices) + list(artifacts.validation_indices)
        ),
        nonmember_indices=list(artifacts.test_indices),
        epochs_trained=result.epochs_trained,
        best_checkpoint_epoch=result.best_checkpoint_epoch,
        best_validation_loss=result.best_validation_loss,
        resample_hz=result.resample_hz,
        normalization=result.normalization,
        classifier_head=result.classifier_head,
        label_smoothing=result.label_smoothing,
        bottleneck_dim=result.bottleneck_dim,
        feature_noise_std=result.feature_noise_std,
        mixup_alpha=result.mixup_alpha,
        confidence_penalty_beta=result.confidence_penalty_beta,
    )
    _save_cache_if_requested(posterior_cache_output, cache)
    return _run_membership_attack_from_cache(
        cache,
        attack_type=attack_type,
        score_type=score_type,
        attack_seed=attack_seed,
    )


def run_federated_eegnet_membership_inference_from_artifacts(
    artifacts,
    *,
    subjects: list[int] | None = None,
    task_model: str = "federated_eegnet",
    attack_type: str = "threshold",
    score_type: str = "label_known_log_probability",
    attack_seed: int | None = None,
    posterior_cache_output: str | Path | None = None,
) -> MembershipInferenceResult:
    """Attack a fitted federated EEGNet using the standard black-box protocol."""

    logits = batched_eegnet_logits(artifacts)
    probabilities = logits.softmax(dim=1).numpy()
    member_indices = list(artifacts.train_indices) + list(
        artifacts.validation_indices
    )
    nonmember_indices = list(artifacts.test_indices)
    result = artifacts.result
    cache = PosteriorFeatureCache(
        task_model=task_model,
        dataset_key=artifacts.dataset_key,
        protocol=artifacts.protocol,
        seed=artifacts.seed,
        subjects=subjects,
        task_balanced_accuracy=result.balanced_accuracy,
        task_macro_f1=result.macro_f1,
        class_probabilities=probabilities,
        encoded_labels=artifacts.encoded_labels,
        member_indices=member_indices,
        nonmember_indices=nonmember_indices,
        best_validation_loss=result.best_validation_loss,
        resample_hz=result.resample_hz,
        normalization=result.normalization,
        classifier_head=result.classifier_head,
        bottleneck_dim=result.bottleneck_dim,
        confidence_penalty_beta=result.confidence_penalty_beta,
        federated_rounds_trained=result.rounds_trained,
        federated_best_checkpoint_round=result.best_checkpoint_round,
        federated_local_epochs=result.local_epochs,
        federated_clients=result.clients,
        estimated_communication_bytes=result.estimated_communication_bytes,
    )
    _save_cache_if_requested(posterior_cache_output, cache)
    attack_kwargs = dict(
        task_model=cache.task_model,
        dataset_key=cache.dataset_key,
        protocol=cache.protocol,
        seed=cache.seed,
        attack_seed=attack_seed,
        task_balanced_accuracy=cache.task_balanced_accuracy,
        task_macro_f1=cache.task_macro_f1,
        class_probabilities=cache.class_probabilities,
        encoded_labels=cache.encoded_labels,
        member_indices=cache.member_indices,
        nonmember_indices=cache.nonmember_indices,
        score_type=score_type,
        best_validation_loss=cache.best_validation_loss,
    )
    if attack_type == "threshold":
        attack_result = _run_threshold_membership_attack(**attack_kwargs)
    else:
        attack_result = _run_learned_membership_attack(
            attack_type=attack_type,
            **attack_kwargs,
        )
    return replace(
        attack_result,
        resample_hz=result.resample_hz,
        normalization=result.normalization,
        classifier_head=result.classifier_head,
        bottleneck_dim=result.bottleneck_dim,
        confidence_penalty_beta=result.confidence_penalty_beta,
        federated_rounds_trained=result.rounds_trained,
        federated_best_checkpoint_round=result.best_checkpoint_round,
        federated_local_epochs=result.local_epochs,
        federated_clients=result.clients,
        estimated_communication_bytes=result.estimated_communication_bytes,
    )


def run_federated_eegnet_membership_inference_attack(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    federated_rounds: int = 20,
    federated_local_epochs: int = 1,
    task_batch_size: int = 32,
    task_learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    attack_type: str = "threshold",
    score_type: str = "label_known_log_probability",
    attack_seed: int | None = None,
    posterior_cache_output: str | Path | None = None,
) -> MembershipInferenceResult:
    artifacts = fit_federated_eegnet(
        dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        rounds=federated_rounds,
        local_epochs=federated_local_epochs,
        batch_size=task_batch_size,
        learning_rate=task_learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
    )
    return run_federated_eegnet_membership_inference_from_artifacts(
        artifacts,
        subjects=subjects,
        attack_type=attack_type,
        score_type=score_type,
        attack_seed=attack_seed,
        posterior_cache_output=posterior_cache_output,
    )


def run_label_smoothing_eegnet_membership_inference_attack(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    task_epochs: int = 20,
    task_batch_size: int = 32,
    task_learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    attack_type: str = "threshold",
    score_type: str = "label_known_log_probability",
    attack_seed: int | None = None,
    label_smoothing: float = 0.1,
    posterior_cache_output: str | Path | None = None,
) -> MembershipInferenceResult:
    """Run a scalar-score membership-inference attack on label-smoothed EEGNet."""

    artifacts = fit_label_smoothing_eegnet(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        epochs=task_epochs,
        batch_size=task_batch_size,
        learning_rate=task_learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        label_smoothing=label_smoothing,
    )

    logits = batched_eegnet_logits(artifacts)
    probabilities = logits.softmax(dim=1).numpy()

    member_indices = list(artifacts.train_indices) + list(artifacts.validation_indices)
    nonmember_indices = list(artifacts.test_indices)
    _save_cache_if_requested(
        posterior_cache_output,
        PosteriorFeatureCache(
            task_model="label_smoothing_eegnet",
            dataset_key=dataset_key,
            protocol=protocol,
            seed=seed,
            subjects=subjects,
            task_balanced_accuracy=artifacts.result.balanced_accuracy,
            task_macro_f1=artifacts.result.macro_f1,
            class_probabilities=probabilities,
            encoded_labels=artifacts.encoded_labels,
            member_indices=member_indices,
            nonmember_indices=nonmember_indices,
            epochs_trained=artifacts.result.epochs_trained,
            best_validation_loss=artifacts.result.best_validation_loss,
            label_smoothing=artifacts.result.label_smoothing,
        ),
    )
    attack_kwargs = dict(
        task_model="label_smoothing_eegnet",
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        attack_seed=attack_seed,
        task_balanced_accuracy=artifacts.result.balanced_accuracy,
        task_macro_f1=artifacts.result.macro_f1,
        class_probabilities=probabilities,
        member_indices=member_indices,
        nonmember_indices=nonmember_indices,
        epochs_trained=artifacts.result.epochs_trained,
        best_validation_loss=artifacts.result.best_validation_loss,
        label_smoothing=artifacts.result.label_smoothing,
    )
    if attack_type == "threshold":
        return _run_threshold_membership_attack(
            encoded_labels=artifacts.encoded_labels,
            score_type=score_type,
            **attack_kwargs,
        )
    return _run_learned_membership_attack(
        attack_type=attack_type,
        encoded_labels=artifacts.encoded_labels,
        score_type=score_type,
        **attack_kwargs,
    )


def run_bottleneck_eegnet_membership_inference_attack(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    task_epochs: int = 20,
    task_batch_size: int = 32,
    task_learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    attack_type: str = "threshold",
    score_type: str = "label_known_log_probability",
    attack_seed: int | None = None,
    bottleneck_dim: int = 8,
    posterior_cache_output: str | Path | None = None,
) -> MembershipInferenceResult:
    """Run a scalar-score membership-inference attack on bottleneck EEGNet."""

    artifacts = fit_bottleneck_eegnet(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        epochs=task_epochs,
        batch_size=task_batch_size,
        learning_rate=task_learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        bottleneck_dim=bottleneck_dim,
    )

    logits = batched_eegnet_logits(artifacts)
    probabilities = logits.softmax(dim=1).numpy()

    member_indices = list(artifacts.train_indices) + list(artifacts.validation_indices)
    nonmember_indices = list(artifacts.test_indices)
    _save_cache_if_requested(
        posterior_cache_output,
        PosteriorFeatureCache(
            task_model="bottleneck_eegnet",
            dataset_key=dataset_key,
            protocol=protocol,
            seed=seed,
            subjects=subjects,
            task_balanced_accuracy=artifacts.result.balanced_accuracy,
            task_macro_f1=artifacts.result.macro_f1,
            class_probabilities=probabilities,
            encoded_labels=artifacts.encoded_labels,
            member_indices=member_indices,
            nonmember_indices=nonmember_indices,
            epochs_trained=artifacts.result.epochs_trained,
            best_validation_loss=artifacts.result.best_validation_loss,
            label_smoothing=artifacts.result.label_smoothing,
            bottleneck_dim=artifacts.result.bottleneck_dim,
            feature_noise_std=artifacts.result.feature_noise_std,
            mixup_alpha=artifacts.result.mixup_alpha,
            confidence_penalty_beta=artifacts.result.confidence_penalty_beta,
        ),
    )
    attack_kwargs = dict(
        task_model="bottleneck_eegnet",
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        attack_seed=attack_seed,
        task_balanced_accuracy=artifacts.result.balanced_accuracy,
        task_macro_f1=artifacts.result.macro_f1,
        class_probabilities=probabilities,
        member_indices=member_indices,
        nonmember_indices=nonmember_indices,
        epochs_trained=artifacts.result.epochs_trained,
        best_validation_loss=artifacts.result.best_validation_loss,
        label_smoothing=artifacts.result.label_smoothing,
        bottleneck_dim=artifacts.result.bottleneck_dim,
        feature_noise_std=artifacts.result.feature_noise_std,
        mixup_alpha=artifacts.result.mixup_alpha,
        confidence_penalty_beta=artifacts.result.confidence_penalty_beta,
    )
    if attack_type == "threshold":
        return _run_threshold_membership_attack(
            encoded_labels=artifacts.encoded_labels,
            score_type=score_type,
            **attack_kwargs,
        )
    return _run_learned_membership_attack(
        attack_type=attack_type,
        encoded_labels=artifacts.encoded_labels,
        score_type=score_type,
        **attack_kwargs,
    )


def run_feature_noise_eegnet_membership_inference_attack(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    task_epochs: int = 20,
    task_batch_size: int = 32,
    task_learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    attack_type: str = "threshold",
    score_type: str = "label_known_log_probability",
    attack_seed: int | None = None,
    feature_noise_std: float = 0.1,
    posterior_cache_output: str | Path | None = None,
) -> MembershipInferenceResult:
    """Run a membership-inference attack on feature-noise EEGNet."""

    artifacts = fit_feature_noise_eegnet(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        epochs=task_epochs,
        batch_size=task_batch_size,
        learning_rate=task_learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        feature_noise_std=feature_noise_std,
    )

    logits = batched_eegnet_logits(artifacts)
    probabilities = logits.softmax(dim=1).numpy()

    member_indices = list(artifacts.train_indices) + list(artifacts.validation_indices)
    nonmember_indices = list(artifacts.test_indices)
    _save_cache_if_requested(
        posterior_cache_output,
        PosteriorFeatureCache(
            task_model="feature_noise_eegnet",
            dataset_key=dataset_key,
            protocol=protocol,
            seed=seed,
            subjects=subjects,
            task_balanced_accuracy=artifacts.result.balanced_accuracy,
            task_macro_f1=artifacts.result.macro_f1,
            class_probabilities=probabilities,
            encoded_labels=artifacts.encoded_labels,
            member_indices=member_indices,
            nonmember_indices=nonmember_indices,
            epochs_trained=artifacts.result.epochs_trained,
            best_validation_loss=artifacts.result.best_validation_loss,
            label_smoothing=artifacts.result.label_smoothing,
            bottleneck_dim=artifacts.result.bottleneck_dim,
            feature_noise_std=artifacts.result.feature_noise_std,
            mixup_alpha=artifacts.result.mixup_alpha,
            confidence_penalty_beta=artifacts.result.confidence_penalty_beta,
        ),
    )
    attack_kwargs = dict(
        task_model="feature_noise_eegnet",
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        attack_seed=attack_seed,
        task_balanced_accuracy=artifacts.result.balanced_accuracy,
        task_macro_f1=artifacts.result.macro_f1,
        class_probabilities=probabilities,
        member_indices=member_indices,
        nonmember_indices=nonmember_indices,
        epochs_trained=artifacts.result.epochs_trained,
        best_validation_loss=artifacts.result.best_validation_loss,
        label_smoothing=artifacts.result.label_smoothing,
        bottleneck_dim=artifacts.result.bottleneck_dim,
        feature_noise_std=artifacts.result.feature_noise_std,
        mixup_alpha=artifacts.result.mixup_alpha,
        confidence_penalty_beta=artifacts.result.confidence_penalty_beta,
    )
    if attack_type == "threshold":
        return _run_threshold_membership_attack(
            encoded_labels=artifacts.encoded_labels,
            score_type=score_type,
            **attack_kwargs,
        )
    return _run_learned_membership_attack(
        attack_type=attack_type,
        encoded_labels=artifacts.encoded_labels,
        score_type=score_type,
        **attack_kwargs,
    )


def run_confidence_penalty_eegnet_membership_inference_attack(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    task_epochs: int = 20,
    task_batch_size: int = 32,
    task_learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    attack_type: str = "threshold",
    score_type: str = "label_known_log_probability",
    attack_seed: int | None = None,
    confidence_penalty_beta: float = 0.05,
    posterior_cache_output: str | Path | None = None,
) -> MembershipInferenceResult:
    """Run a membership-inference attack on confidence-penalty EEGNet."""

    artifacts = fit_confidence_penalty_eegnet(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        epochs=task_epochs,
        batch_size=task_batch_size,
        learning_rate=task_learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        confidence_penalty_beta=confidence_penalty_beta,
    )

    logits = batched_eegnet_logits(artifacts)
    probabilities = logits.softmax(dim=1).numpy()

    member_indices = list(artifacts.train_indices) + list(artifacts.validation_indices)
    nonmember_indices = list(artifacts.test_indices)
    _save_cache_if_requested(
        posterior_cache_output,
        PosteriorFeatureCache(
            task_model="confidence_penalty_eegnet",
            dataset_key=dataset_key,
            protocol=protocol,
            seed=seed,
            subjects=subjects,
            task_balanced_accuracy=artifacts.result.balanced_accuracy,
            task_macro_f1=artifacts.result.macro_f1,
            class_probabilities=probabilities,
            encoded_labels=artifacts.encoded_labels,
            member_indices=member_indices,
            nonmember_indices=nonmember_indices,
            epochs_trained=artifacts.result.epochs_trained,
            best_validation_loss=artifacts.result.best_validation_loss,
            label_smoothing=artifacts.result.label_smoothing,
            bottleneck_dim=artifacts.result.bottleneck_dim,
            feature_noise_std=artifacts.result.feature_noise_std,
            mixup_alpha=artifacts.result.mixup_alpha,
            confidence_penalty_beta=artifacts.result.confidence_penalty_beta,
        ),
    )
    attack_kwargs = dict(
        task_model="confidence_penalty_eegnet",
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        attack_seed=attack_seed,
        task_balanced_accuracy=artifacts.result.balanced_accuracy,
        task_macro_f1=artifacts.result.macro_f1,
        class_probabilities=probabilities,
        member_indices=member_indices,
        nonmember_indices=nonmember_indices,
        epochs_trained=artifacts.result.epochs_trained,
        best_validation_loss=artifacts.result.best_validation_loss,
        label_smoothing=artifacts.result.label_smoothing,
        bottleneck_dim=artifacts.result.bottleneck_dim,
        feature_noise_std=artifacts.result.feature_noise_std,
        mixup_alpha=artifacts.result.mixup_alpha,
        confidence_penalty_beta=artifacts.result.confidence_penalty_beta,
    )
    if attack_type == "threshold":
        return _run_threshold_membership_attack(
            encoded_labels=artifacts.encoded_labels,
            score_type=score_type,
            **attack_kwargs,
        )
    return _run_learned_membership_attack(
        attack_type=attack_type,
        encoded_labels=artifacts.encoded_labels,
        score_type=score_type,
        **attack_kwargs,
    )


def run_mixup_eegnet_membership_inference_attack(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    task_epochs: int = 20,
    task_batch_size: int = 32,
    task_learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    attack_type: str = "threshold",
    score_type: str = "label_known_log_probability",
    attack_seed: int | None = None,
    mixup_alpha: float = 0.2,
    posterior_cache_output: str | Path | None = None,
) -> MembershipInferenceResult:
    """Run a membership-inference attack on mixup EEGNet."""

    artifacts = fit_mixup_eegnet(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        epochs=task_epochs,
        batch_size=task_batch_size,
        learning_rate=task_learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        mixup_alpha=mixup_alpha,
    )

    logits = batched_eegnet_logits(artifacts)
    probabilities = logits.softmax(dim=1).numpy()

    member_indices = list(artifacts.train_indices) + list(artifacts.validation_indices)
    nonmember_indices = list(artifacts.test_indices)
    _save_cache_if_requested(
        posterior_cache_output,
        PosteriorFeatureCache(
            task_model="mixup_eegnet",
            dataset_key=dataset_key,
            protocol=protocol,
            seed=seed,
            subjects=subjects,
            task_balanced_accuracy=artifacts.result.balanced_accuracy,
            task_macro_f1=artifacts.result.macro_f1,
            class_probabilities=probabilities,
            encoded_labels=artifacts.encoded_labels,
            member_indices=member_indices,
            nonmember_indices=nonmember_indices,
            epochs_trained=artifacts.result.epochs_trained,
            best_validation_loss=artifacts.result.best_validation_loss,
            label_smoothing=artifacts.result.label_smoothing,
            bottleneck_dim=artifacts.result.bottleneck_dim,
            feature_noise_std=artifacts.result.feature_noise_std,
            mixup_alpha=artifacts.result.mixup_alpha,
        ),
    )
    attack_kwargs = dict(
        task_model="mixup_eegnet",
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        attack_seed=attack_seed,
        task_balanced_accuracy=artifacts.result.balanced_accuracy,
        task_macro_f1=artifacts.result.macro_f1,
        class_probabilities=probabilities,
        member_indices=member_indices,
        nonmember_indices=nonmember_indices,
        epochs_trained=artifacts.result.epochs_trained,
        best_validation_loss=artifacts.result.best_validation_loss,
        label_smoothing=artifacts.result.label_smoothing,
        bottleneck_dim=artifacts.result.bottleneck_dim,
        feature_noise_std=artifacts.result.feature_noise_std,
        mixup_alpha=artifacts.result.mixup_alpha,
    )
    if attack_type == "threshold":
        return _run_threshold_membership_attack(
            encoded_labels=artifacts.encoded_labels,
            score_type=score_type,
            **attack_kwargs,
        )
    return _run_learned_membership_attack(
        attack_type=attack_type,
        encoded_labels=artifacts.encoded_labels,
        score_type=score_type,
        **attack_kwargs,
    )


def run_csp_membership_inference_attack(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    n_components: int = 8,
    attack_type: str = "threshold",
    score_type: str = "label_known_log_probability",
    attack_seed: int | None = None,
    posterior_cache_output: str | Path | None = None,
) -> MembershipInferenceResult:
    """Run a scalar-score membership-inference attack on CSP+LDA."""

    artifacts = fit_csp_lda(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        n_components=n_components,
    )
    probabilities = artifacts.pipeline.named_steps["lda"].predict_proba(
        artifacts.all_features
    )
    member_indices = list(artifacts.train_indices)
    nonmember_indices = list(artifacts.test_indices)
    _save_cache_if_requested(
        posterior_cache_output,
        PosteriorFeatureCache(
            task_model="csp_lda",
            dataset_key=dataset_key,
            protocol=protocol,
            seed=seed,
            subjects=subjects,
            task_balanced_accuracy=artifacts.result.balanced_accuracy,
            task_macro_f1=artifacts.result.macro_f1,
            class_probabilities=probabilities,
            encoded_labels=artifacts.encoded_labels,
            member_indices=member_indices,
            nonmember_indices=nonmember_indices,
        ),
    )

    attack_kwargs = dict(
        task_model="csp_lda",
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        attack_seed=attack_seed,
        task_balanced_accuracy=artifacts.result.balanced_accuracy,
        task_macro_f1=artifacts.result.macro_f1,
        class_probabilities=probabilities,
        member_indices=member_indices,
        nonmember_indices=nonmember_indices,
    )
    if attack_type == "threshold":
        return _run_threshold_membership_attack(
            encoded_labels=artifacts.encoded_labels,
            score_type=score_type,
            **attack_kwargs,
        )
    return _run_learned_membership_attack(
        attack_type=attack_type,
        encoded_labels=artifacts.encoded_labels,
        score_type=score_type,
        **attack_kwargs,
    )


def run_adversarial_eegnet_membership_inference_attack(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    task_epochs: int = 20,
    task_batch_size: int = 32,
    task_learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    adversarial_weight: float = 0.5,
    adversarial_schedule: str = "constant",
    ramp_up_fraction: float = 0.4,
    attack_type: str = "threshold",
    score_type: str = "label_known_log_probability",
    attack_seed: int | None = None,
    posterior_cache_output: str | Path | None = None,
) -> MembershipInferenceResult:
    """Run a scalar-score membership-inference attack on adversarial EEGNet."""

    artifacts = fit_adversarial_eegnet(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        epochs=task_epochs,
        batch_size=task_batch_size,
        learning_rate=task_learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        adversarial_weight=adversarial_weight,
        adversarial_schedule=adversarial_schedule,
        ramp_up_fraction=ramp_up_fraction,
    )

    base_model = artifacts.model.base_model
    logits = batched_eegnet_logits(
        artifacts,
        batch_size=task_batch_size,
        model=base_model,
    )
    probabilities = logits.softmax(dim=1).numpy()

    member_indices = list(artifacts.train_indices) + list(artifacts.validation_indices)
    nonmember_indices = list(artifacts.test_indices)
    _save_cache_if_requested(
        posterior_cache_output,
        PosteriorFeatureCache(
            task_model="adversarial_eegnet",
            dataset_key=dataset_key,
            protocol=protocol,
            seed=seed,
            subjects=subjects,
            task_balanced_accuracy=artifacts.result.task_balanced_accuracy,
            task_macro_f1=artifacts.result.task_macro_f1,
            class_probabilities=probabilities,
            encoded_labels=artifacts.encoded_labels,
            member_indices=member_indices,
            nonmember_indices=nonmember_indices,
            epochs_trained=artifacts.result.epochs_trained,
            best_validation_loss=artifacts.result.best_validation_loss,
            adversarial_weight=artifacts.result.adversarial_weight,
            adversarial_schedule=artifacts.result.adversarial_schedule,
            ramp_up_fraction=artifacts.result.ramp_up_fraction,
        ),
    )
    attack_kwargs = dict(
        task_model="adversarial_eegnet",
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        attack_seed=attack_seed,
        task_balanced_accuracy=artifacts.result.task_balanced_accuracy,
        task_macro_f1=artifacts.result.task_macro_f1,
        class_probabilities=probabilities,
        member_indices=member_indices,
        nonmember_indices=nonmember_indices,
        epochs_trained=artifacts.result.epochs_trained,
        best_validation_loss=artifacts.result.best_validation_loss,
        adversarial_weight=artifacts.result.adversarial_weight,
        adversarial_schedule=artifacts.result.adversarial_schedule,
        ramp_up_fraction=artifacts.result.ramp_up_fraction,
    )
    if attack_type == "threshold":
        return _run_threshold_membership_attack(
            encoded_labels=artifacts.encoded_labels,
            score_type=score_type,
            **attack_kwargs,
        )
    return _run_learned_membership_attack(
        attack_type=attack_type,
        encoded_labels=artifacts.encoded_labels,
        score_type=score_type,
        **attack_kwargs,
    )
