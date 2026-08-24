"""Subject-identification attacks on frozen task-model features."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from eeg_privacy_benchmark.models import (
    fit_bottleneck_eegnet,
    fit_confidence_penalty_eegnet,
    fit_csp_lda,
    fit_feature_noise_eegnet,
    fit_label_smoothing_eegnet,
    fit_mixup_eegnet,
)
from eeg_privacy_benchmark.models.eegnet_adversarial import fit_adversarial_eegnet
from eeg_privacy_benchmark.models.eegnet_federated import fit_federated_eegnet
from eeg_privacy_benchmark.models.eegnet import (
    batched_eegnet_embeddings,
    fit_eegnet,
)


def _import_probe_dependencies():
    try:
        import numpy as np
        import torch
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.preprocessing import LabelEncoder
    except ImportError as exc:
        raise ImportError(
            "Subject-identification probe requires numpy, torch, and scikit-learn."
        ) from exc
    return {
        "np": np,
        "torch": torch,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
        "LabelEncoder": LabelEncoder,
    }


@dataclass(frozen=True)
class SubjectIdentificationResult:
    task_model: str
    dataset_key: str
    protocol: str
    seed: int
    task_balanced_accuracy: float
    task_macro_f1: float | None
    attacker_train_trials: int
    attacker_test_trials: int
    num_subjects: int
    subject_id_accuracy: float
    subject_id_macro_f1: float
    subject_id_top3_accuracy: float
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

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _top_k_accuracy(predicted_probabilities, y_true, *, top_k: int) -> float:
    hits = (predicted_probabilities.argsort(axis=1)[:, -top_k:] == y_true[:, None]).any(axis=1)
    return float(hits.mean())


def _run_linear_subject_probe(
    *,
    task_model: str,
    dataset_key: str,
    protocol: str,
    seed: int,
    task_balanced_accuracy: float,
    task_macro_f1: float | None,
    embeddings,
    trial_records,
    attacker_train_indices: list[int],
    attacker_test_indices: list[int],
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
) -> SubjectIdentificationResult:
    deps = _import_probe_dependencies()
    subject_encoder = deps["LabelEncoder"]()
    encoded_subjects = subject_encoder.fit_transform(
        [record.subject_id for record in trial_records]
    )
    if len(subject_encoder.classes_) < 2:
        raise ValueError("Subject-identification probe needs at least two subjects.")

    classifier = deps["LogisticRegression"](max_iter=2000)
    classifier.fit(
        embeddings[attacker_train_indices],
        encoded_subjects[attacker_train_indices],
    )
    predicted_subjects = classifier.predict(embeddings[attacker_test_indices])
    predicted_probabilities = classifier.predict_proba(embeddings[attacker_test_indices])
    y_true = encoded_subjects[attacker_test_indices]
    top_k = min(3, len(subject_encoder.classes_))

    return SubjectIdentificationResult(
        task_model=task_model,
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        task_balanced_accuracy=task_balanced_accuracy,
        task_macro_f1=task_macro_f1,
        attacker_train_trials=len(attacker_train_indices),
        attacker_test_trials=len(attacker_test_indices),
        num_subjects=len(subject_encoder.classes_),
        subject_id_accuracy=float(deps["accuracy_score"](y_true, predicted_subjects)),
        subject_id_macro_f1=float(
            deps["f1_score"](y_true, predicted_subjects, average="macro")
        ),
        subject_id_top3_accuracy=_top_k_accuracy(
            predicted_probabilities,
            y_true,
            top_k=top_k,
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


def run_eegnet_subject_id_probe(
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
) -> SubjectIdentificationResult:
    """Train EEGNet, freeze embeddings, and fit a linear subject-ID probe."""

    if protocol != "cross_session":
        raise ValueError(
            "The initial subject-identification probe currently supports only "
            "cross-session evaluation, where the same subjects appear across "
            "train and test trials."
        )

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

    return run_eegnet_subject_id_from_artifacts(artifacts)


def run_eegnet_subject_id_from_artifacts(
    artifacts,
    *,
    task_model: str = "eegnet",
) -> SubjectIdentificationResult:
    """Probe subject leakage in an already-fitted centralized EEGNet."""

    if artifacts.protocol != "cross_session":
        raise ValueError(
            "The subject-identification probe requires cross-session evaluation, "
            "where the same subjects appear in train and test trials."
        )
    result = artifacts.result
    probe_result = _run_linear_subject_probe(
        task_model=task_model,
        dataset_key=artifacts.dataset_key,
        protocol=artifacts.protocol,
        seed=artifacts.seed,
        task_balanced_accuracy=result.balanced_accuracy,
        task_macro_f1=result.macro_f1,
        embeddings=batched_eegnet_embeddings(artifacts).numpy(),
        trial_records=artifacts.trial_records,
        attacker_train_indices=(
            list(artifacts.train_indices) + list(artifacts.validation_indices)
        ),
        attacker_test_indices=list(artifacts.test_indices),
        epochs_trained=result.epochs_trained,
        best_validation_loss=result.best_validation_loss,
        label_smoothing=result.label_smoothing,
        bottleneck_dim=result.bottleneck_dim,
        feature_noise_std=result.feature_noise_std,
        mixup_alpha=result.mixup_alpha,
        confidence_penalty_beta=result.confidence_penalty_beta,
    )
    return replace(
        probe_result,
        best_checkpoint_epoch=result.best_checkpoint_epoch,
        resample_hz=result.resample_hz,
        normalization=result.normalization,
        classifier_head=result.classifier_head,
    )


def run_federated_eegnet_subject_id_from_artifacts(
    artifacts,
    *,
    task_model: str = "federated_eegnet",
) -> SubjectIdentificationResult:
    """Probe subject leakage in a fitted federated EEGNet's embeddings."""

    if artifacts.protocol != "cross_session":
        raise ValueError(
            "The subject-identification probe requires cross-session evaluation, "
            "where the same subjects appear in train and test trials."
        )
    embeddings = batched_eegnet_embeddings(artifacts).numpy()
    result = artifacts.result
    probe_result = _run_linear_subject_probe(
        task_model=task_model,
        dataset_key=artifacts.dataset_key,
        protocol=artifacts.protocol,
        seed=artifacts.seed,
        task_balanced_accuracy=result.balanced_accuracy,
        task_macro_f1=result.macro_f1,
        embeddings=embeddings,
        trial_records=artifacts.trial_records,
        attacker_train_indices=(
            list(artifacts.train_indices) + list(artifacts.validation_indices)
        ),
        attacker_test_indices=list(artifacts.test_indices),
        best_validation_loss=result.best_validation_loss,
        bottleneck_dim=result.bottleneck_dim,
        confidence_penalty_beta=result.confidence_penalty_beta,
    )
    return replace(
        probe_result,
        federated_rounds_trained=result.rounds_trained,
        federated_best_checkpoint_round=result.best_checkpoint_round,
        federated_local_epochs=result.local_epochs,
        federated_clients=result.clients,
        estimated_communication_bytes=result.estimated_communication_bytes,
        resample_hz=result.resample_hz,
        normalization=result.normalization,
        classifier_head=result.classifier_head,
        confidence_penalty_beta=result.confidence_penalty_beta,
    )


def run_federated_eegnet_subject_id_probe(
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
) -> SubjectIdentificationResult:
    if protocol != "cross_session":
        raise ValueError(
            "The subject-identification probe requires cross-session evaluation, "
            "where the same subjects appear in train and test trials."
        )
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
    return run_federated_eegnet_subject_id_from_artifacts(artifacts)


def run_csp_subject_id_probe(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    n_components: int = 8,
) -> SubjectIdentificationResult:
    """Fit CSP, freeze CSP features, and fit a linear subject-ID probe."""

    if protocol != "cross_session":
        raise ValueError(
            "The initial subject-identification probe currently supports only "
            "cross-session evaluation, where the same subjects appear across "
            "train and test trials."
        )

    artifacts = fit_csp_lda(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        n_components=n_components,
    )
    return _run_linear_subject_probe(
        task_model="csp_lda",
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        task_balanced_accuracy=artifacts.result.balanced_accuracy,
        task_macro_f1=artifacts.result.macro_f1,
        embeddings=artifacts.all_features,
        trial_records=artifacts.trial_records,
        attacker_train_indices=list(artifacts.train_indices),
        attacker_test_indices=list(artifacts.test_indices),
    )


def run_adversarial_eegnet_subject_id_probe(
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
) -> SubjectIdentificationResult:
    """Train subject-adversarial EEGNet, freeze embeddings, and fit a probe."""

    if protocol != "cross_session":
        raise ValueError(
            "The initial subject-identification probe currently supports only "
            "cross-session evaluation, where the same subjects appear across "
            "train and test trials."
        )

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
    embeddings = batched_eegnet_embeddings(
        artifacts,
        batch_size=task_batch_size,
        feature_model=base_model,
    ).numpy()

    attacker_train_indices = list(artifacts.train_indices) + list(artifacts.validation_indices)
    attacker_test_indices = list(artifacts.test_indices)
    return _run_linear_subject_probe(
        task_model="adversarial_eegnet",
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        task_balanced_accuracy=artifacts.result.task_balanced_accuracy,
        task_macro_f1=artifacts.result.task_macro_f1,
        embeddings=embeddings,
        trial_records=artifacts.trial_records,
        attacker_train_indices=attacker_train_indices,
        attacker_test_indices=attacker_test_indices,
        epochs_trained=artifacts.result.epochs_trained,
        best_validation_loss=artifacts.result.best_validation_loss,
        adversarial_weight=artifacts.result.adversarial_weight,
        adversarial_schedule=artifacts.result.adversarial_schedule,
        ramp_up_fraction=artifacts.result.ramp_up_fraction,
    )


def run_label_smoothing_eegnet_subject_id_probe(
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
    label_smoothing: float = 0.1,
) -> SubjectIdentificationResult:
    """Train label-smoothed EEGNet, freeze embeddings, and fit a linear subject-ID probe."""

    if protocol != "cross_session":
        raise ValueError(
            "The initial subject-identification probe currently supports only "
            "cross-session evaluation, where the same subjects appear across "
            "train and test trials."
        )

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

    model = artifacts.model
    embeddings = batched_eegnet_embeddings(artifacts).numpy()

    attacker_train_indices = list(artifacts.train_indices) + list(artifacts.validation_indices)
    attacker_test_indices = list(artifacts.test_indices)
    return _run_linear_subject_probe(
        task_model="label_smoothing_eegnet",
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        task_balanced_accuracy=artifacts.result.balanced_accuracy,
        task_macro_f1=artifacts.result.macro_f1,
        embeddings=embeddings,
        trial_records=artifacts.trial_records,
        attacker_train_indices=attacker_train_indices,
        attacker_test_indices=attacker_test_indices,
        epochs_trained=artifacts.result.epochs_trained,
        best_validation_loss=artifacts.result.best_validation_loss,
        label_smoothing=artifacts.result.label_smoothing,
        bottleneck_dim=artifacts.result.bottleneck_dim,
        feature_noise_std=artifacts.result.feature_noise_std,
        mixup_alpha=artifacts.result.mixup_alpha,
        confidence_penalty_beta=artifacts.result.confidence_penalty_beta,
    )


def run_feature_noise_eegnet_subject_id_probe(
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
    feature_noise_std: float = 0.1,
) -> SubjectIdentificationResult:
    """Train feature-noise EEGNet, freeze embeddings, and fit a linear subject-ID probe."""

    if protocol != "cross_session":
        raise ValueError(
            "The initial subject-identification probe currently supports only "
            "cross-session evaluation, where the same subjects appear across "
            "train and test trials."
        )

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

    embeddings = batched_eegnet_embeddings(artifacts).numpy()
    attacker_train_indices = list(artifacts.train_indices) + list(artifacts.validation_indices)
    attacker_test_indices = list(artifacts.test_indices)
    return _run_linear_subject_probe(
        task_model="feature_noise_eegnet",
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        task_balanced_accuracy=artifacts.result.balanced_accuracy,
        task_macro_f1=artifacts.result.macro_f1,
        embeddings=embeddings,
        trial_records=artifacts.trial_records,
        attacker_train_indices=attacker_train_indices,
        attacker_test_indices=attacker_test_indices,
        epochs_trained=artifacts.result.epochs_trained,
        best_validation_loss=artifacts.result.best_validation_loss,
        label_smoothing=artifacts.result.label_smoothing,
        bottleneck_dim=artifacts.result.bottleneck_dim,
        feature_noise_std=artifacts.result.feature_noise_std,
        mixup_alpha=artifacts.result.mixup_alpha,
        confidence_penalty_beta=artifacts.result.confidence_penalty_beta,
    )


def run_confidence_penalty_eegnet_subject_id_probe(
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
    confidence_penalty_beta: float = 0.05,
) -> SubjectIdentificationResult:
    """Train confidence-penalty EEGNet, freeze embeddings, and fit a probe."""

    if protocol != "cross_session":
        raise ValueError(
            "The initial subject-identification probe currently supports only "
            "cross-session evaluation, where the same subjects appear across "
            "train and test trials."
        )

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

    embeddings = batched_eegnet_embeddings(artifacts).numpy()
    attacker_train_indices = list(artifacts.train_indices) + list(artifacts.validation_indices)
    attacker_test_indices = list(artifacts.test_indices)
    return _run_linear_subject_probe(
        task_model="confidence_penalty_eegnet",
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        task_balanced_accuracy=artifacts.result.balanced_accuracy,
        task_macro_f1=artifacts.result.macro_f1,
        embeddings=embeddings,
        trial_records=artifacts.trial_records,
        attacker_train_indices=attacker_train_indices,
        attacker_test_indices=attacker_test_indices,
        epochs_trained=artifacts.result.epochs_trained,
        best_validation_loss=artifacts.result.best_validation_loss,
        label_smoothing=artifacts.result.label_smoothing,
        bottleneck_dim=artifacts.result.bottleneck_dim,
        feature_noise_std=artifacts.result.feature_noise_std,
        mixup_alpha=artifacts.result.mixup_alpha,
        confidence_penalty_beta=artifacts.result.confidence_penalty_beta,
    )


def run_mixup_eegnet_subject_id_probe(
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
    mixup_alpha: float = 0.2,
) -> SubjectIdentificationResult:
    """Train mixup EEGNet, freeze embeddings, and fit a linear subject-ID probe."""

    if protocol != "cross_session":
        raise ValueError(
            "The initial subject-identification probe currently supports only "
            "cross-session evaluation, where the same subjects appear across "
            "train and test trials."
        )

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

    embeddings = batched_eegnet_embeddings(artifacts).numpy()
    attacker_train_indices = list(artifacts.train_indices) + list(artifacts.validation_indices)
    attacker_test_indices = list(artifacts.test_indices)
    return _run_linear_subject_probe(
        task_model="mixup_eegnet",
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        task_balanced_accuracy=artifacts.result.balanced_accuracy,
        task_macro_f1=artifacts.result.macro_f1,
        embeddings=embeddings,
        trial_records=artifacts.trial_records,
        attacker_train_indices=attacker_train_indices,
        attacker_test_indices=attacker_test_indices,
        epochs_trained=artifacts.result.epochs_trained,
        best_validation_loss=artifacts.result.best_validation_loss,
        label_smoothing=artifacts.result.label_smoothing,
        bottleneck_dim=artifacts.result.bottleneck_dim,
        feature_noise_std=artifacts.result.feature_noise_std,
        mixup_alpha=artifacts.result.mixup_alpha,
        confidence_penalty_beta=artifacts.result.confidence_penalty_beta,
    )


def run_bottleneck_eegnet_subject_id_probe(
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
    bottleneck_dim: int = 8,
) -> SubjectIdentificationResult:
    """Train bottleneck EEGNet, freeze embeddings, and fit a linear subject-ID probe."""

    if protocol != "cross_session":
        raise ValueError(
            "The initial subject-identification probe currently supports only "
            "cross-session evaluation, where the same subjects appear across "
            "train and test trials."
        )

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

    model = artifacts.model
    embeddings = batched_eegnet_embeddings(artifacts).numpy()

    attacker_train_indices = list(artifacts.train_indices) + list(artifacts.validation_indices)
    attacker_test_indices = list(artifacts.test_indices)
    return _run_linear_subject_probe(
        task_model="bottleneck_eegnet",
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        task_balanced_accuracy=artifacts.result.balanced_accuracy,
        task_macro_f1=artifacts.result.macro_f1,
        embeddings=embeddings,
        trial_records=artifacts.trial_records,
        attacker_train_indices=attacker_train_indices,
        attacker_test_indices=attacker_test_indices,
        epochs_trained=artifacts.result.epochs_trained,
        best_validation_loss=artifacts.result.best_validation_loss,
        label_smoothing=artifacts.result.label_smoothing,
        bottleneck_dim=artifacts.result.bottleneck_dim,
        feature_noise_std=artifacts.result.feature_noise_std,
        mixup_alpha=artifacts.result.mixup_alpha,
        confidence_penalty_beta=artifacts.result.confidence_penalty_beta,
    )
