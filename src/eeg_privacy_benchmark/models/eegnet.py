"""Minimal EEGNet baseline for the EEG privacy benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random

from eeg_privacy_benchmark.datasets.base import DatasetManifest
from eeg_privacy_benchmark.datasets.factory import build_dataset_loader
from eeg_privacy_benchmark.datasets.splits import (
    generate_cross_run_split,
    generate_cross_session_split,
    generate_cross_subject_split,
)


def _import_model_dependencies():
    try:
        import numpy as np
        import torch
        from sklearn.metrics import balanced_accuracy_score, f1_score
        from sklearn.preprocessing import LabelEncoder
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(
            "EEGNet baseline requires numpy, torch, and scikit-learn."
        ) from exc
    return {
        "np": np,
        "torch": torch,
        "balanced_accuracy_score": balanced_accuracy_score,
        "f1_score": f1_score,
        "LabelEncoder": LabelEncoder,
        "nn": nn,
        "DataLoader": DataLoader,
        "TensorDataset": TensorDataset,
    }


def _build_eegnet_model(
    torch,
    nn,
    n_channels: int,
    n_times: int,
    n_classes: int,
    *,
    bottleneck_dim: int | None = None,
    feature_noise_std: float = 0.0,
    classifier_head: str = "flatten",
):
    if classifier_head not in {"flatten", "global_average"}:
        raise ValueError(
            "classifier_head must be either 'flatten' or 'global_average'."
        )

    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            f1 = 8
            depth_multiplier = 2
            f2 = 16
            self.features = nn.Sequential(
                nn.Conv2d(1, f1, kernel_size=(1, 64), padding=(0, 32), bias=False),
                nn.BatchNorm2d(f1),
                nn.Conv2d(
                    f1,
                    f1 * depth_multiplier,
                    kernel_size=(n_channels, 1),
                    groups=f1,
                    bias=False,
                ),
                nn.BatchNorm2d(f1 * depth_multiplier),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4)),
                nn.Dropout(p=0.25),
                nn.Conv2d(
                    f1 * depth_multiplier,
                    f1 * depth_multiplier,
                    kernel_size=(1, 16),
                    padding=(0, 8),
                    groups=f1 * depth_multiplier,
                    bias=False,
                ),
                nn.Conv2d(f1 * depth_multiplier, f2, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(f2),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 8)),
                nn.Dropout(p=0.25),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_times)
                feature_map = self.features(dummy)
                flattened_dim = feature_map.reshape(1, -1).shape[1]
            self.classifier_head = classifier_head
            self.global_pool = (
                nn.AdaptiveAvgPool2d((1, 1))
                if classifier_head == "global_average"
                else None
            )
            self.bottleneck = None
            self.bottleneck_activation = None
            self.feature_noise_std = feature_noise_std
            classifier_input_dim = (
                feature_map.shape[1]
                if classifier_head == "global_average"
                else flattened_dim
            )
            if bottleneck_dim is not None:
                self.bottleneck = nn.Linear(classifier_input_dim, bottleneck_dim)
                self.bottleneck_activation = nn.ELU()
                classifier_input_dim = bottleneck_dim
            self.classifier = nn.Linear(classifier_input_dim, n_classes)

        def forward(self, inputs):
            features = self.forward_features(inputs)
            if self.training and self.feature_noise_std > 0.0:
                features = features + (
                    torch.randn_like(features) * self.feature_noise_std
                )
            logits = self.classifier(features)
            return logits

        def forward_features(self, inputs):
            features = self.features(inputs)
            if self.global_pool is not None:
                features = self.global_pool(features)
            flattened = features.reshape(inputs.shape[0], -1)
            if self.bottleneck is None or self.bottleneck_activation is None:
                return flattened
            return self.bottleneck_activation(self.bottleneck(flattened))

    return EEGNet()


def _set_all_seeds(torch, np, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_validation_indices(np, train_indices: list[int], labels, seed: int, validation_fraction: float) -> tuple[list[int], list[int]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    label_to_indices: dict[int, list[int]] = {}
    for index in train_indices:
        label_to_indices.setdefault(int(labels[index]), []).append(index)

    train_subset: list[int] = []
    validation_subset: list[int] = []
    rng = np.random.default_rng(seed)

    for indices in label_to_indices.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
        if validation_count >= len(shuffled):
            validation_count = max(1, len(shuffled) - 1)
        validation_subset.extend(shuffled[:validation_count])
        train_subset.extend(shuffled[validation_count:])

    if not train_subset or not validation_subset:
        raise ValueError("Unable to construct non-empty train/validation split.")

    return sorted(train_subset), sorted(validation_subset)


def _normalize_features(
    np,
    features,
    train_indices: list[int],
    *,
    normalization: str = "global_train_channel",
):
    if normalization == "global_train_channel":
        train_stats = features[train_indices]
        mean = train_stats.mean(axis=(0, 2), keepdims=True)
        std = train_stats.std(axis=(0, 2), keepdims=True)
    elif normalization == "per_trial_channel":
        mean = features.mean(axis=2, keepdims=True)
        std = features.std(axis=2, keepdims=True)
    else:
        raise ValueError(
            "normalization must be 'global_train_channel' or 'per_trial_channel'."
        )
    std = np.where(std < 1e-6, 1.0, std)
    normalized = (features - mean) / std
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


@dataclass(frozen=True)
class DeepBaselineResult:
    dataset_key: str
    protocol: str
    seed: int
    train_trials: int
    validation_trials: int
    test_trials: int
    epochs_requested: int
    epochs_trained: int
    best_checkpoint_epoch: int
    best_validation_loss: float
    best_validation_balanced_accuracy: float
    checkpoint_validation_balanced_accuracy: float
    balanced_accuracy: float
    macro_f1: float
    label_smoothing: float | None = None
    bottleneck_dim: int | None = None
    feature_noise_std: float | None = None
    mixup_alpha: float | None = None
    confidence_penalty_beta: float | None = None
    resample_hz: float | None = None
    normalization: str = "global_train_channel"
    classifier_head: str = "flatten"

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class EEGNetTrainingArtifacts:
    dataset_key: str
    protocol: str
    seed: int
    model: object
    result: DeepBaselineResult
    all_features: object
    encoded_labels: object
    trial_records: tuple
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


def _batched_model_forward(
    torch,
    model,
    features,
    *,
    batch_size: int,
    forward_fn=None,
):
    outputs = []
    for start in range(0, len(features), batch_size):
        stop = start + batch_size
        batch_inputs = torch.tensor(
            features[start:stop][:, None, :, :],
            dtype=torch.float32,
        )
        batch_outputs = forward_fn(batch_inputs) if forward_fn else model(batch_inputs)
        outputs.append(batch_outputs.detach().cpu())
    return torch.cat(outputs, dim=0)


def batched_eegnet_logits(
    artifacts: EEGNetTrainingArtifacts,
    *,
    batch_size: int = 64,
    model=None,
):
    deps = _import_model_dependencies()
    torch = deps["torch"]
    prediction_model = model or artifacts.model
    prediction_model.eval()
    with torch.no_grad():
        return _batched_model_forward(
            torch,
            prediction_model,
            artifacts.all_features,
            batch_size=batch_size,
        )


def batched_eegnet_embeddings(
    artifacts: EEGNetTrainingArtifacts,
    *,
    batch_size: int = 64,
    feature_model=None,
):
    deps = _import_model_dependencies()
    torch = deps["torch"]
    model = feature_model or artifacts.model
    model.eval()
    with torch.no_grad():
        return _batched_model_forward(
            torch,
            model,
            artifacts.all_features,
            batch_size=batch_size,
            forward_fn=model.forward_features,
        )


def fit_eegnet(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    label_smoothing: float = 0.0,
    bottleneck_dim: int | None = None,
    feature_noise_std: float = 0.0,
    mixup_alpha: float = 0.0,
    confidence_penalty_beta: float = 0.0,
    resample_hz: float | None = None,
    normalization: str = "global_train_channel",
    classifier_head: str = "flatten",
) -> EEGNetTrainingArtifacts:
    """Train EEGNet and return the fitted model plus aligned metadata."""

    deps = _import_model_dependencies()
    np = deps["np"]
    torch = deps["torch"]
    nn = deps["nn"]
    _set_all_seeds(torch, np, seed)

    loader = build_dataset_loader(dataset_key, resample_hz=resample_hz)
    loaded = loader.load_array_data(subjects=subjects)
    manifest = DatasetManifest(
        dataset_key=dataset_key,
        trial_records=loaded.trial_records,
        available_labels=tuple(sorted({str(label) for label in loaded.labels})),
        notes=loader.dataset_spec.notes,
    )

    if protocol == "cross_subject":
        split_manifest = generate_cross_subject_split(manifest, seed=seed)
    elif protocol == "cross_session":
        split_manifest = generate_cross_session_split(manifest, seed=seed)
    elif protocol == "cross_run":
        split_manifest = generate_cross_run_split(manifest, seed=seed)
    else:
        raise ValueError(f"Unsupported protocol for EEGNet baseline: {protocol}")

    split_lookup = {
        assignment.trial_id: assignment.split for assignment in split_manifest.assignments
    }
    train_indices = [
        index
        for index, record in enumerate(loaded.trial_records)
        if split_lookup[record.trial_id] == "train"
    ]
    test_indices = [
        index
        for index, record in enumerate(loaded.trial_records)
        if split_lookup[record.trial_id] == "test"
    ]
    if not train_indices or not test_indices:
        raise ValueError(f"Protocol {protocol} did not produce both train and test trials.")

    features = np.asarray(loaded.features, dtype=np.float32)
    label_encoder = deps["LabelEncoder"]()
    encoded_labels = label_encoder.fit_transform(loaded.labels)
    train_indices, validation_indices = _build_validation_indices(
        np,
        train_indices,
        encoded_labels,
        seed,
        validation_fraction,
    )
    features, _, _ = _normalize_features(
        np,
        features,
        train_indices,
        normalization=normalization,
    )

    x_train = torch.tensor(features[train_indices][:, None, :, :], dtype=torch.float32)
    y_train = torch.tensor(encoded_labels[train_indices], dtype=torch.long)
    y_validation = torch.tensor(
        encoded_labels[validation_indices], dtype=torch.long
    )
    y_test = torch.tensor(encoded_labels[test_indices], dtype=torch.long)

    train_loader = deps["DataLoader"](
        deps["TensorDataset"](x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    model = _build_eegnet_model(
        torch,
        nn,
        n_channels=x_train.shape[2],
        n_times=x_train.shape[3],
        n_classes=len(label_encoder.classes_),
        bottleneck_dim=bottleneck_dim,
        feature_noise_std=feature_noise_std,
        classifier_head=classifier_head,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0.0, 1.0).")
    if feature_noise_std < 0.0:
        raise ValueError("feature_noise_std must be non-negative.")
    if mixup_alpha < 0.0:
        raise ValueError("mixup_alpha must be non-negative.")
    if confidence_penalty_beta < 0.0:
        raise ValueError("confidence_penalty_beta must be non-negative.")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_state = None
    best_validation_loss = float("inf")
    best_validation_balanced_accuracy = 0.0
    best_checkpoint_epoch = 0
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch_idx in range(epochs):
        model.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            if mixup_alpha > 0.0 and len(batch_labels) > 1:
                mixup_lambda = float(np.random.beta(mixup_alpha, mixup_alpha))
                permutation = torch.randperm(len(batch_labels))
                mixed_features = (
                    mixup_lambda * batch_features
                    + (1.0 - mixup_lambda) * batch_features[permutation]
                )
                logits = model(mixed_features)
                loss = (
                    mixup_lambda * loss_fn(logits, batch_labels)
                    + (1.0 - mixup_lambda) * loss_fn(logits, batch_labels[permutation])
                )
            else:
                logits = model(batch_features)
                loss = loss_fn(logits, batch_labels)
            if confidence_penalty_beta > 0.0:
                probabilities = logits.softmax(dim=1)
                confidence_penalty = (
                    probabilities
                    * torch.log(probabilities.clamp(min=1e-8, max=1.0))
                ).sum(dim=1).mean()
                loss = loss + confidence_penalty_beta * confidence_penalty
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = _batched_model_forward(
                torch,
                model,
                features[validation_indices],
                batch_size=batch_size,
            )
            validation_loss = float(loss_fn(validation_logits, y_validation).item())
            validation_predictions = validation_logits.argmax(dim=1).cpu().numpy()
            validation_balanced_accuracy = float(
                deps["balanced_accuracy_score"](
                    y_validation.cpu().numpy(),
                    validation_predictions,
                )
            )

        epochs_trained = epoch_idx + 1
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_balanced_accuracy = validation_balanced_accuracy
            best_checkpoint_epoch = epoch_idx + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("EEGNet training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = _batched_model_forward(
            torch,
            model,
            features[test_indices],
            batch_size=batch_size,
        )
        predictions = logits.argmax(dim=1).cpu().numpy()

    result = DeepBaselineResult(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        train_trials=len(train_indices),
        validation_trials=len(validation_indices),
        test_trials=len(test_indices),
        epochs_requested=epochs,
        epochs_trained=epochs_trained,
        best_checkpoint_epoch=best_checkpoint_epoch,
        best_validation_loss=best_validation_loss,
        best_validation_balanced_accuracy=best_validation_balanced_accuracy,
        checkpoint_validation_balanced_accuracy=best_validation_balanced_accuracy,
        balanced_accuracy=float(
            deps["balanced_accuracy_score"](y_test.cpu().numpy(), predictions)
        ),
        macro_f1=float(
            deps["f1_score"](y_test.cpu().numpy(), predictions, average="macro")
        ),
        label_smoothing=label_smoothing,
        bottleneck_dim=bottleneck_dim,
        feature_noise_std=feature_noise_std,
        mixup_alpha=mixup_alpha,
        confidence_penalty_beta=confidence_penalty_beta,
        resample_hz=resample_hz,
        normalization=normalization,
        classifier_head=classifier_head,
    )
    return EEGNetTrainingArtifacts(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        model=model,
        result=result,
        all_features=features,
        encoded_labels=encoded_labels,
        trial_records=loaded.trial_records,
        train_indices=tuple(train_indices),
        validation_indices=tuple(validation_indices),
        test_indices=tuple(test_indices),
    )


def run_eegnet(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    label_smoothing: float = 0.0,
    bottleneck_dim: int | None = None,
    feature_noise_std: float = 0.0,
    mixup_alpha: float = 0.0,
    confidence_penalty_beta: float = 0.0,
    resample_hz: float | None = None,
    normalization: str = "global_train_channel",
    classifier_head: str = "flatten",
) -> DeepBaselineResult:
    """Train and evaluate the EEGNet baseline on one dataset/protocol."""

    return fit_eegnet(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        label_smoothing=label_smoothing,
        bottleneck_dim=bottleneck_dim,
        feature_noise_std=feature_noise_std,
        mixup_alpha=mixup_alpha,
        confidence_penalty_beta=confidence_penalty_beta,
        resample_hz=resample_hz,
        normalization=normalization,
        classifier_head=classifier_head,
    ).result


def fit_label_smoothing_eegnet(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    label_smoothing: float = 0.1,
) -> EEGNetTrainingArtifacts:
    """Train EEGNet with label smoothing enabled."""

    return fit_eegnet(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        label_smoothing=label_smoothing,
    )


def run_label_smoothing_eegnet(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    label_smoothing: float = 0.1,
) -> DeepBaselineResult:
    """Train and evaluate EEGNet with label smoothing enabled."""

    return fit_label_smoothing_eegnet(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        label_smoothing=label_smoothing,
    ).result


def fit_bottleneck_eegnet(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    bottleneck_dim: int = 8,
) -> EEGNetTrainingArtifacts:
    """Train EEGNet with a low-dimensional feature bottleneck."""

    if bottleneck_dim < 1:
        raise ValueError("bottleneck_dim must be at least 1.")
    return fit_eegnet(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        bottleneck_dim=bottleneck_dim,
    )


def fit_feature_noise_eegnet(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    feature_noise_std: float = 0.1,
) -> EEGNetTrainingArtifacts:
    """Train EEGNet with Gaussian noise injected into learned features during training."""

    return fit_eegnet(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        feature_noise_std=feature_noise_std,
    )


def fit_mixup_eegnet(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    mixup_alpha: float = 0.2,
) -> EEGNetTrainingArtifacts:
    """Train EEGNet with mixup augmentation during task training."""

    return fit_eegnet(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        mixup_alpha=mixup_alpha,
    )


def fit_confidence_penalty_eegnet(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    confidence_penalty_beta: float = 0.05,
) -> EEGNetTrainingArtifacts:
    """Train EEGNet with a confidence penalty on low-entropy predictions."""

    return fit_eegnet(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        confidence_penalty_beta=confidence_penalty_beta,
    )


def run_feature_noise_eegnet(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    feature_noise_std: float = 0.1,
) -> DeepBaselineResult:
    """Train and evaluate EEGNet with representation noise during training."""

    return fit_feature_noise_eegnet(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        feature_noise_std=feature_noise_std,
    ).result


def run_mixup_eegnet(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    mixup_alpha: float = 0.2,
) -> DeepBaselineResult:
    """Train and evaluate EEGNet with mixup augmentation during training."""

    return fit_mixup_eegnet(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        mixup_alpha=mixup_alpha,
    ).result


def run_confidence_penalty_eegnet(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    confidence_penalty_beta: float = 0.05,
) -> DeepBaselineResult:
    """Train and evaluate EEGNet with confidence-penalty regularization."""

    return fit_confidence_penalty_eegnet(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        confidence_penalty_beta=confidence_penalty_beta,
    ).result


def run_bottleneck_eegnet(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    bottleneck_dim: int = 8,
) -> DeepBaselineResult:
    """Train and evaluate EEGNet with a low-dimensional feature bottleneck."""

    return fit_bottleneck_eegnet(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        bottleneck_dim=bottleneck_dim,
    ).result
