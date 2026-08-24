"""Subject-adversarial EEGNet baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from eeg_privacy_benchmark.models.eegnet import (
    EEGNetTrainingArtifacts,
    _build_eegnet_model,
    _batched_model_forward,
    _build_validation_indices,
    _import_model_dependencies,
    _normalize_features,
    _set_all_seeds,
)
from eeg_privacy_benchmark.datasets.base import DatasetManifest
from eeg_privacy_benchmark.datasets.factory import build_dataset_loader
from eeg_privacy_benchmark.datasets.splits import (
    generate_cross_run_split,
    generate_cross_session_split,
    generate_cross_subject_split,
)


@dataclass(frozen=True)
class AdversarialBaselineResult:
    dataset_key: str
    protocol: str
    seed: int
    train_trials: int
    validation_trials: int
    test_trials: int
    epochs_requested: int
    epochs_trained: int
    adversarial_weight: float
    adversarial_schedule: str
    ramp_up_fraction: float
    best_validation_loss: float
    task_balanced_accuracy: float
    task_macro_f1: float
    history: tuple[dict[str, float | int], ...]

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


class _GradientReversal:
    @staticmethod
    def build(torch):
        class GradientReversalFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inputs, scale):
                ctx.scale = scale
                return inputs.view_as(inputs)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output.neg() * ctx.scale, None

        return GradientReversalFunction


def _compute_adversarial_scale(
    *,
    epoch_idx: int,
    total_epochs: int,
    adversarial_weight: float,
    adversarial_schedule: str,
    ramp_up_fraction: float,
) -> float:
    if adversarial_schedule == "constant":
        return adversarial_weight
    if adversarial_schedule != "linear_ramp":
        raise ValueError(f"Unsupported adversarial schedule: {adversarial_schedule}")

    if not 0.0 < ramp_up_fraction <= 1.0:
        raise ValueError("ramp_up_fraction must be between 0 and 1.")

    ramp_epochs = max(1, int(round(total_epochs * ramp_up_fraction)))
    progress = min((epoch_idx + 1) / ramp_epochs, 1.0)
    return float(adversarial_weight * progress)


def fit_adversarial_eegnet(
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
    adversarial_weight: float = 0.5,
    adversarial_schedule: str = "constant",
    ramp_up_fraction: float = 0.4,
) -> EEGNetTrainingArtifacts:
    """Train EEGNet with a subject-adversarial head."""

    deps = _import_model_dependencies()
    np = deps["np"]
    torch = deps["torch"]
    nn = deps["nn"]
    _set_all_seeds(torch, np, seed)

    loader = build_dataset_loader(dataset_key)
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
        raise ValueError(f"Unsupported protocol for adversarial EEGNet baseline: {protocol}")

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

    task_encoder = deps["LabelEncoder"]()
    encoded_task_labels = task_encoder.fit_transform(loaded.labels)
    train_indices, validation_indices = _build_validation_indices(
        np,
        train_indices,
        encoded_task_labels,
        seed,
        validation_fraction,
    )

    features = np.asarray(loaded.features, dtype=np.float32)
    features, _, _ = _normalize_features(np, features, train_indices)

    subject_encoder = deps["LabelEncoder"]()
    encoded_subject_labels = subject_encoder.fit_transform(
        [record.subject_id for record in loaded.trial_records]
    )

    x_train = torch.tensor(features[train_indices][:, None, :, :], dtype=torch.float32)
    y_train_task = torch.tensor(encoded_task_labels[train_indices], dtype=torch.long)
    y_train_subject = torch.tensor(encoded_subject_labels[train_indices], dtype=torch.long)
    y_validation_task = torch.tensor(
        encoded_task_labels[validation_indices], dtype=torch.long
    )
    y_test_task = torch.tensor(encoded_task_labels[test_indices], dtype=torch.long)

    TensorDataset = deps["TensorDataset"]
    DataLoader = deps["DataLoader"]
    train_loader = DataLoader(
        TensorDataset(x_train, y_train_task, y_train_subject),
        batch_size=batch_size,
        shuffle=True,
    )

    base_model = _build_eegnet_model(
        torch,
        nn,
        n_channels=x_train.shape[2],
        n_times=x_train.shape[3],
        n_classes=len(task_encoder.classes_),
    )
    feature_dim = base_model.classifier.in_features
    subject_head = nn.Linear(feature_dim, len(subject_encoder.classes_))

    optimizer = torch.optim.Adam(
        list(base_model.parameters()) + list(subject_head.parameters()),
        lr=learning_rate,
    )
    task_loss_fn = nn.CrossEntropyLoss()
    subject_loss_fn = nn.CrossEntropyLoss()
    gradient_reversal = _GradientReversal.build(torch)

    best_state = None
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    epochs_trained = 0
    history: list[dict[str, float | int]] = []

    for epoch_idx in range(epochs):
        current_adversarial_scale = _compute_adversarial_scale(
            epoch_idx=epoch_idx,
            total_epochs=epochs,
            adversarial_weight=adversarial_weight,
            adversarial_schedule=adversarial_schedule,
            ramp_up_fraction=ramp_up_fraction,
        )
        base_model.train()
        subject_head.train()
        running_task_loss = 0.0
        running_subject_loss = 0.0
        batch_count = 0
        for batch_x, batch_task_labels, batch_subject_labels in train_loader:
            optimizer.zero_grad()
            features_batch = base_model.forward_features(batch_x)
            task_logits = base_model.classifier(features_batch)
            subject_logits = subject_head(
                gradient_reversal.apply(features_batch, current_adversarial_scale)
            )
            task_loss = task_loss_fn(task_logits, batch_task_labels)
            subject_loss = subject_loss_fn(subject_logits, batch_subject_labels)
            loss = task_loss + subject_loss
            loss.backward()
            optimizer.step()
            running_task_loss += float(task_loss.item())
            running_subject_loss += float(subject_loss.item())
            batch_count += 1

        base_model.eval()
        with torch.no_grad():
            validation_task_logits = _batched_model_forward(
                torch,
                base_model,
                features[validation_indices],
                batch_size=batch_size,
            )
            validation_loss = float(
                task_loss_fn(validation_task_logits, y_validation_task).item()
            )
            validation_predictions = validation_task_logits.argmax(dim=1).cpu().numpy()
            validation_balanced_accuracy = float(
                deps["balanced_accuracy_score"](
                    y_validation_task.cpu().numpy(),
                    validation_predictions,
                )
            )

        history.append(
            {
                "epoch": epoch_idx + 1,
                "adversarial_scale": current_adversarial_scale,
                "train_task_loss": running_task_loss / max(batch_count, 1),
                "train_subject_loss": running_subject_loss / max(batch_count, 1),
                "validation_task_loss": validation_loss,
                "validation_task_balanced_accuracy": validation_balanced_accuracy,
            }
        )

        epochs_trained = epoch_idx + 1
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                "base_model": {
                    key: value.detach().cpu().clone()
                    for key, value in base_model.state_dict().items()
                },
                "subject_head": {
                    key: value.detach().cpu().clone()
                    for key, value in subject_head.state_dict().items()
                },
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("Adversarial EEGNet training did not produce a checkpoint.")

    base_model.load_state_dict(best_state["base_model"])
    subject_head.load_state_dict(best_state["subject_head"])

    base_model.eval()
    with torch.no_grad():
        test_task_logits = _batched_model_forward(
            torch,
            base_model,
            features[test_indices],
            batch_size=batch_size,
        )
        task_predictions = test_task_logits.argmax(dim=1).cpu().numpy()

    result = AdversarialBaselineResult(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        train_trials=len(train_indices),
        validation_trials=len(validation_indices),
        test_trials=len(test_indices),
        epochs_requested=epochs,
        epochs_trained=epochs_trained,
        adversarial_weight=adversarial_weight,
        adversarial_schedule=adversarial_schedule,
        ramp_up_fraction=ramp_up_fraction,
        best_validation_loss=best_validation_loss,
        task_balanced_accuracy=float(
            deps["balanced_accuracy_score"](y_test_task.cpu().numpy(), task_predictions)
        ),
        task_macro_f1=float(
            deps["f1_score"](y_test_task.cpu().numpy(), task_predictions, average="macro")
        ),
        history=tuple(history),
    )

    defense_model = nn.Module()
    defense_model.base_model = base_model
    defense_model.subject_head = subject_head

    return EEGNetTrainingArtifacts(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        model=defense_model,
        result=result,
        all_features=features,
        encoded_labels=encoded_task_labels,
        trial_records=loaded.trial_records,
        train_indices=tuple(train_indices),
        validation_indices=tuple(validation_indices),
        test_indices=tuple(test_indices),
    )
