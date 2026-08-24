"""Subject-partitioned federated EEGNet baseline using FedAvg."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import random

from eeg_privacy_benchmark.datasets.base import DatasetManifest, LoadedArrayDataset
from eeg_privacy_benchmark.datasets.factory import build_dataset_loader
from eeg_privacy_benchmark.datasets.splits import (
    generate_cross_run_split,
    generate_cross_session_split,
    generate_cross_subject_split,
)
from eeg_privacy_benchmark.models.eegnet import (
    _batched_model_forward,
    _build_eegnet_model,
    _build_validation_indices,
    _import_model_dependencies,
    _set_all_seeds,
)


@dataclass(frozen=True)
class FederatedRoundResult:
    round: int
    mean_local_loss: float
    validation_loss: float
    validation_balanced_accuracy: float


@dataclass(frozen=True)
class FederatedEEGNetResult:
    dataset_key: str
    protocol: str
    seed: int
    aggregation: str
    client_partition: str
    normalization: str
    clients: int
    client_trial_counts: dict[str, int]
    train_trials: int
    validation_trials: int
    test_trials: int
    rounds_requested: int
    rounds_trained: int
    local_epochs: int
    best_checkpoint_round: int
    best_validation_loss: float
    best_validation_balanced_accuracy: float
    checkpoint_validation_balanced_accuracy: float
    balanced_accuracy: float
    macro_f1: float
    model_state_bytes: int
    normalization_communication_bytes: int
    estimated_communication_bytes: int
    history: tuple[FederatedRoundResult, ...]
    resample_hz: float | None = None
    classifier_head: str = "flatten"
    bottleneck_dim: int | None = None
    confidence_penalty_beta: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FederatedEEGNetTrainingArtifacts:
    dataset_key: str
    protocol: str
    seed: int
    model: object
    result: FederatedEEGNetResult
    all_features: object
    encoded_labels: object
    trial_records: tuple
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    client_indices: dict[str, tuple[int, ...]]


def _build_split_indices(
    loaded: LoadedArrayDataset,
    *,
    notes: str,
    protocol: str,
    seed: int,
) -> tuple[list[int], list[int]]:
    manifest = DatasetManifest(
        dataset_key=loaded.dataset_key,
        trial_records=loaded.trial_records,
        available_labels=tuple(sorted({str(label) for label in loaded.labels})),
        notes=notes,
    )
    if protocol == "cross_subject":
        split_manifest = generate_cross_subject_split(manifest, seed=seed)
    elif protocol == "cross_session":
        split_manifest = generate_cross_session_split(manifest, seed=seed)
    elif protocol == "cross_run":
        split_manifest = generate_cross_run_split(manifest, seed=seed)
    else:
        raise ValueError(f"Unsupported protocol for federated EEGNet: {protocol}")

    split_lookup = {
        assignment.trial_id: assignment.split
        for assignment in split_manifest.assignments
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
    return train_indices, test_indices


def partition_indices_by_subject(
    trial_records: tuple,
    train_indices: list[int],
) -> dict[str, tuple[int, ...]]:
    """Create deterministic subject clients from task-training trials only."""

    grouped: dict[str, list[int]] = {}
    for index in train_indices:
        grouped.setdefault(trial_records[index].subject_id, []).append(index)
    return {
        subject_id: tuple(sorted(grouped[subject_id]))
        for subject_id in sorted(grouped)
        if grouped[subject_id]
    }


def federated_normalize_features(np, features, client_indices):
    """Normalize from aggregated client sums without pooling raw client trials."""

    if not client_indices:
        raise ValueError("Federated normalization requires at least one client.")
    channel_sum = np.zeros(features.shape[1], dtype=np.float64)
    channel_square_sum = np.zeros(features.shape[1], dtype=np.float64)
    observation_count = 0
    for indices in client_indices.values():
        local = features[list(indices)]
        channel_sum += local.sum(axis=(0, 2), dtype=np.float64)
        channel_square_sum += np.square(local, dtype=np.float64).sum(axis=(0, 2))
        observation_count += local.shape[0] * local.shape[2]
    if observation_count == 0:
        raise ValueError("Federated normalization received no observations.")

    mean = channel_sum / observation_count
    variance = np.maximum(channel_square_sum / observation_count - mean**2, 0.0)
    std = np.sqrt(variance)
    std = np.where(std < 1e-6, 1.0, std)
    broadcast_mean = mean.reshape(1, -1, 1)
    broadcast_std = std.reshape(1, -1, 1)
    normalized = (features - broadcast_mean) / broadcast_std

    client_count = len(client_indices)
    uplink_bytes = client_count * (2 * features.shape[1] + 1) * 8
    broadcast_bytes = client_count * 2 * features.shape[1] * 4
    return (
        normalized.astype(np.float32),
        broadcast_mean.astype(np.float32),
        broadcast_std.astype(np.float32),
        uplink_bytes + broadcast_bytes,
    )


def normalize_federated_features(np, features, client_indices, normalization):
    """Apply the registered federated or trial-local normalization policy."""

    if normalization == "federated_sufficient_statistics":
        return federated_normalize_features(np, features, client_indices)
    if normalization == "per_trial_channel":
        mean = features.mean(axis=2, keepdims=True)
        std = features.std(axis=2, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        normalized = (features - mean) / std
        return (
            normalized.astype(np.float32),
            mean.astype(np.float32),
            std.astype(np.float32),
            0,
        )
    raise ValueError(
        "normalization must be 'federated_sufficient_statistics' or "
        "'per_trial_channel'."
    )


def fedavg_state_dicts(torch, weighted_states: list[tuple[int, dict]]) -> dict:
    """Aggregate model states by sample count, preserving non-floating dtypes."""

    if not weighted_states:
        raise ValueError("FedAvg requires at least one client state.")
    if any(weight <= 0 for weight, _ in weighted_states):
        raise ValueError("FedAvg client weights must be positive.")

    reference_keys = tuple(weighted_states[0][1])
    for _, state in weighted_states[1:]:
        if tuple(state) != reference_keys:
            raise ValueError("FedAvg client state dictionaries have different keys.")

    total_weight = sum(weight for weight, _ in weighted_states)
    largest_client_state = max(
        enumerate(weighted_states),
        key=lambda item: (item[1][0], -item[0]),
    )[1][1]
    aggregated = {}
    for key in reference_keys:
        tensors = [state[key] for _, state in weighted_states]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError(f"FedAvg tensor shape mismatch for {key}.")
        if torch.is_floating_point(tensors[0]) or torch.is_complex(tensors[0]):
            value = torch.zeros_like(tensors[0])
            for weight, state in weighted_states:
                value.add_(state[key], alpha=weight / total_weight)
            aggregated[key] = value
        else:
            aggregated[key] = largest_client_state[key].clone()
    return aggregated


def _train_local_model(
    deps,
    global_model,
    features,
    encoded_labels,
    indices: tuple[int, ...],
    *,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    loader_seed: int,
    confidence_penalty_beta: float,
) -> tuple[dict, float]:
    torch = deps["torch"]
    local_model = deepcopy(global_model)
    x_local = torch.tensor(
        features[list(indices)][:, None, :, :],
        dtype=torch.float32,
    )
    y_local = torch.tensor(encoded_labels[list(indices)], dtype=torch.long)
    generator = torch.Generator().manual_seed(loader_seed)
    loader = deps["DataLoader"](
        deps["TensorDataset"](x_local, y_local),
        batch_size=min(batch_size, len(indices)),
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(local_model.parameters(), lr=learning_rate)
    loss_fn = deps["nn"].CrossEntropyLoss()
    loss_total = 0.0
    observations = 0
    local_model.train()
    for _ in range(local_epochs):
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            logits = local_model(batch_features)
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
            batch_observations = len(batch_labels)
            loss_total += float(loss.item()) * batch_observations
            observations += batch_observations
    state = {
        key: value.detach().cpu().clone()
        for key, value in local_model.state_dict().items()
    }
    return state, loss_total / observations


def _load_checkpoint(torch, path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save_checkpoint(
    torch,
    np,
    path: Path,
    *,
    config: dict,
    completed_rounds: int,
    model,
    best_state: dict,
    best_validation_loss: float,
    best_validation_balanced_accuracy: float,
    rounds_without_improvement: int,
    history: list[FederatedRoundResult],
) -> None:
    payload = {
        "schema_version": 1,
        "config": config,
        "completed_rounds": completed_rounds,
        "model_state": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "best_state": best_state,
        "best_validation_loss": best_validation_loss,
        "best_validation_balanced_accuracy": best_validation_balanced_accuracy,
        "rounds_without_improvement": rounds_without_improvement,
        "history": [asdict(result) for result in history],
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def fit_federated_eegnet_loaded(
    loaded: LoadedArrayDataset,
    *,
    notes: str = "",
    protocol: str,
    seed: int,
    rounds: int = 20,
    local_epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    checkpoint_path: str | Path | None = None,
    checkpoint_interval: int = 1,
    resume_from_checkpoint: bool = True,
    resample_hz: float | None = None,
    normalization: str = "federated_sufficient_statistics",
    classifier_head: str = "flatten",
    bottleneck_dim: int | None = None,
    confidence_penalty_beta: float = 0.0,
) -> FederatedEEGNetTrainingArtifacts:
    """Fit all-client FedAvg on an already loaded EEG dataset."""

    if rounds <= 0:
        raise ValueError("rounds must be positive.")
    if local_epochs <= 0:
        raise ValueError("local_epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive.")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive.")
    if confidence_penalty_beta < 0.0:
        raise ValueError("confidence_penalty_beta cannot be negative.")

    deps = _import_model_dependencies()
    np = deps["np"]
    torch = deps["torch"]
    _set_all_seeds(torch, np, seed)

    train_indices, test_indices = _build_split_indices(
        loaded,
        notes=notes,
        protocol=protocol,
        seed=seed,
    )
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
    client_indices = partition_indices_by_subject(
        loaded.trial_records,
        train_indices,
    )
    if len(client_indices) < 2:
        raise ValueError(
            "Federated EEGNet requires at least two subject clients after validation splitting."
        )
    features, _, _, normalization_communication_bytes = normalize_federated_features(
        np,
        features,
        client_indices,
        normalization,
    )

    model = _build_eegnet_model(
        torch,
        deps["nn"],
        n_channels=features.shape[1],
        n_times=features.shape[2],
        n_classes=len(label_encoder.classes_),
        classifier_head=classifier_head,
        bottleneck_dim=bottleneck_dim,
    )
    loss_fn = deps["nn"].CrossEntropyLoss()
    y_validation = torch.tensor(
        encoded_labels[validation_indices],
        dtype=torch.long,
    )
    y_test = torch.tensor(encoded_labels[test_indices], dtype=torch.long)

    best_state = None
    best_validation_loss = float("inf")
    best_validation_balanced_accuracy = 0.0
    rounds_without_improvement = 0
    history: list[FederatedRoundResult] = []
    start_round = 0
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint_config = {
        "dataset_key": loaded.dataset_key,
        "protocol": protocol,
        "seed": seed,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "validation_fraction": validation_fraction,
        "early_stopping_patience": early_stopping_patience,
        "feature_shape": tuple(features.shape[1:]),
        "class_count": len(label_encoder.classes_),
        "client_trial_counts": {
            client_id: len(indices) for client_id, indices in client_indices.items()
        },
        "resample_hz": resample_hz,
        "normalization": normalization,
        "classifier_head": classifier_head,
        "bottleneck_dim": bottleneck_dim,
        "confidence_penalty_beta": confidence_penalty_beta,
    }
    if checkpoint is not None and checkpoint.exists() and resume_from_checkpoint:
        payload = _load_checkpoint(torch, checkpoint)
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported federated checkpoint schema version.")
        if payload.get("config") != checkpoint_config:
            raise ValueError("Federated checkpoint configuration does not match this run.")
        start_round = int(payload["completed_rounds"])
        if start_round > rounds:
            raise ValueError(
                f"Checkpoint has {start_round} rounds but only {rounds} were requested."
            )
        model.load_state_dict(payload["model_state"])
        best_state = payload["best_state"]
        best_validation_loss = float(payload["best_validation_loss"])
        best_validation_balanced_accuracy = float(
            payload["best_validation_balanced_accuracy"]
        )
        rounds_without_improvement = int(payload["rounds_without_improvement"])
        history = [FederatedRoundResult(**row) for row in payload["history"]]
        random.setstate(payload["python_random_state"])
        np.random.set_state(payload["numpy_random_state"])
        torch.set_rng_state(payload["torch_random_state"])
        if torch.cuda.is_available() and payload["torch_cuda_random_states"] is not None:
            torch.cuda.set_rng_state_all(payload["torch_cuda_random_states"])

    resume_stop = (
        start_round > 0
        and rounds_without_improvement >= early_stopping_patience
    )
    round_stop = start_round if resume_stop else rounds
    for round_index in range(start_round, round_stop):
        weighted_states = []
        weighted_local_loss = 0.0
        for client_offset, indices in enumerate(client_indices.values()):
            state, local_loss = _train_local_model(
                deps,
                model,
                features,
                encoded_labels,
                indices,
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                loader_seed=seed + (round_index + 1) * 10_000 + client_offset,
                confidence_penalty_beta=confidence_penalty_beta,
            )
            client_weight = len(indices)
            weighted_states.append((client_weight, state))
            weighted_local_loss += local_loss * client_weight

        model.load_state_dict(fedavg_state_dicts(torch, weighted_states))
        model.eval()
        with torch.no_grad():
            validation_logits = _batched_model_forward(
                torch,
                model,
                features[validation_indices],
                batch_size=batch_size,
            )
            validation_loss = float(
                loss_fn(validation_logits, y_validation).item()
            )
            validation_predictions = validation_logits.argmax(dim=1).cpu().numpy()
            validation_balanced_accuracy = float(
                deps["balanced_accuracy_score"](
                    y_validation.cpu().numpy(),
                    validation_predictions,
                )
            )
        history.append(
            FederatedRoundResult(
                round=round_index + 1,
                mean_local_loss=weighted_local_loss / len(train_indices),
                validation_loss=validation_loss,
                validation_balanced_accuracy=validation_balanced_accuracy,
            )
        )

        should_stop = False
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_balanced_accuracy = validation_balanced_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
            if rounds_without_improvement >= early_stopping_patience:
                should_stop = True

        completed_rounds = round_index + 1
        if checkpoint is not None and (
            completed_rounds % checkpoint_interval == 0
            or completed_rounds == rounds
            or should_stop
        ):
            _save_checkpoint(
                torch,
                np,
                checkpoint,
                config=checkpoint_config,
                completed_rounds=completed_rounds,
                model=model,
                best_state=best_state,
                best_validation_loss=best_validation_loss,
                best_validation_balanced_accuracy=best_validation_balanced_accuracy,
                rounds_without_improvement=rounds_without_improvement,
                history=history,
            )
        if should_stop:
            break

    if best_state is None:
        raise RuntimeError("Federated EEGNet training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = _batched_model_forward(
            torch,
            model,
            features[test_indices],
            batch_size=batch_size,
        )
        test_predictions = test_logits.argmax(dim=1).cpu().numpy()

    model_state_bytes = sum(
        value.numel() * value.element_size()
        for value in model.state_dict().values()
    )
    rounds_trained = len(history)
    best_checkpoint_round = min(
        history,
        key=lambda round_result: round_result.validation_loss,
    ).round
    estimated_communication_bytes = (
        2 * model_state_bytes * len(client_indices) * rounds_trained
        + normalization_communication_bytes
    )
    result = FederatedEEGNetResult(
        dataset_key=loaded.dataset_key,
        protocol=protocol,
        seed=seed,
        aggregation="fedavg_sample_weighted",
        client_partition="subject_id",
        normalization=normalization,
        clients=len(client_indices),
        client_trial_counts={
            client_id: len(indices) for client_id, indices in client_indices.items()
        },
        train_trials=len(train_indices),
        validation_trials=len(validation_indices),
        test_trials=len(test_indices),
        rounds_requested=rounds,
        rounds_trained=rounds_trained,
        local_epochs=local_epochs,
        best_checkpoint_round=best_checkpoint_round,
        best_validation_loss=best_validation_loss,
        best_validation_balanced_accuracy=best_validation_balanced_accuracy,
        checkpoint_validation_balanced_accuracy=best_validation_balanced_accuracy,
        balanced_accuracy=float(
            deps["balanced_accuracy_score"](
                y_test.cpu().numpy(),
                test_predictions,
            )
        ),
        macro_f1=float(
            deps["f1_score"](
                y_test.cpu().numpy(),
                test_predictions,
                average="macro",
            )
        ),
        model_state_bytes=model_state_bytes,
        normalization_communication_bytes=normalization_communication_bytes,
        estimated_communication_bytes=estimated_communication_bytes,
        history=tuple(history),
        resample_hz=resample_hz,
        classifier_head=classifier_head,
        bottleneck_dim=bottleneck_dim,
        confidence_penalty_beta=(
            confidence_penalty_beta if confidence_penalty_beta > 0.0 else None
        ),
    )
    return FederatedEEGNetTrainingArtifacts(
        dataset_key=loaded.dataset_key,
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
        client_indices=client_indices,
    )


def fit_federated_eegnet(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    rounds: int = 20,
    local_epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    checkpoint_path: str | Path | None = None,
    checkpoint_interval: int = 1,
    resume_from_checkpoint: bool = True,
    resample_hz: float | None = None,
    normalization: str = "federated_sufficient_statistics",
    classifier_head: str = "flatten",
    bottleneck_dim: int | None = None,
    confidence_penalty_beta: float = 0.0,
) -> FederatedEEGNetTrainingArtifacts:
    loader = build_dataset_loader(dataset_key, resample_hz=resample_hz)
    loaded = loader.load_array_data(subjects=subjects)
    return fit_federated_eegnet_loaded(
        loaded,
        notes=loader.dataset_spec.notes,
        protocol=protocol,
        seed=seed,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=checkpoint_interval,
        resume_from_checkpoint=resume_from_checkpoint,
        resample_hz=resample_hz,
        normalization=normalization,
        classifier_head=classifier_head,
        bottleneck_dim=bottleneck_dim,
        confidence_penalty_beta=confidence_penalty_beta,
    )


def run_federated_eegnet(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    rounds: int = 20,
    local_epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    checkpoint_path: str | Path | None = None,
    checkpoint_interval: int = 1,
    resume_from_checkpoint: bool = True,
    resample_hz: float | None = None,
    normalization: str = "federated_sufficient_statistics",
    classifier_head: str = "flatten",
    bottleneck_dim: int | None = None,
    confidence_penalty_beta: float = 0.0,
) -> FederatedEEGNetResult:
    return fit_federated_eegnet(
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
        checkpoint_path=checkpoint_path,
        checkpoint_interval=checkpoint_interval,
        resume_from_checkpoint=resume_from_checkpoint,
        resample_hz=resample_hz,
        normalization=normalization,
        classifier_head=classifier_head,
        bottleneck_dim=bottleneck_dim,
        confidence_penalty_beta=confidence_penalty_beta,
    ).result
