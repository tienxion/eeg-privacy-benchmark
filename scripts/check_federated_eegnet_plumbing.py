from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_privacy_benchmark.cli import make_parser
from eeg_privacy_benchmark.datasets.base import LoadedArrayDataset, TrialRecord
from eeg_privacy_benchmark.models.eegnet_federated import (
    fedavg_state_dicts,
    federated_normalize_features,
    fit_federated_eegnet_loaded,
    normalize_federated_features,
    partition_indices_by_subject,
)
from eeg_privacy_benchmark.privacy import (
    load_posterior_feature_cache,
    run_federated_eegnet_membership_inference_from_artifacts,
    run_federated_eegnet_subject_id_from_artifacts,
)


def _build_toy_dataset():
    import numpy as np

    rng = np.random.default_rng(2026)
    features = []
    labels = []
    records = []
    time = np.linspace(0.0, 1.0, 128, endpoint=False)
    class_wave = np.sin(2.0 * np.pi * 10.0 * time).astype(np.float32)
    for subject in range(1, 6):
        for trial in range(10):
            label = "left_hand" if trial % 2 == 0 else "right_hand"
            direction = 1.0 if label == "left_hand" else -1.0
            sample = rng.normal(0.0, 0.15, size=(3, 128)).astype(np.float32)
            sample[0] += direction * class_wave
            sample[1] += subject * 0.02
            features.append(sample)
            labels.append(label)
            records.append(
                TrialRecord(
                    dataset="federated_toy",
                    subject_id=f"S{subject:02d}",
                    session_id="session_1" if trial < 5 else "session_2",
                    run_id="run_1",
                    trial_id=f"S{subject:02d}_T{trial:02d}",
                    raw_label=label,
                    canonical_label=label,
                )
            )
    return LoadedArrayDataset(
        dataset_key="federated_toy",
        features=np.stack(features),
        labels=np.asarray(labels),
        trial_records=tuple(records),
    )


def check_weighted_fedavg() -> None:
    import torch

    first = {
        "weight": torch.tensor([1.0, 3.0]),
        "counter": torch.tensor(2, dtype=torch.long),
    }
    second = {
        "weight": torch.tensor([5.0, 7.0]),
        "counter": torch.tensor(9, dtype=torch.long),
    }
    aggregate = fedavg_state_dicts(torch, [(1, first), (3, second)])
    torch.testing.assert_close(aggregate["weight"], torch.tensor([4.0, 6.0]))
    if aggregate["counter"].item() != 9:
        raise AssertionError("non-floating state did not come from largest client")


def check_subject_partition_is_deterministic() -> None:
    loaded = _build_toy_dataset()
    selected = [12, 2, 11, 1]
    clients = partition_indices_by_subject(loaded.trial_records, selected)
    if clients != {"S01": (1, 2), "S02": (11, 12)}:
        raise AssertionError(f"unexpected client partition: {clients}")


def check_federated_normalization_matches_pooled_statistics() -> None:
    import numpy as np

    loaded = _build_toy_dataset()
    features = np.asarray(loaded.features, dtype=np.float32)
    train_indices = list(range(20))
    clients = partition_indices_by_subject(loaded.trial_records, train_indices)
    normalized, mean, std, communication_bytes = federated_normalize_features(
        np,
        features,
        clients,
    )
    pooled = features[train_indices]
    expected_mean = pooled.mean(axis=(0, 2), keepdims=True)
    expected_std = pooled.std(axis=(0, 2), keepdims=True)
    expected_std = np.where(expected_std < 1e-6, 1.0, expected_std)
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(std, expected_std, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        normalized,
        (features - expected_mean) / expected_std,
        rtol=1e-5,
        atol=1e-5,
    )
    if communication_bytes <= 0:
        raise AssertionError("normalization communication was not accounted for")


def check_compact_federated_path() -> None:
    import numpy as np
    import torch

    loaded = _build_toy_dataset()
    normalized, _, _, communication_bytes = normalize_federated_features(
        np,
        loaded.features,
        {"all": tuple(range(len(loaded.features)))},
        "per_trial_channel",
    )
    np.testing.assert_allclose(normalized.mean(axis=2), 0.0, atol=1e-5)
    np.testing.assert_allclose(normalized.std(axis=2), 1.0, atol=1e-5)
    if communication_bytes != 0:
        raise AssertionError("per-trial normalization should require no exchange")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        artifacts = fit_federated_eegnet_loaded(
            loaded,
            protocol="cross_session",
            seed=31,
            rounds=1,
            local_epochs=1,
            batch_size=8,
            early_stopping_patience=1,
            resample_hz=250,
            normalization="per_trial_channel",
            classifier_head="global_average",
            bottleneck_dim=6,
            confidence_penalty_beta=0.05,
        )
        attack = run_federated_eegnet_membership_inference_from_artifacts(
            artifacts,
            task_model="compact_federated_bottleneck_eegnet",
        )
        probe = run_federated_eegnet_subject_id_from_artifacts(
            artifacts,
            task_model="compact_federated_bottleneck_eegnet",
        )
    finally:
        torch.set_num_threads(previous_threads)

    result = artifacts.result
    if result.normalization != "per_trial_channel":
        raise AssertionError("compact federated normalization provenance was lost")
    if result.normalization_communication_bytes != 0:
        raise AssertionError("compact normalization communication must be zero")
    if result.classifier_head != "global_average" or result.bottleneck_dim != 6:
        raise AssertionError("compact federated architecture provenance was lost")
    if result.resample_hz != 250:
        raise AssertionError("compact federated sampling provenance was lost")
    if result.confidence_penalty_beta != 0.05:
        raise AssertionError("compact federated confidence penalty was lost")
    if attack.attack_seed_policy != "task_seed_default":
        raise AssertionError("compact federated attack lost attacker-seed policy")
    for payload in (attack, probe):
        if payload.task_model != "compact_federated_bottleneck_eegnet":
            raise AssertionError("compact federated model identity was lost")
        if payload.normalization != "per_trial_channel":
            raise AssertionError("compact privacy result lost normalization")
        if payload.classifier_head != "global_average":
            raise AssertionError("compact privacy result lost classifier head")
        if payload.resample_hz != 250 or payload.bottleneck_dim != 6:
            raise AssertionError("compact privacy result lost architecture metadata")
        if payload.confidence_penalty_beta != 0.05:
            raise AssertionError("compact privacy result lost confidence penalty")


def check_cli_contract() -> None:
    args = make_parser().parse_args(
        [
            "run-federated-eegnet",
            "--dataset",
            "bnci2014_001",
            "--protocol",
            "cross_subject",
            "--rounds",
            "7",
            "--local-epochs",
            "2",
            "--checkpoint",
            "/tmp/federated.pt",
            "--checkpoint-interval",
            "3",
            "--resample-hz",
            "250",
            "--normalization",
            "per_trial_channel",
            "--classifier-head",
            "global_average",
            "--bottleneck-dim",
            "6",
            "--confidence-penalty-beta",
            "0.05",
        ]
    )
    if args.rounds != 7 or args.local_epochs != 2:
        raise AssertionError("federated CLI did not preserve round configuration")
    if args.checkpoint != "/tmp/federated.pt" or args.checkpoint_interval != 3:
        raise AssertionError("federated CLI did not preserve checkpoint configuration")
    if (
        args.resample_hz != 250
        or args.normalization != "per_trial_channel"
        or args.classifier_head != "global_average"
        or args.bottleneck_dim != 6
        or args.confidence_penalty_beta != 0.05
    ):
        raise AssertionError("federated CLI lost compact architecture options")


def check_end_to_end_training() -> None:
    import torch

    loaded = _build_toy_dataset()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        artifacts = fit_federated_eegnet_loaded(
            loaded,
            protocol="cross_subject",
            seed=13,
            rounds=2,
            local_epochs=1,
            batch_size=8,
            learning_rate=1e-3,
            validation_fraction=0.2,
            early_stopping_patience=2,
        )
    finally:
        torch.set_num_threads(previous_threads)

    train = set(artifacts.train_indices)
    validation = set(artifacts.validation_indices)
    test = set(artifacts.test_indices)
    if train & validation or train & test or validation & test:
        raise AssertionError("federated train/validation/test partitions overlap")
    client_union = set().union(*map(set, artifacts.client_indices.values()))
    if client_union != train:
        raise AssertionError("client partitions do not exactly cover task training trials")
    if sum(artifacts.result.client_trial_counts.values()) != len(train):
        raise AssertionError("reported client trial counts do not cover training data")
    if artifacts.result.clients != len(artifacts.client_indices):
        raise AssertionError("reported client count differs from client artifacts")
    expected_communication = (
        2
        * artifacts.result.model_state_bytes
        * artifacts.result.clients
        * artifacts.result.rounds_trained
        + artifacts.result.normalization_communication_bytes
    )
    if artifacts.result.estimated_communication_bytes != expected_communication:
        raise AssertionError("communication estimate is inconsistent")
    if not 0.0 <= artifacts.result.balanced_accuracy <= 1.0:
        raise AssertionError("task balanced accuracy is outside [0, 1]")
    json.dumps(artifacts.result.to_dict())

    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        cache_path = Path(tmp) / "federated_posteriors.npz"
        attack = run_federated_eegnet_membership_inference_from_artifacts(
            artifacts,
            attack_type="threshold",
            score_type="label_known_log_probability",
            posterior_cache_output=cache_path,
        )
        cached = load_posterior_feature_cache(cache_path)
    if attack.task_model != "federated_eegnet":
        raise AssertionError("membership attack lost federated model identity")
    if attack.federated_clients != artifacts.result.clients:
        raise AssertionError("membership result lost federated client count")
    if cached.federated_rounds_trained != artifacts.result.rounds_trained:
        raise AssertionError("posterior cache lost federated round count")
    if artifacts.result.best_checkpoint_round not in range(
        1, artifacts.result.rounds_trained + 1
    ):
        raise AssertionError("best federated checkpoint round is out of range")
    if (
        cached.federated_best_checkpoint_round
        != artifacts.result.best_checkpoint_round
    ):
        raise AssertionError("posterior cache lost best checkpoint round")
    if (
        attack.federated_best_checkpoint_round
        != artifacts.result.best_checkpoint_round
    ):
        raise AssertionError("membership result lost best checkpoint round")


def check_cross_session_subject_probe() -> None:
    import torch

    loaded = _build_toy_dataset()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        artifacts = fit_federated_eegnet_loaded(
            loaded,
            protocol="cross_session",
            seed=13,
            rounds=1,
            local_epochs=1,
            batch_size=8,
            validation_fraction=0.2,
            early_stopping_patience=1,
        )
        probe = run_federated_eegnet_subject_id_from_artifacts(artifacts)
    finally:
        torch.set_num_threads(previous_threads)
    if probe.task_model != "federated_eegnet":
        raise AssertionError("subject probe lost federated model identity")
    if probe.num_subjects != 5:
        raise AssertionError(f"subject probe expected 5 subjects, found {probe.num_subjects}")
    if probe.federated_clients != 5:
        raise AssertionError("subject probe did not report all five clients")


def check_checkpoint_resume_matches_uninterrupted_training() -> None:
    import torch

    loaded = _build_toy_dataset()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        uninterrupted = fit_federated_eegnet_loaded(
            loaded,
            protocol="cross_subject",
            seed=29,
            rounds=2,
            local_epochs=1,
            batch_size=8,
            early_stopping_patience=10,
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            checkpoint_path = Path(tmp) / "federated_checkpoint.pt"
            fit_federated_eegnet_loaded(
                loaded,
                protocol="cross_subject",
                seed=29,
                rounds=1,
                local_epochs=1,
                batch_size=8,
                early_stopping_patience=10,
                checkpoint_path=checkpoint_path,
            )
            resumed = fit_federated_eegnet_loaded(
                loaded,
                protocol="cross_subject",
                seed=29,
                rounds=2,
                local_epochs=1,
                batch_size=8,
                early_stopping_patience=10,
                checkpoint_path=checkpoint_path,
            )
    finally:
        torch.set_num_threads(previous_threads)

    if uninterrupted.result.history != resumed.result.history:
        raise AssertionError("resumed round history differs from uninterrupted training")
    if uninterrupted.result.to_dict() != resumed.result.to_dict():
        raise AssertionError("resumed metrics differ from uninterrupted training")
    for key, expected in uninterrupted.model.state_dict().items():
        torch.testing.assert_close(expected, resumed.model.state_dict()[key])


def main() -> None:
    checks = [
        ("weighted FedAvg", check_weighted_fedavg),
        ("subject partition", check_subject_partition_is_deterministic),
        (
            "federated normalization",
            check_federated_normalization_matches_pooled_statistics,
        ),
        ("compact federated path", check_compact_federated_path),
        ("CLI contract", check_cli_contract),
        ("end-to-end training", check_end_to_end_training),
        ("cross-session subject probe", check_cross_session_subject_probe),
        (
            "checkpoint resume",
            check_checkpoint_resume_matches_uninterrupted_training,
        ),
    ]
    for label, check in checks:
        check()
        print(f"[PASS] {label}")
    print(f"federated_eegnet_checks={len(checks)}")


if __name__ == "__main__":
    main()
