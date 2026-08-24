from __future__ import annotations

import os
import json
import subprocess
import tempfile
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_privacy_benchmark.privacy.membership_inference import (
    PosteriorFeatureCache,
    _run_learned_membership_attack,
    _run_threshold_membership_attack,
    load_posterior_feature_cache,
    run_cached_membership_inference_attack,
    write_posterior_feature_cache,
)
from eeg_privacy_benchmark.experiments import (
    _single_attack_seed_policy,
    run_cached_membership_inference_sweep,
)


def _cli_help(*args: str) -> str:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    result = subprocess.run(
        [sys.executable, "-m", "eeg_privacy_benchmark.cli", *args, "--help"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def _check_cache_cli_help() -> None:
    top_level_help = _cli_help()
    for command in [
        "run-membership-inference-from-cache",
        "run-membership-inference-cache-sweep",
    ]:
        if command not in top_level_help:
            raise AssertionError(f"top-level CLI help is missing {command}")

    sweep_help = _cli_help("run-membership-inference-sweep")
    for flag in ["--posterior-cache-dir", "--attack-type", "--score-type"]:
        if flag not in sweep_help:
            raise AssertionError(f"membership sweep help is missing {flag}")

    cache_sweep_help = _cli_help("run-membership-inference-cache-sweep")
    for flag in [
        "--posterior-caches",
        "--posterior-cache-dir",
        "--attack-seed",
        "--output-dir",
    ]:
        if flag not in cache_sweep_help:
            raise AssertionError(f"cached sweep help is missing {flag}")

    single_cache_help = _cli_help("run-membership-inference-from-cache")
    for flag in ["--posterior-cache", "--attack-type", "--score-type", "--output"]:
        if flag not in single_cache_help:
            raise AssertionError(f"single-cache help is missing {flag}")

    print("PASS cached_cli_help")


def _check_cache_cli_rejects_missing_file() -> None:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "eeg_privacy_benchmark.cli",
            "run-membership-inference-cache-sweep",
            "--posterior-caches",
            "outputs/cache/does_not_exist.npz",
            "--attack-type",
            "threshold",
            "--score-type",
            "max_probability",
            "--output-dir",
            "outputs/cache_sweep_missing_file_should_fail",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    combined_output = result.stdout + result.stderr
    if result.returncode == 0:
        raise AssertionError("cached sweep CLI accepted a missing cache file")
    if "Posterior cache path does not exist" not in combined_output:
        raise AssertionError("cached sweep CLI failed with an unclear missing-file error")
    if "Traceback" in combined_output:
        raise AssertionError("cached sweep CLI emitted a traceback for a missing file")
    print("PASS cached_cli_rejects_missing_file")


def _metadata_payload(cache: PosteriorFeatureCache, **overrides) -> dict:
    payload = {
        "cache_schema_version": 1,
        "task_model": cache.task_model,
        "dataset_key": cache.dataset_key,
        "protocol": cache.protocol,
        "seed": cache.seed,
        "subjects": cache.subjects,
        "task_balanced_accuracy": cache.task_balanced_accuracy,
        "task_macro_f1": cache.task_macro_f1,
        "epochs_trained": cache.epochs_trained,
        "best_validation_loss": cache.best_validation_loss,
        "adversarial_weight": cache.adversarial_weight,
        "adversarial_schedule": cache.adversarial_schedule,
        "ramp_up_fraction": cache.ramp_up_fraction,
        "label_smoothing": cache.label_smoothing,
        "bottleneck_dim": cache.bottleneck_dim,
        "feature_noise_std": cache.feature_noise_std,
        "mixup_alpha": cache.mixup_alpha,
        "confidence_penalty_beta": cache.confidence_penalty_beta,
    }
    payload.update(overrides)
    return payload


def _write_malformed_index_cache(path: Path, cache: PosteriorFeatureCache) -> None:
    np.savez_compressed(
        path,
        metadata=json.dumps(_metadata_payload(cache), sort_keys=True),
        class_probabilities=np.asarray(cache.class_probabilities),
        encoded_labels=np.asarray(cache.encoded_labels),
        member_indices=np.asarray([0, 1, 999], dtype=int),
        nonmember_indices=np.asarray(cache.nonmember_indices, dtype=int),
    )


def _write_malformed_probability_cache(path: Path, cache: PosteriorFeatureCache) -> None:
    probabilities = np.asarray(cache.class_probabilities).copy()
    probabilities[0] = np.array([0.75, 0.75], dtype=float)
    np.savez_compressed(
        path,
        metadata=json.dumps(_metadata_payload(cache), sort_keys=True),
        class_probabilities=probabilities,
        encoded_labels=np.asarray(cache.encoded_labels),
        member_indices=np.asarray(cache.member_indices, dtype=int),
        nonmember_indices=np.asarray(cache.nonmember_indices, dtype=int),
    )


def _write_malformed_metadata_cache(path: Path, cache: PosteriorFeatureCache) -> None:
    np.savez_compressed(
        path,
        metadata=json.dumps(
            _metadata_payload(cache, seed="not-an-integer"),
            sort_keys=True,
        ),
        class_probabilities=np.asarray(cache.class_probabilities),
        encoded_labels=np.asarray(cache.encoded_labels),
        member_indices=np.asarray(cache.member_indices, dtype=int),
        nonmember_indices=np.asarray(cache.nonmember_indices, dtype=int),
    )


def _write_malformed_optional_metadata_cache(
    path: Path,
    cache: PosteriorFeatureCache,
) -> None:
    np.savez_compressed(
        path,
        metadata=json.dumps(
            _metadata_payload(cache, mixup_alpha="not-a-number"),
            sort_keys=True,
        ),
        class_probabilities=np.asarray(cache.class_probabilities),
        encoded_labels=np.asarray(cache.encoded_labels),
        member_indices=np.asarray(cache.member_indices, dtype=int),
        nonmember_indices=np.asarray(cache.nonmember_indices, dtype=int),
    )


def _synthetic_cache(*, seed: int = 13) -> PosteriorFeatureCache:
    probabilities = np.array(
        [
            [0.90, 0.10],
            [0.82, 0.18],
            [0.74, 0.26],
            [0.66, 0.34],
            [0.58, 0.42],
            [0.50, 0.50],
            [0.42, 0.58],
            [0.34, 0.66],
            [0.26, 0.74],
            [0.18, 0.82],
            [0.10, 0.90],
            [0.05, 0.95],
        ],
        dtype=float,
    )
    encoded_labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=int)
    return PosteriorFeatureCache(
        task_model="eegnet",
        dataset_key="synthetic",
        protocol="cross_session",
        seed=seed,
        subjects=[1, 2],
        task_balanced_accuracy=0.75,
        task_macro_f1=0.74,
        class_probabilities=probabilities,
        encoded_labels=encoded_labels,
        member_indices=[0, 1, 2, 3, 4, 5],
        nonmember_indices=[6, 7, 8, 9, 10, 11],
        epochs_trained=3,
        best_validation_loss=0.5,
    )


def _synthetic_bottleneck_cache(*, seed: int = 13, bottleneck_dim: int = 6) -> PosteriorFeatureCache:
    cache = _synthetic_cache(seed=seed)
    return PosteriorFeatureCache(
        **{
            **cache.__dict__,
            "task_model": "bottleneck_eegnet",
            "bottleneck_dim": bottleneck_dim,
        }
    )


def _synthetic_bad_writer_cache() -> PosteriorFeatureCache:
    cache = _synthetic_cache(seed=19)
    probabilities = np.asarray(cache.class_probabilities).copy()
    probabilities[0] = np.array([0.25, 0.25], dtype=float)
    return PosteriorFeatureCache(
        **{
            **cache.__dict__,
            "class_probabilities": probabilities,
        }
    )


def _direct_threshold(cache: PosteriorFeatureCache):
    return _run_threshold_membership_attack(
        task_model=cache.task_model,
        dataset_key=cache.dataset_key,
        protocol=cache.protocol,
        seed=cache.seed,
        attack_seed=None,
        task_balanced_accuracy=cache.task_balanced_accuracy,
        task_macro_f1=cache.task_macro_f1,
        class_probabilities=cache.class_probabilities,
        encoded_labels=cache.encoded_labels,
        member_indices=cache.member_indices,
        nonmember_indices=cache.nonmember_indices,
        score_type="max_probability",
        epochs_trained=cache.epochs_trained,
        best_validation_loss=cache.best_validation_loss,
    )


def _direct_logistic(cache: PosteriorFeatureCache):
    return _run_learned_membership_attack(
        attack_type="logistic_regression",
        task_model=cache.task_model,
        dataset_key=cache.dataset_key,
        protocol=cache.protocol,
        seed=cache.seed,
        attack_seed=101,
        task_balanced_accuracy=cache.task_balanced_accuracy,
        task_macro_f1=cache.task_macro_f1,
        class_probabilities=cache.class_probabilities,
        encoded_labels=cache.encoded_labels,
        member_indices=cache.member_indices,
        nonmember_indices=cache.nonmember_indices,
        score_type="posterior_probabilities",
        epochs_trained=cache.epochs_trained,
        best_validation_loss=cache.best_validation_loss,
    )


def _assert_same_result(name: str, direct, cached) -> None:
    if direct.to_dict() != cached.to_dict():
        raise AssertionError(f"{name} cached result differs from direct result")
    print(f"PASS {name}")


def main() -> None:
    cache = _synthetic_cache()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cache_path = tmp_path / "posterior_cache_seed13.npz"
        write_posterior_feature_cache(cache, cache_path)
        loaded = load_posterior_feature_cache(cache_path)
        if loaded.task_model != cache.task_model or loaded.subjects != cache.subjects:
            raise AssertionError("posterior cache metadata did not round-trip")
        if not np.array_equal(loaded.class_probabilities, cache.class_probabilities):
            raise AssertionError("posterior cache probabilities did not round-trip")
        print("PASS posterior_cache_round_trip")

        _assert_same_result(
            "cached_threshold_attack",
            _direct_threshold(cache),
            run_cached_membership_inference_attack(
                cache_path,
                attack_type="threshold",
                score_type="max_probability",
            ),
        )
        _assert_same_result(
            "cached_logistic_attack",
            _direct_logistic(cache),
            run_cached_membership_inference_attack(
                cache_path,
                attack_type="logistic_regression",
                score_type="posterior_probabilities",
                attack_seed=101,
            ),
        )

        second_cache_path = tmp_path / "posterior_cache_seed17.npz"
        write_posterior_feature_cache(_synthetic_cache(seed=17), second_cache_path)
        sweep_summary = run_cached_membership_inference_sweep(
            posterior_cache_paths=[cache_path, second_cache_path],
            output_dir=tmp_path / "cached_sweep",
            attack_type="logistic_regression",
            score_type="posterior_probabilities",
            attack_seed=101,
        )
        if sweep_summary["num_runs"] != 2 or sweep_summary["seeds"] != [13, 17]:
            raise AssertionError("cached sweep summary did not preserve cache seeds")
        if len(sweep_summary["posterior_cache_paths"]) != 2:
            raise AssertionError("cached sweep summary did not record cache paths")
        if sweep_summary["attack_seed_policy"] != "explicit_fixed_seed":
            raise AssertionError("cached sweep summary lost attacker-seed policy")
        print("PASS cached_attack_sweep")

        try:
            run_cached_membership_inference_sweep(
                posterior_cache_paths=[cache_path, cache_path],
                output_dir=tmp_path / "duplicate_path_cached_sweep",
                attack_type="threshold",
                score_type="max_probability",
            )
        except ValueError as exc:
            if "Duplicate posterior cache path" not in str(exc):
                raise AssertionError(
                    "duplicate cache path sweep failed for the wrong reason"
                ) from exc
            print("PASS cached_sweep_rejects_duplicate_paths")
        else:
            raise AssertionError("cached sweep accepted duplicate cache paths")

        duplicate_seed_path = tmp_path / "posterior_cache_seed13_duplicate.npz"
        write_posterior_feature_cache(_synthetic_cache(seed=13), duplicate_seed_path)
        try:
            run_cached_membership_inference_sweep(
                posterior_cache_paths=[cache_path, duplicate_seed_path],
                output_dir=tmp_path / "duplicate_cached_sweep",
                attack_type="threshold",
                score_type="max_probability",
            )
        except ValueError as exc:
            if "Duplicate task seed" not in str(exc):
                raise AssertionError(
                    "duplicate cache sweep failed for the wrong reason"
                ) from exc
            print("PASS cached_sweep_rejects_duplicate_seeds")
        else:
            raise AssertionError("cached sweep accepted duplicate task seeds")

        bottleneck_6_path = tmp_path / "bottleneck_dim6_seed13.npz"
        bottleneck_12_path = tmp_path / "bottleneck_dim12_seed17.npz"
        write_posterior_feature_cache(
            _synthetic_bottleneck_cache(seed=13, bottleneck_dim=6),
            bottleneck_6_path,
        )
        write_posterior_feature_cache(
            _synthetic_bottleneck_cache(seed=17, bottleneck_dim=12),
            bottleneck_12_path,
        )
        try:
            run_cached_membership_inference_sweep(
                posterior_cache_paths=[bottleneck_6_path, bottleneck_12_path],
                output_dir=tmp_path / "mixed_cached_sweep",
                attack_type="threshold",
                score_type="max_probability",
            )
        except ValueError as exc:
            if "Mixed bottleneck_dim values" not in str(exc):
                raise AssertionError(
                    "mixed cache sweep failed for the wrong reason"
                ) from exc
            print("PASS cached_sweep_rejects_mixed_settings")
        else:
            raise AssertionError("cached sweep accepted mixed bottleneck settings")

        malformed_cache_path = tmp_path / "malformed_index_cache.npz"
        _write_malformed_index_cache(malformed_cache_path, cache)
        try:
            load_posterior_feature_cache(malformed_cache_path)
        except ValueError as exc:
            if "indices must be within probability rows" not in str(exc):
                raise AssertionError(
                    "malformed cache failed for the wrong reason"
                ) from exc
            print("PASS cached_loader_rejects_bad_indices")
        else:
            raise AssertionError("posterior cache loader accepted out-of-range indices")

        malformed_probability_path = tmp_path / "malformed_probability_cache.npz"
        _write_malformed_probability_cache(malformed_probability_path, cache)
        try:
            load_posterior_feature_cache(malformed_probability_path)
        except ValueError as exc:
            if "rows must sum to one" not in str(exc):
                raise AssertionError(
                    "malformed probability cache failed for the wrong reason"
                ) from exc
            print("PASS cached_loader_rejects_bad_probabilities")
        else:
            raise AssertionError("posterior cache loader accepted invalid probabilities")

        malformed_metadata_path = tmp_path / "malformed_metadata_cache.npz"
        _write_malformed_metadata_cache(malformed_metadata_path, cache)
        try:
            load_posterior_feature_cache(malformed_metadata_path)
        except ValueError as exc:
            if "metadata seed must be an integer" not in str(exc):
                raise AssertionError(
                    "malformed metadata cache failed for the wrong reason"
                ) from exc
            print("PASS cached_loader_rejects_bad_metadata")
        else:
            raise AssertionError("posterior cache loader accepted invalid metadata")

        malformed_optional_path = tmp_path / "malformed_optional_metadata_cache.npz"
        _write_malformed_optional_metadata_cache(malformed_optional_path, cache)
        try:
            load_posterior_feature_cache(malformed_optional_path)
        except ValueError as exc:
            if "metadata mixup_alpha must be null or finite" not in str(exc):
                raise AssertionError(
                    "malformed optional metadata cache failed for the wrong reason"
                ) from exc
            print("PASS cached_loader_rejects_bad_optional_metadata")
        else:
            raise AssertionError(
                "posterior cache loader accepted invalid optional metadata"
            )

        try:
            write_posterior_feature_cache(
                _synthetic_bad_writer_cache(),
                tmp_path / "writer_should_reject_bad_probabilities.npz",
            )
        except ValueError as exc:
            if "rows must sum to one" not in str(exc):
                raise AssertionError(
                    "writer rejected invalid cache for the wrong reason"
                ) from exc
            print("PASS cached_writer_rejects_bad_probabilities")
        else:
            raise AssertionError("posterior cache writer accepted invalid probabilities")

    _check_cache_cli_help()
    _check_cache_cli_rejects_missing_file()

    assert _single_attack_seed_policy(
        [
            {"attack_seed_policy": "explicit_fixed_seed"},
            {"attack_seed_policy": "explicit_fixed_seed"},
        ]
    ) == "explicit_fixed_seed"
    print("PASS sweep_attack_seed_policy_consistent")
    for malformed in (
        [{"attack_seed_policy": "task_seed_default"}, {}],
        [
            {"attack_seed_policy": "task_seed_default"},
            {"attack_seed_policy": "explicit_fixed_seed"},
        ],
    ):
        try:
            _single_attack_seed_policy(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("sweep accepted missing or mixed attacker-seed policy")
    print("PASS sweep_attack_seed_policy_rejects_mixed")

    print("PASS posterior-cache plumbing checks=16")


if __name__ == "__main__":
    main()
