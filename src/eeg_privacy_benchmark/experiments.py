"""Small experiment runners for repeated benchmark sweeps."""

from __future__ import annotations

import json
from pathlib import Path

from eeg_privacy_benchmark.privacy import (
    load_posterior_feature_cache,
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
    run_label_smoothing_eegnet_membership_inference_attack,
    run_label_smoothing_eegnet_subject_id_probe,
    run_mixup_eegnet_membership_inference_attack,
    run_mixup_eegnet_subject_id_probe,
)


def _mean(results: list[dict], metric: str) -> float:
    return float(sum(result[metric] for result in results) / len(results))


def _std(results: list[dict], metric: str) -> float:
    mean = _mean(results, metric)
    variance = sum((result[metric] - mean) ** 2 for result in results) / len(results)
    return float(variance ** 0.5)


def _has_numeric_metric(results: list[dict], metric: str) -> bool:
    return all(metric in result and result[metric] is not None for result in results)


def _single_attack_seed_policy(results: list[dict]) -> str:
    policies = {result.get("attack_seed_policy") for result in results}
    if None in policies:
        raise ValueError("Membership results are missing attack_seed_policy metadata.")
    if len(policies) != 1:
        raise ValueError(f"Mixed attack_seed_policy values in sweep: {sorted(policies)}")
    return policies.pop()


CACHE_SWEEP_SETTING_FIELDS = [
    "adversarial_weight",
    "adversarial_schedule",
    "ramp_up_fraction",
    "label_smoothing",
    "bottleneck_dim",
    "feature_noise_std",
    "mixup_alpha",
    "confidence_penalty_beta",
]


def _validate_cache_sweep_compatible(cache_paths, caches) -> None:
    first_cache = caches[0]
    seen_paths = set()
    seen_seeds = {}
    for path, cache in zip(cache_paths, caches):
        resolved_path = Path(path).resolve()
        if resolved_path in seen_paths:
            raise ValueError(f"Duplicate posterior cache path: {path}.")
        seen_paths.add(resolved_path)
        if cache.seed in seen_seeds:
            raise ValueError(
                f"Duplicate task seed in posterior caches: {path} and "
                f"{seen_seeds[cache.seed]} both use seed {cache.seed}."
            )
        seen_seeds[cache.seed] = path
        checks = [
            ("task_model", cache.task_model, first_cache.task_model),
            ("dataset_key", cache.dataset_key, first_cache.dataset_key),
            ("protocol", cache.protocol, first_cache.protocol),
            ("subjects", cache.subjects, first_cache.subjects),
        ]
        for field, actual, expected in checks:
            if actual != expected:
                raise ValueError(
                    f"Mixed {field} values in posterior caches: {path} has "
                    f"{actual}, expected {expected}."
                )
        for field in CACHE_SWEEP_SETTING_FIELDS:
            actual = getattr(cache, field)
            expected = getattr(first_cache, field)
            if actual != expected:
                raise ValueError(
                    f"Mixed {field} values in posterior caches: {path} has "
                    f"{actual}, expected {expected}."
                )


def run_subject_id_sweep(
    *,
    model: str,
    dataset_key: str,
    protocol: str,
    seeds: list[int],
    subjects: list[int] | None,
    output_dir: str | Path,
    task_epochs: int = 20,
    task_batch_size: int = 32,
    task_learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    adversarial_weight: float = 0.5,
    adversarial_schedule: str = "constant",
    ramp_up_fraction: float = 0.4,
    n_components: int = 8,
    label_smoothing: float = 0.1,
    bottleneck_dim: int = 8,
    feature_noise_std: float = 0.1,
    mixup_alpha: float = 0.2,
    confidence_penalty_beta: float = 0.05,
) -> dict:
    """Run repeated subject-ID experiments and write per-run plus summary files."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        if model == "eegnet":
            result = run_eegnet_subject_id_probe(
                dataset_key=dataset_key,
                protocol=protocol,
                seed=seed,
                subjects=subjects,
                task_epochs=task_epochs,
                task_batch_size=task_batch_size,
                task_learning_rate=task_learning_rate,
                validation_fraction=validation_fraction,
                early_stopping_patience=early_stopping_patience,
            )
        elif model == "label_smoothing_eegnet":
            result = run_label_smoothing_eegnet_subject_id_probe(
                dataset_key=dataset_key,
                protocol=protocol,
                seed=seed,
                subjects=subjects,
                task_epochs=task_epochs,
                task_batch_size=task_batch_size,
                task_learning_rate=task_learning_rate,
                validation_fraction=validation_fraction,
                early_stopping_patience=early_stopping_patience,
                label_smoothing=label_smoothing,
            )
        elif model == "bottleneck_eegnet":
            result = run_bottleneck_eegnet_subject_id_probe(
                dataset_key=dataset_key,
                protocol=protocol,
                seed=seed,
                subjects=subjects,
                task_epochs=task_epochs,
                task_batch_size=task_batch_size,
                task_learning_rate=task_learning_rate,
                validation_fraction=validation_fraction,
                early_stopping_patience=early_stopping_patience,
                bottleneck_dim=bottleneck_dim,
            )
        elif model == "feature_noise_eegnet":
            result = run_feature_noise_eegnet_subject_id_probe(
                dataset_key=dataset_key,
                protocol=protocol,
                seed=seed,
                subjects=subjects,
                task_epochs=task_epochs,
                task_batch_size=task_batch_size,
                task_learning_rate=task_learning_rate,
                validation_fraction=validation_fraction,
                early_stopping_patience=early_stopping_patience,
                feature_noise_std=feature_noise_std,
            )
        elif model == "mixup_eegnet":
            result = run_mixup_eegnet_subject_id_probe(
                dataset_key=dataset_key,
                protocol=protocol,
                seed=seed,
                subjects=subjects,
                task_epochs=task_epochs,
                task_batch_size=task_batch_size,
                task_learning_rate=task_learning_rate,
                validation_fraction=validation_fraction,
                early_stopping_patience=early_stopping_patience,
                mixup_alpha=mixup_alpha,
            )
        elif model == "confidence_penalty_eegnet":
            result = run_confidence_penalty_eegnet_subject_id_probe(
                dataset_key=dataset_key,
                protocol=protocol,
                seed=seed,
                subjects=subjects,
                task_epochs=task_epochs,
                task_batch_size=task_batch_size,
                task_learning_rate=task_learning_rate,
                validation_fraction=validation_fraction,
                early_stopping_patience=early_stopping_patience,
                confidence_penalty_beta=confidence_penalty_beta,
            )
        elif model == "adversarial_eegnet":
            result = run_adversarial_eegnet_subject_id_probe(
                dataset_key=dataset_key,
                protocol=protocol,
                seed=seed,
                subjects=subjects,
                task_epochs=task_epochs,
                task_batch_size=task_batch_size,
                task_learning_rate=task_learning_rate,
                validation_fraction=validation_fraction,
                early_stopping_patience=early_stopping_patience,
                adversarial_weight=adversarial_weight,
                adversarial_schedule=adversarial_schedule,
                ramp_up_fraction=ramp_up_fraction,
            )
        elif model == "csp_lda":
            result = run_csp_subject_id_probe(
                dataset_key=dataset_key,
                protocol=protocol,
                seed=seed,
                subjects=subjects,
                n_components=n_components,
            )
        else:
            raise ValueError(f"Unsupported subject-ID sweep model: {model}")

        payload = result.to_dict()
        results.append(payload)
        run_path = out_dir / f"{model}_{dataset_key}_{protocol}_seed{seed}.json"
        run_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    summary = {
        "model": model,
        "dataset_key": dataset_key,
        "protocol": protocol,
        "subjects": subjects,
        "num_runs": len(results),
        "seeds": seeds,
        "task_balanced_accuracy_mean": _mean(results, "task_balanced_accuracy"),
        "task_balanced_accuracy_std": _std(results, "task_balanced_accuracy"),
        "subject_id_accuracy_mean": _mean(results, "subject_id_accuracy"),
        "subject_id_accuracy_std": _std(results, "subject_id_accuracy"),
        "subject_id_macro_f1_mean": _mean(results, "subject_id_macro_f1"),
        "subject_id_macro_f1_std": _std(results, "subject_id_macro_f1"),
    }
    if model == "adversarial_eegnet":
        summary["adversarial_weight"] = adversarial_weight
        summary["adversarial_schedule"] = adversarial_schedule
        summary["ramp_up_fraction"] = ramp_up_fraction
    elif model == "label_smoothing_eegnet":
        summary["label_smoothing"] = label_smoothing
    elif model == "bottleneck_eegnet":
        summary["bottleneck_dim"] = bottleneck_dim
    elif model == "feature_noise_eegnet":
        summary["feature_noise_std"] = feature_noise_std
    elif model == "mixup_eegnet":
        summary["mixup_alpha"] = mixup_alpha
    elif model == "confidence_penalty_eegnet":
        summary["confidence_penalty_beta"] = confidence_penalty_beta
    optional_metrics = [
        "epochs_trained",
        "best_validation_loss",
        "task_macro_f1",
    ]
    for metric in optional_metrics:
        if _has_numeric_metric(results, metric):
            summary[f"{metric}_mean"] = _mean(results, metric)
            summary[f"{metric}_std"] = _std(results, metric)
    summary_path = out_dir / f"{model}_{dataset_key}_{protocol}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_adversarial_weight_sweep(
    *,
    dataset_key: str,
    protocol: str,
    seeds: list[int],
    subjects: list[int] | None,
    adversarial_weights: list[float],
    output_dir: str | Path,
    task_epochs: int = 20,
    task_batch_size: int = 32,
    task_learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    adversarial_schedule: str = "constant",
    ramp_up_fraction: float = 0.4,
) -> dict:
    """Run subject-adversarial sweeps across multiple gradient-reversal weights."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for weight in adversarial_weights:
        weight_slug = str(weight).replace(".", "p")
        summary = run_subject_id_sweep(
            model="adversarial_eegnet",
            dataset_key=dataset_key,
            protocol=protocol,
            seeds=seeds,
            subjects=subjects,
            output_dir=out_dir / f"weight_{weight_slug}",
            task_epochs=task_epochs,
            task_batch_size=task_batch_size,
            task_learning_rate=task_learning_rate,
            validation_fraction=validation_fraction,
            early_stopping_patience=early_stopping_patience,
            adversarial_weight=weight,
            adversarial_schedule=adversarial_schedule,
            ramp_up_fraction=ramp_up_fraction,
        )
        summaries.append(summary)

    ranked = sorted(summaries, key=lambda summary: summary["subject_id_accuracy_mean"])
    aggregate = {
        "model": "adversarial_eegnet",
        "dataset_key": dataset_key,
        "protocol": protocol,
        "subjects": subjects,
        "num_weights": len(adversarial_weights),
        "weights": adversarial_weights,
        "seeds": seeds,
        "adversarial_schedule": adversarial_schedule,
        "ramp_up_fraction": ramp_up_fraction,
        "best_privacy_weight": ranked[0]["adversarial_weight"],
        "best_privacy_subject_id_accuracy_mean": ranked[0]["subject_id_accuracy_mean"],
        "best_privacy_task_balanced_accuracy_mean": ranked[0]["task_balanced_accuracy_mean"],
        "weight_summaries": summaries,
    }
    aggregate_path = out_dir / "adversarial_eegnet_weight_sweep_summary.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate


def run_membership_inference_sweep(
    *,
    model: str,
    dataset_key: str,
    protocol: str,
    seeds: list[int],
    subjects: list[int] | None,
    output_dir: str | Path,
    task_epochs: int = 20,
    task_batch_size: int = 32,
    task_learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    adversarial_weight: float = 0.5,
    adversarial_schedule: str = "constant",
    ramp_up_fraction: float = 0.4,
    n_components: int = 8,
    label_smoothing: float = 0.1,
    bottleneck_dim: int = 8,
    feature_noise_std: float = 0.1,
    mixup_alpha: float = 0.2,
    confidence_penalty_beta: float = 0.05,
    attack_type: str = "threshold",
    score_type: str = "label_known_log_probability",
    attack_seed: int | None = None,
    posterior_cache_dir: str | Path | None = None,
) -> dict:
    """Run repeated membership-inference experiments and summarize attack metrics."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(posterior_cache_dir) if posterior_cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    results = []
    posterior_cache_paths = []

    for seed in seeds:
        posterior_cache_output = None
        if cache_dir is not None:
            posterior_cache_output = (
                cache_dir / f"{model}_{dataset_key}_{protocol}_seed{seed}_posterior_cache.npz"
            )
            posterior_cache_paths.append(str(posterior_cache_output))
        if model == "eegnet":
            result = run_eegnet_membership_inference_attack(
                dataset_key=dataset_key,
                protocol=protocol,
                seed=seed,
                subjects=subjects,
                task_epochs=task_epochs,
                task_batch_size=task_batch_size,
                task_learning_rate=task_learning_rate,
                validation_fraction=validation_fraction,
                early_stopping_patience=early_stopping_patience,
                attack_type=attack_type,
                score_type=score_type,
                attack_seed=attack_seed,
                posterior_cache_output=posterior_cache_output,
            )
        elif model == "label_smoothing_eegnet":
            result = run_label_smoothing_eegnet_membership_inference_attack(
                dataset_key=dataset_key,
                protocol=protocol,
                seed=seed,
                subjects=subjects,
                task_epochs=task_epochs,
                task_batch_size=task_batch_size,
                task_learning_rate=task_learning_rate,
                validation_fraction=validation_fraction,
                early_stopping_patience=early_stopping_patience,
                attack_type=attack_type,
                score_type=score_type,
                attack_seed=attack_seed,
                label_smoothing=label_smoothing,
                posterior_cache_output=posterior_cache_output,
            )
        elif model == "bottleneck_eegnet":
            result = run_bottleneck_eegnet_membership_inference_attack(
                dataset_key=dataset_key,
                protocol=protocol,
                seed=seed,
                subjects=subjects,
                task_epochs=task_epochs,
                task_batch_size=task_batch_size,
                task_learning_rate=task_learning_rate,
                validation_fraction=validation_fraction,
                early_stopping_patience=early_stopping_patience,
                attack_type=attack_type,
                score_type=score_type,
                attack_seed=attack_seed,
                bottleneck_dim=bottleneck_dim,
                posterior_cache_output=posterior_cache_output,
            )
        elif model == "feature_noise_eegnet":
            result = run_feature_noise_eegnet_membership_inference_attack(
                dataset_key=dataset_key,
                protocol=protocol,
                seed=seed,
                subjects=subjects,
                task_epochs=task_epochs,
                task_batch_size=task_batch_size,
                task_learning_rate=task_learning_rate,
                validation_fraction=validation_fraction,
                early_stopping_patience=early_stopping_patience,
                attack_type=attack_type,
                score_type=score_type,
                attack_seed=attack_seed,
                feature_noise_std=feature_noise_std,
                posterior_cache_output=posterior_cache_output,
            )
        elif model == "mixup_eegnet":
            result = run_mixup_eegnet_membership_inference_attack(
                dataset_key=dataset_key,
                protocol=protocol,
                seed=seed,
                subjects=subjects,
                task_epochs=task_epochs,
                task_batch_size=task_batch_size,
                task_learning_rate=task_learning_rate,
                validation_fraction=validation_fraction,
                early_stopping_patience=early_stopping_patience,
                attack_type=attack_type,
                score_type=score_type,
                attack_seed=attack_seed,
                mixup_alpha=mixup_alpha,
                posterior_cache_output=posterior_cache_output,
            )
        elif model == "confidence_penalty_eegnet":
            result = run_confidence_penalty_eegnet_membership_inference_attack(
                dataset_key=dataset_key,
                protocol=protocol,
                seed=seed,
                subjects=subjects,
                task_epochs=task_epochs,
                task_batch_size=task_batch_size,
                task_learning_rate=task_learning_rate,
                validation_fraction=validation_fraction,
                early_stopping_patience=early_stopping_patience,
                attack_type=attack_type,
                score_type=score_type,
                attack_seed=attack_seed,
                confidence_penalty_beta=confidence_penalty_beta,
                posterior_cache_output=posterior_cache_output,
            )
        elif model == "adversarial_eegnet":
            result = run_adversarial_eegnet_membership_inference_attack(
                dataset_key=dataset_key,
                protocol=protocol,
                seed=seed,
                subjects=subjects,
                task_epochs=task_epochs,
                task_batch_size=task_batch_size,
                task_learning_rate=task_learning_rate,
                validation_fraction=validation_fraction,
                early_stopping_patience=early_stopping_patience,
                adversarial_weight=adversarial_weight,
                adversarial_schedule=adversarial_schedule,
                ramp_up_fraction=ramp_up_fraction,
                attack_type=attack_type,
                score_type=score_type,
                attack_seed=attack_seed,
                posterior_cache_output=posterior_cache_output,
            )
        elif model == "csp_lda":
            result = run_csp_membership_inference_attack(
                dataset_key=dataset_key,
                protocol=protocol,
                seed=seed,
                subjects=subjects,
                n_components=n_components,
                attack_type=attack_type,
                score_type=score_type,
                attack_seed=attack_seed,
                posterior_cache_output=posterior_cache_output,
            )
        else:
            raise ValueError(f"Unsupported membership-inference sweep model: {model}")

        payload = result.to_dict()
        results.append(payload)
        attack_seed_slug = (
            f"_attackseed{payload['attack_seed']}" if attack_seed is not None else ""
        )
        run_path = out_dir / f"{model}_{dataset_key}_{protocol}_seed{seed}{attack_seed_slug}.json"
        run_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    summary = {
        "model": model,
        "dataset_key": dataset_key,
        "protocol": protocol,
        "subjects": subjects,
        "num_runs": len(results),
        "seeds": seeds,
        "attack_seed": attack_seed,
        "attack_seed_policy": _single_attack_seed_policy(results),
        "effective_attack_seeds": [result["attack_seed"] for result in results],
        "attack_type": results[0]["attack_type"],
        "score_type": results[0]["score_type"],
        "task_balanced_accuracy_mean": _mean(results, "task_balanced_accuracy"),
        "task_balanced_accuracy_std": _std(results, "task_balanced_accuracy"),
        "attack_auc_mean": _mean(results, "attack_auc"),
        "attack_auc_std": _std(results, "attack_auc"),
        "attack_average_precision_mean": _mean(results, "attack_average_precision"),
        "attack_average_precision_std": _std(results, "attack_average_precision"),
        "attack_balanced_accuracy_mean": _mean(results, "attack_balanced_accuracy"),
        "attack_balanced_accuracy_std": _std(results, "attack_balanced_accuracy"),
        "member_score_mean_mean": _mean(results, "member_score_mean"),
        "member_score_mean_std": _std(results, "member_score_mean"),
        "nonmember_score_mean_mean": _mean(results, "nonmember_score_mean"),
        "nonmember_score_mean_std": _std(results, "nonmember_score_mean"),
    }
    if posterior_cache_paths:
        summary["posterior_cache_paths"] = posterior_cache_paths
    if model == "adversarial_eegnet":
        summary["adversarial_weight"] = adversarial_weight
        summary["adversarial_schedule"] = adversarial_schedule
        summary["ramp_up_fraction"] = ramp_up_fraction
    elif model == "label_smoothing_eegnet":
        summary["label_smoothing"] = label_smoothing
    elif model == "bottleneck_eegnet":
        summary["bottleneck_dim"] = bottleneck_dim
    elif model == "feature_noise_eegnet":
        summary["feature_noise_std"] = feature_noise_std
    elif model == "mixup_eegnet":
        summary["mixup_alpha"] = mixup_alpha
    elif model == "confidence_penalty_eegnet":
        summary["confidence_penalty_beta"] = confidence_penalty_beta
    optional_metrics = [
        "epochs_trained",
        "best_validation_loss",
        "task_macro_f1",
    ]
    for metric in optional_metrics:
        if _has_numeric_metric(results, metric):
            summary[f"{metric}_mean"] = _mean(results, metric)
            summary[f"{metric}_std"] = _std(results, metric)
    summary_slug = (
        f"_attackseed{results[0]['attack_seed']}" if attack_seed is not None else ""
    )
    summary_path = (
        out_dir / f"{model}_{dataset_key}_{protocol}{summary_slug}_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_cached_membership_inference_sweep(
    *,
    posterior_cache_paths: list[str | Path],
    output_dir: str | Path,
    attack_type: str,
    score_type: str,
    attack_seed: int | None = None,
) -> dict:
    """Run repeated membership-inference attacks from cached posterior features."""

    if not posterior_cache_paths:
        raise ValueError("At least one posterior cache path is required.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = [Path(path) for path in posterior_cache_paths]
    caches = [load_posterior_feature_cache(path) for path in cache_paths]
    first_cache = caches[0]
    _validate_cache_sweep_compatible(cache_paths, caches)

    results = []
    for cache_path, cache in zip(cache_paths, caches):
        result = run_cached_membership_inference_attack(
            cache_path,
            attack_type=attack_type,
            score_type=score_type,
            attack_seed=attack_seed,
        )
        payload = result.to_dict()
        payload["posterior_cache_path"] = str(cache_path)
        results.append(payload)
        attack_seed_slug = (
            f"_attackseed{payload['attack_seed']}" if attack_seed is not None else ""
        )
        run_path = (
            out_dir
            / f"{cache.task_model}_{cache.dataset_key}_{cache.protocol}_seed{cache.seed}{attack_seed_slug}.json"
        )
        run_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    summary = {
        "model": first_cache.task_model,
        "dataset_key": first_cache.dataset_key,
        "protocol": first_cache.protocol,
        "subjects": first_cache.subjects,
        "num_runs": len(results),
        "seeds": [cache.seed for cache in caches],
        "attack_seed": attack_seed,
        "attack_seed_policy": _single_attack_seed_policy(results),
        "effective_attack_seeds": [result["attack_seed"] for result in results],
        "attack_type": results[0]["attack_type"],
        "score_type": results[0]["score_type"],
        "posterior_cache_paths": [str(path) for path in cache_paths],
        "task_balanced_accuracy_mean": _mean(results, "task_balanced_accuracy"),
        "task_balanced_accuracy_std": _std(results, "task_balanced_accuracy"),
        "attack_auc_mean": _mean(results, "attack_auc"),
        "attack_auc_std": _std(results, "attack_auc"),
        "attack_average_precision_mean": _mean(results, "attack_average_precision"),
        "attack_average_precision_std": _std(results, "attack_average_precision"),
        "attack_balanced_accuracy_mean": _mean(results, "attack_balanced_accuracy"),
        "attack_balanced_accuracy_std": _std(results, "attack_balanced_accuracy"),
        "member_score_mean_mean": _mean(results, "member_score_mean"),
        "member_score_mean_std": _std(results, "member_score_mean"),
        "nonmember_score_mean_mean": _mean(results, "nonmember_score_mean"),
        "nonmember_score_mean_std": _std(results, "nonmember_score_mean"),
    }
    for setting in CACHE_SWEEP_SETTING_FIELDS:
        value = getattr(first_cache, setting)
        if value is not None:
            summary[setting] = value
    optional_metrics = [
        "epochs_trained",
        "best_validation_loss",
        "task_macro_f1",
    ]
    for metric in optional_metrics:
        if _has_numeric_metric(results, metric):
            summary[f"{metric}_mean"] = _mean(results, metric)
            summary[f"{metric}_std"] = _std(results, metric)

    summary_slug = (
        f"_attackseed{results[0]['attack_seed']}" if attack_seed is not None else ""
    )
    summary_path = (
        out_dir
        / f"{first_cache.task_model}_{first_cache.dataset_key}_{first_cache.protocol}{summary_slug}_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
