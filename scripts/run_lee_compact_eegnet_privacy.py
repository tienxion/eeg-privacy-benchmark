from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_privacy_benchmark.cli import _configure_mne_data_dir
from eeg_privacy_benchmark.models import fit_eegnet
from eeg_privacy_benchmark.privacy import (
    run_cached_membership_inference_attack,
    run_eegnet_membership_inference_from_artifacts,
    run_eegnet_subject_id_from_artifacts,
)
from eeg_privacy_benchmark.results import write_result_json


DATASET_KEY = "lee2019_mi"
PROTOCOL = "cross_session"
SUBJECTS = [1, 2, 3, 4, 5, 6]
SEEDS = (13, 17, 19, 23)
TASK_MODEL = "compact_eegnet"
UTILITY_GATE = 0.60
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "lee_compact_eegnet_privacy"
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
    unsupported = sorted(set(seeds) - set(SEEDS))
    if unsupported:
        raise ValueError(f"Seeds are outside the preregistered set: {unsupported}")
    return seeds


def _seed_paths(output_root: Path, seed: int) -> dict[str, Path]:
    seed_root = output_root / f"seed{seed}"
    paths = {
        "task": seed_root / "compact_eegnet_task.json",
        "subject_id": seed_root / "compact_eegnet_subject_id.json",
        "posterior_cache": seed_root / "compact_eegnet_posteriors.npz",
        "suite": seed_root / "compact_eegnet_suite.json",
    }
    for label, _, _ in ATTACKS:
        paths[label] = seed_root / f"compact_eegnet_membership_{label}.json"
    return paths


def _component_paths(paths: dict[str, Path]) -> list[Path]:
    return [
        paths["task"],
        paths["subject_id"],
        paths["posterior_cache"],
        *(paths[label] for label, _, _ in ATTACKS),
    ]


def _suite_complete(paths: dict[str, Path]) -> bool:
    if not paths["suite"].exists() or not all(
        path.exists() for path in _component_paths(paths)
    ):
        return False
    try:
        suite = json.loads(paths["suite"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return suite.get("metadata_schema_version") == 1


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _write_suite_from_outputs(
    paths: dict[str, Path],
    seed: int,
    *,
    task_model: str = TASK_MODEL,
    bottleneck_dim: int | None = None,
    confidence_penalty_beta: float | None = None,
) -> dict:
    task = json.loads(paths["task"].read_text(encoding="utf-8"))
    subject = json.loads(paths["subject_id"].read_text(encoding="utf-8"))
    attacks = {
        label: json.loads(paths[label].read_text(encoding="utf-8"))
        for label, _, _ in ATTACKS
    }
    suite = {
        "metadata_schema_version": 1,
        "task_model": task_model,
        "dataset_key": DATASET_KEY,
        "protocol": PROTOCOL,
        "subjects": SUBJECTS,
        "seed": seed,
        "resample_hz": 250,
        "normalization": "per_trial_channel",
        "classifier_head": "global_average",
        "bottleneck_dim": bottleneck_dim,
        "confidence_penalty_beta": confidence_penalty_beta,
        "task_balanced_accuracy": task["balanced_accuracy"],
        "task_macro_f1": task["macro_f1"],
        "best_checkpoint_epoch": task["best_checkpoint_epoch"],
        "subject_id_accuracy": subject["subject_id_accuracy"],
        "membership_auc": {
            label: result["attack_auc"] for label, result in attacks.items()
        },
        "outputs": {key: _display_path(path) for key, path in paths.items()},
    }
    write_result_json(suite, paths["suite"])
    return suite


def _run_seed(
    output_root: Path,
    seed: int,
    *,
    task_model: str = TASK_MODEL,
    bottleneck_dim: int | None = None,
    confidence_penalty_beta: float = 0.0,
) -> dict:
    paths = _seed_paths(output_root, seed)
    print(f"seed={seed} status=training", flush=True)
    artifacts = fit_eegnet(
        DATASET_KEY,
        protocol=PROTOCOL,
        seed=seed,
        subjects=SUBJECTS,
        epochs=60,
        batch_size=32,
        learning_rate=1e-3,
        validation_fraction=0.2,
        early_stopping_patience=15,
        resample_hz=250,
        normalization="per_trial_channel",
        classifier_head="global_average",
        bottleneck_dim=bottleneck_dim,
        confidence_penalty_beta=confidence_penalty_beta,
    )
    task = artifacts.result.to_dict()
    task.update(
        {
            "task_model": task_model,
            "subjects": SUBJECTS,
            "model_parameters": sum(
                parameter.numel() for parameter in artifacts.model.parameters()
            ),
            "privacy_metrics_computed": True,
        }
    )
    write_result_json(task, paths["task"])

    first_label, first_attack_type, first_score_type = ATTACKS[0]
    first_attack = run_eegnet_membership_inference_from_artifacts(
        artifacts,
        subjects=SUBJECTS,
        task_model=task_model,
        attack_type=first_attack_type,
        score_type=first_score_type,
        posterior_cache_output=paths["posterior_cache"],
    )
    write_result_json(first_attack.to_dict(), paths[first_label])
    for label, attack_type, score_type in ATTACKS[1:]:
        attack = run_cached_membership_inference_attack(
            paths["posterior_cache"],
            attack_type=attack_type,
            score_type=score_type,
        )
        write_result_json(attack.to_dict(), paths[label])

    subject = run_eegnet_subject_id_from_artifacts(
        artifacts,
        task_model=task_model,
    )
    write_result_json(subject.to_dict(), paths["subject_id"])
    suite = _write_suite_from_outputs(
        paths,
        seed,
        task_model=task_model,
        bottleneck_dim=bottleneck_dim,
        confidence_penalty_beta=(
            confidence_penalty_beta if confidence_penalty_beta > 0.0 else None
        ),
    )
    print(
        f"seed={seed} status=complete "
        f"task_balanced_accuracy={suite['task_balanced_accuracy']:.6f} "
        f"subject_id_accuracy={suite['subject_id_accuracy']:.6f}",
        flush=True,
    )
    return suite


def _write_summary(
    suites: list[dict],
    output_root: Path,
    *,
    task_model: str = TASK_MODEL,
    bottleneck_dim: int | None = None,
    confidence_penalty_beta: float | None = None,
    report_stem: str = "lee_compact_eegnet_privacy",
) -> dict:
    task_scores = [suite["task_balanced_accuracy"] for suite in suites]
    subject_scores = [suite["subject_id_accuracy"] for suite in suites]
    attack_means = {
        label: mean(suite["membership_auc"][label] for suite in suites)
        for label, _, _ in ATTACKS
    }
    summary = {
        "metadata_schema_version": 1,
        "task_model": task_model,
        "dataset_key": DATASET_KEY,
        "protocol": PROTOCOL,
        "subjects": SUBJECTS,
        "seeds": [suite["seed"] for suite in suites],
        "resample_hz": 250,
        "normalization": "per_trial_channel",
        "classifier_head": "global_average",
        "bottleneck_dim": bottleneck_dim,
        "confidence_penalty_beta": confidence_penalty_beta,
        "utility_gate": UTILITY_GATE,
        "mean_task_balanced_accuracy": mean(task_scores),
        "sample_std_task_balanced_accuracy": (
            stdev(task_scores) if len(task_scores) > 1 else 0.0
        ),
        "utility_gate_passed": mean(task_scores) >= UTILITY_GATE,
        "mean_subject_id_accuracy": mean(subject_scores),
        "mean_membership_auc": attack_means,
        "status": (
            "privacy_evaluation_complete"
            if len(suites) == len(SEEDS) and mean(task_scores) >= UTILITY_GATE
            else "partial_privacy_evaluation"
        ),
        "suites": suites,
    }
    json_path = output_root / f"{report_stem}.json"
    markdown_path = output_root / f"{report_stem}.md"
    write_result_json(summary, json_path)
    lines = [
        (
            "# Lee Compact Bottleneck EEGNet Privacy Evaluation"
            if bottleneck_dim is not None
            else "# Lee Compact Confidence-Penalty EEGNet Privacy Evaluation"
            if confidence_penalty_beta is not None
            else "# Lee Compact EEGNet Privacy Evaluation"
        ),
        "",
        (
            "Exact confirmed setting: 250 Hz, per-trial channel normalization, "
            "global-average head."
        ),
        "",
        "| Seed | Task BA | Subject ID accuracy | Label-known AUC | Max-prob AUC | Logistic AUC | MLP AUC |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for suite in suites:
        auc = suite["membership_auc"]
        lines.append(
            f"| {suite['seed']} | {suite['task_balanced_accuracy']:.4f} | "
            f"{suite['subject_id_accuracy']:.4f} | "
            f"{auc['threshold_label_known']:.4f} | "
            f"{auc['threshold_max_probability']:.4f} | "
            f"{auc['logistic_posterior']:.4f} | {auc['mlp_posterior']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- Mean task balanced accuracy: `{summary['mean_task_balanced_accuracy']:.4f}`",
            f"- Sample SD: `{summary['sample_std_task_balanced_accuracy']:.4f}`",
            f"- Mean subject-ID accuracy: `{summary['mean_subject_id_accuracy']:.4f}`",
            f"- Utility gate passed: `{str(summary['utility_gate_passed']).lower()}`",
            f"- Status: `{summary['status']}`",
            "",
            "Mean membership AUC:",
        ]
    )
    for label, value in attack_means.items():
        lines.append(f"- {label}: `{value:.4f}`")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate privacy of the confirmed compact Lee EEGNet."
    )
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--mne-data-dir", default="raw_data/mne_data")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--bottleneck-dim", type=int)
    parser.add_argument("--confidence-penalty-beta", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    if args.bottleneck_dim is not None and args.bottleneck_dim <= 0:
        raise ValueError("--bottleneck-dim must be positive.")
    if args.confidence_penalty_beta < 0.0:
        raise ValueError("--confidence-penalty-beta cannot be negative.")
    if args.bottleneck_dim is not None and args.confidence_penalty_beta > 0.0:
        raise ValueError("Run one compact defense mechanism at a time.")
    confidence_penalty_beta = (
        args.confidence_penalty_beta
        if args.confidence_penalty_beta > 0.0
        else None
    )
    task_model = (
        "compact_bottleneck_eegnet"
        if args.bottleneck_dim is not None
        else "compact_confidence_penalty_eegnet"
        if confidence_penalty_beta is not None
        else TASK_MODEL
    )
    confidence_slug = (
        str(confidence_penalty_beta).replace(".", "p")
        if confidence_penalty_beta is not None
        else None
    )
    output_root = args.output_root or (
        ROOT
        / "outputs"
        / (
            f"lee_compact_bottleneck_dim{args.bottleneck_dim}_privacy"
            if args.bottleneck_dim is not None
            else f"lee_compact_confidence_penalty_beta{confidence_slug}_privacy"
            if confidence_slug is not None
            else "lee_compact_eegnet_privacy"
        )
    )
    report_stem = (
        f"lee_compact_bottleneck_dim{args.bottleneck_dim}_privacy"
        if args.bottleneck_dim is not None
        else f"lee_compact_confidence_penalty_beta{confidence_slug}_privacy"
        if confidence_slug is not None
        else "lee_compact_eegnet_privacy"
    )
    _configure_mne_data_dir(args.mne_data_dir, DATASET_KEY)
    suites = []
    for seed in seeds:
        paths = _seed_paths(output_root, seed)
        if _suite_complete(paths) and not args.force:
            suites.append(json.loads(paths["suite"].read_text(encoding="utf-8")))
            print(f"seed={seed} status=skipped_complete", flush=True)
            continue
        if all(path.exists() for path in _component_paths(paths)) and not args.force:
            suites.append(
                _write_suite_from_outputs(
                    paths,
                    seed,
                    task_model=task_model,
                    bottleneck_dim=args.bottleneck_dim,
                    confidence_penalty_beta=confidence_penalty_beta,
                )
            )
            print(f"seed={seed} status=recovered_suite", flush=True)
            continue
        if args.dry_run:
            print(f"seed={seed} status=planned output={paths['suite']}")
            continue
        suites.append(
            _run_seed(
                output_root,
                seed,
                task_model=task_model,
                bottleneck_dim=args.bottleneck_dim,
                confidence_penalty_beta=args.confidence_penalty_beta,
            )
        )

    if suites:
        summary = _write_summary(
            suites,
            output_root,
            task_model=task_model,
            bottleneck_dim=args.bottleneck_dim,
            confidence_penalty_beta=confidence_penalty_beta,
            report_stem=report_stem,
        )
        print(f"status={summary['status']}")
        print(
            f"mean_task_balanced_accuracy="
            f"{summary['mean_task_balanced_accuracy']:.6f}"
        )


if __name__ == "__main__":
    main()
