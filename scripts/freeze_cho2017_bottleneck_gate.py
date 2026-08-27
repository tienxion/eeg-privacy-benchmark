from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cho2017_confirmatory_validation_v1.yaml"
CACHE_MANIFEST = ROOT / "configs" / "cho2017_confirmatory_cache_manifest_v1.json"
BASELINE_GATE = ROOT / "configs" / "cho2017_confirmatory_baseline_gate_v1.json"
BASELINE_GATE_FREEZER = ROOT / "scripts" / "freeze_cho2017_baseline_gate.py"
BASELINE_RUNNER = ROOT / "scripts" / "run_cho2017_confirmatory_baselines.py"
BOTTLENECK_RUNNER = ROOT / "scripts" / "run_cho2017_confirmatory_bottleneck.py"
BASELINE_OUTPUT = ROOT / "outputs" / "v1_1_cho2017_baseline"
DEFAULT_INPUT = ROOT / "outputs" / "v1_1_cho2017_bottleneck"
DEFAULT_OUTPUT = ROOT / "configs" / "cho2017_confirmatory_bottleneck_gate_v1.json"
DEFAULT_RECORD = ROOT / "docs" / "V1_1_CHO2017_BOTTLENECK_GATE.md"
MODEL = "compact_bottleneck_eegnet_dim6"
BOOTSTRAP_REPLICATES = 100_000
RECORDED_DATE = "2026-08-26"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load required script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_baseline_gate() -> dict[str, Any]:
    tracked = json.loads(BASELINE_GATE.read_text(encoding="utf-8"))
    freezer = _load_script(
        BASELINE_GATE_FREEZER,
        "cho2017_baseline_gate_freezer_for_bottleneck_gate",
    )
    rebuilt = freezer._build_payload(BASELINE_OUTPUT.resolve())
    _require(rebuilt == tracked, "local baseline jobs do not reproduce the tracked gate")
    _require(
        tracked["status"] == "PASS"
        and tracked["plain_eegnet_utility_gate"]["gate_pass"] is True
        and tracked["plain_eegnet_utility_gate"]["next_stage"] == MODEL,
        "baseline utility gate does not authorize the bottleneck stage",
    )
    return tracked


def _validated_bottleneck_jobs(
    input_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runner = _load_script(BASELINE_RUNNER, "cho2017_baseline_runner_for_bottleneck_gate")
    bottleneck = _load_script(
        BOTTLENECK_RUNNER,
        "cho2017_bottleneck_runner_for_gate",
    )
    config, manifest = runner._load_sources()
    folds = {
        fold.fold_index: fold
        for fold in runner.build_outer_folds(config["cohort"]["shuffled_subject_blocks"])
    }
    trials_per_class = {
        int(row["subject"]): int(row["trials_per_class"])
        for row in manifest["rows"]
    }
    seeds = list(config["seeds"]["task"])
    expected_paths = {
        bottleneck._job_path(input_dir, fold_index, seed)
        for fold_index in range(1, 6)
        for seed in seeds
    }
    observed_paths = set((input_dir / "results").glob("*.json"))
    _require(observed_paths == expected_paths, "bottleneck result set is not the frozen 20 jobs")
    expected_caches = {
        bottleneck._cache_path(input_dir, fold_index, seed)
        for fold_index in range(1, 6)
        for seed in seeds
    }
    observed_caches = set((input_dir / "posterior_caches").glob("*.npz"))
    _require(observed_caches == expected_caches, "bottleneck cache set is not the frozen 20 jobs")

    jobs: list[dict[str, Any]] = []
    for fold_index in range(1, 6):
        fold = folds[fold_index]
        role_counts = runner._expected_role_counts(fold, trials_per_class)
        attacker = runner.partition_attacker_subjects(
            fold.fitting_subjects,
            seed=config["seeds"]["attacker_subject_partition"],
            fold_index=fold_index,
        )
        for seed in seeds:
            path = bottleneck._job_path(input_dir, fold_index, seed)
            _require(
                bottleneck._valid_completed_job(
                    path,
                    fold=fold,
                    seed=seed,
                    output_dir=input_dir,
                    expected_role_counts=role_counts,
                    attacker_partition=attacker,
                    runner=runner,
                ),
                f"invalid bottleneck job: {path.name}",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            subject_rows = sorted(payload["subject_metrics"], key=lambda row: row["subject"])
            expected_trials = {
                subject: 2 * trials_per_class[subject] for subject in fold.test_subjects
            }
            _require(
                all(
                    row["trials"] == expected_trials[row["subject"]]
                    and 0.0 <= row["balanced_accuracy"] <= 1.0
                    and 0.0 <= row["macro_f1"] <= 1.0
                    for row in subject_rows
                ),
                f"invalid bottleneck subject metrics: {path.name}",
            )
            jobs.append(
                {
                    "model": MODEL,
                    "fold_index": fold_index,
                    "seed": seed,
                    "result_file": str(path.relative_to(input_dir)),
                    "result_sha256": _sha256(path),
                    "task_balanced_accuracy": float(
                        payload["task_result"]["balanced_accuracy"]
                    ),
                    "task_macro_f1": float(payload["task_result"]["macro_f1"]),
                    "subject_metrics": subject_rows,
                    "posterior_cache": payload["posterior_cache"],
                }
            )
    return jobs, config


def _subject_aggregates(
    jobs: list[dict[str, Any]], subjects: list[int]
) -> list[dict[str, Any]]:
    observations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    folds: dict[int, int] = {}
    for job in jobs:
        for row in job["subject_metrics"]:
            subject = int(row["subject"])
            observations[subject].append(row)
            previous_fold = folds.setdefault(subject, job["fold_index"])
            _require(previous_fold == job["fold_index"], "subject appears in multiple test folds")
    rows = []
    for subject in subjects:
        values = observations[subject]
        _require(len(values) == 4, f"subject {subject} does not have four bottleneck seeds")
        rows.append(
            {
                "subject": subject,
                "test_fold": folds[subject],
                "seed_count": len(values),
                "balanced_accuracy_mean": float(
                    np.mean([row["balanced_accuracy"] for row in values])
                ),
                "balanced_accuracy_std": float(
                    np.std([row["balanced_accuracy"] for row in values], ddof=1)
                ),
                "macro_f1_mean": float(np.mean([row["macro_f1"] for row in values])),
            }
        )
    return rows


def _paired_rows(
    baseline_gate: dict[str, Any],
    bottleneck_rows: list[dict[str, Any]],
    subjects: list[int],
) -> list[dict[str, Any]]:
    baseline_rows = {
        int(row["subject"]): row
        for row in baseline_gate["subject_aggregates"]
        if row["model"] == "compact_eegnet"
    }
    bottleneck_by_subject = {int(row["subject"]): row for row in bottleneck_rows}
    _require(set(baseline_rows) == set(subjects), "baseline gate subject set mismatch")
    _require(set(bottleneck_by_subject) == set(subjects), "bottleneck subject set mismatch")
    rows = []
    for subject in subjects:
        baseline = baseline_rows[subject]
        bottleneck = bottleneck_by_subject[subject]
        _require(
            baseline["test_fold"] == bottleneck["test_fold"]
            and baseline["seed_count"] == bottleneck["seed_count"] == 4,
            f"subject {subject} pairing mismatch",
        )
        loss = float(
            baseline["balanced_accuracy_mean"]
            - bottleneck["balanced_accuracy_mean"]
        )
        rows.append(
            {
                "subject": subject,
                "test_fold": baseline["test_fold"],
                "task_seeds": 4,
                "plain_eegnet_balanced_accuracy": float(
                    baseline["balanced_accuracy_mean"]
                ),
                "bottleneck_balanced_accuracy": float(
                    bottleneck["balanced_accuracy_mean"]
                ),
                "balanced_accuracy_loss_plain_minus_bottleneck": loss,
            }
        )
    return rows


def _result_set_sha256(jobs: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for job in jobs:
        digest.update(job["result_file"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(job["result_sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _build_payload(input_dir: Path) -> dict[str, Any]:
    baseline_gate = _validate_baseline_gate()
    jobs, config = _validated_bottleneck_jobs(input_dir)
    subjects = list(config["cohort"]["confirmatory_subjects"])
    bottleneck_rows = _subject_aggregates(jobs, subjects)
    paired_rows = _paired_rows(baseline_gate, bottleneck_rows, subjects)
    confidence_level = float(
        config["promotion_rules"]["bottleneck_utility_noninferiority"][
            "one_sided_confidence_level"
        ]
    )
    margin = float(
        config["promotion_rules"]["bottleneck_utility_noninferiority"][
            "maximum_mean_balanced_accuracy_loss"
        ]
    )
    bootstrap_seed = int(config["cohort"]["partition_seed"])
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0,
        len(subjects),
        size=(BOOTSTRAP_REPLICATES, len(subjects)),
    )
    losses = np.asarray(
        [row["balanced_accuracy_loss_plain_minus_bottleneck"] for row in paired_rows],
        dtype=float,
    )
    bottleneck_values = np.asarray(
        [row["bottleneck_balanced_accuracy"] for row in paired_rows], dtype=float
    )
    bootstrap_losses = losses[indices].mean(axis=1)
    bootstrap_bottleneck_means = bottleneck_values[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    loss_two_sided = np.quantile(
        bootstrap_losses,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    bottleneck_two_sided = np.quantile(
        bootstrap_bottleneck_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    loss_upper = float(np.quantile(bootstrap_losses, confidence_level))
    mean_loss = float(np.mean(losses))
    mean_pass = mean_loss < margin
    bound_pass = loss_upper < margin
    gate_pass = bool(mean_pass and bound_pass)

    baseline_mean = float(
        baseline_gate["models"]["compact_eegnet"][
            "balanced_accuracy_subject_mean"
        ]
    )
    bottleneck_mean = float(np.mean(bottleneck_values))
    _require(
        np.isclose(baseline_mean - bottleneck_mean, mean_loss, atol=1e-12),
        "paired mean loss does not match model means",
    )
    fold_summaries = []
    for fold_index in range(1, 6):
        fold_rows = [row for row in paired_rows if row["test_fold"] == fold_index]
        fold_summaries.append(
            {
                "fold_index": fold_index,
                "subjects": len(fold_rows),
                "plain_eegnet_balanced_accuracy_mean": float(
                    np.mean([row["plain_eegnet_balanced_accuracy"] for row in fold_rows])
                ),
                "bottleneck_balanced_accuracy_mean": float(
                    np.mean([row["bottleneck_balanced_accuracy"] for row in fold_rows])
                ),
                "balanced_accuracy_loss_mean": float(
                    np.mean(
                        [
                            row["balanced_accuracy_loss_plain_minus_bottleneck"]
                            for row in fold_rows
                        ]
                    )
                ),
            }
        )
    baseline_seed_summaries = {
        int(row["seed"]): row
        for row in baseline_gate["models"]["compact_eegnet"]["seed_summaries"]
    }
    seed_summaries = []
    for seed in config["seeds"]["task"]:
        seed_values = [
            metric["balanced_accuracy"]
            for job in jobs
            if job["seed"] == seed
            for metric in job["subject_metrics"]
        ]
        bottleneck_seed_mean = float(np.mean(seed_values))
        baseline_seed_mean = float(
            baseline_seed_summaries[seed]["balanced_accuracy_mean"]
        )
        seed_summaries.append(
            {
                "seed": seed,
                "subjects": len(seed_values),
                "plain_eegnet_balanced_accuracy_mean": baseline_seed_mean,
                "bottleneck_balanced_accuracy_mean": bottleneck_seed_mean,
                "balanced_accuracy_loss_mean": baseline_seed_mean
                - bottleneck_seed_mean,
            }
        )

    public_jobs = [
        {key: value for key, value in job.items() if key != "subject_metrics"}
        for job in jobs
    ]
    return {
        "schema_version": 1,
        "stage": "cho2017_confirmatory_bottleneck_utility_gate",
        "status": "PASS" if gate_pass else "STOP",
        "recorded_date": RECORDED_DATE,
        "provenance": {
            "contract_sha256": _sha256(CONFIG),
            "cache_manifest_sha256": _sha256(CACHE_MANIFEST),
            "baseline_gate_sha256": _sha256(BASELINE_GATE),
            "bottleneck_result_set_sha256": _result_set_sha256(jobs),
            "validated_result_jobs": len(jobs),
            "validated_posterior_caches": len(jobs),
            "network_or_training": False,
        },
        "inference": {
            "unit": "subject",
            "pairing": "same_subject_plain_eegnet_minus_bottleneck",
            "task_seed_aggregation": "arithmetic_mean_within_subject_before_pairing",
            "cluster_bootstrap": "paired_percentile_resampling_of_50_subjects_with_replacement",
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "one_sided_confidence_level": confidence_level,
            "folds_trials_and_seeds_treated_as_independent": False,
        },
        "bottleneck_utility_noninferiority_gate": {
            "loss_definition": "plain_eegnet_minus_bottleneck_balanced_accuracy",
            "maximum_mean_balanced_accuracy_loss": margin,
            "observed_mean_balanced_accuracy_loss": mean_loss,
            "mean_loss_pass": bool(mean_pass),
            "observed_one_sided_upper_confidence_bound": loss_upper,
            "upper_bound_pass": bool(bound_pass),
            "gate_pass": gate_pass,
            "next_stage": (
                "attacks_from_frozen_posterior_caches_without_task_retraining"
                if gate_pass
                else "stop_negative_bottleneck_utility_result"
            ),
        },
        "models": {
            "compact_eegnet": {
                "subjects": len(subjects),
                "balanced_accuracy_subject_mean": baseline_mean,
            },
            MODEL: {
                "subjects": len(subjects),
                "task_seeds_per_subject": 4,
                "balanced_accuracy_subject_mean": bottleneck_mean,
                "balanced_accuracy_subject_std": float(
                    np.std(bottleneck_values, ddof=1)
                ),
                "balanced_accuracy_subject_min": float(np.min(bottleneck_values)),
                "balanced_accuracy_subject_max": float(np.max(bottleneck_values)),
                "balanced_accuracy_cluster_bootstrap_ci": {
                    "confidence_level": confidence_level,
                    "lower": float(bottleneck_two_sided[0]),
                    "upper": float(bottleneck_two_sided[1]),
                },
            },
        },
        "paired_loss_summary": {
            "subjects": len(subjects),
            "mean": mean_loss,
            "subject_std": float(np.std(losses, ddof=1)),
            "minimum": float(np.min(losses)),
            "maximum": float(np.max(losses)),
            "two_sided_cluster_bootstrap_ci": {
                "confidence_level": confidence_level,
                "lower": float(loss_two_sided[0]),
                "upper": float(loss_two_sided[1]),
            },
            "one_sided_upper_cluster_bootstrap_bound": loss_upper,
        },
        "fold_summaries": fold_summaries,
        "seed_summaries": seed_summaries,
        "paired_subject_aggregates": paired_rows,
        "jobs": public_jobs,
    }


def _format_record(payload: dict[str, Any]) -> str:
    gate = payload["bottleneck_utility_noninferiority_gate"]
    bottleneck = payload["models"][MODEL]
    baseline = payload["models"]["compact_eegnet"]
    paired = payload["paired_loss_summary"]
    status_sentence = (
        "The preregistered bottleneck utility noninferiority gate passed."
        if gate["gate_pass"]
        else "The preregistered bottleneck utility noninferiority gate failed, so privacy promotion stops."
    )
    lines = [
        "# Cho2017 Confirmatory Bottleneck Utility Gate",
        "",
        "## Status",
        "",
        status_sentence,
        "",
        "All 20 frozen compact bottleneck EEGNet (`dim=6`) jobs and all 20",
        "posterior caches were validated before comparison. Producing this record",
        "performed no training, download, model selection, or privacy attack.",
        "",
        "## Paired subject-level comparison",
        "",
        "The four task seeds were averaged within each held-out subject for each model.",
        "The analysis then paired plain EEGNet and bottleneck utility within the same",
        "50 subjects. Loss is defined as plain EEGNet minus bottleneck balanced",
        "accuracy; positive values therefore indicate lower bottleneck utility.",
        "",
        "| Measure | Plain compact EEGNet | Bottleneck dim 6 | Paired loss |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Mean subject balanced accuracy | "
            f"{baseline['balanced_accuracy_subject_mean']:.4f} | "
            f"{bottleneck['balanced_accuracy_subject_mean']:.4f} | "
            f"{paired['mean']:.4f} |"
        ),
        "",
        "The paired two-sided 95% subject-bootstrap interval for mean loss is",
        (
            f"[{paired['two_sided_cluster_bootstrap_ci']['lower']:.4f}, "
            f"{paired['two_sided_cluster_bootstrap_ci']['upper']:.4f}]. The one-sided "
            f"95% upper bound used for the gate is "
            f"{paired['one_sided_upper_cluster_bootstrap_bound']:.4f}."
        ),
        "",
        "## Gate decision",
        "",
        "| Criterion | Required | Observed | Result |",
        "| --- | ---: | ---: | --- |",
        (
            "| Mean balanced-accuracy loss | "
            f"< {gate['maximum_mean_balanced_accuracy_loss']:.3f} | "
            f"{gate['observed_mean_balanced_accuracy_loss']:.4f} | "
            f"{'PASS' if gate['mean_loss_pass'] else 'FAIL'} |"
        ),
        (
            "| One-sided 95% upper subject-bootstrap bound | "
            f"< {gate['maximum_mean_balanced_accuracy_loss']:.3f} | "
            f"{gate['observed_one_sided_upper_confidence_bound']:.4f} | "
            f"{'PASS' if gate['upper_bound_pass'] else 'FAIL'} |"
        ),
        "",
        "A passing result authorizes only the frozen attacker families on the saved",
        "plain and bottleneck posterior caches. It is not evidence of a privacy benefit;",
        "privacy promotion still requires the separate multiplicity-controlled attack gate.",
        "",
        "## Fold diagnostics",
        "",
        "Fold values are descriptive means of ten paired subject aggregates and are not",
        "treated as five independent inferential observations.",
        "",
        "| Fold | Plain EEGNet | Bottleneck dim 6 | Loss |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in payload["fold_summaries"]:
        lines.append(
            f"| {row['fold_index']} | "
            f"{row['plain_eegnet_balanced_accuracy_mean']:.4f} | "
            f"{row['bottleneck_balanced_accuracy_mean']:.4f} | "
            f"{row['balanced_accuracy_loss_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reproducibility record",
            "",
            "The tracked machine-readable artifact is",
            "[`configs/cho2017_confirmatory_bottleneck_gate_v1.json`](../configs/cho2017_confirmatory_bottleneck_gate_v1.json).",
            "It contains the paired subject aggregates, fold and seed diagnostics,",
            "source result hashes, and cache hashes, but no raw EEG or posterior arrays.",
            "",
            "After reproducing the local baseline and bottleneck jobs, verify this record with:",
            "",
            "```bash",
            "PYTHONPATH=src .venv/bin/python scripts/freeze_cho2017_bottleneck_gate.py --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze or verify the Cho2017 bottleneck utility noninferiority gate."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify tracked artifacts against completed local baseline and bottleneck jobs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()
    record_path = args.record.resolve()
    payload = _build_payload(input_dir)
    rendered_json = json.dumps(payload, indent=2) + "\n"
    rendered_record = _format_record(payload)

    if args.check:
        _require(output_path.is_file(), f"tracked gate artifact missing: {output_path}")
        _require(record_path.is_file(), f"tracked gate record missing: {record_path}")
        _require(
            output_path.read_text(encoding="utf-8") == rendered_json,
            "tracked gate artifact does not match completed bottleneck jobs",
        )
        _require(
            record_path.read_text(encoding="utf-8") == rendered_record,
            "tracked gate record does not match completed bottleneck jobs",
        )
        print(
            "PASS Cho2017 bottleneck utility gate "
            f"jobs={payload['provenance']['validated_result_jobs']} "
            f"subjects={payload['paired_loss_summary']['subjects']}"
        )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered_json, encoding="utf-8")
    record_path.write_text(rendered_record, encoding="utf-8")
    print(
        f"Wrote {output_path} and {record_path}; gate={payload['status']} "
        f"mean_loss={payload['bottleneck_utility_noninferiority_gate']['observed_mean_balanced_accuracy_loss']:.4f} "
        f"upper95={payload['bottleneck_utility_noninferiority_gate']['observed_one_sided_upper_confidence_bound']:.4f}"
    )


if __name__ == "__main__":
    main()
