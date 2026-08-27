from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cho2017_confirmatory_validation_v1.yaml"
CACHE_MANIFEST = ROOT / "configs" / "cho2017_confirmatory_cache_manifest_v1.json"
RUNNER = ROOT / "scripts" / "run_cho2017_confirmatory_baselines.py"
DEFAULT_INPUT = ROOT / "outputs" / "v1_1_cho2017_baseline"
DEFAULT_OUTPUT = ROOT / "configs" / "cho2017_confirmatory_baseline_gate_v1.json"
DEFAULT_RECORD = ROOT / "docs" / "V1_1_CHO2017_BASELINE_GATE.md"
MODELS = ("csp_lda", "compact_eegnet")
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


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_cho2017_confirmatory_baselines_for_gate",
        RUNNER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load baseline runner: {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validated_jobs(input_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runner = _load_runner()
    config, manifest = runner._load_sources()
    folds = {
        fold.fold_index: fold
        for fold in runner.build_outer_folds(config["cohort"]["shuffled_subject_blocks"])
    }
    trials_per_class = {
        int(row["subject"]): int(row["trials_per_class"])
        for row in manifest["rows"]
    }
    expected_paths = {
        input_dir / "results" / f"{model}_fold{fold}_seed{seed}.json"
        for model in MODELS
        for fold in range(1, 6)
        for seed in config["seeds"]["task"]
    }
    observed_paths = set((input_dir / "results").glob("*.json"))
    _require(observed_paths == expected_paths, "baseline result file set is not the frozen 40 jobs")

    expected_caches = {
        input_dir / "posterior_caches" / f"compact_eegnet_fold{fold}_seed{seed}.npz"
        for fold in range(1, 6)
        for seed in config["seeds"]["task"]
    }
    observed_caches = set((input_dir / "posterior_caches").glob("*.npz"))
    _require(observed_caches == expected_caches, "posterior cache file set is not the frozen 20 jobs")

    jobs: list[dict[str, Any]] = []
    for model in MODELS:
        for fold_index in range(1, 6):
            fold = folds[fold_index]
            role_counts = runner._expected_role_counts(fold, trials_per_class)
            attacker = runner.partition_attacker_subjects(
                fold.fitting_subjects,
                seed=config["seeds"]["attacker_subject_partition"],
                fold_index=fold_index,
            )
            for seed in config["seeds"]["task"]:
                path = input_dir / "results" / f"{model}_fold{fold_index}_seed{seed}.json"
                _require(
                    runner._valid_completed_job(
                        path,
                        model=model,
                        fold=fold,
                        seed=seed,
                        output_dir=input_dir,
                        expected_role_counts=role_counts,
                        attacker_partition=attacker,
                    ),
                    f"invalid baseline job: {path.name}",
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
                    f"invalid subject metrics: {path.name}",
                )
                task_result = payload["task_result"]
                _require(
                    0.0 <= task_result["balanced_accuracy"] <= 1.0
                    and 0.0 <= task_result["macro_f1"] <= 1.0,
                    f"invalid task metrics: {path.name}",
                )
                if model == "compact_eegnet":
                    _require(
                        task_result["bottleneck_dim"] is None
                        and task_result["resample_hz"] == 250
                        and task_result["normalization"] == "per_trial_channel"
                        and task_result["classifier_head"] == "global_average",
                        f"plain EEGNet provenance mismatch: {path.name}",
                    )
                jobs.append(
                    {
                        "model": model,
                        "fold_index": fold_index,
                        "seed": seed,
                        "result_file": str(path.relative_to(input_dir)),
                        "result_sha256": _sha256(path),
                        "task_balanced_accuracy": float(task_result["balanced_accuracy"]),
                        "task_macro_f1": float(task_result["macro_f1"]),
                        "subject_metrics": subject_rows,
                        "posterior_cache": payload["posterior_cache"],
                    }
                )
    return jobs, config


def _subject_rows(jobs: list[dict[str, Any]], subjects: list[int]) -> list[dict[str, Any]]:
    observations: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    fold_by_subject: dict[int, int] = {}
    for job in jobs:
        for row in job["subject_metrics"]:
            subject = int(row["subject"])
            observations[(job["model"], subject)].append(row)
            previous_fold = fold_by_subject.setdefault(subject, job["fold_index"])
            _require(previous_fold == job["fold_index"], "subject appears in multiple test folds")

    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for subject in subjects:
            values = observations[(model, subject)]
            _require(len(values) == 4, f"{model} subject {subject} does not have four task seeds")
            rows.append(
                {
                    "model": model,
                    "subject": subject,
                    "test_fold": fold_by_subject[subject],
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


def _model_summaries(
    jobs: list[dict[str, Any]],
    subject_rows: list[dict[str, Any]],
    subjects: list[int],
    *,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0,
        len(subjects),
        size=(BOOTSTRAP_REPLICATES, len(subjects)),
    )
    alpha = 1.0 - confidence_level
    summaries: dict[str, Any] = {}
    for model in MODELS:
        model_subject_rows = [row for row in subject_rows if row["model"] == model]
        _require(
            [row["subject"] for row in model_subject_rows] == subjects,
            f"{model} subject summary order mismatch",
        )
        values = np.asarray(
            [row["balanced_accuracy_mean"] for row in model_subject_rows], dtype=float
        )
        bootstrap_means = values[indices].mean(axis=1)
        lower, upper = np.quantile(
            bootstrap_means,
            [alpha / 2.0, 1.0 - alpha / 2.0],
        )
        model_jobs = [job for job in jobs if job["model"] == model]
        fold_summaries = []
        for fold_index in range(1, 6):
            fold_values = [
                row["balanced_accuracy_mean"]
                for row in model_subject_rows
                if row["test_fold"] == fold_index
            ]
            fold_summaries.append(
                {
                    "fold_index": fold_index,
                    "subjects": len(fold_values),
                    "balanced_accuracy_mean": float(np.mean(fold_values)),
                }
            )
        seed_summaries = []
        for seed in sorted({job["seed"] for job in model_jobs}):
            seed_subject_values = [
                metric["balanced_accuracy"]
                for job in model_jobs
                if job["seed"] == seed
                for metric in job["subject_metrics"]
            ]
            seed_summaries.append(
                {
                    "seed": seed,
                    "subjects": len(seed_subject_values),
                    "balanced_accuracy_mean": float(np.mean(seed_subject_values)),
                }
            )
        summaries[model] = {
            "subjects": len(values),
            "task_seeds_per_subject": 4,
            "balanced_accuracy_subject_mean": float(np.mean(values)),
            "balanced_accuracy_subject_std": float(np.std(values, ddof=1)),
            "balanced_accuracy_subject_min": float(np.min(values)),
            "balanced_accuracy_subject_max": float(np.max(values)),
            "balanced_accuracy_cluster_bootstrap_ci": {
                "confidence_level": confidence_level,
                "lower": float(lower),
                "upper": float(upper),
            },
            "macro_f1_subject_mean": float(
                np.mean([row["macro_f1_mean"] for row in model_subject_rows])
            ),
            "fold_summaries": fold_summaries,
            "seed_summaries": seed_summaries,
        }
    return summaries


def _result_set_sha256(jobs: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for job in jobs:
        digest.update(job["result_file"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(job["result_sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _build_payload(input_dir: Path) -> dict[str, Any]:
    jobs, config = _validated_jobs(input_dir)
    subjects = list(config["cohort"]["confirmatory_subjects"])
    confidence_level = float(config["inference"]["confidence_level"])
    bootstrap_seed = int(config["cohort"]["partition_seed"])
    subjects_aggregated = _subject_rows(jobs, subjects)
    models = _model_summaries(
        jobs,
        subjects_aggregated,
        subjects,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    threshold = config["promotion_rules"]["plain_eegnet_utility"]
    eegnet = models["compact_eegnet"]
    mean_pass = (
        eegnet["balanced_accuracy_subject_mean"]
        >= threshold["minimum_mean_subject_balanced_accuracy"]
    )
    lower_bound = eegnet["balanced_accuracy_cluster_bootstrap_ci"]["lower"]
    lower_pass = lower_bound > threshold["lower_confidence_bound_must_exceed"]
    gate_pass = bool(mean_pass and lower_pass)
    public_jobs = [
        {key: value for key, value in job.items() if key != "subject_metrics"}
        for job in jobs
    ]
    return {
        "schema_version": 1,
        "stage": "cho2017_confirmatory_baseline_utility_gate",
        "status": "PASS" if gate_pass else "STOP",
        "recorded_date": RECORDED_DATE,
        "provenance": {
            "contract_sha256": _sha256(CONFIG),
            "cache_manifest_sha256": _sha256(CACHE_MANIFEST),
            "result_set_sha256": _result_set_sha256(jobs),
            "validated_result_jobs": len(jobs),
            "validated_posterior_caches": sum(
                job["posterior_cache"] is not None for job in jobs
            ),
            "network_or_training": False,
        },
        "inference": {
            "unit": "subject",
            "task_seed_aggregation": "arithmetic_mean_within_subject_before_inference",
            "cluster_bootstrap": "percentile_resampling_of_50_subjects_with_replacement",
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "confidence_level": confidence_level,
            "interval_sides": "two_sided",
            "folds_trials_and_seeds_treated_as_independent": False,
        },
        "plain_eegnet_utility_gate": {
            "minimum_mean_subject_balanced_accuracy": threshold[
                "minimum_mean_subject_balanced_accuracy"
            ],
            "observed_mean_subject_balanced_accuracy": eegnet[
                "balanced_accuracy_subject_mean"
            ],
            "mean_threshold_pass": bool(mean_pass),
            "lower_confidence_bound_must_exceed": threshold[
                "lower_confidence_bound_must_exceed"
            ],
            "observed_lower_confidence_bound": lower_bound,
            "lower_bound_pass": bool(lower_pass),
            "gate_pass": gate_pass,
            "next_stage": (
                "compact_bottleneck_eegnet_dim6"
                if gate_pass
                else "stop_negative_task_transfer_result"
            ),
        },
        "models": models,
        "subject_aggregates": subjects_aggregated,
        "jobs": public_jobs,
    }


def _format_record(payload: dict[str, Any]) -> str:
    gate = payload["plain_eegnet_utility_gate"]
    models = payload["models"]
    status_sentence = (
        "The preregistered plain-EEGNet utility gate passed."
        if gate["gate_pass"]
        else "The preregistered plain-EEGNet utility gate failed, so the study stops."
    )
    lines = [
        "# Cho2017 Confirmatory Baseline Utility Gate",
        "",
        "## Status",
        "",
        f"{status_sentence} All 40 frozen baseline jobs and all 20 plain-EEGNet",
        "posterior caches were validated before aggregation. No training, download,",
        "model selection, or privacy attack was performed while producing this record.",
        "",
        "## Subject-level utility",
        "",
        "Repeated task seeds were averaged within each held-out subject first. The 50",
        "subject aggregates were then treated as the inferential observations. The",
        "interval is a deterministic two-sided 95% percentile bootstrap over subjects",
        (
            f"({payload['inference']['bootstrap_replicates']:,} replicates; seed "
            f"`{payload['inference']['bootstrap_seed']}`)."
        ),
        "",
        "| Model | Mean balanced accuracy | Subject SD | 95% subject-bootstrap CI | Mean macro-F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    labels = {"csp_lda": "CSP-LDA", "compact_eegnet": "Plain compact EEGNet"}
    for model in MODELS:
        summary = models[model]
        interval = summary["balanced_accuracy_cluster_bootstrap_ci"]
        lines.append(
            f"| {labels[model]} | {summary['balanced_accuracy_subject_mean']:.4f} | "
            f"{summary['balanced_accuracy_subject_std']:.4f} | "
            f"[{interval['lower']:.4f}, {interval['upper']:.4f}] | "
            f"{summary['macro_f1_subject_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Gate decision",
            "",
            "| Criterion | Required | Observed | Result |",
            "| --- | ---: | ---: | --- |",
            (
                "| Plain-EEGNet mean subject balanced accuracy | "
                f">= {gate['minimum_mean_subject_balanced_accuracy']:.2f} | "
                f"{gate['observed_mean_subject_balanced_accuracy']:.4f} | "
                f"{'PASS' if gate['mean_threshold_pass'] else 'FAIL'} |"
            ),
            (
                "| 95% lower subject-bootstrap bound | "
                f"> {gate['lower_confidence_bound_must_exceed']:.2f} | "
                f"{gate['observed_lower_confidence_bound']:.4f} | "
                f"{'PASS' if gate['lower_bound_pass'] else 'FAIL'} |"
            ),
            "",
            "The gate result authorizes only the frozen compact bottleneck EEGNet",
            "(`dim=6`) stage. It is not evidence of a privacy benefit and does not",
            "authorize attacks until the bottleneck utility noninferiority gate passes.",
            "",
            "## Fold diagnostics",
            "",
            "Fold values are descriptive means of the ten held-out subject aggregates;",
            "they are not treated as five independent inferential observations.",
            "",
            "| Fold | CSP-LDA | Plain compact EEGNet |",
            "| ---: | ---: | ---: |",
        ]
    )
    for fold_index in range(1, 6):
        csp = models["csp_lda"]["fold_summaries"][fold_index - 1]
        eegnet = models["compact_eegnet"]["fold_summaries"][fold_index - 1]
        lines.append(
            f"| {fold_index} | {csp['balanced_accuracy_mean']:.4f} | "
            f"{eegnet['balanced_accuracy_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reproducibility record",
            "",
            "The tracked machine-readable artifact is",
            "[`configs/cho2017_confirmatory_baseline_gate_v1.json`](../configs/cho2017_confirmatory_baseline_gate_v1.json).",
            "It contains the 50 subject aggregates, fold and seed diagnostics, source",
            "result hashes, and posterior-cache hashes, but no raw EEG or posterior arrays.",
            "",
            "After reproducing the 40 local jobs, verify this record with:",
            "",
            "```bash",
            "PYTHONPATH=src .venv/bin/python scripts/freeze_cho2017_baseline_gate.py --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze or verify the Cho2017 confirmatory baseline utility gate."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify tracked artifacts against the completed local baseline jobs.",
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
            "tracked gate artifact does not match the completed baseline jobs",
        )
        _require(
            record_path.read_text(encoding="utf-8") == rendered_record,
            "tracked gate record does not match the completed baseline jobs",
        )
        print(
            "PASS Cho2017 baseline utility gate "
            f"jobs={payload['provenance']['validated_result_jobs']} "
            f"subjects={payload['models']['compact_eegnet']['subjects']}"
        )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered_json, encoding="utf-8")
    record_path.write_text(rendered_record, encoding="utf-8")
    print(
        f"Wrote {output_path} and {record_path}; "
        f"gate={payload['status']} mean="
        f"{payload['plain_eegnet_utility_gate']['observed_mean_subject_balanced_accuracy']:.4f} "
        f"lower95={payload['plain_eegnet_utility_gate']['observed_lower_confidence_bound']:.4f}"
    )


if __name__ == "__main__":
    main()
