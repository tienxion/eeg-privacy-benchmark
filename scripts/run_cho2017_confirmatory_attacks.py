from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

from eeg_privacy_benchmark.privacy.membership_inference import (
    load_posterior_feature_cache,
    run_cached_membership_inference_attack_with_subject_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cho2017_confirmatory_validation_v1.yaml"
BASELINE_GATE = ROOT / "configs" / "cho2017_confirmatory_baseline_gate_v1.json"
BOTTLENECK_GATE = ROOT / "configs" / "cho2017_confirmatory_bottleneck_gate_v1.json"
BOTTLENECK_GATE_FREEZER = ROOT / "scripts" / "freeze_cho2017_bottleneck_gate.py"
BASELINE_RUNNER = ROOT / "scripts" / "run_cho2017_confirmatory_baselines.py"
BASELINE_OUTPUT = ROOT / "outputs" / "v1_1_cho2017_baseline"
BOTTLENECK_OUTPUT = ROOT / "outputs" / "v1_1_cho2017_bottleneck"
DEFAULT_OUTPUT = ROOT / "outputs" / "v1_1_cho2017_attacks"
MODELS = ("compact_eegnet", "compact_bottleneck_eegnet_dim6")


@dataclass(frozen=True)
class AttackSpec:
    family: str
    attack_type: str
    score_type: str

    @property
    def slug(self) -> str:
        return f"{self.attack_type}_{self.score_type}"


@dataclass(frozen=True)
class AttackJob:
    model: str
    fold_index: int
    task_seed: int
    attack_seed: int
    spec: AttackSpec


ATTACK_SPECS = (
    AttackSpec("threshold", "threshold", "label_known_log_probability"),
    AttackSpec("threshold", "threshold", "max_probability"),
    AttackSpec(
        "logistic_regression",
        "logistic_regression",
        "posterior_probabilities",
    ),
    AttackSpec(
        "logistic_regression",
        "logistic_regression",
        "posterior_probabilities_plus_true_label",
    ),
    AttackSpec("mlp", "mlp", "posterior_probabilities"),
)


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load required script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _baseline_runner():
    return _load_script(BASELINE_RUNNER, "cho2017_baseline_runner_for_attacks")


def _load_tracked_gates(runner) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = json.loads(BASELINE_GATE.read_text(encoding="utf-8"))
    bottleneck = json.loads(BOTTLENECK_GATE.read_text(encoding="utf-8"))
    baseline_valid = (
        baseline["schema_version"] == 1
        and baseline["status"] == "PASS"
        and baseline["plain_eegnet_utility_gate"]["gate_pass"] is True
        and baseline["provenance"]["contract_sha256"] == runner._sha256(CONFIG)
        and baseline["provenance"]["validated_posterior_caches"] == 20
    )
    bottleneck_valid = (
        bottleneck["schema_version"] == 1
        and bottleneck["status"] == "PASS"
        and bottleneck["bottleneck_utility_noninferiority_gate"]["gate_pass"] is True
        and bottleneck["bottleneck_utility_noninferiority_gate"]["next_stage"]
        == "attacks_from_frozen_posterior_caches_without_task_retraining"
        and bottleneck["provenance"]["contract_sha256"] == runner._sha256(CONFIG)
        and bottleneck["provenance"]["baseline_gate_sha256"]
        == runner._sha256(BASELINE_GATE)
        and bottleneck["provenance"]["validated_posterior_caches"] == 20
    )
    if not baseline_valid or not bottleneck_valid:
        raise RuntimeError("tracked utility gates do not authorize Cho2017 attacks")
    return baseline, bottleneck


def _verify_local_bottleneck_gate() -> dict[str, Any]:
    tracked = json.loads(BOTTLENECK_GATE.read_text(encoding="utf-8"))
    freezer = _load_script(
        BOTTLENECK_GATE_FREEZER,
        "cho2017_bottleneck_gate_freezer_for_attacks",
    )
    rebuilt = freezer._build_payload(BOTTLENECK_OUTPUT.resolve())
    if rebuilt != tracked:
        raise RuntimeError(
            "local baseline/bottleneck artifacts do not reproduce the tracked gate"
        )
    return tracked


def _source_records(
    baseline_gate: dict[str, Any],
    bottleneck_gate: dict[str, Any],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    records: dict[tuple[str, int, int], dict[str, Any]] = {}
    for source_stage, gate, root in (
        ("baseline", baseline_gate, BASELINE_OUTPUT),
        ("bottleneck", bottleneck_gate, BOTTLENECK_OUTPUT),
    ):
        for row in gate["jobs"]:
            model = row["model"]
            if model not in MODELS:
                continue
            cache_record = row["posterior_cache"]
            relative_path = cache_record["relative_path"]
            path = root / relative_path
            key = (model, int(row["fold_index"]), int(row["seed"]))
            if key in records:
                raise RuntimeError(f"duplicate source cache record: {key}")
            records[key] = {
                "source_stage": source_stage,
                "root": root,
                "relative_path": relative_path,
                "path": path,
                "sha256": cache_record["sha256"],
            }
    expected = {
        (model, fold_index, seed)
        for model in MODELS
        for fold_index in range(1, 6)
        for seed in (13, 17, 19, 23)
    }
    if set(records) != expected:
        raise RuntimeError("tracked utility gates do not contain the frozen 40 caches")
    return records


def _jobs(config: dict[str, Any]) -> list[AttackJob]:
    return [
        AttackJob(model, fold_index, task_seed, attack_seed, spec)
        for model in MODELS
        for fold_index in range(1, 6)
        for task_seed in config["seeds"]["task"]
        for attack_seed in config["seeds"]["membership_attacker"]
        for spec in ATTACK_SPECS
    ]


def _job_path(output_dir: Path, job: AttackJob) -> Path:
    return (
        output_dir
        / "results"
        / (
            f"{job.model}_fold{job.fold_index}_taskseed{job.task_seed}_"
            f"attackseed{job.attack_seed}_{job.spec.slug}.json"
        )
    )


def _finite_probability(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= value <= 1.0


def _valid_completed_job(
    path: Path,
    *,
    job: AttackJob,
    source: dict[str, Any],
    runner,
    expected_evaluation_subjects: list[int],
) -> bool:
    if not path.is_file() or not source["path"].is_file():
        return False
    try:
        if runner._sha256(source["path"]) != source["sha256"]:
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload["attack_result"]
        subject_metrics = payload["subject_metrics"]
        valid = (
            payload["schema_version"] == 1
            and payload["status"] == "complete"
            and payload["stage"] == "cho2017_confirmatory_membership_attack"
            and payload["model"] == job.model
            and payload["fold_index"] == job.fold_index
            and payload["task_seed"] == job.task_seed
            and payload["attack_seed"] == job.attack_seed
            and payload["attacker_family"] == job.spec.family
            and payload["attack_type"] == job.spec.attack_type
            and payload["score_type"] == job.spec.score_type
            and payload["protocol"] == f"cho2017_confirmatory_fold_{job.fold_index}"
            and payload["contract_sha256"] == runner._sha256(CONFIG)
            and payload["baseline_gate_sha256"] == runner._sha256(BASELINE_GATE)
            and payload["bottleneck_gate_sha256"] == runner._sha256(BOTTLENECK_GATE)
            and payload["task_model_training"] is False
            and payload["source_cache"]["stage"] == source["source_stage"]
            and payload["source_cache"]["relative_path"] == source["relative_path"]
            and payload["source_cache"]["sha256"] == source["sha256"]
            and result["task_model"] == job.model
            and result["dataset_key"] == "cho2017"
            and result["protocol"] == f"cho2017_confirmatory_fold_{job.fold_index}"
            and result["seed"] == job.task_seed
            and result["attack_seed"] == job.attack_seed
            and result["attack_seed_policy"] == "explicit_fixed_seed"
            and result["attack_type"] == job.spec.attack_type
            and result["score_type"] == job.spec.score_type
            and result["attacker_evaluation_subjects"]
            == expected_evaluation_subjects
            and len(subject_metrics) == 15
            and [row["subject"] for row in subject_metrics]
            == expected_evaluation_subjects
            and all(
                row["member_trials"] > 0
                and row["nonmember_trials"] > 0
                and _finite_probability(row["attack_auc"])
                and _finite_probability(row["attack_average_precision"])
                and _finite_probability(row["attack_balanced_accuracy"])
                and math.isfinite(row["member_score_mean"])
                and math.isfinite(row["nonmember_score_mean"])
                for row in subject_metrics
            )
        )
        if job.model == "compact_eegnet":
            valid = valid and result.get("bottleneck_dim") is None
        else:
            valid = valid and result["bottleneck_dim"] == 6
        return bool(valid)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _run_job(
    *,
    job: AttackJob,
    source: dict[str, Any],
    runner,
    expected_training_subjects: list[int],
    expected_evaluation_subjects: list[int],
) -> dict[str, Any]:
    if runner._sha256(source["path"]) != source["sha256"]:
        raise RuntimeError("source posterior cache hash does not match the utility gate")
    cache = load_posterior_feature_cache(source["path"])
    expected_protocol = f"cho2017_confirmatory_fold_{job.fold_index}"
    valid_cache = (
        cache.task_model == job.model
        and cache.dataset_key == "cho2017"
        and cache.protocol == expected_protocol
        and cache.seed == job.task_seed
        and cache.attacker_training_subjects == expected_training_subjects
        and cache.attacker_evaluation_subjects == expected_evaluation_subjects
        and (cache.bottleneck_dim == 6 if job.model.endswith("dim6") else cache.bottleneck_dim is None)
    )
    if not valid_cache:
        raise RuntimeError("source posterior cache does not match the frozen attack job")
    result, subject_metrics = run_cached_membership_inference_attack_with_subject_metrics(
        source["path"],
        attack_type=job.spec.attack_type,
        score_type=job.spec.score_type,
        attack_seed=job.attack_seed,
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "stage": "cho2017_confirmatory_membership_attack",
        "model": job.model,
        "fold_index": job.fold_index,
        "task_seed": job.task_seed,
        "attack_seed": job.attack_seed,
        "attacker_family": job.spec.family,
        "attack_type": job.spec.attack_type,
        "score_type": job.spec.score_type,
        "protocol": expected_protocol,
        "contract_sha256": runner._sha256(CONFIG),
        "baseline_gate_sha256": runner._sha256(BASELINE_GATE),
        "bottleneck_gate_sha256": runner._sha256(BOTTLENECK_GATE),
        "task_model_training": False,
        "source_cache": {
            "stage": source["source_stage"],
            "relative_path": source["relative_path"],
            "sha256": source["sha256"],
        },
        "attack_result": result.to_dict(),
        "subject_metrics": subject_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or run the frozen Cho2017 posterior-cache membership attacks. "
            "Dry-run is the default and never trains a task model."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument(
        "--max-jobs",
        type=int,
        help="Stop after this many newly completed cache-only attack jobs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_jobs is not None and args.max_jobs < 1:
        raise ValueError("max-jobs must be positive")
    runner = _baseline_runner()
    config, _ = runner._load_sources()
    baseline_gate, bottleneck_gate = _load_tracked_gates(runner)
    source_records = _source_records(baseline_gate, bottleneck_gate)
    folds = {
        fold.fold_index: fold
        for fold in runner.build_outer_folds(config["cohort"]["shuffled_subject_blocks"])
    }
    attacker_partitions = {
        index: runner.partition_attacker_subjects(
            fold.fitting_subjects,
            seed=config["seeds"]["attacker_subject_partition"],
            fold_index=index,
        )
        for index, fold in folds.items()
    }
    jobs = _jobs(config)
    output_dir = args.output_dir.resolve()
    completed = []
    for job in jobs:
        source = source_records[(job.model, job.fold_index, job.task_seed)]
        attacker_partition = attacker_partitions[job.fold_index]
        if _valid_completed_job(
            _job_path(output_dir, job),
            job=job,
            source=source,
            runner=runner,
            expected_evaluation_subjects=list(attacker_partition.evaluation_subjects),
        ):
            completed.append(job)
    plan = {
        "status": "DRY_RUN" if not args.confirm_run else "AUTHORIZED",
        "network_downloads": False,
        "task_model_training": False,
        "models": list(MODELS),
        "folds": [1, 2, 3, 4, 5],
        "task_seeds": list(config["seeds"]["task"]),
        "attack_seeds": list(config["seeds"]["membership_attacker"]),
        "attack_specs": [
            {
                "family": spec.family,
                "attack_type": spec.attack_type,
                "score_type": spec.score_type,
            }
            for spec in ATTACK_SPECS
        ],
        "total_jobs": len(jobs),
        "completed_jobs": len(completed),
        "pending_jobs": len(jobs) - len(completed),
        "one_attack_job_at_a_time": True,
        "baseline_gate": baseline_gate["status"],
        "bottleneck_gate": bottleneck_gate["status"],
    }
    print(json.dumps(plan, indent=2))
    if not args.confirm_run:
        print("DRY RUN only; add --confirm-run to execute pending attack jobs.")
        return

    _verify_local_bottleneck_gate()
    newly_completed = 0
    for position, job in enumerate(jobs, start=1):
        path = _job_path(output_dir, job)
        source = source_records[(job.model, job.fold_index, job.task_seed)]
        attacker_partition = attacker_partitions[job.fold_index]
        if _valid_completed_job(
            path,
            job=job,
            source=source,
            runner=runner,
            expected_evaluation_subjects=list(attacker_partition.evaluation_subjects),
        ):
            print(
                f"[{position}/{len(jobs)}] SKIP {job.model} fold={job.fold_index} "
                f"task_seed={job.task_seed} attack_seed={job.attack_seed} "
                f"attack={job.spec.slug}"
            )
            continue
        print(
            f"[{position}/{len(jobs)}] RUN {job.model} fold={job.fold_index} "
            f"task_seed={job.task_seed} attack_seed={job.attack_seed} "
            f"attack={job.spec.slug}"
        )
        payload = _run_job(
            job=job,
            source=source,
            runner=runner,
            expected_training_subjects=list(attacker_partition.training_subjects),
            expected_evaluation_subjects=list(attacker_partition.evaluation_subjects),
        )
        runner._write_json_atomic(path, payload)
        print(
            f"COMPLETE {job.spec.slug} attack_auc="
            f"{payload['attack_result']['attack_auc']:.4f}"
        )
        newly_completed += 1
        if args.max_jobs is not None and newly_completed >= args.max_jobs:
            print(f"Reached --max-jobs={args.max_jobs}; resume with the same command.")
            break


if __name__ == "__main__":
    main()
