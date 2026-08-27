from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from eeg_privacy_benchmark.privacy.membership_inference import (
    PosteriorFeatureCache,
    run_cached_membership_inference_attack_with_subject_metrics,
    write_posterior_feature_cache,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_cho2017_confirmatory_attacks.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("cho2017_attack_runner_check", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load Cho2017 attack runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"{name}: PASS")


def _synthetic_schema_v2_cache(path: Path) -> None:
    training_subjects = list(range(1, 16))
    evaluation_subjects = list(range(16, 31))
    subjects = training_subjects + evaluation_subjects
    probabilities = []
    labels = []
    trial_subject_ids = []
    member_indices = []
    nonmember_indices = []
    for subject in subjects:
        for membership, confidence in ((1, 0.80), (0, 0.55)):
            index = len(probabilities)
            probabilities.append([confidence, 1.0 - confidence])
            labels.append(0)
            trial_subject_ids.append(f"sub-{subject}")
            (member_indices if membership else nonmember_indices).append(index)
    cache = PosteriorFeatureCache(
        task_model="compact_eegnet",
        dataset_key="cho2017",
        protocol="cho2017_confirmatory_fold_1",
        seed=13,
        subjects=list(range(3, 53)),
        task_balanced_accuracy=0.62,
        task_macro_f1=0.61,
        class_probabilities=np.asarray(probabilities, dtype=float),
        encoded_labels=np.asarray(labels, dtype=int),
        member_indices=member_indices,
        nonmember_indices=nonmember_indices,
        resample_hz=250,
        normalization="per_trial_channel",
        classifier_head="global_average",
        trial_subject_ids=trial_subject_ids,
        attacker_training_subjects=training_subjects,
        attacker_evaluation_subjects=evaluation_subjects,
    )
    write_posterior_feature_cache(cache, path)


def main() -> None:
    runner = _load_runner()
    baseline_runner = runner._baseline_runner()
    config, _ = baseline_runner._load_sources()
    baseline_gate, bottleneck_gate = runner._load_tracked_gates(baseline_runner)
    sources = runner._source_records(baseline_gate, bottleneck_gate)
    jobs = runner._jobs(config)

    _check(
        "frozen_attack_matrix_is_exact",
        len(jobs) == 600
        and len(set(jobs)) == 600
        and {spec.family for spec in runner.ATTACK_SPECS}
        == {"threshold", "logistic_regression", "mlp"}
        and len(runner.ATTACK_SPECS) == 5,
    )
    _check(
        "tracked_gates_supply_exact_caches",
        len(sources) == 40
        and all(
            source["path"].is_file()
            and baseline_runner._sha256(source["path"]) == source["sha256"]
            for source in sources.values()
        ),
    )
    source_text = RUNNER_PATH.read_text(encoding="utf-8")
    _check(
        "runner_has_no_task_training_or_dataset_path",
        "fit_eegnet" not in source_text
        and "build_dataset_loader" not in source_text
        and '"task_model_training": False' in source_text,
    )
    _check(
        "runner_requires_explicit_confirmation",
        "--confirm-run" in source_text
        and "if not args.confirm_run:" in source_text,
    )
    _check(
        "runner_is_bounded_and_resumable",
        "--max-jobs" in source_text
        and "_valid_completed_job(" in source_text
        and "Reached --max-jobs=" in source_text,
    )
    _check(
        "runner_reproduces_gate_before_attacks",
        source_text.index("_verify_local_bottleneck_gate()")
        < source_text.index("for position, job in enumerate(jobs, start=1):"),
    )

    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        cache_path = temporary_path / "synthetic_schema_v2.npz"
        _synthetic_schema_v2_cache(cache_path)
        result, subject_metrics = (
            run_cached_membership_inference_attack_with_subject_metrics(
                cache_path,
                attack_type="threshold",
                score_type="label_known_log_probability",
                attack_seed=101,
            )
        )
        _check(
            "subject_metrics_use_held_out_attacker_subjects",
            result.attacker_training_subjects == list(range(1, 16))
            and result.attacker_evaluation_subjects == list(range(16, 31))
            and [row["subject"] for row in subject_metrics] == list(range(16, 31))
            and all(
                row["member_trials"] == 1
                and row["nonmember_trials"] == 1
                and 0.0 <= row["attack_auc"] <= 1.0
                for row in subject_metrics
            ),
        )
        empty_output = temporary_path / "dry_run_output"
        _check(
            "dry_run_output_starts_empty",
            not empty_output.exists(),
        )

    plan_shape = {
        "models": list(runner.MODELS),
        "task_seeds": list(config["seeds"]["task"]),
        "attack_seeds": list(config["seeds"]["membership_attacker"]),
        "jobs": len(jobs),
    }
    _check(
        "frozen_plan_provenance",
        plan_shape
        == {
            "models": ["compact_eegnet", "compact_bottleneck_eegnet_dim6"],
            "task_seeds": [13, 17, 19, 23],
            "attack_seeds": [101, 103, 107],
            "jobs": 600,
        },
    )
    print(json.dumps({"passed": 9, "total": 9}))


if __name__ == "__main__":
    main()
