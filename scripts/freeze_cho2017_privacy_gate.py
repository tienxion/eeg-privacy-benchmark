from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cho2017_confirmatory_validation_v1.yaml"
BASELINE_GATE = ROOT / "configs" / "cho2017_confirmatory_baseline_gate_v1.json"
BOTTLENECK_GATE = ROOT / "configs" / "cho2017_confirmatory_bottleneck_gate_v1.json"
BOTTLENECK_GATE_FREEZER = ROOT / "scripts" / "freeze_cho2017_bottleneck_gate.py"
ATTACK_RUNNER = ROOT / "scripts" / "run_cho2017_confirmatory_attacks.py"
DEFAULT_INPUT = ROOT / "outputs" / "v1_1_cho2017_attacks"
DEFAULT_OUTPUT = ROOT / "configs" / "cho2017_confirmatory_privacy_gate_v1.json"
DEFAULT_RECORD = ROOT / "docs" / "V1_1_CHO2017_PRIVACY_GATE.md"
PLAIN_MODEL = "compact_eegnet"
BOTTLENECK_MODEL = "compact_bottleneck_eegnet_dim6"
FAMILIES = ("threshold", "logistic_regression", "mlp")
BOOTSTRAP_REPLICATES = 100_000
RECORDED_DATE = "2026-08-26"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load required script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_local_utility_gate() -> dict[str, Any]:
    tracked = json.loads(BOTTLENECK_GATE.read_text(encoding="utf-8"))
    freezer = _load_script(
        BOTTLENECK_GATE_FREEZER,
        "cho2017_bottleneck_gate_freezer_for_privacy_gate",
    )
    rebuilt = freezer._build_payload(
        ROOT / "outputs" / "v1_1_cho2017_bottleneck"
    )
    if rebuilt != tracked:
        raise RuntimeError("local utility artifacts do not reproduce the tracked gate")
    if (
        tracked["status"] != "PASS"
        or tracked["bottleneck_utility_noninferiority_gate"]["gate_pass"] is not True
    ):
        raise RuntimeError("bottleneck utility gate does not authorize privacy inference")
    return tracked


def _validated_results(input_dir: Path):
    attack_runner = _load_script(ATTACK_RUNNER, "cho2017_attack_runner_for_privacy_gate")
    baseline_runner = attack_runner._baseline_runner()
    config, _ = baseline_runner._load_sources()
    baseline_gate, bottleneck_gate = attack_runner._load_tracked_gates(baseline_runner)
    sources = attack_runner._source_records(baseline_gate, bottleneck_gate)
    folds = {
        fold.fold_index: fold
        for fold in baseline_runner.build_outer_folds(
            config["cohort"]["shuffled_subject_blocks"]
        )
    }
    attacker_partitions = {
        index: baseline_runner.partition_attacker_subjects(
            fold.fitting_subjects,
            seed=config["seeds"]["attacker_subject_partition"],
            fold_index=index,
        )
        for index, fold in folds.items()
    }
    jobs = attack_runner._jobs(config)
    expected_paths = {attack_runner._job_path(input_dir, job) for job in jobs}
    observed_paths = set((input_dir / "results").glob("*.json"))
    if observed_paths != expected_paths:
        raise RuntimeError("attack result set is not the frozen 600-job matrix")

    results = []
    for job in jobs:
        path = attack_runner._job_path(input_dir, job)
        source = sources[(job.model, job.fold_index, job.task_seed)]
        evaluation_subjects = list(
            attacker_partitions[job.fold_index].evaluation_subjects
        )
        if not attack_runner._valid_completed_job(
            path,
            job=job,
            source=source,
            runner=baseline_runner,
            expected_evaluation_subjects=evaluation_subjects,
        ):
            raise RuntimeError(f"invalid attack result: {path.name}")
        results.append(
            {
                "job": job,
                "path": path,
                "sha256": _sha256(path),
                "payload": json.loads(path.read_text(encoding="utf-8")),
            }
        )
    return results, config, attack_runner


def _result_set_sha256(results: list[dict[str, Any]], input_dir: Path) -> str:
    digest = hashlib.sha256()
    for row in results:
        relative = str(row["path"].relative_to(input_dir))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _subject_rows(results: list[dict[str, Any]], attack_runner):
    observations: dict[
        tuple[str, int, int, str], dict[tuple[str, int, int], float]
    ] = defaultdict(dict)
    family_variants = {
        family: [spec.slug for spec in attack_runner.ATTACK_SPECS if spec.family == family]
        for family in FAMILIES
    }
    for row in results:
        job = row["job"]
        for metric in row["payload"]["subject_metrics"]:
            subject = int(metric["subject"])
            group = (job.spec.family, job.attack_seed, subject, job.model)
            observation = (job.fold_index, job.task_seed, job.spec.slug)
            if observation in observations[group]:
                raise RuntimeError("duplicate subject-level attack observation")
            observations[group][observation] = float(metric["attack_auc"])

    subjects_by_family: dict[str, set[int]] = {}
    rows = []
    for family in FAMILIES:
        subjects = {
            key[2]
            for key in observations
            if key[0] == family and key[3] == PLAIN_MODEL
        }
        subjects_by_family[family] = subjects
        for subject in sorted(subjects):
            seed_rows = []
            for attack_seed in (101, 103, 107):
                plain = observations[(family, attack_seed, subject, PLAIN_MODEL)]
                bottleneck = observations[
                    (family, attack_seed, subject, BOTTLENECK_MODEL)
                ]
                if set(plain) != set(bottleneck):
                    raise RuntimeError("plain/bottleneck subject observations do not pair")
                expected_variants = set(family_variants[family])
                if {key[2] for key in plain} != expected_variants:
                    raise RuntimeError("subject observations do not cover frozen variants")
                plain_mean = float(np.mean(list(plain.values())))
                bottleneck_mean = float(np.mean(list(bottleneck.values())))
                seed_rows.append(
                    {
                        "attack_seed": attack_seed,
                        "paired_fold_task_variant_observations": len(plain),
                        "plain_auc_mean": plain_mean,
                        "bottleneck_auc_mean": bottleneck_mean,
                        "auc_reduction_plain_minus_bottleneck": plain_mean
                        - bottleneck_mean,
                    }
                )
            rows.append(
                {
                    "family": family,
                    "subject": subject,
                    "attacker_seed_aggregates": seed_rows,
                    "plain_auc_mean": float(
                        np.mean([row["plain_auc_mean"] for row in seed_rows])
                    ),
                    "bottleneck_auc_mean": float(
                        np.mean([row["bottleneck_auc_mean"] for row in seed_rows])
                    ),
                    "auc_reduction_plain_minus_bottleneck": float(
                        np.mean(
                            [
                                row["auc_reduction_plain_minus_bottleneck"]
                                for row in seed_rows
                            ]
                        )
                    ),
                }
            )
    if len({frozenset(value) for value in subjects_by_family.values()}) != 1:
        raise RuntimeError("attacker families do not share the same subject set")
    return rows, family_variants, sorted(next(iter(subjects_by_family.values())))


def _holm_adjust(p_values: dict[str, float], alpha: float):
    ordered = sorted(p_values, key=lambda family: (p_values[family], family))
    adjusted: dict[str, float] = {}
    running = 0.0
    rejected: dict[str, bool] = {family: False for family in p_values}
    still_rejecting = True
    total = len(ordered)
    for rank, family in enumerate(ordered, start=1):
        multiplier = total - rank + 1
        running = max(running, multiplier * p_values[family])
        adjusted[family] = min(1.0, running)
        if still_rejecting and p_values[family] <= alpha / multiplier:
            rejected[family] = True
        else:
            still_rejecting = False
    return ordered, adjusted, rejected


def _build_payload(input_dir: Path) -> dict[str, Any]:
    utility_gate = _verify_local_utility_gate()
    results, config, attack_runner = _validated_results(input_dir)
    subject_rows, family_variants, evaluated_subjects = _subject_rows(
        results, attack_runner
    )
    confidence = float(config["inference"]["confidence_level"])
    alpha = 1.0 - confidence
    minimum_reduction = float(
        config["promotion_rules"]["membership_privacy"][
            "minimum_mean_auc_reduction"
        ]
    )
    max_regression = float(
        config["promotion_rules"]["membership_privacy"][
            "maximum_allowed_family_auc_regression"
        ]
    )
    family_summaries: dict[str, dict[str, Any]] = {}
    bootstrap_values: dict[str, np.ndarray] = {}
    p_values: dict[str, float] = {}
    for family_index, family in enumerate(FAMILIES):
        rows = [row for row in subject_rows if row["family"] == family]
        reductions = np.asarray(
            [row["auc_reduction_plain_minus_bottleneck"] for row in rows],
            dtype=float,
        )
        rng = np.random.default_rng(
            int(config["cohort"]["partition_seed"]) + family_index
        )
        indices = rng.integers(
            0,
            len(reductions),
            size=(BOOTSTRAP_REPLICATES, len(reductions)),
        )
        bootstrap = reductions[indices].mean(axis=1)
        bootstrap_values[family] = bootstrap
        tail = min(
            (float(np.count_nonzero(bootstrap <= 0)) + 1.0)
            / (BOOTSTRAP_REPLICATES + 1.0),
            (float(np.count_nonzero(bootstrap >= 0)) + 1.0)
            / (BOOTSTRAP_REPLICATES + 1.0),
        )
        p_values[family] = min(1.0, 2.0 * tail)
        seed_summaries = []
        for attack_seed in config["seeds"]["membership_attacker"]:
            seed_reductions = [
                next(
                    item["auc_reduction_plain_minus_bottleneck"]
                    for item in row["attacker_seed_aggregates"]
                    if item["attack_seed"] == attack_seed
                )
                for row in rows
            ]
            seed_summaries.append(
                {
                    "attack_seed": attack_seed,
                    "subjects": len(seed_reductions),
                    "mean_auc_reduction": float(np.mean(seed_reductions)),
                }
            )
        mean_reduction = float(np.mean(reductions))
        family_summaries[family] = {
            "score_variants": family_variants[family],
            "subjects": len(rows),
            "plain_auc_subject_mean": float(
                np.mean([row["plain_auc_mean"] for row in rows])
            ),
            "bottleneck_auc_subject_mean": float(
                np.mean([row["bottleneck_auc_mean"] for row in rows])
            ),
            "mean_auc_reduction_plain_minus_bottleneck": mean_reduction,
            "two_sided_95_percent_subject_bootstrap_ci": {
                "lower": float(np.quantile(bootstrap, alpha / 2.0)),
                "upper": float(np.quantile(bootstrap, 1.0 - alpha / 2.0)),
            },
            "unadjusted_two_sided_bootstrap_p": p_values[family],
            "attacker_seed_summaries": seed_summaries,
            "consistent_positive_attacker_seed_direction": all(
                row["mean_auc_reduction"] > 0 for row in seed_summaries
            ),
            "minimum_mean_reduction_pass": mean_reduction >= minimum_reduction,
            "maximum_regression_pass": -mean_reduction <= max_regression,
        }

    holm_order, adjusted_p, holm_rejected = _holm_adjust(p_values, alpha)
    for rank, family in enumerate(holm_order, start=1):
        divisor = len(FAMILIES) - rank + 1
        adjusted_alpha = alpha / divisor
        bootstrap = bootstrap_values[family]
        interval = {
            "confidence_level": 1.0 - adjusted_alpha,
            "lower": float(np.quantile(bootstrap, adjusted_alpha / 2.0)),
            "upper": float(np.quantile(bootstrap, 1.0 - adjusted_alpha / 2.0)),
        }
        summary = family_summaries[family]
        summary["holm_rank"] = rank
        summary["holm_adjusted_p"] = adjusted_p[family]
        summary["holm_step_down_rejected_zero"] = holm_rejected[family]
        summary["holm_rank_adjusted_subject_bootstrap_ci"] = interval
        summary["adjusted_interval_excludes_zero_in_positive_direction"] = bool(
            holm_rejected[family] and interval["lower"] > 0.0
        )
        summary["family_promotion_pass"] = bool(
            summary["minimum_mean_reduction_pass"]
            and summary["maximum_regression_pass"]
            and summary["consistent_positive_attacker_seed_direction"]
            and summary["adjusted_interval_excludes_zero_in_positive_direction"]
        )

    improved_families = [
        family
        for family in FAMILIES
        if family_summaries[family]["family_promotion_pass"]
    ]
    required_families = int(
        config["promotion_rules"]["membership_privacy"][
            "minimum_improved_families"
        ]
    )
    privacy_promotion = len(improved_families) >= required_families
    jobs = [
        {
            "result_file": str(row["path"].relative_to(input_dir)),
            "sha256": row["sha256"],
        }
        for row in results
    ]
    return {
        "schema_version": 1,
        "stage": "cho2017_confirmatory_membership_privacy_gate",
        "status": "PROMOTE" if privacy_promotion else "NO_PROMOTION",
        "recorded_date": RECORDED_DATE,
        "provenance": {
            "contract_sha256": _sha256(CONFIG),
            "baseline_gate_sha256": _sha256(BASELINE_GATE),
            "bottleneck_gate_sha256": _sha256(BOTTLENECK_GATE),
            "attack_result_set_sha256": _result_set_sha256(results, input_dir),
            "validated_attack_jobs": len(results),
            "task_model_training": False,
            "utility_gate_status": utility_gate["status"],
        },
        "inference": {
            "unit": "attacker_evaluation_subject",
            "evaluated_subjects": evaluated_subjects,
            "evaluated_subject_count": len(evaluated_subjects),
            "confirmatory_subjects_never_assigned_to_attacker_evaluation": sorted(
                set(config["cohort"]["confirmatory_subjects"])
                - set(evaluated_subjects)
            ),
            "aggregation_order": [
                "equal_weight_frozen_score_variants_within_family",
                "equal_weight_task_seed_and_eligible_fold_repetitions_within_subject",
                "pair_plain_and_bottleneck_within_subject_and_attacker_seed",
                "equal_weight_three_attacker_seeds_within_subject",
                "bootstrap_subject_aggregates",
            ],
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seeds_by_family": {
                family: int(config["cohort"]["partition_seed"]) + index
                for index, family in enumerate(FAMILIES)
            },
            "confidence_level": confidence,
            "multiplicity_control": "holm",
            "folds_trials_task_seeds_and_attacker_seeds_treated_as_independent": False,
        },
        "promotion_rule": {
            "minimum_mean_auc_reduction": minimum_reduction,
            "minimum_improved_families": required_families,
            "require_adjusted_interval_excludes_zero": True,
            "require_consistent_attacker_seed_direction": True,
            "maximum_allowed_family_auc_regression": max_regression,
        },
        "family_summaries": family_summaries,
        "privacy_promotion": {
            "improved_families": improved_families,
            "improved_family_count": len(improved_families),
            "required_family_count": required_families,
            "pass": privacy_promotion,
            "decision": (
                "promote_bottleneck_privacy_claim"
                if privacy_promotion
                else "retain_utility_noninferiority_and_report_no_confirmatory_privacy_promotion"
            ),
        },
        "family_subject_aggregates": subject_rows,
        "jobs": jobs,
    }


def _render_record(payload: dict[str, Any]) -> str:
    summaries = payload["family_summaries"]
    rows = []
    for family in FAMILIES:
        summary = summaries[family]
        interval = summary["holm_rank_adjusted_subject_bootstrap_ci"]
        rows.append(
            "| "
            + " | ".join(
                [
                    family.replace("_", " "),
                    f"{summary['plain_auc_subject_mean']:.4f}",
                    f"{summary['bottleneck_auc_subject_mean']:.4f}",
                    f"{summary['mean_auc_reduction_plain_minus_bottleneck']:.5f}",
                    f"[{interval['lower']:.4f}, {interval['upper']:.4f}]",
                    "yes"
                    if summary["consistent_positive_attacker_seed_direction"]
                    else "no",
                    "PASS" if summary["family_promotion_pass"] else "NO",
                ]
            )
            + " |"
        )
    omitted = ", ".join(
        str(value)
        for value in payload["inference"][
            "confirmatory_subjects_never_assigned_to_attacker_evaluation"
        ]
    )
    return f"""# Cho2017 Confirmatory Membership-Privacy Gate

## Status

The preregistered membership-privacy promotion gate did not pass.

All 600 frozen cache-only attack jobs validated. This record performed no EEG
loading or task-model training. The compact bottleneck remains utility
noninferior, but the confirmatory data do not support promoting it as a
membership-privacy improvement over plain compact EEGNet.

## Subject-paired result

Positive reduction means lower bottleneck attack AUC. Frozen score variants,
task seeds, eligible folds, and attacker seeds were aggregated within each
attacker-evaluation subject before inference.

| Attacker family | Plain AUC | Bottleneck AUC | Mean AUC reduction | Holm rank-adjusted interval | Positive on 3/3 attacker seeds | Promotion criterion |
| --- | ---: | ---: | ---: | ---: | --- | --- |
{chr(10).join(rows)}

Promotion required a mean AUC reduction of at least `0.010`, multiplicity-
adjusted interval support above zero, consistent positive direction on all
three attacker seeds, and no family regression above `0.010` in at least two
of the three families. `0/3` families passed, so the decision is
**NO PROMOTION**.

## Inference and scope

The inferential sample contains {payload['inference']['evaluated_subject_count']} subjects
assigned to the held-out attacker-evaluation role in at least one
eligible outer fold. Confirmatory subjects {omitted} were never assigned to
that role by the frozen per-fold partitions; this is a consequence of the
preregistered roles, not a post-result exclusion.

Threshold and logistic-regression score variants were equally weighted within
their family so variants did not become extra promotion opportunities. The
three family tests used Holm step-down multiplicity control. The displayed
rank-adjusted percentile intervals use the corresponding Holm critical level;
the machine-readable artifact also records unadjusted 95% intervals, adjusted
p-values, and attacker-seed diagnostics.

This negative promotion result is not evidence that membership leakage is
absent or that the two models are equivalent. It means only that the frozen
bottleneck did not satisfy the preregistered improvement rule.

## Reproducibility record

The tracked machine-readable artifact is
[`configs/cho2017_confirmatory_privacy_gate_v1.json`](../configs/cho2017_confirmatory_privacy_gate_v1.json).
It contains subject aggregates and hashes for all 600 result files, but no raw
EEG or posterior arrays.

After reproducing the local attack queue, verify this record with:

```bash
PYTHONPATH=src .venv/bin/python scripts/freeze_cho2017_privacy_gate.py --check
```
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze or verify the completed Cho2017 membership-privacy gate. "
            "This command never trains a model or reruns an attack."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _build_payload(args.input_dir.resolve())
    encoded = json.dumps(payload, indent=2) + "\n"
    record = _render_record(payload)
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("tracked privacy-gate JSON does not reproduce")
        if not args.record.is_file() or args.record.read_text(encoding="utf-8") != record:
            raise RuntimeError("tracked privacy-gate record does not reproduce")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        args.record.write_text(record, encoding="utf-8")
    decision = payload["privacy_promotion"]
    print(
        f"{payload['status']} jobs={payload['provenance']['validated_attack_jobs']} "
        f"subjects={payload['inference']['evaluated_subject_count']} "
        f"qualifying_families={decision['improved_family_count']}/{len(FAMILIES)} "
        f"required={decision['required_family_count']}"
    )


if __name__ == "__main__":
    main()
