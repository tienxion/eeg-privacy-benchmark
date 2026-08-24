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

from eeg_privacy_benchmark.privacy import run_cached_membership_inference_attack
from eeg_privacy_benchmark.results import write_result_json
from run_federated_bnci_multiseed import _seed_paths


TASK_SEEDS = (13, 17, 19, 23)
ATTACK_SEEDS = (101, 103, 107)
FEDERATED_ROOT = ROOT / "outputs" / "federated_physionet_multiseed"
CENTRAL_REFIT_ROOT = (
    ROOT
    / "outputs/sweeps/attacker_refit_v1/physionet_motor_imagery/"
    "subject1to46/eegnet/mlp_posterior"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "physionet_federated_mlp_attacker_refit"
COMPARISON_PATH = (
    FEDERATED_ROOT
    / "physionet_motor_imagery_four_seed_centralized_vs_federated.json"
)


def _result_path(output_root: Path, task_seed: int, attack_seed: int) -> Path:
    return output_root / f"attackseed{attack_seed}" / f"taskseed{task_seed}.json"


def _central_result_path(task_seed: int, attack_seed: int) -> Path:
    return CENTRAL_REFIT_ROOT / f"attackseed{attack_seed}" / (
        f"eegnet_physionet_motor_imagery_cross_subject_seed{task_seed}_"
        f"attackseed{attack_seed}.json"
    )


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_summary(results: dict, output_root: Path) -> dict:
    comparison = _read(COMPARISON_PATH)
    default_delta = next(
        row["mean_paired_delta"]
        for row in comparison["summaries"]
        if row["metric"] == "membership_mlp_posterior_auc"
    )
    rows = []
    deltas = []
    for attack_seed in ATTACK_SEEDS:
        central_aucs = [
            _read(_central_result_path(task_seed, attack_seed))["attack_auc"]
            for task_seed in TASK_SEEDS
        ]
        federated_aucs = [
            results[attack_seed][task_seed]["attack_auc"]
            for task_seed in TASK_SEEDS
        ]
        paired = [
            current - reference
            for current, reference in zip(federated_aucs, central_aucs)
        ]
        delta = mean(paired)
        deltas.append(delta)
        rows.append(
            {
                "attack_seed": attack_seed,
                "centralized_mean_auc": mean(central_aucs),
                "centralized_sample_std_auc": stdev(central_aucs),
                "federated_mean_auc": mean(federated_aucs),
                "federated_sample_std_auc": stdev(federated_aucs),
                "mean_paired_delta": delta,
                "privacy_improving_task_seeds": sum(value < 0.0 for value in paired),
            }
        )
    same_direction = all(delta < 0.0 for delta in deltas) or all(
        delta > 0.0 for delta in deltas
    )
    default_sign_consistent = (
        default_delta < 0.0 and all(delta < 0.0 for delta in deltas)
    ) or (default_delta > 0.0 and all(delta > 0.0 for delta in deltas))
    decision = (
        "refit_direction_stable_and_default_consistent"
        if same_direction and default_sign_consistent
        else "attacker_seed_protocol_sensitive_no_general_mlp_claim"
    )
    summary = {
        "metadata_schema_version": 1,
        "dataset_key": "physionet_motor_imagery",
        "protocol": "cross_subject",
        "subjects": list(range(1, 47)),
        "task_seeds": list(TASK_SEEDS),
        "attacker_seeds": list(ATTACK_SEEDS),
        "task_model_retraining": False,
        "attack_seed_policy": "explicit_fixed_seed",
        "default_task_seed_policy_delta": default_delta,
        "attacker_seed_deltas": deltas,
        "mean_delta_across_attacker_seeds": mean(deltas),
        "same_direction_across_refits": same_direction,
        "default_sign_consistent": default_sign_consistent,
        "decision": decision,
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    stem = "physionet_federated_mlp_attacker_refit"
    write_result_json(summary, output_root / f"{stem}.json")
    lines = [
        "# PhysioNet Federated MLP Attacker Refit",
        "",
        "Posterior caches are reused without task-model retraining.",
        "",
        "| Attacker seed | Centralized AUC | Federated AUC | Paired delta | "
        "Improving task seeds |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['attack_seed']} | {row['centralized_mean_auc']:.4f} | "
            f"{row['federated_mean_auc']:.4f} | "
            f"{row['mean_paired_delta']:+.4f} | "
            f"{row['privacy_improving_task_seeds']}/4 |"
        )
    lines.extend(
        [
            "",
            f"- Default task-seed-policy delta: `{default_delta:+.4f}`",
            "- Fixed-seed deltas: "
            f"`{','.join(f'{value:+.4f}' for value in deltas)}`",
            f"- Mean fixed-seed delta: `{mean(deltas):+.4f}`",
            f"- Decision: `{decision}`",
        ]
    )
    (output_root / f"{stem}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refit fixed-seed MLP attackers for PhysioNet federation."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    results = {attack_seed: {} for attack_seed in ATTACK_SEEDS}
    for attack_seed in ATTACK_SEEDS:
        for task_seed in TASK_SEEDS:
            output_path = _result_path(args.output_root, task_seed, attack_seed)
            if output_path.exists() and not args.force:
                result = _read(output_path)
                status = "skipped_complete"
            elif args.dry_run:
                print(f"task_seed={task_seed} attack_seed={attack_seed} status=planned")
                continue
            else:
                cache_path = _seed_paths(FEDERATED_ROOT, task_seed)[
                    "posterior_cache"
                ]
                result = run_cached_membership_inference_attack(
                    cache_path,
                    attack_type="mlp",
                    score_type="posterior_probabilities",
                    attack_seed=attack_seed,
                ).to_dict()
                write_result_json(result, output_path)
                status = "complete"
            results[attack_seed][task_seed] = result
            print(
                f"task_seed={task_seed} attack_seed={attack_seed} status={status} "
                f"auc={result['attack_auc']:.6f}"
            )
    if not args.dry_run:
        summary = _write_summary(results, args.output_root)
        print(f"decision={summary['decision']}")


if __name__ == "__main__":
    main()
