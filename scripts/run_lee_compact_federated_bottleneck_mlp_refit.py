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
from run_lee_compact_eegnet_privacy import SEEDS
from run_lee_compact_federated_mlp_refit import (
    DEFAULT_OUTPUT_ROOT as FEDERATED_BASELINE_REFIT_ROOT,
    _result_path as federated_baseline_result_path,
)
from run_lee_compact_mlp_attacker_refit import (
    ATTACK_SEEDS,
    DEFAULT_OUTPUT_ROOT as CENTRAL_REFIT_ROOT,
    _result_path as central_result_path,
)


FEDERATED_BOTTLENECK_ROOT = (
    ROOT / "outputs" / "federated_lee_compact_bottleneck_dim6_multiseed"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "outputs" / "lee_compact_federated_bottleneck_mlp_attacker_refit"
)


def _result_path(output_root: Path, task_seed: int, attack_seed: int) -> Path:
    return output_root / f"attackseed{attack_seed}" / f"taskseed{task_seed}.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _central_auc(model: str, task_seed: int, attack_seed: int) -> float:
    return _read(
        central_result_path(
            CENTRAL_REFIT_ROOT,
            model,
            task_seed,
            attack_seed,
        )
    )["attack_auc"]


def _write_summary(results: dict, output_root: Path) -> dict:
    comparison = _read(
        ROOT
        / "outputs/lee_compact_federated_bottleneck_comparison/"
        "lee_compact_federated_bottleneck_comparison.json"
    )
    default_delta = next(
        row["federated_bottleneck_mean_paired_effect"]
        for row in comparison["aggregates"]
        if row["metric"] == "membership_mlp_posterior_auc"
    )
    rows = []
    federated_deltas = []
    interactions = []
    for attack_seed in ATTACK_SEEDS:
        central_baseline = [
            _central_auc("compact_eegnet", seed, attack_seed) for seed in SEEDS
        ]
        central_bottleneck = [
            _central_auc("compact_bottleneck_dim6", seed, attack_seed)
            for seed in SEEDS
        ]
        fed_baseline = [
            _read(
                federated_baseline_result_path(
                    FEDERATED_BASELINE_REFIT_ROOT,
                    seed,
                    attack_seed,
                )
            )["attack_auc"]
            for seed in SEEDS
        ]
        fed_bottleneck = [results[attack_seed][seed]["attack_auc"] for seed in SEEDS]
        central_effect = [
            current - reference
            for current, reference in zip(central_bottleneck, central_baseline)
        ]
        fed_effect = [
            current - reference
            for current, reference in zip(fed_bottleneck, fed_baseline)
        ]
        interaction = [
            fed_value - central_value
            for fed_value, central_value in zip(fed_effect, central_effect)
        ]
        federated_delta = mean(fed_effect)
        federated_deltas.append(federated_delta)
        interactions.append(mean(interaction))
        rows.append(
            {
                "attack_seed": attack_seed,
                "federated_baseline_mean_auc": mean(fed_baseline),
                "federated_baseline_sample_std_auc": stdev(fed_baseline),
                "federated_bottleneck_mean_auc": mean(fed_bottleneck),
                "federated_bottleneck_sample_std_auc": stdev(fed_bottleneck),
                "centralized_bottleneck_mean_paired_effect": mean(central_effect),
                "federated_bottleneck_mean_paired_effect": federated_delta,
                "mean_federation_interaction": mean(interaction),
                "federated_privacy_improving_task_seeds": sum(
                    value < 0.0 for value in fed_effect
                ),
            }
        )
    same_direction = all(delta < 0.0 for delta in federated_deltas) or all(
        delta > 0.0 for delta in federated_deltas
    )
    default_sign_consistent = (
        default_delta < 0.0 and all(delta < 0.0 for delta in federated_deltas)
    ) or (default_delta > 0.0 and all(delta > 0.0 for delta in federated_deltas))
    decision = (
        "refit_direction_stable_and_default_consistent"
        if same_direction and default_sign_consistent
        else "attacker_seed_protocol_sensitive_no_general_mlp_claim"
    )
    summary = {
        "metadata_schema_version": 1,
        "dataset_key": "lee2019_mi",
        "protocol": "cross_session",
        "subjects": [1, 2, 3, 4, 5, 6],
        "task_seeds": list(SEEDS),
        "attacker_seeds": list(ATTACK_SEEDS),
        "bottleneck_dim": 6,
        "task_model_retraining": False,
        "attack_seed_policy": "explicit_fixed_seed",
        "default_task_seed_policy_delta": default_delta,
        "attacker_seed_federated_deltas": federated_deltas,
        "attacker_seed_interactions": interactions,
        "mean_federated_delta_across_attacker_seeds": mean(federated_deltas),
        "mean_interaction_across_attacker_seeds": mean(interactions),
        "same_direction_across_refits": same_direction,
        "default_sign_consistent": default_sign_consistent,
        "decision": decision,
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    stem = "lee_compact_federated_bottleneck_mlp_attacker_refit"
    write_result_json(summary, output_root / f"{stem}.json")
    lines = [
        "# Lee Compact Federated Bottleneck MLP Attacker Refit",
        "",
        "Posterior caches are reused without task-model retraining.",
        "",
        "| Attacker seed | Fed baseline | Fed bottleneck | Fed effect | Central "
        "effect | Interaction | Improving seeds |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['attack_seed']} | "
            f"{row['federated_baseline_mean_auc']:.4f} | "
            f"{row['federated_bottleneck_mean_auc']:.4f} | "
            f"{row['federated_bottleneck_mean_paired_effect']:+.4f} | "
            f"{row['centralized_bottleneck_mean_paired_effect']:+.4f} | "
            f"{row['mean_federation_interaction']:+.4f} | "
            f"{row['federated_privacy_improving_task_seeds']}/4 |"
        )
    lines.extend(
        [
            "",
            f"- Default task-seed-policy delta: `{default_delta:+.4f}`",
            "- Fixed-seed federated deltas: "
            f"`{','.join(f'{value:+.4f}' for value in federated_deltas)}`",
            "- Mean fixed-seed federated delta: "
            f"`{mean(federated_deltas):+.4f}`",
            "- Mean fixed-seed interaction: "
            f"`{mean(interactions):+.4f}`",
            f"- Decision: `{decision}`",
        ]
    )
    (output_root / f"{stem}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refit fixed-seed MLP attackers for federated bottleneck."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    results = {attack_seed: {} for attack_seed in ATTACK_SEEDS}
    for attack_seed in ATTACK_SEEDS:
        for task_seed in SEEDS:
            output_path = _result_path(args.output_root, task_seed, attack_seed)
            if output_path.exists() and not args.force:
                result = _read(output_path)
                status = "skipped_complete"
            elif args.dry_run:
                print(f"task_seed={task_seed} attack_seed={attack_seed} status=planned")
                continue
            else:
                cache_path = _seed_paths(FEDERATED_BOTTLENECK_ROOT, task_seed)[
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
