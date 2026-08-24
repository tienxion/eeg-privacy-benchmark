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
from run_federated_bnci_multiseed import _seed_paths as federated_seed_paths
from run_lee_compact_eegnet_privacy import SEEDS
from run_lee_compact_mlp_attacker_refit import (
    ATTACK_SEEDS,
    DEFAULT_OUTPUT_ROOT as CENTRAL_REFIT_ROOT,
    _result_path as central_result_path,
)


FEDERATED_ROOT = ROOT / "outputs" / "federated_lee_compact_multiseed"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "lee_compact_federated_mlp_attacker_refit"
DEFAULT_POLICY_DELTA = 0.02750416666666658


def _result_path(output_root: Path, task_seed: int, attack_seed: int) -> Path:
    return output_root / f"attackseed{attack_seed}" / f"taskseed{task_seed}.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_summary(results: dict, output_root: Path) -> dict:
    rows = []
    deltas = []
    for attack_seed in ATTACK_SEEDS:
        central_aucs = [
            _read(
                central_result_path(
                    CENTRAL_REFIT_ROOT,
                    "compact_eegnet",
                    task_seed,
                    attack_seed,
                )
            )["attack_auc"]
            for task_seed in SEEDS
        ]
        federated_aucs = [
            results[attack_seed][task_seed]["attack_auc"] for task_seed in SEEDS
        ]
        paired = [
            current - baseline
            for current, baseline in zip(federated_aucs, central_aucs)
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
    disagrees = (
        DEFAULT_POLICY_DELTA > 0.0 and all(delta < 0.0 for delta in deltas)
    ) or (DEFAULT_POLICY_DELTA < 0.0 and all(delta > 0.0 for delta in deltas))
    summary = {
        "metadata_schema_version": 1,
        "dataset_key": "lee2019_mi",
        "protocol": "cross_session",
        "subjects": [1, 2, 3, 4, 5, 6],
        "task_seeds": list(SEEDS),
        "attacker_seeds": list(ATTACK_SEEDS),
        "task_model_retraining": False,
        "attack_seed_policy": "explicit_fixed_seed",
        "default_task_seed_policy_delta": DEFAULT_POLICY_DELTA,
        "attacker_seed_deltas": deltas,
        "mean_delta_across_attacker_seeds": mean(deltas),
        "same_direction_across_refits": same_direction,
        "refit_direction_disagrees_with_default_policy": disagrees,
        "decision": (
            "attacker_seed_protocol_sensitive_no_general_mlp_claim"
            if disagrees or not same_direction
            else "refit_direction_stable_and_default_consistent"
        ),
        "rows": rows,
    }
    json_path = output_root / "lee_compact_federated_mlp_attacker_refit.json"
    markdown_path = output_root / "lee_compact_federated_mlp_attacker_refit.md"
    write_result_json(summary, json_path)
    lines = [
        "# Lee Compact Federated MLP Attacker Refit",
        "",
        "Posterior caches are reused without task-model retraining.",
        "",
        "| Attacker seed | Centralized AUC | Federated AUC | Paired delta | Improving task seeds |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['attack_seed']} | {row['centralized_mean_auc']:.4f} | "
            f"{row['federated_mean_auc']:.4f} | {row['mean_paired_delta']:+.4f} | "
            f"{row['privacy_improving_task_seeds']}/4 |"
        )
    lines.extend(
        [
            "",
            f"- Default task-seed-policy delta: `{DEFAULT_POLICY_DELTA:+.4f}`",
            f"- Fixed-seed deltas: `{','.join(f'{value:+.4f}' for value in deltas)}`",
            f"- Mean fixed-seed delta: `{mean(deltas):+.4f}`",
            f"- Decision: `{summary['decision']}`",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refit MLP attackers for compact centralized/federated Lee models."
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
                print(
                    f"task_seed={task_seed} attack_seed={attack_seed} status=planned"
                )
                continue
            else:
                cache_path = federated_seed_paths(FEDERATED_ROOT, task_seed)[
                    "posterior_cache"
                ]
                attack = run_cached_membership_inference_attack(
                    cache_path,
                    attack_type="mlp",
                    score_type="posterior_probabilities",
                    attack_seed=attack_seed,
                )
                result = attack.to_dict()
                write_result_json(result, output_path)
                status = "complete"
            results[attack_seed][task_seed] = result
            print(
                f"task_seed={task_seed} attack_seed={attack_seed} status={status} "
                f"auc={result['attack_auc']:.6f}"
            )
    if args.dry_run:
        return
    summary = _write_summary(results, args.output_root)
    print(f"decision={summary['decision']}")


if __name__ == "__main__":
    main()
