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
    ATTACK_SEEDS,
    DEFAULT_OUTPUT_ROOT as BASELINE_REFIT_ROOT,
    _result_path as baseline_result_path,
)


DEFENSE_ROOT = (
    ROOT
    / "outputs"
    / "federated_lee_compact_confidence_penalty_beta0p05_multiseed"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "outputs" / "lee_compact_federated_confidence_penalty_mlp_attacker_refit"
)


def _result_path(output_root: Path, task_seed: int, attack_seed: int) -> Path:
    return output_root / f"attackseed{attack_seed}" / f"taskseed{task_seed}.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_summary(results: dict, output_root: Path) -> dict:
    comparison = _read(
        ROOT
        / "outputs/lee_compact_federated_confidence_penalty_comparison/"
        "lee_compact_federated_confidence_penalty_comparison.json"
    )
    default_delta = next(
        row["mean_paired_delta"]
        for row in comparison["aggregates"]
        if row["metric"] == "membership_mlp_posterior_auc"
    )
    rows = []
    deltas = []
    for attack_seed in ATTACK_SEEDS:
        baseline_aucs = [
            _read(baseline_result_path(BASELINE_REFIT_ROOT, seed, attack_seed))[
                "attack_auc"
            ]
            for seed in SEEDS
        ]
        defense_aucs = [results[attack_seed][seed]["attack_auc"] for seed in SEEDS]
        paired = [
            current - reference
            for current, reference in zip(defense_aucs, baseline_aucs)
        ]
        delta = mean(paired)
        deltas.append(delta)
        rows.append(
            {
                "attack_seed": attack_seed,
                "baseline_mean_auc": mean(baseline_aucs),
                "baseline_sample_std_auc": stdev(baseline_aucs),
                "confidence_penalty_mean_auc": mean(defense_aucs),
                "confidence_penalty_sample_std_auc": stdev(defense_aucs),
                "mean_paired_delta": delta,
                "privacy_improving_task_seeds": sum(value < 0.0 for value in paired),
            }
        )
    same_direction = all(delta < 0.0 for delta in deltas) or all(
        delta > 0.0 for delta in deltas
    )
    default_sign_consistent = (
        default_delta < 0.0 and all(d < 0.0 for d in deltas)
    ) or (default_delta > 0.0 and all(d > 0.0 for d in deltas))
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
        "baseline_model": "compact_federated_eegnet",
        "defense_model": "compact_federated_confidence_penalty_eegnet",
        "confidence_penalty_beta": 0.05,
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
    stem = "lee_compact_federated_confidence_penalty_mlp_attacker_refit"
    write_result_json(summary, output_root / f"{stem}.json")
    lines = [
        "# Lee Compact Federated Confidence-Penalty MLP Attacker Refit",
        "",
        "Posterior caches are reused without task-model retraining.",
        "",
        "| Attacker seed | Baseline AUC | Confidence penalty AUC | Paired "
        "delta | Improving task seeds |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['attack_seed']} | {row['baseline_mean_auc']:.4f} | "
            f"{row['confidence_penalty_mean_auc']:.4f} | "
            f"{row['mean_paired_delta']:+.4f} | "
            f"{row['privacy_improving_task_seeds']}/4 |"
        )
    lines.extend(
        [
            "",
            f"- Default task-seed-policy delta: `{default_delta:+.4f}`",
            f"- Fixed-seed deltas: `{','.join(f'{value:+.4f}' for value in deltas)}`",
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
        description="Refit fixed-seed MLP attackers for federated confidence penalty."
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
                cache_path = _seed_paths(DEFENSE_ROOT, task_seed)["posterior_cache"]
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
