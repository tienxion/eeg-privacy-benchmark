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
from run_lee_compact_eegnet_privacy import SEEDS, _seed_paths


ATTACK_SEEDS = (101, 103, 107)
PRACTICAL_TIE_AUC = 0.01
MODELS = {
    "compact_eegnet": ROOT / "outputs" / "lee_compact_eegnet_privacy",
    "compact_bottleneck_dim6": (
        ROOT / "outputs" / "lee_compact_bottleneck_dim6_privacy"
    ),
    "compact_confidence_penalty_beta0p05": (
        ROOT / "outputs" / "lee_compact_confidence_penalty_beta0p05_privacy"
    ),
}
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "lee_compact_mlp_attacker_refit"


def _result_path(output_root: Path, model: str, task_seed: int, attack_seed: int) -> Path:
    return (
        output_root
        / model
        / f"attackseed{attack_seed}"
        / f"taskseed{task_seed}.json"
    )


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _decision(deltas: list[float]) -> str:
    if all(abs(delta) <= PRACTICAL_TIE_AUC for delta in deltas):
        return "refit_stable_practical_tie"
    if all(delta < 0.0 for delta in deltas):
        return "refit_stable_privacy_improvement"
    if all(delta > 0.0 for delta in deltas):
        return "refit_stable_privacy_regression"
    return "attacker_refit_direction_mixed"


def _write_summary(results: dict, output_root: Path) -> dict:
    rows = []
    baseline = "compact_eegnet"
    for model in MODELS:
        for attack_seed in ATTACK_SEEDS:
            aucs = [
                results[model][attack_seed][task_seed]["attack_auc"]
                for task_seed in SEEDS
            ]
            row = {
                "model": model,
                "attack_seed": attack_seed,
                "mean_attack_auc": mean(aucs),
                "sample_std_attack_auc": stdev(aucs),
                "mean_paired_delta_vs_baseline": 0.0,
                "privacy_improving_task_seeds": 0,
            }
            if model != baseline:
                baseline_aucs = [
                    results[baseline][attack_seed][task_seed]["attack_auc"]
                    for task_seed in SEEDS
                ]
                deltas = [
                    current - reference
                    for current, reference in zip(aucs, baseline_aucs)
                ]
                row["mean_paired_delta_vs_baseline"] = mean(deltas)
                row["privacy_improving_task_seeds"] = sum(
                    delta < 0.0 for delta in deltas
                )
            rows.append(row)

    decisions = {}
    for model in MODELS:
        if model == baseline:
            continue
        model_rows = [row for row in rows if row["model"] == model]
        deltas = [row["mean_paired_delta_vs_baseline"] for row in model_rows]
        baseline_default = mean(
            _read(_seed_paths(MODELS[baseline], seed)["suite"])["membership_auc"][
                "mlp_posterior"
            ]
            for seed in SEEDS
        )
        model_default = mean(
            _read(_seed_paths(MODELS[model], seed)["suite"])["membership_auc"][
                "mlp_posterior"
            ]
            for seed in SEEDS
        )
        default_delta = model_default - baseline_default
        direction_disagrees = (
            default_delta > 0.0 and all(delta < 0.0 for delta in deltas)
        ) or (default_delta < 0.0 and all(delta > 0.0 for delta in deltas))
        decisions[model] = {
            "attacker_seed_deltas": deltas,
            "mean_delta_across_attacker_seeds": mean(deltas),
            "decision": _decision(deltas),
            "default_task_seed_coupled_delta": default_delta,
            "refit_direction_disagrees_with_default_policy": direction_disagrees,
            "cross_policy_decision": (
                "attacker_seed_protocol_sensitive_no_general_mlp_claim"
                if direction_disagrees
                else "refit_result_consistent_with_default_policy"
            ),
        }

    summary = {
        "metadata_schema_version": 1,
        "dataset_key": "lee2019_mi",
        "protocol": "cross_session",
        "subjects": [1, 2, 3, 4, 5, 6],
        "task_seeds": list(SEEDS),
        "attacker_seeds": list(ATTACK_SEEDS),
        "attack_type": "mlp",
        "score_type": "posterior_probabilities",
        "task_model_retraining": False,
        "practical_tie_auc": PRACTICAL_TIE_AUC,
        "rows": rows,
        "decisions": decisions,
    }
    json_path = output_root / "lee_compact_mlp_attacker_refit.json"
    markdown_path = output_root / "lee_compact_mlp_attacker_refit.md"
    write_result_json(summary, json_path)
    lines = [
        "# Lee Compact MLP Attacker Refit",
        "",
        "Task-model posteriors are reused without retraining. Each row averages task seeds `13,17,19,23`.",
        "",
        "| Model | Attacker seed | Mean AUC (SD) | Delta vs baseline | Improving task seeds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['attack_seed']} | "
            f"{row['mean_attack_auc']:.4f} ({row['sample_std_attack_auc']:.4f}) | "
            f"{row['mean_paired_delta_vs_baseline']:+.4f} | "
            f"{row['privacy_improving_task_seeds']}/4 |"
        )
    lines.extend(["", "Decisions:"])
    for model, decision in decisions.items():
        deltas = ", ".join(
            f"{delta:+.4f}" for delta in decision["attacker_seed_deltas"]
        )
        lines.append(
            f"- {model}: `{decision['decision']}`; attacker-seed deltas `{deltas}`; "
            f"mean `{decision['mean_delta_across_attacker_seeds']:+.4f}`."
        )
        lines.append(
            f"  Default task-seed-coupled delta: "
            f"`{decision['default_task_seed_coupled_delta']:+.4f}`; "
            f"cross-policy read: `{decision['cross_policy_decision']}`."
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refit Lee compact MLP attackers from posterior caches."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results = {model: {attack_seed: {} for attack_seed in ATTACK_SEEDS} for model in MODELS}
    for model, source_root in MODELS.items():
        for attack_seed in ATTACK_SEEDS:
            for task_seed in SEEDS:
                output_path = _result_path(
                    args.output_root,
                    model,
                    task_seed,
                    attack_seed,
                )
                if output_path.exists() and not args.force:
                    result = _read(output_path)
                    print(
                        f"model={model} task_seed={task_seed} "
                        f"attack_seed={attack_seed} status=skipped_complete"
                    )
                elif args.dry_run:
                    print(
                        f"model={model} task_seed={task_seed} "
                        f"attack_seed={attack_seed} status=planned"
                    )
                    continue
                else:
                    cache_path = _seed_paths(source_root, task_seed)["posterior_cache"]
                    attack = run_cached_membership_inference_attack(
                        cache_path,
                        attack_type="mlp",
                        score_type="posterior_probabilities",
                        attack_seed=attack_seed,
                    )
                    result = attack.to_dict()
                    write_result_json(result, output_path)
                    print(
                        f"model={model} task_seed={task_seed} "
                        f"attack_seed={attack_seed} status=complete "
                        f"auc={result['attack_auc']:.6f}"
                    )
                results[model][attack_seed][task_seed] = result

    if args.dry_run:
        return
    summary = _write_summary(results, args.output_root)
    for model, decision in summary["decisions"].items():
        print(f"model={model} decision={decision['decision']}")


if __name__ == "__main__":
    main()
