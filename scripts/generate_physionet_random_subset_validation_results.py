from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "outputs" / "reports"
DEFAULT_PLAN_CSV = REPORT_ROOT / "physionet_random_subset_validation_plan_v1.csv"


@dataclass(frozen=True)
class ResultRow:
    tier: str
    subset_id: str
    model: str
    attack_type: str
    score_type: str
    status: str
    task_balanced_accuracy_mean: float | None
    attack_auc_mean: float | None
    delta_auc_vs_eegnet: float | None
    delta_task_vs_eegnet: float | None
    summary_path: str | None


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _summary_path(output_dir: Path) -> Path | None:
    matches = sorted(output_dir.glob("*_summary.json"))
    if not matches:
        return None
    return matches[0]


def _read_plan(plan_csv: Path) -> list[dict[str, str]]:
    with plan_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _build_rows(plan_rows: list[dict[str, str]]) -> list[ResultRow]:
    raw: list[dict] = []
    for plan_row in plan_rows:
        output_dir = ROOT / plan_row["output_dir"]
        summary_path = _summary_path(output_dir)
        payload = _load_json(summary_path) if summary_path is not None else None
        raw.append(
            {
                "tier": plan_row["tier"],
                "subset_id": plan_row["subset_id"],
                "model": plan_row["model"],
                "attack_type": plan_row["attack_type"],
                "score_type": plan_row["score_type"],
                "status": "complete" if payload else "missing",
                "task": None if payload is None else payload.get("task_balanced_accuracy_mean"),
                "auc": None if payload is None else payload.get("attack_auc_mean"),
                "summary_path": None
                if summary_path is None
                else str(summary_path.relative_to(ROOT)),
            }
        )

    baselines: dict[tuple[str, str, str, str], dict] = {}
    for row in raw:
        if row["model"] == "eegnet" and row["status"] == "complete":
            baselines[(row["tier"], row["subset_id"], row["attack_type"], row["score_type"])] = row

    result_rows: list[ResultRow] = []
    for row in raw:
        baseline = baselines.get((row["tier"], row["subset_id"], row["attack_type"], row["score_type"]))
        delta_auc = None
        delta_task = None
        if baseline is not None and row["auc"] is not None and baseline["auc"] is not None:
            delta_auc = float(row["auc"]) - float(baseline["auc"])
        if baseline is not None and row["task"] is not None and baseline["task"] is not None:
            delta_task = float(row["task"]) - float(baseline["task"])
        result_rows.append(
            ResultRow(
                tier=row["tier"],
                subset_id=row["subset_id"],
                model=row["model"],
                attack_type=row["attack_type"],
                score_type=row["score_type"],
                status=row["status"],
                task_balanced_accuracy_mean=row["task"],
                attack_auc_mean=row["auc"],
                delta_auc_vs_eegnet=delta_auc,
                delta_task_vs_eegnet=delta_task,
                summary_path=row["summary_path"],
            )
        )

    return result_rows


def _write_csv(rows: list[ResultRow]) -> Path:
    out_path = REPORT_ROOT / "physionet_random_subset_validation_results_v1.csv"
    fieldnames = list(asdict(rows[0]).keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return out_path


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _tier_completion(rows: list[ResultRow]) -> dict[str, tuple[int, int]]:
    tiers = sorted({row.tier for row in rows})
    return {
        tier: (
            sum(1 for row in rows if row.tier == tier and row.status == "complete"),
            sum(1 for row in rows if row.tier == tier),
        )
        for tier in tiers
    }


def _tier_a_mixup_read(rows: list[ResultRow]) -> list[str]:
    mixup_rows = [
        row
        for row in rows
        if row.tier == "A"
        and row.model == "mixup_eegnet"
        and row.status == "complete"
        and row.delta_auc_vs_eegnet is not None
    ]
    if not mixup_rows:
        return [
            "Tier A mixup read: no completed random-subset mixup rows yet.",
            "Current decision: unresolved; run Tier A before changing PhysioNet claims.",
        ]

    deltas = [row.delta_auc_vs_eegnet for row in mixup_rows if row.delta_auc_vs_eegnet is not None]
    task_deltas = [
        row.delta_task_vs_eegnet
        for row in mixup_rows
        if row.delta_task_vs_eegnet is not None
    ]
    improving = sum(1 for delta in deltas if delta < 0.0)
    improvement_rate = improving / len(deltas)
    mean_delta = mean(deltas)
    mean_task_delta = mean(task_deltas) if task_deltas else None

    if abs(mean_delta) <= 0.005 or improvement_rate < 0.75:
        decision = "practical tie or unresolved"
    elif mean_delta <= -0.01 and improvement_rate >= 0.75 and (
        mean_task_delta is None or mean_task_delta >= -0.03
    ):
        decision = "candidate protocol-dependent mixup win"
    else:
        decision = "directional but not promotable"

    return [
        f"Tier A mixup completed subsets: `{len(deltas)}`.",
        f"Mean delta AUC versus EEGNet: `{mean_delta:+.4f}`; privacy-improving subsets: `{improving}/{len(deltas)}`.",
        f"Mean delta task balanced accuracy: `{_format_float(mean_task_delta)}`.",
        f"Current decision: `{decision}`.",
    ]


def _threshold_read(rows: list[ResultRow], tier: str, score_label: str) -> list[str]:
    aggregates = _aggregate_tier(rows, tier)
    if not aggregates:
        return [
            f"Tier {tier} threshold read: no completed {score_label} threshold defense rows yet.",
        ]

    lines = [f"Tier {tier} {score_label} threshold aggregate:"]
    stable_candidates: list[str] = []
    for aggregate in aggregates:
        completed = int(aggregate["completed_subsets"])
        wins = int(aggregate["privacy_improving_subsets"])
        mean_delta_auc = float(aggregate["mean_delta_auc"])
        mean_delta_task = float(aggregate["mean_delta_task"])
        lines.append(
            f"`{aggregate['model']}` mean delta AUC `{mean_delta_auc:+.4f}`, "
            f"privacy-improving subsets `{wins}/{completed}`, "
            f"mean delta task BA `{mean_delta_task:+.4f}`."
        )
        if (
            completed >= 4
            and wins / completed >= 0.75
            and mean_delta_auc <= -0.01
            and mean_delta_task >= -0.03
        ):
            stable_candidates.append(str(aggregate["model"]))

    if stable_candidates:
        joined = ", ".join(f"`{model}`" for model in stable_candidates)
        lines.append(
            f"Current Tier {tier} threshold-family decision: candidate threshold-score-specific defense: {joined}."
        )
    else:
        lines.append(
            f"Current Tier {tier} threshold-family decision: no stable promotable default; keep the result attack-family-specific."
        )
    return lines


def _tier_b_threshold_read(rows: list[ResultRow]) -> list[str]:
    return _threshold_read(rows, "B", "max-probability")


def _tier_c_threshold_read(rows: list[ResultRow]) -> list[str]:
    lines = _threshold_read(rows, "C", "label-known-log-probability")
    if lines and "candidate threshold-score-specific defense" in lines[-1]:
        lines.append(
            "Interpretation: this supports a CSP-LDA label-known threshold anchor, not a learned-attack PhysioNet defense default."
        )
    return lines


def _aggregate_tier(rows: list[ResultRow], tier: str) -> list[dict[str, float | int | str]]:
    aggregates: list[dict[str, float | int | str]] = []
    models = sorted(
        {
            row.model
            for row in rows
            if row.tier == tier and row.model != "eegnet" and row.status == "complete"
        }
    )
    for model in models:
        model_rows = [
            row
            for row in rows
            if row.tier == tier
            and row.model == model
            and row.status == "complete"
            and row.delta_auc_vs_eegnet is not None
        ]
        if not model_rows:
            continue
        auc_deltas = [
            row.delta_auc_vs_eegnet
            for row in model_rows
            if row.delta_auc_vs_eegnet is not None
        ]
        task_deltas = [
            row.delta_task_vs_eegnet
            for row in model_rows
            if row.delta_task_vs_eegnet is not None
        ]
        aggregates.append(
            {
                "model": model,
                "completed_subsets": len(model_rows),
                "mean_delta_auc": mean(auc_deltas),
                "privacy_improving_subsets": sum(delta < 0.0 for delta in auc_deltas),
                "mean_delta_task": mean(task_deltas) if task_deltas else 0.0,
            }
        )
    return aggregates


def _write_markdown(rows: list[ResultRow], csv_path: Path) -> Path:
    out_path = REPORT_ROOT / "physionet_random_subset_validation_results_v1.md"
    completion = _tier_completion(rows)
    complete_rows = [row for row in rows if row.status == "complete"]

    lines = [
        "# PhysioNet Random Subject-Subset Validation Results",
        "",
        "This report summarizes completed commands from the deterministic PhysioNet random-subset validation plan. Lower membership-inference AUC is better for privacy. Deltas are computed versus same-subset plain EEGNet for the same attack tier.",
        "",
        f"Companion table: [{csv_path.relative_to(ROOT)}]({csv_path.relative_to(ROOT)})",
        "",
        "## Completion",
        "",
    ]
    for tier, (complete, total) in completion.items():
        lines.append(f"- Tier {tier}: `{complete}/{total}` commands complete.")

    lines.extend(["", "## Current Decision Read", ""])
    lines.extend(f"- {line}" for line in _tier_a_mixup_read(rows))
    lines.extend(f"- {line}" for line in _tier_b_threshold_read(rows))
    lines.extend(f"- {line}" for line in _tier_c_threshold_read(rows))

    for tier in sorted({row.tier for row in rows}):
        tier_aggregates = _aggregate_tier(rows, tier)
        lines.extend(
            [
                "",
                f"## Tier {tier} Aggregate Model Read",
                "",
                "| Model | Completed subsets | Mean delta AUC | Privacy-improving subsets | Mean delta task BA |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        if tier_aggregates:
            for aggregate in tier_aggregates:
                lines.append(
                    f"| `{aggregate['model']}` | "
                    f"{aggregate['completed_subsets']} | "
                    f"{aggregate['mean_delta_auc']:+.4f} | "
                    f"{aggregate['privacy_improving_subsets']}/{aggregate['completed_subsets']} | "
                    f"{aggregate['mean_delta_task']:+.4f} |"
                )
        else:
            lines.append("|  |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Completed Rows",
            "",
            "| Tier | Subset | Model | Attack | Task BA | Attack AUC | Delta AUC | Delta Task |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in complete_rows:
        lines.append(
            f"| `{row.tier}` | `{row.subset_id}` | `{row.model}` | "
            f"`{row.attack_type}::{row.score_type}` | "
            f"{_format_float(row.task_balanced_accuracy_mean)} | "
            f"{_format_float(row.attack_auc_mean)} | "
            f"{_format_float(row.delta_auc_vs_eegnet)} | "
            f"{_format_float(row.delta_task_vs_eegnet)} |"
        )
    if not complete_rows:
        lines.append("|  |  | No completed validation sweeps yet. |  |  |  |  |  |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize completed PhysioNet random-subset validation sweeps."
    )
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    plan_rows = _read_plan(args.plan_csv)
    rows = _build_rows(plan_rows)
    csv_path = _write_csv(rows)
    markdown_path = _write_markdown(rows, csv_path)
    complete = sum(1 for row in rows if row.status == "complete")
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {markdown_path.relative_to(ROOT)}")
    print(f"Completed validation commands: {complete}/{len(rows)}")


if __name__ == "__main__":
    main()
