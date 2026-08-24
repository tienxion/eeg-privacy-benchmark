from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "outputs" / "reports"
PLAN_CSV = REPORT_ROOT / "attacker_refit_plan_v1.csv"
BASELINE_MODEL = "eegnet"
MODEL_ORDER = {
    "eegnet": 0,
    "bottleneck_eegnet": 1,
}


@dataclass(frozen=True)
class ResultRow:
    priority: str
    dataset: str
    model: str
    attack_seed: int
    complete: bool
    summary_path: str
    task_seeds: str
    task_balanced_accuracy_mean: float | None
    attack_auc_mean: float | None
    attack_auc_std: float | None
    delta_auc_vs_eegnet_same_attack_seed: float | None
    delta_task_vs_eegnet_same_attack_seed: float | None
    decision: str


@dataclass(frozen=True)
class AggregateRow:
    priority: str
    dataset: str
    model: str
    completed_attack_seeds: str
    completed_refits: int
    mean_task_balanced_accuracy: float | None
    mean_attack_auc: float | None
    std_attack_auc_across_refits: float | None
    mean_delta_auc_vs_eegnet: float | None
    std_delta_auc_vs_eegnet: float | None
    privacy_improving_refits: int | None
    decision: str


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _summary_path(output_dir: Path) -> Path | None:
    matches = sorted(output_dir.glob("*_summary.json"))
    return matches[0] if matches else None


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _signed(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.4f}"


def _load_result_rows() -> list[ResultRow]:
    if not PLAN_CSV.exists():
        raise FileNotFoundError(f"Missing plan CSV: {PLAN_CSV}")

    raw_rows: list[dict[str, str]]
    with PLAN_CSV.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))

    baseline_by_key: dict[tuple[str, int], tuple[float | None, float | None]] = {}
    partial_rows: list[ResultRow] = []
    for raw in raw_rows:
        output_dir = ROOT / raw["output_dir"]
        summary = _summary_path(output_dir)
        payload = _load_json(summary) if summary else None
        complete = payload is not None
        task = (
            float(payload["task_balanced_accuracy_mean"])
            if payload and payload.get("task_balanced_accuracy_mean") is not None
            else None
        )
        auc = (
            float(payload["attack_auc_mean"])
            if payload and payload.get("attack_auc_mean") is not None
            else None
        )
        auc_std = (
            float(payload["attack_auc_std"])
            if payload and payload.get("attack_auc_std") is not None
            else None
        )
        attack_seed = int(raw["attack_seed"])
        if raw["model"] == BASELINE_MODEL and complete:
            baseline_by_key[(raw["dataset"], attack_seed)] = (task, auc)

        partial_rows.append(
            ResultRow(
                priority=raw["priority"],
                dataset=raw["dataset"],
                model=raw["model"],
                attack_seed=attack_seed,
                complete=complete,
                summary_path=str(summary.relative_to(ROOT)) if summary else "",
                task_seeds=raw["task_seeds"],
                task_balanced_accuracy_mean=task,
                attack_auc_mean=auc,
                attack_auc_std=auc_std,
                delta_auc_vs_eegnet_same_attack_seed=None,
                delta_task_vs_eegnet_same_attack_seed=None,
                decision="pending",
            )
        )

    rows: list[ResultRow] = []
    for row in partial_rows:
        base_task, base_auc = baseline_by_key.get((row.dataset, row.attack_seed), (None, None))
        delta_auc = (
            row.attack_auc_mean - base_auc
            if row.attack_auc_mean is not None and base_auc is not None
            else None
        )
        delta_task = (
            row.task_balanced_accuracy_mean - base_task
            if row.task_balanced_accuracy_mean is not None and base_task is not None
            else None
        )
        updated = ResultRow(
            **{
                **asdict(row),
                "delta_auc_vs_eegnet_same_attack_seed": delta_auc,
                "delta_task_vs_eegnet_same_attack_seed": delta_task,
                "decision": "",
            }
        )
        rows.append(ResultRow(**{**asdict(updated), "decision": _decision(updated)}))
    return rows


def _decision(row: ResultRow) -> str:
    if not row.complete:
        return "pending"
    if row.model == BASELINE_MODEL:
        return "baseline"
    if row.delta_auc_vs_eegnet_same_attack_seed is None:
        return "missing same-attack-seed baseline"
    if row.delta_auc_vs_eegnet_same_attack_seed <= -0.010 and (
        row.delta_task_vs_eegnet_same_attack_seed is None
        or row.delta_task_vs_eegnet_same_attack_seed >= -0.030
    ):
        return "refit privacy-favorable"
    if row.delta_auc_vs_eegnet_same_attack_seed <= 0.005:
        return "refit practical tie"
    return "refit privacy regression"


def _aggregate_rows(rows: list[ResultRow]) -> list[AggregateRow]:
    aggregates: list[AggregateRow] = []
    keys = sorted(
        {(row.priority, row.dataset, row.model) for row in rows},
        key=lambda item: (item[0], item[1], MODEL_ORDER.get(item[2], 99), item[2]),
    )
    for priority, dataset, model in keys:
        completed = [
            row
            for row in rows
            if row.priority == priority
            and row.dataset == dataset
            and row.model == model
            and row.complete
        ]
        auc_values = [
            row.attack_auc_mean
            for row in completed
            if row.attack_auc_mean is not None
        ]
        task_values = [
            row.task_balanced_accuracy_mean
            for row in completed
            if row.task_balanced_accuracy_mean is not None
        ]
        delta_values = [
            row.delta_auc_vs_eegnet_same_attack_seed
            for row in completed
            if row.delta_auc_vs_eegnet_same_attack_seed is not None
        ]
        improving = sum(delta < 0.0 for delta in delta_values) if delta_values else None
        mean_delta = mean(delta_values) if delta_values else None
        aggregates.append(
            AggregateRow(
                priority=priority,
                dataset=dataset,
                model=model,
                completed_attack_seeds=",".join(str(row.attack_seed) for row in completed),
                completed_refits=len(completed),
                mean_task_balanced_accuracy=mean(task_values) if task_values else None,
                mean_attack_auc=mean(auc_values) if auc_values else None,
                std_attack_auc_across_refits=stdev(auc_values) if len(auc_values) > 1 else None,
                mean_delta_auc_vs_eegnet=mean_delta,
                std_delta_auc_vs_eegnet=stdev(delta_values) if len(delta_values) > 1 else None,
                privacy_improving_refits=improving,
                decision=_aggregate_decision(
                    model=model,
                    completed_refits=len(completed),
                    mean_delta_auc=mean_delta,
                    improving=improving,
                ),
            )
        )
    return aggregates


def _aggregate_decision(
    *,
    model: str,
    completed_refits: int,
    mean_delta_auc: float | None,
    improving: int | None,
) -> str:
    if completed_refits == 0:
        return "pending"
    if model == BASELINE_MODEL:
        return "baseline"
    if mean_delta_auc is None or improving is None:
        return "missing baseline overlap"
    if completed_refits < 3:
        return "partial refit evidence"
    if mean_delta_auc <= -0.010 and improving == completed_refits:
        return "refit-stable privacy improvement"
    if abs(mean_delta_auc) <= 0.005:
        return "refit-stable practical tie"
    if mean_delta_auc > 0.005:
        return "refit-stable privacy regression"
    return "mixed refit evidence"


def _write_rows_csv(rows: list[ResultRow]) -> Path:
    out_path = REPORT_ROOT / "attacker_refit_results_rows_v1.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.priority, item.dataset, item.attack_seed, MODEL_ORDER.get(item.model, 99), item.model)):
            writer.writerow(asdict(row))
    return out_path


def _write_aggregate_csv(rows: list[AggregateRow]) -> Path:
    out_path = REPORT_ROOT / "attacker_refit_results_v1.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return out_path


def _write_markdown(
    rows: list[ResultRow],
    aggregates: list[AggregateRow],
    aggregate_csv: Path,
    rows_csv: Path,
) -> Path:
    out_path = REPORT_ROOT / "attacker_refit_results_v1.md"
    complete = sum(1 for row in rows if row.complete)
    p1_complete = sum(1 for row in rows if row.priority == "P1" and row.complete)
    p1_total = sum(1 for row in rows if row.priority == "P1")
    p2_complete = sum(1 for row in rows if row.priority == "P2" and row.complete)
    p2_total = sum(1 for row in rows if row.priority == "P2")

    lines = [
        "# Attacker Refit Results",
        "",
        "This report summarizes completed learned membership attacker-refit jobs. Lower membership AUC is better for privacy; deltas compare each model to plain EEGNet under the same dataset and attacker seed.",
        "",
        "Companion files:",
        f"- [{aggregate_csv.relative_to(ROOT)}]({aggregate_csv.relative_to(ROOT)})",
        f"- [{rows_csv.relative_to(ROOT)}]({rows_csv.relative_to(ROOT)})",
        "",
        "## Completion",
        "",
        f"- Overall: `{complete}/{len(rows)}` refit commands complete.",
        f"- P1: `{p1_complete}/{p1_total}` refit commands complete.",
        f"- P2: `{p2_complete}/{p2_total}` refit commands complete.",
        "",
        "## Aggregate Read",
        "",
        "| Priority | Dataset | Model | Attack seeds | Mean AUC | Mean delta AUC | Improving refits | Decision |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in aggregates:
        improving = "" if row.privacy_improving_refits is None else str(row.privacy_improving_refits)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.priority}`",
                    f"`{row.dataset}`",
                    f"`{row.model}`",
                    f"`{row.completed_attack_seeds}`",
                    _fmt(row.mean_attack_auc),
                    _signed(row.mean_delta_auc_vs_eegnet),
                    improving,
                    f"`{row.decision}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Completed Refit Rows",
            "",
            "| Priority | Dataset | Model | Attack seed | Task BA | AUC | Delta AUC | Decision |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    completed_rows = [row for row in rows if row.complete]
    for row in sorted(completed_rows, key=lambda item: (item.priority, item.dataset, item.attack_seed, MODEL_ORDER.get(item.model, 99), item.model)):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.priority}`",
                    f"`{row.dataset}`",
                    f"`{row.model}`",
                    str(row.attack_seed),
                    _fmt(row.task_balanced_accuracy_mean),
                    _fmt(row.attack_auc_mean),
                    _signed(row.delta_auc_vs_eegnet_same_attack_seed),
                    f"`{row.decision}`",
                ]
            )
            + " |"
        )
    if not completed_rows:
        lines.append("|  |  |  |  |  |  |  | `pending` |")

    pending_rows = [row for row in rows if not row.complete]
    lines.extend(["", "## Pending Rows", ""])
    if pending_rows:
        for row in pending_rows[:20]:
            lines.append(
                f"- `{row.priority}` `{row.dataset}` `{row.model}` attack seed `{row.attack_seed}`."
            )
        if len(pending_rows) > 20:
            lines.append(f"- ...and `{len(pending_rows) - 20}` more pending rows.")
    else:
        lines.append("- None.")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = _load_result_rows()
    aggregates = _aggregate_rows(rows)
    rows_csv = _write_rows_csv(rows)
    aggregate_csv = _write_aggregate_csv(aggregates)
    markdown_path = _write_markdown(rows, aggregates, aggregate_csv, rows_csv)
    print(f"Wrote {rows_csv.relative_to(ROOT)}")
    print(f"Wrote {aggregate_csv.relative_to(ROOT)}")
    print(f"Wrote {markdown_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
