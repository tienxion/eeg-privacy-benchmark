from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "outputs" / "reports"
PLAN_CSV = REPORT_ROOT / "physionet_defense_grid_plan_v1.csv"
BASELINE_ID = "eegnet_baseline"
ATTACK_ORDER = {
    "mlp_posterior": 0,
    "threshold_max_probability": 1,
    "threshold_label_known": 2,
}


@dataclass(frozen=True)
class ResultRow:
    ordinal: int
    row_type: str
    candidate_id: str
    model: str
    setting: str
    attack_label: str
    attack_type: str
    score_type: str
    complete: bool
    summary_path: str
    task_balanced_accuracy_mean: float | None
    attack_auc_mean: float | None
    delta_task_vs_eegnet: float | None
    delta_auc_vs_eegnet: float | None
    decision: str


@dataclass(frozen=True)
class AggregateRow:
    candidate_id: str
    row_type: str
    model: str
    setting: str
    completed_attack_families: str
    completed_rows: int
    mean_delta_task_vs_eegnet: float | None
    mean_delta_auc_vs_eegnet: float | None
    auc_win_families: int
    utility_safe_rows: int
    decision: str


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _summary_path(raw: dict[str, str]) -> Path | None:
    if raw["reference_summary_path"]:
        path = ROOT / raw["reference_summary_path"]
        return path if path.exists() else None
    output_dir = ROOT / raw["output_dir"]
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


def _row_decision(row: ResultRow) -> str:
    if not row.complete:
        return "pending"
    if row.candidate_id == BASELINE_ID:
        return "baseline"
    if row.delta_auc_vs_eegnet is None or row.delta_task_vs_eegnet is None:
        return "missing baseline"
    if row.delta_auc_vs_eegnet <= -0.010 and row.delta_task_vs_eegnet >= -0.030:
        return "family win within utility"
    if row.delta_auc_vs_eegnet <= -0.010:
        return "family win with utility loss"
    if abs(row.delta_auc_vs_eegnet) <= 0.005 and row.delta_task_vs_eegnet >= -0.030:
        return "practical tie"
    if row.delta_auc_vs_eegnet > 0.005:
        return "privacy regression"
    return "mixed"


def _load_rows() -> list[ResultRow]:
    if not PLAN_CSV.exists():
        raise FileNotFoundError(f"Missing plan CSV: {PLAN_CSV}")

    with PLAN_CSV.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))

    partial: list[ResultRow] = []
    baseline_by_attack: dict[str, tuple[float | None, float | None]] = {}

    for raw in raw_rows:
        summary_path = _summary_path(raw)
        payload = _load_json(summary_path) if summary_path else None
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
        if raw["candidate_id"] == BASELINE_ID and complete:
            baseline_by_attack[raw["attack_label"]] = (task, auc)
        partial.append(
            ResultRow(
                ordinal=int(raw["ordinal"]),
                row_type=raw["row_type"],
                candidate_id=raw["candidate_id"],
                model=raw["model"],
                setting=raw["setting"],
                attack_label=raw["attack_label"],
                attack_type=raw["attack_type"],
                score_type=raw["score_type"],
                complete=complete,
                summary_path=str(summary_path.relative_to(ROOT)) if summary_path else "",
                task_balanced_accuracy_mean=task,
                attack_auc_mean=auc,
                delta_task_vs_eegnet=None,
                delta_auc_vs_eegnet=None,
                decision="pending",
            )
        )

    rows: list[ResultRow] = []
    for row in partial:
        base_task, base_auc = baseline_by_attack.get(row.attack_label, (None, None))
        delta_task = (
            row.task_balanced_accuracy_mean - base_task
            if row.task_balanced_accuracy_mean is not None and base_task is not None
            else None
        )
        delta_auc = (
            row.attack_auc_mean - base_auc
            if row.attack_auc_mean is not None and base_auc is not None
            else None
        )
        updated = ResultRow(
            **{
                **asdict(row),
                "delta_task_vs_eegnet": delta_task,
                "delta_auc_vs_eegnet": delta_auc,
                "decision": "",
            }
        )
        rows.append(ResultRow(**{**asdict(updated), "decision": _row_decision(updated)}))
    return rows


def _aggregate_rows(rows: list[ResultRow]) -> list[AggregateRow]:
    aggregates: list[AggregateRow] = []
    keys = []
    seen = set()
    for row in sorted(rows, key=lambda item: item.ordinal):
        key = (row.candidate_id, row.row_type, row.model, row.setting)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)

    for candidate_id, row_type, model, setting in keys:
        completed = [
            row
            for row in rows
            if row.candidate_id == candidate_id and row.complete
        ]
        delta_auc_values = [
            row.delta_auc_vs_eegnet
            for row in completed
            if row.delta_auc_vs_eegnet is not None
        ]
        delta_task_values = [
            row.delta_task_vs_eegnet
            for row in completed
            if row.delta_task_vs_eegnet is not None
        ]
        auc_win_families = sum(delta <= -0.010 for delta in delta_auc_values)
        utility_safe_rows = sum(
            delta is not None and delta >= -0.030
            for delta in (row.delta_task_vs_eegnet for row in completed)
        )
        aggregates.append(
            AggregateRow(
                candidate_id=candidate_id,
                row_type=row_type,
                model=model,
                setting=setting,
                completed_attack_families=",".join(
                    row.attack_label
                    for row in sorted(completed, key=lambda item: ATTACK_ORDER[item.attack_label])
                ),
                completed_rows=len(completed),
                mean_delta_task_vs_eegnet=mean(delta_task_values) if delta_task_values else None,
                mean_delta_auc_vs_eegnet=mean(delta_auc_values) if delta_auc_values else None,
                auc_win_families=auc_win_families,
                utility_safe_rows=utility_safe_rows,
                decision=_aggregate_decision(
                    candidate_id=candidate_id,
                    row_type=row_type,
                    completed_rows=len(completed),
                    auc_win_families=auc_win_families,
                    utility_safe_rows=utility_safe_rows,
                    mean_delta_auc=mean(delta_auc_values) if delta_auc_values else None,
                    mean_delta_task=mean(delta_task_values) if delta_task_values else None,
                ),
            )
        )
    return aggregates


def _aggregate_decision(
    *,
    candidate_id: str,
    row_type: str,
    completed_rows: int,
    auc_win_families: int,
    utility_safe_rows: int,
    mean_delta_auc: float | None,
    mean_delta_task: float | None,
) -> str:
    if candidate_id == BASELINE_ID:
        return "baseline"
    if row_type == "planned" and completed_rows == 0:
        return "pending"
    if completed_rows >= 2 and completed_rows < 3:
        remaining_rows = 3 - completed_rows
        if mean_delta_task is not None and mean_delta_task < -0.030:
            return "early stop: utility boundary failed"
        if auc_win_families + remaining_rows < 2:
            return "early stop: cannot reach two-family rule"
    if completed_rows < 3:
        return "partial evidence"
    if auc_win_families >= 2 and utility_safe_rows == completed_rows:
        return "candidate clears pre-registered rule"
    if mean_delta_task is not None and mean_delta_task < -0.030:
        return "utility boundary failed"
    if mean_delta_auc is not None and mean_delta_auc > 0.005:
        return "privacy regression"
    if auc_win_families == 1:
        return "attack-specific only"
    return "no stable defense upgrade"


def _write_rows_csv(rows: list[ResultRow]) -> Path:
    out_path = REPORT_ROOT / "physionet_defense_grid_results_rows_v1.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return out_path


def _write_aggregate_csv(rows: list[AggregateRow]) -> Path:
    out_path = REPORT_ROOT / "physionet_defense_grid_results_v1.csv"
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
    out_path = REPORT_ROOT / "physionet_defense_grid_results_v1.md"
    planned_rows = [row for row in rows if row.row_type == "planned"]
    complete_planned = sum(1 for row in planned_rows if row.complete)
    aggregate_by_candidate = {row.candidate_id: row for row in aggregates}
    deferred_rows = [
        row
        for row in planned_rows
        if not row.complete
        and aggregate_by_candidate[row.candidate_id].decision.startswith("early stop:")
    ]
    lines = [
        "# PhysioNet Defense Grid Results",
        "",
        "This report applies the pre-registered PhysioNet defense-grid stop rule. Lower membership AUC is better; deltas compare each row to the same-attack EEGNet reference.",
        "",
        "Companion files:",
        f"- [{aggregate_csv.relative_to(ROOT)}]({aggregate_csv.relative_to(ROOT)})",
        f"- [{rows_csv.relative_to(ROOT)}]({rows_csv.relative_to(ROOT)})",
        "",
        "## Completion",
        "",
        f"- Planned commands complete: `{complete_planned}/{len(planned_rows)}`.",
        f"- Planned commands deferred by stop rule: `{len(deferred_rows)}/{len(planned_rows)}`.",
        "",
        "## Aggregate Read",
        "",
        "| Candidate | Type | Model | Setting | Completed attacks | Mean delta AUC | Mean delta task BA | AUC-win families | Utility-safe rows | Decision |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in aggregates:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.candidate_id}`",
                    f"`{row.row_type}`",
                    f"`{row.model}`",
                    row.setting,
                    f"`{row.completed_attack_families}`",
                    _signed(row.mean_delta_auc_vs_eegnet),
                    _signed(row.mean_delta_task_vs_eegnet),
                    str(row.auc_win_families),
                    str(row.utility_safe_rows),
                    f"`{row.decision}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Row Read",
            "",
            "| Candidate | Attack | Task BA | AUC | Delta AUC | Delta task BA | Decision |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.candidate_id}`",
                    f"`{row.attack_label}`",
                    _fmt(row.task_balanced_accuracy_mean),
                    _fmt(row.attack_auc_mean),
                    _signed(row.delta_auc_vs_eegnet),
                    _signed(row.delta_task_vs_eegnet),
                    f"`{row.decision}`",
                ]
            )
            + " |"
        )

    pending = [
        row
        for row in planned_rows
        if not row.complete
        and not aggregate_by_candidate[row.candidate_id].decision.startswith("early stop:")
    ]
    lines.extend(["", "## Pending Rows", ""])
    if pending:
        for row in pending[:30]:
            lines.append(f"- `{row.candidate_id}` `{row.attack_label}`.")
        if len(pending) > 30:
            lines.append(f"- ...and `{len(pending) - 30}` more pending rows.")
    else:
        lines.append("- None.")
    lines.extend(["", "## Deferred By Stop Rule", ""])
    if deferred_rows:
        for row in deferred_rows:
            decision = aggregate_by_candidate[row.candidate_id].decision
            lines.append(f"- `{row.candidate_id}` `{row.attack_label}`: `{decision}`.")
    else:
        lines.append("- None.")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    aggregates = _aggregate_rows(rows)
    rows_csv = _write_rows_csv(rows)
    aggregate_csv = _write_aggregate_csv(aggregates)
    markdown_path = _write_markdown(rows, aggregates, aggregate_csv, rows_csv)
    print(f"Wrote {rows_csv.relative_to(ROOT)}")
    print(f"Wrote {aggregate_csv.relative_to(ROOT)}")
    print(f"Wrote {markdown_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
