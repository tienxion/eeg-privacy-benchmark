from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "outputs" / "reports"
PLAN_CSV = REPORT_ROOT / "physionet_cross_protocol_holdout_plan_v1.csv"

MD_OUT = REPORT_ROOT / "physionet_cross_protocol_holdout_results_v1.md"
CSV_OUT = REPORT_ROOT / "physionet_cross_protocol_holdout_results_v1.csv"


@dataclass(frozen=True)
class ResultRow:
    ordinal: int
    candidate_id: str
    model: str
    setting: str
    protocol: str
    attack_label: str
    attack_type: str
    score_type: str
    status: str
    summary_path: str
    task_balanced_accuracy_mean: float | None
    attack_auc_mean: float | None
    delta_task_vs_eegnet: float | None
    delta_auc_vs_eegnet: float | None
    decision: str


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _markdown_path(path: Path) -> str:
    try:
        return str(ROOT / path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _summary_path(output_dir: Path) -> Path | None:
    matches = sorted(output_dir.glob("*_summary.json"))
    return matches[0] if matches else None


def _summary_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_plan_rows() -> list[dict[str, str]]:
    if not PLAN_CSV.exists():
        raise FileNotFoundError(f"Missing plan CSV: {PLAN_CSV}")
    with PLAN_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_decision(row: ResultRow) -> str:
    if row.status != "complete":
        return "pending"
    if row.candidate_id == "eegnet_baseline":
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


def _build_rows(plan_rows: list[dict[str, str]]) -> list[ResultRow]:
    partial: list[ResultRow] = []
    baseline_by_attack: dict[str, tuple[float | None, float | None]] = {}

    for raw in plan_rows:
        summary_path = _summary_path(ROOT / raw["output_dir"])
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
        if raw["candidate_id"] == "eegnet_baseline" and complete:
            baseline_by_attack[raw["attack_label"]] = (task, auc)
        partial.append(
            ResultRow(
                ordinal=int(raw["ordinal"]),
                candidate_id=raw["candidate_id"],
                model=raw["model"],
                setting=raw["setting"],
                protocol=raw["protocol"],
                attack_label=raw["attack_label"],
                attack_type=raw["attack_type"],
                score_type=raw["score_type"],
                status="complete" if complete else "missing",
                summary_path=_summary_display_path(summary_path) if summary_path else "",
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
            }
        )
        rows.append(ResultRow(**{**asdict(updated), "decision": _row_decision(updated)}))
    return rows


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _signed(value: float | None) -> str:
    return "" if value is None else f"{value:+.4f}"


def _write_csv(rows: list[ResultRow]) -> Path:
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return CSV_OUT


def _aggregate_decision(rows: list[ResultRow]) -> str:
    completed = [row for row in rows if row.status == "complete"]
    if not completed:
        return "pending: no cross-protocol holdout commands have completed."
    non_baseline = [
        row
        for row in completed
        if row.candidate_id != "eegnet_baseline"
        and row.delta_auc_vs_eegnet is not None
        and row.delta_task_vs_eegnet is not None
    ]
    if not non_baseline:
        return "partial: baseline rows exist but no candidate deltas are available."
    if len(completed) < len(rows):
        return "mixed or partial protocol-sensitivity evidence; keep PhysioNet unresolved until all planned rows complete."
    wins = sum(row.delta_auc_vs_eegnet <= -0.010 for row in non_baseline)
    mean_delta_auc = mean(row.delta_auc_vs_eegnet for row in non_baseline)
    mean_delta_task = mean(row.delta_task_vs_eegnet for row in non_baseline)
    winning_attack_types_by_candidate: dict[str, set[str]] = {}
    for row in non_baseline:
        if row.delta_auc_vs_eegnet <= -0.010 and row.delta_task_vs_eegnet >= -0.030:
            winning_attack_types_by_candidate.setdefault(row.candidate_id, set()).add(
                row.attack_type
            )
    if any(
        len(attack_types) >= 2
        for attack_types in winning_attack_types_by_candidate.values()
    ):
        return "candidate evidence may clear the two-family rule; inspect per-family rows."
    if mean_delta_task < -0.030:
        return "utility boundary failed on average; keep PhysioNet unresolved."
    if mean_delta_auc > 0.005:
        return "privacy regression on average; keep PhysioNet unresolved."
    if wins:
        return "threshold-only or attack-specific privacy wins; keep PhysioNet unresolved."
    return "mixed or partial protocol-sensitivity evidence; keep PhysioNet unresolved until all planned rows complete."


def _write_markdown(rows: list[ResultRow], csv_path: Path) -> Path:
    completed = sum(row.status == "complete" for row in rows)
    lines = [
        "# PhysioNet Cross-Protocol Holdout Results",
        "",
        "This report summarizes completed rows from the `cross_protocol_physionet_holdout` plan. Lower membership-inference AUC is better for privacy. Deltas are computed versus the same-attack plain EEGNet baseline when available.",
        "",
        f"Companion CSV: [physionet_cross_protocol_holdout_results_v1.csv]({_markdown_path(csv_path)})",
        "",
        "## Completion",
        "",
        f"- Completed rows: `{completed}/{len(rows)}`.",
        f"- Current decision: {_aggregate_decision(rows)}",
        "",
        "## Rows",
        "",
        "| Ordinal | Candidate | Attack | Status | Task BA | Attack AUC | Delta task | Delta AUC | Decision |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.ordinal}` | `{row.candidate_id}` | `{row.attack_type}::{row.score_type}` | "
            f"`{row.status}` | `{_fmt(row.task_balanced_accuracy_mean)}` | "
            f"`{_fmt(row.attack_auc_mean)}` | `{_signed(row.delta_task_vs_eegnet)}` | "
            f"`{_signed(row.delta_auc_vs_eegnet)}` | {row.decision} |"
        )
    lines.extend(
        [
            "",
            "## Claim Guardrail",
            "",
            "Do not change the paper-facing PhysioNet claim until completed cross-run rows satisfy the pre-registered two-family privacy and utility rule. Pending or mixed results preserve the current unresolved/protocol-sensitive framing.",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")
    return MD_OUT


def main() -> None:
    rows = _build_rows(_load_plan_rows())
    csv_path = _write_csv(rows)
    md_path = _write_markdown(rows, csv_path)
    print(f"Wrote {_display_path(csv_path)}")
    print(f"Wrote {_display_path(md_path)}")
    print(f"Completed rows={sum(row.status == 'complete' for row in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
