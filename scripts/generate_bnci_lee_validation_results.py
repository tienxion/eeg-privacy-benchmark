from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "outputs" / "reports"
PLAN_CSV = REPORT_ROOT / "bnci_lee_validation_plan_v1.csv"
MODEL_ORDER = {
    "eegnet": 0,
    "bottleneck_eegnet": 1,
    "adversarial_eegnet": 2,
    "csp_lda": 3,
}
BASELINE_MODEL = "eegnet"


@dataclass(frozen=True)
class ResultRow:
    tier: str
    dataset: str
    subjects: str
    model: str
    probe: str
    attack_type: str
    score_type: str
    complete: bool
    summary_path: str
    seeds: str
    task_balanced_accuracy: float | None
    privacy_metric: float | None
    privacy_metric_name: str
    delta_task_vs_eegnet: float | None
    delta_privacy_vs_eegnet: float | None
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


def _privacy_metric(payload: dict, probe: str) -> tuple[str, float | None]:
    if probe == "subject_id":
        value = payload.get("subject_id_accuracy_mean")
        return "subject_id_accuracy", float(value) if value is not None else None
    value = payload.get("attack_auc_mean")
    return "attack_auc", float(value) if value is not None else None


def _decision(row: ResultRow) -> str:
    if not row.complete:
        return "pending"
    if row.model == BASELINE_MODEL:
        return "baseline"
    if row.delta_privacy_vs_eegnet is None:
        return "missing baseline"

    # Lower subject-ID accuracy and lower membership AUC both mean less leakage.
    if row.model == "bottleneck_eegnet":
        if row.probe == "subject_id":
            if row.delta_privacy_vs_eegnet <= -0.020 and (
                row.delta_task_vs_eegnet is None or row.delta_task_vs_eegnet >= -0.030
            ):
                return "supports bottleneck subject-ID claim"
            if row.delta_privacy_vs_eegnet > 0.005:
                return "bottleneck subject-ID regression"
            return "bottleneck subject-ID tie"
        if row.delta_privacy_vs_eegnet <= 0.005:
            return "no material membership regression"
        return "membership regression"

    if row.delta_privacy_vs_eegnet <= -0.010:
        return "privacy-favorable"
    if row.delta_privacy_vs_eegnet > 0.005:
        return "more leaky than eegnet"
    return "practical tie"


def _load_rows() -> list[ResultRow]:
    if not PLAN_CSV.exists():
        raise FileNotFoundError(f"Missing plan CSV: {PLAN_CSV}")

    raw_rows: list[dict[str, str]] = []
    with PLAN_CSV.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))

    baseline_by_key: dict[tuple[str, str, str, str, str], tuple[float | None, float | None]] = {}
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
        metric_name, privacy = _privacy_metric(payload or {}, raw["probe"])
        key = (
            raw["dataset"],
            raw["subjects"],
            raw["probe"],
            raw["attack_type"],
            raw["score_type"],
        )
        if raw["model"] == BASELINE_MODEL and complete:
            baseline_by_key[key] = (task, privacy)

        partial_rows.append(
            ResultRow(
                tier=raw["tier"],
                dataset=raw["dataset"],
                subjects=raw["subjects"],
                model=raw["model"],
                probe=raw["probe"],
                attack_type=raw["attack_type"],
                score_type=raw["score_type"],
                complete=complete,
                summary_path=str(summary.relative_to(ROOT)) if summary else "",
                seeds=raw["seeds"],
                task_balanced_accuracy=task,
                privacy_metric=privacy,
                privacy_metric_name=metric_name,
                delta_task_vs_eegnet=None,
                delta_privacy_vs_eegnet=None,
                decision="pending",
            )
        )

    rows: list[ResultRow] = []
    for row in partial_rows:
        key = (row.dataset, row.subjects, row.probe, row.attack_type, row.score_type)
        base_task, base_privacy = baseline_by_key.get(key, (None, None))
        delta_task = (
            row.task_balanced_accuracy - base_task
            if row.task_balanced_accuracy is not None and base_task is not None
            else None
        )
        delta_privacy = (
            row.privacy_metric - base_privacy
            if row.privacy_metric is not None and base_privacy is not None
            else None
        )
        updated = ResultRow(
            **{
                **asdict(row),
                "delta_task_vs_eegnet": delta_task,
                "delta_privacy_vs_eegnet": delta_privacy,
                "decision": "",
            }
        )
        rows.append(ResultRow(**{**asdict(updated), "decision": _decision(updated)}))
    return rows


def _write_csv(rows: list[ResultRow]) -> Path:
    out_path = REPORT_ROOT / "bnci_lee_validation_results_v1.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                item.dataset,
                item.tier,
                item.probe,
                item.attack_type,
                item.score_type,
                MODEL_ORDER.get(item.model, 99),
            ),
        ):
            writer.writerow(asdict(row))
    return out_path


def _write_markdown(rows: list[ResultRow], csv_path: Path) -> Path:
    out_path = REPORT_ROOT / "bnci_lee_validation_results_v1.md"
    complete = sum(1 for row in rows if row.complete)
    by_tier = {
        tier: (sum(1 for row in rows if row.tier == tier and row.complete), sum(1 for row in rows if row.tier == tier))
        for tier in ("A", "B", "C")
    }
    lines = [
        "# BNCI/Lee Validation Results",
        "",
        "This report summarizes whichever staged BNCI/Lee validation commands have completed. Lower subject-ID accuracy and lower membership AUC mean less leakage; deltas are paired against the same-scope plain EEGNet row.",
        "",
        "Companion files:",
        f"- [{csv_path.relative_to(ROOT)}]({csv_path.relative_to(ROOT)})",
        "",
        "## Completion",
        "",
        f"- Overall: `{complete}/{len(rows)}` commands complete.",
    ]
    for tier, (done, total) in by_tier.items():
        lines.append(f"- Tier {tier}: `{done}/{total}` commands complete.")

    lines.extend(
        [
            "",
            "## Completed Rows",
            "",
            "| Tier | Dataset | Probe | Attack/score | Model | Task BA | Privacy metric | Delta privacy | Decision |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    completed_rows = [row for row in rows if row.complete]
    for row in sorted(
        completed_rows,
        key=lambda item: (
            item.dataset,
            item.tier,
            item.probe,
            item.attack_type,
            item.score_type,
            MODEL_ORDER.get(item.model, 99),
        ),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.tier}`",
                    f"`{row.dataset}`",
                    f"`{row.probe}`",
                    f"`{row.attack_type}::{row.score_type}`",
                    f"`{row.model}`",
                    _fmt(row.task_balanced_accuracy),
                    _fmt(row.privacy_metric),
                    _signed(row.delta_privacy_vs_eegnet),
                    f"`{row.decision}`",
                ]
            )
            + " |"
        )
    if not completed_rows:
        lines.append("|  |  |  |  |  |  |  |  | `pending` |")

    lines.extend(
        [
            "",
            "## Current Read",
            "",
        ]
    )
    bottleneck_rows = [
        row
        for row in completed_rows
        if row.model == "bottleneck_eegnet" and row.probe == "subject_id"
    ]
    if bottleneck_rows:
        for row in bottleneck_rows:
            lines.append(
                f"- `{row.dataset}` bottleneck subject-ID delta is `{_signed(row.delta_privacy_vs_eegnet)}` with task delta `{_signed(row.delta_task_vs_eegnet)}`: `{row.decision}`."
            )
    else:
        lines.append("- No bottleneck subject-ID validation row is complete yet.")
    bottleneck_membership_rows = [
        row
        for row in completed_rows
        if row.model == "bottleneck_eegnet"
        and row.probe == "membership"
        and row.attack_type == "mlp"
        and row.score_type == "posterior_probabilities"
    ]
    if bottleneck_membership_rows:
        for row in bottleneck_membership_rows:
            lines.append(
                f"- `{row.dataset}` bottleneck MLP-membership delta is `{_signed(row.delta_privacy_vs_eegnet)}` with task delta `{_signed(row.delta_task_vs_eegnet)}`: `{row.decision}`."
            )
    else:
        lines.append("- No bottleneck MLP-membership validation row is complete yet.")

    lines.extend(
        [
            "",
            "## Pending Rows",
            "",
        ]
    )
    pending_rows = [row for row in rows if not row.complete]
    for row in pending_rows[:20]:
        lines.append(
            f"- Tier `{row.tier}` `{row.dataset}` `{row.model}` `{row.attack_type}::{row.score_type}`."
        )
    if len(pending_rows) > 20:
        lines.append(f"- ...and `{len(pending_rows) - 20}` more pending rows.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    csv_path = _write_csv(rows)
    markdown_path = _write_markdown(rows, csv_path)
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {markdown_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
