from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkflowSpec:
    key: str
    description: str
    plan_csv: str
    build_command: str
    run_command: str
    summarize_command: str
    row_type_field: str | None = None
    row_type_value: str | None = None
    status_note: str = ""


@dataclass(frozen=True)
class WorkflowStatus:
    key: str
    description: str
    plan_csv: str
    plan_exists: bool
    planned_rows: int
    completed_rows: int
    status: str
    status_note: str
    build_command: str
    run_command: str
    summarize_command: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


CLI_PREFIX = "PYTHONPATH=src .venv/bin/python -m eeg_privacy_benchmark.cli"

WORKFLOW_SPECS = (
    WorkflowSpec(
        key="attacker-refit",
        description="Learned membership-attacker refit calibration.",
        plan_csv="outputs/reports/attacker_refit_plan_v1.csv",
        build_command=f"{CLI_PREFIX} build-attacker-refit-plan",
        run_command=f"{CLI_PREFIX} run-attacker-refit-plan --dry-run --limit 2",
        summarize_command=f"{CLI_PREFIX} summarize-attacker-refit",
    ),
    WorkflowSpec(
        key="bnci-lee-validation",
        description="Expanded-subject BNCI/Lee leakage validation.",
        plan_csv="outputs/reports/bnci_lee_validation_plan_v1.csv",
        build_command=f"{CLI_PREFIX} build-bnci-lee-validation-plan",
        run_command=f"{CLI_PREFIX} run-bnci-lee-validation-plan --tier A --dry-run --limit 2",
        summarize_command=f"{CLI_PREFIX} summarize-bnci-lee-validation",
    ),
    WorkflowSpec(
        key="physionet-random-subset",
        description="Tiered deterministic PhysioNet subset validation.",
        plan_csv="outputs/reports/physionet_random_subset_validation_plan_v1.csv",
        build_command=f"{CLI_PREFIX} build-physionet-random-subset-validation-plan",
        run_command=(
            f"{CLI_PREFIX} run-physionet-random-subset-validation-plan "
            "--tier A --dry-run --limit 2"
        ),
        summarize_command=f"{CLI_PREFIX} summarize-physionet-random-subset-validation",
    ),
    WorkflowSpec(
        key="physionet-defense-grid",
        description="Pre-registered PhysioNet non-adversarial defense grid.",
        plan_csv="outputs/reports/physionet_defense_grid_plan_v1.csv",
        build_command=f"{CLI_PREFIX} build-physionet-defense-grid-plan",
        run_command=f"{CLI_PREFIX} run-physionet-defense-grid-plan --dry-run --limit 2",
        summarize_command=f"{CLI_PREFIX} summarize-physionet-defense-grid",
        row_type_field="row_type",
        row_type_value="planned",
        status_note=(
            "Remaining rows are intentionally deferred by the pre-registered stop rule; "
            "this is not an execution blocker."
        ),
    ),
    WorkflowSpec(
        key="physionet-cross-protocol",
        description="Frozen PhysioNet cross-run holdout validation.",
        plan_csv="outputs/reports/physionet_cross_protocol_holdout_plan_v1.csv",
        build_command=f"{CLI_PREFIX} build-physionet-cross-protocol-holdout-plan",
        run_command=(
            f"{CLI_PREFIX} run-physionet-cross-protocol-holdout-plan "
            "--dry-run --limit 2"
        ),
        summarize_command=f"{CLI_PREFIX} summarize-physionet-cross-protocol-holdout",
    ),
    WorkflowSpec(
        key="physionet-mlp-seed-extension",
        description="PhysioNet learned-attack seed-stability extension.",
        plan_csv="outputs/reports/physionet_mlp_seed_extension_plan_v1.csv",
        build_command=f"{CLI_PREFIX} build-physionet-mlp-seed-extension-plan",
        run_command=(
            f"{CLI_PREFIX} run-physionet-mlp-seed-extension-plan "
            "--dry-run --limit 2"
        ),
        summarize_command=f"{CLI_PREFIX} summarize-physionet-mlp-seed-extension",
    ),
)


def _load_planned_rows(spec: WorkflowSpec, repo_root: Path) -> list[dict[str, str]]:
    plan_path = repo_root / spec.plan_csv
    if not plan_path.exists():
        return []
    with plan_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if spec.row_type_field is None:
        return rows
    return [
        row
        for row in rows
        if row.get(spec.row_type_field) == spec.row_type_value
    ]


def _resolve_output_dir(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def inspect_workflow(spec: WorkflowSpec, repo_root: Path) -> WorkflowStatus:
    plan_path = repo_root / spec.plan_csv
    rows = _load_planned_rows(spec, repo_root)
    completed = sum(
        any(_resolve_output_dir(row["output_dir"], repo_root).glob("*_summary.json"))
        for row in rows
    )
    if not plan_path.exists():
        status = "missing-plan"
    elif not rows:
        status = "empty-plan"
    elif completed == len(rows):
        status = "complete"
    elif completed:
        status = "partial"
    else:
        status = "not-started"
    return WorkflowStatus(
        key=spec.key,
        description=spec.description,
        plan_csv=spec.plan_csv,
        plan_exists=plan_path.exists(),
        planned_rows=len(rows),
        completed_rows=completed,
        status=status,
        status_note=spec.status_note,
        build_command=spec.build_command,
        run_command=spec.run_command,
        summarize_command=spec.summarize_command,
    )


def inspect_workflows(repo_root: Path) -> list[WorkflowStatus]:
    return [inspect_workflow(spec, repo_root) for spec in WORKFLOW_SPECS]
