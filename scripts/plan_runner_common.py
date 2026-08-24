from __future__ import annotations

import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar


class RunnablePlanRow(Protocol):
    label: str
    output_dir: Path
    command: str


PlanRow = TypeVar("PlanRow")


@dataclass(frozen=True)
class ExecutionSummary:
    completed: int
    skipped: int


def filter_rows(
    rows: Sequence[PlanRow],
    attribute: str,
    selected_values: Iterable[object] | None,
) -> list[PlanRow]:
    if not selected_values:
        return list(rows)
    selected = set(selected_values)
    return [row for row in rows if getattr(row, attribute) in selected]


def limit_rows(rows: Sequence[PlanRow], limit: int | None) -> list[PlanRow]:
    if limit is None:
        return list(rows)
    return list(rows[:limit])


def is_complete(output_dir: Path) -> bool:
    return any(output_dir.glob("*_summary.json"))


def append_log(log_path: Path, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _run_command(row: RunnablePlanRow, log_path: Path, root: Path) -> int:
    append_log(log_path, f"\n=== START {row.label} ===")
    append_log(log_path, row.command)
    result = subprocess.run(
        row.command,
        cwd=root,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    append_log(log_path, result.stdout or "")
    append_log(log_path, f"=== END {row.label} exit={result.returncode} ===")
    return result.returncode


def execute_rows(
    rows: Sequence[RunnablePlanRow],
    *,
    root: Path,
    log_path: Path,
    dry_run: bool,
    force: bool,
) -> ExecutionSummary:
    skipped = 0
    completed = 0
    for row in rows:
        if is_complete(row.output_dir) and not force:
            skipped += 1
            print(f"SKIP complete: {row.label}")
            append_log(log_path, f"SKIP complete: {row.label}")
            continue
        if dry_run:
            print(f"DRY RUN: {row.label}")
            append_log(log_path, f"DRY RUN: {row.label}\n{row.command}")
            continue

        print(f"RUN: {row.label}", flush=True)
        returncode = _run_command(row, log_path, root)
        if returncode != 0:
            raise SystemExit(f"Command failed with exit code {returncode}: {row.label}")
        completed += 1

    return ExecutionSummary(completed=completed, skipped=skipped)
