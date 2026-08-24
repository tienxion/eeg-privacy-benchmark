from __future__ import annotations

import contextlib
import io
import tempfile
from dataclasses import dataclass
from pathlib import Path

from plan_runner_common import execute_rows, filter_rows, is_complete, limit_rows


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATHS = (
    ROOT / "scripts" / "run_attacker_refit_plan.py",
    ROOT / "scripts" / "run_bnci_lee_validation_plan.py",
    ROOT / "scripts" / "run_physionet_random_subset_validation_plan.py",
    ROOT / "scripts" / "run_physionet_defense_grid_plan.py",
    ROOT / "scripts" / "run_physionet_cross_protocol_holdout_plan.py",
    ROOT / "scripts" / "run_physionet_mlp_seed_extension_plan.py",
)


@dataclass(frozen=True)
class ToyRow:
    label: str
    group: str
    output_dir: Path
    command: str


def check_filter_and_limit_preserve_order() -> None:
    rows = [
        ToyRow("one", "A", Path("one"), "true"),
        ToyRow("two", "B", Path("two"), "true"),
        ToyRow("three", "A", Path("three"), "true"),
    ]
    filtered = filter_rows(rows, "group", ["A"])
    if [row.label for row in filtered] != ["one", "three"]:
        raise AssertionError("shared filtering changed row order")
    if [row.label for row in limit_rows(filtered, 1)] != ["one"]:
        raise AssertionError("shared limit did not retain the first selected row")
    if limit_rows(filtered, 0):
        raise AssertionError("zero limit should select no rows")
    if filter_rows(rows, "group", None) != rows:
        raise AssertionError("empty filtering should preserve all rows")


def check_completion_detection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "output"
        output_dir.mkdir()
        if is_complete(output_dir):
            raise AssertionError("fresh output directory should not be complete")
        (output_dir / "toy_summary.json").write_text("{}\n", encoding="utf-8")
        if not is_complete(output_dir):
            raise AssertionError("summary JSON should mark the row complete")


def check_execute_rows_dry_run_skip_and_success() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dry_log = tmp_path / "dry.log"
        dry_row = ToyRow("dry", "A", tmp_path / "dry-output", "exit 99")
        with contextlib.redirect_stdout(io.StringIO()):
            dry_summary = execute_rows(
                [dry_row],
                root=tmp_path,
                log_path=dry_log,
                dry_run=True,
                force=False,
            )
        if dry_summary.completed != 0 or dry_summary.skipped != 0:
            raise AssertionError("dry run should neither complete nor skip a fresh row")
        if "DRY RUN: dry" not in dry_log.read_text(encoding="utf-8"):
            raise AssertionError("dry run did not record the selected row")

        complete_dir = tmp_path / "complete-output"
        complete_dir.mkdir()
        (complete_dir / "toy_summary.json").write_text("{}\n", encoding="utf-8")
        skip_log = tmp_path / "skip.log"
        skip_row = ToyRow("skip", "A", complete_dir, "exit 99")
        with contextlib.redirect_stdout(io.StringIO()):
            skip_summary = execute_rows(
                [skip_row],
                root=tmp_path,
                log_path=skip_log,
                dry_run=False,
                force=False,
            )
        if skip_summary.completed != 0 or skip_summary.skipped != 1:
            raise AssertionError("completed row should be skipped without force")

        run_log = tmp_path / "run.log"
        run_row = ToyRow("run", "A", tmp_path / "run-output", "printf 'shared-runner-ok\\n'")
        with contextlib.redirect_stdout(io.StringIO()):
            run_summary = execute_rows(
                [run_row],
                root=tmp_path,
                log_path=run_log,
                dry_run=False,
                force=False,
            )
        if run_summary.completed != 1 or run_summary.skipped != 0:
            raise AssertionError("successful row did not increment completion count")
        if "shared-runner-ok" not in run_log.read_text(encoding="utf-8"):
            raise AssertionError("successful command output was not captured in the log")


def check_all_plan_runners_use_shared_execution() -> None:
    for path in RUNNER_PATHS:
        text = path.read_text(encoding="utf-8")
        if "from plan_runner_common import" not in text:
            raise AssertionError(f"runner does not import the shared utility: {path.name}")
        if "subprocess.run(" in text:
            raise AssertionError(f"runner still duplicates shell execution: {path.name}")
        if "execute_rows(" not in text:
            raise AssertionError(f"runner does not use shared execution: {path.name}")


def main() -> None:
    checks = [
        check_filter_and_limit_preserve_order,
        check_completion_detection,
        check_execute_rows_dry_run_skip_and_success,
        check_all_plan_runners_use_shared_execution,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print(f"PASS shared plan-runner checks={len(checks)}")


if __name__ == "__main__":
    main()
