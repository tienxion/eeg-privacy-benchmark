from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from plan_runner_common import execute_rows, filter_rows, is_complete, limit_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN_CSV = ROOT / "outputs" / "reports" / "attacker_refit_plan_v1.csv"
LOG_ROOT = ROOT / "outputs" / "logs"


@dataclass(frozen=True)
class PlannedCommand:
    ordinal: int
    priority: str
    dataset: str
    model: str
    attack_seed: int
    output_dir: Path
    command: str

    @property
    def label(self) -> str:
        return (
            f"{self.ordinal:02d} priority={self.priority} dataset={self.dataset} "
            f"model={self.model} attack_seed={self.attack_seed}"
        )


def _load_plan(path: Path) -> list[PlannedCommand]:
    if not path.exists():
        raise FileNotFoundError(f"plan CSV not found: {path}")

    rows: list[PlannedCommand] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for ordinal, row in enumerate(reader, start=1):
            rows.append(
                PlannedCommand(
                    ordinal=ordinal,
                    priority=row["priority"],
                    dataset=row["dataset"],
                    model=row["model"],
                    attack_seed=int(row["attack_seed"]),
                    output_dir=ROOT / row["output_dir"],
                    command=row["command"],
                )
            )
    return rows


def _is_complete(command: PlannedCommand) -> bool:
    return is_complete(command.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the generated attacker-refit plan.")
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    parser.add_argument(
        "--priority",
        action="append",
        choices=["P1", "P2"],
        help="Priority to run. Can be repeated. Default: all priorities.",
    )
    parser.add_argument("--dataset", action="append", help="Dataset key to run. Can be repeated.")
    parser.add_argument("--model", action="append", help="Model key to run. Can be repeated.")
    parser.add_argument("--attack-seed", type=int, action="append", help="Attack seed to run. Can be repeated.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = _load_plan(args.plan_csv)
    commands = filter_rows(commands, "priority", args.priority)
    commands = filter_rows(commands, "dataset", args.dataset)
    commands = filter_rows(commands, "model", args.model)
    commands = filter_rows(commands, "attack_seed", args.attack_seed)
    commands = limit_rows(commands, args.limit)

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_ROOT / f"attacker_refit_{timestamp}.log"

    print(f"Loaded {len(commands)} attacker-refit commands")
    print(f"Log: {log_path.relative_to(ROOT)}")

    summary = execute_rows(
        commands,
        root=ROOT,
        log_path=log_path,
        dry_run=args.dry_run,
        force=args.force,
    )
    print(
        f"Completed {summary.completed}; skipped {summary.skipped}; "
        f"dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
