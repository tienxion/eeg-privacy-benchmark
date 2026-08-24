from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from plan_runner_common import execute_rows, is_complete, limit_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN_CSV = ROOT / "outputs" / "reports" / "physionet_random_subset_validation_plan_v1.csv"
LOG_ROOT = ROOT / "outputs" / "logs"


@dataclass(frozen=True)
class PlannedCommand:
    ordinal: int
    tier: str
    subset_id: str
    model: str
    attack_type: str
    score_type: str
    output_dir: Path
    command: str

    @property
    def label(self) -> str:
        return (
            f"{self.ordinal:03d} tier={self.tier} subset={self.subset_id} "
            f"model={self.model} attack={self.attack_type}::{self.score_type}"
        )


def _load_plan(path: Path, tier: str) -> list[PlannedCommand]:
    if not path.exists():
        raise FileNotFoundError(f"plan CSV not found: {path}")

    rows: list[PlannedCommand] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for ordinal, row in enumerate(reader, start=1):
            if row["tier"] != tier:
                continue
            rows.append(
                PlannedCommand(
                    ordinal=ordinal,
                    tier=row["tier"],
                    subset_id=row["subset_id"],
                    model=row["model"],
                    attack_type=row["attack_type"],
                    score_type=row["score_type"],
                    output_dir=ROOT / row["output_dir"],
                    command=row["command"],
                )
            )
    return rows


def _is_complete(command: PlannedCommand) -> bool:
    return is_complete(command.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tier from the generated PhysioNet random-subset validation plan."
    )
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    parser.add_argument("--tier", choices=["A", "B", "C"], default="A")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = _load_plan(args.plan_csv, args.tier)
    commands = limit_rows(commands, args.limit)

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_ROOT / f"physionet_random_subset_validation_tier_{args.tier.lower()}_{timestamp}.log"

    print(f"Loaded {len(commands)} Tier {args.tier} commands")
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
