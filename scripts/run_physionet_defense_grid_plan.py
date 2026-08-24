from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from plan_runner_common import execute_rows, is_complete, limit_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN_CSV = ROOT / "outputs" / "reports" / "physionet_defense_grid_plan_v1.csv"
LOG_ROOT = ROOT / "outputs" / "logs"


@dataclass(frozen=True)
class PlannedCommand:
    ordinal: int
    candidate_id: str
    model: str
    attack_label: str
    attack_type: str
    score_type: str
    output_dir: Path
    command: str

    @property
    def label(self) -> str:
        return (
            f"{self.ordinal:02d} candidate={self.candidate_id} model={self.model} "
            f"attack={self.attack_type}::{self.score_type}"
        )


def _load_plan(path: Path, candidate: str | None, attack_label: str | None) -> list[PlannedCommand]:
    if not path.exists():
        raise FileNotFoundError(f"plan CSV not found: {path}")

    rows: list[PlannedCommand] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if raw["row_type"] != "planned":
                continue
            if candidate and raw["candidate_id"] != candidate:
                continue
            if attack_label and raw["attack_label"] != attack_label:
                continue
            rows.append(
                PlannedCommand(
                    ordinal=int(raw["ordinal"]),
                    candidate_id=raw["candidate_id"],
                    model=raw["model"],
                    attack_label=raw["attack_label"],
                    attack_type=raw["attack_type"],
                    score_type=raw["score_type"],
                    output_dir=ROOT / raw["output_dir"],
                    command=raw["command"],
                )
            )
    return rows


def _is_complete(command: PlannedCommand) -> bool:
    return is_complete(command.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run planned rows from the PhysioNet defense-grid plan."
    )
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    parser.add_argument("--candidate", default=None)
    parser.add_argument(
        "--attack-label",
        choices=["mlp_posterior", "threshold_max_probability", "threshold_label_known"],
        default=None,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = _load_plan(args.plan_csv, args.candidate, args.attack_label)
    commands = limit_rows(commands, args.limit)

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_ROOT / f"physionet_defense_grid_{timestamp}.log"

    print(f"Loaded {len(commands)} planned defense-grid commands")
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
