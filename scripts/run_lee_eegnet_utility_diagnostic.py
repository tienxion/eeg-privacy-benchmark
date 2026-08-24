from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_privacy_benchmark.cli import _configure_mne_data_dir
from eeg_privacy_benchmark.models import fit_eegnet
from eeg_privacy_benchmark.results import write_result_json


DATASET_KEY = "lee2019_mi"
PROTOCOL = "cross_session"
SUBJECTS = [1, 2, 3, 4, 5, 6]
CALIBRATION_SEED = 13
DEFAULT_LEARNING_RATES = (1e-4, 3e-4, 1e-3)
UTILITY_GATE = 0.60
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "lee_eegnet_utility_diagnostic"


def _parse_learning_rates(value: str) -> tuple[float, ...]:
    rates = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not rates or any(rate <= 0.0 for rate in rates):
        raise ValueError("Learning rates must be a non-empty list of positive values.")
    if len(rates) != len(set(rates)):
        raise ValueError("Learning rates must be unique.")
    return rates


def _rate_slug(learning_rate: float) -> str:
    return f"{learning_rate:.0e}".replace("-", "m").replace("+", "p")


def select_candidate(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("At least one completed candidate is required.")
    return max(
        candidates,
        key=lambda candidate: (
            candidate["checkpoint_validation_balanced_accuracy"],
            -candidate["learning_rate"],
        ),
    )


def _write_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# Lee EEGNet Utility Diagnostic",
        "",
        "Selection uses validation metrics only. Privacy metrics are not computed.",
        f"Resample frequency: `{summary['resample_hz']}` Hz.",
        "",
        "| Learning rate | Epochs trained | Checkpoint epoch | Checkpoint val BA | Test BA |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in summary["candidates"]:
        lines.append(
            "| {learning_rate:.1e} | {epochs_trained} | {best_checkpoint_epoch} | "
            "{checkpoint_validation_balanced_accuracy:.4f} | "
            "{balanced_accuracy:.4f} |".format(**candidate)
        )
    lines.extend(
        [
            "",
            f"- Selected learning rate: `{summary['selected_learning_rate']:.1e}`",
            f"- Utility gate: `{summary['utility_gate']:.2f}`",
            f"- Selected test balanced accuracy: `{summary['selected_test_balanced_accuracy']:.4f}`",
            f"- Gate passed: `{str(summary['utility_gate_passed']).lower()}`",
            f"- Status: `{summary['status']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Lee EEGNet utility diagnostic."
    )
    parser.add_argument(
        "--learning-rates",
        default=",".join(str(rate) for rate in DEFAULT_LEARNING_RATES),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resample-hz", type=float)
    parser.add_argument("--mne-data-dir", default="raw_data/mne_data")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    learning_rates = _parse_learning_rates(args.learning_rates)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _configure_mne_data_dir(args.mne_data_dir, DATASET_KEY)
    candidates = []
    resample_slug = (
        "" if args.resample_hz is None else f"resample{args.resample_hz:g}_"
    )
    for learning_rate in learning_rates:
        output_path = (
            args.output_root
            / f"seed13_{resample_slug}lr{_rate_slug(learning_rate)}.json"
        )
        if output_path.exists():
            candidate = json.loads(output_path.read_text(encoding="utf-8"))
            print(f"learning_rate={learning_rate:.1e} status=skipped_complete")
        elif args.dry_run:
            print(
                f"learning_rate={learning_rate:.1e} status=planned "
                f"output={output_path}"
            )
            continue
        else:
            print(f"learning_rate={learning_rate:.1e} status=training")
            artifacts = fit_eegnet(
                DATASET_KEY,
                protocol=PROTOCOL,
                seed=CALIBRATION_SEED,
                subjects=SUBJECTS,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=learning_rate,
                validation_fraction=0.2,
                early_stopping_patience=args.early_stopping_patience,
                resample_hz=args.resample_hz,
            )
            candidate = artifacts.result.to_dict()
            candidate.update(
                {
                    "subjects": SUBJECTS,
                    "learning_rate": learning_rate,
                    "selection_scope": "validation_only",
                    "privacy_metrics_computed": False,
                }
            )
            write_result_json(candidate, output_path)
            print(
                f"learning_rate={learning_rate:.1e} status=complete "
                "checkpoint_validation_balanced_accuracy="
                f"{candidate['checkpoint_validation_balanced_accuracy']:.6f}"
            )
        candidates.append(candidate)

    if args.dry_run:
        return
    selected = select_candidate(candidates)
    gate_passed = selected["balanced_accuracy"] >= UTILITY_GATE
    summary = {
        "dataset_key": DATASET_KEY,
        "protocol": PROTOCOL,
        "subjects": SUBJECTS,
        "calibration_seed": CALIBRATION_SEED,
        "resample_hz": args.resample_hz,
        "selection_source": "validation_only",
        "checkpoint_selection_metric": "minimum_validation_loss",
        "candidate_selection_metric": "checkpoint_validation_balanced_accuracy",
        "test_metrics_used_for_selection": False,
        "privacy_metrics_computed": False,
        "utility_gate": UTILITY_GATE,
        "selected_learning_rate": selected["learning_rate"],
        "selected_test_balanced_accuracy": selected["balanced_accuracy"],
        "utility_gate_passed": gate_passed,
        "status": (
            "gate_passed_ready_for_confirmation"
            if gate_passed
            else "gate_failed_architecture_diagnostic_required"
        ),
        "candidates": candidates,
    }
    json_path = args.output_root / f"{resample_slug}lee_eegnet_utility_diagnostic.json"
    markdown_path = args.output_root / f"{resample_slug}lee_eegnet_utility_diagnostic.md"
    write_result_json(summary, json_path)
    _write_markdown(summary, markdown_path)
    print(f"status={summary['status']}")
    print(f"wrote_json={json_path.resolve()}")
    print(f"wrote_markdown={markdown_path.resolve()}")


if __name__ == "__main__":
    main()
