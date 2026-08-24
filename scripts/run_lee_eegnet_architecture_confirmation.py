from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev


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
SEEDS = (13, 17, 19, 23)
UTILITY_GATE = 0.60
SEED13_REFERENCE = (
    ROOT
    / "outputs/lee_eegnet_architecture_diagnostic/"
    "seed13_per_trial_channel__global_average.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "lee_eegnet_architecture_confirmation"


def _write_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# Lee Compact EEGNet Four-Seed Confirmation",
        "",
        "Exact selected setting: 250 Hz, per-trial channel normalization, global-average head.",
        "Privacy metrics are not computed in this confirmation stage.",
        "",
        "| Seed | Checkpoint epoch | Checkpoint val BA | Test BA | Macro F1 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in summary["results"]:
        lines.append(
            "| {seed} | {best_checkpoint_epoch} | "
            "{checkpoint_validation_balanced_accuracy:.4f} | "
            "{balanced_accuracy:.4f} | {macro_f1:.4f} |".format(**result)
        )
    lines.extend(
        [
            "",
            f"- Mean test balanced accuracy: `{summary['mean_balanced_accuracy']:.4f}`",
            f"- Sample SD: `{summary['sample_std_balanced_accuracy']:.4f}`",
            f"- Utility gate: `{summary['utility_gate']:.2f}`",
            f"- Gate passed: `{str(summary['utility_gate_passed']).lower()}`",
            f"- Status: `{summary['status']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirm the selected compact Lee EEGNet on four task seeds."
    )
    parser.add_argument("--mne-data-dir", default="raw_data/mne_data")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    _configure_mne_data_dir(args.mne_data_dir, DATASET_KEY)
    results = []
    for seed in SEEDS:
        output_path = args.output_root / f"seed{seed}.json"
        if output_path.exists():
            result = json.loads(output_path.read_text(encoding="utf-8"))
            print(f"seed={seed} status=skipped_complete")
        elif args.dry_run:
            print(f"seed={seed} status=planned output={output_path}")
            continue
        elif seed == 13 and SEED13_REFERENCE.exists():
            result = json.loads(SEED13_REFERENCE.read_text(encoding="utf-8"))
            result["imported_reference"] = str(SEED13_REFERENCE.relative_to(ROOT))
            write_result_json(result, output_path)
            print("seed=13 status=imported_reference")
        else:
            print(f"seed={seed} status=training")
            artifacts = fit_eegnet(
                DATASET_KEY,
                protocol=PROTOCOL,
                seed=seed,
                subjects=SUBJECTS,
                epochs=60,
                batch_size=32,
                learning_rate=1e-3,
                validation_fraction=0.2,
                early_stopping_patience=15,
                resample_hz=250,
                normalization="per_trial_channel",
                classifier_head="global_average",
            )
            result = artifacts.result.to_dict()
            result.update(
                {
                    "subjects": SUBJECTS,
                    "model_parameters": sum(
                        parameter.numel()
                        for parameter in artifacts.model.parameters()
                    ),
                    "privacy_metrics_computed": False,
                }
            )
            write_result_json(result, output_path)
            print(
                f"seed={seed} status=complete "
                f"balanced_accuracy={result['balanced_accuracy']:.6f}"
            )
        results.append(result)

    if args.dry_run:
        return
    balanced_accuracies = [result["balanced_accuracy"] for result in results]
    mean_balanced_accuracy = mean(balanced_accuracies)
    gate_passed = mean_balanced_accuracy >= UTILITY_GATE
    summary = {
        "dataset_key": DATASET_KEY,
        "protocol": PROTOCOL,
        "subjects": SUBJECTS,
        "seeds": list(SEEDS),
        "resample_hz": 250,
        "normalization": "per_trial_channel",
        "classifier_head": "global_average",
        "learning_rate": 1e-3,
        "privacy_metrics_computed": False,
        "utility_gate": UTILITY_GATE,
        "mean_balanced_accuracy": mean_balanced_accuracy,
        "sample_std_balanced_accuracy": stdev(balanced_accuracies),
        "utility_gate_passed": gate_passed,
        "status": (
            "confirmed_ready_for_privacy_evaluation"
            if gate_passed
            else "confirmation_failed_stop"
        ),
        "results": results,
    }
    json_path = args.output_root / "lee_eegnet_architecture_confirmation.json"
    markdown_path = args.output_root / "lee_eegnet_architecture_confirmation.md"
    write_result_json(summary, json_path)
    _write_markdown(summary, markdown_path)
    print(f"status={summary['status']}")
    print(f"wrote_json={json_path.resolve()}")
    print(f"wrote_markdown={markdown_path.resolve()}")


if __name__ == "__main__":
    main()
