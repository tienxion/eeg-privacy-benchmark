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
SEED = 13
UTILITY_GATE = 0.60
REFERENCE_PATH = (
    ROOT
    / "outputs/lee_eegnet_resample250_diagnostic/"
    "seed13_resample250_lr1em03.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "lee_eegnet_architecture_diagnostic"
CANDIDATES = (
    ("global_train_channel", "flatten"),
    ("per_trial_channel", "flatten"),
    ("global_train_channel", "global_average"),
    ("per_trial_channel", "global_average"),
)


def _candidate_name(normalization: str, classifier_head: str) -> str:
    return f"{normalization}__{classifier_head}"


def select_candidate(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("At least one completed candidate is required.")
    order = {
        _candidate_name(normalization, head): index
        for index, (normalization, head) in enumerate(CANDIDATES)
    }
    return max(
        candidates,
        key=lambda candidate: (
            candidate["checkpoint_validation_balanced_accuracy"],
            candidate["classifier_head"] == "global_average",
            -order[candidate["candidate"]],
        ),
    )


def _write_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# Lee EEGNet Normalization and Head Diagnostic",
        "",
        "All candidates use 250 Hz inputs. Selection is validation-only and no privacy metrics are computed.",
        "",
        "| Normalization | Head | Parameters | Checkpoint epoch | Checkpoint val BA | Test BA |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for candidate in summary["candidates"]:
        lines.append(
            "| {normalization} | {classifier_head} | {model_parameters} | "
            "{best_checkpoint_epoch} | "
            "{checkpoint_validation_balanced_accuracy:.4f} | "
            "{balanced_accuracy:.4f} |".format(**candidate)
        )
    lines.extend(
        [
            "",
            f"- Selected candidate: `{summary['selected_candidate']}`",
            f"- Selected test balanced accuracy: `{summary['selected_test_balanced_accuracy']:.4f}`",
            f"- Utility gate: `{summary['utility_gate']:.2f}`",
            f"- Gate passed: `{str(summary['utility_gate_passed']).lower()}`",
            f"- Status: `{summary['status']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parameter_count(artifacts) -> int:
    return sum(parameter.numel() for parameter in artifacts.model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Lee EEGNet normalization/head diagnostic."
    )
    parser.add_argument("--mne-data-dir", default="raw_data/mne_data")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    _configure_mne_data_dir(args.mne_data_dir, DATASET_KEY)
    results = []
    for normalization, classifier_head in CANDIDATES:
        candidate_name = _candidate_name(normalization, classifier_head)
        output_path = args.output_root / f"seed13_{candidate_name}.json"
        if output_path.exists() and not args.force:
            result = json.loads(output_path.read_text(encoding="utf-8"))
            print(f"candidate={candidate_name} status=skipped_complete")
        elif args.dry_run:
            print(f"candidate={candidate_name} status=planned output={output_path}")
            continue
        elif (
            normalization == "global_train_channel"
            and classifier_head == "flatten"
            and REFERENCE_PATH.exists()
            and not args.force
        ):
            result = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
            result.update(
                {
                    "candidate": candidate_name,
                    "normalization": normalization,
                    "classifier_head": classifier_head,
                    "model_parameters": 3090,
                    "imported_reference": str(REFERENCE_PATH.relative_to(ROOT)),
                }
            )
            write_result_json(result, output_path)
            print(f"candidate={candidate_name} status=imported_reference")
        else:
            print(f"candidate={candidate_name} status=training")
            artifacts = fit_eegnet(
                DATASET_KEY,
                protocol=PROTOCOL,
                seed=SEED,
                subjects=SUBJECTS,
                epochs=60,
                batch_size=32,
                learning_rate=1e-3,
                validation_fraction=0.2,
                early_stopping_patience=15,
                resample_hz=250,
                normalization=normalization,
                classifier_head=classifier_head,
            )
            result = artifacts.result.to_dict()
            result.update(
                {
                    "candidate": candidate_name,
                    "subjects": SUBJECTS,
                    "model_parameters": _parameter_count(artifacts),
                    "selection_scope": "validation_only",
                    "privacy_metrics_computed": False,
                }
            )
            write_result_json(result, output_path)
            print(
                f"candidate={candidate_name} status=complete "
                "checkpoint_validation_balanced_accuracy="
                f"{result['checkpoint_validation_balanced_accuracy']:.6f}"
            )
        results.append(result)

    if args.dry_run:
        return
    selected = select_candidate(results)
    gate_passed = selected["balanced_accuracy"] >= UTILITY_GATE
    summary = {
        "dataset_key": DATASET_KEY,
        "protocol": PROTOCOL,
        "subjects": SUBJECTS,
        "seed": SEED,
        "resample_hz": 250,
        "selection_source": "validation_only",
        "checkpoint_selection_metric": "minimum_validation_loss",
        "candidate_selection_metric": "checkpoint_validation_balanced_accuracy",
        "test_metrics_used_for_selection": False,
        "privacy_metrics_computed": False,
        "utility_gate": UTILITY_GATE,
        "selected_candidate": selected["candidate"],
        "selected_test_balanced_accuracy": selected["balanced_accuracy"],
        "utility_gate_passed": gate_passed,
        "status": (
            "gate_passed_ready_for_confirmation"
            if gate_passed
            else "gate_failed_stop_eegnet_architecture_family"
        ),
        "candidates": results,
    }
    json_path = args.output_root / "lee_eegnet_architecture_diagnostic.json"
    markdown_path = args.output_root / "lee_eegnet_architecture_diagnostic.md"
    write_result_json(summary, json_path)
    _write_markdown(summary, markdown_path)
    print(f"status={summary['status']}")
    print(f"wrote_json={json_path.resolve()}")
    print(f"wrote_markdown={markdown_path.resolve()}")


if __name__ == "__main__":
    main()
