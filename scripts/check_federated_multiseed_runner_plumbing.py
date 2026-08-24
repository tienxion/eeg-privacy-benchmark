from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_federated_bnci_multiseed as runner
from eeg_privacy_benchmark.privacy import (
    PosteriorFeatureCache,
    load_posterior_feature_cache,
    write_posterior_feature_cache,
)


def main() -> None:
    checks: list[tuple[str, bool]] = []
    checks.append(("compact_subject_range", runner._parse_subjects("1-3,5") == [1, 2, 3, 5]))
    try:
        runner._parse_subjects("3-1")
    except ValueError:
        checks.append(("descending_subject_range_rejected", True))
    else:
        checks.append(("descending_subject_range_rejected", False))

    with TemporaryDirectory() as directory:
        paths = runner._seed_paths(Path(directory), 13)
        paths["task"].parent.mkdir(parents=True)
        task = {
            "rounds_trained": 2,
            "local_epochs": 1,
            "balanced_accuracy": 0.6,
            "macro_f1": 0.59,
            "estimated_communication_bytes": 123,
            "history": [
                {"round": 1, "validation_loss": 0.7, "validation_balanced_accuracy": 0.5},
                {"round": 2, "validation_loss": 0.6, "validation_balanced_accuracy": 0.55},
            ],
        }
        paths["task"].write_text(json.dumps(task), encoding="utf-8")
        for label, _, _ in runner.ATTACKS:
            paths[label].write_text(json.dumps({"attack_auc": 0.5}), encoding="utf-8")
        write_posterior_feature_cache(
            PosteriorFeatureCache(
                task_model="federated_eegnet",
                dataset_key="physionet_motor_imagery",
                protocol="cross_subject",
                seed=13,
                subjects=[1, 2],
                task_balanced_accuracy=0.6,
                task_macro_f1=0.59,
                class_probabilities=np.asarray(
                    [[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]]
                ),
                encoded_labels=np.asarray([0, 1, 0, 1]),
                member_indices=[0, 1],
                nonmember_indices=[2, 3],
                federated_rounds_trained=2,
                federated_local_epochs=1,
                federated_clients=2,
                estimated_communication_bytes=123,
            ),
            paths["posterior_cache"],
        )
        checks.append(
            (
                "cross_subject_components_complete",
                runner._component_outputs_complete(paths, include_subject_id=False),
            )
        )
        runner._write_suite_from_outputs(
            paths=paths,
            dataset_key="physionet_motor_imagery",
            protocol="cross_subject",
            subjects=[1, 2],
            seed=13,
            include_subject_id=False,
        )
        suite = json.loads(paths["suite"].read_text(encoding="utf-8"))
        task = json.loads(paths["task"].read_text(encoding="utf-8"))
        cache = load_posterior_feature_cache(paths["posterior_cache"])
        checks.append(
            (
                "cross_subject_suite_contract",
                suite["subject_id_accuracy"] is None
                and "subject_id" not in suite["outputs"],
            )
        )
        checks.append(
            (
                "legacy_provenance_upgraded",
                suite["metadata_schema_version"] == 2
                and task["best_checkpoint_round"] == 2
                and cache.federated_best_checkpoint_round == 2,
            )
        )
        checks.append(
            (
                "upgraded_suite_is_complete",
                runner._suite_complete(paths, include_subject_id=False),
            )
        )

    for name, passed in checks:
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    passed_count = sum(passed for _, passed in checks)
    print(f"passed={passed_count}/{len(checks)}")
    if passed_count != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
