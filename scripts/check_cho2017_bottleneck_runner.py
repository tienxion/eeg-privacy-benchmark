from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from eeg_privacy_benchmark.models.eegnet import fit_eegnet


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_cho2017_confirmatory_bottleneck.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_cho2017_confirmatory_bottleneck_under_test",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load bottleneck runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture_main(runner, arguments: list[str]) -> str:
    output = io.StringIO()
    with patch.object(sys, "argv", [str(RUNNER_PATH), *arguments]), redirect_stdout(output):
        runner.main()
    return output.getvalue()


def _check(name: str, condition: bool, checks: list[tuple[str, bool]]) -> None:
    checks.append((name, bool(condition)))


def main() -> None:
    runner = _load_runner()
    checks: list[tuple[str, bool]] = []
    baseline = runner._baseline_runner()
    gate = runner._load_gate(baseline)

    _check(
        "tracked_gate_authorizes_only_frozen_bottleneck",
        gate["status"] == "PASS"
        and gate["plain_eegnet_utility_gate"]["next_stage"]
        == "compact_bottleneck_eegnet_dim6"
        and runner.MODEL == "compact_bottleneck_eegnet_dim6"
        and runner.BOTTLENECK_DIM == 6,
        checks,
    )
    fit_signature = inspect.signature(fit_eegnet)
    _check(
        "model_api_accepts_frozen_bottleneck_inputs",
        all(
            parameter in fit_signature.parameters
            for parameter in (
                "bottleneck_dim",
                "frequency_band_hz",
                "epoch_seconds",
                "explicit_trial_split",
                "loaded_dataset",
            )
        ),
        checks,
    )
    source = inspect.getsource(runner._run_job)
    _check(
        "runner_freezes_dim6_and_preloaded_data_path",
        "bottleneck_dim=BOTTLENECK_DIM" in source
        and "loaded_dataset=loaded" in source
        and 'task_model=MODEL' in source,
        checks,
    )
    complete_source = RUNNER_PATH.read_text(encoding="utf-8")
    _check(
        "runner_never_executes_privacy_attacks",
        "run_membership_inference" not in complete_source
        and "run_subject_identification" not in complete_source
        and '"attacks_authorized": False' in complete_source,
        checks,
    )

    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary) / "bottleneck"
        with patch.object(
            runner,
            "_run_job",
            side_effect=AssertionError("dry run attempted training"),
        ):
            dry_output = _capture_main(runner, ["--output-dir", str(output_dir)])
        dry_plan = json.loads(dry_output.split("\nDRY RUN", 1)[0])
        _check(
            "default_dry_run_is_full_frozen_plan",
            dry_plan["status"] == "DRY_RUN"
            and dry_plan["network_downloads"] is False
            and dry_plan["model"] == runner.MODEL
            and dry_plan["bottleneck_dim"] == 6
            and dry_plan["folds"] == [1, 2, 3, 4, 5]
            and dry_plan["seeds"] == [13, 17, 19, 23]
            and dry_plan["total_jobs"] == 20
            and dry_plan["attacks_authorized"] is False,
            checks,
        )
        _check("dry_run_does_not_write", not output_dir.exists(), checks)

        calls: list[tuple[int, int]] = []

        def fake_run_job(*, fold, seed, **kwargs):
            del kwargs
            calls.append((fold.fold_index, seed))
            return {
                "status": "complete",
                "model": runner.MODEL,
                "fold_index": fold.fold_index,
                "seed": seed,
                "task_result": {"balanced_accuracy": 0.75},
            }

        def fake_valid(path, **kwargs):
            del kwargs
            return path.is_file()

        confirmed_args = [
            "--output-dir",
            str(output_dir),
            "--confirm-run",
            "--folds",
            "1",
            "--seeds",
            "13,17",
            "--max-jobs",
            "1",
        ]
        with (
            patch.object(runner, "_baseline_runner", return_value=baseline),
            patch.object(runner, "_verify_local_baseline_gate", return_value=gate),
            patch.object(baseline, "_verify_local_cache"),
            patch.object(baseline, "_configure_cache"),
            patch.object(runner, "_valid_completed_job", side_effect=fake_valid),
            patch.object(runner, "_run_job", side_effect=fake_run_job),
        ):
            _capture_main(runner, confirmed_args)
            _capture_main(runner, confirmed_args)
        _check(
            "confirmed_runner_is_bounded_and_resumable",
            calls == [(1, 13), (1, 17)]
            and len(list((output_dir / "results").glob("*.json"))) == 2,
            checks,
        )

    gate_failure_stops_before_cache = False
    with tempfile.TemporaryDirectory() as temporary:
        with (
            patch.object(runner, "_baseline_runner", return_value=baseline),
            patch.object(
                runner,
                "_verify_local_baseline_gate",
                side_effect=RuntimeError("gate mismatch"),
            ),
            patch.object(
                baseline,
                "_verify_local_cache",
                side_effect=AssertionError("cache checked after gate failure"),
            ),
        ):
            try:
                _capture_main(
                    runner,
                    [
                        "--output-dir",
                        str(Path(temporary) / "bottleneck"),
                        "--confirm-run",
                        "--folds",
                        "1",
                        "--seeds",
                        "13",
                    ],
                )
            except RuntimeError as exc:
                gate_failure_stops_before_cache = str(exc) == "gate mismatch"
    _check("confirmed_run_fails_closed_on_gate_mismatch", gate_failure_stops_before_cache, checks)

    _check(
        "missing_output_is_not_resumable",
        not runner._valid_completed_job(
            Path("does-not-exist.json"),
            fold=baseline.build_outer_folds(
                baseline._load_sources()[0]["cohort"]["shuffled_subject_blocks"]
            )[0],
            seed=13,
            output_dir=Path("unused"),
            expected_role_counts={},
            attacker_partition=None,
            runner=baseline,
        ),
        checks,
    )
    _check(
        "bottleneck_outputs_are_isolated_from_baselines",
        runner.DEFAULT_OUTPUT != runner.BASELINE_OUTPUT
        and "bottleneck" in runner.DEFAULT_OUTPUT.name,
        checks,
    )

    for name, passed in checks:
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    passed_count = sum(passed for _, passed in checks)
    print(f"passed={passed_count}/{len(checks)}")
    if passed_count != len(checks):
        raise AssertionError("Cho2017 bottleneck runner check failed")


if __name__ == "__main__":
    main()
