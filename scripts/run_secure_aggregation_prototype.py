from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import gc
import json
from pathlib import Path
from statistics import median
import time

from eeg_privacy_benchmark.models.eegnet_federated import fedavg_state_dicts
from eeg_privacy_benchmark.models.secure_aggregation import (
    simulate_pairwise_masked_fedavg,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT
    / "outputs"
    / "federated_physionet_multiseed"
    / "seed13"
    / "federated_eegnet_checkpoint.pt"
)
DEFAULT_TASK = (
    ROOT
    / "outputs"
    / "federated_physionet_multiseed"
    / "seed13"
    / "federated_eegnet_task.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "secure_aggregation_prototype"


def _synthetic_client_states(torch, base_state: dict, weights: list[int]) -> list:
    generator = torch.Generator(device="cpu").manual_seed(20260824)
    coefficients = [
        float(torch.randn((), generator=generator, dtype=torch.float64).item())
        for _ in weights[:-1]
    ]
    weighted_sum = sum(weight * value for weight, value in zip(weights[:-1], coefficients))
    coefficients.append(-weighted_sum / weights[-1])

    states = [{key: value.clone() for key, value in base_state.items()} for _ in weights]
    for key, base in base_state.items():
        if not (torch.is_floating_point(base) or torch.is_complex(base)):
            continue
        if torch.is_complex(base):
            real = torch.randn(base.shape, generator=generator, dtype=torch.float64)
            imag = torch.randn(base.shape, generator=generator, dtype=torch.float64)
            pattern = torch.complex(real, imag).to(dtype=base.dtype)
        else:
            pattern = torch.randn(base.shape, generator=generator, dtype=base.dtype)
        for state, coefficient in zip(states, coefficients):
            state[key].add_(pattern, alpha=coefficient * 1e-4)
    return list(zip(weights, states))


def _max_abs_error(torch, first: dict, second: dict) -> float:
    errors = []
    for key in first:
        if torch.is_floating_point(first[key]) or torch.is_complex(first[key]):
            errors.append(float((first[key] - second[key]).abs().max().item()))
        elif not torch.equal(first[key], second[key]):
            return float("inf")
    return max(errors, default=0.0)


def _benchmark(callable_, repeats: int) -> list[float]:
    durations = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        callable_()
        durations.append(time.perf_counter() - started)
    return durations


def _write_report(payload: dict, output_root: Path) -> tuple[Path, Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "secure_aggregation_prototype.json"
    csv_path = output_root / "secure_aggregation_prototype_metrics.csv"
    md_path = output_root / "secure_aggregation_prototype.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    metric_rows = [
        {"metric": key, "value": value}
        for key, value in payload["metrics"].items()
        if not isinstance(value, (dict, list))
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(metric_rows)

    metrics = payload["metrics"]
    lines = [
        "# Secure Aggregation Functional Prototype",
        "",
        "Decision: `go_limited_functional_prototype`.",
        "",
        "The simulator applies canceling pairwise masks to sample-weighted client model states and exposes only their aggregate. It is isolated from training and uses the completed PhysioNet seed-13 checkpoint shape with 36 synthetic client updates whose weighted reference aggregate preserves that checkpoint.",
        "",
        "## Verification",
        "",
        f"- Aggregate matches plain FedAvg: `{payload['verification']['aggregate_matches_reference']}`.",
        f"- Maximum parameter error versus FedAvg: `{metrics['max_abs_error_vs_fedavg']:.3e}`.",
        f"- Maximum parameter error versus the preserved checkpoint: `{metrics['max_abs_error_vs_checkpoint']:.3e}`.",
        f"- Pairwise masks: `{metrics['pair_count']}` pairs and `{metrics['mask_applications']}` tensor-mask applications.",
        "",
        "## Overhead",
        "",
        f"- Plain FedAvg median aggregation runtime: `{metrics['plain_median_runtime_seconds']:.6f}` seconds.",
        f"- Masked median aggregation runtime: `{metrics['secure_median_runtime_seconds']:.6f}` seconds.",
        f"- Runtime ratio: `{metrics['runtime_ratio']:.2f}x`.",
        f"- Per-round model payload overhead: `{metrics['per_round_payload_overhead_bytes']}` bytes.",
        f"- Estimated one-time pairwise-seed setup communication: `{metrics['estimated_seed_setup_communication_bytes']}` bytes.",
        f"- Full 80-round communication increase: `{metrics['full_run_communication_increase_percent']:.4f}%`.",
        "",
        "## Threat Model",
        "",
        "The intended protection is confidentiality of each floating-point client update from an honest-but-curious aggregator when every client participates, pairwise secrets remain unknown to the aggregator, and clients do not collude.",
        "",
        "This prototype does not handle dropout recovery, malicious clients, collusion, authentication, poisoning, metadata leakage, global-model or output leakage, membership inference, or differential privacy. The deterministic local PRG and centralized simulator are not production cryptography. Non-floating model state follows the existing public largest-client FedAvg rule.",
        "",
        "## Claim Boundary",
        "",
        "The prototype establishes functional aggregation equivalence and a local overhead estimate only. It does not convert federation into a privacy guarantee and does not change the Phase 4 utility-gate failure.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, csv_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the secure-aggregation prototype.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    task = json.loads(args.task.read_text(encoding="utf-8"))
    base_state = checkpoint["model_state"]
    weights = list(task["client_trial_counts"].values())
    weighted_states = _synthetic_client_states(torch, base_state, weights)

    reference = fedavg_state_dicts(torch, weighted_states)
    secure, metadata = simulate_pairwise_masked_fedavg(
        torch,
        weighted_states,
        master_seed=20260824,
        round_nonce=1,
    )
    error_vs_reference = _max_abs_error(torch, secure, reference)
    error_vs_checkpoint = _max_abs_error(torch, secure, base_state)
    tolerance = 1e-6
    aggregate_matches = error_vs_reference <= tolerance
    checkpoint_preserved = error_vs_checkpoint <= tolerance
    if not aggregate_matches or not checkpoint_preserved:
        raise RuntimeError(
            "Secure aggregate failed equivalence: "
            f"reference_error={error_vs_reference} checkpoint_error={error_vs_checkpoint}"
        )

    plain_durations = _benchmark(
        lambda: fedavg_state_dicts(torch, weighted_states),
        args.repeats,
    )
    secure_durations = _benchmark(
        lambda: simulate_pairwise_masked_fedavg(
            torch,
            weighted_states,
            master_seed=20260824,
            round_nonce=1,
        ),
        args.repeats,
    )
    plain_median = median(plain_durations)
    secure_median = median(secure_durations)
    baseline_communication = int(task["estimated_communication_bytes"])
    secure_communication = (
        baseline_communication + metadata.estimated_seed_setup_communication_bytes
    )

    payload = {
        "schema_version": 1,
        "decision": "go_limited_functional_prototype",
        "reference": {
            "checkpoint": str(args.checkpoint.relative_to(ROOT)),
            "task": str(args.task.relative_to(ROOT)),
            "dataset": task["dataset_key"],
            "seed": task["seed"],
            "clients": task["clients"],
            "rounds": task["rounds_trained"],
        },
        "verification": {
            "aggregate_matches_reference": aggregate_matches,
            "checkpoint_preserved": checkpoint_preserved,
            "absolute_tolerance": tolerance,
            "relative_tolerance": tolerance,
            "reference_aggregation": "fedavg_state_dicts",
        },
        "metrics": {
            **asdict(metadata),
            "max_abs_error_vs_fedavg": error_vs_reference,
            "max_abs_error_vs_checkpoint": error_vs_checkpoint,
            "plain_median_runtime_seconds": plain_median,
            "secure_median_runtime_seconds": secure_median,
            "runtime_ratio": secure_median / plain_median,
            "baseline_full_run_communication_bytes": baseline_communication,
            "secure_full_run_communication_bytes": secure_communication,
            "full_run_communication_increase_percent": (
                (secure_communication / baseline_communication - 1.0) * 100.0
            ),
            "benchmark_repeats": args.repeats,
        },
        "threat_model": {
            "protects": "individual floating-point updates from an honest-but-curious aggregator",
            "assumptions": [
                "all clients are present",
                "pairwise secrets are unknown to the aggregator",
                "clients do not collude",
            ],
            "does_not_protect": [
                "dropout recovery",
                "malicious clients or collusion",
                "authentication or poisoning",
                "metadata",
                "global-model or output leakage",
                "membership inference",
                "differential privacy",
            ],
            "implementation_caveat": "functional centralized simulator; deterministic PRG is not production cryptography",
        },
        "claim": "functional_equivalence_and_local_overhead_only",
    }
    paths = _write_report(payload, args.output_root)
    print(f"aggregate_matches_reference={aggregate_matches}")
    print(f"max_abs_error={error_vs_reference:.3e}")
    print(f"runtime_ratio={payload['metrics']['runtime_ratio']:.2f}")
    for path in paths:
        print(f"wrote={path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
