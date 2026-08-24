from __future__ import annotations

import json
from pathlib import Path

from eeg_privacy_benchmark.models.eegnet_federated import fedavg_state_dicts
from eeg_privacy_benchmark.models.secure_aggregation import (
    simulate_pairwise_masked_fedavg,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "v1" / "summaries" / "secure-aggregation-prototype.json"


def _expect_value_error(callable_, phrase: str) -> None:
    try:
        callable_()
    except ValueError as error:
        assert phrase in str(error)
    else:
        raise AssertionError(f"Expected ValueError containing {phrase!r}")


def main() -> None:
    import torch

    first = {
        "weight": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "counter": torch.tensor(3, dtype=torch.int64),
    }
    second = {
        "weight": torch.tensor([5.0, 6.0], dtype=torch.float32),
        "counter": torch.tensor(7, dtype=torch.int64),
    }
    weighted = [(1, first), (3, second)]
    reference = fedavg_state_dicts(torch, weighted)
    secure, metadata = simulate_pairwise_masked_fedavg(
        torch,
        weighted,
        master_seed=11,
        round_nonce=2,
    )
    torch.testing.assert_close(secure["weight"], reference["weight"], rtol=1e-6, atol=1e-6)
    assert torch.equal(secure["counter"], reference["counter"])
    assert metadata.client_count == 2
    assert metadata.pair_count == 1
    assert metadata.mask_applications == 1
    assert metadata.unique_pairwise_seed_bytes == 32
    assert metadata.estimated_seed_setup_communication_bytes == 64
    assert metadata.per_round_payload_overhead_bytes == 0
    assert metadata.max_abs_mask > 0.0
    print("PASS masked_fedavg_equivalence")

    _expect_value_error(
        lambda: simulate_pairwise_masked_fedavg(torch, []),
        "at least one client",
    )
    _expect_value_error(
        lambda: simulate_pairwise_masked_fedavg(torch, [(0, first)]),
        "positive",
    )
    _expect_value_error(
        lambda: simulate_pairwise_masked_fedavg(
            torch,
            [(1, first), (1, {"other": second["weight"]})],
        ),
        "different keys",
    )
    _expect_value_error(
        lambda: simulate_pairwise_masked_fedavg(
            torch,
            [(1, first), (1, {"weight": torch.ones(3), "counter": second["counter"]})],
        ),
        "shape mismatch",
    )
    _expect_value_error(
        lambda: simulate_pairwise_masked_fedavg(torch, weighted, mask_scale=0.0),
        "mask_scale",
    )
    print("PASS secure_aggregation_input_guards")

    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    assert payload["decision"] == "go_limited_functional_prototype"
    assert payload["verification"]["aggregate_matches_reference"] is True
    assert payload["verification"]["checkpoint_preserved"] is True
    assert metrics["client_count"] == 36
    assert metrics["pair_count"] == 630
    assert metrics["max_abs_error_vs_fedavg"] <= 1e-6
    assert metrics["max_abs_error_vs_checkpoint"] <= 1e-6
    assert metrics["per_round_payload_overhead_bytes"] == 0
    assert metrics["estimated_seed_setup_communication_bytes"] == 40320
    assert metrics["full_run_communication_increase_percent"] > 0.0
    assert metrics["runtime_ratio"] > 0.0
    assert payload["claim"] == "functional_equivalence_and_local_overhead_only"
    assert "membership inference" in payload["threat_model"]["does_not_protect"]
    print("PASS secure_aggregation_report")
    print("PASS secure aggregation checks=3")


if __name__ == "__main__":
    main()
