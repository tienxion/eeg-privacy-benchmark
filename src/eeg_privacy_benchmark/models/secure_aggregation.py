"""Functional pairwise-mask simulation for sample-weighted FedAvg."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib


PAIRWISE_SEED_BYTES = 32


@dataclass(frozen=True)
class SecureAggregationMetadata:
    """Measured protocol-shape metadata from one simulated aggregation round."""

    client_count: int
    pair_count: int
    floating_tensor_count: int
    nonfloating_tensor_count: int
    mask_applications: int
    unique_pairwise_seed_bytes: int
    estimated_seed_setup_communication_bytes: int
    client_update_payload_bytes: int
    per_round_payload_overhead_bytes: int
    max_abs_mask: float


def _validate_weighted_states(torch, weighted_states: list[tuple[int, dict]]) -> tuple:
    if not weighted_states:
        raise ValueError("Secure aggregation requires at least one client state.")
    if any(weight <= 0 for weight, _ in weighted_states):
        raise ValueError("Secure aggregation client weights must be positive.")

    reference_keys = tuple(weighted_states[0][1])
    for _, state in weighted_states[1:]:
        if tuple(state) != reference_keys:
            raise ValueError(
                "Secure aggregation client state dictionaries have different keys."
            )
    for key in reference_keys:
        tensors = [state[key] for _, state in weighted_states]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError(f"Secure aggregation tensor shape mismatch for {key}.")
        if any(tensor.dtype != tensors[0].dtype for tensor in tensors[1:]):
            raise ValueError(f"Secure aggregation tensor dtype mismatch for {key}.")
    return reference_keys


def _pairwise_seed(master_seed: int, first: int, second: int) -> bytes:
    payload = f"eeg-secagg-v1:{master_seed}:{first}:{second}".encode("ascii")
    return hashlib.sha256(payload).digest()


def _generator_seed(pair_seed: bytes, key: str, component: str) -> int:
    digest = hashlib.sha256(
        pair_seed + b":" + key.encode("utf-8") + b":" + component.encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="little") % (2**63 - 1)


def _pairwise_mask(torch, tensor, pair_seed: bytes, key: str, mask_scale: float):
    working_dtype = torch.complex128 if torch.is_complex(tensor) else torch.float64
    real_generator = torch.Generator(device="cpu")
    real_generator.manual_seed(_generator_seed(pair_seed, key, "real"))
    real = torch.randn(tensor.shape, generator=real_generator, dtype=torch.float64)
    if torch.is_complex(tensor):
        imag_generator = torch.Generator(device="cpu")
        imag_generator.manual_seed(_generator_seed(pair_seed, key, "imag"))
        imag = torch.randn(tensor.shape, generator=imag_generator, dtype=torch.float64)
        mask = torch.complex(real, imag)
    else:
        mask = real
    return mask.to(device=tensor.device, dtype=working_dtype) * mask_scale


def simulate_pairwise_masked_fedavg(
    torch,
    weighted_states: list[tuple[int, dict]],
    *,
    master_seed: int = 0,
    round_nonce: int = 0,
    mask_scale: float = 1.0,
) -> tuple[dict, SecureAggregationMetadata]:
    """Simulate all-client pairwise masking and return only the aggregate.

    This is a functional, single-process prototype. Pairwise masks cancel in the
    server sum, but the deterministic local PRG is not a production cryptographic
    protocol and the simulator itself can reconstruct every mask.
    """

    if mask_scale <= 0.0:
        raise ValueError("Secure aggregation mask_scale must be positive.")
    reference_keys = _validate_weighted_states(torch, weighted_states)
    client_count = len(weighted_states)
    pair_count = client_count * (client_count - 1) // 2
    total_weight = sum(weight for weight, _ in weighted_states)
    largest_client_state = max(
        enumerate(weighted_states),
        key=lambda item: (item[1][0], -item[0]),
    )[1][1]

    aggregate = {}
    floating_tensor_count = 0
    nonfloating_tensor_count = 0
    max_abs_mask = 0.0
    client_update_payload_bytes = sum(
        value.numel() * value.element_size()
        for _, state in weighted_states
        for value in state.values()
    )

    for key in reference_keys:
        tensors = [state[key] for _, state in weighted_states]
        if not (torch.is_floating_point(tensors[0]) or torch.is_complex(tensors[0])):
            nonfloating_tensor_count += 1
            aggregate[key] = largest_client_state[key].clone()
            continue

        floating_tensor_count += 1
        working_dtype = torch.complex128 if torch.is_complex(tensors[0]) else torch.float64
        masked_contributions = [
            state[key].to(dtype=working_dtype) * (weight / total_weight)
            for weight, state in weighted_states
        ]
        for first in range(client_count):
            for second in range(first + 1, client_count):
                seed = _pairwise_seed(master_seed ^ round_nonce, first, second)
                mask = _pairwise_mask(torch, tensors[0], seed, key, mask_scale)
                masked_contributions[first].add_(mask)
                masked_contributions[second].sub_(mask)
                max_abs_mask = max(max_abs_mask, float(mask.abs().max().item()))

        value = torch.zeros_like(masked_contributions[0])
        for contribution in masked_contributions:
            value.add_(contribution)
        aggregate[key] = value.to(dtype=tensors[0].dtype)

    unique_seed_bytes = pair_count * PAIRWISE_SEED_BYTES
    metadata = SecureAggregationMetadata(
        client_count=client_count,
        pair_count=pair_count,
        floating_tensor_count=floating_tensor_count,
        nonfloating_tensor_count=nonfloating_tensor_count,
        mask_applications=pair_count * floating_tensor_count,
        unique_pairwise_seed_bytes=unique_seed_bytes,
        estimated_seed_setup_communication_bytes=2 * unique_seed_bytes,
        client_update_payload_bytes=client_update_payload_bytes,
        per_round_payload_overhead_bytes=0,
        max_abs_mask=max_abs_mask,
    )
    return aggregate, metadata
