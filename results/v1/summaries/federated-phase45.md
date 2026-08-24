# Federated Phase 4 and Secure-Aggregation Phase 5 Summary

## Phase 4: Frozen PhysioNet Replication

The four-seed, subjects 1-46, cross-subject PhysioNet replication is complete under 80 all-client FedAvg rounds. Mean task balanced accuracy falls from `0.6574` to `0.5185` (paired delta `-0.1389`), so the frozen 0.60 utility gate fails.

Membership AUC paired deltas are `-0.1009` label-known threshold, `-0.0714` max-probability threshold, `+0.0220` logistic posterior, and `-0.0583` MLP posterior. Fixed MLP attacker seeds give mean delta `-0.0427` with decision `refit_direction_stable_and_default_consistent`. All privacy differences are diagnostic only because task utility fails.

## Phase 5: Limited Secure-Aggregation Prototype

The pairwise-mask simulator matches plain FedAvg within `2.384e-07` and preserves the reference checkpoint within `1.819e-12`. It uses `630` client pairs and `11340` tensor-mask applications.

Median local aggregation runtime rises from `0.000464` to `0.101083` seconds (`217.77x`). Per-round model payload overhead is `0` bytes; estimated one-time pairwise-seed setup is `40320` bytes, a `0.0649%` increase over the measured 80-round run.

## Claim Boundary

Federation is not a privacy defense, and this functional prototype is not production cryptography. It assumes all clients are present, pairwise secrets are hidden from the aggregator, and no collusion. It does not address dropout recovery, malicious clients, authentication, poisoning, metadata, output leakage, membership inference, or differential privacy.
