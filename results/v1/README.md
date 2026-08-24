# Curated Benchmark v1 Evidence

This directory contains aggregate, participant-anonymous evidence selected from
the frozen private benchmark checkpoint. It contains no raw EEG, trial-level
predictions, checkpoints, posterior caches, or sweep directories.

## Primary evidence

- [Benchmark summary](summaries/benchmark-summary.md)
- [Claim traceability](summaries/claim-traceability.md)
- [Claims audit](summaries/claims-audit.md)
- [Protocol audit](summaries/protocol-audit.md)
- [Statistical intervals](summaries/statistical-intervals.md)
- [BNCI/Lee validation](summaries/bnci-lee-validation.md)
- [PhysioNet cross-protocol holdout](summaries/physionet-cross-protocol-holdout.md)
- [Federated and secure-aggregation summary](summaries/federated-phase45.md)

## Figures and tables

- [Seed-paired deltas](figures/seed-stability-deltas.png)
- [Attack-family split AUC](figures/attack-family-split-auc.png)
- [CSP anchor AUC](figures/csp-anchor-auc.png)
- [Core findings table](tables/core-findings.tex)
- [Systems diagnostics table](tables/systems-diagnostics.tex)

PhysioNet federation privacy differences are diagnostic because utility failed
the frozen 0.60 gate. Secure aggregation is a functional-only simulator with a
217.77x measured local runtime ratio and no output-level privacy protection.

Curated figures, tables, and report assets are licensed under CC BY 4.0.
Checksums are in `release/SHA256SUMS`.
