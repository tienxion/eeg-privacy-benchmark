# Cho2017 Confirmatory Membership-Privacy Gate

## Status

The preregistered membership-privacy promotion gate did not pass.

All 600 frozen cache-only attack jobs validated. This record performed no EEG
loading or task-model training. The compact bottleneck remains utility
noninferior, but the confirmatory data do not support promoting it as a
membership-privacy improvement over plain compact EEGNet.

## Subject-paired result

Positive reduction means lower bottleneck attack AUC. Frozen score variants,
task seeds, eligible folds, and attacker seeds were aggregated within each
attacker-evaluation subject before inference.

| Attacker family | Plain AUC | Bottleneck AUC | Mean AUC reduction | Holm rank-adjusted interval | Positive on 3/3 attacker seeds | Promotion criterion |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| threshold | 0.5168 | 0.5168 | -0.00003 | [-0.0042, 0.0044] | no | NO |
| logistic regression | 0.4962 | 0.4962 | 0.00002 | [-0.0053, 0.0051] | yes | NO |
| mlp | 0.5035 | 0.5021 | 0.00140 | [-0.0014, 0.0042] | yes | NO |

Promotion required a mean AUC reduction of at least `0.010`, multiplicity-
adjusted interval support above zero, consistent positive direction on all
three attacker seeds, and no family regression above `0.010` in at least two
of the three families. `0/3` families passed, so the decision is
**NO PROMOTION**.

## Inference and scope

The inferential sample contains 46 subjects
assigned to the held-out attacker-evaluation role in at least one
eligible outer fold. Confirmatory subjects 6, 13, 17, 40 were never assigned to
that role by the frozen per-fold partitions; this is a consequence of the
preregistered roles, not a post-result exclusion.

Threshold and logistic-regression score variants were equally weighted within
their family so variants did not become extra promotion opportunities. The
three family tests used Holm step-down multiplicity control. The displayed
rank-adjusted percentile intervals use the corresponding Holm critical level;
the machine-readable artifact also records unadjusted 95% intervals, adjusted
p-values, and attacker-seed diagnostics.

This negative promotion result is not evidence that membership leakage is
absent or that the two models are equivalent. It means only that the frozen
bottleneck did not satisfy the preregistered improvement rule.

## Reproducibility record

The tracked machine-readable artifact is
[`configs/cho2017_confirmatory_privacy_gate_v1.json`](../configs/cho2017_confirmatory_privacy_gate_v1.json).
It contains subject aggregates and hashes for all 600 result files, but no raw
EEG or posterior arrays.

After reproducing the local attack queue, verify this record with:

```bash
PYTHONPATH=src .venv/bin/python scripts/freeze_cho2017_privacy_gate.py --check
```
