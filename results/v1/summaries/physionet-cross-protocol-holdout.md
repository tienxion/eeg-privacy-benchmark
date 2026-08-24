# PhysioNet Cross-Protocol Holdout Results

This report summarizes completed rows from the `cross_protocol_physionet_holdout` plan. Lower membership-inference AUC is better for privacy. Deltas are computed versus the same-attack plain EEGNet baseline when available.

The table below is the frozen public record for this protocol.

## Completion

- Completed rows: `11/11`.
- Conclusion: threshold-only or attack-specific privacy wins; PhysioNet remains unresolved.

## Rows

| Ordinal | Candidate | Attack | Status | Task BA | Attack AUC | Delta task | Delta AUC | Decision |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `1` | `eegnet_baseline` | `mlp::posterior_probabilities` | `complete` | `0.5229` | `0.5184` | `+0.0000` | `+0.0000` | baseline |
| `2` | `mixup_alpha_0p20` | `mlp::posterior_probabilities` | `complete` | `0.5043` | `0.5109` | `-0.0185` | `-0.0075` | mixed |
| `3` | `bottleneck_dim_6` | `mlp::posterior_probabilities` | `complete` | `0.5366` | `0.5350` | `+0.0137` | `+0.0166` | privacy regression |
| `4` | `eegnet_baseline` | `threshold::max_probability` | `complete` | `0.5229` | `0.5174` | `+0.0000` | `+0.0000` | baseline |
| `5` | `mixup_alpha_0p20` | `threshold::max_probability` | `complete` | `0.5043` | `0.5008` | `-0.0185` | `-0.0166` | family win within utility |
| `6` | `bottleneck_dim_6` | `threshold::max_probability` | `complete` | `0.5366` | `0.5210` | `+0.0137` | `+0.0036` | practical tie |
| `7` | `csp_lda_anchor` | `threshold::max_probability` | `complete` | `0.6182` | `0.5036` | `+0.0953` | `-0.0138` | family win within utility |
| `8` | `eegnet_baseline` | `threshold::label_known_log_probability` | `complete` | `0.5229` | `0.5750` | `+0.0000` | `+0.0000` | baseline |
| `9` | `mixup_alpha_0p20` | `threshold::label_known_log_probability` | `complete` | `0.5043` | `0.5603` | `-0.0185` | `-0.0147` | family win within utility |
| `10` | `bottleneck_dim_6` | `threshold::label_known_log_probability` | `complete` | `0.5366` | `0.5665` | `+0.0137` | `-0.0085` | mixed |
| `11` | `csp_lda_anchor` | `threshold::label_known_log_probability` | `complete` | `0.6182` | `0.5196` | `+0.0953` | `-0.0555` | family win within utility |

## Interpretation Boundary

The completed cross-run rows do not satisfy the pre-registered two-family privacy and utility rule. They therefore preserve the unresolved, protocol-sensitive PhysioNet conclusion.
