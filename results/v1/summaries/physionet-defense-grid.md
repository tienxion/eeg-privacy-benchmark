# PhysioNet Defense Grid Results

This report applies the pre-registered PhysioNet defense-grid stop rule. Lower membership AUC is better; deltas compare each row to the same-attack EEGNet reference.

The aggregate values below are the frozen public record for this protocol.

## Completion

- Evaluated commands: `13/18`.
- Commands omitted by the pre-registered stop rule: `5/18`.

## Aggregate Read

| Candidate | Type | Model | Setting | Completed attacks | Mean delta AUC | Mean delta task BA | AUC-win families | Utility-safe rows | Decision |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `eegnet_baseline` | `reference` | `eegnet` | plain EEGNet | `mlp_posterior,threshold_max_probability,threshold_label_known` | +0.0000 | +0.0000 | 0 | 3 | `baseline` |
| `mixup_alpha_0p20_reference` | `reference` | `mixup_eegnet` | mixup alpha=0.20 | `mlp_posterior,threshold_max_probability,threshold_label_known` | -0.0086 | -0.0351 | 1 | 0 | `utility boundary failed` |
| `confidence_beta_0p025_reference` | `reference` | `confidence_penalty_eegnet` | confidence penalty beta=0.025 | `mlp_posterior,threshold_max_probability,threshold_label_known` | +0.0010 | -0.0034 | 0 | 3 | `no stable defense upgrade` |
| `mixup_alpha_0p10` | `candidate` | `mixup_eegnet` | mixup alpha=0.10 | `mlp_posterior,threshold_max_probability,threshold_label_known` | +0.0030 | -0.0110 | 1 | 3 | `attack-specific only` |
| `mixup_alpha_0p35` | `candidate` | `mixup_eegnet` | mixup alpha=0.35 | `mlp_posterior,threshold_max_probability` | +0.0043 | -0.0071 | 0 | 2 | `early stop: cannot reach two-family rule` |
| `confidence_beta_0p010` | `candidate` | `confidence_penalty_eegnet` | confidence penalty beta=0.010 | `mlp_posterior,threshold_max_probability` | +0.0011 | -0.0009 | 0 | 2 | `early stop: cannot reach two-family rule` |
| `confidence_beta_0p075` | `candidate` | `confidence_penalty_eegnet` | confidence penalty beta=0.075 | `mlp_posterior,threshold_max_probability` | +0.0018 | -0.0105 | 0 | 2 | `early stop: cannot reach two-family rule` |
| `bottleneck_dim_4` | `candidate` | `bottleneck_eegnet` | bottleneck dim=4 | `mlp_posterior,threshold_max_probability` | -0.0083 | -0.0483 | 1 | 0 | `early stop: utility boundary failed` |
| `bottleneck_dim_12` | `candidate` | `bottleneck_eegnet` | bottleneck dim=12 | `mlp_posterior,threshold_max_probability` | -0.0253 | -0.0631 | 2 | 0 | `early stop: utility boundary failed` |

## Row Read

| Candidate | Attack | Task BA | AUC | Delta AUC | Delta task BA | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `eegnet_baseline` | `mlp_posterior` | 0.6574 | 0.5157 | +0.0000 | +0.0000 | `baseline` |
| `eegnet_baseline` | `threshold_max_probability` | 0.6574 | 0.5875 | +0.0000 | +0.0000 | `baseline` |
| `eegnet_baseline` | `threshold_label_known` | 0.6574 | 0.6162 | +0.0000 | +0.0000 | `baseline` |
| `mixup_alpha_0p20_reference` | `mlp_posterior` | 0.6223 | 0.5154 | -0.0003 | -0.0351 | `mixed` |
| `mixup_alpha_0p20_reference` | `threshold_max_probability` | 0.6223 | 0.5648 | -0.0227 | -0.0351 | `family win with utility loss` |
| `mixup_alpha_0p20_reference` | `threshold_label_known` | 0.6223 | 0.6133 | -0.0029 | -0.0351 | `mixed` |
| `confidence_beta_0p025_reference` | `mlp_posterior` | 0.6540 | 0.5174 | +0.0017 | -0.0034 | `practical tie` |
| `confidence_beta_0p025_reference` | `threshold_max_probability` | 0.6540 | 0.5886 | +0.0011 | -0.0034 | `practical tie` |
| `confidence_beta_0p025_reference` | `threshold_label_known` | 0.6540 | 0.6165 | +0.0003 | -0.0034 | `practical tie` |
| `mixup_alpha_0p10` | `mlp_posterior` | 0.6463 | 0.5410 | +0.0253 | -0.0110 | `privacy regression` |
| `mixup_alpha_0p10` | `threshold_max_probability` | 0.6463 | 0.5713 | -0.0162 | -0.0110 | `family win within utility` |
| `mixup_alpha_0p10` | `threshold_label_known` | 0.6463 | 0.6161 | -0.0001 | -0.0110 | `practical tie` |
| `mixup_alpha_0p35` | `mlp_posterior` | 0.6503 | 0.5303 | +0.0146 | -0.0071 | `privacy regression` |
| `mixup_alpha_0p35` | `threshold_max_probability` | 0.6503 | 0.5815 | -0.0060 | -0.0071 | `mixed` |
| `mixup_alpha_0p35` | `threshold_label_known` |  |  |  |  | `not run (stop rule)` |
| `confidence_beta_0p010` | `mlp_posterior` | 0.6565 | 0.5170 | +0.0013 | -0.0009 | `practical tie` |
| `confidence_beta_0p010` | `threshold_max_probability` | 0.6565 | 0.5885 | +0.0010 | -0.0009 | `practical tie` |
| `confidence_beta_0p010` | `threshold_label_known` |  |  |  |  | `not run (stop rule)` |
| `confidence_beta_0p075` | `mlp_posterior` | 0.6469 | 0.5174 | +0.0017 | -0.0105 | `practical tie` |
| `confidence_beta_0p075` | `threshold_max_probability` | 0.6469 | 0.5894 | +0.0019 | -0.0105 | `practical tie` |
| `confidence_beta_0p075` | `threshold_label_known` |  |  |  |  | `not run (stop rule)` |
| `bottleneck_dim_4` | `mlp_posterior` | 0.6091 | 0.5262 | +0.0104 | -0.0483 | `privacy regression` |
| `bottleneck_dim_4` | `threshold_max_probability` | 0.6091 | 0.5605 | -0.0270 | -0.0483 | `family win with utility loss` |
| `bottleneck_dim_4` | `threshold_label_known` |  |  |  |  | `not run (stop rule)` |
| `bottleneck_dim_12` | `mlp_posterior` | 0.5943 | 0.4914 | -0.0243 | -0.0631 | `family win with utility loss` |
| `bottleneck_dim_12` | `threshold_max_probability` | 0.5943 | 0.5612 | -0.0263 | -0.0631 | `family win with utility loss` |
| `bottleneck_dim_12` | `threshold_label_known` |  |  |  |  | `not run (stop rule)` |

## Omitted By Stop Rule

- `mixup_alpha_0p35` `threshold_label_known`: `early stop: cannot reach two-family rule`.
- `confidence_beta_0p010` `threshold_label_known`: `early stop: cannot reach two-family rule`.
- `confidence_beta_0p075` `threshold_label_known`: `early stop: cannot reach two-family rule`.
- `bottleneck_dim_4` `threshold_label_known`: `early stop: utility boundary failed`.
- `bottleneck_dim_12` `threshold_label_known`: `early stop: utility boundary failed`.
