# PhysioNet Random Subject-Subset Validation Results

This report summarizes completed commands from the deterministic PhysioNet random-subset validation plan. Lower membership-inference AUC is better for privacy. Deltas are computed versus same-subset plain EEGNet for the same attack tier.

Companion private row tables are not distributed; aggregate values are frozen in this summary.

## Completion

- Tier A: `20/20` commands complete.
- Tier B: `25/25` commands complete.
- Tier C: `25/25` commands complete.

## Current Decision Read

- Tier A mixup completed subsets: `5`.
- Mean delta AUC versus EEGNet: `+0.0022`; privacy-improving subsets: `3/5`.
- Mean delta task balanced accuracy: `-0.0365`.
- Current decision: `practical tie or unresolved`.
- Tier B max-probability threshold aggregate:
- `adversarial_eegnet` mean delta AUC `+0.0083`, privacy-improving subsets `2/5`, mean delta task BA `-0.0625`.
- `confidence_penalty_eegnet` mean delta AUC `+0.0001`, privacy-improving subsets `3/5`, mean delta task BA `-0.0016`.
- `csp_lda` mean delta AUC `-0.0139`, privacy-improving subsets `3/5`, mean delta task BA `+0.0632`.
- `mixup_eegnet` mean delta AUC `-0.0223`, privacy-improving subsets `4/5`, mean delta task BA `-0.0365`.
- Current Tier B threshold-family decision: no stable promotable default; keep the result attack-family-specific.
- Tier C label-known-log-probability threshold aggregate:
- `adversarial_eegnet` mean delta AUC `-0.0398`, privacy-improving subsets `4/5`, mean delta task BA `-0.0625`.
- `confidence_penalty_eegnet` mean delta AUC `+0.0000`, privacy-improving subsets `3/5`, mean delta task BA `-0.0016`.
- `csp_lda` mean delta AUC `-0.0641`, privacy-improving subsets `5/5`, mean delta task BA `+0.0632`.
- `mixup_eegnet` mean delta AUC `-0.0108`, privacy-improving subsets `4/5`, mean delta task BA `-0.0365`.
- Current Tier C threshold-family decision: candidate threshold-score-specific defense: `csp_lda`.
- Interpretation: this supports a CSP-LDA label-known threshold anchor, not a learned-attack PhysioNet defense default.

## Tier A Aggregate Model Read

| Model | Completed subsets | Mean delta AUC | Privacy-improving subsets | Mean delta task BA |
| --- | ---: | ---: | ---: | ---: |
| `adversarial_eegnet` | 5 | +0.0016 | 3/5 | -0.0625 |
| `confidence_penalty_eegnet` | 5 | +0.0002 | 2/5 | -0.0016 |
| `mixup_eegnet` | 5 | +0.0022 | 3/5 | -0.0365 |

## Tier B Aggregate Model Read

| Model | Completed subsets | Mean delta AUC | Privacy-improving subsets | Mean delta task BA |
| --- | ---: | ---: | ---: | ---: |
| `adversarial_eegnet` | 5 | +0.0083 | 2/5 | -0.0625 |
| `confidence_penalty_eegnet` | 5 | +0.0001 | 3/5 | -0.0016 |
| `csp_lda` | 5 | -0.0139 | 3/5 | +0.0632 |
| `mixup_eegnet` | 5 | -0.0223 | 4/5 | -0.0365 |

## Tier C Aggregate Model Read

| Model | Completed subsets | Mean delta AUC | Privacy-improving subsets | Mean delta task BA |
| --- | ---: | ---: | ---: | ---: |
| `adversarial_eegnet` | 5 | -0.0398 | 4/5 | -0.0625 |
| `confidence_penalty_eegnet` | 5 | +0.0000 | 3/5 | -0.0016 |
| `csp_lda` | 5 | -0.0641 | 5/5 | +0.0632 |
| `mixup_eegnet` | 5 | -0.0108 | 4/5 | -0.0365 |

## Completed Rows

| Tier | Subset | Model | Attack | Task BA | Attack AUC | Delta AUC | Delta Task |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `A` | `subset_01_n32_seed20260703` | `eegnet` | `mlp::posterior_probabilities` | 0.6462 | 0.5177 | 0.0000 | 0.0000 |
| `A` | `subset_01_n32_seed20260703` | `mixup_eegnet` | `mlp::posterior_probabilities` | 0.5358 | 0.4942 | -0.0236 | -0.1104 |
| `A` | `subset_01_n32_seed20260703` | `confidence_penalty_eegnet` | `mlp::posterior_probabilities` | 0.6453 | 0.5192 | 0.0015 | -0.0009 |
| `A` | `subset_01_n32_seed20260703` | `adversarial_eegnet` | `mlp::posterior_probabilities` | 0.5086 | 0.4807 | -0.0370 | -0.1376 |
| `A` | `subset_02_n32_seed20260703` | `eegnet` | `mlp::posterior_probabilities` | 0.4884 | 0.5079 | 0.0000 | 0.0000 |
| `A` | `subset_02_n32_seed20260703` | `mixup_eegnet` | `mlp::posterior_probabilities` | 0.5250 | 0.5470 | 0.0391 | 0.0366 |
| `A` | `subset_02_n32_seed20260703` | `confidence_penalty_eegnet` | `mlp::posterior_probabilities` | 0.4876 | 0.5078 | -0.0001 | -0.0008 |
| `A` | `subset_02_n32_seed20260703` | `adversarial_eegnet` | `mlp::posterior_probabilities` | 0.4978 | 0.5636 | 0.0557 | 0.0094 |
| `A` | `subset_03_n32_seed20260703` | `eegnet` | `mlp::posterior_probabilities` | 0.5566 | 0.5570 | 0.0000 | 0.0000 |
| `A` | `subset_03_n32_seed20260703` | `mixup_eegnet` | `mlp::posterior_probabilities` | 0.5249 | 0.5410 | -0.0160 | -0.0317 |
| `A` | `subset_03_n32_seed20260703` | `confidence_penalty_eegnet` | `mlp::posterior_probabilities` | 0.5638 | 0.5574 | 0.0004 | 0.0072 |
| `A` | `subset_03_n32_seed20260703` | `adversarial_eegnet` | `mlp::posterior_probabilities` | 0.5107 | 0.5489 | -0.0080 | -0.0459 |
| `A` | `subset_04_n32_seed20260703` | `eegnet` | `mlp::posterior_probabilities` | 0.6001 | 0.4650 | 0.0000 | 0.0000 |
| `A` | `subset_04_n32_seed20260703` | `mixup_eegnet` | `mlp::posterior_probabilities` | 0.5370 | 0.4916 | 0.0266 | -0.0631 |
| `A` | `subset_04_n32_seed20260703` | `confidence_penalty_eegnet` | `mlp::posterior_probabilities` | 0.5884 | 0.4639 | -0.0011 | -0.0118 |
| `A` | `subset_04_n32_seed20260703` | `adversarial_eegnet` | `mlp::posterior_probabilities` | 0.5100 | 0.4873 | 0.0223 | -0.0902 |
| `A` | `subset_05_n32_seed20260703` | `eegnet` | `mlp::posterior_probabilities` | 0.5604 | 0.5138 | 0.0000 | 0.0000 |
| `A` | `subset_05_n32_seed20260703` | `mixup_eegnet` | `mlp::posterior_probabilities` | 0.5465 | 0.4985 | -0.0153 | -0.0139 |
| `A` | `subset_05_n32_seed20260703` | `confidence_penalty_eegnet` | `mlp::posterior_probabilities` | 0.5586 | 0.5140 | 0.0002 | -0.0018 |
| `A` | `subset_05_n32_seed20260703` | `adversarial_eegnet` | `mlp::posterior_probabilities` | 0.5122 | 0.4890 | -0.0248 | -0.0482 |
| `B` | `subset_01_n32_seed20260703` | `eegnet` | `threshold::max_probability` | 0.6462 | 0.4919 | 0.0000 | 0.0000 |
| `B` | `subset_01_n32_seed20260703` | `mixup_eegnet` | `threshold::max_probability` | 0.5358 | 0.4682 | -0.0238 | -0.1104 |
| `B` | `subset_01_n32_seed20260703` | `confidence_penalty_eegnet` | `threshold::max_probability` | 0.6453 | 0.4935 | 0.0016 | -0.0009 |
| `B` | `subset_01_n32_seed20260703` | `adversarial_eegnet` | `threshold::max_probability` | 0.5086 | 0.4905 | -0.0015 | -0.1376 |
| `B` | `subset_01_n32_seed20260703` | `csp_lda` | `threshold::max_probability` | 0.6723 | 0.4849 | -0.0071 | 0.0261 |
| `B` | `subset_02_n32_seed20260703` | `eegnet` | `threshold::max_probability` | 0.4884 | 0.5160 | 0.0000 | 0.0000 |
| `B` | `subset_02_n32_seed20260703` | `mixup_eegnet` | `threshold::max_probability` | 0.5250 | 0.4628 | -0.0532 | 0.0366 |
| `B` | `subset_02_n32_seed20260703` | `confidence_penalty_eegnet` | `threshold::max_probability` | 0.4876 | 0.5156 | -0.0004 | -0.0008 |
| `B` | `subset_02_n32_seed20260703` | `adversarial_eegnet` | `threshold::max_probability` | 0.4978 | 0.5303 | 0.0144 | 0.0094 |
| `B` | `subset_02_n32_seed20260703` | `csp_lda` | `threshold::max_probability` | 0.5896 | 0.4807 | -0.0353 | 0.1013 |
| `B` | `subset_03_n32_seed20260703` | `eegnet` | `threshold::max_probability` | 0.5566 | 0.5543 | 0.0000 | 0.0000 |
| `B` | `subset_03_n32_seed20260703` | `mixup_eegnet` | `threshold::max_probability` | 0.5249 | 0.5146 | -0.0397 | -0.0317 |
| `B` | `subset_03_n32_seed20260703` | `confidence_penalty_eegnet` | `threshold::max_probability` | 0.5638 | 0.5535 | -0.0008 | 0.0072 |
| `B` | `subset_03_n32_seed20260703` | `adversarial_eegnet` | `threshold::max_probability` | 0.5107 | 0.5278 | -0.0264 | -0.0459 |
| `B` | `subset_03_n32_seed20260703` | `csp_lda` | `threshold::max_probability` | 0.6284 | 0.4921 | -0.0622 | 0.0718 |
| `B` | `subset_04_n32_seed20260703` | `eegnet` | `threshold::max_probability` | 0.6001 | 0.4639 | 0.0000 | 0.0000 |
| `B` | `subset_04_n32_seed20260703` | `mixup_eegnet` | `threshold::max_probability` | 0.5370 | 0.4822 | 0.0183 | -0.0631 |
| `B` | `subset_04_n32_seed20260703` | `confidence_penalty_eegnet` | `threshold::max_probability` | 0.5884 | 0.4641 | 0.0002 | -0.0118 |
| `B` | `subset_04_n32_seed20260703` | `adversarial_eegnet` | `threshold::max_probability` | 0.5100 | 0.5052 | 0.0413 | -0.0902 |
| `B` | `subset_04_n32_seed20260703` | `csp_lda` | `threshold::max_probability` | 0.6409 | 0.4811 | 0.0172 | 0.0407 |
| `B` | `subset_05_n32_seed20260703` | `eegnet` | `threshold::max_probability` | 0.5604 | 0.4945 | 0.0000 | 0.0000 |
| `B` | `subset_05_n32_seed20260703` | `mixup_eegnet` | `threshold::max_probability` | 0.5465 | 0.4813 | -0.0132 | -0.0139 |
| `B` | `subset_05_n32_seed20260703` | `confidence_penalty_eegnet` | `threshold::max_probability` | 0.5586 | 0.4942 | -0.0004 | -0.0018 |
| `B` | `subset_05_n32_seed20260703` | `adversarial_eegnet` | `threshold::max_probability` | 0.5122 | 0.5084 | 0.0139 | -0.0482 |
| `B` | `subset_05_n32_seed20260703` | `csp_lda` | `threshold::max_probability` | 0.6364 | 0.5125 | 0.0179 | 0.0761 |
| `C` | `subset_01_n32_seed20260703` | `eegnet` | `threshold::label_known_log_probability` | 0.6462 | 0.5575 | 0.0000 | 0.0000 |
| `C` | `subset_01_n32_seed20260703` | `mixup_eegnet` | `threshold::label_known_log_probability` | 0.5358 | 0.5469 | -0.0106 | -0.1104 |
| `C` | `subset_01_n32_seed20260703` | `confidence_penalty_eegnet` | `threshold::label_known_log_probability` | 0.6453 | 0.5584 | 0.0009 | -0.0009 |
| `C` | `subset_01_n32_seed20260703` | `adversarial_eegnet` | `threshold::label_known_log_probability` | 0.5086 | 0.5427 | -0.0148 | -0.1376 |
| `C` | `subset_01_n32_seed20260703` | `csp_lda` | `threshold::label_known_log_probability` | 0.6723 | 0.4842 | -0.0733 | 0.0261 |
| `C` | `subset_02_n32_seed20260703` | `eegnet` | `threshold::label_known_log_probability` | 0.4884 | 0.5320 | 0.0000 | 0.0000 |
| `C` | `subset_02_n32_seed20260703` | `mixup_eegnet` | `threshold::label_known_log_probability` | 0.5250 | 0.5559 | 0.0239 | 0.0366 |
| `C` | `subset_02_n32_seed20260703` | `confidence_penalty_eegnet` | `threshold::label_known_log_probability` | 0.4876 | 0.5320 | -0.0000 | -0.0008 |
| `C` | `subset_02_n32_seed20260703` | `adversarial_eegnet` | `threshold::label_known_log_probability` | 0.4978 | 0.5352 | 0.0032 | 0.0094 |
| `C` | `subset_02_n32_seed20260703` | `csp_lda` | `threshold::label_known_log_probability` | 0.5896 | 0.5241 | -0.0080 | 0.1013 |
| `C` | `subset_03_n32_seed20260703` | `eegnet` | `threshold::label_known_log_probability` | 0.5566 | 0.6004 | 0.0000 | 0.0000 |
| `C` | `subset_03_n32_seed20260703` | `mixup_eegnet` | `threshold::label_known_log_probability` | 0.5249 | 0.5741 | -0.0263 | -0.0317 |
| `C` | `subset_03_n32_seed20260703` | `confidence_penalty_eegnet` | `threshold::label_known_log_probability` | 0.5638 | 0.6004 | 0.0000 | 0.0072 |
| `C` | `subset_03_n32_seed20260703` | `adversarial_eegnet` | `threshold::label_known_log_probability` | 0.5107 | 0.5165 | -0.0838 | -0.0459 |
| `C` | `subset_03_n32_seed20260703` | `csp_lda` | `threshold::label_known_log_probability` | 0.6284 | 0.5048 | -0.0956 | 0.0718 |
| `C` | `subset_04_n32_seed20260703` | `eegnet` | `threshold::label_known_log_probability` | 0.6001 | 0.5536 | 0.0000 | 0.0000 |
| `C` | `subset_04_n32_seed20260703` | `mixup_eegnet` | `threshold::label_known_log_probability` | 0.5370 | 0.5342 | -0.0193 | -0.0631 |
| `C` | `subset_04_n32_seed20260703` | `confidence_penalty_eegnet` | `threshold::label_known_log_probability` | 0.5884 | 0.5534 | -0.0002 | -0.0118 |
| `C` | `subset_04_n32_seed20260703` | `adversarial_eegnet` | `threshold::label_known_log_probability` | 0.5100 | 0.5314 | -0.0222 | -0.0902 |
| `C` | `subset_04_n32_seed20260703` | `csp_lda` | `threshold::label_known_log_probability` | 0.6409 | 0.4939 | -0.0597 | 0.0407 |
| `C` | `subset_05_n32_seed20260703` | `eegnet` | `threshold::label_known_log_probability` | 0.5604 | 0.5909 | 0.0000 | 0.0000 |
| `C` | `subset_05_n32_seed20260703` | `mixup_eegnet` | `threshold::label_known_log_probability` | 0.5465 | 0.5693 | -0.0216 | -0.0139 |
| `C` | `subset_05_n32_seed20260703` | `confidence_penalty_eegnet` | `threshold::label_known_log_probability` | 0.5586 | 0.5903 | -0.0006 | -0.0018 |
| `C` | `subset_05_n32_seed20260703` | `adversarial_eegnet` | `threshold::label_known_log_probability` | 0.5122 | 0.5094 | -0.0815 | -0.0482 |
| `C` | `subset_05_n32_seed20260703` | `csp_lda` | `threshold::label_known_log_probability` | 0.6364 | 0.5068 | -0.0841 | 0.0761 |
