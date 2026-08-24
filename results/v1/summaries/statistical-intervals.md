# Statistical Intervals Summary

This artifact adds paired seed-level uncertainty checks to the current report layer. Intervals are 95% t intervals over the four task-model seeds for privacy-metric deltas versus same-seed plain EEGNet. Because `n=4`, these intervals are intentionally conservative and mainly prevent overclaiming tiny margins.

Companion table: statistical_intervals_matrix_v1.csv

## Key Comparisons

| Source | Dataset | Attack | Model | Metric | Mean Δ | 95% CI | Privacy seeds | Sign p | Interpretation |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `bnci_lee_validation` | BNCI | `validation:subject-id` | Bottleneck | `subject_id_accuracy` | `-0.2683` | `[-0.3488, -0.1878]` | `4/4` | `0.0625` | `directionally-supported` |
| `bnci_lee_validation` | BNCI | `validation:threshold(label-known)` | Bottleneck | `attack_auc` | `-0.0041` | `[-0.0096, +0.0013]` | `4/4` | `0.0625` | `practical-tie` |
| `bnci_lee_validation` | Lee | `validation:subject-id` | Bottleneck | `subject_id_accuracy` | `-0.2340` | `[-0.3254, -0.1425]` | `4/4` | `0.0625` | `directionally-supported` |
| `bnci_lee_validation` | Lee | `validation:threshold(label-known)` | Bottleneck | `attack_auc` | `+0.0145` | `[-0.0508, +0.0798]` | `2/4` | `0.6875` | `no-privacy-improvement` |
| `bnci_lee_validation` | Lee | `validation:threshold(max-prob)` | Bottleneck | `attack_auc` | `+0.0176` | `[+0.0036, +0.0316]` | `0/4` | `1.0000` | `no-privacy-improvement` |
| `bnci_lee_validation` | Lee | `validation:threshold(max-prob)` | CSP-LDA | `attack_auc` | `-0.0540` | `[-0.0802, -0.0278]` | `4/4` | `0.0625` | `directionally-supported` |
| `seed_stability` | BNCI | `learned+label` | Bottleneck | `attack_auc` | `-0.0028` | `[-0.0364, +0.0308]` | `2/4` | `0.6875` | `practical-tie` |
| `seed_stability` | BNCI | `learned(mlp)` | CSP-LDA | `attack_auc` | `-0.0097` | `[-0.0527, +0.0332]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | BNCI | `threshold(max-prob)` | Adv. const | `attack_auc` | `-0.0243` | `[-0.0831, +0.0346]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | Lee | `learned(mlp)` | Adv. const | `attack_auc` | `-0.0167` | `[-0.0622, +0.0289]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | Lee | `threshold(max-prob)` | CSP-LDA | `attack_auc` | `-0.0708` | `[-0.1277, -0.0140]` | `4/4` | `0.0625` | `directionally-supported` |
| `seed_stability` | PhysioNet | `learned(mlp)` | Mixup | `attack_auc` | `+0.0023` | `[-0.0922, +0.0968]` | `2/4` | `0.6875` | `practical-tie` |
| `seed_stability` | PhysioNet | `threshold(max-prob)` | CSP-LDA | `attack_auc` | `-0.0973` | `[-0.2431, +0.0485]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | PhysioNet | `learned` | Adv. ramp | `attack_auc` | `-0.0253` | `[-0.0828, +0.0323]` | `3/4` | `0.3125` | `promising-but-underpowered` |

## Interpretation

- In the original four-seed interval layer, PhysioNet `learned(mlp)` mixup was a practical tie: mean ΔAUC was essentially zero and the interval crossed zero widely. The later eight-seed extension supersedes this as a paper-facing claim and rejects the apparent mixup win.
- The expanded-subject BNCI/Lee validation rows now have the same paired interval treatment as the older seed-stability layer. BNCI bottleneck subject-ID reduction is directionally supported; Lee bottleneck threshold regressions remain caveats because their intervals are wide even when point estimates regress.
- Threshold-family effects are larger than learned-family effects in several places, but the four-seed interval layer still supports caveated wording rather than formal significance claims.

## BNCI

| Source | Attack | Model | Metric | Mean Δ | 95% CI | Privacy seeds | Sign p | Interpretation |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `bnci_lee_validation` | `validation:subject-id` | Bottleneck | `subject_id_accuracy` | `-0.2683` | `[-0.3488, -0.1878]` | `4/4` | `0.0625` | `directionally-supported` |
| `bnci_lee_validation` | `validation:subject-id` | Adv. const | `subject_id_accuracy` | `-0.0860` | `[-0.2207, +0.0487]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `bnci_lee_validation` | `validation:subject-id` | CSP-LDA | `subject_id_accuracy` | `-0.0060` | `[-0.0682, +0.0562]` | `2/4` | `0.6875` | `unstable-privacy-edge` |
| `bnci_lee_validation` | `validation:learned(mlp)` | CSP-LDA | `attack_auc` | `-0.0215` | `[-0.0365, -0.0066]` | `4/4` | `0.0625` | `directionally-supported` |
| `bnci_lee_validation` | `validation:learned(mlp)` | Bottleneck | `attack_auc` | `+0.0007` | `[-0.0363, +0.0378]` | `3/4` | `0.3125` | `practical-tie` |
| `bnci_lee_validation` | `validation:learned(mlp)` | Adv. const | `attack_auc` | `+0.0122` | `[-0.0395, +0.0640]` | `2/4` | `0.6875` | `no-privacy-improvement` |
| `bnci_lee_validation` | `validation:learned` | Bottleneck | `attack_auc` | `+0.0020` | `[-0.0322, +0.0363]` | `1/4` | `0.9375` | `practical-tie` |
| `bnci_lee_validation` | `validation:learned` | CSP-LDA | `attack_auc` | `+0.0213` | `[-0.0269, +0.0696]` | `2/4` | `0.6875` | `no-privacy-improvement` |
| `bnci_lee_validation` | `validation:learned` | Adv. const | `attack_auc` | `+0.0381` | `[-0.0180, +0.0943]` | `1/4` | `0.9375` | `no-privacy-improvement` |
| `bnci_lee_validation` | `validation:learned+label` | Bottleneck | `attack_auc` | `+0.0001` | `[-0.0160, +0.0162]` | `2/4` | `0.6875` | `practical-tie` |
| `bnci_lee_validation` | `validation:learned+label` | CSP-LDA | `attack_auc` | `+0.0169` | `[-0.0479, +0.0818]` | `2/4` | `0.6875` | `no-privacy-improvement` |
| `bnci_lee_validation` | `validation:learned+label` | Adv. const | `attack_auc` | `+0.0242` | `[-0.0313, +0.0798]` | `2/4` | `0.6875` | `no-privacy-improvement` |
| `bnci_lee_validation` | `validation:threshold(label-known)` | CSP-LDA | `attack_auc` | `-0.0387` | `[-0.0530, -0.0244]` | `4/4` | `0.0625` | `directionally-supported` |
| `bnci_lee_validation` | `validation:threshold(label-known)` | Adv. const | `attack_auc` | `-0.0129` | `[-0.0468, +0.0211]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `bnci_lee_validation` | `validation:threshold(label-known)` | Bottleneck | `attack_auc` | `-0.0041` | `[-0.0096, +0.0013]` | `4/4` | `0.0625` | `practical-tie` |
| `bnci_lee_validation` | `validation:threshold(max-prob)` | CSP-LDA | `attack_auc` | `-0.0076` | `[-0.0568, +0.0416]` | `2/4` | `0.6875` | `unstable-privacy-edge` |
| `bnci_lee_validation` | `validation:threshold(max-prob)` | Bottleneck | `attack_auc` | `-0.0008` | `[-0.0132, +0.0115]` | `3/4` | `0.3125` | `practical-tie` |
| `bnci_lee_validation` | `validation:threshold(max-prob)` | Adv. const | `attack_auc` | `+0.0071` | `[-0.0297, +0.0439]` | `2/4` | `0.6875` | `no-privacy-improvement` |
| `bnci_lee_validation` | `validation:threshold(neg-entropy)` | CSP-LDA | `attack_auc` | `-0.0076` | `[-0.0568, +0.0416]` | `2/4` | `0.6875` | `unstable-privacy-edge` |
| `bnci_lee_validation` | `validation:threshold(neg-entropy)` | Bottleneck | `attack_auc` | `-0.0008` | `[-0.0132, +0.0115]` | `3/4` | `0.3125` | `practical-tie` |
| `bnci_lee_validation` | `validation:threshold(neg-entropy)` | Adv. const | `attack_auc` | `+0.0071` | `[-0.0297, +0.0439]` | `2/4` | `0.6875` | `no-privacy-improvement` |
| `seed_stability` | `learned+label` | Bottleneck | `attack_auc` | `-0.0028` | `[-0.0364, +0.0308]` | `2/4` | `0.6875` | `practical-tie` |
| `seed_stability` | `learned+label` | Adv. ramp | `attack_auc` | `+0.0014` | `[-0.0545, +0.0572]` | `2/4` | `0.6875` | `practical-tie` |
| `seed_stability` | `learned+label` | CSP-LDA | `attack_auc` | `+0.0054` | `[-0.0450, +0.0558]` | `1/4` | `0.9375` | `no-privacy-improvement` |
| `seed_stability` | `learned+label` | Adv. const | `attack_auc` | `+0.0240` | `[+0.0145, +0.0335]` | `0/4` | `1.0000` | `no-privacy-improvement` |
| `seed_stability` | `learned` | Bottleneck | `attack_auc` | `-0.0015` | `[-0.0302, +0.0271]` | `2/4` | `0.6875` | `practical-tie` |
| `seed_stability` | `learned` | CSP-LDA | `attack_auc` | `+0.0042` | `[-0.0322, +0.0406]` | `1/4` | `0.9375` | `practical-tie` |
| `seed_stability` | `learned` | Adv. ramp | `attack_auc` | `+0.0252` | `[-0.0306, +0.0809]` | `1/4` | `0.9375` | `no-privacy-improvement` |
| `seed_stability` | `learned` | Adv. const | `attack_auc` | `+0.0412` | `[+0.0005, +0.0820]` | `0/4` | `1.0000` | `no-privacy-improvement` |
| `seed_stability` | `learned(mlp)` | CSP-LDA | `attack_auc` | `-0.0097` | `[-0.0527, +0.0332]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | `learned(mlp)` | Adv. const | `attack_auc` | `+0.0131` | `[-0.0402, +0.0664]` | `1/4` | `0.9375` | `no-privacy-improvement` |
| `seed_stability` | `learned(mlp)` | Bottleneck | `attack_auc` | `+0.0143` | `[-0.0274, +0.0560]` | `2/4` | `0.6875` | `no-privacy-improvement` |
| `seed_stability` | `learned(mlp)` | Adv. ramp | `attack_auc` | `+0.0197` | `[-0.0693, +0.1087]` | `1/4` | `0.9375` | `no-privacy-improvement` |
| `seed_stability` | `threshold(label-known)` | Adv. const | `attack_auc` | `-0.0387` | `[-0.0614, -0.0160]` | `4/4` | `0.0625` | `directionally-supported` |
| `seed_stability` | `threshold(label-known)` | CSP-LDA | `attack_auc` | `-0.0253` | `[-0.0484, -0.0023]` | `4/4` | `0.0625` | `directionally-supported` |
| `seed_stability` | `threshold(label-known)` | Adv. ramp | `attack_auc` | `-0.0089` | `[-0.0509, +0.0330]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | `threshold(label-known)` | Bottleneck | `attack_auc` | `+0.0086` | `[-0.0130, +0.0302]` | `1/4` | `0.9375` | `no-privacy-improvement` |
| `seed_stability` | `threshold(max-prob)` | Adv. const | `attack_auc` | `-0.0243` | `[-0.0831, +0.0346]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | `threshold(max-prob)` | CSP-LDA | `attack_auc` | `-0.0222` | `[-0.0429, -0.0016]` | `4/4` | `0.0625` | `directionally-supported` |
| `seed_stability` | `threshold(max-prob)` | Adv. ramp | `attack_auc` | `-0.0174` | `[-0.0497, +0.0149]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | `threshold(max-prob)` | Bottleneck | `attack_auc` | `+0.0049` | `[-0.0197, +0.0295]` | `2/4` | `0.6875` | `practical-tie` |
| `seed_stability` | `threshold(neg-entropy)` | Adv. const | `attack_auc` | `-0.0243` | `[-0.0831, +0.0345]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | `threshold(neg-entropy)` | CSP-LDA | `attack_auc` | `-0.0222` | `[-0.0429, -0.0016]` | `4/4` | `0.0625` | `directionally-supported` |
| `seed_stability` | `threshold(neg-entropy)` | Adv. ramp | `attack_auc` | `-0.0174` | `[-0.0497, +0.0149]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | `threshold(neg-entropy)` | Bottleneck | `attack_auc` | `+0.0049` | `[-0.0197, +0.0295]` | `2/4` | `0.6875` | `practical-tie` |

## Lee

| Source | Attack | Model | Metric | Mean Δ | 95% CI | Privacy seeds | Sign p | Interpretation |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `bnci_lee_validation` | `validation:subject-id` | Bottleneck | `subject_id_accuracy` | `-0.2340` | `[-0.3254, -0.1425]` | `4/4` | `0.0625` | `directionally-supported` |
| `bnci_lee_validation` | `validation:subject-id` | CSP-LDA | `subject_id_accuracy` | `-0.0340` | `[-0.0978, +0.0299]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `bnci_lee_validation` | `validation:subject-id` | Adv. const | `subject_id_accuracy` | `-0.0146` | `[-0.2358, +0.2067]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `bnci_lee_validation` | `validation:learned(mlp)` | Adv. const | `attack_auc` | `-0.0120` | `[-0.0374, +0.0134]` | `4/4` | `0.0625` | `promising-but-underpowered` |
| `bnci_lee_validation` | `validation:learned(mlp)` | CSP-LDA | `attack_auc` | `-0.0006` | `[-0.0776, +0.0765]` | `2/4` | `0.6875` | `practical-tie` |
| `bnci_lee_validation` | `validation:learned(mlp)` | Bottleneck | `attack_auc` | `+0.0011` | `[-0.0154, +0.0177]` | `2/4` | `0.6875` | `practical-tie` |
| `bnci_lee_validation` | `validation:learned` | Adv. const | `attack_auc` | `-0.0034` | `[-0.0415, +0.0348]` | `3/4` | `0.3125` | `practical-tie` |
| `bnci_lee_validation` | `validation:learned` | Bottleneck | `attack_auc` | `-0.0019` | `[-0.0201, +0.0163]` | `3/4` | `0.3125` | `practical-tie` |
| `bnci_lee_validation` | `validation:learned` | CSP-LDA | `attack_auc` | `+0.0141` | `[-0.0350, +0.0631]` | `1/4` | `0.9375` | `no-privacy-improvement` |
| `bnci_lee_validation` | `validation:learned+label` | Bottleneck | `attack_auc` | `-0.0040` | `[-0.0153, +0.0073]` | `3/4` | `0.3125` | `practical-tie` |
| `bnci_lee_validation` | `validation:learned+label` | Adv. const | `attack_auc` | `+0.0014` | `[-0.0300, +0.0328]` | `2/4` | `0.6875` | `practical-tie` |
| `bnci_lee_validation` | `validation:learned+label` | CSP-LDA | `attack_auc` | `+0.0095` | `[-0.0238, +0.0429]` | `1/4` | `0.9375` | `no-privacy-improvement` |
| `bnci_lee_validation` | `validation:threshold(label-known)` | CSP-LDA | `attack_auc` | `-0.0621` | `[-0.0966, -0.0275]` | `4/4` | `0.0625` | `directionally-supported` |
| `bnci_lee_validation` | `validation:threshold(label-known)` | Adv. const | `attack_auc` | `-0.0234` | `[-0.0492, +0.0025]` | `4/4` | `0.0625` | `promising-but-underpowered` |
| `bnci_lee_validation` | `validation:threshold(label-known)` | Bottleneck | `attack_auc` | `+0.0145` | `[-0.0508, +0.0798]` | `2/4` | `0.6875` | `no-privacy-improvement` |
| `bnci_lee_validation` | `validation:threshold(max-prob)` | CSP-LDA | `attack_auc` | `-0.0540` | `[-0.0802, -0.0278]` | `4/4` | `0.0625` | `directionally-supported` |
| `bnci_lee_validation` | `validation:threshold(max-prob)` | Adv. const | `attack_auc` | `-0.0054` | `[-0.0288, +0.0180]` | `2/4` | `0.6875` | `unstable-privacy-edge` |
| `bnci_lee_validation` | `validation:threshold(max-prob)` | Bottleneck | `attack_auc` | `+0.0176` | `[+0.0036, +0.0316]` | `0/4` | `1.0000` | `no-privacy-improvement` |
| `bnci_lee_validation` | `validation:threshold(neg-entropy)` | CSP-LDA | `attack_auc` | `-0.0540` | `[-0.0802, -0.0278]` | `4/4` | `0.0625` | `directionally-supported` |
| `bnci_lee_validation` | `validation:threshold(neg-entropy)` | Adv. const | `attack_auc` | `-0.0054` | `[-0.0288, +0.0180]` | `2/4` | `0.6875` | `unstable-privacy-edge` |
| `bnci_lee_validation` | `validation:threshold(neg-entropy)` | Bottleneck | `attack_auc` | `+0.0176` | `[+0.0036, +0.0316]` | `0/4` | `1.0000` | `no-privacy-improvement` |
| `seed_stability` | `learned+label` | Bottleneck | `attack_auc` | `+0.0055` | `[-0.0742, +0.0851]` | `2/4` | `0.6875` | `no-privacy-improvement` |
| `seed_stability` | `learned+label` | Adv. ramp | `attack_auc` | `+0.0106` | `[-0.0239, +0.0450]` | `1/4` | `0.9375` | `no-privacy-improvement` |
| `seed_stability` | `learned+label` | Adv. const | `attack_auc` | `+0.0162` | `[-0.0189, +0.0512]` | `1/4` | `0.9375` | `no-privacy-improvement` |
| `seed_stability` | `learned+label` | CSP-LDA | `attack_auc` | `+0.0176` | `[-0.1011, +0.1363]` | `2/4` | `0.6875` | `no-privacy-improvement` |
| `seed_stability` | `learned` | Bottleneck | `attack_auc` | `+0.0039` | `[-0.0889, +0.0967]` | `2/4` | `0.6875` | `practical-tie` |
| `seed_stability` | `learned` | CSP-LDA | `attack_auc` | `+0.0070` | `[-0.0793, +0.0932]` | `2/4` | `0.6875` | `no-privacy-improvement` |
| `seed_stability` | `learned` | Adv. ramp | `attack_auc` | `+0.0167` | `[-0.0213, +0.0547]` | `1/4` | `0.9375` | `no-privacy-improvement` |
| `seed_stability` | `learned` | Adv. const | `attack_auc` | `+0.0206` | `[-0.0179, +0.0591]` | `1/4` | `0.9375` | `no-privacy-improvement` |
| `seed_stability` | `learned(mlp)` | Adv. const | `attack_auc` | `-0.0167` | `[-0.0622, +0.0289]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | `learned(mlp)` | Bottleneck | `attack_auc` | `-0.0150` | `[-0.1549, +0.1248]` | `2/4` | `0.6875` | `unstable-privacy-edge` |
| `seed_stability` | `learned(mlp)` | Adv. ramp | `attack_auc` | `-0.0025` | `[-0.0563, +0.0513]` | `2/4` | `0.6875` | `practical-tie` |
| `seed_stability` | `learned(mlp)` | CSP-LDA | `attack_auc` | `+0.0172` | `[-0.0881, +0.1225]` | `2/4` | `0.6875` | `no-privacy-improvement` |
| `seed_stability` | `threshold(label-known)` | CSP-LDA | `attack_auc` | `-0.0625` | `[-0.1087, -0.0163]` | `4/4` | `0.0625` | `directionally-supported` |
| `seed_stability` | `threshold(label-known)` | Bottleneck | `attack_auc` | `-0.0277` | `[-0.0653, +0.0099]` | `4/4` | `0.0625` | `promising-but-underpowered` |
| `seed_stability` | `threshold(label-known)` | Adv. const | `attack_auc` | `-0.0266` | `[-0.0493, -0.0038]` | `4/4` | `0.0625` | `directionally-supported` |
| `seed_stability` | `threshold(label-known)` | Adv. ramp | `attack_auc` | `-0.0211` | `[-0.0378, -0.0045]` | `4/4` | `0.0625` | `directionally-supported` |
| `seed_stability` | `threshold(max-prob)` | CSP-LDA | `attack_auc` | `-0.0708` | `[-0.1277, -0.0140]` | `4/4` | `0.0625` | `directionally-supported` |
| `seed_stability` | `threshold(max-prob)` | Adv. const | `attack_auc` | `-0.0150` | `[-0.0576, +0.0275]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | `threshold(max-prob)` | Bottleneck | `attack_auc` | `-0.0066` | `[-0.0992, +0.0860]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | `threshold(max-prob)` | Adv. ramp | `attack_auc` | `-0.0027` | `[-0.0602, +0.0547]` | `3/4` | `0.3125` | `practical-tie` |
| `seed_stability` | `threshold(neg-entropy)` | CSP-LDA | `attack_auc` | `-0.0709` | `[-0.1277, -0.0140]` | `4/4` | `0.0625` | `directionally-supported` |
| `seed_stability` | `threshold(neg-entropy)` | Adv. const | `attack_auc` | `-0.0150` | `[-0.0576, +0.0275]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | `threshold(neg-entropy)` | Bottleneck | `attack_auc` | `-0.0066` | `[-0.0992, +0.0860]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | `threshold(neg-entropy)` | Adv. ramp | `attack_auc` | `-0.0027` | `[-0.0602, +0.0547]` | `3/4` | `0.3125` | `practical-tie` |

## PhysioNet

| Source | Attack | Model | Metric | Mean Δ | 95% CI | Privacy seeds | Sign p | Interpretation |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `seed_stability` | `learned+label` | Adv. ramp | `attack_auc` | `-0.0245` | `[-0.0708, +0.0217]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | `learned+label` | Mixup | `attack_auc` | `-0.0123` | `[-0.0635, +0.0389]` | `2/4` | `0.6875` | `unstable-privacy-edge` |
| `seed_stability` | `learned+label` | Conf. penalty | `attack_auc` | `-0.0022` | `[-0.0059, +0.0016]` | `4/4` | `0.0625` | `practical-tie` |
| `seed_stability` | `learned+label` | CSP-LDA | `attack_auc` | `+0.0756` | `[-0.0369, +0.1880]` | `1/4` | `0.9375` | `no-privacy-improvement` |
| `seed_stability` | `learned` | Adv. ramp | `attack_auc` | `-0.0253` | `[-0.0828, +0.0323]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | `learned` | Mixup | `attack_auc` | `-0.0092` | `[-0.0799, +0.0614]` | `1/4` | `0.9375` | `unstable-privacy-edge` |
| `seed_stability` | `learned` | Conf. penalty | `attack_auc` | `-0.0005` | `[-0.0015, +0.0004]` | `3/4` | `0.3125` | `practical-tie` |
| `seed_stability` | `learned` | CSP-LDA | `attack_auc` | `+0.0774` | `[-0.0306, +0.1853]` | `1/4` | `0.9375` | `no-privacy-improvement` |
| `seed_stability` | `learned(mlp)` | Mixup | `attack_auc` | `+0.0023` | `[-0.0922, +0.0968]` | `2/4` | `0.6875` | `practical-tie` |
| `seed_stability` | `learned(mlp)` | Conf. penalty | `attack_auc` | `+0.0043` | `[-0.1074, +0.1160]` | `2/4` | `0.6875` | `practical-tie` |
| `seed_stability` | `learned(mlp)` | Adv. ramp | `attack_auc` | `+0.0052` | `[-0.0430, +0.0533]` | `3/4` | `0.3125` | `no-privacy-improvement` |
| `seed_stability` | `learned(mlp)` | CSP-LDA | `attack_auc` | `+0.0075` | `[-0.1802, +0.1953]` | `2/4` | `0.6875` | `no-privacy-improvement` |
| `seed_stability` | `threshold(label-known)` | Adv. ramp | `attack_auc` | `-0.1015` | `[-0.1465, -0.0565]` | `4/4` | `0.0625` | `directionally-supported` |
| `seed_stability` | `threshold(label-known)` | CSP-LDA | `attack_auc` | `-0.0974` | `[-0.1363, -0.0585]` | `4/4` | `0.0625` | `directionally-supported` |
| `seed_stability` | `threshold(label-known)` | Mixup | `attack_auc` | `-0.0029` | `[-0.0212, +0.0155]` | `2/4` | `0.6875` | `practical-tie` |
| `seed_stability` | `threshold(label-known)` | Conf. penalty | `attack_auc` | `+0.0003` | `[-0.0015, +0.0021]` | `1/4` | `0.9375` | `practical-tie` |
| `seed_stability` | `threshold(max-prob)` | CSP-LDA | `attack_auc` | `-0.0973` | `[-0.2431, +0.0485]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | `threshold(max-prob)` | Adv. ramp | `attack_auc` | `-0.0496` | `[-0.1084, +0.0093]` | `4/4` | `0.0625` | `promising-but-underpowered` |
| `seed_stability` | `threshold(max-prob)` | Mixup | `attack_auc` | `-0.0227` | `[-0.0493, +0.0039]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | `threshold(max-prob)` | Conf. penalty | `attack_auc` | `+0.0011` | `[-0.0064, +0.0085]` | `1/4` | `0.9375` | `practical-tie` |
| `seed_stability` | `threshold(neg-entropy)` | CSP-LDA | `attack_auc` | `-0.0973` | `[-0.2431, +0.0485]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | `threshold(neg-entropy)` | Adv. ramp | `attack_auc` | `-0.0496` | `[-0.1084, +0.0093]` | `4/4` | `0.0625` | `promising-but-underpowered` |
| `seed_stability` | `threshold(neg-entropy)` | Mixup | `attack_auc` | `-0.0227` | `[-0.0494, +0.0039]` | `3/4` | `0.3125` | `promising-but-underpowered` |
| `seed_stability` | `threshold(neg-entropy)` | Conf. penalty | `attack_auc` | `+0.0011` | `[-0.0064, +0.0085]` | `1/4` | `0.9375` | `practical-tie` |
