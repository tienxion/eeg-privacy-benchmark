# Seed Stability Summary

This artifact checks whether the current defense rankings survive paired seed-level comparison against plain `eegnet`. Lower membership AUC is better. A negative `mean ΔAUC` means the model improves privacy versus the same-seed EEGNet baseline.

Related files:

- seed_stability_matrix_v1.csv
- seed_stability_seed_matrix_v1.csv
- seed_stability_delta_plot_v1.png

## At A Glance

| Dataset | Attack | Mean-AUC winner | Winner mean ΔAUC vs EEGNet | Winner seed privacy wins | Winner rank-1 seeds |
| --- | --- | --- | ---: | ---: | ---: |
| `bnci2014_001` | `learned+label` | `bottleneck_eegnet (dim=6)` | `-0.0028` | `2/4` | `1/4` |
| `bnci2014_001` | `learned` | `bottleneck_eegnet (dim=6)` | `-0.0015` | `2/4` | `2/4` |
| `bnci2014_001` | `learned(mlp)` | `csp_lda` | `-0.0097` | `3/4` | `2/4` |
| `bnci2014_001` | `threshold(label-known)` | `adversarial_eegnet (constant@1.00)` | `-0.0387` | `4/4` | `3/4` |
| `bnci2014_001` | `threshold(max-prob)` | `adversarial_eegnet (constant@1.00)` | `-0.0243` | `3/4` | `3/4` |
| `bnci2014_001` | `threshold(neg-entropy)` | `adversarial_eegnet (constant@1.00)` | `-0.0243` | `3/4` | `3/4` |
| `lee2019_mi` | `learned+label` | `eegnet` | `+0.0000` | `0/4` | `1/4` |
| `lee2019_mi` | `learned` | `eegnet` | `+0.0000` | `0/4` | `1/4` |
| `lee2019_mi` | `learned(mlp)` | `adversarial_eegnet (constant@1.00)` | `-0.0167` | `3/4` | `1/4` |
| `lee2019_mi` | `threshold(label-known)` | `csp_lda` | `-0.0625` | `4/4` | `3/4` |
| `lee2019_mi` | `threshold(max-prob)` | `csp_lda` | `-0.0708` | `4/4` | `4/4` |
| `lee2019_mi` | `threshold(neg-entropy)` | `csp_lda` | `-0.0709` | `4/4` | `4/4` |
| `physionet_motor_imagery` | `learned+label` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `-0.0245` | `3/4` | `3/4` |
| `physionet_motor_imagery` | `learned` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `-0.0253` | `3/4` | `2/4` |
| `physionet_motor_imagery` | `learned(mlp)` | `eegnet` | `+0.0000` | `0/4` | `0/4` |
| `physionet_motor_imagery` | `threshold(label-known)` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `-0.1015` | `4/4` | `1/4` |
| `physionet_motor_imagery` | `threshold(max-prob)` | `csp_lda` | `-0.0973` | `3/4` | `3/4` |
| `physionet_motor_imagery` | `threshold(neg-entropy)` | `csp_lda` | `-0.0973` | `3/4` | `3/4` |

## Interpretation

- BNCI’s nonlinear learned-attack `csp_lda` win is not just a rounding artifact: it improves on same-seed EEGNet on most seeds and has the lowest mean AUC in the combined family. The broader BNCI recommendation still needs the threshold caveat, because adversarial constant remains the threshold-family privacy winner with much lower utility.
- Lee remains attack-sensitive but seed-stable enough to discuss: adversarial constant wins the nonlinear learned attack on mean AUC, while `csp_lda` dominates label-free thresholds and bottleneck remains strongest when subject-ID defense is part of the objective.
- PhysioNet’s nonlinear learned-attack winner should be treated as unresolved. `mixup_eegnet (alpha=0.20)` has only a tiny mean AUC edge over EEGNet under `learned(mlp)`, so the claim should be framed as a tie unless widened or repeated.

## BNCI 2014-001

| Attack | Model | Mean task | Mean AUC | Mean ΔAUC vs EEGNet | ΔAUC std | Privacy-improving seeds | Rank-1 seeds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned+label` | `csp_lda` | `0.7179` | `0.5078` | `+0.0054` | `0.0317` | `1/4` | `1/4` |
| `learned+label` | `eegnet` | `0.6819` | `0.5024` | `+0.0000` | `0.0000` | `0/4` | `0/4` |
| `learned+label` | `bottleneck_eegnet (dim=6)` | `0.7409` | `0.4996` | `-0.0028` | `0.0211` | `2/4` | `1/4` |
| `learned+label` | `adversarial_eegnet (constant@1.00)` | `0.5373` | `0.5264` | `+0.0240` | `0.0060` | `0/4` | `0/4` |
| `learned+label` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5473` | `0.5038` | `+0.0014` | `0.0351` | `2/4` | `2/4` |
| `learned` | `csp_lda` | `0.7179` | `0.5085` | `+0.0042` | `0.0229` | `1/4` | `1/4` |
| `learned` | `eegnet` | `0.6819` | `0.5043` | `+0.0000` | `0.0000` | `0/4` | `0/4` |
| `learned` | `bottleneck_eegnet (dim=6)` | `0.7409` | `0.5028` | `-0.0015` | `0.0180` | `2/4` | `2/4` |
| `learned` | `adversarial_eegnet (constant@1.00)` | `0.5373` | `0.5455` | `+0.0412` | `0.0256` | `0/4` | `0/4` |
| `learned` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5473` | `0.5295` | `+0.0252` | `0.0350` | `1/4` | `1/4` |
| `learned(mlp)` | `csp_lda` | `0.7179` | `0.4960` | `-0.0097` | `0.0270` | `3/4` | `2/4` |
| `learned(mlp)` | `eegnet` | `0.6819` | `0.5057` | `+0.0000` | `0.0000` | `0/4` | `0/4` |
| `learned(mlp)` | `bottleneck_eegnet (dim=6)` | `0.7409` | `0.5200` | `+0.0143` | `0.0262` | `2/4` | `0/4` |
| `learned(mlp)` | `adversarial_eegnet (constant@1.00)` | `0.5373` | `0.5188` | `+0.0131` | `0.0335` | `1/4` | `1/4` |
| `learned(mlp)` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5473` | `0.5254` | `+0.0197` | `0.0559` | `1/4` | `1/4` |
| `threshold(label-known)` | `csp_lda` | `0.7179` | `0.5311` | `-0.0253` | `0.0145` | `4/4` | `1/4` |
| `threshold(label-known)` | `eegnet` | `0.6819` | `0.5564` | `+0.0000` | `0.0000` | `0/4` | `0/4` |
| `threshold(label-known)` | `bottleneck_eegnet (dim=6)` | `0.7409` | `0.5650` | `+0.0086` | `0.0136` | `1/4` | `0/4` |
| `threshold(label-known)` | `adversarial_eegnet (constant@1.00)` | `0.5373` | `0.5178` | `-0.0387` | `0.0143` | `4/4` | `3/4` |
| `threshold(label-known)` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5473` | `0.5475` | `-0.0089` | `0.0264` | `3/4` | `0/4` |
| `threshold(max-prob)` | `csp_lda` | `0.7179` | `0.4977` | `-0.0222` | `0.0130` | `4/4` | `1/4` |
| `threshold(max-prob)` | `eegnet` | `0.6819` | `0.5199` | `+0.0000` | `0.0000` | `0/4` | `0/4` |
| `threshold(max-prob)` | `bottleneck_eegnet (dim=6)` | `0.7409` | `0.5248` | `+0.0049` | `0.0154` | `2/4` | `0/4` |
| `threshold(max-prob)` | `adversarial_eegnet (constant@1.00)` | `0.5373` | `0.4957` | `-0.0243` | `0.0370` | `3/4` | `3/4` |
| `threshold(max-prob)` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5473` | `0.5025` | `-0.0174` | `0.0203` | `3/4` | `0/4` |
| `threshold(neg-entropy)` | `csp_lda` | `0.7179` | `0.4977` | `-0.0222` | `0.0130` | `4/4` | `1/4` |
| `threshold(neg-entropy)` | `eegnet` | `0.6819` | `0.5199` | `+0.0000` | `0.0000` | `0/4` | `0/4` |
| `threshold(neg-entropy)` | `bottleneck_eegnet (dim=6)` | `0.7409` | `0.5248` | `+0.0049` | `0.0154` | `2/4` | `0/4` |
| `threshold(neg-entropy)` | `adversarial_eegnet (constant@1.00)` | `0.5373` | `0.4957` | `-0.0243` | `0.0370` | `3/4` | `3/4` |
| `threshold(neg-entropy)` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5473` | `0.5025` | `-0.0174` | `0.0203` | `3/4` | `0/4` |

## Lee2019 MI

| Attack | Model | Mean task | Mean AUC | Mean ΔAUC vs EEGNet | ΔAUC std | Privacy-improving seeds | Rank-1 seeds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned+label` | `csp_lda` | `0.6575` | `0.5085` | `+0.0176` | `0.0746` | `2/4` | `1/4` |
| `learned+label` | `eegnet` | `0.5012` | `0.4910` | `+0.0000` | `0.0000` | `0/4` | `1/4` |
| `learned+label` | `bottleneck_eegnet (dim=6)` | `0.5158` | `0.4964` | `+0.0055` | `0.0501` | `2/4` | `2/4` |
| `learned+label` | `adversarial_eegnet (constant@1.00)` | `0.5204` | `0.5071` | `+0.0162` | `0.0220` | `1/4` | `0/4` |
| `learned+label` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5129` | `0.5015` | `+0.0106` | `0.0216` | `1/4` | `0/4` |
| `learned` | `csp_lda` | `0.6575` | `0.5070` | `+0.0070` | `0.0542` | `2/4` | `1/4` |
| `learned` | `eegnet` | `0.5012` | `0.5000` | `+0.0000` | `0.0000` | `0/4` | `1/4` |
| `learned` | `bottleneck_eegnet (dim=6)` | `0.5158` | `0.5039` | `+0.0039` | `0.0583` | `2/4` | `2/4` |
| `learned` | `adversarial_eegnet (constant@1.00)` | `0.5204` | `0.5206` | `+0.0206` | `0.0242` | `1/4` | `0/4` |
| `learned` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5129` | `0.5167` | `+0.0167` | `0.0239` | `1/4` | `0/4` |
| `learned(mlp)` | `csp_lda` | `0.6575` | `0.5416` | `+0.0172` | `0.0662` | `2/4` | `0/4` |
| `learned(mlp)` | `eegnet` | `0.5012` | `0.5243` | `+0.0000` | `0.0000` | `0/4` | `1/4` |
| `learned(mlp)` | `bottleneck_eegnet (dim=6)` | `0.5158` | `0.5093` | `-0.0150` | `0.0879` | `2/4` | `2/4` |
| `learned(mlp)` | `adversarial_eegnet (constant@1.00)` | `0.5204` | `0.5077` | `-0.0167` | `0.0286` | `3/4` | `1/4` |
| `learned(mlp)` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5129` | `0.5219` | `-0.0025` | `0.0338` | `2/4` | `0/4` |
| `threshold(label-known)` | `csp_lda` | `0.6575` | `0.5429` | `-0.0625` | `0.0290` | `4/4` | `3/4` |
| `threshold(label-known)` | `eegnet` | `0.5012` | `0.6054` | `+0.0000` | `0.0000` | `0/4` | `0/4` |
| `threshold(label-known)` | `bottleneck_eegnet (dim=6)` | `0.5158` | `0.5777` | `-0.0277` | `0.0236` | `4/4` | `0/4` |
| `threshold(label-known)` | `adversarial_eegnet (constant@1.00)` | `0.5204` | `0.5788` | `-0.0266` | `0.0143` | `4/4` | `1/4` |
| `threshold(label-known)` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5129` | `0.5842` | `-0.0211` | `0.0105` | `4/4` | `0/4` |
| `threshold(max-prob)` | `csp_lda` | `0.6575` | `0.4598` | `-0.0708` | `0.0357` | `4/4` | `4/4` |
| `threshold(max-prob)` | `eegnet` | `0.5012` | `0.5307` | `+0.0000` | `0.0000` | `0/4` | `0/4` |
| `threshold(max-prob)` | `bottleneck_eegnet (dim=6)` | `0.5158` | `0.5241` | `-0.0066` | `0.0582` | `3/4` | `0/4` |
| `threshold(max-prob)` | `adversarial_eegnet (constant@1.00)` | `0.5204` | `0.5156` | `-0.0150` | `0.0267` | `3/4` | `0/4` |
| `threshold(max-prob)` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5129` | `0.5280` | `-0.0027` | `0.0361` | `3/4` | `0/4` |
| `threshold(neg-entropy)` | `csp_lda` | `0.6575` | `0.4598` | `-0.0709` | `0.0357` | `4/4` | `4/4` |
| `threshold(neg-entropy)` | `eegnet` | `0.5012` | `0.5307` | `+0.0000` | `0.0000` | `0/4` | `0/4` |
| `threshold(neg-entropy)` | `bottleneck_eegnet (dim=6)` | `0.5158` | `0.5241` | `-0.0066` | `0.0582` | `3/4` | `0/4` |
| `threshold(neg-entropy)` | `adversarial_eegnet (constant@1.00)` | `0.5204` | `0.5156` | `-0.0150` | `0.0267` | `3/4` | `0/4` |
| `threshold(neg-entropy)` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5129` | `0.5280` | `-0.0027` | `0.0361` | `3/4` | `0/4` |

## PhysioNet Motor Imagery (`subjects 1-46`)

| Attack | Model | Mean task | Mean AUC | Mean ΔAUC vs EEGNet | ΔAUC std | Privacy-improving seeds | Rank-1 seeds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `learned+label` | `csp_lda` | `0.6216` | `0.5918` | `+0.0756` | `0.0707` | `1/4` | `0/4` |
| `learned+label` | `eegnet` | `0.6574` | `0.5162` | `+0.0000` | `0.0000` | `0/4` | `0/4` |
| `learned+label` | `confidence_penalty_eegnet (beta=0.025)` | `0.6540` | `0.5140` | `-0.0022` | `0.0023` | `4/4` | `1/4` |
| `learned+label` | `mixup_eegnet (alpha=0.20)` | `0.6223` | `0.5039` | `-0.0123` | `0.0322` | `2/4` | `0/4` |
| `learned+label` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5087` | `0.4917` | `-0.0245` | `0.0291` | `3/4` | `3/4` |
| `learned` | `csp_lda` | `0.6216` | `0.5881` | `+0.0774` | `0.0678` | `1/4` | `1/4` |
| `learned` | `eegnet` | `0.6574` | `0.5108` | `+0.0000` | `0.0000` | `0/4` | `1/4` |
| `learned` | `confidence_penalty_eegnet (beta=0.025)` | `0.6540` | `0.5103` | `-0.0005` | `0.0006` | `3/4` | `0/4` |
| `learned` | `mixup_eegnet (alpha=0.20)` | `0.6223` | `0.5015` | `-0.0092` | `0.0444` | `1/4` | `0/4` |
| `learned` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5087` | `0.4855` | `-0.0253` | `0.0362` | `3/4` | `2/4` |
| `learned(mlp)` | `csp_lda` | `0.6216` | `0.5207` | `+0.0075` | `0.1180` | `2/4` | `2/4` |
| `learned(mlp)` | `eegnet` | `0.6574` | `0.5131` | `+0.0000` | `0.0000` | `0/4` | `0/4` |
| `learned(mlp)` | `confidence_penalty_eegnet (beta=0.025)` | `0.6540` | `0.5174` | `+0.0043` | `0.0702` | `2/4` | `1/4` |
| `learned(mlp)` | `mixup_eegnet (alpha=0.20)` | `0.6223` | `0.5154` | `+0.0023` | `0.0594` | `2/4` | `0/4` |
| `learned(mlp)` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5087` | `0.5183` | `+0.0052` | `0.0303` | `3/4` | `1/4` |
| `threshold(label-known)` | `csp_lda` | `0.6216` | `0.5188` | `-0.0974` | `0.0244` | `4/4` | `3/4` |
| `threshold(label-known)` | `eegnet` | `0.6574` | `0.6162` | `+0.0000` | `0.0000` | `0/4` | `0/4` |
| `threshold(label-known)` | `confidence_penalty_eegnet (beta=0.025)` | `0.6540` | `0.6165` | `+0.0003` | `0.0011` | `1/4` | `0/4` |
| `threshold(label-known)` | `mixup_eegnet (alpha=0.20)` | `0.6223` | `0.6133` | `-0.0029` | `0.0115` | `2/4` | `0/4` |
| `threshold(label-known)` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5087` | `0.5147` | `-0.1015` | `0.0283` | `4/4` | `1/4` |
| `threshold(max-prob)` | `csp_lda` | `0.6216` | `0.4902` | `-0.0973` | `0.0916` | `3/4` | `3/4` |
| `threshold(max-prob)` | `eegnet` | `0.6574` | `0.5875` | `+0.0000` | `0.0000` | `0/4` | `0/4` |
| `threshold(max-prob)` | `confidence_penalty_eegnet (beta=0.025)` | `0.6540` | `0.5886` | `+0.0011` | `0.0047` | `1/4` | `0/4` |
| `threshold(max-prob)` | `mixup_eegnet (alpha=0.20)` | `0.6223` | `0.5648` | `-0.0227` | `0.0167` | `3/4` | `0/4` |
| `threshold(max-prob)` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5087` | `0.5379` | `-0.0496` | `0.0370` | `4/4` | `1/4` |
| `threshold(neg-entropy)` | `csp_lda` | `0.6216` | `0.4902` | `-0.0973` | `0.0916` | `3/4` | `3/4` |
| `threshold(neg-entropy)` | `eegnet` | `0.6574` | `0.5875` | `+0.0000` | `0.0000` | `0/4` | `0/4` |
| `threshold(neg-entropy)` | `confidence_penalty_eegnet (beta=0.025)` | `0.6540` | `0.5886` | `+0.0011` | `0.0047` | `1/4` | `0/4` |
| `threshold(neg-entropy)` | `mixup_eegnet (alpha=0.20)` | `0.6223` | `0.5648` | `-0.0227` | `0.0167` | `3/4` | `0/4` |
| `threshold(neg-entropy)` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5087` | `0.5380` | `-0.0496` | `0.0370` | `4/4` | `1/4` |
