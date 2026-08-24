# MLP Posterior Attack Summary

This report evaluates a nonlinear learned membership attacker with `attack_type=mlp` and `score_type=posterior_probabilities`. It tests whether conclusions obtained with linear logistic-regression attacks survive a nonlinear attacker.

## At A Glance

| Dataset | Linear posterior winner | MLP posterior winner | Winner changed? | Plain `eegnet` linear AUC | Plain `eegnet` MLP AUC |
| --- | --- | --- | --- | --- | --- |
| `bnci2014_001` | `bottleneck_eegnet (dim=6)` | `csp_lda` | `yes` | `0.5043` | `0.5057` |
| `lee2019_mi` | `eegnet` | `adversarial_eegnet (constant@1.00)` | `yes` | `0.5000` | `0.5243` |
| `physionet_motor_imagery` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `eegnet` | `yes` | `0.5108` | `0.5131` |

## BNCI 2014-001

| Model | Task | Linear posterior AUC | MLP posterior AUC | ΔAUC (`mlp - linear`) | MLP rank |
| --- | --- | --- | --- | --- | --- |
| `csp_lda` | `0.7179` | `0.5085` | `0.4960` | `-0.0125` | `1` |
| `eegnet` | `0.6819` | `0.5043` | `0.5057` | `+0.0014` | `2` |
| `adversarial_eegnet (constant@1.00)` | `0.5373` | `0.5455` | `0.5188` | `-0.0268` | `3` |
| `bottleneck_eegnet (dim=6)` | `0.7409` | `0.5028` | `0.5200` | `+0.0172` | `4` |
| `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5473` | `0.5295` | `0.5254` | `-0.0041` | `5` |

MLP winner: `csp_lda` at `0.4960`. The winner changes from `bottleneck_eegnet (dim=6)` under the linear posterior attack.  Largest AUC increase: `bottleneck_eegnet (dim=6)` `+0.0172`. Largest AUC decrease: `adversarial_eegnet (constant@1.00)` `-0.0268`.

## Lee2019 MI

| Model | Task | Linear posterior AUC | MLP posterior AUC | ΔAUC (`mlp - linear`) | MLP rank |
| --- | --- | --- | --- | --- | --- |
| `adversarial_eegnet (constant@1.00)` | `0.5204` | `0.5206` | `0.5077` | `-0.0129` | `1` |
| `bottleneck_eegnet (dim=6)` | `0.5158` | `0.5039` | `0.5093` | `+0.0054` | `2` |
| `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5129` | `0.5167` | `0.5219` | `+0.0051` | `3` |
| `eegnet` | `0.5012` | `0.5000` | `0.5243` | `+0.0243` | `4` |
| `csp_lda` | `0.6575` | `0.5070` | `0.5416` | `+0.0346` | `5` |

MLP winner: `adversarial_eegnet (constant@1.00)` at `0.5077`. The winner changes from `eegnet` under the linear posterior attack.  Largest AUC increase: `csp_lda` `+0.0346`. Largest AUC decrease: `adversarial_eegnet (constant@1.00)` `-0.0129`.

## PhysioNet Motor Imagery (`subjects 1-46`)

| Model | Task | Linear posterior AUC | MLP posterior AUC | ΔAUC (`mlp - linear`) | MLP rank |
| --- | --- | --- | --- | --- | --- |
| `eegnet` | `0.6574` | `0.5108` | `0.5131` | `+0.0024` | `1` |
| `mixup_eegnet (alpha=0.20)` | `0.6223` | `0.5015` | `0.5154` | `+0.0139` | `2` |
| `confidence_penalty_eegnet (beta=0.025)` | `0.6540` | `0.5103` | `0.5174` | `+0.0072` | `3` |
| `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5087` | `0.4855` | `0.5183` | `+0.0328` | `4` |
| `csp_lda` | `0.6216` | `0.5881` | `0.5207` | `-0.0675` | `5` |

MLP winner: `eegnet` at `0.5131`. The winner changes from `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` under the linear posterior attack.  Largest AUC increase: `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` `+0.0328`. Largest AUC decrease: `csp_lda` `-0.0675`.
