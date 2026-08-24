# Cross-Attack Robustness Summary

This report consolidates the completed matched four-seed membership-defense families across BNCI, Lee, and the `subjects 1-46` PhysioNet evaluation. It provides a single view of attack sensitivity.

Related figure: [attack-family split AUC](../figures/attack-family-split-auc.png)

## At A Glance

| Dataset | `learned+label` winner | `learned` winner | `learned(mlp)` winner | `threshold(label-known)` winner | `threshold(max-prob)` winner | `threshold(neg-entropy)` winner |
| --- | --- | --- | --- | --- | --- | --- |
| `bnci2014_001` | `bottleneck_eegnet (dim=6)` | `bottleneck_eegnet (dim=6)` | `eegnet` | `adversarial_eegnet (constant@1.00)` | `adversarial_eegnet (constant@1.00)` | `adversarial_eegnet (constant@1.00)` |
| `lee2019_mi` | `eegnet` | `eegnet` | `adversarial_eegnet (constant@1.00)` | `bottleneck_eegnet (dim=6)` | `adversarial_eegnet (constant@1.00)` | `adversarial_eegnet (constant@1.00)` |
| `physionet_motor_imagery` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `eegnet` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` |

## BNCI 2014-001

| Attack | `eegnet` task / AUC | `bottleneck_eegnet (dim=6)` task / AUC | `adversarial_eegnet (constant@1.00)` task / AUC | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` task / AUC | Lowest AUC |
| --- | --- | --- | --- | --- | --- |
| `learned+label` | `0.6819 / 0.5024` | `0.7409 / 0.4996` | `0.5373 / 0.5264` | `0.5473 / 0.5038` | `bottleneck_eegnet (dim=6)` |
| `learned` | `0.6819 / 0.5043` | `0.7409 / 0.5028` | `0.5373 / 0.5455` | `0.5473 / 0.5295` | `bottleneck_eegnet (dim=6)` |
| `learned(mlp)` | `0.6819 / 0.5057` | `0.7409 / 0.5200` | `0.5373 / 0.5188` | `0.5473 / 0.5254` | `eegnet` |
| `threshold(label-known)` | `0.6819 / 0.5564` | `0.7409 / 0.5650` | `0.5373 / 0.5178` | `0.5473 / 0.5475` | `adversarial_eegnet (constant@1.00)` |
| `threshold(max-prob)` | `0.6819 / 0.5199` | `0.7409 / 0.5248` | `0.5373 / 0.4957` | `0.5473 / 0.5025` | `adversarial_eegnet (constant@1.00)` |
| `threshold(neg-entropy)` | `0.6819 / 0.5199` | `0.7409 / 0.5248` | `0.5373 / 0.4957` | `0.5473 / 0.5025` | `adversarial_eegnet (constant@1.00)` |

Label-free threshold stability:
`max_probability` vs `negative_entropy` AUC diffs by model are `eegnet`: `0.000002`, `bottleneck_eegnet (dim=6)`: `0.000001`, `adversarial_eegnet (constant@1.00)`: `0.000018`, `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: `0.000017`.

## Lee2019 MI

| Attack | `eegnet` task / AUC | `bottleneck_eegnet (dim=6)` task / AUC | `adversarial_eegnet (constant@1.00)` task / AUC | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` task / AUC | Lowest AUC |
| --- | --- | --- | --- | --- | --- |
| `learned+label` | `0.5012 / 0.4910` | `0.5158 / 0.4964` | `0.5204 / 0.5071` | `0.5129 / 0.5015` | `eegnet` |
| `learned` | `0.5012 / 0.5000` | `0.5158 / 0.5039` | `0.5204 / 0.5206` | `0.5129 / 0.5167` | `eegnet` |
| `learned(mlp)` | `0.5012 / 0.5243` | `0.5158 / 0.5093` | `0.5204 / 0.5077` | `0.5129 / 0.5219` | `adversarial_eegnet (constant@1.00)` |
| `threshold(label-known)` | `0.5012 / 0.6054` | `0.5158 / 0.5777` | `0.5204 / 0.5788` | `0.5129 / 0.5842` | `bottleneck_eegnet (dim=6)` |
| `threshold(max-prob)` | `0.5012 / 0.5307` | `0.5158 / 0.5241` | `0.5204 / 0.5156` | `0.5129 / 0.5280` | `adversarial_eegnet (constant@1.00)` |
| `threshold(neg-entropy)` | `0.5012 / 0.5307` | `0.5158 / 0.5241` | `0.5204 / 0.5156` | `0.5129 / 0.5280` | `adversarial_eegnet (constant@1.00)` |

Label-free threshold stability:
`max_probability` vs `negative_entropy` AUC diffs by model are `eegnet`: `0.000006`, `bottleneck_eegnet (dim=6)`: `0.000007`, `adversarial_eegnet (constant@1.00)`: `0.000000`, `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: `0.000001`.

## PhysioNet Motor Imagery (`subjects 1-46`)

| Attack | `eegnet` task / AUC | `confidence_penalty_eegnet (beta=0.025)` task / AUC | `mixup_eegnet (alpha=0.20)` task / AUC | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` task / AUC | Lowest AUC |
| --- | --- | --- | --- | --- | --- |
| `learned+label` | `0.6574 / 0.5162` | `0.6540 / 0.5140` | `0.6223 / 0.5039` | `0.5087 / 0.4917` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` |
| `learned` | `0.6574 / 0.5108` | `0.6540 / 0.5103` | `0.6223 / 0.5015` | `0.5087 / 0.4855` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` |
| `learned(mlp)` | `0.6574 / 0.5131` | `0.6540 / 0.5174` | `0.6223 / 0.5154` | `0.5087 / 0.5183` | `eegnet` |
| `threshold(label-known)` | `0.6574 / 0.6162` | `0.6540 / 0.6165` | `0.6223 / 0.6133` | `0.5087 / 0.5147` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` |
| `threshold(max-prob)` | `0.6574 / 0.5875` | `0.6540 / 0.5886` | `0.6223 / 0.5648` | `0.5087 / 0.5379` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` |
| `threshold(neg-entropy)` | `0.6574 / 0.5875` | `0.6540 / 0.5886` | `0.6223 / 0.5648` | `0.5087 / 0.5380` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` |

Label-free threshold stability:
`max_probability` vs `negative_entropy` AUC diffs by model are `eegnet`: `0.000000`, `confidence_penalty_eegnet (beta=0.025)`: `0.000000`, `mixup_eegnet (alpha=0.20)`: `0.000001`, `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: `0.000008`.
