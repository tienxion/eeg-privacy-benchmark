# CSP Baseline Anchor Summary

This artifact positions `csp_lda` against the current BNCI, Lee, and PhysioNet defense stories. It keeps the existing four-model robustness report intact and adds the classical baseline as an external anchor.

Companion artifacts:
- csp_anchor_matrix_v1.csv
- csp_anchor_auc_plot_v1.png

## At A Glance

| Dataset | CSP task | CSP subject-ID | CSP attack wins (`/6`) | Mean rank with family | Read |
| --- | --- | --- | --- | --- | --- |
| `bnci2014_001` | `0.7179` | `0.8177` | `1` | `2.33` | Task-strong but not a serious defense. CSP stays highly identity-leaky and never wins the matched family. |
| `lee2019_mi` | `0.6575` | `0.6554` | `3` | `2.67` | Threshold-only privacy winner. CSP loses the learned attacks, but it wins all three threshold attacks while keeping heavy identity leakage. |
| `physionet_motor_imagery` | `0.6216` | `n/a` | `2` | `3.17` | Classical threshold winner, learned-attack loser. CSP dominates both label-free threshold attacks and stays competitive on label-known threshold, but it fails learned attacks. |

## BNCI 2014-001

| Attack | CSP task | CSP AUC | Δtask vs `eegnet` | ΔAUC vs `eegnet` | Existing family winner | Winner AUC | CSP rank with family | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `learned+label` | `0.7179` | `0.5078` | `+0.0360` | `+0.0054` | `bottleneck_eegnet (dim=6)` | `0.4996` | `4` | loses matched family |
| `learned` | `0.7179` | `0.5085` | `+0.0360` | `+0.0042` | `bottleneck_eegnet (dim=6)` | `0.5028` | `3` | loses matched family |
| `learned(mlp)` | `0.7179` | `0.4960` | `+0.0360` | `-0.0097` | `eegnet` | `0.5057` | `1` | wins matched family |
| `threshold(label-known)` | `0.7179` | `0.5311` | `+0.0360` | `-0.0253` | `adversarial_eegnet (constant@1.00)` | `0.5178` | `2` | loses matched family |
| `threshold(max-prob)` | `0.7179` | `0.4977` | `+0.0360` | `-0.0222` | `adversarial_eegnet (constant@1.00)` | `0.4957` | `2` | loses matched family |
| `threshold(neg-entropy)` | `0.7179` | `0.4977` | `+0.0360` | `-0.0222` | `adversarial_eegnet (constant@1.00)` | `0.4957` | `2` | loses matched family |

Read: `csp_lda` raises task balanced accuracy to `0.7179`, but subject-ID remains very high at `0.8177`. It never beats the matched family winner on raw membership AUC, and under the learned attacks it is worse than the bottleneck.

## Lee2019 MI

| Attack | CSP task | CSP AUC | Δtask vs `eegnet` | ΔAUC vs `eegnet` | Existing family winner | Winner AUC | CSP rank with family | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `learned+label` | `0.6575` | `0.5085` | `+0.1562` | `+0.0176` | `eegnet` | `0.4910` | `5` | loses matched family |
| `learned` | `0.6575` | `0.5070` | `+0.1562` | `+0.0070` | `eegnet` | `0.5000` | `3` | loses matched family |
| `learned(mlp)` | `0.6575` | `0.5416` | `+0.1562` | `+0.0172` | `adversarial_eegnet (constant@1.00)` | `0.5077` | `5` | loses matched family |
| `threshold(label-known)` | `0.6575` | `0.5429` | `+0.1562` | `-0.0625` | `bottleneck_eegnet (dim=6)` | `0.5777` | `1` | wins matched family |
| `threshold(max-prob)` | `0.6575` | `0.4598` | `+0.1562` | `-0.0708` | `adversarial_eegnet (constant@1.00)` | `0.5156` | `1` | wins matched family |
| `threshold(neg-entropy)` | `0.6575` | `0.4598` | `+0.1562` | `-0.0709` | `adversarial_eegnet (constant@1.00)` | `0.5156` | `1` | wins matched family |

Read: `csp_lda` is a clean attack-family split case. Subject-ID stays high at `0.6554`, and the learned attacks do not favor it, but all three threshold attacks do. Under both label-free threshold attacks it reaches `0.4598` AUC, which is much lower than the current neural family.

## PhysioNet Motor Imagery (`subjects 1-46`)

| Attack | CSP task | CSP AUC | Δtask vs `eegnet` | ΔAUC vs `eegnet` | Existing family winner | Winner AUC | CSP rank with family | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `learned+label` | `0.6216` | `0.5918` | `-0.0358` | `+0.0756` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.4917` | `5` | loses matched family |
| `learned` | `0.6216` | `0.5881` | `-0.0358` | `+0.0774` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.4855` | `5` | loses matched family |
| `learned(mlp)` | `0.6216` | `0.5207` | `-0.0358` | `+0.0075` | `eegnet` | `0.5131` | `5` | loses matched family |
| `threshold(label-known)` | `0.6216` | `0.5188` | `-0.0358` | `-0.0974` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5147` | `2` | loses matched family |
| `threshold(max-prob)` | `0.6216` | `0.4902` | `-0.0358` | `-0.0973` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5379` | `1` | wins matched family |
| `threshold(neg-entropy)` | `0.6216` | `0.4902` | `-0.0358` | `-0.0973` | `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)` | `0.5380` | `1` | wins matched family |

Read: the PhysioNet story is now split between classical and neural lines. Under the completed learned attacks, `csp_lda` is clearly leaky. Under threshold attacks, it becomes competitive immediately and wins both label-free thresholds at `0.4902` while keeping task balanced accuracy `0.6216`.
