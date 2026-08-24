# Claims Audit

This record separates supported conclusions from conclusions that require caveats or are not supported by benchmark v1. It is derived from the frozen attack-family, shortlist, and seed-stability evidence.

Companion table: [`claims-audit.csv`](claims-audit.csv)

## Summary

| ID | Dataset | Verdict | Claim | Paper use |
| --- | --- | --- | --- | --- |
| `C01` | `bnci2014_001` | `qualified-support` | Use bottleneck EEGNet as the BNCI default defense. | Main claim, with attack-sensitivity caveat. |
| `C02` | `bnci2014_001` | `not-supported` | Bottleneck EEGNet is the best BNCI membership defense across attacks. | Do not claim this. |
| `C03` | `lee2019_mi` | `qualified-support` | Use bottleneck EEGNet as the Lee default when subject-ID privacy is part of the objective. | Main claim, explicitly subject-ID-first. |
| `C04` | `lee2019_mi` | `not-supported` | Lee membership privacy has a single best model. | Do not claim this. |
| `C05` | `physionet_motor_imagery` | `not-supported` | Mixup is the PhysioNet nonlinear learned-attack winner. | Do not claim this; report the eight-seed extension as rejecting the apparent win. |
| `C06` | `physionet_motor_imagery` | `qualified-support` | Adversarial linear-ramp is the PhysioNet privacy extreme. | Use only as a privacy-extreme tradeoff point. |
| `C07` | `physionet_motor_imagery` | `not-supported` | PhysioNet has a stable default defense. | Do not claim this. |
| `C08` | `all` | `supported` | The two label-free threshold attacks are redundant at the ranking level. | Use to simplify the attack discussion. |
| `C09` | `all` | `supported` | Adding the nonlinear posterior MLP attacker changes the learned-attack read. | Use as an attack-robustness motivation. |
| `C10` | `physionet_motor_imagery` | `not-supported` | Federation or the secure-aggregation prototype provides a privacy guarantee. | Use as a systems result and explicit threat-model boundary only. |

## Evidence

### C01: Use bottleneck EEGNet as the BNCI default defense.

- Verdict: `qualified-support`
- Evidence: Bottleneck has the best task accuracy (0.7409, +0.0590 vs EEGNet) and the lowest subject-ID accuracy (0.4618). Tier A expanded-subject validation on BNCI subjects 1-9 preserves this: subject-ID delta -0.2683, task delta +0.0106, and MLP-membership delta +0.0007. Tier B learned-logistic validation also has no material membership regression: posterior-only delta +0.0020 and label-aware posterior delta +0.0001. Tier C threshold validation also does not regress versus EEGNet: label-known delta -0.0041, max-probability delta -0.0008, and negative-entropy delta -0.0008. Attacker-refit calibration keeps the learned-MLP bottleneck read as a practical tie: mean delta AUC -0.0025 across 3 attacker seeds, with 2/3 privacy-improving refits.
- Caveat: Raw membership is not attack-invariant: the bottleneck is not the raw winner in every threshold cell, and nonlinear learned attack favors `csp_lda`.

### C02: Bottleneck EEGNet is the best BNCI membership defense across attacks.

- Verdict: `not-supported`
- Evidence: Adversarial constant wins more raw attack-AUC cells in the shortlist (3.0000 of 6) and has lower mean AUC than bottleneck (0.5166 vs 0.5228).
- Caveat: Bottleneck remains the pragmatic default only because utility and subject-ID matter.

### C03: Use bottleneck EEGNet as the Lee default when subject-ID privacy is part of the objective.

- Verdict: `qualified-support`
- Evidence: Bottleneck has the lowest Lee subject-ID accuracy (0.2625) and improves task (+0.0146 vs EEGNet). Tier A expanded-subject validation on Lee subjects 1-12 preserves this: subject-ID delta -0.2340, task delta +0.0019, and MLP-membership delta +0.0011. Tier B learned-logistic validation also has no material membership regression: posterior-only delta -0.0019 and label-aware posterior delta -0.0040. Tier C threshold validation narrows the claim: bottleneck regresses on label-known (+0.0145), max-probability (+0.0176), and negative-entropy (+0.0176) membership AUC. Attacker-refit calibration is also unfavorable for a Lee learned-membership claim: bottleneck mean delta AUC +0.0135 across 3 attacker seeds, with 0/3 privacy-improving refits.
- Caveat: Membership winners split by attack family; CSP owns the expanded-subject threshold family but has much higher subject-ID leakage.

### C04: Lee membership privacy has a single best model.

- Verdict: `not-supported`
- Evidence: The nonlinear learned attack winner is `adversarial_eegnet (constant@1.00)` with mean AUC 0.5077, while threshold-family summaries favor CSP.
- Caveat: Frame Lee as attack-sensitive, not model-invariant.

### C05: Mixup is the PhysioNet nonlinear learned-attack winner.

- Verdict: `not-supported`
- Evidence: The original four-seed `learned(mlp)` read was already too small: mixup was only +0.0023 AUC versus EEGNet, with 95% CI [-0.0922, +0.0968], improves privacy on 2/4 seeds, and has 0/4 unique rank-1 seeds. The completed random-subset validation independently keeps the claim unresolved: Tier A mixup has mean delta AUC +0.0022, privacy improvement on 3/5 subsets, and mean task delta -0.0365. The completed eight-seed subjects `1-46` extension rejects the apparent win: mean delta AUC +0.0096, privacy improvement on 2/8 seeds, and mean task delta +0.0014. The completed 11-row cross-run holdout also falls short for the nonlinear attacker: mixup MLP delta AUC is -0.0075 with task delta -0.0185, below the pre-registered 0.010 AUC-improvement rule.
- Caveat: Frame PhysioNet mixup under the nonlinear learned attacker as not supported, not merely underpowered.

### C06: Adversarial linear-ramp is the PhysioNet privacy extreme.

- Verdict: `qualified-support`
- Evidence: It has the best neural-family mean attack rank (1.5000) and largest mean AUC reduction (-0.0409 vs EEGNet).
- Caveat: The task drop is severe (-0.1486), and CSP wins the combined label-free threshold family once included.

### C07: PhysioNet has a stable default defense.

- Verdict: `not-supported`
- Evidence: Attack-family winners disagree: learned-family AUC favors adversarial linear-ramp, while label-free thresholds favor CSP (0.4902). The completed random-subset validation reinforces the split: Tier B max-probability mixup has mean delta AUC -0.0223 on 4/5 subsets with mean task delta -0.0365, while Tier C label-known CSP-LDA has mean delta AUC -0.0641 on 5/5 subsets with mean task delta +0.0632. The pre-registered non-adversarial defense grid also fails to produce a stable default: `mixup_alpha_0p10` is `attack-specific only` with mean delta AUC +0.0030, `confidence_beta_0p010` is `early stop: cannot reach two-family rule`, and `bottleneck_dim_12` is `early stop: utility boundary failed` despite 2 AUC-win families. The completed 11-row cross-run holdout reaches the same conclusion: mixup reduces AUC on both threshold rows within utility (-0.0166 max-probability, -0.0147 label-known), but not MLP (-0.0075); bottleneck regresses on MLP (+0.0166); and CSP-LDA is a threshold anchor (-0.0555 label-known) rather than a learned-attack defense.
- Caveat: Frame PhysioNet as unresolved and protocol-sensitive. CSP-LDA is a label-known threshold anchor, not a learned-attack default; cross-run wins remain threshold-only or attack-specific.

### C08: The two label-free threshold attacks are redundant at the ranking level.

- Verdict: `supported`
- Evidence: Winner pairs match across datasets: bnci2014_001: `adversarial_eegnet (constant@1.00)` / `adversarial_eegnet (constant@1.00)`; lee2019_mi: `csp_lda` / `csp_lda`; physionet_motor_imagery: `csp_lda` / `csp_lda`.
- Caveat: This supports ranking redundancy, not that the threshold family is unimportant.

### C09: Adding the nonlinear posterior MLP attacker changes the learned-attack read.

- Verdict: `supported`
- Evidence: The `learned(mlp)` mean-AUC winners are BNCI `csp_lda`, Lee `adversarial_eegnet (constant@1.00)`, PhysioNet `eegnet`.
- Caveat: PhysioNet's original MLP rank winner does not survive the eight-seed extension, so use this as an attack-sensitivity result rather than a PhysioNet defense claim.

### C10: Federation or the secure-aggregation prototype provides a privacy guarantee.

- Verdict: `not-supported`
- Evidence: The frozen PhysioNet replication changes mean task balanced accuracy from 0.6574 to 0.5185 (paired delta -0.1389), so the 0.60 utility gate fails. The pairwise-mask simulator matches FedAvg within 2.384e-07 and adds an estimated 0.0649% communication, but it is a functional single-process prototype.
- Caveat: Privacy deltas are diagnostic only. The prototype assumes all clients are present, hidden pairwise secrets, and no collusion; it does not protect against malicious clients, dropout, metadata, output leakage, membership inference, or provide differential privacy.
