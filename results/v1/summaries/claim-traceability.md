# Claim Traceability Matrix

This matrix connects each claim-audit entry to manuscript locations, evidence assets, caveats, and reporting constraints. It preserves the boundary between supported, qualified, and unsupported conclusions across future venue formats.

Companion CSV: [`claim-traceability.csv`](claim-traceability.csv)

## Traceability Matrix

| Claim | Verdict | Role | Manuscript location | Primary asset | Supporting assets | Editing rule |
| --- | --- | --- | --- | --- | --- | --- |
| `C01` | `qualified-support` | main positive qualified claim | Abstract; Introduction; Results 5.2; Discussion; Conclusion | M-T1 | S-F1, S-F3, S-T1, S-T3 | Keep the attack-sensitivity caveat next to the BNCI bottleneck recommendation. |
| `C02` | `not-supported` | guardrail / claim not to make | Discussion; Limitations | S-T3 | S-T1, S-F3 | Use only to explain why the paper does not claim bottleneck is membership-best across attacks. |
| `C03` | `qualified-support` | main positive qualified claim | Abstract; Introduction; Results 5.3; Discussion; Conclusion | M-T1 | S-F1, S-F4, S-T1, S-T3 | Keep the subject-ID-first condition explicit. |
| `C04` | `not-supported` | guardrail / claim not to make | Discussion; Limitations | S-T3 | S-T1, S-F1 | Use only to support the attack-sensitive Lee framing. |
| `C05` | `not-supported` | rejected preliminary claim | Abstract; Results 5.4; Discussion; Limitations | M-F1 | S-F5, S-T2, S-T3, S-T4 | Do not call mixup a stable PhysioNet win under the nonlinear learned attacker. |
| `C06` | `qualified-support` | privacy-extreme tradeoff claim | Abstract; Results 5.4; Discussion | M-T1 | S-F2, S-F3, S-T1, S-T3, S-T4 | Mention severe utility cost whenever the privacy improvement is discussed. |
| `C07` | `not-supported` | guardrail / unresolved-dataset claim | Abstract; Results 5.4; Discussion; Limitations; Conclusion | M-T1 | S-F1, S-F2, S-T1, S-T3, S-T4 | Frame PhysioNet as unresolved and protocol-sensitive. |
| `C08` | `supported` | attack-discussion simplification | Methods; Results | S-T1 | S-F1, S-F3 | Say ranking redundancy only; do not imply threshold attacks are unimportant. |
| `C09` | `supported` | attack-robustness motivation | Abstract; Introduction; Results 5.1 and 5.4 | M-F1 | S-F1, S-F5, S-T1, S-T2, S-T4 | Use as motivation for nonlinear attack checks while preserving the PhysioNet tie caveat. |
| `C10` | `not-supported` | systems-result threat-model guardrail | Abstract; Threat Model 3.2; Results 5.5-5.6; Discussion; Limitations; Conclusion | M-T2 | evidence/federated_phase45_summary, S-T3 | Keep the failed utility gate and functional-only secure-aggregation scope adjacent; never state a privacy guarantee. |

## Claim Details

### C01: Use bottleneck EEGNet as the BNCI default defense.

- Dataset: `bnci2014_001`
- Verdict: `qualified-support`
- Evidence: Bottleneck has the best task accuracy (0.7409, +0.0590 vs EEGNet) and the lowest subject-ID accuracy (0.4618). Tier A expanded-subject validation on BNCI subjects 1-9 preserves this: subject-ID delta -0.2683, task delta +0.0106, and MLP-membership delta +0.0007. Tier B learned-logistic validation also has no material membership regression: posterior-only delta +0.0020 and label-aware posterior delta +0.0001. Tier C threshold validation also does not regress versus EEGNet: label-known delta -0.0041, max-probability delta -0.0008, and negative-entropy delta -0.0008. Attacker-refit calibration keeps the learned-MLP bottleneck read as a practical tie: mean delta AUC -0.0025 across 3 attacker seeds, with 2/3 privacy-improving refits.
- Caveat: Raw membership is not attack-invariant: the bottleneck is not the raw winner in every threshold cell, and nonlinear learned attack favors `csp_lda`.

### C02: Bottleneck EEGNet is the best BNCI membership defense across attacks.

- Dataset: `bnci2014_001`
- Verdict: `not-supported`
- Evidence: Adversarial constant wins more raw attack-AUC cells in the shortlist (3.0000 of 6) and has lower mean AUC than bottleneck (0.5166 vs 0.5228).
- Caveat: Bottleneck remains the pragmatic default only because utility and subject-ID matter.

### C03: Use bottleneck EEGNet as the Lee default when subject-ID privacy is part of the objective.

- Dataset: `lee2019_mi`
- Verdict: `qualified-support`
- Evidence: Bottleneck has the lowest Lee subject-ID accuracy (0.2625) and improves task (+0.0146 vs EEGNet). Tier A expanded-subject validation on Lee subjects 1-12 preserves this: subject-ID delta -0.2340, task delta +0.0019, and MLP-membership delta +0.0011. Tier B learned-logistic validation also has no material membership regression: posterior-only delta -0.0019 and label-aware posterior delta -0.0040. Tier C threshold validation narrows the claim: bottleneck regresses on label-known (+0.0145), max-probability (+0.0176), and negative-entropy (+0.0176) membership AUC. Attacker-refit calibration is also unfavorable for a Lee learned-membership claim: bottleneck mean delta AUC +0.0135 across 3 attacker seeds, with 0/3 privacy-improving refits.
- Caveat: Membership winners split by attack family; CSP owns the expanded-subject threshold family but has much higher subject-ID leakage.

### C04: Lee membership privacy has a single best model.

- Dataset: `lee2019_mi`
- Verdict: `not-supported`
- Evidence: The nonlinear learned attack winner is `adversarial_eegnet (constant@1.00)` with mean AUC 0.5077, while threshold-family summaries favor CSP.
- Caveat: Frame Lee as attack-sensitive, not model-invariant.

### C05: Mixup is the PhysioNet nonlinear learned-attack winner.

- Dataset: `physionet_motor_imagery`
- Verdict: `not-supported`
- Evidence: The original four-seed `learned(mlp)` read was already too small: mixup was only +0.0023 AUC versus EEGNet, with 95% CI [-0.0922, +0.0968], improves privacy on 2/4 seeds, and has 0/4 unique rank-1 seeds. The completed random-subset validation independently keeps the claim unresolved: Tier A mixup has mean delta AUC +0.0022, privacy improvement on 3/5 subsets, and mean task delta -0.0365. The completed eight-seed subjects `1-46` extension rejects the apparent win: mean delta AUC +0.0096, privacy improvement on 2/8 seeds, and mean task delta +0.0014. The completed 11-row cross-run holdout also falls short for the nonlinear attacker: mixup MLP delta AUC is -0.0075 with task delta -0.0185, below the pre-registered 0.010 AUC-improvement rule.
- Caveat: Frame PhysioNet mixup under the nonlinear learned attacker as not supported, not merely underpowered.

### C06: Adversarial linear-ramp is the PhysioNet privacy extreme.

- Dataset: `physionet_motor_imagery`
- Verdict: `qualified-support`
- Evidence: It has the best neural-family mean attack rank (1.5000) and largest mean AUC reduction (-0.0409 vs EEGNet).
- Caveat: The task drop is severe (-0.1486), and CSP wins the combined label-free threshold family once included.

### C07: PhysioNet has a stable default defense.

- Dataset: `physionet_motor_imagery`
- Verdict: `not-supported`
- Evidence: Attack-family winners disagree: learned-family AUC favors adversarial linear-ramp, while label-free thresholds favor CSP (0.4902). The completed random-subset validation reinforces the split: Tier B max-probability mixup has mean delta AUC -0.0223 on 4/5 subsets with mean task delta -0.0365, while Tier C label-known CSP-LDA has mean delta AUC -0.0641 on 5/5 subsets with mean task delta +0.0632. The pre-registered non-adversarial defense grid also fails to produce a stable default: `mixup_alpha_0p10` is `attack-specific only` with mean delta AUC +0.0030, `confidence_beta_0p010` is `early stop: cannot reach two-family rule`, and `bottleneck_dim_12` is `early stop: utility boundary failed` despite 2 AUC-win families. The completed 11-row cross-run holdout reaches the same conclusion: mixup reduces AUC on both threshold rows within utility (-0.0166 max-probability, -0.0147 label-known), but not MLP (-0.0075); bottleneck regresses on MLP (+0.0166); and CSP-LDA is a threshold anchor (-0.0555 label-known) rather than a learned-attack defense.
- Caveat: Frame PhysioNet as unresolved and protocol-sensitive. CSP-LDA is a label-known threshold anchor, not a learned-attack default; cross-run wins remain threshold-only or attack-specific.

### C08: The two label-free threshold attacks are redundant at the ranking level.

- Dataset: `all`
- Verdict: `supported`
- Evidence: Winner pairs match across datasets: bnci2014_001: `adversarial_eegnet (constant@1.00)` / `adversarial_eegnet (constant@1.00)`; lee2019_mi: `csp_lda` / `csp_lda`; physionet_motor_imagery: `csp_lda` / `csp_lda`.
- Caveat: This supports ranking redundancy, not that the threshold family is unimportant.

### C09: Adding the nonlinear posterior MLP attacker changes the learned-attack read.

- Dataset: `all`
- Verdict: `supported`
- Evidence: The `learned(mlp)` mean-AUC winners are BNCI `csp_lda`, Lee `adversarial_eegnet (constant@1.00)`, PhysioNet `eegnet`.
- Caveat: PhysioNet's original MLP rank winner does not survive the eight-seed extension, so use this as an attack-sensitivity result rather than a PhysioNet defense claim.

### C10: Federation or the secure-aggregation prototype provides a privacy guarantee.

- Dataset: `physionet_motor_imagery`
- Verdict: `not-supported`
- Evidence: The frozen PhysioNet replication changes mean task balanced accuracy from 0.6574 to 0.5185 (paired delta -0.1389), so the 0.60 utility gate fails. The pairwise-mask simulator matches FedAvg within 2.384e-07 and adds an estimated 0.0649% communication, but it is a functional single-process prototype.
- Caveat: Privacy deltas are diagnostic only. The prototype assumes all clients are present, hidden pairwise secrets, and no collusion; it does not protect against malicious clients, dropout, metadata, output leakage, membership inference, or provide differential privacy.

## Final Editing Rule

Every abstract, contribution, result, and conclusion sentence should be traceable to a `supported` or `qualified-support` claim unless it is explicitly framed as a limitation, guardrail, or future-work item.
