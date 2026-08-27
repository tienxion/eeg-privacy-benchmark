# Cho2017 v1.1 Claim Audit

## Audit decision

The preregistered Cho2017 confirmatory extension is complete. It supports a
task-utility noninferiority statement for the fixed compact bottleneck EEGNet
(`dim=6`), but it does not support a membership-privacy improvement claim.

This audit governs project, outreach, resume, and future release language for
the Cho2017 extension. The public v1 evidence boundary remains unchanged.

## Evidence-to-claim map

| Evidence layer | Result | Claim status |
| --- | --- | --- |
| Cache acquisition and objective QA | 50/50 confirmatory subjects passed; zero objective exclusions | Supported as data/provenance evidence |
| Plain compact EEGNet utility | Mean subject BA `0.6287`; 95% subject-bootstrap CI `[0.6012, 0.6585]` | Baseline utility gate passed |
| Bottleneck utility | Mean BA `0.6256`; paired loss `0.0032`; one-sided 95% upper bound `0.0075` against margin `0.030` | Utility noninferiority supported |
| Membership attacks | 600/600 frozen cache-only jobs validated across threshold, logistic-regression, and MLP families | Attack evaluation complete |
| Membership promotion | Family reductions `-0.00003`, `+0.00002`, and `+0.00140`; all adjusted intervals cross zero; `0/3` families qualify versus `2/3` required | No membership-privacy promotion |
| Subject identification | Secondary diagnostic was not promoted by this confirmatory sequence | No new Cho2017 subject-ID claim |

## Approved wording

The following statement is supported:

> On 50 previously untouched Cho2017 subjects, the fixed compact bottleneck
> EEGNet was task-utility noninferior to architecture-matched compact EEGNet
> under the preregistered `0.030` margin. A subsequent 600-job cache-only
> membership evaluation did not meet the preregistered privacy-promotion rule.

For a concise resume or outreach line:

> Built and executed a preregistered 50-subject Cho2017 validation with
> checksum-backed data QA, resumable model/attack runners, subject-level
> bootstrap inference, and multiplicity-controlled claim auditing; preserved a
> negative privacy result rather than tuning after evaluation.

## Prohibited or unsupported wording

- Do not say the bottleneck improves Cho2017 membership privacy.
- Do not say the bottleneck is a Cho2017 privacy defense without immediately
  limiting the statement to utility noninferiority and the failed promotion
  gate.
- Do not interpret near-chance mean AUC as proof that membership leakage is
  absent.
- Do not claim the two task models are privacy-equivalent; the study tested an
  improvement rule, not an equivalence margin.
- Do not make a new Cho2017 subject-identification or cross-session identity
  claim from this sequence.
- Do not add a defense, bottleneck dimension, attacker, fold, seed, or threshold
  to the confirmatory result after seeing the outcome. Any such work is a
  separately labelled exploratory study.
- Do not imply that the four subjects never assigned to an attacker-evaluation
  role were removed after results. Subjects `6,13,17,40` were absent from the
  inferential membership sample because of the frozen per-fold role assignment.

## Release boundary

The appropriate v1.1 presentation is a reproducible confirmatory extension with
mixed results: successful baseline transfer, successful bottleneck utility
noninferiority, and unsuccessful membership-privacy promotion. This is a
stronger and more credible research record than presenting only favorable
attack variants.

Public v1.1 summaries and release notes must follow this audit. Raw EEG, local
posterior arrays, model checkpoints, and generated attack-result directories
are outside the public release boundary.

## Source records

- [Validation contract](./V1_1_CHO2017_VALIDATION_CONTRACT.md)
- [Cache QA](./V1_1_CHO2017_CACHE_QA.md)
- [Baseline utility gate](./V1_1_CHO2017_BASELINE_GATE.md)
- [Bottleneck utility gate](./V1_1_CHO2017_BOTTLENECK_GATE.md)
- [Membership-privacy gate](./V1_1_CHO2017_PRIVACY_GATE.md)

Reproduce the three tracked decisions with:

```bash
PYTHONPATH=src .venv/bin/python scripts/freeze_cho2017_baseline_gate.py --check
PYTHONPATH=src .venv/bin/python scripts/freeze_cho2017_bottleneck_gate.py --check
PYTHONPATH=src .venv/bin/python scripts/freeze_cho2017_privacy_gate.py --check
```
