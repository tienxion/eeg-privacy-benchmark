# Shin2017A v1.2 Claim Audit

## Audit decision

The preregistered Shin2017A temporal-generalization study is complete at its
frozen utility stop boundary. Plain compact EEGNet passed the baseline utility
gate and showed detectable cross-session subject identity in its cached
representations. The fixed compact bottleneck EEGNet (`dim=6`) then completed,
but task-utility noninferiority was not established under the preregistered
one-sided confidence-bound rule.

Accordingly, the confirmatory sequence does not support a Shin2017A bottleneck
privacy-improvement claim. The bottleneck subject-identification promotion probe
and secondary membership attacks were not run because the utility stop rule was
reached first. This audit governs project, outreach, resume, manuscript, and any
future release language for v1.2.

## Evidence-to-claim map

| Evidence layer | Result | Claim status |
| --- | --- | --- |
| Cache acquisition and outcome-blind QA | Subjects `3-29`; 81 subject-sessions; 895 selected files; zero objective exclusions | Supported as data/provenance evidence |
| Plain compact EEGNet utility | Mean subject balanced accuracy `0.6341`; 95% subject-bootstrap CI `[0.6008, 0.6705]` | Baseline temporal-utility gate passed |
| Plain EEGNet identity detectability | Mean cross-session subject-ID accuracy `0.3198` versus chance `0.0370`; `+0.2827` above chance | Frozen detectability gate passed for the named linear probe |
| Bottleneck execution | 12/12 fixed `dim=6` jobs and 12/12 six-dimensional feature caches validated | Execution complete |
| Bottleneck utility point estimate | Mean balanced accuracy `0.6137`; paired mean loss `0.02037` versus margin `0.030` | Point-estimate criterion passed |
| Bottleneck utility uncertainty | One-sided 95% upper paired subject-bootstrap bound `0.03071` versus strict `<0.030` requirement | Noninferiority not established |
| Bottleneck identity reduction | Not evaluated because the utility stop rule fired first | No Shin2017A bottleneck identity claim |
| Membership privacy | Not evaluated; secondary stage remained unauthorized and was blocked by the stop rule | No Shin2017A membership claim |

## Approved wording

The following statement is supported:

> In a preregistered 27-subject, three-session Shin2017A validation, plain
> compact EEGNet passed the temporal-utility gate and its cached representations
> supported above-chance cross-session subject identification. The fixed
> six-dimensional bottleneck completed with a mean paired utility loss of
> `0.02037`, but its one-sided 95% upper bound was `0.03071`, narrowly exceeding
> the frozen `<0.030` noninferiority requirement. The study therefore stopped
> before bottleneck privacy promotion or membership evaluation.

For a concise resume or outreach line:

> Designed and executed a preregistered three-session EEG validation with
> checksum-backed data QA, resumable training, cross-session identity probes,
> subject-level bootstrap inference, and a fail-closed claim audit; preserved a
> near-boundary negative noninferiority result instead of retuning after
> evaluation.

## Prohibited or unsupported wording

- Do not say the Shin2017A bottleneck reduced subject-identification leakage; it
  was not evaluated after the utility stop.
- Do not say the bottleneck improves Shin2017A membership privacy; membership
  attacks were not run.
- Do not say the bottleneck was utility-noninferior. Its mean loss was within
  the margin, but the preregistered one-sided bound did not pass.
- Do not characterize the result as a large or catastrophic utility failure.
  The upper bound missed the margin by approximately `0.00071`.
- Do not round `0.03071` to `0.030` and declare a pass.
- Do not interpret detectable identity under one linear probe as proof of a
  general privacy harm, re-identification in deployment, or clinical risk.
- Do not infer that the bottleneck lacks a privacy effect merely because the
  privacy stage was stopped; that effect remains unevaluated under this
  confirmatory sequence.
- Do not add seeds, subjects, alternate bottleneck dimensions, thresholds, or
  attackers to rescue the confirmatory result. Any follow-up must be separately
  labelled exploratory or preregistered validation work.

## Release boundary

The appropriate v1.2 presentation is a reproducible temporal-validation study
with a narrow negative utility result and an honored stop rule. That is useful
evidence about cross-session transfer and protocol sensitivity, but it is not a
new defense claim.

This public record includes the historical contract, aggregate gate artifacts,
and claim audit. It omits authorization records, detailed execution documents,
raw EEG, local feature caches, ignored result directories, and internal handoff
notes.

## Source records

- [Baseline utility gate](./V1_2_SHIN2017A_BASELINE_UTILITY_GATE.md)
- [Plain-EEGNet identity-detectability gate](./V1_2_SHIN2017A_SUBJECT_ID_DETECTABILITY_GATE.md)
- [Bottleneck utility gate](./V1_2_SHIN2017A_BOTTLENECK_UTILITY_GATE.md)

Verify the public historical contract and aggregate decisions with:

```bash
PYTHONPATH=src python scripts/check_shin2017a_public_evidence.py
```
