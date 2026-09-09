# Research log

This is a running development summary after the frozen v1.2 release. Update this
file in place; experiment contracts and reproducible evidence belong in their
own versioned artifacts when released. This log is not a new release, a complete
reproduction bundle, or an update to the published v1.0–v1.2 evidence.

## Current conclusion — 2026-09-09

No new subject-membership privacy defense is supported. Matched Cho2017
inclusion experiments show a positive average output-score change, but both
frozen stages missed their subject-direction gate. More model seeds did not
resolve that limitation. Development now focuses on replication design and
dataset feasibility before more training.

## Follow-on study history

This retrospective index records work done September 3–9. Study numbers below
are internal experiment versions, not public release numbers. The new runners
and full reproducibility bundle are not included in this documentation update.

| Study | Result | Interpretation |
| --- | --- | --- |
| Subject-bag membership, Sep 3 | Fixed-score AUC 0.5389; learned AUC 0.5187. | Neither attack met its frozen validity rule; no defense promotion. |
| Calibrated membership, Sep 3 | A Gaussian reference score initially reached AUC 0.8245, but arbitrary fold-label placebos reached 0.73–0.81. | Membership interpretation retracted: the method recognized cross-seed fold signatures. The original score must not be cited as validated leakage. |
| Matched inclusion v3, Sep 3 | Ten targets, task seeds 13/17, 20 IN fits; mean true-label-logit difference +0.08558; 7/10 targets positive. | Missed the frozen 8/10 direction requirement: `CAUSAL_SIGNAL_NOT_VALIDATED`. |
| New-seed replication v4, Sep 4 | Same ten targets, seeds 19/23, 20 further IN fits; mean +0.11786; again 7/10 positive. | `SEED_REPLICATION_NOT_VALIDATED`. New initializations are not new participants. |
| Four-seed descriptive audit, Sep 4 | Pooled mean +0.10172; 8/10 positive on average, but only 2/10 positive at every seed. | Post-hoc pooling does not reverse either frozen verdict. |
| Cohort-extension feasibility, Sep 9 | Another 40 existing Cho targets would require 160 IN fits, about 44 fit-hours using observed runtimes. | Design only; no new fits. Characterizes a fixed cohort, not untouched independent replication. |

The matched experiments compare an existing OUT model with an IN counterpart
using the same seed, background fitting data and validation roles, adding only
the target's nonquery trials. Both query the same 40 unseen target trials. The
reported effect is IN-minus-OUT mean true-label logit, **not an attack AUC**.

Targets still share fitted backgrounds and OUT models across the design.
Subject-bootstrap intervals cannot be assumed to have population coverage;
whole-fold resampling is only a sensitivity check because those folds overlap
too. Neither failed gate proves privacy, and a positive average score effect
does not establish a general membership attacker or a defense benefit.

## Sep 9 — replication feasibility from existing local caches

The following checks used existing benchmark cohorts, with no data downloads
and no model fitting. They are not untouched confirmation data.

| Dataset | Local structural check | Decision |
| --- | --- | --- |
| PhysioNet motor imagery | 46 subjects, 138 left/right imagery files; 45 eligible annotations per subject. Reserving 20 queries per class leaves only five target training trials. | Cannot reproduce Cho's 160–200-trial training addition at the same query budget. A different dose/query-budget experiment needs its own design. |
| BNCI2014-001 | Nine subjects, all 18 expected MAT files with the expected top-level variable structure. | Available for development; nested labels, query budgets and signal quality not yet checked in this screen. |
| Lee2019 MI | Twelve subjects, all 24 expected MAT files; both phases have consistent labels and 50 trials per class. The benchmark uses only the offline phase: 200 trials across two sessions per subject. | Pooling the two offline sessions supports 40 balanced queries and 160 target training trials. This is a development candidate, not cross-session or untouched confirmation. |

Only PhysioNet runs 4/8/12 represent left/right imagery; runs 6/10/14 are
bilateral hand/foot imagery. See the [provider's run and annotation
mapping](https://archive.physionet.org/pn4/eegmmidb/). Counts are upper bounds before
signal-quality rejection, not final usable-epoch counts. BNCI's MAT inventory
does not certify nested labels or recording integrity.

Lee's deeper check covered 48 offline/online runs, consistent numeric/text/one-hot
labels, channel order, nonoverlapping in-bounds events, and 14,400 sampled
comparisons between stored epochs and continuous recordings. It found labels in
the cached online phases too, but these are excluded from the primary budget to
preserve the existing offline-only benchmark selection. The phases have different
feedback conditions; see the [original study](https://doi.org/10.1093/gigascience/giz002).
Full local file hashes were recorded, without comparison to upstream checksums.
Signal quality and complete epoch-array equality were not assessed.

The budget depends on the design: pooling both offline sessions leaves 160
training trials after 40 queries; a single-session design leaves 60; training on
all of session 1 and querying session 2 provides 100 training trials and leaves
60 session-2 trials unused. Pooling sessions is not a cross-session test.

Next: define the target cohort, disjoint background training and validation roles,
fixed queries, training addition, estimand, and prior-use disclosure, then verify
the actual preprocessing and split construction before new matched effects.
Dataset-source separation alone does not establish statistical independence or
outcome-blind replication.

Existing release claims remain in [RESULTS.md](../RESULTS.md). Dataset provenance
and acquisition guidance remain in [DATASETS.md](../DATASETS.md). No recordings,
participant-level outputs, model weights, or submission materials are included
in this update.
