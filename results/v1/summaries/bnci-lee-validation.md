# BNCI/Lee Validation Results

This report summarizes whichever staged BNCI/Lee validation commands have completed. Lower subject-ID accuracy and lower membership AUC mean less leakage; deltas are paired against the same-scope plain EEGNet row.

Companion files:
- Companion private row tables are not distributed; aggregate values are frozen in this summary.

## Completion

- Overall: `56/56` commands complete.
- Tier A: `16/16` commands complete.
- Tier B: `16/16` commands complete.
- Tier C: `24/24` commands complete.

## Completed Rows

| Tier | Dataset | Probe | Attack/score | Model | Task BA | Privacy metric | Delta privacy | Decision |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `A` | `bnci2014_001` | `membership` | `mlp::posterior_probabilities` | `eegnet` | 0.7199 | 0.5207 | +0.0000 | `baseline` |
| `A` | `bnci2014_001` | `membership` | `mlp::posterior_probabilities` | `bottleneck_eegnet` | 0.7305 | 0.5215 | +0.0007 | `no material membership regression` |
| `A` | `bnci2014_001` | `membership` | `mlp::posterior_probabilities` | `adversarial_eegnet` | 0.5347 | 0.5329 | +0.0122 | `more leaky than eegnet` |
| `A` | `bnci2014_001` | `membership` | `mlp::posterior_probabilities` | `csp_lda` | 0.7070 | 0.4992 | -0.0215 | `privacy-favorable` |
| `A` | `bnci2014_001` | `subject_id` | `subject_id::linear_probe` | `eegnet` | 0.7199 | 0.7106 | +0.0000 | `baseline` |
| `A` | `bnci2014_001` | `subject_id` | `subject_id::linear_probe` | `bottleneck_eegnet` | 0.7305 | 0.4423 | -0.2683 | `supports bottleneck subject-ID claim` |
| `A` | `bnci2014_001` | `subject_id` | `subject_id::linear_probe` | `adversarial_eegnet` | 0.5347 | 0.6246 | -0.0860 | `privacy-favorable` |
| `A` | `bnci2014_001` | `subject_id` | `subject_id::linear_probe` | `csp_lda` | 0.7070 | 0.7047 | -0.0060 | `practical tie` |
| `B` | `bnci2014_001` | `membership` | `logistic_regression::posterior_probabilities` | `eegnet` | 0.7199 | 0.4967 | +0.0000 | `baseline` |
| `B` | `bnci2014_001` | `membership` | `logistic_regression::posterior_probabilities` | `bottleneck_eegnet` | 0.7305 | 0.4987 | +0.0020 | `no material membership regression` |
| `B` | `bnci2014_001` | `membership` | `logistic_regression::posterior_probabilities` | `adversarial_eegnet` | 0.5347 | 0.5348 | +0.0381 | `more leaky than eegnet` |
| `B` | `bnci2014_001` | `membership` | `logistic_regression::posterior_probabilities` | `csp_lda` | 0.7070 | 0.5180 | +0.0213 | `more leaky than eegnet` |
| `B` | `bnci2014_001` | `membership` | `logistic_regression::posterior_probabilities_plus_true_label` | `eegnet` | 0.7199 | 0.4991 | +0.0000 | `baseline` |
| `B` | `bnci2014_001` | `membership` | `logistic_regression::posterior_probabilities_plus_true_label` | `bottleneck_eegnet` | 0.7305 | 0.4992 | +0.0001 | `no material membership regression` |
| `B` | `bnci2014_001` | `membership` | `logistic_regression::posterior_probabilities_plus_true_label` | `adversarial_eegnet` | 0.5347 | 0.5233 | +0.0242 | `more leaky than eegnet` |
| `B` | `bnci2014_001` | `membership` | `logistic_regression::posterior_probabilities_plus_true_label` | `csp_lda` | 0.7070 | 0.5160 | +0.0169 | `more leaky than eegnet` |
| `C` | `bnci2014_001` | `membership` | `threshold::label_known_log_probability` | `eegnet` | 0.7199 | 0.5425 | +0.0000 | `baseline` |
| `C` | `bnci2014_001` | `membership` | `threshold::label_known_log_probability` | `bottleneck_eegnet` | 0.7305 | 0.5384 | -0.0041 | `no material membership regression` |
| `C` | `bnci2014_001` | `membership` | `threshold::label_known_log_probability` | `adversarial_eegnet` | 0.5347 | 0.5297 | -0.0129 | `privacy-favorable` |
| `C` | `bnci2014_001` | `membership` | `threshold::label_known_log_probability` | `csp_lda` | 0.7070 | 0.5039 | -0.0387 | `privacy-favorable` |
| `C` | `bnci2014_001` | `membership` | `threshold::max_probability` | `eegnet` | 0.7199 | 0.5007 | +0.0000 | `baseline` |
| `C` | `bnci2014_001` | `membership` | `threshold::max_probability` | `bottleneck_eegnet` | 0.7305 | 0.4999 | -0.0008 | `no material membership regression` |
| `C` | `bnci2014_001` | `membership` | `threshold::max_probability` | `adversarial_eegnet` | 0.5347 | 0.5078 | +0.0071 | `more leaky than eegnet` |
| `C` | `bnci2014_001` | `membership` | `threshold::max_probability` | `csp_lda` | 0.7070 | 0.4931 | -0.0076 | `practical tie` |
| `C` | `bnci2014_001` | `membership` | `threshold::negative_entropy` | `eegnet` | 0.7199 | 0.5007 | +0.0000 | `baseline` |
| `C` | `bnci2014_001` | `membership` | `threshold::negative_entropy` | `bottleneck_eegnet` | 0.7305 | 0.4999 | -0.0008 | `no material membership regression` |
| `C` | `bnci2014_001` | `membership` | `threshold::negative_entropy` | `adversarial_eegnet` | 0.5347 | 0.5078 | +0.0071 | `more leaky than eegnet` |
| `C` | `bnci2014_001` | `membership` | `threshold::negative_entropy` | `csp_lda` | 0.7070 | 0.4931 | -0.0076 | `practical tie` |
| `A` | `lee2019_mi` | `membership` | `mlp::posterior_probabilities` | `eegnet` | 0.5169 | 0.5067 | +0.0000 | `baseline` |
| `A` | `lee2019_mi` | `membership` | `mlp::posterior_probabilities` | `bottleneck_eegnet` | 0.5188 | 0.5079 | +0.0011 | `no material membership regression` |
| `A` | `lee2019_mi` | `membership` | `mlp::posterior_probabilities` | `adversarial_eegnet` | 0.5138 | 0.4948 | -0.0120 | `privacy-favorable` |
| `A` | `lee2019_mi` | `membership` | `mlp::posterior_probabilities` | `csp_lda` | 0.5617 | 0.5062 | -0.0006 | `practical tie` |
| `A` | `lee2019_mi` | `subject_id` | `subject_id::linear_probe` | `eegnet` | 0.5169 | 0.3781 | +0.0000 | `baseline` |
| `A` | `lee2019_mi` | `subject_id` | `subject_id::linear_probe` | `bottleneck_eegnet` | 0.5188 | 0.1442 | -0.2340 | `supports bottleneck subject-ID claim` |
| `A` | `lee2019_mi` | `subject_id` | `subject_id::linear_probe` | `adversarial_eegnet` | 0.5138 | 0.3635 | -0.0146 | `privacy-favorable` |
| `A` | `lee2019_mi` | `subject_id` | `subject_id::linear_probe` | `csp_lda` | 0.5617 | 0.3442 | -0.0340 | `privacy-favorable` |
| `B` | `lee2019_mi` | `membership` | `logistic_regression::posterior_probabilities` | `eegnet` | 0.5169 | 0.5034 | +0.0000 | `baseline` |
| `B` | `lee2019_mi` | `membership` | `logistic_regression::posterior_probabilities` | `bottleneck_eegnet` | 0.5188 | 0.5015 | -0.0019 | `no material membership regression` |
| `B` | `lee2019_mi` | `membership` | `logistic_regression::posterior_probabilities` | `adversarial_eegnet` | 0.5138 | 0.5000 | -0.0034 | `practical tie` |
| `B` | `lee2019_mi` | `membership` | `logistic_regression::posterior_probabilities` | `csp_lda` | 0.5617 | 0.5174 | +0.0141 | `more leaky than eegnet` |
| `B` | `lee2019_mi` | `membership` | `logistic_regression::posterior_probabilities_plus_true_label` | `eegnet` | 0.5169 | 0.4979 | +0.0000 | `baseline` |
| `B` | `lee2019_mi` | `membership` | `logistic_regression::posterior_probabilities_plus_true_label` | `bottleneck_eegnet` | 0.5188 | 0.4939 | -0.0040 | `no material membership regression` |
| `B` | `lee2019_mi` | `membership` | `logistic_regression::posterior_probabilities_plus_true_label` | `adversarial_eegnet` | 0.5138 | 0.4993 | +0.0014 | `practical tie` |
| `B` | `lee2019_mi` | `membership` | `logistic_regression::posterior_probabilities_plus_true_label` | `csp_lda` | 0.5617 | 0.5074 | +0.0095 | `more leaky than eegnet` |
| `C` | `lee2019_mi` | `membership` | `threshold::label_known_log_probability` | `eegnet` | 0.5169 | 0.5776 | +0.0000 | `baseline` |
| `C` | `lee2019_mi` | `membership` | `threshold::label_known_log_probability` | `bottleneck_eegnet` | 0.5188 | 0.5921 | +0.0145 | `membership regression` |
| `C` | `lee2019_mi` | `membership` | `threshold::label_known_log_probability` | `adversarial_eegnet` | 0.5138 | 0.5542 | -0.0234 | `privacy-favorable` |
| `C` | `lee2019_mi` | `membership` | `threshold::label_known_log_probability` | `csp_lda` | 0.5617 | 0.5156 | -0.0621 | `privacy-favorable` |
| `C` | `lee2019_mi` | `membership` | `threshold::max_probability` | `eegnet` | 0.5169 | 0.4994 | +0.0000 | `baseline` |
| `C` | `lee2019_mi` | `membership` | `threshold::max_probability` | `bottleneck_eegnet` | 0.5188 | 0.5170 | +0.0176 | `membership regression` |
| `C` | `lee2019_mi` | `membership` | `threshold::max_probability` | `adversarial_eegnet` | 0.5138 | 0.4940 | -0.0054 | `practical tie` |
| `C` | `lee2019_mi` | `membership` | `threshold::max_probability` | `csp_lda` | 0.5617 | 0.4454 | -0.0540 | `privacy-favorable` |
| `C` | `lee2019_mi` | `membership` | `threshold::negative_entropy` | `eegnet` | 0.5169 | 0.4994 | +0.0000 | `baseline` |
| `C` | `lee2019_mi` | `membership` | `threshold::negative_entropy` | `bottleneck_eegnet` | 0.5188 | 0.5170 | +0.0176 | `membership regression` |
| `C` | `lee2019_mi` | `membership` | `threshold::negative_entropy` | `adversarial_eegnet` | 0.5138 | 0.4940 | -0.0054 | `practical tie` |
| `C` | `lee2019_mi` | `membership` | `threshold::negative_entropy` | `csp_lda` | 0.5617 | 0.4454 | -0.0540 | `privacy-favorable` |

## Current Read

- `bnci2014_001` bottleneck subject-ID delta is `-0.2683` with task delta `+0.0106`: `supports bottleneck subject-ID claim`.
- `lee2019_mi` bottleneck subject-ID delta is `-0.2340` with task delta `+0.0019`: `supports bottleneck subject-ID claim`.
- `bnci2014_001` bottleneck MLP-membership delta is `+0.0007` with task delta `+0.0106`: `no material membership regression`.
- `lee2019_mi` bottleneck MLP-membership delta is `+0.0011` with task delta `+0.0019`: `no material membership regression`.

## Pending Rows
