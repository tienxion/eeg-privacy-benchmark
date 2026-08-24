# Cross-Dataset Defense Summary

As of the current local-cache benchmark state, the completed four-seed results support a dataset-specific defense story rather than a single universally best method.

Companion artifact: attack_robustness_summary_v1.md consolidates the matched BNCI, Lee, and PhysioNet family results by attack type, with the plot at attack_robustness_auc_plot_v1.png.

Nonlinear learned-attack check: mlp_posterior_attack_summary_v1.md adds `attack_type=mlp`, `score_type=posterior_probabilities`. It materially changes the learned-attack winners on all three datasets: BNCI flips to `csp_lda`, Lee flips to `adversarial_eegnet (constant@1.00)`, and PhysioNet `subjects 1-46` flips from adversarial `linear_ramp@1.0` to `mixup_eegnet (alpha=0.20)`.

Seed-stability check: seed_stability_summary_v1.md adds paired same-seed deltas against plain EEGNet. It confirms that the PhysioNet `learned(mlp)` mixup edge is effectively a tie (`-0.0003` mean AUC vs EEGNet, `2/4` privacy-improving seeds, `0/4` rank-1 seeds), while BNCI's `learned(mlp)` `csp_lda` win is more credible (`-0.0097`, `3/4`, `2/4`) and Lee remains genuinely attack-sensitive.

Paper-claims audit: paper_claims_audit_v1.md separates supported, qualified, and unsupported paper claims. The short version is: BNCI and Lee bottleneck defaults are qualified claims, PhysioNet has no stable default, the MLP attacker is worth reporting, and PhysioNet mixup under `learned(mlp)` should be called a tie rather than a win.

Practical shortlist: defense_shortlist_summary_v1.md compresses the same benchmark state into paper-facing defaults, privacy extremes, and fallback baselines, with the matrix at defense_shortlist_matrix_v1.csv.

Classical anchor: csp_anchor_summary_v1.md adds `csp_lda` back into the matched cross-dataset read. It leaves BNCI unchanged, but it matters on Lee and PhysioNet: CSP wins all three Lee threshold attacks and both label-free PhysioNet threshold attacks, while still failing the learned attacks.

Attack-family view: attack_family_split_summary_v1.md compresses that disagreement into one table. It makes the current paper-safe read explicit: BNCI still defaults to the bottleneck, Lee defaults to the bottleneck only if subject-ID matters, and PhysioNet remains unresolved because learned and threshold attackers prefer different model families.

## At a Glance

| Dataset | Protocol | Strongest privacy axis available | Best subject-ID defense | Best membership defense | Current recommendation |
| --- | --- | --- | --- | --- | --- |
| `bnci2014_001` | `cross_session` | subject-ID + membership | `bottleneck_eegnet (dim=6)` | strongly attack-sensitive: bottleneck wins the two linear learned attacks, plain EEGNet wins the nonlinear learned attack inside the neural family, `csp_lda` wins once the classical anchor is included, and `adversarial_eegnet (constant@1.00)` wins all threshold attacks | `bottleneck_eegnet (dim=6)` with a stronger learned-attack caveat |
| `lee2019_mi` | `cross_session` | subject-ID + membership | `bottleneck_eegnet (dim=6)` | attack-sensitive: `eegnet` wins the two linear learned attacks, `adversarial_eegnet (constant@1.00)` wins the nonlinear learned attack, and `csp_lda` wins all threshold attacks | `bottleneck_eegnet (dim=6)` only if subject-ID matters |
| `physionet_motor_imagery` | `cross_subject` | membership only | N/A | now split three ways: `linear_ramp@1.0` wins the two linear learned attacks and the threshold family inside the neural slice, `mixup_eegnet (alpha=0.20)` only ties plain EEGNet under the nonlinear learned attack, and `csp_lda` wins the threshold family once the classical anchor is included | treat PhysioNet as unresolved; the nonlinear learned-attack winner is too small to claim without more repeats |

## BNCI 2014-001

- `eegnet`: task `0.6819`, subject-ID `0.7088`, membership AUC `0.5024`
- `adversarial_eegnet (constant@1.00)`: task `0.5373`, subject-ID `0.6141`, membership AUC `0.5264`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5473`, subject-ID `0.6619`, membership AUC `0.5038`
- `bottleneck_eegnet (dim=6)`: task `0.7409`, subject-ID `0.4618`, membership AUC `0.4996`

Takeaway: under the learned posterior-plus-label attack, the bottleneck is the clear BNCI winner. It improves task accuracy, strongly reduces subject leakage, and has the lowest membership AUC of the four widened models. The adversarial schedules do reduce subject leakage relative to plain EEGNet, but `constant@1.0` fails badly on membership there and `linear_ramp@1.0` still does not beat the bottleneck overall.

#### Learned-attack robustness check (`posterior_probabilities`)

- `eegnet`: task `0.6819`, membership AUC `0.5043`
- `bottleneck_eegnet (dim=6)`: task `0.7409`, membership AUC `0.5028`
- `adversarial_eegnet (constant@1.00)`: task `0.5373`, membership AUC `0.5455`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5473`, membership AUC `0.5295`

Takeaway: the posterior-only learned attack is much less hostile to the BNCI bottleneck line than the threshold family. `bottleneck_eegnet (dim=6)` regains a slight raw membership edge over plain EEGNet and stays far better than both adversarial baselines. That makes the BNCI bottleneck problem look threshold-specific rather than universal, even though the raw learned-attack lead is still small enough that it should not be oversold as attack-invariant.

#### Threshold-attack robustness check (`max_probability`)

- `eegnet`: task `0.6819`, membership AUC `0.5199`
- `bottleneck_eegnet (dim=6)`: task `0.7409`, membership AUC `0.5248`
- `adversarial_eegnet (constant@1.00)`: task `0.5373`, membership AUC `0.4957`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5473`, membership AUC `0.5025`

Takeaway: the BNCI bottleneck story is not attack-robust on raw membership AUC. Under the label-free threshold attack, `bottleneck_eegnet (dim=6)` becomes the most leaky model in this four-model family, while `adversarial_eegnet (constant@1.00)` becomes the raw privacy winner. The bottleneck still keeps the strongest task and subject-ID profile, so it remains the pragmatic BNCI defense recommendation, but the membership claim now needs an explicit attack-sensitivity caveat.

Label-free threshold note: the matched `negative_entropy` rerun is now complete on BNCI and is effectively identical to `max_probability`. The ranking is unchanged, and the AUCs match to about `1e-5`: `eegnet 0.5199`, `bottleneck_eegnet (dim=6) 0.5248`, `adversarial_eegnet (constant@1.00) 0.4957`, `adversarial_eegnet (linear_ramp@1.00, ramp=0.40) 0.5025`.

#### Threshold-attack robustness check (`label_known_log_probability`)

- `eegnet`: task `0.6819`, membership AUC `0.5564`
- `bottleneck_eegnet (dim=6)`: task `0.7409`, membership AUC `0.5650`
- `adversarial_eegnet (constant@1.00)`: task `0.5373`, membership AUC `0.5178`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5473`, membership AUC `0.5475`

Takeaway: the stronger label-known threshold attack confirms the BNCI caveat rather than rescuing the bottleneck line. `bottleneck_eegnet (dim=6)` is again the most leaky model in the main four-model family, and `adversarial_eegnet (constant@1.00)` is again the raw privacy winner. So BNCI’s bottleneck recommendation remains justified by task and subject-ID, not by an attack-robust raw membership lead.

## Lee2019 MI

- `eegnet`: task `0.5012`, subject-ID `0.5267`, membership AUC `0.4910`
- `adversarial_eegnet (constant@1.00)`: task `0.5204`, subject-ID `0.4558`, membership AUC `0.5071`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5129`, subject-ID `0.4950`, membership AUC `0.5015`
- `bottleneck_eegnet (dim=6)`: task `0.5158`, subject-ID `0.2625`, membership AUC `0.4964`

Takeaway: Lee matches BNCI on the main conclusion under the learned attack. The bottleneck is the strongest overall defense because it produces the largest subject-ID reduction by far while keeping near-chance membership leakage. The one caveat is that plain EEGNet has the lowest Lee membership AUC by a small margin, so the bottleneck is a pragmatic best model rather than a clean win on every metric.

#### Learned-attack robustness check (`posterior_probabilities`)

- `eegnet`: task `0.5012`, membership AUC `0.5000`
- `bottleneck_eegnet (dim=6)`: task `0.5158`, membership AUC `0.5039`
- `adversarial_eegnet (constant@1.00)`: task `0.5204`, membership AUC `0.5206`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5129`, membership AUC `0.5167`

Takeaway: the posterior-only learned attack keeps Lee cleaner than the threshold family as well. Plain EEGNet remains the raw privacy winner, but `bottleneck_eegnet (dim=6)` stays close to chance and still clearly beats both adversarial baselines. So Lee’s bottleneck recommendation remains a subject-ID-first compromise rather than a claim that the bottleneck is always the lowest-leakage model.

#### Threshold-attack robustness check (`max_probability`)

- `eegnet`: task `0.5012`, membership AUC `0.5307`
- `bottleneck_eegnet (dim=6)`: task `0.5158`, membership AUC `0.5241`
- `adversarial_eegnet (constant@1.00)`: task `0.5204`, membership AUC `0.5156`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5129`, membership AUC `0.5280`

Takeaway: Lee is also attack-sensitive, but less sharply than BNCI. The bottleneck remains better than plain EEGNet on this threshold attack, yet it is no longer the raw privacy winner; `adversarial_eegnet (constant@1.00)` takes the lowest AUC. That keeps the bottleneck as the best overall Lee defense because the subject-ID reduction is still much stronger, but it weakens any claim that Lee membership privacy is cleanly bottleneck-led across attacks.

Label-free threshold note: the completed low-CPU `negative_entropy` rerun is effectively identical to `max_probability` on Lee as well. The ranking is unchanged, and the AUCs match to about `1e-5`: `eegnet 0.5307`, `bottleneck_eegnet (dim=6) 0.5241`, `adversarial_eegnet (constant@1.00) 0.5156`, `adversarial_eegnet (linear_ramp@1.00, ramp=0.40) 0.5280`.

#### Threshold-attack robustness check (`label_known_log_probability`)

- `eegnet`: task `0.5012`, membership AUC `0.6054`
- `bottleneck_eegnet (dim=6)`: task `0.5158`, membership AUC `0.5777`
- `adversarial_eegnet (constant@1.00)`: task `0.5204`, membership AUC `0.5788`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5129`, membership AUC `0.5842`

Takeaway: Lee is mixed but materially better-behaved than BNCI. Under the label-known threshold attack, `bottleneck_eegnet (dim=6)` narrowly becomes the raw privacy winner, edging `adversarial_eegnet (constant@1.00)` while also beating plain EEGNet clearly. So Lee’s bottleneck story survives this stronger threshold attack better than BNCI’s, even though the max-probability threshold attack still prefers the adversarial constant model.

## PhysioNet Motor Imagery

- Valid protocol here is `cross_subject`, not `cross_session`, so the benchmark currently supports membership comparison but not a comparable subject-ID probe.
- `eegnet`: task `0.4960`, membership AUC `0.5524`
- `label_smoothing_eegnet (eps=0.10)`: task `0.5077`, membership AUC `0.5483`
- `adversarial_eegnet (constant@1.00)`: task `0.4970`, membership AUC `0.5352`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.4975`, membership AUC `0.5232`
- `adversarial_eegnet (linear_ramp@0.75, ramp=0.40)`: task `0.4758`, membership AUC `0.5240`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.4758`, membership AUC `0.5206`
- `adversarial_eegnet (linear_ramp@0.55, ramp=0.40)`: task `0.4758`, membership AUC `0.5213`
- `adversarial_eegnet (linear_ramp@0.50, ramp=0.40)`: task `0.4434`, membership AUC `0.5114`
- `bottleneck_eegnet (dim=6)`: task `0.4988`, membership AUC `0.6428`

Takeaway: PhysioNet is the transfer failure for the bottleneck line and the one dataset where adversarial training looked worth keeping early. On the smallest completed slice (`subjects 1-4`), the adversarial frontier initially looked like `linear_ramp@0.50` for pure privacy and `linear_ramp@0.60` for compromise. The widened checks below changed that ranking materially, so this small-slice result should be treated as exploratory rather than definitive. The small-slice bottleneck result fails badly on membership; the wider `1-44` bottleneck check is more nuanced but still not a privacy win.

### Widened PhysioNet Check (`subjects 1-6`, seeds `13,17,19,23`)

- `eegnet`: task `0.4743`, membership AUC `0.5372`
- `label_smoothing_eegnet (eps=0.10)`: task `0.4743`, membership AUC `0.5373`
- `adversarial_eegnet (constant@1.00)`: task `0.4604`, membership AUC `0.5271`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.4670`, membership AUC `0.5156`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.4930`, membership AUC `0.5314`
- `adversarial_eegnet (linear_ramp@0.50, ramp=0.40)`: task `0.4930`, membership AUC `0.5313`

Takeaway: the widened `1-6` check no longer sits near chance for any of these models, but it still favors the adversarial line over plain EEGNet. `linear_ramp@1.00` gives the lowest widened membership AUC (`0.5156`), but it pays a task-accuracy cost versus the tuned `0.50/0.60` settings (`0.4670` vs `0.4930`). `constant@1.0` is a middle point on privacy (`0.5271`) but is worse than both tuned tradeoff settings on utility. `linear_ramp@0.50` and `0.60` remain effectively tied on the utility/privacy tradeoff, and `label_smoothing_eegnet` is still indistinguishable from plain EEGNet on the widened slice.

### Widened PhysioNet Sanity Check (`subjects 1-8`, seeds `13,17,19,23`)

- `eegnet`: task `0.4796`, membership AUC `0.5452`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.4884`, membership AUC `0.5330`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5199`, membership AUC `0.5119`
- `adversarial_eegnet (constant@1.00)`: task `0.5575`, membership AUC `0.5115`

Takeaway: the completed four-seed `1-8` sanity check now resolves most of the earlier ambiguity. On this larger slice, both weight-`1.0` adversarial schedules beat plain EEGNet and `linear_ramp@0.60` on task accuracy and membership AUC, and `constant@1.00` edges `linear_ramp@1.00` on both metrics (`0.5575` / `0.5115` vs `0.5199` / `0.5119`). That makes `constant@1.00` the current leader on the largest completed PhysioNet slice, while also showing that the best schedule is not stable across widened subsets.

### Widened PhysioNet Check (`subjects 1-10`)

#### Two seeds (`13,17`)

- `eegnet`: task `0.4923`, membership AUC `0.4920`
- `adversarial_eegnet (constant@1.00)`: task `0.5212`, membership AUC `0.5968`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.4864`, membership AUC `0.4936`

#### Four seeds (`13,17,19,23`)

- `eegnet`: task `0.4632`, membership AUC `0.4977`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.4868`, membership AUC `0.4938`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.4811`, membership AUC `0.4871`
- `adversarial_eegnet (constant@1.00)`: task `0.4998`, membership AUC `0.5391`

Takeaway: `subjects 1-10` is the strongest widened PhysioNet check so far because it rejects the flashy `constant@1.0` result from `subjects 1-8` twice, once at two seeds and again at four. On the matched four-seed `1-10` slice, both ramp schedules are directionally better than plain EEGNet, but the gains are modest: `linear_ramp@1.0` gives the lowest AUC (`0.4871`) while `linear_ramp@0.6` gives the highest task accuracy among the non-failing schedules (`0.4868`). `constant@1.0` keeps a utility gain but clearly leaks more than plain EEGNet (`0.5391` vs `0.4977` AUC). That leaves `linear_ramp@1.0` as the safest default and `linear_ramp@0.6` as a near-tied alternative rather than a distinct winner.

### Widened PhysioNet Check (`subjects 1-12`)

#### Two seeds (`13,17`)

- `eegnet`: task `0.4808`, membership AUC `0.5769`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.4978`, membership AUC `0.5325`
- `adversarial_eegnet (constant@1.00)`: task `0.5052`, membership AUC `0.5415`

#### Four seeds (`13,17,19,23`)

- `eegnet`: task `0.4751`, membership AUC `0.5370`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.4959`, membership AUC `0.5201`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.4867`, membership AUC `0.5169`
- `adversarial_eegnet (constant@1.00)`: task `0.4806`, membership AUC `0.5165`

Takeaway: the widened `1-12` four-seed check still favors the adversarial line over plain EEGNet, and it now restores the same basic tradeoff shape seen on `subjects 1-10`. `linear_ramp@1.0` keeps the lowest AUC (`0.5169`), `linear_ramp@0.6` keeps the highest task accuracy among the non-failing schedules (`0.4959`), and `constant@1.0` sits in between on this slice even though it failed badly on `subjects 1-10`. That keeps `linear_ramp@1.0` as the safer default and `linear_ramp@0.6` as the utility-leaning alternative, while reinforcing that the exact schedule ranking is still unstable under widening.

### Widened PhysioNet Check (`subjects 1-14`)

#### Two seeds (`13,17`)

- `eegnet`: task `0.4487`, membership AUC `0.5647`
- `adversarial_eegnet (constant@1.00)`: task `0.4789`, membership AUC `0.5566`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.4787`, membership AUC `0.6289`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.4708`, membership AUC `0.6321`

#### Four seeds (`13,17,19,23`)

- `eegnet`: task `0.4532`, membership AUC `0.5136`
- `adversarial_eegnet (constant@1.00)`: task `0.5005`, membership AUC `0.5337`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.4842`, membership AUC `0.5479`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.4934`, membership AUC `0.5586`

Takeaway: the widest completed PhysioNet slice breaks the earlier adversarial-default story. On matched four seeds, all three adversarial schedules are worse than plain EEGNet on membership AUC, even though they improve task accuracy. `constant@1.0` is the least-bad adversarial point at `0.5337`, but it still leaks more than plain EEGNet at `0.5136`, while both ramp schedules are worse again (`0.5479`, `0.5586`). That means the current honest read is no longer “pick the best adversarial schedule”; it is “PhysioNet is unresolved, and the apparent adversarial gains through `subjects 1-12` do not survive the next widening pass.”

### Widened PhysioNet Check (`subjects 1-16`)

#### Two seeds (`13,17`)

- `eegnet`: task `0.4785`, membership AUC `0.5014`
- `adversarial_eegnet (constant@1.00)`: task `0.4947`, membership AUC `0.5133`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5002`, membership AUC `0.4931`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.5001`, membership AUC `0.4914`

Takeaway: the next widened slice immediately reopens the ramped line, but only at two seeds. Both ramp settings are slightly better than plain EEGNet on utility and membership, while `constant@1.0` is still worse than plain EEGNet on privacy. The important constraint is that this does not override the `subjects 1-14` four-seed result. The right interpretation is still instability under widening, not a restored adversarial win.

#### Four seeds (`13,17,19,23`)

- `eegnet`: task `0.4963`, membership AUC `0.5113`
- `adversarial_eegnet (constant@1.00)`: task `0.4956`, membership AUC `0.5005`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5002`, membership AUC `0.4923`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.5014`, membership AUC `0.4920`

Takeaway: the completed `subjects 1-16` four-seed family strengthens the adversarial line again. `linear_ramp@0.60` is the best completed `1-16` point by a hair (`0.5014` / `0.4920`), `linear_ramp@1.0` is essentially tied just behind it (`0.5002` / `0.4923`), `constant@1.0` sits in the middle (`0.4956` / `0.5005`), and plain EEGNet is worst of the four on membership (`0.5113`). That is enough to keep the PhysioNet story unresolved rather than closed negative, but it is still not enough to promote a stable default because the previous `subjects 1-14` four-seed slice contradicted it sharply.

### Widened PhysioNet Check (`subjects 1-18`)

#### Two seeds (`13,17`)

- `eegnet`: task `0.5269`, membership AUC `0.5066`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5325`, membership AUC `0.5019`
- `adversarial_eegnet (constant@1.00)`: task `0.5123`, membership AUC `0.4949`

#### Four seeds (`13,17,19,23`)

- `eegnet`: task `0.5039`, membership AUC `0.5004`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5188`, membership AUC `0.4933`
- `adversarial_eegnet (constant@1.00)`: task `0.4993`, membership AUC `0.4929`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.4968`, membership AUC `0.4976`

Takeaway: the widest completed PhysioNet slice swings back toward the adversarial line. On matched four seeds, all three adversarial schedules now beat plain EEGNet on membership AUC. `constant@1.0` is narrowly best on privacy (`0.4929`), `linear_ramp@1.0` is essentially tied just behind it (`0.4933`) with the best task accuracy (`0.5188`), and `linear_ramp@0.6` is weaker than both weight-`1.0` schedules (`0.4976`). That does not resolve PhysioNet, because the `subjects 1-14` four-seed slice still contradicted it sharply, but it does mean the strongest current PhysioNet read is no longer purely negative.

### Widened PhysioNet Check (`subjects 1-20`)

#### Two seeds (`13,17`)

- `eegnet`: task `0.5198`, membership AUC `0.5126`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5198`, membership AUC `0.5203`
- `adversarial_eegnet (constant@1.00)`: task `0.4924`, membership AUC `0.5129`

Takeaway: the current widest completed PhysioNet slice weakens the reopened adversarial story again. On the matched `subjects 1-20` two-seed slice, `linear_ramp@1.0` is essentially tied with plain EEGNet on task and slightly worse on membership (`0.5203` vs `0.5126`), while `constant@1.0` is effectively tied with plain EEGNet on membership (`0.5129` vs `0.5126`) but worse on task. This is not strong enough to overturn the completed `subjects 1-18` four-seed advantage for the adversarial family, but it is enough to keep PhysioNet unresolved under widening.

#### Four seeds (`13,17,19,23`)

- `eegnet`: task `0.5013`, membership AUC `0.4986`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.4728`, membership AUC `0.5041`
- `adversarial_eegnet (constant@1.00)`: task `0.4492`, membership AUC `0.5052`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.4762`, membership AUC `0.5099`

Takeaway: the stronger `subjects 1-20` four-seed rerun overturns the noisy two-seed wobble and reestablishes the negative PhysioNet read at the widest completed slice. Plain EEGNet is now best on both task and membership across the completed `1-20` family, with AUC `0.4986` versus `0.5041` for `linear_ramp@0.6`, `0.5052` for `constant@1.0`, and `0.5099` for `linear_ramp@1.0`. That means the widest completed PhysioNet slice no longer supports the reopened adversarial story from `subjects 1-18`.

### Widened PhysioNet Check (`subjects 1-22`)

#### Two seeds (`13,17`)

- `eegnet`: task `0.4882`, membership AUC `0.5819`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5132`, membership AUC `0.5545`
- `adversarial_eegnet (constant@1.00)`: task `0.5203`, membership AUC `0.5540`

Takeaway: the next widening pass flips again relative to the completed `1-20` four-seed slice. On the matched `subjects 1-22` two-seed checkpoint, both weight-`1.0` adversarial schedules beat plain EEGNet on task and membership, with `constant@1.0` narrowly ahead of `linear_ramp@1.0` on privacy and clearly ahead on task. That is still only a two-seed result, so it is not strong enough to overturn the negative `1-20` four-seed slice on its own, but it does keep the PhysioNet widening story unstable.

#### Four seeds (`13,17,19,23`)

- `eegnet`: task `0.5073`, membership AUC `0.5436`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.4727`, membership AUC `0.5097`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.4882`, membership AUC `0.5125`
- `adversarial_eegnet (constant@1.00)`: task `0.5001`, membership AUC `0.5241`

Takeaway: the stronger `subjects 1-22` four-seed rerun preserves the reopened adversarial line, but not symmetrically. The plain `1-22` baseline stays quite leaky at AUC `0.5436`. All three adversarial schedules beat that, with `linear_ramp@0.6` giving the lowest AUC at `0.5097`, `linear_ramp@1.0` close behind at `0.5125`, and `constant@1.0` weaker on privacy at `0.5241`. The tradeoff is utility: `linear_ramp@0.6` gives up the most task accuracy, while `constant@1.0` keeps the best task of the three adversarial settings. So the widened `1-22` slice no longer looks like a clean negative repeat of `1-20`; it reopens the adversarial family again, while still leaving the schedule ranking unstable under widening.

### Widened PhysioNet Check (`subjects 1-24`)

#### Two seeds (`13,17`)

- `eegnet`: task `0.5122`, membership AUC `0.5229`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5445`, membership AUC `0.4794`
- `adversarial_eegnet (constant@1.00)`: task `0.5135`, membership AUC `0.4985`

Takeaway: the first `subjects 1-24` checkpoint strongly reopens the adversarial line. Both weight-`1.0` adversarial schedules beat plain EEGNet on membership, and `linear_ramp@1.0` is clearly ahead of `constant@1.0` on both privacy and task at two seeds.

#### Four seeds (`13,17,19,23`)

- `eegnet`: task `0.5016`, membership AUC `0.5195`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.5132`, membership AUC `0.5094`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5095`, membership AUC `0.5046`
- `adversarial_eegnet (constant@1.00)`: task `0.5225`, membership AUC `0.5144`

Takeaway: the stronger `subjects 1-24` four-seed reruns keep the adversarial family ahead of plain EEGNet, but sharpen the tradeoff. `linear_ramp@1.0` is the lowest-leakage point at AUC `0.5046`. `linear_ramp@0.6` is close behind on privacy at `0.5094` and keeps slightly better task than `linear_ramp@1.0`. `constant@1.0` gives the best task accuracy at `0.5225`, but leaks more than both ramped schedules at `0.5144`. So the widest completed slice currently still favors adversarial training overall, but not a single schedule on both objectives.

### Widened PhysioNet Check (`subjects 1-26`)

#### Two seeds (`13,17`)

- `eegnet`: task `0.5301`, membership AUC `0.5253`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.5170`, membership AUC `0.5510`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5103`, membership AUC `0.5235`
- `adversarial_eegnet (constant@1.00)`: task `0.5023`, membership AUC `0.5380`

Takeaway: the next widening pass weakens the reopened `1-24` story again. At `subjects 1-26`, plain EEGNet comes back out on top at two seeds. `linear_ramp@1.0` is only slightly worse than plain EEGNet on privacy but clearly worse on task, while `constant@1.0` and `linear_ramp@0.6` are both materially worse on privacy. So the current widest checkpoint is negative again, but only at two seeds. A four-seed plain rerun is active, so this slice is not yet strong enough to replace the completed `1-24` four-seed family as the main PhysioNet reference point.

#### Four seeds (`13,17,19,23`)

- `eegnet`: task `0.5216`, membership AUC `0.5139`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.5090`, membership AUC `0.5141`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5047`, membership AUC `0.5046`
- `adversarial_eegnet (constant@1.00)`: task `0.5023`, membership AUC `0.5207`

Takeaway: the stronger `subjects 1-26` reruns soften the negative two-seed read, but do not turn it into another broad reopening like `1-24`. `linear_ramp@1.0` is the only adversarial setting that improves privacy over plain EEGNet, lowering AUC from `0.5139` to `0.5046`, but it does so with lower task accuracy. `linear_ramp@0.6` ends up essentially tied with plain EEGNet on privacy and worse on task, while `constant@1.0` is worse than plain EEGNet on both privacy and task. So the current `1-26` four-seed slice is mixed: the ramped line is still viable, but only in the narrower `linear_ramp@1.0` form.

### Widened PhysioNet Check (`subjects 1-28`)

#### Two seeds (`13,17`)

- `eegnet`: task `0.4779`, membership AUC `0.5195`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.4735`, membership AUC `0.5339`
- `adversarial_eegnet (constant@1.00)`: task `0.5094`, membership AUC `0.4889`

Takeaway: the first `subjects 1-28` checkpoint reverses the `1-26` read sharply. `constant@1.0` beats plain EEGNet on both utility and privacy, while `linear_ramp@1.0` is worse than plain EEGNet on both. So `1-28` initially looks like a constant-specific reopening rather than a broad adversarial win.

#### Four seeds (`13,17,19,23`)

- `eegnet`: task `0.4715`, membership AUC `0.5192`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.4860`, membership AUC `0.4867`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.4764`, membership AUC `0.5214`
- `adversarial_eegnet (constant@1.00)`: task `0.4873`, membership AUC `0.4945`

Takeaway: the stronger `subjects 1-28` four-seed reruns keep the reopening, but only for two of the three schedules. `linear_ramp@0.6` is now the best privacy point at AUC `0.4867`, and `constant@1.0` is a close second at `0.4945` while also giving the best task accuracy. `linear_ramp@1.0` does not hold up here: it is slightly worse than plain EEGNet on privacy (`0.5214` vs `0.5192`). So the current widest completed slice still favors the adversarial family overall, but not the same schedule that looked best at `1-26`.

### Widened PhysioNet Check (`subjects 1-30`)

#### Two seeds (`13,17`)

- `eegnet`: task `0.5356`, membership AUC `0.4882`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.4781`, membership AUC `0.4805`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.4964`, membership AUC `0.4944`
- `adversarial_eegnet (constant@1.00)`: task `0.4920`, membership AUC `0.4852`

Takeaway: the first `subjects 1-30` checkpoint narrows the `1-28` reopening. Plain EEGNet is now the highest-utility model by a large margin and is already below chance on membership AUC. `linear_ramp@0.6` still gives the lowest AUC (`0.4805`), but the improvement over plain EEGNet is small (`-0.0076`) and costs substantial task accuracy (`-0.0575`). `constant@1.0` is the middle privacy point (`0.4852`) with less task loss than `0.6`, while `linear_ramp@1.0` is worse than plain EEGNet on privacy. This two-seed slice is not enough to replace the completed `1-28` four-seed reference point yet, but it suggests the adversarial advantage may shrink again as the subject set widens.

#### Four seeds (`13,17,19,23`)

- `eegnet`: task `0.5077`, membership AUC `0.4996`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.4853`, membership AUC `0.5008`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.4986`, membership AUC `0.4917`
- `adversarial_eegnet (constant@1.00)`: task `0.4973`, membership AUC `0.4837`

Takeaway: the stronger `subjects 1-30` four-seed rerun reopens the adversarial line, but only as a privacy/utility tradeoff. `constant@1.0` is the best privacy point at AUC `0.4837`, and `linear_ramp@1.0` is second at `0.4917`; both beat plain EEGNet on membership. However, plain EEGNet is still the highest-utility model at task `0.5077`, so the current widest slice does not give a clean Pareto win. `linear_ramp@0.6` does not hold up on this slice: it is worse than plain EEGNet on both task and membership.

### Widened PhysioNet Check (`subjects 1-32`)

#### Two seeds (`13,17`)

- `eegnet`: task `0.5483`, membership AUC `0.5004`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.5201`, membership AUC `0.5188`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5096`, membership AUC `0.5341`
- `adversarial_eegnet (constant@1.00)`: task `0.4905`, membership AUC `0.5364`

Takeaway: the first `subjects 1-32` checkpoint is negative for the adversarial family. Plain EEGNet is best on both task accuracy and membership AUC. `linear_ramp@0.6` is the least-bad adversarial point, but it is still worse than plain EEGNet on privacy (`0.5188` vs `0.5004`) and task (`0.5201` vs `0.5483`). Both weight-`1.0` schedules are materially worse on privacy, with `constant@1.0` also taking the largest utility hit. Because this is only a two-seed checkpoint, it should not replace the `subjects 1-30` four-seed slice as the main reference, but it is another widening warning against promoting a stable adversarial default.

#### Four seeds (`13,17,19,23`)

- `eegnet`: task `0.5611`, membership AUC `0.5250`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.5018`, membership AUC `0.5106`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.4985`, membership AUC `0.5414`
- `adversarial_eegnet (constant@1.00)`: task `0.4992`, membership AUC `0.5223`

Takeaway: the stronger `subjects 1-32` four-seed rerun changes the two-seed read from adversarial-negative to a privacy-only tradeoff. `linear_ramp@0.6` now improves membership AUC versus plain EEGNet (`0.5106` vs `0.5250`), and `constant@1.0` is slightly privacy-favorable (`0.5223` vs `0.5250`). Neither is close on utility: plain EEGNet is far ahead on task (`0.5611`), while every adversarial setting sits near `0.50`. `linear_ramp@1.0` fails on both objectives here. The widened PhysioNet conclusion remains unresolved schedule instability, not a stable default.

### Widened PhysioNet Check (`subjects 1-34`)

#### Four seeds (`13,17,19,23`)

- `eegnet`: task `0.5376`, membership AUC `0.5333`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.5091`, membership AUC `0.5082`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5127`, membership AUC `0.5499`
- `adversarial_eegnet (constant@1.00)`: task `0.4891`, membership AUC `0.5230`

Takeaway: the widest completed PhysioNet slice keeps the adversarial privacy line alive but still only as a tradeoff. `linear_ramp@0.6` is the best privacy point by a clear margin (`0.5082` vs plain `0.5333` AUC), and `constant@1.0` is also privacy-favorable (`0.5230`), but both lose task accuracy to plain EEGNet. `linear_ramp@1.0` fails again, worse than plain on both utility and membership AUC. This extends the `1-32` pattern but shifts the exact numbers: adversarial training can reduce leakage on the widest slice, yet the schedule winner is still unstable and the utility cost remains material.

### Widened PhysioNet Check (`subjects 1-36`)

#### Four seeds (`13,17,19,23`)

- `eegnet`: task `0.5413`, membership AUC `0.5314`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.4833`, membership AUC `0.4983`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5020`, membership AUC `0.5004`
- `adversarial_eegnet (constant@1.00)`: task `0.5055`, membership AUC `0.5475`

Takeaway: the new widest completed PhysioNet slice keeps the ramped adversarial line alive, but still not as a Pareto win. `linear_ramp@0.6` is the lowest-leakage point at AUC `0.4983`, and `linear_ramp@1.0` is close at `0.5004` while retaining more task accuracy. Plain EEGNet remains clearly best on task (`0.5413`). `constant@1.0` fails on this slice, leaking more than plain EEGNet. The practical read is that the last three widest slices now favor `linear_ramp@0.6` on privacy, but the broader widening history still argues against a stable universal PhysioNet default.

### Widened PhysioNet Check (`subjects 1-38`)

#### Four seeds (`13,17,19,23`)

- `eegnet`: task `0.5510`, membership AUC `0.4871`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.4971`, membership AUC `0.4970`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5120`, membership AUC `0.4985`
- `adversarial_eegnet (constant@1.00)`: task `0.4970`, membership AUC `0.5435`

Takeaway: the new widest completed PhysioNet slice breaks the previous `linear_ramp@0.6` privacy-frontier streak. Plain EEGNet is best on both task accuracy and membership AUC, while both ramped schedules land near chance but no longer beat the baseline. `constant@1.0` fails again on privacy. This strengthens the paper-safe interpretation that PhysioNet remains unresolved and slice-dependent rather than supporting a stable adversarial default.

### Widened PhysioNet Check (`subjects 1-40`)

#### Four seeds (`13,17,19,23`)

- `eegnet`: task `0.5423`, membership AUC `0.5205`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.5092`, membership AUC `0.4996`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.4977`, membership AUC `0.5265`
- `adversarial_eegnet (constant@1.00)`: task `0.5139`, membership AUC `0.5228`

Takeaway: the new widest completed PhysioNet slice reopens the `linear_ramp@0.6` privacy line, but only as a tradeoff. `linear_ramp@0.6` reduces membership AUC by `0.0209` versus plain EEGNet, while losing `0.0331` task balanced accuracy. `linear_ramp@1.0` and `constant@1.0` both leak more than plain EEGNet. The last two widest slices now disagree, which reinforces instability rather than a stable PhysioNet default.

### Widened PhysioNet Check (`subjects 1-42`)

#### Four seeds (`13,17,19,23`)

- `eegnet`: task `0.6460`, membership AUC `0.5425`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.5154`, membership AUC `0.5714`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.4992`, membership AUC `0.5170`
- `adversarial_eegnet (constant@1.00)`: task `0.4971`, membership AUC `0.5672`

Takeaway: this completed PhysioNet slice shifts the best adversarial privacy point from `linear_ramp@0.6` to `linear_ramp@1.0`, but the tradeoff is severe. `linear_ramp@1.0` lowers membership AUC by `0.0255` versus plain EEGNet while losing `0.1467` task balanced accuracy. `linear_ramp@0.6` and `constant@1.0` both leak more than plain EEGNet. This continues the widening-instability story rather than resolving PhysioNet.

#### Confidence-penalty robustness check (`13,17,19,23`)

- `confidence_penalty_eegnet (beta=0.025)`: task `0.6446`, membership AUC `0.5431`
- `confidence_penalty_eegnet (beta=0.05)`: task `0.6376`, membership AUC `0.5455`

Takeaway: the confidence-penalty gain from `subjects 1-44` does not transfer backward to `subjects 1-42`. Both tested beta values are neutral-to-worse than plain EEGNet on membership AUC, and neither approaches the `linear_ramp@1.0` privacy point on this slice. This keeps confidence penalty promising but not yet stable.

### Widened PhysioNet Check (`subjects 1-44`)

#### Four seeds (`13,17,19,23`)

- `eegnet`: task `0.6354`, membership AUC `0.5119`
- `bottleneck_eegnet (dim=6)`: task `0.6813`, membership AUC `0.5149`
- `confidence_penalty_eegnet (beta=0.025)`: task `0.6447`, membership AUC `0.5049`
- `confidence_penalty_eegnet (beta=0.05)`: task `0.6395`, membership AUC `0.5046`
- `confidence_penalty_eegnet (beta=0.10)`: task `0.6413`, membership AUC `0.5094`
- `feature_noise_eegnet (sigma=0.05)`: task `0.6628`, membership AUC `0.5121`
- `mixup_eegnet (alpha=0.20)`: task `0.5969`, membership AUC `0.5019`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.5054`, membership AUC `0.5354`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5023`, membership AUC `0.5237`
- `adversarial_eegnet (constant@1.00)`: task `0.5026`, membership AUC `0.5258`

Takeaway: the new widest completed PhysioNet slice reverses the apparent adversarial privacy gains from `1-40` and `1-42`, but opens a better non-adversarial regularization line. `confidence_penalty_eegnet` is Pareto-improving across the tested beta values, with `beta=0.025` the best balanced point (`0.6447` task, `0.5049` AUC) and `beta=0.05` the lowest-AUC confidence-penalty point (`0.5046`). `mixup_eegnet (alpha=0.20)` gives the lowest AUC overall (`0.5019`) but gives up utility, and `feature_noise_eegnet (sigma=0.05)` improves utility while staying essentially tied with plain on AUC. All three adversarial settings reduce utility by about `0.13` task balanced accuracy and leak more than plain EEGNet, so the current widest slice argues against an adversarial default and for validating the regularized branch on adjacent slices.

### Widened PhysioNet Check (`subjects 1-46`)

#### Four seeds (`13,17,19,23`)

- `eegnet`: task `0.6574`, membership AUC `0.5162`, attack balanced accuracy `0.5469`
- `bottleneck_eegnet (dim=6)`: task `0.6475`, membership AUC `0.5301`, attack balanced accuracy `0.5498`
- `confidence_penalty_eegnet (beta=0.025)`: task `0.6540`, membership AUC `0.5140`, attack balanced accuracy `0.5406`
- `mixup_eegnet (alpha=0.20)`: task `0.6223`, membership AUC `0.5039`, attack balanced accuracy `0.5305`
- `feature_noise_eegnet (sigma=0.05)`: task `0.5981`, membership AUC `0.5431`, attack balanced accuracy `0.5583`
- `adversarial_eegnet (linear_ramp@0.60, ramp=0.40)`: task `0.4998`, membership AUC `0.5346`, attack balanced accuracy `0.5481`
- `adversarial_eegnet (linear_ramp@1.00, ramp=0.40)`: task `0.5087`, membership AUC `0.4917`, attack balanced accuracy `0.5195`
- `adversarial_eegnet (constant@1.00)`: task `0.5004`, membership AUC `0.5143`, attack balanced accuracy `0.5353`

Takeaway: the forward defense check preserves the same broad shape as `subjects 1-44` for mixup, but weakens the confidence-penalty claim and rejects the utility-heavy alternatives. Relative to matched plain EEGNet on `subjects 1-46`, `beta=0.025` lowers membership AUC by `0.0022` and attack balanced accuracy by `0.0063`, while task balanced accuracy drops by `0.0034`. `mixup_eegnet (alpha=0.20)` is the strongest non-adversarial privacy point, lowering AUC by `0.0123` and attack balanced accuracy by `0.0164`, but it costs `0.0351` task balanced accuracy. The raw lowest AUC is adversarial `linear_ramp@1.0` (`0.4917`), but it costs about `0.149` task balanced accuracy versus plain. `linear_ramp@0.6` is worse than plain on privacy, and `constant@1.0` is only slightly lower AUC than plain with a similarly severe task collapse. `bottleneck_eegnet (dim=6)` loses the `1-44` utility advantage and leaks more than plain; `feature_noise_eegnet (sigma=0.05)` is worse than plain on both task and privacy. This keeps PhysioNet framed as unstable rather than solved.

#### Learned-attack robustness check (`posterior_probabilities`)

- `eegnet`: task `0.6574`, membership AUC `0.5108`, attack balanced accuracy `0.5534`
- `confidence_penalty_eegnet (beta=0.025)`: task `0.6540`, membership AUC `0.5103`, attack balanced accuracy `0.5541`
- `mixup_eegnet (alpha=0.20)`: task `0.6223`, membership AUC `0.5015`, attack balanced accuracy `0.5362`
- `adversarial_eegnet (linear_ramp@1.0, ramp=0.40)`: task `0.5087`, membership AUC `0.4855`, attack balanced accuracy `0.5233`

Takeaway: removing the true-label one-hot feature does not weaken the learned attack on this slice. In fact, all four key models are slightly lower-AUC here than under `posterior_probabilities_plus_true_label`: plain EEGNet goes from `0.5162` to `0.5108`, `confidence_penalty_eegnet (beta=0.025)` from `0.5140` to `0.5103`, `mixup_eegnet (alpha=0.20)` from `0.5039` to `0.5015`, and `linear_ramp@1.0` from `0.4917` to `0.4855`. The ranking is unchanged, so the learned-attack story on `subjects 1-46` is not being driven by explicit label knowledge.

#### Threshold-attack robustness check (`label_known_log_probability`)

- `eegnet`: task `0.6574`, membership AUC `0.6162`, attack balanced accuracy `0.5995`
- `confidence_penalty_eegnet (beta=0.025)`: task `0.6540`, membership AUC `0.6165`, attack balanced accuracy `0.6011`
- `mixup_eegnet (alpha=0.20)`: task `0.6223`, membership AUC `0.6133`, attack balanced accuracy `0.5938`
- `adversarial_eegnet (linear_ramp@1.0, ramp=0.40)`: task `0.5087`, membership AUC `0.5147`, attack balanced accuracy `0.5335`

Takeaway: the alternate threshold attack changes the non-adversarial ranking materially. `mixup_eegnet (alpha=0.20)` is still privacy-favorable, but only by `0.0029` AUC versus plain EEGNet, while `confidence_penalty_eegnet (beta=0.025)` is slightly worse than plain (`0.6165` vs `0.6162`). `linear_ramp@1.0` remains strongly privacy-favorable under this attack as well, with a much larger gap (`-0.1015` AUC), but it still carries the same severe utility collapse. So the current honest read is not just slice-instability but attack-sensitivity in the non-adversarial branch.

#### Threshold-attack robustness check (`max_probability`)

- `eegnet`: task `0.6574`, membership AUC `0.5875`, attack balanced accuracy `0.5828`
- `confidence_penalty_eegnet (beta=0.025)`: task `0.6540`, membership AUC `0.5886`, attack balanced accuracy `0.5836`
- `mixup_eegnet (alpha=0.20)`: task `0.6223`, membership AUC `0.5648`, attack balanced accuracy `0.5660`
- `adversarial_eegnet (linear_ramp@1.0, ramp=0.40)`: task `0.5087`, membership AUC `0.5379`, attack balanced accuracy `0.5498`

Takeaway: the label-free threshold attack sits between the learned attack and the harsher label-known threshold attack. `mixup_eegnet (alpha=0.20)` still lowers leakage relative to plain EEGNet, now by `0.0227` AUC, while `confidence_penalty_eegnet (beta=0.025)` is again slightly worse than plain (`0.5886` vs `0.5875`). `linear_ramp@1.0` still gives the lowest AUC, but its gap versus plain shrinks to `-0.0496`, much smaller than under the label-known threshold attack, and it keeps the same severe utility collapse. This keeps the core PhysioNet read attack-sensitive, with only the adversarial line staying privacy-favorable across all three evaluated attacks.

#### Threshold-attack robustness check (`negative_entropy`)

- `eegnet`: task `0.6574`, membership AUC `0.5875`, attack balanced accuracy `0.5828`
- `confidence_penalty_eegnet (beta=0.025)`: task `0.6540`, membership AUC `0.5886`, attack balanced accuracy `0.5836`
- `mixup_eegnet (alpha=0.20)`: task `0.6223`, membership AUC `0.5648`, attack balanced accuracy `0.5660`
- `adversarial_eegnet (linear_ramp@1.0, ramp=0.40)`: task `0.5087`, membership AUC `0.5380`, attack balanced accuracy `0.5498`

Takeaway: this attack is effectively redundant with `max_probability` on the `subjects 1-46` key-model slice. The ranking is identical, and the AUCs match the `max_probability` values to about `1e-5` or better across all four models. That means the current label-free threshold story is not just directionally but numerically stable: `mixup_eegnet (alpha=0.20)` remains modestly privacy-favorable, `confidence_penalty_eegnet (beta=0.025)` remains slightly worse than plain, and `linear_ramp@1.0` remains the best privacy point with severe utility collapse.

## Current Project Read

- There is no single defense that wins everywhere.
- `bottleneck_eegnet (dim=6)` is still the strongest overall candidate on the datasets where both subject-ID and membership are available (`bnci2014_001` and `lee2019_mi`), but the raw membership ranking on both datasets is now attack-sensitive.
- PhysioNet still does not support a stable promoted defense on the widest completed slices; the `1-46` check gives the lowest raw AUC to adversarial `linear_ramp@1.0`, but only with severe task collapse. Among non-adversarial defenses, mixup is strongest on privacy under the learned attack, confidence penalty is lower-cost but small-effect there, and the threshold attacks weaken both non-adversarial gains sharply.
- The learned logistic-regression attack is also stable to removing the true-label feature on the `1-46` key-model slice: the ranking is unchanged and every model is slightly lower-AUC under posterior-only features than under posterior-plus-label features.
- On BNCI, the attack-family split is now explicit: the bottleneck is slightly favorable under both learned attacks, but both threshold attacks reverse the raw membership ranking hard enough that `adversarial_eegnet (constant@1.00)` becomes the privacy winner instead.
- On Lee, the bottleneck stays near chance under both learned attacks and trades the raw lead across attacks: plain EEGNet is best under learned attacks, `adversarial_eegnet (constant@1.00)` is best under the label-free thresholds, and the bottleneck narrowly leads under the label-known threshold. The completed low-CPU `negative_entropy` rerun confirms that the full Lee label-free threshold family is effectively identical to `max_probability`.
- Across the two label-free threshold attacks evaluated so far (`max_probability` and `negative_entropy`), the `1-46` key-model rankings are effectively identical, so the label-free threshold read now looks stable even though the broader cross-attack story remains attack-sensitive.
- The dedicated widening consolidation is now in physionet_motor_imagery_widening_4seed_summary_v1.md, with the companion plot at physionet_widening_4seed_tradeoff.png.
- The current widest matched forward-defense check is `subjects 1-46`: adversarial `linear_ramp@1.0` is the raw privacy winner (`0.4917` AUC) but with severe utility loss (`0.5087` task); `mixup_eegnet (alpha=0.20)` is the best non-adversarial privacy point (`0.5039` AUC, `0.6223` task); `confidence_penalty_eegnet (beta=0.025)` is a lower-cost privacy improvement (`0.6540` task, `0.5140` AUC); bottleneck and feature noise both fail against plain.
- The posterior-only learned attack at `subjects 1-46` keeps the same ordering as the label-assisted learned attack but is slightly harsher on every model: plain `0.5108`, confidence penalty `0.5103`, mixup `0.5015`, adversarial `linear_ramp@1.0` `0.4855`.
- The alternate `threshold` / `label_known_log_probability` attack at `subjects 1-46` is more skeptical of the non-adversarial line: `mixup_eegnet (alpha=0.20)` is only slightly better than plain (`0.6133` vs `0.6162` AUC), `confidence_penalty_eegnet (beta=0.025)` is slightly worse than plain (`0.6165`), and `linear_ramp@1.0` still shows a large privacy win (`0.5147`) at the same severe utility cost.
- The label-free `threshold` / `max_probability` attack at `subjects 1-46` keeps the same qualitative ranking but with a smaller adversarial gap: `mixup_eegnet (alpha=0.20)` is modestly better than plain (`0.5648` vs `0.5875` AUC), `confidence_penalty_eegnet (beta=0.025)` is slightly worse than plain (`0.5886`), and `linear_ramp@1.0` remains best on privacy (`0.5379`) with the same severe utility collapse.
- The label-free `threshold` / `negative_entropy` attack at `subjects 1-46` is numerically almost identical to `max_probability`: the ranking is unchanged and every model lands within about `1e-5` AUC of its `max_probability` value.
- The broader `subjects 1-44` regularized slice still favors `mixup_eegnet (alpha=0.20)` on membership AUC (`0.5019`) and `bottleneck_eegnet (dim=6)` on task (`0.6813`); `confidence_penalty_eegnet (beta=0.025)` remains the best balanced `1-44` point because it improves both task and AUC versus plain EEGNet without the mixup utility loss.
- The widened PhysioNet `subjects 1-6` four-seed slice still supports the earlier tradeoff read where `0.5`/`0.6` are the better utility/privacy compromise points and `1.0` is the pure-privacy extreme.
- The newer `subjects 1-8` four-seed slice favored `constant@1.0`, the `subjects 1-10` and `1-12` slices favored parts of the ramped line, the `subjects 1-14` four-seed slice favored plain EEGNet on membership AUC, and the wider `subjects 1-16` and `1-18` four-seed slices swung back toward the adversarial line.
- The next `subjects 1-16` two-seed slice reopens the ramped line slightly: both `linear_ramp@1.0` and `linear_ramp@0.6` are again a little better than plain EEGNet on membership AUC, while `constant@1.0` still loses on privacy.
- The completed `subjects 1-16` four-seed reruns reopen the adversarial line again: both ramp settings beat plain EEGNet on task accuracy and membership AUC there, with `linear_ramp@0.6` edging `linear_ramp@1.0` by a very small margin.
- The completed `subjects 1-18` four-seed slice keeps the reopened line alive there: all three adversarial schedules beat plain EEGNet on membership AUC, with `constant@1.0` and `linear_ramp@1.0` essentially tied for the lead and `linear_ramp@0.6` trailing them.
- The wider `subjects 1-20` slice weakens that reopened line at two seeds and then breaks it again at four seeds: plain EEGNet comes back out on top of the completed adversarial family on task and membership.
- The next `subjects 1-22` slice reopens the adversarial line again under both two-seed and four-seed validation, but with a different schedule ranking: `linear_ramp@0.6` is the lowest-leakage point at four seeds, `linear_ramp@1.0` is a close second with better task, and `constant@1.0` keeps the best task of the three adversarial settings while leaking more than both ramped schedules.
- The widest completed `subjects 1-24` slice reopens the adversarial line again under both two-seed and four-seed validation, but with yet another ranking shift: `linear_ramp@1.0` is best on privacy, `linear_ramp@0.6` is close behind with slightly better task, and `constant@1.0` is the best utility point but weakest of the three on privacy.
- The next `subjects 1-26` slice weakens that reopened line again: the two-seed read is negative, and the stronger four-seed reruns only partly recover it. At four seeds, `linear_ramp@1.0` still improves privacy over plain EEGNet, but `linear_ramp@0.6` no longer does and `constant@1.0` is worse than plain EEGNet on both privacy and task.
- The newer `subjects 1-28` four-seed slice reopens the adversarial line again, but with another schedule shuffle: `linear_ramp@0.6` is best on privacy, `constant@1.0` is best on task and close behind on privacy, and `linear_ramp@1.0` is now worse than plain EEGNet on privacy.
- The `subjects 1-30` four-seed slice kept the adversarial privacy line alive but not as a Pareto win: `constant@1.0` was best on membership AUC (`0.4837`), `linear_ramp@1.0` was second (`0.4917`), and plain EEGNet remained best on task (`0.5077`).
- The `subjects 1-32` four-seed slice kept the same privacy/utility-tradeoff shape with another schedule shuffle: `linear_ramp@0.6` was best on privacy (`0.5106`), plain EEGNet remained best on task (`0.5611`), and `linear_ramp@1.0` was worse than plain on both metrics.
- The `subjects 1-34` four-seed slice again favored `linear_ramp@0.6` on privacy (`0.5082`) and plain EEGNet on task (`0.5376`), while `linear_ramp@1.0` failed on both metrics.
- The `subjects 1-36` four-seed slice strengthened the ramped privacy line, `subjects 1-38` reversed that read in favor of plain EEGNet on both axes, `subjects 1-40` reopened only `linear_ramp@0.6`, `subjects 1-42` shifted the privacy point to `linear_ramp@1.0`, and the current widest `subjects 1-44` returns membership privacy to plain EEGNet while moving the utility lead to the bottleneck model.
- The PhysioNet schedule winner is therefore not stable even under four-seed widening.
- The strongest current PhysioNet claim is mixed rather than steadily improving: the family failed on the completed `subjects 1-14`, `1-20`, `1-38`, and `1-44` four-seed slices, reopened on several intermediate checkpoints, and still has no stable adversarial default.
- The `subjects 1-44` bottleneck check weakens the simple “PhysioNet bottleneck fails” story: `dim=6` is the best utility model at task `0.6813`, but its membership AUC `0.5149` remains slightly worse than plain EEGNet `0.5119`.
- The `subjects 1-44` regularized check is the most useful new PhysioNet branch: `confidence_penalty_eegnet` is Pareto-improving at `beta=0.025`, `0.05`, and `0.10`, `mixup_eegnet (alpha=0.20)` is the lowest-AUC privacy point, and `feature_noise_eegnet (sigma=0.05)` is a utility-heavy near-tie on privacy.
- The adjacent `subjects 1-42` confidence-penalty check is negative: `beta=0.025` and `beta=0.05` both leak slightly more than plain EEGNet and are far behind the `linear_ramp@1.0` privacy point on that slice.
- The forward `subjects 1-46` defense check is mixed and attack-sensitive: adversarial `linear_ramp@1.0` keeps the strongest privacy improvement under all five evaluated attacks but with severe utility collapse; `mixup_eegnet (alpha=0.20)` is the best non-adversarial privacy point under both learned attacks, only slightly better than plain under the label-known threshold attack, and modestly better under the two label-free threshold attacks; `confidence_penalty_eegnet (beta=0.025)` improves AUC only slightly under both learned attacks and is slightly worse than plain under all threshold attacks; both `bottleneck_eegnet (dim=6)` and `feature_noise_eegnet (sigma=0.05)` are worse than plain on privacy.
- `label_smoothing_eegnet` is also neutral on widened PhysioNet, not just on BNCI and Lee-style checks.
- `adversarial_eegnet (constant@1.0)` is still not a stable lead candidate. It fails on `subjects 1-10`, loses to plain EEGNet on membership at `subjects 1-14`, wins privacy at `subjects 1-30`, remains mixed through `subjects 1-36`, fails badly at `subjects 1-38`, leaks slightly more than plain EEGNet at `subjects 1-40`, fails again at `subjects 1-42`, and remains worse than plain at `subjects 1-44`.
- `label_smoothing_eegnet` is mostly neutral. It is cleaner than the failed bottleneck transfer on PhysioNet, but it is not a major privacy lever.

## Recommended Next Move

1. Treat `bottleneck_eegnet (dim=6)` as the main defense line for BNCI and Lee.
2. Treat PhysioNet adversarial training as unresolved rather than promoting it. The completed four-seed slices still disagree materially, and the `subjects 1-44` adversarial family loses to plain EEGNet while regularized non-adversarial models look more promising but not yet stable.
3. Stop assuming a universal defense and either:
   - add a new defense family aimed at preserving the BNCI/Lee bottleneck gains while avoiding the PhysioNet failure, or
   - expand the benchmark to another dataset so the current cross-dataset split becomes harder to dismiss as dataset-specific noise.
