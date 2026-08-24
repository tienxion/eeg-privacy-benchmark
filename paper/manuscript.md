# Attack-Sensitive Privacy Evaluation for EEG Motor-Imagery Decoding

## Abstract

Privacy evaluations of EEG decoders can give different answers depending on whether leakage means subject identity or training-set membership, which attacker is used, and how data are split. We present a reproducible benchmark of binary motor-imagery decoding on BNCI 2014-001, Lee2019 MI, and PhysioNet Motor Imagery. The benchmark evaluates CSP-LDA and compact neural decoders with bottleneck, subject-adversarial, mixup, confidence-penalty, label-smoothing, and feature-noise variants. Privacy is measured separately with frozen-feature subject-identification probes and threshold, logistic, and nonlinear posterior-based membership attacks.

Bottleneck EEGNet provides a qualified utility/subject-identity tradeoff on BNCI and Lee, but it is not an attack-invariant membership defense. BNCI expanded-subject validation reduces subject-identification accuracy by 0.2683 while improving task balanced accuracy by 0.0106 relative to EEGNet; Lee reduces subject-identification accuracy by 0.2340 while improving task balanced accuracy by 0.0019, but its threshold membership AUCs regress by 0.0145 to 0.0176. PhysioNet supports no stable default: defense rankings change by attack family and protocol, the eight-seed mixup check worsens nonlinear membership AUC by 0.0096 on average, and the strongest adversarial privacy point costs 0.1486 task balanced accuracy. A federated PhysioNet replication also fails a 0.60 utility gate. A pairwise-mask secure-aggregation simulator matches FedAvg within 2.384e-07, but adds 217.77x local aggregation runtime and protects neither global-model outputs nor membership. The resulting evidence is mixed rather than uniformly positive, showing why EEG privacy claims must be objective-specific, attack-sensitive, and utility-gated.

## 1. Introduction

EEG is a person-derived biosignal. Even when a decoder is trained for a narrow task such as left-versus-right motor imagery, its representations and outputs may retain information about the people represented in the data. Public motor-imagery corpora make this risk experimentally tractable, but they also create a benchmarking problem: datasets differ in session structure, subject count, acquisition protocol, and attainable task accuracy [tangermann2012bcicompiv; lee2019openbmi; schalk2009eegmmidb; jayaram2018moabb].

Two privacy questions are often conflated. Subject identification asks whether a representation reveals who produced a trial. Membership inference asks whether a trial was included in the task model's training set. These objectives use different adversaries, targets, and evaluation splits, so a reduction in one metric does not establish a reduction in the other. A defensible EEG privacy benchmark must report them separately and test whether conclusions survive changes in membership attacker family [shokri2017membership].

We evaluate classical and neural decoders across three public datasets using frozen protocols and seed-matched comparisons. Rather than forcing one overall winner, we use a claim audit that distinguishes supported, qualified, and unsupported conclusions. Bottleneck EEGNet improves the task/subject-identity tradeoff on BNCI and Lee, with stronger membership qualifications on Lee. PhysioNet remains unresolved after random-subset, eight-seed, cross-run, and defense-grid checks. Changing the training topology through federation or hiding individual updates with a limited secure-aggregation mechanism does not by itself protect released model outputs.

The contributions are:

1. A cross-dataset benchmark that separates task utility, subject-identification leakage, and membership leakage.
2. An attack-sensitive membership evaluation spanning scalar thresholds, learned logistic attacks, and a nonlinear MLP attacker.
3. Seed-level and expanded-scope validation that preserves qualified BNCI/Lee conclusions while rejecting a preliminary PhysioNet mixup claim.
4. A utility-gated federated diagnostic and a narrowly scoped secure-aggregation functional test.
5. A frozen claim-to-evidence package that retains negative and mixed findings rather than selecting only favorable attack/model pairs.

## 2. Related Work

The benchmark uses MOABB-backed versions of BNCI2014-001, Lee2019_MI/OpenBMI, and the PhysioNet EEG Motor Movement/Imagery corpus [jayaram2018moabb; tangermann2012bcicompiv; lee2019openbmi; schalk2009eegmmidb; schalk2004bci2000; goldberger2000physionet]. CSP-LDA provides a classical spatial-filter anchor, while EEGNet provides a compact convolutional baseline [ramoser2000csp; lawhern2018eegnet]. Their inclusion matters because classical and neural models expose different confidence and representation structures.

Membership inference estimates whether an example influenced model fitting, commonly from confidence scores or posterior vectors [shokri2017membership]. EEG privacy also includes broader identity and side-channel risks [martinovic2012bci_side_channel; meng2023user_identity]. Our scope is intentionally narrower: linear subject-identification probes on frozen learned features and membership attacks on released task-model outputs. We do not infer clinical attributes or reconstruct raw signals.

The evaluated defenses are lightweight benchmark baselines. Bottlenecks restrict representation dimension; adversarial training discourages subject-predictive features; mixup interpolates examples and targets; confidence penalties discourage sharply peaked posteriors; and label smoothing or feature noise regularize training [ganin2016dann; zhang2018mixup; pereyra2017confidence]. None provides a formal privacy guarantee.

## 3. Threat Model and Research Questions

### 3.1 Distinct Leakage Objectives

For subject identification, the adversary receives frozen task-model features and predicts subject identity with a linear probe. This objective is evaluated only where cross-session structure permits identities learned from one session to be tested on another. Lower probe accuracy indicates less identity information accessible to this adversary.

For membership inference, members are task-model training and validation trials and nonmembers are held-out test trials. The adversary receives a scalar confidence score or posterior vector and predicts membership. Lower attack AUC indicates less separability; 0.5 is chance ranking. Learned logistic and MLP attacks use a 50/50 attacker train/evaluation split. The best threshold orientation is a diagnostic attacker choice, not a deployable defense setting.

### 3.2 Attacker Scope and Non-Guarantees

The benchmark assumes access to released model scores or representations appropriate to each probe. It does not cover raw-signal reconstruction, attribute inference, model extraction, adaptive white-box attacks, poisoning, or composition over repeated releases. A favorable result bounds measured leakage under the named protocol; it is not a general privacy guarantee.

Federation changes where training data reside and how updates are combined. It does not remove membership signals from a released global model. Secure aggregation can hide individual client updates from an aggregator under its assumptions, but it does not hide the final aggregate or task-model outputs. We therefore treat federation and secure aggregation as systems diagnostics, not as substitutes for output-level privacy evaluation.

## 4. Methods

### 4.1 Datasets, Preprocessing, and Splits

All tasks use binary left-hand versus right-hand motor imagery. BNCI 2014-001 and Lee2019 MI use cross-session evaluation; PhysioNet uses cross-subject evaluation for the primary benchmark and a cross-run holdout as a protocol-sensitivity check. Trial extraction uses MOABB 1.2.0 LeftRightImagery defaults: events left_hand and right_hand, an 8-32 Hz filter, tmin 0.0, no explicit tmax, no resampling, and no channel restriction. The audited environment uses MNE 1.8.0, PyTorch 2.8.0, and scikit-learn 1.5.2.

### 4.2 Models and Defenses

The model set includes CSP-LDA and compact EEGNet. Completed defense sweeps include a six-dimensional bottleneck, constant and ramped subject-adversarial training, mixup, confidence penalties, label smoothing, and feature noise. Comparisons are paired by task-model seed where the archived protocol supports pairing. Four task-model seeds (13, 17, 19, and 23) form the primary matched layer; the PhysioNet nonlinear-attacker extension adds seeds 29, 31, 37, and 41.

### 4.3 Attacks and Metrics

Threshold membership attacks use label-known log probability, maximum probability, and negative entropy. Learned logistic attacks use posterior probabilities either alone or with the true-label indicator. The nonlinear attack is a one-hidden-layer MLP with 32 hidden units. Task utility is summarized by balanced accuracy, subject leakage by linear-probe accuracy, and membership leakage by attack AUC. Deltas are defense minus EEGNet, so negative privacy deltas favor the defense and positive task deltas favor the defense.

### 4.4 Claim Rules and Validation

Claims are not based on rank alone. We inspect paired seed direction, effect size, uncertainty where available, task cost, expanded-subject behavior, and disagreement across attack families. The PhysioNet checks include random subject subsets, an eight-seed nonlinear-attacker extension, an 11-row cross-run holdout, and a pre-registered non-adversarial defense grid. These analyses were completed before freezing benchmark v1; no post-freeze training is used in this manuscript.

### 4.5 Federated and Secure-Aggregation Diagnostics

The federated diagnostic uses the frozen 46-subject PhysioNet cross-subject scope over seeds 13, 17, 19, 23. A mean task balanced-accuracy gate of 0.60 is required before interpreting privacy deltas. The secure-aggregation prototype applies cancelling pairwise masks to 36 client updates and compares the result with ordinary FedAvg. Runtime and communication are local implementation diagnostics, not deployment benchmarks.

## 5. Results

### 5.1 Dataset-Level Overview

No defense dominates all datasets, objectives, and attack families (Table 1). The two constructive findings concern subject-identification reduction on cross-session BNCI and Lee. Membership-only rankings are less stable: nonlinear and threshold attacks select different models, and the two label-free threshold scores are rank-redundant without making the threshold family unimportant. Figure 1 shows why seed-paired deltas are necessary before promoting a mean-rank winner.

### 5.2 BNCI: Qualified Bottleneck Default

In the matched shortlist, bottleneck EEGNet reaches task balanced accuracy 0.7409, a 0.0590 improvement over EEGNet, while reducing subject-identification accuracy to 0.4618. Expanded validation on subjects 1-9 preserves the result: task balanced accuracy improves by 0.0106 and subject-identification accuracy decreases by 0.2683. Membership behavior is materially weaker than the identity result. The expanded MLP delta is +0.0007; logistic deltas are +0.0020 and +0.0001; threshold deltas are -0.0041, -0.0008, and -0.0008. A three-seed attacker refit gives mean MLP delta -0.0025 with privacy improvement in two of three refits.

Accordingly, bottleneck EEGNet is the BNCI default only under a joint utility/subject-identity criterion. It is not the best membership defense across attacks: adversarial constant has lower shortlist mean AUC (0.5166 versus 0.5228), and CSP-LDA wins the nonlinear learned family.

### 5.3 Lee: Subject-Identification-First Bottleneck Result

On Lee, bottleneck EEGNet reaches subject-identification accuracy 0.2625 and improves task balanced accuracy by 0.0146 over EEGNet in the matched shortlist. Expanded validation on subjects 1-12 again reduces subject-identification accuracy by 0.2340 while improving task balanced accuracy by 0.0019. This result does not extend cleanly to membership: the expanded MLP delta is +0.0011, logistic deltas are -0.0019 and -0.0040, but threshold deltas regress by +0.0145, +0.0176, and +0.0176. The attacker-refit MLP delta is +0.0135 with zero of three privacy-improving refits.

Lee therefore supports a subject-identification-first bottleneck choice, not a single membership winner. The nonlinear learned attack favors adversarial constant, whereas threshold summaries favor CSP-LDA. CSP-LDA's stronger threshold result must also be balanced against its substantially higher subject-identification leakage.

### 5.4 PhysioNet: No Supported Default

PhysioNet produces the clearest negative result. Learned-family AUC favors adversarial linear-ramp, while label-free thresholds favor CSP-LDA at AUC 0.4902. The adversarial ramp is only a privacy-extreme point: it reduces mean AUC by 0.0409 but reduces task balanced accuracy by 0.1486. The non-adversarial defense grid yields no candidate satisfying the utility and two-family support rules.

The preliminary nonlinear mixup ranking also fails validation. In the original four-seed comparison, its AUC difference is +0.0023 with 95% interval [-0.0922, +0.0968], privacy improvement in two of four seeds, and no unique rank-1 seed. Across eight seeds, mixup changes AUC by +0.0096 and improves privacy in only two of eight seeds while task balanced accuracy changes by +0.0014. On the cross-run holdout, mixup improves threshold AUCs by 0.0166 and 0.0147 but improves MLP AUC by only 0.0075, below the 0.010 decision rule, with a 0.0185 task loss. These checks reject a stable PhysioNet mixup claim rather than merely leaving it underpowered.

### 5.5 Federation Fails the Utility Gate

The frozen PhysioNet replication changes mean task balanced accuracy from 0.6574 centrally to 0.5185 under federation, a paired delta of -0.1389. It therefore fails the task-utility gate of 0.60. Membership deltas range from -0.1009 to +0.0220 across attacks, and the fixed-MLP refit delta is -0.0427; all are diagnostic only because utility is invalid. Federation does not support a defense claim.

### 5.6 Secure Aggregation Has Narrow Functional Scope

The pairwise-mask simulator matches FedAvg with maximum absolute error 2.384e-07 and preserves the checkpoint within 1.819e-12. It uses 630 client pairs and 11,340 mask applications. Estimated one-time seed setup is 40,320 bytes, corresponding to a 0.0649% increase over the full run, with zero added per-round model payload in this accounting. Local median aggregation runtime rises from 0.000464 s to 0.101083 s, a 217.77x ratio (Table 2).

This is a functional single-process simulator using a deterministic pseudorandom generator rather than production cryptography. It assumes all clients participate, pairwise secrets remain hidden from the aggregator, and clients do not collude. It does not address dropout recovery, malicious clients, authentication, poisoning, metadata, global-model or output leakage, membership inference, or differential privacy.

## 6. Discussion

The results support a benchmark contribution more strongly than a universal defense contribution. BNCI and Lee show that representation bottlenecks can materially reduce linear subject-identification leakage while maintaining or improving task utility. Their membership results differ, especially under Lee threshold attacks. Reporting only one privacy metric would turn a qualified result into an unsupported generalization.

PhysioNet demonstrates why negative findings are informative. Wider subject scope, additional seeds, cross-run evaluation, and a bounded defense grid do not converge on one defense. Different attacks reward different model properties, and the strongest neural privacy extreme sacrifices substantial utility. The appropriate conclusion is unresolved, not solved and not awaiting selective confirmation.

The systems diagnostics reinforce the same separation of mechanisms. Federation limits raw-data movement but leaves a global model that can still leak membership. Secure aggregation can conceal individual updates from a narrow honest-but-curious aggregator, but the aggregate and final outputs remain available. Privacy claims must therefore name the protected object, adversary, and failure modes.

## 7. Limitations

- The primary matched uncertainty layer uses four task-model seeds, so small AUC deltas remain imprecise even when ranks differ.
- Subject-identification results are available for cross-session BNCI and Lee but are not directly comparable with the cross-subject PhysioNet protocol.
- The attack set is output- and feature-focused; adaptive white-box, reconstruction, attribute, and repeated-query adversaries are outside scope.
- Defense sweeps are bounded benchmark baselines rather than exhaustive hyperparameter searches or formally private mechanisms.
- The secure-aggregation implementation is a local functional simulator with no malicious-client, dropout, collusion, authentication, or production-cryptography guarantee.
- Results are specific to binary motor imagery, the audited preprocessing defaults, and the frozen dataset scopes.

## 8. Ethics and Data Use

This study analyzes public EEG datasets and introduces no new human-subject data collection. The work evaluates aggregate privacy leakage to improve risk reporting; it does not attempt to identify named individuals, infer health conditions, or release subject-level predictions. Public availability does not make EEG nonsensitive, so all findings are framed as bounded benchmark results rather than permission for unrestricted downstream use.

## 9. Reproducibility

Benchmark v1 is frozen. Paper-facing values were generated from archived sweep summaries through deterministic report scripts. The public release records claim-level evidence, protocol and dependency audits, selected assets, and release checksums. Public-release QA rejects missing claim boundaries, host-specific paths, prohibited artifacts, incomplete documentation, and checksum drift. No model training is required to inspect or validate the curated public evidence.

## 10. Conclusion

EEG privacy evaluation cannot be reduced to one defense or one leakage score. Bottleneck EEGNet is a qualified utility/subject-identity choice on BNCI and Lee, but membership behavior remains attack-sensitive. PhysioNet provides no stable default after expanded validation. Federation and secure aggregation protect different parts of the training system and do not eliminate output leakage. A credible benchmark should preserve these distinctions, enforce utility gates, and report negative or mixed evidence alongside positive results.

## References

- Ganin, Y., et al. (2016). Domain-Adversarial Training of Neural Networks. Journal of Machine Learning Research.
- Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. Circulation.
- Jayaram, V., and Barachant, A. (2018). MOABB: trustworthy algorithm benchmarking for BCIs. Journal of Neural Engineering.
- Lawhern, V. J., et al. (2018). EEGNet. Journal of Neural Engineering.
- Lee, M.-H., et al. (2019). EEG dataset and OpenBMI toolbox. GigaScience.
- Martinovic, I., et al. (2012). On the Feasibility of Side-Channel Attacks with Brain-Computer Interfaces. USENIX Security Symposium.
- Meng, L., et al. (2023). User Identity Protection in EEG-Based Brain-Computer Interfaces. IEEE TNSRE.
- Pereyra, G., et al. (2017). Regularizing Neural Networks by Penalizing Confident Output Distributions. ICLR Workshop.
- Ramoser, H., et al. (2000). Optimal spatial filtering of single trial EEG during imagined hand movement. IEEE TRE.
- Schalk, G. (2009). EEG Motor Movement/Imagery Dataset. PhysioNet.
- Schalk, G., et al. (2004). BCI2000. IEEE TBME.
- Shokri, R., et al. (2017). Membership Inference Attacks Against Machine Learning Models. IEEE S&P.
- Tangermann, M., et al. (2012). Review of the BCI Competition IV. Frontiers in Neuroscience.
- Zhang, H., et al. (2018). mixup. ICLR.
