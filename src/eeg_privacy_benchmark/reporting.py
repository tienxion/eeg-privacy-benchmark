"""Reporting helpers for benchmark experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path


def _load_summary(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _format_optional_stat(summary: dict, mean_key: str, std_key: str, precision: int = 4) -> str:
    if mean_key not in summary or std_key not in summary:
        return "n/a"
    return f"{summary[mean_key]:.{precision}f} +- {summary[std_key]:.{precision}f}"


def _format_baseline_delta(value: float, baseline: float, *, lower_is_better: bool) -> str:
    delta = value - baseline
    if abs(delta) < 0.00005:
        return "`0.0000`, effectively tied with the plain EEGNet baseline"

    direction = "lower" if delta < 0 else "higher"
    outcome = "better" if (delta < 0) == lower_is_better else "worse"
    return f"`{abs(delta):.4f}` {direction} than the plain EEGNet baseline ({outcome})"


def _format_defense_setting(summary: dict) -> str:
    schedule = summary.get("adversarial_schedule", "constant")
    weight = summary.get("adversarial_weight")
    if weight is None:
        return schedule
    if schedule == "linear_ramp":
        ramp_fraction = summary.get("ramp_up_fraction", 0.0)
        return f"linear_ramp@{weight:.2f}, ramp={ramp_fraction:.2f}"
    return f"constant@{weight:.2f}"


def _format_model_label(summary: dict) -> str:
    model = summary["model"]
    if model == "adversarial_eegnet":
        return f"{model} ({_format_defense_setting(summary)})"
    if model == "label_smoothing_eegnet":
        label_smoothing = summary.get("label_smoothing")
        if label_smoothing is not None:
            return f"{model} (eps={float(label_smoothing):.2f})"
    if model == "bottleneck_eegnet":
        bottleneck_dim = summary.get("bottleneck_dim")
        if bottleneck_dim is not None:
            return f"{model} (dim={int(bottleneck_dim)})"
    if model == "feature_noise_eegnet":
        feature_noise_std = summary.get("feature_noise_std")
        if feature_noise_std is not None:
            return f"{model} (sigma={float(feature_noise_std):.2f})"
    if model == "mixup_eegnet":
        mixup_alpha = summary.get("mixup_alpha")
        if mixup_alpha is not None:
            return f"{model} (alpha={float(mixup_alpha):.2f})"
    if model == "confidence_penalty_eegnet":
        confidence_penalty_beta = summary.get("confidence_penalty_beta")
        if confidence_penalty_beta is not None:
            return f"{model} (beta={float(confidence_penalty_beta):.2f})"
    return model


def _membership_score_note(attack_type: str, score_type: str) -> str:
    if attack_type == "logistic_regression":
        if score_type == "posterior_probabilities_plus_true_label":
            return (
                "- This attack fits a logistic-regression membership classifier on held-out "
                "task-model posterior vectors concatenated with the true-label one-hot feature, "
                "then evaluates on a disjoint member/non-member split."
            )
        return (
            "- This attack fits a logistic-regression membership classifier on held-out "
            "task-model posterior vectors and reports evaluation metrics on a disjoint "
            "member/non-member split."
        )
    if attack_type == "mlp":
        if score_type == "posterior_probabilities_plus_true_label":
            return (
                "- This attack fits a small MLP membership classifier on held-out "
                "task-model posterior vectors concatenated with the true-label one-hot feature, "
                "then evaluates on a disjoint member/non-member split."
            )
        return (
            "- This attack fits a small MLP membership classifier on held-out "
            "task-model posterior vectors and reports evaluation metrics on a disjoint "
            "member/non-member split."
        )
    if score_type == "label_known_log_probability":
        return (
            "- This attack is label-known and score-threshold based, using the true-label "
            "log probability as the membership score."
        )
    if score_type == "max_probability":
        return (
            "- This attack is label-free and score-threshold based, using the model's "
            "maximum predicted class probability as the membership score."
        )
    if score_type == "negative_entropy":
        return (
            "- This attack is label-free and score-threshold based, using negative "
            "predictive entropy as the membership score."
        )
    return "- This report summarizes a scalar score-threshold membership attack."


def build_subject_id_comparison_report(
    *,
    summary_paths: list[str | Path],
    output_path: str | Path,
) -> None:
    """Build a Markdown comparison report from subject-ID sweep summaries."""

    summaries = [_load_summary(path) for path in summary_paths]
    if not summaries:
        raise ValueError("At least one summary path is required.")

    dataset_keys = {summary["dataset_key"] for summary in summaries}
    protocols = {summary["protocol"] for summary in summaries}
    subject_sets = {tuple(summary.get("subjects") or []) for summary in summaries}
    if len(dataset_keys) != 1 or len(protocols) != 1 or len(subject_sets) != 1:
        raise ValueError(
            "All summaries in one comparison report must share dataset, protocol, and subjects."
        )

    dataset_key = next(iter(dataset_keys))
    protocol = next(iter(protocols))
    subjects = list(next(iter(subject_sets)))
    num_subjects = len(subjects)
    chance_accuracy = (1.0 / num_subjects) if num_subjects else 0.0

    ranked = sorted(
        summaries,
        key=lambda summary: summary["subject_id_accuracy_mean"],
        reverse=True,
    )
    lines = [
        f"# Subject-ID Comparison Report: {dataset_key}",
        "",
        f"- protocol: `{protocol}`",
        f"- subjects: `{subjects}`",
        f"- subject-count: `{num_subjects}`",
        f"- chance subject-ID accuracy: `{chance_accuracy:.4f}`",
        "",
        "## Model Summary",
        "",
        "| Model | Task bal. acc. mean +- std | Subject-ID acc. mean +- std | Subject-ID macro-F1 mean +- std |",
        "| --- | --- | --- | --- |",
    ]
    for summary in ranked:
        lines.append(
            "| "
            f"{_format_model_label(summary)} | "
            f"{summary['task_balanced_accuracy_mean']:.4f} +- {summary['task_balanced_accuracy_std']:.4f} | "
            f"{summary['subject_id_accuracy_mean']:.4f} +- {summary['subject_id_accuracy_std']:.4f} | "
            f"{summary['subject_id_macro_f1_mean']:.4f} +- {summary['subject_id_macro_f1_std']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"The highest leakage in this comparison comes from `{_format_model_label(ranked[0])}`, "
                f"with mean subject-ID accuracy `{ranked[0]['subject_id_accuracy_mean']:.4f}`, "
                f"which is `{ranked[0]['subject_id_accuracy_mean'] - chance_accuracy:.4f}` above chance."
            ),
            (
                f"The lowest leakage in this comparison comes from `{_format_model_label(ranked[-1])}`, "
                f"with mean subject-ID accuracy `{ranked[-1]['subject_id_accuracy_mean']:.4f}`."
            ),
        ]
    )

    if len(ranked) >= 2:
        leakage_gap = ranked[0]["subject_id_accuracy_mean"] - ranked[-1]["subject_id_accuracy_mean"]
        task_gap = ranked[0]["task_balanced_accuracy_mean"] - ranked[-1]["task_balanced_accuracy_mean"]
        lines.extend(
            [
                (
                    f"The leakage gap between highest- and lowest-leakage models is "
                    f"`{leakage_gap:.4f}` subject-ID accuracy points."
                ),
                (
                    f"The corresponding task balanced-accuracy gap is "
                    f"`{task_gap:.4f}`."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `subject_id_top3_accuracy` is not summarized here because it becomes uninformative when the subject count is small.",
            "- The next useful extension is the same comparison on `Lee2019_MI` and then a first defense baseline.",
            "",
        ]
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def build_adversarial_defense_report(
    *,
    baseline_summary_path: str | Path,
    defense_summary_paths: list[str | Path],
    output_path: str | Path,
) -> None:
    """Build a Markdown report comparing plain and adversarial EEGNet sweeps."""

    baseline = _load_summary(baseline_summary_path)
    defenses = [_load_summary(path) for path in defense_summary_paths]
    if not defenses:
        raise ValueError("At least one defense summary path is required.")

    dataset_keys = {baseline["dataset_key"], *(summary["dataset_key"] for summary in defenses)}
    protocols = {baseline["protocol"], *(summary["protocol"] for summary in defenses)}
    subject_sets = {
        tuple(baseline.get("subjects") or []),
        *(tuple(summary.get("subjects") or []) for summary in defenses),
    }
    if len(dataset_keys) != 1 or len(protocols) != 1 or len(subject_sets) != 1:
        raise ValueError(
            "Baseline and defense summaries must share dataset, protocol, and subjects."
        )

    dataset_key = baseline["dataset_key"]
    protocol = baseline["protocol"]
    subjects = list(next(iter(subject_sets)))
    chance_accuracy = (1.0 / len(subjects)) if subjects else 0.0
    if "subject_id_accuracy_mean" in baseline:
        privacy_mean_key = "subject_id_accuracy_mean"
        privacy_std_key = "subject_id_accuracy_std"
        privacy_label = "Subject-ID acc."
        privacy_note_label = "subject-ID accuracy"
        chance_line = f"- chance subject-ID accuracy: `{chance_accuracy:.4f}`"
        baseline_privacy_line = (
            f"- subject-ID accuracy: `{baseline[privacy_mean_key]:.4f}` "
            f"+- `{baseline[privacy_std_key]:.4f}`"
        )
    elif "attack_auc_mean" in baseline:
        privacy_mean_key = "attack_auc_mean"
        privacy_std_key = "attack_auc_std"
        privacy_label = "Membership AUC"
        privacy_note_label = "membership AUC"
        chance_line = "- chance membership AUC: `0.5000`"
        baseline_privacy_line = (
            f"- membership AUC: `{baseline[privacy_mean_key]:.4f}` "
            f"+- `{baseline[privacy_std_key]:.4f}`"
        )
    else:
        raise ValueError(
            "Defense reports require either subject-ID or membership summary metrics."
        )
    ranked_defenses = sorted(
        defenses,
        key=lambda summary: (
            summary.get("adversarial_schedule", "constant"),
            float(summary.get("adversarial_weight", 0.0)),
        ),
    )
    lowest_leakage = min(defenses, key=lambda summary: summary[privacy_mean_key])
    highest_utility = max(defenses, key=lambda summary: summary["task_balanced_accuracy_mean"])

    lines = [
        f"# Adversarial Defense Report: {dataset_key}",
        "",
        f"- protocol: `{protocol}`",
        f"- subjects: `{subjects}`",
        chance_line,
        "",
        "## Plain EEGNet Baseline",
        "",
        (
            f"- task balanced accuracy: `{baseline['task_balanced_accuracy_mean']:.4f}` "
            f"+- `{baseline['task_balanced_accuracy_std']:.4f}`"
        ),
        baseline_privacy_line,
        "",
        "## Adversarial Sweep",
        "",
        f"| Setting | Task bal. acc. mean +- std | {privacy_label} mean +- std | Epochs trained mean +- std | Best val. loss mean +- std |",
        "| --- | --- | --- | --- | --- |",
    ]

    for summary in ranked_defenses:
        lines.append(
            "| "
            f"{_format_defense_setting(summary)} | "
            f"{summary['task_balanced_accuracy_mean']:.4f} +- {summary['task_balanced_accuracy_std']:.4f} | "
            f"{summary[privacy_mean_key]:.4f} +- {summary[privacy_std_key]:.4f} | "
            f"{_format_optional_stat(summary, 'epochs_trained_mean', 'epochs_trained_std', precision=2)} | "
            f"{_format_optional_stat(summary, 'best_validation_loss_mean', 'best_validation_loss_std')} |"
        )

    baseline_leakage = baseline[privacy_mean_key]
    baseline_task = baseline["task_balanced_accuracy_mean"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"The lowest-leakage defense setting is `{_format_defense_setting(lowest_leakage)}`, "
                f"with {privacy_note_label} `{lowest_leakage[privacy_mean_key]:.4f}`. "
                f"That is {_format_baseline_delta(lowest_leakage[privacy_mean_key], baseline_leakage, lower_is_better=True)}."
            ),
            (
                f"The highest-utility defense setting is `{_format_defense_setting(highest_utility)}`, "
                f"with task balanced accuracy `{highest_utility['task_balanced_accuracy_mean']:.4f}`. "
                f"That is {_format_baseline_delta(highest_utility['task_balanced_accuracy_mean'], baseline_task, lower_is_better=False)}."
            ),
            "",
            "## Notes",
            "",
            "- These defense results should be read as privacy/utility tradeoffs, not as standalone accuracy numbers.",
            "- Epoch and validation-loss summaries are included to show whether a weight is merely undertraining the task model.",
            "",
        ]
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def build_membership_inference_report(
    *,
    summary_paths: list[str | Path],
    output_path: str | Path,
) -> None:
    """Build a Markdown comparison report from membership-inference summaries."""

    summaries = [_load_summary(path) for path in summary_paths]
    if not summaries:
        raise ValueError("At least one membership summary path is required.")

    dataset_keys = {summary["dataset_key"] for summary in summaries}
    protocols = {summary["protocol"] for summary in summaries}
    subject_sets = {tuple(summary.get("subjects") or []) for summary in summaries}
    attack_types = {summary["attack_type"] for summary in summaries}
    score_types = {summary["score_type"] for summary in summaries}
    if (
        len(dataset_keys) != 1
        or len(protocols) != 1
        or len(subject_sets) != 1
        or len(attack_types) != 1
        or len(score_types) != 1
    ):
        raise ValueError(
            "All membership summaries in one report must share dataset, protocol, "
            "subjects, attack type, and score type."
        )

    dataset_key = next(iter(dataset_keys))
    protocol = next(iter(protocols))
    subjects = list(next(iter(subject_sets)))
    attack_type = next(iter(attack_types))
    score_type = next(iter(score_types))

    ranked = sorted(summaries, key=lambda summary: summary["attack_auc_mean"], reverse=True)
    lines = [
        f"# Membership-Inference Report: {dataset_key}",
        "",
        f"- protocol: `{protocol}`",
        f"- subjects: `{subjects}`",
        f"- attack type: `{attack_type}`",
        f"- score type: `{score_type}`",
        "",
        "## Model Summary",
        "",
        "| Model | Task bal. acc. mean +- std | Attack AUC mean +- std | Attack AP mean +- std | Attack bal. acc. mean +- std |",
        "| --- | --- | --- | --- | --- |",
    ]
    for summary in ranked:
        lines.append(
            "| "
            f"{_format_model_label(summary)} | "
            f"{summary['task_balanced_accuracy_mean']:.4f} +- {summary['task_balanced_accuracy_std']:.4f} | "
            f"{summary['attack_auc_mean']:.4f} +- {summary['attack_auc_std']:.4f} | "
            f"{summary['attack_average_precision_mean']:.4f} +- {summary['attack_average_precision_std']:.4f} | "
            f"{summary['attack_balanced_accuracy_mean']:.4f} +- {summary['attack_balanced_accuracy_std']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"The strongest membership signal in this comparison comes from "
                f"`{_format_model_label(ranked[0])}`, with mean attack AUC "
                f"`{ranked[0]['attack_auc_mean']:.4f}`."
            ),
            (
                f"The weakest membership signal in this comparison comes from "
                f"`{_format_model_label(ranked[-1])}`, with mean attack AUC "
                f"`{ranked[-1]['attack_auc_mean']:.4f}`."
            ),
        ]
    )

    if len(ranked) >= 2:
        auc_gap = ranked[0]["attack_auc_mean"] - ranked[-1]["attack_auc_mean"]
        lines.append(
            f"The attack-AUC gap between strongest and weakest membership leakage is `{auc_gap:.4f}`."
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            _membership_score_note(attack_type, score_type),
            "- Treat this as a baseline privacy audit, not a final attack upper bound.",
            "",
        ]
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
