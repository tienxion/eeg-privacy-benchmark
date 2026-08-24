from __future__ import annotations

import argparse
import csv
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "outputs" / "reports"
PLAN_ROOT = ROOT / "outputs" / "plans"

DEFAULT_SEEDS = "13,17,19,23"
DEFAULT_TASK_EPOCHS = 20
DEFAULT_TASK_BATCH_SIZE = 32
DEFAULT_MNE_DATA_DIR = "raw_data/mne_data"
ENV_PREFIX = (
    "OMP_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
    "VECLIB_MAXIMUM_THREADS=1",
    "NUMEXPR_NUM_THREADS=1",
    "PYTHONPATH=src",
)


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str
    label: str
    subjects: tuple[int, ...]
    prior_subjects: tuple[int, ...]


@dataclass(frozen=True)
class ModelConfig:
    model: str
    label: str
    slug: str
    extra_args: tuple[str, ...]


@dataclass(frozen=True)
class ProbeConfig:
    tier: str
    probe: str
    attack_type: str
    score_type: str
    label: str


@dataclass(frozen=True)
class CommandRow:
    ordinal: int
    tier: str
    dataset: str
    model: str
    probe: str
    attack_type: str
    score_type: str
    seeds: str
    subjects: str
    output_dir: str
    command: str
    purpose: str


DATASETS = (
    DatasetConfig(
        dataset="bnci2014_001",
        label="BNCI 2014-001 full nine-subject cross-session expansion",
        subjects=tuple(range(1, 10)),
        prior_subjects=tuple(range(1, 5)),
    ),
    DatasetConfig(
        dataset="lee2019_mi",
        label="Lee2019 MI twelve-subject cross-session expansion",
        subjects=tuple(range(1, 13)),
        prior_subjects=tuple(range(1, 7)),
    ),
)

MODELS = (
    ModelConfig("eegnet", "plain EEGNet baseline", "eegnet", ()),
    ModelConfig(
        "bottleneck_eegnet",
        "bottleneck EEGNet dim=6",
        "bottleneck_dim6",
        ("--bottleneck-dim", "6"),
    ),
    ModelConfig(
        "adversarial_eegnet",
        "subject-adversarial EEGNet constant@1.0",
        "adversarial_constant_w1p0",
        ("--adversarial-weight", "1.0", "--adversarial-schedule", "constant"),
    ),
    ModelConfig("csp_lda", "CSP-LDA classical anchor", "csp_lda", ()),
)

PROBES = (
    ProbeConfig("A", "subject_id", "subject_id", "linear_probe", "subject-ID leakage"),
    ProbeConfig("A", "membership", "mlp", "posterior_probabilities", "nonlinear learned membership"),
    ProbeConfig(
        "B",
        "membership",
        "logistic_regression",
        "posterior_probabilities_plus_true_label",
        "label-known learned membership",
    ),
    ProbeConfig(
        "B",
        "membership",
        "logistic_regression",
        "posterior_probabilities",
        "posterior-only learned membership",
    ),
    ProbeConfig(
        "C",
        "membership",
        "threshold",
        "label_known_log_probability",
        "label-known threshold membership",
    ),
    ProbeConfig(
        "C",
        "membership",
        "threshold",
        "max_probability",
        "label-free max-probability threshold membership",
    ),
    ProbeConfig(
        "C",
        "membership",
        "threshold",
        "negative_entropy",
        "label-free negative-entropy threshold membership",
    ),
)


def _subject_csv(subjects: tuple[int, ...]) -> str:
    return ",".join(str(subject) for subject in subjects)


def _subject_slug(subjects: tuple[int, ...]) -> str:
    return "subject" + "".join(str(subject) for subject in subjects)


def _score_slug(probe: ProbeConfig) -> str:
    if probe.probe == "subject_id":
        return "subject_id"
    if probe.attack_type == "mlp":
        return "mlp_posterior"
    if probe.attack_type == "logistic_regression" and probe.score_type == "posterior_probabilities":
        return "posterior_lr"
    if probe.attack_type == "logistic_regression":
        return "posterior_label_lr"
    if probe.score_type == "label_known_log_probability":
        return "threshold_label_known"
    if probe.score_type == "max_probability":
        return "threshold_maxprob"
    if probe.score_type == "negative_entropy":
        return "threshold_negative_entropy"
    return f"{probe.attack_type}_{probe.score_type}"


def _output_dir(dataset: DatasetConfig, model: ModelConfig, probe: ProbeConfig) -> Path:
    return (
        Path("outputs")
        / "sweeps"
        / "bnci_lee_validation_v1"
        / dataset.dataset
        / _subject_slug(dataset.subjects)
        / model.slug
        / _score_slug(probe)
    )


def _command_for(
    *,
    dataset: DatasetConfig,
    model: ModelConfig,
    probe: ProbeConfig,
    seeds: str,
    task_epochs: int,
    task_batch_size: int,
    mne_data_dir: str,
) -> tuple[str, str]:
    output_dir = _output_dir(dataset, model, probe)
    subcommand = (
        "run-subject-id-sweep" if probe.probe == "subject_id" else "run-membership-inference-sweep"
    )
    args = [
        *ENV_PREFIX,
        ".venv/bin/python",
        "-m",
        "eeg_privacy_benchmark.cli",
        subcommand,
        "--dataset",
        dataset.dataset,
        "--model",
        model.model,
        "--protocol",
        "cross_session",
        "--subjects",
        _subject_csv(dataset.subjects),
        "--seeds",
        seeds,
        "--task-epochs",
        str(task_epochs),
        "--task-batch-size",
        str(task_batch_size),
        "--mne-data-dir",
        mne_data_dir,
        "--output-dir",
        str(output_dir),
        *model.extra_args,
    ]
    if probe.probe == "membership":
        args.extend(["--attack-type", probe.attack_type, "--score-type", probe.score_type])
    return str(output_dir), shlex.join(args)


def _build_rows(args: argparse.Namespace) -> list[CommandRow]:
    rows: list[CommandRow] = []
    ordinal = 1
    tiers = set(args.tiers.split(",")) if args.tiers else {"A", "B", "C"}
    for dataset in DATASETS:
        for probe in PROBES:
            if probe.tier not in tiers:
                continue
            for model in MODELS:
                output_dir, command = _command_for(
                    dataset=dataset,
                    model=model,
                    probe=probe,
                    seeds=args.seeds,
                    task_epochs=args.task_epochs,
                    task_batch_size=args.task_batch_size,
                    mne_data_dir=args.mne_data_dir,
                )
                rows.append(
                    CommandRow(
                        ordinal=ordinal,
                        tier=probe.tier,
                        dataset=dataset.dataset,
                        model=model.model,
                        probe=probe.probe,
                        attack_type=probe.attack_type,
                        score_type=probe.score_type,
                        seeds=args.seeds,
                        subjects=_subject_csv(dataset.subjects),
                        output_dir=output_dir,
                        command=command,
                        purpose=f"{dataset.label}: {probe.label} for {model.label}.",
                    )
                )
                ordinal += 1
    return rows


def _write_csv(rows: list[CommandRow]) -> Path:
    out_path = REPORT_ROOT / "bnci_lee_validation_plan_v1.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return out_path


def _write_shell(rows: list[CommandRow]) -> Path:
    out_path = PLAN_ROOT / "bnci_lee_validation_commands_v1.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by scripts/generate_bnci_lee_validation_plan.py.",
        "# Runs the staged BNCI/Lee subject-coverage validation sweeps.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"# {row.ordinal:02d} | tier {row.tier} | {row.dataset} | {row.model} | {row.attack_type}::{row.score_type}",
                row.command,
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    out_path.chmod(0o755)
    return out_path


def _write_markdown(rows: list[CommandRow], csv_path: Path, shell_path: Path) -> Path:
    out_path = REPORT_ROOT / "bnci_lee_validation_plan_v1.md"
    by_tier = {tier: sum(1 for row in rows if row.tier == tier) for tier in ("A", "B", "C")}
    lines = [
        "# BNCI/Lee Validation Plan",
        "",
        "This plan turns the remaining P1 item into exact local-cache commands. It expands the current frozen BNCI and Lee subject subsets while keeping the model set, seed set, and attack labels narrow enough to test the bottleneck claim without starting a broad hyperparameter sweep.",
        "",
        "Companion files:",
        f"- [{csv_path.relative_to(ROOT)}]({csv_path.relative_to(ROOT)})",
        f"- [{shell_path.relative_to(ROOT)}]({shell_path.relative_to(ROOT)})",
        "",
        "## Scope",
        "",
    ]
    for dataset in DATASETS:
        lines.append(
            f"- `{dataset.dataset}`: prior subjects `{_subject_csv(dataset.prior_subjects)}`; validation subjects `{_subject_csv(dataset.subjects)}`."
        )
    lines.extend(
        [
            f"- Seeds: `{rows[0].seeds}`.",
            "- Models: `eegnet`, `bottleneck_eegnet --bottleneck-dim 6`, `adversarial_eegnet --adversarial-weight 1.0 --adversarial-schedule constant`, and `csp_lda`.",
            "- Thread caps are embedded in every command to avoid high-concurrency BLAS oversubscription.",
            f"- Generated commands: `{len(rows)}`.",
            "",
            "## Tiers",
            "",
            f"- Tier A (`{by_tier['A']}` commands): subject-ID leakage plus MLP-posterior membership. Run this first because it directly tests whether the bottleneck story survives larger subject coverage.",
            f"- Tier B (`{by_tier['B']}` commands): logistic-regression learned attacks with posterior-only and posterior-plus-true-label features.",
            f"- Tier C (`{by_tier['C']}` commands): threshold attacks using label-known log-probability, max probability, and negative entropy.",
            "",
            "## Stop Criteria",
            "",
            "- Keep bottleneck as the BNCI/Lee main defense only if task balanced accuracy and subject-ID reductions persist versus plain EEGNet.",
            "- Do not upgrade bottleneck to an attack-invariant membership defense unless learned and threshold attack AUCs are consistently no worse than EEGNet by more than +0.005.",
            "- If CSP-LDA or adversarial constant wins raw membership AUC while bottleneck keeps better task/subject-ID utility, keep the current qualified-default wording rather than changing the paper's main claim.",
            "- Stop after Tier A if bottleneck loses both task utility and subject-ID advantage on either dataset.",
            "",
            "## Command Preview",
            "",
            "```bash",
            rows[0].command,
            "```",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate staged BNCI/Lee validation commands.")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--task-epochs", type=int, default=DEFAULT_TASK_EPOCHS)
    parser.add_argument("--task-batch-size", type=int, default=DEFAULT_TASK_BATCH_SIZE)
    parser.add_argument("--mne-data-dir", default=DEFAULT_MNE_DATA_DIR)
    parser.add_argument(
        "--tiers",
        default="A,B,C",
        help="Comma-separated tiers to include. Default: A,B,C.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    PLAN_ROOT.mkdir(parents=True, exist_ok=True)
    rows = _build_rows(args)
    if not rows:
        raise SystemExit("No commands generated; check --tiers.")
    csv_path = _write_csv(rows)
    shell_path = _write_shell(rows)
    markdown_path = _write_markdown(rows, csv_path, shell_path)
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {shell_path.relative_to(ROOT)}")
    print(f"Wrote {markdown_path.relative_to(ROOT)}")
    print(f"Generated {len(rows)} commands")


if __name__ == "__main__":
    main()
