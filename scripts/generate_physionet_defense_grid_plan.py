from __future__ import annotations

import argparse
import csv
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "outputs" / "reports"
PLAN_ROOT = ROOT / "outputs" / "plans"

DATASET = "physionet_motor_imagery"
PROTOCOL = "cross_subject"
DEFAULT_SUBJECTS = tuple(range(1, 47))
DEFAULT_SEEDS = "13,17,19,23"
DEFAULT_TASK_EPOCHS = 20
DEFAULT_TASK_BATCH_SIZE = 32
DEFAULT_MNE_DATA_DIR = "raw_data/mne_data"
THREAD_ENV = (
    "OMP_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
    "VECLIB_MAXIMUM_THREADS=1",
    "NUMEXPR_NUM_THREADS=1",
)


@dataclass(frozen=True)
class AttackConfig:
    attack_label: str
    attack_type: str
    score_type: str
    purpose: str


@dataclass(frozen=True)
class CandidateConfig:
    candidate_id: str
    row_type: str
    model: str
    setting: str
    extra_args: tuple[str, ...]
    purpose: str
    reference_paths: dict[str, str] | None = None


@dataclass(frozen=True)
class CommandRow:
    ordinal: int
    priority: str
    row_type: str
    candidate_id: str
    model: str
    setting: str
    attack_label: str
    attack_type: str
    score_type: str
    subjects: str
    seeds: str
    output_dir: str
    command: str
    reference_summary_path: str
    purpose: str


ATTACKS = (
    AttackConfig(
        attack_label="mlp_posterior",
        attack_type="mlp",
        score_type="posterior_probabilities",
        purpose="Primary nonlinear learned membership attack.",
    ),
    AttackConfig(
        attack_label="threshold_max_probability",
        attack_type="threshold",
        score_type="max_probability",
        purpose="Label-free confidence threshold attack.",
    ),
    AttackConfig(
        attack_label="threshold_label_known",
        attack_type="threshold",
        score_type="label_known_log_probability",
        purpose="Label-known threshold attack.",
    ),
)

REFERENCE_CANDIDATES = (
    CandidateConfig(
        candidate_id="eegnet_baseline",
        row_type="reference",
        model="eegnet",
        setting="plain EEGNet",
        extra_args=(),
        purpose="Same-protocol reference baseline for all defense deltas.",
        reference_paths={
            "mlp_posterior": "outputs/sweeps/membership_eegnet_physionet_motor_imagery_subject12345678910111213141516171819202122232425262728293031323334353637383940414243444546_mlp_posterior_4seed_v1/eegnet_physionet_motor_imagery_cross_subject_summary.json",
            "threshold_max_probability": "outputs/sweeps/membership_eegnet_physionet_motor_imagery_subject12345678910111213141516171819202122232425262728293031323334353637383940414243444546_threshold_maxprob_4seed_v1/eegnet_physionet_motor_imagery_cross_subject_summary.json",
            "threshold_label_known": "outputs/sweeps/membership_eegnet_physionet_motor_imagery_subject12345678910111213141516171819202122232425262728293031323334353637383940414243444546_threshold_label_known_logprob_4seed_v1/eegnet_physionet_motor_imagery_cross_subject_summary.json",
        },
    ),
    CandidateConfig(
        candidate_id="mixup_alpha_0p20_reference",
        row_type="reference",
        model="mixup_eegnet",
        setting="mixup alpha=0.20",
        extra_args=(),
        purpose="Previously completed mixup default; included as a frozen reference.",
        reference_paths={
            "mlp_posterior": "outputs/sweeps/membership_mixup_eegnet_physionet_motor_imagery_subject12345678910111213141516171819202122232425262728293031323334353637383940414243444546_alpha02_mlp_posterior_4seed_v1/mixup_eegnet_physionet_motor_imagery_cross_subject_summary.json",
            "threshold_max_probability": "outputs/sweeps/membership_mixup_eegnet_physionet_motor_imagery_subject12345678910111213141516171819202122232425262728293031323334353637383940414243444546_alpha0p20_threshold_maxprob_4seed_v1/mixup_eegnet_physionet_motor_imagery_cross_subject_summary.json",
            "threshold_label_known": "outputs/sweeps/membership_mixup_eegnet_physionet_motor_imagery_subject12345678910111213141516171819202122232425262728293031323334353637383940414243444546_alpha0p20_threshold_label_known_logprob_4seed_v1/mixup_eegnet_physionet_motor_imagery_cross_subject_summary.json",
        },
    ),
    CandidateConfig(
        candidate_id="confidence_beta_0p025_reference",
        row_type="reference",
        model="confidence_penalty_eegnet",
        setting="confidence penalty beta=0.025",
        extra_args=(),
        purpose="Previously completed confidence-penalty default; included as a frozen reference.",
        reference_paths={
            "mlp_posterior": "outputs/sweeps/membership_confidence_penalty_eegnet_physionet_motor_imagery_subject12345678910111213141516171819202122232425262728293031323334353637383940414243444546_beta0025_mlp_posterior_4seed_v1/confidence_penalty_eegnet_physionet_motor_imagery_cross_subject_summary.json",
            "threshold_max_probability": "outputs/sweeps/membership_confidence_penalty_eegnet_physionet_motor_imagery_subject12345678910111213141516171819202122232425262728293031323334353637383940414243444546_beta0p025_threshold_maxprob_4seed_v1/confidence_penalty_eegnet_physionet_motor_imagery_cross_subject_summary.json",
            "threshold_label_known": "outputs/sweeps/membership_confidence_penalty_eegnet_physionet_motor_imagery_subject12345678910111213141516171819202122232425262728293031323334353637383940414243444546_beta0p025_threshold_label_known_logprob_4seed_v1/confidence_penalty_eegnet_physionet_motor_imagery_cross_subject_summary.json",
        },
    ),
)

NEW_CANDIDATES = (
    CandidateConfig(
        candidate_id="mixup_alpha_0p10",
        row_type="planned",
        model="mixup_eegnet",
        setting="mixup alpha=0.10",
        extra_args=("--mixup-alpha", "0.10"),
        purpose="Test whether weaker mixup lowers utility cost without worsening leakage.",
    ),
    CandidateConfig(
        candidate_id="mixup_alpha_0p35",
        row_type="planned",
        model="mixup_eegnet",
        setting="mixup alpha=0.35",
        extra_args=("--mixup-alpha", "0.35"),
        purpose="Test whether stronger mixup creates a real privacy edge across attacks.",
    ),
    CandidateConfig(
        candidate_id="confidence_beta_0p010",
        row_type="planned",
        model="confidence_penalty_eegnet",
        setting="confidence penalty beta=0.010",
        extra_args=("--confidence-penalty-beta", "0.010"),
        purpose="Test a lighter confidence penalty around the previous beta=0.025 default.",
    ),
    CandidateConfig(
        candidate_id="confidence_beta_0p075",
        row_type="planned",
        model="confidence_penalty_eegnet",
        setting="confidence penalty beta=0.075",
        extra_args=("--confidence-penalty-beta", "0.075"),
        purpose="Test whether a stronger confidence penalty improves two-family privacy.",
    ),
    CandidateConfig(
        candidate_id="bottleneck_dim_4",
        row_type="planned",
        model="bottleneck_eegnet",
        setting="bottleneck dim=4",
        extra_args=("--bottleneck-dim", "4"),
        purpose="Test a tighter bottleneck after PhysioNet remained unresolved.",
    ),
    CandidateConfig(
        candidate_id="bottleneck_dim_12",
        row_type="planned",
        model="bottleneck_eegnet",
        setting="bottleneck dim=12",
        extra_args=("--bottleneck-dim", "12"),
        purpose="Test a wider bottleneck to recover utility while tracking leakage.",
    ),
)


def _subject_csv(subjects: tuple[int, ...]) -> str:
    return ",".join(str(subject) for subject in subjects)


def _output_dir(candidate: CandidateConfig, attack: AttackConfig) -> Path:
    return (
        Path("outputs")
        / "sweeps"
        / "physionet_defense_grid_v1"
        / candidate.candidate_id
        / attack.attack_label
    )


def _command_for(
    *,
    candidate: CandidateConfig,
    attack: AttackConfig,
    subjects: tuple[int, ...],
    seeds: str,
    task_epochs: int,
    task_batch_size: int,
    mne_data_dir: str,
) -> tuple[str, str]:
    output_dir = _output_dir(candidate, attack)
    args = [
        *THREAD_ENV,
        "PYTHONPATH=src",
        ".venv/bin/python",
        "-m",
        "eeg_privacy_benchmark.cli",
        "run-membership-inference-sweep",
        "--dataset",
        DATASET,
        "--model",
        candidate.model,
        "--protocol",
        PROTOCOL,
        "--subjects",
        _subject_csv(subjects),
        "--seeds",
        seeds,
        "--task-epochs",
        str(task_epochs),
        "--task-batch-size",
        str(task_batch_size),
        "--mne-data-dir",
        mne_data_dir,
        "--attack-type",
        attack.attack_type,
        "--score-type",
        attack.score_type,
        "--output-dir",
        str(output_dir),
        *candidate.extra_args,
    ]
    return str(output_dir), shlex.join(args)


def _build_rows(args: argparse.Namespace) -> list[CommandRow]:
    rows: list[CommandRow] = []
    ordinal = 0
    subjects = _subject_csv(DEFAULT_SUBJECTS)

    for candidate in (*REFERENCE_CANDIDATES, *NEW_CANDIDATES):
        for attack in ATTACKS:
            ordinal += 1
            reference_path = ""
            output_dir = ""
            command = ""
            if candidate.row_type == "reference":
                reference_path = (candidate.reference_paths or {})[attack.attack_label]
                output_dir = str(Path(reference_path).parent)
            else:
                output_dir, command = _command_for(
                    candidate=candidate,
                    attack=attack,
                    subjects=DEFAULT_SUBJECTS,
                    seeds=args.seeds,
                    task_epochs=args.task_epochs,
                    task_batch_size=args.task_batch_size,
                    mne_data_dir=args.mne_data_dir,
                )
            rows.append(
                CommandRow(
                    ordinal=ordinal,
                    priority="P1",
                    row_type=candidate.row_type,
                    candidate_id=candidate.candidate_id,
                    model=candidate.model,
                    setting=candidate.setting,
                    attack_label=attack.attack_label,
                    attack_type=attack.attack_type,
                    score_type=attack.score_type,
                    subjects=subjects,
                    seeds=args.seeds,
                    output_dir=output_dir,
                    command=command,
                    reference_summary_path=reference_path,
                    purpose=f"{candidate.purpose} {attack.purpose}",
                )
            )
    return rows


def _write_csv(rows: list[CommandRow]) -> Path:
    out_path = REPORT_ROOT / "physionet_defense_grid_plan_v1.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return out_path


def _write_shell(rows: list[CommandRow]) -> Path:
    out_path = PLAN_ROOT / "physionet_defense_grid_commands_v1.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by scripts/generate_physionet_defense_grid_plan.py.",
        "# Reference rows are intentionally omitted; they point to existing summaries.",
        "",
    ]
    for row in rows:
        if not row.command:
            continue
        lines.extend(
            [
                f"# {row.ordinal:02d} | {row.candidate_id} | {row.attack_type}::{row.score_type}",
                row.command,
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    out_path.chmod(0o755)
    return out_path


def _write_markdown(rows: list[CommandRow], csv_path: Path, shell_path: Path) -> Path:
    out_path = REPORT_ROOT / "physionet_defense_grid_plan_v1.md"
    planned = [row for row in rows if row.row_type == "planned"]
    references = [row for row in rows if row.row_type == "reference"]
    lines = [
        "# PhysioNet Defense Grid Plan",
        "",
        "This pre-registers the next PhysioNet defense-design step. It is intentionally narrow: two mixup strengths, two confidence-penalty strengths, and two bottleneck widths, all evaluated on the same three attack families already used in the robustness matrix.",
        "",
        "Companion files:",
        f"- [{csv_path.relative_to(ROOT)}]({csv_path.relative_to(ROOT)})",
        f"- [{shell_path.relative_to(ROOT)}]({shell_path.relative_to(ROOT)})",
        "",
        "## Scope",
        "",
        f"- Dataset/protocol: `{DATASET}` / `{PROTOCOL}`.",
        f"- Subjects: `{_subject_csv(DEFAULT_SUBJECTS)}`.",
        f"- Task seeds: `{rows[0].seeds}`.",
        f"- Planned new commands: `{len(planned)}`.",
        f"- Existing reference rows: `{len(references)}`.",
        "",
        "## Candidate Grid",
        "",
        "| Candidate | Model | Setting | Status |",
        "| --- | --- | --- | --- |",
    ]
    seen: set[str] = set()
    for row in rows:
        if row.candidate_id in seen:
            continue
        seen.add(row.candidate_id)
        status = "existing reference" if row.row_type == "reference" else "planned run"
        lines.append(
            f"| `{row.candidate_id}` | `{row.model}` | {row.setting} | {status} |"
        )

    lines.extend(
        [
            "",
            "## Attack Families",
            "",
        ]
    )
    for attack in ATTACKS:
        lines.append(
            f"- `{attack.attack_label}`: `{attack.attack_type}::{attack.score_type}`. {attack.purpose}"
        )

    lines.extend(
        [
            "",
            "## Stop Rule",
            "",
            "- Promote a PhysioNet defense only if it beats EEGNet by at least `0.010` lower AUC on at least two attack families while losing less than `0.030` task balanced accuracy on average.",
            "- If a candidate only wins one attack family, call it attack-specific rather than a default defense.",
            "- If candidates rotate winners or trade away more than `0.030` task balanced accuracy, keep PhysioNet unresolved.",
            "- Do not broaden beyond this grid until all planned rows are complete or a candidate clearly fails the utility boundary on the first two attack families.",
            "",
            "## Command Preview",
            "",
            "```bash",
            planned[0].command if planned else "# no planned commands",
            "```",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a pre-registered PhysioNet defense-grid command plan."
    )
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--task-epochs", type=int, default=DEFAULT_TASK_EPOCHS)
    parser.add_argument("--task-batch-size", type=int, default=DEFAULT_TASK_BATCH_SIZE)
    parser.add_argument("--mne-data-dir", default=DEFAULT_MNE_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    PLAN_ROOT.mkdir(parents=True, exist_ok=True)
    rows = _build_rows(args)
    csv_path = _write_csv(rows)
    shell_path = _write_shell(rows)
    markdown_path = _write_markdown(rows, csv_path, shell_path)
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {shell_path.relative_to(ROOT)}")
    print(f"Wrote {markdown_path.relative_to(ROOT)}")
    print(f"Generated {sum(1 for row in rows if row.command)} planned commands")


if __name__ == "__main__":
    main()
