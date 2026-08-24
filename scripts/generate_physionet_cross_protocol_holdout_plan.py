from __future__ import annotations

import csv
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "outputs" / "reports"
PLAN_ROOT = ROOT / "outputs" / "plans"

DATASET = "physionet_motor_imagery"
PROTOCOL = "cross_run"
SUBJECTS = tuple(range(1, 47))
SEEDS = "13,17,19,23"
TASK_EPOCHS = 20
TASK_BATCH_SIZE = 32
MNE_DATA_DIR = "raw_data/mne_data"
THREAD_ENV = (
    "OMP_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
    "VECLIB_MAXIMUM_THREADS=1",
    "NUMEXPR_NUM_THREADS=1",
)

MD_OUT = REPORT_ROOT / "physionet_cross_protocol_holdout_plan_v1.md"
CSV_OUT = REPORT_ROOT / "physionet_cross_protocol_holdout_plan_v1.csv"
COMMANDS_OUT = PLAN_ROOT / "physionet_cross_protocol_holdout_commands_v1.sh"


@dataclass(frozen=True)
class ModelConfig:
    candidate_id: str
    model: str
    setting: str
    extra_args: tuple[str, ...]
    purpose: str
    include_in_mlp: bool = True
    include_in_threshold: bool = True


@dataclass(frozen=True)
class AttackConfig:
    attack_label: str
    attack_type: str
    score_type: str
    purpose: str


@dataclass(frozen=True)
class CommandRow:
    ordinal: int
    candidate_id: str
    model: str
    setting: str
    protocol: str
    attack_label: str
    attack_type: str
    score_type: str
    subjects: str
    seeds: str
    output_dir: str
    command: str
    purpose: str


MODELS = (
    ModelConfig(
        candidate_id="eegnet_baseline",
        model="eegnet",
        setting="plain EEGNet",
        extra_args=(),
        purpose="Cross-run reference baseline for protocol-sensitivity deltas.",
    ),
    ModelConfig(
        candidate_id="mixup_alpha_0p20",
        model="mixup_eegnet",
        setting="mixup alpha=0.20",
        extra_args=("--mixup-alpha", "0.20"),
        purpose="Retest the previously unsupported apparent learned-attack mixup edge under a different protocol.",
    ),
    ModelConfig(
        candidate_id="bottleneck_dim_6",
        model="bottleneck_eegnet",
        setting="bottleneck dim=6",
        extra_args=("--bottleneck-dim", "6"),
        purpose="Check whether the cross-dataset bottleneck default remains utility-limited or gains relevance on PhysioNet cross-run.",
    ),
    ModelConfig(
        candidate_id="csp_lda_anchor",
        model="csp_lda",
        setting="CSP-LDA",
        extra_args=(),
        purpose="Carry the threshold-family anchor into the protocol-change check.",
        include_in_mlp=False,
    ),
)

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


def _subject_csv() -> str:
    return ",".join(str(subject) for subject in SUBJECTS)


def _markdown_path(path: Path) -> str:
    try:
        return str(ROOT / path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _output_dir(model: ModelConfig, attack: AttackConfig) -> Path:
    return (
        Path("outputs")
        / "sweeps"
        / "physionet_cross_protocol_holdout_v1"
        / model.candidate_id
        / attack.attack_label
    )


def _command_for(model: ModelConfig, attack: AttackConfig) -> tuple[str, str]:
    output_dir = _output_dir(model, attack)
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
        model.model,
        "--protocol",
        PROTOCOL,
        "--subjects",
        _subject_csv(),
        "--seeds",
        SEEDS,
        "--task-epochs",
        str(TASK_EPOCHS),
        "--task-batch-size",
        str(TASK_BATCH_SIZE),
        "--mne-data-dir",
        MNE_DATA_DIR,
        "--attack-type",
        attack.attack_type,
        "--score-type",
        attack.score_type,
        "--output-dir",
        str(output_dir),
        *model.extra_args,
    ]
    return str(output_dir), shlex.join(args)


def _build_rows() -> list[CommandRow]:
    rows = []
    ordinal = 1
    for attack in ATTACKS:
        for model in MODELS:
            if attack.attack_type == "mlp" and not model.include_in_mlp:
                continue
            if attack.attack_type == "threshold" and not model.include_in_threshold:
                continue
            output_dir, command = _command_for(model, attack)
            rows.append(
                CommandRow(
                    ordinal=ordinal,
                    candidate_id=model.candidate_id,
                    model=model.model,
                    setting=model.setting,
                    protocol=PROTOCOL,
                    attack_label=attack.attack_label,
                    attack_type=attack.attack_type,
                    score_type=attack.score_type,
                    subjects=_subject_csv(),
                    seeds=SEEDS,
                    output_dir=output_dir,
                    command=command,
                    purpose=f"{model.purpose} {attack.purpose}",
                )
            )
            ordinal += 1
    return rows


def _write_csv(rows: list[CommandRow]) -> Path:
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return CSV_OUT


def _write_commands(rows: list[CommandRow]) -> Path:
    COMMANDS_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by scripts/generate_physionet_cross_protocol_holdout_plan.py.",
        "# Review the companion Markdown plan before running these commands.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"# {row.ordinal:02d}: {row.candidate_id} {row.attack_type}::{row.score_type}",
                row.command,
                "",
            ]
        )
    COMMANDS_OUT.write_text("\n".join(lines), encoding="utf-8")
    return COMMANDS_OUT


def _write_markdown(
    rows: list[CommandRow],
    csv_path: Path,
    commands_path: Path,
) -> Path:
    lines = [
        "# PhysioNet Cross-Protocol Holdout Plan",
        "",
        "This plan operationalizes the selected `cross_protocol_physionet_holdout` target. It changes the PhysioNet protocol from the saturated subjects `1-46` cross-subject slice to a fixed cross-run holdout while keeping the subject set, seeds, model family, and attack families narrow. PhysioNet exposes one MOABB session for these trials, so the alternate holdout axis is the repeated recording run rather than a session label.",
        "",
        f"Companion CSV: [physionet_cross_protocol_holdout_plan_v1.csv]({_markdown_path(csv_path)})",
        f"Command script: [physionet_cross_protocol_holdout_commands_v1.sh]({_markdown_path(commands_path)})",
        "",
        "## Scope",
        "",
        f"- Dataset/protocol: `{DATASET}` / `{PROTOCOL}`.",
        f"- Subjects: `{_subject_csv()}`.",
        f"- Seeds: `{SEEDS}`.",
        f"- Planned commands: `{len(rows)}`.",
        "- Models: plain EEGNet, mixup alpha=0.20, bottleneck dim=6, and CSP-LDA threshold anchor.",
        "- Attacks: MLP posterior, max-probability threshold, and label-known log-probability threshold.",
        "",
        "## Pre-Registered Decision Rules",
        "",
        "- Do not broaden this into a new hyperparameter grid unless the cross-run protocol first changes the PhysioNet conclusion.",
        "- Promote a PhysioNet defense only if it improves membership AUC by at least `0.010` on at least two attack families, improves at least `75%` of paired comparisons, and loses less than `0.030` mean task balanced accuracy.",
        "- If winners rotate across attacks or utility loss exceeds the threshold, keep PhysioNet framed as unresolved and protocol-sensitive.",
        "- Treat this as protocol-sensitivity evidence, not as a replacement for the completed cross-subject, random-subset, attacker-refit, seed-extension, or defense-grid results.",
        "",
        "## Planned Commands",
        "",
        "| Ordinal | Candidate | Model | Attack | Output directory |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.ordinal}` | `{row.candidate_id}` | `{row.model}` | "
            f"`{row.attack_type}::{row.score_type}` | `{row.output_dir}` |"
        )
    lines.extend(
        [
            "",
            "## Safe Preview",
            "",
            "This generator only writes commands. Do not execute the command script until the protocol-change scope above has been reviewed.",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")
    return MD_OUT


def main() -> None:
    rows = _build_rows()
    csv_path = _write_csv(rows)
    commands_path = _write_commands(rows)
    md_path = _write_markdown(rows, csv_path, commands_path)
    print(f"Wrote {_display_path(csv_path)}")
    print(f"Wrote {_display_path(commands_path)}")
    print(f"Wrote {_display_path(md_path)}")
    print(f"Generated planned commands={len(rows)}")


if __name__ == "__main__":
    main()
