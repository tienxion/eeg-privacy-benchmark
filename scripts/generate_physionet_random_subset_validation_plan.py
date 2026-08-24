from __future__ import annotations

import argparse
import csv
import random
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "outputs" / "reports"
PLAN_ROOT = ROOT / "outputs" / "plans"

DATASET = "physionet_motor_imagery"
PROTOCOL = "cross_subject"
DEFAULT_SUBJECT_POOL = tuple(range(1, 47))
DEFAULT_TASK_EPOCHS = 20
DEFAULT_TASK_BATCH_SIZE = 32
DEFAULT_MNE_DATA_DIR = "raw_data/mne_data"
DEFAULT_SEEDS = "13,17,19,23"


@dataclass(frozen=True)
class ModelConfig:
    model: str
    label: str
    extra_args: tuple[str, ...]
    include_in_mlp: bool = True
    include_in_threshold: bool = True


@dataclass(frozen=True)
class AttackConfig:
    tier: str
    attack_type: str
    score_type: str
    label: str
    purpose: str


@dataclass(frozen=True)
class CommandRow:
    tier: str
    subset_id: str
    subset_size: int
    subjects: str
    model: str
    attack_type: str
    score_type: str
    output_dir: str
    command: str
    purpose: str


MODEL_CONFIGS = (
    ModelConfig("eegnet", "plain EEGNet", ()),
    ModelConfig("mixup_eegnet", "mixup EEGNet alpha=0.20", ("--mixup-alpha", "0.2")),
    ModelConfig(
        "confidence_penalty_eegnet",
        "confidence-penalty EEGNet beta=0.025",
        ("--confidence-penalty-beta", "0.025"),
    ),
    ModelConfig(
        "adversarial_eegnet",
        "subject-adversarial EEGNet linear_ramp@1.0",
        (
            "--adversarial-weight",
            "1.0",
            "--adversarial-schedule",
            "linear_ramp",
            "--ramp-up-fraction",
            "0.4",
        ),
    ),
    ModelConfig("csp_lda", "CSP-LDA anchor", (), include_in_mlp=False),
)

ATTACK_CONFIGS = (
    AttackConfig(
        tier="A",
        attack_type="mlp",
        score_type="posterior_probabilities",
        label="mlp_posterior",
        purpose="Primary learned-attack PhysioNet stability check.",
    ),
    AttackConfig(
        tier="B",
        attack_type="threshold",
        score_type="max_probability",
        label="threshold_max_probability",
        purpose="Label-free threshold robustness check after Tier A.",
    ),
    AttackConfig(
        tier="C",
        attack_type="threshold",
        score_type="label_known_log_probability",
        label="threshold_label_known",
        purpose="Label-known threshold robustness check if A/B remain ambiguous.",
    ),
)


def _subject_csv(subjects: list[int]) -> str:
    return ",".join(str(subject) for subject in subjects)


def _sample_subsets(
    *,
    subject_pool: tuple[int, ...],
    subset_size: int,
    subset_count: int,
    sampling_seed: int,
) -> list[list[int]]:
    if subset_size > len(subject_pool):
        raise ValueError("subset_size cannot exceed the available subject pool")

    rng = random.Random(sampling_seed)
    subsets: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    max_attempts = subset_count * 100
    attempts = 0

    while len(subsets) < subset_count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("failed to sample unique subject subsets")
        subset = tuple(sorted(rng.sample(subject_pool, subset_size)))
        if subset in seen:
            continue
        seen.add(subset)
        subsets.append(list(subset))

    return subsets


def _command_for(
    *,
    model_config: ModelConfig,
    attack_config: AttackConfig,
    subset_id: str,
    subjects: list[int],
    seeds: str,
    task_epochs: int,
    task_batch_size: int,
    mne_data_dir: str,
) -> tuple[str, str]:
    output_dir = (
        Path("outputs")
        / "sweeps"
        / "physionet_random_subset_validation_v1"
        / f"tier_{attack_config.tier.lower()}"
        / subset_id
        / model_config.model
        / attack_config.label
    )
    args = [
        "PYTHONPATH=src",
        ".venv/bin/python",
        "-m",
        "eeg_privacy_benchmark.cli",
        "run-membership-inference-sweep",
        "--dataset",
        DATASET,
        "--model",
        model_config.model,
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
        attack_config.attack_type,
        "--score-type",
        attack_config.score_type,
        "--output-dir",
        str(output_dir),
        *model_config.extra_args,
    ]
    return str(output_dir), shlex.join(args)


def _build_rows(args: argparse.Namespace) -> list[CommandRow]:
    subsets = _sample_subsets(
        subject_pool=DEFAULT_SUBJECT_POOL,
        subset_size=args.subset_size,
        subset_count=args.subset_count,
        sampling_seed=args.sampling_seed,
    )
    rows: list[CommandRow] = []

    for attack_config in ATTACK_CONFIGS:
        for subset_index, subjects in enumerate(subsets, start=1):
            subset_id = f"subset_{subset_index:02d}_n{args.subset_size}_seed{args.sampling_seed}"
            for model_config in MODEL_CONFIGS:
                if attack_config.tier == "A" and not model_config.include_in_mlp:
                    continue
                if attack_config.tier in {"B", "C"} and not model_config.include_in_threshold:
                    continue
                output_dir, command = _command_for(
                    model_config=model_config,
                    attack_config=attack_config,
                    subset_id=subset_id,
                    subjects=subjects,
                    seeds=args.seeds,
                    task_epochs=args.task_epochs,
                    task_batch_size=args.task_batch_size,
                    mne_data_dir=args.mne_data_dir,
                )
                rows.append(
                    CommandRow(
                        tier=attack_config.tier,
                        subset_id=subset_id,
                        subset_size=args.subset_size,
                        subjects=_subject_csv(subjects),
                        model=model_config.model,
                        attack_type=attack_config.attack_type,
                        score_type=attack_config.score_type,
                        output_dir=output_dir,
                        command=command,
                        purpose=attack_config.purpose,
                    )
                )

    return rows


def _write_csv(rows: list[CommandRow]) -> Path:
    out_path = REPORT_ROOT / "physionet_random_subset_validation_plan_v1.csv"
    fieldnames = list(asdict(rows[0]).keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return out_path


def _write_shell(rows: list[CommandRow]) -> Path:
    out_path = PLAN_ROOT / "physionet_random_subset_validation_commands_v1.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by scripts/generate_physionet_random_subset_validation_plan.py.",
        "# Run Tier A first. Only continue to Tier B/C after reading the Tier A deltas.",
        "",
    ]
    current_tier = ""
    for row in rows:
        if row.tier != current_tier:
            current_tier = row.tier
            lines.extend(["", f"# Tier {row.tier}: {row.purpose}"])
        lines.extend(
            [
                f"# {row.subset_id} | {row.model} | {row.attack_type}::{row.score_type}",
                row.command,
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    out_path.chmod(0o755)
    return out_path


def _tier_counts(rows: list[CommandRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.tier] = counts.get(row.tier, 0) + 1
    return counts


def _write_markdown(rows: list[CommandRow], csv_path: Path, shell_path: Path) -> Path:
    out_path = REPORT_ROOT / "physionet_random_subset_validation_plan_v1.md"
    tier_counts = _tier_counts(rows)
    subsets = sorted({row.subset_id: row.subjects for row in rows}.items())

    lines = [
        "# PhysioNet Random Subject-Subset Validation Plan",
        "",
        "This is an operational P1 validation plan for PhysioNet. It does not report new model results; it freezes the random subject subsets and exact membership-inference sweep commands needed to test whether the current PhysioNet defense rankings survive non-sequential subject sampling.",
        "",
        "Companion files:",
        f"- [{csv_path.relative_to(ROOT)}]({csv_path.relative_to(ROOT)})",
        f"- [{shell_path.relative_to(ROOT)}]({shell_path.relative_to(ROOT)})",
        "",
        "## Execution Order",
        "",
        f"- Tier A commands: `{tier_counts.get('A', 0)}` four-seed sweeps. Run these first; they test the key MLP-posterior ambiguity.",
        f"- Tier B commands: `{tier_counts.get('B', 0)}` four-seed sweeps. Run only after Tier A, to check label-free threshold stability.",
        f"- Tier C commands: `{tier_counts.get('C', 0)}` four-seed sweeps. Run only if label-known threshold evidence is needed.",
        f"- Total generated commands: `{len(rows)}`.",
        "",
        "## Frozen Random Subsets",
        "",
        "| Subset | Subjects |",
        "| --- | --- |",
    ]
    for subset_id, subjects in subsets:
        lines.append(f"| `{subset_id}` | `{subjects}` |")

    lines.extend(
        [
            "",
            "## Models",
            "",
        ]
    )
    for config in MODEL_CONFIGS:
        mlp_note = "included in Tier A" if config.include_in_mlp else "threshold anchor only"
        lines.append(f"- `{config.model}`: {config.label}; {mlp_note}.")

    lines.extend(
        [
            "",
            "## Stop Criteria",
            "",
            "- For the MLP-posterior comparison, treat PhysioNet mixup as not supported if mean delta AUC versus plain EEGNet is worse than +0.005 and privacy-improving subsets stay below 75%.",
            "- Call PhysioNet mixup unresolved or a practical tie only if mean delta AUC remains within +/-0.005, or if directional privacy improvement is too inconsistent to promote.",
            "- Promote a PhysioNet learned-attack defense only if the AUC edge is at least 0.01 and the same model wins at least 75% of random subsets without severe task-utility collapse.",
            "- If winners rotate across subsets or attack tiers, keep PhysioNet unresolved and report the protocol sensitivity explicitly.",
            "- Do not broaden the hyperparameter grid until this fixed-subset validation read is complete.",
            "",
            "## Command Preview",
            "",
            "The full shell script contains every command. The first Tier A command is:",
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
    parser = argparse.ArgumentParser(
        description="Generate a deterministic PhysioNet random-subset validation command plan."
    )
    parser.add_argument("--subset-size", type=int, default=32)
    parser.add_argument("--subset-count", type=int, default=5)
    parser.add_argument("--sampling-seed", type=int, default=20260703)
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
    print(f"Generated {len(rows)} commands")


if __name__ == "__main__":
    main()
