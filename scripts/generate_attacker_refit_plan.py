from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "outputs" / "reports"
PLAN_ROOT = ROOT / "outputs" / "plans"

THREAD_LIMITS = (
    "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
    "VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
)
CLI = "PYTHONPATH=src .venv/bin/python -m eeg_privacy_benchmark.cli"
TASK_SEEDS = "13,17,19,23"
ATTACK_SEEDS = [101, 103, 107]
BNCI_SUBJECTS = "1,2,3,4,5,6,7,8,9"
LEE_SUBJECTS = "1,2,3,4,5,6,7,8,9,10,11,12"
PHYSIONET_SUBJECTS = ",".join(str(index) for index in range(1, 47))


@dataclass(frozen=True)
class AttackerRefitJob:
    priority: str
    dataset: str
    protocol: str
    subjects: str
    model: str
    task_seeds: str
    attack_seed: int
    attack_type: str
    score_type: str
    output_dir: str
    command: str
    purpose: str
    stop_rule: str


def _build_command(
    *,
    dataset: str,
    model: str,
    protocol: str,
    subjects: str,
    attack_seed: int,
    output_dir: str,
    extra_args: str = "",
) -> str:
    command = (
        f"{THREAD_LIMITS} {CLI} run-membership-inference-sweep "
        f"--dataset {dataset} --model {model} --protocol {protocol} "
        f"--subjects {subjects} --seeds {TASK_SEEDS} --task-epochs 20 "
        f"--task-batch-size 32 --mne-data-dir raw_data/mne_data "
        f"--attack-type mlp --score-type posterior_probabilities "
        f"--attack-seed {attack_seed} --output-dir {output_dir}"
    )
    if extra_args:
        command = f"{command} {extra_args}"
    return command


def _job(
    *,
    priority: str,
    dataset: str,
    protocol: str,
    subjects: str,
    model: str,
    slug: str,
    attack_seed: int,
    purpose: str,
    stop_rule: str,
    extra_args: str = "",
) -> AttackerRefitJob:
    output_dir = (
        f"outputs/sweeps/attacker_refit_v1/{dataset}/{slug}/"
        f"attackseed{attack_seed}"
    )
    return AttackerRefitJob(
        priority=priority,
        dataset=dataset,
        protocol=protocol,
        subjects=subjects,
        model=model,
        task_seeds=TASK_SEEDS,
        attack_seed=attack_seed,
        attack_type="mlp",
        score_type="posterior_probabilities",
        output_dir=output_dir,
        command=_build_command(
            dataset=dataset,
            model=model,
            protocol=protocol,
            subjects=subjects,
            attack_seed=attack_seed,
            output_dir=output_dir,
            extra_args=extra_args,
        ),
        purpose=purpose,
        stop_rule=stop_rule,
    )


def _jobs() -> list[AttackerRefitJob]:
    jobs: list[AttackerRefitJob] = []
    for attack_seed in ATTACK_SEEDS:
        jobs.extend(
            [
                _job(
                    priority="P1",
                    dataset="bnci2014_001",
                    protocol="cross_session",
                    subjects=BNCI_SUBJECTS,
                    model="eegnet",
                    slug="subject123456789/eegnet/mlp_posterior",
                    attack_seed=attack_seed,
                    purpose="Quantify attacker-refit variance for the BNCI plain EEGNet learned-attack baseline.",
                    stop_rule="If rank ordering changes across attack seeds, report learned-attack winners as unstable.",
                ),
                _job(
                    priority="P1",
                    dataset="bnci2014_001",
                    protocol="cross_session",
                    subjects=BNCI_SUBJECTS,
                    model="bottleneck_eegnet",
                    slug="subject123456789/bottleneck_dim6/mlp_posterior",
                    attack_seed=attack_seed,
                    purpose="Check whether the BNCI bottleneck learned-attack read survives attacker refits.",
                    stop_rule="Keep the bottleneck privacy claim only if the direction is stable across refits.",
                    extra_args="--bottleneck-dim 6",
                ),
                _job(
                    priority="P1",
                    dataset="lee2019_mi",
                    protocol="cross_session",
                    subjects=LEE_SUBJECTS,
                    model="eegnet",
                    slug="subject123456789101112/eegnet/mlp_posterior",
                    attack_seed=attack_seed,
                    purpose="Quantify attacker-refit variance for the Lee plain EEGNet learned-attack baseline.",
                    stop_rule="If Lee refits swing around the current mean, downgrade dataset-specific learned-attack claims.",
                ),
                _job(
                    priority="P1",
                    dataset="lee2019_mi",
                    protocol="cross_session",
                    subjects=LEE_SUBJECTS,
                    model="bottleneck_eegnet",
                    slug="subject123456789101112/bottleneck_dim6/mlp_posterior",
                    attack_seed=attack_seed,
                    purpose="Test whether the Lee bottleneck learned-attack result is attacker-fit dependent.",
                    stop_rule="Treat Lee bottleneck membership results as exploratory unless the sign is refit-stable.",
                    extra_args="--bottleneck-dim 6",
                ),
                _job(
                    priority="P2",
                    dataset="physionet_motor_imagery",
                    protocol="cross_subject",
                    subjects=PHYSIONET_SUBJECTS,
                    model="eegnet",
                    slug="subject1to46/eegnet/mlp_posterior",
                    attack_seed=attack_seed,
                    purpose="Check if the PhysioNet learned-attack baseline remains weak under attacker refits.",
                    stop_rule="Do not promote a PhysioNet learned-attack default unless refits stay directionally consistent.",
                ),
            ]
        )
    return jobs


def _write_csv(jobs: list[AttackerRefitJob]) -> Path:
    out_path = REPORT_ROOT / "attacker_refit_plan_v1.csv"
    fieldnames = list(asdict(jobs[0]).keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for job in jobs:
            writer.writerow(asdict(job))
    return out_path


def _write_commands(jobs: list[AttackerRefitJob]) -> Path:
    out_path = PLAN_ROOT / "attacker_refit_commands_v1.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by scripts/generate_attacker_refit_plan.py.",
        "# These jobs intentionally vary --attack-seed while holding task seeds fixed.",
        "",
    ]
    for job in jobs:
        lines.extend([f"# {job.priority} {job.dataset} {job.model} attack_seed={job.attack_seed}", job.command, ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")
    out_path.chmod(0o755)
    return out_path


def _write_markdown(jobs: list[AttackerRefitJob]) -> Path:
    out_path = REPORT_ROOT / "attacker_refit_plan_v1.md"
    csv_path = REPORT_ROOT / "attacker_refit_plan_v1.csv"
    command_path = PLAN_ROOT / "attacker_refit_commands_v1.sh"
    lines = [
        "# Attacker Refit Calibration Plan",
        "",
        "This plan is the next P1 hardening step after completing BNCI/Lee validation and paired interval reporting. The code now supports `--attack-seed`, which decouples learned membership-attacker randomization from task-model training seeds.",
        "",
        f"Companion table: [{csv_path.name}]({csv_path})",
        f"Runnable command file: [{command_path.name}]({command_path})",
        "",
        "## Scope",
        "",
        "- The jobs hold task seeds fixed at `13,17,19,23` and rerun learned MLP membership attacks with attacker seeds `101,103,107`.",
        "- Current implementation still retrains the task model for each command; a posterior-feature cache would make this cheaper but is not required for correctness.",
        "- Threshold attacks are not prioritized here because they do not fit a stochastic attacker.",
        "",
        "## Job Queue",
        "",
        "| Priority | Dataset | Model | Attack seed | Purpose |",
        "| --- | --- | --- | --- | --- |",
    ]
    for job in jobs:
        lines.append(
            f"| `{job.priority}` | `{job.dataset}` | `{job.model}` | "
            f"`{job.attack_seed}` | {job.purpose} |"
        )

    lines.extend(["", "## Stop Rules", ""])
    for job in jobs:
        lines.append(
            f"- `{job.dataset}` `{job.model}` attack seed `{job.attack_seed}`: {job.stop_rule}"
        )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    PLAN_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = _jobs()
    csv_path = _write_csv(jobs)
    commands_path = _write_commands(jobs)
    markdown_path = _write_markdown(jobs)
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {commands_path.relative_to(ROOT)}")
    print(f"Wrote {markdown_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
