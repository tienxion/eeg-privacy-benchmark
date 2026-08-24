from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "outputs" / "reports"
SWEEP_ROOT = ROOT / "outputs" / "sweeps"

EXISTING_ROOTS = {
    "eegnet": SWEEP_ROOT
    / "membership_eegnet_physionet_motor_imagery_subject12345678910111213141516171819202122232425262728293031323334353637383940414243444546_mlp_posterior_4seed_v1",
    "mixup_eegnet": SWEEP_ROOT
    / "membership_mixup_eegnet_physionet_motor_imagery_subject12345678910111213141516171819202122232425262728293031323334353637383940414243444546_alpha02_mlp_posterior_4seed_v1",
    "confidence_penalty_eegnet": SWEEP_ROOT
    / "membership_confidence_penalty_eegnet_physionet_motor_imagery_subject12345678910111213141516171819202122232425262728293031323334353637383940414243444546_beta0025_mlp_posterior_4seed_v1",
    "adversarial_eegnet": SWEEP_ROOT
    / "membership_adversarial_eegnet_physionet_motor_imagery_subject12345678910111213141516171819202122232425262728293031323334353637383940414243444546_linear_ramp_w1p0_mlp_posterior_4seed_v1",
    "csp_lda": SWEEP_ROOT
    / "membership_csp_lda_physionet_motor_imagery_subject12345678910111213141516171819202122232425262728293031323334353637383940414243444546_mlp_posterior_4seed_v1",
}
EXTENSION_ROOT = SWEEP_ROOT / "physionet_mlp_seed_extension_v1"
MODEL_ORDER = (
    "eegnet",
    "mixup_eegnet",
    "confidence_penalty_eegnet",
    "adversarial_eegnet",
    "csp_lda",
)
BASELINE_MODEL = "eegnet"


@dataclass(frozen=True)
class SeedRow:
    model: str
    seed: int
    task_balanced_accuracy: float
    attack_auc: float
    source: str
    path: str


@dataclass(frozen=True)
class AggregateRow:
    model: str
    seeds: str
    seed_count: int
    mean_task: float
    mean_auc: float
    mean_delta_task_vs_eegnet: float | None
    mean_delta_auc_vs_eegnet: float | None
    std_delta_auc_vs_eegnet: float | None
    privacy_improving_seeds: int | None
    decision: str


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _seed_from_path(path: Path) -> int | None:
    stem = path.stem
    marker = "_seed"
    if marker not in stem:
        return None
    try:
        return int(stem.rsplit(marker, 1)[1])
    except ValueError:
        return None


def _load_seed_rows_for_model(model: str) -> dict[int, SeedRow]:
    rows: dict[int, SeedRow] = {}
    roots = [EXISTING_ROOTS[model], EXTENSION_ROOT / model / "mlp_posterior_seed_extension"]
    sources = ["existing_4seed", "extension"]
    for root, source in zip(roots, sources):
        if not root.exists():
            continue
        for path in sorted(root.glob("*_seed*.json")):
            seed = _seed_from_path(path)
            if seed is None:
                continue
            payload = _load_json(path)
            if not payload:
                continue
            if payload.get("attack_type") != "mlp" or payload.get("score_type") != "posterior_probabilities":
                continue
            rows[seed] = SeedRow(
                model=model,
                seed=seed,
                task_balanced_accuracy=float(payload["task_balanced_accuracy"]),
                attack_auc=float(payload["attack_auc"]),
                source=source,
                path=str(path.relative_to(ROOT)),
            )
    return rows


def _load_seed_rows() -> dict[str, dict[int, SeedRow]]:
    return {model: _load_seed_rows_for_model(model) for model in MODEL_ORDER}


def _decision(
    *,
    model: str,
    mean_delta_auc: float | None,
    mean_delta_task: float | None,
    improving: int | None,
    total: int,
) -> str:
    if model == BASELINE_MODEL:
        return "baseline"
    if mean_delta_auc is None or improving is None:
        return "missing baseline overlap"
    win_rate = improving / total if total else 0.0
    if model == "mixup_eegnet":
        if mean_delta_auc > 0.005 and win_rate < 0.75:
            return "not supported"
        if abs(mean_delta_auc) <= 0.005 or win_rate < 0.75:
            return "practical tie or unresolved"
        if mean_delta_auc <= -0.010 and win_rate >= 0.75 and (
            mean_delta_task is None or mean_delta_task >= -0.030
        ):
            return "candidate learned-attack defense"
        return "directional but not promotable"
    if mean_delta_auc <= -0.010 and win_rate >= 0.75 and (
        mean_delta_task is None or mean_delta_task >= -0.030
    ):
        return "candidate learned-attack defense"
    if mean_delta_auc < 0.0:
        return "privacy-favorable but not promotable"
    return "not privacy-improving"


def _aggregate_rows(seed_rows: dict[str, dict[int, SeedRow]]) -> list[AggregateRow]:
    baseline = seed_rows[BASELINE_MODEL]
    rows: list[AggregateRow] = []
    for model in MODEL_ORDER:
        model_rows = seed_rows[model]
        seeds = sorted(model_rows)
        task_values = [model_rows[seed].task_balanced_accuracy for seed in seeds]
        auc_values = [model_rows[seed].attack_auc for seed in seeds]
        overlap = [seed for seed in seeds if seed in baseline]
        delta_auc_values = [
            model_rows[seed].attack_auc - baseline[seed].attack_auc
            for seed in overlap
        ]
        delta_task_values = [
            model_rows[seed].task_balanced_accuracy - baseline[seed].task_balanced_accuracy
            for seed in overlap
        ]
        mean_delta_auc = mean(delta_auc_values) if delta_auc_values else None
        mean_delta_task = mean(delta_task_values) if delta_task_values else None
        std_delta_auc = stdev(delta_auc_values) if len(delta_auc_values) > 1 else None
        improving = sum(delta < 0.0 for delta in delta_auc_values) if delta_auc_values else None
        rows.append(
            AggregateRow(
                model=model,
                seeds=",".join(str(seed) for seed in seeds),
                seed_count=len(seeds),
                mean_task=mean(task_values) if task_values else float("nan"),
                mean_auc=mean(auc_values) if auc_values else float("nan"),
                mean_delta_task_vs_eegnet=mean_delta_task,
                mean_delta_auc_vs_eegnet=mean_delta_auc,
                std_delta_auc_vs_eegnet=std_delta_auc,
                privacy_improving_seeds=improving,
                decision=_decision(
                    model=model,
                    mean_delta_auc=mean_delta_auc,
                    mean_delta_task=mean_delta_task,
                    improving=improving,
                    total=len(overlap),
                ),
            )
        )
    return rows


def _write_seed_csv(seed_rows: dict[str, dict[int, SeedRow]]) -> Path:
    out_path = REPORT_ROOT / "physionet_mlp_seed_extension_seed_rows_v1.csv"
    rows = [row for model in MODEL_ORDER for row in seed_rows[model].values()]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.model, item.seed)):
            writer.writerow(asdict(row))
    return out_path


def _write_aggregate_csv(rows: list[AggregateRow]) -> Path:
    out_path = REPORT_ROOT / "physionet_mlp_seed_extension_results_v1.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return out_path


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _signed(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.4f}"


def _write_markdown(
    rows: list[AggregateRow],
    seed_rows: dict[str, dict[int, SeedRow]],
    aggregate_csv: Path,
    seed_csv: Path,
) -> Path:
    out_path = REPORT_ROOT / "physionet_mlp_seed_extension_results_v1.md"
    extension_counts = {
        model: sum(1 for row in seed_rows[model].values() if row.source == "extension")
        for model in MODEL_ORDER
    }
    mixup = next(row for row in rows if row.model == "mixup_eegnet")
    lines = [
        "# PhysioNet MLP Seed-Extension Results",
        "",
        "This report combines the existing PhysioNet subjects `1-46` MLP-posterior seed rows with any completed seed-extension rows. Lower membership-inference AUC is better for privacy; deltas are paired against same-seed plain EEGNet.",
        "",
        "Companion files:",
        f"- [{aggregate_csv.relative_to(ROOT)}]({aggregate_csv.relative_to(ROOT)})",
        f"- [{seed_csv.relative_to(ROOT)}]({seed_csv.relative_to(ROOT)})",
        "",
        "## Completion",
        "",
    ]
    for model in MODEL_ORDER:
        lines.append(f"- `{model}` extension seeds complete: `{extension_counts[model]}`.")
    lines.extend(
        [
            "",
            "## Current Decision Read",
            "",
            f"- Mixup seeds evaluated: `{mixup.seed_count}` (`{mixup.seeds}`).",
            f"- Mixup mean delta AUC versus EEGNet: `{_signed(mixup.mean_delta_auc_vs_eegnet)}`.",
            f"- Mixup privacy-improving same-seed comparisons: `{mixup.privacy_improving_seeds}/{mixup.seed_count}`.",
            f"- Mixup mean task balanced-accuracy delta: `{_signed(mixup.mean_delta_task_vs_eegnet)}`.",
            f"- Current decision: `{mixup.decision}`.",
            "",
            "## Aggregate Model Read",
            "",
            "| Model | Seeds | Mean task BA | Mean AUC | Mean delta AUC | Privacy-improving seeds | Mean delta task BA | Decision |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        improving = (
            ""
            if row.privacy_improving_seeds is None
            else f"{row.privacy_improving_seeds}/{row.seed_count}"
        )
        lines.append(
            f"| `{row.model}` | `{row.seeds}` | {_fmt(row.mean_task)} | {_fmt(row.mean_auc)} | "
            f"{_signed(row.mean_delta_auc_vs_eegnet)} | {improving} | "
            f"{_signed(row.mean_delta_task_vs_eegnet)} | `{row.decision}` |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    seed_rows = _load_seed_rows()
    aggregate_rows = _aggregate_rows(seed_rows)
    seed_csv = _write_seed_csv(seed_rows)
    aggregate_csv = _write_aggregate_csv(aggregate_rows)
    markdown_path = _write_markdown(aggregate_rows, seed_rows, aggregate_csv, seed_csv)
    print(f"Wrote {aggregate_csv.relative_to(ROOT)}")
    print(f"Wrote {seed_csv.relative_to(ROOT)}")
    print(f"Wrote {markdown_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
