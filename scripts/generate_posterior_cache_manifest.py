from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_privacy_benchmark.privacy import load_posterior_feature_cache


REPORT_ROOT = ROOT / "outputs" / "reports"
CACHE_ROOT = ROOT / "outputs" / "cache"

CSV_OUT = REPORT_ROOT / "posterior_cache_manifest_v1.csv"
JSON_OUT = REPORT_ROOT / "posterior_cache_manifest_v1.json"
MD_OUT = REPORT_ROOT / "posterior_cache_manifest_v1.md"


@dataclass(frozen=True)
class CacheManifestRow:
    cache_path: str
    size_bytes: int
    sha256: str
    task_model: str
    dataset_key: str
    protocol: str
    seed: int
    subjects: str
    task_balanced_accuracy: float | None
    task_macro_f1: float | None
    probability_rows: int
    class_count: int
    member_trials: int
    nonmember_trials: int
    adversarial_weight: float | None
    adversarial_schedule: str | None
    ramp_up_fraction: float | None
    label_smoothing: float | None
    bottleneck_dim: int | None
    feature_noise_std: float | None
    mixup_alpha: float | None
    confidence_penalty_beta: float | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _markdown_path(path: Path) -> str:
    try:
        return str(ROOT / path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _cache_paths(cache_root: Path = CACHE_ROOT) -> list[Path]:
    if not cache_root.exists():
        return []
    return sorted(cache_root.rglob("*.npz"))


def _row(path: Path) -> CacheManifestRow:
    cache = load_posterior_feature_cache(path)
    probability_rows = int(cache.class_probabilities.shape[0])
    class_count = int(cache.class_probabilities.shape[1])
    return CacheManifestRow(
        cache_path=_display_path(path),
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        task_model=cache.task_model,
        dataset_key=cache.dataset_key,
        protocol=cache.protocol,
        seed=cache.seed,
        subjects=json.dumps(cache.subjects, sort_keys=True),
        task_balanced_accuracy=cache.task_balanced_accuracy,
        task_macro_f1=cache.task_macro_f1,
        probability_rows=probability_rows,
        class_count=class_count,
        member_trials=len(cache.member_indices),
        nonmember_trials=len(cache.nonmember_indices),
        adversarial_weight=cache.adversarial_weight,
        adversarial_schedule=cache.adversarial_schedule,
        ramp_up_fraction=cache.ramp_up_fraction,
        label_smoothing=cache.label_smoothing,
        bottleneck_dim=cache.bottleneck_dim,
        feature_noise_std=cache.feature_noise_std,
        mixup_alpha=cache.mixup_alpha,
        confidence_penalty_beta=cache.confidence_penalty_beta,
    )


def _rows(cache_root: Path = CACHE_ROOT) -> list[CacheManifestRow]:
    return [_row(path) for path in _cache_paths(cache_root)]


def _write_csv(rows: list[CacheManifestRow], out_path: Path = CSV_OUT) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(CacheManifestRow)]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return out_path


def _write_json(
    rows: list[CacheManifestRow],
    out_path: Path = JSON_OUT,
    cache_root: Path = CACHE_ROOT,
) -> Path:
    payload = {
        "cache_root": _display_path(cache_root),
        "cache_count": len(rows),
        "caches": [asdict(row) for row in rows],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _write_markdown(
    rows: list[CacheManifestRow],
    out_path: Path = MD_OUT,
    csv_path: Path = CSV_OUT,
    json_path: Path = JSON_OUT,
    cache_root: Path = CACHE_ROOT,
) -> Path:
    cache_root_display = _display_path(cache_root)
    lines = [
        "# Posterior Cache Manifest",
        "",
        f"This manifest inventories archived posterior-feature `.npz` caches under `{cache_root_display}/`. It validates each cache with `load_posterior_feature_cache` before recording metadata, so malformed caches fail manifest generation.",
        "",
        f"Companion CSV: [posterior_cache_manifest_v1.csv]({_markdown_path(csv_path)})",
        f"Companion JSON: [posterior_cache_manifest_v1.json]({_markdown_path(json_path)})",
        "",
        "## Summary",
        "",
        f"- Cache root: `{cache_root_display}`",
        f"- Archived posterior caches: `{len(rows)}`",
        "",
    ]
    if not rows:
        lines.extend(
            [
                "No archived posterior-feature caches are present. The cache workflow is implemented and tested synthetically, but current package claims do not rely on archived `.npz` caches.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Caches",
                "",
                "| Cache | Model | Dataset | Protocol | Seed | Rows | Classes | Members | Nonmembers | SHA256 prefix |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in rows:
            lines.append(
                f"| `{row.cache_path}` | `{row.task_model}` | `{row.dataset_key}` | `{row.protocol}` | "
                f"`{row.seed}` | `{row.probability_rows}` | `{row.class_count}` | "
                f"`{row.member_trials}` | `{row.nonmember_trials}` | `{row.sha256[:16]}` |"
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_manifest(
    *,
    cache_root: Path = CACHE_ROOT,
    csv_out: Path = CSV_OUT,
    json_out: Path = JSON_OUT,
    md_out: Path = MD_OUT,
) -> tuple[Path, Path, Path]:
    rows = _rows(cache_root)
    csv_path = _write_csv(rows, csv_out)
    json_path = _write_json(rows, json_out, cache_root)
    md_path = _write_markdown(rows, md_out, csv_path, json_path, cache_root)
    return csv_path, json_path, md_path


def main() -> None:
    csv_path, json_path, md_path = write_manifest()
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {json_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
