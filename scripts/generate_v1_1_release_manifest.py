from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "release"
MANIFEST = RELEASE_DIR / "v1.1.0-manifest.json"
CHECKSUMS = RELEASE_DIR / "SHA256SUMS-v1.1.0"
SOURCE_CHECKPOINT = "c737c1a"

EVIDENCE_PATHS = (
    "configs/cho2017_confirmatory_validation_v1.yaml",
    "configs/cho2017_confirmatory_cache_manifest_v1.json",
    "configs/cho2017_confirmatory_baseline_gate_v1.json",
    "configs/cho2017_confirmatory_bottleneck_gate_v1.json",
    "configs/cho2017_confirmatory_privacy_gate_v1.json",
    "docs/V1_1_CHO2017_VALIDATION_CONTRACT.md",
    "docs/V1_1_CHO2017_CACHE_QA.md",
    "docs/V1_1_CHO2017_BASELINE_GATE.md",
    "docs/V1_1_CHO2017_BOTTLENECK_GATE.md",
    "docs/V1_1_CHO2017_PRIVACY_GATE.md",
    "docs/V1_1_CHO2017_CLAIM_AUDIT.md",
    "results/v1.1/README.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _license_for(relative: Path) -> str:
    if relative.parts[0] in {"docs", "results"}:
        return "CC-BY-4.0"
    return "Apache-2.0"


def main() -> None:
    paths = [ROOT / relative for relative in EVIDENCE_PATHS]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing v1.1 evidence files: {missing}")
    entries = []
    for path in paths:
        relative = path.relative_to(ROOT)
        entries.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "media_type": mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
                "license": _license_for(relative),
            }
        )
    payload = {
        "schema_version": 1,
        "release": "1.1.0",
        "source_checkpoint": SOURCE_CHECKPOINT,
        "evidence_file_count": len(entries),
        "files": entries,
    }
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [f"{entry['sha256']}  {entry['path']}" for entry in entries]
    lines.append(f"{_sha256(MANIFEST)}  {MANIFEST.relative_to(ROOT).as_posix()}")
    CHECKSUMS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote={MANIFEST.relative_to(ROOT)} evidence_files={len(entries)}")
    print(f"wrote={CHECKSUMS.relative_to(ROOT)} checksums={len(lines)}")


if __name__ == "__main__":
    main()
