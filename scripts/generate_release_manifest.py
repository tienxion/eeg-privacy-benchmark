from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "release"
MANIFEST = RELEASE_DIR / "v1.0.0-manifest.json"
CHECKSUMS = RELEASE_DIR / "SHA256SUMS"
SOURCE_CHECKPOINT = "5dee43de85a2b4011fc97581efcf65f35fa5a4aa"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _license_for(relative: Path) -> str:
    if relative.parts[0] in {"paper", "results"}:
        return "CC-BY-4.0"
    return "Apache-2.0"


def evidence_files() -> list[Path]:
    files: list[Path] = []
    for directory in (ROOT / "paper", ROOT / "results" / "v1"):
        files.extend(path for path in directory.rglob("*") if path.is_file())
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> None:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in evidence_files():
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
        "release": "1.0.0",
        "source_checkpoint": SOURCE_CHECKPOINT,
        "evidence_file_count": len(entries),
        "files": entries,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    checksum_lines = [f"{entry['sha256']}  {entry['path']}" for entry in entries]
    checksum_lines.append(f"{_sha256(MANIFEST)}  {MANIFEST.relative_to(ROOT).as_posix()}")
    CHECKSUMS.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"wrote={MANIFEST.relative_to(ROOT)} evidence_files={len(entries)}")
    print(f"wrote={CHECKSUMS.relative_to(ROOT)} checksums={len(checksum_lines)}")


if __name__ == "__main__":
    main()
