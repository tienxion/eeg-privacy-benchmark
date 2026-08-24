from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 1024 * 1024
APPROVED_NAME = "Arnav Adepu"
APPROVED_EMAIL = "169912162+tienxion@users.noreply.github.com"

REQUIRED_FILES = {
    ".github/workflows/ci.yml",
    ".gitignore",
    ".python-version",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "DATASETS.md",
    "LICENSE",
    "LICENSES/Apache-2.0.txt",
    "LICENSES/CC-BY-4.0.txt",
    "LICENSE_SCOPE.md",
    "README.md",
    "REPRODUCING.md",
    "RESULTS.md",
    "SECURITY.md",
    "paper/manuscript.md",
    "paper/references.bib",
    "pyproject.toml",
    "release/SHA256SUMS",
    "release/v1.0.0-rc1-manifest.json",
    "requirements-lock.txt",
    "results/v1/README.md",
    "results/v1/PROVENANCE.md",
}

ALLOWED_TOP_LEVEL = {
    ".github",
    ".gitignore",
    ".python-version",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "DATASETS.md",
    "LICENSE",
    "LICENSES",
    "LICENSE_SCOPE.md",
    "README.md",
    "REPRODUCING.md",
    "RESULTS.md",
    "SECURITY.md",
    "configs",
    "data",
    "notebooks",
    "outputs",
    "paper",
    "pyproject.toml",
    "raw_data",
    "release",
    "requirements-lock.txt",
    "results",
    "scripts",
    "src",
}

PRIVATE_FILENAME_MARKERS = {
    "agent_handoff",
    "advisor",
    "human_decisions",
    "lab_email",
    "lab_share",
    "public_release_plan",
    "repo_checkpoint",
}

FORBIDDEN_EXTENSIONS = {
    ".bdf",
    ".ckpt",
    ".edf",
    ".eeg",
    ".fif",
    ".gdf",
    ".joblib",
    ".log",
    ".mat",
    ".npz",
    ".pkl",
    ".pt",
    ".pth",
    ".set",
    ".tar",
    ".vhdr",
    ".zip",
}

TEXT_EXTENSIONS = {
    "",
    ".bib",
    ".cff",
    ".csv",
    ".gitignore",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _files() -> list[Path]:
    return sorted(
        (
            path
            for path in ROOT.rglob("*")
            if path.is_file() and ".git" not in path.relative_to(ROOT).parts
        ),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_required(errors: list[str]) -> None:
    present = {path.relative_to(ROOT).as_posix() for path in _files()}
    for relative in sorted(REQUIRED_FILES - present):
        errors.append(f"missing required file: {relative}")
    unexpected = sorted({path.parts[0] for path in map(lambda p: p.relative_to(ROOT), _files())} - ALLOWED_TOP_LEVEL)
    for name in unexpected:
        errors.append(f"unexpected top-level path: {name}")


def _check_paths_and_contents(errors: list[str]) -> None:
    host_markers = (
        "/" + "Users/",
        "/" + "home/",
        "C:" + "\\Users\\",
        "/private/" + "var/folders/",
        "/var/" + "folders/",
    )
    local_marker = "_local" + "cache"
    secret_patterns = (
        re.compile("-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile("gh" + r"[pousr]_[A-Za-z0-9_]{20,}"),
        re.compile("github_" + r"pat_[A-Za-z0-9_]{20,}"),
        re.compile("AK" + r"IA[0-9A-Z]{16}"),
        re.compile("AI" + r"za[0-9A-Za-z_-]{30,}"),
        re.compile("sk" + r"-[A-Za-z0-9]{20,}"),
    )

    for path in _files():
        relative = path.relative_to(ROOT)
        relative_text = relative.as_posix()
        lower = relative_text.lower()
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds {MAX_FILE_BYTES} bytes: {relative_text}")
        if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
            errors.append(f"prohibited artifact extension: {relative_text}")
        if any(marker in lower for marker in PRIVATE_FILENAME_MARKERS):
            errors.append(f"private/internal filename: {relative_text}")
        if relative.parts[:1] == ("raw_data",) and relative_text != "raw_data/README.md":
            errors.append(f"raw_data contains non-placeholder file: {relative_text}")
        if relative.parts[:1] == ("outputs",) and relative_text != "outputs/README.md":
            errors.append(f"outputs contains generated file: {relative_text}")
        if relative.parts[:2] == ("data", "manifests") and relative.name != ".gitkeep":
            errors.append(f"data manifest is not public metadata: {relative_text}")

        data = path.read_bytes()
        if path.suffix.lower() not in {".png"} and b"\0" in data:
            errors.append(f"unexpected binary file: {relative_text}")
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF-8 public text file: {relative_text}")
            continue
        if local_marker in text:
            errors.append(f"unresolved private artifact marker: {relative_text}")
        for marker in host_markers:
            if marker in text:
                errors.append(f"host-specific path marker in {relative_text}: {marker}")
        for pattern in secret_patterns:
            if pattern.search(text):
                errors.append(f"credential-like signature in {relative_text}")


def _check_markdown_links(errors: list[str]) -> None:
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for path in _files():
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for target in link_pattern.findall(text):
            clean = target.split("#", 1)[0]
            if not clean or clean.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (path.parent / clean).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                errors.append(f"link escapes repository: {path.relative_to(ROOT)} -> {target}")
                continue
            if not resolved.exists():
                errors.append(f"broken link: {path.relative_to(ROOT)} -> {target}")


def _check_manifest(errors: list[str]) -> None:
    manifest_path = ROOT / "release" / "v1.0.0-rc1-manifest.json"
    checksums_path = ROOT / "release" / "SHA256SUMS"
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("release") != "1.0.0-rc1":
        errors.append("release manifest version is not 1.0.0-rc1")
    if payload.get("source_checkpoint") != "5dee43de85a2b4011fc97581efcf65f35fa5a4aa":
        errors.append("release manifest source checkpoint mismatch")
    entries = payload.get("files", [])
    if payload.get("evidence_file_count") != len(entries):
        errors.append("release manifest evidence count mismatch")
    for entry in entries:
        path = ROOT / entry["path"]
        if not path.exists():
            errors.append(f"manifest file missing: {entry['path']}")
        elif _sha256(path) != entry["sha256"]:
            errors.append(f"manifest checksum mismatch: {entry['path']}")
        elif path.stat().st_size != entry["bytes"]:
            errors.append(f"manifest byte-size mismatch: {entry['path']}")
    if checksums_path.exists():
        expected = [f"{entry['sha256']}  {entry['path']}" for entry in entries]
        expected.append(
            f"{_sha256(manifest_path)}  {manifest_path.relative_to(ROOT).as_posix()}"
        )
        actual = checksums_path.read_text(encoding="utf-8").splitlines()
        if actual != expected:
            errors.append("SHA256SUMS does not exactly match the release manifest")


def _check_git(errors: list[str], allowed_remotes: set[str]) -> None:
    if not (ROOT / ".git").exists():
        errors.append("candidate is not a Git repository")
        return
    remote_result = _run_git("remote")
    remotes = {line.strip() for line in remote_result.stdout.splitlines() if line.strip()}
    unexpected = remotes - allowed_remotes
    if unexpected:
        errors.append(f"unexpected Git remotes: {', '.join(sorted(unexpected))}")
    commits = _run_git("rev-list", "--all")
    if commits.returncode != 0 or not commits.stdout.strip():
        return
    identities = _run_git("log", "--all", "--format=%an%x09%ae")
    for line in identities.stdout.splitlines():
        name, email = line.split("\t", 1)
        if name != APPROVED_NAME or email != APPROVED_EMAIL:
            errors.append(f"unapproved commit identity: {name} <{email}>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the public-release boundary.")
    parser.add_argument(
        "--allow-remote",
        action="append",
        default=[],
        help="Allow a named remote, intended only for CI clones.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    _check_required(errors)
    _check_paths_and_contents(errors)
    _check_markdown_links(errors)
    _check_manifest(errors)
    _check_git(errors, set(args.allow_remote))
    if errors:
        for error in errors:
            print(f"FAIL {error}")
        raise SystemExit(f"Public release check failed: errors={len(errors)}")
    print(f"PASS public release files={len(_files())} max_bytes={MAX_FILE_BYTES}")


if __name__ == "__main__":
    main()
