from __future__ import annotations

import os
import py_compile
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
SCRIPTS = ROOT / "scripts"


def _run(script_name: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SOURCE), str(SCRIPTS)])
    subprocess.run(
        [sys.executable, str(SCRIPTS / script_name)],
        cwd=ROOT,
        env=env,
        check=True,
    )


def _compile_python() -> int:
    files = sorted([*SOURCE.rglob("*.py"), *SCRIPTS.glob("*.py")])
    for path in files:
        py_compile.compile(str(path), doraise=True)
    return len(files)


def _parse_metadata() -> int:
    files = sorted((ROOT / "configs").glob("*.yaml")) + [ROOT / "CITATION.cff"]
    for path in files:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise AssertionError(f"Expected mapping in {path}")
    return len(files)


def _check_import_and_cli() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SOURCE)
    command = [sys.executable, "-m", "eeg_privacy_benchmark.cli", "--help"]
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    if result.returncode or "EEG privacy benchmark utilities" not in result.stdout:
        raise AssertionError(result.stderr or result.stdout)
    version = subprocess.run(
        [
            sys.executable,
            "-c",
            "import eeg_privacy_benchmark as p; print(p.__version__)",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if version != "1.1.0":
        raise AssertionError(f"Unexpected package version: {version}")


def _check_manuscript_and_tables() -> None:
    manuscript = (ROOT / "paper" / "manuscript.md").read_text(encoding="utf-8")
    for heading in (
        "## Abstract",
        "## 3. Threat Model and Research Questions",
        "## 5. Results",
        "## 7. Limitations",
        "## 9. Reproducibility",
    ):
        if heading not in manuscript:
            raise AssertionError(f"Missing manuscript heading: {heading}")
    for phrase in (
        "fails the task-utility gate",
        "production cryptography",
        "does not address dropout",
    ):
        if phrase not in manuscript:
            raise AssertionError(f"Missing claim boundary: {phrase}")

    tables = sorted((ROOT / "results" / "v1" / "tables").glob("*.tex"))
    if len(tables) != 6:
        raise AssertionError(f"Expected six LaTeX tables, found {len(tables)}")
    for path in tables:
        text = path.read_text(encoding="utf-8")
        if text.count("\\begin{tabular}") != text.count("\\end{tabular}"):
            raise AssertionError(f"Unbalanced tabular environment: {path}")


def main() -> None:
    python_count = _compile_python()
    metadata_count = _parse_metadata()
    _check_import_and_cli()
    for script in (
        "check_dataset_cache_plumbing.py",
        "check_federated_multiseed_runner_plumbing.py",
        "check_plan_runner_common.py",
        "check_posterior_cache_plumbing.py",
        "check_secure_aggregation_plumbing.py",
    ):
        _run(script)
    _check_manuscript_and_tables()
    print(
        f"PASS public smoke python={python_count} metadata={metadata_count} "
        "training=0 downloads=0 tables=6"
    )


if __name__ == "__main__":
    main()
