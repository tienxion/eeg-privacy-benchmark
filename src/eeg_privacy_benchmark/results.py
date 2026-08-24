"""Small helpers for writing experiment results."""

from __future__ import annotations

import json
from pathlib import Path


def write_result_json(payload: dict, output_path: str | Path) -> None:
    """Write one result payload to JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
