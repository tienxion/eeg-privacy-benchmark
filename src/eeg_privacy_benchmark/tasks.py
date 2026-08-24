"""Canonical task definitions for benchmark experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    label_names: tuple[str, ...]
    description: str


LEFT_RIGHT_MOTOR_IMAGERY = TaskSpec(
    name="left_vs_right_motor_imagery",
    label_names=("left_hand", "right_hand"),
    description=(
        "Binary motor-imagery classification restricted to left-hand and "
        "right-hand imagery so the task is shared cleanly across all "
        "version-1 datasets."
    ),
)
