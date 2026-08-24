"""Evaluation protocol definitions for the benchmark."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationProtocol:
    name: str
    description: str


CROSS_SUBJECT = EvaluationProtocol(
    name="cross_subject",
    description="Train and test on disjoint subject sets within a dataset.",
)

CROSS_SESSION = EvaluationProtocol(
    name="cross_session",
    description="Train and test on disjoint sessions for the same dataset.",
)

CROSS_RUN = EvaluationProtocol(
    name="cross_run",
    description="Train and test on disjoint recording runs for one dataset.",
)

CROSS_DATASET = EvaluationProtocol(
    name="cross_dataset",
    description="Train on one or more datasets and evaluate on a held-out study.",
)
