"""Task-specific label harmonization for public EEG datasets."""

from __future__ import annotations

from collections.abc import Iterable


LEFT_HAND_ALIASES = frozenset(
    {
        "left_hand",
        "left hand",
        "left",
        "hands_left",
        "left_hand_imagery",
    }
)

RIGHT_HAND_ALIASES = frozenset(
    {
        "right_hand",
        "right hand",
        "right",
        "hands_right",
        "right_hand_imagery",
    }
)


def normalize_label(label: str) -> str:
    """Normalize labels to a stable lowercase underscore format."""

    return label.strip().lower().replace("-", "_").replace(" ", "_")


def canonicalize_left_right_label(label: str) -> str | None:
    """Map dataset-specific labels to the version-1 task labels."""

    normalized = normalize_label(label)
    if normalized in LEFT_HAND_ALIASES:
        return "left_hand"
    if normalized in RIGHT_HAND_ALIASES:
        return "right_hand"
    return None


def extract_available_canonical_labels(labels: Iterable[str]) -> tuple[str, ...]:
    """Return canonical labels present in a label collection."""

    present = {
        canonical
        for label in labels
        if (canonical := canonicalize_left_right_label(label)) is not None
    }
    return tuple(sorted(present))
