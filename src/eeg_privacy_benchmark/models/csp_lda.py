"""Classical CSP plus LDA baseline for the EEG privacy benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from eeg_privacy_benchmark.datasets.base import DatasetManifest
from eeg_privacy_benchmark.datasets.factory import build_dataset_loader
from eeg_privacy_benchmark.datasets.splits import (
    generate_cross_run_split,
    generate_cross_session_split,
    generate_cross_subject_split,
)


def _import_model_dependencies():
    try:
        import numpy as np
        from mne.decoding import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.metrics import balanced_accuracy_score, f1_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder
    except ImportError as exc:
        raise ImportError(
            "CSP+LDA baseline requires numpy, mne, and scikit-learn."
        ) from exc
    return {
        "np": np,
        "CSP": CSP,
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
        "balanced_accuracy_score": balanced_accuracy_score,
        "f1_score": f1_score,
        "Pipeline": Pipeline,
        "LabelEncoder": LabelEncoder,
    }


@dataclass(frozen=True)
class BaselineResult:
    dataset_key: str
    protocol: str
    seed: int
    train_trials: int
    test_trials: int
    balanced_accuracy: float
    macro_f1: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class CSPTrainingArtifacts:
    dataset_key: str
    protocol: str
    seed: int
    pipeline: object
    result: BaselineResult
    all_features: object
    encoded_labels: object
    trial_records: tuple
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


def fit_csp_lda(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    n_components: int = 8,
) -> CSPTrainingArtifacts:
    """Fit the CSP+LDA baseline and return the trained pipeline plus metadata."""

    deps = _import_model_dependencies()
    loader = build_dataset_loader(dataset_key)
    loaded = loader.load_array_data(subjects=subjects)
    manifest = DatasetManifest(
        dataset_key=dataset_key,
        trial_records=loaded.trial_records,
        available_labels=tuple(sorted({str(label) for label in loaded.labels})),
        notes=loader.dataset_spec.notes,
    )

    if protocol == "cross_subject":
        split_manifest = generate_cross_subject_split(manifest, seed=seed)
    elif protocol == "cross_session":
        split_manifest = generate_cross_session_split(manifest, seed=seed)
    elif protocol == "cross_run":
        split_manifest = generate_cross_run_split(manifest, seed=seed)
    else:
        raise ValueError(f"Unsupported protocol for CSP+LDA baseline: {protocol}")

    split_lookup = {
        assignment.trial_id: assignment.split for assignment in split_manifest.assignments
    }
    train_indices = [
        index
        for index, record in enumerate(loaded.trial_records)
        if split_lookup[record.trial_id] == "train"
    ]
    test_indices = [
        index
        for index, record in enumerate(loaded.trial_records)
        if split_lookup[record.trial_id] == "test"
    ]
    if not train_indices or not test_indices:
        raise ValueError(f"Protocol {protocol} did not produce both train and test trials.")

    label_encoder = deps["LabelEncoder"]()
    encoded_labels = label_encoder.fit_transform(loaded.labels)

    pipeline = deps["Pipeline"](
        steps=[
            ("csp", deps["CSP"](n_components=n_components, reg=None, log=True, norm_trace=False)),
            ("lda", deps["LinearDiscriminantAnalysis"]()),
        ]
    )
    pipeline.fit(loaded.features[train_indices], encoded_labels[train_indices])
    predictions = pipeline.predict(loaded.features[test_indices])

    result = BaselineResult(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        train_trials=len(train_indices),
        test_trials=len(test_indices),
        balanced_accuracy=float(
            deps["balanced_accuracy_score"](encoded_labels[test_indices], predictions)
        ),
        macro_f1=float(
            deps["f1_score"](encoded_labels[test_indices], predictions, average="macro")
        ),
    )

    csp_features = pipeline.named_steps["csp"].transform(loaded.features)
    return CSPTrainingArtifacts(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        pipeline=pipeline,
        result=result,
        all_features=csp_features,
        encoded_labels=encoded_labels,
        trial_records=loaded.trial_records,
        train_indices=tuple(train_indices),
        test_indices=tuple(test_indices),
    )


def run_csp_lda(
    dataset_key: str,
    *,
    protocol: str,
    seed: int,
    subjects: list[int] | None = None,
    n_components: int = 8,
) -> BaselineResult:
    """Train and evaluate the CSP+LDA baseline on one dataset/protocol."""

    return fit_csp_lda(
        dataset_key=dataset_key,
        protocol=protocol,
        seed=seed,
        subjects=subjects,
        n_components=n_components,
    ).result
