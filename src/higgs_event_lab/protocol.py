"""Shared, leakage-safe experiment protocol for the HIGGS benchmark.

Model owners use this module to create the one stratified train/validation/test
partition required for fair comparison.  Splits are index based so every model
can evaluate the exact same held-out events without persisting the dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

SPLIT_PROTOCOL_VERSION = "higgs-split-v1"
ARTIFACT_SCHEMA_VERSION = "higgs-experiment-v1"


@dataclass(frozen=True)
class SplitConfig:
    """Parameters for the project-wide stratified holdout protocol."""

    seed: int = 42
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    def __post_init__(self) -> None:
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation_fraction + test_fraction must be less than 1")


@dataclass(frozen=True)
class SplitIndices:
    """Integer row positions for a reproducible three-way partition."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    config: SplitConfig

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_version": SPLIT_PROTOCOL_VERSION,
            "seed": self.config.seed,
            "validation_fraction": self.config.validation_fraction,
            "test_fraction": self.config.test_fraction,
            "train_rows": int(self.train.size),
            "validation_rows": int(self.validation.size),
            "test_rows": int(self.test.size),
        }


def create_stratified_splits(
    labels: np.ndarray | list[int], config: SplitConfig = SplitConfig()
) -> SplitIndices:
    """Make deterministic, disjoint splits without inspecting feature values.

    The validation and test sets are never used to fit scalers or model weights.
    A single-class or too-small label vector fails early rather than producing a
    misleading experimental result.
    """
    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    if y.size < 10:
        raise ValueError("at least 10 rows are required to create stable splits")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("labels must contain both binary classes 0 and 1")

    all_indices = np.arange(y.size)
    holdout_fraction = config.validation_fraction + config.test_fraction
    first = StratifiedShuffleSplit(
        n_splits=1, test_size=holdout_fraction, random_state=config.seed
    )
    train_idx, holdout_idx = next(first.split(all_indices, y))

    test_share_of_holdout = config.test_fraction / holdout_fraction
    second = StratifiedShuffleSplit(
        n_splits=1, test_size=test_share_of_holdout, random_state=config.seed
    )
    validation_local, test_local = next(second.split(holdout_idx, y[holdout_idx]))
    validation_idx = holdout_idx[validation_local]
    test_idx = holdout_idx[test_local]

    return SplitIndices(
        train=np.sort(train_idx),
        validation=np.sort(validation_idx),
        test=np.sort(test_idx),
        config=config,
    )


def new_experiment_artifact(
    *,
    run_id: str,
    model_name: str,
    feature_group: str,
    feature_names: list[str],
    split: SplitIndices,
    data_source: str,
    row_count: int,
) -> dict[str, Any]:
    """Create the common JSON envelope consumed by reports and the dashboard."""
    if not run_id.strip() or not model_name.strip():
        raise ValueError("run_id and model_name must be non-empty")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "model": {"name": model_name, "family": "neural_network"},
        "features": {
            "group": feature_group,
            "count": len(feature_names),
            "names": feature_names,
        },
        "data": {"source": data_source, "row_count": row_count},
        "split": split.metadata(),
        "training": {},
        "evaluation": {},
    }
