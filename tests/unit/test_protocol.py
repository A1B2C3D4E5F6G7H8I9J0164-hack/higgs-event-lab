"""Tests for Krishna's shared, leakage-safe split and artifact protocol."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from higgs_event_lab.protocol import ARTIFACT_SCHEMA_VERSION, SplitConfig, create_stratified_splits, new_experiment_artifact


def test_splits_are_deterministic_disjoint_and_stratified():
    labels = np.array([0, 1] * 100)
    first = create_stratified_splits(labels, SplitConfig(seed=17))
    second = create_stratified_splits(labels, SplitConfig(seed=17))
    assert np.array_equal(first.train, second.train)
    assert not set(first.train) & set(first.validation)
    assert not set(first.train) & set(first.test)
    assert not set(first.validation) & set(first.test)
    assert set(np.concatenate([first.train, first.validation, first.test])) == set(range(len(labels)))
    assert labels[first.train].mean() == pytest.approx(0.5, abs=0.02)


def test_artifact_envelope_records_contract_metadata():
    labels = np.array([0, 1] * 10)
    split = create_stratified_splits(labels)
    artifact = new_experiment_artifact(run_id="tiny", model_name="compact_mlp", feature_group="all", feature_names=["x1"], split=split, data_source="synthetic", row_count=len(labels))
    assert artifact["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert artifact["split"]["seed"] == 42
    assert artifact["features"]["names"] == ["x1"]
