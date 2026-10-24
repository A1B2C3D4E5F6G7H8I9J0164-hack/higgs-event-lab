"""Unit tests for shared evaluation and threshold selection module."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from higgs_event_lab.artifacts import save_artifact_json
from higgs_event_lab.evaluation import compute_classification_metrics, select_operating_threshold
from higgs_event_lab.protocol import create_stratified_splits, new_experiment_artifact
from higgs_event_lab.splits import SplitConfig, create_stratified_splits as splits_creator


def test_compute_classification_metrics_returns_standard_keys():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    probs = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.15, 0.65, 0.4])

    metrics = compute_classification_metrics(y_true, probs, n_bins=5)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert len(metrics["calibration"]["fraction_positives"]) == len(metrics["calibration"]["mean_predicted_value"])


def test_compute_classification_metrics_validates_inputs():
    with pytest.raises(ValueError, match="same length"):
        compute_classification_metrics(np.array([0, 1]), np.array([0.5]))

    with pytest.raises(ValueError, match="within"):
        compute_classification_metrics(np.array([0, 1]), np.array([0.5, 1.5]))

    with pytest.raises(ValueError, match="both binary classes"):
        compute_classification_metrics(np.array([1, 1]), np.array([0.5, 0.6]))


def test_select_operating_threshold():
    y_val = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    probs_val = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    threshold = select_operating_threshold(y_val, probs_val, target_signal_efficiency=0.50)
    assert 0.1 <= threshold <= 0.9


def test_splits_and_artifacts_wrappers(tmp_path: Path):
    labels = np.array([0, 1] * 10)
    split = splits_creator(labels, SplitConfig(seed=12))
    assert len(split.train) > 0

    envelope = new_experiment_artifact(
        run_id="wrap_test",
        model_name="test_model",
        feature_group="all",
        feature_names=["x1", "x2"],
        split=split,
        data_source="unit_test",
        row_count=20,
    )
    saved = save_artifact_json(envelope, tmp_path / "art" / "test.json")
    assert saved.exists()
